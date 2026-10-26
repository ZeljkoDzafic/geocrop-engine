"""Presto encoder (Tseng et al., 2023, arXiv:2304.14065) for on-VM embeddings. Requires torch (`[deep]` extra).

Adapted from the official implementation, nasaharvest/presto `single_file_presto.py` at commit
11e207a668a34336ced1d8e492a1bd5849b96c4a, which is distributed under the MIT License:

    Copyright (c) 2024 Presto Authors
    Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
    associated documentation files (the "Software"), to deal in the Software without restriction, including
    without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the
    following conditions: The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF
    ANY KIND, EXPRESS OR IMPLIED.

Only the encoder is kept (the decoder is needed for pre-training, not for embeddings). Parameter names match
the official checkpoint (`encoder.*` keys of `data/default_model.pt`), so the pinned weights load with
``strict=True``. Differences from the original, none of which change the arithmetic: einops replaced by
plain torch, the per-batch "same number of masked tokens" requirement is met by the caller
(`geocrop.embeddings.presto_embed` groups pixels by mask pattern), and the encoder always returns the
pooled, normalised embedding (``eval_task=True`` path).
"""
from __future__ import annotations

import math
from collections import OrderedDict

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

#: Presto channel groups over the 17 normalised input bands (see geocrop.embeddings.PRESTO_BANDS).
BANDS_GROUPS_IDX = OrderedDict([
    ("S1", [0, 1]), ("S2_RGB", [2, 3, 4]), ("S2_Red_Edge", [5, 6, 7]), ("S2_NIR_10m", [8]),
    ("S2_NIR_20m", [9]), ("S2_SWIR", [10, 11]), ("ERA5", [12, 13]), ("SRTM", [14, 15]), ("NDVI", [16]),
])
NUM_DYNAMIC_WORLD_CLASSES = 9          # class index 9 = missing (always masked here)


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        x = F.scaled_dot_product_attention(q, k, v)
        return self.proj(x.transpose(1, 2).reshape(B, N, C))


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


def sinusoid_table(positions: int, d_hid: int, T: float = 1000) -> torch.Tensor:
    pos = np.arange(positions)[:, None]
    hid = np.arange(d_hid)[None, :]
    table = pos / np.power(T, 2 * (hid // 2) / d_hid)
    table[:, 0::2] = np.sin(table[:, 0::2])
    table[:, 1::2] = np.cos(table[:, 1::2])
    return torch.tensor(table, dtype=torch.float32)


def month_table(d_hid: int) -> torch.Tensor:
    """Sinusoidal month encoding for months 0..11 (0 = January)."""
    angles = np.arange(0, 13) / (12 / (2 * np.pi))
    sin = np.sin(np.stack([angles] * (d_hid // 2), axis=-1))
    cos = np.cos(np.stack([angles] * (d_hid // 2), axis=-1))
    return torch.tensor(np.concatenate([sin[:-1], cos[:-1]], axis=-1), dtype=torch.float32)


class PrestoEncoder(nn.Module):
    """Presto encoder with the default (published) hyper-parameters: 128-d, depth 2, 8 heads, MLP ratio 4."""

    def __init__(self, embedding_size: int = 128, channel_embed_ratio: float = 0.25, month_embed_ratio: float = 0.25,
                 depth: int = 2, mlp_ratio: float = 4, num_heads: int = 8, max_sequence_length: int = 24):
        super().__init__()
        self.band_groups = BANDS_GROUPS_IDX
        self.embedding_size = embedding_size
        self.band_group_to_idx = {g: i for i, g in enumerate(self.band_groups)}
        self.band_group_to_idx["dynamic_world"] = len(self.band_groups)
        self.eo_patch_embed = nn.ModuleDict({g: nn.Linear(len(idx), embedding_size)
                                             for g, idx in self.band_groups.items()})
        self.dw_embed = nn.Embedding(NUM_DYNAMIC_WORLD_CLASSES + 1, embedding_size)
        self.latlon_embed = nn.Linear(3, embedding_size)
        self.blocks = nn.ModuleList([Block(embedding_size, num_heads, mlp_ratio) for _ in range(depth)])
        self.norm = nn.LayerNorm(embedding_size)
        self.max_sequence_length = max_sequence_length
        pos_size = int(embedding_size * (1 - (channel_embed_ratio + month_embed_ratio)))
        channel_size = int(embedding_size * channel_embed_ratio)
        month_size = int(embedding_size * month_embed_ratio)
        self.pos_embed = nn.Parameter(sinusoid_table(max_sequence_length, pos_size)[None], requires_grad=False)
        self.month_embed = nn.Embedding.from_pretrained(month_table(month_size), freeze=True)
        self.channel_embed = nn.Embedding(len(self.band_groups) + 1, channel_size)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @staticmethod
    def cartesian(latlons: torch.Tensor) -> torch.Tensor:
        rad = latlons * math.pi / 180
        lats, lons = rad[:, 0], rad[:, 1]
        return torch.stack([torch.cos(lats) * torch.cos(lons), torch.cos(lats) * torch.sin(lons), torch.sin(lats)], -1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor, latlons: torch.Tensor, month: int = 0) -> torch.Tensor:
        """x, mask: (B, T, 17) normalised inputs / 1 = missing; latlons (B, 2) degrees; month: first month 0..11.

        All rows of the batch must have the same token mask (identical missing pattern)."""
        B, T, _ = x.shape
        if not 0 <= int(month) < 12:
            raise ValueError("month must be 0..11 (0 = January)")
        if T > self.max_sequence_length:
            raise ValueError(f"Presto accepts at most {self.max_sequence_length} timesteps")
        months = torch.fmod(torch.arange(int(month), int(month) + T, device=x.device), 12)
        month_emb = self.month_embed(months)[None].expand(B, T, -1)
        pos_emb = self.pos_embed[:, :T, :].expand(B, T, -1)
        tokens, masks = [], []
        for group, idx in self.band_groups.items():
            tok = self.eo_patch_embed[group](x[:, :, idx])
            ch = self.channel_embed.weight[self.band_group_to_idx[group]][None, None].expand(B, T, -1)
            if group == "SRTM":            # static: one token, no month/position encoding
                extra = torch.cat((torch.zeros_like(month_emb[:, :1]), ch[:, :1], torch.zeros_like(pos_emb[:, :1])), -1)
                tok, gmask = tok[:, :1] + extra, mask[:, :1, idx].amax(-1)
            else:
                tok, gmask = tok + torch.cat((month_emb, ch, pos_emb), -1), mask[:, :, idx].amax(-1)
            tokens.append(tok)
            masks.append(gmask)
        dw = torch.full((B, T), NUM_DYNAMIC_WORLD_CLASSES, dtype=torch.long, device=x.device)
        ch = self.channel_embed.weight[self.band_group_to_idx["dynamic_world"]][None, None].expand(B, T, -1)
        tokens.append(self.dw_embed(dw) + torch.cat((month_emb, ch, pos_emb), -1))
        masks.append((dw == NUM_DYNAMIC_WORLD_CLASSES).to(x.dtype))
        allt = torch.cat(tokens, 1)
        keep = ~torch.cat(masks, 1).bool()
        if not bool((keep == keep[:1]).all()):
            raise ValueError("All batch rows must share one token mask; group rows by mask pattern")
        allt = allt[:, keep[0]]
        allt = torch.cat((self.latlon_embed(self.cartesian(latlons))[:, None], allt), 1)
        for blk in self.blocks:
            allt = blk(allt)
        return self.norm(allt.mean(dim=1))


def load(checkpoint: str | None = None, map_location: str = "cpu") -> PrestoEncoder:
    """Encoder with the weights of a full Presto checkpoint (``encoder.*`` keys); random weights if None."""
    model = PrestoEncoder()
    if checkpoint is not None:
        state = torch.load(checkpoint, map_location=map_location, weights_only=True)
        enc = {k[len("encoder."):]: v for k, v in state.items() if k.startswith("encoder.")}
        model.load_state_dict(enc, strict=True)
    return model.eval()
