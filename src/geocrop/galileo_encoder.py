"""Galileo encoder (Tseng et al., 2025, arXiv:2502.09356) for on-VM embeddings. Requires torch (`[deep]` extra).

Adapted from the official implementation, nasaharvest/galileo `single_file_galileo.py` (and the input
handling of `src/data/dataset.py` and `src/eval/cropharvest/cropharvest_eval.py`) at commit
0f0b5b95ac81acef4b74cf4686dc877202f4541b, which is distributed under the MIT License:

    MIT License

    Copyright (c) 2024 Presto Authors

    Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
    associated documentation files (the "Software"), to deal in the Software without restriction, including
    without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the
    following conditions: The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF
    ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR
    A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR
    ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

Only the encoder is kept (decoders are needed for pre-training, not for embeddings). Parameter names match the
official ``encoder.pt`` files, so published weights load with ``strict=True``. Differences from the original,
none of which change the arithmetic: einops replaced by plain torch reshapes, the token-exit / ``exit_after``
training options are removed (the encoder always runs every block and applies the final LayerNorm, as at
evaluation time), FlexiPatchEmbed pseudo-inverses are computed on first use instead of at construction, and the
attention mask is broadcast instead of repeated per head.

Input contract (official training / evaluation normaliser, ``Normalizer(std=True, std_multiplier=2)`` with the
statistics of ``config/normalization.json``): space-time bands ``VV, VH`` in dB, ``B2 ... B12`` as Sentinel-2
digital numbers (reflectance x 10000; pre-training used the L1C ``COPERNICUS/S2_HARMONIZED`` collection) and
``NDVI`` = (B8 - B4) / (B8 + B4), 0 where B8 + B4 <= 0. Every band except NDVI is scaled with
``(x - (mean - 2 std)) / (4 std)``; NDVI is used as is. Months are 0-based (0 = January).
"""
from __future__ import annotations

import json
import math
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

BASE_GSD = 10

S1_BANDS = ["VV", "VH"]
S2_BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
SPACE_TIME_BANDS = S1_BANDS + S2_BANDS + ["NDVI"]
TIME_BANDS = ["temperature_2m", "total_precipitation_sum", "def", "soil", "aet", "avg_rad"]
DW_BANDS = ["DW_water", "DW_trees", "DW_grass", "DW_flooded_vegetation", "DW_crops", "DW_shrub_and_scrub",
            "DW_built", "DW_bare", "DW_snow_and_ice"]
WC_BANDS = ["WC_temporarycrops", "WC_maize", "WC_wintercereals", "WC_springcereals", "WC_irrigation"]
SPACE_BANDS = ["elevation", "slope"] + DW_BANDS + WC_BANDS
LOCATION_BANDS = ["x", "y", "z"]
STATIC_BANDS = ["b1"] + LOCATION_BANDS + [f"{b}_static" for b in DW_BANDS] + [f"{b}_static" for b in WC_BANDS]

SPACE_TIME_GROUPS: OrderedDict[str, list[int]] = OrderedDict([
    ("S1", [0, 1]), ("S2_RGB", [2, 3, 4]), ("S2_Red_Edge", [5, 6, 7]), ("S2_NIR_10m", [8]),
    ("S2_NIR_20m", [9]), ("S2_SWIR", [10, 11]), ("NDVI", [12]),
])
TIME_GROUPS: OrderedDict[str, list[int]] = OrderedDict([("ERA5", [0, 1]), ("TC", [2, 3, 4]), ("VIIRS", [5])])
SPACE_GROUPS: OrderedDict[str, list[int]] = OrderedDict([
    ("SRTM", [0, 1]), ("DW", list(range(2, 11))), ("WC", list(range(11, 16))),
])
STATIC_GROUPS: OrderedDict[str, list[int]] = OrderedDict([
    ("LS", [0]), ("location", [1, 2, 3]), ("DW_static", list(range(4, 13))), ("WC_static", list(range(13, 18))),
])

#: Pre-training statistics of the 13 space-time bands (official config/normalization.json, key "13").
SPACE_TIME_MEAN = np.array([
    -11.728724389184965, -18.85558188024017, 1395.3408730676722, 1338.4026921784578, 1343.09883810357,
    1543.8607982512297, 2186.2022069512263, 2525.0932853316694, 2410.3377187373408, 2750.2854646886753,
    2234.911100061487, 1474.5311266077113, 0.2892116502999044])
SPACE_TIME_STD = np.array([
    4.887145774840316, 5.730270320384293, 917.7041440370853, 913.2988423581528, 1092.678723527555,
    1047.2206083460424, 1048.0101611156767, 1143.6903026819996, 1098.979177731649, 1204.472755085893,
    1145.9774063078878, 980.2429840007796, 0.2720939024500081])
STD_MULTIPLIER = 2.0
SPACE_TIME_SHIFT = np.append(SPACE_TIME_MEAN[:12] - STD_MULTIPLIER * SPACE_TIME_STD[:12], 0.0)
SPACE_TIME_DIV = np.append(2 * STD_MULTIPLIER * SPACE_TIME_STD[:12], 1.0)

#: Encoder hyper-parameters of the published "nano" model (models/nano/config.json).
NANO_CONFIG = {"embedding_size": 128, "depth": 4, "num_heads": 8, "mlp_ratio": 4, "max_sequence_length": 24,
               "freeze_projections": False, "drop_path": 0.1, "max_patch_size": 8}


def normalise_space_time(x: np.ndarray) -> np.ndarray:
    """(..., 13) raw space-time bands (S1 dB, S2 DN, NDVI) -> Galileo's normalised inputs (float32)."""
    x = np.asarray(x, dtype="float64")
    return ((x - SPACE_TIME_SHIFT) / SPACE_TIME_DIV).astype("float32")


def to_cartesian(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    lat, lon = np.asarray(lat) * math.pi / 180, np.asarray(lon) * math.pi / 180
    return np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], axis=-1)


# --------------------------------------------------------------------------- positional encodings


def _sincos_1d(embed_dim: int, pos: torch.Tensor) -> torch.Tensor:
    omega = torch.arange(embed_dim // 2, device=pos.device) / embed_dim / 2.0
    omega = 1.0 / 10000 ** omega
    out = torch.einsum("m,d->md", pos.reshape(-1).to(omega.dtype), omega)
    return torch.cat([torch.sin(out), torch.cos(out)], dim=1)


def _sincos_2d_with_resolution(embed_dim: int, grid_size: int, res: torch.Tensor) -> torch.Tensor:
    """(n, grid_size * grid_size, embed_dim) for n resolutions (official get_2d_sincos_pos_embed_with_resolution)."""
    device = res.device
    ar = torch.arange(grid_size, device=device)
    grid = torch.stack(torch.meshgrid(ar, ar, indexing="xy"), dim=0)
    grid = torch.einsum("chw,n->cnhw", grid.to(res.dtype), res)
    _, n, h, w = grid.shape
    emb = torch.cat([_sincos_1d(embed_dim // 2, grid[0]), _sincos_1d(embed_dim // 2, grid[1])], dim=1)
    return emb.reshape(n, h * w, embed_dim)


def _month_table(embed_dim: int) -> torch.Tensor:
    angles = torch.arange(0, 13) / (12 / (2 * np.pi))
    sin = torch.sin(torch.stack([angles for _ in range(embed_dim // 2)], dim=-1))
    cos = torch.cos(torch.stack([angles for _ in range(embed_dim // 2)], dim=-1))
    return torch.cat([sin[:-1], cos[:-1]], dim=-1)


# --------------------------------------------------------------------------- building blocks


class FlexiPatchEmbed(nn.Module):
    """Patch embedding whose 8 x 8 kernel is resized to smaller patches by pseudo-inverse resampling."""

    def __init__(self, patch_size: int, in_chans: int, embed_dim: int):
        super().__init__()
        self.patch_size = (patch_size, patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=self.patch_size, stride=self.patch_size, bias=True)
        self.pinvs: dict = {}

    def _pinv(self, new: tuple[int, int]) -> torch.Tensor:
        if new not in self.pinvs:
            old = self.patch_size
            mat = []
            for i in range(int(np.prod(old))):
                basis = torch.zeros(old)
                basis[np.unravel_index(i, old)] = 1.0
                mat.append(F.interpolate(basis[None, None], new, mode="bicubic", antialias=True)[0, 0].reshape(-1))
            self.pinvs[new] = torch.linalg.pinv(torch.stack(mat))
        return self.pinvs[new]

    def forward(self, x: torch.Tensor, patch_size: int) -> torch.Tensor:
        # x: [b, h, w, (t), c]
        b = x.shape[0]
        timed = x.dim() == 5
        if timed:
            t = x.shape[3]
            x = x.permute(0, 3, 4, 1, 2).reshape(b * t, x.shape[4], x.shape[1], x.shape[2])
        else:
            x = x.permute(0, 3, 1, 2)
        ps = (int(patch_size), int(patch_size))
        weight = self.proj.weight
        if ps != self.patch_size:
            pinv = self._pinv(ps).to(weight.device)
            o, i = weight.shape[:2]
            weight = (weight.reshape(o * i, -1) @ pinv.T).reshape(o, i, *ps)
        x = F.conv2d(x, weight, bias=self.proj.bias, stride=ps)
        if timed:
            x = x.reshape(b, t, *x.shape[1:]).permute(0, 3, 4, 1, 2)
        else:
            x = x.permute(0, 2, 3, 1)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, attn_mask=None):
        B, N, C = x.shape
        h = self.num_heads

        def heads(t):
            return t.reshape(B, N, h, C // h).transpose(1, 2)

        q, k, v = heads(self.q(x)), heads(self.k(x)), heads(self.v(x))
        if attn_mask is not None:
            attn_mask = attn_mask[:, None, None].expand(B, h, N, N)
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
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

    def forward(self, x, attn_mask):
        x = x + self.attn(self.norm1(x), attn_mask)
        return x + self.mlp(self.norm2(x))


def _init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


# --------------------------------------------------------------------------- encoder


class GalileoEncoder(nn.Module):
    """Galileo encoder; the defaults of the published nano model are in NANO_CONFIG."""

    def __init__(self, max_patch_size: int = 8, embedding_size: int = 128, depth: int = 2, mlp_ratio: float = 2,
                 num_heads: int = 8, max_sequence_length: int = 24, freeze_projections: bool = False,
                 drop_path: float = 0.0):
        super().__init__()
        del freeze_projections, drop_path        # training-only options
        self.embedding_size = embedding_size
        self.base_patch_size = max_patch_size
        self.max_sequence_length = max_sequence_length
        q = int(embedding_size * 0.25)
        self.blocks = nn.ModuleList([Block(embedding_size, num_heads, mlp_ratio) for _ in range(depth)])
        self.pos_embed = nn.Parameter(_sincos_1d(q, torch.arange(max_sequence_length)), requires_grad=False)
        self.month_embed = nn.Embedding.from_pretrained(_month_table(q), freeze=True)
        self.s_t_channel_embed = nn.Parameter(torch.zeros(len(SPACE_TIME_GROUPS), q))
        self.sp_channel_embed = nn.Parameter(torch.zeros(len(SPACE_GROUPS), q))
        self.t_channel_embed = nn.Parameter(torch.zeros(len(TIME_GROUPS), q))
        self.st_channel_embed = nn.Parameter(torch.zeros(len(STATIC_GROUPS), q))
        self.space_time_embed = nn.ModuleDict({g: FlexiPatchEmbed(max_patch_size, len(i), embedding_size)
                                               for g, i in SPACE_TIME_GROUPS.items()})
        self.space_embed = nn.ModuleDict({g: FlexiPatchEmbed(max_patch_size, len(i), embedding_size)
                                          for g, i in SPACE_GROUPS.items()})
        self.time_embed = nn.ModuleDict({g: nn.Linear(len(i), embedding_size) for g, i in TIME_GROUPS.items()})
        self.static_embed = nn.ModuleDict({g: nn.Linear(len(i), embedding_size) for g, i in STATIC_GROUPS.items()})
        self.norm = nn.LayerNorm(embedding_size)
        self.apply(_init_weights)

    # ------------------------------------------------------------------ token bookkeeping
    @staticmethod
    def collapse_and_combine_hwtc(s_t_x, sp_x, t_x, st_x, s_t_m, sp_m, t_m, st_m):
        b, d = s_t_x.shape[0], s_t_x.shape[-1]
        x = torch.cat([s_t_x.reshape(b, -1, d), sp_x.reshape(b, -1, d), t_x.reshape(b, -1, d), st_x], dim=1)
        m = torch.cat([s_t_m.reshape(b, -1), sp_m.reshape(b, -1), t_m.reshape(b, -1), st_m], dim=1)
        return x, m

    @staticmethod
    def split_and_expand_hwtc(x, h, w, t, s_t_c_g, sp_c_g, t_c_g, st_c_g):
        b, _, d = x.shape
        n_s_t, n_t = h * w * t * s_t_c_g, t * t_c_g
        s_t_x = x[:, :n_s_t].reshape(b, h, w, t, s_t_c_g, d)
        sp_x = x[:, n_s_t:-(n_t + st_c_g)].reshape(b, h, w, sp_c_g, d)
        t_x = x[:, -(n_t + st_c_g):-st_c_g].reshape(b, t, t_c_g, d)
        return s_t_x, sp_x, t_x, x[:, -st_c_g:]

    @staticmethod
    def remove_masked_tokens(x, mask):
        dtype = mask.dtype
        mask = mask.bool()
        sorted_mask, indices = torch.sort((~mask).int(), dim=1, descending=True, stable=True)
        x = x.gather(1, indices[:, :, None].expand_as(x))
        x = x * sorted_mask.unsqueeze(-1)
        max_length = sorted_mask.sum(-1).max()
        x = x[:, :max_length]
        return x, indices, (1 - sorted_mask[:, :max_length]).to(dtype=dtype)

    @staticmethod
    def add_removed_tokens(x, indices, mask):
        out = torch.zeros(x.shape[0], indices.shape[1], x.shape[-1], dtype=x.dtype, device=x.device)
        full_mask = torch.cat((mask, torch.ones((x.shape[0], indices.shape[1] - x.shape[1]), device=x.device,
                                                dtype=mask.dtype)), dim=-1)
        out[~full_mask.bool()] = x[~mask.bool()]
        out = out.scatter(1, indices[:, :, None].expand_as(out), out)
        full_mask = full_mask.scatter(1, indices.expand_as(full_mask), full_mask)
        return out, full_mask

    @classmethod
    def average_tokens(cls, s_t_x, sp_x, t_x, st_x, s_t_m, sp_m, t_m, st_m):
        x, m = cls.collapse_and_combine_hwtc(s_t_x, sp_x, t_x, st_x, s_t_m, sp_m, t_m, st_m)
        x, _, m = cls.remove_masked_tokens(x, m)
        x_for_mean = x * (1 - m.unsqueeze(-1))
        return x_for_mean.sum(dim=1) / torch.sum(1 - m, -1, keepdim=True)

    # ------------------------------------------------------------------ forward
    def apply_linear_projection(self, s_t_x, sp_x, t_x, st_x, s_t_m, sp_m, t_m, st_m, patch_size):
        b, h, w, t, _ = s_t_x.shape
        nh, nw, d = h // patch_size, w // patch_size, self.embedding_size
        s_t_l, sp_l, t_l, st_l, s_t_ml, sp_ml, t_ml, st_ml = [], [], [], [], [], [], [], []
        for idx, (g, ch) in enumerate(SPACE_TIME_GROUPS.items()):
            s_t_ml.append(s_t_m[:, 0::patch_size, 0::patch_size, :, idx])
            s_t_l.append(self.space_time_embed[g](s_t_x[..., ch], patch_size) if s_t_ml[-1].min() == 0 else
                         torch.zeros(b, nh, nw, t, d, dtype=s_t_x.dtype, device=s_t_x.device))
        for idx, (g, ch) in enumerate(SPACE_GROUPS.items()):
            sp_ml.append(sp_m[:, 0::patch_size, 0::patch_size, idx])
            sp_l.append(self.space_embed[g](sp_x[..., ch], patch_size) if sp_ml[-1].min() == 0 else
                        torch.zeros(b, nh, nw, d, dtype=sp_x.dtype, device=sp_x.device))
        for idx, (g, ch) in enumerate(TIME_GROUPS.items()):
            t_ml.append(t_m[:, :, idx])
            t_l.append(self.time_embed[g](t_x[:, :, ch]) if t_ml[-1].min() == 0 else
                       torch.zeros(b, t, d, dtype=t_x.dtype, device=t_x.device))
        for idx, (g, ch) in enumerate(STATIC_GROUPS.items()):
            st_ml.append(st_m[:, idx])
            st_l.append(self.static_embed[g](st_x[:, ch]) if st_ml[-1].min() == 0 else
                        torch.zeros(b, d, dtype=st_x.dtype, device=st_x.device))
        return (torch.stack(s_t_l, -2), torch.stack(sp_l, -2), torch.stack(t_l, -2), torch.stack(st_l, -2),
                torch.stack(s_t_ml, -1), torch.stack(sp_ml, -1), torch.stack(t_ml, -1), torch.stack(st_ml, -1))

    def apply_encodings(self, s_t_x, sp_x, t_x, st_x, months, patch_size, input_res):
        b, h, w, t, s_t_c_g, _ = s_t_x.shape
        sp_c_g, t_c_g, st_c_g = sp_x.shape[-2], t_x.shape[-2], st_x.shape[-2]
        q = int(self.embedding_size * 0.25)
        dev = s_t_x.device
        s_t_channel = self.s_t_channel_embed.expand(b, h, w, t, -1, -1)
        t_channel = self.t_channel_embed.expand(b, t, -1, -1)
        st_channel = self.st_channel_embed.expand(b, -1, -1)
        sp_channel = self.sp_channel_embed.expand(b, h, w, -1, -1)
        pos = self.pos_embed[:t]
        month = self.month_embed(months)                                   # b t q
        pos_s_t = pos[None, None, None, :, None].expand(b, h, w, t, s_t_c_g, q)
        m_s_t = month[:, None, None, :, None].expand(b, h, w, t, s_t_c_g, q)
        pos_t = pos[None, :, None].expand(b, t, t_c_g, q)
        m_t = month[:, :, None].expand(b, t, t_c_g, q)
        t_zeros = torch.zeros(b, t, t_c_g, q, device=dev)
        sp_zeros = torch.zeros(b, h, w, sp_c_g, sp_channel.shape[-1] * 2, device=dev)
        st_zeros = torch.zeros(b, st_c_g, st_channel.shape[-1] * 3, device=dev)
        if patch_size is None:
            patch_size = self.base_patch_size
        gsd_ratio = input_res * patch_size / BASE_GSD
        if h != w:
            raise ValueError("Galileo's spatial encoding requires h == w")
        spatial = _sincos_2d_with_resolution(q, h, torch.ones(b, device=dev) * gsd_ratio).reshape(b, h, w, q)
        spatial_s_t = spatial[:, :, :, None, None].expand(b, h, w, t, s_t_c_g, q)
        spatial_s = spatial[:, :, :, None].expand(b, h, w, sp_c_g, q)
        s_t_embed = torch.cat([s_t_channel, pos_s_t, m_s_t, spatial_s_t], dim=-1)
        sp_embed = torch.cat([sp_channel, sp_zeros, spatial_s], dim=-1)
        t_embed = torch.cat([t_channel, pos_t, m_t, t_zeros], dim=-1)
        st_embed = torch.cat([st_channel, st_zeros], dim=-1)
        return s_t_x + s_t_embed, sp_x + sp_embed, t_x + t_embed, st_x + st_embed

    def apply_attn(self, s_t_x, sp_x, t_x, st_x, s_t_m, sp_m, t_m, st_m, months, patch_size, input_res):
        _, h, w, t, s_t_c_g, _ = s_t_x.shape
        sp_c_g, t_c_g, st_c_g = sp_x.shape[3], t_x.shape[-2], st_x.shape[-2]
        s_t_x, sp_x, t_x, st_x = self.apply_encodings(s_t_x, sp_x, t_x, st_x, months, patch_size, input_res)
        x, m = self.collapse_and_combine_hwtc(s_t_x, sp_x, t_x, st_x, s_t_m, sp_m, t_m, st_m)
        x, indices, new_m = self.remove_masked_tokens(x, m >= 1)
        for blk in self.blocks:
            x = blk(x, attn_mask=~new_m.bool())
        x, _ = self.add_removed_tokens(x, indices, new_m)
        return (*self.split_and_expand_hwtc(x, h, w, t, s_t_c_g, sp_c_g, t_c_g, st_c_g), s_t_m, sp_m, t_m, st_m)

    def forward(self, s_t_x, sp_x, t_x, st_x, s_t_m, sp_m, t_m, st_m, months, patch_size: int,
                input_resolution_m: int = BASE_GSD):
        """Inputs as the official ``MaskedOutput`` ([B, H, W, T, 13], [B, H, W, 16], [B, T, 6], [B, 18] and the
        group masks, 1 = missing); returns the encoded tokens, the per-patch masks and the months."""
        if s_t_x.shape[3] > self.max_sequence_length:
            raise ValueError(f"Galileo accepts at most {self.max_sequence_length} timesteps")
        s_t_x, sp_x, t_x, st_x, s_t_m, sp_m, t_m, st_m = self.apply_linear_projection(
            s_t_x, sp_x, t_x, st_x, s_t_m, sp_m, t_m, st_m, patch_size)
        s_t_x, sp_x, t_x, st_x, s_t_m, sp_m, t_m, st_m = self.apply_attn(
            s_t_x, sp_x, t_x, st_x, s_t_m, sp_m, t_m, st_m, months, patch_size, input_resolution_m)
        return (self.norm(s_t_x), self.norm(sp_x), self.norm(t_x), self.norm(st_x), s_t_m, sp_m, t_m, st_m,
                months)


# --------------------------------------------------------------------------- loading and pixel encoding


def load(folder: str | Path | None = None, map_location: str = "cpu", *, config: str | Path | None = None,
         weights: str | Path | None = None) -> GalileoEncoder:
    """Encoder from ``folder/config.json`` + ``folder/encoder.pt`` (or explicit `config` / `weights` paths);
    random nano weights when nothing is given."""
    if folder is not None:
        config = config or Path(folder) / "config.json"
        weights = weights or Path(folder) / "encoder.pt"
    if config is not None:
        kwargs = json.loads(Path(config).read_text(encoding="utf8"))["model"]["encoder"]
    else:
        kwargs = dict(NANO_CONFIG)
    model = GalileoEncoder(**kwargs)
    if weights is not None:
        state = torch.load(weights, map_location=map_location, weights_only=True)
        state = {k.replace(".backbone", ""): v for k, v in state.items()}
        model.load_state_dict(state, strict=True)
    return model.eval()


def encode_pixels(model: GalileoEncoder, x: np.ndarray, mask: np.ndarray, months: np.ndarray,
                  latlons: np.ndarray | None = None, batch_size: int = 1024) -> np.ndarray:
    """Pixel time series -> (N, D) embeddings, as the official CropHarvest evaluation (1 x 1 pixel, patch 1).

    x: (N, T, 13) normalised space-time bands (see normalise_space_time); mask: (N, T, 7) group mask, 1 = missing;
    months: (T,) or (N, T), 0 = January; latlons: (N, 2) degrees or None (location token masked). Space and time
    groups (SRTM, Dynamic World, ERA5, ...) are masked. Rows without any space-time observation are NaN."""
    x = np.asarray(x, dtype="float32")
    mask = np.asarray(mask).astype(bool)
    n, t, _ = x.shape
    months = np.asarray(months, dtype="int64")
    months = np.broadcast_to(months, (n, t)) if months.ndim == 1 else months
    dims = model.embedding_size
    out = np.full((n, dims), np.nan, dtype="float32")
    rows = np.flatnonzero(~mask.all(axis=(1, 2)))
    if not len(rows):
        return out
    # zero-fill masked or invalid values so that NaN cannot leak into the (masked) tokens
    groups = list(SPACE_TIME_GROUPS.values())
    band_mask = np.zeros_like(x, dtype=bool)
    for g, idx in enumerate(groups):
        band_mask[:, :, idx] = mask[:, :, g:g + 1]
    x = np.where(band_mask | ~np.isfinite(x), 0, x).astype("float32")
    location = list(STATIC_GROUPS).index("location")
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            part = rows[start:start + batch_size]
            b = len(part)
            s_t_x = torch.from_numpy(x[part])[:, None, None]
            s_t_m = torch.from_numpy(mask[part].astype("float32"))[:, None, None]
            sp_x = torch.zeros(b, 1, 1, len(SPACE_BANDS))
            sp_m = torch.ones(b, 1, 1, len(SPACE_GROUPS))
            t_x = torch.zeros(b, t, len(TIME_BANDS))
            t_m = torch.ones(b, t, len(TIME_GROUPS))
            st_x = torch.zeros(b, len(STATIC_BANDS))
            st_m = torch.ones(b, len(STATIC_GROUPS))
            if latlons is not None:
                ll = np.asarray(latlons, dtype="float64")[part]
                st_x[:, STATIC_GROUPS["location"]] = torch.from_numpy(to_cartesian(ll[:, 0], ll[:, 1])).float()
                st_m[:, location] = 0
            enc = model(s_t_x, sp_x, t_x, st_x, s_t_m, sp_m, t_m, st_m, torch.from_numpy(months[part].copy()),
                        patch_size=1)
            out[part] = model.average_tokens(*enc[:8]).numpy()
    return out
