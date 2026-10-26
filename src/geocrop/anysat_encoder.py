"""AnySat encoder (Astruc et al., CVPR 2025, arXiv:2412.14123), inference path for Sentinel time series.

Requires torch (`[deep]` extra). Adapted from the official implementation, gastruc/AnySat at commit
5f6f475e1a22ce5e3a56a5b18f4ed6d24eca2a4a (``hubconf.py``, ``src/models/networks/encoder/Any_multi.py``,
``Transformer.py``, ``utils/ltae.py``, ``utils/utils_ViT.py``, ``utils/irpe.py``, ``utils/pos_embed.py``,
``utils/utils.py``), which is distributed under the MIT License:

    MIT License
    Copyright (c) 2023 Damien ROBERT
    Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
    associated documentation files (the "Software"), to deal in the Software without restriction, including
    without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the
    following conditions: The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF
    ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR
    A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
    FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

``irpe.py`` in that repository carries no header of its own; it is the image relative position encoding of
Wu et al. (ICCV 2021, microsoft/Cream ``iRPE``, MIT License, Copyright (c) Microsoft Corporation), as modified
by the AnySat authors, and is covered by the AnySat MIT licence above.

The pretrained weights (``models/AnySat.pth`` of the Hugging Face repository g-astruc/AnySat) are published
under the MIT licence according to the model card.

Only the release (feature-extraction) path of the time-series modalities ``s2`` (B2 B3 B4 B5 B6 B7 B8 B8A B11
B12), ``s1-asc`` (VV, VH) and ``s1`` (VV, VH, ratio) is kept. Parameter names equal the official ``AnyModule``
state-dict keys, so the matching subset of the official checkpoint loads with ``strict=True`` (projectors of
the other modalities are skipped). Differences from the original, none of which change the arithmetic:
einops/timm/hydra are not used; dropout, drop-path, patch dropout, layer scale and flash attention (all inactive
or identity at inference with ``flash_attn=False``) are omitted; the CUDA ``rpe_index`` op is replaced by the
official pure-PyTorch fallback; iRPE bucket ids are recomputed per call instead of being cached in a class
buffer; the training-time positional tables (empty for the released model) are not built. The official
quirk that the contextual key RPE of multi-head attention is taken from head 0 only (``gather`` with a
singleton head index) is reproduced deliberately.
"""
from __future__ import annotations

import math
from datetime import date

import numpy as np
import torch
from torch import nn

#: Channel order of the time-series modalities (official README).
S2_BANDS = ("B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12")
S1_ASC_BANDS = ("VV", "VH")
S1_BANDS = ("VV", "VH", "ratio")
MODALITY_CHANNELS = {"s2": len(S2_BANDS), "s1-asc": len(S1_ASC_BANDS), "s1": len(S1_BANDS)}
#: Ground resolution (m) of each modality (``hubconf.AnySat.res`` / spatial-encoder ``input_res``).
RESOLUTION = {"s2": 10, "s1-asc": 10, "s1": 10}

SIZES = {
    "base": {"dim": 768, "depth": 6, "heads": 12},          # the released checkpoint
    "tiny": {"dim": 64, "depth": 2, "heads": 4},            # random-weight stand-in for tests
}


def day_of_year(d: date) -> int:
    """Day-of-year convention of the pre-training data (TreeSatAI-TS, PLANTED, S2NAIP): ``tm_yday`` (1 Jan = 1)."""
    return d.timetuple().tm_yday


# --------------------------------------------------------------------------- positional encodings


def _sincos_1d(embed_dim: int, pos: torch.Tensor) -> torch.Tensor:
    omega = torch.arange(embed_dim // 2, dtype=torch.float32, device=pos.device)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000 ** omega
    out = torch.einsum("m,d->md", pos.reshape(-1), omega)
    return torch.cat([torch.sin(out), torch.cos(out)], dim=1)


def _sincos_2d_grid(embed_dim: int, grid: torch.Tensor) -> torch.Tensor:
    return torch.cat([_sincos_1d(embed_dim // 2, grid[0]), _sincos_1d(embed_dim // 2, grid[1])], dim=1)


def pos_embed_with_scale(embed_dim: int, grid_size: int, scale: float, modis: bool = False) -> torch.Tensor:
    """``get_2d_sincos_pos_embed_with_scale(..., cls_token=True)``: (1, 1 [+1] + grid², D)."""
    g = torch.arange(grid_size, dtype=torch.float32)
    grid = torch.stack(torch.meshgrid(g, g, indexing="xy"), dim=0)
    grid = torch.einsum("chw,n->cnhw", grid, torch.tensor([scale], dtype=torch.float32))
    _, n, h, w = grid.shape
    pe = _sincos_2d_grid(embed_dim, grid).reshape(n, h * w, embed_dim)
    pe = torch.cat([torch.zeros([n, 1, embed_dim]), pe], dim=1)
    if modis:
        pe = torch.cat([torch.zeros([n, 1, embed_dim]), pe], dim=1)
    return pe


def pos_embed_with_resolution(embed_dim: int, grid_size: int, res: float) -> torch.Tensor:
    """``get_2d_sincos_pos_embed_with_resolution(..., cls_token=True)`` for one modality."""
    size = max(1, int(grid_size * 10 / res))
    g = torch.arange(size, dtype=torch.float32)
    grid = torch.stack(torch.meshgrid(g, g, indexing="xy"), dim=0)
    grid = torch.einsum("chw,n->cnhw", grid, torch.tensor([res]))
    _, n, h, w = grid.shape
    pe = _sincos_2d_grid(embed_dim, grid).reshape(n, h * w, embed_dim)
    return torch.cat([torch.zeros([n, 1, embed_dim]), pe], dim=1)


# --------------------------------------------------------------------------- iRPE (euclidean, contextual, keys)

_RPE_RATIO = 1.9
_ALPHA, _BETA, _GAMMA = 1 * _RPE_RATIO, 2 * _RPE_RATIO, 8 * _RPE_RATIO
_NUM_BUCKETS = 2 * int(_BETA) + 1 + 1          # [-beta, beta] plus one bucket for the skip (class) token


def _piecewise_index(rp: torch.Tensor, alpha: float, beta: float, gamma: float) -> torch.Tensor:
    rp_abs = rp.abs()
    not_mask = ~(rp_abs <= alpha)
    rp_out, rp_abs_out = rp[not_mask], rp_abs[not_mask]
    y_out = (torch.sign(rp_out) * (alpha + torch.log(rp_abs_out / alpha) / math.log(gamma / alpha) *
                                   (beta - alpha)).round().clip(max=beta)).to(torch.long)
    idx = rp.clone().round().to(torch.long)
    idx[not_mask] = y_out
    return idx


def rpe_bucket_ids(height: int, width: int, skip: int) -> torch.Tensor:
    """Euclidean iRPE bucket ids, (skip + h*w, skip + h*w); the skip tokens use the extra bucket."""
    rows = torch.arange(height, dtype=torch.long).view(height, 1).repeat(1, width)
    cols = torch.arange(width, dtype=torch.long).view(1, width).repeat(height, 1)
    pos = torch.stack([rows, cols], 2)
    L = height * width
    diff = pos.view(L, 1, 2) - pos.view(1, L, 2)
    dis = diff.square().sum(2).float().sqrt().round()
    ids = _piecewise_index(dis, _ALPHA, _BETA, _GAMMA) + int(_BETA)
    if skip > 0:
        full = ids.new_empty((skip + L, skip + L))
        full[:skip] = _NUM_BUCKETS - 1
        full[:, :skip] = _NUM_BUCKETS - 1
        full[skip:, skip:] = ids
        ids = full
    return ids


class IRPE(nn.Module):
    """Contextual image relative position encoding on keys with one table shared by all heads."""

    def __init__(self, head_dim: int, n_modalities: int = 1):
        super().__init__()
        self.head_dim, self.n_modalities = head_dim, n_modalities
        self.lookup_table_weight = nn.Parameter(torch.zeros(1, head_dim, _NUM_BUCKETS))
        nn.init.trunc_normal_(self.lookup_table_weight, std=0.02)

    def forward(self, x: torch.Tensor, height: int, width: int, modis: bool = False) -> torch.Tensor:
        """x: (B, H, L, head_dim) -> (B, 1, L, L) bias (broadcast over heads, as in the original)."""
        B, _, L, _ = x.shape
        skip = 1 + int(modis)
        bucket = rpe_bucket_ids(height, width, skip).to(x.device).unsqueeze(0).repeat(B, 1, 1)
        n = self.n_modalities
        bucket = torch.cat([bucket[:, :1, :]] + [bucket[:, 1:, :] for _ in range(n)], dim=1)
        bucket = torch.cat([bucket[:, :, :1]] + [bucket[:, :, 1:] for _ in range(n)], dim=2)
        lq, lk = bucket.shape[1], bucket.shape[2]
        table = torch.matmul(x.transpose(0, 1).reshape(-1, B * L, self.head_dim), self.lookup_table_weight)
        table = table.view(-1, B, L, _NUM_BUCKETS).transpose(0, 1)               # (B, H, L, buckets)
        offset = torch.arange(0, L * _NUM_BUCKETS, _NUM_BUCKETS, dtype=bucket.dtype,
                              device=x.device).view(-1, 1).unsqueeze(0).repeat(B, 1, 1)
        flat = (bucket + offset).flatten(1, 2)
        return torch.gather(table.flatten(2), dim=2, index=flat.unsqueeze(1)).view(B, -1, lq, lk)


# --------------------------------------------------------------------------- transformer blocks


class Mlp(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class Attention(nn.Module):
    """ViT self-attention, explicit softmax path (``flash_attn=False``)."""

    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.num_heads, self.head_dim = num_heads, dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        q, k, v = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4).unbind(0)
        attn = ((q @ k.transpose(-2, -1)) * self.scale).softmax(dim=-1)
        return self.proj((attn @ v).transpose(1, 2).reshape(B, N, C))


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, eps: float = 1e-6):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=eps)
        self.attn = Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim, eps=eps)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class RPEAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.rpe_k = IRPE(dim // num_heads, n_modalities=1)

    def forward(self, x):
        B, N, C = x.shape
        height = int(N ** 0.5)
        q, k, v = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn + self.rpe_k(q, height=height, width=height)
        attn = attn.softmax(dim=-1)
        return self.proj((attn @ v).transpose(1, 2).reshape(B, N, C))


class BlockTransformer(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = RPEAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class TransformerMulti(nn.Module):
    """Spatial encoder: pools the sub-patch tokens of each patch into one token (class token output)."""

    def __init__(self, dim: int, depth: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.predictor_blocks = nn.ModuleList([BlockTransformer(dim, num_heads, mlp_ratio) for _ in range(depth)])
        self.predictor_norm = nn.LayerNorm(dim)

    def forward(self, x, modality: str, scale: int):
        B, N, C = x.shape
        x = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1)
        x = x + pos_embed_with_resolution(C, scale, RESOLUTION[modality]).to(x.device)
        for blk in self.predictor_blocks:
            x = blk(x)
        return self.predictor_norm(x)[:, 0]


class CrossRPEAttentionMulti(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.wk = nn.Linear(dim, dim, bias=True)
        self.wv = nn.Linear(dim, dim, bias=True)
        self.q_learned = nn.Parameter(torch.zeros(1, 1, dim))
        self.rpe_k = IRPE(dim // num_heads, n_modalities=1)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, n_modalities: int = 1, modis: bool = False, scale: int = 1):
        B, N, C = x.shape
        num_patches = N // n_modalities + int((N - int(modis)) % n_modalities > 0) + int(modis)
        pe = pos_embed_with_scale(C, int(num_patches ** 0.5), scale, modis=modis).to(x.device)
        q = (self.q_learned.expand(B, num_patches, -1) + pe.expand(B, -1, -1))
        q = q.reshape(B, num_patches, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.wk(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.wv(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        height = int(num_patches ** 0.5)
        rpe = self.rpe_k(q, height=height, width=height, modis=modis)
        attn = attn + torch.cat([rpe[:, :, :, :(1 + modis)],
                                 rpe[:, :, :, (1 + modis):].repeat(1, 1, 1, n_modalities)], dim=-1)
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, num_patches, C)
        x = torch.cat([x[:, :1], x[:, (1 + modis):]], dim=1)
        return self.proj(x)


class CrossBlockMulti(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, eps: float = 1e-6):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=eps)
        self.attn = CrossRPEAttentionMulti(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim, eps=eps)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))

    def forward(self, x, n_modalities: int = 1, modis: bool = False, scale: int = 1):
        x = self.attn(self.norm1(x), n_modalities=n_modalities, modis=modis, scale=scale)   # no residual (official)
        return x + self.mlp(self.norm2(x))


# --------------------------------------------------------------------------- L-TAE projector


class PositionalEncoder(nn.Module):
    def __init__(self, d: int, T: float = 1000, repeat: int | None = None):
        super().__init__()
        self.repeat = repeat
        self.register_buffer("denom", torch.pow(T, 2 * (torch.arange(0, d).float() // 2) / d), persistent=False)

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        table = positions[:, :, None] / self.denom[None, None, :]
        table[:, :, 0::2] = torch.sin(table[:, :, 0::2])
        table[:, :, 1::2] = torch.cos(table[:, :, 1::2])
        if self.repeat is not None:
            table = torch.cat([table for _ in range(self.repeat)], dim=-1)
        return table


class MultiHeadAttention(nn.Module):
    """L-TAE attention with one learned master query per head."""

    def __init__(self, n_head: int, d_k: int, d_in: int):
        super().__init__()
        self.n_head, self.d_k, self.d_in = n_head, d_k, d_in
        self.Q = nn.Parameter(torch.zeros((n_head, d_k)))
        nn.init.normal_(self.Q, mean=0, std=np.sqrt(2.0 / d_k))
        self.fc1_k = nn.Linear(d_in, n_head * d_k)
        nn.init.normal_(self.fc1_k.weight, mean=0, std=np.sqrt(2.0 / d_k))
        self.temperature = float(np.power(d_k, 0.5))

    def forward(self, v):
        d_k, n_head = self.d_k, self.n_head
        sz_b, seq_len, _ = v.size()
        q = torch.stack([self.Q for _ in range(sz_b)], dim=1).view(-1, d_k)
        k = self.fc1_k(v).view(sz_b, seq_len, n_head, d_k).permute(2, 0, 1, 3).contiguous().view(-1, seq_len, d_k)
        v = torch.stack(v.split(v.shape[-1] // n_head, dim=-1)).view(n_head * sz_b, seq_len, -1)
        attn = torch.matmul(q.unsqueeze(1), k.transpose(1, 2)) / self.temperature
        attn = torch.softmax(attn, dim=2)
        out = torch.matmul(attn, v)
        return out.view(n_head, sz_b, 1, self.d_in // n_head).squeeze(dim=2)


class LTAE2d(nn.Module):
    """Lightweight Temporal Attention Encoder applied to every pixel of an image time series."""

    def __init__(self, in_channels: int, n_head: int, d_k: int, mlp: list[int], mlp_in: list[int], T: float,
                 in_norm: bool):
        super().__init__()
        self.n_head = n_head
        dims = [in_channels, *mlp_in]
        layers: list[nn.Module] = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers.extend([nn.Linear(a, b), nn.GroupNorm(4, b), nn.ReLU(), nn.Dropout(0.0)])
        self.inconv = nn.Sequential(*layers)
        self.d_model = mlp_in[-1]
        self.positional_encoder = PositionalEncoder(self.d_model // n_head, T=T, repeat=n_head)
        self.attention_heads = MultiHeadAttention(n_head=n_head, d_k=d_k, d_in=self.d_model)
        self.in_norm = nn.GroupNorm(num_groups=n_head, num_channels=mlp_in[-1]) if in_norm else nn.Identity()
        self.out_norm = nn.GroupNorm(num_groups=n_head, num_channels=mlp[-1])
        widths = [self.d_model, *mlp]
        layers = []
        for a, b in zip(widths[:-1], widths[1:]):
            layers.extend([nn.Linear(a, b), nn.GroupNorm(4, b), nn.ReLU()])
        self.mlp = nn.Sequential(*layers)

    def forward(self, x, batch_positions):
        sz_b, seq_len, d, h, w = x.shape
        out = x.permute(0, 3, 4, 1, 2).contiguous().view(sz_b * h * w, seq_len, d)
        out = self.inconv(out.view(-1, out.size(-1))).view(out.shape[0], out.shape[1], -1)
        out = self.in_norm(out.permute(0, 2, 1)).permute(0, 2, 1)
        bp = batch_positions.unsqueeze(-1).repeat((1, 1, h)).unsqueeze(-1).repeat((1, 1, 1, w))
        bp = bp.permute(0, 2, 3, 1).contiguous().view(sz_b * h * w, seq_len)
        out = out + self.positional_encoder(bp)
        out = self.attention_heads(out)
        out = out.permute(1, 0, 2).contiguous().view(sz_b * h * w, -1)
        out = self.out_norm(self.mlp(out))
        return out.view(sz_b, h, w, -1).permute(0, 3, 1, 2)


class PatchLTAEMulti(nn.Module):
    def __init__(self, in_channels: int, dim: int, in_norm: bool, T: float = 367, n_head: int = 16, d_k: int = 8):
        super().__init__()
        self.patch_embed = LTAE2d(in_channels, n_head, d_k, mlp=[dim],
                                  mlp_in=[dim // 8, dim // 2, dim, dim * 2, dim], T=T, in_norm=in_norm)
        self.pad_parameter = nn.Parameter(torch.tensor(0.0), requires_grad=False)

    def forward(self, x, dates, scale: int, mask=None):
        if mask is not None:
            x = x.masked_fill(torch.logical_not(mask.bool()), self.pad_parameter)
        x = self.patch_embed(x, dates)
        x = x.unfold(2, scale, scale).unfold(3, scale, scale)
        x = x.flatten(2, 3).flatten(3, 4).permute(0, 2, 3, 1)
        return x.flatten(0, 1)


# --------------------------------------------------------------------------- the encoder


class AnySatEncoder(nn.Module):
    """AnySat feature extractor for Sentinel time-series modalities (``AnyModule.forward_release``)."""

    def __init__(self, size: str = "base", modalities: tuple[str, ...] = ("s2",)):
        super().__init__()
        cfg = SIZES[size]
        dim, depth, heads = cfg["dim"], cfg["depth"], cfg["heads"]
        unknown = set(modalities) - set(MODALITY_CHANNELS)
        if unknown:
            raise ValueError(f"Unsupported AnySat modalities {sorted(unknown)}; choose from {list(MODALITY_CHANNELS)}")
        self.size, self.modalities, self.embedding_size = size, tuple(modalities), dim
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for m in sorted(self.modalities):
            setattr(self, f"projector_{m}", PatchLTAEMulti(MODALITY_CHANNELS[m], dim, in_norm=(m == "s2")))
        self.spatial_encoder = TransformerMulti(dim, depth, heads)
        self.blocks = nn.ModuleList([Block(dim, heads) for _ in range(depth)] + [CrossBlockMulti(dim, heads)])

    def forward(self, x: dict, patch_size: int = 10, output: str = "tile") -> torch.Tensor:
        """x: {"s2": (B, T, 10, H, W) standardised, "s2_dates": (B, T) day of year, ["s2_mask"]: valid=1}.

        ``output="tile"`` -> (B, D); ``output="patch"`` -> (B, H', W', D) with H' = H * 10 / patch_size."""
        if output not in ("tile", "patch"):
            raise ValueError("output must be 'tile' or 'patch'")
        scale = int(patch_size) // 10
        modalities = [m for m in x if not (m.endswith("_dates") or m.endswith("_mask"))]
        for m in modalities:
            if m not in self.modalities:
                raise ValueError(f"Modality {m!r} not built into this encoder ({self.modalities})")
            if x[m].ndim != 5 or x[m].shape[-1] != x[m].shape[-2]:
                raise ValueError(f"{m} must be (B, T, C, H, W) with H == W")
        batch_size = x[modalities[0]].shape[0]
        tokens, pos_embed, N, num_patches, C = [], None, 0, 0, 0
        for m in modalities:
            token = getattr(self, f"projector_{m}")(x[m], x[f"{m}_dates"], scale, x.get(f"{m}_mask"))
            if pos_embed is None:
                B, _, C = token.shape
                N = B // batch_size
                num_patches = int(N ** 0.5)
                pos_embed = pos_embed_with_scale(C, num_patches, scale).to(token.device)
            token = self.spatial_encoder(token, m, scale).view(-1, N, self.embedding_size)
            tokens.append(token + pos_embed[:, 1:, :])
        tokens = torch.cat(tokens, dim=1)
        tokens = torch.cat(((self.cls_token + pos_embed[:, :1, :]).expand(tokens.shape[0], -1, -1), tokens), dim=1)
        for blk in self.blocks[:-1]:
            tokens = blk(tokens)
        tokens = self.blocks[-1](tokens, n_modalities=len(modalities), modis=False, scale=scale)
        if output == "tile":
            return tokens[:, 0, :]
        return tokens[:, 1:, :].view(batch_size, num_patches, num_patches, C)


def load(checkpoint: str | None = None, size: str = "base", modalities: tuple[str, ...] = ("s2",),
         map_location: str = "cpu") -> AnySatEncoder:
    """Encoder with the weights of the official checkpoint (``{"state_dict": ...}``); random weights if None.

    Only the keys of the built modules are taken from the checkpoint; they must all be present (strict)."""
    model = AnySatEncoder(size, modalities)
    if checkpoint is not None:
        state = torch.load(checkpoint, map_location=map_location, weights_only=True)
        state = state.get("state_dict", state)
        own = model.state_dict()
        model.load_state_dict({k: v for k, v in state.items() if k in own}, strict=True)
    return model.eval()


# --------------------------------------------------------------------------- input contract and pixel encoding


def channel_stats(x: np.ndarray, valid: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel mean and standard deviation of (..., C) data over valid, finite observations.

    ``valid`` has the shape of ``x`` without the channel axis. AnySat standardises every dataset with its own
    statistics (``(x - mean) / std``, official README and dataset loaders)."""
    x = np.asarray(x, dtype="float64")
    flat = x.reshape(-1, x.shape[-1])
    ok = np.isfinite(flat).all(axis=1)
    if valid is not None:
        ok &= np.asarray(valid, dtype=bool).reshape(-1)
    if not ok.any():
        raise ValueError("No valid observation to compute channel statistics from")
    v = flat[ok]
    std = v.std(axis=0)
    return v.mean(axis=0).astype("float32"), np.where(std > 0, std, 1.0).astype("float32")


def standardise(x: np.ndarray, mean, std) -> np.ndarray:
    """(x - mean) / std over the last (channel) axis."""
    return ((np.asarray(x, dtype="float32") - np.asarray(mean, dtype="float32")) /
            np.asarray(std, dtype="float32")).astype("float32")


def encode_pixels(model: AnySatEncoder, s2: np.ndarray, valid: np.ndarray, doy, batch_size: int = 256) -> np.ndarray:
    """Pixel time series -> (N, D) AnySat embeddings on CPU; rows without a valid timestep are NaN.

    s2: (N, T, 10) standardised S2 (band order `S2_BANDS`); valid: (N, T) bool; doy: (T,) day of year
    (`day_of_year`). Every pixel is a 1 x 1 tile at 10 m encoded with ``patch_size=10, output="tile"``, as in
    the AnySat baseline of nasaharvest/galileo for CropHarvest pixel time series. Missing timesteps are left
    out: rows are grouped by validity pattern and only their valid timesteps and dates are passed."""
    s2 = np.asarray(s2, dtype="float32")
    valid = np.asarray(valid, dtype=bool) & np.isfinite(s2).all(axis=-1)
    doy = np.asarray(doy, dtype="float32").reshape(-1)
    n, T = valid.shape
    if s2.shape[:2] != (n, T) or len(doy) != T:
        raise ValueError("s2 (N, T, C), valid (N, T) and doy (T,) disagree")
    out = np.full((n, model.embedding_size), np.nan, dtype="float32")
    rows = np.flatnonzero(valid.any(axis=1))
    if not len(rows):
        return out
    patterns, inverse = np.unique(valid[rows], axis=0, return_inverse=True)
    inverse = np.asarray(inverse).ravel()
    with torch.inference_mode():
        for p, pattern in enumerate(patterns):
            members = rows[inverse == p]
            steps = np.flatnonzero(pattern)
            for start in range(0, len(members), batch_size):
                part = members[start:start + batch_size]
                x = torch.from_numpy(np.ascontiguousarray(s2[part][:, steps]))[..., None, None]
                dates = torch.from_numpy(doy[steps]).expand(len(part), -1)
                out[part] = model({"s2": x, "s2_dates": dates}, patch_size=10, output="tile").numpy()
    return out
