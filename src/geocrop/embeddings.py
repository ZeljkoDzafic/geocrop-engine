"""Foundation-model embeddings as the optional feature group ``emb`` (gaps G2 and G3).

Two families share one output contract:

* **pre-computed annual embeddings** - AlphaEarth Foundations Satellite Embedding V1 (64-d) and TESSERA
  (128-d). `geocrop.embedding_sources` reads only the tile windows covering the study grid; this module
  dequantises them and regrids them onto the grid.
* **Presto** (Tseng et al., 2023) embeddings computed locally (CPU) from GeoCrop's own monthly
  Sentinel-1/Sentinel-2 composites, per pixel (blockwise) or per parcel (from parcel-mean series).

Output: ``embeddings/<source>_<year>.tif`` on the reference grid, float32, nodata -9999, one band per
embedding dimension with descriptions ``emb_00``, ``emb_01``, ... and a feature manifest
(``<source>_<year>.features.json``, group ``emb``) so that `geocrop.features.attach_embeddings` (or
``geocrop merge-stacks``) can append it to a handcrafted feature stack. Only one embedding source can be part
of a stack at a time (feature names are the band descriptions).
"""
from __future__ import annotations

import json
import warnings
from contextlib import ExitStack
from dataclasses import asdict
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine
from rasterio.warp import Resampling, reproject

from .features import Feature, manifest_path
from .raster import NODATA, blocks, ensure_parent, float_profile, read_aligned, same_grid

SOURCES = ("alphaearth", "tessera", "presto")

SOURCE_INFO: dict[str, dict] = {
    "alphaearth": {
        "dims": 64, "sensor": "AlphaEarth",
        "product": "AlphaEarth Foundations Satellite Embedding V1 (annual)",
        "licence": "CC-BY-4.0",
        "attribution": "The AlphaEarth Foundations Satellite Embedding dataset is produced by Google and "
                       "Google DeepMind.",
        "dequantisation": "((q / 127.5) ** 2) * sign(q); q = -128 is masked",
    },
    "tessera": {
        "dims": 128, "sensor": "TESSERA",
        "product": "TESSERA v1 annual embeddings (0.1 degree tiles)",
        "licence": "CC0-1.0 per the producer's terms of use (see docs/embeddings.md for conflicting statements)",
        "attribution": "TESSERA embeddings, University of Cambridge (Feng et al., 2025, arXiv:2506.20380).",
        "dequantisation": "int8 q * float32 scales (per pixel); non-finite scales are masked",
    },
    "presto": {
        "dims": 128, "sensor": "Presto",
        "product": "Presto encoder embeddings computed from GeoCrop monthly S1/S2 composites",
        "licence": "MIT (Presto code and weights)",
        "attribution": "Presto: Tseng et al. (2023), arXiv:2304.14065, NASA Harvest (MIT).",
        "dequantisation": "none (float32 encoder output)",
    },
}


def band_names(dims: int) -> list[str]:
    return [f"emb_{i:02d}" for i in range(dims)]


def output_path(directory: str | Path, source: str, year: int) -> Path:
    if source not in SOURCES:
        raise ValueError(f"Unknown embedding source {source!r}; choose from {', '.join(SOURCES)}")
    return Path(directory) / f"{source}_{int(year)}.tif"


# --------------------------------------------------------------------------- dequantisation


def dequantise_alphaearth(raw: np.ndarray) -> np.ndarray:
    """AlphaEarth int8 -> float32 in [-1, 1]: divide by 127.5, square, keep the sign; -128 -> NaN."""
    raw = np.asarray(raw)
    v = raw.astype("float32") / 127.5
    out = (v * v * np.sign(v)).astype("float32")
    out[raw == -128] = np.nan
    return out


def dequantise_tessera(quantised: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """TESSERA int8 (H, W, D) x float32 scales (H, W) or (H, W, D) -> float32; non-finite scales -> NaN."""
    scales = np.asarray(scales, dtype="float32")
    scales = np.where(np.isfinite(scales), scales, np.nan)
    if scales.ndim == quantised.ndim - 1:
        scales = scales[..., None]
    return quantised.astype("float32") * scales


# --------------------------------------------------------------------------- regridding


def north_up(data: np.ndarray, transform: Affine) -> tuple[np.ndarray, Affine]:
    """Flip a bottom-up array (positive y pixel size, as in the AlphaEarth COGs) to north-up."""
    if transform.b or transform.d:
        raise ValueError("Rotated source grids are not supported")
    if transform.e < 0:
        return data, transform
    rows = data.shape[-2]
    return data[..., ::-1, :], Affine(transform.a, 0, transform.c, 0, -transform.e, transform.f + transform.e * rows)


def regrid(data: np.ndarray, src_transform: Affine, src_crs, dst_transform: Affine, dst_crs, shape: tuple[int, int],
           resampling: Resampling = Resampling.nearest) -> np.ndarray:
    """Reproject a (bands, rows, cols) float array (NaN = no data) onto a target grid window."""
    data, src_transform = north_up(np.asarray(data, dtype="float32"), src_transform)
    out = np.full((data.shape[0], *shape), np.nan, dtype="float32")
    reproject(np.ascontiguousarray(data), out, src_transform=src_transform, src_crs=src_crs, src_nodata=np.nan,
              dst_transform=dst_transform, dst_crs=dst_crs, dst_nodata=np.nan, resampling=resampling)
    return out


# --------------------------------------------------------------------------- output contract


class EmbeddingWriter:
    """Create ``<source>_<year>.tif`` on a reference grid and its ``emb`` feature manifest."""

    def __init__(self, output: str | Path, reference: str | Path, source: str, year: int, dims: int | None = None,
                 extra: dict | None = None):
        self.output, self.reference, self.source, self.year = ensure_parent(output), Path(reference), source, int(year)
        self.dims = int(dims or SOURCE_INFO[source]["dims"])
        self.extra = dict(extra or {})
        self._sink = None

    def __enter__(self):
        with rasterio.open(self.reference) as ref:
            if ref.crs is None:
                raise ValueError("Reference grid needs a CRS")
            self.crs, self.transform, self.shape = ref.crs, ref.transform, ref.shape
            profile = float_profile(ref, count=self.dims)
        self._sink = rasterio.open(self.output, "w", **profile)
        for i, n in enumerate(band_names(self.dims), 1):
            self._sink.set_band_description(i, n)
        return self

    def write(self, data: np.ndarray, window=None):
        data = np.asarray(data, dtype="float32")
        self._sink.write(np.where(np.isfinite(data), data, NODATA).astype("float32"), window=window)

    def __exit__(self, *exc):
        info = {k: v for k, v in SOURCE_INFO[self.source].items() if k != "dims"}
        self._sink.update_tags(embedding_source=self.source, year=str(self.year), licence=info["licence"],
                               attribution=info["attribution"], dequantisation=info["dequantisation"],
                               **{k: str(v) for k, v in self.extra.items()})
        self._sink.close()
        if exc[0] is None:
            write_manifest(self.output, self.source, self.year, self.dims, self.extra)
        return False


def write_manifest(output: str | Path, source: str, year: int, dims: int, extra: dict | None = None) -> Path:
    info = SOURCE_INFO[source]
    feats = [Feature(n, "emb", info["sensor"], None, (), f"{info['product']} {year}, dimension {i}")
             for i, n in enumerate(band_names(dims))]
    path = manifest_path(output)
    path.write_text(json.dumps({
        "stack": Path(output).name, "nodata": NODATA,
        "embedding": {"source": source, "year": int(year), "dims": dims,
                      **{k: v for k, v in info.items() if k != "dims"}, **(extra or {})},
        "features": [asdict(f) for f in feats]}, indent=2, default=str), encoding="utf8")
    return path


def summary(path: str | Path, sample: int = 200_000, seed: int = 0) -> dict:
    """Value ranges of an embeddings raster (for reports): valid fraction, min/max/mean, mean vector norm."""
    with rasterio.open(path) as ds:
        data = ds.read().astype("float64")
        data[data == ds.nodata] = np.nan
    flat = data.reshape(data.shape[0], -1).T
    valid = np.isfinite(flat).all(axis=1)
    v = flat[valid]
    if len(v) > sample:
        v = v[np.random.default_rng(seed).choice(len(v), sample, replace=False)]
    if not len(v):
        return {"pixels": int(len(flat)), "valid_fraction": 0.0}
    norms = np.linalg.norm(v, axis=1)
    return {"pixels": int(len(flat)), "bands": int(data.shape[0]), "valid_fraction": float(valid.mean()),
            "min": float(v.min()), "max": float(v.max()), "mean": float(v.mean()),
            "band_std_mean": float(v.std(axis=0).mean()),
            "vector_norm_min": float(norms.min()), "vector_norm_mean": float(norms.mean()),
            "vector_norm_max": float(norms.max())}


# --------------------------------------------------------------------------- Presto input contract

#: Presto's normalised band order (official `NORMED_BANDS`): B1, B9 and B10 are not used.
PRESTO_BANDS = ("VV", "VH", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12",
                "temperature_2m", "total_precipitation", "elevation", "slope", "NDVI")
#: Presto band -> GeoCrop composite file stem.
GEOCROP_S2 = {"B2": "B02", "B3": "B03", "B4": "B04", "B5": "B05", "B6": "B06", "B7": "B07", "B8": "B08",
              "B8A": "B8A", "B11": "B11", "B12": "B12"}
PRESTO_S2 = tuple(GEOCROP_S2)
S1_SHIFT, S1_DIV = 25.0, 25.0          # (dB + 25) / 25
PRESTO_MAX_TIMESTEPS = 24               # positional table length; pre-training used 12 monthly steps
#: Groups whose token mask is set together (tokens are per group, see presto_encoder.BANDS_GROUPS_IDX).
_GROUP_SLICES = {"S1": [0, 1], "S2_RGB": [2, 3, 4], "S2_Red_Edge": [5, 6, 7], "S2_NIR_10m": [8],
                 "S2_NIR_20m": [9], "S2_SWIR": [10, 11], "ERA5": [12, 13], "SRTM": [14, 15], "NDVI": [16]}


def presto_inputs(s2: dict[str, np.ndarray] | None, s1: dict[str, np.ndarray] | None) -> tuple[np.ndarray, np.ndarray]:
    """Build Presto's normalised input and missing mask from monthly series.

    s2: GeoCrop band stem (B02 ...) -> (T, N) surface reflectance in [0, 1] (= Presto's DN / 10000);
    s1: VV / VH -> (T, N) linear backscatter (converted to dB, then (dB + 25) / 25).
    Returns x, mask of shape (N, T, 17); mask 1 = missing. A timestep's Sentinel-2 groups and NDVI are masked
    together when any S2 band is missing; ERA5 and SRTM are not produced on-VM and are always masked.
    """
    ref = next(iter((s2 or s1 or {}).values()), None)
    if ref is None:
        raise ValueError("Presto needs Sentinel-2 and/or Sentinel-1 series")
    T, N = np.asarray(ref).shape
    if T > PRESTO_MAX_TIMESTEPS:
        raise ValueError(f"Presto accepts at most {PRESTO_MAX_TIMESTEPS} monthly timesteps")
    x = np.zeros((N, T, len(PRESTO_BANDS)), dtype="float32")
    mask = np.ones_like(x)
    if s1:
        with np.errstate(divide="ignore", invalid="ignore"):
            for i, pol in enumerate(("VV", "VH")):
                lin = np.asarray(s1[pol], dtype="float64")
                ok = np.isfinite(lin) & (lin > 0)
                db = 10 * np.log10(np.where(ok, lin, 1.0))
                x[:, :, i] = ((db + S1_SHIFT) / S1_DIV).T
            both = (np.isfinite(np.asarray(s1["VV"])) & (np.asarray(s1["VV"]) > 0) &
                    np.isfinite(np.asarray(s1["VH"])) & (np.asarray(s1["VH"]) > 0)).T
        x[:, :, :2] = np.where(both[..., None], x[:, :, :2], 0)
        mask[:, :, :2] = (~both)[..., None]
    if s2:
        ok = np.ones((N, T), dtype=bool)
        for j, band in enumerate(PRESTO_S2, start=2):
            v = np.asarray(s2[GEOCROP_S2[band]], dtype="float32").T
            ok &= np.isfinite(v)
            x[:, :, j] = v
        x[:, :, 2:12] = np.where(ok[..., None], x[:, :, 2:12], 0)
        nir, red = x[:, :, 8], x[:, :, 4]
        with np.errstate(divide="ignore", invalid="ignore"):
            x[:, :, 16] = np.where((nir + red) > 0, (nir - red) / (nir + red), 0)
        x[:, :, 16] = np.where(ok, x[:, :, 16], 0)
        mask[:, :, 2:12] = (~ok)[..., None]
        mask[:, :, 16] = ~ok
    return x, mask


def token_mask(mask: np.ndarray) -> np.ndarray:
    """(N, T, 17) band mask -> (N, T * groups + 1) token mask, the unit Presto drops tokens by."""
    groups = [mask[:, :, idx].max(axis=-1) for g, idx in _GROUP_SLICES.items() if g != "SRTM"]
    srtm = mask[:, :1, _GROUP_SLICES["SRTM"]].max(axis=-1)
    return np.concatenate(groups + [srtm], axis=1).astype(bool)


def presto_embed(model, x: np.ndarray, mask: np.ndarray, latlons: np.ndarray, month: int,
                 batch_size: int = 512) -> np.ndarray:
    """Encode (N, T, 17) inputs to (N, D) embeddings on CPU; rows without any observation are NaN.

    Presto drops masked tokens, so a batch must share one token mask: rows are grouped by mask pattern.
    `model` is a `presto_encoder.PrestoEncoder` (or any module with the same forward signature)."""
    import torch

    n = len(x)
    tokens = token_mask(mask)
    eo_tokens = tokens.copy()
    eo_tokens[:, -1] = True
    dims = getattr(model, "embedding_size", 128)
    out = np.full((n, dims), np.nan, dtype="float32")
    observed = ~eo_tokens.all(axis=1)
    if not observed.any():
        return out
    rows = np.flatnonzero(observed)
    _, inverse = np.unique(np.packbits(tokens[rows], axis=1), axis=0, return_inverse=True)
    inverse = np.asarray(inverse).ravel()
    with torch.inference_mode():
        for pattern in np.unique(inverse):
            members = rows[inverse == pattern]
            for start in range(0, len(members), batch_size):
                part = members[start:start + batch_size]
                emb = model(torch.from_numpy(x[part]), torch.from_numpy(mask[part]),
                            torch.from_numpy(latlons[part].astype("float32")), month=int(month))
                out[part] = emb.numpy()
    return out


def load_presto(checkpoint: str | Path | None = None):
    """Presto encoder from a local checkpoint (random weights if None, for tests). Needs torch."""
    try:
        from . import presto_encoder
    except ImportError as exc:     # pragma: no cover - depends on the environment
        raise RuntimeError('Presto embeddings need torch: pip install -e ".[deep]"') from exc
    return presto_encoder.load(None if checkpoint is None else str(checkpoint))


# --------------------------------------------------------------------------- monthly series


def month_names(start: str, months: int = 12) -> list[str]:
    """Consecutive month folder names ``YYYY-MM`` starting at ``start`` (``YYYY-MM``)."""
    y, m = (int(p) for p in str(start).split("-")[:2])
    names = []
    for i in range(int(months)):
        k = m - 1 + i
        names.append(f"{y + k // 12:04d}-{k % 12 + 1:02d}")
    return names


def monthly_windows(start: str, months: int = 12) -> dict[str, tuple[date, date]]:
    out = {}
    for name in month_names(start, months):
        y, m = (int(p) for p in name.split("-"))
        first = date(y, m, 1)
        nxt = date(y + (m == 12), m % 12 + 1, 1)
        out[name] = (first, nxt - timedelta(days=1))
    return out


def monthly_composites(reference: str | Path, out: str | Path, start: str, months: int = 12,
                       s2_scenes: str | Path | None = None, s1_scenes: str | Path | None = None,
                       **s2_options) -> dict[str, dict]:
    """Calendar-month S2 (cloud-masked median) and S1 (linear mean) composites in ``out/<YYYY-MM>/``.

    Months without scenes are skipped; Presto masks them. Returns {month: {"s2": n_scenes, "s1": n_scenes}}."""
    from . import s1 as s1mod
    from . import s2 as s2mod
    from . import scenes

    windows = monthly_windows(start, months)
    report = {}
    s2_by = scenes.assign([s for s in scenes.discover(s2_scenes) if s.get("sensor", "S2") == "S2"], windows) \
        if s2_scenes else {}
    s1_by = scenes.assign([s for s in scenes.discover(s1_scenes) if s.get("sensor") == "S1"], windows) \
        if s1_scenes else {}
    for name in windows:
        a, b = s2_by.get(name, []), s1_by.get(name, [])
        if a:
            s2mod.composite([s["path"] for s in a], Path(out) / name, reference=reference, **s2_options)
        if b:
            s1mod.composite([s["path"] for s in b], Path(out) / name, reference=reference)
        report[name] = {"s2": len(a), "s1": len(b)}
    return report


def _monthly_sources(stack: ExitStack, monthly: Path, names: list[str]) -> list[dict]:
    """Open per-month S2 band and S1 polarisation rasters (VV/VH, or the mean of ASC_/DESC_ orbits)."""
    out = []
    for name in names:
        folder = monthly / name
        src: dict = {"s2": None, "s1": None}
        if all((folder / f"{b}.tif").exists() for b in GEOCROP_S2.values()):
            src["s2"] = {b: stack.enter_context(rasterio.open(folder / f"{b}.tif")) for b in GEOCROP_S2.values()}
        prefixes = [p for p in ("", "ASC_", "DESC_") if all((folder / f"{p}{q}.tif").exists() for q in ("VV", "VH"))]
        if prefixes:
            use = [""] if "" in prefixes else prefixes
            src["s1"] = {q: [stack.enter_context(rasterio.open(folder / f"{p}{q}.tif")) for p in use]
                         for q in ("VV", "VH")}
        out.append(src)
    if not any(s["s2"] or s["s1"] for s in out):
        raise ValueError(f"No monthly S2/S1 composites in {monthly} for {names[0]}..{names[-1]}")
    return out


def _read(ds, ref, window, shape) -> np.ndarray:
    if same_grid(ds, ref):
        v = ds.read(1, window=window).astype("float32")
    else:
        v = read_aligned(ds, ref.window_transform(window), ref.crs, shape)
    if ds.nodata is not None:
        v[v == ds.nodata] = np.nan
    return v


def _block_series(sources, ref, window) -> tuple[dict | None, dict | None]:
    shape = (int(window.height), int(window.width))
    T, n = len(sources), shape[0] * shape[1]
    has_s2, has_s1 = any(s["s2"] for s in sources), any(s["s1"] for s in sources)
    s2 = {b: np.full((T, n), np.nan, dtype="float32") for b in GEOCROP_S2.values()} if has_s2 else None
    s1 = {q: np.full((T, n), np.nan, dtype="float32") for q in ("VV", "VH")} if has_s1 else None
    for t, src in enumerate(sources):
        if src["s2"]:
            for b, ds in src["s2"].items():
                s2[b][t] = _read(ds, ref, window, shape).ravel()
        if src["s1"]:
            for q, handles in src["s1"].items():
                layers = np.stack([_read(ds, ref, window, shape).ravel() for ds in handles])
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    s1[q][t] = np.nanmean(np.where(layers > 0, layers, np.nan), axis=0)
    return s2, s1


def _latlon(transform, crs, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    from pyproj import Transformer

    xs, ys = rasterio.transform.xy(transform, rows, cols, offset="center")
    lon, lat = Transformer.from_crs(crs, "EPSG:4326", always_xy=True).transform(np.asarray(xs), np.asarray(ys))
    return np.column_stack([lat, lon]).astype("float32")


def presto_pixels(monthly: str | Path, reference: str | Path, output: str | Path, start: str, months: int = 12,
                  model=None, checkpoint: str | Path | None = None, block_size: int = 256,
                  batch_size: int = 512, extra: dict | None = None) -> Path:
    """Pixel-level Presto embeddings from monthly composites, block by block on CPU (memory O(block²))."""
    names = month_names(start, months)
    model = model if model is not None else load_presto(checkpoint)
    month0 = int(names[0].split("-")[1]) - 1
    year = int(names[-1].split("-")[0])
    info = {"start": names[0], "months": len(names), "start_month_index": month0, **(extra or {})}
    with ExitStack() as stack, rasterio.open(reference) as ref:
        sources = _monthly_sources(stack, Path(monthly), names)
        info["months_with_s2"] = sum(bool(s["s2"]) for s in sources)
        info["months_with_s1"] = sum(bool(s["s1"]) for s in sources)
        with EmbeddingWriter(output, reference, "presto", year, getattr(model, "embedding_size", 128), info) as sink:
            for w in blocks(ref.width, ref.height, block_size):
                s2, s1 = _block_series(sources, ref, w)
                x, mask = presto_inputs(s2, s1)
                rr, cc = np.mgrid[int(w.row_off):int(w.row_off + w.height), int(w.col_off):int(w.col_off + w.width)]
                emb = presto_embed(model, x, mask, _latlon(ref.transform, ref.crs, rr.ravel(), cc.ravel()), month0,
                                   batch_size)
                sink.write(emb.T.reshape(-1, int(w.height), int(w.width)), window=w)
    return Path(output)


def presto_parcels(monthly: str | Path, reference: str | Path, parcels, output: str | Path, start: str,
                   months: int = 12, id_field: str = "parcel_id", model=None, checkpoint: str | Path | None = None,
                   block_size: int = 512, min_pixels: int = 1):
    """Parcel-level Presto embeddings from parcel-mean monthly series (S2 reflectance mean, S1 linear mean).

    Writes ``output`` (.csv or .parquet) with id, n_pixels, emb_00 ... One Presto sequence per parcel: much
    cheaper than pixel level and less affected by mixed pixels, but blind to within-parcel heterogeneity."""
    import geopandas as gpd
    import pandas as pd
    from rasterio.features import rasterize

    names = month_names(start, months)
    model = model if model is not None else load_presto(checkpoint)
    month0 = int(names[0].split("-")[1]) - 1
    gdf = parcels if isinstance(parcels, gpd.GeoDataFrame) else gpd.read_file(parcels)
    if id_field not in gdf.columns:
        raise ValueError(f"Parcel layer lacks {id_field!r}")
    with ExitStack() as stack, rasterio.open(reference) as ref:
        gdf = gdf.to_crs(ref.crs)
        P, T = len(gdf), len(names)
        sources = _monthly_sources(stack, Path(monthly), names)
        keys = [*GEOCROP_S2.values(), "VV", "VH"]
        sums = {k: np.zeros((T, P + 1)) for k in keys}
        counts = {k: np.zeros((T, P + 1)) for k in keys}
        pixels = np.zeros(P + 1)
        shapes = list(zip(gdf.geometry, range(1, P + 1)))
        for w in blocks(ref.width, ref.height, block_size):
            shape = (int(w.height), int(w.width))
            ids = rasterize(shapes, out_shape=shape, transform=ref.window_transform(w), fill=0, dtype="int32").ravel()
            if not ids.any():
                continue
            pixels += np.bincount(ids, minlength=P + 1)
            s2, s1 = _block_series(sources, ref, w)
            for series in (s2, s1):
                for k, arr in (series or {}).items():
                    for t in range(T):
                        ok = np.isfinite(arr[t]) & (ids > 0)
                        sums[k][t] += np.bincount(ids[ok], weights=arr[t][ok], minlength=P + 1)
                        counts[k][t] += np.bincount(ids[ok], minlength=P + 1)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = {k: (sums[k] / counts[k])[:, 1:].astype("float32") for k in keys}
        has_s2, has_s1 = any(s["s2"] for s in sources), any(s["s1"] for s in sources)
        x, mask = presto_inputs({k: mean[k] for k in GEOCROP_S2.values()} if has_s2 else None,
                                {k: mean[k] for k in ("VV", "VH")} if has_s1 else None)
        pts = gdf.geometry.representative_point().to_crs("EPSG:4326")
        latlons = np.column_stack([pts.y, pts.x]).astype("float32")
    emb = presto_embed(model, x, mask, latlons, month0)
    emb[pixels[1:] < min_pixels] = np.nan
    cols = band_names(emb.shape[1])
    table = pd.concat([pd.DataFrame({id_field: gdf[id_field].to_numpy(), "n_pixels": pixels[1:].astype(int)}),
                       pd.DataFrame(emb, columns=cols)], axis=1)
    out = ensure_parent(output)
    if out.suffix == ".parquet":
        table.to_parquet(out, index=False)
    else:
        table.to_csv(out, index=False)
    return table
