"""Feature stack from regular-grid time series of individual scenes (alternative/complement to windows).

For every pixel and selected index, clear observations of all scenes in the season are binned to a
regular grid (default 5 days), gap-filled with the weighted Whittaker smoother and summarised as

    ts        smoothed values every `output_step_days`         ts_NDVI_d150 (day of year)
    phenology season metrics (SOS, EOS, peak, LOS, ...)     phen_NDVI_sos
    harmonic  harmonic regression a0 / amplitude / phase    harm_NDVI_amp1

Sentinel-2 scenes are masked with the same SCL/cloud-probability rules as compositing; Sentinel-1
scenes contribute radar indices (orbit directions pooled; use `orbit` to restrict to one direction).
The output is a regular feature stack with a feature manifest, so sampling, training, ablation and
prediction work unchanged; `merge_stacks` combines it with the window stack.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from datetime import date
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import Resampling

from . import indices
from . import scenes as scene_meta
from . import timeseries as ts
from .features import Feature, load_manifest, manifest_path
from .raster import NODATA, blocks, ensure_parent, float_profile, read_aligned
from .s2 import DEFAULT_SCL_INVALID, clear_mask

PHENOLOGY = ("sos", "eos", "los", "peak_doy", "peak_value", "amplitude", "greenup_rate", "senescence_rate", "integral")


def _doy(d: date) -> int:
    return d.timetuple().tm_yday


def build_timeseries_stack(scene_root: str | Path, reference: str | Path, output: str | Path, start: date, end: date,
                           s2_indices=("NDVI", "NDRE", "NDMI"), s1_indices=("VHVV_dB",), step_days: int = 5,
                           output_step_days: int = 15, lam: float = 50.0, envelope_iterations: int = 0,
                           harmonic_order: int = 2, scl_invalid=DEFAULT_SCL_INVALID, cloud_probability_max=None,
                           orbit: str | None = None, block_size: int = 256) -> Path:
    roots = [scene_root] if isinstance(scene_root, (str, Path)) else list(scene_root)
    metas = sorted((m for r in roots if Path(r).exists() for m in scene_meta.discover(r) if start <= m["date"] <= end),
                   key=lambda m: m["date"])
    s2 = [m for m in metas if m["sensor"] == "S2"] if s2_indices else []
    s1 = [m for m in metas if m["sensor"] == "S1" and (orbit is None or str(m.get("orbit_direction", "")).upper() == orbit.upper())] if s1_indices else []
    if not s2 and not s1:
        raise ValueError("No scenes in the season for the requested indices")
    grid = ts.make_grid(start, end, step_days)
    days = np.array([_doy(d) for d in grid], dtype="float64")
    out_idx = list(range(0, len(grid), max(1, output_step_days // step_days)))
    series = [("S2", i, s2) for i in s2_indices if s2] + [("S1", i, s1) for i in s1_indices if s1]
    features: list[Feature] = []
    for sensor, name, _ in series:
        features += [Feature(f"ts_{name}_d{int(days[k])}", "ts", sensor, None, (), f"Whittaker-smoothed {name}, DOY {int(days[k])}")
                     for k in out_idx]
        features += [Feature(f"phen_{name}_{m}", "phenology", sensor, None, (), f"{m} of smoothed {name}") for m in PHENOLOGY]
        features += [Feature(f"harm_{name}_{c}", "harmonic", sensor, None, (), f"harmonic {c} of {name}")
                     for c in ["a0", "rmse"] + [f"{p}{k}" for k in range(1, harmonic_order + 1) for p in ("amp", "phase")]]
    names = [f.name for f in features]
    for m in s2 + s1:
        need = ("SCL",) + tuple({b for i in s2_indices for b in indices.get(i).inputs}) if m["sensor"] == "S2" \
            else tuple({b for i in s1_indices for b in indices.get(i).inputs})
        missing = [b for b in need if not (m["path"] / f"{b}.tif").exists()]
        if missing:
            raise FileNotFoundError(f"{m['path']} lacks {missing}")
    out = ensure_parent(output)
    with rasterio.open(reference) as ref, rasterio.open(out, "w", **float_profile(ref, count=len(names))) as sink:
        for i, n in enumerate(names, 1):
            sink.set_band_description(i, n)
        handles = {}
        try:
            for w in blocks(ref.width, ref.height, block_size):
                transform, shape = ref.window_transform(w), (int(w.height), int(w.width))
                npx = shape[0] * shape[1]
                values: dict[str, np.ndarray] = {}
                for sensor, name, scenes in series:
                    obs = np.full((npx, len(scenes)), np.nan)
                    ok = np.zeros((npx, len(scenes)), dtype=bool)
                    idx = indices.get(name)
                    for j, m in enumerate(scenes):
                        bands = {}
                        for b in idx.inputs + (("SCL",) if sensor == "S2" else ()):
                            key = (str(m["path"]), b)
                            if key not in handles:
                                handles[key] = rasterio.open(m["path"] / f"{b}.tif")
                            if b == "SCL":
                                bands[b] = read_aligned(handles[key], transform, ref.crs, shape, Resampling.nearest,
                                                        dtype="uint8", fill=0)
                            else:
                                bands[b] = read_aligned(handles[key], transform, ref.crs, shape)
                        v = idx.compute(bands)
                        good = np.isfinite(v)
                        if sensor == "S2":
                            opt = {}
                            for extra in ("CLDPRB", "CLOUDMASK", "SHADOW"):
                                fp = m["path"] / f"{extra}.tif"
                                if fp.exists():
                                    key = (str(m["path"]), extra)
                                    if key not in handles:
                                        handles[key] = rasterio.open(fp)
                                    opt[extra] = read_aligned(handles[key], transform, ref.crs, shape,
                                                              Resampling.nearest, dtype="uint8", fill=255)
                            good &= clear_mask(bands["SCL"], opt.get("CLDPRB"), scl_invalid, cloud_probability_max,
                                               external=opt.get("CLOUDMASK"), shadow=opt.get("SHADOW"))
                        obs[:, j] = v.ravel()
                        ok[:, j] = good.ravel()
                    y, wgt = ts.bin_to_grid([m["date"] for m in scenes], obs, ok, grid)
                    z = ts.whittaker(y, wgt, lam=lam, envelope_iterations=envelope_iterations)
                    for k in out_idx:
                        values[f"ts_{name}_d{int(days[k])}"] = z[:, k]
                    for metric, arr in ts.phenology(z, days).items():
                        values[f"phen_{name}_{metric}"] = arr
                    for c, arr in ts.harmonics(y, wgt, days, harmonic_order).items():
                        values[f"harm_{name}_{c}"] = arr
                for i, n in enumerate(names, 1):
                    v = values[n].reshape(shape)
                    sink.write(np.where(np.isfinite(v), v, NODATA).astype("float32"), i, window=w)
        finally:
            for h in handles.values():
                h.close()
    manifest_path(out).write_text(json.dumps({
        "stack": out.name, "kind": "timeseries", "grid_start": start.isoformat(), "grid_end": end.isoformat(),
        "step_days": step_days, "lambda": lam, "envelope_iterations": envelope_iterations,
        "scenes": {"S2": len(s2), "S1": len(s1)}, "nodata": NODATA,
        "features": [asdict(f) for f in features]}, indent=2), encoding="utf8")
    return out


def merge_stacks(stacks, output: str | Path, block_size: int = 512) -> Path:
    """Concatenate feature stacks on one grid (bands and manifests) into a single stack."""
    handles = [rasterio.open(s) for s in stacks]
    try:
        first = handles[0]
        for h in handles[1:]:
            if h.shape != first.shape or h.crs != first.crs or not h.transform.almost_equals(first.transform):
                raise ValueError(f"{h.name} is not on the grid of {first.name}")
        manifests = [load_manifest(s) for s in stacks]
        feats = [f for m in manifests for f in m]
        names = [f.name for f in feats]
        if len(set(names)) != len(names):
            raise ValueError("Duplicate feature names across stacks")
        out = ensure_parent(output)
        with rasterio.open(out, "w", **float_profile(first, count=len(names))) as sink:
            for i, n in enumerate(names, 1):
                sink.set_band_description(i, n)
            for w in blocks(first.width, first.height, block_size):
                band = 1
                for h in handles:
                    data = h.read(window=w)
                    sink.write(data, list(range(band, band + h.count)), window=w)
                    band += h.count
    finally:
        for h in handles:
            h.close()
    manifest_path(out).write_text(json.dumps({"stack": out.name, "merged_from": [str(s) for s in stacks],
                                              "nodata": NODATA, "features": [asdict(f) for f in feats]}, indent=2),
                                  encoding="utf8")
    return out
