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

from . import attribution, contracts, indices
from . import scenes as scene_meta
from . import temporal_quality as tq
from . import timeseries as ts
from .aoi import AoiMask, creation_options
from .features import Feature, load_manifest, write_manifest
from .raster import NODATA, blocks, ensure_parent, float_profile, read_aligned
from .s2 import DEFAULT_SCL_INVALID, clear_mask

PHENOLOGY = ("sos", "eos", "los", "peak_doy", "peak_value", "amplitude", "greenup_rate", "senescence_rate", "integral")
QUALITY_METRICS = ("observation_count", "observed_bin_fraction", "longest_gap_days", "passes_policy")


def quality_path(output: str | Path) -> Path:
    path = Path(output)
    return path.with_name(path.stem + "_quality.tif")


def support_path(output: str | Path) -> Path:
    """Per-pixel temporal-support raster written next to the stack (`temporal_support=True`)."""
    path = Path(output)
    return path.with_name(path.stem + "_temporal_support.tif")


def _doy(d: date) -> int:
    return d.timetuple().tm_yday


def build_timeseries_stack(scene_root: str | Path, reference: str | Path, output: str | Path, start: date, end: date,
                           s2_indices=("NDVI", "NDRE", "NDMI"), s1_indices=("VHVV_dB",), step_days: int = 5,
                           output_step_days: int = 15, lam: float = 50.0, envelope_iterations: int = 0,
                           harmonic_order: int = 2, scl_invalid=DEFAULT_SCL_INVALID, cloud_probability_max=None,
                           orbit: str | None = None, block_size: int = 256,
                           min_observed_fraction: float = 0, max_gap_days: float | None = None,
                           temporal_support: bool = False, support_tolerance_days: float = 10,
                           aoi: AoiMask | None = None) -> Path:
    """Regular-grid time-series feature stack plus a ``_quality.tif`` QA raster.

    With `temporal_support=True` a third raster ``<stack>_temporal_support.tif`` (+ ``.json`` manifest)
    separates observed from interpolated values per pixel and index (`geocrop.temporal_quality`): clear
    acquisitions, longest gap, season coverage and the distance of SOS/peak/EOS to the nearest clear
    acquisition. The feature stack itself is unchanged.

    With an AOI mask (`geocrop.aoi`) only the blocks that touch the AOI (plus its halo) are reconstructed; the
    rest of every output stays no-data (sparse). Reconstruction is per pixel, so computed blocks are identical
    to an unmasked run. The pixel totals in the quality JSON then count the computed blocks only.
    """
    if not np.isfinite(min_observed_fraction) or not 0 <= min_observed_fraction <= 1:
        raise ValueError("min_observed_fraction must be between zero and one")
    if max_gap_days is not None and (not np.isfinite(max_gap_days) or max_gap_days < 0):
        raise ValueError("max_gap_days must be nonnegative or None")
    if step_days < 1 or output_step_days < 1 or output_step_days % step_days:
        raise ValueError("output_step_days must be a positive multiple of step_days")
    if start.year != end.year:
        raise ValueError("Time-series DOY features require a single calendar year; split cross-year seasons explicitly")
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
    if temporal_support and (not np.isfinite(support_tolerance_days) or support_tolerance_days <= 0):
        raise ValueError("support_tolerance_days must be positive")
    out = ensure_parent(output)
    qa_path = quality_path(out)
    qa_names = [f"qa_{name}_{metric}" for _, name, _ in series for metric in QUALITY_METRICS]
    tq_path = support_path(out)
    tq_names = [f"tq_{name}_{metric}" for _, name, _ in series for metric in tq.PIXEL_METRICS] \
        if temporal_support else []
    tq_sink = None
    totals = {name: {"pixels": 0, "passes_policy": 0, "without_observations": 0} for _, name, _ in series}
    sparse = creation_options(aoi)
    with rasterio.open(reference) as ref, \
            rasterio.open(out, "w", **float_profile(ref, count=len(names), **sparse)) as sink, \
            rasterio.open(qa_path, "w", **float_profile(ref, count=len(qa_names), **sparse)) as qa_sink:
        if aoi is not None:
            aoi.check_grid(ref, "the reference grid")
        for i, n in enumerate(names, 1):
            sink.set_band_description(i, n)
        for i, n in enumerate(qa_names, 1):
            qa_sink.set_band_description(i, n)
        scene_bands = [m["path"] / f"{('B04' if sensor == 'S2' else 'VV')}.tif" for sensor, _, scenes in series
                       for m in scenes]
        credit = attribution.read(scene_bands)
        for target in (sink, qa_sink) if credit else ():
            target.update_tags(**{attribution.TAG: credit, attribution.LICENCE_TAG: attribution.COPERNICUS_LICENCE})
        if temporal_support:
            tq_sink = rasterio.open(tq_path, "w", **float_profile(ref, count=len(tq_names), **sparse))
            for i, n in enumerate(tq_names, 1):
                tq_sink.set_band_description(i, n)
        handles = {}
        try:
            for w in blocks(ref.width, ref.height, block_size):
                if aoi is not None and not aoi.active(w):
                    continue
                transform, shape = ref.window_transform(w), (int(w.height), int(w.width))
                npx = shape[0] * shape[1]
                values: dict[str, np.ndarray] = {}
                qa_values = {}
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
                    support = ts.temporal_support(wgt, step_days)
                    passes = ((wgt > 0).sum(axis=1) >= 4) & (support["observed_bin_fraction"] >= min_observed_fraction)
                    if max_gap_days is not None:
                        passes &= support["longest_gap_days"] <= max_gap_days
                    support["passes_policy"] = passes.astype(float)
                    for metric, arr in support.items():
                        qa_values[f"qa_{name}_{metric}"] = arr
                    totals[name]["pixels"] += npx
                    totals[name]["passes_policy"] += int(passes.sum())
                    totals[name]["without_observations"] += int((support["observation_count"] == 0).sum())
                    z = ts.whittaker(y, wgt, lam=lam, envelope_iterations=envelope_iterations)
                    z[~passes] = np.nan
                    for k in out_idx:
                        values[f"ts_{name}_d{int(days[k])}"] = z[:, k]
                    phen = ts.phenology(z, days)
                    for metric, arr in phen.items():
                        values[f"phen_{name}_{metric}"] = arr
                    if temporal_support:
                        scene_doy = np.array([_doy(m["date"]) for m in scenes], dtype="float64")
                        support = tq.support_metrics(np.where(ok, scene_doy[None, :], np.nan), days,
                                                     support_tolerance_days, phen, days[out_idx])
                        for metric in tq.PIXEL_METRICS:
                            qa_values[f"tq_{name}_{metric}"] = support[metric]
                    for c, arr in ts.harmonics(y, wgt, days, harmonic_order).items():
                        arr[~passes] = np.nan
                        values[f"harm_{name}_{c}"] = arr
                for i, n in enumerate(qa_names, 1):
                    qa_sink.write(qa_values[n].reshape(shape).astype("float32"), i, window=w)
                for i, n in enumerate(names, 1):
                    v = values[n].reshape(shape)
                    sink.write(np.where(np.isfinite(v), v, NODATA).astype("float32"), i, window=w)
                for i, n in enumerate(tq_names, 1):
                    v = qa_values[n].reshape(shape)
                    tq_sink.write(np.where(np.isfinite(v), v, NODATA).astype("float32"), i, window=w)
        finally:
            for h in handles.values():
                h.close()
            if tq_sink is not None:
                tq_sink.close()
    write_manifest(out, {
        "stack": out.name, "kind": "timeseries", "grid_start": start.isoformat(), "grid_end": end.isoformat(),
        "step_days": step_days, "lambda": lam, "envelope_iterations": envelope_iterations,
        "scenes": {"S2": len(s2), "S1": len(s1)}, "nodata": NODATA,
        "features": [asdict(f) for f in features],
        **({"aoi_mask": aoi.path.name, "aoi_halo_px": aoi.halo} if aoi is not None else {})})
    qa_path.with_suffix(".json").write_text(json.dumps({
        "schema": "geocrop-temporal-support/1", "quality_raster": qa_path.name,
        "start": start.isoformat(), "end": end.isoformat(), "grid_bins": len(grid), "step_days": step_days,
        "min_observed_fraction": min_observed_fraction, "max_gap_days": max_gap_days,
        "min_occupied_bins": 4, "bands": qa_names, "series": totals,
        "scope": "Before interpolation; gap = consecutive empty grid bins times step_days, including edges. "
                 "Input observations sharing a bin do not increase coverage. Policy pass does not guarantee "
                 "all phenology/harmonic features are finite or classification is accurate. "
                 "QA bands are separate from classifier features."}, indent=2), encoding="utf8")
    if temporal_support:
        tq_path.with_suffix(".json").write_text(json.dumps(contracts.stamp("temporal_support", {
            "schema": "geocrop-temporal-support-pixels", "raster": tq_path.name, "stack": out.name,
            "start": start.isoformat(), "end": end.isoformat(), "step_days": step_days,
            "output_step_days": output_step_days, "lambda": lam, "tolerance_days": support_tolerance_days,
            "nodata": NODATA, "scenes": {"S2": [m["date"].isoformat() for m in s2],
                                         "S1": [m["date"].isoformat() for m in s1]},
            "bands": [{"band": i, "name": f"tq_{name}_{metric}", "index": name, "metric": metric,
                       "description": tq.PIXEL_DESCRIPTIONS[metric]}
                      for i, (name, metric) in enumerate(((n, m) for _, n, _ in series for m in tq.PIXEL_METRICS), 1)],
            "scope": "Observed = clear acquisition of the pixel (S2: SCL/cloud masks; S1: finite backscatter). "
                     "Everything between observations is Whittaker interpolation. Distances use exact "
                     "acquisition dates. Phenology distances are nodata where the series fails the policy "
                     "or has no detectable season."}), indent=2), encoding="utf8")
    return out


def merge_stacks(stacks, output: str | Path, block_size: int = 512, aoi: AoiMask | None = None) -> Path:
    """Concatenate feature stacks on one grid (bands and manifests) into a single stack.

    With an AOI mask (`geocrop.aoi`) only the blocks that touch the AOI (plus its halo) are copied."""
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
        with rasterio.open(out, "w", **float_profile(first, count=len(names), **creation_options(aoi))) as sink:
            for i, n in enumerate(names, 1):
                sink.set_band_description(i, n)
            attribution.propagate(stacks, sink)
            for w in blocks(first.width, first.height, block_size):
                if aoi is not None and not aoi.active(w):
                    continue
                band = 1
                for h in handles:
                    data = h.read(window=w)
                    sink.write(data, list(range(band, band + h.count)), window=w)
                    band += h.count
    finally:
        for h in handles:
            h.close()
    write_manifest(out, {"stack": out.name, "merged_from": [str(s) for s in stacks],
                         "nodata": NODATA, "features": [asdict(f) for f in feats]})
    return out
