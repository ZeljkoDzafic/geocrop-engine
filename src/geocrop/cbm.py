"""Checks-by-Monitoring style parcel time series, markers and traffic-light outcome.

Operational CAP monitoring (Area Monitoring System, Reg. (EU) 2021/2116 and 2022/1173; Sen4CAP; JRC CbM)
works on parcel-mean time series and *markers* — evidence of agricultural events such as emergence,
peak, harvest or bare soil — combined into a green / yellow / red outcome per parcel. This module:

* `parcel_signatures`  per-parcel, per-acquisition mean/std/count of indices over interior pixels
                       (cloud-screened), written as a long table (CSV or Parquet)
* `markers`            Whittaker-smoothed parcel series -> SOS/peak/EOS, harvest / rapid-senescence date (NDVI drop
                       after peak), bare-soil days, consistency with a crop calendar of the declared crop
* `traffic_light`      combines declaration status, conformal prediction set and markers:
                       green  = classification agrees, unambiguous, markers consistent
                       red    = confident discrepancy (declared crop outside the prediction set) or
                                markers contradict the declared crop's calendar
                       yellow = everything else that has data; grey = insufficient data
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.warp import Resampling

from . import indices
from . import scenes as scene_meta
from . import timeseries as ts
from .raster import blocks, read_aligned
from .s2 import DEFAULT_SCL_INVALID, clear_mask


def parcel_signatures(scene_root: str | Path, parcels: str | Path | gpd.GeoDataFrame, reference: str | Path,
                      output: str | Path, start: date, end: date, index_names=("NDVI", "BSI"),
                      id_field: str = "parcel_id", interior_buffer_px: float = 1, scl_invalid=DEFAULT_SCL_INVALID,
                      block_size: int = 1024) -> pd.DataFrame:
    gdf = gpd.read_file(parcels) if not isinstance(parcels, gpd.GeoDataFrame) else parcels.copy()
    metas = [m for m in scene_meta.discover(scene_root) if m["sensor"] == "S2" and start <= m["date"] <= end]
    if not metas:
        raise ValueError("No Sentinel-2 scenes in the period")
    with rasterio.open(reference) as ref:
        gdf = gdf.to_crs(ref.crs).reset_index(drop=True)
        pixel = abs(ref.transform.a)
        geoms = gdf.geometry.buffer(-interior_buffer_px * pixel) if interior_buffer_px else gdf.geometry
        geoms = [g if not g.is_empty else o for g, o in zip(geoms, gdf.geometry)]
        labels = np.arange(1, len(gdf) + 1)
        tree = gpd.GeoSeries(geoms, crs=gdf.crs).sindex
        n = len(gdf) + 1
        acc = {(j, i): [np.zeros(n), np.zeros(n), np.zeros(n)] for j in range(len(metas)) for i in index_names}
        total = np.zeros(n)
        need = sorted({b for i in index_names for b in indices.get(i).inputs})
        handles = {}
        try:
            for w in blocks(ref.width, ref.height, block_size):
                transform, shape = ref.window_transform(w), (int(w.height), int(w.width))
                bounds = rasterio.windows.bounds(w, ref.transform)
                hits = tree.query(_box(bounds), predicate="intersects")
                if len(hits) == 0:
                    continue
                lab = rasterize(((geoms[k], int(labels[k])) for k in hits), out_shape=shape, transform=transform,
                                fill=0, dtype="int32")
                inside = lab > 0
                if not inside.any():
                    continue
                total += np.bincount(lab[inside], minlength=n)
                for j, m in enumerate(metas):
                    bands = {}
                    for b in need + ["SCL"]:
                        key = (j, b)
                        if key not in handles:
                            handles[key] = rasterio.open(m["path"] / f"{b}.tif")
                        bands[b] = (read_aligned(handles[key], transform, ref.crs, shape, Resampling.nearest,
                                                 dtype="uint8", fill=0) if b == "SCL"
                                    else read_aligned(handles[key], transform, ref.crs, shape))
                    clear = clear_mask(bands["SCL"], None, scl_invalid) & inside
                    for i in index_names:
                        v = indices.get(i).compute(bands)
                        ok = clear & np.isfinite(v)
                        s, ss, c = acc[(j, i)]
                        s += np.bincount(lab[ok], weights=v[ok], minlength=n)
                        ss += np.bincount(lab[ok], weights=v[ok] ** 2, minlength=n)
                        c += np.bincount(lab[ok], minlength=n)
        finally:
            for h in handles.values():
                h.close()
    rows = []
    ids = gdf[id_field].astype(str).to_numpy()
    for (j, i), (s, ss, c) in acc.items():
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = s / c
            std = np.sqrt(np.maximum(ss / c - mean**2, 0))
        for k in range(1, n):
            rows.append({"parcel_id": ids[k - 1], "date": metas[j]["date"].isoformat(), "index": i,
                         "mean": mean[k] if c[k] else np.nan, "std": std[k] if c[k] else np.nan,
                         "n_valid": int(c[k]), "n_pixels": int(total[k])})
    table = pd.DataFrame(rows)
    _write_table(table, output)
    return table


def _box(bounds):
    from shapely.geometry import box
    return box(*bounds)


def _write_table(table: pd.DataFrame, output: str | Path) -> Path:
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix.lower() == ".parquet":
        table.to_parquet(out, index=False)
    else:
        table.to_csv(out, index=False)
    return out


def markers(signatures: pd.DataFrame, start: date, end: date, index: str = "NDVI", step_days: int = 5,
            lam: float = 20.0, min_coverage: float = 0.5, harvest_drop: float = 0.2, harvest_window_days: int = 30,
            bare_soil_ndvi: float = 0.25) -> pd.DataFrame:
    """Per-parcel markers from parcel signatures. Observations covering < min_coverage of the parcel are ignored."""
    sig = signatures[(signatures["index"] == index)].copy()
    sig = sig[(sig.n_pixels > 0) & (sig.n_valid >= min_coverage * sig.n_pixels) & sig["mean"].notna()]
    grid = ts.make_grid(start, end, step_days)
    days = np.array([d.timetuple().tm_yday for d in grid], dtype=float)
    parcels = sorted(signatures.parcel_id.astype(str).unique())
    pos = {p: k for k, p in enumerate(parcels)}
    obs_dates = sorted({date.fromisoformat(d) for d in sig.date})
    dpos = {d.isoformat(): k for k, d in enumerate(obs_dates)}
    values = np.full((len(parcels), len(obs_dates)), np.nan)
    for r in sig.itertuples():
        values[pos[str(r.parcel_id)], dpos[r.date]] = r.mean
    y, w = ts.bin_to_grid(obs_dates, values, np.isfinite(values), grid) if obs_dates else \
        (np.full((len(parcels), len(grid)), np.nan), np.zeros((len(parcels), len(grid))))
    z = ts.whittaker(y, w, lam=lam, min_valid=4)
    phen = ts.phenology(z, days)
    k = max(1, harvest_window_days // step_days)
    harvest = np.full(len(parcels), np.nan)
    pk = np.where(np.isfinite(z).all(axis=1), np.nanargmax(np.nan_to_num(z, nan=-np.inf), axis=1), -1)
    for p in range(len(parcels)):
        if pk[p] < 0:
            continue
        drops = z[p, pk[p]:-k] - z[p, pk[p] + k:] if pk[p] + k < len(days) else np.array([])
        hit = np.flatnonzero(drops > harvest_drop)
        if hit.size:
            harvest[p] = days[pk[p] + hit[0] + k]
    season = (days >= 91) & (days <= 273)   # April - September
    bare = np.where(np.isfinite(z), (z < bare_soil_ndvi) & season[None, :], False).sum(axis=1) * step_days
    out = pd.DataFrame({"parcel_id": parcels, "clear_observations": (w > 0).sum(axis=1),
                        "sos_doy": phen["sos"], "peak_doy": phen["peak_doy"], "peak_value": phen["peak_value"],
                        "eos_doy": phen["eos"], "harvest_doy": harvest, "bare_soil_days": bare})
    out.loc[out.clear_observations < 4, ["sos_doy", "peak_doy", "peak_value", "eos_doy", "harvest_doy"]] = np.nan
    return out


def calendar_consistency(markers_df: pd.DataFrame, declared: pd.Series, calendar: dict) -> pd.Series:
    """True/False/None: do peak and harvest dates fall inside the declared crop's expected ranges?

    calendar: {class: {"peak": [doy_min, doy_max], "harvest": [doy_min, doy_max]}} — regional knowledge,
    to be set by agronomists for the study area and year."""
    result = []
    for (_, m), d in zip(markers_df.iterrows(), declared):
        spec = calendar.get(str(d))
        if spec is None or not np.isfinite(m.peak_doy):
            result.append(None)
            continue
        ok = True
        if "peak" in spec:
            ok &= spec["peak"][0] <= m.peak_doy <= spec["peak"][1]
        if "harvest" in spec and np.isfinite(m.harvest_doy):
            ok &= spec["harvest"][0] <= m.harvest_doy <= spec["harvest"][1]
        result.append(bool(ok))
    return pd.Series(result, index=markers_df.index, dtype=object)


def traffic_light(frame: pd.DataFrame) -> pd.Series:
    """Combine `declaration_status`, optional `cp_status` and optional `calendar_consistent` columns."""
    out = []
    for r in frame.itertuples():
        status = getattr(r, "declaration_status", None)
        cp = getattr(r, "cp_status", None)
        cal = getattr(r, "calendar_consistent", None)
        if status in ("insufficient_resolution", "not_declared", None):
            out.append("grey")
        elif cal is False or cp == "discrepancy" or (status == "discrepancy" and cp is None):
            out.append("red")
        elif status == "agreement" and cp in (None, "agreement") and cal is not False:
            out.append("green")
        else:
            out.append("yellow")
    return pd.Series(out, index=frame.index)
