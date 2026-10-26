"""Versioned, automatically executable ACCEPTANCE run of the whole chain on a small real area.

    geocrop acceptance --out outputs/acceptance                 # real Sentinel-2 + Sentinel-1, Semberija 2023
    geocrop acceptance --out outputs/acceptance-ci --offline    # same chain and checks on synthetic scenes

Chain: STAC -> Sentinel-2 L2A (Earth Search) + Sentinel-1 RTC (Planetary Computer) -> scene QA -> window
composites -> feature stack -> OBIA segmentation of the peak composite -> pseudo-reference -> ML training ->
prediction -> parcel analytics on segments -> validation on held-out segments -> export (COG, STAC, GeoParquet)
-> `acceptance_report.json` + `acceptance_report.md`.

No field labels can be published, so training uses a DOCUMENTED PSEUDO-REFERENCE: k-means clusters of the
per-segment NDVI trajectory. These are "pseudo-classes for software acceptance, not crop types"; the
validation score shows that the chain can learn and reproduce a structure present in the data, NOT crop-type
accuracy. Every check states its value, threshold and method; the verdict is PASS only if all checks pass.
"""
from __future__ import annotations

import json
import platform
import resource
import shutil
import sys
import time
import tracemalloc
import warnings
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from shapely.geometry import box

from .raster import NODATA

SCHEMA = "geocrop-acceptance/1"
DEFAULT_BBOX = (19.120, 44.730, 19.145, 44.748)      # ~2.0 x 2.0 km arable land, Semberija (BiH), UTM 34N
DEFAULT_START, DEFAULT_END = "2023-06-01", "2023-09-30"
PSEUDO_NOTE = "pseudo-classes for software acceptance, not crop types"
BOUNDED_INDICES = ("NDVI", "NDRE", "NDMI", "BSI")       # normalised differences: physically within [-1, 1]
SCL_INVALID = (0, 1, 3, 8, 9, 10, 11)
DATA_SOURCES = {
    "sentinel-2": {"endpoint": "https://earth-search.aws.element84.com/v1", "collection": "sentinel-2-l2a",
                   "licence": "Copernicus Sentinel data terms (free, full and open)",
                   "attribution": "Contains modified Copernicus Sentinel data {year}"},
    "sentinel-1": {"endpoint": "https://planetarycomputer.microsoft.com/api/stac/v1", "collection": "sentinel-1-rtc",
                   "licence": "CC-BY-4.0 (Microsoft Planetary Computer; RTC processing by Catalyst)",
                   "attribution": "Contains modified Copernicus Sentinel data {year}; Sentinel-1 RTC from Microsoft "
                                  "Planetary Computer (CC BY 4.0), processed by Catalyst"},
}


class DataUnavailable(RuntimeError):
    """External data could not be obtained (service outage or no usable acquisitions): run is INCOMPLETE."""


@dataclass
class Check:
    id: str
    stage: str
    description: str
    passed: bool
    value: object
    threshold: str
    method: str = ""


@dataclass
class Run:
    out: Path
    checks: list = field(default_factory=list)
    stages: list = field(default_factory=list)
    info: dict = field(default_factory=dict)

    def check(self, id_, stage, description, passed, value, threshold, method=""):
        self.checks.append(Check(id_, stage, description, bool(passed), _jsonable(value), threshold, method))

    @contextmanager
    def stage(self, name):
        tracemalloc.start()
        t0 = time.perf_counter()
        record = {"stage": name, "status": "ok"}
        try:
            yield record
        except BaseException as exc:
            record["status"] = "error"
            record["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            record["seconds"] = round(time.perf_counter() - t0, 2)
            record["python_peak_mb"] = round(tracemalloc.get_traced_memory()[1] / 2**20, 1)
            tracemalloc.stop()
            record["process_peak_rss_mb"] = _peak_rss_mb()
            self.stages.append(record)

    def rel(self, path) -> str:
        try:
            return Path(path).resolve().relative_to(self.out.resolve()).as_posix()
        except ValueError:
            return str(path)


def _peak_rss_mb() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(peak / (2**20 if sys.platform == "darwin" else 2**10), 1)


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else round(float(value), 4)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def windows_for(start: str, end: str) -> dict[str, tuple[date, date]]:
    from .quicklook import monthly_windows
    months = monthly_windows(date.fromisoformat(start), date.fromisoformat(end))
    if len(months) < 2:
        raise ValueError("The acceptance period must span at least two calendar months")
    return {f"P{i}": span for i, span in enumerate(months.values(), 1)}


# --------------------------------------------------------------------------- checks

def _same_grid(path, ref) -> bool:
    with rasterio.open(path) as ds:
        return ds.crs == ref.crs and ds.shape == ref.shape and ds.transform.almost_equals(ref.transform)


def check_grid_identity(run: Run, grid: Path, rasters: list[Path]):
    with rasterio.open(grid) as ref:
        bad = [run.rel(p) for p in rasters if not _same_grid(p, ref)]
        crs = ref.crs.to_string()
    run.check("grid_identity", "all", "Every raster (scenes, composites, stack, predictions, exports) shares the "
              "study grid", not bad and rasters, {"rasters_checked": len(rasters), "mismatches": bad[:10], "crs": crs},
              "0 mismatches", "CRS equality, identical shape, affine transform equal within 1e-9")


def s2_radiometry(scene_dirs) -> dict:
    reds, nirs = [], []
    for d in scene_dirs:
        d = Path(d)
        with rasterio.open(d / "SCL.tif") as s, rasterio.open(d / "B04.tif") as r, rasterio.open(d / "B08.tif") as n:
            scl, red, nir = s.read(1), r.read(1), n.read(1)
        ok = ~np.isin(scl, SCL_INVALID) & (red != NODATA) & (nir != NODATA) & np.isfinite(red) & np.isfinite(nir)
        reds.append(red[ok])
        nirs.append(nir[ok])
    red, nir = np.concatenate(reds), np.concatenate(nirs)
    if not red.size:
        return {"clear_pixels": 0}
    return {"clear_pixels": int(red.size), "negative_red_share": float((red < 0).mean()),
            "negative_nir_share": float((nir < 0).mean()), "red_median": float(np.median(red)),
            "nir_median": float(np.median(nir))}


def _ndvi(folder: Path) -> np.ndarray:
    arr = {}
    for b in ("B04", "B08"):
        with rasterio.open(folder / f"{b}.tif") as ds:
            v = ds.read(1).astype("float64")
        v[(v == NODATA) | ~np.isfinite(v)] = np.nan
        arr[b] = v
    with np.errstate(invalid="ignore", divide="ignore"):
        return (arr["B08"] - arr["B04"]) / (arr["B08"] + arr["B04"])


def coregistration(comp_root: Path, periods, max_shift: int = 3, smooth: int = 5) -> dict:
    """S1/S2 co-registration by spatial cross-correlation of S2 NDVI and S1 VH backscatter (dB).

    Green vegetation (crops, tree lines) raises cross-polarised volume scattering, so NDVI and VH dB are positively
    correlated over arable land. Both window composites are smoothed (NaN-aware moving mean, reduces speckle) and
    Pearson r is computed with the radar field shifted by up to `max_shift` pixels; the surface is averaged over
    windows and its peak refined to sub-pixel precision. Correct co-registration puts the peak at (0, 0); a
    residual offset of k pixels moves it to k.
    """
    from .s1_stac import nan_smooth, shift_surface, surface_peak
    surfaces, per_window = [], {}
    for p in periods:
        folder = comp_root / p
        with rasterio.open(folder / "VH.tif") as ds:
            vh = ds.read(1).astype("float64")
        vh[(vh == NODATA) | ~np.isfinite(vh) | (vh <= 0)] = np.nan
        surf = shift_surface(nan_smooth(_ndvi(folder), smooth), nan_smooth(10 * np.log10(vh), smooth), max_shift)
        surfaces.append(surf)
        per_window[p] = round(float(surf[max_shift, max_shift]), 3)
    peak = surface_peak(np.nanmean(np.stack(surfaces), axis=0))
    return {"windows": list(periods), "best_shift_px": [peak["dy"], peak["dx"]], "r_at_zero_shift": peak["r_zero"],
            "r_at_best_shift": peak["r_peak"], "mean_r_at_max_shift": peak["r_ring"],
            "r_zero_shift_per_window": per_window, "max_shift_px": max_shift}


def nodata_leakage(stack: Path, comp_root: Path, periods) -> dict:
    """Pixels with no-data in a window's composite must be no-data in every feature derived from it (per sensor,
    from the feature manifest), and no valid feature value may be the sentinel or numerically close to it."""
    from .features import load_manifest
    leaks = {}
    with rasterio.open(stack) as ds:
        names = list(ds.descriptions)
        declared = ds.nodata
        data = ds.read()
    nonfinite = int((~np.isfinite(data)).sum())
    near_sentinel = int(((data != NODATA) & (data < -1000)).sum())
    sensor_of = {f.name: (f.sensor, f.periods) for f in load_manifest(stack)}
    missing = {}
    for p in periods:
        for sensor, band in (("S2", "B04"), ("S1", "VV")):
            path = comp_root / p / f"{band}.tif"
            if path.exists():
                with rasterio.open(path) as c:
                    v = c.read(1)
                missing[sensor, p] = (v == NODATA) | ~np.isfinite(v)
    for k, n in enumerate(names):
        sensor, spans = sensor_of[n]
        gaps = [missing[sensor, p] for p in spans if (sensor, p) in missing]
        if not gaps or n.startswith("season_") and n.endswith(("_max", "_min", "_range")):
            continue        # nan-aware seasonal statistics legitimately skip a missing window
        hole = np.logical_or.reduce(gaps)
        count = int((data[k][hole] != NODATA).sum())
        if count:
            leaks[n] = count
    return {"declared_nodata": declared, "window_nodata_leaks": sum(leaks.values()), "leaking_features": leaks,
            "near_sentinel_values": near_sentinel, "nonfinite_values": nonfinite}


def index_ranges(stack: Path) -> dict:
    out = {}
    with rasterio.open(stack) as ds:
        for k, n in enumerate(ds.descriptions, 1):
            if n.split("_")[0].startswith("P") and n.split("_", 1)[-1] in BOUNDED_INDICES:
                v = ds.read(k)
                v = v[(v != NODATA) & np.isfinite(v)]
                if v.size:
                    out[n] = [round(float(v.min()), 4), round(float(v.max()), 4)]
    return out


def probability_checks(pred_dir: Path) -> dict:
    with rasterio.open(pred_dir / "probabilities.tif") as p, rasterio.open(pred_dir / "classes.tif") as c:
        prob, cls = p.read(), c.read(1)
    ok = (prob != NODATA).all(axis=0) & np.isfinite(prob).all(axis=0)
    z = prob[:, ok]
    sums = z.sum(axis=0)
    return {"valid_pixels": int(ok.sum()), "min": float(z.min()), "max": float(z.max()),
            "max_abs_sum_error": float(np.abs(sums - 1).max()),
            "argmax_matches_classes": bool(np.array_equal(z.argmax(axis=0), cls[ok])),
            "nodata_class_outside": bool(np.all(cls[~ok] == 255))}


def cog_checks(paths) -> dict:
    out = {}
    for p in paths:
        with rasterio.open(p) as ds:
            out[Path(p).name] = {"layout": ds.tags(ns="IMAGE_STRUCTURE").get("LAYOUT"),
                                 "tiled": ds.profile.get("tiled", False), "crs": str(ds.crs),
                                 "overviews": ds.overviews(1), "readable": bool(ds.read(1, window=((0, 1), (0, 1))).size)}
    return out


# --------------------------------------------------------------------------- data

def _grid_from_bbox(bbox, out: Path) -> Path:
    from .quicklook import utm_epsg
    from .raster import make_grid
    west, south, east, north = map(float, bbox)
    if not (west < east and south < north):
        raise ValueError("bbox must be west south east north (degrees)")
    epsg = utm_epsg((west + east) / 2, (south + north) / 2)
    aoi = gpd.GeoDataFrame(geometry=[box(west, south, east, north)], crs="EPSG:4326").to_crs(epsg)
    if aoi.area.iloc[0] / 1e6 > 100:
        raise ValueError("The acceptance area must not exceed 100 km2")
    return make_grid(aoi, f"EPSG:{epsg}", 10, out / "grid.tif")


def _real_scenes(run: Run, grid: Path, start: str, end: str, max_items: int) -> tuple[list, list]:
    from . import s1_stac
    from .stac_import import import_season
    with run.stage("s2_import") as rec:
        try:
            s2 = import_season(grid, run.out / "scenes" / "s2", start, end, max_cloud=60, min_clear_fraction=0.5,
                               max_items=max_items)
        except Exception as exc:  # catalogue outage is not an engine failure
            raise DataUnavailable(f"Sentinel-2 STAC service unavailable: {exc}") from exc
        rec["items"] = len(s2)
        rec["status_counts"] = pd.Series([r["status"].split(":")[0] for r in s2]).value_counts().to_dict()
    with run.stage("s1_import") as rec:
        try:
            s1 = s1_stac.import_season(grid, run.out / "scenes" / "s1", start, end, max_items=max_items)
        except s1_stac.ServiceUnavailable as exc:
            raise DataUnavailable(str(exc)) from exc
        rec["items"] = len(s1)
        rec["status_counts"] = pd.Series([r["status"].split(":")[0] for r in s1]).value_counts().to_dict()
    s2_ok = [r for r in s2 if r["status"] in ("imported", "cached")]
    s1_ok = [r for r in s1 if r["status"] in ("imported", "cached")]
    run.info["stac_items"] = {
        "sentinel-2": sorted(r["item"] for r in s2_ok),
        "sentinel-1": sorted(i for r in s1_ok for i in r["item"]),
        "sentinel-2_rejected": {r["item"]: r["status"] for r in s2 if r not in s2_ok},
        "sentinel-1_rejected": {";".join(r["item"]): r["status"] for r in s1 if r not in s1_ok}}
    if not s2_ok or not s1_ok:
        raise DataUnavailable(f"No usable acquisitions (Sentinel-2: {len(s2_ok)}, Sentinel-1: {len(s1_ok)})")
    return [Path(r["path"]) for r in s2_ok], [Path(r["path"]) for r in s1_ok]


def _synthetic_scenes(run: Run, seed: int) -> tuple[Path, list, list]:
    from .demo import generate_scenes
    with run.stage("synthetic_scenes") as rec:
        data = generate_scenes(run.out / "synthetic", parcels_per_side=10, parcel_px=16, seed=seed)
        rec["s2_scenes"], rec["s1_scenes"] = len(data["s2_scenes"]), len(data["s1_scenes"])
    grid = Path(data["reference"])
    run.info["stac_items"] = {"sentinel-2": [], "sentinel-1": [],
                              "note": "offline mode: synthetic scene folders, no STAC items"}
    return grid, [Path(p) for p in data["s2_scenes"]], [Path(p) for p in data["s1_scenes"]]


# --------------------------------------------------------------------------- pseudo-reference

def _segment_table(stack: Path, segments: gpd.GeoDataFrame, names: list[str]) -> pd.DataFrame:
    from rasterio.features import rasterize
    with rasterio.open(stack) as ds:
        labels = rasterize(((g, int(i)) for g, i in zip(segments.geometry, segments.segment_id)),
                           out_shape=ds.shape, transform=ds.transform, fill=0, dtype="int32")
        desc = list(ds.descriptions)
        bands = np.stack([ds.read(desc.index(n) + 1) for n in names])
    ok = (bands != NODATA).all(axis=0) & np.isfinite(bands).all(axis=0) & (labels > 0)
    n = int(labels.max()) + 1
    count = np.bincount(labels[ok], minlength=n)
    table = pd.DataFrame({"segment_id": np.arange(n), "n_valid": count})
    for k, name in enumerate(names):
        sums = np.bincount(labels[ok], weights=bands[k][ok], minlength=n)
        with np.errstate(invalid="ignore", divide="ignore"):
            table[name] = sums / count
    return table[table.segment_id > 0].set_index("segment_id")


def pseudo_reference(stack: Path, segments: gpd.GeoDataFrame, periods, k: int = 4, seed: int = 42,
                     holdout: float = 0.3, min_pixels: int = 9) -> tuple[gpd.GeoDataFrame, dict]:
    """k-means of per-segment mean NDVI trajectories -> pseudo-classes, plus a stratified segment holdout."""
    from sklearn.cluster import KMeans
    from sklearn.model_selection import train_test_split

    names = [f"{p}_NDVI" for p in periods]
    table = _segment_table(stack, segments, names)
    table = table[table.n_valid >= min_pixels].dropna()
    if len(table) < 20:
        raise ValueError(f"Only {len(table)} segments with valid NDVI trajectories; area too small or too cloudy")
    X = table[names].to_numpy()
    for kk in range(k, 1, -1):
        km = KMeans(n_clusters=kk, n_init=10, random_state=seed).fit(X)
        if np.bincount(km.labels_).min() >= 4:
            break
    order = np.argsort(km.cluster_centers_.mean(axis=1))          # stable names: ascending seasonal NDVI
    rank = {int(c): r for r, c in enumerate(order)}
    labels = np.array([f"pseudo_{chr(65 + rank[int(c)])}" for c in km.labels_])
    train_ids, test_ids = train_test_split(table.index.to_numpy(), test_size=holdout, random_state=seed,
                                           stratify=labels)
    frame = segments.set_index("segment_id").loc[table.index].reset_index()
    frame["pseudo_class"] = labels
    frame["split"] = np.where(frame.segment_id.isin(test_ids), "holdout", "train")
    centres = {f"pseudo_{chr(65 + r)}": [round(float(v), 3) for v in km.cluster_centers_[c]]
               for r, c in enumerate(order)}
    info = {"method": f"k-means (k={kk}, n_init=10, seed={seed}) of per-segment mean NDVI per window "
                      f"({', '.join(names)}); classes ordered by mean seasonal NDVI",
            "note": PSEUDO_NOTE, "k": kk, "segments_used": int(len(frame)),
            "segments_excluded": int(len(segments) - len(frame)), "train_segments": int(len(train_ids)),
            "holdout_segments": int(len(test_ids)), "holdout_fraction": holdout,
            "class_counts": pd.Series(labels).value_counts().sort_index().to_dict(),
            "ndvi_centroids": centres}
    return frame, info


# --------------------------------------------------------------------------- main

def run(out: str | Path, bbox=DEFAULT_BBOX, start: str = DEFAULT_START, end: str = DEFAULT_END,
        offline: bool = False, seed: int = 42, k: int = 4, max_items: int = 200, coregister_s1: bool = True,
        log=print) -> dict:
    """Execute the acceptance chain and write acceptance_report.json/.md into `out`. Returns the report."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    acc = Run(out)
    started = datetime.now(timezone.utc)
    t0 = time.perf_counter()
    windows = windows_for(start, end)
    acc.info.update({"mode": "offline (synthetic)" if offline else "network (real Sentinel-1/2)",
                     "period": [start, end], "seed": seed, "s1_coregistration": coregister_s1,
                     "windows": {p: [a.isoformat(), b.isoformat()] for p, (a, b) in windows.items()}})
    error = None
    verdict = None
    try:
        _chain(acc, bbox, start, end, offline, seed, k, max_items, windows, coregister_s1, log)
    except DataUnavailable as exc:
        error, verdict = f"{type(exc).__name__}: {exc}", "INCOMPLETE"
    except Exception as exc:  # engine failure: recorded, verdict FAIL
        import traceback
        error = f"{type(exc).__name__}: {exc}"
        acc.info["traceback"] = traceback.format_exc(limit=6)
    if verdict is None:
        verdict = "PASS" if acc.checks and error is None and all(c.passed for c in acc.checks) else "FAIL"
    report = _report(acc, verdict, error, started, time.perf_counter() - t0, offline)
    (out / "acceptance_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str),
                                                encoding="utf8")
    (out / "acceptance_report.md").write_text(render_markdown(report), encoding="utf8")
    log(f"acceptance: {verdict} ({sum(c['passed'] for c in report['checks'])}/{len(report['checks'])} checks)")
    return report


def _chain(acc: Run, bbox, start, end, offline, seed, k, max_items, windows, coregister_s1, log):  # noqa: C901 - linear chain
    from . import export, features, model, parcels, s1, s2, scenes, segmentation, validation
    from .provenance import run_manifest
    from .s1_stac import plausibility
    from .samples import extract

    out = acc.out
    periods = list(windows)
    # ---------------------------------------------------------------- data
    if offline:
        grid, s2_dirs, s1_dirs = _synthetic_scenes(acc, seed)
    else:
        with acc.stage("grid"):
            grid = _grid_from_bbox(bbox, out)
        s2_dirs, s1_dirs = _real_scenes(acc, grid, start, end, max_items)
    with rasterio.open(grid) as g:
        from rasterio.warp import transform_bounds
        acc.info["area"] = {"bbox_wgs84": [round(v, 5) for v in transform_bounds(g.crs, "EPSG:4326", *g.bounds)],
                            "crs": g.crs.to_string(), "shape": list(g.shape), "pixel_m": abs(g.transform.a),
                            "area_km2": round(g.width * g.height * abs(g.transform.a * g.transform.e) / 1e6, 3)}
    log(f"acceptance: {len(s2_dirs)} Sentinel-2 and {len(s1_dirs)} Sentinel-1 acquisitions")
    # ---------------------------------------------------------------- scene QA
    with acc.stage("qa") as rec:
        rad = s2_radiometry(s2_dirs)
        rec["s2"] = rad
        acc.check("s2_negative_reflectance", "qa", "Share of negative red reflectance in clear pixels (double "
                  "BOA offset would give ~70 %)", rad.get("negative_red_share", 1) <= 0.02,
                  rad.get("negative_red_share"), "<= 0.02", "SCL-clear pixels of all imported scenes, B04 < 0")
        acc.check("s2_red_nir_medians", "qa", "Median red and NIR reflectance of clear pixels are physically "
                  "plausible for summer arable land",
                  0 < rad.get("red_median", -1) < 0.2 and 0.1 < rad.get("nir_median", -1) < 0.6,
                  {"red_median": rad.get("red_median"), "nir_median": rad.get("nir_median")},
                  "0 < red < 0.2 and 0.1 < NIR < 0.6")
        stats = [plausibility(d) for d in s1_dirs]
        rec["s1_scenes"] = len(stats)
        vv_med = [s["vv_db_p1_p50_p99"][1] for s in stats if "vv_db_p1_p50_p99" in s]
        vh_med = [s["vh_db_p1_p50_p99"][1] for s in stats if "vh_db_p1_p50_p99" in s]
        lin = min(s.get("linear_in_1e-4_1", 0) for s in stats)
        vv_gt = min(s.get("vv_gt_vh", 0) for s in stats)
        acc.check("s1_linear_range", "qa", "Linear gamma0 values within 1e-4..1 (worst scene)", lin >= 0.95,
                  lin, ">= 0.95 of valid pixels per scene", "VV and VH pooled, per scene")
        acc.check("s1_db_ranges", "qa", "Median backscatter of every scene in the expected dB range",
                  bool(vv_med) and all(-20 <= v <= -3 for v in vv_med) and all(-28 <= v <= -10 for v in vh_med),
                  {"vv_median_db": [min(vv_med, default=None), max(vv_med, default=None)],
                   "vh_median_db": [min(vh_med, default=None), max(vh_med, default=None)]},
                  "VV median in [-20, -3] dB, VH median in [-28, -10] dB",
                  "10 log10 of linear gamma0, valid pixels")
        acc.check("s1_vv_gt_vh", "qa", "Co-polarised exceeds cross-polarised backscatter (worst scene)",
                  vv_gt >= 0.9, vv_gt, ">= 0.9 of pixels per scene", "pixelwise VV > VH")
    # ---------------------------------------------------------------- composites
    comp = out / "composites"
    for stale in (comp, out / "s1_uncorrected"):     # windows of an earlier run must not enter the stack
        if stale.exists():
            shutil.rmtree(stale)
    with acc.stage("s2_composites") as rec:
        s2_meta = [scenes.read(d) for d in s2_dirs]
        s1_meta = [scenes.read(d) for d in s1_dirs]
        g2, g1 = scenes.assign(s2_meta, windows), scenes.assign(s1_meta, windows)
        rec["s2_per_window"] = {p: len(v) for p, v in g2.items()}
        rec["s1_per_window"] = {p: len(v) for p, v in g1.items()}
        missing = [p for p in periods if not g2[p] or not g1[p]]
        if missing:
            acc.check("window_acquisitions", "composites", "Every window has Sentinel-2 and Sentinel-1 acquisitions",
                      False, {"s2": rec["s2_per_window"], "s1": rec["s1_per_window"]}, ">= 1 S2 and >= 1 S1 per window")
            raise DataUnavailable(f"Windows without acquisitions: {missing}")
        acc.check("window_acquisitions", "composites", "Every window has Sentinel-2 and Sentinel-1 acquisitions",
                  True, {"s2": rec["s2_per_window"], "s1": rec["s1_per_window"]}, ">= 1 S2 and >= 1 S1 per window")
        for p in periods:
            s2.composite([m["path"] for m in g2[p]], comp / p, reference=grid, mask_dilation_px=1,
                         temporal_screening=len(g2[p]) >= 3)
    calib, verify = periods[: len(periods) // 2], periods[len(periods) // 2:]
    s1_used = s1_dirs
    with acc.stage("s1_coregistration") as rec:
        rec.update({"calibration_windows": calib, "verification_windows": verify, "enabled": coregister_s1})
        if coregister_s1:
            from .s1_stac import apply_offsets, estimate_offsets
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                ref_ndvi = np.nanmean(np.stack([_ndvi(comp / p) for p in calib]), axis=0)
            offsets = estimate_offsets([m["path"] for p in calib for m in g1[p]], ref_ndvi)
            rec["offsets_px"] = offsets
            acc.info["s1_geolocation_offsets"] = offsets
            target = out / "scenes" / "s1_coregistered"
            if target.exists():
                shutil.rmtree(target)
            s1_used = apply_offsets(s1_dirs, offsets, target, min_offset_px=0.5)
            for o in offsets.values():
                o["applied"] = bool(o["accepted"] and np.hypot(o["dy"], o["dx"]) >= 0.5)
            g1 = scenes.assign([scenes.read(d) for d in s1_used], windows)
    with acc.stage("s1_composites") as rec:
        coverage = {}
        for p in periods:
            s1.composite([m["path"] for m in g1[p]], comp / p, reference=grid)
            cov = {}
            for b, key in (("B04", "s2"), ("VV", "s1")):
                with rasterio.open(comp / p / f"{b}.tif") as ds:
                    v = ds.read(1)
                cov[key] = float(((v != NODATA) & np.isfinite(v)).mean())
            coverage[p] = cov
        rec["valid_coverage"] = coverage
        worst = min(min(c.values()) for c in coverage.values())
        acc.check("valid_coverage", "composites", "Valid-pixel coverage of every window composite (S2 after cloud "
                  "screening, S1 after co-registration)", worst >= 0.9, coverage, ">= 0.90 per window and sensor",
                  "fraction of grid pixels with a composite value")
        if coregister_s1:   # uncorrected composites of the verification windows, for the before/after diagnostic
            raw = scenes.assign([scenes.read(d) for d in s1_dirs], windows)
            for p in verify:
                for b in ("B04", "B08"):
                    (out / "s1_uncorrected" / p).mkdir(parents=True, exist_ok=True)
                    shutil.copy(comp / p / f"{b}.tif", out / "s1_uncorrected" / p / f"{b}.tif")
                s1.composite([m["path"] for m in raw[p]], out / "s1_uncorrected" / p, reference=grid)
    with acc.stage("coregistration_check") as rec:
        co = coregistration(comp, verify)
        rec["corrected" if coregister_s1 else "uncorrected"] = co
        value = {"verification_windows": verify, "residual_shift_px": co["best_shift_px"],
                 "r_at_zero_shift": co["r_at_zero_shift"], "r_at_best_shift": co["r_at_best_shift"],
                 "correction_applied": coregister_s1}
        if coregister_s1:
            before = coregistration(out / "s1_uncorrected", verify)
            rec["uncorrected"] = before
            value["shift_without_correction_px"] = before["best_shift_px"]
            value["offsets_per_geometry_px"] = {k: [v["dy"], v["dx"]] for k, v in offsets.items()}
        acc.check("s1_s2_coregistration", "coregistration", "Sentinel-1 and Sentinel-2 composites are "
                  "co-registered (windows independent of the offset estimation)",
                  max(map(abs, co["best_shift_px"])) <= 1 and co["r_at_zero_shift"] >= 0.2, value,
                  "residual shift within +/-1 px and r(0,0) >= 0.2",
                  "Pearson r between S2 NDVI and S1 VH (dB), 5x5 NaN-aware smoothing, radar shifted by -3..3 px, "
                  "surface averaged over the verification windows, sub-pixel peak; S1 offsets per orbit geometry "
                  "are estimated only from the calibration windows (" + ", ".join(calib) + ")")
    # ---------------------------------------------------------------- features
    stack = out / "features" / "stack.tif"
    with acc.stage("features") as rec:
        features.build_stack(comp, stack)
        mf = features.load_manifest(stack)
        rec["features"] = len(mf)
        rec["groups"] = pd.Series([f.group for f in mf]).value_counts().to_dict()
        leak = nodata_leakage(stack, comp, periods)
        acc.check("nodata_leakage", "features", "No no-data sentinel leaks into features",
                  leak["window_nodata_leaks"] == 0 and leak["near_sentinel_values"] == 0
                  and leak["nonfinite_values"] == 0 and leak["declared_nodata"] == NODATA, leak,
                  "0 leaks, 0 near-sentinel values, 0 NaN/inf, nodata declared -9999",
                  "composite no-data of a sensor and window -> no-data in every feature derived from it (feature "
                  "manifest; nan-aware seasonal max/min/range excepted); values < -1000 other than the sentinel")
        ranges = index_ranges(stack)
        bad = {n: r for n, r in ranges.items() if r[0] < -1 or r[1] > 1}
        acc.check("index_ranges", "features", "Normalised-difference indices within [-1, 1]",
                  bool(ranges) and not bad, {"checked": len(ranges), "out_of_range": bad,
                                             "ndvi": {n: r for n, r in ranges.items() if n.endswith("_NDVI")}},
                  "all in [-1, 1]", f"min/max of {', '.join(BOUNDED_INDICES)} per window")
        with rasterio.open(stack) as ds:
            ok = (ds.read() != NODATA).all(axis=0).mean()
        rec["valid_fraction"] = float(ok)
        acc.check("stack_s1_s2", "features", "Feature stack holds Sentinel-2 and Sentinel-1 features",
                  {"S1", "S2"} <= {f.sensor for f in mf}, sorted({f.sensor for f in mf}), "S1 and S2 present")
    # ---------------------------------------------------------------- segmentation
    with acc.stage("segmentation") as rec:
        with rasterio.open(stack) as ds:
            names = list(ds.descriptions)
            medians = {}
            for p in periods:
                v = ds.read(names.index(f"{p}_NDVI") + 1)
                medians[p] = float(np.median(v[v != NODATA]))
        peak = max(medians, key=medians.get)
        seg = segmentation.obia(stack, out / "segmentation" / "segments.gpkg",
                                bands=[f"{peak}_B04", f"{peak}_B08", f"{peak}_B11", f"{peak}_NDVI"],
                                segments_per_km2=50)
        rec.update({"peak_window": peak, "median_ndvi_per_window": medians, **{k_: v for k_, v in seg.items()
                                                                              if k_ != "path"}})
        segments = gpd.read_file(seg["path"])
        acc.check("segments_valid", "segmentation", "OBIA segments are valid, non-empty, unique polygons",
                  len(segments) >= 20 and segments.is_valid.all() and segments.segment_id.is_unique,
                  {"segments": len(segments), "invalid": int((~segments.is_valid).sum())}, ">= 20 valid segments")
    # ---------------------------------------------------------------- pseudo-reference
    with acc.stage("pseudo_reference") as rec:
        ref, info = pseudo_reference(stack, segments, periods, k=k, seed=seed)
        rec.update(info)
        acc.info["pseudo_reference"] = info
        ref_dir = out / "reference"
        ref_dir.mkdir(parents=True, exist_ok=True)
        train_poly = ref[ref.split == "train"].assign(polygon_id=lambda d: "S" + d.segment_id.astype(str),
                                                      class_name=lambda d: d.pseudo_class)
        train_path = ref_dir / "pseudo_training.gpkg"
        train_path.unlink(missing_ok=True)
        train_poly[["polygon_id", "class_name", "geometry"]].to_file(train_path, driver="GPKG")
    # ---------------------------------------------------------------- training
    params = {"n_estimators": 100, "max_features": "sqrt", "class_weight": "balanced_subsample", "n_jobs": -1}
    with acc.stage("training") as rec:
        samples = extract(stack, train_path, max_per_polygon=50, seed=seed)
        samples.save(out / "features" / "samples.npz")
        rep = model.train(samples, out / "model", params=params, n_splits=3, spatial_block_m=500, seed=seed,
                          bootstrap=100)
        rec.update({"samples": rep["samples"], "polygons": rep["polygons"], "features": rep["features"],
                    "cv_macro_f1": rep["cross_validation"]["macro_f1"], "model_sha256": rep["model_sha256"]})
    with acc.stage("prediction") as rec:
        pred_dir = out / "prediction"
        model.predict(stack, out / "model" / "model.joblib", pred_dir)
        pc = probability_checks(pred_dir)
        rec.update(pc)
        acc.check("probabilities", "prediction", "Class probabilities lie in [0, 1] and sum to 1",
                  pc["valid_pixels"] > 0 and pc["min"] >= 0 and pc["max"] <= 1 and pc["max_abs_sum_error"] <= 1e-4
                  and pc["argmax_matches_classes"] and pc["nodata_class_outside"], pc,
                  "min >= 0, max <= 1, |sum - 1| <= 1e-4, classes = argmax, 255 outside valid pixels")
    with acc.stage("reproducibility") as rec:
        model.train(samples, out / "model_repeat", params=params, n_splits=3, spatial_block_m=500, seed=seed,
                    bootstrap=100)
        model.predict(stack, out / "model_repeat" / "model.joblib", out / "prediction_repeat")
        same = {}
        for name in ("classes.tif", "probabilities.tif"):
            with rasterio.open(pred_dir / name) as a, rasterio.open(out / "prediction_repeat" / name) as b:
                same[name] = bool(np.array_equal(a.read(), b.read()))
        rec["identical"] = same
        acc.check("reproducibility", "reproducibility", "Training twice with the same seed gives identical "
                  "predictions", all(same.values()), same, "bit-identical classes and probabilities",
                  "independent second training + prediction run")
    # ---------------------------------------------------------------- parcels
    with acc.stage("parcels") as rec:
        parcels_in = ref.assign(parcel_id=lambda d: "S" + d.segment_id.astype(str),
                                declared_crop=lambda d: d.pseudo_class)[["parcel_id", "declared_crop", "split",
                                                                         "geometry"]]
        ps = parcels.aggregate(pred_dir / "probabilities.tif", parcels_in, out / "parcels" / "parcels.gpkg",
                               strategy="mean_probability", conformal=out / "model" / "conformal.json", alpha=0.1,
                               aoa=pred_dir / "aoa.tif")
        rec.update({k_: ps[k_] for k_ in ("parcels", "decision_counts", "declaration_status_counts",
                                           "traffic_light_counts")})
        table = gpd.read_file(out / "parcels" / "parcels.gpkg")
        classes = json.loads((pred_dir / "class_legend.json").read_text(encoding="utf8")).values()
        required = ["parcel_id", "declared_class", "predicted_class", "decision", "decision_reasons",
                    "declaration_status", "traffic_light", "n_valid", "confidence", "margin", "entropy",
                    "prediction_set", "cp_status", "aoa_fraction", "area_m2"] + [f"p_{c}" for c in classes]
        missing_cols = [c for c in required if c not in table]
        with rasterio.open(grid) as g:
            same_crs = table.crs == g.crs
        acc.check("parcel_schema", "parcels", "Parcel output schema, one row per segment, CRS of the grid",
                  not missing_cols and len(table) == len(parcels_in) and table.parcel_id.is_unique and same_crs,
                  {"rows": len(table), "expected_rows": len(parcels_in), "missing_columns": missing_cols,
                   "crs": str(table.crs)}, "all required columns, row count equal, unique ids, grid CRS",
                  "required: " + ", ".join(required[:15]) + ", p_<class>")
    # ---------------------------------------------------------------- validation
    with acc.stage("validation") as rec:
        hold = table[table.split == "holdout"]
        decided = hold[hold.predicted_class.notna()]
        from .metrics import summary
        seg_acc = summary(decided.declared_class, decided.predicted_class)
        pts = gpd.GeoDataFrame({"class_name": ref[ref.split == "holdout"].pseudo_class.to_numpy(),
                                "observed_at": f"{start[:4]}-07-15T10:00:00Z"},
                               geometry=ref[ref.split == "holdout"].geometry.representative_point().to_numpy(),
                               crs=ref.crs)
        pv = validation.evaluate_points(pred_dir, pts, out / "validation", year=int(start[:4]), training=train_path,
                                        bootstrap=200, seed=seed)
        rec["segment_level"] = {k_: seg_acc[k_] for k_ in ("n", "overall_accuracy", "kappa", "macro_f1")}
        rec["point_level"] = {k_: pv["accuracy"][k_] for k_ in ("n", "overall_accuracy", "kappa", "macro_f1")}
        rec["holdout_segments"], rec["holdout_decided"] = int(len(hold)), int(len(decided))
        acc.info["validation"] = {"segment_level": rec["segment_level"], "point_level": rec["point_level"],
                                  "note": "Held-out PSEUDO-labelled segments (not used in training). Measures "
                                          "recovery of k-means pseudo-classes, NOT crop-type accuracy."}
        acc.check("holdout_recovery", "validation", "Held-out pseudo-labelled segments are recovered by the "
                  "trained chain", seg_acc["overall_accuracy"] >= 0.7 and seg_acc["macro_f1"] >= 0.6
                  and len(decided) >= 0.9 * len(hold),
                  {"overall_accuracy": seg_acc["overall_accuracy"], "macro_f1": seg_acc["macro_f1"],
                   "decided": len(decided), "holdout": len(hold),
                   "point_overall_accuracy": pv["accuracy"]["overall_accuracy"]},
                  "OA >= 0.70, macro-F1 >= 0.60, >= 90 % of held-out segments decided",
                  "parcel-level mean-probability class vs pseudo-class on segments excluded from training; "
                  "point check via geocrop.validation at one interior point per held-out segment")
    # ---------------------------------------------------------------- export
    with acc.stage("export") as rec:
        exp = out / "export"
        if exp.exists():
            shutil.rmtree(exp)
        exp.mkdir(parents=True)
        export.to_cog(pred_dir / "classes.tif", exp / "classes.tif", "nearest")
        export.to_cog(pred_dir / "probabilities.tif", exp / "probabilities.tif", "average")
        export.to_cog(pred_dir / "uncertainty.tif", exp / "uncertainty.tif", "average")
        shutil.copy(pred_dir / "class_legend.json", exp / "class_legend.json")
        item_id = f"geocrop-acceptance-{start[:4]}-{'offline' if offline else 'semberija'}"
        item = export.stac_item(exp, item_id, start, end, output=exp / "item.json")
        cogs = cog_checks([exp / "classes.tif", exp / "probabilities.tif", exp / "uncertainty.tif"])
        rec["cogs"] = cogs
        hrefs_ok = all((exp / a["href"]).is_file() for a in item["assets"].values())
        stac_ok = (item["stac_version"] == "1.0.0" and item["properties"]["start_datetime"].startswith(start)
                   and item["properties"]["proj:epsg"] == int(acc.info["area"]["crs"].split(":")[1])
                   and len(item["assets"]["classes"]["classification:classes"]) == len(classes) and hrefs_ok)
        pystac_ok = None
        try:
            import pystac
            pystac_ok = pystac.Item.from_dict(item).id == item_id
        except ImportError:
            pass
        acc.check("export_cog", "export", "Exported rasters are readable Cloud-Optimised GeoTIFFs",
                  all(c["layout"] == "COG" and c["readable"] for c in cogs.values()), cogs,
                  "IMAGE_STRUCTURE LAYOUT=COG and readable", "GDAL COG driver metadata")
        acc.check("export_stac", "export", "STAC Item has required fields and resolvable assets",
                  stac_ok and pystac_ok is not False,
                  {"assets": sorted(item["assets"]), "hrefs_resolve": hrefs_ok, "pystac_parse": pystac_ok,
                   "proj:epsg": item["properties"]["proj:epsg"]},
                  "stac_version 1.0.0, period, proj:epsg, classification classes, all hrefs exist")
        try:
            import pyarrow  # noqa: F401
        except ImportError:
            rec["geoparquet"] = "skipped: pyarrow not installed"
        else:
            pq = export.to_geoparquet(out / "parcels" / "parcels.gpkg", exp / "parcels.parquet")
            back = gpd.read_parquet(pq)
            import pyarrow.parquet as papq
            geo = b"geo" in (papq.read_schema(pq).metadata or {})
            acc.check("export_geoparquet", "export", "GeoParquet parcels round-trip", len(back) == len(table) and geo
                      and back.crs == table.crs, {"rows": len(back), "geo_metadata": geo}, "rows equal, geo metadata")
        rasters = ([d / f for d in s2_dirs for f in ("B02.tif", "B04.tif", "B08.tif", "B11.tif", "SCL.tif")]
                   + [d / f"{p}.tif" for d in {*s1_dirs, *s1_used} for p in ("VV", "VH")]
                   + sorted(comp.rglob("*.tif")) + [stack]
                   + [pred_dir / f for f in ("classes.tif", "probabilities.tif", "uncertainty.tif", "aoa.tif")]
                   + sorted(exp.glob("*.tif")))
        check_grid_identity(acc, grid, rasters)
    # ---------------------------------------------------------------- manifest
    with acc.stage("manifest") as rec:
        hashed = [stack, out / "features" / "samples.npz", out / "model" / "model.joblib",
                  pred_dir / "classes.tif", pred_dir / "probabilities.tif", out / "parcels" / "parcels.gpkg",
                  exp / "item.json", train_path, grid]
        notes = {"acceptance": True, "mode": acc.info["mode"], "bbox": list(bbox), "period": [start, end],
                 "seed": seed, "windows": acc.info["windows"], "stac_items": acc.info["stac_items"]}
        path = run_manifest(out, None, hashed, notes=notes)
        body = json.loads(path.read_text(encoding="utf8"))
        acc.info["manifest"] = {"path": acc.rel(path), "git_commit": body.get("git_commit"),
                                "git_dirty": body.get("git_dirty"),
                                "hashes": {acc.rel(f["path"]): f["sha256"] for f in body["inputs"]}}
        acc.check("run_manifest", "manifest", "Run manifest records versions and output hashes",
                  len(body["inputs"]) == len(hashed) and {"numpy", "rasterio", "scikit-learn"} <= set(body["packages"]),
                  {"hashed_files": len(body["inputs"]), "packages": len(body["packages"])},
                  "every listed output hashed; core package versions present")


# --------------------------------------------------------------------------- report

def _report(acc: Run, verdict, error, started, seconds, offline) -> dict:
    from . import __version__
    from .provenance import _run, _versions

    here = Path(__file__).parent
    year = acc.info.get("period", [DEFAULT_START])[0][:4]
    sources = {} if offline else {k: {**v, "attribution": v["attribution"].format(year=year)}
                                  for k, v in DATA_SOURCES.items()}
    return {
        "schema": SCHEMA, "verdict": verdict, "error": error,
        "engine": {"version": __version__, "git_commit": _run(["git", "rev-parse", "HEAD"], cwd=here),
                   "git_describe": _run(["git", "describe", "--always", "--dirty", "--tags"], cwd=here)},
        "started_utc": started.isoformat(), "runtime_s": round(seconds, 1),
        "environment": {"python": sys.version.split()[0], "platform": platform.platform(),
                        "machine": platform.machine(), "packages": _versions()},
        "data_sources": sources,
        **{k: v for k, v in acc.info.items() if k != "traceback"},
        "traceback": acc.info.get("traceback"),
        "stages": acc.stages,
        "checks": [asdict(c) for c in acc.checks],
        "summary": {"checks": len(acc.checks), "passed": sum(c.passed for c in acc.checks),
                    "failed": [c.id for c in acc.checks if not c.passed]},
        "scope": {"proves": OFFLINE_PROVES if offline else PROVES, "does_not_prove": DOES_NOT_PROVE},
    }


PROVES = [
    "The complete chain runs unattended on public data for a real area and season, from STAC search to exports.",
    "Imported Sentinel-2 reflectance (BOA offset handling) and Sentinel-1 gamma0 are physically plausible.",
    "All rasters share one grid; after per-geometry offset correction, Sentinel-1 and Sentinel-2 are "
    "co-registered within one 10 m pixel on windows independent of the correction.",
    "No no-data sentinel leaks into features; indices and probabilities respect their mathematical ranges.",
    "A model trained on the stack recovers a structure present in the data on segments it did not see.",
    "Training is reproducible (same seed -> bit-identical predictions); outputs follow the documented schemas "
    "(parcel table, COG, STAC Item, GeoParquet) and are traceable through versions, git commit and hashes.",
]
OFFLINE_PROVES = [
    "OFFLINE MODE: synthetic scenes. The chain, checks and report work without network access; nothing is "
    "shown about real imagery.",
] + PROVES[3:]
DOES_NOT_PROVE = [
    "Crop-type accuracy: the classes are k-means PSEUDO-classes of the same NDVI the model sees, not crops; "
    "their recovery is expected to be high and is not comparable with crop-classification accuracy.",
    "That OBIA segments are agricultural parcels (no reference boundaries are used).",
    "Calibration or conformal coverage on independent data (held-out segments neighbour training segments).",
    "Transferability to other regions, seasons, processing baselines or data providers (Sentinel-1 offsets must be "
    "re-estimated elsewhere), absolute geolocation, or regional-scale performance.",
]


def _fmt(value) -> str:
    if isinstance(value, float):
        return f"{value:.4g}"
    if isinstance(value, dict):
        return "; ".join(f"{k}: {_fmt(v)}" for k, v in value.items())
    if isinstance(value, list):
        return "[" + ", ".join(_fmt(v) for v in value) + "]"
    return str(value)


def render_markdown(r: dict) -> str:
    e = r["engine"]
    lines = [f"# GeoCrop Engine acceptance report — {r['verdict']}", "",
             f"Schema `{r['schema']}` · engine {e['version']} (`{e.get('git_describe') or e.get('git_commit')}`) · "
             f"started {r['started_utc'][:19]}Z · runtime {r['runtime_s']} s · mode: {r.get('mode')}", ""]
    s = r["summary"]
    lines += [f"**Verdict: {r['verdict']}** — {s['passed']}/{s['checks']} checks passed"
              + (f"; failed: {', '.join(s['failed'])}" if s["failed"] else "") + ".", ""]
    if r.get("error"):
        lines += [f"Run error: `{r['error']}`", ""]
    a = r.get("area", {})
    if a:
        lines += ["## Area and period", "",
                  "| Item | Value |", "|---|---|",
                  f"| Bounding box (WGS 84) | {', '.join(str(v) for v in a['bbox_wgs84'])} |",
                  f"| Grid | {a['crs']}, {a['shape'][0]} x {a['shape'][1]} px at {a['pixel_m']:g} m, "
                  f"{a['area_km2']} km² |",
                  f"| Period | {' to '.join(r.get('period', []))} |",
                  "| Windows | " + ", ".join(f"{k} {v[0]}..{v[1]}" for k, v in r.get("windows", {}).items()) + " |",
                  ""]
    if r.get("data_sources"):
        lines += ["## Data sources and attribution", ""]
        for name, src in r["data_sources"].items():
            lines.append(f"- **{name}**: `{src['collection']}` at {src['endpoint']} — {src['licence']}. "
                         f"*{src['attribution']}.*")
        items = r.get("stac_items", {})
        for key in ("sentinel-2", "sentinel-1"):
            ids = items.get(key, [])
            lines += ["", f"<details><summary>{key}: {len(ids)} STAC items used</summary>", ""]
            lines += [f"- `{i}`" for i in ids] + ["", "</details>"]
        lines.append("")
    if r.get("s1_geolocation_offsets"):
        lines += ["## Sentinel-1 geolocation relative to Sentinel-2", "",
                  "Offsets of the Sentinel-1 RTC scenes per acquisition geometry, estimated from the calibration "
                  "windows by NDVI / VH cross-correlation (pixels of 10 m; reference[y, x] ~ S1[y + dy, x + dx]). "
                  "Offsets with a clear correlation peak and >= 0.5 px are removed before compositing; the "
                  "co-registration check is evaluated on the remaining (independent) windows only.", "",
                  "| Geometry | Scenes | dy | dx | Offset (m) | r at peak | r at (0, 0) | Applied |",
                  "|---|---|---|---|---|---|---|---|"]
        for key, o in r["s1_geolocation_offsets"].items():
            lines.append(f"| {key} | {o['scenes']} | {o['dy']:+.2f} | {o['dx']:+.2f} | "
                         f"{10 * (o['dy'] ** 2 + o['dx'] ** 2) ** 0.5:.0f} | {o['r_peak']:.2f} | {o['r_zero']:.2f} | "
                         f"{'yes' if o.get('applied') else 'no'} |")
        lines.append("")
    if r.get("pseudo_reference"):
        p = r["pseudo_reference"]
        lines += ["## Pseudo-reference (" + p["note"] + ")", "", f"{p['method']}.", "",
                  f"Segments used: {p['segments_used']} ({p['train_segments']} training, {p['holdout_segments']} "
                  f"held out); class counts: {_fmt(p['class_counts'])}.", "",
                  "| Pseudo-class | Mean NDVI per window |", "|---|---|"]
        lines += [f"| {k} | {', '.join(f'{v:.2f}' for v in c)} |" for k, c in p["ndvi_centroids"].items()] + [""]
    lines += ["## Checks", "", "| Check | Stage | Result | Value | Threshold | Method |", "|---|---|---|---|---|---|"]
    for c in r["checks"]:
        value = _fmt(c["value"]).replace("|", "/")
        if len(value) > 300:
            value = value[:297] + "..."
        lines.append(f"| `{c['id']}` — {c['description']} | {c['stage']} | {'PASS' if c['passed'] else '**FAIL**'} "
                     f"| {value} | {c['threshold']} | {c['method'].replace('|', '/')} |")
    lines += ["", "## Stages (runtime and memory)", "",
              "| Stage | Status | Seconds | Python peak MB | Process peak RSS MB |", "|---|---|---|---|---|"]
    lines += [f"| {st['stage']} | {st['status']} | {st['seconds']} | {st['python_peak_mb']} | "
              f"{st['process_peak_rss_mb']} |" for st in r["stages"]]
    lines += ["", "Python peak = peak of Python/NumPy allocations traced during the stage (tracemalloc; GDAL "
              "buffers excluded). Process peak RSS is the high-water mark of the process up to the end of the stage.",
              ""]
    if r.get("validation"):
        v = r["validation"]
        lines += ["## Validation on held-out pseudo-labelled segments", "", v["note"], "",
                  "| Level | n | OA | kappa | macro-F1 |", "|---|---|---|---|---|"]
        for lvl in ("segment_level", "point_level"):
            x = v[lvl]
            lines.append(f"| {lvl.replace('_', ' ')} | {x['n']} | {x['overall_accuracy']:.3f} | {x['kappa']:.3f} | "
                         f"{x['macro_f1']:.3f} |")
        lines.append("")
    lines += ["## What this acceptance run proves", ""] + [f"- {x}" for x in r["scope"]["proves"]]
    lines += ["", "## What it does NOT prove", ""] + [f"- {x}" for x in r["scope"]["does_not_prove"]]
    env = r["environment"]
    pk = env["packages"]
    lines += ["", "## Environment", "", f"Python {env['python']} on {env['platform']}; "
              + ", ".join(f"{k} {v}" for k, v in pk.items()), ""]
    m = r.get("manifest")
    if m:
        lines += [f"Run manifest: `{m['path']}` (git commit `{m['git_commit']}`, dirty: {m['git_dirty']}).", "",
                  "| Output | SHA-256 |", "|---|---|"] + [f"| `{k}` | `{v[:16]}…` |" for k, v in m["hashes"].items()]
        lines.append("")
    return "\n".join(lines)
