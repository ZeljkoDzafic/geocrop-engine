"""Study preflight: can a configured run succeed, and roughly what will it cost?

`geocrop preflight` extends the read-only input audit (:mod:`geocrop.audit`) with the checks that decide
whether a long regional run is worth starting: scene availability per window, expected clear-observation
coverage, training-class support under the configured spatial blocks/folds/holdout, CRS/grid
compatibility, and a resource estimate (disk, RAM, runtime) compared with the free space on each output
volume. Every check is ``pass``, ``warn`` or ``error``; the verdict is ``ready``, ``ready-with-warnings``
or ``blocked``.

Nothing here reads full-resolution pixels: masks are read decimated onto a coarse version of the reference
grid and class support uses polygon centroids with the same block/fold/holdout functions as training
(:mod:`geocrop.model`). Resource figures are planning estimates, not measurements; the assumptions are
listed in every report and in ``docs/preflight.md``.

The same disk estimate backs the pipeline's free-space guard (:func:`check_disk`).
"""
from __future__ import annotations

import csv
import math
import os
import shutil
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np

from . import __version__, applicability, parallel, scenes
from . import config as cfgmod
from .parallel import CGROUP_CPU_FILES, available_cpus  # noqa: F401  (defined in parallel; re-exported)

SCHEMA = "geocrop-preflight/1"
GB = 1e9
S2_COMPOSITE_BYTES = 10 * 4 + 2          # ten float32 bands + uint16 valid_count per pixel and window
BLOCK = 512                              # block size of compositing, feature building and prediction
TS_BLOCK = 256                           # block size of time-series reconstruction
HEAVY_STAGES = ("composite", "features", "predict")

# Rough single-machine runtime coefficients (seconds), measured with `geocrop run` on synthetic demo scenes
# (0.6 Mpx grid, 20 S2 + 8 S1 acquisitions, 97 features, ~61k samples, 100 trees) on an 8-core laptop and
# rounded up; the prediction/AOA terms were recalibrated on real 25-1000 km2 runs (docs/performance.md). Real
# runs differ by I/O, reprojection, tree depth and core count: treat results as an order of magnitude and
# recalibrate from a pilot's stages.json (seconds per stage).
RUNTIME = {
    "composite_s_per_mpx_scene": 1.0,      # per million grid pixels per S2 acquisition (measured 0.65)
    "composite_s1_s_per_mpx_scene": 0.3,   # per million grid pixels per S1 acquisition
    "features_s_per_mpx_feature": 0.1,     # per million pixels per output feature (measured 0.08)
    "timeseries_s_per_mpx_scene_series": 0.5,
    "train_s_per_msample_tree": 1.5,       # per million samples per tree, per CV fold + final fit (measured 1.2),
                                           # wall time on >= TRAIN_REFERENCE_CPUS cores (scaled up on fewer)
    "predict_s_per_mpx_tree": 0.1,         # per million pixels per tree, RF probabilities + uncertainty bands
    # Area-of-applicability map: exact nearest-neighbour search of every pixel among up to 20 000 training points,
    # per process (the map is spread over `workers` processes, see estimate()). Measured 35 s per Mpx with 97
    # features on real Semberija stacks with the matrix-product search (146-150 s with the k-d tree of earlier
    # releases; docs/performance.md); scaled linearly with the feature count and rounded up.
    "aoa_s_per_mpx_100_features": 40.0,
}
# Training (parallel tree building) was timed on 8 cores; with fewer usable CPUs its wall time grows in proportion
# (measured 2.1x the 8-core estimate in a 2-CPU Ubuntu container, docs/performance.md). More cores are not credited.
TRAIN_REFERENCE_CPUS = 8
# Python interpreter, GDAL block cache and imported libraries: measured ~1 GB above the per-stage working sets
# on 25-1000 km2 real runs (docs/performance.md).
RAM_OVERHEAD_BYTES = 1.0 * GB
WORKER_OVERHEAD_BYTES = 0.5 * GB          # interpreter, GDAL and libraries of one block worker process
ASSUMPTIONS = [
    "Rasters are written over the full reference-grid extent (pixels outside the AOI included).",
    "Expected on-disk size = uncompressed float32 size x resources.compression_ratio (deflate + "
    "floating-point predictor); the uncompressed size is the upper bound.",
    "Feature count follows the pipeline's naming rules from the config and the sensors found per window.",
    "Peak RAM is the largest per-stage working set from the block sizes (512 px; 256 px for time series) "
    "plus Random-Forest model size plus ~1 GB of Python/GDAL/library overhead; parallel tree building is not "
    "included.",
    "Training samples = sum over polygons of min(area in pixels, model.max_samples_per_polygon).",
    "Random-Forest model size (upper bound; separable data gives far smaller trees) ~ trees x 2 x 0.63 x "
    "samples / min_samples_leaf nodes x (64 + 8 x classes) bytes; aoa.json ~ 2 x the float32 sample matrix.",
    "Runtime uses rough per-pixel coefficients (see RUNTIME in geocrop.preflight; prediction includes the "
    "area-of-applicability map, divided by the block workers: resources.workers, else model.n_jobs if positive, "
    "else min(usable CPUs, 4)); order of magnitude only. Scene import (network) is not estimated.",
    "Each block worker process adds a model copy, its block buffers and ~0.5 GB of interpreter/library memory to "
    "the prediction RAM estimate.",
    "Training wall time is scaled by 8 / usable CPUs below 8 (affinity set or cgroup quota); more cores are not credited.",
    "Class support uses polygon centroids; training uses sampled pixel centres, so buffered folds may differ slightly.",
    "Coverage uses SCL/CLDPRB/CLOUDMASK/SHADOW at reduced resolution without mask dilation or temporal screening.",
]


class DiskSpaceError(RuntimeError):
    """Raised by the pipeline guard when an output volume lacks the space a stage needs."""


# ----------------------------------------------------------------------------------------- helpers
def _windows(cfg) -> dict:
    return cfgmod.parse_windows(cfg["windows"])


def _existing_ancestor(path: Path) -> Path:
    path = Path(path).absolute()
    while not path.exists() and path != path.parent:
        path = path.parent
    return path


def free_bytes(path) -> int:
    """Free bytes on the volume that will hold `path` (nearest existing ancestor)."""
    return int(shutil.disk_usage(_existing_ancestor(Path(path))).free)


CGROUP_MEMORY_FILES = ("/sys/fs/cgroup/memory.max",                       # cgroup v2 (Docker, systemd, k8s)
                       "/sys/fs/cgroup/memory/memory.limit_in_bytes")     # cgroup v1


def memory_limit_bytes(cgroup_files=CGROUP_MEMORY_FILES) -> tuple[int | None, str]:
    """Memory this process may use: installed RAM, or a smaller cgroup limit (container, systemd MemoryMax).

    Returns ``(bytes or None, source)``; ``source`` is ``"installed"`` or the cgroup file that lowered it. On
    Linux, ``SC_PHYS_PAGES`` reports the host's (or VM's) RAM even inside a container started with ``--memory``.
    """
    try:
        total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        total = None
    source = "installed"
    for name in cgroup_files:
        try:
            raw = Path(name).read_text(encoding="ascii").strip()
        except OSError:
            continue
        if raw.isdigit() and int(raw) > 0 and (total is None or int(raw) < total):
            total, source = int(raw), name           # "max" or a huge v1 sentinel means: no limit
        break
    return total, source


def _volume(path) -> str:
    anchor = _existing_ancestor(Path(path))
    try:
        return f"dev:{os.stat(anchor).st_dev}"
    except OSError:
        return str(anchor)


def _discover(root) -> tuple[list[dict], list[str]]:
    root = Path(root)
    if not root.exists():
        return [], []
    found, bad = [], []
    for sidecar in sorted(root.rglob(scenes.SIDE_CAR)):
        try:
            found.append(scenes.read(sidecar.parent))
        except (ValueError, KeyError, TypeError) as exc:
            bad.append(f"{sidecar.parent}: {exc}")
    return sorted(found, key=lambda m: m["date"]), bad


def _s1_prefixes(cfg, s1_by_window) -> list[str]:
    """Orbit prefixes of S1 composites ('' unless orbits are kept separate), as features.plan names them."""
    from .s1 import ORBITS
    if not cfg["s1"]["separate_orbits"]:
        return [""]
    found = {ORBITS.get(str(m.get("orbit_direction", "")).upper(), "") for v in s1_by_window.values() for m in v}
    return sorted(f"{p}_" if p else "" for p in found) or [""]


def feature_count(cfg, n_windows: int, use_s2: bool, s1_prefixes: list[str] | None) -> int:
    """Window-feature count with the naming rules of :func:`geocrop.features.plan`."""
    f = cfg["features"]
    per_period = (list(f["s2_bands"]) + list(f["s2_indices"])) if use_s2 else []
    s1_names = [f"{pre}{n}" for pre in (s1_prefixes or []) for n in f["s1"]] if f["s1"] else []
    per_period += s1_names
    count = n_windows * len(per_period)
    deltas = [n for n in per_period if n in f["deltas"] or
              (n.split("_", 1)[-1] in f["deltas"] and n.startswith(("ASC_", "DESC_")))]
    count += len(deltas) * max(0, n_windows - 1)
    if n_windows > 1:
        count += 4 * len([n for n in per_period if n in f["seasonal"]])
    return count


def timeseries_layout(cfg, s2_dates, s1_dates) -> dict:
    """Series and feature count of the time-series stack (see :mod:`geocrop.tsfeatures`)."""
    t = cfg["timeseries"]
    if not t["enabled"] or not (t["start"] and t["end"]):
        return {"series": 0, "features": 0, "scenes": 0}
    from .timeseries import make_grid
    from .tsfeatures import PHENOLOGY
    start, end = date.fromisoformat(str(t["start"])), date.fromisoformat(str(t["end"]))
    n_s2 = sum(start <= d <= end for d in s2_dates)
    n_s1 = sum(start <= d <= end for d in s1_dates)
    series = (len(t["s2_indices"]) if n_s2 else 0) + (len(t["s1_indices"]) if n_s1 else 0)
    grid = make_grid(start, end, t["step_days"])
    outputs = len(range(0, len(grid), max(1, t["output_step_days"] // t["step_days"])))
    per = outputs + len(PHENOLOGY) + 2 + 2 * t["harmonic_order"]
    return {"series": series, "features": series * per, "scenes": n_s2 + n_s1}


def _grid_info(cfg, s2_scenes) -> dict | None:
    import rasterio
    ref = cfg["grid"].get("reference")
    candidates = [("grid.reference", Path(ref))] if ref else []
    candidates += [("first S2 scene (B02.tif)", Path(s["path"]) / "B02.tif") for s in s2_scenes[:1]]
    for source, path in candidates:
        if not path.is_file():
            continue
        try:
            ds = rasterio.open(path)
        except rasterio.errors.RasterioError as exc:   # estimates degrade to "grid unknown"; preflight reports it
            return {"source": source, "path": str(path), "error": str(exc)}
        with ds:
            return {"source": source, "path": str(path), "crs": ds.crs.to_string() if ds.crs else None,
                    "width": ds.width, "height": ds.height, "transform": ds.transform,
                    "res": tuple(ds.res), "bounds": tuple(ds.bounds),
                    "aoi_mask": ds.dtypes[0] == "uint8" and ds.nodata == 0 and source == "grid.reference"}
    return None


def plan(cfg) -> dict:
    """Cheap study layout used by preflight and the disk guard: scenes per window, grid, feature counts.

    Reads only scene sidecars and raster headers."""
    windows = _windows(cfg)
    s2, bad2 = _discover(cfg["paths"]["s2_scenes"])
    s1, bad1 = _discover(cfg["paths"]["s1_scenes"])
    s2 = [m for m in s2 if str(m.get("sensor", "S2")).upper() == "S2"]
    s1 = [m for m in s1 if str(m.get("sensor", "S1")).upper() == "S1"]
    s2w, s1w = scenes.assign(s2, windows), scenes.assign(s1, windows)
    use_s2 = any(s2w.values())
    use_s1 = bool(cfg["features"]["s1"]) and any(s1w.values())
    prefixes = _s1_prefixes(cfg, s1w) if use_s1 else []
    n_windows = len(windows)
    ts = timeseries_layout(cfg, [m["date"] for m in s2], [m["date"] for m in s1])
    grid = _grid_info(cfg, s2)
    grid_error = grid.pop("error") if grid and "error" in grid else None
    if grid_error:
        grid = None
    return {"windows": windows, "s2": s2, "s1": s1, "s2_by_window": s2w, "s1_by_window": s1w,
            "unreadable_sidecars": bad2 + bad1, "use_s2": use_s2, "use_s1": use_s1, "s1_prefixes": prefixes,
            "window_features": feature_count(cfg, n_windows, use_s2, prefixes) if n_windows else 0,
            "timeseries": ts, "grid": grid, "grid_error": grid_error,
            "pixels": grid["width"] * grid["height"] if grid else None}


def _n_classes(cfg) -> int:
    report = Path(cfg["paths"]["output"]) / "model" / "train_report.json"
    try:
        import json
        return max(2, len(json.loads(report.read_text(encoding="utf8")).get("class_counts", {})))
    except (OSError, ValueError):
        pass
    try:
        import geopandas as gpd
        frame = gpd.read_file(cfg["paths"]["training"], ignore_geometry=True)
        return max(2, int(frame[cfg["classes"]["label_field"]].dropna().astype(str).nunique()))
    except Exception:  # noqa: BLE001 - any reader failure falls back to a conservative default
        return 10


def stage_bytes(cfg, layout: dict, stage: str, n_classes: int | None = None) -> int:
    """Uncompressed bytes a stage writes (0 when the grid is unknown)."""
    px = layout["pixels"] or 0
    if stage == "composite":
        s2w = sum(bool(v) for v in layout["s2_by_window"].values())
        s1w = sum(bool(v) for v in layout["s1_by_window"].values())
        layers = 2 * max(1, len(_s1_prefixes(cfg, layout["s1_by_window"])))
        return int(px * (s2w * S2_COMPOSITE_BYTES + s1w * layers * 4))
    if stage == "features":
        fw, ft = layout["window_features"], layout["timeseries"]["features"]
        total = fw + ft + layout["timeseries"]["series"] * 2   # + quality layers
        if fw and ft:
            total += fw + ft                                      # parts are kept next to the merged stack
        return int(px * total * 4)
    if stage == "predict":
        k = n_classes or _n_classes(cfg)
        return int(px * ((k + 3 + 2) * 4 + 1))                    # probabilities, uncertainty, aoa, classes
    return 0


STAGE_TARGET = {"composite": "composites", "features": "features", "predict": "output"}


def check_disk(cfg, stage: str, existing_bytes: int = 0, layout: dict | None = None) -> dict | None:
    """Free-space guard before a heavy stage; raises :class:`DiskSpaceError` when space is short.

    need = expected bytes (compression_ratio x uncompressed) x safety_factor + min_free_gb, minus the
    bytes of this stage's existing outputs that the rerun will overwrite."""
    r = cfg["resources"]
    if not r["disk_guard"] or stage not in HEAVY_STAGES:
        return None
    estimate_error = None
    try:
        layout = layout or plan(cfg)
        if stage == "predict" and Path(cfg["paths"]["features"]).is_file():
            import rasterio
            with rasterio.open(cfg["paths"]["features"]) as ds:
                layout = {**layout, "pixels": ds.width * ds.height}
        expected = stage_bytes(cfg, layout, stage) * r["compression_ratio"]
    except Exception as exc:  # noqa: BLE001 - an estimate must never mask the stage's own error reporting
        expected, estimate_error = 0.0, f"{type(exc).__name__}: {exc}"
    target = Path(cfg["paths"][STAGE_TARGET[stage]])
    need = max(0.0, expected * r["safety_factor"] - existing_bytes) + r["min_free_gb"] * GB
    free = free_bytes(target)
    info = {"stage": stage, "volume_path": str(_existing_ancestor(target)), "free_gb": round(free / GB, 2),
            "need_gb": round(need / GB, 2), "estimate_gb": round(expected / GB, 2),
            "replaced_gb": round(existing_bytes / GB, 2)}
    if estimate_error:
        info["estimate_error"] = estimate_error
    if free < need:
        raise DiskSpaceError(
            f"Not enough free disk for stage {stage!r} on {info['volume_path']}: {info['free_gb']} GB free, "
            f"{info['need_gb']} GB needed (estimate {info['estimate_gb']} GB x safety_factor "
            f"{r['safety_factor']} - {info['replaced_gb']} GB of outputs being replaced + min_free_gb "
            f"{r['min_free_gb']}). Free space (e.g. `geocrop clean "
            "--config ... --all-intermediates`), move paths to a larger volume, or adjust resources.* deliberately.")
    return info


# ----------------------------------------------------------------------------------------- report
class _Report:
    def __init__(self):
        self.checks: list[dict] = []

    def add(self, code, section, status, message, **details):
        item = {"code": code, "section": section, "status": status, "message": message}
        if details:
            item["details"] = details
        self.checks.append(item)
        return item


def _inventory_rows(paths) -> tuple[list[dict], list[str]]:
    rows, problems = [], []
    for path in paths:
        try:
            with open(path, newline="", encoding="utf8") as fh:
                seen = set()
                for r in csv.DictReader(fh):
                    ident = r.get("id") or r.get("item_id") or r.get("name")
                    when = r.get("sensing_start") or r.get("datetime")
                    collection = (r.get("collection") or r.get("name") or "").upper()
                    if not when or ident in seen:
                        continue
                    seen.add(ident)
                    sensor = "S1" if ("SENTINEL-1" in collection or collection.startswith("S1")) else (
                        "S2" if ("SENTINEL-2" in collection or collection.startswith("S2")) else None)
                    if sensor is None:
                        continue
                    day = datetime.fromisoformat(when.replace("Z", "+00:00")).date()
                    rows.append({"sensor": sensor, "date": day,
                                 "orbit_direction": (r.get("orbit_direction") or r.get("orbit_state") or "").upper()})
        except (OSError, ValueError, csv.Error) as exc:
            problems.append(f"{path}: {exc}")
    return rows, problems


def _check_windows(cfg, layout, rep: _Report, inventories) -> list[dict]:
    p = cfg["preflight"]
    windows = layout["windows"]
    if not windows:
        rep.add("windows.defined", "scenes", "error", "Define windows: in the configuration (composite stage requires them).")
        return []
    for problem in layout["unreadable_sidecars"]:
        rep.add("scenes.sidecar", "scenes", "error", f"Unreadable scene.json: {problem}")
    inv, problems = _inventory_rows(inventories)
    for problem in problems:
        rep.add("inventory.readable", "scenes", "error", f"Cannot read inventory CSV {problem}")
    inv_by = {s: scenes.assign([r for r in inv if r["sensor"] == s], windows) for s in ("S2", "S1")}
    imported = {"S2": bool(layout["s2"]), "S1": bool(layout["s1"])}
    active = {"S2": layout["use_s2"] or (not imported["S2"] and any(inv_by["S2"].values())),
              "S1": layout["use_s1"] or (not imported["S1"] and bool(cfg["features"]["s1"]) and any(inv_by["S1"].values()))}
    rows = []
    for w in windows:
        s2, s1 = layout["s2_by_window"][w], layout["s1_by_window"][w]
        row = {"window": w, "start": windows[w][0].isoformat(), "end": windows[w][1].isoformat(),
               "s2_imported": len(s2), "s1_imported": len(s1),
               "s1_by_orbit": {d or "UNKNOWN": sum(str(m.get("orbit_direction") or "").upper() == d for m in s1)
                               for d in sorted({str(m.get("orbit_direction") or "").upper() for m in s1})},
               "s2_inventory": len(inv_by["S2"][w]), "s1_inventory": len(inv_by["S1"][w]),
               "s1_inventory_by_orbit": {d or "UNKNOWN": sum(r["orbit_direction"] == d for r in inv_by["S1"][w])
                                         for d in sorted({r["orbit_direction"] for r in inv_by["S1"][w]})}}
        rows.append(row)
        # Effective availability: imported scene folders when a sensor has any, otherwise the inventory.
        n2 = row["s2_imported"] if imported["S2"] else row["s2_inventory"]
        n1 = row["s1_imported"] if imported["S1"] else row["s1_inventory"]
        source = "imported scene folders" if imported["S2"] or imported["S1"] else "inventory CSV"
        if n2 + n1 == 0:
            rep.add("scenes.window_empty", "scenes", "error",
                    f"Window {w} has no S2 or S1 scenes ({source}); the composite stage would stop.", window=w)
            continue
        for sensor, n, minimum in (("S2", n2, p["min_s2_scenes_per_window"]), ("S1", n1, p["min_s1_scenes_per_window"])):
            if active[sensor] and n < minimum:
                rep.add("scenes.window_minimum", "scenes", "warn",
                        f"Window {w}: {n} {sensor} scene(s), below the configured minimum {minimum}.",
                        window=w, sensor=sensor, count=n, minimum=minimum)
        if not imported["S2"] and not imported["S1"]:
            rep.add("scenes.not_imported", "scenes", "warn",
                    f"Window {w}: counts come from inventories only; no scene folders imported yet.", window=w)
    if layout["use_s2"]:
        lacking = [w for w in windows if not layout["s2_by_window"][w]]
        rep.add("scenes.s2_every_window", "scenes", "error" if lacking else "pass",
                f"Sentinel-2 must be present in every window or none; missing in {lacking}." if lacking
                else "Sentinel-2 scenes present in every window.", windows=lacking)
    if layout["use_s1"]:
        lacking = [w for w in windows if not layout["s1_by_window"][w]]
        layouts = {w: sorted({str(m.get("orbit_direction") or "").upper() for m in layout["s1_by_window"][w]})
                   for w in windows if layout["s1_by_window"][w]}
        mixed = cfg["s1"]["separate_orbits"] and len({tuple(v) for v in layouts.values()}) > 1
        bad = bool(lacking) or mixed
        rep.add("scenes.s1_layout", "scenes", "error" if bad else "pass",
                (f"Sentinel-1 composites need the same orbit layout in every window; missing in {lacking}, "
                 f"orbit directions per window {layouts}.") if bad else "Sentinel-1 layout consistent across windows.",
                windows=lacking, orbits=layouts)
    # Scene folder completeness (file existence only).
    from .indices import S2_BANDS
    incomplete = [str(m["path"]) for m in layout["s2"] if not all((m["path"] / f"{b}.tif").is_file()
                                                                   for b in (*S2_BANDS, "SCL"))]
    incomplete += [str(m["path"]) for m in layout["s1"] if not all((m["path"] / f"{b}.tif").is_file()
                                                                    for b in ("VV", "VH"))]
    if layout["s2"] or layout["s1"]:
        rep.add("scenes.files", "scenes", "error" if incomplete else "pass",
                f"{len(incomplete)} scene folder(s) lack required rasters." if incomplete
                else "All scene folders contain their required rasters.", folders=incomplete[:20])
    return rows


def _coarse_grid(grid, max_px):
    from affine import Affine
    factor = max(1, math.ceil(max(grid["width"], grid["height"]) / max_px))
    shape = (math.ceil(grid["height"] / factor), math.ceil(grid["width"] / factor))
    return grid["transform"] * Affine.scale(factor), shape, factor


def _check_coverage(cfg, layout, rep: _Report) -> list[dict]:
    """Fraction of the AOI with fewer clear S2 observations than compositing.min_valid_observations."""
    import rasterio
    from rasterio.warp import Resampling

    from .raster import read_aligned
    from .s2 import clear_mask
    grid, c, p = layout["grid"], cfg["compositing"], cfg["preflight"]
    if grid is None or not layout["s2"]:
        return []
    transform, shape, factor = _coarse_grid(grid, p["coverage_max_px"])
    if grid["aoi_mask"]:
        with rasterio.open(grid["path"]) as ds:
            aoi = ds.read(1, out_shape=shape, resampling=Resampling.nearest) > 0
    else:
        aoi = np.ones(shape, dtype=bool)
    rows = []
    for w, members in layout["s2_by_window"].items():
        if not members:
            continue
        count = np.zeros(shape, dtype="uint16")
        days: dict = {}
        for m in members:      # same-day tiles are mosaicked by the composite: count one observation per day
            days.setdefault(m["date"], []).append(m["path"])
        for folders in days.values():
            day_ok = np.zeros(shape, dtype=bool)
            for folder in folders:
                if not (folder / "SCL.tif").is_file():
                    continue
                with rasterio.open(folder / "SCL.tif") as src:
                    scl = read_aligned(src, transform, grid["crs"], shape, Resampling.nearest, dtype="uint8", fill=0)
                extra = {}
                for name in ("CLDPRB", "CLOUDMASK", "SHADOW"):
                    if (folder / f"{name}.tif").is_file() and (name != "CLDPRB" or c["cloud_probability_max"] is not None):
                        with rasterio.open(folder / f"{name}.tif") as src:
                            extra[name] = read_aligned(src, transform, grid["crs"], shape, Resampling.nearest,
                                                       dtype="uint8", fill=255)
                day_ok |= clear_mask(scl, extra.get("CLDPRB"), c["scl_invalid"], c["cloud_probability_max"], 0,
                                     extra.get("CLOUDMASK"), extra.get("SHADOW"))
            count += day_ok
        n_aoi = int(aoi.sum())
        zero = float(((count == 0) & aoi).sum() / n_aoi) if n_aoi else 1.0
        below = float(((count < c["min_valid_observations"]) & aoi).sum() / n_aoi) if n_aoi else 1.0
        row = {"window": w, "scenes": len(members), "decimation": factor,
               "aoi_fraction_zero_clear": round(zero, 4), "aoi_fraction_below_min_valid": round(below, 4),
               "median_clear_observations": float(np.median(count[aoi])) if n_aoi else 0.0}
        rows.append(row)
        if below >= 1.0:
            status, msg = "error", f"Window {w}: no AOI pixel reaches {c['min_valid_observations']} clear observation(s); the composite would be empty."
        elif below > p["max_unobserved_fraction"]:
            status, msg = "warn", (f"Window {w}: {below:.1%} of the AOI has fewer than "
                                   f"{c['min_valid_observations']} clear observation(s) (no-data in the composite).")
        else:
            status, msg = "pass", f"Window {w}: {below:.1%} of the AOI below the clear-observation minimum."
        rep.add("coverage.clear_observations", "coverage", status, msg, **row)
    return rows


def _metric(crs) -> bool:
    from .audit import _metric as metric
    try:
        return metric(crs)
    except Exception:  # noqa: BLE001 - unparsable CRS is simply not metric
        return False


def _check_crs(cfg, layout, rep: _Report) -> dict:
    """Grid CRS and the overlap of each reference layer with the study grid footprint."""
    import geopandas as gpd
    from pyproj import CRS
    from shapely.geometry import box
    grid, out = layout["grid"], {}
    if layout.get("grid_error"):
        rep.add("grid.readable", "crs", "error", f"Cannot read the grid raster: {layout['grid_error']}")
    if grid is None:
        rep.add("grid.available", "crs", "warn" if layout["s2"] or layout["s1"] else "error",
                "No grid.reference raster and no S2 scene to derive the grid from; "
                "grid size, coverage and resources cannot be estimated.")
        return out
    out["grid"] = {k: grid[k] for k in ("source", "path", "crs", "width", "height", "res")}
    metric = grid["crs"] is not None and _metric(grid["crs"])
    rep.add("grid.metric_crs", "crs", "pass" if metric else "error",
            f"Grid CRS {grid['crs']} is a projected metric CRS." if metric else
            f"Grid CRS {grid['crs']} is not a projected CRS in metres (spatial blocks, buffers and areas need one).")
    declared = cfg["grid"].get("crs")
    if declared and grid["crs"]:
        same = CRS.from_user_input(declared) == CRS.from_user_input(grid["crs"])
        rep.add("grid.declared_crs", "crs", "pass" if same else "error",
                "grid.crs matches the reference grid." if same else
                f"grid.crs {declared} differs from the {grid['source']} CRS {grid['crs']}; recreate the grid "
                "or correct the configuration.")
    resolution = cfg["grid"].get("resolution_m")
    if resolution and grid["res"] and not all(math.isclose(r, float(resolution), rel_tol=1e-6) for r in grid["res"]):
        rep.add("grid.resolution", "crs", "warn",
                f"grid.resolution_m {resolution} differs from the grid pixel size {grid['res']}.")
    if grid["crs"] is None:
        return out
    footprint = box(*grid["bounds"])
    for layer, required in (("training", True), ("parcels", False), ("observations", False)):
        path = Path(cfg["paths"][layer])
        if not path.exists():
            if required:
                rep.add(f"{layer}.present", "crs", "error", f"Training layer {path} not found; the samples stage needs it.")
            else:
                rep.add(f"{layer}.present", "crs", "warn",
                        f"{layer.capitalize()} layer {path} not found; the dependent stages will be skipped.")
            continue
        try:
            frame = gpd.read_file(path)
        except Exception as exc:  # noqa: BLE001 - reported by the audit as well
            rep.add(f"{layer}.readable", "crs", "error", f"Cannot read {path}: {exc}")
            continue
        if frame.crs is None or not len(frame):
            continue  # reported by the input audit
        try:
            geoms = frame.to_crs(grid["crs"]).geometry
            valid = geoms[~(geoms.isna() | geoms.is_empty)]
            inside = valid.intersects(footprint)
            n_in = int(inside.sum())
        except Exception as exc:  # noqa: BLE001
            rep.add(f"{layer}.reprojection", "crs", "error", f"Cannot reproject {layer} ({frame.crs}) to the grid: {exc}")
            continue
        out[layer] = {"crs": frame.crs.to_string(), "features": int(len(frame)), "inside_grid": n_in}
        if n_in == 0:
            rep.add(f"{layer}.overlap", "crs", "error" if required else "warn",
                    f"No {layer} feature intersects the study grid (layer CRS {frame.crs.to_string()}); "
                    "check the CRS definition and the AOI.", inside=0, features=len(valid))
        elif n_in < len(valid):
            rep.add(f"{layer}.overlap", "crs", "warn",
                    f"{len(valid) - n_in} of {len(valid)} {layer} features lie outside the study grid.",
                    inside=n_in, features=len(valid))
        else:
            rep.add(f"{layer}.overlap", "crs", "pass", f"All {layer} features intersect the study grid.")
    return out


def _check_classes(cfg, layout, rep: _Report) -> dict:
    """Class support under the configured blocks, folds, calibration split and locked holdout (centroids)."""
    import geopandas as gpd

    from . import model
    from .samples import Samples
    path, cl, m = Path(cfg["paths"]["training"]), cfg["classes"], cfg["model"]
    if not path.exists():
        return {}
    try:
        frame = gpd.read_file(path)
    except Exception:  # noqa: BLE001 - reported elsewhere
        return {}
    if frame.crs is None or cl["label_field"] not in frame or cl["id_field"] not in frame:
        return {}
    frame = frame[~(frame.geometry.isna() | frame.geometry.is_empty) & frame[cl["label_field"]].notna()]
    grid_crs = layout["grid"]["crs"] if layout["grid"] else None
    crs = grid_crs if grid_crs and _metric(grid_crs) else (frame.crs if _metric(frame.crs) else None)
    if crs is None:
        rep.add("classes.metric_crs", "classes", "error",
                "Spatial blocks need a projected metric CRS (grid or training layer).")
        return {}
    projected = frame.to_crs(crs).geometry
    pts = projected.centroid
    y = frame[cl["label_field"]].astype(str).to_numpy()
    ids = frame[cl["id_field"]].astype(str).to_numpy()
    s = Samples(np.zeros((len(frame), 1), "float32"), y, ids, pts.x.to_numpy(), pts.y.to_numpy(), ["centroid"],
                str(crs))
    block_m, k, seed = m["spatial_block_m"], m["cv_folds"], cfg["seed"]
    groups = model.spatial_blocks(s, block_m)
    table = {}
    for c in sorted(set(y)):
        sel = y == c
        table[c] = {"polygons": int(len(set(ids[sel]))), "blocks": int(len(set(groups[sel])))}
    out = {"spatial_block_m": block_m, "cv_folds": k, "spatial_groups": int(len(set(groups))), "per_class": table}
    if layout["grid"]:
        pixel_area = abs(layout["grid"]["res"][0] * layout["grid"]["res"][1])
        out["expected_samples"] = int(np.minimum(projected.area.to_numpy() / pixel_area,
                                                 m["max_samples_per_polygon"]).sum())
    if len(set(groups)) < 2:
        rep.add("classes.spatial_groups", "classes", "error",
                f"Only {len(set(groups))} spatial block(s) at {block_m} m; cross-validation needs at least two.")
        return out
    single = sorted(c for c, v in table.items() if v["blocks"] < 2)
    few = sorted(c for c, v in table.items() if 2 <= v["blocks"] < k)
    rep.add("classes.blocks", "classes", "warn" if single else "pass",
            f"Classes in a single spatial block cannot be both trained and tested in any fold: {single}."
            if single else "Every class occurs in at least two spatial blocks.", classes=single)
    if few:
        rep.add("classes.blocks_per_fold", "classes", "warn",
                f"Classes with fewer blocks than cv_folds ({k}) cannot appear in every CV test fold: {few}.",
                classes=few)
    try:
        dev, test = model.split_final_holdout(s, block_m, m["holdout_fraction"], seed)
    except ValueError as exc:
        rep.add("classes.holdout", "classes", "error", f"Locked holdout cannot be drawn: {exc}")
        return out
    if test is not None:
        missing_test = sorted(set(y) - set(test.y))
        missing_dev = sorted(set(y) - set(dev.y))
        out["holdout"] = {"polygons": int(len(set(test.polygon))), "classes_missing": missing_test}
        rep.add("classes.holdout_coverage", "classes", "warn" if missing_test else "pass",
                f"Classes absent from the locked holdout (no holdout accuracy for them): {missing_test}."
                if missing_test else "Every class is represented in the locked holdout.", classes=missing_test)
        if missing_dev:
            rep.add("classes.development", "classes", "error",
                    f"The locked holdout takes every polygon of {missing_dev}; they cannot be trained.",
                    classes=missing_dev)
    try:
        proper, _ = model.selection_samples(dev, block_m, seed, m["spatial_buffer_m"], m["calibration_mode"],
                                            m["calibration_fraction"])
        dev_groups = model.spatial_blocks(proper, block_m)
        plan_ = model._fold_plan(proper.y, dev_groups, k, seed, polygon=proper.polygon,
                                 coordinates=np.column_stack([proper.x, proper.yy]), crs=proper.crs,
                                 spatial_buffer_m=m["spatial_buffer_m"])
    except ValueError as exc:
        rep.add("classes.folds", "classes", "error", f"Training split/folds would fail: {exc}")
        return out
    classes = sorted(set(proper.y))
    absent_test = {c: sum(c not in set(proper.y[te]) for _, te, _ in plan_) for c in classes}
    absent_train = {c: sum(c not in set(proper.y[tr]) for tr, _, _ in plan_) for c in classes}
    out["folds"] = {"n_folds": len(plan_), "folds_without_class_in_test": absent_test,
                    "folds_without_class_in_train": absent_train}
    lacking = {c: n for c, n in absent_test.items() if n}
    rep.add("classes.fold_coverage", "classes", "warn" if lacking else "pass",
            f"Classes missing from some CV test folds (count of folds): {lacking}." if lacking
            else f"Every class appears in all {len(plan_)} CV test folds.", classes=lacking)
    untrained = {c: n for c, n in absent_train.items() if n}
    if untrained:
        rep.add("classes.fold_training", "classes", "warn",
                f"Classes missing from the training side of some folds (unpredictable there): {untrained}.",
                classes=untrained)
    return out


def _rf_model_bytes(cfg, n_samples, n_classes) -> int:
    m = cfg["model"]
    trees = int(m["n_estimators"] or 100)
    leaf = max(1, int(m["min_samples_leaf"] or 1))
    return int(trees * 2 * 0.63 * n_samples / leaf * (64 + 8 * n_classes))


def estimate(cfg, layout, n_samples: int | None = None, n_classes: int | None = None,
             cpus: float | None = None) -> dict:
    """Rough grid, feature, disk, RAM and runtime figures (see ASSUMPTIONS); `cpus` defaults to the usable CPUs."""
    r = cfg["resources"]
    px = layout["pixels"]
    n_classes = n_classes or _n_classes(cfg)
    fw, ts = layout["window_features"], layout["timeseries"]
    n_features = fw + ts["features"]
    out = {"grid_pixels": px, "grid_megapixels": round(px / 1e6, 2) if px else None,
           "area_km2": (round(px * abs(layout["grid"]["res"][0] * layout["grid"]["res"][1]) / 1e6, 1)
                        if px else None),
           "features": {"windows": fw, "timeseries": ts["features"], "total": n_features},
           "classes": n_classes, "compression_ratio": r["compression_ratio"]}
    if not px:
        out["note"] = "Grid unknown: disk, RAM and runtime cannot be estimated."
        return out
    disk = {}
    for stage in HEAVY_STAGES:
        raw = stage_bytes(cfg, layout, stage, n_classes)
        disk[stage] = {"target": cfg["paths"][STAGE_TARGET[stage]], "uncompressed_gb": round(raw / GB, 3),
                       "expected_gb": round(raw * r["compression_ratio"] / GB, 3)}
    n_samples = int(n_samples or 0)
    model_bytes = _rf_model_bytes(cfg, n_samples, n_classes)
    samples_bytes = n_samples * max(1, n_features) * 4
    train_bytes = 3 * samples_bytes + model_bytes      # samples.npz + aoa.json (~2x samples) + model (upper bound)
    disk["samples+model"] = {"target": cfg["paths"]["output"], "uncompressed_gb": round(train_bytes / GB, 3),
                             "expected_gb": round(train_bytes / GB, 3)}
    out["stack_gb"] = {"uncompressed": round(px * n_features * 4 / GB, 3),
                       "expected": round(px * n_features * 4 * r["compression_ratio"] / GB, 3)}
    out["disk"] = disk
    max_scenes = max([len(v) for v in layout["s2_by_window"].values()] + [0])
    halo = int(cfg["compositing"]["mask_dilation_px"])
    w, h = layout["grid"]["width"], layout["grid"]["height"]
    block = min(BLOCK, w) * min(BLOCK, h)
    if cpus is None:
        cpus = available_cpus()[0]
    # Blockwise prediction / AOA map run in `workers` processes (resources.workers, else model.n_jobs, else
    # min(usable CPUs, 4)) when the grid is large enough for a pool; each holds a model copy and block buffers.
    workers = min(parallel.resolve_workers(r.get("workers"), cfg["model"].get("n_jobs"), cpus=cpus), max(1, int(cpus)))
    workers = workers if px >= applicability.POOL_MIN_PIXELS else 1
    ram = {
        "composite": max_scenes * 11 * (min(BLOCK, w) + 2 * halo) * (min(BLOCK, h) + 2 * halo) * 4 * 3,
        "features": max(1, fw) * block * 4 * 2,
        "timeseries": ts["scenes"] * max(1, ts["series"]) * min(TS_BLOCK, w) * min(TS_BLOCK, h) * 8 * 3,
        "train": samples_bytes * 3 + model_bytes,
        "predict": (max(1, n_features) * block * 4 * 2 + n_classes * block * 4 * 3 + model_bytes
                    + (workers * (max(1, n_features) * block * 4 * 2 + model_bytes + WORKER_OVERHEAD_BYTES)
                       if workers > 1 else 0)),
    }
    ram = {k: v + RAM_OVERHEAD_BYTES for k, v in ram.items()}
    out["peak_ram_gb"] = {k: round(v / GB, 2) for k, v in ram.items()}
    out["peak_ram_gb"]["max"] = round(max(ram.values()) / GB, 2)
    mpx = px / 1e6
    trees = int(cfg["model"]["n_estimators"] or 100)
    out["cpus"] = round(cpus, 2)
    out["workers"] = workers
    s1_scenes = sum(len(v) for v in layout["s1_by_window"].values())
    s2_scenes = sum(len(v) for v in layout["s2_by_window"].values())
    secs = {
        "composite": mpx * (s2_scenes * RUNTIME["composite_s_per_mpx_scene"]
                            + s1_scenes * RUNTIME["composite_s1_s_per_mpx_scene"]),
        "features": mpx * fw * RUNTIME["features_s_per_mpx_feature"],
        "timeseries": mpx * ts["scenes"] * max(1, ts["series"]) * RUNTIME["timeseries_s_per_mpx_scene_series"]
        if ts["features"] else 0.0,
        "train": n_samples / 1e6 * trees * RUNTIME["train_s_per_msample_tree"] * (cfg["model"]["cv_folds"] + 1)
        * TRAIN_REFERENCE_CPUS / min(TRAIN_REFERENCE_CPUS, max(1.0, cpus)),
        "predict": mpx * trees * RUNTIME["predict_s_per_mpx_tree"]
        + mpx * RUNTIME["aoa_s_per_mpx_100_features"] * max(1, n_features) / 100 / workers,
    }
    out["runtime_hours_rough"] = {k: round(v / 3600, 3) for k, v in secs.items()}
    out["runtime_hours_rough"]["total"] = round(sum(secs.values()) / 3600, 3)
    return out


def _check_resources(cfg, est, rep: _Report) -> list[dict]:
    r = cfg["resources"]
    if "disk" not in est:
        rep.add("resources.estimate", "resources", "warn", est.get("note", "No resource estimate."))
        return []
    volumes: dict[str, dict] = {}
    for stage, d in est["disk"].items():
        key = _volume(d["target"])
        v = volumes.setdefault(key, {"path": str(_existing_ancestor(Path(d["target"]))), "stages": [],
                                     "expected_gb": 0.0, "uncompressed_gb": 0.0})
        v["stages"].append(stage)
        v["expected_gb"] += d["expected_gb"]
        v["uncompressed_gb"] += d["uncompressed_gb"]
    rows = []
    for v in volumes.values():
        free = free_bytes(v["path"]) / GB
        need = v["expected_gb"] * r["safety_factor"] + r["min_free_gb"]
        upper = v["uncompressed_gb"] * r["safety_factor"] + r["min_free_gb"]
        row = {**v, "expected_gb": round(v["expected_gb"], 2), "uncompressed_gb": round(v["uncompressed_gb"], 2),
               "free_gb": round(free, 2), "need_gb": round(need, 2), "need_uncompressed_gb": round(upper, 2)}
        rows.append(row)
        if free < need:
            rep.add("resources.disk", "resources", "error",
                    f"{row['path']}: {row['free_gb']} GB free, about {row['need_gb']} GB needed for {v['stages']}.", **row)
        elif free < upper:
            rep.add("resources.disk", "resources", "warn",
                    f"{row['path']}: enough space only if compression reaches {r['compression_ratio']} "
                    f"({row['free_gb']} GB free, {row['need_uncompressed_gb']} GB uncompressed).", **row)
        else:
            rep.add("resources.disk", "resources", "pass",
                    f"{row['path']}: {row['free_gb']} GB free, about {row['need_gb']} GB needed.", **row)
    limit, source = memory_limit_bytes()
    total_ram = limit / GB if limit else None
    peak = est["peak_ram_gb"]["max"]
    if total_ram:
        status = "error" if peak > total_ram else ("warn" if peak > 0.7 * total_ram else "pass")
        what = "installed" if source == "installed" else f"allowed by the cgroup limit ({source})"
        rep.add("resources.ram", "resources", status,
                f"Rough peak RAM {peak} GB vs {total_ram:.1f} GB {what}.", peak_gb=peak,
                installed_gb=round(total_ram, 1), memory_source=source)
    return rows


def run(cfg, config_path=None, inventories=()) -> dict:
    """Build the preflight report for a resolved configuration."""
    from .audit import audit_inputs
    rep = _Report()
    layout = plan(cfg)
    paths = cfg["paths"]
    windows = _check_windows(cfg, layout, rep, [*cfg["preflight"]["inventories"], *inventories])
    coverage = _check_coverage(cfg, layout, rep)
    crs_info = _check_crs(cfg, layout, rep)
    distance_crs = layout["grid"]["crs"] if layout["grid"] and _metric(layout["grid"]["crs"]) else None
    layers = {k: paths[k] for k in ("training", "parcels", "observations") if Path(paths[k]).exists()}
    if layers:
        audit = audit_inputs(**layers, label_field=cfg["classes"]["label_field"], id_field=cfg["classes"]["id_field"],
                             parcel_id_field=cfg["parcels"]["id_field"],
                             observation_label_field=cfg["validation"]["label_field"],
                             training_exclusion_m=cfg["validation"]["training_exclusion_m"], distance_crs=distance_crs)
        for c in audit["checks"]:
            if c["status"] == "fail":
                rep.add(f"audit.{c['code']}", "inputs", "error" if c["severity"] == "error" else "warn",
                        f"{c['layer']}: {c['message']}", **({"count": c["count"]} if "count" in c else {}))
        rep.add("audit.inputs", "inputs", "pass" if audit["ok"] else "error",
                f"Input audit: {audit['summary']['passed']} passed, {audit['summary']['errors']} error(s), "
                f"{audit['summary']['warnings']} warning(s).")
    classes = _check_classes(cfg, layout, rep)
    from .estimators import backend_error
    algorithm = cfg["model"].get("algorithm") or "rf"
    unavailable = backend_error(algorithm)
    rep.add("model.backend", "classes", "error" if unavailable else "pass",
            f"Estimator {algorithm!r} cannot be used: {unavailable}" if unavailable
            else f"Estimator {algorithm!r} and its optional backend load.")
    if cfg["timeseries"]["enabled"] and not (cfg["timeseries"]["start"] and cfg["timeseries"]["end"]):
        rep.add("timeseries.season", "scenes", "error", "timeseries.enabled needs timeseries.start and timeseries.end.")
    n_classes = len(classes.get("per_class", {})) or None
    est = estimate(cfg, layout, classes.get("expected_samples"), n_classes)
    volumes = _check_resources(cfg, est, rep)
    est["volumes"] = volumes
    counts = {s: sum(c["status"] == s for c in rep.checks) for s in ("pass", "warn", "error")}
    verdict = "blocked" if counts["error"] else ("ready-with-warnings" if counts["warn"] else "ready")
    return {"schema": SCHEMA, "verdict": verdict, "engine": __version__,
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "config": str(config_path) if config_path else None, "config_fingerprint": cfgmod.fingerprint(cfg),
            "summary": counts, "checks": rep.checks, "windows": windows, "coverage": coverage,
            "layers": crs_info, "classes": classes, "resources": est, "assumptions": ASSUMPTIONS,
            "scope": "Read-only planning check: decimated mask reads, polygon centroids and rough resource "
                     "coefficients; not a substitute for a pilot run or for accuracy assessment."}


# ----------------------------------------------------------------------------------------- markdown
def _md_table(rows, columns) -> list[str]:
    if not rows:
        return ["_none_", ""]
    out = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for r in rows:
        out.append("| " + " | ".join(str(r.get(c, "")).replace("|", "/") for c in columns) + " |")
    return out + [""]


def to_markdown(report: dict) -> str:
    badge = {"ready": "READY", "ready-with-warnings": "READY WITH WARNINGS", "blocked": "BLOCKED"}[report["verdict"]]
    s = report["summary"]
    lines = [f"# GeoCrop preflight: {badge}", "",
             f"Config `{report['config']}` (fingerprint `{report['config_fingerprint']}`), engine {report['engine']}, "
             f"{report['generated_at']}.", "",
             f"**{s['error']} error(s), {s['warn']} warning(s), {s['pass']} passed.**", ""]
    for status, title in (("error", "Errors (must fix before running)"), ("warn", "Warnings (review)")):
        items = [c for c in report["checks"] if c["status"] == status]
        if items:
            lines += [f"## {title}", ""] + [f"- `{c['code']}`: {c['message']}" for c in items] + [""]
    lines += ["## Scenes per window", ""]
    lines += _md_table(report["windows"], ["window", "start", "end", "s2_imported", "s1_imported", "s1_by_orbit",
                                           "s2_inventory", "s1_inventory"])
    lines += ["## Clear-observation coverage (decimated)", ""]
    lines += _md_table(report["coverage"], ["window", "scenes", "aoi_fraction_zero_clear",
                                            "aoi_fraction_below_min_valid", "median_clear_observations", "decimation"])
    cl = report["classes"]
    if cl.get("per_class"):
        lines += [f"## Training class support (blocks of {cl['spatial_block_m']} m, {cl['cv_folds']} folds)", ""]
        folds = cl.get("folds", {})
        rows = [{"class": c, **v,
                 "folds_without_test": folds.get("folds_without_class_in_test", {}).get(c, ""),
                 "in_holdout": ("no" if c in cl.get("holdout", {}).get("classes_missing", []) else "yes")
                 if "holdout" in cl else "n/a"} for c, v in cl["per_class"].items()]
        lines += _md_table(rows, ["class", "polygons", "blocks", "folds_without_test", "in_holdout"])
    r = report["resources"]
    lines += ["## Resource estimate (ROUGH planning figures)", "",
              f"- Grid: {r.get('grid_megapixels')} Mpx, ~{r.get('area_km2')} km²; features: {r['features']}",
              f"- Feature stack: {r.get('stack_gb', {})} GB (expected uses compression ratio {r['compression_ratio']})",
              f"- Peak RAM (GB): {r.get('peak_ram_gb', 'n/a')}",
              f"- Runtime (hours, rough): {r.get('runtime_hours_rough', 'n/a')}", ""]
    lines += _md_table(r.get("volumes", []), ["path", "stages", "expected_gb", "uncompressed_gb", "need_gb", "free_gb"])
    lines += ["## Assumptions", ""] + [f"- {a}" for a in report["assumptions"]] + ["", f"_{report['scope']}_", ""]
    return "\n".join(lines)
