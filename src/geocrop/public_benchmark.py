"""Frozen-protocol public benchmark (gap G1): crop types of public EuroCrops parcels with Sentinel-2 from STAC.

A versioned protocol YAML (``configs/benchmarks/*.yaml``) fixes everything that influences the numbers: the
label source and its checksum, tiles, year, HCAT class definitions, parcel sampling, spatial-block holdout,
imagery selection, compositing windows, feature set, models and metrics. ``run`` executes it end to end with
the engine's own building blocks:

    EuroCrops parcels (runtime download, md5-pinned) -> HCAT harmonisation (`geocrop.hcat`) -> stratified sample
    -> per tile: reference grid -> STAC COG import (`stac_import`) -> window composites (`s2.composite`)
       -> feature stack (`features.build_stack`) -> pixel samples (`samples.extract`)
    -> parcel-level split: test blocks | calibration blocks | proper-training blocks (+ distance buffer)
    -> per model: fit on training pixels, parcel probability = mean pixel probability,
       OA / macro-F1 / per-class F1 with block-bootstrap CIs, Mondrian split-conformal coverage and set size,
       review burden at a target accuracy (threshold chosen on calibration parcels), wall time, peak RSS.

The results JSON records the protocol SHA-256, scene IDs, sample fingerprint and software versions. Scores
describe this public task only; they are not evidence about accuracy in other regions or years.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timezone
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import yaml
from shapely.geometry import box, shape

from . import __version__, hcat, metrics, public_data, uncertainty
from .benchmark import _json_safe
from .config import parse_windows

SCHEMA = "geocrop-public-benchmark/1"
PROTOCOL_SCHEMA = "geocrop-public-benchmark-protocol/1"
REQUIRED = ("protocol", "version", "dataset", "year", "grid", "tiles", "classes", "sampling", "split", "imagery",
            "windows", "features", "models", "evaluation")


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


# --------------------------------------------------------------------------- protocol

def load_protocol(path: str | Path) -> tuple[dict, str]:
    """(protocol, sha256 of the file bytes). Any edit, including comments, yields a new hash."""
    raw = Path(path).read_bytes()
    protocol = yaml.safe_load(raw) or {}
    validate_protocol(protocol)
    return protocol, hashlib.sha256(raw).hexdigest()


def validate_protocol(p: dict) -> None:
    if p.get("schema") != PROTOCOL_SCHEMA:
        raise ValueError(f"Protocol schema must be {PROTOCOL_SCHEMA!r}")
    missing = [k for k in REQUIRED if k not in p]
    if missing:
        raise ValueError(f"Protocol lacks {missing}")
    hcat.rules_from_spec(p["classes"])
    windows = parse_windows(p["windows"])
    years = {d.year for pair in windows.values() for d in pair}
    if years != {int(p["year"])}:
        raise ValueError("All windows must lie in the protocol year")
    if not p["tiles"].get("origins"):
        raise ValueError("Protocol needs at least one tile origin")
    if not 0 < p["split"]["test_fraction"] < 0.5 or not 0 < p["split"]["calibration_fraction"] < 1:
        raise ValueError("split.test_fraction must be in (0, 0.5) and calibration_fraction in (0, 1)")
    if not 0 < p["evaluation"]["conformal_alpha"] < 1 or not 0 < p["evaluation"]["target_accuracy"] <= 1:
        raise ValueError("conformal_alpha must be in (0, 1) and target_accuracy in (0, 1]")
    if not p["models"]:
        raise ValueError("Protocol lists no models")


def tile_boxes(p: dict) -> dict:
    size = float(p["tiles"]["size_m"])
    return {str(k): box(x, y, x + size, y + size) for k, (x, y) in p["tiles"]["origins"].items()}


# --------------------------------------------------------------------------- labels

def label_parcels(raw: gpd.GeoDataFrame, p: dict, tile: str, footprint) -> gpd.GeoDataFrame:
    """Harmonise one tile's EuroCrops parcels to protocol classes; keep whole, valid parcels >= min area."""
    ds = p["dataset"]
    g = raw.to_crs(p["grid"]["crs"])
    g = g[g.geometry.notna() & ~g.geometry.is_empty]
    bad = ~g.geometry.is_valid
    if bad.any():
        g.loc[bad, "geometry"] = g.loc[bad, "geometry"].make_valid()
    g = g[g.geometry.within(footprint)]
    out = gpd.GeoDataFrame({"parcel_id": g[ds["id_field"]].astype(str).to_numpy(),
                            "hcat_code": g[ds["code_field"]].astype(str).to_numpy(),
                            "class_name": hcat.harmonise(g[ds["code_field"]], hcat.rules_from_spec(p["classes"])).to_numpy(),
                            "tile": tile}, geometry=g.geometry.to_numpy(), crs=g.crs)
    out["area_m2"] = out.geometry.area
    return out[out["class_name"].notna() & (out["area_m2"] >= float(p["sampling"]["min_area_m2"]))]


def sample_parcels(population: gpd.GeoDataFrame, n_per_class: int, seed: int) -> gpd.GeoDataFrame:
    """Stratified random draw without replacement; deterministic for a given population and seed."""
    rng = np.random.default_rng(seed)
    parts = []
    for cls in sorted(population["class_name"].unique()):
        group = population[population["class_name"] == cls].sort_values("parcel_id")
        k = min(n_per_class, len(group))
        parts.append(group.iloc[np.sort(rng.choice(len(group), size=k, replace=False))])
    return gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=population.crs)


def prepare_labels(p: dict, cache: str | Path | None = None, layer: str | Path | None = None):
    """(population, sample) of harmonised parcels in all tiles; `layer` overrides the downloaded archive."""
    ds = p["dataset"]
    if layer is None:
        layer = public_data.eurocrops_layer(ds["archive"], ds.get("checksum"), str(ds["zenodo_record"]), cache)
    frames = []
    for tile, footprint in tile_boxes(p).items():
        raw = public_data.read_parcels(layer, footprint.bounds, p["grid"]["crs"],
                                       columns=[ds["id_field"], ds["code_field"]])
        frames.append(label_parcels(raw, p, tile, footprint))
    population = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=p["grid"]["crs"])
    population = population.drop_duplicates("parcel_id")
    sample = sample_parcels(population, int(p["sampling"]["parcels_per_class"]), int(p["sampling"]["seed"]))
    return population, sample


# --------------------------------------------------------------------------- imagery

def _baseline(item) -> tuple:
    text = str(item.properties.get("s2:processing_baseline") or "0")
    try:
        return tuple(int(x) for x in text.split("."))
    except ValueError:
        return (0,)


def window_candidates(items, start: date, end: date, aoi_wgs84, min_coverage: float = 0.98) -> list:
    """One item per acquisition date (highest processing baseline, best AOI coverage), least cloudy first.

    Earth Search lists several processing versions of the same granule and overlapping MGRS tiles; both would
    double-count an acquisition and waste reads."""
    best: dict[str, tuple] = {}
    for item in items:
        day = date.fromisoformat(str(item.properties.get("datetime"))[:10])
        if not start <= day <= end:
            continue
        cover = shape(item.geometry).intersection(aoi_wgs84).area / aoi_wgs84.area if item.geometry else 0.0
        if cover < min_coverage:
            continue
        key = (round(cover, 3), _baseline(item), -float(item.properties.get("eo:cloud_cover") or 100), item.id)
        if day.isoformat() not in best or key > best[day.isoformat()][0]:
            best[day.isoformat()] = (key, item)
    chosen = [v[1] for v in best.values()]
    return sorted(chosen, key=lambda i: (float(i.properties.get("eo:cloud_cover") or 100), i.id))


def _import(item, grid: Path, target: Path, min_clear: float) -> dict:
    from .stac_import import cached_scene_valid, grid_identity, import_item

    if cached_scene_valid(target, grid_identity(grid)):
        return {"item": item.id, "status": "cached", "path": str(target)}
    try:
        return import_item(item, grid, target, None, min_clear)
    except Exception as exc:  # a broken remote asset is a recorded outcome, the next candidate is tried
        return {"item": item.id, "status": f"failed: {type(exc).__name__}: {exc}"}


def tile_stack(p: dict, tile: str, footprint, work: Path, workers: int = 4, log=None) -> tuple[Path, dict]:
    """Reference grid, scene import, window composites and feature stack of one tile (resumable)."""
    from rasterio.warp import transform_bounds

    from . import features
    from .raster import make_grid
    from .s2 import composite

    log = log or _log
    tdir = work / "tiles" / tile
    stack, meta_path = tdir / "stack.tif", tdir / "imagery.json"
    if stack.exists() and meta_path.exists() and features.manifest_path(stack).exists():
        return stack, json.loads(meta_path.read_text(encoding="utf8"))
    crs = p["grid"]["crs"]
    grid = make_grid(gpd.GeoDataFrame(geometry=[footprint], crs=crs), crs, float(p["grid"]["resolution_m"]),
                     tdir / "grid.tif")
    windows = parse_windows(p["windows"])
    img = p["imagery"]
    bbox = transform_bounds(crs, "EPSG:4326", *footprint.bounds)
    items = public_data.search_items(img["endpoint"], img["collection"], bbox,
                                     min(a for a, _ in windows.values()).isoformat(),
                                     max(b for _, b in windows.values()).isoformat() + "T23:59:59Z",
                                     img.get("max_cloud"))
    aoi = box(*bbox)
    meta = {"tile": tile, "bbox_wgs84": [round(v, 6) for v in bbox], "items_found": len(items), "windows": {}}
    k, cap, min_clear = int(img["scenes_per_window"]), int(img.get("max_candidates", 12)), float(img["min_clear_fraction"])
    comp = p.get("compositing", {})
    for name, (start, end) in windows.items():
        candidates = window_candidates(items, start, end, aoi)[:cap]
        used, tried = [], []
        pos = 0
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            while len(used) < k and pos < len(candidates):
                batch = candidates[pos: pos + (k - len(used))]
                pos += len(batch)
                for res in pool.map(lambda it: _import(it, grid, tdir / "scenes" / it.id, min_clear), batch):
                    tried.append({key: res.get(key) for key in ("item", "status", "clear_fraction")})
                    if res["status"] in ("imported", "cached"):
                        used.append(res)
        if not used:
            raise RuntimeError(f"Tile {tile} window {name}: no acquisition with clear fraction >= {min_clear}")
        log(f"[{tile}] {name}: {len(used)} scenes ({', '.join(u['item'] for u in used)})")
        composite([u["path"] for u in used], tdir / "composites" / name, reference=grid,
                  statistic=comp.get("statistic", "median"), mask_dilation_px=int(comp.get("mask_dilation_px", 0)),
                  min_valid_observations=int(comp.get("min_valid_observations", 1)))
        meta["windows"][name] = {"start": start.isoformat(), "end": end.isoformat(), "candidates": len(candidates),
                                 "used": [u["item"] for u in used], "tried": tried}
    f = p["features"]
    features.build_stack(tdir / "composites", stack, periods=list(windows), s2_bands=tuple(f["s2_bands"]),
                         s2_indices=tuple(f["s2_indices"]), s1=(), deltas=tuple(f.get("deltas", ())),
                         seasonal=tuple(f.get("seasonal", ())))
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf8")
    return stack, meta


# --------------------------------------------------------------------------- split

def spatial_block(points: gpd.GeoSeries, block_m: float) -> np.ndarray:
    return (np.floor(points.x / block_m).astype(int).astype(str) + ":" +
            np.floor(points.y / block_m).astype(int).astype(str)).to_numpy()


def split_parcels(parcels: gpd.GeoDataFrame, split: dict) -> pd.Series:
    """Role per parcel: ``test`` | ``calibration`` | ``fit`` | ``buffer_excluded`` (index aligned)."""
    from scipy.spatial import cKDTree
    from sklearn.model_selection import GroupShuffleSplit

    from .model import holdout_mask

    centroids = parcels.geometry.centroid
    blocks = spatial_block(centroids, float(split["spatial_block_m"]))
    y = parcels["class_name"].to_numpy()
    seed = int(split["seed"])
    test = holdout_mask(y, blocks, float(split["test_fraction"]), seed)
    roles = np.where(test, "test", "fit").astype(object)
    rest = np.flatnonzero(~test)
    gss = GroupShuffleSplit(n_splits=1, test_size=float(split["calibration_fraction"]), random_state=seed)
    _, cal = next(gss.split(rest, groups=blocks[rest]))
    roles[rest[cal]] = "calibration"
    buffer_m = float(split.get("buffer_m", 0) or 0)
    if buffer_m and test.any():
        xy = np.column_stack([centroids.x, centroids.y])
        near = cKDTree(xy[test]).query(xy[~test], k=1)[0] <= buffer_m
        roles[rest[near]] = "buffer_excluded"
    for role in ("fit", "calibration"):
        missing = sorted(set(y) - set(y[roles == role]))
        if missing:
            raise ValueError(f"Split leaves no {role} parcels for classes {missing}; change the protocol")
    return pd.Series(roles, index=parcels.index, name="role")


# --------------------------------------------------------------------------- evaluation helpers

def peak_rss_mb() -> float | None:
    """Process high-water mark of resident memory in MB (None where `resource` is unavailable, e.g. Windows)."""
    try:
        import resource
    except ImportError:
        return None
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(peak / (1024 * 1024) if sys.platform == "darwin" else peak / 1024, 1)


def parcel_probabilities(proba: np.ndarray, polygon: np.ndarray, classes: list[str]) -> pd.DataFrame:
    frame = pd.DataFrame(proba, columns=classes)
    frame["parcel_id"] = polygon
    return frame.groupby("parcel_id", sort=True)[classes].mean()


def review_threshold(confidence, correct, target: float) -> float:
    """Smallest confidence threshold whose accepted set reaches `target` accuracy (inf when none does)."""
    order = np.argsort(-np.asarray(confidence, dtype=float))
    conf = np.asarray(confidence, dtype=float)[order]
    hits = np.cumsum(np.asarray(correct, dtype=bool)[order])
    accuracy = hits / np.arange(1, len(conf) + 1)
    ok = np.flatnonzero(accuracy >= target)
    # accepted set = all parcels with confidence >= threshold; ties must stay together
    for i in ok[::-1]:
        if i == len(conf) - 1 or conf[i + 1] < conf[i]:
            return float(conf[i])
    return float("inf")


def block_bootstrap(y_true, y_pred, blocks, labels, n_boot: int, seed: int, confidence: float) -> dict:
    """Percentile CIs of OA, macro-F1 and per-class F1, resampling spatial blocks of test parcels."""
    y_true, y_pred, blocks = (np.asarray(v).astype(str) for v in (y_true, y_pred, blocks))
    units = np.unique(blocks)
    members = {u: np.flatnonzero(blocks == u) for u in units}
    rng = np.random.default_rng(seed)
    oa, mf1, per = [], [], {c: [] for c in labels}
    for _ in range(n_boot):
        idx = np.concatenate([members[u] for u in rng.choice(units, size=len(units), replace=True)])
        s = metrics.summary(y_true[idx], y_pred[idx], labels=labels)
        oa.append(s["overall_accuracy"])
        mf1.append(s["macro_f1"])
        for c in labels:
            if s["per_class"][c]["reference_count"]:
                per[c].append(s["per_class"][c]["f1"])
    alpha = 1 - confidence
    q = (100 * alpha / 2, 100 * (1 - alpha / 2))

    def ci(v):
        return [float(np.percentile(v, q[0])), float(np.percentile(v, q[1]))] if len(v) else None
    return {"n_boot": n_boot, "units": "spatial blocks", "n_units": int(len(units)), "confidence": confidence,
            "overall_accuracy_ci": ci(oa), "macro_f1_ci": ci(mf1), "per_class_f1_ci": {c: ci(per[c]) for c in labels}}


def evaluate(name: str, params: dict, data: dict, p: dict, classes: list[str], seed: int) -> dict:
    """Fit one model on training pixels; parcel-level metrics on test parcels, calibrated on calibration parcels."""
    from .estimators import make_estimator

    ev = p["evaluation"]
    try:
        est = make_estimator(dict(params), seed)
    except RuntimeError as exc:          # optional backend missing
        return {"status": "skipped", "reason": str(exc), "params": params}
    t0 = time.perf_counter()
    est.fit(data["X_fit"], data["y_fit"])
    fit_s = time.perf_counter() - t0
    order = [list(map(str, est.classes_)).index(c) for c in classes]
    t0 = time.perf_counter()
    proba_cal = np.asarray(est.predict_proba(data["X_cal"]))[:, order]
    proba_test = np.asarray(est.predict_proba(data["X_test"]))[:, order]
    predict_s = time.perf_counter() - t0
    pixels = len(data["X_cal"]) + len(data["X_test"])

    pc = parcel_probabilities(proba_cal, data["poly_cal"], classes)
    pt = parcel_probabilities(proba_test, data["poly_test"], classes)
    y_cal = data["truth"].reindex(pc.index).to_numpy()
    y_test = data["truth"].reindex(pt.index).to_numpy()
    pred_test = np.array(classes)[pt.to_numpy().argmax(axis=1)]
    pred_cal = np.array(classes)[pc.to_numpy().argmax(axis=1)]
    summary = metrics.summary(y_test, pred_test, labels=classes)
    pixel_pred = np.array(classes)[proba_test.argmax(axis=1)]
    pixel = metrics.summary(data["y_test"], pixel_pred, labels=classes)
    boot = block_bootstrap(y_test, pred_test, data["block"].reindex(pt.index).to_numpy(), classes,
                           int(ev["bootstrap"]), int(ev["bootstrap_seed"]), float(ev["confidence"]))

    alpha = float(ev["conformal_alpha"])
    conf = uncertainty.conformal_thresholds(pc.to_numpy(), y_cal, classes, alphas=(alpha,), mondrian=True,
                                            sparse_class_policy="include")
    sets = uncertainty.prediction_sets(pt.to_numpy(), classes, conf["thresholds"][str(alpha)])
    cov = uncertainty.coverage(sets, y_test)
    cov["per_class_coverage"] = {c: float(np.mean([c in s for s, t in zip(sets, y_test) if t == c]))
                                 for c in classes if (y_test == c).any()}
    cov["target"] = 1 - alpha
    cov["always_include_classes"] = conf["always_include_classes"].get(str(alpha), [])

    target = float(ev["target_accuracy"])
    threshold = review_threshold(pc.max(axis=1).to_numpy(), pred_cal == y_cal, target)
    test_conf = pt.max(axis=1).to_numpy()
    accepted = test_conf >= threshold
    review = {"target_accuracy": target, "threshold_from_calibration": None if not np.isfinite(threshold) else threshold,
              "review_share": float(1 - accepted.mean()),
              "accepted_accuracy": float((pred_test[accepted] == y_test[accepted]).mean()) if accepted.any() else None,
              "accepted_parcels": int(accepted.sum())}
    return {"status": "ok", "params": params, "parcel": summary, "pixel": {k: pixel[k] for k in (
                "n", "overall_accuracy", "macro_f1", "kappa")},
            "bootstrap": boot, "conformal": cov, "review": review,
            "time": {"fit_s": round(fit_s, 3), "predict_s": round(predict_s, 3),
                     "predict_pixels_per_s": round(pixels / max(predict_s, 1e-9), 1),
                     "fit_pixels": int(len(data["y_fit"]))},
            "peak_rss_mb_after": peak_rss_mb()}


# --------------------------------------------------------------------------- run

def _fingerprint(*arrays) -> str:
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(str(a.dtype).encode() + str(a.shape).encode())
        h.update(json.dumps(a.tolist()).encode() if a.dtype.kind in "OUS" else memoryview(a).cast("B"))
    return h.hexdigest()


def run(protocol_path: str | Path, work_dir: str | Path, cache: str | Path | None = None,
        models: list[str] | None = None, workers: int = 4, layer: str | Path | None = None, log=None) -> dict:
    """Execute a protocol; write ``results.json`` and ``results.md`` into `work_dir` and return the report."""
    from .provenance import _run, _versions
    from .samples import Samples, extract

    log = log or _log
    started = datetime.now(timezone.utc)
    here = Path(__file__).parent
    engine = {"version": __version__, "git_commit": _run(["git", "rev-parse", "HEAD"], cwd=here),
              # tracked source/config changes at start; untracked run artefacts (e.g. catboost_info/) do not count
              "git_dirty": bool(_run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=here))}
    p, digest = load_protocol(protocol_path)
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    stages = {}

    def stage(name, t0):
        stages[name] = {"seconds": round(time.perf_counter() - t0, 1), "peak_rss_mb": peak_rss_mb()}
        log(f"stage {name}: {stages[name]['seconds']} s, peak RSS {stages[name]['peak_rss_mb']} MB")

    t0 = time.perf_counter()
    population, parcels = prepare_labels(p, cache, layer)
    stage("labels", t0)

    t0 = time.perf_counter()
    boxes = tile_boxes(p)
    imagery, parts = {}, []
    samp = p["sampling"]
    for tile, footprint in boxes.items():
        stack, meta = tile_stack(p, tile, footprint, work, workers, log)
        imagery[tile] = meta
        mine = parcels[parcels["tile"] == tile]
        if len(mine):
            parts.append(extract(stack, mine[["parcel_id", "class_name", "geometry"]], label_field="class_name",
                                 id_field="parcel_id", max_per_polygon=int(samp["max_pixels_per_parcel"]),
                                 inner_buffer_m=float(samp["inner_buffer_m"]), seed=int(samp["seed"])))
    stage("imagery_and_samples", t0)
    if any(s.features != parts[0].features for s in parts):
        raise RuntimeError("Tiles produced different feature sets")
    s = Samples(np.concatenate([q.X for q in parts]), np.concatenate([q.y for q in parts]),
                np.concatenate([q.polygon for q in parts]), np.concatenate([q.x for q in parts]),
                np.concatenate([q.yy for q in parts]), parts[0].features, parts[0].crs)

    with_pixels = parcels[parcels["parcel_id"].isin(set(s.polygon))].reset_index(drop=True)
    roles = split_parcels(with_pixels, p["split"])
    with_pixels["role"] = roles.to_numpy()
    with_pixels["block"] = spatial_block(with_pixels.geometry.centroid, float(p["split"]["spatial_block_m"]))
    role_of = with_pixels.set_index("parcel_id")["role"]
    pix_role = role_of.reindex(s.polygon).to_numpy()
    classes = sorted(with_pixels["class_name"].unique())
    data = {"X_fit": s.X[pix_role == "fit"], "y_fit": s.y[pix_role == "fit"],
            "X_cal": s.X[pix_role == "calibration"], "poly_cal": s.polygon[pix_role == "calibration"],
            "X_test": s.X[pix_role == "test"], "poly_test": s.polygon[pix_role == "test"],
            "y_test": s.y[pix_role == "test"],
            "truth": with_pixels.set_index("parcel_id")["class_name"],
            "block": with_pixels.set_index("parcel_id")["block"]}

    results = {}
    t_models = time.perf_counter()
    chosen = models or list(p["models"])
    unknown = [m for m in chosen if m not in p["models"]]
    if unknown:
        raise ValueError(f"Models {unknown} are not in the protocol")
    for name in chosen:
        t0 = time.perf_counter()
        log(f"model {name} ...")
        try:
            results[name] = evaluate(name, p["models"][name], data, p, classes, int(p["split"]["seed"]))
        except Exception as exc:  # a failing model is a reported outcome, others still run
            results[name] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}", "params": p["models"][name]}
        results[name]["wall_s"] = round(time.perf_counter() - t0, 2)
        log(f"model {name}: {results[name]['status']} in {results[name]['wall_s']} s")

    stage("models", t_models)
    counts = with_pixels.groupby(["class_name", "role"]).size().unstack(fill_value=0)
    report = {
        "schema": SCHEMA,
        "protocol": {"id": p["protocol"], "version": p["version"], "file": Path(protocol_path).name,
                     "sha256": digest},
        "command": " ".join(["geocrop", "benchmark-public", "--protocol", str(protocol_path), "--work", str(work_dir),
                             *(["--models", *models] if models else [])]),
        "started_utc": started.isoformat(timespec="seconds"),
        "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "engine": engine,
        "runtime": {"python": platform.python_version(), "platform": platform.platform(),
                    "cpu_count": os.cpu_count(), **_versions()},
        "dataset": {k: p["dataset"][k] for k in ("name", "zenodo_record", "archive", "checksum", "licence", "citation")},
        "labels": {"population_per_class": {str(k): int(v) for k, v in population["class_name"].value_counts().sort_index().items()},
                   "sampled_per_class": {str(k): int(v) for k, v in parcels["class_name"].value_counts().sort_index().items()},
                   "with_pixels_per_class": {str(k): int(v) for k, v in with_pixels["class_name"].value_counts().sort_index().items()},
                   "per_tile": {str(k): int(v) for k, v in parcels["tile"].value_counts().sort_index().items()}},
        "split": {"parcels_per_class_and_role": {c: {r: int(counts.loc[c, r]) for r in counts.columns} for c in counts.index},
                  "blocks_per_role": {r: int(with_pixels.loc[with_pixels.role == r, "block"].nunique())
                                      for r in sorted(with_pixels.role.unique())},
                  "pixels_per_role": {r: int((pix_role == r).sum()) for r in sorted(set(pix_role))}},
        "samples": {"pixels": int(len(s.y)), "features": len(s.features), "crs": s.crs,
                    "fingerprint": _fingerprint(s.X, s.y.astype(str), s.polygon.astype(str))},
        "imagery": imagery,
        "classes": classes,
        "models": results,
        "stages": stages,
        "peak_rss_mb": peak_rss_mb(),
        "note": "Parcel-level scores on a spatial-block holdout of public EuroCrops declarations (declared, not "
                "field-verified, labels). Valid for this protocol only; not evidence of accuracy elsewhere.",
    }
    report = _json_safe(json.loads(json.dumps(report, default=_json_default)))
    (work / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf8")
    (work / "results.md").write_text(markdown(report), encoding="utf8")
    return report


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


# --------------------------------------------------------------------------- report

def _pct(v, digits=1):
    return "–" if v is None else f"{100 * v:.{digits}f}"


def _ci(v):
    return "" if not v else f" [{100 * v[0]:.1f}, {100 * v[1]:.1f}]"


def markdown(r: dict) -> str:
    ok = {k: v for k, v in r["models"].items() if v.get("status") == "ok"}
    lines = [f"# Public benchmark: {r['protocol']['id']} v{r['protocol']['version']}", "",
             f"* Protocol: `configs/benchmarks/{r['protocol']['file']}`, SHA-256 `{r['protocol']['sha256']}`",
             f"* Command: `{r['command']}`",
             f"* Engine: geocrop-engine {r['engine']['version']}, commit `{r['engine']['git_commit']}`"
             f"{' (uncommitted changes)' if r['engine']['git_dirty'] else ''}",
             f"* Run: {r['started_utc']} – {r['finished_utc']} on {r['runtime']['platform']} "
             f"({r['runtime']['cpu_count']} logical CPUs), peak RSS {r['peak_rss_mb']} MB",
             f"* Labels: {r['dataset']['name']} `{r['dataset']['archive']}` (Zenodo {r['dataset']['zenodo_record']}, "
             f"`{r['dataset']['checksum']}`), licence {r['dataset']['licence']}. {r['dataset']['citation'].strip()}",
             f"* Samples: {r['samples']['pixels']} pixels, {r['samples']['features']} features, fingerprint "
             f"`{r['samples']['fingerprint'][:16]}`", "",
             "Scores are parcel-level (class = argmax of mean pixel probability) on spatial-block test parcels; "
             f"intervals are {int(100 * next(iter(ok.values()))['bootstrap']['confidence']) if ok else 95} % "
             "block-bootstrap percentiles. Labels are farmers' declarations, not field observations.", "",
             "## Models", "",
             "| Model | OA % | Macro-F1 % | Conformal coverage % (target) | Mean set size | Review share % "
             "(accepted acc. %) | Fit s | Predict s | Pixel OA % |",
             "|---|---|---|---|---|---|---|---|---|"]
    for name, m in r["models"].items():
        if m.get("status") != "ok":
            lines.append(f"| {name} | {m['status']}: {m.get('reason') or m.get('error')} | | | | | | | |")
            continue
        c, rv, b = m["conformal"], m["review"], m["bootstrap"]
        lines.append(f"| {name} | {_pct(m['parcel']['overall_accuracy'])}{_ci(b['overall_accuracy_ci'])} | "
                     f"{_pct(m['parcel']['macro_f1'])}{_ci(b['macro_f1_ci'])} | {_pct(c['coverage'])} ({_pct(c['target'], 0)}) | "
                     f"{c['mean_set_size']:.2f} | {_pct(rv['review_share'])} ({_pct(rv['accepted_accuracy'])}) | "
                     f"{m['time']['fit_s']:.1f} | {m['time']['predict_s']:.1f} | {_pct(m['pixel']['overall_accuracy'])} |")
    lines += ["", "Review share: parcels whose maximum probability is below the threshold that reaches "
              f"{_pct(next(iter(ok.values()))['review']['target_accuracy'], 0) if ok else '–'} % accuracy on the "
              "calibration parcels; the accepted accuracy is then measured on the test parcels.", ""]
    if ok:
        classes = r["classes"]
        lines += ["## Per-class F1 % (test parcels, block-bootstrap CI)", "",
                  "| Class | Test parcels | " + " | ".join(ok) + " |", "|---|---|" + "---|" * len(ok)]
        for c in classes:
            n = r["split"]["parcels_per_class_and_role"].get(c, {}).get("test", 0)
            cells = [f"{_pct(m['parcel']['per_class'][c]['f1'])}{_ci(m['bootstrap']['per_class_f1_ci'].get(c))}"
                     for m in ok.values()]
            lines.append(f"| {c} | {n} | " + " | ".join(cells) + " |")
        lines.append("")
    lines += ["## Data and split", "", "| Class | Population | Sampled | With pixels | Fit | Calibration | Test | Buffer-excluded |",
              "|---|---|---|---|---|---|---|---|"]
    lab, sp = r["labels"], r["split"]["parcels_per_class_and_role"]
    for c in r["classes"]:
        roles = sp.get(c, {})
        lines.append(f"| {c} | {lab['population_per_class'].get(c, 0)} | {lab['sampled_per_class'].get(c, 0)} | "
                     f"{lab['with_pixels_per_class'].get(c, 0)} | {roles.get('fit', 0)} | {roles.get('calibration', 0)} | "
                     f"{roles.get('test', 0)} | {roles.get('buffer_excluded', 0)} |")
    lines += ["", f"Spatial blocks per role: {r['split']['blocks_per_role']}; pixels per role: {r['split']['pixels_per_role']}.",
              "", "## Imagery (Sentinel-2 L2A, Earth Search)", "", "| Tile | Window | Scenes used |", "|---|---|---|"]
    for tile, meta in r["imagery"].items():
        for w, info in meta["windows"].items():
            lines.append(f"| {tile} | {w} {info['start']}–{info['end']} | {', '.join(info['used'])} |")
    lines += ["", "## Stages", "", "| Stage | Seconds | Peak RSS MB (process high-water mark) |", "|---|---|---|"]
    for name, st in r["stages"].items():
        lines.append(f"| {name} | {st['seconds']} | {st['peak_rss_mb']} |")
    lines += ["", f"_{r['note']}_", ""]
    return "\n".join(lines)
