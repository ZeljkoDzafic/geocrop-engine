"""Reproducible benchmarks: classifiers on identical spatial folds, and OBIA vs U-Net segmentation.

Model benchmark
    Every algorithm is evaluated on the same samples, the same spatial folds and (optionally) the same
    locked holdout blocks; tuning grids are applied with nested CV, so no algorithm is favoured by
    selection on its own test folds. Each run is repeated over `repeats` seeds and reported as
    mean ± SD, together with training time, prediction throughput and a McNemar test against the
    first (reference) algorithm on out-of-fold and holdout predictions.

Segmentation benchmark
    OBIA (with and without seam reconciliation) and, when a trained checkpoint is given, U-Net +
    watershed are run on the same raster and scored against the same reference parcels with
    object-level OS/US/D and matched-IoU F1, and with boundary precision/recall/F1 at a tolerance.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from . import metrics
from .model import cross_validate, folds, holdout_mask, make_estimator, nested_cross_validate, spatial_blocks
from .samples import Samples

DEFAULT_MODELS = {
    "rf": {"algorithm": "rf", "n_estimators": 500, "max_features": "sqrt", "class_weight": "balanced_subsample",
           "n_jobs": -1},
    "extra_trees": {"algorithm": "extra_trees", "n_estimators": 500, "max_features": "sqrt",
                    "class_weight": "balanced_subsample", "n_jobs": -1},
    "hist_gb": {"algorithm": "hist_gb", "n_estimators": 300, "learning_rate": 0.05},
    "lightgbm": {"algorithm": "lightgbm", "n_estimators": 400, "learning_rate": 0.05, "num_leaves": 31,
                 "class_weight": "balanced", "n_jobs": -1},
    "catboost": {"algorithm": "catboost", "n_estimators": 500, "max_depth": 6, "class_weight": "balanced"},
}


def _available(params: dict) -> str | None:
    try:
        make_estimator(params, 0)
        return None
    except RuntimeError as exc:
        return str(exc)


def models(samples: Samples, output_dir: str | Path, features: list[str] | None = None, candidates: dict | None = None,
           grids: dict | None = None, n_splits: int = 5, spatial_block_m: float | None = 5000, seed: int = 42,
           repeats: int = 3, holdout_fraction: float = 0.0, targets=()) -> pd.DataFrame:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    candidates = candidates or DEFAULT_MODELS
    grids = grids or {}
    X_all = samples.subset(features) if features else samples.X
    y_all = samples.y
    g_all = spatial_blocks(samples, spatial_block_m)
    hold = holdout_mask(y_all, g_all, holdout_fraction, seed) if holdout_fraction else np.zeros(len(y_all), bool)
    dev = ~hold
    X, y, poly, groups = X_all[dev], y_all[dev], samples.polygon[dev], g_all[dev]
    rows, details, reference = [], {}, {}
    for name, params in candidates.items():
        missing = _available(params)
        if missing:
            rows.append({"model": name, "status": f"skipped: {missing}"})
            continue
        runs = []
        for r in range(repeats):
            s = seed + 101 * r
            t0 = time.perf_counter()
            if name in grids:
                cv = nested_cross_validate(X, y, poly, groups, params, grids[name], n_splits, 3, s)
                oof = None
            else:
                cv = cross_validate(X, y, poly, groups, params, n_splits, s)
                oof = cv.pop("_oof")
                cv.pop("_oof_proba"), cv.pop("_classes")
            cv_seconds = time.perf_counter() - t0
            t0 = time.perf_counter()
            est = make_estimator(params, s).fit(X, y)
            fit_seconds = time.perf_counter() - t0
            probe = X[: min(len(X), 20000)]
            t0 = time.perf_counter()
            est.predict_proba(probe)
            px_per_s = len(probe) / max(time.perf_counter() - t0, 1e-9)
            run = {"seed": s, "overall_accuracy": cv["overall_accuracy"], "kappa": cv["kappa"],
                   "macro_f1": cv["macro_f1"], "cv_seconds": cv_seconds, "fit_seconds": fit_seconds,
                   "predict_pixels_per_second": px_per_s}
            for t in targets:
                run[f"f1_{t}"] = cv["per_class"].get(t, {}).get("f1")
            if r == 0 and oof is not None:
                reference.setdefault("oof", (name, oof))
                run["mcnemar_p_vs_reference"] = metrics.mcnemar(y, oof, reference["oof"][1])["p_value"]
            if hold.any():
                pred = np.asarray(est.predict(X_all[hold])).astype(str)
                h = metrics.summary(y_all[hold], pred, labels=np.unique(y_all))
                run.update({"holdout_overall_accuracy": h["overall_accuracy"], "holdout_macro_f1": h["macro_f1"]})
                if r == 0:
                    reference.setdefault("hold", (name, pred))
                    run["holdout_mcnemar_p_vs_reference"] = metrics.mcnemar(y_all[hold], pred,
                                                                            reference["hold"][1])["p_value"]
            runs.append(run)
        frame = pd.DataFrame(runs)
        row = {"model": name, "status": "ok", "repeats": repeats, "tuned": name in grids}
        for col in frame.columns:
            if col == "seed":
                continue
            vals = frame[col].dropna().astype(float)
            if len(vals):
                row[col] = vals.mean()
                if col in ("overall_accuracy", "macro_f1", "holdout_macro_f1") and len(vals) > 1:
                    row[f"{col}_sd"] = vals.std(ddof=1)
        rows.append(row)
        details[name] = {"params": params, "grid": grids.get(name), "runs": runs}
    table = pd.DataFrame(rows)
    table.to_csv(out / "benchmark_models.csv", index=False)
    (out / "benchmark_models.json").write_text(json.dumps({
        "reference_model": reference.get("oof", (None,))[0], "n_splits": n_splits, "spatial_block_m": spatial_block_m,
        "holdout_fraction": holdout_fraction, "samples": int(len(y_all)), "details": details,
        "fold_signature": fold_signature(samples, spatial_block_m, n_splits, seed) if spatial_block_m else None,
        "note": "Same samples and spatial folds for every model; tuned models evaluated with nested CV."},
        indent=2, default=str), encoding="utf8")
    return table


# --------------------------------------------------------------------------- segmentation

def rasterize_boundaries(parcels: gpd.GeoDataFrame, like: str | Path) -> np.ndarray:
    """One-pixel boundary mask of polygon outlines on the grid of `like`."""
    import rasterio
    from rasterio.features import rasterize

    with rasterio.open(like) as ds:
        gdf = parcels.to_crs(ds.crs)
        lines = [(g.boundary, 1) for g in gdf.geometry if g is not None and not g.is_empty]
        return rasterize(lines, out_shape=ds.shape, transform=ds.transform, fill=0, all_touched=True,
                         dtype="uint8").astype(bool)


def segmentation(raster: str | Path, reference: str | Path, output_dir: str | Path, bands=None,
                 unet_model: str | Path | None = None, unet_bands=None, tile_size: int = 1024,
                 segments_per_km2: float = 50, tolerance_px: int = 1, iou_threshold: float = 0.5) -> pd.DataFrame:
    from .segmentation import boundaries_to_parcels, obia, unet_infer
    from .segmentation_metrics import boundary_scores, object_scores

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ref = gpd.read_file(reference)
    ref_mask = rasterize_boundaries(ref, raster)
    methods = {
        "obia_tiles": lambda p: obia(raster, p, bands, tile_size, segments_per_km2, merge_seams=False),
        "obia_seam_merged": lambda p: obia(raster, p, bands, tile_size, segments_per_km2, merge_seams=True),
    }
    if unet_model:
        def unet(p):
            prob = unet_infer(raster, unet_model, out / "unet_boundary.tif", bands=unet_bands)
            return boundaries_to_parcels(prob, p)
        methods["unet_watershed"] = unet
    rows = []
    for name, run in methods.items():
        path = out / f"{name}.gpkg"
        t0 = time.perf_counter()
        try:
            info = run(path)
        except (ValueError, RuntimeError) as exc:      # e.g. no objects found: record, keep benchmarking
            rows.append({"method": name, "status": f"failed: {exc}", "seconds": time.perf_counter() - t0})
            continue
        seconds = time.perf_counter() - t0
        seg = gpd.read_file(path)
        obj = object_scores(seg, ref, iou_threshold)
        bnd = boundary_scores(rasterize_boundaries(seg, raster), ref_mask, tolerance_px)
        rows.append({"method": name, "status": "ok", "segments": info["segments"], "seconds": seconds,
                     **{k: obj[k] for k in ("over_segmentation_mean", "under_segmentation_mean", "D_mean",
                                            "object_precision", "object_recall", "object_f1", "matched_iou_mean")},
                     **{k: bnd[k] for k in ("boundary_precision", "boundary_recall", "boundary_f1")},
                     "tolerance_px": tolerance_px})
    table = pd.DataFrame(rows)
    table.to_csv(out / "benchmark_segmentation.csv", index=False)
    return table


def fold_signature(samples: Samples, spatial_block_m: float, n_splits: int, seed: int) -> str:
    """Hash of the fold assignment, recorded so that benchmark runs can prove they used identical folds."""
    import hashlib
    groups = spatial_blocks(samples, spatial_block_m)
    h = hashlib.sha256()
    for k, (_, te) in enumerate(folds(samples.y, groups, n_splits, seed)):
        h.update(f"{k}:{','.join(map(str, te[:2000]))}".encode())
    return h.hexdigest()[:16]
