"""Benchmarks on identical sampled support and spatial folds.

`run`          strict internal model-selection report (single seed, sample/fold fingerprints, failures kept).
`models`       extended comparison: repeated seeds, nested tuning, locked holdout, timing, McNemar tests.
`segmentation` OBIA (with/without seam reconciliation) vs U-Net + watershed against reference parcels.

These are internal diagnostics, not evidence of external accuracy or superiority over other software.
"""
from __future__ import annotations

import hashlib
import json
import platform
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import sklearn

from . import metrics
from .estimators import (
    available_algorithms,
    make_estimator,
    predict_labels,
    sequence_params,
)
from .model import _fold_plan, cross_validate, fold_plan_signature, selection_samples, spatial_blocks
from .samples import Samples, check_classes


def _digest_array(values) -> str:
    values = np.ascontiguousarray(values)
    h = hashlib.sha256()
    h.update(str(values.dtype).encode())
    h.update(str(values.shape).encode())
    if values.dtype.hasobject:
        h.update(json.dumps(values.tolist(), ensure_ascii=False).encode())
    else:
        h.update(memoryview(values).cast("B"))
    return h.hexdigest()


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value.item() if isinstance(value, np.generic) else value


def run(samples: Samples, output_dir: str | Path,
        candidates: dict[str, dict] | None = None, n_splits: int = 5,
        spatial_block_m: float | None = 5000, seed: int = 42,
        spatial_buffer_m: float = 0, calibration_mode: str = "oof", calibration_fraction: float = 0.2,
        holdout_fraction: float = 0.0, allow_missing_classes: bool = False) -> dict:
    """Run named parameter sets on the same folds, persisting successful and failed outcomes.

    The locked final holdout (`holdout_fraction`, same deterministic split as `train`) is removed before
    the calibration split, so the report never selects a model on objects later used as holdout evidence.

    Optional-model dependency failures remain visible as failed candidates. A partial run
    has status ``incomplete``; callers must inspect it instead of selecting from only the
    survivors. No winner is chosen automatically and no mapping model is fitted afterward.
    """
    if candidates is None:
        candidates = {a: {"algorithm": a, "n_estimators": 100, "n_jobs": 1,
                           "class_weight": "balanced_subsample"}
                      for a in ("rf", "extra_trees", "hist_gb")}
    if not candidates:
        raise ValueError("Supply at least one benchmark candidate")
    for name, params in candidates.items():
        if not isinstance(name, str) or not name.strip() or not isinstance(params, dict):
            raise ValueError("Candidates must map nonempty names to estimator parameter dictionaries")
        if params.get("algorithm", "rf") not in available_algorithms():
            raise ValueError(f"Unknown algorithm in candidate {name!r}")
    from .model import split_final_holdout

    missing_classes = check_classes(samples, allow_missing_classes, "benchmark")
    t_run = time.perf_counter()
    input_count = len(samples.y)
    samples, final_test = split_final_holdout(samples, spatial_block_m, holdout_fraction, seed)
    samples, calibration_split = selection_samples(samples, spatial_block_m, seed, spatial_buffer_m,
                                                   calibration_mode, calibration_fraction)
    groups = spatial_blocks(samples, spatial_block_m)
    spatial = {"polygon": samples.polygon, "coordinates": np.column_stack([samples.x, samples.yy]),
               "crs": samples.crs, "spatial_buffer_m": spatial_buffer_m}
    # Validate the shared design before doing costly candidate fits or creating a report.
    plan = _fold_plan(samples.y, groups, n_splits, seed, **spatial)
    identity = {key: _digest_array(getattr(samples, key)) for key in ("X", "y", "polygon", "x", "yy")}
    identity.update({"features": samples.features, "crs": samples.crs})
    sample_fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    fold_records = [{"fold": k, **diagnostics, "n_test": len(te),
                     "train_index_sha256": _digest_array(tr), "test_index_sha256": _digest_array(te)}
                    for k, (tr, te, diagnostics) in enumerate(plan)]
    fold_fingerprint = hashlib.sha256(json.dumps(fold_records, sort_keys=True).encode()).hexdigest()
    rows, results = [], {}
    for name, raw in candidates.items():
        params = dict(raw)
        started = time.perf_counter()
        try:
            params = sequence_params(params, samples.features)
            cv = cross_validate(samples.X, samples.y, samples.polygon, groups, params, n_splits, seed,
                                coordinates=spatial["coordinates"], crs=samples.crs,
                                spatial_buffer_m=spatial_buffer_m)
            for key in ("_oof", "_oof_proba", "_classes"):
                cv.pop(key)
            row = {"candidate": name, "algorithm": params.get("algorithm", "rf"), "status": "ok",
                   **{k: cv[k] for k in ("overall_accuracy", "macro_f1", "kappa", "fold_macro_f1_std")},
                   "ece": cv["calibration"]["ece"]}
            results[name] = {"status": "ok", "params": params, "cv": cv}
        except Exception as exc:  # Candidate/backend failures are report outcomes; interrupts still propagate.
            row = {"candidate": name, "algorithm": params.get("algorithm", "rf"), "status": "failed",
                   "error": f"{type(exc).__name__}: {exc}"}
            results[name] = {"status": "failed", "params": params, "error": row["error"]}
        row["seconds"] = round(time.perf_counter() - started, 6)
        rows.append(row)
    report = {"schema": "geocrop-benchmark/1", "status": "complete" if all(
        r["status"] == "ok" for r in rows) else "incomplete", "internal_only": True,
        "note": "Internal model-selection diagnostics on supplied reference samples. No external accuracy or "
                "cross-engine superiority is established. Timings include CV fit/prediction on this machine.",
        "samples": len(samples.y), "input_samples": input_count, "features": samples.features,
        "reference_classes_without_samples": missing_classes,
        "calibration_mode": calibration_mode, "calibration_split": calibration_split,
        "holdout_fraction": holdout_fraction,
        "holdout_samples_excluded": 0 if final_test is None else int(len(final_test.y)),
        "sample_fingerprint": sample_fingerprint,
        "fold_fingerprint": fold_fingerprint, "folds": fold_records,
        "seed": seed, "requested_folds": n_splits, "actual_folds": len(plan),
        "spatial_block_m": spatial_block_m, "spatial_buffer_m": spatial_buffer_m,
        "runtime": {"python": platform.python_version(), "platform": platform.platform(),
                    "sklearn": sklearn.__version__, "numpy": np.__version__},
        "results": results, "summary": rows}
    report = _json_safe(report)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "benchmark.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf8")
    pd.DataFrame(rows).to_csv(out / "benchmark.csv", index=False)
    from . import registry
    registry.record_classification("benchmark", out, rows, samples, n_splits=n_splits, seeds=[seed],
                                   spatial_block_m=spatial_block_m, spatial_buffer_m=spatial_buffer_m,
                                   holdout=final_test, started=t_run,
                                   protocol={"calibration_mode": calibration_mode, "holdout_fraction": holdout_fraction})
    return report


# --------------------------------------------------------------------------- extended comparison

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

#: Temporal deep models (need `ts` features only; torch). Not in the default comparison; select by name.
SEQUENCE_MODELS = {
    "tempcnn": {"algorithm": "tempcnn", "epochs": 60, "patience": 10},
    "ltae": {"algorithm": "ltae", "epochs": 100, "patience": 10},
}


def candidate(name: str) -> dict:
    """Default settings for a model name: built-in defaults, temporal models, else any registered algorithm."""
    if name in DEFAULT_MODELS:
        return dict(DEFAULT_MODELS[name])
    if name in SEQUENCE_MODELS:
        return dict(SEQUENCE_MODELS[name])
    if name in available_algorithms():
        return {"algorithm": name}
    raise ValueError(f"Unknown model {name!r}; choose from {available_algorithms()}")


def _available(params: dict) -> str | None:
    from .estimators import backend_error

    reason = backend_error(params.get("algorithm", "rf"))
    if reason:
        return reason
    try:
        make_estimator(params, 0)
        return None
    except RuntimeError as exc:
        return str(exc)


def models(samples: Samples, output_dir: str | Path, features: list[str] | None = None, candidates: dict | None = None,
           grids: dict | None = None, n_splits: int = 5, spatial_block_m: float | None = 5000, seed: int = 42,
           repeats: int = 3, holdout_fraction: float = 0.0, targets=(), spatial_buffer_m: float = 0,
           calibration_mode: str = "oof", calibration_fraction: float = 0.2,
           allow_missing_classes: bool = False) -> pd.DataFrame:
    """Extended comparison: repeated seeds, nested tuning, locked holdout, timing and McNemar tests.

    Uses only the objects allowed for model selection under the declared calibration design
    (`selection_samples`) and the same optional spatial buffer as training. A sample set that lost a whole
    reference class is refused (`samples.check_classes`) unless `allow_missing_classes`.

    Every run records the signature of the folds it actually used (``fold_signature`` per run and
    ``fold_signatures`` per model in the JSON report); identical signatures prove identical folds."""
    from .model import nested_cross_validate, split_final_holdout, tune

    missing_classes = check_classes(samples, allow_missing_classes, "benchmark-models")

    t_run = time.perf_counter()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    candidates = candidates or DEFAULT_MODELS
    grids = grids or {}
    samples, test = split_final_holdout(samples, spatial_block_m, holdout_fraction, seed)
    samples, calibration_split = selection_samples(samples, spatial_block_m, seed, spatial_buffer_m,
                                                   calibration_mode, calibration_fraction)
    X = samples.subset(features) if features else samples.X
    y, poly, groups = samples.y, samples.polygon, spatial_blocks(samples, spatial_block_m)
    spatial = {"coordinates": np.column_stack([samples.x, samples.yy]), "crs": samples.crs,
               "spatial_buffer_m": spatial_buffer_m}
    hold = test is not None
    rows, details, reference = [], {}, {}
    names = features or samples.features
    for name, params in candidates.items():
        try:
            params = sequence_params(params, names)
        except ValueError as exc:      # e.g. a temporal model on non-`ts` features
            rows.append({"model": name, "status": f"skipped: {exc}"})
            continue
        missing = _available(params)
        if missing:
            rows.append({"model": name, "status": f"skipped: {missing}"})
            continue
        runs = []
        for r in range(repeats):
            s = seed + 101 * r
            t0 = time.perf_counter()
            if name in grids:
                cv = nested_cross_validate(X, y, poly, groups, params, grids[name], n_splits, 3, s, **spatial)
                oof = None
            else:
                cv = cross_validate(X, y, poly, groups, params, n_splits, s, **spatial)
                oof = cv.pop("_oof")
                cv.pop("_oof_proba"), cv.pop("_classes")
            cv_seconds = time.perf_counter() - t0
            fit_params = params
            if name in grids:      # select on development data only, then fit the selected configuration
                chosen = tune(X, y, poly, groups, params, grids[name], n_splits, s, **spatial)["best_params"]
                fit_params = {**params, **chosen}
                run_selected = chosen
            else:
                run_selected = None
            t0 = time.perf_counter()
            est = make_estimator(fit_params, s).fit(X, y)
            fit_seconds = time.perf_counter() - t0
            probe = X[: min(len(X), 20000)]
            t0 = time.perf_counter()
            est.predict_proba(probe)
            px_per_s = len(probe) / max(time.perf_counter() - t0, 1e-9)
            run = {"seed": s, "fold_signature": cv.get("fold_signature"), "selected_params": run_selected,
                   "overall_accuracy": cv["overall_accuracy"], "kappa": cv["kappa"],
                   "macro_f1": cv["macro_f1"], "cv_seconds": cv_seconds, "fit_seconds": fit_seconds,
                   "predict_pixels_per_second": px_per_s}
            for t in targets:
                run[f"f1_{t}"] = cv["per_class"].get(t, {}).get("f1")
            if r == 0 and oof is not None:
                reference.setdefault("oof", (name, oof))
                paired = metrics.grouped_mcnemar(y, oof, reference["oof"][1], poly)
                run["mcnemar_p_vs_reference"] = paired["p_value"]
                run["mcnemar_n_units"] = paired["n_units"]
            if hold:
                Xt = test.subset(features) if features else test.X
                pred = predict_labels(est, Xt).astype(str)
                h = metrics.summary(test.y, pred, labels=np.unique(np.concatenate([y, test.y])))
                run.update({"holdout_overall_accuracy": h["overall_accuracy"], "holdout_macro_f1": h["macro_f1"]})
                if r == 0:
                    reference.setdefault("hold", (name, pred))
                    paired = metrics.grouped_mcnemar(test.y, pred, reference["hold"][1], test.polygon)
                    run["holdout_mcnemar_p_vs_reference"] = paired["p_value"]
                    run["holdout_mcnemar_n_units"] = paired["n_units"]
            runs.append(run)
        frame = pd.DataFrame(runs).drop(columns=["fold_signature"])
        row = {"model": name, "status": "ok", "repeats": repeats, "tuned": name in grids}
        for col in frame.columns:
            if col in ("seed", "selected_params"):
                continue
            vals = frame[col].dropna().astype(float)
            if len(vals):
                row[col] = vals.mean()
                if col in ("overall_accuracy", "macro_f1", "holdout_macro_f1") and len(vals) > 1:
                    row[f"{col}_sd"] = vals.std(ddof=1)
        row["mcnemar_unit"] = "reference_polygon"
        rows.append(row)
        details[name] = {"params": params, "grid": grids.get(name), "runs": runs,
                         "fold_signatures": [r["fold_signature"] for r in runs]}
    for column, ref_key in (("mcnemar_p_vs_reference", "oof"), ("holdout_mcnemar_p_vs_reference", "hold")):
        baseline = reference.get(ref_key, (None,))[0]
        comparisons = [row for row in rows if column in row and row["model"] != baseline]
        for row, adjusted in zip(comparisons, metrics.holm_adjust([row[column] for row in comparisons])):
            row[column + "_holm"] = adjusted
            details[row["model"]]["runs"][0][column + "_holm"] = adjusted
        if baseline in details:
            details[baseline]["runs"][0][column + "_holm"] = 1.0
            next(row for row in rows if row["model"] == baseline)[column + "_holm"] = 1.0
    table = pd.DataFrame(rows)
    table.to_csv(out / "benchmark_models.csv", index=False)
    (out / "benchmark_models.json").write_text(json.dumps({
        "reference_model": reference.get("oof", (None,))[0], "n_splits": n_splits, "spatial_block_m": spatial_block_m,
        "holdout_reference_model": reference.get("hold", (None,))[0],
        "paired_comparison": {"unit": "reference_polygon", "aggregation": "majority_vote",
                              "tie_rule": "lexicographic", "repeat": "first repeat only",
                              "adjustment": "Holm against baseline, excluding self; OOF and holdout families separate"},
        "holdout_fraction": holdout_fraction, "samples": int(len(y)), "details": details,
        "spatial_buffer_m": spatial_buffer_m, "calibration_mode": calibration_mode,
        "calibration_split": calibration_split,
        "fold_signature": fold_signature(samples, spatial_block_m, n_splits, seed) if spatial_block_m else None,
        "fold_signatures_per_model": {m: d["fold_signatures"] for m, d in details.items()},
        "identical_folds_across_models": len({tuple(d["fold_signatures"]) for d in details.values()}) <= 1,
        "reference_classes_without_samples": missing_classes,
        "note": "Same samples and spatial folds for every model; tuned models evaluated with nested CV."},
        indent=2, default=str), encoding="utf8")
    from . import registry
    registry.record_classification("benchmark-models", out, rows, samples, n_splits=n_splits,
                                   seeds=[seed + 101 * r for r in range(repeats)], spatial_block_m=spatial_block_m,
                                   spatial_buffer_m=spatial_buffer_m, holdout=test, candidate_key="model",
                                   started=t_run, protocol={"repeats": repeats, "tuned": sorted(grids),
                                                            "features": len(names)})
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
                 segments_per_km2: float = 50, tolerance_px: int = 1, iou_threshold: float = 0.5,
                 methods: dict[str, dict] | None = None) -> pd.DataFrame:
    """Score segmenters from the registry (`geocrop.segmenters`) against reference parcels.

    Default methods: OBIA with and without seam merging, plus U-Net + watershed when `unet_model` is given.
    `methods` adds (or overrides) entries ``{label: {"segmenter": <registered name>, **params}}``, e.g. a
    plugin segmenter, so new algorithms are benchmarked on identical metrics without code changes.
    """
    from . import segmenters
    from .segmentation_metrics import boundary_scores, object_scores

    t_run = time.perf_counter()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ref = gpd.read_file(reference)
    ref_mask = rasterize_boundaries(ref, raster)
    obia = {"segmenter": "obia", "bands": bands, "tile_size": tile_size, "segments_per_km2": segments_per_km2}
    specs = {"obia_tiles": {**obia, "merge_seams": False}, "obia_seam_merged": {**obia, "merge_seams": True}}
    if unet_model:
        specs["unet_watershed"] = {"segmenter": "unet_watershed", "model": unet_model, "bands": unet_bands,
                                   "boundary_output": out / "unet_boundary.tif"}
    specs.update(methods or {})
    methods = {}
    for label, spec in specs.items():
        params = dict(spec)
        segmenter = segmenters.get(params.pop("segmenter"))
        methods[label] = lambda p, s=segmenter, kw=params: s.segment(raster, p, **kw)
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
    from . import registry
    registry.record_segmentation(out, rows, raster, reference, tolerance_px=tolerance_px, iou_threshold=iou_threshold,
                                 started=t_run)
    return table


def fold_signature(samples: Samples, spatial_block_m: float, n_splits: int, seed: int) -> str:
    """Hash of the fold assignment, recorded so that benchmark runs can prove they used identical folds."""
    groups = spatial_blocks(samples, spatial_block_m)
    plan = _fold_plan(samples.y, groups, n_splits, seed)
    return fold_plan_signature([te for _, te, _ in plan])
