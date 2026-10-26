"""Pixel classifier: spatially blocked cross-validation, tuning, final fit and blockwise prediction.

Leakage control: all pixels of a reference polygon share one group, and polygons are grouped
into square spatial blocks (default 5 km), so no block contributes to both training and
testing in any fold. Folds are stratified by class where the data allow
(StratifiedGroupKFold). Cross-validated scores are *internal* diagnostics; external accuracy
must come from independent field observations (see `geocrop.validation`).
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import sklearn
from joblib import dump, load
from pyproj import CRS
from scipy.spatial import cKDTree
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

from . import applicability, contracts, metrics, uncertainty
from .estimators import channels_from_names, importances, make_estimator, predict_labels
from .hashing import sha256
from .raster import NODATA, blocks, gtiff_profile, valid_mask
from .samples import Samples

RF_KEYS = ("algorithm", "n_estimators", "max_features", "min_samples_leaf", "max_depth", "class_weight", "n_jobs",
           "max_samples", "learning_rate", "num_leaves", "epochs", "n_channels")


def spatial_blocks(samples: Samples, block_m: float | None) -> np.ndarray:
    """Block id per sample; every polygon is assigned to the block containing its pixel centroid."""
    if not block_m:
        return samples.polygon.copy()
    crs = CRS.from_user_input(samples.crs)
    if not crs.is_projected or "met" not in (crs.axis_info[0].unit_name or "").lower():
        raise ValueError("Spatial blocks require a projected CRS in metres (e.g. UTM)")
    frame = pd.DataFrame({"p": samples.polygon, "x": samples.x, "y": samples.yy})
    centre = frame.groupby("p")[["x", "y"]].mean()
    block = (np.floor(centre.x / block_m).astype(int).astype(str) + ":" +
             np.floor(centre.y / block_m).astype(int).astype(str))
    return block.reindex(samples.polygon).to_numpy()


def make_rf(params: dict, seed: int):
    """Estimator for `params["algorithm"]` (default Random Forest); see `geocrop.estimators`."""
    return make_estimator(params, seed)


def _fold_plan(y, groups, n_splits: int, seed: int, *, polygon=None, coordinates=None, crs=None,
               spatial_buffer_m: float = 0):
    """Build deterministic folds, optionally purging whole training polygons near test samples.

    Distances refer to sampled pixel centres, not unsampled polygon boundaries. Test sets stay
    unchanged. A buffer therefore cannot establish separation of complete polygon geometries.
    """
    if not np.isfinite(spatial_buffer_m) or spatial_buffer_m < 0:
        raise ValueError("spatial_buffer_m must be finite and non-negative")
    y, groups = np.asarray(y), np.asarray(groups)
    points = None
    if coordinates is not None:
        points = np.asarray(coordinates, dtype=float)
        if points.shape != (len(y), 2) or not np.isfinite(points).all():
            raise ValueError("coordinates must be a finite (n_samples, 2) array")
    metric_crs = False
    if crs is not None:
        parsed = CRS.from_user_input(crs)
        metric_crs = parsed.is_projected and all(
            np.isclose(axis.unit_conversion_factor, 1.0) for axis in parsed.axis_info[:2])
    if spatial_buffer_m:
        if points is None or polygon is None:
            raise ValueError("Spatial buffering needs coordinates and polygon IDs")
        if not metric_crs:
            raise ValueError("Spatial buffering requires a projected CRS in metres")
    ids = np.asarray(polygon) if polygon is not None else np.arange(len(y))
    if ids.shape != (len(y),):
        raise ValueError("polygon IDs must have one value per sample")
    n_groups = len(np.unique(groups))
    if n_groups < 2:
        raise ValueError("Cross-validation needs at least two spatial groups")
    splitter = StratifiedGroupKFold(n_splits=min(n_splits, n_groups), shuffle=True, random_state=seed)
    out = []
    for k, (tr, te) in enumerate(splitter.split(np.zeros(len(y)), y, groups)):
        if set(groups[tr]) & set(groups[te]):
            raise AssertionError("Spatial group leakage between folds")
        original = tr.copy()
        tree = cKDTree(points[te]) if points is not None and metric_crs else None
        if spatial_buffer_m:
            distances, _ = tree.query(points[tr], k=1)
            purged = np.unique(ids[tr][distances <= spatial_buffer_m])
            tr = tr[~np.isin(ids[tr], purged)]
            if len(tr) == 0:
                raise ValueError(f"Spatial buffer {spatial_buffer_m:g} m leaves no training samples in fold {k}")
            lost = set(y[original]) - set(y[tr])
            if lost:
                raise ValueError(f"Spatial buffer removes training classes in fold {k}: {sorted(lost)}")
        diagnostics = {
            "n_train": int(len(tr)), "buffer_excluded_samples": int(len(original) - len(tr)),
            "buffer_excluded_polygons": int(len(set(ids[original]) - set(ids[tr]))),
            "min_train_test_distance_m": (float(tree.query(points[tr], k=1)[0].min())
                                          if tree is not None else None),
        }
        out.append((tr, te, diagnostics))
    return out


def folds(y, groups, n_splits: int, seed: int, *, polygon=None, coordinates=None, crs=None,
          spatial_buffer_m: float = 0):
    """Spatial-group folds; an optional metric buffer purges whole nearby training polygons.

    The buffer applies to sampled pixel centres only. Held-out samples are never dropped.
    """
    return [(tr, te) for tr, te, _ in _fold_plan(
        y, groups, n_splits, seed, polygon=polygon, coordinates=coordinates, crs=crs,
        spatial_buffer_m=spatial_buffer_m)]


def cross_validate(X, y, polygon, groups, params: dict, n_splits: int = 5, seed: int = 42,
                   bootstrap: int = 0, *, coordinates=None, crs=None, spatial_buffer_m: float = 0) -> dict:
    """Out-of-fold predictions across spatial folds; metrics pooled over all folds plus per-fold spread."""
    oof = np.empty(len(y), dtype=object)
    oof_proba = None
    per_fold = []
    classes = np.unique(y)
    for k, (tr, te, diagnostics) in enumerate(_fold_plan(
            y, groups, n_splits, seed, polygon=polygon, coordinates=coordinates, crs=crs,
            spatial_buffer_m=spatial_buffer_m)):
        rf = make_rf(params, seed + k).fit(X[tr], y[tr])
        oof[te] = predict_labels(rf, X[te])
        proba = np.zeros((len(te), len(classes)), dtype="float32")
        proba[:, np.searchsorted(classes, rf.classes_)] = rf.predict_proba(X[te])
        if oof_proba is None:
            oof_proba = np.zeros((len(y), len(classes)), dtype="float32")
        oof_proba[te] = proba
        s = metrics.summary(y[te], oof[te], labels=classes)
        per_fold.append({"fold": k, **diagnostics, "n_test": int(len(te)), "test_groups": int(len(np.unique(groups[te]))),
                         "missing_train_classes": sorted(set(classes) - set(rf.classes_)),
                         "overall_accuracy": s["overall_accuracy"], "macro_f1": s["macro_f1"], "kappa": s["kappa"]})
    pooled = metrics.summary(y, oof.astype(str), labels=classes)
    pooled["_oof"] = oof.astype(str)   # out-of-fold predictions and probabilities; removed before JSON export
    pooled["_oof_proba"] = oof_proba
    pooled["_classes"] = [str(c) for c in classes]
    pooled["folds"] = per_fold
    pooled["spatial_buffer_m"] = spatial_buffer_m
    pooled["distance_support"] = "sampled pixel centres; not full polygon geometries"
    pooled["fold_macro_f1_mean"] = float(np.mean([f["macro_f1"] for f in per_fold]))
    pooled["fold_macro_f1_std"] = float(np.std([f["macro_f1"] for f in per_fold]))
    pooled["calibration"] = metrics.calibration(oof_proba, y, classes)
    if bootstrap:
        pooled["bootstrap"] = metrics.bootstrap(y, oof.astype(str), groups=polygon, n_boot=bootstrap, seed=seed)
    pooled["note"] = "Internal spatial cross-validation on interpreted reference polygons; not independent field validation."
    return pooled


def tune(X, y, polygon, groups, base: dict, grid: dict, n_splits: int = 5, seed: int = 42, *,
         coordinates=None, crs=None, spatial_buffer_m: float = 0) -> dict:
    """Exhaustive grid search scored by spatial-CV macro-F1. The winning score is optimistically biased
    (selected on the same folds) and must not be reported as final accuracy."""
    keys = list(grid)
    trials = []
    for values in itertools.product(*(grid[k] for k in keys)):
        params = {**base, **dict(zip(keys, values))}
        cv = cross_validate(X, y, polygon, groups, params, n_splits, seed, coordinates=coordinates,
                            crs=crs, spatial_buffer_m=spatial_buffer_m)
        trials.append({**dict(zip(keys, values)), "macro_f1": cv["macro_f1"], "overall_accuracy": cv["overall_accuracy"]})
    best = max(trials, key=lambda t: t["macro_f1"])
    return {"best_params": {k: best[k] for k in keys}, "trials": trials}


def holdout_mask(y, groups, fraction: float, seed: int = 42) -> np.ndarray:
    """Lock away ~`fraction` of the spatial groups (class-stratified) as a final test set."""
    if not 0 < fraction < 0.5:
        raise ValueError("holdout fraction must be in (0, 0.5)")
    k = max(2, int(round(1 / fraction)))
    splitter = StratifiedGroupKFold(n_splits=min(k, len(np.unique(groups))), shuffle=True, random_state=seed + 1000)
    _, te = next(splitter.split(np.zeros(len(y)), y, groups))
    mask = np.zeros(len(y), dtype=bool)
    mask[te] = True
    return mask


def nested_cross_validate(X, y, polygon, groups, base: dict, grid: dict, n_outer: int = 5, n_inner: int = 3,
                          seed: int = 42, *, coordinates=None, crs=None, spatial_buffer_m: float = 0) -> dict:
    """Unbiased estimate of the *tuned* procedure: hyperparameters are selected by an inner spatial CV that
    sees only the outer training blocks; the outer test blocks are predicted once (Varma & Simon, 2006).
    The same optional spatial buffer is applied in the outer and inner folds."""
    classes = np.unique(y)
    oof = np.empty(len(y), dtype=object)
    chosen = []
    xy = None if coordinates is None else np.asarray(coordinates)
    for k, (tr, te) in enumerate(folds(y, groups, n_outer, seed, polygon=polygon, coordinates=xy, crs=crs,
                                       spatial_buffer_m=spatial_buffer_m)):
        inner = tune(X[tr], y[tr], polygon[tr], groups[tr], base, grid, n_inner, seed + k,
                     coordinates=None if xy is None else xy[tr], crs=crs, spatial_buffer_m=spatial_buffer_m)
        params = {**base, **inner["best_params"]}
        oof[te] = predict_labels(make_estimator(params, seed + k).fit(X[tr], y[tr]), X[te])
        chosen.append({"fold": k, **inner["best_params"]})
    s = metrics.summary(y, oof.astype(str), labels=classes)
    s["selected_params_per_fold"] = chosen
    s["note"] = "Nested spatial CV: outer-fold scores of the full tuning procedure (not optimistic)."
    return s


def split_final_holdout(samples: Samples, spatial_block_m: float | None, fraction: float, seed: int):
    """(development, locked test or None): the single definition of the final holdout used by train, ablation
    and both benchmarks, so that model selection never sees the objects later reported as holdout accuracy."""
    if not fraction:
        return samples, None
    mask = holdout_mask(samples.y, spatial_blocks(samples, spatial_block_m), fraction, seed)
    return _subset(samples, ~mask), _subset(samples, mask)


def _subset(samples: Samples, mask) -> Samples:
    return Samples(samples.X[mask], samples.y[mask], samples.polygon[mask], samples.x[mask], samples.yy[mask],
                   samples.features, samples.crs)


def _calibration_split(samples: Samples, groups, fraction: float, seed: int, spatial_buffer_m: float):
    """One deterministic group holdout; its labels never select the split or model settings.

    An unsuitable split fails rather than searching for a more favourable calibration sample.
    Buffers apply to sampled support and purge complete proper-training polygons.
    """
    if not np.isfinite(fraction) or not 0 < fraction < 1:
        raise ValueError("calibration_fraction must be finite and between zero and one")
    if not np.isfinite(spatial_buffer_m) or spatial_buffer_m < 0:
        raise ValueError("spatial_buffer_m must be finite and non-negative")
    n_groups = len(np.unique(groups))
    if n_groups < 3:
        raise ValueError("Holdout calibration needs at least three spatial groups (two for training CV)")
    if n_groups - int(np.ceil(fraction * n_groups)) < 2:
        raise ValueError("Holdout calibration leaves fewer than two proper-training groups; reduce fraction or add data")
    splitter = GroupShuffleSplit(n_splits=1, test_size=fraction, random_state=seed)
    fit, calibration = next(splitter.split(samples.X, groups=groups))
    original = fit.copy()
    if spatial_buffer_m:
        crs = CRS.from_user_input(samples.crs)
        if not crs.is_projected or not all(np.isclose(a.unit_conversion_factor, 1.0) for a in crs.axis_info[:2]):
            raise ValueError("Spatial buffering requires a projected CRS in metres")
        xy = np.column_stack([samples.x, samples.yy])
        if not np.isfinite(xy).all():
            raise ValueError("Spatial buffering needs finite sample coordinates")
        near = cKDTree(xy[calibration]).query(xy[fit], k=1)[0] <= spatial_buffer_m
        purge = np.unique(samples.polygon[fit][near])
        fit = fit[~np.isin(samples.polygon[fit], purge)]
    if len(np.unique(groups[fit])) < 2:
        raise ValueError("Holdout calibration leaves fewer than two proper-training groups; reduce fraction/buffer or add data")
    missing = sorted(set(samples.y) - set(samples.y[fit]))
    if missing:
        raise ValueError(f"Proper-training split lacks classes {missing}; add class support or change the declared split design")
    excluded = np.setdiff1d(original, fit)
    metadata = {
        "mode": "holdout", "seed": seed, "fraction_of_groups": fraction,
        "fit_samples": int(len(fit)), "calibration_samples": int(len(calibration)),
        "buffer_excluded_samples": int(len(excluded)),
        "fit_polygon_ids": sorted(set(samples.polygon[fit].astype(str))),
        "calibration_polygon_ids": sorted(set(samples.polygon[calibration].astype(str))),
        "buffer_excluded_polygon_ids": sorted(set(samples.polygon[excluded].astype(str))),
        "fit_groups": sorted(set(np.asarray(groups)[fit].astype(str))),
        "calibration_groups": sorted(set(np.asarray(groups)[calibration].astype(str))),
        "distance_support": "sampled pixel centres; not full polygon geometries",
    }
    if set(metadata["fit_polygon_ids"]) & set(metadata["calibration_polygon_ids"]):
        raise ValueError("Calibration split must keep every polygon in one spatial group")
    return fit, calibration, metadata


def selection_samples(samples: Samples, spatial_block_m: float | None = 5000, seed: int = 42,
                      spatial_buffer_m: float = 0, calibration_mode: str = "oof",
                      calibration_fraction: float = 0.2) -> tuple[Samples, dict | None]:
    """Return only the objects available for model/feature selection under the declared calibration design.

    Holdout mode reuses the exact deterministic spatial-group split used by ``train``. Callers
    must not use excluded/calibration objects to choose models, features, thresholds or settings.
    The input is never modified; OOF mode returns it unchanged after reference-label checks.
    """
    uncertainty.validate_polygon_labels(samples.y, samples.polygon)
    if calibration_mode not in ("oof", "holdout"):
        raise ValueError("calibration_mode must be oof or holdout")
    if calibration_mode == "oof":
        return samples, None
    groups = spatial_blocks(samples, spatial_block_m)
    fit, _, split = _calibration_split(samples, groups, calibration_fraction, seed, spatial_buffer_m)
    proper = Samples(samples.X[fit], samples.y[fit], samples.polygon[fit], samples.x[fit], samples.yy[fit],
                     samples.features, samples.crs)
    return proper, split


def train(samples: Samples, output_dir: str | Path, features: list[str] | None = None, params: dict | None = None,
          n_splits: int = 5, spatial_block_m: float | None = 5000, seed: int = 42, tune_grid: dict | None = None,
          bootstrap: int = 1000, provenance: dict | None = None, calibrate: bool = True,
          conformal_alphas=(0.05, 0.1, 0.2), spatial_buffer_m: float = 0,
          calibration_mode: str = "oof", calibration_fraction: float = 0.2, holdout_fraction: float = 0.0,
          refit_all: bool = True, inner_splits: int = 3) -> dict:
    """Spatial CV (optionally after tuning), then fit the mapping model on its proper-training samples.

    From the out-of-fold probabilities it also fits a temperature (calibration), empirical
    conformal-style thresholds on polygon probabilities (`conformal.json`) and lists suspicious labels.
    Default ``oof`` uses exploratory OOF thresholds. Opt-in ``holdout`` reserves spatial groups
    before tuning/CV, fits temperature on proper-training OOF predictions, then calibrates sets
    using a frozen proper-training predictor. It never refits that predictor on calibration data.
    Spatial grouping does not establish target exchangeability or observed parcel coverage.

    Further safeguards:
    * holdout_fraction > 0 first locks away whole spatial blocks as a final test set, before the
      calibration split, tuning or model comparison; it is scored once (`report["holdout"]`). In ``oof``
      mode the mapping model is then refitted on all samples (`refit_all`); in ``holdout`` calibration
      mode the predictor stays frozen so that the conformal thresholds remain bound to it.
    * With a tuning grid, `report["nested_cv"]` is the unbiased score of the tuning procedure; the
      re-run `cross_validation` with the selected parameters serves calibration and diagnostics only.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    features = features or samples.features
    samples, test = split_final_holdout(samples, spatial_block_m, holdout_fraction, seed)
    input_samples = samples
    samples, split = selection_samples(samples, spatial_block_m, seed, spatial_buffer_m,
                                       calibration_mode, calibration_fraction)
    groups = spatial_blocks(samples, spatial_block_m)
    calibration = None
    if split is not None:
        held_out = np.isin(input_samples.polygon.astype(str), split["calibration_polygon_ids"])
        calibration = Samples(input_samples.X[held_out], input_samples.y[held_out], input_samples.polygon[held_out],
                              input_samples.x[held_out], input_samples.yy[held_out], input_samples.features,
                              input_samples.crs)
    X = samples.subset(features)
    y = samples.y
    params = dict(params or {"n_estimators": 500, "max_features": "sqrt", "class_weight": "balanced_subsample", "n_jobs": -1})
    if params.get("algorithm") == "tempcnn" and "n_channels" not in params:
        params["n_channels"] = channels_from_names(features)
    report: dict = {"features": len(features), "samples": int(len(y)), "polygons": int(len(np.unique(samples.polygon))),
                    "spatial_groups": int(len(np.unique(groups))), "spatial_block_m": spatial_block_m,
                    "class_counts": pd.Series(y).value_counts().to_dict(),
                    "polygon_counts": pd.Series(samples.polygon).groupby(y).nunique().to_dict()}
    spatial = {"coordinates": np.column_stack([samples.x, samples.yy]), "crs": samples.crs,
               "spatial_buffer_m": spatial_buffer_m}
    report["spatial_buffer_m"] = spatial_buffer_m
    report["calibration_mode"] = calibration_mode
    if test is not None:
        report["holdout_design"] = {"fraction": holdout_fraction, "samples": int(len(test.y)),
                                    "polygons": int(len(np.unique(test.polygon))),
                                    "classes_missing": sorted(set(samples.y) - set(test.y))}
    if split is not None:
        report["input_samples"] = int(len(input_samples.y))
        report["calibration_split"] = split
    if tune_grid:
        report["nested_cv"] = nested_cross_validate(X, y, samples.polygon, groups, params, tune_grid, n_splits,
                                                    inner_splits, seed, **spatial)
        report["tuning"] = tune(X, y, samples.polygon, groups, params, tune_grid, n_splits, seed, **spatial)
        params.update(report["tuning"]["best_params"])
    report["params"] = params
    cv = cross_validate(X, y, samples.polygon, groups, params, n_splits, seed, bootstrap, **spatial)
    if tune_grid:
        cv["note"] += " Parameters were tuned on these folds: use nested_cv for performance."
    report["cross_validation"] = cv
    cv.pop("_oof")
    oof_proba, classes = cv.pop("_oof_proba"), cv.pop("_classes")
    temperature = 1.0
    if calibrate:
        temperature = uncertainty.fit_temperature(oof_proba, y, classes)
        oof_proba = uncertainty.apply_temperature(oof_proba, temperature)
        cv["calibration_after_temperature"] = metrics.calibration(oof_proba, y, classes)
    cv["temperature"] = temperature
    poly = uncertainty.polygon_oof(oof_proba, y, samples.polygon, classes)
    poly.to_csv(out / "oof_polygons.csv", index=False)
    suspects = uncertainty.suspect_labels(poly)
    model = make_estimator(params, seed).fit(X, y)
    if calibration is not None:
        cp = uncertainty.apply_temperature(model.predict_proba(calibration.subset(features)), temperature)
        poly = uncertainty.polygon_oof(cp, calibration.y, calibration.polygon, classes)
        poly.to_csv(out / "calibration_polygons.csv", index=False)
        (out / "calibration_split.json").write_text(json.dumps(split, indent=2), encoding="utf8")
    conformal = uncertainty.conformal_thresholds(
        poly[[f"p_{c}" for c in classes]].to_numpy(), poly.label, classes, conformal_alphas,
        sparse_class_policy="include" if calibration is not None else "pooled")
    conformal["level"] = "polygon (out-of-fold mean probabilities)"
    conformal["method"] = "empirical OOF conformal-style thresholds"
    conformal["coverage_guarantee"] = False
    conformal["coverage_evaluation"] = "same objects used to fit thresholds; not independent validation"
    conformal["note"] = ("OOF models differ from the final all-sample model, temperature uses the same labels, "
                         "and calibration objects are reused for coverage checks. No split-conformal "
                         "finite-sample coverage guarantee is established.")
    if calibration is not None:
        conformal.update({
            "level": "polygon (held-out spatial-group mean probabilities)",
            "method": "split conformal scores from a frozen proper-training predictor",
            "calibration_mode": "holdout",
            "temperature_fit": "proper-training OOF labels only; held-out calibration labels unused",
            "predictor_refit_after_calibration": False,
            "exchangeability_verified": False,
            "note": "The predictor and temperature were fixed before scoring held-out groups. Class-conditional "
                    "split-conformal theory requires exchangeable calibration/target objects; this is not "
                    "established by spatial group splitting. Sparse classes are always included. Coverage "
                    "must still be assessed on independent parcels.",
        })
    conformal["coverage_check"] = {}
    for a in conformal_alphas:
        sets = uncertainty.prediction_sets(poly[[f"p_{c}" for c in classes]].to_numpy(), classes,
                                           conformal["thresholds"][str(a)])
        conformal["coverage_check"][str(a)] = uncertainty.coverage(sets, poly.label)
    suspects.to_csv(out / "suspect_labels.csv", index=False)
    report["suspect_labels"] = int(len(suspects))
    if test is not None:
        proba = uncertainty.apply_temperature(model.predict_proba(test.subset(features)), temperature)
        pred = np.asarray(model.classes_)[proba.argmax(axis=1)].astype(str)
        h = metrics.summary(test.y, pred, labels=np.unique(np.concatenate([y, test.y])))
        if bootstrap:
            h["bootstrap"] = metrics.bootstrap(test.y, pred, groups=test.polygon, n_boot=bootstrap, seed=seed)
        h["note"] = "Locked spatial holdout, scored once; interpreted polygons, not independent field data."
        report["holdout"] = h
        pd.DataFrame({"polygon": test.polygon, "label": test.y, "predicted": pred}).to_csv(
            out / "holdout_predictions.csv", index=False)
        if refit_all and calibration_mode == "oof":
            samples = Samples(np.concatenate([samples.X, test.X]), np.concatenate([samples.y, test.y]),
                              np.concatenate([samples.polygon, test.polygon]), np.concatenate([samples.x, test.x]),
                              np.concatenate([samples.yy, test.yy]), samples.features, samples.crs)
            X, y = samples.subset(features), samples.y
            groups = spatial_blocks(samples, spatial_block_m)
            spatial = {"coordinates": np.column_stack([samples.x, samples.yy]), "crs": samples.crs,
                       "spatial_buffer_m": spatial_buffer_m}
            model = make_estimator(params, seed).fit(X, y)
        report["refit_after_holdout"] = bool(refit_all and calibration_mode == "oof")
    report["final_model_samples"] = int(len(y))
    imp = importances(model, len(features))
    importance = pd.DataFrame({"feature": features, "importance": imp})
    fold_id = np.empty(len(y), dtype=int)
    for k, (_, te) in enumerate(folds(y, groups, n_splits, seed, polygon=samples.polygon, **spatial)):
        fold_id[te] = k
    aoa = applicability.fit(X, fold_id, features, None if np.isnan(imp).all() else imp, seed=seed)
    aoa.to_json(out / "aoa.json")
    report["aoa_threshold"] = aoa.threshold
    importance.sort_values("importance", ascending=False).to_csv(out / "feature_importance.csv", index=False)
    pd.DataFrame(cv["confusion_matrix"], index=cv["labels"], columns=cv["labels"]).to_csv(out / "cv_confusion.csv")
    payload = {"schema_version": contracts.SCHEMA_VERSIONS["model"],
               "model": model, "features": features, "classes": [str(c) for c in model.classes_],
               "params": params, "seed": seed, "sklearn": sklearn.__version__, "temperature": temperature,
               "provenance": provenance or {}, "calibration_mode": calibration_mode,
               "calibration_split": split}
    dump(payload, out / "model.joblib", compress=3)
    model_sha256 = sha256(out / "model.joblib")
    conformal["model_sha256"] = model_sha256
    report["model_sha256"] = model_sha256
    (out / "conformal.json").write_text(json.dumps(conformal, indent=2), encoding="utf8")
    (out / "train_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf8")
    return report


def predict(stack: str | Path, model_path: str | Path, output_dir: str | Path, block_size: int = 512) -> dict:
    """Blockwise inference -> classes.tif (uint8 index, see class_legend.json), probabilities.tif
    (one band per class) and uncertainty.tif (max probability, top-2 margin, normalised entropy)."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    model_sha256 = sha256(model_path)
    payload = contracts.check("model", load(model_path), model_path)
    model, wanted = payload["model"], payload["features"]
    temperature = float(payload.get("temperature", 1.0))
    classes = [str(c) for c in model.classes_]
    if len(classes) > 254:
        raise ValueError("At most 254 classes fit the uint8 class raster")
    with rasterio.open(stack) as src:
        names = list(src.descriptions)
        missing = [f for f in wanted if f not in names]
        if missing:
            raise ValueError(f"Stack lacks model features: {missing[:5]}")
        bands = [names.index(f) + 1 for f in wanted]
        base = gtiff_profile(src)
        cls_prof = {**base, "count": 1, "dtype": "uint8", "nodata": 255}
        prob_prof = {**base, "count": len(classes), "dtype": "float32", "nodata": NODATA, "predictor": 3}
        unc_prof = {**base, "count": 3, "dtype": "float32", "nodata": NODATA, "predictor": 3}
        with rasterio.open(out / "classes.tif", "w", **cls_prof) as c_dst, \
                rasterio.open(out / "probabilities.tif", "w", **prob_prof) as p_dst, \
                rasterio.open(out / "uncertainty.tif", "w", **unc_prof) as u_dst:
            for destination in (c_dst, p_dst, u_dst):
                destination.update_tags(model_sha256=model_sha256)
            for i, c in enumerate(classes, 1):
                p_dst.set_band_description(i, c)
            for i, n in enumerate(("max_probability", "margin", "entropy"), 1):
                u_dst.set_band_description(i, n)
            for w in blocks(src.width, src.height, block_size):
                cube = src.read(bands, window=w)
                ok = valid_mask(cube, src.nodata)
                shape = ok.shape
                c = np.full(shape, 255, dtype="uint8")
                p = np.full((len(classes), *shape), NODATA, dtype="float32")
                u = np.full((3, *shape), NODATA, dtype="float32")
                if ok.any():
                    proba = model.predict_proba(cube[:, ok].T)
                    proba = uncertainty.apply_temperature(proba, temperature).astype("float32")
                    order = np.sort(proba, axis=1)
                    c[ok] = proba.argmax(axis=1).astype("uint8")
                    p[:, ok] = proba.T
                    u[0, ok] = order[:, -1]
                    u[1, ok] = order[:, -1] - (order[:, -2] if proba.shape[1] > 1 else 0)
                    ent = -np.sum(np.where(proba > 0, proba * np.log(np.maximum(proba, 1e-12)), 0), axis=1)
                    u[2, ok] = ent / np.log(len(classes)) if len(classes) > 1 else 0
                c_dst.write(c, 1, window=w)
                p_dst.write(p, window=w)
                u_dst.write(u, window=w)
    legend = {str(i): c for i, c in enumerate(classes)}
    aoa_json = Path(model_path).with_name("aoa.json")
    if aoa_json.exists():
        applicability.map_aoa(stack, applicability.Applicability.from_json(aoa_json), out / "aoa.tif", block_size)
    (out / "class_legend.json").write_text(json.dumps(legend, indent=2), encoding="utf8")
    manifest = out / "prediction_manifest.json"
    manifest.write_text(json.dumps({
        "schema": "geocrop-prediction/1", "model_sha256": model_sha256,
        "features": wanted, "classes": classes, "temperature": temperature,
        "calibration_mode": payload.get("calibration_mode", "oof"),
        "scope": "Artifact identity and inference settings only; not a scientific validity assessment.",
    }, indent=2), encoding="utf8")
    return {"classes": str(out / "classes.tif"), "probabilities": str(out / "probabilities.tif"),
            "uncertainty": str(out / "uncertainty.tif"), "legend": legend,
            "aoa": str(out / "aoa.tif") if aoa_json.exists() else None, "manifest": str(manifest)}
