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
from sklearn.model_selection import StratifiedGroupKFold

from . import applicability, metrics, uncertainty
from .estimators import channels_from_names, importances, make_estimator
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


def folds(y, groups, n_splits: int, seed: int):
    n_groups = len(np.unique(groups))
    if n_groups < 2:
        raise ValueError("Cross-validation needs at least two spatial groups")
    splitter = StratifiedGroupKFold(n_splits=min(n_splits, n_groups), shuffle=True, random_state=seed)
    out = []
    for tr, te in splitter.split(np.zeros(len(y)), y, groups):
        if set(groups[tr]) & set(groups[te]):
            raise AssertionError("Spatial group leakage between folds")
        out.append((tr, te))
    return out


def cross_validate(X, y, polygon, groups, params: dict, n_splits: int = 5, seed: int = 42,
                   bootstrap: int = 0) -> dict:
    """Out-of-fold predictions across spatial folds; metrics pooled over all folds plus per-fold spread."""
    oof = np.empty(len(y), dtype=object)
    oof_proba = None
    per_fold = []
    classes = np.unique(y)
    for k, (tr, te) in enumerate(folds(y, groups, n_splits, seed)):
        rf = make_rf(params, seed + k).fit(X[tr], y[tr])
        oof[te] = rf.predict(X[te])
        proba = np.zeros((len(te), len(classes)), dtype="float32")
        proba[:, np.searchsorted(classes, rf.classes_)] = rf.predict_proba(X[te])
        if oof_proba is None:
            oof_proba = np.zeros((len(y), len(classes)), dtype="float32")
        oof_proba[te] = proba
        s = metrics.summary(y[te], oof[te], labels=classes)
        per_fold.append({"fold": k, "n_test": int(len(te)), "test_groups": int(len(np.unique(groups[te]))),
                         "missing_train_classes": sorted(set(classes) - set(rf.classes_)),
                         "overall_accuracy": s["overall_accuracy"], "macro_f1": s["macro_f1"], "kappa": s["kappa"]})
    pooled = metrics.summary(y, oof.astype(str), labels=classes)
    pooled["_oof"] = oof.astype(str)   # out-of-fold predictions and probabilities; removed before JSON export
    pooled["_oof_proba"] = oof_proba
    pooled["_classes"] = [str(c) for c in classes]
    pooled["folds"] = per_fold
    pooled["fold_macro_f1_mean"] = float(np.mean([f["macro_f1"] for f in per_fold]))
    pooled["fold_macro_f1_std"] = float(np.std([f["macro_f1"] for f in per_fold]))
    pooled["calibration"] = metrics.calibration(oof_proba, y, classes)
    if bootstrap:
        pooled["bootstrap"] = metrics.bootstrap(y, oof.astype(str), groups=polygon, n_boot=bootstrap, seed=seed)
    pooled["note"] = "Internal spatial cross-validation on interpreted reference polygons; not independent field validation."
    return pooled


def tune(X, y, polygon, groups, base: dict, grid: dict, n_splits: int = 5, seed: int = 42) -> dict:
    """Exhaustive grid search scored by spatial-CV macro-F1. The winning score is optimistically biased
    (selected on the same folds) and must not be reported as final accuracy."""
    keys = list(grid)
    trials = []
    for values in itertools.product(*(grid[k] for k in keys)):
        params = {**base, **dict(zip(keys, values))}
        cv = cross_validate(X, y, polygon, groups, params, n_splits, seed)
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
                          seed: int = 42) -> dict:
    """Unbiased estimate of the *tuned* procedure: hyperparameters are selected by an inner spatial CV that
    sees only the outer training blocks; the outer test blocks are predicted once (Varma & Simon, 2006)."""
    classes = np.unique(y)
    oof = np.empty(len(y), dtype=object)
    chosen = []
    for k, (tr, te) in enumerate(folds(y, groups, n_outer, seed)):
        inner = tune(X[tr], y[tr], polygon[tr], groups[tr], base, grid, n_inner, seed + k)
        params = {**base, **inner["best_params"]}
        oof[te] = make_estimator(params, seed + k).fit(X[tr], y[tr]).predict(X[te])
        chosen.append({"fold": k, **inner["best_params"]})
    s = metrics.summary(y, oof.astype(str), labels=classes)
    s["selected_params_per_fold"] = chosen
    s["note"] = "Nested spatial CV: outer-fold scores of the full tuning procedure (not optimistic)."
    return s


def train(samples: Samples, output_dir: str | Path, features: list[str] | None = None, params: dict | None = None,
          n_splits: int = 5, spatial_block_m: float | None = 5000, seed: int = 42, tune_grid: dict | None = None,
          bootstrap: int = 1000, provenance: dict | None = None, calibrate: bool = True,
          conformal_alphas=(0.05, 0.1, 0.2), holdout_fraction: float = 0.0, refit_all: bool = True,
          inner_splits: int = 3) -> dict:
    """Spatial CV (nested when tuning), optional locked holdout, then the mapping model.

    * holdout_fraction > 0 locks away whole spatial blocks before any tuning, calibration or model
      comparison; they are scored once at the end (`report["holdout"]`).
    * With a tuning grid, `report["nested_cv"]` gives the unbiased score of the tuning procedure;
      `report["cross_validation"]` (re-run with the selected parameters) then serves only calibration,
      conformal thresholds and diagnostics.
    * From the out-of-fold probabilities: temperature (calibration), split-conformal thresholds on
      polygon-level probabilities (`conformal.json`) and suspicious reference labels (`suspect_labels.csv`).
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    features = features or samples.features
    X_all = samples.subset(features)
    y_all = samples.y
    groups_all = spatial_blocks(samples, spatial_block_m)
    params = dict(params or {"n_estimators": 500, "max_features": "sqrt", "class_weight": "balanced_subsample", "n_jobs": -1})
    if params.get("algorithm") == "tempcnn" and "n_channels" not in params:
        params["n_channels"] = channels_from_names(features)
    report: dict = {"features": len(features), "samples": int(len(y_all)),
                    "polygons": int(len(np.unique(samples.polygon))),
                    "spatial_groups": int(len(np.unique(groups_all))), "spatial_block_m": spatial_block_m,
                    "class_counts": pd.Series(y_all).value_counts().to_dict(),
                    "polygon_counts": pd.Series(samples.polygon).groupby(y_all).nunique().to_dict()}
    hold = holdout_mask(y_all, groups_all, holdout_fraction, seed) if holdout_fraction else np.zeros(len(y_all), bool)
    dev = ~hold
    X, y, polygon, groups = X_all[dev], y_all[dev], samples.polygon[dev], groups_all[dev]
    if hold.any():
        report["holdout_design"] = {"fraction": holdout_fraction, "samples": int(hold.sum()),
                                    "polygons": int(len(np.unique(samples.polygon[hold]))),
                                    "blocks": int(len(np.unique(groups_all[hold]))),
                                    "classes_missing": sorted(set(y_all) - set(y_all[hold]))}
    if tune_grid:
        report["nested_cv"] = nested_cross_validate(X, y, polygon, groups, params, tune_grid, n_splits,
                                                    inner_splits, seed)
        report["tuning"] = tune(X, y, polygon, groups, params, tune_grid, n_splits, seed)
        params.update(report["tuning"]["best_params"])
    report["params"] = params
    cv = cross_validate(X, y, polygon, groups, params, n_splits, seed, bootstrap)
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
    poly = uncertainty.polygon_oof(oof_proba, y, polygon, classes)
    poly.to_csv(out / "oof_polygons.csv", index=False)
    conformal = uncertainty.conformal_thresholds(poly[[f"p_{c}" for c in classes]].to_numpy(), poly.label, classes,
                                                 conformal_alphas)
    conformal["level"] = "polygon (out-of-fold mean probabilities)"
    conformal["coverage_check"] = {}
    for a in conformal_alphas:
        sets = uncertainty.prediction_sets(poly[[f"p_{c}" for c in classes]].to_numpy(), classes,
                                           conformal["thresholds"][str(a)])
        conformal["coverage_check"][str(a)] = uncertainty.coverage(sets, poly.label)
    (out / "conformal.json").write_text(json.dumps(conformal, indent=2), encoding="utf8")
    suspects = uncertainty.suspect_labels(poly)
    suspects.to_csv(out / "suspect_labels.csv", index=False)
    report["suspect_labels"] = int(len(suspects))
    model = make_estimator(params, seed).fit(X, y)
    if hold.any():
        proba = uncertainty.apply_temperature(model.predict_proba(X_all[hold]), temperature)
        pred = np.asarray(model.classes_)[proba.argmax(axis=1)].astype(str)
        h = metrics.summary(y_all[hold], pred, labels=np.unique(y_all))
        if bootstrap:
            h["bootstrap"] = metrics.bootstrap(y_all[hold], pred, groups=samples.polygon[hold], n_boot=bootstrap,
                                               seed=seed)
        h["note"] = "Locked spatial holdout, scored once; interpreted polygons, not independent field data."
        report["holdout"] = h
        pd.DataFrame({"polygon": samples.polygon[hold], "label": y_all[hold], "predicted": pred}).to_csv(
            out / "holdout_predictions.csv", index=False)
        if refit_all:
            X, y, groups = X_all, y_all, groups_all
            model = make_estimator(params, seed).fit(X, y)
    report["final_model_samples"] = int(len(y))
    imp = importances(model, len(features))
    importance = pd.DataFrame({"feature": features, "importance": imp})
    fold_id = np.empty(len(y), dtype=int)
    for k, (_, te) in enumerate(folds(y, groups, n_splits, seed)):
        fold_id[te] = k
    aoa = applicability.fit(X, fold_id, features, None if np.isnan(imp).all() else imp, seed=seed)
    aoa.to_json(out / "aoa.json")
    report["aoa_threshold"] = aoa.threshold
    importance.sort_values("importance", ascending=False).to_csv(out / "feature_importance.csv", index=False)
    pd.DataFrame(cv["confusion_matrix"], index=cv["labels"], columns=cv["labels"]).to_csv(out / "cv_confusion.csv")
    payload = {"model": model, "features": features, "classes": [str(c) for c in model.classes_],
               "params": params, "seed": seed, "sklearn": sklearn.__version__, "temperature": temperature,
               "provenance": provenance or {}}
    dump(payload, out / "model.joblib", compress=3)
    (out / "train_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf8")
    return report


def predict(stack: str | Path, model_path: str | Path, output_dir: str | Path, block_size: int = 512) -> dict:
    """Blockwise inference -> classes.tif (uint8 index, see class_legend.json), probabilities.tif
    (one band per class) and uncertainty.tif (max probability, top-2 margin, normalised entropy)."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = load(model_path)
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
    return {"classes": str(out / "classes.tif"), "probabilities": str(out / "probabilities.tif"),
            "uncertainty": str(out / "uncertainty.tif"), "legend": legend,
            "aoa": str(out / "aoa.tif") if aoa_json.exists() else None}
