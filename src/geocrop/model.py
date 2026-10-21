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
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold

from . import metrics
from .raster import NODATA, blocks, gtiff_profile, valid_mask
from .samples import Samples

RF_KEYS = ("n_estimators", "max_features", "min_samples_leaf", "max_depth", "class_weight", "n_jobs", "max_samples")


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


def make_rf(params: dict, seed: int) -> RandomForestClassifier:
    kwargs = {k: v for k, v in params.items() if k in RF_KEYS and v is not None}
    return RandomForestClassifier(random_state=seed, **kwargs)


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
    pooled["_oof"] = oof.astype(str)   # out-of-fold predictions; removed before JSON export
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


def train(samples: Samples, output_dir: str | Path, features: list[str] | None = None, params: dict | None = None,
          n_splits: int = 5, spatial_block_m: float | None = 5000, seed: int = 42, tune_grid: dict | None = None,
          bootstrap: int = 1000, provenance: dict | None = None) -> dict:
    """Spatial CV (optionally after tuning), then fit the mapping model on all samples."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    features = features or samples.features
    X = samples.subset(features)
    y = samples.y
    groups = spatial_blocks(samples, spatial_block_m)
    params = dict(params or {"n_estimators": 500, "max_features": "sqrt", "class_weight": "balanced_subsample", "n_jobs": -1})
    report: dict = {"features": len(features), "samples": int(len(y)), "polygons": int(len(np.unique(samples.polygon))),
                    "spatial_groups": int(len(np.unique(groups))), "spatial_block_m": spatial_block_m,
                    "class_counts": pd.Series(y).value_counts().to_dict(),
                    "polygon_counts": pd.Series(samples.polygon).groupby(y).nunique().to_dict()}
    if tune_grid:
        report["tuning"] = tune(X, y, samples.polygon, groups, params, tune_grid, n_splits, seed)
        params.update(report["tuning"]["best_params"])
    report["params"] = params
    report["cross_validation"] = cross_validate(X, y, samples.polygon, groups, params, n_splits, seed, bootstrap)
    report["cross_validation"].pop("_oof")
    model = make_rf(params, seed).fit(X, y)
    importance = pd.DataFrame({"feature": features, "importance": model.feature_importances_})
    importance.sort_values("importance", ascending=False).to_csv(out / "feature_importance.csv", index=False)
    cv = report["cross_validation"]
    pd.DataFrame(cv["confusion_matrix"], index=cv["labels"], columns=cv["labels"]).to_csv(out / "cv_confusion.csv")
    payload = {"model": model, "features": features, "classes": [str(c) for c in model.classes_],
               "params": params, "seed": seed, "sklearn": sklearn.__version__, "provenance": provenance or {}}
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
                    proba = model.predict_proba(cube[:, ok].T).astype("float32")
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
    (out / "class_legend.json").write_text(json.dumps(legend, indent=2), encoding="utf8")
    return {"classes": str(out / "classes.tif"), "probabilities": str(out / "probabilities.tif"),
            "uncertainty": str(out / "uncertainty.tif"), "legend": legend}
