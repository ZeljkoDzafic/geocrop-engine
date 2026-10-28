"""Pixel classifier: spatially blocked cross-validation, tuning, final fit and blockwise prediction.

Leakage control: all pixels of a reference polygon share one group, and polygons are grouped
into square spatial blocks (default 5 km), so no block contributes to both training and
testing in any fold. Folds are stratified by class where the data allow
(StratifiedGroupKFold). Cross-validated scores are *internal* diagnostics; external accuracy
must come from independent field observations (see `geocrop.validation`).

Transfer validation (``cv_strategy`` = ``region``, ``year`` or ``region_year``) holds out one whole region,
season or region-season cell per fold (see `geocrop.splits.leave_one_group_out`) and reports per-group
metrics and the gap to in-domain spatial CV; see docs/validation-protocols.md.
"""
from __future__ import annotations

import itertools
import json
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import sklearn
from joblib import dump, load
from pyproj import CRS
from rasterio.windows import Window
from scipy.spatial import cKDTree
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

from . import applicability, attribution, contracts, faultinject, metrics, splits, uncertainty
from .aoi import AoiMask, creation_options
from .estimators import importances, make_estimator, predict_labels, sequence_params
from .hashing import sha256
from .raster import (
    NODATA,
    RasterInputError,
    blocks,
    feature_names,
    gtiff_profile,
    open_input,
    read_block,
    valid_mask,
)
from .samples import Samples, check_classes

RF_KEYS = ("algorithm", "n_estimators", "max_features", "min_samples_leaf", "max_depth", "class_weight", "n_jobs",
           "max_samples", "learning_rate", "num_leaves", "epochs", "n_channels")
_MISSING = object()


def config_params(model_cfg: dict, keep_none: bool = False) -> dict:
    """Estimator parameters of a `model:` config section: the shared `RF_KEYS` plus `model.params`.

    `model.params` holds estimator-specific hyperparameters (for example of a plugin estimator); they
    override the shared keys and are checked with `check_params`. Without `model.params` the result is
    exactly the historical RF_KEYS subset (`keep_none` keeps shared keys set to null, as the train stage of
    `geocrop run` always did), so existing configurations train identical models.
    """
    params = {k: model_cfg[k] for k in RF_KEYS if k in model_cfg and (keep_none or model_cfg[k] is not None)}
    extra = dict(model_cfg.get("params") or {})
    if extra:
        params.update(extra)
        check_params(params, extra)
    return params


def _same(a, b) -> bool:
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    try:
        return bool(a == b)
    except Exception:  # array-like or exotic values: identity is the only safe comparison
        return a is b


def check_params(params: dict, extra: dict, where: str = "model.params") -> None:
    """Fail unless every key of `extra` is a constructor parameter of the estimator that `params` selects
    (built-in or plugin) and actually reaches it with the configured value.

    The estimator is constructed (not fitted) through its registered factory. A key the estimator does not
    accept, or one its factory silently drops (the CatBoost factory, for example, maps only the shared keys),
    raises ValueError naming the accepted parameters.
    """
    import inspect

    algo = params.get("algorithm", "rf")
    est = make_estimator(dict(params), 0)
    current = None
    if callable(getattr(est, "get_params", None)):
        try:
            current = dict(est.get_params(deep=False))
        except Exception:  # an estimator whose get_params needs fitted state
            current = None
    signature = {name for name, p in inspect.signature(type(est).__init__).parameters.items()
                 if name != "self" and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)}
    accepted = signature | set(current or ())
    for key, value in extra.items():
        if key not in accepted:
            raise ValueError(f"{where}.{key} is not a parameter of the {algo!r} estimator "
                             f"({type(est).__name__}); accepted: {', '.join(sorted(accepted))}")
        actual = current[key] if current is not None and key in current else getattr(est, key, _MISSING)
        if actual is _MISSING or not _same(actual, value):
            raise ValueError(f"The {algo!r} estimator factory does not pass {where}.{key} on to "
                             f"{type(est).__name__} (configured {value!r}, estimator has "
                             f"{None if actual is _MISSING else actual!r}); use a supported key or a plugin "
                             "factory that forwards its parameters")


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


def cv_groups(samples: Samples, strategy: str = "spatial_blocks", block_m: float | None = 5000) -> np.ndarray:
    """Group id per sample for a cross-validation strategy (spatial blocks, region, year or region-year)."""
    splits.check_strategy(strategy)
    if strategy == "spatial_blocks":
        return spatial_blocks(samples, block_m)
    return splits.strategy_groups(strategy, region=samples.region, year=samples.year, polygon=samples.polygon)


def make_rf(params: dict, seed: int):
    """Estimator for `params["algorithm"]` (default Random Forest); see `geocrop.estimators`."""
    return make_estimator(params, seed)


def _fold_plan(y, groups, n_splits: int, seed: int, *, polygon=None, coordinates=None, crs=None,
               spatial_buffer_m: float = 0, strategy: str = "spatial_blocks"):
    """Build deterministic folds, optionally purging whole training polygons near test samples.

    Distances refer to sampled pixel centres, not unsampled polygon boundaries. Test sets stay
    unchanged. A buffer therefore cannot establish separation of complete polygon geometries.
    With a group strategy (region/year/region_year) each fold holds out exactly one group and
    `n_splits` is ignored; diagnostics then name the ``held_out_group``.
    """
    splits.check_strategy(strategy)
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
    if strategy == "spatial_blocks":
        splitter = StratifiedGroupKFold(n_splits=min(n_splits, n_groups), shuffle=True, random_state=seed)
        plan = [(None, tr, te) for tr, te in splitter.split(np.zeros(len(y)), y, groups)]
    else:
        plan = list(splits.leave_one_group_out(groups, strategy))
    out = []
    for k, (held_out, tr, te) in enumerate(plan):
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
        if held_out is not None:
            diagnostics["held_out_group"] = str(held_out)
        out.append((tr, te, diagnostics))
    return out


def fold_plan_signature(test_indices) -> str:
    """Short hash of a fold assignment (the ordered test-index arrays): equal signatures = identical folds.

    Same value as `geocrop.benchmark.fold_signature` for the same samples, block size, folds and seed."""
    import hashlib

    def digest(values) -> str:
        values = np.ascontiguousarray(values)
        h = hashlib.sha256()
        h.update(str(values.dtype).encode())
        h.update(str(values.shape).encode())
        h.update(memoryview(values).cast("B"))
        return h.hexdigest()
    return hashlib.sha256(json.dumps([[k, digest(te)] for k, te in enumerate(test_indices)]).encode()).hexdigest()[:16]


def folds(y, groups, n_splits: int, seed: int, *, polygon=None, coordinates=None, crs=None,
          spatial_buffer_m: float = 0, strategy: str = "spatial_blocks"):
    """Spatial-group folds; an optional metric buffer purges whole nearby training polygons.

    The buffer applies to sampled pixel centres only. Held-out samples are never dropped.
    """
    return [(tr, te) for tr, te, _ in _fold_plan(
        y, groups, n_splits, seed, polygon=polygon, coordinates=coordinates, crs=crs,
        spatial_buffer_m=spatial_buffer_m, strategy=strategy)]


def cross_validate(X, y, polygon, groups, params: dict, n_splits: int = 5, seed: int = 42,
                   bootstrap: int = 0, *, coordinates=None, crs=None, spatial_buffer_m: float = 0,
                   strategy: str = "spatial_blocks") -> dict:
    """Out-of-fold predictions across spatial folds; metrics pooled over all folds plus per-fold spread.

    With a group `strategy` (region, year, region_year) `groups` must hold those group ids (see `cv_groups`);
    every fold holds out one group and the per-fold rows are the per-held-out-group metrics. Samples that
    no fold can evaluate (region_year cells without data from another region *and* year) are excluded
    from pooled metrics and counted in ``unevaluated_samples``.
    """
    oof = np.full(len(y), None, dtype=object)
    oof_proba = None
    per_fold = []
    classes = np.unique(y)
    plan = _fold_plan(y, groups, n_splits, seed, polygon=polygon, coordinates=coordinates, crs=crs,
                      spatial_buffer_m=spatial_buffer_m, strategy=strategy)
    for k, (tr, te, diagnostics) in enumerate(plan):
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
    done = np.array([v is not None for v in oof])
    pooled = metrics.summary(y[done], oof[done].astype(str), labels=classes)
    pooled["_oof"] = oof.astype(str)   # out-of-fold predictions and probabilities; removed before JSON export
    pooled["_oof_proba"] = oof_proba
    pooled["_classes"] = [str(c) for c in classes]
    if strategy != "spatial_blocks":
        pooled["_evaluated"] = done   # removed before JSON export
    pooled["strategy"] = strategy
    pooled["fold_signature"] = fold_plan_signature([te for _, te, _ in plan])
    pooled["unevaluated_samples"] = int((~done).sum())
    pooled["folds"] = per_fold
    pooled["spatial_buffer_m"] = spatial_buffer_m
    pooled["distance_support"] = "sampled pixel centres; not full polygon geometries"
    pooled["fold_macro_f1_mean"] = float(np.mean([f["macro_f1"] for f in per_fold]))
    pooled["fold_macro_f1_std"] = float(np.std([f["macro_f1"] for f in per_fold]))
    pooled["calibration"] = metrics.calibration(oof_proba[done], y[done], classes)
    if bootstrap:
        pooled["bootstrap"] = metrics.bootstrap(y[done], oof[done].astype(str), groups=np.asarray(polygon)[done],
                                                n_boot=bootstrap, seed=seed)
    pooled["note"] = "Internal spatial cross-validation on interpreted reference polygons; not independent field validation."
    if strategy != "spatial_blocks":
        pooled["note"] = (f"Leave-one-{strategy.replace('_', '-')}-out validation on interpreted reference polygons "
                          "(out-of-domain transfer); not independent field validation.")
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
                          seed: int = 42, *, coordinates=None, crs=None, spatial_buffer_m: float = 0,
                          strategy: str = "spatial_blocks", inner_groups=None) -> dict:
    """Unbiased estimate of the *tuned* procedure: hyperparameters are selected by an inner spatial CV that
    sees only the outer training blocks; the outer test blocks are predicted once (Varma & Simon, 2006).
    The same optional spatial buffer is applied in the outer and inner folds.

    With a group `strategy` the outer loop holds out one region/year (`groups` are those ids) and the inner
    tuning runs k-fold over `inner_groups` (normally spatial blocks) of the outer training data only, so the
    held-out region/year never influences the selected hyperparameters."""
    classes = np.unique(y)
    oof = np.full(len(y), None, dtype=object)
    chosen = []
    xy = None if coordinates is None else np.asarray(coordinates)
    inner_groups = np.asarray(groups if inner_groups is None else inner_groups)
    plan = _fold_plan(y, groups, n_outer, seed, polygon=polygon, coordinates=xy, crs=crs,
                      spatial_buffer_m=spatial_buffer_m, strategy=strategy)
    for k, (tr, te, diag) in enumerate(plan):
        inner = tune(X[tr], y[tr], polygon[tr], inner_groups[tr], base, grid, n_inner, seed + k,
                     coordinates=None if xy is None else xy[tr], crs=crs, spatial_buffer_m=spatial_buffer_m)
        params = {**base, **inner["best_params"]}
        oof[te] = predict_labels(make_estimator(params, seed + k).fit(X[tr], y[tr]), X[te])
        row = {"fold": k, **inner["best_params"]}
        if "held_out_group" in diag:
            row["held_out_group"] = diag["held_out_group"]
        chosen.append(row)
    done = np.array([v is not None for v in oof])
    s = metrics.summary(y[done], oof[done].astype(str), labels=classes)
    s["selected_params_per_fold"] = chosen
    s["fold_signature"] = fold_plan_signature([te for _, te, _ in plan])
    s["note"] = "Nested spatial CV: outer-fold scores of the full tuning procedure (not optimistic)."
    if strategy != "spatial_blocks":
        s["strategy"] = strategy
        s["unevaluated_samples"] = int((~done).sum())
        s["note"] = (f"Nested leave-one-{strategy.replace('_', '-')}-out CV; hyperparameters tuned by spatial-block "
                     "CV inside each outer training set (not optimistic).")
    return s


def transfer_validation(X, y, polygon, groups, in_domain_groups, params: dict, strategy: str,
                        n_splits: int = 5, seed: int = 42, *, coordinates=None, crs=None,
                        spatial_buffer_m: float = 0, in_domain: dict | None = None) -> dict:
    """Out-of-domain (leave-one-region/year-out) vs in-domain (spatial-block CV) accuracy on the same samples.

    `groups` are the strategy's group ids, `in_domain_groups` the spatial blocks. `in_domain` may be a
    `cross_validate` result computed earlier on the same arrays (it must still contain ``_oof``).
    Returns pooled metrics for both regimes (scored on the samples evaluated out-of-domain), one row per
    held-out group with its in-domain and out-of-domain scores, and the transfer gap
    (in-domain minus out-of-domain; positive values mean accuracy lost when moving to an unseen domain).
    """
    if strategy == "spatial_blocks":
        raise ValueError("transfer_validation needs a group strategy: region, year or region_year")
    y, polygon, groups = np.asarray(y), np.asarray(polygon), np.asarray(groups).astype(str)
    spatial = {"coordinates": coordinates, "crs": crs, "spatial_buffer_m": spatial_buffer_m}
    if in_domain is None:
        in_domain = cross_validate(X, y, polygon, in_domain_groups, params, n_splits, seed, **spatial)
    out = cross_validate(X, y, polygon, groups, params, n_splits, seed, strategy=strategy, **spatial)
    classes = out["_classes"]
    done = out["_evaluated"]
    keys = ("overall_accuracy", "kappa", "macro_f1")
    ind = metrics.summary(y[done], np.asarray(in_domain["_oof"])[done], labels=classes)
    ood = {k: out[k] for k in keys}
    rows = []
    for fold in out["folds"]:
        g = fold["held_out_group"]
        m = groups == g
        a = metrics.summary(y[m], np.asarray(in_domain["_oof"])[m], labels=classes)
        b = metrics.summary(y[m], out["_oof"][m], labels=classes)
        row = {"held_out_group": g, "n_test": int(m.sum()), "n_train": fold["n_train"],
               "test_polygons": int(len(np.unique(polygon[m]))),
               "missing_train_classes": fold["missing_train_classes"],
               "classes_in_test": sorted(set(y[m].astype(str)))}
        for k in keys:
            row[f"in_domain_{k}"], row[f"out_of_domain_{k}"] = a[k], b[k]
            row[f"gap_{k}"] = a[k] - b[k]
        row["out_of_domain_per_class_f1"] = {c: v["f1"] for c, v in b["per_class"].items() if v["reference_count"]}
        rows.append(row)
    skipped = sorted(set(groups) - {r["held_out_group"] for r in rows})
    group_f1 = np.array([r["out_of_domain_macro_f1"] for r in rows], dtype=float)
    return {
        "strategy": strategy, "n_groups": int(len(np.unique(groups))), "groups_not_evaluable": skipped,
        "evaluated_samples": int(done.sum()), "unevaluated_samples": int((~done).sum()),
        "in_domain": {k: ind[k] for k in keys} | {"protocol": "spatial-block k-fold CV on all groups"},
        "out_of_domain": ood | {"protocol": f"leave-one-{strategy.replace('_', '-')}-out",
                                "per_class": out["per_class"], "labels": out["labels"],
                                "confusion_matrix": out["confusion_matrix"]},
        "transfer_gap": {k: ind[k] - ood[k] for k in keys},
        "group_macro_f1_mean": float(np.nanmean(group_f1)) if len(group_f1) else None,
        "group_macro_f1_std": float(np.nanstd(group_f1)) if len(group_f1) else None,
        "worst_group": min(rows, key=lambda r: r["out_of_domain_macro_f1"])["held_out_group"] if rows else None,
        "per_group": rows,
        "note": "Both regimes are scored on the same samples. Interpreted reference polygons; the gap is an "
                "internal estimate of domain shift, not independent field accuracy.",
        "_oof": out["_oof"],
    }


def split_final_holdout(samples: Samples, spatial_block_m: float | None, fraction: float, seed: int):
    """(development, locked test or None): the single definition of the final holdout used by train, ablation
    and both benchmarks, so that model selection never sees the objects later reported as holdout accuracy."""
    if not fraction:
        return samples, None
    mask = holdout_mask(samples.y, spatial_blocks(samples, spatial_block_m), fraction, seed)
    return _subset(samples, ~mask), _subset(samples, mask)


def _subset(samples: Samples, mask) -> Samples:
    return samples.take(mask)


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
    return samples.take(fit), split


def train(samples: Samples, output_dir: str | Path, features: list[str] | None = None, params: dict | None = None,
          n_splits: int = 5, spatial_block_m: float | None = 5000, seed: int = 42, tune_grid: dict | None = None,
          bootstrap: int = 1000, provenance: dict | None = None, calibrate: bool = True,
          conformal_alphas=(0.05, 0.1, 0.2), spatial_buffer_m: float = 0,
          calibration_mode: str = "oof", calibration_fraction: float = 0.2, holdout_fraction: float = 0.0,
          refit_all: bool = True, inner_splits: int = 3, cv_strategy: str = "spatial_blocks",
          allow_missing_classes: bool = False) -> dict:
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
    * cv_strategy ``region``/``year``/``region_year`` adds `report["transfer_validation"]`: leave-one-group-out
      metrics per held-out region/year, the in-domain spatial CV on the same samples and the transfer gap.
      `cross_validation`, calibration, conformal thresholds and AOA keep using spatial blocks, so the default
      outputs are unchanged.
    """
    splits.check_strategy(cv_strategy)
    missing_classes = check_classes(samples, allow_missing_classes, "train")
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
        calibration = input_samples.take(held_out)
    X = samples.subset(features)
    y = samples.y
    params = dict(params or {"n_estimators": 500, "max_features": "sqrt", "class_weight": "balanced_subsample", "n_jobs": -1})
    params = sequence_params(params, features)
    report: dict = {"features": len(features), "samples": int(len(y)), "polygons": int(len(np.unique(samples.polygon))),
                    "spatial_groups": int(len(np.unique(groups))), "spatial_block_m": spatial_block_m,
                    "class_counts": pd.Series(y).value_counts().to_dict(),
                    "polygon_counts": pd.Series(samples.polygon).groupby(y).nunique().to_dict()}
    if missing_classes:
        report["reference_classes_without_samples"] = missing_classes
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
    if cv_strategy != "spatial_blocks":
        tgroups = cv_groups(samples, cv_strategy, spatial_block_m)
        tv = transfer_validation(X, y, samples.polygon, tgroups, groups, params, cv_strategy, n_splits, seed,
                                 in_domain=cv, **spatial)
        tv.pop("_oof")
        if tune_grid:
            tv["nested"] = nested_cross_validate(X, y, samples.polygon, tgroups, params, tune_grid, n_splits,
                                                 inner_splits, seed, strategy=cv_strategy, inner_groups=groups,
                                                 **spatial)
        report["transfer_validation"] = tv
        pd.DataFrame([{k: v for k, v in r.items() if not isinstance(v, (list, dict))} for r in tv["per_group"]]
                     ).to_csv(out / "transfer_per_group.csv", index=False)
    report["cv_strategy"] = cv_strategy
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
            meta = {k: (np.concatenate([getattr(samples, k), getattr(test, k)])
                        if getattr(samples, k) is not None else None) for k in ("region", "year")}
            samples = Samples(np.concatenate([samples.X, test.X]), np.concatenate([samples.y, test.y]),
                              np.concatenate([samples.polygon, test.polygon]), np.concatenate([samples.x, test.x]),
                              np.concatenate([samples.yy, test.yy]), samples.features, samples.crs, **meta)
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
    # Reference of the training/calibration domain for prediction-time shift checks (geocrop.shift).
    from . import shift
    ref = shift.build_reference(input_samples.subset(features), input_samples.polygon, features, classes,
                                poly["polygon"], poly[[f"p_{c}" for c in classes]].to_numpy(), poly["label"],
                                train_di=aoa.train_di, aoa_threshold=aoa.threshold,
                                importance=None if np.isnan(imp).all() else imp,
                                region=input_samples.region, year=input_samples.year, seed=seed)
    shift.save_reference(ref, out / shift.REFERENCE_FILE)
    importance.sort_values("importance", ascending=False).to_csv(out / "feature_importance.csv", index=False)
    pd.DataFrame(cv["confusion_matrix"], index=cv["labels"], columns=cv["labels"]).to_csv(out / "cv_confusion.csv")
    payload = {"schema_version": contracts.SCHEMA_VERSIONS["model"],
               "model": model, "features": features, "classes": [str(c) for c in model.classes_],
               "params": params, "seed": seed, "sklearn": sklearn.__version__, "temperature": temperature,
               "provenance": provenance or {}, "calibration_mode": calibration_mode,
               "calibration_split": split}
    dump(payload, out / "model.joblib", compress=3)
    faultinject.point("train")
    model_sha256 = sha256(out / "model.joblib")
    conformal["model_sha256"] = model_sha256
    report["model_sha256"] = model_sha256
    (out / "conformal.json").write_text(json.dumps(conformal, indent=2), encoding="utf8")
    (out / "train_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf8")
    return report


SMOOTHING_DEFAULTS = {"enabled": False, "window": 7, "smoothness": 10.0}


def smoothing_settings(smoothing: dict | bool | None) -> dict | None:
    """Validated smoothing settings (None when disabled); accepts the `predict.smoothing` config section."""
    if not smoothing:
        return None
    cfg = {**SMOOTHING_DEFAULTS, **({"enabled": True} if smoothing is True else smoothing)}
    if not cfg["enabled"]:
        return None
    window, smoothness = cfg["window"], cfg["smoothness"]
    if isinstance(window, bool) or not isinstance(window, int) or window < 3 or window % 2 == 0:
        raise ValueError("smoothing window must be an odd integer >= 3")
    if isinstance(smoothness, bool) or not isinstance(smoothness, (int, float)) or not np.isfinite(smoothness) \
            or smoothness <= 0:
        raise ValueError("smoothing smoothness must be a positive finite number")
    return {"enabled": True, "window": int(window), "smoothness": float(smoothness)}


def smooth_probabilities(proba: np.ndarray, valid: np.ndarray, window: int = 7, smoothness: float = 10.0,
                         eps: float = 1e-4) -> np.ndarray:
    """Edge-aware Bayesian-style smoothing of a class-probability cube ``(classes, rows, cols)``.

    Each class probability is mapped to log-odds. For every pixel the local mean ``m`` and variance ``v`` of the
    log-odds over the valid pixels of a ``window`` x ``window`` neighbourhood act as a prior; the pixel's own
    value is the observation with variance `smoothness`. The posterior mean is
    ``(v * x + smoothness * m) / (v + smoothness)``: in homogeneous areas (small ``v``) pixels are pulled
    towards their neighbourhood, across field edges (large ``v``) they keep their own value. Probabilities are
    renormalised to sum to one. Invalid pixels neither contribute nor change (returned unchanged).
    Independent implementation of the idea behind the Bayesian smoothing of the sits R package.
    """
    from scipy.ndimage import uniform_filter

    proba = np.asarray(proba, dtype="float64")
    valid = np.asarray(valid, dtype=bool)
    out = proba.copy()
    if not valid.any():
        return out.astype("float32")
    w = valid.astype("float64")
    count = uniform_filter(w, size=window, mode="constant", cval=0.0)
    count = np.where(count > 0, count, np.nan)
    smoothed = np.empty_like(proba)
    for c in range(proba.shape[0]):
        p = np.clip(np.where(valid, proba[c], 0.5), eps, 1 - eps)
        x = np.log(p / (1 - p)) * w
        mean = uniform_filter(x, size=window, mode="constant", cval=0.0) / count
        var = np.maximum(uniform_filter(x * x, size=window, mode="constant", cval=0.0) / count - mean ** 2, 0.0)
        post = (var * x + smoothness * mean) / (var + smoothness)
        smoothed[c] = 1 / (1 + np.exp(-post))
    total = smoothed.sum(axis=0)
    smoothed /= np.where(total > 0, total, 1)
    out[:, valid] = smoothed[:, valid]
    return out.astype("float32")


def _uncertainty_bands(proba: np.ndarray, n_classes: int) -> np.ndarray:
    """(3, n): max probability, top-2 margin, normalised entropy for an ``(n, classes)`` probability matrix."""
    order = np.sort(proba, axis=1)
    u = np.empty((3, len(proba)), dtype="float32")
    u[0] = order[:, -1]
    u[1] = order[:, -1] - (order[:, -2] if proba.shape[1] > 1 else 0)
    ent = -np.sum(np.where(proba > 0, proba * np.log(np.maximum(proba, 1e-12)), 0), axis=1)
    u[2] = ent / np.log(n_classes) if n_classes > 1 else 0
    return u


def predict(stack: str | Path, model_path: str | Path, output_dir: str | Path, block_size: int = 512,
            smoothing: dict | bool | None = None, shift: dict | None = None, aoi: AoiMask | None = None) -> dict:
    """Blockwise inference -> classes.tif (uint8 index, see class_legend.json), probabilities.tif
    (one band per class) and uncertainty.tif (max probability, top-2 margin, normalised entropy).

    `smoothing` (the `predict.smoothing` config section, e.g. ``{"enabled": True, "window": 7,
    "smoothness": 10}``) applies `smooth_probabilities` before the argmax, so classes, uncertainty and any
    later parcel aggregation use smoothed probabilities; the raw probabilities are kept in
    ``probabilities_unsmoothed.tif``. Blocks are read with a halo, so results do not depend on block size.

    `shift` (``{"config": <uncertainty section>, "parcels": path|None, "id_field": str, "alpha": float}``)
    compares a pixel sample of the stack (inside the parcels when given) with the model's
    ``domain_reference.npz`` and writes ``shift_report.json`` (`geocrop.shift`); the manifest then states
    whether conformal coverage is established. Without it the outputs are unchanged.

    `aoi` (`geocrop.aoi.AoiMask`) restricts prediction to the AOI: only AOI pixels are classified, every other
    pixel of classes / probabilities / uncertainty / AOA map (and of the unsmoothed probabilities) is no-data,
    and blocks without AOI pixels are neither read nor written. Probabilities of the pixels within the smoothing
    radius of the AOI are computed first (so smoothing at the AOI border sees the same neighbours as an unmasked
    run), which needs `aoi.halo` >= the smoothing radius; pixels inside the AOI equal an unmasked run.
    """
    smooth = smoothing_settings(smoothing)
    smooth_halo = smooth["window"] // 2 if smooth is not None else 0
    if aoi is not None and aoi.halo < smooth_halo:
        raise ValueError(f"AOI halo {aoi.halo} px is smaller than the smoothing radius {smooth_halo} px; "
                         "pass an AoiMask with halo >= window // 2 (and build the stack with the same halo)")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    model_sha256 = sha256(model_path)
    payload = contracts.check("model", load(model_path), model_path)
    model, wanted = payload["model"], payload["features"]
    temperature = float(payload.get("temperature", 1.0))
    classes = [str(c) for c in model.classes_]
    if len(classes) > 254:
        raise ValueError("At most 254 classes fit the uint8 class raster")
    raw_path = out / "probabilities_unsmoothed.tif"
    with open_input(stack, "Feature stack") as src:
        names = feature_names(src)
        missing = [f for f in wanted if f not in names]
        if missing:
            raise RasterInputError(f"Feature stack {stack} lacks {len(missing)} model features: {missing[:5]}")
        any_valid = False
        bands = [names.index(f) + 1 for f in wanted]
        base = gtiff_profile(src)
        cls_prof = {**base, "count": 1, "dtype": "uint8", "nodata": 255}
        prob_prof = {**base, "count": len(classes), "dtype": "float32", "nodata": NODATA, "predictor": 3}
        unc_prof = {**base, "count": 3, "dtype": "float32", "nodata": NODATA, "predictor": 3}
        width, height = src.width, src.height
        if aoi is not None:
            aoi.check_grid(src, "the feature stack")
            cls_prof, prob_prof, unc_prof = ({**p, **creation_options(aoi)} for p in (cls_prof, prob_prof, unc_prof))

        def describe(p_dst):
            p_dst.update_tags(model_sha256=model_sha256)
            for i, c in enumerate(classes, 1):
                p_dst.set_band_description(i, c)

        def model_proba(w, restrict=True):
            cube = read_block(src, bands, window=w)
            ok = valid_mask(cube, src.nodata)
            if aoi is not None and restrict:
                ok &= aoi.read(w)        # classify AOI pixels only
            proba = None
            if ok.any():
                proba = model.predict_proba(cube[:, ok].T)
                proba = uncertainty.apply_temperature(proba, temperature).astype("float32")
            return ok, proba

        if smooth is not None:
            with rasterio.open(raw_path, "w", **prob_prof) as r_dst:
                describe(r_dst)
                r_dst.update_tags(smoothing="none (input to smoothing)")
                for w in blocks(width, height, block_size):
                    if aoi is not None and not aoi.active(w, smooth_halo):
                        continue
                    ok, proba = model_proba(w, restrict=False)
                    p = np.full((len(classes), *ok.shape), NODATA, dtype="float32")
                    if proba is not None:
                        p[:, ok] = proba.T
                    r_dst.write(p, window=w)
        with rasterio.open(out / "classes.tif", "w", **cls_prof) as c_dst, \
                rasterio.open(out / "probabilities.tif", "w", **prob_prof) as p_dst, \
                rasterio.open(out / "uncertainty.tif", "w", **unc_prof) as u_dst, \
                (rasterio.open(raw_path) if smooth is not None else nullcontext()) as raw:
            for destination in (c_dst, u_dst):
                destination.update_tags(model_sha256=model_sha256)
            describe(p_dst)
            for destination in (c_dst, p_dst, u_dst):    # Copernicus attribution of the feature stack, if any
                attribution.propagate([stack], destination)
            if smooth is not None:
                for destination in (c_dst, p_dst, u_dst):
                    destination.update_tags(smoothing=json.dumps(smooth))
            for i, n in enumerate(("max_probability", "margin", "entropy"), 1):
                u_dst.set_band_description(i, n)
            halo = smooth_halo
            for w in blocks(width, height, block_size):
                if aoi is not None and not aoi.active(w, 0):
                    continue
                if smooth is None:
                    ok, proba = model_proba(w)
                else:
                    c0, r0 = max(0, w.col_off - halo), max(0, w.row_off - halo)
                    c1 = min(width, w.col_off + w.width + halo)
                    r1 = min(height, w.row_off + w.height + halo)
                    cube = raw.read(window=Window(c0, r0, c1 - c0, r1 - r0))
                    big_ok = np.all(cube != NODATA, axis=0)
                    cube = smooth_probabilities(cube, big_ok, smooth["window"], smooth["smoothness"])
                    inner = (slice(w.row_off - r0, w.row_off - r0 + w.height),
                             slice(w.col_off - c0, w.col_off - c0 + w.width))
                    ok = big_ok[inner]
                    if aoi is not None:
                        ok = ok & aoi.read(w)
                    proba = cube[:, inner[0], inner[1]][:, ok].T if ok.any() else None
                any_valid = any_valid or bool(ok.any())
                shape = ok.shape
                c = np.full(shape, 255, dtype="uint8")
                p = np.full((len(classes), *shape), NODATA, dtype="float32")
                u = np.full((3, *shape), NODATA, dtype="float32")
                if proba is not None:
                    c[ok] = proba.argmax(axis=1).astype("uint8")
                    p[:, ok] = proba.T
                    u[:, ok] = _uncertainty_bands(proba, len(classes))
                c_dst.write(c, 1, window=w)
                p_dst.write(p, window=w)
                u_dst.write(u, window=w)
                faultinject.point("predict")
    if aoi is not None and smooth is not None:
        _mask_raw_probabilities(raw_path, aoi, smooth_halo, block_size)
    if not any_valid:
        for name in ("classes.tif", "probabilities.tif", "uncertainty.tif"):
            (out / name).unlink(missing_ok=True)
        raise RasterInputError(f"Feature stack {stack}: no pixel has valid values for all {len(wanted)} model "
                               "features (all nodata/NaN); the classification would be empty")
    legend = {str(i): c for i, c in enumerate(classes)}
    aoa_json = Path(model_path).with_name("aoa.json")
    if aoa_json.exists():
        applicability.map_aoa(stack, applicability.Applicability.from_json(aoa_json), out / "aoa.tif", block_size,
                              aoi=aoi)
    (out / "class_legend.json").write_text(json.dumps(legend, indent=2), encoding="utf8")
    manifest = out / "prediction_manifest.json"
    body = {
        "schema": "geocrop-prediction/1", "model_sha256": model_sha256,
        "features": wanted, "classes": classes, "temperature": temperature,
        "calibration_mode": payload.get("calibration_mode", "oof"),
        "scope": "Artifact identity and inference settings only; not a scientific validity assessment.",
    }
    if aoi is not None:
        body["aoi"] = {"mask": aoi.path.name, "halo_px": aoi.halo,
                       "note": "Only AOI pixels are classified; every pixel outside the AOI is no-data in "
                               "classes, probabilities, uncertainty, aoa and the unsmoothed probabilities."}
    if smooth is not None:
        body["smoothing"] = {**smooth, "method": "edge-aware Bayesian-style log-odds smoothing",
                             "unsmoothed_probabilities": raw_path.name,
                             "note": "Conformal thresholds in conformal.json were fitted on unsmoothed "
                                     "out-of-fold probabilities; their coverage on smoothed maps is not established."}
    if shift is not None:
        info = shift_check(stack, model_path, model, classes, wanted, out, shift)
        if info is not None:
            body["domain_shift"] = info
    manifest.write_text(json.dumps(body, indent=2), encoding="utf8")
    result = {"classes": str(out / "classes.tif"), "probabilities": str(out / "probabilities.tif"),
              "uncertainty": str(out / "uncertainty.tif"), "legend": legend,
              "aoa": str(out / "aoa.tif") if aoa_json.exists() else None, "manifest": str(manifest)}
    if smooth is not None:
        result["probabilities_unsmoothed"] = str(raw_path)
    return result


def _mask_raw_probabilities(path: Path, aoi: AoiMask, halo: int, block_size: int) -> None:
    """After smoothing: set the unsmoothed probabilities of pixels outside the AOI (computed only as smoothing
    context) to no-data, so no product holds a prediction outside the AOI."""
    with rasterio.open(path, "r+") as ds:
        for w in blocks(ds.width, ds.height, block_size):
            if not aoi.active(w, halo):
                continue
            inside = aoi.read(w)
            if inside.all():
                continue
            data = ds.read(window=w)
            data[:, ~inside] = NODATA
            ds.write(data, window=w)



def shift_check(stack, model_path, model, classes, features, out: Path, options: dict) -> dict | None:
    """Prediction-time domain-shift check (`geocrop.shift`); writes ``shift_report.json`` into `out`."""
    from . import shift as shiftmod
    cfg = shiftmod.settings(options.get("config"))
    target = out / "shift_report.json"
    if not cfg["shift_detection"]:
        target.unlink(missing_ok=True)
        return None
    ref_path = Path(model_path).with_name(shiftmod.REFERENCE_FILE)
    if not ref_path.exists():
        report = shiftmod.unavailable(f"the model has no {shiftmod.REFERENCE_FILE}; retrain to enable shift checks",
                                      cfg)
    else:
        ref = shiftmod.load_reference(ref_path)
        X, groups, note = shiftmod.sample_stack(stack, features, options.get("parcels"),
                                                options.get("id_field", "parcel_id"),
                                                n=cfg["shift_sample_pixels"], seed=int(options.get("seed", 42)))
        keep = np.isfinite(X).all(axis=1)
        X, groups = X[keep], groups[keep]
        order = [classes.index(c) for c in ref["meta"]["classes"]]

        def proba(Z):
            return np.asarray(model.predict_proba(Z))[:, order]
        aoa_json = Path(model_path).with_name("aoa.json")
        aoa = applicability.Applicability.from_json(aoa_json) if aoa_json.exists() else None
        conf_json = Path(model_path).with_name("conformal.json")
        conformal = json.loads(conf_json.read_text(encoding="utf8")) if conf_json.exists() else None
        report = shiftmod.detect(ref, X, groups, proba, aoa=aoa, conformal=conformal,
                                 metadata=options.get("metadata"), config=cfg,
                                 seed=int(options.get("seed", 42)), sample_note=note)
    target.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf8")
    return {"verdict": report["verdict"], "report": target.name, "policy": report["policy"],
            "conformal_mode": report["conformal_mode"],
            "conformal_coverage": shiftmod.coverage_statement(report, options.get("alpha", 0.1)),
            "reasons": [r["message"] for r in report.get("reasons", [])]}
