"""Ablation experiments on identical samples and identical spatial folds.

Each experiment is a feature subset defined by groups / periods / explicit names (see
`geocrop.features.select`). Because samples and folds are shared, differences between rows
are attributable to the feature set alone. Output: `ablation.csv` (one row per experiment,
ready for a manuscript table) and `ablation.json` (full metrics, per-class F1, fold spread).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import metrics
from .features import Feature, select
from .model import cross_validate, selection_samples, spatial_blocks
from .samples import Samples

# Default design mirrors the study protocol; override in the YAML `experiments:` section.
DEFAULT_EXPERIMENTS = {
    "E0_S2_bands": {"groups": ["s2_band"]},
    "E1_S2_bands_indices": {"groups": ["s2_band", "s2_index"]},
    "E2_S2_full_temporal": {"groups": ["s2_band", "s2_index", "delta", "seasonal"]},
    "E3_S1_only": {"groups": ["s1"]},
    "E4_S1_S2_fusion": {"groups": ["s2_band", "s2_index", "s1", "delta", "seasonal"]},
}


def run(samples: Samples, manifest: list[Feature], output_dir: str | Path, experiments: dict | None = None,
        params: dict | None = None, n_splits: int = 5, spatial_block_m: float | None = 5000, seed: int = 42,
        targets: list[str] | None = None, holdout_fraction: float = 0.0, spatial_buffer_m: float = 0,
        calibration_mode: str = "oof", calibration_fraction: float = 0.2) -> pd.DataFrame:
    """Compare feature subsets on identical samples and spatial folds.

    * holdout_fraction > 0 locks away whole spatial blocks first: every experiment is additionally trained
      on the remaining data and scored once on the holdout (`holdout_*` columns, McNemar on the holdout).
    * In ``holdout`` calibration mode only the proper-training objects (same deterministic split as `train`)
      are used, so calibration objects never influence feature selection.
    * spatial_buffer_m purges training polygons near test samples in every fold.
    """
    from .model import _subset, holdout_mask, make_estimator

    test = None
    if holdout_fraction:
        test_mask = holdout_mask(samples.y, spatial_blocks(samples, spatial_block_m), holdout_fraction, seed)
        test, samples = _subset(samples, test_mask), _subset(samples, ~test_mask)
    samples, calibration_split = selection_samples(samples, spatial_block_m, seed, spatial_buffer_m,
                                                   calibration_mode, calibration_fraction)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    experiments = experiments or DEFAULT_EXPERIMENTS
    params = params or {"n_estimators": 300, "max_features": "sqrt", "class_weight": "balanced_subsample", "n_jobs": -1}
    groups = spatial_blocks(samples, spatial_block_m)
    spatial = {"coordinates": np.column_stack([samples.x, samples.yy]), "crs": samples.crs,
               "spatial_buffer_m": spatial_buffer_m}
    ref_hold = None
    available = [f for f in manifest if f.name in samples.features]
    rows, full = [], {}
    reference_name, reference_oof = None, None   # first successful experiment is the paired-test baseline
    for name, spec in experiments.items():
        try:
            names = select(available, groups=spec.get("groups"), periods=spec.get("periods"),
                           names=spec.get("features"), exclude=spec.get("exclude"))
        except ValueError:
            rows.append({"experiment": name, "status": "skipped: no matching features"})
            continue
        X = samples.subset(names)
        cv = cross_validate(X, samples.y, samples.polygon, groups, params, n_splits, seed, **spatial)
        oof = cv.pop("_oof")
        cv.pop("_oof_proba"), cv.pop("_classes")
        if reference_oof is None:
            reference_name, reference_oof = name, oof
        test_p = metrics.mcnemar(samples.y, oof, reference_oof)
        row = {"experiment": name, "status": "ok", "n_features": len(names),
               "overall_accuracy": cv["overall_accuracy"], "kappa": cv["kappa"], "macro_f1": cv["macro_f1"],
               "macro_f1_fold_std": cv["fold_macro_f1_std"], "ece": cv["calibration"]["ece"],
               "vs": reference_name, "mcnemar_p": test_p["p_value"]}
        for t in targets or []:
            row[f"f1_{t}"] = cv["per_class"].get(t, {}).get("f1")
        if test is not None:
            pred = np.asarray(make_estimator(params, seed).fit(X, samples.y).predict(test.subset(names))).astype(str)
            h = metrics.summary(test.y, pred, labels=np.unique(np.concatenate([samples.y, test.y])))
            if ref_hold is None:
                ref_hold = pred
            row.update({"holdout_overall_accuracy": h["overall_accuracy"], "holdout_macro_f1": h["macro_f1"],
                        "holdout_mcnemar_p": metrics.mcnemar(test.y, pred, ref_hold)["p_value"]})
        rows.append(row)
        full[name] = {"spec": spec, "features": names, "cv": cv}
    table = pd.DataFrame(rows)
    table.to_csv(out / "ablation.csv", index=False)
    (out / "ablation.json").write_text(json.dumps({
        "spatial_block_m": spatial_block_m, "spatial_buffer_m": spatial_buffer_m, "n_splits": n_splits, "seed": seed,
        "params": params, "calibration_mode": calibration_mode, "calibration_split": calibration_split,
        "holdout_fraction": holdout_fraction, "holdout_samples": 0 if test is None else int(len(test.y)),
        "n_samples": int(len(samples.y)), "n_groups": int(len(np.unique(groups))), "experiments": full},
        indent=2, default=str), encoding="utf8")
    return table


def group_importance(samples: Samples, manifest: list[Feature], features: list[str], params: dict,
                                    spatial_block_m: float | None = 5000, n_splits: int = 5, seed: int = 42,
                                    by: str = "group", spatial_buffer_m: float = 0,
                                    calibration_mode: str = "oof", calibration_fraction: float = 0.2) -> pd.DataFrame:
    """Permute-and-retrain group importance: drop in spatial-CV macro-F1 when all features of a
    group (or window) are shuffled jointly before cross-validation, which removes their information.

    Working on whole groups avoids the dilution of single-feature importance across correlated
    features (e.g. the same index in adjacent windows)."""
    samples, calibration_split = selection_samples(samples, spatial_block_m, seed, spatial_buffer_m,
                                                   calibration_mode, calibration_fraction)
    rng = np.random.default_rng(seed)
    groups = spatial_blocks(samples, spatial_block_m)
    spatial = {"coordinates": np.column_stack([samples.x, samples.yy]), "crs": samples.crs,
               "spatial_buffer_m": spatial_buffer_m}
    X = samples.subset(features).copy()
    base = cross_validate(X, samples.y, samples.polygon, groups, params, n_splits, seed, **spatial)["macro_f1"]
    lookup = {f.name: f for f in manifest}
    keys = {}
    for i, n in enumerate(features):
        f = lookup[n]
        key = f.group if by == "group" else (f.period or "multi-period")
        keys.setdefault(key, []).append(i)
    rows = []
    for key, cols in keys.items():
        Xp = X.copy()
        perm = rng.permutation(len(Xp))
        Xp[:, cols] = Xp[perm][:, cols]
        score = cross_validate(Xp, samples.y, samples.polygon, groups, params, n_splits, seed, **spatial)["macro_f1"]
        rows.append({by: key, "n_features": len(cols), "macro_f1_drop": base - score})
    table = pd.DataFrame(rows).sort_values("macro_f1_drop", ascending=False)
    table.attrs["calibration_split"] = calibration_split
    return table


def feature_reduction(samples: Samples, features: list[str], params: dict, output_dir: str | Path,
                      spatial_block_m: float | None = 5000, n_splits: int = 5, seed: int = 42,
                      step: float = 0.2, min_features: int = 5, spatial_buffer_m: float = 0,
                      calibration_mode: str = "oof", calibration_fraction: float = 0.2) -> pd.DataFrame:
    """Recursive elimination: repeatedly drop the `step` fraction of least important features
    (impurity importance of a model fitted on all samples) and record spatial-CV accuracy.

    The resulting curve supports choosing a reduced model with comparable macro-F1. Because the
    ranking uses all samples, report the curve as model-selection evidence, not as test accuracy.
    """
    from .estimators import importances
    from .model import make_rf

    samples, calibration_split = selection_samples(samples, spatial_block_m, seed, spatial_buffer_m,
                                                   calibration_mode, calibration_fraction)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    groups = spatial_blocks(samples, spatial_block_m)
    spatial = {"coordinates": np.column_stack([samples.x, samples.yy]), "crs": samples.crs,
               "spatial_buffer_m": spatial_buffer_m}
    current = list(features)
    rows = []
    while True:
        X = samples.subset(current)
        cv = cross_validate(X, samples.y, samples.polygon, groups, params, n_splits, seed, **spatial)
        rows.append({"n_features": len(current), "macro_f1": cv["macro_f1"],
                     "overall_accuracy": cv["overall_accuracy"], "features": ";".join(current)})
        if len(current) <= min_features:
            break
        rf = make_rf(params, seed).fit(X, samples.y)
        order = np.argsort(importances(rf, len(current)))
        drop = min(max(1, int(len(current) * step)), len(current) - min_features)
        current = [current[i] for i in sorted(order[drop:])]
    table = pd.DataFrame(rows)
    table.to_csv(out / "feature_reduction.csv", index=False)
    (out / "feature_reduction.json").write_text(json.dumps({
        "calibration_mode": calibration_mode, "calibration_split": calibration_split,
        "note": "Model selection on proper-training samples only; not independent test accuracy.",
        "samples": len(samples.y), "curve": rows}, indent=2), encoding="utf8")
    return table
