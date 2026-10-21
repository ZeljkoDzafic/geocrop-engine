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
from .model import cross_validate, spatial_blocks
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
        targets: list[str] | None = None) -> pd.DataFrame:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    experiments = experiments or DEFAULT_EXPERIMENTS
    params = params or {"n_estimators": 300, "max_features": "sqrt", "class_weight": "balanced_subsample", "n_jobs": -1}
    groups = spatial_blocks(samples, spatial_block_m)
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
        cv = cross_validate(samples.subset(names), samples.y, samples.polygon, groups, params, n_splits, seed)
        oof = cv.pop("_oof")
        cv.pop("_oof_proba"), cv.pop("_classes")
        if reference_oof is None:
            reference_name, reference_oof = name, oof
        test = metrics.mcnemar(samples.y, oof, reference_oof)
        row = {"experiment": name, "status": "ok", "n_features": len(names),
               "overall_accuracy": cv["overall_accuracy"], "kappa": cv["kappa"], "macro_f1": cv["macro_f1"],
               "macro_f1_fold_std": cv["fold_macro_f1_std"], "ece": cv["calibration"]["ece"],
               "vs": reference_name, "mcnemar_p": test["p_value"]}
        for t in targets or []:
            row[f"f1_{t}"] = cv["per_class"].get(t, {}).get("f1")
        rows.append(row)
        full[name] = {"spec": spec, "features": names, "cv": cv}
    table = pd.DataFrame(rows)
    table.to_csv(out / "ablation.csv", index=False)
    (out / "ablation.json").write_text(json.dumps({
        "spatial_block_m": spatial_block_m, "n_splits": n_splits, "seed": seed, "params": params,
        "n_samples": int(len(samples.y)), "n_groups": int(len(np.unique(groups))), "experiments": full},
        indent=2, default=str), encoding="utf8")
    return table


def group_importance(samples: Samples, manifest: list[Feature], features: list[str], params: dict,
                                    spatial_block_m: float | None = 5000, n_splits: int = 5, seed: int = 42,
                                    by: str = "group") -> pd.DataFrame:
    """Permute-and-retrain group importance: drop in spatial-CV macro-F1 when all features of a
    group (or window) are shuffled jointly before cross-validation, which removes their information.

    Working on whole groups avoids the dilution of single-feature importance across correlated
    features (e.g. the same index in adjacent windows)."""
    rng = np.random.default_rng(seed)
    groups = spatial_blocks(samples, spatial_block_m)
    X = samples.subset(features).copy()
    base = cross_validate(X, samples.y, samples.polygon, groups, params, n_splits, seed)["macro_f1"]
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
        score = cross_validate(Xp, samples.y, samples.polygon, groups, params, n_splits, seed)["macro_f1"]
        rows.append({by: key, "n_features": len(cols), "macro_f1_drop": base - score})
    return pd.DataFrame(rows).sort_values("macro_f1_drop", ascending=False)


def feature_reduction(samples: Samples, features: list[str], params: dict, output_dir: str | Path,
                      spatial_block_m: float | None = 5000, n_splits: int = 5, seed: int = 42,
                      step: float = 0.2, min_features: int = 5) -> pd.DataFrame:
    """Recursive elimination: repeatedly drop the `step` fraction of least important features
    (impurity importance of a model fitted on all samples) and record spatial-CV accuracy.

    The resulting curve supports choosing a reduced model with comparable macro-F1. Because the
    ranking uses all samples, report the curve as model-selection evidence, not as test accuracy.
    """
    from .estimators import importances
    from .model import make_rf

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    groups = spatial_blocks(samples, spatial_block_m)
    current = list(features)
    rows = []
    while True:
        X = samples.subset(current)
        cv = cross_validate(X, samples.y, samples.polygon, groups, params, n_splits, seed)
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
    return table
