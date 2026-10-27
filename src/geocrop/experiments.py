"""Ablation experiments on identical samples and identical spatial folds.

Each experiment is a feature subset defined by groups / periods / explicit names (see
`geocrop.features.select`). Because samples and folds are shared, differences between rows
are attributable to the feature set alone. Output: `ablation.csv` (one row per experiment,
ready for a manuscript table) and `ablation.json` (full metrics, per-class F1, fold spread).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import metrics
from .features import GROUPS, Feature, select
from .model import cross_validate, cv_groups, selection_samples, spatial_blocks, transfer_validation
from .samples import Samples

# Default design mirrors the study protocol; override in the YAML `experiments:` section.
DEFAULT_EXPERIMENTS = {
    "E0_S2_bands": {"groups": ["s2_band"]},
    "E1_S2_bands_indices": {"groups": ["s2_band", "s2_index"]},
    "E2_S2_full_temporal": {"groups": ["s2_band", "s2_index", "delta", "seasonal"], "sensors": ["S2"]},
    "E3_S1_only": {"groups": ["s1"]},
    "E4_S1_S2_fusion": {"groups": ["s2_band", "s2_index", "s1", "delta", "seasonal"]},
}
# Added to the defaults only when the stack carries an `emb` group (geocrop.embeddings): handcrafted vs
# foundation-model embeddings vs both, on the same samples and folds.
HANDCRAFTED_GROUPS = [g for g in GROUPS if g != "emb"]
EMBEDDING_EXPERIMENTS = {
    "E5_handcrafted_all": {"groups": HANDCRAFTED_GROUPS},
    "E6_embeddings_only": {"groups": ["emb"]},
    "E7_handcrafted_plus_embeddings": {"groups": [*HANDCRAFTED_GROUPS, "emb"]},
}


def run(samples: Samples, manifest: list[Feature], output_dir: str | Path, experiments: dict | None = None,
        params: dict | None = None, n_splits: int = 5, spatial_block_m: float | None = 5000, seed: int = 42,
        targets: list[str] | None = None, holdout_fraction: float = 0.0, spatial_buffer_m: float = 0,
        calibration_mode: str = "oof", calibration_fraction: float = 0.2,
        cv_strategy: str = "spatial_blocks") -> pd.DataFrame:
    """Compare feature subsets on identical samples and spatial folds.

    * cv_strategy ``region``/``year``/``region_year`` additionally scores every experiment by leave-one-group-out
      validation (`transfer_*` columns: out-of-domain accuracy, macro-F1, transfer gap, worst group).
    * holdout_fraction > 0 locks away whole spatial blocks first: every experiment is additionally trained
      on the remaining data and scored once on the holdout (`holdout_*` columns, McNemar on the holdout).
    * In ``holdout`` calibration mode only the proper-training objects (same deterministic split as `train`)
      are used, so calibration objects never influence feature selection.
    * spatial_buffer_m purges training polygons near test samples in every fold.
    """
    from .estimators import predict_labels
    from .model import make_estimator, split_final_holdout

    t_run = time.perf_counter()
    samples, test = split_final_holdout(samples, spatial_block_m, holdout_fraction, seed)
    samples, calibration_split = selection_samples(samples, spatial_block_m, seed, spatial_buffer_m,
                                                   calibration_mode, calibration_fraction)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    available = [f for f in manifest if f.name in samples.features]
    if not experiments:
        experiments = dict(DEFAULT_EXPERIMENTS)
        if any(f.group == "emb" for f in available):
            experiments.update(EMBEDDING_EXPERIMENTS)
    params = params or {"n_estimators": 300, "max_features": "sqrt", "class_weight": "balanced_subsample", "n_jobs": -1}
    groups = spatial_blocks(samples, spatial_block_m)
    spatial = {"coordinates": np.column_stack([samples.x, samples.yy]), "crs": samples.crs,
               "spatial_buffer_m": spatial_buffer_m}
    transfer_groups = None if cv_strategy == "spatial_blocks" else cv_groups(samples, cv_strategy, spatial_block_m)
    ref_hold = None
    rows, full = [], {}
    reference_name, reference_oof = None, None   # first successful experiment is the paired-test baseline
    for name, spec in experiments.items():
        try:
            names = select(available, groups=spec.get("groups"), periods=spec.get("periods"),
                           names=spec.get("features"), exclude=spec.get("exclude"), sensors=spec.get("sensors"))
        except ValueError:
            rows.append({"experiment": name, "status": "skipped: no matching features"})
            continue
        X = samples.subset(names)
        cv = cross_validate(X, samples.y, samples.polygon, groups, params, n_splits, seed, **spatial)
        tv = None
        if transfer_groups is not None:
            tv = transfer_validation(X, samples.y, samples.polygon, transfer_groups, groups, params, cv_strategy,
                                     n_splits, seed, in_domain=cv, **spatial)
            tv.pop("_oof")
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
        if tv is not None:
            row.update({"transfer_strategy": cv_strategy,
                        "transfer_overall_accuracy": tv["out_of_domain"]["overall_accuracy"],
                        "transfer_macro_f1": tv["out_of_domain"]["macro_f1"],
                        "transfer_gap_macro_f1": tv["transfer_gap"]["macro_f1"],
                        "transfer_worst_group": tv["worst_group"]})
        if test is not None:
            pred = predict_labels(make_estimator(params, seed).fit(X, samples.y), test.subset(names)).astype(str)
            h = metrics.summary(test.y, pred, labels=np.unique(np.concatenate([samples.y, test.y])))
            if ref_hold is None:
                ref_hold = pred
            row.update({"holdout_overall_accuracy": h["overall_accuracy"], "holdout_macro_f1": h["macro_f1"],
                        "holdout_mcnemar_p": metrics.mcnemar(test.y, pred, ref_hold)["p_value"]})
        rows.append(row)
        full[name] = {"spec": spec, "features": names, "cv": cv}
        if tv is not None:
            full[name]["transfer_validation"] = tv
    table = pd.DataFrame(rows)
    table.to_csv(out / "ablation.csv", index=False)
    (out / "ablation.json").write_text(json.dumps({
        "spatial_block_m": spatial_block_m, "spatial_buffer_m": spatial_buffer_m, "n_splits": n_splits, "seed": seed,
        "cv_strategy": cv_strategy,
        "params": params, "calibration_mode": calibration_mode, "calibration_split": calibration_split,
        "holdout_fraction": holdout_fraction, "holdout_samples": 0 if test is None else int(len(test.y)),
        "n_samples": int(len(samples.y)), "n_groups": int(len(np.unique(groups))), "experiments": full},
        indent=2, default=str), encoding="utf8")
    from . import registry
    registry.record_classification("ablation", out, rows, samples, n_splits=n_splits, seeds=[seed],
                                   spatial_block_m=spatial_block_m, spatial_buffer_m=spatial_buffer_m, holdout=test,
                                   strategies=() if cv_strategy == "spatial_blocks" else (cv_strategy,),
                                   candidate_key="experiment", started=t_run,
                                   protocol={"params": params, "calibration_mode": calibration_mode})
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


def transfer(samples: Samples, output_dir: str | Path, features: list[str] | None = None, params: dict | None = None,
             strategies=("region", "year"), n_splits: int = 5, spatial_block_m: float | None = 5000, seed: int = 42,
             spatial_buffer_m: float = 0) -> pd.DataFrame:
    """Spatio-temporal transfer study: leave-one-region-out and/or leave-one-year-out next to spatial CV.

    Writes ``transfer.csv`` (one row per strategy and held-out group, plus a pooled ``ALL`` row per strategy)
    and ``transfer.json`` (full metrics). Strategies whose metadata are missing are reported as skipped.
    """
    t_run = time.perf_counter()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    features = features or samples.features
    params = params or {"n_estimators": 300, "max_features": "sqrt", "class_weight": "balanced_subsample",
                        "n_jobs": -1}
    X = samples.subset(features)
    blocks = spatial_blocks(samples, spatial_block_m)
    spatial = {"coordinates": np.column_stack([samples.x, samples.yy]), "crs": samples.crs,
               "spatial_buffer_m": spatial_buffer_m}
    in_domain = cross_validate(X, samples.y, samples.polygon, blocks, params, n_splits, seed, **spatial)
    rows, full = [], {}
    for strategy in strategies:
        try:
            groups = cv_groups(samples, strategy, spatial_block_m)
            tv = transfer_validation(X, samples.y, samples.polygon, groups, blocks, params, strategy, n_splits,
                                     seed, in_domain=in_domain, **spatial)
        except ValueError as exc:
            rows.append({"strategy": strategy, "held_out_group": None, "status": f"skipped: {exc}"})
            continue
        tv.pop("_oof")
        full[strategy] = tv
        rows.append({"strategy": strategy, "held_out_group": "ALL", "status": "ok",
                     "n_test": tv["evaluated_samples"],
                     **{f"in_domain_{k}": v for k, v in tv["in_domain"].items() if k != "protocol"},
                     **{f"out_of_domain_{k}": tv["out_of_domain"][k]
                        for k in ("overall_accuracy", "kappa", "macro_f1")},
                     **{f"gap_{k}": v for k, v in tv["transfer_gap"].items()}})
        for r in tv["per_group"]:
            rows.append({"strategy": strategy, "status": "ok",
                         **{k: (";".join(v) if isinstance(v, list) else v) for k, v in r.items()
                            if not isinstance(v, dict)}})
    table = pd.DataFrame(rows)
    table.to_csv(out / "transfer.csv", index=False)
    (out / "transfer.json").write_text(json.dumps({
        "spatial_block_m": spatial_block_m, "spatial_buffer_m": spatial_buffer_m, "n_splits": n_splits,
        "seed": seed, "params": params, "n_samples": int(len(samples.y)), "features": len(features),
        "in_domain_spatial_cv": {k: in_domain[k] for k in ("overall_accuracy", "kappa", "macro_f1")},
        "strategies": full}, indent=2, default=str), encoding="utf8")
    from . import registry
    registry.record_transfer(out, rows, samples, features=features, strategies=strategies, n_splits=n_splits,
                             seed=seed, spatial_block_m=spatial_block_m, spatial_buffer_m=spatial_buffer_m,
                             params=params, started=t_run)
    return table


def _synthetic_map(rng, size: int, parcel: int, n_classes: int, n_features: int, centres, noise: float,
                   parcel_noise: float):
    """Synthetic parcel map: square parcels of random class; features = class centre + parcel offset + pixel noise."""
    k = size // parcel
    labels = rng.integers(0, n_classes, size=(k, k))
    label = np.kron(labels, np.ones((parcel, parcel), dtype=int))
    pid = np.kron(np.arange(k * k).reshape(k, k), np.ones((parcel, parcel), dtype=int))
    offset = rng.normal(0, parcel_noise, size=(k * k, n_features))
    X = centres[label] + offset[pid] + rng.normal(0, noise, size=(size, size, n_features))
    edge = np.zeros((size, size), dtype=bool)
    edge[np.r_[0:size:parcel], :] = edge[np.r_[parcel - 1:size:parcel], :] = True
    edge[:, np.r_[0:size:parcel]] = edge[:, np.r_[parcel - 1:size:parcel]] = True
    return X.reshape(-1, n_features), label.ravel(), pid.ravel(), edge.ravel()


def smoothing_comparison(output_dir: str | Path | None = None, seed: int = 42, size: int = 160, parcel: int = 10,
                         n_classes: int = 4, n_features: int = 6, noise: float = 2.0, parcel_noise: float = 0.3,
                         settings=({"window": 5, "smoothness": 10.0}, {"window": 7, "smoothness": 10.0},
                                   {"window": 7, "smoothness": 1.0}),
                         alpha: float = 0.1, n_estimators: int = 100) -> dict:
    """SYNTHETIC experiment: accuracy and conformal coverage with and without probability smoothing.

    Three independent synthetic parcel maps (same class signatures) are used for training, conformal
    calibration and testing, so no pixel is used twice. For each smoothing setting the report gives pixel
    accuracy (all pixels and parcel-edge pixels), parcel accuracy after mean-probability aggregation and
    parcel-level conformal coverage at level 1 - alpha with thresholds calibrated (a) on probabilities
    processed the same way and (b) on unsmoothed probabilities (the situation of a model whose
    conformal.json was fitted on unsmoothed out-of-fold probabilities). Results are synthetic and only
    show the mechanics; they are not evidence about real Sentinel-2 maps.
    """
    from . import metrics as m
    from . import uncertainty as u
    from .model import make_estimator, smooth_probabilities

    rng = np.random.default_rng(seed)
    centres = rng.normal(0, 1.5, size=(n_classes, n_features))
    maps = [_synthetic_map(rng, size, parcel, n_classes, n_features, centres, noise, parcel_noise) for _ in range(3)]
    (Xt, yt, _, _), (Xc, yc, pc, _), (Xs, ys, ps, es) = maps
    classes = [str(c) for c in range(n_classes)]
    model = make_estimator({"n_estimators": n_estimators, "n_jobs": -1, "min_samples_leaf": 5}, seed).fit(
        Xt, np.asarray(yt).astype(str))

    def proba_map(X):
        p = model.predict_proba(X).reshape(size, size, n_classes)
        return np.moveaxis(p, -1, 0)

    raw_c, raw_s = proba_map(Xc), proba_map(Xs)
    valid = np.ones((size, size), dtype=bool)

    def parcel_mean(p, pid):
        flat = p.reshape(n_classes, -1).T
        frame = pd.DataFrame(flat).groupby(pid).mean()
        return frame.to_numpy(), frame.index.to_numpy()

    def parcel_labels(y, pid):
        return pd.Series(y).groupby(pid).first().astype(str).to_numpy()

    yc_parcel, ys_parcel = parcel_labels(yc, pc), parcel_labels(ys, ps)
    raw_cal = parcel_mean(raw_c, pc)[0]
    raw_thr = u.conformal_thresholds(raw_cal, yc_parcel, classes, (alpha,), sparse_class_policy="include")

    def evaluate(cal, test, name, params):
        pix = np.asarray(classes)[test.reshape(n_classes, -1).argmax(axis=0)]
        truth = np.asarray(ys).astype(str)
        all_px = m.summary(truth, pix, labels=classes)
        edge_px = m.summary(truth[es], pix[es], labels=classes)
        cal_parcel = parcel_mean(cal, pc)[0]
        test_parcel = parcel_mean(test, ps)[0]
        parcel_pred = np.asarray(classes)[test_parcel.argmax(axis=1)]
        par = m.summary(ys_parcel, parcel_pred, labels=classes)
        own = u.conformal_thresholds(cal_parcel, yc_parcel, classes, (alpha,), sparse_class_policy="include")
        cov_own = u.coverage(u.prediction_sets(test_parcel, classes, own["thresholds"][str(alpha)]), ys_parcel)
        cov_raw = u.coverage(u.prediction_sets(test_parcel, classes, raw_thr["thresholds"][str(alpha)]), ys_parcel)
        return {"variant": name, **params,
                "pixel_overall_accuracy": all_px["overall_accuracy"], "pixel_macro_f1": all_px["macro_f1"],
                "edge_pixel_overall_accuracy": edge_px["overall_accuracy"],
                "parcel_overall_accuracy": par["overall_accuracy"], "parcel_macro_f1": par["macro_f1"],
                "coverage_own_calibration": cov_own["coverage"], "set_size_own_calibration": cov_own["mean_set_size"],
                "coverage_unsmoothed_calibration": cov_raw["coverage"],
                "set_size_unsmoothed_calibration": cov_raw["mean_set_size"]}

    rows = [evaluate(raw_c, raw_s, "unsmoothed", {"window": None, "smoothness": None})]
    for st in settings:
        sc = smooth_probabilities(raw_c, valid, st["window"], st["smoothness"])
        ss = smooth_probabilities(raw_s, valid, st["window"], st["smoothness"])
        rows.append(evaluate(sc, ss, "smoothed", dict(st)))
    report = {"synthetic": True, "seed": seed, "alpha": alpha, "target_coverage": 1 - alpha,
              "design": {"size_px": size, "parcel_px": parcel, "classes": n_classes, "features": n_features,
                         "pixel_noise_sd": noise, "parcel_noise_sd": parcel_noise,
                         "maps": "independent training / calibration / test maps",
                         "test_parcels": int(len(ys_parcel)), "calibration_parcels": int(len(yc_parcel))},
              "rows": rows,
              "note": "SYNTHETIC data with square parcels and i.i.d. pixel noise; illustrates the mechanics only."}
    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out / "smoothing_comparison.csv", index=False)
        (out / "smoothing_comparison.json").write_text(json.dumps(report, indent=2), encoding="utf8")
    return report
