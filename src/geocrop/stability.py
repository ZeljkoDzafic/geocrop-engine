"""How much do cross-validation scores depend on one particular spatial-block partition?

A single 5 km grid is one arbitrary draw of blocks. This module repeats the spatially blocked cross-validation
over several partitions (block sizes and shifted grid origins), keeps every polygon inside one block in each of
them, and reports the spread of overall accuracy, macro F1 and target macro F1. Optional training-only controls
(`train_exclude`, `train_max_samples_per_polygon`) are applied identically to every partition.

The spread describes sensitivity to the partition, not a confidence interval and not independent validation.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import metrics, model
from .samples import Samples


def partitions(block_sizes_m, n_offsets: int = 1) -> list[tuple[float, tuple[float, float]]]:
    """(block size, grid origin shift) pairs; shifts walk the diagonal in ``n_offsets`` equal steps."""
    if n_offsets < 1:
        raise ValueError("n_offsets must be >= 1")
    out = []
    for size in block_sizes_m:
        if size <= 0:
            raise ValueError("block sizes must be positive")
        for k in range(n_offsets):
            shift = size * k / n_offsets
            out.append((float(size), (shift, shift)))
    return out


def _centroids(samples: Samples) -> pd.DataFrame:
    return pd.DataFrame({"p": samples.polygon, "x": samples.x, "y": samples.yy}).groupby("p")[["x", "y"]].mean()


def _watch_rows(variant: dict, samples: Samples, groups, oof, folds, watch, radii_m, train_ok,
                proba=None, classes=()) -> list[dict]:
    """OOF decision of each watched polygon plus how much same-class training data stayed nearby."""
    centre = _centroids(samples)
    poly_class = pd.Series(samples.y, index=samples.polygon).groupby(level=0).first()
    fold_of = pd.Series(folds, index=samples.polygon).groupby(level=0).first()
    group_of = pd.Series(groups, index=samples.polygon).groupby(level=0).first()
    ok = np.ones(len(samples.y), dtype=bool) if train_ok is None else np.asarray(train_ok)
    rows = []
    for pid in watch:
        pid = str(pid)
        if pid not in centre.index:
            continue
        mask = samples.polygon == pid
        pred = pd.Series(oof[mask]).value_counts()
        true = poly_class[pid]
        fold = fold_of[pid]
        in_train = pd.Series(ok & (folds != fold) & (samples.y == true), index=samples.polygon).groupby(level=0).any()
        same = centre.loc[in_train[in_train].index]
        dist = np.hypot(same.x - centre.loc[pid, "x"], same.y - centre.loc[pid, "y"])
        row = {**variant, "polygon_id": pid, "true_class": true, "pixels": int(mask.sum()),
               "majority_prediction": pred.index[0], "share_majority": float(pred.iloc[0] / mask.sum()),
               "share_true_class": float(pred.get(true, 0) / mask.sum()), "fold": int(fold), "group": str(group_of[pid]),
               "watched_polygons_in_same_fold": int(sum(fold_of.get(str(w), -1) == fold for w in watch if str(w) != pid)),
               "same_class_polygons_in_same_group": int(((poly_class == true) & (group_of == group_of[pid])).sum() - 1),
               "same_class_training_polygons": int(len(same))}
        if proba is not None:
            for c, v in zip(classes, proba[mask].mean(axis=0)):
                row[f"mean_p_{c}"] = float(v)
        for r in radii_m:
            row[f"same_class_training_within_{int(r)}m"] = int((dist <= r).sum())
        rows.append(row)
    return rows


def run(samples: Samples, params: dict, block_sizes_m=(5000,), n_offsets: int = 4, n_splits: int = 5,
        seed: int = 42, targets=(), features: list[str] | None = None, train_exclude=(),
        train_max_samples_per_polygon: int | None = None, out: str | Path | None = None,
        polygon_repeats: int = 0, watch=(), radii_m=(2000, 5000, 10000), train_relabel: dict | None = None) -> dict:
    """Spatial-block CV over alternative partitions, optionally plus stratified polygon-grouped CV.

    ``polygon_repeats`` > 0 adds that many repeats of stratified polygon-grouped k-fold CV (whole polygons are held
    out, neighbouring polygons stay in training, so it is *less* strict than block CV). ``watch`` lists polygon
    ids whose out-of-fold decision, fold mates and nearby same-class training data are recorded per variant.
    """
    X = samples.subset(features or samples.features)
    y = samples.y
    train_y = model.train_labels(y, samples.polygon, train_relabel) if train_relabel else None
    train_ok = None
    if len(train_exclude) or train_max_samples_per_polygon:
        train_ok = model.train_mask(samples.polygon, train_exclude, train_max_samples_per_polygon, seed)
    variants = [{"variant": f"block_{int(size)}m_shift_{int(shift[0])}m", "kind": "spatial_blocks", "block_m": size,
                 "offset_m": shift[0], "seed": seed, "_groups": None} for size, shift in partitions(block_sizes_m, n_offsets)]
    variants += [{"variant": f"polygon_stratified_seed_{seed + k}", "kind": "polygon_stratified", "block_m": None,
                  "offset_m": None, "seed": seed + k, "_groups": samples.polygon} for k in range(polygon_repeats)]
    rows, watch_rows, confusions = [], [], {}
    polygons_per_class = pd.Series(y, index=samples.polygon).groupby(level=0).first().value_counts()
    for v in variants:
        meta = {k: v[k] for k in ("variant", "kind", "block_m", "offset_m", "seed")}
        if v["_groups"] is None:
            groups = model.spatial_blocks(samples, v["block_m"], (v["offset_m"], v["offset_m"]))
        else:
            groups = v["_groups"]
        if len(np.unique(groups)) < n_splits:
            rows.append({**meta, "status": f"skipped: {len(np.unique(groups))} groups"})
            continue
        cv = model.cross_validate(X, y, samples.polygon, groups, params, n_splits, v["seed"], train_ok=train_ok,
                                  train_y=train_y)
        oof = cv["_oof"]
        plan = model._fold_plan(y, groups, n_splits, v["seed"], polygon=samples.polygon)
        folds = np.zeros(len(y), dtype=int)
        for k, (_, te, _) in enumerate(plan):
            folds[te] = k
        yp, pp = metrics.polygon_majority(y, oof, samples.polygon)
        par = metrics.summary(yp, pp, labels=np.unique(y))
        row = {**meta, "status": "ok", "groups": int(len(np.unique(groups))),
               "overall_accuracy": cv["overall_accuracy"], "macro_f1": cv["macro_f1"],
               "parcel_overall_accuracy": par["overall_accuracy"]}
        per_fold_classes = [pd.Series(y[te], index=samples.polygon[te]).groupby(level=0).first().value_counts()
                            for _, te, _ in plan]
        row["min_test_polygons_per_class"] = int(min(
            fc.reindex(polygons_per_class.index, fill_value=0).min() for fc in per_fold_classes))
        row["classes_absent_from_a_test_fold"] = int(sum(
            (fc.reindex(polygons_per_class.index, fill_value=0) == 0).any() for fc in per_fold_classes))
        for t in targets:
            row[f"f1_{t}"] = cv["per_class"][t]["f1"] if t in cv["per_class"] else None
        if train_relabel:     # scores on the polygons that were NOT relabelled: did the others change?
            rest = ~np.isin(samples.polygon, [str(k) for k in train_relabel])
            other = metrics.summary(y[rest], oof[rest], labels=np.unique(y))
            row["overall_accuracy_unrelabelled"] = other["overall_accuracy"]
            row["kappa"] = cv["kappa"]
            for t in targets:
                pc = other["per_class"].get(t, {})
                row[f"precision_{t}_unrelabelled"] = pc.get("users_accuracy")
                row[f"recall_{t}_unrelabelled"] = pc.get("producers_accuracy")
                row[f"f1_{t}_unrelabelled"] = pc.get("f1")
        if targets:
            row["target_macro_f1"] = metrics.target_macro_f1(cv, targets)
            row["parcel_target_macro_f1"] = metrics.target_macro_f1(par, targets)
        rows.append(row)
        labels = np.unique(y)
        confusions[v["variant"]] = pd.DataFrame(pd.crosstab(pd.Categorical(y, labels), pd.Categorical(oof, labels),
                                                            dropna=False).to_numpy(), index=labels, columns=labels)
        watch_rows += _watch_rows(meta, samples, groups, oof, folds, watch, radii_m, train_ok, cv["_oof_proba"], cv["_classes"])
    table = pd.DataFrame(rows)
    ok = table[table["status"] == "ok"]
    cols = [c for c in ("overall_accuracy", "macro_f1", "target_macro_f1", "parcel_target_macro_f1")
            + tuple(f"f1_{t}" for t in targets) if c in ok.columns]
    summary = {c: {"mean": float(ok[c].mean()), "std": float(ok[c].std(ddof=1)) if len(ok) > 1 else 0.0,
                   "min": float(ok[c].min()), "max": float(ok[c].max())} for c in cols}
    watch_table = pd.DataFrame(watch_rows)
    result = {"partitions": int(len(ok)), "summary": summary, "train_controls": {
        "excluded_polygons": sorted(map(str, train_exclude)),
        "max_samples_per_polygon": train_max_samples_per_polygon,
        "relabelled_for_training_only": {str(k): str(v) for k, v in (train_relabel or {}).items()}},
        "note": ("Spread over alternative partitions of the same samples. It shows sensitivity to the "
                 "partition, not a confidence interval; folds of different partitions are not independent. "
                 "Polygon-grouped CV keeps neighbouring polygons in training and is therefore optimistic "
                 "compared with block CV.")}
    if len(watch_table):
        wt = watch_table[watch_table["true_class"].notna()]
        result["watch"] = {pid: {"variants": int(len(g)),
                                 "variants_not_predicted_as_true_class": int((g["majority_prediction"] != g["true_class"]).sum())}
                           for pid, g in wt.groupby("polygon_id")}
    if out is not None:
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        table.to_csv(out / "cv_stability.csv", index=False)
        (out / "cv_stability.json").write_text(json.dumps(result, indent=2), encoding="utf8")
        if len(watch_table):
            watch_table.to_csv(out / "watch_polygons.csv", index=False)
        (out / "confusion").mkdir(exist_ok=True)
        for name, cm in confusions.items():
            cm.to_csv(out / "confusion" / f"{name}.csv")
    result["table"] = table
    result["watch_table"] = watch_table
    return result
