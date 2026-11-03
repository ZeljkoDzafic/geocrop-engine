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


def run(samples: Samples, params: dict, block_sizes_m=(5000,), n_offsets: int = 4, n_splits: int = 5,
        seed: int = 42, targets=(), features: list[str] | None = None, train_exclude=(),
        train_max_samples_per_polygon: int | None = None, out: str | Path | None = None) -> dict:
    X = samples.subset(features or samples.features)
    y = samples.y
    train_ok = None
    if len(train_exclude) or train_max_samples_per_polygon:
        train_ok = model.train_mask(samples.polygon, train_exclude, train_max_samples_per_polygon, seed)
    rows = []
    for size, shift in partitions(block_sizes_m, n_offsets):
        groups = model.spatial_blocks(samples, size, shift)
        if len(np.unique(groups)) < n_splits:
            rows.append({"block_m": size, "offset_m": shift[0], "status": f"skipped: {len(np.unique(groups))} blocks"})
            continue
        cv = model.cross_validate(X, y, samples.polygon, groups, params, n_splits, seed, train_ok=train_ok)
        oof = cv["_oof"]
        yp, pp = metrics.polygon_majority(y, oof, samples.polygon)
        par = metrics.summary(yp, pp, labels=np.unique(y))
        row = {"block_m": size, "offset_m": shift[0], "status": "ok", "blocks": int(len(np.unique(groups))),
               "overall_accuracy": cv["overall_accuracy"], "macro_f1": cv["macro_f1"],
               "parcel_overall_accuracy": par["overall_accuracy"]}
        for t in targets:
            row[f"f1_{t}"] = cv["per_class"][t]["f1"] if t in cv["per_class"] else None
        if targets:
            row["target_macro_f1"] = metrics.target_macro_f1(cv, targets)
            row["parcel_target_macro_f1"] = metrics.target_macro_f1(par, targets)
        rows.append(row)
    table = pd.DataFrame(rows)
    ok = table[table["status"] == "ok"]
    cols = [c for c in ("overall_accuracy", "macro_f1", "target_macro_f1", "parcel_target_macro_f1")
            if c in ok.columns]
    summary = {c: {"mean": float(ok[c].mean()), "std": float(ok[c].std(ddof=1)) if len(ok) > 1 else 0.0,
                   "min": float(ok[c].min()), "max": float(ok[c].max())} for c in cols}
    result = {"partitions": int(len(ok)), "summary": summary, "train_controls": {
        "excluded_polygons": sorted(map(str, train_exclude)),
        "max_samples_per_polygon": train_max_samples_per_polygon},
        "note": ("Spread over alternative block partitions of the same samples. It shows sensitivity to the "
                 "partition, not a confidence interval; folds of different partitions are not independent.")}
    if out is not None:
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        table.to_csv(out / "cv_stability.csv", index=False)
        (out / "cv_stability.json").write_text(json.dumps(result, indent=2), encoding="utf8")
    result["table"] = table
    return result
