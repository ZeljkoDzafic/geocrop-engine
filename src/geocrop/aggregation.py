"""Parcel-level aggregation of pixel out-of-fold predictions: majority vote, mean probability, thresholded.

Input is the pixel table written by `geocrop train` (`oof_pixels.csv`): polygon, spatial_group, label and one
``p_<class>`` column per class. Three decisions per polygon are compared on the same polygons:

* ``majority``      class with most pixel votes; confidence = its vote share.
* ``mean_prob``     class with the highest mean pixel probability; confidence = that mean.
* ``combined``      mean-probability class, accepted only if it equals the majority class and its confidence
                    reaches the threshold; otherwise the polygon is left for review.

The confidence threshold is chosen without the polygons it is applied to: for every spatial group the threshold is
fitted on the polygons of all OTHER groups (smallest value whose accepted polygons reach ``target_accuracy``) and
applied to that group. The out-of-fold probabilities of those other polygons come from models that had the held-out
group in training, so the selection is close to, but not identical with, a fully nested procedure. A fixed
``threshold`` can be given instead. Reported accuracies describe these reference polygons only.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import metrics

GRID = np.round(np.arange(0.40, 0.96, 0.05), 2)


def polygon_table(pixels: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """One row per polygon: reference, group, majority/mean-probability class and their confidences."""
    pcols = [c for c in pixels.columns if c.startswith("p_")]
    if not pcols:
        raise ValueError("The pixel table has no p_<class> probability columns")
    classes = [c[2:] for c in pcols]
    done = pixels[pixels["oof_done"]] if "oof_done" in pixels else pixels
    rows = []
    for pid, g in done.groupby("polygon", sort=True):
        votes = g[pcols].to_numpy().argmax(axis=1)
        share = np.bincount(votes, minlength=len(classes)) / len(votes)
        mean = g[pcols].to_numpy().mean(axis=0)
        # ties go to the alphabetically first class (columns are sorted), as in metrics.polygon_majority
        rows.append({"polygon": pid, "reference": str(g["label"].iloc[0]), "group": str(g["spatial_group"].iloc[0]),
                     "pixels": int(len(g)), "majority": classes[int(share.argmax())], "majority_share": float(share.max()),
                     "mean_prob_class": classes[int(mean.argmax())], "mean_prob": float(mean.max()),
                     "dominant_share_of_reference": float(share[classes.index(str(g["label"].iloc[0]))]
                                                          if str(g["label"].iloc[0]) in classes else 0.0)})
    return pd.DataFrame(rows), classes


def choose_threshold(conf: np.ndarray, correct: np.ndarray, target_accuracy: float, min_accepted: int = 5) -> float:
    """Smallest grid threshold whose accepted polygons are at least `target_accuracy` correct."""
    for t in GRID:
        keep = conf >= t
        if keep.sum() >= min_accepted and correct[keep].mean() >= target_accuracy:
            return float(t)
    return float(GRID[-1])


def compare(pixels: pd.DataFrame, targets=(), target_accuracy: float = 0.95, threshold: float | None = None,
            out: str | Path | None = None) -> dict:
    table, classes = polygon_table(pixels)
    labels = sorted(set(table["reference"]) | set(classes))
    table["maj_ok"] = table["majority"] == table["reference"]
    table["mean_ok"] = table["mean_prob_class"] == table["reference"]
    table["comb_conf"] = table["mean_prob"]
    agree = table["majority"] == table["mean_prob_class"]
    thr = pd.Series(np.nan, index=table.index)
    for g in table["group"].unique():
        test = table["group"] == g
        if threshold is not None:
            thr[test] = threshold
            continue
        train = ~test
        ok = (table.loc[train, "mean_prob_class"] == table.loc[train, "reference"]).to_numpy()
        thr[test] = choose_threshold(table.loc[train, "mean_prob"].to_numpy(), ok, target_accuracy)
    table["threshold"] = thr
    table["combined_accepted"] = agree & (table["mean_prob"] >= table["threshold"])
    methods = {"majority": ("majority", None), "mean_prob": ("mean_prob_class", None),
               "combined": ("mean_prob_class", table["combined_accepted"])}
    report = {"polygons": int(len(table)), "target_accuracy": None if threshold is not None else target_accuracy,
              "fixed_threshold": threshold, "methods": {}}
    matrices = {}
    for name, (col, accepted) in methods.items():
        s = metrics.summary(table["reference"], table[col], labels=labels)
        entry = {"overall_accuracy": s["overall_accuracy"], "kappa": s["kappa"], "macro_f1": s["macro_f1"],
                 "per_class": {c: {"precision": s["per_class"][c]["users_accuracy"],
                                   "recall": s["per_class"][c]["producers_accuracy"], "f1": s["per_class"][c]["f1"]}
                               for c in labels},
                 "target_macro_f1": metrics.target_macro_f1(s, targets) if targets else None}
        if accepted is not None:
            acc = table[accepted]
            entry["accepted_share"] = float(accepted.mean())
            entry["rejected_share"] = float(1 - accepted.mean())
            entry["accuracy_on_accepted"] = float((acc["mean_prob_class"] == acc["reference"]).mean()) if len(acc) else None
            if len(acc):
                sa = metrics.summary(acc["reference"], acc["mean_prob_class"], labels=labels)
                entry["target_macro_f1_on_accepted"] = metrics.target_macro_f1(sa, targets) if targets else None
            entry["rejected_by_reference_class"] = table[~accepted]["reference"].value_counts().to_dict()
            entry["mean_threshold"] = float(table["threshold"].mean())
        matrices[name] = pd.DataFrame(s["confusion_matrix"], index=labels, columns=labels)
        report["methods"][name] = entry
    report["note"] = ("Thresholds are chosen on polygons of the other spatial groups and applied to the held-out "
                      "group; reference polygons only, not independent validation.")
    if out is not None:
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        table.to_csv(out / "parcel_aggregation_polygons.csv", index=False)
        for name, cm in matrices.items():
            cm.to_csv(out / f"parcel_confusion_{name}.csv")
        (out / "parcel_aggregation.json").write_text(json.dumps(report, indent=2), encoding="utf8")
    report["table"] = table
    return report
