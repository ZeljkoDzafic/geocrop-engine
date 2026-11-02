"""Jeffries-Matusita (JM) separability of class pairs, per feature, from parcel means.

A feature-wise, pairwise diagnostic of *when* and *with what* two crops become separable. The unit is the
reference polygon: each feature is first averaged inside a polygon, then the class distributions are compared
over polygons, so a large parcel carries no more weight than a small one. Univariate Gaussian classes are
assumed; with few polygons per class the variances are uncertain, so read it as a ranking, not a test.

    B  = (m1 - m2)^2 / (8 v) + 0.5 ln( v / sqrt(v1 v2) ),   v = (v1 + v2) / 2
    JM = 2 (1 - exp(-B))                                      0 (identical) .. 2 (separable)

JM is not a model: a high JM for one feature does not guarantee a good multivariate classifier.
"""
from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from . import features as featmod
from .samples import extract

MIN_POLYGONS = 3


def jeffries_matusita(a, b) -> float:
    """JM distance of two 1-D samples (NaN if a class has < 3 finite values or zero variance)."""
    a, b = (np.asarray(v, dtype=float) for v in (a, b))
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < MIN_POLYGONS or len(b) < MIN_POLYGONS:
        return float("nan")
    m1, m2, v1, v2 = a.mean(), b.mean(), a.var(ddof=1), b.var(ddof=1)
    if v1 <= 0 or v2 <= 0:
        return float("nan")
    v = (v1 + v2) / 2
    bhatt = (m1 - m2) ** 2 / (8 * v) + 0.5 * np.log(v / np.sqrt(v1 * v2))
    return float(2 * (1 - np.exp(-bhatt)))


def parcel_means(stack, polygons, label_field: str, classes: list[str], id_field: str = "polygon_id",
                 features: list[str] | None = None) -> pd.DataFrame:
    """One row per polygon: id, class and the mean of every feature over its valid pixels."""
    names = features or [f.name for f in featmod.load_manifest(stack)]
    s = extract(stack, polygons, label_field=label_field, id_field=id_field, max_per_polygon=None, features=names)
    df = pd.DataFrame(s.X, columns=s.features)
    df.insert(0, "class_name", s.y)
    df.insert(0, "polygon_id", s.polygon)
    df = df[df["class_name"].isin(classes)]
    return df.groupby(["polygon_id", "class_name"], sort=True).mean().reset_index()


def pairwise(means: pd.DataFrame, classes: list[str], period_of: dict[str, str] | None = None) -> pd.DataFrame:
    """Long table (feature, period, pair, jm, n_a, n_b) for every class pair and feature."""
    rows = []
    for feature in [c for c in means.columns if c not in ("polygon_id", "class_name")]:
        for a, b in itertools.combinations(classes, 2):
            xa = means.loc[means["class_name"] == a, feature].to_numpy()
            xb = means.loc[means["class_name"] == b, feature].to_numpy()
            rows.append({"feature": feature, "period": (period_of or {}).get(feature, ""), "pair": f"{a}-{b}",
                         "jm": jeffries_matusita(xa, xb), "n_a": int(np.isfinite(xa).sum()),
                         "n_b": int(np.isfinite(xb).sum())})
    return pd.DataFrame(rows)


def best_per_pair(table: pd.DataFrame) -> pd.DataFrame:
    """Best feature per (period, pair) and the mean over pairs of those maxima, per period."""
    ok = table.dropna(subset=["jm"])
    if ok.empty:
        return ok
    best = ok.loc[ok.groupby(["period", "pair"])["jm"].idxmax()].reset_index(drop=True)
    mean = best.groupby("period")["jm"].mean().rename("mean_best_pairwise_jm")
    return best.merge(mean, on="period")


def run(stack, polygons, label_field: str, classes: list[str], out: str | Path, id_field: str = "polygon_id",
        groups: list[str] | None = None) -> dict:
    """Parcel means -> JM tables written to `out` (parcel_means.csv, jm_all.csv, jm_best_per_pair.csv)."""
    manifest = featmod.load_manifest(stack)
    keep = [f for f in manifest if f.period and (groups is None or f.group in groups)]
    period_of = {f.name: f.period for f in keep}
    means = parcel_means(stack, polygons, label_field, classes, id_field, [f.name for f in keep])
    table = pairwise(means, classes, period_of)
    best = best_per_pair(table)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    means.to_csv(out / "parcel_means.csv", index=False)
    table.to_csv(out / "jm_all.csv", index=False)
    best.to_csv(out / "jm_best_per_pair.csv", index=False)
    return {"polygons": means["class_name"].value_counts().to_dict(), "features": len(keep),
            "mean_best_pairwise_jm": (best.groupby("period")["mean_best_pairwise_jm"].first().to_dict()
                                      if len(best) else {}), "out": str(out)}
