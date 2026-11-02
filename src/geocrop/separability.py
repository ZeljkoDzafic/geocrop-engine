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
    """One row per polygon: id, class and the mean of every feature over *that feature's* valid pixels.

    Validity is judged per band (nodata or non-finite), so a cloudy window never removes pixels from the mean of
    another window; a polygon without any valid pixel of a feature gets NaN for it. Polygons that cover no
    pixel centre of the stack are skipped."""
    import geopandas as gpd
    import rasterio
    from rasterio.mask import mask as rmask

    names = list(features) if features is not None else [f.name for f in featmod.load_manifest(stack)]
    if not names:
        raise ValueError("No features selected for the separability analysis")
    gdf = polygons if isinstance(polygons, gpd.GeoDataFrame) else gpd.read_file(polygons)
    for col in (label_field, id_field):
        if col not in gdf.columns:
            raise ValueError(f"Reference polygons have no field {col!r}")
    gdf = gdf[gdf[label_field].astype(str).isin(classes)]
    rows = []
    with rasterio.open(stack) as src:
        index = {d: i for i, d in enumerate(src.descriptions)}
        missing = [n for n in names if n not in index]
        if missing:
            raise ValueError(f"Features not in the stack: {missing[:5]}")
        bands = [index[n] for n in names]
        gdf = gdf.to_crs(src.crs)
        for _, r in gdf.iterrows():
            try:
                a, _ = rmask(src, [r.geometry], crop=True, filled=False, indexes=[b + 1 for b in bands])
            except ValueError:          # polygon outside the stack
                continue
            vals = np.ma.filled(a.astype("float64"), np.nan).reshape(len(bands), -1)
            if src.nodata is not None:
                vals[vals == src.nodata] = np.nan
            vals[~np.isfinite(vals)] = np.nan
            valid = np.isfinite(vals)
            if not valid.any():
                continue
            counts = valid.sum(axis=1)
            sums = np.where(valid, vals, 0.0).sum(axis=1)
            mean = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)
            rows.append([str(r[id_field]), str(r[label_field]), *mean])
    return pd.DataFrame(rows, columns=["polygon_id", "class_name", *names])


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
    if not keep:
        raise ValueError(f"No period features match groups={groups!r}; groups in the stack with periods: "
                         f"{sorted({f.group for f in manifest if f.period})}")
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
