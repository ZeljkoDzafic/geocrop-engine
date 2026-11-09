"""Temporal profiles of selected reference polygons against their own class and a comparison class.

For every time-series feature family (``<family>_d<day of year>``, e.g. ``ts_NDVI_d121``) the parcel-mean profile of
each highlighted polygon is compared with the per-class median and inter-quartile band of the other polygons.
A distance to every class median curve (root mean square, over the days where both exist) states which class the
polygon resembles; it is a descriptive diagnostic for label review, not a classifier and not evidence that a label
is wrong. Highlighted polygons are excluded from the class bands they are compared with.

Writes ``profiles.csv`` (long format), ``profile_distances.csv`` and, with matplotlib installed, one PNG per family.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from . import features as featmod
from . import separability

_DAY = re.compile(r"^(?P<family>.+)_d(?P<doy>\d+)$")


def run(stack, polygons, label_field: str, highlight: list[str], classes: list[str], out: str | Path,
        id_field: str = "polygon_id", families: list[str] | None = None) -> dict:
    manifest = [f.name for f in featmod.load_manifest(stack)]
    parsed = {n: m.groupdict() for n in manifest if (m := _DAY.match(n))}
    if families is not None:
        parsed = {n: g for n, g in parsed.items() if g["family"] in families}
    if not parsed:
        raise ValueError("No time-series features named <family>_d<day> in the stack"
                         + (f" for families {families}" if families else ""))
    means = separability.parcel_means(stack, polygons, label_field, classes, id_field, list(parsed))
    id_col, cls_col = "polygon_id", "class_name"
    missing = sorted(set(map(str, highlight)) - set(means[id_col].astype(str)))
    if missing:
        raise ValueError(f"Highlighted polygons without data in the stack: {missing}")
    means[id_col] = means[id_col].astype(str)
    long = means.melt(id_vars=[id_col, cls_col], value_vars=list(parsed), var_name="feature", value_name="value")
    long["family"] = long["feature"].map(lambda n: parsed[n]["family"])
    long["doy"] = long["feature"].map(lambda n: int(parsed[n]["doy"]))
    long["role"] = np.where(long[id_col].isin(set(map(str, highlight))), "highlight", "reference")
    long = long.rename(columns={id_col: "polygon", cls_col: "class"}).drop(columns="feature")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    long.to_csv(out / "profiles.csv", index=False)
    band = long[long["role"] == "reference"].groupby(["class", "family", "doy"])["value"]
    median = band.median().rename("median")
    rows = []
    for (pid, family), g in long[long["role"] == "highlight"].groupby(["polygon", "family"]):
        curve = g.set_index("doy")["value"]
        own = g["class"].iloc[0]
        for c in classes:
            ref = median.loc[c, family] if (c, family) in median.index.droplevel(2).unique() else None
            if ref is None:
                continue
            joint = curve.dropna().index.intersection(ref.dropna().index)
            if len(joint) == 0:
                continue
            rows.append({"polygon": pid, "own_class": own, "family": family, "compared_with": c,
                         "rmse": float(np.sqrt(np.mean((curve[joint] - ref[joint]) ** 2))), "days": int(len(joint))})
    dist = pd.DataFrame(rows)
    dist.to_csv(out / "profile_distances.csv", index=False)
    summary = {}
    if len(dist):
        for pid, g in dist.groupby("polygon"):
            mean_by_class = g.groupby("compared_with")["rmse"].mean()
            summary[pid] = {"own_class": g["own_class"].iloc[0], "closest_class_by_mean_rmse": mean_by_class.idxmin(),
                            "mean_rmse": {k: float(v) for k, v in mean_by_class.items()}}
    plotted = _plot(long, median, classes, out)
    return {"highlighted": sorted(map(str, highlight)), "families": sorted({g["family"] for g in parsed.values()}),
            "summary": summary, "figures": plotted, "out": str(out)}


def _plot(long: pd.DataFrame, median: pd.Series, classes: list[str], out: Path) -> list[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    figures = []
    palette = dict(zip(classes, ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#e6ab02"]))
    for family, fam in long.groupby("family"):
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ref = fam[fam["role"] == "reference"]
        for c in classes:
            d = ref[ref["class"] == c]
            if d.empty:
                continue
            q = d.groupby("doy")["value"].quantile([0.25, 0.5, 0.75]).unstack()
            ax.fill_between(q.index, q[0.25], q[0.75], color=palette.get(c), alpha=0.18, linewidth=0)
            ax.plot(q.index, q[0.5], color=palette.get(c), lw=2, label=f"{c} median, IQR")
        for pid, g in fam[fam["role"] == "highlight"].groupby("polygon"):
            g = g.sort_values("doy")
            ax.plot(g["doy"], g["value"], color="black", lw=1.2, ls="--", marker=".", label=f"{pid} ({g['class'].iloc[0]})")
        ax.set_xlabel("day of year")
        ax.set_ylabel(family)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, ncol=2)
        path = out / f"profile_{family}.png"
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        figures.append(str(path))
    return figures
