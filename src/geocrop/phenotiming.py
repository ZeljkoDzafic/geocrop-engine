"""Timing of the seasonal curve (start, peak, end, length) and the effect of time alignment, from ``profiles.csv``.

For every polygon and time-series family the curve is interpolated to a daily grid. With the season minimum as the
base and the maximum as the peak, ``sos`` / ``eos`` are the first rising and last falling day at which the curve
crosses ``base + fraction * (peak - base)`` (default 0.5), ``gsl`` = ``eos - sos`` and ``peak_doy`` the day of the
maximum. Highlighted polygons are placed against the distribution of the reference polygons of every class (offset in
days from the class median and rank within the class).

Time alignment answers a narrow question: if a highlighted polygon's curve is moved in time by the shift that
minimises its RMSE to a class median curve, how close does it get, and how large is the shift? A polygon that matches
a class much better after a shift of a few weeks has a development shifted in time (sowing date, variety); one that
does not has a different curve shape. Both are descriptive diagnostics for review, not a classifier and not evidence
that a label is wrong. Curves are as dense as the time series (here: composite steps), so timing has that resolution.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _daily(curve: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    c = curve.dropna().sort_index()
    if len(c) < 3:
        return np.array([]), np.array([])
    days = np.arange(int(c.index.min()), int(c.index.max()) + 1)
    return days, np.interp(days, c.index.to_numpy(float), c.to_numpy(float))


def timing(curve: pd.Series, fraction: float = 0.5) -> dict | None:
    days, v = _daily(curve)
    if len(days) == 0:
        return None
    base, peak = float(v.min()), float(v.max())
    level = base + fraction * (peak - base)
    ip = int(np.argmax(v))
    above = v >= level
    if peak - base <= 0 or not above.any():
        return None
    sos = float(days[int(np.argmax(above))])
    eos = float(days[len(v) - 1 - int(np.argmax(above[::-1]))])
    return {"sos": sos, "peak_doy": float(days[ip]), "eos": eos, "gsl": eos - sos, "peak": peak, "base": base}


def _rmse_shift(curve: pd.Series, ref: pd.Series, shifts) -> tuple[float, int, float]:
    """RMSE without shift, best shift (days, positive = later) and the RMSE at that shift."""
    days, v = _daily(curve)
    if len(days) == 0:
        return np.nan, 0, np.nan
    r = ref.dropna()
    grid = r.index.to_numpy(float)
    rv = r.to_numpy(float)

    def err(s):
        x = np.interp(grid - s, days, v, left=np.nan, right=np.nan)
        ok = ~np.isnan(x)
        return float(np.sqrt(np.mean((x[ok] - rv[ok]) ** 2))) if ok.sum() >= 3 else np.nan

    scores = {int(s): err(s) for s in shifts}
    valid = {s: e for s, e in scores.items() if not np.isnan(e)}
    best = min(valid, key=valid.get) if valid else 0
    return scores.get(0, np.nan), best, valid.get(best, np.nan)


def run(profiles_csv: str | Path, classes: list[str], out: str | Path, fraction: float = 0.5,
        max_shift_days: int = 45) -> dict:
    long = pd.read_csv(profiles_csv)
    need = {"polygon", "class", "value", "family", "doy", "role"}
    if not need <= set(long.columns):
        raise ValueError(f"profiles file needs columns {sorted(need)}")
    if not 0 < fraction < 1:
        raise ValueError("fraction must be in (0, 1)")
    rows = []
    for (pid, family), g in long.groupby(["polygon", "family"]):
        t = timing(g.set_index("doy")["value"], fraction)
        if t:
            rows.append({"polygon": pid, "class": g["class"].iloc[0], "role": g["role"].iloc[0], "family": family, **t})
    per_polygon = pd.DataFrame(rows)
    ref = per_polygon[per_polygon["role"] == "reference"]
    metrics = ["sos", "peak_doy", "eos", "gsl", "peak"]
    class_stats = ref.groupby(["family", "class"])[metrics].agg(["median", "min", "max", "count"]).round(1)
    med = long[long["role"] == "reference"].groupby(["class", "family", "doy"])["value"].median()
    rel = []
    shifts = range(-max_shift_days, max_shift_days + 1)
    for _, h in per_polygon[per_polygon["role"] == "highlight"].iterrows():
        fam = h["family"]
        own = ref[(ref["family"] == fam) & (ref["class"] == h["class"])]
        row = {"polygon": h["polygon"], "class": h["class"], "family": fam}
        for m in metrics:
            for c in classes:
                sel = ref[(ref["family"] == fam) & (ref["class"] == c)][m]
                if len(sel):
                    row[f"{m}_minus_{c}_median"] = float(h[m] - sel.median())
            if len(own):
                row[f"{m}_percentile_in_own_class"] = float((own[m] < h[m]).mean() * 100)
        curve = long[(long["polygon"] == h["polygon"]) & (long["family"] == fam)].set_index("doy")["value"]
        for c in classes:
            if (c, fam) in med.index.droplevel(2).unique():
                e0, s, e1 = _rmse_shift(curve, med.loc[c, fam], shifts)
                row[f"rmse_{c}"] = e0
                row[f"best_shift_days_{c}"] = s
                row[f"rmse_aligned_{c}"] = e1
        rel.append(row)
    relative = pd.DataFrame(rel)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    per_polygon.to_csv(out / "phenology_timing.csv", index=False)
    class_stats.to_csv(out / "phenology_class_stats.csv")
    relative.to_csv(out / "phenology_highlight_vs_classes.csv", index=False)
    return {"polygons": int(per_polygon["polygon"].nunique()), "families": sorted(per_polygon["family"].unique()),
            "fraction": fraction, "max_shift_days": max_shift_days, "out": str(out),
            "note": "Descriptive timing from interpolated composite curves; resolution equals the composite step."}
