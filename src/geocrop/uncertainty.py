"""Probability calibration, conformal prediction sets and label-noise screening.

* Temperature scaling (Guo et al., 2017), fitted on out-of-fold probabilities: p_T ∝ p^(1/T).
  This one-parameter fit can still overfit; ECE measured on the fitting labels is descriptive.
* Class-conditional ("Mondrian") conformal score quantiles (Vovk et al., 2005; Angelopoulos & Bates,
  2023). Standard split-conformal guarantees require a fixed predictor and separate exchangeable
  calibration objects. The training workflow instead pools out-of-fold polygon means and fits
  temperature on the same labels: its prediction sets are empirical diagnostics, not guaranteed
  split-conformal coverage for the final refitted model. Evaluate coverage on independent parcels.
* Label-noise screening (confident-learning style; Northcutt et al., 2021): reference polygons whose
  out-of-fold probability for their own label is low are listed for re-interpretation.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar


def apply_temperature(proba: np.ndarray, temperature: float) -> np.ndarray:
    p = np.clip(np.asarray(proba, dtype="float64"), 1e-12, 1.0)
    if temperature == 1.0:
        return p / p.sum(axis=-1, keepdims=True)
    logp = np.log(p) / temperature
    logp -= logp.max(axis=-1, keepdims=True)
    e = np.exp(logp)
    return e / e.sum(axis=-1, keepdims=True)


def fit_temperature(proba: np.ndarray, y, classes) -> float:
    """Temperature minimising the negative log-likelihood of out-of-fold probabilities."""
    classes = list(map(str, classes))
    idx = np.array([classes.index(str(v)) for v in y])

    def nll(log_t):
        p = apply_temperature(proba, math.exp(log_t))
        return -np.mean(np.log(np.clip(p[np.arange(len(idx)), idx], 1e-12, 1)))
    res = minimize_scalar(nll, bounds=(-3, 3), method="bounded")
    return float(math.exp(res.x))


def conformal_thresholds(proba: np.ndarray, y, classes, alphas=(0.05, 0.1, 0.2), mondrian: bool = True,
                         sparse_class_policy: str = "pooled") -> dict:
    """Score quantiles; ``include`` conservatively includes classes with insufficient calibration objects.

    The default pooled fallback preserves the exploratory OOF workflow. Class-conditional split
    calibration must use ``include``: a scarce class gets threshold 1, never an unjustified pooled
    class-conditional threshold. A proper calibration protocol and exchangeability remain necessary.
    """
    if sparse_class_policy not in ("pooled", "include"):
        raise ValueError("sparse_class_policy must be pooled or include")
    alphas = tuple(alphas)
    if any(not np.isfinite(a) or not 0 < a < 1 for a in alphas):
        raise ValueError("Conformal alphas must be finite and between zero and one")
    classes = [str(c) for c in classes]
    y = np.asarray(y).astype(str)
    idx = np.array([classes.index(v) for v in y])
    scores = 1 - np.asarray(proba)[np.arange(len(y)), idx]
    def q(s, a):
        """Conformal quantile, or None when n is too small for level 1-a (needs n >= 1/a - 1)."""
        n = len(s)
        k = math.ceil((n + 1) * (1 - a))
        return None if n == 0 or k > n else float(np.sort(s)[k - 1])

    out, fallback = {}, {}
    for a in alphas:
        pooled = q(scores, a)
        pooled = 1.0 if pooled is None else pooled
        if mondrian:
            per = {c: q(scores[y == c], a) for c in classes}
            # classes with too few calibration objects use the pooled (marginal) threshold
            fallback[str(a)] = sorted(c for c, v in per.items() if v is None)
            out[str(a)] = {c: ((1.0 if sparse_class_policy == "include" else pooled) if v is None else v)
                           for c, v in per.items()}
        else:
            out[str(a)] = {c: pooled for c in classes}
    return {"alphas": list(map(float, alphas)), "mondrian": mondrian, "thresholds": out,
            "pooled_fallback_classes": fallback if sparse_class_policy == "pooled" else {},
            "always_include_classes": fallback if sparse_class_policy == "include" else {},
            "sparse_class_policy": sparse_class_policy,
            "n_calibration": {c: int((y == c).sum()) for c in classes},
            "note": "Coverage guarantees require a fixed predictor and separate exchangeable calibration data; "
                    "OOF calibration in geocrop train does not establish them. Pooled fallback does not "
                    "establish class-conditional coverage."}


def prediction_sets(proba: np.ndarray, classes, thresholds: dict) -> list[list[str]]:
    classes = [str(c) for c in classes]
    q = np.array([thresholds[c] for c in classes])
    member = (1 - np.asarray(proba)) <= q[None, :]
    return [[classes[j] for j in np.flatnonzero(row)] for row in member]


def coverage(sets: list[list[str]], y) -> dict:
    y = [str(v) for v in y]
    hit = np.array([t in s for s, t in zip(sets, y)])
    size = np.array([len(s) for s in sets])
    return {"coverage": float(hit.mean()), "mean_set_size": float(size.mean()),
            "singleton_rate": float((size == 1).mean()), "empty_rate": float((size == 0).mean())}


def validate_polygon_labels(y, polygon) -> None:
    """One physical calibration object must have one label, including after ID string conversion."""
    if len(y) != len(polygon):
        raise ValueError("Each reference label needs one polygon ID")
    pairs = pd.DataFrame({"polygon": np.asarray(polygon).astype(str), "label": np.asarray(y).astype(str)})
    counts = pairs.groupby("polygon")["label"].nunique()
    mixed = counts[counts > 1].index.tolist()
    if mixed:
        raise ValueError(f"Polygon IDs must have one class label; mixed labels for {mixed[:5]}")


def polygon_oof(proba: np.ndarray, y, polygon, classes) -> pd.DataFrame:
    """Polygon-level mean of out-of-fold pixel probabilities (one calibration object per polygon)."""
    validate_polygon_labels(y, polygon)
    frame = pd.DataFrame(np.asarray(proba), columns=[f"p_{c}" for c in classes])
    frame["polygon"] = np.asarray(polygon).astype(str)
    frame["label"] = np.asarray(y).astype(str)
    out = frame.groupby(["polygon", "label"], as_index=False).mean()
    pcols = [f"p_{c}" for c in classes]
    out["predicted"] = [str(classes[i]) for i in out[pcols].to_numpy().argmax(axis=1)]
    out["p_own"] = [row[f"p_{row['label']}"] for _, row in out.iterrows()]
    return out


def suspect_labels(poly: pd.DataFrame, own_max: float = 0.2, other_min: float = 0.6) -> pd.DataFrame:
    """Polygons whose own-label OOF probability is below `own_max` while another class exceeds `other_min`
    (or, failing that, simply the lowest own-label probabilities). Candidates for re-interpretation, not errors."""
    pcols = [c for c in poly.columns if c.startswith("p_") and c != "p_own"]
    best = poly[pcols].max(axis=1)
    flagged = poly[(poly.p_own < own_max) & (best >= other_min) & (poly.predicted != poly.label)]
    return flagged.sort_values("p_own")[["polygon", "label", "predicted", "p_own"]]
