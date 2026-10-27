"""Accuracy assessment following good-practice remote-sensing protocols.

* `summary`           OA, Cohen's kappa, macro/weighted F1, per-class user's/producer's accuracy and F1
* `bootstrap`         percentile confidence intervals of OA and macro-F1 (resampling reference units)
* `area_adjusted`     stratified area and accuracy estimators with standard errors (Olofsson et al., 2014)
* `calibration`       expected calibration error and reliability-curve bins for class probabilities

Convention: confusion matrices have rows = reference (true) class, columns = map (predicted) class,
matching scikit-learn. `area_adjusted` transposes internally to the map-stratum convention.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix


def summary(y_true, y_pred, labels=None) -> dict:
    y_true = np.asarray(y_true).astype(str)
    y_pred = np.asarray(y_pred).astype(str)
    labels = list(labels) if labels is not None else sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return from_confusion(cm, labels)


def from_confusion(cm, labels) -> dict:
    cm = np.asarray(cm, dtype=float)
    n = cm.sum()
    if n == 0:
        raise ValueError("Empty confusion matrix")
    diag = np.diag(cm)
    ref_tot, map_tot = cm.sum(axis=1), cm.sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        producers = np.where(ref_tot > 0, diag / ref_tot, np.nan)   # recall
        users = np.where(map_tot > 0, diag / map_tot, np.nan)       # precision
        f1 = np.where(producers + users > 0, 2 * producers * users / (producers + users), 0.0)
    oa = diag.sum() / n
    pe = (ref_tot * map_tot).sum() / n**2
    kappa = (oa - pe) / (1 - pe) if pe < 1 else 1.0
    present = ref_tot > 0
    per_class = {
        str(lab): {"producers_accuracy": _f(producers[i]), "users_accuracy": _f(users[i]), "f1": _f(f1[i]),
                   "reference_count": int(ref_tot[i]), "map_count": int(map_tot[i])}
        for i, lab in enumerate(labels)}
    return {
        "n": int(n), "overall_accuracy": float(oa), "kappa": float(kappa),
        "macro_f1": float(np.mean(f1[present])) if present.any() else float("nan"),
        "weighted_f1": float(np.sum(f1 * ref_tot) / ref_tot.sum()),
        "labels": [str(x) for x in labels], "confusion_matrix": cm.astype(int).tolist(),
        "per_class": per_class,
    }


def _f(x):
    return None if x is None or not np.isfinite(x) else float(x)


def bootstrap(y_true, y_pred, groups=None, n_boot: int = 1000, seed: int = 42, alpha: float = 0.05) -> dict:
    """Percentile CIs for OA and macro-F1. Resamples `groups` (e.g. parcels) as units when given,
    which avoids overconfident intervals from spatially autocorrelated pixels."""
    y_true = np.asarray(y_true).astype(str)
    y_pred = np.asarray(y_pred).astype(str)
    labels = sorted(set(y_true) | set(y_pred))
    rng = np.random.default_rng(seed)
    if groups is None:
        units = np.arange(len(y_true))
        members = {u: np.array([u]) for u in units}
    else:
        groups = np.asarray(groups)
        units = np.unique(groups)
        members = {u: np.flatnonzero(groups == u) for u in units}
    oa, mf1 = [], []
    for _ in range(n_boot):
        pick = rng.choice(units, size=len(units), replace=True)
        idx = np.concatenate([members[u] for u in pick])
        s = from_confusion(confusion_matrix(y_true[idx], y_pred[idx], labels=labels), labels)
        oa.append(s["overall_accuracy"])
        mf1.append(s["macro_f1"])
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return {"n_boot": n_boot, "units": int(len(units)), "confidence": 1 - alpha,
            "overall_accuracy_ci": [float(np.percentile(oa, lo)), float(np.percentile(oa, hi))],
            "macro_f1_ci": [float(np.nanpercentile(mf1, lo)), float(np.nanpercentile(mf1, hi))]}


def area_adjusted(cm, labels, mapped_area: dict) -> dict:
    """Olofsson et al. (2014) stratified estimators.

    cm: rows = reference, columns = map (scikit-learn convention), counts of a sample that was
        stratified by MAP class; mapped_area: {class: mapped area (any unit)} for every label.
    Returns error-adjusted area per class, UA/PA/OA and their standard errors.
    """
    n = np.asarray(cm, dtype=float).T          # rows = map stratum i, cols = reference j
    labels = [str(x) for x in labels]
    if any(lab not in mapped_area for lab in labels):
        raise ValueError("mapped_area must contain every label")
    area = np.array([float(mapped_area[lab]) for lab in labels])
    total = area.sum()
    w = area / total
    n_i = n.sum(axis=1)
    if (n_i[w > 0] < 2).any():
        raise ValueError("Each mapped stratum needs at least two sample units")
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = np.where(n_i[:, None] > 0, n / n_i[:, None], 0.0)
    p = w[:, None] * frac                       # estimated cell proportions
    p_ref = p.sum(axis=0)                       # true-class proportions
    p_map = p.sum(axis=1)
    oa = np.trace(p)
    denom = np.where(n_i > 1, n_i - 1, np.inf)
    se_p_ref = np.sqrt(np.sum((w[:, None] ** 2) * frac * (1 - frac) / denom[:, None], axis=0))
    ua = np.divide(np.diag(p), p_map, out=np.full(len(labels), np.nan), where=p_map > 0)
    pa = np.divide(np.diag(p), p_ref, out=np.full(len(labels), np.nan), where=p_ref > 0)
    var_oa = np.sum(w**2 * np.diag(frac) * (1 - np.diag(frac)) / denom)
    var_ua = np.diag(frac) * (1 - np.diag(frac)) / denom
    # Producer's accuracy variance, Olofsson et al. (2014) eq. 7.
    var_pa = np.full(len(labels), np.nan)
    n_hat_j = (area[:, None] * frac).sum(axis=0)  # estimated total per reference class
    for j in range(len(labels)):
        if n_hat_j[j] <= 0:
            continue
        others = sum(area[i] ** 2 * frac[i, j] * (1 - frac[i, j]) / denom[i] for i in range(len(labels)) if i != j)
        first = area[j] ** 2 * (1 - pa[j]) ** 2 * frac[j, j] * (1 - frac[j, j]) / denom[j]
        var_pa[j] = (first + pa[j] ** 2 * others) / n_hat_j[j] ** 2 if np.isfinite(pa[j]) else np.nan
    per_class = {lab: {"mapped_area": float(area[k]), "adjusted_area": float(p_ref[k] * total),
                       "adjusted_area_se": float(se_p_ref[k] * total),
                       "adjusted_area_ci95": [float((p_ref[k] - 1.96 * se_p_ref[k]) * total),
                                              float((p_ref[k] + 1.96 * se_p_ref[k]) * total)],
                       "users_accuracy": _f(ua[k]), "users_accuracy_se": _f(np.sqrt(var_ua[k])),
                       "producers_accuracy": _f(pa[k]), "producers_accuracy_se": _f(np.sqrt(var_pa[k]))}
                 for k, lab in enumerate(labels)}
    return {"overall_accuracy": float(oa), "overall_accuracy_se": float(np.sqrt(var_oa)),
            "total_area": float(total), "per_class": per_class, "method": "Olofsson et al. (2014)"}


def calibration(probabilities, y_true, classes, bins: int = 10) -> dict:
    """Top-label expected calibration error (ECE) and reliability bins.

    probabilities: (n, k) array; y_true: n labels; classes: column order of probabilities.
    """
    probs = np.asarray(probabilities, dtype=float)
    classes = np.asarray([str(c) for c in classes])
    y_true = np.asarray(y_true).astype(str)
    confidence = probs.max(axis=1)
    correct = classes[probs.argmax(axis=1)] == y_true
    edges = np.linspace(0, 1, bins + 1)
    rows, ece = [], 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (confidence > lo) & (confidence <= hi) if lo > 0 else (confidence >= lo) & (confidence <= hi)
        if not sel.any():
            continue
        acc, conf = float(correct[sel].mean()), float(confidence[sel].mean())
        ece += sel.mean() * abs(acc - conf)
        rows.append({"bin": [float(lo), float(hi)], "count": int(sel.sum()), "accuracy": acc, "confidence": conf})
    return {"ece": float(ece), "bins": rows, "n": int(len(y_true))}


def reject_curve(confidence, correct, thresholds=None) -> list[dict]:
    """Accuracy vs. coverage when predictions below a confidence threshold are sent to manual review."""
    confidence = np.asarray(confidence, dtype=float)
    correct = np.asarray(correct, dtype=bool)
    thresholds = np.linspace(0, 0.95, 20) if thresholds is None else thresholds
    rows = []
    for t in thresholds:
        keep = confidence >= t
        rows.append({"threshold": float(t), "coverage": float(keep.mean()),
                     "accuracy": float(correct[keep].mean()) if keep.any() else None,
                     "review_count": int((~keep).sum())})
    return rows


def mcnemar(y_true, pred_a, pred_b) -> dict:
    """Paired comparison of two classifiers on the same samples (McNemar, 1947).

    Exact binomial test when discordant pairs < 25, otherwise chi-square with continuity correction.
    Pixels from one parcel are not independent; prefer parcel- or point-level predictions.
    """
    from scipy.stats import binomtest, chi2

    y = np.asarray(y_true).astype(str)
    a = np.asarray(pred_a).astype(str) == y
    b = np.asarray(pred_b).astype(str) == y
    n01, n10 = int((a & ~b).sum()), int((~a & b).sum())
    if n01 + n10 == 0:
        return {"a_only_correct": n01, "b_only_correct": n10, "statistic": 0.0, "p_value": 1.0, "test": "none"}
    if n01 + n10 < 25:
        p = binomtest(n01, n01 + n10, 0.5).pvalue
        return {"a_only_correct": n01, "b_only_correct": n10, "statistic": float(min(n01, n10)),
                "p_value": float(p), "test": "exact binomial"}
    stat = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
    return {"a_only_correct": n01, "b_only_correct": n10, "statistic": float(stat),
            "p_value": float(chi2.sf(stat, 1)), "test": "chi-square (continuity corrected)"}


def grouped_mcnemar(y_true, pred_a, pred_b, groups) -> dict:
    """One majority-vote comparison per reference polygon, with deterministic lexical ties.

    Repeated pixels in a polygon are not independent trials. Distinct polygons can still be spatially
    dependent; grouping alone does not establish independence or make internal CV an external assessment.
    """
    y, a, b, g = (np.asarray(v).astype(str) for v in (y_true, pred_a, pred_b, groups))
    if y.ndim != 1 or not (y.shape == a.shape == b.shape == g.shape) or not len(y):
        raise ValueError("Grouped McNemar needs equally sized, non-empty one-dimensional arrays")

    def majority(values):
        labels, counts = np.unique(values, return_counts=True)
        return labels[counts.argmax()]

    truth, vote_a, vote_b = [], [], []
    order = np.argsort(g, kind="stable")
    sorted_groups = g[order]
    boundaries = np.r_[0, np.flatnonzero(sorted_groups[1:] != sorted_groups[:-1]) + 1, len(g)]
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        members = order[start:end]  # group once; never scan all pixels once per polygon
        labels = np.unique(y[members])
        if len(labels) != 1:
            raise ValueError(f"Reference group {sorted_groups[start]!r} has conflicting labels")
        truth.append(labels[0])
        vote_a.append(majority(a[members]))
        vote_b.append(majority(b[members]))
    return {**mcnemar(truth, vote_a, vote_b), "n_units": len(truth), "unit": "reference_polygon",
            "aggregation": "majority_vote", "tie_rule": "lexicographic"}


def holm_adjust(p_values) -> list[float]:
    """Holm step-down family-wise adjustment, returned in the original comparison order."""
    p = np.asarray(p_values, dtype=float)
    if p.ndim != 1 or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("P-values must be a one-dimensional sequence of finite numbers in [0, 1]")
    order = np.argsort(p, kind="stable")
    adjusted = np.empty_like(p)
    adjusted[order] = np.minimum(1, np.maximum.accumulate(p[order] * np.arange(len(p), 0, -1)))
    return adjusted.tolist()
