"""Domain-shift detection at prediction time and shift-aware conformal prediction sets.

A classifier trained and calibrated in one region/year can predict well in another one while its
uncertainty statements no longer hold: split-conformal coverage assumes that calibration and target
objects are exchangeable, which a new area or season breaks. This module compares the prediction domain
with the training/calibration domain and says so explicitly (``shift_report.json``).

Signals (each graded 0 = none, 1 = mild, 2 = severe; the verdict is the maximum):

* ``aoa``          share of target pixels outside the area of applicability (Meyer & Pebesma, 2021), in
                   excess of the share of training samples outside it under spatial cross-validation;
* ``dissimilarity`` median dissimilarity index of the target sample relative to the median cross-validated
                   training DI;
* ``domain_classifier`` cross-validated ROC AUC of a classifier separating reference (training/calibration)
                   pixels from target pixels (classifier two-sample test; Rabanser et al., 2019). Folds are
                   grouped by polygon/parcel, and the reference is re-weighted to the target's *predicted*
                   class mix, so that a different crop mix alone (label shift) does not count as covariate
                   shift within classes. The unadjusted AUC is reported too;
* ``feature_smd``  importance-weighted mean absolute standardised mean difference of the features
                   (Austin, 2009), with the same class-mix adjustment;
* ``class_prior``  total-variation distance between predicted class shares of target and reference
                   (label shift; Lipton et al., 2018). Mondrian (class-conditional) sets are robust to pure
                   label shift, so this signal is at most *mild*;
* ``metadata``     prediction year/region not among the calibration years/regions (mild: exchangeability
                   with the calibration objects is then not plausible, whatever the features say).

Any verdict other than ``in-domain`` marks conformal coverage as **not established**. Optional adjusted
thresholds (``shift_conformal``):

* ``conservative``  per class the larger of the class-conditional and the pooled quantile at level
                    1 - alpha/2. A heuristic safety margin; no coverage guarantee under shift.
* ``weighted``      covariate-shift weighted split conformal (Tibshirani et al., 2019) per class: calibration
                    scores weighted by estimated likelihood ratios w(x) = p/(1-p) of the class-mix adjusted
                    domain classifier; the test-point weight is a high quantile (default 0.9) of the target
                    objects' weights, which is more conservative than the per-object rule for most objects.
                    Valid only if P(y|x) is unchanged (pure covariate shift) and w is estimated well -
                    neither is verified, so coverage stays "not established".

Pure local computation (numpy, scikit-learn); no network.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from . import uncertainty

SCHEMA = "geocrop-shift-report/1"
REFERENCE_FILE = "domain_reference.npz"
VERDICTS = ("in-domain", "mild", "severe")
POLICIES = ("warn", "review", "abstain")
CONFORMAL_MODES = ("none", "conservative", "weighted")
REASON_CODE = "DOMAIN_SHIFT"

#: (mild, severe) limits per signal; None = the signal cannot reach that level. Fixed a priori (documented in
#: docs/domain-shift.md), not tuned on the transfer benchmark.
THRESHOLDS: dict[str, tuple[float | None, float | None]] = {
    "aoa_excess_outside": (0.05, 0.20),
    "di_median_ratio": (1.25, 2.0),
    "domain_auc": (0.70, 0.85),
    "feature_smd": (0.20, 0.50),
    "class_prior_tvd": (0.15, None),
}

DEFAULTS = {
    "shift_detection": True,     # write shift_report.json at prediction (needs model/domain_reference.npz)
    "shift_policy": "warn",      # warn | review | abstain
    "shift_conformal": "none",   # none | conservative | weighted
    "shift_sample_pixels": 5000,
    "shift_test_weight_quantile": 0.9,
    "region": None,              # region id of the prediction domain (compared with the training regions)
    "year": None,                # year of the prediction domain (default: cv.year or timeseries.start year)
    "shift_thresholds": {},      # overrides of THRESHOLDS, e.g. {domain_auc: [0.75, 0.9]}
}

NOT_ESTABLISHED = "not established (domain shift)"


def settings(raw: dict | None) -> dict:
    """Validated settings (the ``uncertainty`` config section); raises ValueError."""
    s = {**DEFAULTS, **(raw or {})}
    unknown = set(s) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown uncertainty keys {sorted(unknown)}")
    if not isinstance(s["shift_detection"], bool):
        raise ValueError("uncertainty.shift_detection must be true or false")
    if s["shift_policy"] not in POLICIES:
        raise ValueError(f"uncertainty.shift_policy must be one of {', '.join(POLICIES)}")
    if s["shift_conformal"] not in CONFORMAL_MODES:
        raise ValueError(f"uncertainty.shift_conformal must be one of {', '.join(CONFORMAL_MODES)}")
    n = s["shift_sample_pixels"]
    if isinstance(n, bool) or not isinstance(n, int) or n < 200:
        raise ValueError("uncertainty.shift_sample_pixels must be an integer >= 200")
    q = s["shift_test_weight_quantile"]
    if isinstance(q, bool) or not isinstance(q, (int, float)) or not 0.5 <= q <= 1:
        raise ValueError("uncertainty.shift_test_weight_quantile must be in [0.5, 1]")
    if not isinstance(s["shift_thresholds"], dict) or set(s["shift_thresholds"]) - set(THRESHOLDS):
        raise ValueError(f"uncertainty.shift_thresholds keys must be among {sorted(THRESHOLDS)}")
    for k, v in s["shift_thresholds"].items():
        if not (isinstance(v, (list, tuple)) and len(v) == 2
                and all(x is None or (isinstance(x, (int, float)) and not isinstance(x, bool)) for x in v)):
            raise ValueError(f"uncertainty.shift_thresholds.{k} must be [mild, severe] (numbers or null)")
    for k in ("region", "year"):
        if s[k] is not None and not isinstance(s[k], (str, int)):
            raise ValueError(f"uncertainty.{k} must be a string, an integer or null")
    return s


# --------------------------------------------------------------------------- reference (training time)

def build_reference(X, polygon, features, classes, cal_polygon, cal_proba, cal_label, *, train_di=None,
                    aoa_threshold=None, importance=None, region=None, year=None, max_pixels: int = 4000,
                    per_polygon: int = 10, seed: int = 42) -> dict:
    """Reference of the training/calibration domain.

    X, polygon       labelled pixels of the training domain (fit and calibration objects) and their polygon id
    cal_*            calibration objects (polygon id, class-probability rows, label) used for conformal sets
    train_di         cross-validated training dissimilarity indices (``Applicability.train_di``)
    region, year     per-pixel region/year ids of `X` (optional metadata)
    Pixels of calibration objects come first (at most `per_polygon` each) so that every calibration object
    can receive a likelihood-ratio weight; the rest of `max_pixels` is filled with other training pixels.
    """
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype="float32")
    polygon = np.asarray(polygon).astype(str)
    cal_polygon = np.asarray(cal_polygon).astype(str)
    is_cal_px = np.isin(polygon, cal_polygon)
    chosen = []
    cal_idx = np.flatnonzero(is_cal_px)
    if len(cal_idx):
        order = rng.permutation(cal_idx)
        seen: dict[str, int] = {}
        for i in order:
            p = polygon[i]
            if seen.get(p, 0) < per_polygon:
                seen[p] = seen.get(p, 0) + 1
                chosen.append(i)
    chosen = np.asarray(chosen, dtype=int)
    if len(chosen) > max_pixels:
        chosen = rng.choice(chosen, max_pixels, replace=False)
    rest = np.setdiff1d(np.arange(len(X)), chosen)
    room = max(0, max_pixels - len(chosen))
    if room and len(rest):
        chosen = np.concatenate([chosen, rng.choice(rest, min(room, len(rest)), replace=False)])
    chosen = np.sort(chosen)
    meta = {"features": list(map(str, features)), "classes": list(map(str, classes)),
            "aoa_threshold": None if aoa_threshold is None else float(aoa_threshold),
            "regions": _counts(region), "years": _counts(year),
            "calibration_regions": _counts(None if region is None else np.asarray(region)[is_cal_px]),
            "calibration_years": _counts(None if year is None else np.asarray(year)[is_cal_px]),
            "n_training_pixels": int(len(X)), "n_calibration_objects": int(len(cal_polygon))}
    di = np.asarray(train_di, dtype="float64") if train_di is not None else np.empty(0)
    if len(di) > 5000:
        di = rng.choice(di, 5000, replace=False)
    imp = np.ones(X.shape[1]) if importance is None else np.nan_to_num(np.asarray(importance, dtype="float64"))
    return {"X": X[chosen], "polygon": polygon[chosen], "is_calibration": is_cal_px[chosen],
            "cal_polygon": cal_polygon, "cal_proba": np.asarray(cal_proba, dtype="float64"),
            "cal_label": np.asarray(cal_label).astype(str), "train_di": di, "importance": imp, "meta": meta}


def _counts(values) -> dict:
    if values is None:
        return {}
    v = np.asarray(values).astype(str)
    v = v[(v != "") & (v != "None") & (v != "nan")]
    u, c = np.unique(v, return_counts=True)
    return {str(a): int(b) for a, b in zip(u, c)}


def save_reference(ref: dict, path: str | Path) -> Path:
    path = Path(path)
    np.savez_compressed(path, **{k: v for k, v in ref.items() if k != "meta"},
                        meta=np.array(json.dumps(ref["meta"])))
    return path


def load_reference(path: str | Path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        ref = {k: z[k] for k in z.files if k != "meta"}
        ref["meta"] = json.loads(str(z["meta"]))
    return ref


# --------------------------------------------------------------------------- conformal helpers

def weighted_threshold(scores, weights, test_weight: float, alpha: float) -> float:
    """Weighted split-conformal quantile (Tibshirani et al., 2019): the (1-alpha) quantile of
    sum_i p_i delta_{s_i} + p_{n+1} delta_{+inf}, p_i = w_i / (sum_j w_j + w_test). Returns 1.0 (always
    include) when the quantile is +inf. With equal weights it equals the usual ceil((n+1)(1-alpha))-th score."""
    s = np.asarray(scores, dtype="float64")
    w = np.asarray(weights, dtype="float64")
    if len(s) == 0:
        return 1.0
    order = np.argsort(s, kind="stable")
    s, w = s[order], w[order]
    cum = np.cumsum(w) / (w.sum() + float(test_weight))
    k = int(np.searchsorted(cum, 1 - alpha - 1e-12, side="left"))
    return 1.0 if k >= len(s) else float(s[k])


def scores_of(proba, labels, classes) -> np.ndarray:
    classes = [str(c) for c in classes]
    idx = np.array([classes.index(str(v)) for v in labels])
    return 1 - np.asarray(proba)[np.arange(len(idx)), idx]


def conservative_thresholds(proba, labels, classes, alpha: float) -> dict:
    """max(class-conditional, pooled) split-conformal quantile at level 1 - alpha/2 (sparse classes: 1)."""
    classes = [str(c) for c in classes]
    a = alpha / 2
    per = uncertainty.conformal_thresholds(proba, labels, classes, (a,), mondrian=True,
                                           sparse_class_policy="include")["thresholds"][str(a)]
    pooled = uncertainty.conformal_thresholds(proba, labels, classes, (a,), mondrian=False)["thresholds"][str(a)]
    return {c: float(max(per[c], pooled[c])) for c in classes}


def weighted_thresholds(proba, labels, classes, alpha: float, weights, test_weight: float) -> dict:
    """Class-conditional (Mondrian) weighted split-conformal thresholds."""
    classes = [str(c) for c in classes]
    labels = np.asarray(labels).astype(str)
    s = scores_of(proba, labels, classes)
    w = np.asarray(weights, dtype="float64")
    return {c: weighted_threshold(s[labels == c], w[labels == c], test_weight, alpha) for c in classes}


# --------------------------------------------------------------------------- detection

def _level(value, limits) -> int:
    if value is None or not np.isfinite(value):
        return 0
    mild, severe = limits
    if severe is not None and value >= severe:
        return 2
    if mild is not None and value >= mild:
        return 1
    return 0


def _shares(pred, classes) -> np.ndarray:
    idx = {c: i for i, c in enumerate(classes)}
    counts = np.bincount([idx[p] for p in pred], minlength=len(classes)).astype("float64")
    return counts / max(counts.sum(), 1)


def _smd(Xr, wr, Xt) -> np.ndarray:
    wr = wr / wr.sum()
    mr = np.nansum(Xr * wr[:, None], axis=0)
    vr = np.nansum(wr[:, None] * (Xr - mr) ** 2, axis=0)
    mt, vt = np.nanmean(Xt, axis=0), np.nanvar(Xt, axis=0)
    den = np.sqrt((vr + vt) / 2)
    with np.errstate(invalid="ignore", divide="ignore"):
        d = np.abs(mt - mr) / den
    return np.where(den > 0, np.nan_to_num(d), 0.0)


def _weighted_bins(X: np.ndarray, w: np.ndarray, n_bins: int = 255) -> np.ndarray:
    """Replace every feature by its weighted-quantile bin index (NaN stays NaN).

    HistGradientBoosting bins each feature at every ``fit``; with sample weights recent scikit-learn
    versions compute 254 weighted percentiles per feature in Python, which dominated the shift check
    (seconds per fit). Features with at most 255 distinct values are split on their midpoints, so
    binning once here with the same weighted quantiles gives the same kind of model at a fraction
    of the cost, and the bin edges use only unlabelled feature values.
    """
    out = np.full(X.shape, np.nan)
    levels = np.arange(1, n_bins) / n_bins
    for j in range(X.shape[1]):
        col = X[:, j]
        ok = ~np.isnan(col)
        if not ok.any():
            continue
        v, wj = col[ok], w[ok]
        order = np.argsort(v, kind="stable")
        cw = np.cumsum(wj[order])
        edges = np.unique(v[order][np.minimum(np.searchsorted(cw / cw[-1], levels), len(v) - 1)])
        out[ok, j] = np.searchsorted(edges, v, side="left")
    return out


def _domain_classifier(Xr, wr, gr, Xt, gt, seed: int, folds: int = 5) -> tuple[float, np.ndarray]:
    """Grouped out-of-fold P(target | x) for all rows of [Xr; Xt] and the weighted ROC AUC."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold

    X = np.vstack([Xr, Xt]).astype("float64")
    y = np.r_[np.zeros(len(Xr)), np.ones(len(Xt))]
    w = np.r_[wr / wr.sum() * len(Xt), np.ones(len(Xt))]      # both sides carry equal total weight
    groups = np.r_[np.char.add("r:", np.asarray(gr).astype(str)), np.char.add("t:", np.asarray(gt).astype(str))]
    k = int(min(folds, len(np.unique(groups))))
    p = np.full(len(y), 0.5)
    if k < 2:
        return float("nan"), p
    X = _weighted_bins(X, w)
    for tr, te in GroupKFold(n_splits=k).split(X, y, groups):
        if len(np.unique(y[tr])) < 2:
            continue
        clf = HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=15, learning_rate=0.1,
                                             l2_regularization=1.0, random_state=seed)
        clf.fit(X[tr], y[tr], sample_weight=w[tr])
        p[te] = clf.predict_proba(X[te])[:, 1]
    return float(roc_auc_score(y, p, sample_weight=w)), p


def detect(reference: dict, X_target, groups_target, predict_proba, *, aoa=None, conformal: dict | None = None,
           metadata: dict | None = None, config: dict | None = None, seed: int = 42,
           sample_note: str = "parcel pixels") -> dict:
    """Compare a target pixel sample with the reference; return the shift report (JSON-ready dict).

    predict_proba  callable X -> class probabilities (columns ordered as ``reference['meta']['classes']``)
    aoa            `applicability.Applicability` of the model (optional)
    conformal      ``conformal.json`` content (thresholds per alpha) of the model (optional)
    metadata       ``{"region": ..., "year": ...}`` of the prediction domain
    """
    cfg = settings(config)
    limits = {**THRESHOLDS, **{k: tuple(v) for k, v in cfg["shift_thresholds"].items()}}
    rng = np.random.default_rng(seed)
    meta = reference["meta"]
    classes = meta["classes"]
    Xt = np.asarray(X_target, dtype="float64")
    gt = np.asarray(groups_target).astype(str)
    if len(Xt) < 20:
        return unavailable(f"too few target pixels ({len(Xt)}) for a shift check", cfg)
    n = int(cfg["shift_sample_pixels"])
    if len(Xt) > n:
        keep = rng.choice(len(Xt), n, replace=False)
        Xt, gt = Xt[keep], gt[keep]
    Xr = np.asarray(reference["X"], dtype="float64")
    gr = np.asarray(reference["polygon"]).astype(str)
    signals: dict = {}
    reasons: list[dict] = []

    def grade(name, value, key, message, extra=None):
        lvl = _level(value, limits[key]) if key else int(value is not None and value > 0)
        signals[name] = {"value": None if value is None or not np.isfinite(value) else float(value),
                         "level": lvl, "limits": list(limits[key]) if key else None, **(extra or {})}
        if lvl:
            reasons.append({"signal": name, "level": VERDICTS[lvl], "message": message})
        return lvl

    # class priors (predicted)
    pr = np.asarray(predict_proba(Xr))
    pt = np.asarray(predict_proba(Xt))
    pred_r = np.array(classes)[pr.argmax(axis=1)]
    pred_t = np.array(classes)[pt.argmax(axis=1)]
    sr, st = _shares(pred_r, classes), _shares(pred_t, classes)
    tvd = float(0.5 * np.abs(sr - st).sum())
    grade("class_prior", tvd, "class_prior_tvd",
          f"predicted class mix differs from the reference (total variation {tvd:.2f})",
          {"reference_shares": dict(zip(classes, sr.round(4).tolist())),
           "target_shares": dict(zip(classes, st.round(4).tolist()))})
    # class-mix adjustment of the reference: weight r_k = target share / reference share of the predicted class
    ratio = np.divide(st, sr, out=np.zeros_like(st), where=sr > 0)
    wr = ratio[[classes.index(c) for c in pred_r]]
    if wr.sum() <= 0:
        wr = np.ones(len(Xr))

    # feature distributions
    imp = np.asarray(reference.get("importance", np.ones(Xr.shape[1])), dtype="float64")
    imp = imp / imp.sum() if imp.sum() > 0 else np.full(len(imp), 1 / len(imp))
    smd = _smd(Xr, wr, Xt)
    smd_raw = _smd(Xr, np.ones(len(Xr)), Xt)
    top = np.argsort(smd)[::-1][:5]
    grade("feature_smd", float((imp * smd).sum()), "feature_smd",
          f"feature means differ (importance-weighted mean |SMD| {float((imp * smd).sum()):.2f})",
          {"unadjusted": float((imp * smd_raw).sum()), "max": float(smd.max()),
           "share_features_above_0.25": float((smd > 0.25).mean()),
           "top_features": {meta["features"][i]: float(smd[i]) for i in top}})

    # domain classifier (class-mix adjusted, plus unadjusted for information)
    auc, p_oof = _domain_classifier(Xr, wr, gr, Xt, gt, seed)
    auc_raw, _ = _domain_classifier(Xr, np.ones(len(Xr)), gr, Xt, gt, seed)
    grade("domain_classifier", auc, "domain_auc",
          f"a classifier separates target from reference pixels (class-mix adjusted AUC {auc:.2f})",
          {"unadjusted_auc": auc_raw, "folds": "grouped by polygon/parcel", "model": "HistGradientBoosting"})

    # AOA / dissimilarity
    if aoa is not None:
        di_t = aoa.dissimilarity(Xt)
        out_t = float((di_t > aoa.threshold).mean())
        tdi = np.asarray(reference.get("train_di", np.empty(0)), dtype="float64")
        out_r = float((tdi > aoa.threshold).mean()) if len(tdi) else 0.0
        excess = max(0.0, out_t - out_r)
        grade("aoa", excess, "aoa_excess_outside",
              f"{100 * out_t:.1f} % of target pixels lie outside the AOA (training CV: {100 * out_r:.1f} %)",
              {"target_outside_share": out_t, "reference_outside_share": out_r, "threshold": float(aoa.threshold)})
        med_r = float(np.median(tdi)) if len(tdi) else float("nan")
        r = float(np.median(di_t) / med_r) if len(tdi) and med_r > 0 else float("nan")
        grade("dissimilarity", r, "di_median_ratio",
              f"median dissimilarity index is {r:.2f} x the training CV median",
              {"target_median_di": float(np.median(di_t)), "reference_median_di": med_r,
               "target_q90_di": float(np.quantile(di_t, 0.9)),
               "reference_q90_di": float(np.quantile(tdi, 0.9)) if len(tdi) else None})
    else:
        signals["aoa"] = {"value": None, "level": 0, "note": "no AOA model"}

    # metadata
    md = {"region": cfg["region"], "year": cfg["year"], **{k: v for k, v in (metadata or {}).items()
                                                             if v is not None}}
    cal_years = meta.get("calibration_years") or meta.get("years") or {}
    cal_regions = meta.get("calibration_regions") or meta.get("regions") or {}
    diffs, unknown = [], []
    for key, known in (("year", cal_years), ("region", cal_regions)):
        v = md.get(key)
        if v is None or not known:
            unknown.append(key)
        elif str(v) not in known:
            diffs.append(f"{key} {v} not among calibration {key}s {sorted(known)}")
    signals["metadata"] = {"value": len(diffs), "level": int(bool(diffs)), "prediction": {k: md.get(k) for k in
                                                                                          ("region", "year")},
                           "calibration_years": cal_years, "calibration_regions": cal_regions,
                           "unknown": unknown}
    for d in diffs:
        reasons.append({"signal": "metadata", "level": "mild", "message": d})

    level = max(s["level"] for s in signals.values())
    verdict = VERDICTS[level]

    # conformal: object-level likelihood ratios and adjusted thresholds
    conf_block = {"status": NOT_ESTABLISHED if level else "assumed exchangeable (no shift detected)"}
    p_ref = np.clip(p_oof[:len(Xr)], 0.01, 0.99)
    p_tgt = np.clip(p_oof[len(Xr):], 0.01, 0.99)
    lr_ref, lr_tgt = p_ref / (1 - p_ref), p_tgt / (1 - p_tgt)
    obj_ref = pd.Series(lr_ref).groupby(gr).mean()
    obj_tgt = pd.Series(lr_tgt).groupby(gt).mean()
    cal_poly = np.asarray(reference["cal_polygon"]).astype(str)
    w_cal = obj_ref.reindex(cal_poly).to_numpy()
    fill = float(np.nanmedian(w_cal)) if np.isfinite(w_cal).any() else 1.0
    missing = int((~np.isfinite(w_cal)).sum())
    w_cal = np.where(np.isfinite(w_cal), w_cal, fill)
    q = float(cfg["shift_test_weight_quantile"])
    w_test = float(np.quantile(obj_tgt.to_numpy(), q)) if len(obj_tgt) else 1.0
    ess = float(w_cal.sum() ** 2 / (w_cal ** 2).sum()) if len(w_cal) else 0.0
    conf_block["weights"] = {"calibration_objects": int(len(w_cal)), "without_sampled_pixels": missing,
                             "effective_sample_size": ess, "test_weight": w_test, "test_weight_quantile": q,
                             "calibration_weight_median": float(np.median(w_cal)) if len(w_cal) else None}
    if conformal is not None and len(reference["cal_label"]):
        thr = {}
        for a in conformal.get("alphas", []):
            a = float(a)
            thr[str(a)] = {
                "none": {c: float(v) for c, v in conformal["thresholds"][str(a)].items()},
                "conservative": conservative_thresholds(reference["cal_proba"], reference["cal_label"], classes, a),
                "weighted": weighted_thresholds(reference["cal_proba"], reference["cal_label"], classes, a,
                                                w_cal, w_test)}
        conf_block["thresholds"] = thr
    conf_block["applied"] = cfg["shift_conformal"] if level else "none"
    report = {
        "schema": SCHEMA, "verdict": verdict, "level": level, "reasons": reasons, "signals": signals,
        "policy": cfg["shift_policy"], "conformal_mode": cfg["shift_conformal"], "conformal": conf_block,
        "sample": {"target_pixels": int(len(Xt)), "target_groups": int(len(np.unique(gt))),
                   "reference_pixels": int(len(Xr)), "reference_groups": int(len(np.unique(gr))),
                   "target": sample_note},
        "thresholds": {k: list(v) for k, v in limits.items()},
        "references": REFERENCES,
        "note": "Shift signals are diagnostics with a-priori limits, not hypothesis tests with error control. "
                "Under any detected shift, conformal coverage is not established; adjusted sets are a "
                "safeguard, not a guarantee.",
    }
    return _plain(report)


def unavailable(reason: str, config: dict | None = None) -> dict:
    cfg = settings(config)
    return {"schema": SCHEMA, "verdict": "unknown", "level": None, "reasons": [], "signals": {},
            "policy": cfg["shift_policy"], "conformal_mode": cfg["shift_conformal"],
            "conformal": {"status": "not checked for domain shift", "applied": "none"},
            "status": "unavailable", "unavailable_reason": reason, "references": REFERENCES}


def coverage_statement(report: dict | None, alpha: float | None) -> str | None:
    """Human-readable conformal coverage statement for outputs (None without conformal sets)."""
    if alpha is None:
        return None
    if report is None or report.get("verdict") == "unknown":
        return f"target {100 * (1 - alpha):.0f} % assumes exchangeable calibration and target parcels " \
               "(not checked for domain shift)"
    if report["level"]:
        applied = report["conformal"].get("applied", "none")
        return f"{NOT_ESTABLISHED}; sets: {applied}" + ("" if applied == "none" else " (shift-adjusted)")
    return f"target {100 * (1 - alpha):.0f} % assumes exchangeable calibration and target parcels " \
           "(no domain shift detected)"


def applied_thresholds(report: dict | None, alpha: float) -> dict | None:
    """Adjusted per-class thresholds to use instead of conformal.json, or None (use the stored ones)."""
    if not report or not report.get("level"):
        return None
    mode = report.get("conformal", {}).get("applied", "none")
    if mode == "none":
        return None
    return report["conformal"].get("thresholds", {}).get(str(alpha), {}).get(mode)


# --------------------------------------------------------------------------- policy

MODEL_CODES = {"review_decision", "conformal_discrepancy", "declaration_discrepancy", "ambiguous_prediction_set",
               "mixed_evidence"}


def apply_policy(frame, light, codes: list[list[str]], report: dict | None, min_aoa_fraction: float | None = 0.5):
    """Route parcels by the shift policy. Returns (light Series, codes, withheld bool array).

    review   every parcel whose class comes from the model (not grey) becomes yellow with ``DOMAIN_SHIFT``;
             a red light that also rests on non-model evidence (calendar, markers) stays red, with the code.
    abstain  as review; under a *severe* shift, parcels outside the AOA (aoa_fraction below the minimum or
             unknown) are grey with ``DOMAIN_SHIFT``: the model decision is withheld.
    warn     nothing changes.
    """
    light = pd.Series(light, index=frame.index).copy()
    codes = [list(c) for c in codes]
    withheld = np.zeros(len(frame), dtype=bool)
    if not report or not report.get("level") or report.get("policy", "warn") == "warn":
        return light, codes, withheld
    severe = report["level"] >= 2
    for k, idx in enumerate(frame.index):
        colour = light.loc[idx]
        if colour == "grey":
            continue
        if report["policy"] == "abstain" and severe and "aoa_fraction" in frame:
            a = frame.at[idx, "aoa_fraction"]
            outside = a is None or not np.isfinite(float(a)) or (min_aoa_fraction is not None
                                                               and float(a) < min_aoa_fraction)
            if outside:
                light.loc[idx] = "grey"
                withheld[k] = True
                codes[k].append(REASON_CODE)
                continue
        if colour == "green" or (colour == "red" and set(codes[k]) <= MODEL_CODES):
            light.loc[idx] = "yellow"
        codes[k].append(REASON_CODE)
    return light, codes, withheld


# --------------------------------------------------------------------------- raster sampling (pipeline)

def sample_stack(stack: str | Path, features: list[str], parcels=None, id_field: str = "parcel_id",
                 n: int = 5000, per_group: int = 10, seed: int = 42) -> tuple[np.ndarray, np.ndarray, str]:
    """Target pixel sample of a feature stack: inside parcels (groups = parcel ids) when given, else from
    random 64 x 64 windows of the whole map (groups = window ids; includes non-agricultural land, so the
    shift signals are then less specific)."""
    import geopandas as gpd
    import rasterio
    from rasterio.features import geometry_mask
    from rasterio.windows import Window

    from .raster import valid_mask
    from .samples import polygon_window

    rng = np.random.default_rng(seed)
    rows, groups = [], []
    with rasterio.open(stack) as src:
        names = list(src.descriptions)
        missing = [f for f in features if f not in names]
        if missing:
            raise ValueError(f"Feature stack lacks model features {missing[:5]}")
        bands = [names.index(f) + 1 for f in features]
        if parcels is not None:
            gdf = gpd.read_file(parcels) if not isinstance(parcels, gpd.GeoDataFrame) else parcels
            gdf = gdf.to_crs(src.crs)
            order = rng.permutation(len(gdf))[: max(1, 3 * n // per_group)]
            for i in order:
                if sum(len(r) for r in rows) >= n:
                    break
                geom = gdf.geometry.iloc[i]
                if geom is None or geom.is_empty:
                    continue
                win = polygon_window(geom, src)
                if win is None:
                    continue
                cube = src.read(bands, window=win)
                inside = geometry_mask([geom], out_shape=cube.shape[1:], transform=src.window_transform(win),
                                       invert=True) & valid_mask(cube, src.nodata)
                px = cube[:, inside].T
                if len(px) == 0:
                    continue
                px = px[rng.permutation(len(px))[:per_group]]
                rows.append(px)
                groups.append(np.full(len(px), str(gdf[id_field].iloc[i]) if id_field in gdf else str(i)))
            note = "pixels inside parcels"
        else:
            size = 64
            for t in range(max(1, n // per_group)):
                if sum(len(r) for r in rows) >= n:
                    break
                c0 = int(rng.integers(0, max(1, src.width - size + 1)))
                r0 = int(rng.integers(0, max(1, src.height - size + 1)))
                win = Window(c0, r0, min(size, src.width), min(size, src.height))
                cube = src.read(bands, window=win)
                ok = valid_mask(cube, src.nodata)
                px = cube[:, ok].T
                if len(px) == 0:
                    continue
                px = px[rng.permutation(len(px))[:per_group * 4]]
                rows.append(px)
                groups.append(np.full(len(px), f"w{t}"))
            note = "whole map (random windows; includes non-agricultural land)"
    if not rows:
        return np.empty((0, len(features))), np.empty(0, dtype=str), note
    return np.vstack(rows).astype("float64"), np.concatenate(groups), note


def _plain(v):
    if isinstance(v, dict):
        return {str(k): _plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    if isinstance(v, np.generic):
        v = v.item()
    if isinstance(v, float) and not math.isfinite(v):
        return None
    return v


REFERENCES = [
    "Meyer, H., Pebesma, E. (2021). Predicting into unknown space? Estimating the area of applicability of "
    "spatial prediction models. Methods in Ecology and Evolution 12, 1620-1633.",
    "Rabanser, S., Guennemann, S., Lipton, Z. C. (2019). Failing Loudly: An Empirical Study of Methods for "
    "Detecting Dataset Shift. NeurIPS 32.",
    "Lipton, Z. C., Wang, Y.-X., Smola, A. (2018). Detecting and Correcting for Label Shift with Black Box "
    "Predictors. ICML.",
    "Austin, P. C. (2009). Balance diagnostics for comparing the distribution of baseline covariates between "
    "treatment groups in propensity-score matched samples. Statistics in Medicine 28, 3083-3107.",
    "Tibshirani, R. J., Barber, R. F., Candes, E. J., Ramdas, A. (2019). Conformal Prediction Under Covariate "
    "Shift. NeurIPS 32.",
    "Barber, R. F., Candes, E. J., Ramdas, A., Tibshirani, R. J. (2023). Conformal prediction beyond "
    "exchangeability. Annals of Statistics 51(2), 816-845.",
]
