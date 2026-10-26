"""Event and homogeneity markers for Checks by Monitoring, and configurable traffic-light rules.

Markers are computed from long parcel-signature tables (see `geocrop.cbm.parcel_signatures` and
`geocrop.cbm.s1_signatures`) and are pure table operations — no raster access:

* harvest   abrupt Sentinel-2 index drop after the season peak, confirmed (or found) by a step change in
            Sentinel-1 VH / VH-VV backscatter. A gradual decline without a backscatter step is reported as
            `senescence` (maturation or drought senescence of a standing crop), not as harvest.
* tillage   bare soil after harvest (low index) with a persistent rise of VV backscatter (surface roughness).
* mowing    index drops followed by regrowth (grassland), counted with dates; Sentinel-1 support per event.
* homogeneity  within-parcel dispersion and bimodality of the index per acquisition; persistent bimodal
            dispersion points to a split or mixed declaration.
* data sufficiency  clear Sentinel-2 observations, longest gap, Sentinel-1 acquisitions per orbit group.

Sentinel-1 series are analysed separately per orbit group (orbit direction or relative orbit) so that
acquisition geometries are never mixed; evidence from different groups is combined afterwards.

Every marker has an evidence category and a heuristic confidence in [0, 1]. Confidences are ordinal
scores built from evidence strength and data sufficiency (documented in docs/markers.md); they are not
calibrated probabilities. Thresholds are defaults from the literature and synthetic tests and must be
tuned on regional reference data before operational use.
"""
from __future__ import annotations

import math
from datetime import date

import numpy as np
import pandas as pd

from . import timeseries as ts

EVENT_DEFAULTS: dict = {
    "index": "NDVI",
    "step_days": 5,
    "lambda": 5.0,
    "min_coverage": 0.5,
    "min_s2_observations": 8,
    "max_s2_gap_days": 45,
    "min_s1_acquisitions": 6,
    "s1_group_by": "orbit_direction",
    "s1_window_days": 24,
    "s1_min_obs": 2,
    "s1_step_db": 1.5,
    "s1_min_z": 3.0,
    "s1_match_days": 18,
    "harvest_drop": 0.2,
    "abrupt_drop": 0.2,
    "abrupt_window_days": 20,
    "harvest_s1_sign": "negative",
    "bare_soil_ndvi": 0.25,
    "min_bare_days": 15,
    "tillage_step_db": 1.5,
    "mowing_drop": 0.15,
    "mowing_window_days": 30,
    "mowing_lambda": 2.0,
    "recovery_fraction": 0.5,
    "recovery_days": 40,
    "min_event_observations": 2,
    "heterogeneity_std": 0.1,
    "bimodality_threshold": 5 / 9,
    "heterogeneity_min_dates": 2,
    "heterogeneity_min_fraction": 0.25,
    "min_pixels_homogeneity": 10,
}

OUTCOMES = ("ignore", "yellow", "red")

RULE_DEFAULTS: dict = {
    "enabled": False,
    "min_marker_confidence": 0.5,
    "harvest_expected_classes": [],
    "no_harvest_outcome": "yellow",
    "mowing_expected_classes": [],
    "min_mowing_events": 1,
    "no_mowing_outcome": "yellow",
    "tillage_forbidden_classes": [],
    "tillage_outcome": "yellow",
    "heterogeneity_outcome": "yellow",
    "mixed_class_share": None,
    "mixed_class_outcome": "yellow",
    "insufficient_data_outcome": "ignore",
}

#: Columns produced by `event_markers` (besides parcel_id); `parcels.aggregate` copies them when rules apply.
EVENT_COLUMNS = (
    "s2_clear_observations", "s2_observed_fraction", "s2_longest_gap_days", "s1_acquisitions",
    "s1_orbit_groups", "s2_sufficient", "s1_sufficient", "data_sufficiency",
    "event_peak_doy", "harvest_evidence", "harvest_detected", "harvest_event_doy", "harvest_s2_drop",
    "harvest_s1_step_db", "harvest_s1_doy", "harvest_s1_groups", "harvest_confidence",
    "tillage_evidence", "tillage_detected", "tillage_doy", "tillage_s1_step_db", "bare_days_after_harvest",
    "tillage_confidence",
    "mowing_events", "mowing_doys", "mowing_s1_supported", "mowing_confidence",
    "homogeneity", "homogeneity_basis", "homogeneity_dates", "heterogeneous_dates", "heterogeneous_fraction",
    "max_within_std", "max_bimodality", "homogeneity_confidence",
)

_POSITIVE_INT = ("step_days", "min_s2_observations", "min_s1_acquisitions", "s1_min_obs", "min_event_observations",
                 "heterogeneity_min_dates", "min_pixels_homogeneity")
_POSITIVE = ("lambda", "mowing_lambda", "max_s2_gap_days", "s1_window_days", "s1_step_db", "s1_match_days", "harvest_drop",
             "abrupt_drop", "abrupt_window_days", "tillage_step_db", "mowing_drop", "mowing_window_days",
             "recovery_days", "heterogeneity_std", "min_bare_days")
_FRACTION = ("min_coverage", "recovery_fraction", "bimodality_threshold", "heterogeneity_min_fraction")


def _number(value) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def resolve_event_params(params: dict | None = None) -> dict:
    """Defaults overridden by `params`; unknown keys and out-of-range values raise ValueError."""
    p = dict(EVENT_DEFAULTS)
    unknown = set(params or {}) - set(p)
    if unknown:
        raise ValueError(f"Unknown marker parameter(s): {sorted(unknown)}")
    p.update(params or {})
    for key in _POSITIVE_INT:
        if isinstance(p[key], bool) or not isinstance(p[key], (int, np.integer)) or p[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    for key in _POSITIVE:
        if not _number(p[key]) or p[key] <= 0:
            raise ValueError(f"{key} must be a positive number")
    for key in _FRACTION:
        if not _number(p[key]) or not 0 < p[key] <= 1:
            raise ValueError(f"{key} must be in (0, 1]")
    if not _number(p["s1_min_z"]) or p["s1_min_z"] < 0:
        raise ValueError("s1_min_z must be a nonnegative number")
    if not _number(p["bare_soil_ndvi"]):
        raise ValueError("bare_soil_ndvi must be a number")
    if p["s1_group_by"] not in ("orbit_direction", "relative_orbit"):
        raise ValueError("s1_group_by must be orbit_direction or relative_orbit")
    if p["harvest_s1_sign"] not in ("negative", "any"):
        raise ValueError("harvest_s1_sign must be negative or any")
    if not isinstance(p["index"], str) or not p["index"]:
        raise ValueError("index must be an index name")
    return p


def resolve_rules(rules: dict | None = None) -> dict:
    """Traffic-light marker rules with defaults; validates outcomes, class lists and thresholds."""
    r = dict(RULE_DEFAULTS)
    unknown = set(rules or {}) - set(r)
    if unknown:
        raise ValueError(f"Unknown marker rule key(s): {sorted(unknown)}")
    r.update(rules or {})
    if not isinstance(r["enabled"], bool):
        raise ValueError("rules.enabled must be true or false")
    if not _number(r["min_marker_confidence"]) or not 0 <= r["min_marker_confidence"] <= 1:
        raise ValueError("min_marker_confidence must be in [0, 1]")
    for key in ("harvest_expected_classes", "mowing_expected_classes", "tillage_forbidden_classes"):
        if not isinstance(r[key], (list, tuple)) or not all(isinstance(c, (str, int)) for c in r[key]):
            raise ValueError(f"{key} must be a list of class names")
        r[key] = [str(c) for c in r[key]]
    for key in ("no_harvest_outcome", "no_mowing_outcome", "tillage_outcome", "heterogeneity_outcome",
                "mixed_class_outcome", "insufficient_data_outcome"):
        if r[key] not in OUTCOMES:
            raise ValueError(f"{key} must be one of {OUTCOMES}")
    if isinstance(r["min_mowing_events"], bool) or not isinstance(r["min_mowing_events"], int) \
            or r["min_mowing_events"] < 1:
        raise ValueError("min_mowing_events must be a positive integer")
    share = r["mixed_class_share"]
    if share is not None and (not _number(share) or not 0 < share <= 0.5):
        raise ValueError("mixed_class_share must be null or in (0, 0.5]")
    return r


# --------------------------------------------------------------------------- Sentinel-1 step changes

def _doy(d: str) -> float:
    return float(date.fromisoformat(str(d)[:10]).timetuple().tm_yday)


def s1_steps(doy: np.ndarray, values: np.ndarray, window_days: float, min_obs: int) -> pd.DataFrame:
    """Step changes of one backscatter series (dB) of a single orbit group.

    At every acquisition date d the step is median(values in [d, d + window)) − median(values in
    [d − window, d)), requiring `min_obs` acquisitions on each side. `z` divides the step by its standard
    error, from a robust noise estimate (1.4826 · MAD of consecutive differences / √2, floored at 0.1 dB)
    and the asymptotic efficiency of the median (≈1.2533 σ/√n).
    """
    doy = np.asarray(doy, dtype=float)
    values = np.asarray(values, dtype=float)
    ok = np.isfinite(doy) & np.isfinite(values)
    doy, values = doy[ok], values[ok]
    order = np.argsort(doy)
    doy, values = doy[order], values[order]
    cols = ["doy", "step", "z", "n_before", "n_after"]
    if len(doy) < 2 * min_obs:
        return pd.DataFrame(columns=cols)
    diffs = np.diff(values)
    sigma = 1.4826 * np.median(np.abs(diffs - np.median(diffs))) / math.sqrt(2) if diffs.size else 0.0
    sigma = max(float(sigma), 0.1)
    rows = []
    for d in np.unique(doy):
        before = values[(doy >= d - window_days) & (doy < d)]
        after = values[(doy >= d) & (doy < d + window_days)]
        if len(before) < min_obs or len(after) < min_obs:
            continue
        step = float(np.median(after) - np.median(before))
        se = 1.2533 * sigma * math.sqrt(1 / len(before) + 1 / len(after))
        rows.append((float(d), step, step / se, len(before), len(after)))
    return pd.DataFrame(rows, columns=cols)


def _s1_tables(s1: pd.DataFrame | None, group_by: str, window: float, min_obs: int) -> tuple[dict, dict, dict]:
    """Per parcel: steps per (group, feature), acquisitions per group and the set of acquisition dates."""
    steps: dict[str, list[tuple[str, str, pd.DataFrame]]] = {}
    acquisitions: dict[str, dict[str, int]] = {}
    dates: dict[str, set] = {}
    if s1 is None or s1.empty:
        return steps, acquisitions, dates
    frame = s1[s1["mean"].notna() & (s1["n_valid"] > 0)].copy()
    if group_by not in frame:
        raise ValueError(f"Sentinel-1 signatures need a {group_by!r} column to separate acquisition geometries")
    if group_by == "relative_orbit":
        frame["_group"] = frame.get("orbit_direction", "").astype(str) + ":" + frame[group_by].astype(str)
    else:
        frame["_group"] = frame[group_by].astype(str)
    frame["_doy"] = frame["date"].map(_doy)
    frame["parcel_id"] = frame["parcel_id"].astype(str)
    for (pid, group, feature), part in frame.groupby(["parcel_id", "_group", "index"], sort=True):
        steps.setdefault(pid, []).append((group, feature, s1_steps(part["_doy"], part["mean"], window, min_obs)))
        acquisitions.setdefault(pid, {})
        acquisitions[pid][group] = max(acquisitions[pid].get(group, 0), int(part["_doy"].nunique()))
        dates.setdefault(pid, set()).update(part["date"].astype(str).str[:10])
    return steps, acquisitions, dates


def _best_step(entries, features, lo, hi, sign, step_db, min_z):
    """Strongest qualifying step within [lo, hi] over groups and features.

    Returns (best row or None, supporting groups, whether any step could be computed in the interval)."""
    best, groups, available = None, set(), False
    for group, feature, table in entries or []:
        if feature not in features or table.empty:
            continue
        sel = table[(table.doy >= lo) & (table.doy <= hi)]
        if sel.empty:
            continue
        available = True
        if sign == "negative":
            sel = sel[sel.step < 0]
        elif sign == "positive":
            sel = sel[sel.step > 0]
        sel = sel[(sel.step.abs() >= step_db) & (sel.z.abs() >= min_z)]
        if sel.empty:
            continue
        groups.add(group)
        row = sel.loc[sel.z.abs().idxmax()]
        if best is None or abs(row.z) > abs(best["z"]):
            best = {"doy": float(row.doy), "step": float(row.step), "z": float(row.z), "group": group,
                    "feature": feature}
    return best, sorted(groups), available


# --------------------------------------------------------------------------- markers

def _s2_matrix(s2: pd.DataFrame, parcels: list[str], p: dict, grid: list[date]):
    sig = s2[s2["index"] == p["index"]]
    sig = sig[(sig.n_pixels > 0) & (sig.n_valid >= p["min_coverage"] * sig.n_pixels) & sig["mean"].notna()]
    pos = {pid: k for k, pid in enumerate(parcels)}
    obs_dates = sorted({date.fromisoformat(str(d)[:10]) for d in sig.date})
    dpos = {d.isoformat(): k for k, d in enumerate(obs_dates)}
    values = np.full((len(parcels), len(obs_dates)), np.nan)
    for r in sig.itertuples():
        values[pos[str(r.parcel_id)], dpos[str(r.date)[:10]]] = r.mean
    if obs_dates:
        return ts.bin_to_grid(obs_dates, values, np.isfinite(values), grid)
    return np.full((len(parcels), len(grid)), np.nan), np.zeros((len(parcels), len(grid)))


def _low_observations(y, w, event, step_days, recovery_days):
    """Clear observations supporting the low phase of an event (guards against single cloud outliers)."""
    a = event["start_index"] + 1
    b = event["end_index"] + max(1, int(recovery_days // step_days) // 2)
    level = event["_start_value"] - 0.5 * event["drop"]
    seg_y, seg_w = y[a:b + 1], w[a:b + 1]
    return int(((seg_w > 0) & np.isfinite(seg_y) & (seg_y <= level)).sum())


def _events(z_row, y_row, w_row, days, p, min_drop, window):
    found = ts.drop_events(z_row[None], days, min_drop, window, p["recovery_fraction"], p["recovery_days"])[0]
    for e in found:
        e["_start_value"] = float(z_row[e["start_index"]])
        e["observations"] = _low_observations(y_row, w_row, e, p["step_days"], p["recovery_days"])
    return [e for e in found if e["observations"] >= p["min_event_observations"]]


def event_markers(s2_signatures: pd.DataFrame | None, start: date, end: date,
                  s1_signatures: pd.DataFrame | None = None, params: dict | None = None) -> pd.DataFrame:
    """Harvest, tillage, mowing and homogeneity markers with confidence and data-sufficiency fields.

    s2_signatures  output of `cbm.parcel_signatures` (with `moments=True` for the bimodality test)
    s1_signatures  output of `cbm.s1_signatures` (dB parcel means per acquisition and orbit group), optional
    params         overrides of `EVENT_DEFAULTS`
    Days are day-of-year of the grid / acquisition dates (one season within one calendar year).
    """
    p = resolve_event_params(params)
    s2 = s2_signatures if s2_signatures is not None else pd.DataFrame(
        columns=["parcel_id", "date", "index", "mean", "std", "n_valid", "n_pixels"])
    ids = set(s2.parcel_id.astype(str)) if len(s2) else set()
    if s1_signatures is not None and len(s1_signatures):
        ids |= set(s1_signatures.parcel_id.astype(str))
    parcels = sorted(ids)
    grid = ts.make_grid(start, end, p["step_days"])
    days = np.array([d.timetuple().tm_yday for d in grid], dtype=float)
    y, w = _s2_matrix(s2, parcels, p, grid)
    z = ts.whittaker(y, w, lam=p["lambda"], min_valid=4) if len(parcels) else np.zeros((0, len(grid)))
    # mowing dips are short (cut and regrowth within weeks): lighter smoothing keeps their depth
    z_mow = ts.whittaker(y, w, lam=p["mowing_lambda"], min_valid=4) if len(parcels) else z
    support = ts.temporal_support(w, p["step_days"]) if len(parcels) else {
        "observation_count": np.zeros(0), "observed_bin_fraction": np.zeros(0), "longest_gap_days": np.zeros(0)}
    s1_steps_by, s1_acq, s1_dates = _s1_tables(s1_signatures, p["s1_group_by"], p["s1_window_days"], p["s1_min_obs"])
    homog = homogeneity(s2, p).set_index("parcel_id") if len(s2) else pd.DataFrame()
    rows = []
    for k, pid in enumerate(parcels):
        clear = int((w[k] > 0).sum())
        s2_ok = clear >= p["min_s2_observations"] and support["longest_gap_days"][k] <= p["max_s2_gap_days"] \
            and np.isfinite(z[k]).all()
        acq = s1_acq.get(pid, {})
        s1_ok = bool(acq) and max(acq.values()) >= p["min_s1_acquisitions"]
        row = {"parcel_id": pid, "s2_clear_observations": clear,
               "s2_observed_fraction": float(support["observed_bin_fraction"][k]),
               "s2_longest_gap_days": float(support["longest_gap_days"][k]),
               "s1_acquisitions": len(s1_dates.get(pid, ())),
               "s1_orbit_groups": ";".join(f"{g}:{n}" for g, n in sorted(acq.items())),
               "s2_sufficient": bool(s2_ok), "s1_sufficient": bool(s1_ok),
               "data_sufficiency": ("s2+s1" if s2_ok and s1_ok else "s2" if s2_ok else "s1" if s1_ok
                                    else "insufficient")}
        row.update(_harvest_tillage_mowing(z[k], z_mow[k], y[k], w[k], days, s2_ok, s1_ok, s1_steps_by.get(pid), p))
        if pid in homog.index:
            row.update(homog.loc[pid].to_dict())
        else:
            row.update({"homogeneity": "insufficient_data", "homogeneity_basis": None, "homogeneity_dates": 0,
                        "heterogeneous_dates": 0, "heterogeneous_fraction": np.nan, "max_within_std": np.nan,
                        "max_bimodality": np.nan, "homogeneity_confidence": 0.0})
        rows.append(row)
    return pd.DataFrame(rows, columns=["parcel_id", *EVENT_COLUMNS])


def _strength(value, threshold):
    return float(min(1.0, abs(value) / (2 * threshold))) if threshold else 0.0


def _harvest_tillage_mowing(z, z_mow, y, w, days, s2_ok, s1_ok, s1_entries, p) -> dict:
    out = {"event_peak_doy": np.nan, "harvest_evidence": "insufficient_data", "harvest_detected": False,
           "harvest_event_doy": np.nan, "harvest_s2_drop": np.nan, "harvest_s1_step_db": np.nan,
           "harvest_s1_doy": np.nan, "harvest_s1_groups": "", "harvest_confidence": 0.0,
           "tillage_evidence": "insufficient_data", "tillage_detected": False, "tillage_doy": np.nan,
           "tillage_s1_step_db": np.nan, "bare_days_after_harvest": np.nan, "tillage_confidence": 0.0,
           "mowing_events": 0, "mowing_doys": "", "mowing_s1_supported": 0, "mowing_confidence": np.nan}
    harvest_features = ("VH_dB", "VHVV_dB")
    sign = "negative" if p["harvest_s1_sign"] == "negative" else "any"
    match, z_min, step_db = p["s1_match_days"], p["s1_min_z"], p["s1_step_db"]
    end_doy = float(days[-1])
    if not s2_ok:
        if s1_ok:   # S1-only evidence: any qualifying VH/VH-VV decrease in the season
            best, groups, _ = _best_step(s1_entries, harvest_features, days[0], end_doy, sign, step_db, z_min)
            if best is not None:
                out.update({"harvest_evidence": "harvest_s1", "harvest_detected": True,
                            "harvest_event_doy": best["doy"], "harvest_s1_doy": best["doy"],
                            "harvest_s1_step_db": best["step"], "harvest_s1_groups": ";".join(groups),
                            "harvest_confidence": round(0.7 * (0.4 + 0.3 * _strength(best["z"], z_min)), 3)})
            else:
                out.update({"harvest_evidence": "none", "harvest_confidence": 0.3})
        return out
    pk = int(np.argmax(z))
    peak_doy = float(days[pk])
    out["event_peak_doy"] = peak_doy
    # -------- harvest
    abrupt = [e for e in _events(z, y, w, days, p, p["abrupt_drop"], p["abrupt_window_days"])
              if e["start_index"] >= pk]
    decline = float(z[pk] - z[pk:].min())
    harvest = None
    if abrupt:
        e = max(abrupt, key=lambda e: e["drop"])
        s2s = _strength(e["drop"], p["abrupt_drop"])
        best, groups, available = _best_step(s1_entries, harvest_features, e["drop_doy"] - match,
                                             e["drop_doy"] + match, sign, step_db, z_min)
        out.update({"harvest_detected": True, "harvest_event_doy": e["drop_doy"], "harvest_s2_drop": e["drop"]})
        if best is not None:
            out.update({"harvest_evidence": "harvest_s2_s1", "harvest_s1_doy": best["doy"],
                        "harvest_s1_step_db": best["step"], "harvest_s1_groups": ";".join(groups),
                        "harvest_confidence": 0.6 + 0.2 * s2s + 0.2 * _strength(best["z"], z_min)})
        elif available:
            out.update({"harvest_evidence": "harvest_s2_unconfirmed", "harvest_confidence": 0.3 + 0.3 * s2s})
        else:
            out.update({"harvest_evidence": "harvest_s2", "harvest_confidence": 0.4 + 0.3 * s2s})
        harvest = e["drop_doy"]
    else:
        best, groups, available = _best_step(s1_entries, harvest_features, peak_doy, end_doy, sign, step_db, z_min)
        if best is not None and decline >= p["harvest_drop"] / 2:
            out.update({"harvest_evidence": "harvest_s1", "harvest_detected": True, "harvest_event_doy": best["doy"],
                        "harvest_s1_doy": best["doy"], "harvest_s1_step_db": best["step"],
                        "harvest_s2_drop": decline, "harvest_s1_groups": ";".join(groups),
                        "harvest_confidence": 0.4 + 0.3 * _strength(best["z"], z_min)
                        + 0.1 * min(1.0, decline / p["harvest_drop"])})
            harvest = best["doy"]
        elif decline >= p["harvest_drop"]:
            if available and s1_ok:
                out.update({"harvest_evidence": "senescence", "harvest_s2_drop": decline,
                            "harvest_confidence": 0.5 + 0.3 * _strength(decline, p["harvest_drop"])})
            else:
                out.update({"harvest_evidence": "senescence_or_harvest", "harvest_s2_drop": decline,
                            "harvest_confidence": 0.2})
        else:
            out.update({"harvest_evidence": "none", "harvest_s2_drop": decline,
                        "harvest_confidence": 0.7 if s1_ok else 0.5})
    out["harvest_confidence"] = round(float(min(1.0, out["harvest_confidence"])), 3)
    # -------- tillage: bare soil after harvest (or after the peak) + persistent VV rise
    after = harvest if harvest is not None else peak_doy
    bare = (z < p["bare_soil_ndvi"]) & (days > after)
    bare_days = float(bare.sum() * p["step_days"])
    out["bare_days_after_harvest"] = bare_days
    if bare_days >= p["min_bare_days"]:
        first, last = float(days[bare][0]), float(days[bare][-1])
        lo = max(first - match, after + match)
        best, _, available = _best_step(s1_entries, ("VV_dB", "VH_dB"), lo, last + p["step_days"], "positive",
                                        p["tillage_step_db"], z_min)
        bare_strength = min(1.0, bare_days / (2 * p["min_bare_days"]))
        if best is not None:
            out.update({"tillage_evidence": "tillage", "tillage_detected": True, "tillage_doy": best["doy"],
                        "tillage_s1_step_db": best["step"],
                        "tillage_confidence": 0.5 + 0.25 * bare_strength + 0.25 * _strength(best["z"], z_min)})
        elif available and s1_ok:
            out.update({"tillage_evidence": "bare_soil", "tillage_confidence": 0.5})
        else:
            out.update({"tillage_evidence": "bare_soil_no_s1", "tillage_confidence": 0.3})
    else:
        out.update({"tillage_evidence": "none", "tillage_confidence": 0.6 if s1_ok else 0.5})
    out["tillage_confidence"] = round(float(min(1.0, out["tillage_confidence"])), 3)
    # -------- mowing: drops with regrowth
    mows = [e for e in _events(z_mow, y, w, days, p, p["mowing_drop"], p["mowing_window_days"]) if e["recovered"]]
    scores, supported = [], 0
    for e in mows:
        best, _, _ = _best_step(s1_entries, harvest_features, e["drop_doy"] - match, e["drop_doy"] + match, "any",
                                step_db, z_min)
        supported += best is not None
        scores.append(0.4 + 0.3 * _strength(e["drop"], p["mowing_drop"]) + 0.3 * (best is not None))
    out.update({"mowing_events": len(mows), "mowing_doys": ";".join(str(int(e["drop_doy"])) for e in mows),
                "mowing_s1_supported": int(supported),
                "mowing_confidence": round(float(np.mean(scores)), 3) if scores else (0.7 if s1_ok else 0.5)})
    return out


def homogeneity(s2_signatures: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
    """Within-parcel heterogeneity of the index per acquisition, summarised per parcel.

    A date is heterogeneous when the within-parcel standard deviation reaches `heterogeneity_std` and —
    when the signatures carry `bimodality` (Sarle's coefficient, `cbm.parcel_signatures(moments=True)`) —
    the coefficient exceeds `bimodality_threshold` (5/9, the value of a uniform distribution). A parcel is
    `heterogeneous` when at least `heterogeneity_min_dates` dates and `heterogeneity_min_fraction` of its
    usable dates are heterogeneous: a persistent two-population pattern, as produced by two crops or a
    partly used parcel under one declaration. Fewer than three usable dates give `insufficient_data`.
    """
    p = resolve_event_params(params)
    sig = s2_signatures[s2_signatures["index"] == p["index"]]
    sig = sig[(sig.n_pixels > 0) & (sig.n_valid >= p["min_coverage"] * sig.n_pixels)
              & (sig.n_valid >= p["min_pixels_homogeneity"]) & sig["std"].notna()]
    has_bc = "bimodality" in sig and sig["bimodality"].notna().any()
    rows = []
    for pid, part in sig.groupby(sig.parcel_id.astype(str)):
        het = part["std"] >= p["heterogeneity_std"]
        if has_bc:
            het &= part["bimodality"].fillna(0) >= p["bimodality_threshold"]
        n, h = len(part), int(het.sum())
        fraction = h / n if n else np.nan
        if n < 3:
            flag, conf = "insufficient_data", 0.0
        elif h >= p["heterogeneity_min_dates"] and fraction >= p["heterogeneity_min_fraction"]:
            flag, conf = "heterogeneous", (0.5 + 0.5 * min(1.0, fraction / (2 * p["heterogeneity_min_fraction"])))
        else:
            flag, conf = "homogeneous", 0.5 + 0.5 * (1 - min(1.0, fraction / p["heterogeneity_min_fraction"]))
        conf *= min(1.0, n / 6) * (1.0 if has_bc else 0.7)
        rows.append({"parcel_id": pid, "homogeneity": flag, "homogeneity_basis": "std+bimodality" if has_bc
                     else "std_only", "homogeneity_dates": n, "heterogeneous_dates": h,
                     "heterogeneous_fraction": fraction, "max_within_std": float(part["std"].max()),
                     "max_bimodality": float(part["bimodality"].max()) if has_bc else np.nan,
                     "homogeneity_confidence": round(float(conf), 3)})
    return pd.DataFrame(rows, columns=["parcel_id", "homogeneity", "homogeneity_basis", "homogeneity_dates",
                                       "heterogeneous_dates", "heterogeneous_fraction", "max_within_std",
                                       "max_bimodality", "homogeneity_confidence"])


def bimodality_coefficient(skewness, kurtosis):
    """Sarle's bimodality coefficient (g1² + 1) / g2 from skewness and (non-excess) kurtosis."""
    with np.errstate(invalid="ignore", divide="ignore"):
        return (np.asarray(skewness, dtype=float) ** 2 + 1) / np.asarray(kurtosis, dtype=float)


def class_mixture(frame: pd.DataFrame) -> pd.Series:
    """Second-largest pixel-vote share per parcel from `f_<class>` columns of `parcels.aggregate` (NaN if absent)."""
    cols = [c for c in frame.columns if c.startswith("f_")]
    if len(cols) < 2:
        return pd.Series(np.nan, index=frame.index)
    values = np.sort(frame[cols].to_numpy(dtype=float), axis=1)
    return pd.Series(values[:, -2], index=frame.index)


# --------------------------------------------------------------------------- traffic-light rules

_SEVERITY = {"grey": -1, "green": 0, "yellow": 1, "red": 2}


def _value(row, name, default=None):
    v = row.get(name, default)
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return default
    return v


def marker_rule_reasons(row: dict, rules: dict, mixture: float | None = None) -> list[tuple[str, str]]:
    """(reason code, outcome) pairs triggered by the marker columns of one parcel row."""
    out = []
    declared = _value(row, "declared_class")
    declared = None if declared is None else str(declared)
    min_conf = rules["min_marker_confidence"]
    sufficiency = _value(row, "data_sufficiency")

    def insufficient(code):
        out.append((f"insufficient_marker_data_{code}", rules["insufficient_data_outcome"]))

    if declared is not None and declared in rules["harvest_expected_classes"]:
        evidence = _value(row, "harvest_evidence")
        if evidence is None or evidence == "insufficient_data" or sufficiency in (None, "insufficient"):
            insufficient("harvest")
        elif not bool(_value(row, "harvest_detected", False)) and \
                float(_value(row, "harvest_confidence", 0)) >= min_conf:
            out.append(("no_harvest_evidence", rules["no_harvest_outcome"]))
    if declared is not None and declared in rules["mowing_expected_classes"]:
        if not bool(_value(row, "s2_sufficient", False)):
            insufficient("mowing")
        elif int(_value(row, "mowing_events", 0)) < rules["min_mowing_events"] and \
                float(_value(row, "mowing_confidence", 0)) >= min_conf:
            out.append(("insufficient_mowing_events", rules["no_mowing_outcome"]))
    if declared is not None and declared in rules["tillage_forbidden_classes"]:
        if bool(_value(row, "tillage_detected", False)) and float(_value(row, "tillage_confidence", 0)) >= min_conf:
            out.append(("tillage_detected", rules["tillage_outcome"]))
    if _value(row, "homogeneity") == "heterogeneous" and \
            float(_value(row, "homogeneity_confidence", 0)) >= min_conf:
        out.append(("heterogeneous_parcel", rules["heterogeneity_outcome"]))
    if rules["mixed_class_share"] is not None and mixture is not None and np.isfinite(mixture) \
            and mixture >= rules["mixed_class_share"]:
        out.append(("mixed_class_pixels", rules["mixed_class_outcome"]))
    return [(code, outcome) for code, outcome in out if outcome != "ignore"]


def apply_rules(frame: pd.DataFrame, base: pd.Series, base_reasons: list[list[str]], rules: dict) -> pd.DataFrame:
    """Escalate base colours with marker rules; grey stays grey and review decisions stay yellow."""
    mixture = class_mixture(frame)
    colours, reasons = [], []
    for (idx, row), colour, codes in zip(frame.iterrows(), base, base_reasons):
        triggered = marker_rule_reasons(row.to_dict(), rules, mixture.loc[idx])
        codes = list(codes) + [code for code, _ in triggered]
        locked = colour == "grey" or row.get("declaration_status") == "uncertain" or row.get("decision") == "review"
        if not locked:
            for _, outcome in triggered:
                if _SEVERITY[outcome] > _SEVERITY[colour]:
                    colour = outcome
        colours.append(colour)
        reasons.append(";".join(codes))
    return pd.DataFrame({"traffic_light": colours, "traffic_light_reasons": reasons}, index=frame.index)
