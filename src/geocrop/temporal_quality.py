"""Temporal quality: which values were OBSERVED and which were INTERPOLATED.

Whittaker smoothing (`geocrop.timeseries.whittaker`) fills every gap of a regular grid, so a smoothed
series, its phenology metrics (SOS, peak, EOS), the CbM markers and the parcel-classifier inputs look
equally complete whether they rest on 25 clear acquisitions or on four. This module makes the
difference explicit and actionable:

* observation matrices  ``obs`` (n_rows, k): day of year of each *actual* clear observation of a row
                        (pixel or parcel), NaN where an acquisition was cloudy/missing. Built from the
                        exact acquisition dates, not from grid bins.
* support metrics       clear observations, longest interval without an observation (including the
                        season edges), share of the season within ``tolerance_days`` of an observation,
                        and the interpolation dependence of every phenology metric: distance in days
                        from the metric's date to the nearest real observation and the length of the
                        observation gap that contains it.
* per-pixel rasters     `PIXEL_METRICS`, written by `geocrop.tsfeatures.build_timeseries_stack(...,
                        temporal_support=True)` next to the time-series stack with a manifest.
* per-parcel table      `parcel_support` on CbM signatures (``tq_*`` columns).
* routing rules         `RULE_DEFAULTS` (config ``cbm.temporal_support``): weakly supported decisions go to
                        review (yellow) with reason codes ``LOW_OBS``, ``LONG_GAP``, ``LONG_GAP_AT_PEAK``,
                        ``INTERPOLATED_PHENOLOGY``, ``INTERPOLATED_FEATURES`` and ``NO_TEMPORAL_SUPPORT``.
                        Disabled by default, so existing traffic lights are unchanged.

Thresholds are defaults chosen for a 5-day Sentinel-2 revisit; tune them on regional reference data.
"""
from __future__ import annotations

import math
from datetime import date

import numpy as np
import pandas as pd

from . import timeseries as ts

#: Phenology dates whose interpolation dependence is measured (name -> key of `timeseries.phenology`).
PHENOLOGY_DATES = {"sos": "sos", "peak": "peak_doy", "eos": "eos"}

#: Bands of the per-pixel temporal-support raster (prefixed ``tq_<index>_``).
PIXEL_METRICS = ("clear_observations", "longest_gap_days", "season_coverage", "sos_obs_distance_days",
                 "peak_obs_distance_days", "eos_obs_distance_days", "peak_gap_days",
                 "interpolated_phenology", "interpolated_feature_fraction")

PIXEL_DESCRIPTIONS = {
    "clear_observations": "number of clear (cloud-free, valid) acquisitions in the season",
    "longest_gap_days": "longest interval in days without a clear acquisition, season edges included",
    "season_coverage": "share of grid days within tolerance_days of a clear acquisition",
    "sos_obs_distance_days": "days from start of season to the nearest clear acquisition",
    "peak_obs_distance_days": "days from the peak to the nearest clear acquisition",
    "eos_obs_distance_days": "days from end of season to the nearest clear acquisition",
    "peak_gap_days": "length in days of the observation gap containing the peak (0 = peak observed)",
    "interpolated_phenology": "number of SOS/peak/EOS farther than tolerance_days from a clear acquisition",
    "interpolated_feature_fraction": "share of smoothed-series and phenology features farther than "
                                     "tolerance_days from a clear acquisition",
}

REASON_CODES = {
    "LOW_OBS": "fewer clear observations than min_clear_observations",
    "LONG_GAP": "an observation gap longer than max_gap_days",
    "LONG_GAP_AT_PEAK": "the season peak lies in an observation gap longer than max_gap_at_peak_days",
    "INTERPOLATED_PHENOLOGY": "more than max_interpolated_phenology of SOS/peak/EOS rest on interpolation",
    "INTERPOLATED_FEATURES": "share of interpolation-dependent features above max_interpolated_feature_fraction",
    "NO_TEMPORAL_SUPPORT": "no temporal-support record for the parcel",
}

OUTCOMES = ("ignore", "yellow")

RULE_DEFAULTS: dict = {
    "enabled": False,
    # measurement (per-parcel support from signatures)
    "index": "NDVI",
    "step_days": 5,
    "lambda": 20.0,
    "min_coverage": 0.5,
    "output_step_days": 15,
    "tolerance_days": 10,
    # routing thresholds (null disables a check)
    "min_clear_observations": 6,
    "max_gap_days": 60,
    "max_gap_at_peak_days": 30,
    "max_interpolated_phenology": 0,
    "max_interpolated_feature_fraction": None,
    "outcome": "yellow",
    "missing_support_outcome": "ignore",
    "demote_red": False,
}

_SEVERITY = {"grey": -1, "green": 0, "yellow": 1, "red": 2}


def _number(value) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float, np.integer, np.floating)) \
        and math.isfinite(value)


def resolve(rules: dict | None = None) -> dict:
    """Defaults overridden by `rules`; unknown keys and invalid values raise ValueError."""
    r = dict(RULE_DEFAULTS)
    unknown = set(rules or {}) - set(r)
    if unknown:
        raise ValueError(f"Unknown temporal_support key(s): {sorted(unknown)}")
    r.update(rules or {})
    for key in ("enabled", "demote_red"):
        if not isinstance(r[key], bool):
            raise ValueError(f"temporal_support.{key} must be true or false")
    if not isinstance(r["index"], str) or not r["index"]:
        raise ValueError("temporal_support.index must be an index name")
    for key in ("step_days", "output_step_days"):
        if isinstance(r[key], bool) or not isinstance(r[key], (int, np.integer)) or r[key] < 1:
            raise ValueError(f"temporal_support.{key} must be a positive integer")
    if r["output_step_days"] % r["step_days"]:
        raise ValueError("temporal_support.output_step_days must be a multiple of step_days")
    for key in ("lambda", "tolerance_days"):
        if not _number(r[key]) or r[key] <= 0:
            raise ValueError(f"temporal_support.{key} must be a positive number")
    if not _number(r["min_coverage"]) or not 0 < r["min_coverage"] <= 1:
        raise ValueError("temporal_support.min_coverage must be in (0, 1]")
    for key in ("min_clear_observations", "max_interpolated_phenology"):
        v = r[key]
        if v is not None and (isinstance(v, bool) or not isinstance(v, (int, np.integer)) or v < 0):
            raise ValueError(f"temporal_support.{key} must be null or a nonnegative integer")
    for key in ("max_gap_days", "max_gap_at_peak_days"):
        if r[key] is not None and (not _number(r[key]) or r[key] <= 0):
            raise ValueError(f"temporal_support.{key} must be null or a positive number")
    f = r["max_interpolated_feature_fraction"]
    if f is not None and (not _number(f) or not 0 <= f <= 1):
        raise ValueError("temporal_support.max_interpolated_feature_fraction must be null or in [0, 1]")
    for key in ("outcome", "missing_support_outcome"):
        if r[key] not in OUTCOMES:
            raise ValueError(f"temporal_support.{key} must be one of {OUTCOMES}")
    return r


# --------------------------------------------------------------------------- vectorised metrics

def _obs(obs) -> np.ndarray:
    a = np.asarray(obs, dtype="float64")
    if a.ndim != 2:
        raise ValueError("obs must be a (rows, observations) array of day-of-year values (NaN = not observed)")
    return a


def _target(target, n) -> np.ndarray:
    t = np.asarray(target, dtype="float64")
    return np.broadcast_to(t, (n,)) if t.ndim == 0 else t


def clear_observations(obs) -> np.ndarray:
    """Number of distinct observed days per row."""
    o = np.sort(_obs(obs), axis=1)
    finite = np.isfinite(o)
    distinct = finite & np.concatenate([np.ones((len(o), 1), bool), np.diff(o, axis=1) != 0], axis=1)
    return distinct.sum(axis=1)


def distance_to_observation(obs, target) -> np.ndarray:
    """Days from `target` (scalar or per row) to the nearest observation; NaN target -> NaN, no obs -> inf."""
    o = _obs(obs)
    t = _target(target, len(o))
    d = np.abs(o - t[:, None])
    d = np.where(np.isfinite(d), d, np.inf)
    out = d.min(axis=1) if o.shape[1] else np.full(len(o), np.inf)
    return np.where(np.isfinite(t), out, np.nan)


def gap_containing(obs, target, season_start: float, season_end: float) -> np.ndarray:
    """Length of the observation gap containing `target`: next observation minus previous one.

    0 when the target day itself is observed. A missing side is bounded by the season edge, so a
    target before the first observation reports first - season_start. NaN target -> NaN.
    """
    o = _obs(obs)
    t = _target(target, len(o))
    before = np.where(o <= t[:, None], o, -np.inf).max(axis=1) if o.shape[1] else np.full(len(o), -np.inf)
    after = np.where(o >= t[:, None], o, np.inf).min(axis=1) if o.shape[1] else np.full(len(o), np.inf)
    before = np.where(np.isfinite(before), before, season_start)
    after = np.where(np.isfinite(after), after, season_end)
    return np.where(np.isfinite(t), np.maximum(after - before, 0.0), np.nan)


def longest_gap(obs, season_start: float, season_end: float) -> np.ndarray:
    """Longest interval in days without an observation, the season edges counting as bounds."""
    o = _obs(obs)
    lo = np.full((len(o), 1), float(season_start))
    hi = np.full((len(o), 1), float(season_end))
    inside = np.where((o >= season_start) & (o <= season_end), o, np.nan)
    s = np.sort(np.concatenate([lo, inside, hi], axis=1), axis=1)      # NaN sorted last
    s = np.where(np.isfinite(s), s, season_end)
    return np.diff(s, axis=1).max(axis=1)


def season_coverage(obs, grid_days, tolerance_days: float) -> np.ndarray:
    """Share of grid days that lie within `tolerance_days` of an observation."""
    o = _obs(obs)
    g = np.asarray(grid_days, dtype="float64")
    if not o.shape[1] or not len(g):
        return np.zeros(len(o))
    near = np.zeros((len(o), len(g)), dtype=bool)
    for j in range(o.shape[1]):
        col = o[:, j]
        ok = np.isfinite(col)
        if ok.any():
            near[ok] |= np.abs(g[None, :] - col[ok, None]) <= tolerance_days
    return near.mean(axis=1)


def support_metrics(obs, grid_days, tolerance_days: float, phenology: dict | None = None,
                    feature_days=None) -> dict:
    """All support metrics of `PIXEL_METRICS` for each row (pixel or parcel).

    obs           (n, k) observed day of year (NaN = not observed)
    grid_days     regular grid (day of year) on which the series was interpolated
    phenology     output of `timeseries.phenology` (sos, peak_doy, eos ...) or None
    feature_days  days of year of smoothed-series features (e.g. the ``ts_*_d<doy>`` outputs); default
                  every grid day. Together with the phenology dates they make up the denominator of
                  ``interpolated_feature_fraction``.
    """
    o = _obs(obs)
    g = np.asarray(grid_days, dtype="float64")
    lo, hi = float(g[0]), float(g[-1])
    n = len(o)
    out = {"clear_observations": clear_observations(o).astype("float64"),
           "longest_gap_days": longest_gap(o, lo, hi),
           "season_coverage": season_coverage(o, g, tolerance_days)}
    fd = g if feature_days is None else np.asarray(feature_days, dtype="float64")
    dependent = np.zeros(n)
    total = np.zeros(n)
    for d in fd:
        dist = distance_to_observation(o, d)
        dependent += dist > tolerance_days
        total += 1
    interpolated = np.zeros(n)
    counted = np.zeros(n)
    for name, key in PHENOLOGY_DATES.items():
        doy = np.full(n, np.nan) if phenology is None else np.asarray(phenology[key], dtype="float64")
        dist = distance_to_observation(o, doy)
        out[f"{name}_obs_distance_days"] = dist
        finite = np.isfinite(doy)
        far = finite & (dist > tolerance_days)
        interpolated += far
        counted += finite
        dependent += far
        total += finite
    peak = np.full(n, np.nan) if phenology is None else np.asarray(phenology["peak_doy"], dtype="float64")
    out["peak_gap_days"] = gap_containing(o, peak, lo, hi)
    out["interpolated_phenology"] = np.where(counted > 0, interpolated, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        out["interpolated_feature_fraction"] = np.where(total > 0, dependent / total, np.nan)
    return out


# --------------------------------------------------------------------------- per parcel

def _doy(d) -> float:
    return float(date.fromisoformat(str(d)[:10]).timetuple().tm_yday)


def observation_matrix(signatures: pd.DataFrame, index: str, min_coverage: float, parcel_ids=None):
    """(ids, obs (n, k) DOY of clear observations, observed dates per parcel) from a signature table."""
    sig = signatures[signatures["index"] == index]
    ids = sorted(signatures.parcel_id.astype(str).unique()) if parcel_ids is None else [str(p) for p in parcel_ids]
    pos = {p: k for k, p in enumerate(ids)}
    clear = sig[(sig.n_pixels > 0) & (sig.n_valid >= min_coverage * sig.n_pixels) & sig["mean"].notna()
                & sig.parcel_id.astype(str).isin(pos)]
    dates = sorted({str(d)[:10] for d in clear.date})
    dpos = {d: k for k, d in enumerate(dates)}
    obs = np.full((len(ids), len(dates)), np.nan)
    observed: dict[str, list[str]] = {p: [] for p in ids}
    for r in clear.itertuples():
        pid, d = str(r.parcel_id), str(r.date)[:10]
        obs[pos[pid], dpos[d]] = _doy(d)
        observed[pid].append(d)
    return ids, obs, {p: sorted(set(v)) for p, v in observed.items()}


def parcel_series(signatures: pd.DataFrame, start: date, end: date, params: dict | None = None, parcel_ids=None) -> dict:
    """Observed and interpolated parcel series on the season grid (the `cbm.markers` recipe).

    Returns ids, grid (dates), days (DOY), y/w (binned observations and counts), z (Whittaker curve),
    obs (DOY of clear observations), observed (clear dates per parcel), phenology and resolved params.
    """
    p = resolve(params)
    if start.year != end.year:
        raise ValueError("Temporal support uses day-of-year; split cross-year seasons explicitly")
    grid = ts.make_grid(start, end, p["step_days"])
    days = np.array([d.timetuple().tm_yday for d in grid], dtype="float64")
    ids, obs, observed = observation_matrix(signatures, p["index"], p["min_coverage"], parcel_ids)
    dates = sorted({d for v in observed.values() for d in v})
    if dates:
        dpos = {d: k for k, d in enumerate(dates)}
        pos = {pid: k for k, pid in enumerate(ids)}
        values = np.full((len(ids), len(dates)), np.nan)
        sig = signatures[signatures["index"] == p["index"]]
        sig = sig[(sig.n_pixels > 0) & (sig.n_valid >= p["min_coverage"] * sig.n_pixels) & sig["mean"].notna()]
        for r in sig.itertuples():
            if str(r.parcel_id) in pos:
                values[pos[str(r.parcel_id)], dpos[str(r.date)[:10]]] = r.mean
        y, w = ts.bin_to_grid([date.fromisoformat(d) for d in dates], values, np.isfinite(values), grid)
    else:
        y, w = np.full((len(ids), len(grid)), np.nan), np.zeros((len(ids), len(grid)))
    z = ts.whittaker(y, w, lam=p["lambda"], min_valid=4) if len(ids) else np.zeros((0, len(grid)))
    phen = ts.phenology(z, days) if len(ids) else {k: np.zeros(0) for k in ("sos", "peak_doy", "eos")}
    return {"ids": ids, "grid": grid, "days": days, "y": y, "w": w, "z": z, "obs": obs, "observed": observed,
            "phenology": phen, "params": p}


def parcel_support(signatures: pd.DataFrame, start: date, end: date, params: dict | None = None,
                   markers: pd.DataFrame | None = None, parcel_ids=None) -> pd.DataFrame:
    """Per-parcel temporal support (``tq_*`` columns) from a CbM signature table.

    The phenology is recomputed with the same grid/Whittaker settings as `cbm.markers` (index, step_days,
    lambda, min_coverage of `params`), so SOS/peak/EOS match the markers used by the traffic light.
    With `markers` (from `cbm.markers` or `cbm_markers.event_markers`) the distance of the harvest date
    to the nearest observation is added (``tq_harvest_obs_distance_days``).
    """
    series = parcel_series(signatures, start, end, params, parcel_ids)
    p, ids, days, obs = series["params"], series["ids"], series["days"], series["obs"]
    observed, phen = series["observed"], series["phenology"]
    feature_days = days[::max(1, p["output_step_days"] // p["step_days"])]
    m = support_metrics(obs, days, p["tolerance_days"], phen, feature_days)
    frame = pd.DataFrame({"parcel_id": ids})
    frame["tq_clear_observations"] = m["clear_observations"].astype(int)
    frame["tq_first_observation"] = [v[0] if v else None for v in (observed[i] for i in ids)]
    frame["tq_last_observation"] = [v[-1] if v else None for v in (observed[i] for i in ids)]
    frame["tq_longest_gap_days"] = m["longest_gap_days"]
    frame["tq_season_coverage"] = m["season_coverage"]
    for name, key in PHENOLOGY_DATES.items():
        frame[f"tq_{name}_doy"] = phen[key]
        frame[f"tq_{name}_obs_distance_days"] = m[f"{name}_obs_distance_days"]
    frame["tq_sos_gap_days"] = gap_containing(obs, phen["sos"], days[0], days[-1])
    frame["tq_peak_gap_days"] = m["peak_gap_days"]
    frame["tq_eos_gap_days"] = gap_containing(obs, phen["eos"], days[0], days[-1])
    tol = p["tolerance_days"]
    frame["tq_interpolated_phenology"] = m["interpolated_phenology"]
    frame["tq_interpolated_metrics"] = [
        ";".join(n for n in PHENOLOGY_DATES if np.isfinite(frame.at[k, f"tq_{n}_obs_distance_days"])
                 and frame.at[k, f"tq_{n}_obs_distance_days"] > tol) for k in range(len(frame))]
    frame["tq_interpolated_feature_fraction"] = m["interpolated_feature_fraction"]
    if markers is not None and len(markers):
        col = next((c for c in ("harvest_event_doy", "harvest_doy") if c in markers), None)
        if col is not None:
            h = markers.assign(parcel_id=markers.parcel_id.astype(str)).set_index("parcel_id")[col]
            h = pd.to_numeric(h.reindex(ids), errors="coerce").to_numpy(dtype="float64")
            frame["tq_harvest_obs_distance_days"] = distance_to_observation(obs, h)
    frame["tq_tolerance_days"] = float(tol)
    frame["tq_observed_dates"] = [";".join(observed[i]) for i in ids]
    return frame


SUPPORT_COLUMNS = ("tq_clear_observations", "tq_longest_gap_days", "tq_season_coverage", "tq_peak_gap_days",
                   "tq_sos_obs_distance_days", "tq_peak_obs_distance_days", "tq_eos_obs_distance_days",
                   "tq_interpolated_phenology", "tq_interpolated_metrics", "tq_interpolated_feature_fraction")


# --------------------------------------------------------------------------- rules

def _get(row: dict, key):
    v = row.get(key)
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    return v


def evaluate(row: dict, rules: dict) -> list[dict]:
    """Every temporal-support check of one parcel row with inputs, threshold and result.

    Returns dicts ``{"rule", "code", "inputs", "threshold", "triggered", "outcome"}``; `outcome` is the
    colour the check asks for when triggered (``ignore`` = recorded only). Checks with a null threshold
    are skipped.
    """
    r = rules
    checks = []
    has_support = _get(row, "tq_clear_observations") is not None
    if not has_support:
        return [{"rule": "temporal_support.present", "code": "NO_TEMPORAL_SUPPORT", "inputs": {},
                 "threshold": None, "triggered": True, "outcome": r["missing_support_outcome"]}]

    def add(rule, code, inputs, threshold, triggered):
        checks.append({"rule": f"temporal_support.{rule}", "code": code, "inputs": inputs, "threshold": threshold,
                       "triggered": bool(triggered), "outcome": r["outcome"]})

    n = int(_get(row, "tq_clear_observations"))
    if r["min_clear_observations"] is not None:
        add("min_clear_observations", "LOW_OBS", {"tq_clear_observations": n}, r["min_clear_observations"],
            n < r["min_clear_observations"])
    if r["max_gap_days"] is not None:
        g = _get(row, "tq_longest_gap_days")
        add("max_gap_days", "LONG_GAP", {"tq_longest_gap_days": g}, r["max_gap_days"],
            g is not None and float(g) > r["max_gap_days"])
    if r["max_gap_at_peak_days"] is not None:
        g = _get(row, "tq_peak_gap_days")
        add("max_gap_at_peak_days", "LONG_GAP_AT_PEAK", {"tq_peak_gap_days": g, "tq_peak_doy": _get(row, "tq_peak_doy")},
            r["max_gap_at_peak_days"], g is not None and float(g) > r["max_gap_at_peak_days"])
    if r["max_interpolated_phenology"] is not None:
        k = _get(row, "tq_interpolated_phenology")
        add("max_interpolated_phenology", "INTERPOLATED_PHENOLOGY",
            {"tq_interpolated_phenology": k, "tq_interpolated_metrics": _get(row, "tq_interpolated_metrics") or "",
             "tolerance_days": _get(row, "tq_tolerance_days")},
            r["max_interpolated_phenology"], k is not None and float(k) > r["max_interpolated_phenology"])
    if r["max_interpolated_feature_fraction"] is not None:
        f = _get(row, "tq_interpolated_feature_fraction")
        add("max_interpolated_feature_fraction", "INTERPOLATED_FEATURES", {"tq_interpolated_feature_fraction": f},
            r["max_interpolated_feature_fraction"], f is not None and float(f) > r["max_interpolated_feature_fraction"])
    return checks


def rule_reasons(row: dict, rules: dict) -> list[tuple[str, str]]:
    """(reason code, outcome) of the triggered, non-ignored checks."""
    return [(c["code"], c["outcome"]) for c in evaluate(row, rules) if c["triggered"] and c["outcome"] != "ignore"]


def route(colour: str, row: dict, triggered: list[tuple[str, str]], rules: dict) -> str:
    """Colour after temporal-support routing. Grey stays grey; review decisions stay yellow; green goes to
    yellow when a check triggers; red stays red unless `demote_red` (then yellow: a discrepancy resting on
    interpolated evidence is sent to review instead of being reported as confirmed)."""
    if not triggered or colour == "grey":
        return colour
    if colour == "red":
        return "yellow" if rules["demote_red"] else "red"
    target = max((o for _, o in triggered), key=_SEVERITY.__getitem__)
    return target if _SEVERITY[target] > _SEVERITY[colour] else colour


def apply_rules(frame: pd.DataFrame, colours, reasons: list[list[str]], rules: dict) -> pd.DataFrame:
    """Route `colours` (with their reason-code lists) of `frame` rows through the temporal-support checks."""
    out_colour, out_reasons = [], []
    for (_, row), colour, codes in zip(frame.iterrows(), colours, reasons):
        d = row.to_dict()
        triggered = rule_reasons(d, rules)
        out_colour.append(route(colour, d, triggered, rules))
        out_reasons.append(";".join(list(codes) + [code for code, _ in triggered]))
    return pd.DataFrame({"traffic_light": out_colour, "traffic_light_reasons": out_reasons}, index=frame.index)


def attach(frame: pd.DataFrame, support: pd.DataFrame, id_field: str = "parcel_id") -> pd.DataFrame:
    """`frame` with the ``tq_*`` columns of `support` joined on the parcel id (missing parcels get NaN)."""
    s = support.assign(parcel_id=support.parcel_id.astype(str)).set_index("parcel_id")
    ids = frame[id_field].astype(str) if id_field in frame else frame.index.astype(str)
    out = frame.copy()
    for col in s.columns:
        if col.startswith("tq_"):
            out[col] = s[col].reindex(ids).to_numpy()
    return out
