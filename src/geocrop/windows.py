"""Temporal-window analysis: when, and with which sensors, do crop classes become separable and classifiable?

A generic, season- and crop-agnostic replacement for hand-made window comparisons. Nothing here knows an area, a
year or a crop: windows are named date ranges (explicit, or generated from start/end/length/step/overlap), classes
and variables are whatever the observation table holds.

Input is a long table of *raw* observations, one row per unit (reference polygon), date and variable:
``unit_id, class_name, x, y, date, variable, value`` (``x``/``y`` in a projected CRS in metres, used for spatial
blocks; an optional ``block`` column overrides them). Units are analysed as parcel-level objects, as in
`geocrop.separability`.

For every window the module can compare, on the same units and the same spatial folds,

* ``single``      only observations inside the window,
* ``cumulative``  all observations from the season start to the window end (what an operational in-season
                  classification at that date could see),
* ``optimised``   the start of the window is chosen per window end by Jeffries-Matusita on the *training* units of
                  each fold only (nested), so the held-out units never influence the choice,

with JM separability of chosen class pairs (reusing `separability.pairwise`), spatial-CV performance of S1-only,
S2-only and combined feature groups (reusing the fold plan and estimator of `geocrop.model`), a group permutation
importance, and the *earliest reliable window* under criteria that are frozen in the manifest before any result is
computed.

In-season leakage guard. Every feature set records the date of the latest observation it used; `check_provenance`
raises `LeakageError` when that date is after the window end (the classification date). Temporal smoothing in
in-season mode is `causal_smooth`: it is refitted for each cutoff on the observations up to that cutoff only.
`geocrop.timeseries.whittaker` on a whole-season series is *not* causal (a value at day D depends on observations
after D); it must not feed an in-season feature. See docs/temporal-windows.md for failure modes.

JM and CV on few units are rankings and internal estimates, a recommendation for human review, not a test and not
independent validation.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import __version__, metrics, model, phenofeatures, separability
from . import timeseries as ts
from .estimators import predict_labels
from .hashing import sha256
from .samples import Samples

KINDS = ("single", "cumulative", "optimised")
STATS = ("mean", "std", "min", "max", "median", "smoothed_mean")
FILL = -9999.0          # value written where a unit has no observation in a window (trees; JM ignores it)
OBS_COLUMNS = ("unit_id", "class_name", "date", "variable", "value")
CAVEAT = ("Recommendation for human review, not a definitive conclusion. JM on few units is a ranking, not a "
          "test; cross-validation scores are internal estimates on interpreted reference units, not independent "
          "validation.")


class LeakageError(ValueError):
    """An in-season feature used an observation dated after the classification date."""


# --- windows ------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Window:
    name: str
    start: date
    end: date

    @property
    def days(self) -> int:
        return (self.end - self.start).days + 1

    def as_dict(self) -> dict:
        return {"name": self.name, "start": self.start.isoformat(), "end": self.end.isoformat(), "days": self.days}


def _date(value) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value))
    except ValueError as exc:
        raise ValueError(f"Not an ISO date (YYYY-MM-DD): {value!r}") from exc


def parse_windows(spec) -> list[Window]:
    """Explicit windows from ``{name: {"start", "end"}}`` or ``[{"name", "start", "end"}, ...]`` (ISO dates,
    both ends inclusive; names are arbitrary, windows may overlap or leave gaps)."""
    items = ([{"name": k, **v} for k, v in spec.items()] if isinstance(spec, dict) else list(spec))
    out = []
    for item in items:
        w = Window(str(item["name"]), _date(item["start"]), _date(item["end"]))
        if w.end < w.start:
            raise ValueError(f"Window {w.name!r}: end precedes start")
        out.append(w)
    if len({w.name for w in out}) != len(out):
        raise ValueError("Window names must be unique")
    return out


def generate_windows(start, end, length_days: int, step_days: int | None = None, overlap_days: int | None = None,
                     prefix: str = "W", partial: str = "drop") -> list[Window]:
    """Windows of `length_days` days from `start`, advancing by `step_days` (or length - overlap_days).

    Both ends are inclusive, so a window starting on day s ends on s + length - 1. ``step_days`` defaults to the
    length (adjacent, non-overlapping windows). ``partial`` decides what happens to a last window that would end
    after `end`: ``drop`` (default) or ``clip`` it to `end`. Names are ``<prefix>01``, ``<prefix>02``, ...
    """
    start, end = _date(start), _date(end)
    if end < start:
        raise ValueError("end precedes start")
    if length_days < 1:
        raise ValueError("length_days must be >= 1")
    if overlap_days is not None:
        if not 0 <= overlap_days < length_days:
            raise ValueError("overlap_days must satisfy 0 <= overlap_days < length_days")
        if step_days is not None and step_days != length_days - overlap_days:
            raise ValueError("step_days and overlap_days disagree (step = length - overlap)")
        step_days = length_days - overlap_days
    step_days = length_days if step_days is None else step_days
    if step_days < 1:
        raise ValueError("step_days must be >= 1")
    if partial not in ("drop", "clip"):
        raise ValueError("partial must be 'drop' or 'clip'")
    out, k, s = [], 1, start
    while s <= end:
        e = s + timedelta(days=length_days - 1)
        if e > end:
            if partial == "drop":
                break
            e = end
        out.append(Window(f"{prefix}{k:02d}", s, e))
        k += 1
        s += timedelta(days=step_days)
    if not out:
        raise ValueError("No window fits between start and end")
    return out


def cumulative_windows(windows: list[Window], season_start) -> list[Window]:
    """One window per distinct end date, from the season start to that end (first window name kept)."""
    season_start = _date(season_start)
    out, seen = [], set()
    for w in sorted(windows, key=lambda w: (w.end, w.start)):
        if w.end < season_start:
            raise ValueError(f"Window {w.name!r} ends before the season start")
        if w.end not in seen:
            seen.add(w.end)
            out.append(Window(w.name, season_start, w.end))
    return out


# --- leakage guard ------------------------------------------------------------------------------------------

def check_observation_dates(dates, cutoff, what: str = "observations") -> None:
    """Raise `LeakageError` when any date is after `cutoff`."""
    cutoff = pd.Timestamp(_date(cutoff))
    late = sorted({pd.Timestamp(d) for d in np.asarray(dates).ravel() if pd.Timestamp(d) > cutoff})
    if late:
        raise LeakageError(f"{what}: {len(late)} date(s) after the cutoff {cutoff.date()} "
                           f"(latest {late[-1].date()})")


def check_provenance(used: dict, cutoff) -> None:
    """`used` maps feature name -> latest observation date the feature depends on (None: no observation).

    Raises `LeakageError` naming every feature that depends on data after `cutoff`."""
    cutoff = pd.Timestamp(_date(cutoff))
    bad = {n: pd.Timestamp(d).date().isoformat() for n, d in used.items() if d is not None and pd.Timestamp(d) > cutoff}
    if bad:
        shown = ", ".join(f"{n} (uses {d})" for n, d in list(bad.items())[:5])
        raise LeakageError(f"{len(bad)} feature(s) use observations after the cutoff {cutoff.date()}: {shown}")


def check_feature_names(names, cutoff, window_ends: dict) -> None:
    """Guard for *existing* feature columns (e.g. of a samples file): raise when a name is not provably observable
    at `cutoff`. Reuses the rules of `phenofeatures.feature_sets`: columns of windows ending after the cutoff,
    ``ts_`` columns after it, whole-season (``season_*``, ``harm_*``, ``phen_*``) and unrecognised columns are
    refused. ``window_ends`` maps window name -> end date."""
    cut = _date(cutoff)
    doy = {n: _date(e).timetuple().tm_yday for n, e in window_ends.items()}
    allowed = set(phenofeatures.feature_sets(list(names), [], cutoff=cut.timetuple().tm_yday, windows=doy)["A"])
    bad = [n for n in names if n not in allowed]
    if bad:
        raise LeakageError(f"{len(bad)} feature name(s) not observable by {cut}: {bad[:5]}")


def causal_smooth(obs_dates, values, cutoff, season_start, step_days: int = 5, lam: float = 20.0,
                  envelope_iterations: int = 0, min_valid: int = 4) -> dict:
    """Whittaker smoothing of irregular observations that sees only observations dated in [season_start, cutoff].

    obs_dates: sequence of dates; values: (n_units, n_obs) with NaN for missing. The series is truncated at the
    cutoff *before* binning and smoothing, and the grid ends at the cutoff, so the result at any grid date depends
    on observations <= cutoff only. Calling it again with a later cutoff refits (values near the end of a short
    series are less constrained, which is the honest in-season behaviour). Observations are binned to the grid
    step that contains them (floor), so no observation is moved to a grid date after itself.

    Returns ``{"grid": [date, ...], "z": (n_units, n_grid), "last_observation": date | None}``."""
    cutoff, season_start = _date(cutoff), _date(season_start)
    if cutoff < season_start:
        raise ValueError("cutoff precedes the season start")
    dates = [_date(d) for d in obs_dates]
    v = np.asarray(values, dtype="float64")
    if v.ndim != 2 or v.shape[1] != len(dates):
        raise ValueError("values must be (n_units, n_obs) matching obs_dates")
    keep = np.array([season_start <= d <= cutoff for d in dates], dtype=bool)
    grid = ts.make_grid(season_start, cutoff, step_days)
    y = np.full((v.shape[0], len(grid)), np.nan)
    if keep.any():
        idx = np.array([(d - season_start).days // step_days for d in np.array(dates, dtype=object)[keep]])
        total, count = np.zeros_like(y), np.zeros_like(y)
        vk = v[:, keep]
        ok = np.isfinite(vk)
        for j in range(vk.shape[1]):
            total[ok[:, j], idx[j]] += vk[ok[:, j], j]
            count[ok[:, j], idx[j]] += 1
        with np.errstate(invalid="ignore", divide="ignore"):
            y = np.where(count > 0, total / count, np.nan)
        w = count
    else:
        w = np.zeros_like(y)
    z = ts.whittaker(y, w, lam, 2, envelope_iterations, min_valid=min_valid)
    last = max((d for d, k in zip(dates, keep) if k), default=None)
    return {"grid": grid, "z": z, "last_observation": last}


def causal_smooth_per_cutoff(obs_dates, values, cutoffs, season_start, **options) -> dict:
    """`causal_smooth` refitted for every cutoff: ``{cutoff: result}``."""
    return {_date(c): causal_smooth(obs_dates, values, c, season_start, **options) for c in cutoffs}


# --- observations -------------------------------------------------------------------------------------------

def load_observations(source) -> pd.DataFrame:
    """Observation table from a CSV/Parquet path or a DataFrame, validated and with dates normalised."""
    if isinstance(source, pd.DataFrame):
        df = source.copy()
    elif str(source).lower().endswith((".parquet", ".pq")):
        df = pd.read_parquet(source)
    else:
        df = pd.read_csv(source)
    missing = [c for c in OBS_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Observation table lacks column(s) {missing}; need {list(OBS_COLUMNS)} plus x, y")
    df = df.copy()
    df["unit_id"] = df["unit_id"].astype(str)
    df["class_name"] = df["class_name"].astype(str)
    df["variable"] = df["variable"].astype(str)
    df["date"] = pd.to_datetime(df["date"], errors="raise").dt.tz_localize(None).dt.normalize()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df


def build_units(obs: pd.DataFrame) -> pd.DataFrame:
    """One row per unit (class, coordinates, optional block); a class that varies within a unit is an error."""
    if obs.groupby("unit_id")["class_name"].nunique().gt(1).any():
        raise ValueError("A unit has more than one class_name")
    cols = ["class_name"] + [c for c in ("x", "y", "block") if c in obs.columns]
    units = obs.groupby("unit_id", sort=True)[cols].first().reset_index()
    if not {"x", "y"} <= set(units.columns) and "block" not in units.columns:
        raise ValueError("Spatial cross-validation needs x and y (or block) columns in the observation table")
    return units


# --- configuration ------------------------------------------------------------------------------------------

DEFAULTS = {
    "seed": 42, "kinds": list(KINDS), "stats": ["mean"],
    "smoothing": {"step_days": 5, "lam": 20.0, "envelope_iterations": 0, "min_valid": 4},
    "cv": {"block_m": 2000.0, "n_splits": 5, "crs": "EPSG:3857"},
    "model": {"n_estimators": 100, "max_features": "sqrt", "class_weight": "balanced_subsample", "n_jobs": 1},
    "bootstrap": {"n": 500, "alpha": 0.05},
    "permutation_repeats": 3,
    "optimise": {"min_length_days": 10},
}
CRITERIA_DEFAULTS = {
    "configuration": "combined", "kind": "cumulative", "target_classes": None,
    "min_f1": None, "min_recall": None,
    "jm": {"pairs": [], "min": None},
    "lower_bound": {"metric": "f1", "min": None, "method": "bootstrap"},
    "stability": {"mode": "all_later", "k": 2},
}


def _merge(base: dict, over: dict) -> dict:
    out = json.loads(json.dumps(base))
    for k, v in (over or {}).items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def _sha_json(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode("utf8")).hexdigest()


def _pair(a, b) -> str:
    a, b = sorted((str(a), str(b)))
    return f"{a}-{b}"


def resolve_config(cfg: dict, need_groups: bool = True) -> dict:
    """Complete and validate a configuration (defaults filled, windows resolved, groups/criteria normalised).

    Keys: ``season`` {start,end}; ``windows`` {name: {start,end}} and/or ``generate`` {start,end,length_days,
    step_days,overlap_days,prefix,partial} (start/end default to the season); ``groups`` {S1: [var,...], S2: [...]}
    (variables of the observation table; names are arbitrary); ``stats``; ``smoothing``; ``classes``; ``pairs``;
    ``cv``; ``model``; ``bootstrap``; ``permutation_repeats``; ``optimise``; ``criteria``; ``seed``."""
    rc = _merge(DEFAULTS, {k: v for k, v in cfg.items() if k not in ("criteria",)})
    rc["criteria"] = _merge(CRITERIA_DEFAULTS, cfg.get("criteria") or {})
    if need_groups and not rc.get("groups"):
        raise ValueError("config needs groups, e.g. {S1: [VV, VH], S2: [NDVI]}")
    rc["groups"] = {str(g): [str(v) for v in vs] for g, vs in (rc.get("groups") or {}).items()}
    windows = parse_windows(rc["windows"]) if rc.get("windows") else []
    season = rc.get("season") or {}
    if rc.get("generate"):
        g = dict(rc["generate"])
        windows += generate_windows(g.get("start", season.get("start")), g.get("end", season.get("end")),
                                    int(g["length_days"]), g.get("step_days"), g.get("overlap_days"),
                                    g.get("prefix", "W"), g.get("partial", "drop"))
    if not windows:
        raise ValueError("config needs windows and/or generate")
    if len({w.name for w in windows}) != len(windows):
        raise ValueError("Window names must be unique")
    windows.sort(key=lambda w: (w.end, w.start))
    season_start = _date(season.get("start") or min(w.start for w in windows))
    season_end = _date(season.get("end") or max(w.end for w in windows))
    if any(w.start < season_start or w.end > season_end for w in windows):
        raise ValueError("Every window must lie inside the season")
    rc["season"] = {"start": season_start.isoformat(), "end": season_end.isoformat()}
    rc["windows"] = {w.name: {"start": w.start.isoformat(), "end": w.end.isoformat()} for w in windows}
    rc.pop("generate", None)
    bad = [k for k in rc["kinds"] if k not in KINDS] + [s for s in rc["stats"] if s not in STATS]
    if bad:
        raise ValueError(f"Unknown kind/stat: {bad}; kinds {KINDS}, stats {STATS}")
    c = rc["criteria"]
    if c["kind"] not in KINDS or c["kind"] not in rc["kinds"]:
        raise ValueError("criteria.kind must be one of the analysed kinds")
    if c["lower_bound"]["metric"] not in ("f1", "recall") or c["lower_bound"]["method"] not in ("bootstrap", "fold_min"):
        raise ValueError("criteria.lower_bound: metric f1|recall, method bootstrap|fold_min")
    if c["stability"]["mode"] not in ("all_later", "next_k", "none"):
        raise ValueError("criteria.stability.mode must be all_later, next_k or none")
    c["jm"]["pairs"] = [_pair(*p.split("-", 1)) if isinstance(p, str) else _pair(*p) for p in c["jm"]["pairs"]]
    if rc.get("pairs"):
        rc["pairs"] = sorted({_pair(*p) if not isinstance(p, str) else _pair(*p.split("-", 1)) for p in rc["pairs"]})
    return rc


def criteria_hash(criteria: dict) -> str:
    return _sha_json(criteria)


def _configurations(groups: list[str]) -> dict[str, tuple[str, ...]]:
    """Feature-group configurations to evaluate: each group alone, all together, and (3+ groups) each drop-one."""
    out = {g: (g,) for g in groups}
    if len(groups) > 1:
        out["combined"] = tuple(groups)
    if len(groups) > 2:
        out.update({f"drop_{g}": tuple(h for h in groups if h != g) for g in groups})
    return out


# --- features -----------------------------------------------------------------------------------------------

class _Data:
    """Observations indexed for window queries; every feature records the latest observation date it used."""

    def __init__(self, obs: pd.DataFrame, units: pd.DataFrame, variables: list[str], season_start: date,
                 smoothing: dict, stats=("mean",)):
        self.units, self.smoothing, self.season_start = units, smoothing, season_start
        self.variables, self.stats = list(variables), list(stats)
        index = {u: i for i, u in enumerate(units["unit_id"])}
        obs = obs[obs["unit_id"].isin(index) & obs["variable"].isin(variables) & obs["value"].notna()]
        self.by_var = {v: g.assign(u=g["unit_id"].map(index))[["u", "date", "value"]].reset_index(drop=True)
                       for v, g in obs.groupby("variable")}
        self.n = len(units)
        self._smooth: dict = {}
        self._features: dict = {}
        self.guard_rows: list[dict] = []

    def _select(self, var: str, start: date, end: date) -> pd.DataFrame:
        df = self.by_var.get(var)
        if df is None:
            return pd.DataFrame({"u": [], "date": [], "value": []})
        return df[(df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(end))]

    def _smoothed(self, var: str, cutoff: date):
        key = (var, cutoff)
        if key not in self._smooth:
            sub = self._select(var, self.season_start, cutoff)      # truncated BEFORE smoothing
            dates = sorted(sub["date"].unique())
            mat = np.full((self.n, len(dates)), np.nan)
            if len(dates):
                col = {d: j for j, d in enumerate(dates)}
                mat[sub["u"].to_numpy(int), sub["date"].map(col).to_numpy(int)] = sub["value"].to_numpy()
            r = causal_smooth([pd.Timestamp(d).date() for d in dates], mat, cutoff, self.season_start,
                              **self.smoothing)
            self._smooth[key] = r
        return self._smooth[key]

    def features(self, start: date, end: date) -> tuple[pd.DataFrame, dict]:
        """Unit x (variable__stat) features of the window [start, end] and their provenance; guarded at `end`."""
        key = (start, end)
        if key in self._features:
            return self._features[key]
        cols, used = {}, {}
        for var in self.variables:
            sub = self._select(var, start, end)
            last = pd.Timestamp(sub["date"].max()).date() if len(sub) else None
            grouped = sub.groupby("u")["value"]
            for stat in self.stats:
                name = f"{var}__{stat}"
                if stat == "smoothed_mean":
                    r = self._smoothed(var, end)
                    inside = np.array([start <= g <= end for g in r["grid"]])
                    with np.errstate(all="ignore"):
                        cols[name] = (np.nanmean(r["z"][:, inside], axis=1) if inside.any()
                                      else np.full(self.n, np.nan))
                    used[name] = r["last_observation"]
                else:
                    agg = {"mean": grouped.mean, "std": lambda g=grouped: g.std(ddof=0), "min": grouped.min,
                           "max": grouped.max, "median": grouped.median}[stat]()
                    cols[name] = agg.reindex(range(self.n)).to_numpy(float)
                    used[name] = last
        check_provenance(used, end)
        latest = max((d for d in used.values() if d is not None), default=None)
        self.guard_rows.append({"start": start.isoformat(), "end": end.isoformat(), "cutoff": end.isoformat(),
                                "features": len(cols), "latest_observation_used": latest.isoformat() if latest else "",
                                "ok": True})
        self._features[key] = (pd.DataFrame(cols), used)
        return self._features[key]


def window_features(obs, units, variables, window: Window, stats=("mean",), season_start=None,
                    smoothing: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """Features of one window (unit x ``variable__stat``) and their provenance (latest observation date used).

    Public wrapper around the machinery used by `evaluate`; ``season_start`` only matters for ``smoothed_mean``."""
    data = _Data(load_observations(obs), units, variables, _date(season_start or window.start),
                 {**DEFAULTS["smoothing"], **(smoothing or {})}, stats)
    return data.features(window.start, window.end)


# --- metrics ------------------------------------------------------------------------------------------------

def _prf(cm: np.ndarray):
    """Per-class precision, recall, F1 from a confusion matrix (rows reference); NaN where undefined."""
    tp = np.diag(cm).astype(float)
    ref, pred = cm.sum(axis=1), cm.sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return (np.where(pred > 0, tp / pred, np.nan), np.where(ref > 0, tp / ref, np.nan),
                np.where(ref + pred > 0, 2 * tp / (ref + pred), np.nan))


def _bootstrap_lower(y_idx, p_idx, k: int, draws: np.ndarray, alpha: float):
    """alpha-quantile of per-class (f1, recall) over unit bootstrap resamples (`draws`: B x n indices)."""
    f1s, recs = [], []
    for idx in draws:
        cm = np.bincount(y_idx[idx] * k + p_idx[idx], minlength=k * k).reshape(k, k)
        _, rec, f1 = _prf(cm)
        f1s.append(f1)
        recs.append(rec)
    with np.errstate(all="ignore"):
        return np.nanquantile(np.array(f1s), alpha, axis=0), np.nanquantile(np.array(recs), alpha, axis=0)


# --- evaluation ---------------------------------------------------------------------------------------------

def _oof(get_x, y, plan, params, seed, perm_groups=None, repeats=0):
    """Out-of-fold labels (and, per feature group, labels with that group's columns permuted in the test fold)."""
    oof = np.empty(len(y), dtype=object)
    perm = {g: [np.empty(len(y), dtype=object) for _ in range(repeats)] for g in (perm_groups or {})}
    for k, (tr, te, _) in enumerate(plan):
        X = get_x(k, tr)
        rf = model.make_rf(params, seed + k).fit(X[tr], y[tr])
        oof[te] = predict_labels(rf, X[te])
        for g, cols in (perm_groups or {}).items():
            for r in range(repeats):
                rng = np.random.default_rng([seed, k, r])
                Xp = X[te].copy()
                Xp[:, cols] = Xp[rng.permutation(len(te))][:, cols]    # rows shuffled jointly: keeps within-group structure
                perm[g][r][te] = predict_labels(rf, Xp)
    return oof.astype(str), {g: [p.astype(str) for p in ps] for g, ps in perm.items()}


def _jm_pairs(frame: pd.DataFrame, y, idx, pairs, classes) -> pd.DataFrame:
    means = frame.iloc[idx].copy()
    means.insert(0, "class_name", np.asarray(y)[idx])
    means.insert(0, "polygon_id", [str(i) for i in idx])
    means = means.replace(FILL, np.nan)
    table = separability.pairwise(means, classes)
    return table[table["pair"].isin(pairs)]


def _score(table: pd.DataFrame) -> float:
    """Selection objective: mean over pairs of the best feature's JM (NaN pairs count as 0)."""
    if table.empty:
        return float("nan")
    best = table.groupby("pair")["jm"].max().fillna(0.0)
    return float(best.mean())


def evaluate(observations, config: dict, out: str | Path | None = None) -> dict:
    """Run the whole analysis. Returns tables (DataFrames), the earliest-reliable result and the manifest.

    With `out`, the resolved configuration and the evaluation criteria are written to ``manifest.json`` *before*
    any result is computed (status ``criteria_registered``), then tables, plots and the final manifest follow."""
    rc = resolve_config(config)
    seed = int(rc["seed"])
    crit = rc["criteria"]
    reg_hash = criteria_hash(crit)
    out = Path(out) if out is not None else None
    manifest = {"schema": "geocrop-windows/1", "geocrop_version": __version__, "status": "criteria_registered",
                "registered_utc": datetime.now(timezone.utc).isoformat(), "seed": seed,
                "config_sha256": _sha_json(rc), "criteria": crit, "criteria_sha256": reg_hash,
                "python": sys.version.split()[0], "packages": _versions(), "caveat": CAVEAT,
                "inputs": ([{"path": str(observations), "sha256": sha256(observations)}]
                           if isinstance(observations, (str, Path)) else [])}
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        (out / "resolved_config.json").write_text(json.dumps(rc, indent=2), encoding="utf8")
        (out / "criteria.json").write_text(json.dumps(crit, indent=2), encoding="utf8")
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf8")

    windows = parse_windows(rc["windows"])
    season_start, season_end = _date(rc["season"]["start"]), _date(rc["season"]["end"])
    obs = load_observations(observations)
    n_obs_in = len(obs)
    obs = obs[(obs["date"] >= pd.Timestamp(season_start)) & (obs["date"] <= pd.Timestamp(season_end))]
    classes = sorted(rc.get("classes") or obs["class_name"].unique())
    obs = obs[obs["class_name"].isin(classes)]
    units = build_units(obs)
    y = units["class_name"].to_numpy(str)
    n_per_class = pd.Series(y).value_counts().to_dict()
    pairs = rc.get("pairs") or [_pair(a, b) for a, b in itertools.combinations(classes, 2)]
    targets = sorted(crit["target_classes"] or classes)
    groups = rc["groups"]
    variables = sorted({v for vs in groups.values() for v in vs})
    var_group = {v: g for g, vs in groups.items() for v in vs}
    configs = _configurations(list(groups))

    data = _Data(obs, units, variables, season_start, rc["smoothing"], rc["stats"])

    # one spatial fold plan for every window, kind and configuration
    cv = rc["cv"]
    if "block" in units.columns:
        blocks = units["block"].astype(str).to_numpy()
    else:
        dummy = Samples(np.zeros((len(units), 1), "float32"), y, units["unit_id"].to_numpy(str),
                        units["x"].to_numpy(float), units["y"].to_numpy(float), ["_"], cv["crs"])
        blocks = model.spatial_blocks(dummy, cv["block_m"])
    plan = model._fold_plan(y, blocks, int(cv["n_splits"]), seed, polygon=units["unit_id"].to_numpy(str))
    signature = model.fold_plan_signature([te for _, te, _ in plan])
    params = dict(rc["model"])
    k = len(classes)
    y_idx = np.searchsorted(classes, y)
    n_boot = int(rc["bootstrap"]["n"])
    alpha = float(rc["bootstrap"]["alpha"])
    draws = np.random.default_rng(seed).integers(0, len(y), size=(n_boot, len(y)))

    cum = cumulative_windows(windows, season_start)
    by_kind = {"single": windows, "cumulative": cum, "optimised": cum}
    ends = sorted({w.end for w in windows})
    starts = sorted({season_start, *(w.start for w in windows)})
    min_len = int(rc["optimise"]["min_length_days"])
    jm_rows, perf_rows, class_rows, contrib_rows, opt_rows = [], [], [], [], []

    def cols_of(config):
        gs = configs[config]
        return [f"{v}__{s}" for v in variables if var_group[v] in gs for s in rc["stats"]]

    for kind in rc["kinds"]:
        for w in by_kind[kind]:
            if kind in ("single", "cumulative"):
                F, _ = data.features(w.start, w.end)
                fixed = F.fillna(FILL).to_numpy(float)
                for pr in _jm_pairs(F, y, np.arange(len(y)), pairs, classes).itertuples():
                    feat = pr.feature
                    jm_rows.append({"kind": kind, "window": w.name, "start": w.start, "end": w.end, "feature": feat,
                                    "group": var_group[feat.split("__")[0]], "pair": pr.pair, "jm": pr.jm,
                                    "n_a": pr.n_a, "n_b": pr.n_b})
            for cname, gs in configs.items():
                cols = cols_of(cname)
                if kind == "optimised":
                    chosen = {}
                    cand = [s for s in starts if (w.end - s).days + 1 >= min_len]

                    def get_x(fold, tr, w=w, cols=cols, cand=cand, chosen=chosen):
                        best, best_s = -np.inf, cand[0]
                        for s in cand:                       # earliest start wins ties: iterate ascending, strict >
                            Fc, _ = data.features(s, w.end)
                            sc = _score(_jm_pairs(Fc[cols], y, tr, pairs, classes))
                            if np.isfinite(sc) and sc > best:
                                best, best_s = sc, s
                        chosen[fold] = (best_s, best)
                        return data.features(best_s, w.end)[0][cols].fillna(FILL).to_numpy(float)
                else:
                    X = fixed[:, [list(F.columns).index(c) for c in cols]]

                    def get_x(fold, tr, X=X):
                        return X
                perm_groups = None
                reps = int(rc["permutation_repeats"])
                if cname == "combined" and reps:
                    perm_groups = {g: [i for i, c in enumerate(cols) if var_group[c.split("__")[0]] == g]
                                   for g in gs}
                oof, perm = _oof(get_x, y, plan, params, seed, perm_groups, reps)
                if kind == "optimised":
                    for fold, (s, sc) in sorted(chosen.items()):
                        opt_rows.append({"configuration": cname, "window": w.name, "end": w.end, "fold": fold,
                                         "start": s, "train_objective_jm": sc})
                p_idx = np.searchsorted(classes, oof)
                cm = np.bincount(y_idx * k + p_idx, minlength=k * k).reshape(k, k)
                pooled = metrics.from_confusion(cm, classes)
                lo_f1, lo_rec = _bootstrap_lower(y_idx, p_idx, k, draws, alpha)
                fold_f1, fold_rec = [], []
                for _, te, _ in plan:
                    fcm = np.bincount(y_idx[te] * k + p_idx[te], minlength=k * k).reshape(k, k)
                    _, rec, f1 = _prf(fcm)
                    fold_f1.append(f1)
                    fold_rec.append(rec)
                with np.errstate(all="ignore"):
                    fmin_f1, fmin_rec = np.nanmin(fold_f1, axis=0), np.nanmin(fold_rec, axis=0)
                missing = float((fixed == FILL).mean()) if kind != "optimised" else float("nan")
                row = {"configuration": cname, "kind": kind, "window": w.name, "start": w.start, "end": w.end,
                       "features": len(cols), "n_units": len(y), "overall_accuracy": pooled["overall_accuracy"],
                       "macro_f1": pooled["macro_f1"], "missing_share": missing, "fold_signature": signature}
                if perm_groups:
                    for g, ps in perm.items():
                        drops = [pooled["macro_f1"] - metrics.summary(y, p, labels=classes)["macro_f1"] for p in ps]
                        row[f"perm_drop_macro_f1_{g}"] = float(np.mean(drops))
                perf_rows.append(row)
                for i, c in enumerate(classes):
                    pc = pooled["per_class"][c]
                    class_rows.append({"configuration": cname, "kind": kind, "window": w.name, "end": w.end,
                                       "class_name": c, "n_units": int(n_per_class[c]), "f1": pc["f1"],
                                       "recall": pc["producers_accuracy"], "precision": pc["users_accuracy"],
                                       "f1_lower": _f(lo_f1[i]), "recall_lower": _f(lo_rec[i]),
                                       "f1_fold_min": _f(fmin_f1[i]), "recall_fold_min": _f(fmin_rec[i])})

    jm = pd.DataFrame(jm_rows)
    perf = pd.DataFrame(perf_rows)
    by_class = pd.DataFrame(class_rows)
    jm_best = _jm_best(jm, configs, groups)
    contribution = _contribution(perf, configs, groups)
    earliest = earliest_reliable(by_class, jm_best, crit, [w.name for w in cum], {w.name: w.end for w in cum})
    if earliest["criteria_sha256"] != reg_hash:
        raise RuntimeError("Criteria changed after registration")
    guard = pd.DataFrame(data.guard_rows).drop_duplicates(["start", "end"]).sort_values(["end", "start"])
    result = {"config": rc, "windows": pd.DataFrame([w.as_dict() for w in windows]), "jm": jm, "jm_best": jm_best,
              "performance": perf, "performance_by_class": by_class, "contribution": contribution,
              "optimised_selection": pd.DataFrame(opt_rows), "leakage_guard": guard, "earliest": earliest,
              "fold_signature": signature,
              "summary": {"units_per_class": n_per_class, "observations_in_season": int(len(obs)),
                          "observations_outside_season_or_classes": int(n_obs_in - len(obs)),
                          "folds": len(plan), "fold_signature": signature, "earliest_reliable": earliest["window"],
                          "caveat": CAVEAT}}
    if out is not None:
        export(result, out, manifest)
    return result


def _f(x):
    return None if x is None or not np.isfinite(x) else float(x)


def _versions() -> dict:
    import sklearn
    return {"numpy": np.__version__, "pandas": pd.__version__, "scikit-learn": sklearn.__version__}


def _jm_best(jm: pd.DataFrame, configs: dict, groups: dict) -> pd.DataFrame:
    """Best-feature JM per (configuration, kind, window, pair)."""
    if jm.empty:
        return jm
    rows = []
    for cname, gs in configs.items():
        sub = jm[jm["group"].isin(gs)].dropna(subset=["jm"])
        if sub.empty:
            continue
        best = sub.loc[sub.groupby(["kind", "window", "pair"])["jm"].idxmax()].copy()
        best.insert(0, "configuration", cname)
        rows.append(best.rename(columns={"feature": "best_feature"}).drop(columns=["group"]))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _contribution(perf: pd.DataFrame, configs: dict, groups: dict) -> pd.DataFrame:
    """Drop-group contribution: macro F1 of the combined set minus the set without each group (same folds)."""
    if "combined" not in configs:
        return pd.DataFrame()
    rows = []
    for (kind, window), g in perf.groupby(["kind", "window"], sort=False):
        score = g.set_index("configuration")["macro_f1"]
        for grp in groups:
            rest = [c for c, gs in configs.items() if set(gs) == set(groups) - {grp}]
            row = {"kind": kind, "window": window, "end": g["end"].iloc[0], "group": grp,
                   "macro_f1_combined": score["combined"], "macro_f1_without_group": score[rest[0]] if rest else np.nan,
                   "macro_f1_group_only": score.get(grp, np.nan)}
            row["drop_group_gain"] = row["macro_f1_combined"] - row["macro_f1_without_group"]
            col = f"perm_drop_macro_f1_{grp}"
            row["permutation_drop"] = g.loc[g["configuration"] == "combined", col].iloc[0] if col in g else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


# --- earliest reliable window -------------------------------------------------------------------------------

def earliest_reliable(by_class: pd.DataFrame, jm_best: pd.DataFrame, criteria: dict, order: list[str],
                      ends: dict) -> dict:
    """Earliest cumulative window meeting criteria fixed beforehand, or None.

    Per window every enabled criterion is checked for the configured configuration and kind: class-wise F1 and
    recall of the target classes, the lower bound of the chosen metric (bootstrap quantile or minimum over
    folds), and JM >= min for each listed pair (best feature of the configuration). Stability: with
    ``all_later`` the window counts only if it and every later window meet them; with ``next_k`` it and the next
    k windows (fewer than k later windows left: not reliable). ``none`` checks the window alone."""
    c = criteria
    sub = by_class[(by_class["configuration"] == c["configuration"]) & (by_class["kind"] == c["kind"])]
    targets = c["target_classes"] or sorted(sub["class_name"].unique())
    lb = c["lower_bound"]
    lb_col = {("f1", "bootstrap"): "f1_lower", ("recall", "bootstrap"): "recall_lower",
              ("f1", "fold_min"): "f1_fold_min", ("recall", "fold_min"): "recall_fold_min"}[(lb["metric"], lb["method"])]
    rows = []
    for name in order:
        w = sub[sub["window"] == name].set_index("class_name")
        flags, vals = {}, {}
        for label, col, thr in (("f1", "f1", c["min_f1"]), ("recall", "recall", c["min_recall"]),
                                (f"{lb['metric']}_{lb['method']}_lower", lb_col, lb["min"])):
            if thr is None:
                continue
            v = [w[col].get(t) if t in w.index else None for t in targets]
            ok = all(x is not None and np.isfinite(x) for x in v)
            vals[label] = float(min(v)) if ok else np.nan      # the weakest target class decides
            flags[label] = bool(np.isfinite(vals[label]) and vals[label] >= thr)
        if c["jm"]["min"] is not None and c["jm"]["pairs"]:
            j = jm_best[(jm_best["configuration"] == c["configuration"]) & (jm_best["kind"] == c["kind"])
                        & (jm_best["window"] == name)].set_index("pair")["jm"] if len(jm_best) else pd.Series(dtype=float)
            v = [j.get(p, np.nan) for p in c["jm"]["pairs"]]
            vals["jm_min_over_pairs"] = float(np.min(v)) if np.isfinite(v).all() else np.nan
            flags["jm"] = bool(np.isfinite(vals["jm_min_over_pairs"]) and vals["jm_min_over_pairs"] >= c["jm"]["min"])
        rows.append({"window": name, "end": ends[name], **{f"value_{k}": v for k, v in vals.items()},
                     **{f"meets_{k}": v for k, v in flags.items()}, "meets_all": bool(flags) and all(flags.values())})
    table = pd.DataFrame(rows)
    meets = table["meets_all"].to_numpy(bool) if len(table) else np.array([], bool)
    stab = c["stability"]
    reliable = np.zeros(len(meets), bool)
    for i in range(len(meets)):
        if stab["mode"] == "all_later":
            reliable[i] = meets[i:].all()
        elif stab["mode"] == "next_k":
            reliable[i] = i + int(stab["k"]) < len(meets) and meets[i:i + int(stab["k"]) + 1].all()
        else:
            reliable[i] = meets[i]
    table["reliable"] = reliable
    hit = table[table["reliable"]]
    return {"window": None if hit.empty else hit.iloc[0]["window"],
            "end": None if hit.empty else str(hit.iloc[0]["end"]), "criteria": c,
            "criteria_sha256": criteria_hash(c), "table": table,
            "note": ("No window meets the criteria with the required stability." if hit.empty else
                     "First window meeting the pre-registered criteria; internal CV, a recommendation for human review.")}


# --- exports ------------------------------------------------------------------------------------------------

def to_markdown(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    """Pipe table without optional dependencies."""
    if df.empty:
        return "(empty)\n"
    def cell(v):
        if isinstance(v, (float, np.floating)):
            return "" if not np.isfinite(v) else floatfmt.format(v)
        return str(v)
    head = "| " + " | ".join(map(str, df.columns)) + " |\n|" + "|".join("---" for _ in df.columns) + "|\n"
    return head + "".join("| " + " | ".join(cell(v) for v in r) + " |\n" for r in df.itertuples(index=False))


def export(result: dict, out: str | Path, manifest: dict | None = None) -> dict:
    """Write CSV + markdown tables, PNG plots and the final manifest into `out`."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    tables = {"windows": result["windows"], "jm_by_feature": result["jm"], "jm_best": result["jm_best"],
              "performance": result["performance"], "performance_by_class": result["performance_by_class"],
              "contribution": result["contribution"], "optimised_selection": result["optimised_selection"],
              "earliest_reliable": result["earliest"]["table"], "leakage_guard": result["leakage_guard"]}
    for name, df in tables.items():
        df.to_csv(out / f"{name}.csv", index=False)
        (out / f"{name}.md").write_text(f"# {name}\n\n{CAVEAT}\n\n" + to_markdown(df), encoding="utf8")
    plots = plot(result, out)
    registered = json.loads((out / "manifest.json").read_text(encoding="utf8")) if (out / "manifest.json").exists() \
        else (manifest or {})
    if registered.get("criteria_sha256") != result["earliest"]["criteria_sha256"]:
        raise RuntimeError("Criteria in the manifest differ from the criteria used for the evaluation")
    outputs = sorted(p.name for p in out.iterdir() if p.is_file() and p.name != "manifest.json")
    registered.update({"status": "completed", "completed_utc": datetime.now(timezone.utc).isoformat(),
                       "fold_signature": result["fold_signature"], "earliest_reliable": result["earliest"]["window"],
                       "summary": result["summary"], "plots": plots,
                       "outputs": {n: sha256(out / n) for n in outputs}})
    (out / "manifest.json").write_text(json.dumps(registered, indent=2, default=str), encoding="utf8")
    return registered


_PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]   # Okabe-Ito, colour-blind safe


def plot(result: dict, out: str | Path) -> list[str]:
    """PNG figures (needs matplotlib, otherwise returns []): JM by window end, cumulative performance, groups."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    out = Path(out)
    written = []
    jm, perf, by_class = result["jm_best"], result["performance"], result["performance_by_class"]
    configs = list(dict.fromkeys(perf["configuration"])) if len(perf) else []
    colour = {c: _PALETTE[i % len(_PALETTE)] for i, c in enumerate(configs)}
    style = {"single": ":", "cumulative": "-", "optimised": "--"}

    if len(jm):
        pairs = sorted(jm["pair"].unique())
        fig, axes = plt.subplots(1, len(pairs), figsize=(4.6 * len(pairs), 3.6), squeeze=False, sharey=True)
        for ax, pair in zip(axes[0], pairs):
            for (cfg, kind), g in jm[jm["pair"] == pair].groupby(["configuration", "kind"]):
                g = g.sort_values("end")
                ax.plot(pd.to_datetime(g["end"]), g["jm"], style[kind], color=colour[cfg], marker="o", ms=3, label=f"{cfg} {kind}")
            if result["config"]["criteria"]["jm"]["min"]:
                ax.axhline(result["config"]["criteria"]["jm"]["min"], color="grey", lw=0.8, ls="-.")
            ax.set(title=pair, xlabel="window end", ylim=(0, 2.05))
            ax.tick_params(axis="x", rotation=45)
        axes[0][0].set_ylabel("JM (best feature)")
        axes[0][-1].legend(fontsize=7, frameon=False)
        fig.tight_layout()
        fig.savefig(out / "jm_by_window.png", dpi=150)
        plt.close(fig)
        written.append("jm_by_window.png")

    if len(perf):
        crit = result["config"]["criteria"]
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8))
        for (cfg, kind), g in perf.groupby(["configuration", "kind"]):
            g = g.sort_values("end")
            a1.plot(pd.to_datetime(g["end"]), g["macro_f1"], style[kind], color=colour[cfg], marker="o", ms=3, label=f"{cfg} {kind}")
        a1.set(title="Macro F1 by window end (spatial CV)", xlabel="window end", ylabel="macro F1", ylim=(0, 1.02))
        a1.tick_params(axis="x", rotation=45)
        a1.legend(fontsize=7, frameon=False)
        sel = by_class[(by_class["configuration"] == crit["configuration"]) & (by_class["kind"] == crit["kind"])]
        lb = {"f1": "f1_lower", "recall": "recall_lower"}[crit["lower_bound"]["metric"]] \
            if crit["lower_bound"]["method"] == "bootstrap" else f"{crit['lower_bound']['metric']}_fold_min"
        for i, (cls, g) in enumerate(sel.groupby("class_name")):
            g = g.sort_values("end")
            x = pd.to_datetime(g["end"])
            a2.plot(x, g["f1"], color=_PALETTE[i % 6], marker="o", ms=3, label=cls)
            a2.plot(x, g[lb], color=_PALETTE[i % 6], ls=":", lw=1)
        if crit["min_f1"]:
            a2.axhline(crit["min_f1"], color="grey", lw=0.8, ls="-.")
        e = result["earliest"]["end"]
        if e:
            a2.axvline(pd.Timestamp(e), color="black", lw=0.8)
        a2.set(title=f"Class F1 ({crit['configuration']}, {crit['kind']}); dotted = lower bound", xlabel="window end", ylim=(0, 1.02))
        a2.tick_params(axis="x", rotation=45)
        a2.legend(fontsize=7, frameon=False)
        fig.tight_layout()
        fig.savefig(out / "performance_by_window.png", dpi=150)
        plt.close(fig)
        written.append("performance_by_window.png")

        cum = perf[(perf["kind"] == crit["kind"])]
        if len(cum):
            wide = cum.pivot(index="end", columns="configuration", values="macro_f1").sort_index()
            fig, ax = plt.subplots(figsize=(1.0 * len(wide) * max(1, len(wide.columns)) / 2 + 3, 3.6))
            wide.plot.bar(ax=ax, color=[colour[c] for c in wide.columns], width=0.8)
            ax.set(title=f"Contribution of feature groups ({crit['kind']} windows)", ylabel="macro F1", xlabel="window end",
                   ylim=(0, 1.02))
            ax.set_xticklabels([pd.Timestamp(i).strftime("%m-%d") for i in wide.index], rotation=45)
            ax.legend(fontsize=7, frameon=False)
            fig.tight_layout()
            fig.savefig(out / "group_contribution.png", dpi=150)
            plt.close(fig)
            written.append("group_contribution.png")
    return written


def windows_yaml(windows: list[Window], title: str = "windows") -> str:
    """`windows:` block for the engine configuration (feeds `geocrop features` / `run`, see docs)."""
    lines = [f"# {title}", "windows:"]
    lines += [f'  {w.name}: {{start: "{w.start.isoformat()}", end: "{w.end.isoformat()}"}}' for w in windows]
    return "\n".join(lines) + "\n"
