"""Phenology-derived and time-aligned features from the per-pixel time-series columns of a sample set.

The sample files written by the ``samples`` stage carry ``ts_<index>_d<doy>`` columns (a smoothed curve per index
on a regular day-of-year grid). From them this module derives, per index:

* ``ph_<idx>_*``  season metrics (start, peak day, end, length, peak value, green-up and senescence rate,
  integral) from :func:`geocrop.timeseries.phenology`;
* ``rel_<idx>_*`` curve values re-expressed in *relative* phenological time: normalised to the season amplitude and
  read at fixed offsets from the start of season (``sos``) and, when the season is complete, from the peak
  (``pk``). Two pixels whose development is shifted by a few weeks then get similar values.

Post-season and in-season sets differ in what they may look at. With ``cutoff`` the curves are truncated at that
day of year before any feature is derived, so nothing after the cutoff enters: the end of season, season length and
senescence are then undefined and are not produced; the peak is the peak so far. Caveat: the smoothing that
produced the ``ts_`` columns saw the whole season, so an in-season set built here is an optimistic proxy; a strictly
causal test recomputes the curves with the season ended at the cutoff. Undefined values are filled with ``-1``.

Descriptive, label-free, and a recommendation for human review, not a validated operational model.
"""
from __future__ import annotations

import re

import numpy as np

from . import timeseries as ts

SOS_OFFSETS = (0, 15, 30, 45, 60, 90)
PEAK_OFFSETS = (-60, -45, -30, -15, 15, 30, 45, 60)
FILL = -1.0
_TS = re.compile(r"^ts_(?P<idx>.+)_d(?P<doy>\d+)$")


def series_columns(names: list[str], index: str) -> tuple[list[int], np.ndarray]:
    """Column positions and day-of-year of the ``ts_<index>_d<doy>`` columns, sorted by day."""
    found = []
    for i, n in enumerate(names):
        m = _TS.match(n)
        if m and m["idx"] == index:
            found.append((int(m["doy"]), i))
    found.sort()
    return [i for _, i in found], np.array([d for d, _ in found], dtype=float)


def _at(z: np.ndarray, days: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Linear interpolation of every row of ``z`` at per-row days ``t``; values outside the curve hold the edge."""
    t = np.clip(t, days[0], days[-1])
    j = np.clip(np.searchsorted(days, t, side="right") - 1, 0, len(days) - 2)
    rows = np.arange(len(z))
    w = (t - days[j]) / (days[j + 1] - days[j])
    return z[rows, j] * (1 - w) + z[rows, j + 1] * w


def derive(z: np.ndarray, days: np.ndarray, index: str, cutoff: float | None = None, threshold: float = 0.5,
           min_amplitude: float = 0.1, sos_offsets=SOS_OFFSETS, peak_offsets=PEAK_OFFSETS,
           fill: float = FILL) -> dict[str, np.ndarray]:
    """Feature columns for one index from curves ``z`` (n, t) at ``days``; ``cutoff`` truncates the curve first.

    ``threshold`` is the share of the amplitude that defines start and end, ``min_amplitude`` the smallest season
    amplitude that counts as a season (below it the metrics are undefined), ``sos_offsets`` / ``peak_offsets`` the
    days from start / peak at which the normalised curve is read, ``fill`` the value written where undefined.
    """
    z = np.asarray(z, dtype="float64")
    days = np.asarray(days, dtype="float64")
    if cutoff is not None:
        keep = days <= cutoff
        z, days = z[:, keep], days[keep]
    if len(days) < 4:
        raise ValueError(f"{index}: fewer than 4 time steps up to the cutoff")
    p = ts.phenology(z, days, threshold, min_amplitude)
    out: dict[str, np.ndarray] = {f"ph_{index}_sos": p["sos"], f"ph_{index}_peak_doy": p["peak_doy"],
                                  f"ph_{index}_peak": p["peak_value"], f"ph_{index}_amplitude": p["amplitude"],
                                  f"ph_{index}_greenup_rate": p["greenup_rate"], f"ph_{index}_integral": p["integral"]}
    post = cutoff is None
    if post:
        out.update({f"ph_{index}_eos": p["eos"], f"ph_{index}_gsl": p["los"],
                    f"ph_{index}_senescence_rate": p["senescence_rate"]})
    base = p["peak_value"] - p["amplitude"]
    amp = np.where(p["amplitude"] > 1e-6, p["amplitude"], np.nan)
    norm = (z - base[:, None]) / amp[:, None]
    for off in sos_offsets:
        out[f"rel_{index}_sos{off:+d}"] = _at(norm, days, p["sos"] + off)
    if post:
        for off in peak_offsets:
            out[f"rel_{index}_pk{off:+d}"] = _at(norm, days, p["peak_doy"] + off)
    return {k: np.where(np.isfinite(v), v, fill) for k, v in out.items()}


def add_features(X: np.ndarray, names: list[str], indices=("NDVI", "NDRE"), cutoff: float | None = None, **options):
    """Samples matrix plus derived columns -> (X_new, names_new, new_names); cutoff columns end in ``_c<doy>``."""
    cols, new = [], []
    tag = "" if cutoff is None else f"_c{int(cutoff)}"
    for idx in indices:
        pos, days = series_columns(names, idx)
        if len(pos) < 4:
            raise ValueError(f"no usable ts_{idx}_d* columns in the samples")
        for k, v in derive(X[:, pos], days, idx, cutoff, **options).items():
            cols.append(v.astype("float32"))
            new.append(k + tag)
    return np.column_stack([X, *cols]).astype("float32"), list(names) + new, new


def feature_sets(names: list[str], derived: list[str], cutoff: float | None = None,
                 windows: dict[str, float] | None = None) -> dict[str, list[str]]:
    """Reference (A), + season metrics (B) and + relative profiles (C) sets.

    Without ``cutoff`` A is every original column. With a cutoff A keeps only what is observable by then: window
    columns (``P1_`` ...) of windows ending on or before it, ``ts_`` columns up to it and the deltas between those
    windows; whole-season columns (``season_``, ``harm_``) are dropped. ``windows`` maps window name to end day.
    """
    orig = [n for n in names if n not in set(derived)]
    if cutoff is not None:
        ok = {w for w, end in (windows or {}).items() if end <= cutoff}

        def keep(n: str) -> bool:
            if n.startswith(("season_", "harm_")):
                return False
            m = _TS.match(n)
            if m:
                return int(m["doy"]) <= cutoff
            if n.startswith("delta_"):
                return all(w in ok for w in re.findall(r"P\d", n))
            m = re.match(r"^(P\d)_", n)
            return m is None or m[1] in ok

        orig = [n for n in orig if keep(n)]
    mine = [n for n in derived if n.endswith(f"_c{int(cutoff)}")] if cutoff is not None else \
        [n for n in derived if not re.search(r"_c\d+$", n)]
    return {"A": orig, "B": orig + [n for n in mine if n.startswith("ph_")],
            "C": orig + [n for n in mine if n.startswith("rel_")]}
