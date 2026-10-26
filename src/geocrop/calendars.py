"""Crop calendars: data-driven derivation from training-parcel signatures and external priors.

`derive`            per-class percentiles of phenological events (SOS, peak, EOS, harvest) from parcel markers
                    (`geocrop.cbm.markers`) of labelled parcels
`to_calendar`       the expected-range calendar used by `geocrop.cbm.calendar_consistency`
                    ({class: {"peak": [lo, hi], "harvest": [lo, hi], "sos": ..., "eos": ...}})
`suggest_windows`   phenological compositing windows whose boundaries follow the class-median events, as a
                    ``windows:`` block for the configuration
`worldcereal_priors` SOS/EOS day of year read from the WorldCereal global crop-calendar rasters at a location
                    (files obtained by `geocrop.public_data.worldcereal_calendar_files`)

A derived calendar describes the training parcels of one region and season; it replaces hand-written
ranges with reproducible ones but inherits every bias of the labels and of the markers (e.g. NDVI cannot
separate harvest from drought senescence). Calendars derived from CC-BY-SA data inherit the ShareAlike terms.
"""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

EVENTS = {"sos": "sos_doy", "peak": "peak_doy", "eos": "eos_doy", "harvest": "harvest_doy"}


def derive(markers_df: pd.DataFrame, labels, percentiles=(10, 50, 90), min_parcels: int = 5,
           id_field: str = "parcel_id") -> dict:
    """Per-class percentiles of marker days of year.

    labels: mapping or Series parcel_id -> class, or a frame with columns ``parcel_id`` and ``class_name``.
    Classes with fewer than `min_parcels` finite values of an event get ``None`` for that event.
    """
    if isinstance(labels, pd.DataFrame):
        labels = labels.set_index(id_field)["class_name"]
    labels = pd.Series(labels, dtype=object)
    labels.index = labels.index.astype(str)
    frame = markers_df.copy()
    frame[id_field] = frame[id_field].astype(str)
    frame["class_name"] = frame[id_field].map(labels)
    frame = frame[frame["class_name"].notna()]
    if frame.empty:
        raise ValueError("No marker rows match the labelled parcels")
    out = {}
    for cls, group in frame.groupby("class_name"):
        entry = {"n_parcels": int(len(group))}
        for event, column in EVENTS.items():
            values = group[column].to_numpy(dtype=float) if column in group else np.array([])
            values = values[np.isfinite(values)]
            if len(values) < min_parcels:
                entry[event] = None
                continue
            entry[event] = {"n": int(len(values)),
                            **{f"p{p:g}": float(np.percentile(values, p)) for p in percentiles}}
        out[str(cls)] = entry
    return {"percentiles": list(percentiles), "min_parcels": min_parcels, "classes": out}


def to_calendar(derived: dict, low: float | None = None, high: float | None = None) -> dict:
    """Expected ranges [p_low, p_high] (rounded outward to whole days) per class and event."""
    pct = derived["percentiles"]
    low = pct[0] if low is None else low
    high = pct[-1] if high is None else high
    cal = {}
    for cls, entry in derived["classes"].items():
        spec = {}
        for event in EVENTS:
            stats = entry.get(event)
            if stats and f"p{low:g}" in stats and f"p{high:g}" in stats:
                spec[event] = [int(np.floor(stats[f"p{low:g}"])), int(np.ceil(stats[f"p{high:g}"]))]
        if spec:
            cal[cls] = spec
    return cal


def _date(year: int, doy: float) -> date:
    return date(year, 1, 1) + timedelta(days=int(round(doy)) - 1)


def suggest_windows(derived: dict, year: int, min_days: int = 30, max_windows: int = 6,
                    events=("sos", "peak", "eos"), margin_days: int = 15) -> dict:
    """Windows ``{"P1": {"start", "end"}, ...}`` with boundaries at class-median phenological events.

    The season spans the earliest low-percentile SOS minus `margin_days` to the latest high-percentile EOS plus
    `margin_days`. Candidate boundaries (medians of `events` of every class) closer than `min_days` to a
    neighbour are merged until no window is shorter than `min_days` and there are at most `max_windows`.
    """
    pct = derived["percentiles"]
    lo_key, mid_key, hi_key = f"p{pct[0]:g}", f"p{pct[len(pct) // 2]:g}", f"p{pct[-1]:g}"
    starts, ends, marks = [], [], []
    for entry in derived["classes"].values():
        if entry.get("sos"):
            starts.append(entry["sos"][lo_key])
        if entry.get("eos"):
            ends.append(entry["eos"][hi_key])
        marks += [entry[e][mid_key] for e in events if entry.get(e)]
    if not starts or not ends:
        raise ValueError("Need SOS and EOS percentiles for at least one class to suggest windows")
    first, last = max(1.0, min(starts) - margin_days), min(365.0, max(ends) + margin_days)
    cuts = sorted(m for m in marks if first + min_days <= m <= last - min_days)
    bounds = [first, *cuts, last]

    while len(bounds) > 2:
        gaps = np.diff(bounds)
        i = int(np.argmin(gaps))
        if gaps[i] >= min_days and len(gaps) <= max_windows:
            break
        # merge the shortest window i into its shorter neighbour by removing the boundary between them
        if i == 0:
            bounds.pop(1)
        elif i == len(gaps) - 1:
            bounds.pop(i)
        else:
            bounds.pop(i if gaps[i - 1] <= gaps[i + 1] else i + 1)
    windows = {}
    for k, (a, b) in enumerate(zip(bounds[:-1], bounds[1:]), 1):
        start = _date(year, a) if k == 1 else _date(year, a) + timedelta(days=1)
        windows[f"P{k}"] = {"start": start.isoformat(), "end": _date(year, b).isoformat()}
    return windows


def windows_yaml(windows: dict, derived: dict | None = None, source: str | None = None) -> str:
    """YAML snippet for the configuration's ``windows:`` section, with provenance comments."""
    lines = ["# Suggested phenological windows derived from training-parcel markers (geocrop calendar derive)."]
    if source:
        lines.append(f"# Source: {source}")
    if derived:
        mid = f"p{derived['percentiles'][len(derived['percentiles']) // 2]:g}"
        for cls, e in sorted(derived["classes"].items()):
            parts = [f"{ev} {e[ev][mid]:.0f}" for ev in ("sos", "peak", "eos") if e.get(ev)]
            lines.append(f"#   {cls} (n={e['n_parcels']}): median DOY " + ", ".join(parts))
    lines.append("# Review against agronomic knowledge and the scene inventory before use.")
    lines.append("windows:")
    for name, w in windows.items():
        lines.append(f'  {name}: {{start: "{w["start"]}", end: "{w["end"]}"}}')
    return "\n".join(lines) + "\n"


def calendar_yaml(calendar: dict, derived: dict | None = None, source: str | None = None) -> str:
    lines = ["# Expected day-of-year ranges per class derived from training-parcel markers (geocrop calendar derive)."]
    if derived:
        pct = derived["percentiles"]
        lines.append(f"# Ranges are the p{pct[0]:g}-p{pct[-1]:g} percentiles; classes with < {derived['min_parcels']} "
                     "parcels per event are omitted.")
    if source:
        lines.append(f"# Source: {source}")
    for cls in sorted(calendar):
        items = ", ".join(f"{k}: [{v[0]}, {v[1]}]" for k, v in calendar[cls].items())
        n = f"  # n={derived['classes'][cls]['n_parcels']}" if derived and cls in derived["classes"] else ""
        lines.append(f"{cls}: {{{items}}}{n}")
    return "\n".join(lines) + "\n"


def worldcereal_priors(files: dict, lon: float, lat: float) -> dict:
    """``{crop: {"sos": doy|None, "eos": doy|None}}`` sampled from WorldCereal calendar rasters at (lon, lat).

    `files` is the output of `geocrop.public_data.worldcereal_calendar_files` (paths per crop and stage).
    Values are days of year of the first season (0 = no data); EOS < SOS means the season crosses the new year.
    """
    import rasterio

    out = {}
    for crop, stages in files.items():
        out[crop] = {}
        for stage, path in stages.items():
            with rasterio.open(Path(path)) as ds:
                value = float(next(ds.sample([(lon, lat)]))[0])
                nodata = ds.nodata if ds.nodata is not None else 0
            out[crop][stage] = None if not np.isfinite(value) or value == nodata or value <= 0 else int(round(value))
    return out
