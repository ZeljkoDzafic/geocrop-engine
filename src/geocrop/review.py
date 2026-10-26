"""Analyst parcel review: queue, evidence per parcel and an append-only decision log (gap G7).

The review works on the outputs of a finished parcel job (``geocrop parcels``/``run``/``demo``/``traffic-light``):

* `discover`         finds the parcel layer(s), signature tables (S2 indices / S1 dB), marker tables, imagery
                     folders (window composites and scene folders) and the model hash among a list of files;
* `query`            filters, sorts and pages the parcel table (traffic light, declared class, model decision,
                     reason code, uncertainty, analyst review state);
* `detail`           attributes, predicted vs declared class, class probabilities, conformal set, markers and
                     reason codes of one parcel;
* `series`           the parcel's time series from signatures, or (without signatures) parcel means computed from
                     the imagery folders;
* `chips`            small true-colour / NDVI image crops around the parcel for every date, with a shared stretch;
* `DecisionLog`      append-only CSV of analyst decisions (agree / disagree with corrected class / can't tell);
* `export_labels`    the decisions as a label table in exactly the format read by `geocrop active-learning ingest`.

Everything here reads and writes local files given by the caller; path confinement is the caller's job.
Analyst decisions are labels for the active-learning loop, not a probability sample: they must never be used to
estimate map accuracy or area (see docs/active-learning.md).
"""
from __future__ import annotations

import csv
import json
import math
import re
import threading
import weakref
from collections import OrderedDict
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .active_learning import LABEL_TEMPLATE_COLUMNS
from .cbm_markers import EVENT_COLUMNS

ID_FIELDS = ("parcel_id", "segment_id")
TRAFFIC_ORDER = {"red": 0, "yellow": 1, "grey": 2, "green": 3}
DECISIONS = ("agree", "disagree", "unsure")
PURPOSES = ("training", "validation")
REASON_COLUMNS = ("traffic_light_reasons", "decision_reasons", "evidence_warnings")
PHENOLOGY_MARKERS = ("clear_observations", "sos_doy", "peak_doy", "peak_value", "eos_doy", "harvest_doy",
                     "bare_soil_days", "calendar_consistent")
MARKER_COLUMNS = frozenset(EVENT_COLUMNS) | frozenset(PHENOLOGY_MARKERS)
SIGNATURE_COLUMNS = frozenset({"date", "index", "mean"})
SORT_KEYS = ("priority", "parcel_id", "confidence", "entropy", "margin", "set_size", "area_m2", "traffic_light")
DECISION_COLUMNS = ("timestamp_utc", "parcel_id", "decision", "label", "predicted_class", "declared_class",
                    "analyst", "comment", "purpose", "confidence", "model_sha256", "parcels_layer")
#: Class names written into label tables: no leading formula characters, no separators or control characters.
LABEL_PATTERN = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_ .()/-]{0,63}$")
SUMMARY_FIELDS = ("declared_class", "predicted_class", "traffic_light", "decision", "declaration_status",
                  "confidence", "entropy", "margin", "set_size", "prediction_set", "area_m2")
TABLE_SUFFIXES = (".csv", ".parquet")
VECTOR_SUFFIXES = (".gpkg", ".geojson")
MAX_TABLE_BYTES = 512 * 1024 * 1024
S2_ORDER = ("NDVI", "EVI", "NDRE", "NDMI", "BSI", "MSAVI2", "GCVI", "RECI")
S1_INDICES = ("VV_dB", "VH_dB", "VHVV_dB")
MAX_SERIES_PER_PANEL = 4
MAX_CHIP_DATES = 48


# ------------------------------------------------------------------------------------------------ small caches

class _Cache:
    """Tiny thread-safe LRU keyed by (path, mtime_ns, size, extra) so edited files are re-read."""

    def __init__(self, size: int = 6):
        self.size, self.data, self.lock = size, OrderedDict(), threading.Lock()

    def get(self, path: Path, loader, extra=()):
        st = path.stat()
        key = (str(path), st.st_mtime_ns, st.st_size, extra)
        with self.lock:
            if key in self.data:
                self.data.move_to_end(key)
                return self.data[key]
        value = loader(path)
        with self.lock:
            self.data[key] = value
            while len(self.data) > self.size:
                self.data.popitem(last=False)
        return value


_LAYERS, _TABLES, _HEADERS = _Cache(4), _Cache(4), _Cache(256)


def _columns(path: Path) -> tuple[str, ...]:
    """Column names of a CSV/Parquet/GPKG/GeoJSON file without reading its rows."""
    def load(p):
        suffix = p.suffix.lower()
        if suffix == ".csv":
            return tuple(pd.read_csv(p, nrows=0).columns)
        if suffix == ".parquet":
            import pyarrow.parquet as pq
            return tuple(pq.read_schema(p).names)
        import geopandas as gpd
        return tuple(gpd.read_file(p, rows=1).columns)
    try:
        return _HEADERS.get(path, load)
    except Exception:  # noqa: BLE001 - unreadable files are simply not review sources
        return ()


def id_field(columns) -> str | None:
    return next((c for c in ID_FIELDS if c in columns), None)


# ------------------------------------------------------------------------------------------------ discovery

def _scene_label(folder: Path) -> tuple[str, str | None]:
    """(label, ISO date or None) of an imagery folder: scene date, composite window, or folder name."""
    meta = folder / "scene.json"
    if meta.exists():
        try:
            day = str(json.loads(meta.read_text())["datetime"])[:10]
            return day, day
        except (ValueError, KeyError, OSError):
            pass
    comp = folder / "composite.json"
    if comp.exists():
        try:
            info = json.loads(comp.read_text())
            start, end = info.get("start"), info.get("end")
            if start and end:
                mid = date.fromordinal((date.fromisoformat(str(start)[:10]).toordinal()
                                        + date.fromisoformat(str(end)[:10]).toordinal()) // 2)
                return f"window {folder.name}", mid.isoformat()
        except (ValueError, TypeError, OSError):
            pass
        return f"window {folder.name}", None
    return folder.name, None


def imagery(files: list[Path]) -> list[dict]:
    """Imagery folders among `files`: S2 (B04+B08, true colour if B02+B03) and/or S1 (VV+VH), chronological."""
    names: dict[Path, set] = {}
    for f in files:
        if f.suffix.lower() == ".tif":
            names.setdefault(f.parent, set()).add(f.stem)
    out = []
    for folder, bands in names.items():
        s2, s1 = {"B04", "B08"} <= bands, {"VV", "VH"} <= bands
        if not (s2 or s1):
            continue
        label, day = _scene_label(folder)
        out.append({"folder": folder, "label": label, "date": day, "s2": s2, "s1": s1,
                    "true_colour": s2 and {"B02", "B03"} <= bands})
    out.sort(key=lambda m: (m["date"] or "", m["label"]))
    return out


def discover(files: list[Path], exclude: Path | None = None) -> dict:
    """Review sources among a job's (already confined) output files; `exclude` = the review folder itself."""
    layers, signatures, markers, manifest = [], [], [], None
    for f in files:
        if exclude is not None and f.is_relative_to(exclude):
            continue
        suffix = f.suffix.lower()
        if suffix in VECTOR_SUFFIXES and f.stat().st_size < MAX_TABLE_BYTES:
            cols = _columns(f)
            if id_field(cols) and ({"traffic_light", "predicted_class"} & set(cols)):
                layers.append(f)
        elif suffix in TABLE_SUFFIXES and f.stat().st_size < MAX_TABLE_BYTES:
            cols = set(_columns(f))
            if not id_field(cols):
                continue
            if SIGNATURE_COLUMNS <= cols:
                signatures.append(f)
            elif cols & MARKER_COLUMNS:
                markers.append(f)
        elif f.name in ("prediction_manifest.json", "parcels.summary.json") and manifest is None:
            manifest = f
    # the reviewed layer (traffic-light output) comes before the plain parcel layer; shortest path first
    layers.sort(key=lambda p: ("traffic_light_reasons" not in _columns(p), len(p.parts), str(p)))
    return {"layers": layers, "signatures": signatures, "markers": markers, "manifest": manifest,
            "imagery": imagery(files)}


def model_sha256(files: list[Path]) -> str:
    """The model hash recorded by predict/parcels (empty when unknown)."""
    for name, key in (("prediction_manifest.json", "model_sha256"), ("parcels.summary.json", "calibration_model_sha256"),
                      ("conformal.json", "model_sha256"), ("train_report.json", "model_sha256")):
        for f in files:
            if f.name == name:
                try:
                    value = json.loads(f.read_text()).get(key)
                except (ValueError, OSError):
                    continue
                if isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value):
                    return value
    return ""


# ------------------------------------------------------------------------------------------------ parcel table

def _load_layer(path: Path):
    import geopandas as gpd
    gdf = gpd.read_file(path)
    field = id_field(gdf.columns)
    if field is None:
        raise ValueError(f"{path.name}: no parcel id column ({' / '.join(ID_FIELDS)})")
    gdf[field] = gdf[field].astype(str)
    if gdf[field].duplicated().any():
        raise ValueError(f"{path.name}: parcel ids are not unique")
    gdf = gdf.set_index(field, drop=False)
    gdf.attrs["id_field"] = field
    return gdf


def load_layer(path: Path):
    return _LAYERS.get(Path(path), _load_layer)


def classes(gdf) -> list[str]:
    """Class names known to the layer: probability columns, then predicted and declared classes."""
    names = [c[2:] for c in gdf.columns if c.startswith("p_") and gdf[c].dtype.kind == "f"]
    for col in ("predicted_class", "declared_class"):
        if col in gdf:
            names += [str(v) for v in gdf[col].dropna().unique() if str(v)]
    return sorted(dict.fromkeys(n for n in names if LABEL_PATTERN.match(n)))


def _codes(value) -> list[str]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    return [c for c in str(value).split(";") if c.strip()]


def reasons(row) -> list[dict]:
    """Reason codes of one parcel with the column they come from (traffic light, decision, evidence)."""
    out, seen = [], set()
    for col, kind in zip(REASON_COLUMNS, ("traffic_light", "decision", "evidence")):
        for code in _codes(row.get(col)):
            if code not in seen:
                seen.add(code)
                out.append({"code": code.strip(), "source": kind})
    status = row.get("declaration_status")
    if isinstance(status, str) and status and status != "agreement" and status not in seen:
        out.insert(0, {"code": status, "source": "declaration"})
    return out


def _plain(value):
    """JSON-safe scalar: NaN/NA -> None, numpy -> Python, floats rounded for display."""
    if value is None:
        return None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return None if not math.isfinite(float(value)) else round(float(value), 6)
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value if isinstance(value, (int, str)) else str(value)


def summary_row(pid: str, row, reviewed: dict | None = None) -> dict:
    item = {"parcel_id": pid}
    for col in SUMMARY_FIELDS:
        if col in row:
            item[col] = _plain(row[col])
    item["reasons"] = [r["code"] for r in reasons(row)]
    if reviewed is not None:
        latest = reviewed.get(pid)
        item["review"] = None if latest is None else {k: latest.get(k) for k in ("decision", "label", "timestamp_utc")}
    return item


_MEMO: dict[tuple[int, str], object] = {}


def _memo(gdf, name: str, compute):
    """Per-layer-object memo (layers are cached by `load_layer`); entries vanish with the layer object."""
    key = (id(gdf), name)
    if key not in _MEMO:
        _MEMO[key] = compute()
        weakref.finalize(gdf, _MEMO.pop, key, None)
    return _MEMO[key]


def _reason_sets(gdf) -> pd.Series:
    """Reason codes per parcel, computed once per loaded layer."""
    def compute():
        t = gdf.drop(columns="geometry")
        return pd.Series([frozenset(r["code"] for r in reasons(row)) for _, row in t.iterrows()], index=t.index)
    return _memo(gdf, "reasons", compute)


def facets(gdf) -> dict:
    """Values available for each filter, with counts."""
    return _memo(gdf, "facets", lambda: _facets(gdf))


def _facets(gdf) -> dict:
    def counts(col):
        if col not in gdf:
            return {}
        return {str(k): int(v) for k, v in gdf[col].fillna("none").astype(str).value_counts().sort_index().items()}
    codes: dict[str, int] = {}
    for found in _reason_sets(gdf):
        for code in found:
            codes[code] = codes.get(code, 0) + 1
    return {"traffic_light": counts("traffic_light"), "declared_class": counts("declared_class"),
            "decision": counts("decision"), "reason": dict(sorted(codes.items())),
            "uncertainty": {"ambiguous_set": int((gdf["set_size"] > 1).sum()) if "set_size" in gdf else 0}}


def query(gdf, *, traffic_light=(), declared_class=(), decision=(), reason=(), max_confidence=None,
          min_set_size=None, review="all", reviewed: dict | None = None, search: str = "", sort: str = "priority",
          descending: bool = False, offset: int = 0, limit: int = 50) -> dict:
    """Filtered, sorted page of the parcel table. Multi-valued filters are OR within, AND across filters."""
    reviewed = reviewed or {}
    t = gdf.drop(columns="geometry")
    mask = pd.Series(True, index=t.index)
    for col, values in (("traffic_light", traffic_light), ("declared_class", declared_class), ("decision", decision)):
        if values:
            mask &= t[col].fillna("none").astype(str).isin(values) if col in t else False
    if reason:
        wanted = set(reason)
        mask &= _reason_sets(gdf).map(lambda found: bool(wanted & found))
    if max_confidence is not None:
        mask &= t["confidence"].astype(float) < max_confidence if "confidence" in t else False
    if min_set_size is not None:
        mask &= t["set_size"].astype(float) >= min_set_size if "set_size" in t else False
    if review in ("pending", "done"):
        done = t.index.isin(list(reviewed))
        mask &= done if review == "done" else ~done
    if search:
        mask &= t.index.str.contains(re.escape(search), case=False, regex=True)
    t = t[mask]
    if sort not in SORT_KEYS:
        raise ValueError(f"sort must be one of {', '.join(SORT_KEYS)}")
    ids = pd.Series(t.index, index=t.index)
    if sort in ("priority", "traffic_light"):
        light = t["traffic_light"].map(TRAFFIC_ORDER).fillna(9) if "traffic_light" in t else pd.Series(9, index=t.index)
        conf = t["confidence"].astype(float).fillna(-1) if "confidence" in t else pd.Series(0.0, index=t.index)
        keys = pd.DataFrame({"a": light, "b": conf, "c": ids})
        order = keys.sort_values(["a", "b", "c"], ascending=[not descending, not descending, True]).index
    elif sort == "parcel_id":
        order = ids.sort_values(ascending=not descending).index
    elif sort in t:
        keys = pd.DataFrame({"a": pd.to_numeric(t[sort], errors="coerce"), "c": ids})
        order = keys.sort_values(["a", "c"], ascending=[not descending, True], na_position="last").index
    else:
        order = ids.sort_values().index
    total = len(order)
    page = order[offset:offset + limit]
    return {"total": total, "offset": offset, "limit": limit,
            "items": [summary_row(pid, t.loc[pid], reviewed) for pid in page],
            "ids": [str(i) for i in order[:5000]]}


# ------------------------------------------------------------------------------------------------ one parcel

def _load_table(path: Path) -> pd.DataFrame:
    t = pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path, low_memory=False)
    field = id_field(t.columns)
    t[field] = t[field].astype(str)
    return t


def _rows_for(path: Path, pid: str) -> pd.DataFrame:
    t = _TABLES.get(path, _load_table)
    return t[t[id_field(t.columns)] == pid]


def marker_values(row, tables: list[Path], pid: str) -> dict:
    """cbm / cbm_markers columns of the parcel layer, completed from marker tables keyed by parcel id."""
    out = {c: _plain(row[c]) for c in row.index if c in MARKER_COLUMNS}
    for path in tables:
        try:
            rows = _rows_for(path, pid)
        except Exception:  # noqa: BLE001 - a broken marker table must not hide the parcel
            continue
        if len(rows):
            first = rows.iloc[0]
            for c in rows.columns:
                if c in MARKER_COLUMNS and c not in out:
                    out[c] = _plain(first[c])
    order = list(EVENT_COLUMNS) + list(PHENOLOGY_MARKERS)
    return {k: out[k] for k in sorted(out, key=lambda k: order.index(k) if k in order else len(order))}


def detail(gdf, pid: str, marker_tables=()) -> dict:
    if pid not in gdf.index:
        raise KeyError(pid)
    row = gdf.loc[pid]
    attrs = row.drop(labels="geometry")
    probs = {c[2:]: _plain(attrs[c]) for c in attrs.index if c.startswith("p_") and _plain(attrs[c]) is not None}
    fractions = {c[2:]: _plain(attrs[c]) for c in attrs.index if c.startswith("f_") and _plain(attrs[c]) is not None}
    conformal = [c for c in str(attrs.get("prediction_set") or "").replace(",", ";").split(";") if c]
    shown = {k for k in attrs.index if k.startswith(("p_", "f_")) or k in MARKER_COLUMNS or k in REASON_COLUMNS}
    return {
        "parcel_id": pid,
        "predicted_class": _plain(attrs.get("predicted_class")),
        "declared_class": _plain(attrs.get("declared_class")),
        "traffic_light": _plain(attrs.get("traffic_light")),
        "decision": _plain(attrs.get("decision")),
        "declaration_status": _plain(attrs.get("declaration_status")),
        "probabilities": dict(sorted(probs.items(), key=lambda kv: -kv[1])),
        "pixel_fractions": fractions,
        "conformal_set": conformal,
        "reasons": reasons(attrs),
        "markers": marker_values(attrs, list(marker_tables), pid),
        "attributes": {k: _plain(attrs[k]) for k in attrs.index if k not in shown},
    }


def geometry_wgs84(gdf, pid: str, tolerance_m: float = 1.0) -> dict:
    import shapely
    one = gdf.loc[[pid], ["geometry"]]
    if one.crs is not None and one.crs.is_projected:
        one["geometry"] = one.geometry.simplify(tolerance_m, preserve_topology=True)
    one = one.to_crs("EPSG:4326") if one.crs is not None else one
    geom = one.geometry.iloc[0]
    w, s, e, n = geom.bounds
    return {"type": "Feature", "properties": {"parcel_id": pid},
            "geometry": json.loads(shapely.to_geojson(geom)), "bounds": [[s, w], [n, e]]}


# ------------------------------------------------------------------------------------------------ time series

def _series_payload(panels: dict, source: str, files) -> dict:
    out = []
    for name, unit in (("Sentinel-2", "index"), ("Sentinel-1", "dB")):
        series = panels.get(name, {})
        keys = sorted(series, key=lambda k: (S2_ORDER + S1_INDICES).index(k.split(" ")[0])
                      if k.split(" ")[0] in S2_ORDER + S1_INDICES else 99)
        if keys:
            out.append({"name": name, "unit": unit,
                        "series": [{"name": k, "points": series[k]} for k in keys[:MAX_SERIES_PER_PANEL]],
                        "more": max(0, len(keys) - MAX_SERIES_PER_PANEL)})
    return {"source": source, "files": [str(f) for f in files], "panels": out}


def series_from_signatures(paths: list[Path], pid: str) -> dict:
    """Parcel series from `geocrop signatures` / `s1-signatures` tables (long format)."""
    panels: dict[str, dict[str, list]] = {}
    used = []
    for path in paths:
        try:
            rows = _rows_for(path, pid)
        except Exception:  # noqa: BLE001
            continue
        if not len(rows):
            continue
        used.append(path)
        orbit = "orbit_direction" in rows.columns
        for _, r in rows.sort_values("date").iterrows():
            index = str(r["index"])
            panel = "Sentinel-1" if index.endswith("_dB") else "Sentinel-2"
            name = index
            if panel == "Sentinel-1" and orbit and isinstance(r.get("orbit_direction"), str):
                name = f"{index} {r['orbit_direction'][:3].lower()}"
            mean = _plain(r.get("mean"))
            n_valid = _plain(r.get("n_valid"))
            panels.setdefault(panel, {}).setdefault(name, []).append({
                "date": str(r["date"])[:10], "label": str(r["date"])[:10], "mean": mean, "std": _plain(r.get("std")),
                "n_valid": n_valid, "observed": mean is not None and (n_valid is None or n_valid > 0)})
    return _series_payload(panels, "signatures", used)


def _window(src, geom, pad_fraction: float, min_pad_px: int):
    """Square read window around a geometry (in the raster's CRS), padded, in pixel coordinates."""
    from rasterio.windows import from_bounds
    w, s, e, n = geom.bounds
    res = abs(src.transform.a)
    side = max(e - w, n - s)
    pad = max(side * pad_fraction, min_pad_px * res)
    cx, cy, half = (w + e) / 2, (s + n) / 2, side / 2 + pad
    win = from_bounds(cx - half, cy - half, cx + half, cy + half, src.transform)
    return win.round_offsets().round_lengths(), (cx - half, cy - half, cx + half, cy + half)


def _geom_in(gdf, pid: str, crs):
    one = gdf.loc[[pid], ["geometry"]]
    return (one.to_crs(crs) if one.crs is not None and crs is not None else one).geometry.iloc[0]


def _read(folder: Path, band: str, window):
    import rasterio
    with rasterio.open(folder / f"{band}.tif") as src:
        data = src.read(1, window=window, boundless=True, fill_value=np.nan if src.nodata is None else src.nodata,
                        out_dtype="float64")
        if src.nodata is not None:
            data[data == src.nodata] = np.nan
        data[data == -9999] = np.nan
    return data


def _parcel_mask(geom, bounds, shape):
    from rasterio.features import geometry_mask
    from rasterio.transform import from_bounds
    transform = from_bounds(*bounds, shape[1], shape[0])
    return geometry_mask([geom], out_shape=shape, transform=transform, invert=True, all_touched=False)


def series_from_imagery(gdf, pid: str, folders: list[dict]) -> dict:
    """Parcel means per imagery folder: NDVI from S2 bands, VV/VH/VH-VV dB from linear S1 (mean in linear)."""
    import rasterio
    panels: dict[str, dict[str, list]] = {}
    for m in folders[:MAX_CHIP_DATES * 2]:
        band = "B04" if m["s2"] else "VV"
        with rasterio.open(m["folder"] / f"{band}.tif") as src:
            geom = _geom_in(gdf, pid, src.crs)
            window, bounds = _window(src, geom, 0.0, 0)
        if window.width < 1 or window.height < 1:
            continue
        point = {"date": m["date"], "label": m["label"]}
        if m["s2"]:
            red, nir = _read(m["folder"], "B04", window), _read(m["folder"], "B08", window)
            inside = _parcel_mask(geom, bounds, red.shape)
            with np.errstate(divide="ignore", invalid="ignore"):
                ndvi = ((nir - red) / (nir + red))[inside]
            ok = np.isfinite(ndvi)
            panels.setdefault("Sentinel-2", {}).setdefault("NDVI", []).append({
                **point, "mean": _plain(ndvi[ok].mean()) if ok.any() else None,
                "std": _plain(ndvi[ok].std()) if ok.any() else None, "n_valid": int(ok.sum()),
                "observed": bool(ok.any())})
        if m["s1"]:
            vv, vh = _read(m["folder"], "VV", window), _read(m["folder"], "VH", window)
            inside = _parcel_mask(geom, bounds, vv.shape)
            ok = inside & np.isfinite(vv) & np.isfinite(vh) & (vv > 0) & (vh > 0)
            means = (vv[ok].mean(), vh[ok].mean()) if ok.any() else (np.nan, np.nan)
            values = {"VV_dB": 10 * np.log10(means[0]), "VH_dB": 10 * np.log10(means[1]),
                      "VHVV_dB": 10 * np.log10(means[1] / means[0])} if ok.any() else dict.fromkeys(S1_INDICES)
            for name, value in values.items():
                panels.setdefault("Sentinel-1", {}).setdefault(name, []).append({
                    **point, "mean": _plain(value), "std": None, "n_valid": int(ok.sum()), "observed": bool(ok.any())})
    return _series_payload(panels, "imagery", sorted({str(m["folder"]) for m in folders}))


# ------------------------------------------------------------------------------------------------ evidence packages

EVIDENCE_SCHEMA = "geocrop-parcel-evidence"
_EVIDENCE = _Cache(20000)


def _evidence_id(path: Path):
    try:
        body = json.loads(path.read_text(encoding="utf8"))
    except (OSError, ValueError):
        return None
    if not isinstance(body, dict) or body.get("schema") != EVIDENCE_SCHEMA:
        return None
    return str(body.get("parcel_id"))


def evidence_packages(files: list[Path], limit: int = 20000) -> dict[str, Path]:
    """parcel id -> `evidence.json` of `geocrop evidence` packages among the (confined) files."""
    out = {}
    for f in files:
        if f.name != "evidence.json" or f.stat().st_size > 16 * 1024 * 1024:
            continue
        pid = _EVIDENCE.get(f, _evidence_id)
        if pid and pid not in out:
            out[pid] = f
            if len(out) >= limit:
                break
    return out


def evidence_summary(path: Path) -> dict:
    """Outcome, triggered rules and temporal support of one evidence package (for the parcel card)."""
    body = json.loads(Path(path).read_text(encoding="utf8"))
    rules = [{k: r.get(k) for k in ("rule", "stage", "outcome", "reason_code")}
             for r in body.get("rules", []) if isinstance(r, dict) and r.get("triggered")]
    support = body.get("temporal_support") or {}
    return {"schema_version": body.get("schema_version"), "outcome": body.get("outcome"), "triggered_rules": rules,
            "temporal_support": {k: support.get(k) for k in ("index", "clear_observations", "longest_gap_days",
                                                             "season_coverage", "peak_gap_days") if k in support},
            "notes": body.get("notes") or []}


def series_from_evidence(path: Path) -> dict:
    """Observed and rejected acquisitions plus the interpolated curve recorded in an evidence package."""
    body = json.loads(Path(path).read_text(encoding="utf8"))
    ds = body.get("data_sources") or {}
    panels: dict[str, dict[str, list]] = {}
    for o in ds.get("observed_acquisitions") or []:
        index = str(o.get("index"))
        panel = "Sentinel-1" if index.endswith("_dB") else "Sentinel-2"
        mean = _plain(o.get("mean"))
        panels.setdefault(panel, {}).setdefault(index, []).append({
            "date": str(o.get("date"))[:10], "label": str(o.get("date"))[:10], "mean": mean, "std": _plain(o.get("std")),
            "n_valid": _plain(o.get("n_valid")), "observed": mean is not None})
    for o in ds.get("rejected_acquisitions") or []:
        index = str(o.get("index"))
        panel = "Sentinel-1" if index.endswith("_dB") else "Sentinel-2"
        panels.setdefault(panel, {}).setdefault(index, []).append({
            "date": str(o.get("date"))[:10], "label": str(o.get("date"))[:10], "mean": None, "std": None,
            "n_valid": 0, "observed": False})
    index = (body.get("temporal_support") or {}).get("index") or "NDVI"
    grid = [g for g in ds.get("interpolated_grid") or [] if isinstance(g, dict)]
    if grid:
        panels.setdefault("Sentinel-2", {})[f"{index} curve"] = [{
            "date": str(g.get("date"))[:10], "label": str(g.get("date"))[:10], "mean": _plain(g.get("value")),
            "std": None, "n_valid": None, "observed": _plain(g.get("value")) is not None} for g in grid]
    for series in panels.values():
        for points in series.values():
            points.sort(key=lambda p: p["date"])
    return _series_payload(panels, "evidence", [path])


# ------------------------------------------------------------------------------------------------ image chips

NDVI_STOPS = ((-0.1, "#a6611a"), (0.2, "#dfc27d"), (0.4, "#f5f5a0"), (0.6, "#80cdc1"), (0.9, "#1a7837"))


def _hex(color: str):
    return tuple(int(color[i:i + 2], 16) for i in (1, 3, 5))


def _shrink(a: np.ndarray, size: int) -> np.ndarray:
    """Nearest-neighbour downsampling to at most `size` pixels per side (never upsampling)."""
    h, w = a.shape[:2]
    if max(h, w) <= size:
        return a
    rows = np.linspace(0, h - 1, size).round().astype(int)
    cols = np.linspace(0, w - 1, size).round().astype(int)
    return a[rows][:, cols]


def chips(gdf, pid: str, folders: list[dict], kind: str = "true", size: int = 96) -> list[dict]:
    """RGBA crops around a parcel for every imagery date, with one stretch shared by all dates.

    kind  `true` (B04/B03/B02, joint 2-98 % stretch over all dates so brightness is comparable) or
          `ndvi` (fixed colour ramp -0.1 ... 0.9).
    Each chip carries the parcel outline in chip coordinates (0..1, origin top-left) and the share of valid
    pixels inside the parcel (clouds/no-data lower it).
    """
    import rasterio
    if kind not in ("true", "ndvi"):
        raise ValueError("kind must be 'true' or 'ndvi'")
    usable = [m for m in folders if (m["true_colour"] if kind == "true" else m["s2"])][:MAX_CHIP_DATES]
    raw = []
    for m in usable:
        with rasterio.open(m["folder"] / "B04.tif") as src:
            geom = _geom_in(gdf, pid, src.crs)
            window, bounds = _window(src, geom, 0.35, 4)
        if window.width < 1 or window.height < 1 or window.width > 4096 or window.height > 4096:
            continue
        bands = ("B04", "B03", "B02") if kind == "true" else ("B04", "B08")
        data = np.stack([_read(m["folder"], b, window) for b in bands])
        inside = _parcel_mask(geom, bounds, data.shape[1:])
        raw.append((m, geom, bounds, data, inside))
    if not raw:
        return []
    if kind == "true":
        pooled = np.concatenate([d.reshape(3, -1) for _, _, _, d, _ in raw], axis=1)
        valid = np.isfinite(pooled).all(axis=0)
        lo, hi = (np.percentile(pooled[:, valid], [2, 98], axis=1) if valid.any()
                  else (np.zeros(3), np.ones(3)))
    out = []
    for m, geom, bounds, data, inside in raw:
        ok = np.isfinite(data).all(axis=0)
        rgba = np.zeros((*ok.shape, 4), dtype="uint8")
        if kind == "true":
            for i in range(3):
                t = np.clip((np.where(ok, data[i], lo[i]) - lo[i]) / max(hi[i] - lo[i], 1e-6), 0, 1) ** (1 / 1.2)
                rgba[..., i] = (t * 255).astype("uint8")
        else:
            with np.errstate(divide="ignore", invalid="ignore"):
                ndvi = (data[1] - data[0]) / (data[1] + data[0])
            ok &= np.isfinite(ndvi)
            xs = np.array([v for v, _ in NDVI_STOPS])
            cols = np.array([_hex(c) for _, c in NDVI_STOPS], dtype=float)
            v = np.clip(np.where(ok, ndvi, 0), xs[0], xs[-1])
            for ch in range(3):
                rgba[..., ch] = np.interp(v, xs, cols[:, ch]).astype("uint8")
        rgba[..., 3] = np.where(ok, 255, 0)
        x0, y0, x1, y1 = bounds
        rings = [geom.exterior] if geom.geom_type == "Polygon" else [g.exterior for g in getattr(geom, "geoms", [])]
        outline = [[[round((x - x0) / (x1 - x0), 4), round((y1 - y) / (y1 - y0), 4)] for x, y in ring.coords]
                   for ring in rings][:8]
        n_inside = int(inside.sum())
        out.append({"label": m["label"], "date": m["date"], "rgba": _shrink(rgba, size), "outline": outline,
                    "valid_fraction": round(float((ok & inside).sum() / n_inside), 3) if n_inside else None,
                    "pixels": int(max(data.shape[1:]))})
    return out


# ------------------------------------------------------------------------------------------------ decisions

def _text(value, limit: int) -> str:
    """Single-line free text for a CSV cell: control characters removed, formula prefixes neutralised."""
    text = re.sub(r"[\x00-\x1f\x7f]+", " ", str(value or "")).strip()[:limit]
    return "'" + text if text[:1] in ("=", "+", "-", "@") else text


def validate_decision(body: dict, gdf, known_classes: list[str]) -> dict:
    """Checked decision row (raises ValueError with a readable message)."""
    pid = str(body.get("parcel_id") or "")
    if pid not in gdf.index:
        raise KeyError(pid)
    decision = body.get("decision")
    if decision not in DECISIONS:
        raise ValueError(f"decision must be one of {', '.join(DECISIONS)}")
    purpose = body.get("purpose")
    if purpose not in PURPOSES:
        raise ValueError("purpose must be 'training' or 'validation' (stated explicitly for every decision)")
    row = gdf.loc[pid]
    predicted = _plain(row.get("predicted_class"))
    label = str(body.get("label") or "").strip()
    if decision == "agree":
        if not predicted:
            raise ValueError("This parcel has no predicted class to agree with; record 'disagree' with a class")
        label = str(predicted)
    elif decision == "disagree":
        if not label:
            raise ValueError("A disagreement needs the corrected class")
        if not LABEL_PATTERN.match(label):
            raise ValueError("Class names use letters, digits, space, '_', '-', '.', '/', '(' or ')' (max 64)")
        if predicted and label == str(predicted):
            raise ValueError("The corrected class equals the prediction; record 'agree' instead")
    else:
        label = ""
    confidence = body.get("confidence")
    if confidence is not None:
        confidence = float(confidence)
        if not 0 <= confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")
    return {"parcel_id": pid, "decision": decision, "label": label, "predicted_class": predicted or "",
            "declared_class": _plain(row.get("declared_class")) or "", "analyst": _text(body.get("analyst"), 80),
            "comment": _text(body.get("comment"), 2000), "purpose": purpose,
            "confidence": "" if confidence is None else f"{confidence:g}",
            "new_class": bool(label and known_classes and label not in known_classes)}


class DecisionLog:
    """Append-only CSV log of analyst decisions; the latest row per (parcel, analyst) is the current one."""

    _locks: dict[str, threading.Lock] = {}
    _guard = threading.Lock()

    def __init__(self, path: Path):
        self.path = Path(path)
        with self._guard:
            self.lock = self._locks.setdefault(str(self.path), threading.Lock())

    def append(self, row: dict, model_sha256: str, layer: str) -> dict:
        record = {**{c: "" for c in DECISION_COLUMNS}, **{k: v for k, v in row.items() if k in DECISION_COLUMNS},
                  "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                  "model_sha256": model_sha256, "parcels_layer": layer}
        with self.lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            new = not self.path.exists() or self.path.stat().st_size == 0
            with self.path.open("a", newline="", encoding="utf8") as fh:
                writer = csv.DictWriter(fh, fieldnames=DECISION_COLUMNS)
                if new:
                    writer.writeheader()
                writer.writerow(record)
        return record

    def rows(self) -> list[dict]:
        if not self.path.exists():
            return []
        with self.lock, self.path.open(newline="", encoding="utf8") as fh:
            return [dict(r) for r in csv.DictReader(fh)]

    def latest(self) -> dict[str, dict]:
        """Most recent decision per parcel (any analyst) for the queue state."""
        out = {}
        for r in self.rows():
            out[r.get("parcel_id", "")] = r
        return out


def export_labels(rows: list[dict], purpose: str, round_id: int | None = None) -> pd.DataFrame:
    """Label table for `geocrop active-learning ingest` (columns exactly `LABEL_TEMPLATE_COLUMNS`).

    One row per (parcel, analyst): the analyst's latest decision. Rows of other purposes are left out; `unsure`
    decisions are kept with an empty label (recorded as `no_label` by ingest). `source` is `analyst`.
    """
    if purpose not in PURPOSES:
        raise ValueError("purpose must be 'training' or 'validation'")
    latest: dict[tuple[str, str], dict] = {}
    for r in rows:
        latest[(r.get("parcel_id", ""), r.get("analyst", ""))] = r
    out = []
    for r in latest.values():
        if r.get("purpose") != purpose:
            continue
        note = f"review:{r.get('decision', '')}" + (f"; {r['comment']}" if r.get("comment") else "")
        out.append({"parcel_id": r["parcel_id"], "label": r.get("label", ""), "source": "analyst",
                    "date": r.get("timestamp_utc", ""), "confidence": r.get("confidence", ""), "purpose": purpose,
                    "observer": r.get("analyst", ""), "notes": note,
                    "al_round": "" if round_id is None else int(round_id)})
    return pd.DataFrame(out, columns=list(LABEL_TEMPLATE_COLUMNS))
