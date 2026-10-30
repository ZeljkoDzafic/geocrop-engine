"""Reference-data intake: read-only checks of training polygons, declared parcels and field observations.

``geocrop intake`` is run when restricted reference data arrive, before a multi-hour run. It composes the existing
input audit (:mod:`geocrop.audit`: schema, geometry validity, duplicate IDs, contradictory overlaps, observation
leakage) and the class-support planning of :mod:`geocrop.preflight` (spatial blocks, folds, locked holdout), and
adds what only the arrival of data can show: file readability, encodings of class names, declaration codes
missing from the crop mapping, dates against the study season, overlap with the AOI, per-municipality counts,
polygon size in pixels and training/parcel overlap.

Privacy: the report contains counts, class names, declaration codes and (at most ``list_offenders``) feature IDs
or row numbers. It never contains geometries, coordinates or other attributes.
"""
from __future__ import annotations

import json
import re
import unicodedata
import warnings
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box

from . import __version__
from . import config as cfgmod
from .adapters import clean_code, parse_date, read_layer, read_mapping_rows

SCHEMA = "geocrop-intake/1"
LAYERS = ("training", "parcels", "observations")
POLYGONS = ("Polygon", "MultiPolygon")
#: Heuristic thresholds (not standards); each is a warning limit, stated in the report.
MIN_POLYGONS_PER_CLASS = 10
MIN_OBSERVATIONS_PER_CLASS = 25
BALANCE_RATIO = 10.0
UNMAPPED_ERROR_SHARE = 0.05
TINY_PARCEL_SHARE = 0.10
INDICATIVE_HOLDOUT = 0.2
MUNICIPALITY_FIELDS = ("municipality", "opstina", "opcina", "general")
BOUNDARY_NAME_FIELDS = ("municipality", "opstina", "opcina", "name", "shapename", "name_3")
SCOPE = ("Read-only check of the supplied layers. Counts and class names only; no geometries or attributes are "
         "written. Heuristic thresholds are marked in the messages; a PASS does not establish label quality or accuracy.")


def fold(value) -> str:
    """Comparison key of a class name: diacritics removed (also đ), case-folded, separators unified."""
    text = str(value).translate(str.maketrans({"đ": "d", "Đ": "D", "ł": "l", "Ł": "L"}))
    text = "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))
    return re.sub(r"[\s_\-]+", "_", text.strip().casefold())


def _mojibake(text: str) -> bool:
    if "�" in text:
        return True
    if text.isascii():
        return False
    for enc in ("cp1252", "cp1250", "latin-1"):
        try:
            if text.encode(enc).decode("utf-8") != text:
                return True
        except (UnicodeEncodeError, UnicodeDecodeError):
            continue
    return False


def text_issues(values: pd.Series) -> dict:
    """Spelling problems in a class-name or code column and a suggested normalisation map."""
    raw = values.dropna().astype(str)
    raw = raw[raw.str.strip() != ""]
    counts = raw.value_counts()
    whitespace = sorted(v for v in counts.index if v != " ".join(v.split()))
    broken = sorted(v for v in counts.index if _mojibake(v))
    groups: dict[str, list[str]] = {}
    for v in counts.index:
        groups.setdefault(fold(v), []).append(v)
    variants = {}
    for members in groups.values():
        if len(members) > 1:
            canonical = " ".join(max(members, key=lambda m: (counts[m], m == " ".join(m.split()), m)).split())
            variants.update({m: canonical for m in members if m != canonical})
    return {"whitespace": whitespace, "mojibake": broken, "variants": variants,
            "distinct": int(len(counts))}


class _Report:
    def __init__(self, limit: int):
        self.checks: list[dict] = []
        self.offenders: dict[str, list] = {}
        self.limit = limit

    def add(self, code, layer, status, message, **details):
        item = {"code": code, "layer": layer, "status": status, "message": message}
        if details:
            item["details"] = details
        self.checks.append(item)
        return item

    def ids(self, code, values):
        values = [v for v in values if v is not None]
        if self.limit > 0 and values:
            self.offenders[code] = [v.item() if hasattr(v, "item") else v for v in values[: self.limit]]


def _find_aoi(cfg, aoi, config_path):
    if aoi:
        return Path(aoi), "argument"
    if config_path:
        near = Path(config_path).resolve().parent / "aoi.geojson"
        if near.exists():
            return near, "next to the configuration"
    return None, None


def _aoi_geometry(cfg, aoi, config_path):
    """(GeoSeries of the AOI polygon, municipality names, description) or (None, [], reason)."""
    path, how = _find_aoi(cfg, aoi, config_path)
    if path is not None:
        frame = gpd.read_file(path)
        names = []
        if "municipalities" in frame:
            for entry in frame["municipalities"].dropna():
                names += list(entry) if isinstance(entry, (list, tuple, np.ndarray)) else re.split(r"[;,]", str(entry))
        return frame.geometry, [n.strip() for n in names if str(n).strip()], f"{path.name} ({how})"
    ref = cfg["grid"].get("reference")
    if ref and Path(ref).exists():
        import rasterio
        with rasterio.open(ref) as ds:
            return gpd.GeoSeries([box(*ds.bounds)], crs=ds.crs), [], "grid reference footprint"
    return None, [], None


def _find_boundaries(cfg, path, config_path):
    if path:
        return Path(path)
    if config_path:
        base = Path(config_path).resolve().parent
        for name in ("municipalities.geojson", "municipalities.gpkg", "municipalities.shp"):
            if (base / name).exists():
                return base / name
    return None


class _Layer:
    """One loaded layer: raw frame, and a working copy (valid, non-empty, study CRS) with source row numbers."""

    def __init__(self, name, source, raw, info):
        self.name, self.source, self.info = name, source, info
        raw = raw.copy()
        raw["_row"] = np.arange(1, len(raw) + 1)
        self.raw, self.work = raw, None
        self.repair = {"invalid": 0, "repairable": 0}


def _prepare(layer: _Layer, crs, rep: _Report, types):
    raw = layer.raw
    geom = raw.geometry
    absent = geom.isna() | geom.is_empty
    work = raw.loc[~absent].copy()
    invalid = ~work.geometry.is_valid
    layer.repair["invalid"] = int(invalid.sum())
    if invalid.any():
        fixed = work.loc[invalid, "geometry"].make_valid()
        ok = fixed.is_valid & ~fixed.is_empty
        layer.repair["repairable"] = int(ok.sum())
        work.loc[invalid, "geometry"] = fixed
        rep.add("geometry.repairable", layer.name, "warn" if ok.all() else "error",
                f"{int(invalid.sum())} invalid geometries; automatic repair (make_valid) fixes {int(ok.sum())} of them. "
                "In QGIS: Vector > Geometry Tools > Fix geometries (or 'Check validity' to see them).",
                invalid=int(invalid.sum()), repairable=int(ok.sum()))
        rep.ids("geometry.repairable", list(work.loc[invalid, "_row"]))
    if types:
        work = work[work.geom_type.isin(types)]
    work = work[~work.geometry.is_empty]
    if raw.crs is not None and len(work):
        try:
            work = work.to_crs(crs)
        except Exception as exc:  # noqa: BLE001 - unusual CRS definitions
            rep.add(f"{layer.name}.reproject", layer.name, "error", f"Cannot reproject from {raw.crs} to {crs}: {exc}")
            work = work.iloc[0:0]
    elif raw.crs is None:
        work = work.iloc[0:0]
    layer.work = work


def _load(name, source, opts, rep: _Report) -> _Layer | None:
    try:
        raw, info = read_layer(source, **opts)
    except Exception as exc:  # noqa: BLE001 - GDAL/pyogrio/pandas raise several unrelated types
        hint = ""
        if Path(source).suffix.lower() == ".shp":
            hint = " (a Shapefile needs its .shx, .dbf and .prj files next to the .shp)"
        elif Path(source).suffix.lower() in (".csv", ".txt", ".tsv"):
            hint = " (for a table pass --wkt-field, or --x-field/--y-field with --crs; try --encoding cp1250)"
        rep.add(f"{name}.read", name, "error",
                f"Cannot read {Path(source).name}: {type(exc).__name__}: {str(exc)[:200]}{hint}")
        return None
    if not isinstance(raw, gpd.GeoDataFrame) or "geometry" not in raw:
        rep.add(f"{name}.read", name, "error", f"{Path(source).name} has no geometry column.")
        return None
    rep.add(f"{name}.read", name, "pass",
            f"Read {len(raw)} features ({', '.join(f'{k}={v}' for k, v in info.items())}).")
    return _Layer(name, str(source), raw, info)


def _text_checks(rep: _Report, layer: str, field: str, values: pd.Series, targets=()) -> dict:
    issues = text_issues(values)
    prefix = f"{layer}.{field}"
    if issues["mojibake"]:
        rep.add(f"{prefix}.encoding", layer, "error",
                f"{len(issues['mojibake'])} value(s) in {field!r} look like wrongly decoded text (e.g. 'Å¡' instead "
                f"of 'š'); re-export as UTF-8 or read the file with --encoding cp1250: {issues['mojibake'][:5]}",
                values=issues["mojibake"][:20])
    if issues["whitespace"]:
        rep.add(f"{prefix}.whitespace", layer, "warn",
                f"{len(issues['whitespace'])} value(s) in {field!r} have leading/trailing/double spaces: "
                f"{[repr(v) for v in issues['whitespace'][:5]]}", values=issues["whitespace"][:20])
    if issues["variants"]:
        rep.add(f"{prefix}.variants", layer, "warn",
                f"Spelling variants of the same value in {field!r} (case, diacritics, spaces): "
                f"{len(issues['variants'])} variant(s) map to {len(set(issues['variants'].values()))} canonical "
                "name(s); apply the suggested normalisation map.", normalisation_map=issues["variants"])
    known = {fold(t): t for t in targets}
    misspelt = {v: known[fold(v)] for v in set(map(str, values.dropna())) if fold(v) in known and v != known[fold(v)]
                and v.strip() != known[fold(v)]}
    if misspelt:
        rep.add(f"{prefix}.target_spelling", layer, "error",
                f"Values that differ from a configured target class only by case/diacritics/separators will not be "
                f"recognised: {dict(list(misspelt.items())[:5])}", normalisation_map=misspelt)
    return issues


def _class_table(rep: _Report, cfg, layer: _Layer, inner_m: float) -> dict:
    """Polygons, area and pixel counts per training class (after the interior buffer), and tiny polygons."""
    label, ident = cfg["classes"]["label_field"], cfg["classes"]["id_field"]
    work = layer.work
    if label not in work or not len(work):
        return {}
    res = float(cfg["grid"].get("resolution_m") or 10)
    work = work[work[label].notna() & (work[label].astype(str).str.strip() != "")]
    geom = work.geometry.buffer(-abs(inner_m)) if inner_m else work.geometry
    px = (geom.area / (res * res)).to_numpy()
    cap = cfg["model"]["max_samples_per_polygon"]
    labels = work[label].astype(str).str.strip()
    table, tiny_rows = {}, []
    for c in sorted(labels.unique()):
        sel = (labels == c).to_numpy()
        v = px[sel]
        table[c] = {"polygons": int(sel.sum()), "area_ha": round(float(work.geometry.area[sel].sum() / 1e4), 2),
                    "pixels": int(v.sum()), "expected_samples": int(np.minimum(v, cap).sum()),
                    "min_pixels": round(float(v.min()), 2), "polygons_below_1px": int((v < 1).sum()),
                    "polygons_below_4px": int((v < 4).sum())}
        tiny_rows += list(work.loc[sel & (px < 1), ident if ident in work else "_row"])
    note = f" after a {inner_m:g} m interior buffer" if inner_m else ""
    tiny = sum(v["polygons_below_1px"] for v in table.values())
    rep.add("training.polygon_size", "training", "warn" if tiny else "pass",
            f"{tiny} polygon(s) are smaller than one {res:g} m pixel{note} and will usually yield no sample."
            if tiny else f"Every polygon covers at least one {res:g} m pixel{note}.",
            polygons_below_1px=tiny, min_pixels=min((v["min_pixels"] for v in table.values()), default=None))
    rep.ids("training.polygon_size", tiny_rows)
    empty = [c for c, v in table.items() if v["expected_samples"] == 0]
    if empty:
        rep.add("training.class_no_pixels", "training", "error",
                f"Classes whose polygons are all smaller than one pixel{note}: {empty}.", classes=empty)
    return table


def _balance(rep: _Report, layer, table, targets, minimum, key, unit):
    if not table:
        return
    counts = {c: v[key] for c, v in table.items()}
    low = {c: n for c, n in counts.items() if n < minimum}
    rep.add(f"{layer}.class_support", layer, "warn" if low else "pass",
            f"Classes with fewer than {minimum} {unit} (heuristic): {low}." if low
            else f"Every class has at least {minimum} {unit} (heuristic limit).", classes=low)
    missing = [t for t in targets if t not in counts]
    if missing:
        rep.add(f"{layer}.targets_missing", layer, "error",
                f"Configured target classes without any {unit} in the layer: {missing}.", classes=missing)
    if len(counts) > 1:
        hi, lo = max(counts.values()), min(counts.values())
        ratio = hi / max(lo, 1)
        rep.add(f"{layer}.class_balance", layer, "warn" if ratio > BALANCE_RATIO else "pass",
                f"Largest/smallest class ratio is {ratio:.1f} in {unit} (heuristic warning above {BALANCE_RATIO:g}); "
                "class weights and the per-polygon sample cap limit the effect but not the evaluation noise.",
                ratio=round(ratio, 2))


def _municipality(rep: _Report, cfg, layer: _Layer, boundaries, expected, label_field=None, area=False) -> dict | None:
    work = layer.work
    col = next((c for c in layer.raw.columns if fold(c) in MUNICIPALITY_FIELDS), None)
    how = None
    if col is not None:
        names = layer.raw.set_index("_row")[col].astype(str).str.strip().replace("", "(blank)")
        names = names.reindex(work["_row"]).to_numpy()
        how = f"field {col!r}"
    elif boundaries is not None and len(work):
        pts = work.geometry.representative_point()
        joined = gpd.sjoin(gpd.GeoDataFrame(geometry=pts, crs=work.crs), boundaries, how="left", predicate="within")
        joined = joined[~joined.index.duplicated()]
        names = joined["_name"].fillna("(outside boundaries)").reindex(work.index).to_numpy()
        how = "spatial join with the municipality boundaries"
    else:
        rep.add(f"{layer.name}.municipality", layer.name, "pass",
                "Per-municipality counts skipped: no `municipality`/`opstina` field and no municipality boundaries "
                "file (--municipalities) available offline.", skipped=True)
        return None
    frame = pd.DataFrame({"m": names, "label": work[label_field].astype(str).str.strip().to_numpy()
                          if label_field and label_field in work else ""})
    if area:
        frame["ha"] = (work.geometry.area / 1e4).to_numpy()
    out = {}
    for m, g in frame.groupby("m"):
        out[str(m)] = {"features": int(len(g))}
        if area:
            out[str(m)]["area_ha"] = round(float(g["ha"].sum()), 1)
        if label_field and layer.name != "parcels":
            out[str(m)]["per_class"] = {str(k): int(v) for k, v in g["label"].value_counts().items()}
    status, msg = "pass", f"Counts per municipality from {how}."
    known = {fold(e) for e in expected}
    if expected:
        unknown = sorted(m for m in out if fold(m) not in known)
        absent = sorted(e for e in expected if fold(e) not in {fold(m) for m in out})
        if unknown or absent:
            status = "warn"
            msg = (f"Municipality names ({how}) differ from the study municipalities {expected}: not recognised "
                   f"{unknown}; no features in {absent}.")
    rep.add(f"{layer.name}.municipality", layer.name, status, msg, per_municipality=out)
    return out


def _aoi_overlap(rep: _Report, layer: _Layer, aoi, points: bool):
    work = layer.work
    if aoi is None or not len(work):
        return
    mask = work.geometry.intersects(aoi)
    outside = int((~mask).sum())
    if points or layer.name == "observations":
        share = float(mask.mean())
    else:
        total = work.geometry.area.sum()
        share = float(work.geometry.intersection(aoi).area.sum() / total) if total else 0.0
    status = "error" if share == 0 else ("warn" if outside else "pass")
    hint = (" Nothing falls inside the AOI: check the CRS definition and swapped x/y (lon/lat) columns."
            if share == 0 else "")
    rep.add(f"{layer.name}.aoi_overlap", layer.name, status,
            f"{share:.1%} of the layer lies inside the AOI; {outside} of {len(work)} feature(s) are outside and will "
            f"not contribute.{hint}", share_inside=round(share, 4), outside=outside)
    rep.ids(f"{layer.name}.aoi_overlap", list(work.loc[~mask, "_row"]))


def _training_checks(rep, cfg, layer, targets, inner_m, holdout, frame_for_cv):
    cl = cfg["classes"]
    label = cl["label_field"]
    for f in (cl["id_field"], label):
        if f not in layer.raw:
            rep.add(f"training.{f}.missing", "training", "error",
                    f"Required field {f!r} is missing (columns: {[c for c in layer.raw.columns if c not in ('geometry', '_row')][:15]}).")
    if label in layer.raw:
        _text_checks(rep, "training", label, layer.raw[label], targets)
    table = _class_table(rep, cfg, layer, inner_m)
    _balance(rep, "training", table, targets, MIN_POLYGONS_PER_CLASS, "polygons", "polygons")
    extra = sorted(set(table) - set(targets)) if targets else []
    if targets:
        rep.add("training.auxiliary_classes", "training", "pass",
                f"Classes besides the configured targets (auxiliary): {extra}.", classes=extra)
    support, checks, assumed = {}, [], False
    from .preflight import class_support
    if len(layer.work) and label in layer.raw and cl["id_field"] in layer.raw and layer.raw.crs is not None:
        frac = cfg["model"]["holdout_fraction"]
        assumed = not frac
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")   # sklearn complains about the very classes reported below
                support, checks = class_support(cfg, frame_for_cv, INDICATIVE_HOLDOUT if assumed else None)
        except Exception as exc:  # noqa: BLE001 - reported, never fatal
            rep.add("training.cv_support", "training", "warn", f"Class-support planning failed: {exc}")
    for c in checks:
        status = c["status"]
        if assumed and status == "error" and c["code"].startswith(("classes.holdout", "classes.development")):
            status = "warn"
        msg = c["message"] + (f" (indicative locked holdout of {INDICATIVE_HOLDOUT:g}; the configuration has none)"
                              if assumed and c["code"].startswith("classes.holdout") else "")
        rep.add(f"training.{c['code']}", "training", status, msg, **c.get("details", {}))
    return table, support, assumed


def _parcel_checks(rep, cfg, layer, mapping_rows, targets, training_classes, inner_px):
    pc = cfg["parcels"]
    field = pc["declared_field"]
    if pc["id_field"] not in layer.raw:
        rep.add("parcels.parcel_id.missing", "parcels", "error",
                f"Required field {pc['id_field']!r} is missing.")
    mapping = {}
    if mapping_rows is not None:
        keys = Counter(k for k, _ in mapping_rows)
        conflict = sorted(k for k, n in keys.items() if n > 1 and len({v for kk, v in mapping_rows if kk == k}) > 1)
        if conflict:
            rep.add("parcels.mapping_conflict", "parcels", "error",
                    f"Crop-mapping codes with several different classes (the last one would win): {conflict[:10]}.",
                    codes=conflict[:50])
        mapping = dict(mapping_rows)
        _text_checks(rep, "parcels", "crop_mapping.class_name", pd.Series(sorted(set(mapping.values()))))
        uncovered = [t for t in targets if t not in set(mapping.values())]
        if uncovered:
            rep.add("parcels.mapping_targets", "parcels", "warn",
                    f"No mapping row produces the target class(es) {uncovered}: no parcel can be declared as such.",
                    classes=uncovered)
    if field not in layer.raw:
        rep.add("parcels.declared_crop.missing", "parcels", "error",
                f"Declaration field {field!r} is missing; map the authority's field with "
                f"`geocrop normalize-parcels --declared-field <your field>` or set parcels.declared_field.")
        return {}
    raw = layer.raw[field]
    nonblank = raw.notna() & (raw.astype(str).str.strip() != "")
    codes = raw[nonblank].map(clean_code)
    blank = int((~nonblank).sum())
    rep.add("parcels.declared_crop.blank", "parcels", "warn" if blank else "pass",
            f"{blank} parcel(s) have no declared crop." if blank else "Every parcel has a declared crop.", count=blank)
    rep.ids("parcels.declared_crop.blank", list(layer.raw.loc[~nonblank, "_row"]))
    artefacts = int(raw[nonblank].astype(str).str.fullmatch(r"\s*\d+\.0\s*").sum())
    if artefacts:
        rep.add("parcels.declared_crop.float_codes", "parcels", "warn",
                f"{artefacts} code(s) look like spreadsheet numbers ('1101.0'); they are read as '1101'. Store the "
                "column as text to keep leading zeros.", count=artefacts)
    _text_checks(rep, "parcels", field, codes.astype(str))
    vocabulary = set(mapping) | set(mapping.values()) if mapping else set(targets) | set(training_classes)
    unmapped = codes[~codes.isin(vocabulary)]
    share = float(len(unmapped) / max(len(codes), 1))
    what = "the crop mapping" if mapping else "the target/training classes (no crop mapping supplied)"
    counts = unmapped.value_counts()
    rep.add("parcels.unmapped_codes", "parcels",
            "pass" if not len(unmapped) else ("error" if share >= UNMAPPED_ERROR_SHARE else "warn"),
            f"{len(unmapped)} of {len(codes)} declarations ({share:.1%}) use {len(counts)} code(s) absent from {what}; "
            f"add them to the mapping CSV (error from {UNMAPPED_ERROR_SHARE:.0%}, heuristic): "
            f"{dict(list(counts.items())[:10])}" if len(unmapped) else f"Every declaration code is covered by {what}.",
            unmapped_codes={str(k): int(v) for k, v in counts.items()}, share=round(share, 4))
    rep.ids("parcels.unmapped_codes", list(layer.raw.loc[codes[~codes.isin(vocabulary)].index, "_row"]))
    classes = codes.map(lambda c: mapping.get(c, c))
    cls_counts = classes.value_counts()
    table = {str(k): int(v) for k, v in cls_counts.items()}
    rep.add("parcels.declared_classes", "parcels", "pass", f"Declared classes after mapping: {table}.", per_class=table)
    work = layer.work
    if len(work):
        wkb = work.geometry.to_wkb()
        dup = int(wkb.duplicated().sum())
        rep.add("parcels.duplicate_geometry", "parcels", "warn" if dup else "pass",
                f"{dup} parcel(s) repeat the exact geometry of another parcel." if dup
                else "No exact duplicate parcel geometries.", count=dup)
        rep.ids("parcels.duplicate_geometry", list(work.loc[wkb.duplicated(keep=False), "_row"]))
        res = float(cfg["grid"].get("resolution_m") or 10)
        core = work.geometry.buffer(-inner_px * res) if inner_px else work.geometry
        px = core.area / (res * res)
        small = int((px < pc["min_valid_pixels"]).sum())
        share = small / len(work)
        rep.add("parcels.polygon_size", "parcels", "warn" if share > TINY_PARCEL_SHARE else "pass",
                f"{small} of {len(work)} parcels ({share:.1%}) have fewer than min_valid_pixels={pc['min_valid_pixels']} "
                f"pixels at {res:g} m after the {inner_px:g}-pixel interior buffer; they get no decision "
                f"(warning above {TINY_PARCEL_SHARE:.0%}, heuristic).", below=small, share=round(share, 4))
        rep.ids("parcels.polygon_size", list(work.loc[px < pc["min_valid_pixels"], "_row"]))
    return table


def _overlap_training_parcels(rep, training: _Layer, parcels: _Layer):
    t, p = training.work, parcels.work
    if not len(t) or not len(p):
        return
    index = p.sindex
    inside, identical, rows = 0, 0, []
    for i, geom in zip(t["_row"], t.geometry):
        best, iou = 0.0, 0.0
        for j in index.query(geom, predicate="intersects"):
            other = p.geometry.iloc[j]
            inter = geom.intersection(other).area
            if inter <= 0:
                continue
            best = max(best, inter / geom.area)
            iou = max(iou, inter / (geom.area + other.area - inter))
        if best >= 0.5:
            inside += 1
            rows.append(i)
        identical += iou >= 0.9
    rep.add("training.parcel_overlap", "training", "warn" if inside else "pass",
            f"{inside} of {len(t)} training polygon(s) lie (>= 50% of their area) inside declared parcels and "
            f"{int(identical)} nearly coincide with one (IoU >= 0.9). Training objects must be excluded from any "
            "parcel-level verification statistics (leakage)." if inside
            else "No training polygon lies inside a declared parcel.", inside=inside, near_identical=int(identical))
    rep.ids("training.parcel_overlap", rows)


def _observation_checks(rep, cfg, layer, targets, training_classes, season, exclusion_m):
    v = cfg["validation"]
    label, date_f, group = v["label_field"], v["date_field"], v["group_field"]
    raw = layer.raw
    present = True
    for f, why in ((label, "class"), (date_f, "date")):
        if f not in raw:
            rep.add(f"observations.{f}.missing", "observations", "error", f"Required {why} field {f!r} is missing.")
            present = False
    if group and group not in raw:
        rep.add(f"observations.{group}.missing", "observations", "error",
                f"validation.group_field is {group!r} but the layer has no such field.")
    table = {}
    if label in raw:
        issues = _text_checks(rep, "observations", label, raw[label], targets)
        del issues
        vals = raw[label].dropna().astype(str).str.strip()
        table = {str(k): {"observations": int(n)} for k, n in vals.value_counts().items()}
        _balance(rep, "observations", table, [], MIN_OBSERVATIONS_PER_CLASS, "observations", "observations")
        seen = set(table)
        reference = set(training_classes)
        if reference:
            absent = sorted(seen - reference)
            untested = sorted((reference & set(targets)) - seen)
            if absent or untested:
                rep.add("observations.class_vocabulary", "observations", "warn",
                        f"Observation classes absent from the training set: {absent}; target classes without any "
                        f"observation (no independent validation): {untested}.")
    if date_f in raw and present:
        text = raw[date_f].astype(str).str.strip()
        parsed = pd.to_datetime(text.where(text.str.match(r"^\d{4}-\d{2}-\d{2}")), errors="coerce", format="ISO8601",
                                utc=True).dt.tz_localize(None)
        bad = parsed.isna()
        dmy = [t for t in text[bad] if parse_date(t) is not None]
        if bad.any():
            rep.add("observations.date_format", "observations", "error",
                    f"{int(bad.sum())} row(s) have an observed_at that is not an ISO date (YYYY-MM-DD)"
                    + (f"; {len(dmy)} look like day.month.year, convert them with `geocrop normalize-observations`"
                       if dmy else "") + f". Examples of the shape: {sorted({re.sub(r'[0-9]', '9', t) for t in text[bad]})[:5]}",
                    count=int(bad.sum()))
            rep.ids("observations.date_format", list(raw.loc[bad, "_row"]))
        else:
            rep.add("observations.date_format", "observations", "pass", "Every observed_at is an ISO date.")
        good = parsed[~bad]
        if len(good):
            start, end = season
            outside = (good < start) | (good > end)
            rep.add("observations.season", "observations", "warn" if outside.any() else "pass",
                    f"{int(outside.sum())} observation(s) are outside the study season {start.date()} to {end.date()} "
                    f"(observed {good.min().date()} to {good.max().date()})." if outside.any()
                    else f"All dates lie in the study season {start.date()} to {end.date()} "
                         f"(observed {good.min().date()} to {good.max().date()}).",
                    outside=int(outside.sum()), observed_from=str(good.min().date()), observed_to=str(good.max().date()))
            rep.ids("observations.season", list(raw.loc[good.index[outside], "_row"]))
            if v["year"] is not None:
                wrong = int((good.dt.year != int(v["year"])).sum())
                if wrong:
                    rep.add("observations.year", "observations", "warn",
                            f"{wrong} observation(s) are not in validation.year={v['year']}.", count=wrong)
    if group and group in raw:
        blank = int((raw[group].isna() | (raw[group].astype(str).str.strip() == "")).sum())
        sizes = raw[group].astype(str).str.strip().value_counts()
        rep.add("observations.groups", "observations", "error" if blank else "pass",
                f"{blank} observation(s) have an empty {group!r}; each needs a field/block identifier." if blank
                else f"{len(sizes)} validation group(s); largest has {int(sizes.max())} observation(s).",
                groups=int(len(sizes)), blank=blank)
    elif not group:
        rep.add("observations.groups", "observations", "pass",
                "validation.group_field is not set: points are treated as independent. If several points lie in the "
                "same field, add a group field and set it, otherwise accuracy intervals are too narrow.")
    work = layer.work
    if len(work) and label in raw and date_f in raw:
        key = pd.DataFrame({"w": work.geometry.to_wkb(), "c": work[label].astype(str).str.strip(),
                            "d": work[date_f].astype(str).str.strip()})
        dup = key.duplicated(keep="first")
        rep.add("observations.duplicates", "observations", "warn" if dup.any() else "pass",
                f"{int(dup.sum())} observation(s) repeat the point, class and date of another row." if dup.any()
                else "No duplicate observations (same point, class and date).", count=int(dup.sum()))
        rep.ids("observations.duplicates", list(work.loc[dup[dup].index, "_row"]))
    return table


def _season(cfg) -> tuple[pd.Timestamp, pd.Timestamp]:
    ts = cfg["timeseries"]
    if ts["enabled"] and ts["start"] and ts["end"]:
        return pd.Timestamp(str(ts["start"])), pd.Timestamp(str(ts["end"]))
    win = cfg.get("windows") or {}
    starts = [pd.Timestamp(str(w["start"])) for w in win.values()]
    ends = [pd.Timestamp(str(w["end"])) for w in win.values()]
    if starts:
        return min(starts), max(ends)
    year = int(cfg["validation"]["year"] or datetime.now().year)
    return pd.Timestamp(f"{year}-01-01"), pd.Timestamp(f"{year}-12-31")


def _leakage_offenders(rep, cfg, training: _Layer, obs: _Layer):
    """Row numbers of observations within the exclusion buffer of a training polygon (listing only)."""
    metres = float(cfg["validation"]["training_exclusion_m"] or 0)
    t, o = training.work, obs.work
    if not len(t) or not len(o):
        return
    zones = t.geometry.buffer(metres) if metres else t.geometry
    index = zones.sindex
    hit = [row for row, pt in zip(o["_row"], o.geometry) if len(index.query(pt, predicate="intersects"))]
    if hit:
        rep.add("observations.training_overlap", "observations", "warn",
                f"{len(hit)} observation(s) lie inside a training polygon or within training_exclusion_m="
                f"{metres:g} m of one; exclude them from independent validation.", count=len(hit))
        rep.ids("observations.training_overlap", hit)
    else:
        rep.add("observations.training_overlap", "observations", "pass",
                f"No observation lies within {metres:g} m of a training polygon.")


def run(cfg, config_path=None, training=None, parcels=None, observations=None, crop_mapping=None, aoi=None,
        municipalities=None, list_offenders: int = 20, inner_buffer_m: float = 0.0, require_all: bool = False,
        read_options: dict | None = None) -> dict:
    """Intake report for the supplied layers (default: the existing files of ``paths.*`` of the configuration).

    `read_options`: per layer ('training', 'parcels', 'observations') keyword arguments of
    :func:`geocrop.adapters.read_layer` (wkt_field, x_field, y_field, crs, encoding, delimiter)."""
    from .audit import audit_inputs
    rep = _Report(list_offenders)
    read_options = read_options or {}
    crs = cfg["grid"]["crs"]
    if not crs:
        raise ValueError("Intake needs grid.crs in the study configuration (a projected CRS in metres, e.g. EPSG:32633)")
    targets = list(cfg["classes"]["targets"])
    sources = {"training": training, "parcels": parcels, "observations": observations}
    for name in LAYERS:
        if sources[name] is None and Path(cfg["paths"][name]).exists():
            sources[name] = cfg["paths"][name]
    layers: dict[str, _Layer] = {}
    for name in LAYERS:
        if sources[name] is None:
            rep.add(f"{name}.supplied", name, "error" if require_all else "warn",
                    f"No {name} layer supplied (and {cfg['paths'][name]} does not exist); not checked.")
        elif not Path(sources[name]).exists():
            rep.add(f"{name}.read", name, "error", f"File not found: {sources[name]}")
        else:
            layer = _load(name, sources[name], read_options.get(name, {}), rep)
            if layer is not None:
                layers[name] = layer
    try:
        aoi_series, expected, aoi_desc = _aoi_geometry(cfg, aoi, config_path)
    except Exception as exc:  # noqa: BLE001
        aoi_series, expected, aoi_desc = None, [], None
        rep.add("aoi.read", "study", "warn", f"AOI could not be read: {exc}")
    if aoi_desc is None:
        rep.add("aoi.available", "study", "warn", "No AOI file found (--aoi or aoi.geojson next to the configuration); "
                "the AOI overlap check is skipped.")
    boundaries = None
    bpath = _find_boundaries(cfg, municipalities, config_path)
    if bpath is not None:
        try:
            boundaries = gpd.read_file(bpath).to_crs(crs)
            name_col = next((c for c in boundaries.columns if fold(c) in BOUNDARY_NAME_FIELDS), None)
            if name_col is None:
                raise ValueError("no name column (municipality/opstina/name/shapeName)")
            boundaries = boundaries.rename(columns={name_col: "_name"})[["_name", "geometry"]]
        except Exception as exc:  # noqa: BLE001
            boundaries = None
            rep.add("municipalities.read", "study", "warn", f"Municipality boundaries {bpath.name} unusable: {exc}")

    # 1. existing audit (schema, validity, IDs, overlaps, leakage) on the unmodified layers
    if layers:
        from .preflight import _metric
        grid_crs = crs if _metric(crs) else None
        audit = audit_inputs(**{n: layer.raw.drop(columns="_row") for n, layer in layers.items()},
                             label_field=cfg["classes"]["label_field"], id_field=cfg["classes"]["id_field"],
                             parcel_id_field=cfg["parcels"]["id_field"],
                             observation_label_field=cfg["validation"]["label_field"],
                             training_exclusion_m=cfg["validation"]["training_exclusion_m"], distance_crs=grid_crs)
        for c in audit["checks"]:
            if c["status"] == "fail":
                rep.add(f"audit.{c['layer']}.{c['code']}", c["layer"], "error" if c["severity"] == "error" else "warn",
                        c["message"], **({"count": c["count"]} if "count" in c else {}))
        rep.add("audit.summary", "study", "pass",
                f"geocrop audit: {audit['summary']['passed']} passed, {audit['summary']['errors']} error(s), "
                f"{audit['summary']['warnings']} warning(s).", **audit["summary"])

    for name, layer in layers.items():
        _prepare(layer, crs, rep, POLYGONS if name != "observations" else ("Point",))

    aoi_geom = None
    if aoi_series is not None:
        aoi_geom = aoi_series.to_crs(crs).union_all() if hasattr(aoi_series, "union_all") else aoi_series.to_crs(crs).unary_union
    classes: dict = {}
    training_classes: list[str] = []
    if "training" in layers:
        layer = layers["training"]
        table, support, assumed = _training_checks(rep, cfg, layer, targets, inner_buffer_m, INDICATIVE_HOLDOUT,
                                                   layer.raw.drop(columns="_row"))
        classes["training"] = {"per_class": table, "support": support, "indicative_holdout": assumed}
        training_classes = list(table)
        _aoi_overlap(rep, layer, aoi_geom, False)
        _municipality(rep, cfg, layer, boundaries, expected, cfg["classes"]["label_field"])
        cv = cfg["cv"]
        if cv["strategy"] in ("region", "region_year"):
            rf = cv["region_field"]
            if not rf or rf not in layer.raw:
                rep.add("training.region_field", "training", "error",
                        f"cv.strategy={cv['strategy']!r} needs the training field {rf!r} (e.g. municipality).")
            elif (layer.raw[rf].isna() | (layer.raw[rf].astype(str).str.strip() == "")).any():
                rep.add("training.region_field", "training", "error", f"Field {rf!r} has empty values.")
    if "parcels" in layers:
        layer = layers["parcels"]
        mapping_path = crop_mapping or cfg["parcels"].get("crop_mapping")
        rows = None
        if mapping_path:
            try:
                rows = read_mapping_rows(mapping_path)
                rep.add("parcels.mapping_read", "parcels", "pass", f"Crop mapping {Path(mapping_path).name}: {len(rows)} row(s).")
            except Exception as exc:  # noqa: BLE001
                rep.add("parcels.mapping_read", "parcels", "error", f"Cannot read crop mapping {mapping_path}: {exc}")
        else:
            rep.add("parcels.mapping_read", "parcels", "warn", "No crop mapping supplied (--crop-mapping or parcels.crop_mapping).")
        classes["parcels"] = {"declared": _parcel_checks(rep, cfg, layer, rows, targets, training_classes,
                                                         cfg["parcels"]["interior_buffer_px"])}
        _aoi_overlap(rep, layer, aoi_geom, False)
        _municipality(rep, cfg, layer, boundaries, expected, area=True)
        if "training" in layers:
            _overlap_training_parcels(rep, layers["training"], layer)
    if "observations" in layers:
        layer = layers["observations"]
        classes["observations"] = {"per_class": _observation_checks(rep, cfg, layer, targets, training_classes,
                                                                    _season(cfg), 0)}
        _aoi_overlap(rep, layer, aoi_geom, True)
        _municipality(rep, cfg, layer, boundaries, expected, cfg["validation"]["label_field"])
        if "training" in layers:
            _leakage_offenders(rep, cfg, layers["training"], layer)

    counts = {s: sum(c["status"] == s for c in rep.checks) for s in ("pass", "warn", "error")}
    verdict = "error" if counts["error"] else ("warn" if counts["warn"] else "pass")
    return {"schema": SCHEMA, "verdict": verdict, "engine": __version__,
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "config": str(config_path) if config_path else None, "config_fingerprint": cfgmod.fingerprint(cfg),
            "layers": {n: {"source": layer.source, "features": int(len(layer.raw)),
                           "crs": str(layer.raw.crs) if layer.raw.crs else None, **layer.info,
                           "invalid_geometries": layer.repair["invalid"], "repairable": layer.repair["repairable"]}
                       for n, layer in layers.items()},
            "aoi": aoi_desc, "summary": counts, "checks": rep.checks, "classes": classes,
            "offenders": rep.offenders, "offenders_note": "Row numbers are 1-based positions in the file; IDs only up to "
            f"the --list-offenders limit ({list_offenders}). No attributes or geometries are reported.",
            "scope": SCOPE}


def to_markdown(report: dict) -> str:
    badge = {"pass": "PASS", "warn": "PASS WITH WARNINGS", "error": "FAIL"}[report["verdict"]]
    s = report["summary"]
    lines = [f"# GeoCrop intake: {badge}", "",
             f"Config `{report['config']}` (fingerprint `{report['config_fingerprint']}`), engine {report['engine']}, "
             f"{report['generated_at']}. AOI: {report['aoi'] or 'not available'}.", "",
             f"**{s['error']} error(s), {s['warn']} warning(s), {s['pass']} passed.**", ""]
    if report["layers"]:
        lines += ["## Layers", "", "| layer | file | features | CRS | invalid | repairable |", "|---|---|---|---|---|---|"]
        for n, v in report["layers"].items():
            lines.append(f"| {n} | {Path(v['source']).name} | {v['features']} | {v['crs']} | "
                         f"{v['invalid_geometries']} | {v['repairable']} |")
        lines.append("")
    for status, title in (("error", "Errors (fix before the run)"), ("warn", "Warnings (review)")):
        items = [c for c in report["checks"] if c["status"] == status]
        if items:
            lines += [f"## {title}", ""]
            for c in items:
                lines.append(f"- `{c['code']}`: {c['message']}")
                ids = report["offenders"].get(c["code"])
                if ids:
                    lines.append(f"  - first {len(ids)} (IDs/rows): {ids}")
                nm = c.get("details", {}).get("normalisation_map")
                if nm:
                    lines += [f"  - `{k}` -> `{v}`" for k, v in list(nm.items())[:20]]
            lines.append("")
    train = report["classes"].get("training", {}).get("per_class")
    if train:
        lines += ["## Training classes", "", "| class | polygons | area ha | pixels | expected samples | min px | < 1 px |",
                  "|---|---|---|---|---|---|---|"]
        lines += [f"| {c} | {v['polygons']} | {v['area_ha']} | {v['pixels']} | {v['expected_samples']} | "
                  f"{v['min_pixels']} | {v['polygons_below_1px']} |" for c, v in train.items()]
        lines.append("")
    for layer in ("parcels", "observations"):
        table = report["classes"].get(layer, {})
        rows = table.get("declared") or table.get("per_class")
        if rows:
            lines += [f"## {layer.capitalize()} classes", "", "| class | count |", "|---|---|"]
            lines += [f"| {c} | {v['observations'] if isinstance(v, dict) else v} |" for c, v in rows.items()]
            lines.append("")
    for c in report["checks"]:
        pm = c.get("details", {}).get("per_municipality")
        if pm:
            lines += [f"## Per municipality: {c['layer']}", "", "| municipality | features | area ha |", "|---|---|---|"]
            lines += [f"| {m} | {v['features']} | {v.get('area_ha', '')} |" for m, v in pm.items()]
            lines.append("")
    notes = [c for c in report["checks"] if c.get("details", {}).get("skipped")]
    if notes:
        lines += ["## Notes", ""] + [f"- `{c['code']}`: {c['message']}" for c in notes] + [""]
    lines += [f"_{report['scope']}_", "", f"_{report['offenders_note']}_", ""]
    return "\n".join(lines)


def write(report: dict, out_dir: str | Path, inputs=()) -> tuple[Path, Path]:
    """Write intake_report.json and intake_report.md; refuses to overwrite an input file."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    targets = (out / "intake_report.json", out / "intake_report.md")
    for t in targets:
        if any(Path(i).resolve() == t.resolve() for i in inputs if i):
            raise ValueError("The intake output must not overwrite an input file")
    targets[0].write_text(json.dumps(report, indent=2, ensure_ascii=False, default=_jsonable, allow_nan=False),
                          encoding="utf8")
    targets[1].write_text(to_markdown(report), encoding="utf8")
    return targets


def _jsonable(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (set, tuple)):
        return list(value)
    return str(value)
