"""Neutral schema adapters: map any provider's parcel / reference fields to the engine contract."""
from __future__ import annotations

import csv
import io
import re
from pathlib import Path

import geopandas as gpd
import pandas as pd

TABLE_SUFFIXES = (".csv", ".txt", ".tsv")
ENCODINGS = ("utf-8", "cp1250")
_LAT = ("lat", "latitude", "y_wgs84", "y")
_LON = ("lon", "lng", "long", "longitude", "x_wgs84", "x")


def detect_encoding(path: str | Path, explicit: str | None = None) -> str:
    """Encoding of a text export: `explicit` when given, UTF-8 (with or without BOM) when the whole file decodes
    as such, otherwise cp1250 (Serbian/Central-European Windows exports). Detection is a heuristic; pass an explicit
    encoding when the diacritics look wrong."""
    if explicit:
        return explicit
    raw = Path(path).read_bytes()
    try:
        raw.decode("utf-8")
        return "utf-8-sig" if raw.startswith(b"\xef\xbb\xbf") else "utf-8"
    except UnicodeDecodeError:
        pass
    try:
        raw.decode("cp1250")
        return "cp1250"
    except UnicodeDecodeError:
        return "latin-1"


def detect_delimiter(text: str) -> str:
    """Delimiter of the header line: the most frequent of ; , tab | (outside double quotes)."""
    header = text.splitlines()[0] if text else ""
    header = re.sub(r'"[^"]*"', "", header)
    counts = {d: header.count(d) for d in (";", ",", "\t", "|")}
    best = max(counts, key=counts.get)
    return best if counts[best] else ","


def read_table(path: str | Path, encoding: str | None = None, delimiter: str | None = None) -> tuple[pd.DataFrame, dict]:
    """Read a CSV export as strings (no type guessing: codes keep leading zeros) plus the detected settings."""
    enc = detect_encoding(path, encoding)
    with Path(path).open(encoding=enc, newline="") as fh:
        text = fh.read()
    sep = delimiter or detect_delimiter(text)
    frame = pd.read_csv(io.StringIO(text), sep=sep, dtype=str, keep_default_na=False, skipinitialspace=False)
    frame.columns = [str(c).strip().lstrip("\ufeff") for c in frame.columns]
    return frame, {"encoding": enc, "delimiter": sep, "encoding_detected": encoding is None}


def _number(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.strip().str.replace(",", ".", regex=False), errors="coerce")


def table_to_geo(frame: pd.DataFrame, wkt_field: str | None = None, x_field: str | None = None,
                 y_field: str | None = None, crs: str | None = None) -> gpd.GeoDataFrame:
    """Geometry from a WKT column or from x/y (lon/lat) columns; decimal commas are accepted.

    Without field names, a `wkt`/`geometry` column or lat/lon-style columns are looked up. The CRS is required for
    WKT and projected x/y; lat/lon column names default to EPSG:4326."""
    from shapely import wkt as shapely_wkt
    lower = {c.lower(): c for c in frame.columns}
    if not (wkt_field or x_field or y_field):
        wkt_field = lower.get("wkt") or lower.get("geometry") or lower.get("geom")
        if not wkt_field:
            lon = next((lower[n] for n in _LON if n in lower), None)
            lat = next((lower[n] for n in _LAT if n in lower), None)
            if lon and lat:
                x_field, y_field = lon, lat
                crs = crs or ("EPSG:4326" if lon.lower() not in ("x",) else None)
    if wkt_field:
        if wkt_field not in frame:
            raise ValueError(f"Missing WKT column {wkt_field!r} (columns: {list(frame.columns)[:20]})")
        if not crs:
            raise ValueError("A table with WKT geometry needs --crs (e.g. EPSG:32633)")
        blank = frame[wkt_field].str.strip() == ""
        geoms = [None if b else shapely_wkt.loads(v) for v, b in zip(frame[wkt_field], blank)]
        return gpd.GeoDataFrame(frame.drop(columns=[wkt_field]), geometry=geoms, crs=crs)
    if x_field and y_field:
        for f in (x_field, y_field):
            if f not in frame:
                raise ValueError(f"Missing coordinate column {f!r} (columns: {list(frame.columns)[:20]})")
        if not crs:
            raise ValueError("x/y columns need --crs (EPSG:4326 for lon/lat, EPSG:32633 for UTM 33N metres)")
        x, y = _number(frame[x_field]), _number(frame[y_field])
        bad = x.isna() | y.isna()
        if bad.any():
            raise ValueError(f"{int(bad.sum())} rows have a missing or non-numeric coordinate in "
                             f"{x_field!r}/{y_field!r} (decimal comma or dot are accepted, nothing else)")
        return gpd.GeoDataFrame(frame.drop(columns=[x_field, y_field]), geometry=gpd.points_from_xy(x, y), crs=crs)
    raise ValueError("Table has no geometry: pass --wkt-field, or --x-field/--y-field with --crs")


def read_layer(path: str | Path, wkt_field: str | None = None, x_field: str | None = None, y_field: str | None = None,
               crs: str | None = None, encoding: str | None = None, delimiter: str | None = None
               ) -> tuple[gpd.GeoDataFrame, dict]:
    """Read a vector layer (GPKG, GeoJSON, Shapefile, FlatGeobuf via GDAL/pyogrio; GeoParquet) or a CSV table with
    WKT or x/y columns. Returns the frame and a small description (format, encoding, delimiter)."""
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix in TABLE_SUFFIXES:
        frame, info = read_table(p, encoding, delimiter)
        return table_to_geo(frame, wkt_field, x_field, y_field, crs), {"format": "table", **info}
    if suffix in (".parquet", ".geoparquet"):
        try:
            return gpd.read_file(p), {"format": "GeoParquet"}
        except Exception:  # noqa: BLE001 - GDAL builds without the Parquet driver
            return gpd.read_parquet(p), {"format": "GeoParquet"}
    kwargs = {"encoding": encoding} if encoding and suffix == ".shp" else {}
    return gpd.read_file(p, **kwargs), {"format": suffix.lstrip(".") or "unknown"}


def clean_code(value) -> str:
    """Declaration code as text: trimmed, spreadsheet float artefacts removed ('1101.0' -> '1101')."""
    text = str(value).strip()
    return text[:-2] if re.fullmatch(r"\d+\.0", text) else text


def read_mapping_rows(path: str | Path) -> list[tuple[str, str]]:
    """Raw (source_code, class_name) pairs of a mapping CSV in file order (header removed, codes cleaned)."""
    enc = detect_encoding(path)
    with Path(path).open(encoding=enc, newline="") as fh:
        text = fh.read()
    rows = list(csv.reader(io.StringIO(text), delimiter=detect_delimiter(text)))
    if rows:
        rows[0] = [c.lstrip("\ufeff") for c in rows[0]]
    if rows and rows[0][:2] == ["source", "class_name"]:
        rows = rows[1:]
    return [(clean_code(r[0]), r[1].strip()) for r in rows if len(r) >= 2 and r[0].strip()]


def read_mapping(path: str | Path | None) -> dict:
    """Two-column CSV (source_code,class_name) mapping declaration or field codes to model classes.

    UTF-8 or cp1250 (detected); comma or semicolon delimited. A repeated code keeps its last class."""
    if not path:
        return {}
    return dict(read_mapping_rows(path))


def _write(out: gpd.GeoDataFrame, output: str | Path) -> Path:
    dest = Path(output)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.unlink(missing_ok=True)
    out.to_file(dest, driver="GPKG" if dest.suffix.lower() == ".gpkg" else "GeoJSON")
    return dest


def normalize_parcels(source: str | Path, output: str | Path, id_field: str, declared_field: str | None = None,
                      mapping: dict | None = None, pseudonymise: bool = False, make_valid: bool = True,
                      wkt_field: str | None = None, x_field: str | None = None, y_field: str | None = None,
                      crs: str | None = None, encoding: str | None = None, delimiter: str | None = None) -> dict:
    """Write parcels as `parcel_id` (+ `declared_crop`) GeoPackage.

    `source` is a vector layer or a CSV table (WKT column or x/y columns, see :func:`read_layer`).
    pseudonymise=True replaces IDs with sequential codes and writes the lookup next to the output
    (keep that lookup private); geometries are repaired with make_valid when requested. Declaration codes
    absent from `mapping` are kept unchanged and counted in `unmapped_codes`.
    """
    src, info = read_layer(source, wkt_field, x_field, y_field, crs, encoding, delimiter)
    if id_field not in src:
        raise ValueError(f"Missing parcel ID field {id_field!r}")
    if src.crs is None:
        raise ValueError("Parcels need a CRS")
    out = gpd.GeoDataFrame({"parcel_id": src[id_field].astype(str).str.strip()}, geometry=src.geometry, crs=src.crs)
    if out.parcel_id.duplicated().any():
        raise ValueError("Duplicate parcel IDs")
    unmapped = 0
    if declared_field:
        if declared_field not in src:
            raise ValueError(f"Missing declaration field {declared_field!r}")
        codes = src[declared_field].map(clean_code)
        known = mapping or {}
        unmapped = int((~codes.isin(set(known) | set(known.values()))).sum()) if known else 0
        out["declared_crop"] = codes.map(lambda c: known.get(c, c))
    repaired = 0
    if make_valid:
        bad = ~out.geometry.is_valid & out.geometry.notna()
        repaired = int(bad.sum())
        if repaired:
            out.loc[bad, "geometry"] = out.loc[bad, "geometry"].make_valid()
    if pseudonymise:
        dest = Path(output)
        lookup = dest.with_name(dest.stem + "_id_lookup.csv")
        codes = [f"P{i:07d}" for i in range(len(out))]
        out[["parcel_id"]].assign(pseudonym=codes).to_csv(lookup, index=False)
        out["parcel_id"] = codes
    dest = _write(out, output)
    return {"path": str(dest), "parcels": int(len(out)), "repaired_geometries": repaired,
            "pseudonymised": pseudonymise, "unmapped_codes": unmapped, "input": info}


def normalize_observations(source: str | Path, output: str | Path, label_field: str, date_field: str,
                           group_field: str | None = None, wkt_field: str | None = None,
                           x_field: str | None = None, y_field: str | None = None, crs: str | None = None,
                           encoding: str | None = None, delimiter: str | None = None) -> dict:
    """Write field observations as `class_name`, `observed_at` (ISO date) and optional `group_id` GeoPackage.

    Dates written as day.month.year (`15.06.2023.`, `15/06/2023`) are converted to ISO; anything else that is not
    an ISO date is refused with the row count, never guessed."""
    src, info = read_layer(source, wkt_field, x_field, y_field, crs, encoding, delimiter)
    for f in (label_field, date_field, group_field):
        if f and f not in src:
            raise ValueError(f"Missing column {f!r} (columns: {list(src.columns)[:20]})")
    if src.crs is None:
        raise ValueError("Observations need a CRS")
    dates = src[date_field].map(parse_date)
    bad = dates.isna()
    if bad.any():
        raise ValueError(f"{int(bad.sum())} rows have a date that is neither ISO (2023-06-15) nor day.month.year")
    out = gpd.GeoDataFrame({"class_name": src[label_field].astype(str).str.strip(), "observed_at": dates},
                           geometry=src.geometry, crs=src.crs)
    if group_field:
        out["group_id"] = src[group_field].astype(str).str.strip()
    dest = _write(out, output)
    return {"path": str(dest), "observations": int(len(out)), "input": info}


_DMY = re.compile(r"^\s*(\d{1,2})[./-](\d{1,2})[./-](\d{4})\.?\s*$")


def parse_date(value) -> str | None:
    """ISO date text (YYYY-MM-DD, optionally with a time part) or day.month.year converted to ISO; else None."""
    text = str(value).strip()
    m = _DMY.match(text)
    try:
        if m:
            return pd.Timestamp(year=int(m[3]), month=int(m[2]), day=int(m[1])).date().isoformat()
        if not re.match(r"^\d{4}-\d{2}-\d{2}", text):
            return None
        return pd.Timestamp(pd.to_datetime(text, format="ISO8601")).date().isoformat()
    except (ValueError, TypeError):
        return None
