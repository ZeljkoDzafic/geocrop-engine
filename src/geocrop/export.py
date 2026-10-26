"""Interoperable publication of results: Cloud-Optimised GeoTIFF, GeoParquet, STAC metadata, HCAT codes."""
from __future__ import annotations

import csv
import json
import math
import os
import re
import tempfile
from datetime import date, datetime, timezone
from pathlib import Path

import rasterio
from rasterio.shutil import copy as rio_copy

from . import attribution


def to_cog(src: str | Path, dst: str | Path, overview_resampling: str = "nearest") -> Path:
    """Copy a GeoTIFF as a Cloud-Optimised GeoTIFF (GDAL COG driver, deflate, internal overviews).

    Use `nearest`/`mode` for class rasters and `average` for continuous layers. Dataset tags, including the
    Copernicus attribution (``GEOCROP_ATTRIBUTION``), are copied into the COG."""
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    rio_copy(str(src), str(dst), driver="COG", COMPRESS="DEFLATE", OVERVIEW_RESAMPLING=overview_resampling.upper(),
             BIGTIFF="IF_SAFER")
    return dst


def to_geoparquet(vector: str | Path, dst: str | Path) -> Path:
    import geopandas as gpd

    try:
        import pyarrow  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("pip install pyarrow") from exc
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    gpd.read_file(vector).to_parquet(dst, index=False)
    return dst


def read_hcat(path: str | Path) -> dict:
    """CSV class_name,hcat_code,hcat_name mapping model classes to the EuroCrops HCAT taxonomy."""
    with Path(path).open(encoding="utf8") as fh:
        return {r["class_name"]: {"code": r.get("hcat_code") or None, "name": r.get("hcat_name") or None}
                for r in csv.DictReader(fh)}


CLASSIFICATION_SCHEMA = "https://stac-extensions.github.io/classification/v2.0.0/schema.json"
PROJECTION_SCHEMA = "https://stac-extensions.github.io/projection/v1.1.0/schema.json"
RASTER_SCHEMA = "https://stac-extensions.github.io/raster/v1.1.0/schema.json"


def _relative_asset(path: Path, base: Path) -> str:
    if not path.is_file():
        raise ValueError(f"STAC asset does not exist: {path}")
    return Path(os.path.relpath(path.resolve(), base.resolve())).as_posix()


def _number(value):
    """JSON-safe raster nodata, using Raster Extension spellings for nonfinite values."""
    if value is None or math.isfinite(value):
        return value
    return "nan" if math.isnan(value) else "inf" if value > 0 else "-inf"


def _raster_asset(path: Path, base: Path, reference, names=None) -> dict:
    with rasterio.open(path) as ds:
        if ds.crs != reference.crs or ds.shape != reference.shape or not ds.transform.almost_equals(reference.transform):
            raise ValueError(f"Prediction asset is not on the classes grid: {path.name}")
        if names is not None and ds.count != len(names):
            raise ValueError(f"Wrong band count for {path.name}: expected {len(names)}, got {ds.count}")
        bands = []
        for i in range(ds.count):
            band = {"data_type": ds.dtypes[i], "scale": ds.scales[i], "offset": ds.offsets[i]}
            if ds.nodatavals[i] is not None:
                band["nodata"] = _number(ds.nodatavals[i])
            description = ds.descriptions[i] or (names[i] if names else None)
            if description:
                band["name"] = description
            if names:
                band["description"] = names[i]
            bands.append(band)
        return {"href": _relative_asset(path, base), "type": "image/tiff; application=geotiff",
                "roles": ["data"], "raster:bands": bands}


def _class_definitions(legend: dict, hcat: dict | None) -> list[dict]:
    if not isinstance(legend, dict) or not legend:
        raise ValueError("class_legend.json must be a nonempty value-to-label mapping")
    try:
        values = [int(k) for k in legend]
    except (TypeError, ValueError) as exc:
        raise ValueError("Class legend keys must be integer pixel values") from exc
    if len(set(values)) != len(values) or any(str(int(k)) != k for k in legend):
        raise ValueError("Class legend keys must be unique canonical integer values")
    labels = list(legend.values())
    if not all(isinstance(label, str) and label.strip() for label in labels) or len(set(labels)) != len(labels):
        raise ValueError("Class legend labels must be nonempty and unique")
    classes, used = [], set()
    for value in sorted(values):
        label = legend[str(value)]
        # Classification v2.0 names are machine identifiers. Preserve arbitrary model labels in title.
        name = re.sub(r"[^0-9A-Za-z_-]+", "_", label).strip("_") or f"class_{value}"
        if name in used:
            name = f"{name}_{value}"
        while name in used:
            name += "_"
        used.add(name)
        entry = {"value": value, "name": name, "title": label}
        if hcat is not None:
            mapping = hcat.get(label)
            if not isinstance(mapping, dict) or not str(mapping.get("code") or "").strip():
                raise ValueError(f"HCAT mapping lacks a code for class {label!r}")
            entry["description"] = f"HCAT {mapping['code']} {mapping.get('name') or ''}".strip()
        classes.append(entry)
    return classes


def _footprint(ds):
    """Transform the actual affine pixel perimeter, including rotated grids, to WGS84."""
    from rasterio.warp import transform_geom
    from shapely.geometry import shape

    corners = [(0, 0), (ds.width, 0), (ds.width, ds.height), (0, ds.height), (0, 0)]
    ring = []
    for (x0, y0), (x1, y1) in zip(corners, corners[1:]):
        for k in range(21):
            x, y = x0 + (x1 - x0) * k / 21, y0 + (y1 - y0) * k / 21
            t = ds.transform
            ring.append([t.a * x + t.b * y + t.c, t.d * x + t.e * y + t.f])
    ring.append(ring[0])
    geometry = json.loads(json.dumps(transform_geom(ds.crs, "EPSG:4326",
                                                   {"type": "Polygon", "coordinates": [ring]})))
    footprint = shape(geometry)
    bounds = list(footprint.bounds)
    if footprint.is_empty or not footprint.is_valid or not all(math.isfinite(v) for v in bounds):
        raise ValueError("Cannot derive a valid geographic prediction footprint")
    return geometry, bounds


def stac_item(prediction_dir: str | Path, item_id: str, start: str, end: str, output: str | Path | None = None,
              hcat: dict | None = None, extra_assets: dict | None = None) -> dict:
    """Build STAC 1.0 classification/projection/raster metadata from actual prediction assets.

    Classes and legend are required; other rasters are included only when present and must share
    the classes grid. HREFs are relative to the Item file's directory (or prediction_dir when no
    file is written), so external Item output locations remain resolvable. The export CLI places
    Item and copied COGs together for a self-contained directory. Additional local assets are
    explicit; remote URLs are not fetched or silently copied. This routine checks structural data
    contracts, not JSON Schema; official schema validation is a separate optional development check.
    """
    if not isinstance(item_id, str) or not item_id.strip():
        raise ValueError("STAC Item ID must be nonempty")
    try:
        first, last = date.fromisoformat(start), date.fromisoformat(end)
    except (TypeError, ValueError) as exc:
        raise ValueError("start and end must be ISO calendar dates") from exc
    if first > last:
        raise ValueError("start must not be after end")
    pred = Path(prediction_dir)
    target = Path(output) if output is not None else None
    base = target.parent if target is not None else pred
    legend_path = pred / "class_legend.json"
    legend = json.loads(legend_path.read_text(encoding="utf8"))
    classes = _class_definitions(legend, hcat)
    class_path = pred / "classes.tif"
    inputs = [legend_path, class_path]
    with rasterio.open(class_path) as ds:
        if ds.crs is None or ds.count != 1 or not ds.dtypes[0].startswith(("int", "uint")):
            raise ValueError("classes.tif must be a one-band integer raster with a CRS")
        import numpy as np
        limits = np.iinfo(ds.dtypes[0])
        if any(c["value"] < limits.min or c["value"] > limits.max or c["value"] == ds.nodata for c in classes):
            raise ValueError("Legend pixel values must fit the classes dtype and differ from nodata")
        geometry, bbox = _footprint(ds)
        assets = {"classes": _raster_asset(class_path, base, ds, ["Crop class index"])}
        assets["classes"]["classification:classes"] = classes
        assets["classes"]["raster:bands"][0]["classification:classes"] = classes
        names = {"probabilities": [legend[str(c["value"])] for c in classes],
                 "uncertainty": ["max_probability", "margin", "entropy"],
                 "aoa": ["dissimilarity_index", "inside_aoa"]}
        for key, band_names in names.items():
            path = pred / f"{key}.tif"
            if path.is_file():
                if key == "probabilities":
                    with rasterio.open(path) as probability:
                        if list(probability.descriptions) != band_names:
                            raise ValueError("Probability band descriptions must match the class legend order")
                assets[key] = _raster_asset(path, base, ds, band_names)
                if key == "aoa":
                    assets[key]["roles"] = ["data", "metadata"]
                inputs.append(path)
        properties = {"datetime": None, "start_datetime": f"{first.isoformat()}T00:00:00Z",
                      "end_datetime": f"{last.isoformat()}T23:59:59Z",
                      "proj:epsg": ds.crs.to_epsg(), "proj:shape": [ds.height, ds.width],
                      "proj:transform": list(ds.transform)[:6], "proj:bbox": list(ds.bounds),
                      "created": datetime.now(timezone.utc).isoformat(),
                      "processing:software": {"geocrop-engine": _ver()}}
        if properties["proj:epsg"] is None:
            properties["proj:wkt2"] = ds.crs.to_wkt(version="WKT2_2019")
        credit = ds.tags().get(attribution.TAG)
    links = []
    if credit:          # derived from Copernicus Sentinel data: attribution, licence link and providers
        properties.update({
            "license": "other", "attribution": credit,
            "providers": [{"name": "European Union Copernicus programme / ESA (Sentinel data)",
                           "roles": ["producer", "licensor"], "url": "https://www.copernicus.eu",
                           "description": credit},
                          {"name": "GeoCrop Engine", "roles": ["processor"],
                           "url": "https://github.com/ZeljkoDzafic/geocrop-engine"}]})
        links.append({"rel": "license", "href": attribution.COPERNICUS_LICENCE.rsplit(" ", 1)[-1],
                      "title": "Copernicus Sentinel data legal notice"})
    for key, filename in (("class_legend", "class_legend.json"), ("run_manifest", "run_manifest.json"),
                          ("train_report", "train_report.json"), ("model_card", "model_card.json")):
        path = pred / filename
        if path.is_file():
            assets[key] = {"href": _relative_asset(path, base), "type": "application/json", "roles": ["metadata"]}
            inputs.append(path)
    for key, asset in (extra_assets or {}).items():
        if key in assets or not isinstance(asset, dict) or not isinstance(asset.get("href"), str):
            raise ValueError(f"Invalid or duplicate extra asset: {key}")
        raw = asset["href"]
        if "://" in raw:
            raise ValueError("Additional STAC assets must be existing local files, not remote URLs")
        path = Path(raw) if Path(raw).is_absolute() else base / raw
        assets[key] = {**asset, "href": _relative_asset(path, base)}
        inputs.append(path)
    item = {"type": "Feature", "stac_version": "1.0.0", "id": item_id,
            "stac_extensions": [CLASSIFICATION_SCHEMA, PROJECTION_SCHEMA, RASTER_SCHEMA],
            "bbox": bbox, "geometry": geometry, "properties": properties, "assets": assets, "links": links}
    if target is not None:
        if any(target.resolve() == source.resolve() or (target.exists() and target.samefile(source)) for source in inputs):
            raise ValueError("STAC Item output must not overwrite an input asset")
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("w", encoding="utf8", dir=target.parent, delete=False) as temporary:
            temporary_path = Path(temporary.name)
            try:
                temporary.write(json.dumps(item, indent=2, ensure_ascii=False, allow_nan=False))
                temporary.close()
                os.replace(temporary_path, target)
            finally:
                temporary_path.unlink(missing_ok=True)
    return item


def _ver() -> str:
    from . import __version__
    return __version__
