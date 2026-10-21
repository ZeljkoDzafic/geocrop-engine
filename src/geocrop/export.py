"""Interoperable publication of results: Cloud-Optimised GeoTIFF, GeoParquet, STAC metadata, HCAT codes."""
from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import rasterio
from rasterio.shutil import copy as rio_copy


def to_cog(src: str | Path, dst: str | Path, overview_resampling: str = "nearest") -> Path:
    """Copy a GeoTIFF as a Cloud-Optimised GeoTIFF (GDAL COG driver, deflate, internal overviews).

    Use `nearest`/`mode` for class rasters and `average` for continuous layers."""
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


def stac_item(prediction_dir: str | Path, item_id: str, start: str, end: str, output: str | Path | None = None,
              hcat: dict | None = None, extra_assets: dict | None = None) -> dict:
    """Minimal STAC 1.0 Item (with the classification extension) describing a prediction directory."""
    pred = Path(prediction_dir)
    legend = json.loads((pred / "class_legend.json").read_text(encoding="utf8"))
    with rasterio.open(pred / "classes.tif") as ds:
        from rasterio.warp import transform_bounds
        west, south, east, north = transform_bounds(ds.crs, "EPSG:4326", *ds.bounds)
        epsg = ds.crs.to_epsg()
        shape = [ds.height, ds.width]
    classes = [{"value": int(k), "name": v, **({"description": f"HCAT {hcat[v]['code']} {hcat[v]['name'] or ''}".strip()}
                                               if hcat and v in hcat and hcat[v]["code"] else {})}
               for k, v in legend.items()]
    assets = {
        "classes": {"href": "classes.tif", "type": "image/tiff; application=geotiff", "roles": ["data"],
                    "classification:classes": classes},
        "probabilities": {"href": "probabilities.tif", "type": "image/tiff; application=geotiff", "roles": ["data"]},
        "uncertainty": {"href": "uncertainty.tif", "type": "image/tiff; application=geotiff", "roles": ["data"]},
    }
    if (pred / "aoa.tif").exists():
        assets["aoa"] = {"href": "aoa.tif", "type": "image/tiff; application=geotiff", "roles": ["metadata"]}
    assets.update(extra_assets or {})
    item = {
        "type": "Feature", "stac_version": "1.0.0", "id": item_id,
        "stac_extensions": ["https://stac-extensions.github.io/classification/v2.0.0/schema.json",
                            "https://stac-extensions.github.io/projection/v1.1.0/schema.json"],
        "bbox": [west, south, east, north],
        "geometry": {"type": "Polygon", "coordinates": [[[west, south], [east, south], [east, north], [west, north],
                                                          [west, south]]]},
        "properties": {"datetime": None, "start_datetime": f"{start}T00:00:00Z", "end_datetime": f"{end}T23:59:59Z",
                       "proj:epsg": epsg, "proj:shape": shape,
                       "created": datetime.now(timezone.utc).isoformat(), "processing:software": {"geocrop-engine": _ver()}},
        "assets": assets, "links": [],
    }
    if output:
        Path(output).write_text(json.dumps(item, indent=2), encoding="utf8")
    return item


def _ver() -> str:
    from . import __version__
    return __version__
