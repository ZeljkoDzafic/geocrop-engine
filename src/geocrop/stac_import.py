"""Cloud-native Sentinel-2 L2A import from STAC items with Cloud-Optimised GeoTIFF assets.

Only the window of each asset that covers the study grid is read over HTTP, so a season for a
region can be prepared without downloading full ~1 GB products. Each acquisition becomes a normal
scene folder (B02 ... B12 reflectance, SCL, scene.json) on the reference grid and is processed by
the same compositing / time-series code as SAFE imports.

Radiometry. Reflectance = DN * scale + offset from the asset's `raster:bands`, EXCEPT when the item
declares that the provider already removed the processing-baseline >= 04.00 offset
(`earthsearch:boa_offset_applied: true`, Element 84 Earth Search): applying the advertised offset again
would shift reflectance by -0.1 (verified on 2023 scenes over northern Bosnia: 71 % negative red values).
A scene whose valid reflectance is implausibly negative is rejected instead of imported.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject, transform_bounds
from rasterio.windows import Window, from_bounds

from . import scenes as scene_meta
from .indices import S2_BANDS
from .raster import NODATA, float_profile, gtiff_profile

EARTH_SEARCH = "https://earth-search.aws.element84.com/v1"
ASSET_KEYS = {  # Earth Search v1 names; CDSE/others can pass their own mapping
    "B02": "blue", "B03": "green", "B04": "red", "B05": "rededge1", "B06": "rededge2", "B07": "rededge3",
    "B08": "nir", "B8A": "nir08", "B11": "swir16", "B12": "swir22", "SCL": "scl",
}
GDAL_ENV = {"AWS_NO_SIGN_REQUEST": "YES", "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
            "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.tiff", "GDAL_HTTP_MAX_RETRY": "4", "GDAL_HTTP_RETRY_DELAY": "2",
            # A stalled TLS connection must fail (and be retried / recorded per item) instead of hanging the
            # import forever: abort a request after 10 min, or when it receives < 1 KB/s for 60 s.
            "GDAL_HTTP_CONNECTTIMEOUT": "30", "GDAL_HTTP_TIMEOUT": "600",
            "GDAL_HTTP_LOW_SPEED_TIME": "60", "GDAL_HTTP_LOW_SPEED_LIMIT": "1024"}


def grid_identity(reference: str | Path) -> str:
    """Fingerprint of the study grid (CRS, transform, shape) a scene folder was cropped to."""
    import hashlib
    with rasterio.open(reference) as ref:
        body = f"{ref.crs.to_wkt()}|{tuple(ref.transform)[:6]}|{ref.shape}"
    return hashlib.sha256(body.encode()).hexdigest()[:20]


def cached_scene_valid(folder: Path, grid_id: str) -> bool:
    """A cached scene is reused only if it was imported for the same grid and all band files are present."""
    meta = folder / scene_meta.SIDE_CAR
    if not meta.exists():
        return False
    try:
        recorded = json.loads(meta.read_text(encoding="utf8")).get("grid_id")
    except ValueError:
        return False
    return recorded == grid_id and all((folder / f"{b}.tif").is_file() for b in (*S2_BANDS, "SCL"))


def radiometry(item, asset) -> tuple[float, float]:
    bands = (asset.extra_fields.get("raster:bands") or [{}])[0]
    scale = bands.get("scale")
    if scale is None:
        raise ValueError(f"{item.id}: asset lacks raster:bands scale; refusing to guess reflectance scaling")
    offset = bands.get("offset", 0.0) or 0.0
    if item.properties.get("earthsearch:boa_offset_applied") is True:
        offset = 0.0
    return float(scale), float(offset)


def read_window_aligned(src, dst_transform, dst_crs, shape, resampling, dtype="float32", fill=np.nan, margin=4):
    """Read only the source window covering the destination grid (plus margin) and reproject it."""
    h, w = shape
    left, top = dst_transform @ (0, 0)
    right, bottom = dst_transform @ (w, h)
    b = transform_bounds(dst_crs, src.crs, left, bottom, right, top)
    win = from_bounds(*b, transform=src.transform).round_offsets().round_lengths()
    win = Window(win.col_off - margin, win.row_off - margin, win.width + 2 * margin, win.height + 2 * margin)
    nodata = src.nodata if src.nodata is not None else 0
    data = src.read(1, window=win, boundless=True, fill_value=nodata)
    out = np.full(shape, fill, dtype=dtype)
    reproject(data, out, src_transform=src.window_transform(win), src_crs=src.crs, src_nodata=nodata,
              dst_transform=dst_transform, dst_crs=dst_crs, dst_nodata=fill, resampling=resampling)
    return out


def import_item(item, reference: str | Path, out_dir: str | Path, asset_keys: dict | None = None,
                min_clear_fraction: float = 0.0, scl_invalid=(0, 1, 3, 8, 9, 10, 11),
                max_negative_fraction: float = 0.05) -> dict:
    """Write one scene folder for `item` on the reference grid; skip it if the AOI is mostly cloudy."""
    keys = asset_keys or ASSET_KEYS
    out = Path(out_dir)
    with rasterio.open(reference) as ref, rasterio.Env(**GDAL_ENV):
        shape, transform, crs = ref.shape, ref.transform, ref.crs
        with rasterio.open(item.assets[keys["SCL"]].href) as src:
            scl = read_window_aligned(src, transform, crs, shape, Resampling.nearest, "uint8", 0)
        footprint = scl > 0
        if not footprint.any():
            return {"item": item.id, "status": "skipped: no coverage"}
        clear = footprint & ~np.isin(scl, list(scl_invalid))
        clear_fraction = float(clear.sum() / footprint.sum())
        if clear_fraction < min_clear_fraction:
            return {"item": item.id, "status": f"skipped: clear fraction {clear_fraction:.2f}"}
        out.mkdir(parents=True, exist_ok=True)
        bands = {}
        for band in S2_BANDS:
            asset = item.assets[keys[band]]
            scale, offset = radiometry(item, asset)
            method = Resampling.nearest if band in ("B02", "B03", "B04", "B08") else Resampling.bilinear
            with rasterio.open(asset.href) as src:
                dn = read_window_aligned(src, transform, crs, shape, method, "float32", 0)
            bands[band] = np.where(dn > 0, dn * scale + offset, np.nan).astype("float32")
        red = bands["B04"][clear & np.isfinite(bands["B04"])]
        if red.size and (red < 0).mean() > max_negative_fraction:
            raise ValueError(f"{item.id}: {(red < 0).mean():.0%} negative red reflectance; check radiometric offset")
        for band, arr in bands.items():
            with rasterio.open(out / f"{band}.tif", "w", **float_profile(ref)) as dst:
                dst.write(np.where(np.isfinite(arr), arr, NODATA).astype("float32"), 1)
        with rasterio.open(out / "SCL.tif", "w", **gtiff_profile(ref, count=1, dtype="uint8", nodata=None)) as dst:
            dst.write(scl, 1)
    p = item.properties
    scene_meta.write(out, sensor="S2", datetime=p.get("datetime"), tile=p.get("s2:mgrs_tile") or p.get("grid:code"),
                     platform=p.get("platform"), processing_baseline=p.get("s2:processing_baseline"),
                     boa_offset_applied=p.get("earthsearch:boa_offset_applied"), source=item.id,
                     stac_self=item.get_self_href(), clear_fraction=round(clear_fraction, 4),
                     grid_id=grid_identity(reference),
                     note="Windowed COG import on the reference grid")
    return {"item": item.id, "status": "imported", "path": str(out), "clear_fraction": clear_fraction}


def import_season(reference: str | Path, out_root: str | Path, start: str, end: str,
                  endpoint: str = EARTH_SEARCH, collection: str = "sentinel-2-l2a", max_cloud: float | None = 80,
                  min_clear_fraction: float = 0.2, max_items: int = 500, asset_keys: dict | None = None) -> list[dict]:
    """Search the STAC API over the reference grid's footprint and import every acquisition as a scene folder."""
    try:
        from pystac_client import Client
    except ImportError as exc:
        raise RuntimeError('pip install "geocrop-engine[stac]"') from exc
    with rasterio.open(reference) as ref:
        bbox = transform_bounds(ref.crs, "EPSG:4326", *ref.bounds)
    query = {"eo:cloud_cover": {"lt": max_cloud}} if max_cloud is not None else None
    items = Client.open(endpoint).search(collections=[collection], bbox=list(bbox), datetime=f"{start}/{end}",
                                         query=query, max_items=max_items).items()
    results = []
    grid_id = grid_identity(reference)
    for item in sorted(items, key=lambda i: i.properties.get("datetime", "")):
        target = Path(out_root) / item.id
        if cached_scene_valid(target, grid_id):
            results.append({"item": item.id, "status": "cached", "path": str(target)})
            continue
        (target / scene_meta.SIDE_CAR).unlink(missing_ok=True)   # stale grid or incomplete: re-import
        try:
            results.append(import_item(item, reference, target, asset_keys, min_clear_fraction))
        except (ValueError, rasterio.errors.RasterioIOError) as exc:
            results.append({"item": item.id, "status": f"failed: {exc}"})
    return results
