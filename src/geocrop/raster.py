"""Small raster helpers shared by all stages: grids, block iteration and aligned reads."""
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject
from rasterio.windows import Window

NODATA = -9999.0


def blocks(width: int, height: int, size: int) -> Iterator[Window]:
    for top in range(0, height, size):
        for left in range(0, width, size):
            yield Window(left, top, min(size, width - left), min(size, height - top))


def expand(window: Window, halo: int) -> Window:
    """Window grown by `halo` pixels on each side (may extend outside the raster)."""
    return Window(window.col_off - halo, window.row_off - halo, window.width + 2 * halo, window.height + 2 * halo)


def read_aligned(src, transform, crs, shape, resampling=Resampling.bilinear, band=1, dtype="float32",
                 fill=np.nan, src_nodata="auto") -> np.ndarray:
    """Reproject one band of an open dataset onto an arbitrary target window grid.

    `src_nodata` defaults to the dataset's nodata; pass a value (e.g. 0 for Sentinel-2 JP2 DN)
    when the file does not declare one, so nodata never bleeds into bilinear interpolation.
    """
    out = np.full(shape, fill, dtype=dtype)
    reproject(rasterio.band(src, band), out, src_transform=src.transform, src_crs=src.crs,
              src_nodata=src.nodata if src_nodata == "auto" else src_nodata, dst_transform=transform, dst_crs=crs, dst_nodata=fill,
              resampling=resampling)
    return out


def same_grid(a, b) -> bool:
    return a.crs == b.crs and a.shape == b.shape and a.transform.almost_equals(b.transform)


def gtiff_profile(reference, **extra) -> dict:
    """GeoTIFF creation options derived from a reference dataset: deflate, BigTIFF if needed, and
    256 x 256 internal tiles when the raster is large enough (striped otherwise)."""
    profile = reference.profile.copy()
    for key in ("blockxsize", "blockysize", "tiled", "interleave", "photometric"):
        profile.pop(key, None)
    profile.update(driver="GTiff", compress="deflate", BIGTIFF="IF_SAFER")
    if profile["width"] >= 256 and profile["height"] >= 256:
        profile.update(tiled=True, blockxsize=256, blockysize=256)
    profile.update(extra)
    return profile


def float_profile(reference, count=1, **extra) -> dict:
    return gtiff_profile(reference, count=count, dtype="float32", nodata=NODATA, predictor=3, **extra)


def ensure_parent(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def valid_mask(stack: np.ndarray, nodata=NODATA) -> np.ndarray:
    """All-band validity for a (bands, rows, cols) stack."""
    ok = np.isfinite(stack).all(axis=0)
    if nodata is not None:
        ok &= (stack != nodata).all(axis=0)
    return ok


def make_grid(aoi, crs: str, resolution: float, output: str | Path, buffer_m: float = 0.0) -> Path:
    """Write the study's reference grid: a uint8 GeoTIFF (1 inside AOI, 0 outside) whose origin is
    snapped to multiples of `resolution`. Every composite, feature and prediction uses this grid.

    Choose ONE projected CRS for the whole AOI even when it spans UTM zones (e.g. EPSG:32634 for
    16-19.5 E); scenes from other zones are reprojected onto it during compositing.
    """
    import geopandas as gpd
    from rasterio.features import geometry_mask
    from rasterio.transform import from_origin

    frame = gpd.read_file(aoi) if not isinstance(aoi, gpd.GeoDataFrame) else aoi
    if frame.crs is None:
        raise ValueError("AOI needs a CRS")
    frame = frame.to_crs(crs)
    geom = frame.geometry.union_all() if hasattr(frame.geometry, "union_all") else frame.geometry.unary_union
    if buffer_m:
        geom = geom.buffer(buffer_m)
    xmin, ymin, xmax, ymax = geom.bounds
    xmin, ymin = np.floor(xmin / resolution) * resolution, np.floor(ymin / resolution) * resolution
    xmax, ymax = np.ceil(xmax / resolution) * resolution, np.ceil(ymax / resolution) * resolution
    width, height = int(round((xmax - xmin) / resolution)), int(round((ymax - ymin) / resolution))
    transform = from_origin(xmin, ymax, resolution, resolution)
    profile = {"driver": "GTiff", "width": width, "height": height, "count": 1, "dtype": "uint8", "crs": crs,
               "transform": transform, "nodata": 0, "compress": "deflate", "tiled": True,
               "blockxsize": 256, "blockysize": 256, "BIGTIFF": "IF_SAFER"}
    out = ensure_parent(output)
    with rasterio.open(out, "w", **profile) as dst:
        for w in blocks(width, height, 2048):
            mask = geometry_mask([geom], out_shape=(int(w.height), int(w.width)),
                                 transform=rasterio.windows.transform(w, transform), invert=True)
            dst.write(mask.astype("uint8"), 1, window=w)
    return out
