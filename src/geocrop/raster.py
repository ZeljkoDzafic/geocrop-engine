"""Small raster helpers shared by all stages: grids, block iteration, aligned reads and checked input access.

`open_input`, `read_block`, `check_grid` and `feature_names` are used at stage boundaries (features, samples,
predict) so that a truncated, mis-gridded or unlabelled input raster fails with a `RasterInputError` that
names the file and the problem, instead of an anonymous GDAL error or a silently wrong product.
"""
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import rasterio
from rasterio.errors import RasterioIOError
from rasterio.warp import Resampling, reproject
from rasterio.windows import Window

NODATA = -9999.0


class RasterInputError(ValueError):
    """An input raster is unreadable or violates the data contract; the message names the file."""


def open_input(path: str | Path, role: str = "input raster"):
    """`rasterio.open` for reading, with an actionable error for missing, truncated or non-raster files."""
    path = Path(path)
    if not path.is_file():
        raise RasterInputError(f"{role} {path} does not exist")
    try:
        return rasterio.open(path)
    except RasterioIOError as exc:
        raise RasterInputError(f"{role} {path} cannot be opened as a raster (truncated, corrupt or not a "
                               f"GeoTIFF): {exc}. Regenerate or re-download it.") from exc


def read_block(ds, indexes=None, window: Window | None = None) -> np.ndarray:
    """`ds.read` that names the file and window when the data are truncated or corrupt."""
    try:
        return ds.read(indexes, window=window)
    except RasterioIOError as exc:
        cause = exc.__cause__ or exc.__context__
        where = (f"rows {int(window.row_off)}-{int(window.row_off + window.height)}, cols "
                 f"{int(window.col_off)}-{int(window.col_off + window.width)}") if window is not None else "full raster"
        raise RasterInputError(f"Raster {ds.name} is truncated or corrupt: reading {where} failed ({exc}"
                               f"{f'; {cause}' if cause else ''}). Regenerate or re-download it.") from exc


def check_grid(ds, reference, what: str = "") -> None:
    """Require `ds` to share CRS, shape and transform with `reference`; the error names both files."""
    label = f" ({what})" if what else ""
    if ds.crs is None:
        raise RasterInputError(f"Raster {ds.name}{label} has no CRS")
    problems = []
    if ds.crs != reference.crs:
        problems.append(f"CRS {ds.crs} != {reference.crs}")
    if ds.shape != reference.shape:
        problems.append(f"shape {ds.shape} != {reference.shape}")
    if not ds.transform.almost_equals(reference.transform):
        problems.append(f"transform {tuple(ds.transform)[:6]} != {tuple(reference.transform)[:6]}")
    if problems:
        raise RasterInputError(f"Grid mismatch: {ds.name}{label} differs from {reference.name}: "
                               f"{'; '.join(problems)}. Composite every input on one reference grid.")


def feature_names(ds) -> list[str]:
    """Band descriptions of a feature stack; every band must be named, names must be unique."""
    names = list(ds.descriptions)
    unnamed = [i for i, n in enumerate(names, 1) if not n or not str(n).strip()]
    if unnamed:
        raise RasterInputError(f"Feature stack {ds.name}: bands {unnamed[:10]} of {len(names)} have no description "
                               "(feature name). Build stacks with geocrop features, which names every band.")
    duplicated = sorted({n for n in names if names.count(n) > 1})
    if duplicated:
        raise RasterInputError(f"Feature stack {ds.name}: duplicate band descriptions {duplicated[:10]}")
    return names


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
