"""External data-cube adapter: xarray / Zarr / NetCDF cubes and odc-stac loading <-> GeoCrop file contracts.

GeoCrop's own preprocessing writes one scene folder per acquisition and one composite folder per window, all on
the study reference grid. This adapter lets prepared spatio-temporal cubes take that place, and lets GeoCrop
outputs leave as cubes, without changing any downstream stage:

* `load_stac_cube`      Sentinel-2 L2A from any STAC API through odc-stac (optional), loaded onto the exact study
                        grid (CRS, transform, shape) with the same SCL screening and BOA-offset handling as
                        `geocrop stac-import` (`stac_import.radiometry` is reused, not re-implemented).
* `normalise_cube`      validates a user / Open Data Cube cube (dims time, y, x and bands as variables or a
                        ``band`` dimension): band names, CRS, grid alignment, units/scale, nodata, UTC timestamps.
* `cube_to_scenes`      canonical cube -> scene folders (B02 ... B12, SCL, scene.json), the `s2-import` contract.
* `cube_to_composites`  canonical cube -> window composites through `s2.composite` (identical masking and
                        statistics), or a pre-composited cube with a ``window`` dimension -> composite folders.
* `scenes_to_cube`, `composites_to_cube`, `stack_to_cube`, `write_cube`
                        GeoCrop scene folders / composites / feature stacks -> xarray Dataset with CF-1.8 metadata,
                        written as Zarr or NetCDF.

The *canonical cube* is an ``xarray.Dataset`` with dims (time, y, x), one float32 variable per band
(`indices.S2_BANDS`, surface reflectance in [0, 1], NaN = no data), a uint8 ``SCL`` variable (0 = no data),
pixel-centre ``x``/``y`` coordinates of the study grid, a ``spatial_ref`` grid-mapping coordinate and naive
``datetime64`` timestamps in UTC. Optional per-time coordinates (``scene_id``, ``tile``, ``platform`` ...) are
copied into scene.json.

xarray is required for every function here; odc-stac only for `load_stac_cube`; zarr / netCDF4 only for writing
or reading those formats. All are imported lazily with an install hint.
"""
from __future__ import annotations

import json
import re
import tempfile
from datetime import date
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine
from rasterio.crs import CRS
from rasterio.warp import Resampling, reproject, transform_bounds

from . import __version__
from . import scenes as scene_meta
from .indices import S2_BANDS
from .raster import NODATA, float_profile, gtiff_profile
from .s2 import DEFAULT_SCL_INVALID, SCL_MAX
from .stac_import import ASSET_KEYS, EARTH_SEARCH, GDAL_ENV, cached_scene_valid, grid_identity, radiometry

XARRAY_HINT = "the data-cube adapter needs xarray: pip install xarray"
ODC_HINT = 'loading STAC cubes needs odc-stac: pip install "odc-stac>=0.4" (Apache-2.0)'
ZARR_HINT = "Zarr cubes need the zarr package: pip install zarr"
NETCDF_HINT = "NetCDF cubes need netCDF4 (or h5netcdf): pip install netCDF4"

#: Per-time coordinates copied into scene.json when present on the cube.
SCENE_FIELDS = ("tile", "platform", "processing_baseline", "boa_offset_applied", "stac_self", "clear_fraction")
CUBE_VERSION = 1
MAX_NEGATIVE_RED = 0.05       # same rejection threshold as stac_import.import_item
MAX_ABOVE_RANGE = 0.5         # > 50 % of valid red values above 1.5 -> not reflectance (likely DN)
SYNTHETIC_SCL_CLEAR = 7       # SCL "unclassified": not in DEFAULT_SCL_INVALID, so treated as usable
CATEGORICAL = {"SCL": "uint8", "valid_count": "uint16"}   # nearest resampling, 0 = no data

# Band aliases: Earth Search / odc-stac asset names, Open Data Cube / Digital Earth Africa measurement names,
# CDSE openEO names and lower-case GeoCrop names. Keys are compared case-insensitively.
_ALIASES = {
    "B02": ("b02", "b2", "blue", "nbart_blue", "coastal_blue"),
    "B03": ("b03", "b3", "green", "nbart_green"),
    "B04": ("b04", "b4", "red", "nbart_red"),
    "B05": ("b05", "b5", "rededge1", "red_edge_1", "nbart_red_edge_1"),
    "B06": ("b06", "b6", "rededge2", "red_edge_2", "nbart_red_edge_2"),
    "B07": ("b07", "b7", "rededge3", "red_edge_3", "nbart_red_edge_3"),
    "B08": ("b08", "b8", "nir", "nir_1", "nbart_nir_1"),
    "B8A": ("b8a", "nir08", "nir_2", "nir_narrow", "nbart_nir_2"),
    "B11": ("b11", "swir16", "swir_1", "swir1", "nbart_swir_2"),
    "B12": ("b12", "swir22", "swir_2", "swir2", "nbart_swir_3"),
    "SCL": ("scl", "scene_classification"),
}
ALIASES = {alias: band for band, names in _ALIASES.items() for alias in (band.lower(), *names)}
_DIM_ALIASES = {"x": ("x", "lon", "longitude", "easting"), "y": ("y", "lat", "latitude", "northing"),
                "time": ("time", "t", "date"), "band": ("band", "bands", "variable", "spectral")}


class CubeError(ValueError):
    """A cube violates GeoCrop's cube contract; the message says what to fix."""


class GridMismatchError(CubeError):
    """The cube is not on the study grid lattice and resampling was not requested."""


def _xr():
    try:
        import xarray as xr
    except ImportError as exc:
        raise RuntimeError(XARRAY_HINT) from exc
    return xr


# --------------------------------------------------------------------------- reference grid helpers

def _ref_grid(reference) -> dict:
    with rasterio.open(reference) as ref:
        if ref.crs is None:
            raise CubeError(f"Reference grid {reference} has no CRS")
        return {"crs": ref.crs, "transform": ref.transform, "shape": ref.shape, "path": str(reference)}


def pixel_centres(transform: Affine, shape) -> tuple[np.ndarray, np.ndarray]:
    """x (columns) and y (rows) pixel-centre coordinates of a north-up grid."""
    h, w = shape
    x = transform.c + transform.a * (np.arange(w) + 0.5)
    y = transform.f + transform.e * (np.arange(h) + 0.5)
    return x, y


def reference_geobox(reference):
    """odc-geo GeoBox with exactly the reference raster's CRS, transform and shape."""
    try:
        from odc.geo.geobox import GeoBox
    except ImportError as exc:
        raise RuntimeError(ODC_HINT) from exc
    g = _ref_grid(reference)
    return GeoBox(g["shape"], g["transform"], g["crs"].to_wkt())


def grid_mapping(crs: CRS, transform: Affine) -> dict:
    """CF grid-mapping attributes (pyproj ``to_cf``) plus GDAL's GeoTransform, readable by GDAL and rioxarray."""
    from pyproj import CRS as PCRS
    attrs = {k: v for k, v in PCRS.from_wkt(crs.to_wkt()).to_cf().items() if v is not None}
    attrs["crs_wkt"] = attrs["spatial_ref"] = crs.to_wkt()
    attrs["GeoTransform"] = " ".join(str(v) for v in transform.to_gdal())
    return attrs


def _grid_coords(xr, crs: CRS, transform: Affine, shape) -> dict:
    x, y = pixel_centres(transform, shape)
    linear = "m" if crs.is_projected else "degrees"
    return {
        "x": xr.Variable("x", x, {"standard_name": "projection_x_coordinate" if crs.is_projected else "longitude",
                                  "units": linear, "axis": "X"}),
        "y": xr.Variable("y", y, {"standard_name": "projection_y_coordinate" if crs.is_projected else "latitude",
                                  "units": linear, "axis": "Y"}),
        "spatial_ref": xr.Variable((), np.int32(0), grid_mapping(crs, transform)),
    }


# --------------------------------------------------------------------------- STAC loading through odc-stac

def search_items(reference, start: str, end: str, endpoint: str = EARTH_SEARCH, collection: str = "sentinel-2-l2a",
                 max_cloud: float | None = 80, max_items: int = 500) -> list:
    """Same STAC query as `stac_import.import_season` (bbox of the grid, period, tile cloud pre-filter)."""
    try:
        from pystac_client import Client
    except ImportError as exc:
        raise RuntimeError('pip install "geocrop-engine[stac]"') from exc
    g = _ref_grid(reference)
    with rasterio.open(reference) as ref:
        bbox = transform_bounds(g["crs"], "EPSG:4326", *ref.bounds)
    query = {"eo:cloud_cover": {"lt": max_cloud}} if max_cloud is not None else None
    items = Client.open(endpoint).search(collections=[collection], bbox=list(bbox), datetime=f"{start}/{end}",
                                         query=query, max_items=max_items).items()
    return sorted(items, key=lambda i: (_item_datetime(i), i.id))


def _item_datetime(item) -> str:
    return item.properties.get("datetime") or (item.datetime.isoformat() if item.datetime else "")


def _odc_load(items, assets, geobox, resampling, dtype, chunks, pool):
    import odc.stac
    return odc.stac.load(items, bands=list(assets), geobox=geobox, groupby=lambda item, parsed, idx: idx,
                         resampling=resampling, dtype=dtype, chunks=chunks, pool=pool, fail_on_error=True)


def load_stac_cube(reference, start: str | None = None, end: str | None = None, endpoint: str = EARTH_SEARCH,
                   collection: str = "sentinel-2-l2a", max_cloud: float | None = 80, min_clear_fraction: float = 0.2,
                   max_items: int = 500, asset_keys: dict | None = None, scl_invalid=DEFAULT_SCL_INVALID,
                   max_negative_fraction: float = MAX_NEGATIVE_RED, chunks: dict | None = None, pool: int = 4,
                   items=None):
    """Sentinel-2 L2A from a STAC API as a canonical GeoCrop cube on the study grid, via odc-stac.

    Mirrors `stac_import.import_item` item by item: SCL is loaded first (nearest); items without coverage or with a
    clear fraction (SCL not in `scl_invalid`, inside the footprint SCL > 0) below `min_clear_fraction` are dropped
    before any reflectance is read; reflectance = DN * scale + offset from the asset's ``raster:bands``, with the
    offset forced to 0 when the item declares ``earthsearch:boa_offset_applied`` (`stac_import.radiometry`);
    DN <= 0 is no data; 10 m bands use nearest, 20 m bands bilinear; items with more than
    `max_negative_fraction` negative clear red values are rejected. Pass `items` to skip the search. With
    `chunks` the reflectance stays lazy (dask) and the negative-red check runs in `cube_to_scenes`.

    The per-item outcome is stored as JSON in ``attrs["geocrop:items"]``.
    """
    xr = _xr()
    try:
        import odc.stac
    except ImportError as exc:
        raise RuntimeError(ODC_HINT) from exc
    keys = asset_keys or ASSET_KEYS
    if items is None:
        if not (start and end):
            raise CubeError("load_stac_cube needs start and end (or items)")
        items = search_items(reference, start, end, endpoint, collection, max_cloud, max_items)
    items = sorted(items, key=lambda i: (_item_datetime(i), i.id))
    geobox = reference_geobox(reference)
    g = _ref_grid(reference)
    report: list[dict] = []
    odc.stac.configure_rio(cloud_defaults=True, **GDAL_ENV)
    with rasterio.Env(**GDAL_ENV):
        if not items:
            raise CubeError("No STAC items found for the grid and period")
        scl_cube = _odc_load(items, [keys["SCL"]], geobox, "nearest", "uint8", None, pool)
        scl_all = np.nan_to_num(np.asarray(scl_cube[keys["SCL"]].values), nan=0).astype("uint8")
        kept, clear_fractions = [], []
        for i, item in enumerate(items):
            footprint = scl_all[i] > 0
            if not footprint.any():
                report.append({"item": item.id, "status": "skipped: no coverage"})
                continue
            clear = footprint & ~np.isin(scl_all[i], list(scl_invalid))
            cf = float(clear.sum() / footprint.sum())
            if cf < min_clear_fraction:
                report.append({"item": item.id, "status": f"skipped: clear fraction {cf:.2f}"})
                continue
            kept.append(i)
            clear_fractions.append(cf)
        if not kept:
            raise CubeError(f"No STAC item passed the clear-fraction screen ({len(items)} searched)")
        sel = [items[i] for i in kept]
        resampling = {keys[b]: ("nearest" if b in ("B02", "B03", "B04", "B08") else "bilinear") for b in S2_BANDS}
        raw = _odc_load(sel, [keys[b] for b in S2_BANDS], geobox, resampling, "float32", chunks, pool)
    data_vars, good = {}, np.ones(len(sel), bool)
    for band in S2_BANDS:
        scale = np.array([radiometry(item, item.assets[keys[band]])[0] for item in sel], "float32")
        offset = np.array([radiometry(item, item.assets[keys[band]])[1] for item in sel], "float32")
        dn = raw[keys[band]]
        refl = xr.where(dn > 0, dn * xr.DataArray(scale, dims="time") + xr.DataArray(offset, dims="time"), np.nan)
        data_vars[band] = (("time", "y", "x"), refl.data.astype("float32"))
    scl = scl_all[kept]
    if chunks is None:
        for t, item in enumerate(sel):
            clear = (scl[t] > 0) & ~np.isin(scl[t], list(scl_invalid))
            red = data_vars["B04"][1][t]
            red = red[clear & np.isfinite(red)]
            if red.size and (red < 0).mean() > max_negative_fraction:
                good[t] = False
                report.append({"item": item.id, "status": f"failed: {item.id}: {(red < 0).mean():.0%} negative red "
                                                          "reflectance; check radiometric offset"})
    data_vars["SCL"] = (("time", "y", "x"), scl)
    times = _utc_index([_item_datetime(item) for item in sel])
    props = [item.properties for item in sel]
    ds = xr.Dataset(data_vars, coords={
        **_grid_coords(xr, g["crs"], g["transform"], g["shape"]), "time": ("time", times.values),
        "scene_id": ("time", [item.id for item in sel]),
        "source_datetime": ("time", [_item_datetime(item) for item in sel]),
        "tile": ("time", [str(p.get("s2:mgrs_tile") or p.get("grid:code") or "") for p in props]),
        "platform": ("time", [str(p.get("platform") or "") for p in props]),
        "processing_baseline": ("time", [str(p.get("s2:processing_baseline") or "") for p in props]),
        "boa_offset_applied": ("time", [str(p.get("earthsearch:boa_offset_applied")) for p in props]),
        "stac_self": ("time", [item.get_self_href() or "" for item in sel]),
        "clear_fraction": ("time", np.round(clear_fractions, 4)),
    })
    ds = ds.isel(time=np.flatnonzero(good))
    for t, item in enumerate(sel):
        if good[t]:
            report.append({"item": item.id, "status": "loaded", "clear_fraction": clear_fractions[t]})
    _finish(ds, g, alignment="exact", source=f"odc-stac:{endpoint}/{collection}")
    ds.attrs["geocrop:items"] = json.dumps(sorted(report, key=lambda r: r["item"]))
    return ds


def _finish(ds, g: dict, alignment: str, source: str):
    """Attach canonical-cube attributes in place."""
    for band in S2_BANDS:
        if band in ds:
            ds[band].attrs.update(standard_name="surface_bidirectional_reflectance", units="1",
                                  long_name=f"Sentinel-2 {band} surface reflectance", grid_mapping="spatial_ref")
    if "SCL" in ds:
        ds["SCL"].attrs.update(long_name="Sentinel-2 scene classification (ESA SCL, 0 = no data)",
                               grid_mapping="spatial_ref", units="1")
    ds["time"].attrs.update(standard_name="time", axis="T", long_name="acquisition time (UTC)")
    ds.attrs.update({"Conventions": "CF-1.8", "geocrop:cube_version": CUBE_VERSION,
                     "geocrop:grid_id": grid_identity(g["path"]), "geocrop:grid_alignment": alignment,
                     "source": source, "history": f"geocrop-engine {__version__} datacube adapter"})
    return ds


# --------------------------------------------------------------------------- reading and validating user cubes

def open_cube(path, group: str | None = None):
    """Open a Zarr store (directory or ``.zarr``) or a NetCDF file as an ``xarray.Dataset`` (lazy)."""
    xr = _xr()
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Cube {path} not found")
    if path.suffix.lower() == ".zarr" or (path.is_dir() and any((path / n).exists() for n in
                                                                  ("zarr.json", ".zgroup", ".zmetadata"))):
        try:
            import zarr  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(ZARR_HINT) from exc
        return xr.open_zarr(path, group=group, consolidated=False)
    if path.suffix.lower() in (".nc", ".nc4", ".netcdf", ".h5"):
        try:
            return xr.open_dataset(path, group=group)
        except ValueError as exc:
            raise RuntimeError(f"{NETCDF_HINT} ({exc})") from exc
    raise CubeError(f"{path}: unknown cube format; expected a .zarr store or a .nc file")


def _as_dataset(obj, band_map: dict | None):
    xr = _xr()
    if isinstance(obj, (str, Path)):
        obj = open_cube(obj)
    if isinstance(obj, xr.DataArray):
        band_dim = next((d for d in obj.dims if d.lower() in _DIM_ALIASES["band"]), None)
        if band_dim is None:
            raise CubeError(f"DataArray cube needs a band dimension (one of {_DIM_ALIASES['band']}); dims {obj.dims}")
        attrs = dict(obj.attrs)
        obj = obj.assign_coords({band_dim: [str(b) for b in obj[band_dim].values]}).to_dataset(dim=band_dim)
        obj.attrs.update(attrs)
    elif not isinstance(obj, xr.Dataset):
        raise CubeError(f"Expected an xarray Dataset/DataArray or a path, got {type(obj).__name__}")
    ds = obj
    band_dim = next((d for d in ds.dims if str(d).lower() in _DIM_ALIASES["band"]), None)
    if band_dim is not None:      # Dataset holding one variable with a band dimension
        carriers = [v for v in ds.data_vars if band_dim in ds[v].dims]
        if len(carriers) != 1:
            raise CubeError(f"Expected one variable with dimension {band_dim!r}, found {carriers}")
        da = ds[carriers[0]]
        ds = da.assign_coords({band_dim: [str(b) for b in da[band_dim].values]}).to_dataset(dim=band_dim)
        ds.attrs.update(obj.attrs)
        for name in obj.coords:
            if band_dim not in obj[name].dims and name not in ds.coords:
                ds = ds.assign_coords({name: obj[name]})
    # rename dims
    renames = {}
    for target, names in _DIM_ALIASES.items():
        if target == "band":
            continue
        found = [d for d in ds.dims if str(d).lower() in names]
        if len(found) > 1:
            raise CubeError(f"Ambiguous {target} dimension: {found}")
        if found and found[0] != target:
            renames[found[0]] = target
    ds = ds.rename(renames)
    # rename bands
    mapping = {}
    for var in ds.data_vars:
        user = (band_map or {}).get(str(var))
        band = user or ALIASES.get(str(var).lower())
        if band is not None:
            if band in mapping.values():
                raise CubeError(f"Two cube variables map to band {band}; pass an explicit band_map")
            mapping[var] = band
    if "valid_count" in ds.data_vars:
        mapping["valid_count"] = "valid_count"
    ds = ds[list(mapping)].rename({k: v for k, v in mapping.items() if k != v})
    missing = [b for b in S2_BANDS if b not in ds]
    if missing:
        raise CubeError(f"Cube lacks Sentinel-2 bands {missing}; variables found {sorted(map(str, obj.data_vars))}. "
                        "Pass band_map={'cube_name': 'B04', ...} for non-standard names")
    return ds, obj


def cube_crs(ds, fallback=None) -> CRS:
    """CRS of a cube from CF grid mapping, ``spatial_ref``/``crs`` variables, rioxarray/odc metadata or attrs."""
    candidates = []
    for var in ds.data_vars:
        gm = ds[var].attrs.get("grid_mapping") or ds[var].encoding.get("grid_mapping")
        if gm and gm in ds.variables:
            candidates.append(ds[gm].attrs)
    for name in ("spatial_ref", "crs"):
        if name in ds.variables:
            candidates.append(ds[name].attrs)
    candidates.append(ds.attrs)
    for attrs in candidates:
        for key in ("crs_wkt", "spatial_ref", "crs", "epsg", "epsg_code"):
            value = attrs.get(key)
            if value not in (None, ""):
                try:
                    return CRS.from_user_input(int(value) if key.startswith("epsg") else value)
                except (rasterio.errors.CRSError, ValueError, TypeError) as exc:
                    raise CubeError(f"Unreadable CRS {key}={value!r}: {exc}") from exc
    if fallback is not None:
        return CRS.from_user_input(fallback)
    raise CubeError("Cube has no CRS (no CF grid_mapping, spatial_ref/crs variable or crs attribute); pass crs=")


def cube_transform(ds) -> Affine:
    """Affine transform of the (north-up) pixel grid implied by regular pixel-centre x/y coordinates."""
    if "x" not in ds.dims or "y" not in ds.dims:
        raise CubeError(f"Cube needs x and y dimensions; dims {tuple(ds.dims)}")
    out = []
    for dim in ("x", "y"):
        v = np.asarray(ds[dim].values, dtype="float64")
        if v.size < 2:
            gt = next((ds[n].attrs.get("GeoTransform") for n in ("spatial_ref", "crs") if n in ds.variables), None)
            if not gt:
                raise CubeError(f"Cannot infer pixel size from a single {dim} coordinate without a GeoTransform")
            t = Affine.from_gdal(*map(float, str(gt).split()))
            out.append((v[0], t.a if dim == "x" else t.e))
            continue
        step = np.diff(v)
        if not np.allclose(step, step[0], rtol=1e-6, atol=1e-9 * max(1.0, abs(v).max())):
            raise CubeError(f"Irregular {dim} coordinates (steps {step.min()} .. {step.max()}); "
                            "GeoCrop needs a regular grid")
        out.append((v[0], float(step.mean())))
    (x0, dx), (y0, dy) = out
    if dx <= 0:
        raise CubeError("x coordinates must increase")
    return Affine(dx, 0.0, x0 - dx / 2, 0.0, dy, y0 - dy / 2)


def _utc_index(values, assume_tz: str = "UTC"):
    import pandas as pd
    try:
        idx = pd.DatetimeIndex(values) if not isinstance(values, pd.DatetimeIndex) else values
    except (TypeError, ValueError):
        try:
            idx = pd.to_datetime(list(values), utc=True)
        except (TypeError, ValueError) as exc:
            raise CubeError(f"Unparseable time coordinate: {exc}") from exc
    if idx.tz is None:
        idx = idx.tz_localize(assume_tz)
    idx = idx.tz_convert("UTC").tz_localize(None)
    if idx.isna().any():
        raise CubeError("Time coordinate contains missing values (NaT)")
    return idx.astype("datetime64[ns]")


def _time_to_utc(ds, assume_tz: str):
    if "time" not in ds.dims:
        raise CubeError(f"Cube needs a time dimension; dims {tuple(ds.dims)}")
    idx = ds.indexes["time"]
    if hasattr(idx, "to_datetimeindex"):      # cftime (non-standard calendars)
        try:
            idx = idx.to_datetimeindex()
        except ValueError as exc:
            raise CubeError(f"Time coordinate uses a calendar that cannot be mapped to UTC: {exc}") from exc
    tz = ds["time"].attrs.get("timezone") or ds["time"].attrs.get("time_zone")
    ds = ds.assign_coords(time=_utc_index(idx, tz or assume_tz).values)
    return ds.sortby("time")


def _decide_units(da, units: str, scale, offset):
    """(scale, offset, kind) turning the band into reflectance; refuses to guess ambiguous DN scaling."""
    attrs = da.attrs
    if scale is None and "scale" in attrs:
        scale = float(attrs["scale"])
        offset = float(attrs.get("offset", 0.0)) if offset is None else offset
    if units == "reflectance":
        return 1.0 if scale is None else float(scale), float(offset or 0.0), "reflectance"
    if units == "dn":
        if scale is None:
            raise CubeError(f"{da.name}: units='dn' needs a scale (e.g. 0.0001) and, for processing baseline "
                            ">= 04.00 data without offset harmonisation, offset=-0.1")
        return float(scale), float(offset or 0.0), "dn"
    if units != "auto":
        raise CubeError(f"units must be auto, reflectance or dn, not {units!r}")
    if scale is not None:
        return float(scale), float(offset or 0.0), "dn"
    if str(attrs.get("units", "")).lower() in ("1", "reflectance", "unitless", "dimensionless"):
        return 1.0, float(offset or 0.0), "reflectance"
    if np.issubdtype(da.dtype, np.integer):
        raise CubeError(f"{da.name}: integer digital numbers without scale metadata; pass units='dn' with "
                        "scale (and offset when the BOA offset was not removed) instead of letting GeoCrop guess")
    return None, None, "auto"   # decided from the data after alignment


def _to_reflectance(da, units, scale, offset, nodata):
    s, o, kind = _decide_units(da, units, scale, offset)
    nd = nodata if nodata is not None else da.attrs.get("nodata", da.attrs.get("_FillValue"))
    v = da.astype("float32")
    if nd is not None and np.isfinite(nd):
        v = v.where(da != nd)
    if kind == "dn":
        v = v.where(da > 0) * np.float32(s) + np.float32(o)    # DN <= 0 is no data (S2 convention, stac_import)
    elif kind == "reflectance":
        v = v * np.float32(s) + np.float32(o) if (s, o) != (1.0, 0.0) else v
    return v.astype("float32"), {"kind": kind, "scale": s, "offset": o, "nodata": nd}


def _align(ds, crs: CRS, transform: Affine, g: dict, resample: bool, resampling: str):
    """Put every variable on the reference grid: exact lattice crop/pad, or reprojection when allowed."""
    xr = _xr()
    rt, (h, w) = g["transform"], g["shape"]
    same_res = np.isclose(transform.a, rt.a, rtol=1e-6) and np.isclose(transform.e, rt.e, rtol=1e-6)
    col = (rt.c - transform.c) / transform.a
    row = (rt.f - transform.f) / transform.e
    on_lattice = (crs == g["crs"] and same_res and np.isclose(col, round(col), atol=1e-4)
                  and np.isclose(row, round(row), atol=1e-4))
    x, y = pixel_centres(rt, g["shape"])
    if on_lattice:
        col, row = int(round(col)), int(round(row))
        nx, ny = ds.sizes["x"], ds.sizes["y"]
        c0, c1, r0, r1 = max(col, 0), min(col + w, nx), max(row, 0), min(row + h, ny)
        if c0 >= c1 or r0 >= r1:
            raise GridMismatchError("Cube does not overlap the study grid")
        sub = ds.isel(x=slice(c0, c1), y=slice(r0, r1))
        sub = sub.assign_coords(x=x[c0 - col:c1 - col], y=y[r0 - row:r1 - row])
        if (c1 - c0, r1 - r0) != (w, h):
            fill = {v: (0 if v in CATEGORICAL else np.nan) for v in sub.data_vars}
            sub = sub.reindex(x=x, y=y, fill_value=fill)
        return sub, "exact" if (c1 - c0, r1 - r0) == (w, h) else "exact (partial coverage, padded with no data)"
    if not resample:
        raise GridMismatchError(
            f"Cube grid (CRS {crs.to_string()}, pixel {transform.a:g} x {abs(transform.e):g}, origin "
            f"{transform.c:g}, {transform.f:g}) is not on the study grid lattice (CRS {g['crs'].to_string()}, "
            f"pixel {rt.a:g} x {abs(rt.e):g}, origin {rt.c:g}, {rt.f:g}); pass resample=True (--resample) to "
            "reproject, or prepare the cube on the reference grid (e.g. odc-stac with geobox from the grid)")
    method = getattr(Resampling, resampling)
    out = {}
    for var in ds.data_vars:
        src = np.asarray(ds[var].values)
        cat = CATEGORICAL.get(var)
        if cat:
            src = np.nan_to_num(src, nan=0)
        dst = np.full((src.shape[0], h, w), 0 if cat else np.nan, dtype=cat or "float32")
        for t in range(src.shape[0]):
            reproject(src[t].astype(cat or "float32"), dst[t], src_transform=transform, src_crs=crs,
                      src_nodata=0 if cat else np.nan, dst_transform=rt, dst_crs=g["crs"],
                      dst_nodata=0 if cat else np.nan, resampling=Resampling.nearest if cat else method)
        out[var] = (("time", "y", "x"), dst)
    keep = {k: v for k, v in ds.coords.items() if k == "time" or set(v.dims) <= {"time"}}
    keep.pop("spatial_ref", None)
    return xr.Dataset(out, coords={**keep, "x": x, "y": y}), f"resampled ({resampling})"


def _check_values(ds, max_negative_fraction: float):
    red = ds["B04"].values
    for t in range(red.shape[0]):
        v = red[t][np.isfinite(red[t])]
        if not v.size:
            continue
        when = str(ds["time"].values[t])[:19]
        if (v < 0).mean() > max_negative_fraction:
            raise CubeError(f"{when}: {(v < 0).mean():.0%} negative red reflectance; check the offset (BOA offset "
                            "applied twice?)")
        if (v > 1.5).mean() > MAX_ABOVE_RANGE:
            raise CubeError(f"{when}: red values up to {v.max():g} are not reflectance; pass units='dn' and scale")


def _scl(ds, assume_masked: bool):
    xr = _xr()
    if "SCL" in ds:
        scl = ds["SCL"]
        values = np.asarray(scl.values)
        if np.issubdtype(values.dtype, np.floating):
            finite = values[np.isfinite(values)]
            if finite.size and not np.allclose(finite, np.round(finite)):
                raise CubeError("SCL holds non-integer values; it must be the ESA scene classification 0-11")
            values = np.nan_to_num(values, nan=0)
        if values.size and (values.min() < 0 or values.max() > SCL_MAX):
            raise CubeError(f"SCL values outside the ESA classes 0-{SCL_MAX}; refusing to treat unknown values as "
                            "clear sky")
        return ds.assign(SCL=(scl.dims, values.astype("uint8"))), False
    if not assume_masked:
        raise CubeError("Cube has no SCL band. GeoCrop screens clouds with SCL; pass assume_masked=True "
                        "(--assume-masked) if the cube is already cloud-masked (masked pixels = NaN)")
    observed = xr.concat([ds[b].notnull() for b in S2_BANDS], dim="band").all("band")
    scl = xr.where(observed, np.uint8(SYNTHETIC_SCL_CLEAR), np.uint8(0)).astype("uint8")
    return ds.assign(SCL=scl.transpose("time", "y", "x")), True


def normalise_cube(cube, reference, *, units: str = "auto", scale: float | None = None, offset: float | None = None,
                   nodata: float | None = None, band_map: dict | None = None, crs=None, assume_tz: str = "UTC",
                   assume_masked: bool = False, resample: bool = False, resampling: str = "bilinear",
                   max_negative_fraction: float = MAX_NEGATIVE_RED):
    """Validate an external cube and return the canonical GeoCrop cube on the study grid.

    `cube` is an xarray Dataset (bands as variables, ODC style), a DataArray / Dataset with a ``band``
    dimension, or a path to a Zarr store / NetCDF file. Checks, in order: band names (aliases or `band_map`),
    dimensions (time, y, x), timestamps (tz-aware -> UTC; naive values are interpreted in `assume_tz`, default
    UTC), units (CF-decoded reflectance, STAC-style ``scale``/``offset`` attributes, or explicit `units='dn'` with
    `scale`/`offset`; integer DN without scale is refused), nodata (attribute, `nodata`, NaN; DN <= 0), CRS,
    grid alignment (exact lattice crop/pad; reprojection only with `resample=True`), plausible reflectance
    (negative red fraction, values > 1.5) and SCL classes (or a synthetic SCL from NaN masks when
    `assume_masked`).
    """
    xr = _xr()
    g = _ref_grid(reference)
    ds, original = _as_dataset(cube, band_map)
    ds = _time_to_utc(ds, assume_tz)
    ds = ds.transpose("time", "y", "x", ...)
    src_crs = cube_crs(original if crs is None else xr.Dataset(), fallback=crs)
    transform = cube_transform(ds)
    if transform.e > 0:          # south-up coordinates: flip to north-up
        ds = ds.isel(y=slice(None, None, -1))
        transform = cube_transform(ds)
    radiometry_info = {}
    pending = []
    for band in S2_BANDS:
        refl, info = _to_reflectance(ds[band], units, scale, offset, nodata)
        ds = ds.assign({band: refl})
        radiometry_info[band] = info
        if info["kind"] == "auto":
            pending.append(band)
    keep = [v for v in (*S2_BANDS, *CATEGORICAL) if v in ds]
    ds = ds[keep]
    ds, alignment = _align(ds, src_crs, transform, g, resample, resampling)
    if pending:      # float data without unit metadata: decide from the values
        red = np.asarray(ds["B04"].values)
        p99 = float(np.nanpercentile(red, 99)) if np.isfinite(red).any() else 0.0
        if p99 > 1.5:
            raise CubeError(f"Float bands without unit metadata look like digital numbers (red p99 = {p99:g}); "
                            "pass units='dn' with scale/offset")
        for band in pending:
            radiometry_info[band].update(kind="reflectance (inferred from value range)", scale=1.0, offset=0.0)
    _check_values(ds, max_negative_fraction)
    ds, synthetic = _scl(ds, assume_masked)
    ds = ds.drop_vars([c for c in ("spatial_ref",) if c in ds.coords])
    ds = ds.assign_coords(_grid_coords(xr, g["crs"], g["transform"], g["shape"]))
    for name in list(ds.coords):
        if name not in ("time", "x", "y", "spatial_ref") and set(ds[name].dims) - {"time"}:
            ds = ds.drop_vars(name)
    _finish(ds, g, alignment, source=str(cube) if isinstance(cube, (str, Path)) else
            original.attrs.get("source", "xarray cube"))
    ds.attrs["geocrop:radiometry"] = json.dumps(radiometry_info, default=str)
    ds.attrs["geocrop:scl_synthesised"] = int(synthetic)
    return ds


# --------------------------------------------------------------------------- cube -> GeoCrop contracts

def _is_canonical(ds, reference) -> bool:
    return ds.attrs.get("geocrop:grid_id") == grid_identity(reference) and "SCL" in ds and \
        all(b in ds for b in S2_BANDS)


def _folder_names(ds) -> list[str]:
    ids = ds["scene_id"].values if "scene_id" in ds.coords else [None] * ds.sizes["time"]
    names, seen = [], {}
    for t, sid in zip(ds["time"].values, ids):
        base = re.sub(r"[^A-Za-z0-9._-]+", "_", str(sid)) if sid not in (None, "") else \
            "S2_" + str(np.datetime_as_string(t, unit="s")).replace("-", "").replace(":", "")
        n = seen.get(base, 0)
        seen[base] = n + 1
        names.append(base if n == 0 else f"{base}_{n}")
    return names


def _iso_utc(t) -> str:
    return str(np.datetime_as_string(np.datetime64(t, "us"), unit="us")) + "Z"


def cube_to_scenes(cube, reference, out_root, *, min_clear_fraction: float = 0.0, scl_invalid=DEFAULT_SCL_INVALID,
                   max_negative_fraction: float = MAX_NEGATIVE_RED, overwrite: bool = False,
                   **normalise_kwargs) -> list[dict]:
    """Write each time step as a GeoCrop S2 scene folder (B02 ... B12, SCL, scene.json) on the reference grid.

    The same rules as `stac_import.import_item` apply: time steps without coverage (SCL all 0) or with a clear
    fraction below `min_clear_fraction` are skipped, more than `max_negative_fraction` negative clear red values
    fail the step. A folder already imported for the same grid is reused unless `overwrite`.
    Non-canonical cubes are first passed through `normalise_cube` with `normalise_kwargs`.
    """
    ds = cube if not isinstance(cube, (str, Path)) and _is_canonical(cube, reference) else \
        normalise_cube(cube, reference, **normalise_kwargs)
    out_root = Path(out_root)
    grid_id = grid_identity(reference)
    synthetic = bool(ds.attrs.get("geocrop:scl_synthesised", 0))
    results = []
    with rasterio.open(reference) as ref:
        if (ds.sizes["y"], ds.sizes["x"]) != ref.shape:
            raise CubeError(f"Cube shape {(ds.sizes['y'], ds.sizes['x'])} differs from the grid {ref.shape}")
        for t, name in enumerate(_folder_names(ds)):
            step = ds.isel(time=t)
            target = out_root / name
            if not overwrite and cached_scene_valid(target, grid_id):
                results.append({"scene": name, "status": "cached", "path": str(target)})
                continue
            scl = np.asarray(step["SCL"].values).astype("uint8")
            footprint = scl > 0
            if not footprint.any():
                results.append({"scene": name, "status": "skipped: no coverage"})
                continue
            clear = footprint & ~np.isin(scl, list(scl_invalid))
            cf = float(clear.sum() / footprint.sum())
            if cf < min_clear_fraction:
                results.append({"scene": name, "status": f"skipped: clear fraction {cf:.2f}"})
                continue
            bands = {b: np.asarray(step[b].values, dtype="float32") for b in S2_BANDS}
            red = bands["B04"][clear & np.isfinite(bands["B04"])]
            if red.size and (red < 0).mean() > max_negative_fraction:
                results.append({"scene": name, "status": f"failed: {(red < 0).mean():.0%} negative red reflectance; "
                                                         "check radiometric offset"})
                continue
            (target / scene_meta.SIDE_CAR).unlink(missing_ok=True)
            target.mkdir(parents=True, exist_ok=True)
            for band, arr in bands.items():
                with rasterio.open(target / f"{band}.tif", "w", **float_profile(ref)) as dst:
                    dst.write(np.where(np.isfinite(arr), arr, NODATA).astype("float32"), 1)
            with rasterio.open(target / "SCL.tif", "w", **gtiff_profile(ref, count=1, dtype="uint8",
                                                                        nodata=None)) as dst:
                dst.write(scl, 1)
            meta = {k: _scalar(step[k].values) for k in SCENE_FIELDS if k in step.coords}
            meta = {k: v for k, v in meta.items() if v not in (None, "", "None")}
            if "boa_offset_applied" in meta:
                meta["boa_offset_applied"] = {"True": True, "False": False}.get(str(meta["boa_offset_applied"]),
                                                                                 meta["boa_offset_applied"])
            meta["clear_fraction"] = round(cf, 4)
            src_dt = _scalar(step["source_datetime"].values) if "source_datetime" in step.coords else None
            scene_meta.write(target, sensor="S2", datetime=src_dt or _iso_utc(step["time"].values),
                             source=_scalar(step["scene_id"].values) if "scene_id" in step.coords else name,
                             grid_id=grid_id, cube_source=ds.attrs.get("source"),
                             grid_alignment=ds.attrs.get("geocrop:grid_alignment"),
                             scl_synthesised=synthetic, note="Imported from a data cube on the reference grid",
                             **meta)
            results.append({"scene": name, "status": "imported", "path": str(target), "clear_fraction": cf})
    return results


def _scalar(v):
    v = v.item() if hasattr(v, "item") else v
    if isinstance(v, bytes):
        v = v.decode()
    if isinstance(v, float) and not np.isfinite(v):
        return None
    return v


def cube_to_composites(cube, reference, out_root, windows: dict | None = None, *, statistic: str = "median",
                       scl_invalid=DEFAULT_SCL_INVALID, mask_dilation_px: int = 0, min_valid_observations: int = 1,
                       temporal_screening: bool = False, scenes_dir=None, **normalise_kwargs) -> dict:
    """Window composites in GeoCrop's composite contract (``<out>/<window>/B02.tif ... valid_count.tif``).

    A cube with a ``time`` dimension is written as scene folders (to `scenes_dir`, or a temporary directory) and
    composited by `s2.composite` with the given parameters, so masking, same-day mosaicking and statistics are
    exactly those of the local pipeline. A pre-composited cube with a ``window`` dimension (e.g. written by
    `composites_to_cube`) is written directly; its ``valid_count`` variable is used when present.
    """
    xr = _xr()
    obj = open_cube(cube) if isinstance(cube, (str, Path)) else cube
    if isinstance(obj, xr.Dataset) and "window" in obj.dims:
        return _write_window_cube(obj, reference, out_root, normalise_kwargs)
    if not windows:
        raise CubeError("A time-series cube needs windows {name: (start, end)} to composite")
    from . import s2
    tmp = None
    if scenes_dir is None:
        tmp = tempfile.TemporaryDirectory(prefix="geocrop-cube-")
        scenes_dir = tmp.name
    try:
        written = cube_to_scenes(obj, reference, scenes_dir, scl_invalid=scl_invalid, **normalise_kwargs)
        found = scene_meta.discover(scenes_dir)
        by_window = scene_meta.assign(found, {k: (_date(a), _date(b)) for k, (a, b) in windows.items()})
        out = {}
        for name, members in by_window.items():
            if not members:
                out[name] = {"status": "skipped: no scenes"}
                continue
            target = Path(out_root) / name
            s2.composite([m["path"] for m in members], target, reference=reference, statistic=statistic,
                         scl_invalid=scl_invalid, mask_dilation_px=mask_dilation_px,
                         min_valid_observations=min_valid_observations, temporal_screening=temporal_screening)
            meta_path = target / "composite.json"
            meta = json.loads(meta_path.read_text(encoding="utf8"))
            meta["cube_source"] = str(cube) if isinstance(cube, (str, Path)) else obj.attrs.get("source")
            meta["scenes_staged"] = tmp is None
            meta_path.write_text(json.dumps(meta, indent=2, default=str), encoding="utf8")
            out[name] = {"status": "written", "path": str(target), "scenes": len(members)}
        return {"windows": out, "scenes": written}
    finally:
        if tmp is not None:
            tmp.cleanup()


def _date(v) -> date:
    return v if isinstance(v, date) else date.fromisoformat(str(v))


def _write_window_cube(ds, reference, out_root, kw: dict) -> dict:
    """Pre-composited (window, y, x) cube -> composite folders, validated like a time-series cube."""
    windows = [str(w) for w in ds["window"].values]
    # windows become placeholder time steps so the same validation and grid alignment apply
    pseudo = ds.assign_coords(window=np.arange(len(windows)).astype("datetime64[D]").astype("datetime64[ns]"))
    pseudo = pseudo.rename({"window": "time"})
    canon = normalise_cube(pseudo, reference, **{"assume_masked": True, **kw})
    has_count = "valid_count" in canon
    out = {}
    with rasterio.open(reference) as ref:
        for i, name in enumerate(windows):
            target = Path(out_root) / name
            target.mkdir(parents=True, exist_ok=True)
            observed = np.ones(ref.shape, bool)
            for b in S2_BANDS:
                arr = np.asarray(canon[b].values[i], dtype="float32")
                observed &= np.isfinite(arr)
                with rasterio.open(target / f"{b}.tif", "w", **float_profile(ref)) as dst:
                    dst.write(np.where(np.isfinite(arr), arr, NODATA).astype("float32"), 1)
            vc = np.nan_to_num(np.asarray(canon["valid_count"].values[i], dtype="float64"), nan=0) if has_count \
                else observed
            with rasterio.open(target / "valid_count.tif", "w",
                               **gtiff_profile(ref, count=1, dtype="uint16", nodata=None)) as dst:
                dst.write(vc.astype("uint16"), 1)
            (target / "composite.json").write_text(json.dumps({
                "sensor": "S2", "acquisitions": {}, "statistic": ds.attrs.get("geocrop:statistic", "external"),
                "valid_count": "from cube" if has_count else "unknown: 1 = observed, 0 = no data",
                "reference_grid": str(reference), "grid_alignment": canon.attrs.get("geocrop:grid_alignment"),
                "cube_source": ds.attrs.get("source", "xarray cube")}, indent=2), encoding="utf8")
            out[name] = {"status": "written", "path": str(target)}
    return {"windows": out}


# --------------------------------------------------------------------------- GeoCrop outputs -> cubes

def _grid_of(path) -> tuple[CRS, Affine, tuple[int, int]]:
    with rasterio.open(path) as src:
        return src.crs, src.transform, src.shape


def _read(path, dtype="float32", nodata=NODATA):
    with rasterio.open(path) as src:
        a = src.read(1).astype(dtype)
    if np.issubdtype(np.dtype(dtype), np.floating):
        a[a == nodata] = np.nan
    return a


def scenes_to_cube(scene_root, start: str | None = None, end: str | None = None):
    """GeoCrop S2 scene folders (one acquisition each) -> canonical time-series cube."""
    xr = _xr()
    found = [s for s in scene_meta.discover(scene_root) if s["sensor"] == "S2"
             and (not start or s["date"] >= _date(start)) and (not end or s["date"] <= _date(end))]
    if not found:
        raise CubeError(f"No S2 scene folders under {scene_root}")
    crs, transform, shape = _grid_of(Path(found[0]["path"]) / "B02.tif")
    data = {b: np.stack([_read(Path(s["path"]) / f"{b}.tif") for s in found]) for b in S2_BANDS}
    data["SCL"] = np.stack([_read(Path(s["path"]) / "SCL.tif", "uint8") for s in found])
    times = _utc_index([str(s["datetime"]) for s in found])
    coords = {**_grid_coords(xr, crs, transform, shape), "time": times.values,
              "scene_id": ("time", [Path(s["path"]).name for s in found]),
              "source_datetime": ("time", [str(s["datetime"]) for s in found])}
    for k in SCENE_FIELDS:
        coords[k] = ("time", [str(s.get(k, "")) for s in found])
    ds = xr.Dataset({k: (("time", "y", "x"), v) for k, v in data.items()}, coords=coords)
    ds.attrs["title"] = "GeoCrop Sentinel-2 scene time series"
    return _finish_export(ds, crs, transform, shape, str(scene_root))


def composites_to_cube(composites_root, windows: dict | None = None):
    """GeoCrop window composites -> Dataset (window, y, x) with B02 ... B12 and valid_count."""
    xr = _xr()
    root = Path(composites_root)
    names = [d.name for d in sorted(root.iterdir()) if (d / "B02.tif").is_file()]
    if windows:
        names = [w for w in windows if w in names]
    elif not names:
        raise CubeError(f"No composite folders (with B02.tif) under {root}")
    if not names:
        raise CubeError(f"None of the windows {list(windows)} has a composite under {root}")
    crs, transform, shape = _grid_of(root / names[0] / "B02.tif")
    data = {b: (("window", "y", "x"), np.stack([_read(root / w / f"{b}.tif") for w in names])) for b in S2_BANDS}
    data["valid_count"] = (("window", "y", "x"), np.stack([_read(root / w / "valid_count.tif", "uint16")
                                                           for w in names]))
    stats = {json.loads((root / w / "composite.json").read_text(encoding="utf8")).get("statistic")
             for w in names if (root / w / "composite.json").is_file()}
    coords = {**_grid_coords(xr, crs, transform, shape), "window": names}
    if windows:
        coords["window_start"] = ("window", np.array([str(windows[w][0]) for w in names], "datetime64[ns]"))
        coords["window_end"] = ("window", np.array([str(windows[w][1]) for w in names], "datetime64[ns]"))
    ds = xr.Dataset(data, coords=coords)
    ds["valid_count"].attrs.update(long_name="number of clear observations in the window", units="1",
                                   grid_mapping="spatial_ref")
    ds.attrs["title"] = "GeoCrop Sentinel-2 window composites"
    if len(stats) == 1 and None not in stats:
        ds.attrs["geocrop:statistic"] = stats.pop()
    return _finish_export(ds, crs, transform, shape, str(root))


def stack_to_cube(stack):
    """GeoCrop feature stack (stack.tif + manifest) -> Dataset with ``features`` (feature, y, x)."""
    xr = _xr()
    from .features import load_manifest
    with rasterio.open(stack) as src:
        arr = src.read().astype("float32")
        crs, transform, shape = src.crs, src.transform, src.shape
        names = [d or f"band_{i + 1}" for i, d in enumerate(src.descriptions)]
    arr[arr == NODATA] = np.nan
    coords = {**_grid_coords(xr, crs, transform, shape), "feature": names}
    try:
        manifest = {f.name: f for f in load_manifest(stack)}
        for key in ("group", "sensor", "period"):
            coords[f"feature_{key}"] = ("feature", [str(getattr(manifest[n], key, "") or "")
                                                    if n in manifest else "" for n in names])
    except FileNotFoundError:
        pass
    ds = xr.Dataset({"features": (("feature", "y", "x"), arr)}, coords=coords)
    ds["features"].attrs.update(long_name="GeoCrop feature stack", grid_mapping="spatial_ref")
    ds.attrs["title"] = "GeoCrop feature stack"
    return _finish_export(ds, crs, transform, shape, str(stack))


def _finish_export(ds, crs, transform, shape, source):
    for band in S2_BANDS:
        if band in ds:
            ds[band].attrs.update(standard_name="surface_bidirectional_reflectance", units="1",
                                  long_name=f"Sentinel-2 {band} surface reflectance", grid_mapping="spatial_ref")
    if "SCL" in ds:
        ds["SCL"].attrs.update(long_name="Sentinel-2 scene classification (ESA SCL)", grid_mapping="spatial_ref")
    if "time" in ds.coords:
        ds["time"].attrs.update(standard_name="time", axis="T", long_name="acquisition time (UTC)")
    ds.attrs.update({"Conventions": "CF-1.8", "source": source, "geocrop:cube_version": CUBE_VERSION,
                     "history": f"geocrop-engine {__version__} datacube export"})
    return ds


def write_cube(ds, path, overwrite: bool = False) -> Path:
    """Write a cube as Zarr (``.zarr``) or NetCDF (``.nc``); float NaN is encoded as _FillValue NaN."""
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; pass overwrite=True (--overwrite)")
    ds = ds.copy()
    for name in list(ds.variables):
        if ds[name].dtype == object:
            ds[name] = ds[name].astype(str)
    for var in ds.data_vars:
        ds[var].encoding.pop("_FillValue", None)
        if np.issubdtype(ds[var].dtype, np.floating):
            ds[var].encoding["_FillValue"] = np.float32(np.nan)
    if path.suffix.lower() == ".zarr":
        try:
            import zarr  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(ZARR_HINT) from exc
        ds.to_zarr(path, mode="w", consolidated=False)
    elif path.suffix.lower() in (".nc", ".nc4"):
        try:
            ds.to_netcdf(path)
        except (ValueError, ImportError) as exc:
            raise RuntimeError(f"{NETCDF_HINT} ({exc})") from exc
    else:
        raise CubeError(f"{path}: choose a .zarr or .nc output")
    return path
