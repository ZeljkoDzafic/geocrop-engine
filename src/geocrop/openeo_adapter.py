"""Optional openEO back-end (Copernicus Data Space Ecosystem) writing results in GeoCrop's file contracts.

Local processing stays the default. This adapter lets a study run the data-heavy part of the chain on an
openEO back-end (default ``https://openeo.dataspace.copernicus.eu``) and download only the results:

* ``s2_composites``  per-window Sentinel-2 composites (SCL masking, optional dilation and cloud probability,
                     reflectance range check, median/mean/percentile) plus ``valid_count`` -> ``<composites>/<window>/``
* ``s1_composites``  per-window Sentinel-1 VV/VH linear backscatter (``sar_backscatter`` or CARD4L
                     ``ard_normalized_radar_backscatter``), optionally per orbit direction
* ``s2_scenes``      per-acquisition scene folders (B02 ... B12, SCL, scene.json): masking then runs locally,
                     exactly as for STAC/SAFE imports
* ``signatures``     parcel signatures (``aggregate_spatial``) in the long table of `geocrop.cbm.parcel_signatures`

Process graphs are built here as plain openEO JSON (API 1.x flat graphs, deterministic node ids), so a study's
recipe can be exported and archived without the openEO client or credentials (`geocrop openeo-graph`,
`geocrop openeo-extract --dry-run`). The ``openeo`` Python client (Apache-2.0) is imported lazily, only to
authenticate and run batch jobs.

Credentials are read from the environment only and never written by GeoCrop:
``CDSE_OPENEO_CLIENT_ID`` / ``CDSE_OPENEO_CLIENT_SECRET`` (OIDC client credentials, optional
``CDSE_OPENEO_PROVIDER_ID``), otherwise the interactive OIDC device flow with ``store_refresh_token=False``.
``CDSE_OPENEO_URL`` overrides the back-end URL.

Downloaded rasters are aligned onto the study reference grid with `geocrop.raster.read_aligned` and validated
(grid, coverage, band count, radiometry) before any contract file is written. Differences from local masking
are listed in `DIFFERENCES` and recorded in every output's provenance; see docs/openeo.md.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import Resampling, transform_bounds

from . import __version__
from . import scenes as scene_meta
from .indices import S1_POLARISATIONS, S2_BANDS
from .raster import NODATA, float_profile, gtiff_profile, read_aligned
from .s2 import DEFAULT_SCL_INVALID

CDSE_URL = "https://openeo.dataspace.copernicus.eu"
S2_COLLECTION = "SENTINEL2_L2A"
S1_COLLECTION = "SENTINEL1_GRD"
PRODUCTS = ("s2_composites", "s1_composites", "s2_scenes", "signatures")
S1_COEFFICIENTS = ("sigma0-ellipsoid", "gamma0-terrain")
ENV_URL = "CDSE_OPENEO_URL"
ENV_CLIENT_ID = "CDSE_OPENEO_CLIENT_ID"
ENV_CLIENT_SECRET = "CDSE_OPENEO_CLIENT_SECRET"
ENV_PROVIDER = "CDSE_OPENEO_PROVIDER_ID"
INSTALL_HINT = 'the openEO back-end needs the optional openeo client: pip install "openeo>=0.30"'

#: Sentinel-2 L2A digital numbers on the back-end -> surface reflectance (BOA offset assumed harmonised).
DN_SCALE = 1e-4
#: Valid DN range: (0, 15000] = reflectance (0, 1.5], the import (DN > 0) and composite (<= 1.5) checks.
DN_MAX = 15000
MAX_NEGATIVE_RED = 0.05
ORBIT_STATE = {"ASC": "ascending", "DESC": "descending"}

#: Where the openEO recipe cannot reproduce local GeoCrop processing exactly (also in docs/openeo.md).
DIFFERENCES = (
    "Resampling: the back-end resamples every band to the study CRS/resolution with nearest neighbour "
    "(local SAFE/STAC import uses bilinear for 20 m bands); results are then aligned to the reference grid locally.",
    "Same-day tiles: the back-end mosaics acquisitions of one day into one observation with its own rule; "
    "GeoCrop takes the first clear tile per pixel.",
    "Shadow projection (SHADOW.tif), external cloud masks (CLOUDMASK.tif) and temporal_screening are not "
    "available in s2_composites; use s2_scenes to keep them local.",
    "Cloud probability uses the back-end CLD band (availability per collection to verify) instead of CLDPRB.tif.",
    "Mask dilation uses apply_kernel with a (2k+1)x(2k+1) box, equal to k iterations of a 3x3 binary dilation; "
    "kernel border handling at the extent edge may differ.",
    "Reflectance: back-end DN are scaled by 1e-4 assuming the processing-baseline 04.00 offset is already "
    "harmonised; composites with > 5 % negative red reflectance are rejected.",
    "Parcel signatures: pixel inclusion in aggregate_spatial follows the back-end rule (GeoCrop rasterises pixel "
    "centres); one clear-pixel count is shared by all indices (GeoCrop also drops non-finite index values).",
    "Sentinel-1: sar_backscatter/CARD4L processing (orbit, noise, terrain correction, DEM) is the back-end's, "
    "not the SNAP graph of `geocrop snap-graph`.",
)


# --------------------------------------------------------------------------- process graph building

class ProcessGraph:
    """A flat openEO process graph with deterministic node ids ``<process_id without _><n>``."""

    def __init__(self):
        self.nodes: dict[str, dict] = {}
        self._count: dict[str, int] = {}

    def add(self, process_id: str, **arguments) -> dict:
        n = self._count.get(process_id, 0) + 1
        self._count[process_id] = n
        node_id = f"{process_id.replace('_', '')}{n}"
        self.nodes[node_id] = {"process_id": process_id, "arguments": arguments}
        return {"from_node": node_id}

    def finish(self, ref: dict) -> dict:
        self.nodes[ref["from_node"]]["result"] = True
        return self.nodes


def callback(build) -> dict:
    """Child process graph (reducer, process, overlap resolver) built by `build(graph) -> result ref`."""
    g = ProcessGraph()
    return {"process_graph": g.finish(build(g))}


DATA = {"from_parameter": "data"}
X = {"from_parameter": "x"}


def _nd(a, b):
    return ("/", ("-", a, b), ("+", a, b))


_MSAVI_T = ("+", ("*", 2, "B08"), 1)
#: GeoCrop indices expressible as openEO band math; inputs are reflectance (DN * 1e-4).
#: Expressions are nested tuples (op, a, b) with ops + - * / sqrt; leaves are band names or numbers.
INDEX_EXPRESSIONS: dict[str, tuple] = {
    "NDVI": _nd("B08", "B04"),
    "NDRE": _nd("B8A", "B05"),
    "NDMI": _nd("B08", "B11"),
    "NDWI": _nd("B03", "B08"),
    "BSI": _nd(("+", "B11", "B04"), ("+", "B08", "B02")),
    "EVI": ("*", 2.5, ("/", ("-", "B08", "B04"), ("+", ("-", ("+", "B08", ("*", 6, "B04")), ("*", 7.5, "B02")), 1))),
    "EVI2": ("*", 2.5, ("/", ("-", "B08", "B04"), ("+", ("+", "B08", ("*", 2.4, "B04")), 1))),
    "MSAVI2": ("/", ("-", _MSAVI_T, ("sqrt", ("-", ("*", _MSAVI_T, _MSAVI_T), ("*", 8, ("-", "B08", "B04"))))), 2),
    "RECI": ("-", ("/", "B08", "B05"), 1),
    "GCVI": ("-", ("/", "B08", "B03"), 1),
    "PSRI": ("/", ("-", "B04", "B02"), "B06"),
}
_OPS = {"+": "add", "-": "subtract", "*": "multiply", "/": "divide"}


def expression_bands(expr) -> list[str]:
    if isinstance(expr, str):
        return [expr]
    if isinstance(expr, tuple):
        return sorted({b for e in expr[1:] for b in expression_bands(e)})
    return []


def evaluate_expression(expr, bands: dict[str, np.ndarray]) -> np.ndarray:
    """NumPy evaluation of an index expression (used to check parity with `geocrop.indices`)."""
    if isinstance(expr, str):
        return np.asarray(bands[expr], dtype="float64")
    if not isinstance(expr, tuple):
        return expr
    op, *args = expr
    v = [evaluate_expression(a, bands) for a in args]
    with np.errstate(divide="ignore", invalid="ignore"):
        if op == "sqrt":
            return np.sqrt(v[0])
        return {"+": np.add, "-": np.subtract, "*": np.multiply, "/": np.divide}[op](v[0], v[1])


def _expression_node(g: ProcessGraph, expr, leaf, memo: dict):
    if isinstance(expr, (int, float)):
        return expr
    if expr in memo:
        return memo[expr]
    if isinstance(expr, str):
        ref = leaf(expr)
    else:
        op, *args = expr
        refs = [_expression_node(g, a, leaf, memo) for a in args]
        ref = g.add("sqrt", x=refs[0]) if op == "sqrt" else g.add(_OPS[op], x=refs[0], y=refs[1])
    memo[expr] = ref
    return ref


def _or_chain(g: ProcessGraph, terms: list):
    if not terms:
        return g.add("constant", x=False)
    acc = terms[0]
    for t in terms[1:]:
        acc = g.add("or", x=acc, y=t)
    return acc


def _invalid_expression(g: ProcessGraph, scl_invalid, range_bands=(), cloud_probability_max=None, elements=None):
    """Boolean "observation unusable" inside a callback over the bands array (parameter ``data``)."""
    elements = {} if elements is None else elements

    def elem(label):
        if label not in elements:
            elements[label] = g.add("array_element", data=DATA, label=label)
        return elements[label]

    terms = [g.add("eq", x=elem("SCL"), y=int(v)) for v in scl_invalid]
    if cloud_probability_max is not None:
        terms.append(g.add("gt", x=elem("CLD"), y=cloud_probability_max))
    for b in range_bands:
        terms += [g.add("lte", x=elem(b), y=0), g.add("gt", x=elem(b), y=DN_MAX)]
    return _or_chain(g, terms)


def reference_grid(reference: str | Path) -> dict:
    """CRS (EPSG), resolution, bounds and fingerprint of the study reference grid."""
    from .stac_import import grid_identity
    with rasterio.open(reference) as ref:
        epsg = ref.crs.to_epsg() if ref.crs else None
        if epsg is None:
            raise ValueError(f"{reference}: the reference grid needs an EPSG-coded CRS for openEO")
        res = abs(ref.transform.a)
        if not np.isclose(res, abs(ref.transform.e)):
            raise ValueError("openEO extraction needs square reference pixels")
        b = ref.bounds
        return {"path": str(reference), "epsg": int(epsg), "resolution": float(res), "shape": list(ref.shape),
                "bounds": [float(b.left), float(b.bottom), float(b.right), float(b.top)],
                "grid_id": grid_identity(reference)}


def spatial_extent(grid: dict, pad_px: int = 2) -> dict:
    """Reference bounds plus a small margin, in the reference CRS (openEO ``spatial_extent`` with ``crs``)."""
    w, s, e, n = grid["bounds"]
    pad = pad_px * grid["resolution"]
    return {"west": w - pad, "south": s - pad, "east": e + pad, "north": n + pad, "crs": grid["epsg"]}


def temporal_extent(start: date, end: date) -> list[str]:
    """GeoCrop windows are inclusive; openEO temporal extents exclude the end date."""
    return [start.isoformat(), (end + timedelta(days=1)).isoformat()]


def _property_filter(process_id: str, value) -> dict:
    return callback(lambda g: g.add(process_id, x={"from_parameter": "value"}, y=value))


def _load(g, collection, grid, start, end, bands, properties=None):
    args = {"id": collection, "spatial_extent": spatial_extent(grid), "temporal_extent": temporal_extent(start, end),
            "bands": list(bands)}
    if properties:
        args["properties"] = properties
    return g.add("load_collection", **args)


def _resample(g, data, grid, method):
    return g.add("resample_spatial", data=data, resolution=grid["resolution"], projection=grid["epsg"],
                 method=method, align="upper-left")


def _statistic(g: ProcessGraph, statistic: str):
    if statistic in ("median", "mean"):
        return g.add(statistic, data=DATA, ignore_nodata=True)
    if statistic.startswith("p"):
        q = g.add("quantiles", data=DATA, probabilities=[float(statistic[1:]) / 100], ignore_nodata=True)
        return g.add("array_element", data=q, index=0)
    raise ValueError(f"Unsupported statistic {statistic}")


def s2_composite_graph(grid: dict, start: date, end: date, statistic: str = "median",
                       scl_invalid=DEFAULT_SCL_INVALID, cloud_probability_max: float | None = None,
                       mask_dilation_px: int = 0, collection: str = S2_COLLECTION,
                       max_cloud_cover: float | None = None) -> dict:
    """Window composite: SCL/CLD/range mask (+ box dilation) -> temporal statistic + valid_count, one GTiff."""
    g = ProcessGraph()
    bands = [*S2_BANDS, "SCL"] + (["CLD"] if cloud_probability_max is not None else [])
    props = {"eo:cloud_cover": _property_filter("lte", max_cloud_cover)} if max_cloud_cover is not None else None
    cube = _resample(g, _load(g, collection, grid, start, end, bands, props), grid, "near")
    bad = g.add("reduce_dimension", data=cube, dimension="bands", reducer=callback(
        lambda c: _invalid_expression(c, scl_invalid, S2_BANDS, cloud_probability_max)))
    cloudy = [v for v in scl_invalid if int(v) != 0]   # s2.clear_mask dilates clouds/shadows, not no-data
    if mask_dilation_px > 0 and (cloudy or cloud_probability_max is not None):
        cloud = g.add("reduce_dimension", data=cube, dimension="bands", reducer=callback(
            lambda c: c.add("if", value=_invalid_expression(c, cloudy, (), cloud_probability_max), accept=1, reject=0)))
        size = 2 * int(mask_dilation_px) + 1
        spread = g.add("apply_kernel", data=cloud, kernel=[[1] * size for _ in range(size)], factor=1, border=0,
                       replace_invalid=0)
        dilated = g.add("apply", data=spread, process=callback(lambda c: c.add("gt", x=X, y=0)))
        bad = g.add("merge_cubes", cube1=bad, cube2=dilated, overlap_resolver=callback(
            lambda c: c.add("or", x=X, y={"from_parameter": "y"})))
    masked = g.add("mask", data=g.add("filter_bands", data=cube, bands=list(S2_BANDS)), mask=bad)
    comp = g.add("reduce_dimension", data=masked, dimension="t", reducer=callback(lambda c: _statistic(c, statistic)))
    count = g.add("reduce_dimension", data=g.add("filter_bands", data=masked, bands=["B02"]), dimension="t",
                  reducer=callback(lambda c: c.add("count", data=DATA)))
    count = g.add("rename_labels", data=count, dimension="bands", target=["valid_count"], source=["B02"])
    merged = g.add("merge_cubes", cube1=comp, cube2=count)
    return g.finish(g.add("save_result", data=merged, format="GTiff", options={}))


def s1_composite_graph(grid: dict, start: date, end: date, statistic: str = "mean", orbit: str | None = None,
                       coefficient: str = "sigma0-ellipsoid", card4l: bool = False,
                       collection: str = S1_COLLECTION) -> dict:
    """Window composite of linear VV/VH backscatter (values <= 0 dropped as in `geocrop.s1.composite`)."""
    if coefficient not in S1_COEFFICIENTS:
        raise ValueError(f"coefficient must be one of {S1_COEFFICIENTS}")
    if statistic not in ("mean", "median"):
        raise ValueError("Sentinel-1 statistic must be mean or median")
    g = ProcessGraph()
    props = {"sat:orbit_state": _property_filter("eq", ORBIT_STATE[orbit])} if orbit else None
    cube = _load(g, collection, grid, start, end, S1_POLARISATIONS, props)
    if card4l:
        cube = g.add("ard_normalized_radar_backscatter", data=cube, elevation_model=None, contributing_area=False,
                     ellipsoid_incidence_angle=False, noise_removal=True)
        cube = g.add("filter_bands", data=cube, bands=list(S1_POLARISATIONS))
    else:
        cube = g.add("sar_backscatter", data=cube, coefficient=coefficient, elevation_model=None, mask=False,
                     contributing_area=False, local_incidence_angle=False, ellipsoid_incidence_angle=False,
                     noise_removal=True, options={})
    cube = _resample(g, cube, grid, "bilinear")
    cube = g.add("apply", data=cube, process=callback(lambda c: c.add("if", value=c.add("gt", x=X, y=0), accept=X)))
    comp = g.add("reduce_dimension", data=cube, dimension="t", reducer=callback(lambda c: _statistic(c, statistic)))
    return g.finish(g.add("save_result", data=comp, format="GTiff", options={}))


def s2_scenes_graph(grid: dict, start: date, end: date, with_cld: bool = False, collection: str = S2_COLLECTION,
                    max_cloud_cover: float | None = None) -> dict:
    """Per-acquisition DN and SCL on the study CRS/resolution; masking stays local (exact GeoCrop rules)."""
    g = ProcessGraph()
    bands = [*S2_BANDS, "SCL"] + (["CLD"] if with_cld else [])
    props = {"eo:cloud_cover": _property_filter("lte", max_cloud_cover)} if max_cloud_cover is not None else None
    cube = _resample(g, _load(g, collection, grid, start, end, bands, props), grid, "near")
    return g.finish(g.add("save_result", data=cube, format="GTiff", options={}))


def signature_labels(index_names) -> list[str]:
    return [f"{i}_{k}" for i in index_names for k in ("sum", "sumsq")] + ["n_valid", "n_pixels"]


def signatures_graph(grid: dict, start: date, end: date, geometries: dict, index_names=("NDVI", "BSI"),
                     scl_invalid=DEFAULT_SCL_INVALID, collection: str = S2_COLLECTION) -> dict:
    """Per-acquisition parcel sums -> mean/std/n_valid/n_pixels (one ``aggregate_spatial`` with ``sum``).

    Bands per index: sum of the index over clear pixels and sum of its square; plus the clear-pixel count and
    the pixel count. Means and standard deviations are derived locally (`read_signatures`)."""
    unknown = [i for i in index_names if i not in INDEX_EXPRESSIONS]
    if unknown:
        raise ValueError(f"Indices {unknown} have no openEO expression; supported: {sorted(INDEX_EXPRESSIONS)}")
    need = sorted({b for i in index_names for b in expression_bands(INDEX_EXPRESSIONS[i])})
    g = ProcessGraph()
    cube = _resample(g, _load(g, collection, grid, start, end, [*need, "SCL"]), grid, "near")

    def build(c: ProcessGraph):
        elements, memo = {}, {}
        bad = _invalid_expression(c, scl_invalid, elements=elements)

        def leaf(band):
            if band not in elements:
                elements[band] = c.add("array_element", data=DATA, label=band)
            return c.add("multiply", x=elements[band], y=DN_SCALE)
        values = []
        for name in index_names:
            v = _expression_node(c, INDEX_EXPRESSIONS[name], leaf, memo)
            values += [c.add("if", value=bad, accept=0, reject=v),
                       c.add("if", value=bad, accept=0, reject=c.add("multiply", x=v, y=v))]
        values += [c.add("if", value=bad, accept=0, reject=1), 1]
        return c.add("array_create", data=values)
    feats = g.add("apply_dimension", data=cube, dimension="bands", process=callback(build))
    feats = g.add("rename_labels", data=feats, dimension="bands", target=signature_labels(index_names))
    agg = g.add("aggregate_spatial", data=feats, geometries=geometries,
                reducer=callback(lambda c: c.add("sum", data=DATA, ignore_nodata=True)))
    return g.finish(g.add("save_result", data=agg, format="JSON", options={}))


def parcel_geometries(parcels, grid: dict, id_field: str = "parcel_id", interior_buffer_px: float = 1):
    """Interior-buffered parcels (as in `cbm.parcel_signatures`) as a WGS84 GeoJSON FeatureCollection."""
    import geopandas as gpd
    from shapely.geometry import mapping
    gdf = gpd.read_file(parcels) if not isinstance(parcels, gpd.GeoDataFrame) else parcels.copy()
    if id_field not in gdf.columns:
        raise ValueError(f"Parcel layer lacks the id field {id_field!r}")
    gdf = gdf.to_crs(epsg=grid["epsg"]).reset_index(drop=True)
    geoms = gdf.geometry.buffer(-interior_buffer_px * grid["resolution"]) if interior_buffer_px else gdf.geometry
    geoms = gpd.GeoSeries([g if not g.is_empty else o for g, o in zip(geoms, gdf.geometry)], crs=gdf.crs)
    geoms = geoms.to_crs("EPSG:4326")
    ids = gdf[id_field].astype(str).tolist()
    features = [{"type": "Feature", "properties": {"parcel_id": pid},
                 "geometry": json.loads(json.dumps(mapping(geom)))} for pid, geom in zip(ids, geoms)]
    return {"type": "FeatureCollection", "features": features}, ids


# --------------------------------------------------------------------------- recipe (all graphs of a study)

def _windows(cfg: dict) -> dict[str, tuple[date, date]]:
    out = {str(k): (date.fromisoformat(str(v["start"])), date.fromisoformat(str(v["end"])))
           for k, v in (cfg.get("windows") or {}).items()}
    return dict(sorted(out.items(), key=lambda kv: kv[1][0]))


def _process(name: str, summary: str, graph: dict) -> dict:
    return {"id": f"geocrop_{name}", "summary": summary,
            "description": "GeoCrop Engine recipe exported as an openEO process graph; "
                           "see docs/openeo.md for differences from local processing.",
            "process_graph": graph}


def graph_sha256(graph: dict) -> str:
    return hashlib.sha256(json.dumps(graph, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def build_recipe(cfg: dict, products=None) -> dict:
    """All openEO graphs for a resolved GeoCrop configuration (no network, no credentials)."""
    o = cfg["openeo"]
    products = list(products or o["products"])
    bad = [p for p in products if p not in PRODUCTS]
    if bad:
        raise ValueError(f"Unknown openEO products {bad}; choose from {PRODUCTS}")
    if not cfg["grid"].get("reference"):
        raise ValueError("grid.reference (the study reference grid from `geocrop grid`) is required for openEO")
    grid = reference_grid(cfg["grid"]["reference"])
    windows = _windows(cfg)
    c = cfg["compositing"]
    paths = cfg["paths"]
    items = []
    if ("s2_composites" in products or "s1_composites" in products) and not windows:
        raise ValueError("Composites need windows: in the configuration")
    if "s2_composites" in products:
        for w, (start, end) in windows.items():
            graph = s2_composite_graph(grid, start, end, c["statistic"], c["scl_invalid"], c["cloud_probability_max"],
                                       int(c["mask_dilation_px"]), o["s2_collection"], o["max_cloud_cover"])
            items.append({"name": f"s2_composite_{w}", "product": "s2_composites", "window": w,
                          "start": start.isoformat(), "end": end.isoformat(),
                          "band_order": [*S2_BANDS, "valid_count"], "target": str(Path(paths["composites"]) / w),
                          "process": _process(f"s2_composite_{w}", f"Sentinel-2 {c['statistic']} composite, "
                                                                    f"window {w}", graph)})
    if "s1_composites" in products:
        orbits = ("ASC", "DESC") if cfg["s1"]["separate_orbits"] else (None,)
        for w, (start, end) in windows.items():
            for orbit in orbits:
                name = f"s1_composite_{w}" + (f"_{orbit}" if orbit else "")
                graph = s1_composite_graph(grid, start, end, cfg["s1"]["statistic"], orbit, o["s1_coefficient"],
                                           bool(o["s1_card4l"]), o["s1_collection"])
                items.append({"name": name, "product": "s1_composites", "window": w, "orbit": orbit,
                              "start": start.isoformat(), "end": end.isoformat(), "band_order": list(S1_POLARISATIONS),
                              "target": str(Path(paths["composites"]) / w),
                              "process": _process(name, f"Sentinel-1 {cfg['s1']['statistic']} backscatter composite "
                                                        f"({o['s1_coefficient'] if not o['s1_card4l'] else 'CARD4L'}),"
                                                        f" window {w}", graph)})
    start, end = _period(cfg, windows)
    if "s2_scenes" in products:
        with_cld = c["cloud_probability_max"] is not None
        graph = s2_scenes_graph(grid, start, end, with_cld, o["s2_collection"], o["max_cloud_cover"])
        items.append({"name": "s2_scenes", "product": "s2_scenes", "start": start.isoformat(), "end": end.isoformat(),
                      "band_order": [*S2_BANDS, "SCL"] + (["CLD"] if with_cld else []),
                      "target": str(Path(paths["s2_scenes"])),
                      "process": _process("s2_scenes", "Sentinel-2 L2A scenes on the study grid", graph)})
    if "signatures" in products:
        p = cfg["parcels"]
        geometries, ids = parcel_geometries(paths["parcels"], grid, p["id_field"], p["interior_buffer_px"])
        names = list(o["signature_indices"])
        graph = signatures_graph(grid, start, end, geometries, names, c["scl_invalid"], o["s2_collection"])
        items.append({"name": "signatures", "product": "signatures", "start": start.isoformat(),
                      "end": end.isoformat(), "band_order": signature_labels(names), "indices": names,
                      "parcel_ids": ids, "target": str(Path(o["work_dir"]) / "signatures.csv"),
                      "process": _process("signatures", "Parcel signatures (aggregate_spatial)", graph)})
    for item in items:
        item["process_graph_sha256"] = graph_sha256(item["process"]["process_graph"])
    return {"geocrop_version": __version__, "backend_url": o["url"], "reference_grid": grid,
            "collections": {"s2": o["s2_collection"], "s1": o["s1_collection"]}, "dn_scale": DN_SCALE,
            "compositing": {k: c[k] for k in ("statistic", "scl_invalid", "cloud_probability_max", "mask_dilation_px",
                                               "min_valid_observations", "temporal_screening")},
            "s1": {"statistic": cfg["s1"]["statistic"], "separate_orbits": cfg["s1"]["separate_orbits"],
                   "coefficient": o["s1_coefficient"], "card4l": bool(o["s1_card4l"])},
            "differences": list(DIFFERENCES), "graphs": items}


def _period(cfg: dict, windows: dict) -> tuple[date, date]:
    o, t = cfg["openeo"], cfg["timeseries"]
    start = o.get("start") or t.get("start") or (min(s for s, _ in windows.values()) if windows else None)
    end = o.get("end") or t.get("end") or (max(e for _, e in windows.values()) if windows else None)
    if not (start and end):
        return date(1970, 1, 1), date(1970, 1, 1)
    return date.fromisoformat(str(start)), date.fromisoformat(str(end))


def write_recipe(recipe: dict, out_dir: str | Path) -> dict:
    """One openEO process JSON per graph plus ``recipe.json`` (metadata, hashes, differences)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    files = []
    for item in recipe["graphs"]:
        path = out / f"{item['name']}.json"
        path.write_text(json.dumps(item["process"], indent=2, sort_keys=True) + "\n", encoding="utf8")
        files.append(str(path))
    meta = {**recipe, "graphs": [{k: v for k, v in i.items() if k not in ("process", "parcel_ids")}
                                 | {"file": f"{i['name']}.json"} for i in recipe["graphs"]]}
    (out / "recipe.json").write_text(json.dumps(meta, indent=2, default=str) + "\n", encoding="utf8")
    return {"recipe": str(out / "recipe.json"), "graphs": files}


# --------------------------------------------------------------------------- connection and execution

def _client():
    try:
        import openeo
    except ImportError as exc:
        raise RuntimeError(INSTALL_HINT) from exc
    return openeo


def credentials_available() -> bool:
    return bool(os.environ.get(ENV_CLIENT_ID) and os.environ.get(ENV_CLIENT_SECRET))


def connect(url: str | None = None, auth: str = "auto"):
    """Authenticated openEO connection; secrets come only from the environment and are never stored."""
    if auth not in ("auto", "client-credentials", "device"):
        raise ValueError("auth must be auto, client-credentials or device")
    openeo = _client()
    conn = openeo.connect(os.environ.get(ENV_URL) or url or CDSE_URL)
    provider = os.environ.get(ENV_PROVIDER) or None
    if auth != "device" and credentials_available():
        return conn.authenticate_oidc_client_credentials(client_id=os.environ[ENV_CLIENT_ID],
                                                         client_secret=os.environ[ENV_CLIENT_SECRET],
                                                         provider_id=provider)
    if auth == "client-credentials":
        raise RuntimeError(f"Set {ENV_CLIENT_ID} and {ENV_CLIENT_SECRET} for client-credentials authentication")
    return conn.authenticate_oidc_device(provider_id=provider, store_refresh_token=False)


def execute(recipe: dict, connection, work_dir: str | Path, job_options: dict | None = None,
            log=print) -> list[dict]:
    """Run every graph as a batch job, download the results and convert them into GeoCrop contracts."""
    work = Path(work_dir)
    reference = recipe["reference_grid"]["path"]
    client = _client()
    results = []
    for item in recipe["graphs"]:
        cube = connection.datacube_from_flat_graph(item["process"]["process_graph"])
        job = cube.create_job(title=f"geocrop {item['name']}", job_options=job_options)
        log(f"[openeo] {item['name']}: job {job.job_id}")
        job.start_and_wait()
        files = [Path(f) for f in job.get_results().download_files(work / "downloads" / item["name"])]
        backend = {"type": "openeo", "url": recipe["backend_url"], "job_id": job.job_id,
                   "process_graph_sha256": item["process_graph_sha256"], "openeo_client": client.__version__,
                   "differences": recipe["differences"]}
        results.append(convert(item, files, reference, backend, recipe))
    return results


def convert(item: dict, files, reference, backend: dict, recipe: dict) -> dict:
    """Write downloaded result files of one graph into the GeoCrop contract named by ``item['product']``."""
    tifs = sorted(Path(f) for f in files if Path(f).suffix.lower() in (".tif", ".tiff"))
    product = item["product"]
    if product == "signatures":
        data = [Path(f) for f in files if Path(f).suffix.lower() in (".json", ".csv")
                and Path(f).name not in ("job-results.json",)]
        if len(data) != 1:
            raise ValueError(f"Expected one signature result (JSON/CSV), got {[str(f) for f in files]}")
        table = read_signatures(data[0], item["parcel_ids"], item["indices"])
        return {"product": product, "rows": len(table), "path": str(write_table(table, item["target"]))}
    if product == "s2_scenes":
        written = write_scenes(tifs, item["target"], reference, item["band_order"], backend)
        return {"product": product, "scenes": len(written), "path": item["target"]}
    if len(tifs) != 1:
        raise ValueError(f"{item['name']}: expected one GeoTIFF, got {[str(t) for t in tifs]}")
    if product == "s2_composites":
        write_s2_composite(tifs[0], item["target"], reference, recipe["compositing"], backend,
                           recipe["compositing"]["min_valid_observations"])
    else:
        write_s1_composite(tifs[0], item["target"], reference, item.get("orbit"), recipe["s1"], backend)
    return {"product": product, "window": item.get("window"), "path": item["target"]}


# --------------------------------------------------------------------------- contract conversion (local files)

def _check_coverage(src, ref):
    left, bottom, right, top = transform_bounds(src.crs, ref.crs, *src.bounds)
    rb = ref.bounds
    tol = abs(ref.transform.a) * 0.5
    if left > rb.left + tol or bottom > rb.bottom + tol or right < rb.right - tol or top < rb.top - tol:
        raise ValueError(f"{src.name}: openEO result {left, bottom, right, top} does not cover the study grid "
                         f"{tuple(rb)}")


def alignment_method(src, ref) -> str:
    """'exact' when the result lies on the reference lattice (same CRS/resolution, integer pixel offset)."""
    if src.crs != ref.crs:
        return "resampled"
    st, rt = src.transform, ref.transform
    if not (np.isclose(st.a, rt.a) and np.isclose(st.e, rt.e) and st.b == 0 and st.d == 0):
        return "resampled"
    dx, dy = (st.c - rt.c) / rt.a, (st.f - rt.f) / rt.e
    return "exact" if np.isclose(dx, round(dx), atol=1e-6) and np.isclose(dy, round(dy), atol=1e-6) else "resampled"


def _read_on_grid(src, ref, band: int, continuous: bool) -> np.ndarray:
    exact = alignment_method(src, ref) == "exact"
    method = Resampling.nearest if exact or not continuous else Resampling.bilinear
    return read_aligned(src, ref.transform, ref.crs, ref.shape, method, band=band, dtype="float32", fill=np.nan)


def _open_result(path, ref, expected_bands: int):
    src = rasterio.open(path)
    if src.count != expected_bands:
        src.close()
        raise ValueError(f"{path}: {src.count} bands, expected {expected_bands}")
    if src.crs is None:
        src.close()
        raise ValueError(f"{path}: result has no CRS")
    _check_coverage(src, ref)
    return src


def _dn_to_reflectance(dn: np.ndarray) -> np.ndarray:
    ok = np.isfinite(dn) & (dn > 0) & (dn <= DN_MAX)
    return np.where(ok, dn * DN_SCALE, np.nan).astype("float32")


def _check_red(red: np.ndarray, where: str):
    red = red[np.isfinite(red)]
    if red.size and (red < 0).mean() > MAX_NEGATIVE_RED:
        raise ValueError(f"{where}: {(red < 0).mean():.0%} negative red reflectance; check the DN offset harmonisation")


def write_s2_composite(result: str | Path, out_dir: str | Path, reference: str | Path, compositing: dict,
                       backend: dict, min_valid_observations: int = 1) -> Path:
    """openEO composite GeoTIFF (B02 ... B12 DN statistic, valid_count) -> GeoCrop composite folder."""
    out = Path(out_dir)
    with rasterio.open(reference) as ref:
        with _open_result(result, ref, len(S2_BANDS) + 1) as src:
            method = alignment_method(src, ref)
            count = _read_on_grid(src, ref, len(S2_BANDS) + 1, continuous=False)
            count = np.where(np.isfinite(count) & (count > 0), np.rint(count), 0)
            enough = count >= max(int(min_valid_observations), 1)
            bands = {}
            for i, b in enumerate(S2_BANDS, 1):
                v = _read_on_grid(src, ref, i, continuous=True) * DN_SCALE
                bands[b] = np.where(enough & np.isfinite(v), v, np.nan).astype("float32")
        _check_red(bands["B04"], str(result))
        out.mkdir(parents=True, exist_ok=True)
        for b, v in bands.items():
            with rasterio.open(out / f"{b}.tif", "w", **float_profile(ref)) as dst:
                dst.write(np.where(np.isfinite(v), v, NODATA).astype("float32"), 1)
        with rasterio.open(out / "valid_count.tif", "w", **gtiff_profile(ref, count=1, dtype="uint16", nodata=None)) as dst:
            dst.write(count.astype("uint16"), 1)
    (out / "composite.json").write_text(json.dumps({
        "sensor": "S2", "acquisitions": {}, "statistic": compositing["statistic"],
        "scl_invalid": list(compositing["scl_invalid"]), "cloud_probability_max": compositing["cloud_probability_max"],
        "mask_dilation_px": compositing["mask_dilation_px"], "min_valid_observations": min_valid_observations,
        "temporal_screening": False, "temporal_screening_requested": bool(compositing.get("temporal_screening")),
        "reference_grid": str(reference), "grid_alignment": method,
        "backend": backend}, indent=2, default=str), encoding="utf8")
    return out


def write_s1_composite(result: str | Path, out_dir: str | Path, reference: str | Path, orbit: str | None,
                       s1: dict, backend: dict) -> Path:
    """openEO VV/VH GeoTIFF -> ``[ASC_|DESC_]VV.tif`` / ``VH.tif`` and ``composite_s1.json`` (merged per window)."""
    out = Path(out_dir)
    prefix = f"{orbit}_" if orbit else ""
    with rasterio.open(reference) as ref:
        with _open_result(result, ref, len(S1_POLARISATIONS)) as src:
            method = alignment_method(src, ref)
            data = {p: _read_on_grid(src, ref, i, continuous=True) for i, p in enumerate(S1_POLARISATIONS, 1)}
        out.mkdir(parents=True, exist_ok=True)
        for pol, v in data.items():
            with rasterio.open(out / f"{prefix}{pol}.tif", "w", **float_profile(ref)) as dst:
                dst.write(np.where(np.isfinite(v) & (v > 0), v, NODATA).astype("float32"), 1)
                dst.set_band_description(1, f"{s1.get('coefficient', 'sigma0')}_{pol}_linear")
    meta_path = out / "composite_s1.json"
    meta = json.loads(meta_path.read_text(encoding="utf8")) if meta_path.exists() else {}
    groups = meta.get("groups", {}) if meta.get("backend", {}).get("type") == "openeo" else {}
    groups[orbit or "ALL"] = [f"openeo:{backend.get('job_id')}"]
    meta_path.write_text(json.dumps({
        "sensor": "S1", "statistic": s1["statistic"], "domain": "linear", "separate_orbits": bool(orbit),
        "groups": groups, "backscatter": "CARD4L gamma0-terrain" if s1.get("card4l") else s1.get("coefficient"),
        "grid_alignment": method, "backend": backend}, indent=2, default=str), encoding="utf8")
    return out


_DATE = re.compile(r"(\d{4})-?(\d{2})-?(\d{2})")


def write_scenes(results, out_root: str | Path, reference: str | Path, band_order, backend: dict) -> list[Path]:
    """Per-date openEO GeoTIFFs (``openEO_YYYY-MM-DDZ.tif``) -> scene folders ``openeo_<date>/``."""
    from .stac_import import grid_identity
    grid_id = grid_identity(reference)
    written = []
    with rasterio.open(reference) as ref:
        for path in sorted(Path(r) for r in results):
            m = _DATE.search(path.stem)
            if not m:
                raise ValueError(f"{path.name}: no acquisition date in the file name")
            day = date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            with _open_result(path, ref, len(band_order)) as src:
                arrays = {name: _read_on_grid(src, ref, i, continuous=name not in ("SCL", "CLD"))
                          for i, name in enumerate(band_order, 1)}
            scl = np.where(np.isfinite(arrays["SCL"]), arrays["SCL"], 0).astype("uint8")
            if not (scl > 0).any():
                continue
            bands = {b: _dn_to_reflectance(arrays[b]) for b in S2_BANDS}
            clear = (scl > 0) & ~np.isin(scl, list(DEFAULT_SCL_INVALID))
            _check_red(np.where(clear, bands["B04"], np.nan), str(path))
            folder = Path(out_root) / f"openeo_{day.isoformat()}"
            folder.mkdir(parents=True, exist_ok=True)
            for b, v in bands.items():
                with rasterio.open(folder / f"{b}.tif", "w", **float_profile(ref)) as dst:
                    dst.write(np.where(np.isfinite(v), v, NODATA).astype("float32"), 1)
            uint8 = gtiff_profile(ref, count=1, dtype="uint8", nodata=None)
            with rasterio.open(folder / "SCL.tif", "w", **uint8) as dst:
                dst.write(scl, 1)
            if "CLD" in arrays:
                cld = np.clip(np.where(np.isfinite(arrays["CLD"]), arrays["CLD"], 0), 0, 100).astype("uint8")
                with rasterio.open(folder / "CLDPRB.tif", "w", **uint8) as dst:
                    dst.write(cld, 1)
            scene_meta.write(folder, sensor="S2", datetime=day.isoformat(), source=f"openeo:{path.name}",
                             grid_id=grid_id, clear_fraction=round(float(clear.sum() / max((scl > 0).sum(), 1)), 4),
                             backend=backend, note="openEO result on the reference grid; DN scaled by 1e-4")
            written.append(folder)
    return written


def _signature_rows(path: Path, n_bands: int):
    """(date, feature_index, values) from the openEO aggregate_spatial JSON (date -> per feature band values)
    or CSV (date, feature_index, band columns in order) result."""
    if path.suffix.lower() == ".json":
        body = json.loads(path.read_text(encoding="utf8"))
        for day, features in body.items():
            for k, values in enumerate(features or []):
                if values:
                    yield day, k, values
        return
    import pandas as pd
    frame = pd.read_csv(path)
    cols = [c for c in frame.columns if c not in ("date", "feature_index")]
    if len(cols) != n_bands:
        raise ValueError(f"{path}: {len(cols)} value columns, expected {n_bands}")
    for row in frame.itertuples(index=False):
        rec = row._asdict()
        yield rec["date"], int(rec["feature_index"]), [rec[c] for c in cols]


def read_signatures(path: str | Path, parcel_ids, index_names):
    """openEO signature sums -> the `cbm.parcel_signatures` long table (parcel_id, date, index, mean, std, ...)."""
    import pandas as pd
    labels = signature_labels(index_names)
    sums: dict[tuple[str, int], np.ndarray] = {}
    for day, k, values in _signature_rows(Path(path), len(labels)):
        if len(values) != len(labels):
            raise ValueError(f"{path}: {len(values)} values per feature, expected {len(labels)}")
        if not 0 <= k < len(parcel_ids):
            raise ValueError(f"{path}: feature index {k} outside the {len(parcel_ids)} uploaded parcels")
        sums[(str(day)[:10], k)] = np.asarray([np.nan if v is None else float(v) for v in values])
    n_pixels = {}
    for (_, k), v in sums.items():
        n_pixels[k] = max(n_pixels.get(k, 0), int(np.nan_to_num(v[-1])))
    rows = []
    for (day, k), v in sorted(sums.items()):
        c = int(np.nan_to_num(v[-2]))
        for j, name in enumerate(index_names):
            s, ss = v[2 * j], v[2 * j + 1]
            mean = s / c if c else np.nan
            std = float(np.sqrt(max(ss / c - mean**2, 0))) if c else np.nan
            rows.append({"parcel_id": str(parcel_ids[k]), "date": day, "index": name, "mean": mean, "std": std,
                         "n_valid": c, "n_pixels": n_pixels[k]})
    return pd.DataFrame(rows, columns=["parcel_id", "date", "index", "mean", "std", "n_valid", "n_pixels"])


def write_table(table, output: str | Path) -> Path:
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix.lower() == ".parquet":
        table.to_parquet(out, index=False)
    else:
        table.to_csv(out, index=False)
    return out
