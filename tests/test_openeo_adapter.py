"""openEO back-end adapter: offline process-graph construction (golden snapshots), parity of the graph's band
math and masks with GeoCrop's local rules, conversion of synthetic downloaded results into GeoCrop contracts,
and credential-gated network checks against CDSE (skipped without CDSE_OPENEO_* variables).

Regenerate the golden snapshots after an intended graph change with GEOCROP_UPDATE_GOLDEN=1.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import date
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

from geocrop import indices
from geocrop import openeo_adapter as oa
from geocrop import scenes as scene_meta
from geocrop.config import ConfigError, load
from geocrop.indices import S2_BANDS
from geocrop.raster import NODATA
from geocrop.s2 import DEFAULT_SCL_INVALID, clear_mask

GOLDEN = Path(__file__).parent / "golden" / "openeo"
ORIGIN = (500000.0, 5000060.0)
SHAPE = (6, 8)


def _reference(path: Path) -> Path:
    with rasterio.open(path, "w", driver="GTiff", width=SHAPE[1], height=SHAPE[0], count=1, dtype="uint8",
                       crs="EPSG:32634", transform=from_origin(*ORIGIN, 10, 10), nodata=0) as ds:
        ds.write(np.ones((1, *SHAPE), "uint8"))
    return path


def _parcels(path: Path) -> Path:
    x0, y1 = ORIGIN
    frame = gpd.GeoDataFrame({"parcel_id": ["P1", "P2"]},
                             geometry=[box(x0, y1 - 60, x0 + 40, y1), box(x0 + 40, y1 - 60, x0 + 80, y1)],
                             crs="EPSG:32634")
    frame.to_file(path, driver="GPKG")
    return path


@pytest.fixture()
def study(tmp_path):
    ref = _reference(tmp_path / "grid.tif")
    parcels = _parcels(tmp_path / "parcels.gpkg")
    cfg = load(None, {
        "grid": {"reference": str(ref)},
        "windows": {"spring": {"start": "2023-04-01", "end": "2023-04-30"},
                    "summer": {"start": "2023-07-01", "end": "2023-07-31"}},
        "compositing": {"mask_dilation_px": 1, "statistic": "p75", "cloud_probability_max": 40},
        "s1": {"separate_orbits": True},
        "paths": {"composites": str(tmp_path / "composites"), "s2_scenes": str(tmp_path / "s2"),
                  "parcels": str(parcels)},
        "openeo": {"products": list(oa.PRODUCTS), "work_dir": str(tmp_path / "openeo"),
                   "signature_indices": ["NDVI", "BSI", "EVI"]},
    })
    return cfg, tmp_path


# --------------------------------------------------------------------------- graph construction

def test_golden_process_graphs(study):
    cfg, _ = study
    recipe = oa.build_recipe(cfg)
    names = [g["name"] for g in recipe["graphs"]]
    assert names == ["s2_composite_spring", "s2_composite_summer", "s1_composite_spring_ASC",
                     "s1_composite_spring_DESC", "s1_composite_summer_ASC", "s1_composite_summer_DESC",
                     "s2_scenes", "signatures"]
    update = os.environ.get("GEOCROP_UPDATE_GOLDEN") == "1"
    for name in ("s2_composite_spring", "s1_composite_spring_ASC", "s2_scenes", "signatures"):
        process = next(g["process"] for g in recipe["graphs"] if g["name"] == name)
        body = json.loads(json.dumps(process, sort_keys=True))
        path = GOLDEN / f"{name}.json"
        if update:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf8")
        assert body == json.loads(path.read_text(encoding="utf8")), f"{name} differs from its golden snapshot"


def test_graph_basics(study):
    cfg, _ = study
    recipe = oa.build_recipe(cfg, ["s2_composites"])
    graph = recipe["graphs"][0]["process"]["process_graph"]
    lc = graph["loadcollection1"]["arguments"]
    assert lc["id"] == "SENTINEL2_L2A"
    assert lc["temporal_extent"] == ["2023-04-01", "2023-05-01"]          # inclusive window -> exclusive end
    assert lc["bands"] == [*S2_BANDS, "SCL", "CLD"]
    assert lc["spatial_extent"] == {"west": 499980.0, "south": 4999980.0, "east": 500100.0, "north": 5000080.0,
                                    "crs": 32634}
    assert graph["resamplespatial1"]["arguments"]["projection"] == 32634
    assert graph["applykernel1"]["arguments"]["kernel"] == [[1, 1, 1]] * 3
    assert [n for n, v in graph.items() if v.get("result")] == ["saveresult1"]
    assert recipe["graphs"][0]["process_graph_sha256"] == oa.graph_sha256(graph)


def test_write_recipe_and_dry_run_cli(study, monkeypatch):
    cfg, tmp = study
    import yaml
    cfg_path = tmp / "study.yaml"
    raw = {k: cfg[k] for k in ("grid", "windows", "compositing", "s1", "paths", "openeo")}
    cfg_path.write_text(yaml.safe_dump(raw), encoding="utf8")
    for var in (oa.ENV_CLIENT_ID, oa.ENV_CLIENT_SECRET):
        monkeypatch.delenv(var, raising=False)
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).parents[1] / "src")}
    env.pop(oa.ENV_CLIENT_ID, None)
    env.pop(oa.ENV_CLIENT_SECRET, None)
    out = subprocess.run([sys.executable, "-m", "geocrop.cli", "openeo-extract", "--config", str(cfg_path),
                          "--products", "s2_composites", "--dry-run", "--graphs", str(tmp / "graphs")],
                         capture_output=True, text=True, env=env, check=True)
    summary = json.loads(out.stdout)
    assert summary["dry_run"] is True and len(summary["graphs"]) == 2
    meta = json.loads((tmp / "graphs" / "recipe.json").read_text())
    assert meta["graphs"][0]["file"] == "s2_composite_spring.json" and meta["differences"]
    process = json.loads((tmp / "graphs" / "s2_composite_spring.json").read_text())
    assert process["id"] == "geocrop_s2_composite_spring" and "process_graph" in process
    out = subprocess.run([sys.executable, "-m", "geocrop.cli", "openeo-graph", "--config", str(cfg_path),
                          "--products", "s2_scenes", "--out", str(tmp / "g2"), "--stdout"],
                         capture_output=True, text=True, env=env, check=True)
    assert set(json.loads(out.stdout)) == {"s2_scenes"}


def test_config_validation():
    with pytest.raises(ConfigError):
        load(None, {"openeo": {"products": ["nonsense"]}})
    with pytest.raises(ConfigError):
        load(None, {"openeo": {"s1_coefficient": "beta0"}})


def test_recipe_needs_reference_grid():
    with pytest.raises(ValueError, match="grid.reference"):
        oa.build_recipe(load(None, {"windows": {"a": {"start": "2023-01-01", "end": "2023-01-31"}}}))


def test_unsupported_signature_index(study):
    cfg, _ = study
    cfg["openeo"]["signature_indices"] = ["NDVI", "VV_dB"]
    with pytest.raises(ValueError, match="no openEO expression"):
        oa.build_recipe(cfg, ["signatures"])


# --------------------------------------------------------------------------- parity with local rules

def _evaluate(graph: dict, params: dict):
    """Tiny evaluator for the openEO processes used in GeoCrop callbacks (NumPy arrays as pixel values)."""
    memo = {}

    def arg(v):
        if isinstance(v, dict) and "from_node" in v:
            return node(v["from_node"])
        if isinstance(v, dict) and "from_parameter" in v:
            return params[v["from_parameter"]]
        if isinstance(v, list):
            return [arg(x) for x in v]
        return v

    def node(name):
        if name not in memo:
            spec = graph[name]
            a = {k: arg(v) for k, v in spec["arguments"].items()}
            p = spec["process_id"]
            ops = {"eq": np.equal, "gt": np.greater, "lte": np.less_equal, "or": np.logical_or, "add": np.add,
                   "subtract": np.subtract, "multiply": np.multiply, "divide": np.divide}
            with np.errstate(divide="ignore", invalid="ignore"):
                if p == "array_element":
                    memo[name] = a["data"][a["label"]]
                elif p in ops:
                    memo[name] = ops[p](a["x"], a["y"])
                elif p == "sqrt":
                    memo[name] = np.sqrt(a["x"])
                elif p == "if":
                    memo[name] = np.where(a["value"], a["accept"], a.get("reject", np.nan))
                elif p == "constant":
                    memo[name] = a["x"]
                elif p == "array_create":
                    memo[name] = a["data"]
                else:
                    raise NotImplementedError(p)
        return memo[name]
    result = next(n for n, v in graph.items() if v.get("result"))
    return node(result)


def _random_scene(rng, n=4000):
    dn = {b: rng.integers(-50, 16500, n).astype("float64") for b in S2_BANDS}
    scl = rng.integers(0, 12, n)
    cld = rng.integers(0, 101, n).astype("float64")
    return dn, scl, cld


def test_mask_reducer_matches_local_masking():
    rng = np.random.default_rng(0)
    dn, scl, cld = _random_scene(rng)
    graph = oa.s2_composite_graph({"bounds": [0, 0, 10, 10], "resolution": 10, "epsg": 32634},
                                  date(2023, 4, 1), date(2023, 4, 30), cloud_probability_max=40)
    reducer = graph["reducedimension1"]["arguments"]["reducer"]["process_graph"]
    bad = _evaluate(reducer, {"data": {**dn, "SCL": scl, "CLD": cld}})
    # local: clear_mask + import (DN > 0) + composite reflectance range (<= 1.5)
    ok = clear_mask(scl, cld, DEFAULT_SCL_INVALID, 40)
    for b in S2_BANDS:
        refl = np.where(dn[b] > 0, dn[b] * oa.DN_SCALE, np.nan)
        ok &= np.isfinite(refl) & (refl >= -0.1) & (refl <= 1.5)
    np.testing.assert_array_equal(~bad, ok)


def test_dilation_kernel_equals_iterated_binary_dilation():
    from scipy.ndimage import binary_dilation, convolve
    rng = np.random.default_rng(1)
    cloud = rng.random((40, 40)) < 0.03
    for k in (1, 2, 3):
        box_kernel = np.ones((2 * k + 1, 2 * k + 1))
        spread = convolve(cloud.astype(int), box_kernel, mode="constant", cval=0) > 0
        np.testing.assert_array_equal(spread, binary_dilation(cloud, np.ones((3, 3), bool), iterations=k))


@pytest.mark.parametrize("name", sorted(oa.INDEX_EXPRESSIONS))
def test_index_expressions_match_registry(name):
    rng = np.random.default_rng(2)
    bands = {b: rng.uniform(0.01, 0.6, 500) for b in S2_BANDS}
    expected = indices.get(name).compute({b: v.astype("float32") for b, v in bands.items()})
    got = oa.evaluate_expression(oa.INDEX_EXPRESSIONS[name], bands)
    ok = np.isfinite(expected)
    np.testing.assert_allclose(got[ok], expected[ok], rtol=2e-4, atol=1e-5)


def test_signature_callback_values():
    rng = np.random.default_rng(3)
    dn, scl, _ = _random_scene(rng, 300)
    for b in S2_BANDS:
        dn[b] = np.abs(dn[b]) + 1
    graph = oa.signatures_graph({"bounds": [0, 0, 10, 10], "resolution": 10, "epsg": 32634},
                                date(2023, 4, 1), date(2023, 4, 30), {"type": "FeatureCollection", "features": []},
                                ("NDVI", "BSI"))
    cb = graph["applydimension1"]["arguments"]["process"]["process_graph"]
    values = _evaluate(cb, {"data": {**dn, "SCL": scl}})
    clear = clear_mask(scl, None)
    refl = {b: dn[b] * oa.DN_SCALE for b in S2_BANDS}
    ndvi = indices.get("NDVI").compute(refl)
    np.testing.assert_allclose(values[0], np.where(clear, ndvi, 0), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(values[1], np.where(clear, ndvi ** 2, 0), rtol=1e-5, atol=1e-6)
    np.testing.assert_array_equal(values[4], clear.astype(int))
    assert values[5] == 1
    assert graph["renamelabels1"]["arguments"]["target"] == oa.signature_labels(("NDVI", "BSI"))


# --------------------------------------------------------------------------- openeo client (optional)

def test_graphs_roundtrip_through_openeo_client(study):
    openeo = pytest.importorskip("openeo")
    from openeo.internal.graph_building import PGNode
    from openeo.rest.datacube import DataCube
    cfg, _ = study
    for item in oa.build_recipe(cfg)["graphs"]:
        flat = item["process"]["process_graph"]
        assert DataCube(graph=PGNode.from_flat_graph(flat), connection=None).flat_graph() == flat, item["name"]
    # the scenes graph is identical to what the client's own API builds
    grid = oa.reference_grid(cfg["grid"]["reference"])
    cube = openeo.DataCube.load_collection("SENTINEL2_L2A", connection=None,
                                           spatial_extent=oa.spatial_extent(grid),
                                           temporal_extent=oa.temporal_extent(date(2023, 4, 1), date(2023, 7, 31)),
                                           bands=[*S2_BANDS, "SCL", "CLD"], fetch_metadata=False)
    cube = cube.resample_spatial(resolution=10.0, projection=32634, method="near", align="upper-left")
    ours = oa.s2_scenes_graph(grid, date(2023, 4, 1), date(2023, 7, 31), with_cld=True)
    assert cube.save_result(format="GTiff").flat_graph() == ours


def test_missing_client_message(monkeypatch):
    monkeypatch.setitem(sys.modules, "openeo", None)
    with pytest.raises(RuntimeError, match="pip install"):
        oa.connect()


# --------------------------------------------------------------------------- contract conversion

def _write_result(path: Path, bands: np.ndarray, offset_px=(-2, -2), res=10.0, nodata=None):
    """A synthetic downloaded openEO GeoTIFF: padded extent (2 px) on the reference lattice."""
    transform = from_origin(ORIGIN[0] + offset_px[1] * res, ORIGIN[1] - offset_px[0] * res, res, res)
    with rasterio.open(path, "w", driver="GTiff", width=bands.shape[2], height=bands.shape[1], count=bands.shape[0],
                       dtype="float32", crs="EPSG:32634", transform=transform, nodata=nodata) as ds:
        ds.write(bands.astype("float32"))
    return path


def test_s2_composite_conversion_feeds_features(study):
    cfg, tmp = study
    recipe = oa.build_recipe(cfg, ["s2_composites"])
    h, w = SHAPE[0] + 4, SHAPE[1] + 4
    data = np.zeros((len(S2_BANDS) + 1, h, w))
    for i in range(len(S2_BANDS)):
        data[i] = 1000 + 100 * i
    data[2] = 500                                  # B04
    data[7] = 3000                                 # B8A
    data[len(S2_BANDS)] = 3                        # valid_count
    data[len(S2_BANDS), 2, 2] = 1                  # below min_valid_observations=2 -> no-data
    data[:len(S2_BANDS), 3, 3] = np.nan            # never clear
    data[len(S2_BANDS), 3, 3] = 0
    (tmp / "dl").mkdir()
    tif = _write_result(tmp / "dl" / "openEO.tif", data)
    recipe["compositing"]["min_valid_observations"] = 2
    for item in recipe["graphs"]:
        oa.convert(item, [tif, tmp / "dl" / "job-results.json"], cfg["grid"]["reference"],
                   {"type": "openeo", "job_id": "j-1"}, recipe)
    comp = Path(cfg["paths"]["composites"]) / "spring"
    # the result is padded by 2 px: its pixel (r, c) is pixel (r - 2, c - 2) of the reference grid
    with rasterio.open(comp / "B04.tif") as b04, rasterio.open(cfg["grid"]["reference"]) as ref:
        assert b04.transform == ref.transform and b04.shape == ref.shape and b04.nodata == NODATA
        red = b04.read(1)
    with rasterio.open(comp / "valid_count.tif") as vc:
        count = vc.read(1)
    assert count.dtype == np.uint16 and count[0, 0] == 1 and count[1, 1] == 0 and count[2, 2] == 3
    assert red[0, 0] == NODATA and red[1, 1] == NODATA and red[2, 2] == pytest.approx(0.05)
    with rasterio.open(comp / "B8A.tif") as ds:
        assert ds.read(1)[2, 2] == pytest.approx(0.3)
    meta = json.loads((comp / "composite.json").read_text())
    assert meta["backend"]["job_id"] == "j-1" and meta["grid_alignment"] == "exact"
    assert meta["temporal_screening"] is False and meta["min_valid_observations"] == 2
    # the summer window too, then the local feature stack builds from openEO composites
    from geocrop.features import build_stack
    stack = build_stack(cfg["paths"]["composites"], tmp / "stack.tif", periods=["spring", "summer"],
                        s1=[], deltas=["NDVI"], seasonal=[])
    with rasterio.open(stack) as ds:
        assert ds.count > 20 and ds.shape == SHAPE


def test_s2_composite_rejects_negative_red(study):
    cfg, tmp = study
    recipe = oa.build_recipe(cfg, ["s2_composites"])
    data = np.full((len(S2_BANDS) + 1, SHAPE[0] + 4, SHAPE[1] + 4), 1000.0)
    data[2] = -800
    tif = _write_result(tmp / "neg.tif", data)
    with pytest.raises(ValueError, match="negative red"):
        oa.write_s2_composite(tif, tmp / "c", cfg["grid"]["reference"], recipe["compositing"], {})


def test_conversion_rejects_wrong_band_count_and_coverage(study):
    cfg, tmp = study
    recipe = oa.build_recipe(cfg, ["s2_composites"])
    tif = _write_result(tmp / "few.tif", np.ones((3, SHAPE[0] + 4, SHAPE[1] + 4)))
    with pytest.raises(ValueError, match="bands"):
        oa.write_s2_composite(tif, tmp / "c", cfg["grid"]["reference"], recipe["compositing"], {})
    small = _write_result(tmp / "small.tif", np.ones((len(S2_BANDS) + 1, 3, 3)), offset_px=(0, 0))
    with pytest.raises(ValueError, match="does not cover"):
        oa.write_s2_composite(small, tmp / "c", cfg["grid"]["reference"], recipe["compositing"], {})


def test_misaligned_result_is_resampled(study):
    cfg, tmp = study
    recipe = oa.build_recipe(cfg, ["s2_composites"])
    data = np.full((len(S2_BANDS) + 1, SHAPE[0] + 4, SHAPE[1] + 4), 2000.0)
    data[-1] = 4
    tif = _write_result(tmp / "shift.tif", data, offset_px=(-2.5, -2.5))
    oa.write_s2_composite(tif, tmp / "c", cfg["grid"]["reference"], recipe["compositing"], {})
    meta = json.loads((tmp / "c" / "composite.json").read_text())
    assert meta["grid_alignment"] == "resampled"
    with rasterio.open(tmp / "c" / "B02.tif") as ds:
        assert np.allclose(ds.read(1), 0.2)


def test_s1_conversion_per_orbit(study):
    cfg, tmp = study
    recipe = oa.build_recipe(cfg, ["s1_composites"])
    items = [i for i in recipe["graphs"] if i["window"] == "spring"]
    for item, value in zip(items, (0.05, 0.02)):
        data = np.stack([np.full((SHAPE[0] + 4, SHAPE[1] + 4), value), np.full((SHAPE[0] + 4, SHAPE[1] + 4), 0.0)])
        tif = _write_result(tmp / f"{item['name']}.tif", data)
        oa.convert(item, [tif], cfg["grid"]["reference"], {"type": "openeo", "job_id": item["name"]}, recipe)
    comp = Path(cfg["paths"]["composites"]) / "spring"
    with rasterio.open(comp / "ASC_VV.tif") as a, rasterio.open(comp / "DESC_VV.tif") as d, \
            rasterio.open(comp / "ASC_VH.tif") as vh:
        assert np.allclose(a.read(1), 0.05) and np.allclose(d.read(1), 0.02)
        assert (vh.read(1) == NODATA).all()                       # zero backscatter is no-data
    meta = json.loads((comp / "composite_s1.json").read_text())
    assert set(meta["groups"]) == {"ASC", "DESC"} and meta["domain"] == "linear"


def test_scene_conversion_then_local_composite(study):
    cfg, tmp = study
    recipe = oa.build_recipe(cfg, ["s2_scenes"])
    item = recipe["graphs"][0]
    assert item["band_order"][-1] == "CLD"
    files = []
    for k, (day, scl_value) in enumerate((("2023-04-05", 4), ("2023-04-10", 9), ("2023-04-20", 5))):
        data = np.zeros((len(item["band_order"]), SHAPE[0] + 4, SHAPE[1] + 4))
        data[:len(S2_BANDS)] = 1000 * (k + 1)
        data[len(S2_BANDS)] = scl_value
        data[len(S2_BANDS) + 1] = 10
        files.append(_write_result(tmp / f"openEO_{day}Z.tif", data))
    oa.convert(item, files, cfg["grid"]["reference"], {"type": "openeo", "job_id": "j-s"}, recipe)
    found = scene_meta.discover(cfg["paths"]["s2_scenes"])
    assert [m["date"].isoformat() for m in found] == ["2023-04-05", "2023-04-10", "2023-04-20"]
    assert all((m["path"] / "CLDPRB.tif").exists() for m in found)
    from geocrop.s2 import composite
    out = composite([m["path"] for m in found], tmp / "local_comp", cfg["grid"]["reference"])
    with rasterio.open(out / "B02.tif") as ds, rasterio.open(out / "valid_count.tif") as vc:
        assert np.allclose(ds.read(1), 0.2)          # median of 0.1 and 0.3; the cloudy (SCL 9) scene is masked
        assert (vc.read(1) == 2).all()


@pytest.mark.parametrize("fmt", ["json", "csv"])
def test_signature_conversion(study, fmt):
    cfg, tmp = study
    recipe = oa.build_recipe(cfg, ["signatures"])
    item = recipe["graphs"][0]
    assert item["parcel_ids"] == ["P1", "P2"]
    geoms = item["process"]["process_graph"]["aggregatespatial1"]["arguments"]["geometries"]
    assert [f["properties"]["parcel_id"] for f in geoms["features"]] == ["P1", "P2"]
    # P1 has 3 clear pixels with NDVI 0.5, 0.6, 0.7 out of 8; P2 has none on that date
    ndvi = np.array([0.5, 0.6, 0.7])
    p1 = [ndvi.sum(), (ndvi ** 2).sum(), 0.3, 0.03, 0.9, 0.27, 3, 8]
    p2 = [0, 0, 0, 0, 0, 0, 0, 8]
    if fmt == "json":
        path = tmp / "timeseries.json"
        path.write_text(json.dumps({"2023-04-05T00:00:00Z": [p1, p2], "2023-04-10T00:00:00Z": [[], None]}))
    else:
        path = tmp / "timeseries.csv"
        cols = item["band_order"]
        lines = ["date,feature_index," + ",".join(cols),
                 "2023-04-05T00:00:00Z,0," + ",".join(map(str, p1)),
                 "2023-04-05T00:00:00Z,1," + ",".join(map(str, p2))]
        path.write_text("\n".join(lines) + "\n")
    res = oa.convert(item, [path], cfg["grid"]["reference"], {}, recipe)
    table = __import__("pandas").read_csv(res["path"])
    assert list(table.columns) == ["parcel_id", "date", "index", "mean", "std", "n_valid", "n_pixels"]
    row = table[(table.parcel_id == "P1") & (table["index"] == "NDVI")].iloc[0]
    assert row["mean"] == pytest.approx(0.6) and row["std"] == pytest.approx(ndvi.std())
    assert row["n_valid"] == 3 and row["n_pixels"] == 8 and row["date"] == "2023-04-05"
    assert np.isnan(table[(table.parcel_id == "P2") & (table["index"] == "BSI")].iloc[0]["mean"])
    # the table feeds the local marker computation
    from geocrop.cbm import markers
    markers(table, date(2023, 4, 1), date(2023, 7, 31))


# --------------------------------------------------------------------------- network (credentials required)

@pytest.mark.network
def test_cdse_validates_recipe(study):
    pytest.importorskip("openeo")
    if not oa.credentials_available():
        pytest.skip(f"set {oa.ENV_CLIENT_ID} and {oa.ENV_CLIENT_SECRET} to run CDSE openEO checks")
    cfg, _ = study
    conn = oa.connect(cfg["openeo"]["url"], auth="client-credentials")
    for item in oa.build_recipe(cfg, ["s2_composites", "s1_composites", "s2_scenes"])["graphs"][:3]:
        errors = conn.validate_process_graph(item["process"]["process_graph"])
        assert not errors, (item["name"], errors)
