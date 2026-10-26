"""Error and edge paths of the data-cube adapter (`geocrop.datacube`) on small synthetic xarray cubes.

Every check is offline: odc-stac loading is replaced by an in-memory loader, STAC search by a fake client.
Skipped without xarray (the minimal CI install), like tests/test_datacube.py.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pystac
import pytest
import rasterio
from rasterio.transform import from_origin

from geocrop import datacube as dc
from geocrop.indices import S2_BANDS

xr = pytest.importorskip("xarray")

CRS = "EPSG:32634"
ORIGIN = (500000.0, 5000080.0)
SHAPE = (6, 8)
TIMES = ["2023-07-03T09:48:18", "2023-07-08T09:48:20"]


def _reference(path: Path, crs=CRS) -> Path:
    with rasterio.open(path, "w", driver="GTiff", width=SHAPE[1], height=SHAPE[0], count=1, dtype="uint8",
                       crs=crs, transform=from_origin(*ORIGIN, 10, 10), nodata=0) as ds:
        ds.write(np.ones((1, *SHAPE), "uint8"))
    return path


@pytest.fixture()
def ref(tmp_path):
    return _reference(tmp_path / "grid.tif")


def _cube(nt=2, scl=4, red=None, times=TIMES, shape=SHAPE, x0=ORIGIN[0], seed=0, crs=CRS):
    rng = np.random.default_rng(seed)
    h, w = shape
    data = {b: (("time", "y", "x"), rng.uniform(0.02, 0.45, (nt, h, w)).astype("float32")) for b in S2_BANDS}
    if red is not None:
        data["B04"] = (("time", "y", "x"), np.broadcast_to(np.asarray(red, "float32"), (nt, h, w)).copy())
    if scl is not None:
        data["SCL"] = (("time", "y", "x"), np.broadcast_to(np.asarray(scl), (nt, h, w)).astype("uint8"))
    coords = {"time": np.array(times[:nt], "datetime64[ns]"), "x": x0 + 10 * (np.arange(w) + 0.5),
              "y": ORIGIN[1] - 10 * (np.arange(h) + 0.5)}
    if crs:
        coords["spatial_ref"] = ((), 0, {"crs_wkt": rasterio.crs.CRS.from_user_input(crs).to_wkt()})
    ds = xr.Dataset(data, coords=coords)
    for b in S2_BANDS:
        ds[b].attrs["units"] = "1"
    return ds


# --------------------------------------------------------------------------- grid helpers

def test_reference_without_crs_and_missing_odc_geo(tmp_path, ref, monkeypatch):
    nocrs = tmp_path / "nocrs.tif"
    with rasterio.open(nocrs, "w", driver="GTiff", width=2, height=2, count=1, dtype="uint8") as ds:
        ds.write(np.ones((1, 2, 2), "uint8"))
    with pytest.raises(dc.CubeError, match="no CRS"):
        dc.normalise_cube(_cube(), nocrs)
    monkeypatch.setitem(sys.modules, "odc.geo.geobox", None)
    with pytest.raises(RuntimeError, match="odc-stac"):
        dc.reference_geobox(ref)


def test_search_items_sorted_by_time(ref, monkeypatch):
    import pystac_client
    late = pystac.Item("b", None, None, datetime(2023, 7, 9, tzinfo=timezone.utc), {"datetime": "2023-07-09T00:00:00Z"})
    early = pystac.Item("a", None, None, datetime(2023, 7, 1, tzinfo=timezone.utc), {})
    seen = []

    class Client:
        def search(self, **kw):
            seen.append(kw)
            return type("S", (), {"items": lambda self: iter([late, early])})()
    monkeypatch.setattr(pystac_client.Client, "open", staticmethod(lambda endpoint: Client()))
    assert [i.id for i in dc.search_items(ref, "2023-07-01", "2023-07-31", max_cloud=None)] == ["a", "b"]
    assert seen[0]["query"] is None
    dc.search_items(ref, "2023-07-01", "2023-07-31")
    assert seen[1]["query"] == {"eo:cloud_cover": {"lt": 80}}
    monkeypatch.setitem(sys.modules, "pystac_client", None)
    with pytest.raises(RuntimeError, match="stac"):
        dc.search_items(ref, "2023-07-01", "2023-07-31")


# --------------------------------------------------------------------------- load_stac_cube with a fake loader

def _stac_item(item_id, when, offset_applied=True):
    item = pystac.Item(item_id, None, None, datetime.fromisoformat(when).replace(tzinfo=timezone.utc),
                       {"datetime": when + "Z", "earthsearch:boa_offset_applied": offset_applied})
    for band in (*S2_BANDS, "SCL"):
        asset = pystac.Asset(href=f"/nonexistent/{item_id}/{band}.tif")
        if band != "SCL":
            asset.extra_fields["raster:bands"] = [{"scale": 1e-4, "offset": -0.1}]
        item.add_asset(dc.ASSET_KEYS[band], asset)
    return item


@pytest.fixture()
def fake_odc(monkeypatch, ref):
    pytest.importorskip("odc.stac")
    state = {"scl": {}, "dn": {}}
    monkeypatch.setattr(dc, "reference_geobox", lambda reference: None)

    def load(items, assets, geobox, resampling, dtype, chunks, pool):
        nt = len(items)
        out = {}
        for key in assets:
            if key == dc.ASSET_KEYS["SCL"]:
                arr = np.stack([np.full(SHAPE, state["scl"].get(i.id, 4)) for i in items]).astype("uint8")
            else:
                red = key == dc.ASSET_KEYS["B04"]
                arr = np.stack([np.full(SHAPE, state["dn"].get(i.id, 3000) if red else 3000) for i in items])
            out[key] = (("time", "y", "x"), arr.astype(dtype))
        return xr.Dataset(out, coords={"time": np.arange(nt)})
    monkeypatch.setattr(dc, "_odc_load", load)
    return state


def test_load_stac_cube_screens_items(ref, fake_odc, monkeypatch):
    with pytest.raises(dc.CubeError, match="start and end"):
        dc.load_stac_cube(ref)
    monkeypatch.setattr(dc, "search_items", lambda *a, **k: [])
    with pytest.raises(dc.CubeError, match="No STAC items"):
        dc.load_stac_cube(ref, "2023-07-01", "2023-07-31")
    items = [_stac_item("EMPTY", "2023-07-01T10:00:00"), _stac_item("CLOUD", "2023-07-02T10:00:00"),
             _stac_item("NEG", "2023-07-03T10:00:00", offset_applied=False), _stac_item("OK", "2023-07-04T10:00:00")]
    fake_odc["scl"].update(EMPTY=0, CLOUD=9)
    fake_odc["dn"]["NEG"] = 500                     # 500 * 1e-4 - 0.1 < 0: offset applied twice
    monkeypatch.setattr(dc, "search_items", lambda *a, **k: items)
    cube = dc.load_stac_cube(ref, "2023-07-01", "2023-07-31", min_clear_fraction=0.5)
    report = {r["item"]: r["status"] for r in json.loads(cube.attrs["geocrop:items"])}
    assert report["EMPTY"] == "skipped: no coverage" and report["CLOUD"].startswith("skipped: clear fraction")
    assert report["NEG"].startswith("failed:") and report["OK"] == "loaded"
    assert list(cube["scene_id"].values) == ["OK"]
    np.testing.assert_allclose(cube["B04"].values, 0.3, rtol=1e-6)
    fake_odc["scl"]["OK"] = 9
    with pytest.raises(dc.CubeError, match="clear-fraction screen"):
        dc.load_stac_cube(ref, items=[items[1], items[3]], min_clear_fraction=0.5)


# --------------------------------------------------------------------------- opening and reading cubes

def test_open_cube_errors(tmp_path, monkeypatch):
    with pytest.raises(FileNotFoundError):
        dc.open_cube(tmp_path / "missing.zarr")
    (tmp_path / "odd.txt").write_text("x")
    with pytest.raises(dc.CubeError, match="unknown cube format"):
        dc.open_cube(tmp_path / "odd.txt")
    store = tmp_path / "store"
    store.mkdir()
    (store / ".zgroup").write_text("{}")
    monkeypatch.setitem(sys.modules, "zarr", None)
    with pytest.raises(RuntimeError, match="zarr"):
        dc.open_cube(store)
    (tmp_path / "bad.nc").write_bytes(b"not netcdf")

    def broken(*a, **k):
        raise ValueError("no engine")
    monkeypatch.setattr(xr, "open_dataset", broken)
    with pytest.raises(RuntimeError, match="netCDF4"):
        dc.open_cube(tmp_path / "bad.nc")


def test_as_dataset_shapes(ref):
    da = xr.DataArray(np.zeros((2, 3)), dims=("y", "x"))
    with pytest.raises(dc.CubeError, match="band dimension"):
        dc.normalise_cube(da, ref)
    with pytest.raises(dc.CubeError, match="Expected an xarray"):
        dc.normalise_cube(np.zeros(3), ref)
    cube = _cube(scl=None)
    stacked = cube.to_array("band").to_dataset(name="reflectance")
    stacked["other"] = stacked["reflectance"].isel(band=0, drop=True)
    stacked = stacked.assign_coords(tile=("time", ["34TDQ", "34TDQ"]))
    out = dc.normalise_cube(stacked, ref, assume_masked=True)        # one band-carrying variable + coords
    np.testing.assert_allclose(out["B08"].values, cube["B08"].values)
    two = stacked.assign(copy=stacked["reflectance"] * 1)
    with pytest.raises(dc.CubeError, match="one variable with dimension"):
        dc.normalise_cube(two, ref)


def test_dimension_and_band_aliases(ref):
    cube = _cube().rename({"x": "easting", "y": "northing"})
    out = dc.normalise_cube(cube, ref)
    assert set(out.dims) == {"time", "y", "x"}
    ambiguous = _cube().rename({"x": "lon"}).expand_dims(longitude=[0.0])
    with pytest.raises(dc.CubeError, match="Ambiguous x"):
        dc.normalise_cube(ambiguous, ref)
    dup = _cube().assign(red=lambda d: d["B04"])
    with pytest.raises(dc.CubeError, match="map to band B04"):
        dc.normalise_cube(dup, ref)


def test_crs_and_transform_errors(ref):
    bad = _cube(crs=None)
    bad.attrs["crs"] = "EPSG:not-a-code"
    with pytest.raises(dc.CubeError, match="Unreadable CRS"):
        dc.normalise_cube(bad, ref)
    flat = _cube().isel(x=0, drop=True)
    with pytest.raises(dc.CubeError, match="x and y dimensions"):
        dc.cube_transform(flat)
    one = _cube().isel(x=[0])
    with pytest.raises(dc.CubeError, match="single x"):
        dc.cube_transform(one)
    one["spatial_ref"].attrs["GeoTransform"] = f"{ORIGIN[0]} 10 0 {ORIGIN[1]} 0 -10"
    t = dc.cube_transform(one)
    assert (t.a, t.c) == (10.0, ORIGIN[0])
    irregular = _cube().assign_coords(x=ORIGIN[0] + np.array([5, 15, 25, 35, 45, 55, 65, 80.0]))
    with pytest.raises(dc.CubeError, match="Irregular x"):
        dc.cube_transform(irregular)
    with pytest.raises(dc.CubeError, match="x coordinates must increase"):
        dc.cube_transform(_cube().isel(x=slice(None, None, -1)))


def test_time_coordinate_handling(ref):
    import pandas as pd
    mixed = dc._utc_index(["2023-07-03T10:00:00+02:00", "2023-07-03T10:00:00+00:00"])
    assert list(mixed) == [pd.Timestamp("2023-07-03T08:00:00"), pd.Timestamp("2023-07-03T10:00:00")]
    with pytest.raises(dc.CubeError, match="Unparseable"):
        dc._utc_index(["yesterday-ish", object()])
    with pytest.raises(dc.CubeError, match="NaT"):
        dc._utc_index(pd.DatetimeIndex(["2023-07-03", None]))
    with pytest.raises(dc.CubeError, match="time dimension"):
        dc.normalise_cube(_cube().isel(time=0, drop=True), ref)


def test_cftime_calendar_is_converted(ref):
    pytest.importorskip("cftime")
    cube = _cube().assign_coords(time=xr.date_range("2023-07-03", periods=2, freq="5D", calendar="noleap",
                                                    use_cftime=True))
    out = dc.normalise_cube(cube, ref)
    assert str(out["time"].values[0])[:10] == "2023-07-03"
    weird = _cube().assign_coords(time=xr.date_range("2023-02-29", periods=2, freq="D", calendar="360_day",
                                                     use_cftime=True))
    with pytest.raises(dc.CubeError, match="calendar"):
        dc.normalise_cube(weird, ref)


def test_units_decisions(ref):
    cube = _cube()
    for b in S2_BANDS:
        cube[b] = cube[b] * 2
    out = dc.normalise_cube(cube, ref, units="reflectance", scale=0.5)
    np.testing.assert_allclose(out["B04"].values, _cube()["B04"].values, rtol=1e-6)
    with pytest.raises(dc.CubeError, match="units='dn' needs a scale"):
        dc.normalise_cube(_cube(), ref, units="dn")
    with pytest.raises(dc.CubeError, match="units must be"):
        dc.normalise_cube(_cube(), ref, units="percent")


def test_alignment_and_value_checks(ref):
    with pytest.raises(dc.GridMismatchError, match="does not overlap"):
        dc.normalise_cube(_cube(x0=ORIGIN[0] + 10000), ref)
    with pytest.raises(dc.CubeError, match="not reflectance"):
        dc.normalise_cube(_cube(red=5.0), ref, units="reflectance")
    gap = _cube()
    gap["B04"][0] = np.nan                                  # an empty time step is skipped by the value check
    assert np.isnan(dc.normalise_cube(gap, ref)["B04"].values[0]).all()


def test_scl_float_values(ref):
    scl = np.full((2, *SHAPE), 4.0)
    scl[0, 0, 0] = np.nan
    cube = _cube(scl=None).assign(SCL=(("time", "y", "x"), scl))
    out = dc.normalise_cube(cube, ref)
    assert out["SCL"].dtype == np.uint8 and out["SCL"].values[0, 0, 0] == 0
    cube["SCL"][1, 1, 1] = 4.5
    with pytest.raises(dc.CubeError, match="non-integer"):
        dc.normalise_cube(cube, ref)


def test_spatial_auxiliary_coordinates_are_dropped(ref):
    cube = _cube().assign_coords(lat2d=(("y", "x"), np.zeros(SHAPE)))
    assert "lat2d" not in dc.normalise_cube(cube, ref).coords


# --------------------------------------------------------------------------- cube -> scenes / composites

def test_cube_to_scenes_shape_mismatch_and_negative_red(ref, tmp_path):
    canon = dc.normalise_cube(_cube(red=-0.05), ref, max_negative_fraction=1.0)
    res = dc.cube_to_scenes(canon, ref, tmp_path / "scenes")
    assert all(r["status"].startswith("failed: 100% negative red") for r in res)
    small = canon.isel(x=slice(0, 4))
    small.attrs = canon.attrs
    with pytest.raises(dc.CubeError, match="differs from the grid"):
        dc.cube_to_scenes(small, ref, tmp_path / "scenes")


def test_scalar_conversion():
    assert dc._scalar(np.array(b"S2A")) == "S2A"
    assert dc._scalar(np.float64("nan")) is None
    assert dc._scalar(np.int64(3)) == 3


def test_cube_to_composites_needs_windows_and_skips_empty(ref, tmp_path):
    with pytest.raises(dc.CubeError, match="needs windows"):
        dc.cube_to_composites(_cube(), ref, tmp_path / "comp")
    res = dc.cube_to_composites(_cube(), ref, tmp_path / "comp",
                                {"july": ("2023-07-01", "2023-07-31"), "may": ("2023-05-01", "2023-05-31")},
                                scenes_dir=tmp_path / "scenes")
    assert res["windows"]["may"] == {"status": "skipped: no scenes"}
    assert res["windows"]["july"]["status"] == "written" and (tmp_path / "scenes").is_dir()


def test_composites_to_cube_errors_and_mixed_statistics(ref, tmp_path):
    root = tmp_path / "comp"
    root.mkdir()
    with pytest.raises(dc.CubeError, match="No composite folders"):
        dc.composites_to_cube(root)
    dc.cube_to_composites(_cube(), ref, root, {"a": ("2023-07-01", "2023-07-05"), "b": ("2023-07-06", "2023-07-31")},
                          statistic="median")
    meta = json.loads((root / "b" / "composite.json").read_text())
    meta["statistic"] = "mean"
    (root / "b" / "composite.json").write_text(json.dumps(meta))
    assert "geocrop:statistic" not in dc.composites_to_cube(root).attrs
    with pytest.raises(dc.CubeError, match="None of the windows"):
        dc.composites_to_cube(root, {"c": ("2023-08-01", "2023-08-31")})


def test_stack_to_cube_without_manifest(ref, tmp_path):
    stack = tmp_path / "stack.tif"
    with rasterio.open(stack, "w", driver="GTiff", width=SHAPE[1], height=SHAPE[0], count=2, dtype="float32",
                       crs=CRS, transform=from_origin(*ORIGIN, 10, 10)) as ds:
        ds.write(np.ones((2, *SHAPE), "float32"))
        ds.set_band_description(1, "NDVI_w1")
    cube = dc.stack_to_cube(stack)
    assert list(cube["feature"].values) == ["NDVI_w1", "band_2"] and "feature_group" not in cube.coords


def test_write_cube_formats(tmp_path, monkeypatch):
    ds = _cube().assign_coords(label=("time", np.array(["a", "b"], dtype=object)))
    with pytest.raises(dc.CubeError, match="choose a .zarr or .nc"):
        dc.write_cube(ds, tmp_path / "cube.tif")
    monkeypatch.setitem(sys.modules, "zarr", None)
    with pytest.raises(RuntimeError, match="zarr"):
        dc.write_cube(ds, tmp_path / "cube.zarr")

    def broken(self, *a, **k):
        raise ValueError("no netCDF engine")
    monkeypatch.setattr(xr.Dataset, "to_netcdf", broken)
    with pytest.raises(RuntimeError, match="netCDF4"):
        dc.write_cube(ds, tmp_path / "cube.nc")


def test_scenes_to_cube_without_scenes(tmp_path):
    with pytest.raises(dc.CubeError, match="No S2 scene folders"):
        dc.scenes_to_cube(tmp_path)
