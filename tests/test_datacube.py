"""External data-cube adapter: validation and conversion of synthetic xarray cubes (no network), odc-stac loading
of local STAC items against `stac_import.import_item`, export/import round trips, and a network check over
Semberija comparing `load_stac_cube` with `geocrop stac-import`."""
from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from geocrop import datacube as dc
from geocrop import s2
from geocrop import scenes as scene_meta
from geocrop.indices import S2_BANDS
from geocrop.raster import NODATA

xr = pytest.importorskip("xarray")

CRS = "EPSG:32634"
ORIGIN = (500000.0, 5000080.0)
SHAPE = (6, 8)
TIMES = ["2023-07-03T09:48:18", "2023-07-08T09:48:20", "2023-07-13T09:48:16"]


def _reference(path: Path) -> Path:
    with rasterio.open(path, "w", driver="GTiff", width=SHAPE[1], height=SHAPE[0], count=1, dtype="uint8",
                       crs=CRS, transform=from_origin(*ORIGIN, 10, 10), nodata=0) as ds:
        ds.write(np.ones((1, *SHAPE), "uint8"))
    return path


@pytest.fixture()
def ref(tmp_path):
    return _reference(tmp_path / "grid.tif")


def _values(nt=3, shape=SHAPE, seed=0):
    rng = np.random.default_rng(seed)
    return {b: rng.uniform(0.02, 0.45, (nt, *shape)).astype("float32") for b in S2_BANDS}


def _cube(values=None, scl=None, origin=ORIGIN, res=10.0, times=TIMES, crs=CRS, ascending_y=False):
    values = values or _values(len(times))
    nt, h, w = values["B02"].shape
    x = origin[0] + res * (np.arange(w) + 0.5)
    y = origin[1] - res * (np.arange(h) + 0.5)
    if scl is None:
        scl = np.full((nt, h, w), 4, "uint8")
    data = {b: (("time", "y", "x"), v) for b, v in values.items()}
    if scl is not False:
        data["SCL"] = (("time", "y", "x"), scl)
    coords = {"time": np.array(times, "datetime64[ns]"), "x": x, "y": y}
    if crs:
        coords["spatial_ref"] = ((), 0, {"crs_wkt": rasterio.crs.CRS.from_user_input(crs).to_wkt()})
    ds = xr.Dataset(data, coords=coords)
    for b in S2_BANDS:
        ds[b].attrs["units"] = "1"
    return ds.isel(y=slice(None, None, -1)) if ascending_y else ds


def _read(path):
    with rasterio.open(path) as src:
        return src.read(1)


# --------------------------------------------------------------------------- validation and conversion

def test_canonical_cube_to_scenes_round_trip(ref, tmp_path):
    values = _values()
    scl = np.full((3, *SHAPE), 4, "uint8")
    scl[1, :2, :] = 9                       # cloud
    values["B04"][2, 0, 0] = np.nan          # no data
    res = dc.cube_to_scenes(_cube(values, scl), ref, tmp_path / "scenes")
    assert [r["status"] for r in res] == ["imported"] * 3
    found = scene_meta.discover(tmp_path / "scenes")
    assert [s["date"] for s in found] == [date(2023, 7, 3), date(2023, 7, 8), date(2023, 7, 13)]
    folder = Path(res[1]["path"])
    np.testing.assert_array_equal(_read(folder / "B08.tif"), values["B08"][1])
    np.testing.assert_array_equal(_read(folder / "SCL.tif"), scl[1])
    assert _read(Path(res[2]["path"]) / "B04.tif")[0, 0] == NODATA
    meta = json.loads((folder / "scene.json").read_text())
    assert meta["datetime"].endswith("Z") and meta["grid_alignment"] == "exact"
    assert meta["clear_fraction"] == pytest.approx(1 - 2 / 6, abs=1e-4)
    # cached on a second call, re-imported with overwrite
    assert {r["status"] for r in dc.cube_to_scenes(_cube(values, scl), ref, tmp_path / "scenes")} == {"cached"}
    back = dc.scenes_to_cube(tmp_path / "scenes")
    np.testing.assert_array_equal(back["B11"].values, values["B11"])
    np.testing.assert_array_equal(back["SCL"].values, scl)


def test_dataarray_with_band_dim_dn_and_aliases(ref, tmp_path):
    vals = _values()
    dn = np.stack([np.round((vals[b] + 0.1) / 1e-4) for b in S2_BANDS], axis=-1).astype("uint16")
    dn[0, 0, 0, :] = 0                                        # DN 0 = no data
    names = [dc.ASSET_KEYS[b] for b in S2_BANDS]              # Earth Search names: blue, green, red ...
    da = xr.DataArray(dn, dims=("time", "y", "x", "band"), coords={
        "time": np.array(TIMES, "datetime64[ns]"), "band": names,
        "x": ORIGIN[0] + 10 * (np.arange(SHAPE[1]) + 0.5), "y": ORIGIN[1] - 10 * (np.arange(SHAPE[0]) + 0.5)},
        attrs={"crs": CRS})
    with pytest.raises(dc.CubeError, match="without scale"):
        dc.normalise_cube(da, ref, assume_masked=True)
    cube = dc.normalise_cube(da, ref, units="dn", scale=1e-4, offset=-0.1, assume_masked=True)
    np.testing.assert_allclose(cube["B05"].values[1], vals["B05"][1], atol=6e-5)
    assert np.isnan(cube["B02"].values[0, 0, 0])
    assert cube["SCL"].values[0, 0, 0] == 0 and cube["SCL"].values[0, 1, 1] == dc.SYNTHETIC_SCL_CLEAR
    assert json.loads(cube.attrs["geocrop:radiometry"])["B04"]["offset"] == -0.1


def test_stac_style_scale_attrs_and_band_map(ref):
    vals = _values()
    ds = _cube({b: np.round(v / 1e-4).astype("uint16") for b, v in vals.items()})
    for b in S2_BANDS:
        ds[b].attrs = {"scale": 1e-4, "offset": 0.0, "nodata": 0}
    ds = ds.rename({"B8A": "narrow_nir"})
    with pytest.raises(dc.CubeError, match="lacks Sentinel-2 bands"):
        dc.normalise_cube(ds, ref)
    cube = dc.normalise_cube(ds, ref, band_map={"narrow_nir": "B8A"})
    np.testing.assert_allclose(cube["B8A"].values, vals["B8A"], atol=6e-5)


def test_units_are_never_guessed(ref):
    vals = _values()
    dn_float = _cube({b: v / 1e-4 for b, v in vals.items()})
    for b in S2_BANDS:
        dn_float[b].attrs = {}
    with pytest.raises(dc.CubeError, match="digital numbers"):
        dc.normalise_cube(dn_float, ref)
    unlabelled = _cube(vals)
    for b in S2_BANDS:
        unlabelled[b].attrs = {}
    cube = dc.normalise_cube(unlabelled, ref)
    assert "inferred" in json.loads(cube.attrs["geocrop:radiometry"])["B02"]["kind"]
    double_offset = _cube({b: v - 0.1 for b, v in vals.items()})       # BOA offset applied twice
    with pytest.raises(dc.CubeError, match="negative red"):
        dc.normalise_cube(double_offset, ref)


def test_nodata_value_is_masked(ref):
    vals = _values()
    vals["B03"][0, 2, 2] = -9999.0
    ds = _cube(vals)
    ds["B03"].attrs["nodata"] = -9999.0
    assert np.isnan(dc.normalise_cube(ds, ref)["B03"].values[0, 2, 2])


def test_crs_required_and_override(ref):
    ds = _cube(crs=None)
    with pytest.raises(dc.CubeError, match="no CRS"):
        dc.normalise_cube(ds, ref)
    assert dc.normalise_cube(ds, ref, crs=CRS).attrs["geocrop:grid_alignment"] == "exact"
    with pytest.raises(dc.GridMismatchError):
        dc.normalise_cube(_cube(crs="EPSG:32633"), ref)


def test_grid_alignment_crop_pad_and_resample(ref):
    big_vals = _values(shape=(10, 12))
    big = _cube(big_vals, scl=np.full((3, 10, 12), 5, "uint8"), origin=(ORIGIN[0] - 20, ORIGIN[1] + 30))
    cube = dc.normalise_cube(big, ref)
    np.testing.assert_array_equal(cube["B04"].values, big_vals["B04"][:, 3:9, 2:10])
    assert cube.attrs["geocrop:grid_alignment"] == "exact"
    small = _cube(_values(shape=(4, 8)), scl=np.full((3, 4, 8), 5, "uint8"), origin=(ORIGIN[0], ORIGIN[1] - 20))
    padded = dc.normalise_cube(small, ref)
    assert np.isnan(padded["B02"].values[:, :2]).all() and (padded["SCL"].values[:, :2] == 0).all()
    assert "partial" in padded.attrs["geocrop:grid_alignment"]
    shifted = _cube(origin=(ORIGIN[0] + 5, ORIGIN[1]))
    with pytest.raises(dc.GridMismatchError, match="not on the study grid lattice"):
        dc.normalise_cube(shifted, ref)
    resampled = dc.normalise_cube(_cube(_values(shape=(8, 10)), scl=np.full((3, 8, 10), 4, "uint8"),
                                        origin=(ORIGIN[0] - 5, ORIGIN[1] + 5)), ref, resample=True)
    assert resampled["B02"].shape == (3, *SHAPE) and resampled.attrs["geocrop:grid_alignment"].startswith("resampled")
    assert np.isfinite(resampled["B02"].values).all()


def test_ascending_y_is_flipped(ref):
    vals = _values()
    cube = dc.normalise_cube(_cube(vals, ascending_y=True), ref)
    np.testing.assert_array_equal(cube["B02"].values, vals["B02"])


def test_timestamps_converted_to_utc(ref):
    ds = _cube().assign_coords(time=["2023-07-03T11:48:18+02:00", "2023-07-08T11:48:20+02:00",
                                     "2023-07-13T11:48:16+02:00"])
    cube = dc.normalise_cube(ds, ref)
    assert str(cube["time"].values[0])[:19] == "2023-07-03T09:48:18"
    local = dc.normalise_cube(_cube(), ref, assume_tz="Europe/Sarajevo")
    assert str(local["time"].values[0])[:19] == "2023-07-03T07:48:18"
    unordered = _cube(times=[TIMES[2], TIMES[0], TIMES[1]])
    assert list(dc.normalise_cube(unordered, ref)["time"].values) == sorted(unordered["time"].values)


def test_scl_contract(ref):
    with pytest.raises(dc.CubeError, match="no SCL"):
        dc.normalise_cube(_cube(scl=False), ref)
    with pytest.raises(dc.CubeError, match="outside the ESA"):
        dc.normalise_cube(_cube(scl=np.full((3, *SHAPE), 40, "uint8")), ref)


def test_cube_to_scenes_screens_like_stac_import(ref, tmp_path):
    scl = np.full((3, *SHAPE), 4, "uint8")
    scl[0] = 9                      # fully cloudy
    scl[1] = 0                      # no coverage
    res = dc.cube_to_scenes(_cube(scl=scl), ref, tmp_path / "s", min_clear_fraction=0.2)
    assert [r["status"].split(":")[0] for r in res] == ["skipped", "skipped", "imported"]


def test_cube_to_composites_matches_local_compositing(ref, tmp_path):
    scl = np.full((3, *SHAPE), 4, "uint8")
    scl[0, 3:, :] = 8
    cube = _cube(scl=scl)
    windows = {"july": (date(2023, 7, 1), date(2023, 7, 31))}
    res = dc.cube_to_composites(cube, ref, tmp_path / "comp", windows, scenes_dir=tmp_path / "staged")
    assert res["windows"]["july"]["scenes"] == 3
    s2.composite(sorted((tmp_path / "staged").iterdir()), tmp_path / "local", reference=ref)
    for name in (*[f"{b}.tif" for b in S2_BANDS], "valid_count.tif"):
        np.testing.assert_array_equal(_read(tmp_path / "comp" / "july" / name), _read(tmp_path / "local" / name))
    assert _read(tmp_path / "comp" / "july" / "valid_count.tif")[4, 0] == 2


@pytest.mark.parametrize("suffix", [".zarr", ".nc"])
def test_composites_export_import_round_trip(ref, tmp_path, suffix):
    pytest.importorskip("zarr" if suffix == ".zarr" else "netCDF4")
    windows = {"early": (date(2023, 7, 1), date(2023, 7, 9)), "late": (date(2023, 7, 10), date(2023, 7, 31))}
    dc.cube_to_composites(_cube(), ref, tmp_path / "comp", windows)
    ds = dc.composites_to_cube(tmp_path / "comp", windows)
    assert ds.attrs["Conventions"] == "CF-1.8" and ds["B04"].attrs["grid_mapping"] == "spatial_ref"
    assert ds["spatial_ref"].attrs["grid_mapping_name"] == "transverse_mercator"
    out = dc.write_cube(ds, tmp_path / f"comp{suffix}")
    with pytest.raises(FileExistsError):
        dc.write_cube(ds, out)
    back = dc.cube_to_composites(out, ref, tmp_path / "comp2")
    assert set(back["windows"]) == {"early", "late"}
    for w in windows:
        for name in (*[f"{b}.tif" for b in S2_BANDS], "valid_count.tif"):
            np.testing.assert_array_equal(_read(tmp_path / "comp2" / w / name), _read(tmp_path / "comp" / w / name))
    if suffix == ".nc":     # GDAL reads the CF grid mapping back onto the same grid
        with rasterio.open(f'NETCDF:"{out}":B04') as src, rasterio.open(ref) as r:
            assert src.crs == r.crs and src.transform.almost_equals(r.transform)


def test_scenes_cube_zarr_round_trip(ref, tmp_path):
    pytest.importorskip("zarr")
    vals = _values()
    dc.cube_to_scenes(_cube(vals), ref, tmp_path / "scenes")
    path = dc.write_cube(dc.scenes_to_cube(tmp_path / "scenes"), tmp_path / "ts.zarr")
    reopened = dc.open_cube(path)
    res = dc.cube_to_scenes(reopened, ref, tmp_path / "again")
    assert [r["status"] for r in res] == ["imported"] * 3
    np.testing.assert_array_equal(_read(Path(res[0]["path"]) / "B12.tif"), vals["B12"][0])
    assert json.loads((Path(res[0]["path"]) / "scene.json").read_text())["datetime"].startswith("2023-07-03T09:48:18")


def test_stack_to_cube(stack, tmp_path):
    ds = dc.stack_to_cube(stack)
    with rasterio.open(stack) as src:
        assert ds["features"].shape == (src.count, src.height, src.width)
        assert list(ds["feature"].values) == list(src.descriptions)
    assert "feature_group" in ds.coords and ds["features"].attrs["grid_mapping"] == "spatial_ref"


def test_cli_cube_import_and_export(ref, tmp_path, capsys):
    pytest.importorskip("netCDF4")
    from geocrop.cli import main
    _cube().to_netcdf(tmp_path / "user.nc")
    main(["cube", "import", "--cube", str(tmp_path / "user.nc"), "--reference", str(ref),
          "--out", str(tmp_path / "scenes")])
    assert len(scene_meta.discover(tmp_path / "scenes")) == 3
    capsys.readouterr()
    main(["cube", "export", "--kind", "scenes", "--src", str(tmp_path / "scenes"), "--out", str(tmp_path / "ts.nc")])
    assert json.loads(capsys.readouterr().out)["dims"]["time"] == 3
    with pytest.raises(SystemExit, match="lattice"):
        _cube(origin=(ORIGIN[0] + 3, ORIGIN[1])).to_netcdf(tmp_path / "shifted.nc")
        main(["cube", "import", "--cube", str(tmp_path / "shifted.nc"), "--reference", str(ref),
              "--out", str(tmp_path / "x")])


def test_config_data_section():
    from geocrop.config import ConfigError, load
    assert load(None)["data"]["source"] == "local"
    assert load(None, {"data": {"source": "odc-stac"}})["data"]["source"] == "odc-stac"
    with pytest.raises(ConfigError, match="data.source"):
        load(None, {"data": {"source": "gee"}})


# --------------------------------------------------------------------------- odc-stac on local STAC items

def _write_band(path, arr, res, dtype="uint16"):
    h, w = arr.shape
    with rasterio.open(path, "w", driver="GTiff", width=w, height=h, count=1, dtype=dtype, crs=CRS,
                       transform=from_origin(ORIGIN[0] - 40, ORIGIN[1] + 40, res, res), nodata=0) as ds:
        ds.write(arr.astype(dtype), 1)


def _local_item(folder: Path, when: str, boa_applied: bool, seed: int):
    pystac = pytest.importorskip("pystac")
    rng = np.random.default_rng(seed)
    folder.mkdir(parents=True)
    item = pystac.Item(id=f"S2X_TEST_{seed}", geometry={"type": "Polygon", "coordinates": [[
        [18.0, 44.0], [21.0, 44.0], [21.0, 46.0], [18.0, 46.0], [18.0, 44.0]]]},
        bbox=[18.0, 44.0, 21.0, 46.0], datetime=datetime.fromisoformat(when).replace(tzinfo=timezone.utc),
        properties={"earthsearch:boa_offset_applied": boa_applied, "s2:processing_baseline": "05.09",
                    "platform": "sentinel-2a", "s2:mgrs_tile": "34TCQ"})
    for band in (*S2_BANDS, "SCL"):
        res = 10 if band in ("B02", "B03", "B04", "B08") else 20
        n = (SHAPE[0] * 10 + 80) // res, (SHAPE[1] * 10 + 80) // res
        arr = rng.integers(1, 12, n) if band == "SCL" else rng.integers(1200, 4000, n)
        if band == "SCL":
            arr[0, :] = 9
        path = folder / f"{band}.tif"
        _write_band(path, arr, res, "uint8" if band == "SCL" else "uint16")
        key = dc.ASSET_KEYS[band]
        asset = pystac.Asset(href=str(path), media_type="image/tiff; application=geotiff", roles=["data"],
                             extra_fields={"proj:epsg": 32634, "proj:shape": list(n),
                                           "proj:transform": list(from_origin(ORIGIN[0] - 40, ORIGIN[1] + 40,
                                                                              res, res))[:6]})
        if band != "SCL":
            asset.extra_fields["raster:bands"] = [{"scale": 1e-4, "offset": -0.1, "nodata": 0,
                                                   "data_type": "uint16"}]
        item.add_asset(key, asset)
    return item


def test_load_stac_cube_matches_stac_import_on_local_items(ref, tmp_path):
    pytest.importorskip("odc.stac")
    from geocrop.stac_import import import_item
    items = [_local_item(tmp_path / "a", "2023-07-03T09:48:18", True, 1),
             _local_item(tmp_path / "b", "2023-07-08T09:48:20", False, 2)]
    cube = dc.load_stac_cube(ref, items=items, min_clear_fraction=0.0, max_negative_fraction=1.0)
    assert cube.sizes["time"] == 2
    res = dc.cube_to_scenes(cube, ref, tmp_path / "cube", max_negative_fraction=1.0)
    for item, r in zip(items, res):
        import_item(item, ref, tmp_path / "stac" / item.id, max_negative_fraction=1.0)
        for name in (*[f"{b}.tif" for b in S2_BANDS], "SCL.tif"):
            a, b = _read(Path(r["path"]) / name), _read(tmp_path / "stac" / item.id / name)
            np.testing.assert_allclose(a, b, atol=1e-6, err_msg=f"{item.id} {name}")
    # BOA offset: applied once for boa_offset_applied=False, not again when the provider already removed it
    assert np.nanmin(cube["B04"].values[0]) >= 0.12 - 1e-6
    assert np.nanmin(cube["B04"].values[1]) >= 0.02 - 1e-6


def test_missing_odc_stac_names_the_extra(monkeypatch, ref):
    import builtins
    real = builtins.__import__

    def fake(name, *args, **kwargs):
        if name.startswith("odc"):
            raise ImportError(name)
        return real(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", fake)
    with pytest.raises(RuntimeError, match="odc-stac"):
        dc.load_stac_cube(ref, "2023-07-01", "2023-07-31")


# --------------------------------------------------------------------------- real data

@pytest.mark.network
def test_semberija_odc_stac_equals_stac_import(tmp_path):
    """Semberija (19.10-19.16E, 44.72-44.76N), July 2023: odc-stac cube vs `geocrop stac-import`."""
    pytest.importorskip("odc.stac")
    pytest.importorskip("pystac_client")
    import geopandas as gpd
    from shapely.geometry import box

    from geocrop.raster import make_grid
    from geocrop.stac_import import import_season
    aoi = tmp_path / "aoi.gpkg"
    gpd.GeoDataFrame(geometry=[box(19.10, 44.72, 19.16, 44.76)], crs="EPSG:4326").to_file(aoi)
    grid = make_grid(aoi, "EPSG:32634", 10, tmp_path / "grid.tif")
    imported = [r for r in import_season(grid, tmp_path / "stac", "2023-07-01", "2023-07-31")
                if r["status"] == "imported"]
    cube = dc.load_stac_cube(grid, "2023-07-01", "2023-07-31")
    res = {r["scene"]: r for r in dc.cube_to_scenes(cube, grid, tmp_path / "cube")}
    assert imported and {r["item"] for r in imported} == {k for k, v in res.items() if v["status"] == "imported"}
    worst = 0.0
    for r in imported:
        for name in (*[f"{b}.tif" for b in S2_BANDS], "SCL.tif"):
            a = _read(tmp_path / "cube" / r["item"] / name).astype("float64")
            b = _read(Path(r["path"]) / name).astype("float64")
            worst = max(worst, float(np.abs(a - b).max()))
    assert worst <= 1e-4, worst
