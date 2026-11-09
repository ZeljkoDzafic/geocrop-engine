"""Crop presence in heterogeneous parcels on a synthetic probability raster (no real data)."""
import json

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

from geocrop import crop_presence

CLASSES = ["forest", "grass", "maize", "soybean"]


def _raster(tmp_path):
    """60 x 60 pixels of 10 m; per-block class maps are painted below, probability 0.9 for the painted class."""
    cls = np.zeros((60, 60), dtype=int)               # forest everywhere
    cls[0:20, 0:20] = 3                               # P1: soybean block
    cls[0:20, 30:34] = 1                              # P2: 50 % forest, 20 % grass, 30 % soybean
    cls[0:20, 34:40] = 3
    cls[20:40, 0:20] = 2                              # P3: maize block, declared soybean
    cls[40:60, 0:20] = 0                              # P4: forest with isolated soybean pixels
    for r, c in ((42, 2), (47, 9), (53, 15), (58, 5)):
        cls[r, c] = 3
    z = np.full((4, 60, 60), 0.02, dtype="float32")
    for i in range(4):
        z[i][cls == i] = 0.94
    path = tmp_path / "probabilities.tif"
    with rasterio.open(path, "w", driver="GTiff", height=60, width=60, count=4, dtype="float32", crs="EPSG:32633",
                       transform=from_origin(500000, 5000600, 10, 10)) as dst:
        dst.write(z)
        dst.descriptions = CLASSES
    return path


def _parcels(tmp_path):
    x0, y0 = 500000, 5000600
    def b(c0, r0, c1, r1):
        return box(x0 + c0 * 10, y0 - r1 * 10, x0 + c1 * 10, y0 - r0 * 10)
    gdf = gpd.GeoDataFrame({"parcel_id": ["P1", "P2", "P3", "P4"], "declared_crop": ["soybean"] * 4,
                            "declared_area_ha": [4.0, 1.2, 4.0, 4.0],
                            "geometry": [b(0, 0, 20, 20), b(20, 0, 40, 20), b(0, 20, 20, 40), b(0, 40, 20, 60)]},
                           crs="EPSG:32633")
    return gdf


def test_methods_and_status(tmp_path):
    rpath, gdf = _raster(tmp_path), _parcels(tmp_path)
    res = crop_presence.analyse(rpath, gdf, tmp_path / "out", declared_area_field="declared_area_ha")
    t = gpd.read_file(res["gpkg"], layer="parcels").set_index("parcel_id")
    assert t.loc["P1", "preliminary_status"] == "supported"
    # P2: soybean is 30 % of the parcel but 100 % of its agricultural part: the whole-parcel majority fails, the agricultural part passes
    assert t.loc["P2", "a_match_whole"] == False and t.loc["P2", "a_match_agricultural"] == True  # noqa: E712
    assert t.loc["P2", "agricultural_area_ha"] == pytest.approx(1.2)
    assert t.loc["P2", "soft_vote_agricultural"] > 0.9 > t.loc["P2", "soft_vote_whole"]
    assert t.loc["P2", "preliminary_status"] == "supported"
    assert t.loc["P3", "preliminary_status"] == "not_supported"
    assert t.loc["P4", "preliminary_status"] == "disputed_review"           # isolated pixels: no connected support
    assert t.loc["P4", "connected_support_ha"] < 0.05 and t.loc["P4", "crop_area_ha"] == pytest.approx(0.04)
    # area comparison is reported, not merged into the status
    assert t.loc["P1", "area_difference_ha"] == pytest.approx(0.0, abs=1e-9)
    seg = gpd.read_file(res["gpkg"], layer="segments")
    assert len(seg) > 4 and {"parcel_id", "mean_p_declared", "supported"} <= set(seg.columns)
    man = json.loads((tmp_path / "out" / "crop_presence_manifest.json").read_text())
    assert man["thresholds_validated"] is False and "administrative" in man["note"]


def test_soft_vote_formula():
    z = np.zeros((2, 2, 2), dtype="float32")
    z[0] = [[0.8, 0.6], [0.2, 0.4]]
    z[1] = 1 - z[0]
    inside = np.ones((2, 2), dtype=bool)
    row, _ = crop_presence.analyse_parcel(z, None, inside, ["a", "b"], "a", 0.01, ("a", "b"), 0.5, 0.0)
    assert row["soft_vote_whole"] == pytest.approx((0.8 + 0.6 + 0.2 + 0.4) / 4)


def test_requires_declared_field_and_unique_ids(tmp_path):
    rpath, gdf = _raster(tmp_path), _parcels(tmp_path)
    with pytest.raises(ValueError):
        crop_presence.analyse(rpath, gdf.drop(columns="declared_crop"), tmp_path / "a")
    dup = gdf.copy()
    dup["parcel_id"] = "P1"
    with pytest.raises(ValueError):
        crop_presence.analyse(rpath, dup, tmp_path / "b")


def test_edge_cases(tmp_path):
    rpath, gdf = _raster(tmp_path), _parcels(tmp_path)
    extra = gpd.GeoDataFrame({"parcel_id": ["OUT", "NAN", "GRASS", "UNMODELLED"],
                              "declared_crop": ["soybean", "soybean", "grass", "wheat"],
                              "declared_area_ha": [1.0, np.nan, 1.0, 1.0],
                              "geometry": [box(900000, 5000000, 900100, 5000100),        # outside the raster
                                           box(500000, 5000400, 500200, 5000600),         # inside, NaN declared area
                                           box(500000, 5000000, 500200, 5000200),         # declared meadow
                                           box(500000, 5000400, 500200, 5000600)]},       # declared crop the model lacks
                             crs="EPSG:32633")
    allp = gpd.GeoDataFrame(pd.concat([gdf, extra], ignore_index=True), crs="EPSG:32633")
    res = crop_presence.analyse(rpath, allp, tmp_path / "out", declared_area_field="declared_area_ha")
    t = gpd.read_file(res["gpkg"], layer="parcels").set_index("parcel_id")
    assert t.loc["OUT", "n_valid_pixels"] == 0 and t.loc["OUT", "review_reason"] == "too_few_valid_pixels"
    assert np.isnan(t.loc["NAN", "declared_area_ha"]) and np.isnan(t.loc["NAN", "area_difference_ha"])
    assert t.loc["UNMODELLED", "review_reason"] == "declared_crop_not_modelled"
    assert t.loc["GRASS", "agricultural_area_ha"] >= t.loc["GRASS", "area_ha_grass"]       # the declared class joins the agricultural part


def test_inputs_are_checked(tmp_path):
    rpath, gdf = _raster(tmp_path), _parcels(tmp_path)
    with pytest.raises(ValueError, match="declared_area_field"):
        crop_presence.analyse(rpath, gdf, tmp_path / "a", declared_area_field="nope")
    with pytest.raises(ValueError, match="interior_buffer_px"):
        crop_presence.analyse(rpath, gdf, tmp_path / "b", interior_buffer_px=-1)
    seg = gpd.GeoDataFrame({"sid": [1], "geometry": [box(0, 0, 1, 1)]}, crs="EPSG:32633")
    seg.to_file(tmp_path / "seg.gpkg", driver="GPKG")
    with pytest.raises(ValueError, match="segment_id_field"):
        crop_presence.analyse(rpath, gdf, tmp_path / "c", segments=tmp_path / "seg.gpkg", segment_id_field="missing")
    bad = tmp_path / "bad.tif"
    with rasterio.open(rpath) as src, rasterio.open(bad, "w", **src.profile) as dst:
        data = src.read()
        data[0, 5, 5] = 3.0
        dst.write(data)
        dst.descriptions = CLASSES
    with pytest.raises(ValueError, match="sum to one"):
        crop_presence.analyse(bad, gdf, tmp_path / "d")


def test_segments_without_overlap_fall_back_and_given_segments_are_used(tmp_path):
    rpath, gdf = _raster(tmp_path), _parcels(tmp_path)
    far = gpd.GeoDataFrame({"sid": [7], "geometry": [box(900000, 5000000, 900100, 5000100)]}, crs="EPSG:32633")
    far.to_file(tmp_path / "far.gpkg", driver="GPKG")
    res = crop_presence.analyse(rpath, gdf, tmp_path / "o1", segments=tmp_path / "far.gpkg", segment_id_field="sid")
    t = gpd.read_file(res["gpkg"], layer="parcels").set_index("parcel_id")
    assert t.loc["P1", "preliminary_status"] == "supported"                                 # same as without segments
    one = gpd.GeoDataFrame({"sid": [0], "geometry": [gdf.geometry.iloc[0]]}, crs="EPSG:32633")   # one OBIA segment, id 0
    one.to_file(tmp_path / "one.gpkg", driver="GPKG")
    res = crop_presence.analyse(rpath, gdf.iloc[:1], tmp_path / "o2", segments=tmp_path / "one.gpkg", segment_id_field="sid")
    t = gpd.read_file(res["gpkg"], layer="parcels")
    assert int(t.loc[0, "segments"]) == 1


def test_cli_smoke(tmp_path):
    from geocrop import cli
    rpath, gdf = _raster(tmp_path), _parcels(tmp_path)
    gdf.to_file(tmp_path / "parcels.gpkg", driver="GPKG")
    cli.main(["crop-presence", "--probabilities", str(rpath), "--parcels", str(tmp_path / "parcels.gpkg"),
              "--declared-area-field", "declared_area_ha", "--out", str(tmp_path / "cli_out")])
    assert (tmp_path / "cli_out" / "crop_presence.csv").is_file()
    assert json.loads((tmp_path / "cli_out" / "crop_presence_manifest.json").read_text())["parcels"] == 4
