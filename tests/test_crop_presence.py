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
    assert t.loc["P2", "p1_match_whole"] == False and t.loc["P2", "p1_match_agricultural"] == True  # noqa: E712
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


# --- audit of P3 and the presence criterion (synthetic parcels of 20 x 15 pixels = 3 ha at 10 m) -----------------------

def _scene(tmp_path, maps, declared="soybean", d_area=None, name="scene"):
    """Stack parcels (15 rows x 20 cols, class index per pixel, -1 = no data) in one raster; 0.94 for the painted class."""
    n = len(maps)
    z = np.full((4, 20 * n, 30), 0.02, dtype="float32")
    z[0] = 0.94                                                  # forest outside the painted parcels
    geoms = []
    x0, y0 = 500000, 5000000 + 20 * n * 10
    for k, (_pid, m) in enumerate(maps.items()):
        r0 = 20 * k
        patch = np.stack([np.where(m == i, 0.94, 0.02) for i in range(4)]).astype("float32")
        patch[:, m == -1] = np.nan
        z[:, r0:r0 + 15, 0:20] = patch
        geoms.append(box(x0, y0 - (r0 + 15) * 10, x0 + 200, y0 - r0 * 10))
    path = tmp_path / f"{name}.tif"
    with rasterio.open(path, "w", driver="GTiff", height=20 * n, width=30, count=4, dtype="float32",
                       crs="EPSG:32633", transform=from_origin(x0, y0, 10, 10), nodata=None) as dst:
        dst.write(z)
        dst.descriptions = CLASSES
    gdf = gpd.GeoDataFrame({"parcel_id": list(maps), "declared_crop": declared, "declared_area_ha": d_area, "geometry": geoms},
                           crs="EPSG:32633")
    return path, gdf


def _blocks(shares_cols):
    """15 x 20 map from column blocks: [(class, n_cols), ...]."""
    return np.concatenate([np.full((15, w), c) for c, w in shares_cols], axis=1)


def _speckle(base, pixels):
    m = np.full((15, 20), base)
    for r, c in pixels:
        m[r, c] = 3
    return m


MAPS = {
    "HETERO": _blocks([(3, 7), (2, 9), (1, 4)]),            # 35 % soybean (connected), 45 % maize, 20 % grass
    "MAJORITY": _blocks([(3, 16), (1, 4)]),                  # 80 % soybean
    "SPECKLE": _speckle(2, [(1, 1), (5, 5), (9, 9), (13, 14)]),   # maize with four isolated soybean pixels (0.04 ha)
    "BLOCK4": _speckle(2, [(1, 1), (1, 2), (2, 1), (2, 2)]),      # maize with one 2 x 2 soybean block (0.04 ha)
    "OTHERCROP": np.full((15, 20), 2),
    "GRASS": np.full((15, 20), 1),
    "NODATA": _blocks([(-1, 16), (3, 4)]),                   # 80 % without observations
}


def test_p3_is_stricter_than_p1_on_heterogeneous_parcels(tmp_path):
    """Audit: a connected 1.05 ha soybean block (35 % of the parcel, 44 % of its agricultural part) is never P3 supported."""
    rpath, gdf = _scene(tmp_path, MAPS)
    t = gpd.read_file(crop_presence.analyse(rpath, gdf, tmp_path / "o")["gpkg"], layer="parcels").set_index("parcel_id")
    h = t.loc["HETERO"]
    assert h["crop_area_ha"] == pytest.approx(1.05) and h["connected_support_ha"] == pytest.approx(1.05)
    assert h["supported_share_agricultural"] == pytest.approx(105 / 240)
    assert h["preliminary_status"] == "disputed_review" and h["review_reason"] == "low_supported_share"
    assert h["p1_match_agricultural"] == False  # noqa: E712
    assert h["presence_status"] == "present"                                  # the new criterion sees the connected block
    assert t.loc["MAJORITY", "preliminary_status"] == "supported" and t.loc["MAJORITY", "presence_status"] == "present"
    # the same parcel with a lower share threshold is supported by P3, so the 0.5 default (= majority) is the cause
    lo = gpd.read_file(crop_presence.analyse(rpath, gdf, tmp_path / "o2", min_supported_share=0.4)["gpkg"],
                       layer="parcels").set_index("parcel_id")
    assert lo.loc["HETERO", "preliminary_status"] == "supported"


def test_p3_speckle_is_never_not_supported_and_presence_needs_connected_area(tmp_path):
    rpath, gdf = _scene(tmp_path, MAPS)
    t = gpd.read_file(crop_presence.analyse(rpath, gdf, tmp_path / "o")["gpkg"], layer="parcels").set_index("parcel_id")
    for pid in ("SPECKLE", "BLOCK4"):
        assert t.loc[pid, "crop_area_ha"] == pytest.approx(0.04)
        assert t.loc[pid, "preliminary_status"] == "disputed_review"          # not_supported needs zero declared pixels
        assert t.loc[pid, "presence_status"] == "absent"                       # 0.04 ha < 0.05 ha
    assert t.loc["SPECKLE", "connected_support_ha"] == pytest.approx(0.01)   # isolated pixels, no 8-neighbours
    assert t.loc["BLOCK4", "connected_support_ha"] == pytest.approx(0.04)
    lo = gpd.read_file(crop_presence.analyse(rpath, gdf, tmp_path / "o2", min_connected_ha=0.03)["gpkg"],
                       layer="parcels").set_index("parcel_id")
    assert lo.loc["BLOCK4", "presence_status"] == "present"                    # a connected 0.04 ha passes a 0.03 ha minimum
    assert lo.loc["SPECKLE", "presence_status"] == "partial_presence" and lo.loc["SPECKLE", "presence_reason"] == "fragmented_support"


def test_p3_supported_is_subset_of_p1_agricultural_up_to_exact_ties():
    """Property on random synthetic parcels: P3 'supported' implies a P1 agricultural match unless the declared crop is
    tied at exactly 50 % of the agricultural part with another class (argmax then picks the first class)."""
    rng = np.random.default_rng(3)
    supported = 0
    for t in range(600):
        h, w = int(rng.integers(4, 25)), int(rng.integers(4, 25))
        cls = np.kron(rng.integers(0, 4, size=((h + 4) // 5, (w + 4) // 5)), np.ones((5, 5), int))[:h, :w]
        z = np.stack([np.where(cls == i, 0.9, 0.1 / 3) for i in range(4)]).astype(float)
        z += rng.dirichlet(np.ones(4) * (0.5, 2.0)[t % 2], size=cls.shape).transpose(2, 0, 1) * 0.3
        z /= z.sum(axis=0)
        row, _ = crop_presence.analyse_parcel(z, None, np.ones((h, w), bool), CLASSES, "soybean", 0.01, ("maize", "soybean"),
                                              float(rng.choice((0.4, 0.5, 0.6))), 0.03)
        status, _ = crop_presence.preliminary_status(row, None, 4, 0.03, 0.05, 0.5, 0.1, 0.6, 0.5)
        if status == "supported":
            supported += 1
            if not row["p1_match_agricultural"]:
                assert row["crop_share_agricultural"] == 0.5
    assert supported > 50


def test_presence_decision_reliability_and_declared_share():
    base = {"n_valid_pixels": 100, "declared_class_modelled": True, "connected_support_ha": 0.3, "connected_mean_p": 0.8,
            "connected_mean_confidence": 0.8, "crop_area_ha": 0.3, "expected_pixels": 100.0, "valid_pixel_fraction": 1.0}
    assert crop_presence.presence_decision(base, None)[0] == "present"
    assert crop_presence.presence_decision(base, 0.4)[0] == "present"                      # 0.3 / 0.4 >= 0.5
    assert crop_presence.presence_decision(base, 1.0) == ("partial_presence", ["connected_area_below_declared_share"])
    weak = dict(base, connected_mean_p=0.52)
    assert crop_presence.presence_decision(weak, None) == ("partial_presence", ["connected_support_unreliable"])
    assert crop_presence.presence_decision(dict(base, valid_pixel_fraction=0.3), None)[0] == "indeterminate"
    assert crop_presence.presence_decision(dict(base, expected_pixels=5.0), None)[0] == "indeterminate"
    assert crop_presence.presence_decision(dict(base, declared_class_modelled=False), None)[0] == "indeterminate"
    none = dict(base, connected_support_ha=0.0, connected_mean_p=None, connected_mean_confidence=None, crop_area_ha=0.0)
    assert crop_presence.presence_decision(none, None)[0] == "absent"


def test_mismatch_taxonomy_primary_reason_and_flags(tmp_path):
    maps = dict(MAPS)
    rpath, gdf = _scene(tmp_path, maps, d_area=[1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    extra = gpd.GeoDataFrame({"parcel_id": ["OUTSIDE", "THIN", "AREA"], "declared_crop": "soybean",
                              "declared_area_ha": [1.0, 0.1, 1.0],
                              "geometry": [box(900000, 5000000, 900100, 5000100),               # outside the raster
                                           box(500206, 5000000 + 140 * 10 - 600, 500210, 5000000 + 140 * 10),   # 4 m x 600 m sliver
                                           gdf.geometry.iloc[1]]},                                # MAJORITY mapped 2.4 ha, declared 1 ha
                             crs="EPSG:32633")
    allp = gpd.GeoDataFrame(pd.concat([gdf, extra], ignore_index=True), crs="EPSG:32633")
    res = crop_presence.analyse(rpath, allp, tmp_path / "o", declared_area_field="declared_area_ha")
    t = gpd.read_file(res["gpkg"], layer="parcels").set_index("parcel_id")
    assert t.loc["MAJORITY", "mismatch_category"] == "none"
    assert t.loc["HETERO", "presence_status"] == "present" and t.loc["HETERO", "mismatch_category"] == "none"
    assert t.loc["OTHERCROP", "mismatch_category"] == "other_crop_present" and t.loc["OTHERCROP", "flag_other_crop_present"]
    assert t.loc["GRASS", "mismatch_category"] == "non_agricultural_dominant"
    assert t.loc["GRASS", "dominant_nonagricultural_class"] == "grass" and t.loc["GRASS", "agricultural_share"] == 0
    assert pd.isna(t.loc["MAJORITY", "dominant_nonagricultural_class"])
    assert t.loc["NODATA", "mismatch_category"] == "no_valid_observations" and t.loc["NODATA", "presence_status"] == "indeterminate"
    assert t.loc["NODATA", "valid_pixel_fraction"] == pytest.approx(0.2) and t.loc["NODATA", "expected_pixels"] == pytest.approx(300)
    assert t.loc["OUTSIDE", "mismatch_category"] == "no_valid_observations" and t.loc["OUTSIDE", "n_valid_pixels"] == 0
    assert t.loc["THIN", "mismatch_category"] == "too_few_pixels" and t.loc["THIN", "n_parcel_pixels"] == 0
    assert t.loc["AREA", "presence_status"] == "present" and t.loc["AREA", "mismatch_category"] == "area_disagreement"
    # MAJORITY: declared 2 ha, mapped 2.4 ha in one connected area: present and within tolerance (20 % off)
    assert not t.loc["AREA", "flag_declared_crop_partial"] and not t.loc["MAJORITY", "flag_area_disagreement"]
    assert t.loc["HETERO", "presence_status"] == "present"
    # declared area 3 ha: the connected 1.05 ha covers 35 % < 50 %, partial
    t3 = gpd.read_file(crop_presence.analyse(rpath, gdf.assign(declared_area_ha=3.0), tmp_path / "o3",
                                             declared_area_field="declared_area_ha")["gpkg"], layer="parcels").set_index("parcel_id")
    assert t3.loc["HETERO", "presence_status"] == "partial_presence"
    assert t3.loc["HETERO", "mismatch_category"] == "declared_crop_partial" and t3.loc["HETERO", "flag_declared_crop_partial"]
    man = json.loads((tmp_path / "o" / "crop_presence_manifest.json").read_text())
    assert man["mismatch_category_counts"]["none"] >= 2 and man["config"]["presence_min_share"] == 0.5


def test_compare_probabilities_flags_model_disagreement(tmp_path):
    rpath, gdf = _scene(tmp_path, {"A": MAPS["HETERO"], "B": MAPS["MAJORITY"]}, d_area=1.0)
    r2, _ = _scene(tmp_path, {"A": MAPS["OTHERCROP"], "B": MAPS["MAJORITY"]}, d_area=1.0, name="second")
    res = crop_presence.analyse(rpath, gdf, tmp_path / "o", declared_area_field="declared_area_ha", compare_probabilities=r2)
    t = gpd.read_file(res["gpkg"], layer="parcels").set_index("parcel_id")
    assert t.loc["A", "flag_model_disagreement"] and not t.loc["B", "flag_model_disagreement"]
    assert t.loc["A", "cmp_presence_status"] == "absent" and t.loc["B", "cmp_presence_status"] == "present"
    assert t.loc["A", "mismatch_category"] == "model_disagreement"              # primary only when nothing else applies
    assert json.loads((tmp_path / "o" / "crop_presence_manifest.json").read_text())["inputs"]["compare_probabilities_sha256"]


def test_sensitivity_sweep_and_crosstab(tmp_path):
    rpath, gdf = _scene(tmp_path, MAPS)
    out = crop_presence.sensitivity(rpath, gdf, tmp_path / "sens.csv")
    s = pd.read_csv(out)
    assert len(s) == 12 and set(s["min_connected_ha"]) == {0.03, 0.05, 0.1, 0.2} and set(s["support_prob"]) == {0.4, 0.5, 0.6}
    d = s[(s["min_connected_ha"] == 0.05) & (s["support_prob"] == 0.5)].iloc[0]
    assert d["present_not_p1_agri"] >= 1 and d["p3_supported_not_p1_agri"] == 0      # HETERO is present, not a P1 match
    assert (s["p3_supported_not_p1_agri"] == 0).all()
    assert (s[[f"presence_{x}" for x in crop_presence.PRESENCE_STATUSES]].sum(axis=1) == s["parcels"]).all()
    lo = s[(s["min_connected_ha"] == 0.03) & (s["support_prob"] == 0.5)].iloc[0]
    assert lo["presence_present"] > d["presence_present"]                              # BLOCK4 appears at 0.03 ha


def test_review_sample_is_stratified_and_exports_layer(tmp_path):
    rng = np.random.default_rng(0)
    cats = ["none"] * 30 + ["other_crop_present"] * 12 + ["area_disagreement"] * 2
    tab = gpd.GeoDataFrame({"parcel_id": [f"S{i}" for i in range(len(cats))], "mismatch_category": cats,
                            "parcel_area_ha": rng.uniform(0.1, 10, len(cats)),
                            "geometry": [box(i, 0, i + 1, 1) for i in range(len(cats))]}, crs="EPSG:32633")
    res = crop_presence.review_sample(tab, 5, tmp_path / "sample", seed=1)
    assert res["per_category"] == {"none": 5, "other_crop_present": 5, "area_disagreement": 2}
    csv = pd.read_csv(res["csv"])
    assert csv["parcel_id"].is_unique and len(csv) == 12
    assert len(gpd.read_file(res["gpkg"], layer="sample")) == 12
    again = crop_presence.review_sample(tab, 5, tmp_path / "again", seed=1)
    assert pd.read_csv(again["csv"])["parcel_id"].tolist() == csv["parcel_id"].tolist()    # reproducible


def test_cli_presence_options(tmp_path):
    from geocrop import cli
    rpath, gdf = _scene(tmp_path, MAPS, d_area=1.0)
    r2, _ = _scene(tmp_path, MAPS, d_area=1.0, name="second")
    gdf.to_file(tmp_path / "parcels.gpkg", driver="GPKG")
    cli.main(["crop-presence", "--probabilities", str(rpath), "--parcels", str(tmp_path / "parcels.gpkg"),
              "--declared-area-field", "declared_area_ha", "--compare-probabilities", str(r2), "--review-sample", "2",
              "--sensitivity", "--presence-min-share", "0.3", "--min-connected-ha", "0.03", "--out", str(tmp_path / "cli_out")])
    out = tmp_path / "cli_out"
    assert (out / "review_sample.csv").is_file() and (out / "presence_sensitivity.csv").is_file()
    t = pd.read_csv(out / "crop_presence.csv")
    assert {"presence_status", "mismatch_category", "cmp_presence_status", "flag_model_disagreement"} <= set(t.columns)
