"""Edge and error paths of the CbM modules: cbm, cbm_markers and temporal_quality (synthetic data)."""
from __future__ import annotations

from datetime import date, timedelta

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

from geocrop import cbm, cbm_markers, scenes, temporal_quality

START, END = date(2023, 4, 1), date(2023, 9, 30)
T = from_origin(500000, 4900000, 10, 10)


def _write(path, array, dtype="float32", nodata=None):
    with rasterio.open(path, "w", driver="GTiff", width=array.shape[1], height=array.shape[0], count=1,
                       dtype=dtype, crs="EPSG:32634", transform=T, nodata=nodata) as ds:
        ds.write(array.astype(dtype), 1)


@pytest.fixture(scope="module")
def blocky(tmp_path_factory):
    """20x20 grid read in 5x5 blocks: a corner parcel touching (but not covering) neighbouring blocks,
    and a parcel spanning two blocks; one S2 and two S1 scenes (one of them without valid backscatter)."""
    root = tmp_path_factory.mktemp("blocky")
    _write(root / "ref.tif", np.ones((20, 20)), dtype="uint8")
    gdf = gpd.GeoDataFrame({"parcel_id": ["corner", "wide"]},
                           geometry=[box(500000, 4899950, 500050, 4900000),      # cols 0-4, rows 0-4
                                     box(500100, 4899850, 500200, 4899900)],     # cols 10-19, rows 10-14
                           crs="EPSG:32634")
    s2 = root / "scenes" / "s2" / "a"
    s2.mkdir(parents=True)
    _write(s2 / "B04.tif", np.full((20, 20), 1000.0))
    _write(s2 / "B08.tif", np.full((20, 20), 3000.0))
    _write(s2 / "SCL.tif", np.full((20, 20), 4), dtype="uint8")
    scenes.write(s2, sensor="S2", datetime="2023-06-01T10:00:00")
    for name, level in (("good", 0.1), ("zero", 0.0)):
        folder = root / "scenes" / "s1" / name
        folder.mkdir(parents=True)
        _write(folder / "VV.tif", np.full((20, 20), level))
        _write(folder / "VH.tif", np.full((20, 20), level / 10))
        scenes.write(folder, sensor="S1", datetime=f"2023-06-0{2 if name == 'good' else 3}T16:00:00",
                     orbit_direction="ascending", relative_orbit=51)
    return {"root": root / "scenes", "ref": root / "ref.tif", "parcels": gdf}


def test_signatures_without_scenes_raise(blocky, tmp_path):
    empty = tmp_path / "none"
    empty.mkdir()
    with pytest.raises(ValueError, match="No Sentinel-2"):
        cbm.parcel_signatures(empty, blocky["parcels"], blocky["ref"], tmp_path / "s.csv", START, END)
    with pytest.raises(ValueError, match="No Sentinel-1"):
        cbm.s1_signatures(empty, blocky["parcels"], blocky["ref"], tmp_path / "s.csv", START, END)


def test_parcel_signatures_block_independent(blocky, tmp_path):
    small = cbm.parcel_signatures(blocky["root"], blocky["parcels"], blocky["ref"], tmp_path / "a.csv", START, END,
                                  ("NDVI",), interior_buffer_px=0, block_size=5)
    whole = cbm.parcel_signatures(blocky["root"], blocky["parcels"], blocky["ref"], tmp_path / "b.csv", START, END,
                                  ("NDVI",), interior_buffer_px=0, block_size=64)
    cols = ["parcel_id", "mean", "std", "n_valid", "n_pixels"]
    pd.testing.assert_frame_equal(small[cols], whole[cols])
    assert small.set_index("parcel_id")["n_pixels"].to_dict() == {"corner": 25, "wide": 50}
    assert np.allclose(small["mean"], 0.5)                       # (3000 - 1000) / (3000 + 1000)
    assert (tmp_path / "a.csv").exists()


def test_s1_signatures_skip_invalid_scene_and_blocks(blocky, tmp_path):
    t = cbm.s1_signatures(blocky["root"], blocky["parcels"], blocky["ref"], tmp_path / "s1.csv", START, END,
                          interior_buffer_px=0, block_size=5)
    good = t[t.date == "2023-06-02"]
    zero = t[t.date == "2023-06-03"]
    assert (good.n_valid == good.n_pixels).all() and good.n_valid.gt(0).all()
    assert (zero.n_valid == 0).all() and zero["mean"].isna().all()   # zero backscatter is not valid
    assert set(t.orbit_direction) == {"ASCENDING"}
    vv = good[good["index"] == "VV_dB"]["mean"]
    assert np.allclose(vv, -10, atol=1e-4)


def test_calendar_consistency_harvest_only_spec():
    m = pd.DataFrame({"peak_doy": [180.0, 180.0, np.nan], "harvest_doy": [200.0, 260.0, 200.0]})
    cal = {"maize": {"harvest": [190, 220]}}
    out = cbm.calendar_consistency(m, pd.Series(["maize", "maize", "maize"]), cal)
    assert out.tolist() == [True, False, None]
    assert cbm.calendar_consistency(m, pd.Series(["x", "x", "x"]), cal).tolist() == [None, None, None]


# --------------------------------------------------------------------------- cbm_markers validation

@pytest.mark.parametrize("params", [{"step_days": 0}, {"step_days": True}, {"min_coverage": 1.5},
                                    {"s1_min_z": -1.0}, {"bare_soil_ndvi": "low"}, {"harvest_s1_sign": "up"},
                                    {"index": ""}, {"lambda": 0}, {"s1_group_by": "tile"}, {"nope": 1}])
def test_resolve_event_params_rejects(params):
    with pytest.raises(ValueError):
        cbm_markers.resolve_event_params(params)


@pytest.mark.parametrize("rules", [{"bogus": 1}, {"enabled": "yes"}, {"min_marker_confidence": 2},
                                   {"harvest_expected_classes": "maize"}, {"tillage_forbidden_classes": [1.5]},
                                   {"min_mowing_events": 0}, {"min_mowing_events": True},
                                   {"tillage_outcome": "blue"}, {"mixed_class_share": 0.9}])
def test_resolve_rules_rejects(rules):
    with pytest.raises(ValueError):
        cbm_markers.resolve_rules(rules)


def test_resolve_rules_normalises_class_names():
    r = cbm_markers.resolve_rules({"harvest_expected_classes": [1101, "maize"]})
    assert r["harvest_expected_classes"] == ["1101", "maize"]


def test_s1_steps_too_short_series_is_empty():
    out = cbm_markers.s1_steps([150, 156, 162], [-15, -15, -18], window_days=24, min_obs=2)
    assert out.empty and list(out.columns) == ["doy", "step", "z", "n_before", "n_after"]


def _s1_series(pid, drop, n=14, group="ASCENDING"):
    rows = []
    for k in range(n):
        d = date(2023, 5, 1) + timedelta(days=6 * k)
        level = -15.0 - (drop if k >= n // 2 else 0.0)
        for feature, value in (("VH_dB", level), ("VHVV_dB", level + 7), ("VV_dB", -8.0)):
            rows.append({"parcel_id": pid, "date": d.isoformat(), "index": feature, "mean": value, "std": 0.5,
                         "n_valid": 20, "n_pixels": 20, "orbit_direction": group, "relative_orbit": 51})
    return rows


def test_event_markers_s1_only_harvest():
    s1 = pd.DataFrame(_s1_series("cut", 4.0) + _s1_series("flat", 0.0))
    out = cbm_markers.event_markers(None, START, END, s1_signatures=s1).set_index("parcel_id")
    assert (~out.s2_sufficient).all() and out.s1_sufficient.all()
    assert out.loc["cut", "harvest_evidence"] == "harvest_s1" and bool(out.loc["cut", "harvest_detected"])
    assert out.loc["cut", "harvest_s1_step_db"] < -1.5
    assert 0 < out.loc["cut", "harvest_confidence"] <= 0.7
    assert out.loc["flat", "harvest_evidence"] == "none" and not out.loc["flat", "harvest_detected"]
    assert (out.homogeneity == "insufficient_data").all()      # no S2 at all
    assert out.data_sufficiency.eq("s1").all()


def test_event_markers_s1_needs_group_column():
    s1 = pd.DataFrame(_s1_series("p", 2.0)).drop(columns="orbit_direction")
    with pytest.raises(ValueError, match="orbit_direction"):
        cbm_markers.event_markers(None, START, END, s1_signatures=s1)


def test_homogeneity_needs_three_dates():
    rows = [{"parcel_id": "p", "date": f"2023-06-0{d}", "index": "NDVI", "mean": 0.5, "std": 0.3,
             "n_valid": 40, "n_pixels": 40} for d in (1, 6)]
    out = cbm_markers.homogeneity(pd.DataFrame(rows))
    assert out.iloc[0]["homogeneity"] == "insufficient_data" and out.iloc[0]["homogeneity_confidence"] == 0


def test_marker_rule_reasons_insufficient_mowing_and_no_tillage():
    rules = cbm_markers.resolve_rules({"enabled": True, "mowing_expected_classes": ["grass"],
                                       "tillage_forbidden_classes": ["grass"],
                                       "insufficient_data_outcome": "yellow"})
    row = {"declared_class": "grass", "s2_sufficient": False, "tillage_detected": False, "tillage_confidence": 0.9}
    assert cbm_markers.marker_rule_reasons(row, rules) == [("insufficient_marker_data_mowing", "yellow")]
    row = {"declared_class": "grass", "s2_sufficient": True, "mowing_events": 2, "mowing_confidence": 0.9,
           "tillage_detected": True, "tillage_confidence": 0.9}
    assert cbm_markers.marker_rule_reasons(row, rules) == [("tillage_detected", "yellow")]


# --------------------------------------------------------------------------- temporal_quality

@pytest.mark.parametrize("rules", [{"enabled": "yes"}, {"index": 5}, {"step_days": 0}, {"min_coverage": 0},
                                   {"output_step_days": 7}, {"tolerance_days": -1}, {"min_clear_observations": -1},
                                   {"max_gap_days": 0}, {"max_interpolated_feature_fraction": 2},
                                   {"outcome": "red"}, {"unknown": 1}])
def test_temporal_resolve_rejects(rules):
    with pytest.raises(ValueError):
        temporal_quality.resolve(rules)


def test_season_coverage_edges():
    with pytest.raises(ValueError, match="rows, observations"):
        temporal_quality.season_coverage(np.array([100.0, 120.0]), [100, 110], 5)
    assert temporal_quality.season_coverage(np.zeros((3, 0)), [100, 110], 5).tolist() == [0, 0, 0]
    obs = np.array([[100.0, np.nan], [np.nan, np.nan]])
    cov = temporal_quality.season_coverage(obs, [100, 110, 120], 5)
    assert cov[0] == pytest.approx(1 / 3) and cov[1] == 0


def test_parcel_series_rejects_cross_year():
    with pytest.raises(ValueError, match="day-of-year"):
        temporal_quality.parcel_series(pd.DataFrame(columns=["parcel_id", "index"]), date(2022, 11, 1),
                                       date(2023, 3, 1))


def test_parcel_support_without_clear_observations():
    sig = pd.DataFrame([{"parcel_id": "p", "date": "2023-06-01", "index": "NDVI", "mean": 0.6, "std": 0.1,
                         "n_valid": 1, "n_pixels": 40}])          # below min_coverage: not a clear observation
    markers = pd.DataFrame({"parcel_id": ["p"], "other": [1]})     # no harvest column
    out = temporal_quality.parcel_support(sig, START, END, markers=markers)
    assert out.loc[0, "tq_clear_observations"] == 0
    assert out.loc[0, "tq_observed_dates"] == ""
    assert "tq_harvest_obs_distance_days" not in out


def test_evaluate_with_disabled_checks_and_attach():
    rules = temporal_quality.resolve({"enabled": True, "min_clear_observations": None, "max_gap_days": None,
                                      "max_gap_at_peak_days": None, "max_interpolated_phenology": None})
    assert temporal_quality.evaluate({"tq_clear_observations": 2}, rules) == []
    support = pd.DataFrame({"parcel_id": [1, 2], "tq_clear_observations": [3, 9], "note": ["x", "y"]})
    frame = pd.DataFrame({"parcel_id": ["2", "3"]})
    out = temporal_quality.attach(frame, support)
    assert "note" not in out and out["tq_clear_observations"].iloc[0] == 9
    assert np.isnan(out["tq_clear_observations"].iloc[1])


def _ndvi_rows(pid, days, value=0.6):
    return [{"parcel_id": pid, "date": (date(2023, 5, 1) + timedelta(days=d)).isoformat(), "index": "NDVI",
             "mean": value, "std": 0.05, "n_valid": 40, "n_pixels": 40} for d in days]


def test_markers_sparse_parcel_has_no_phenology():
    sig = pd.DataFrame(_ndvi_rows("dense", range(0, 120, 6)) + _ndvi_rows("sparse", (0, 30)))
    out = cbm.markers(sig, START, END).set_index("parcel_id")
    assert out.loc["sparse", "clear_observations"] < 4
    assert out.loc["sparse", ["peak_doy", "harvest_doy", "sos_doy"]].isna().all()
    assert np.isfinite(out.loc["dense", "peak_doy"])


def test_parcel_series_restricted_to_requested_parcels():
    sig = pd.DataFrame(_ndvi_rows("p", range(0, 60, 10)) + _ndvi_rows("q", range(5, 60, 10), value=0.2))
    series = temporal_quality.parcel_series(sig, START, END, parcel_ids=["p"])
    assert series["ids"] == ["p"] and series["y"].shape[0] == 1
    assert np.nanmax(series["y"]) == pytest.approx(0.6)            # q's values never leak into p
