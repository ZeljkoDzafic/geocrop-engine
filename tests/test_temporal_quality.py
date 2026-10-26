"""Temporal quality: observed vs interpolated support, gap metrics and traffic-light routing (synthetic)."""
import json
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin

from geocrop import cbm, config, contracts, scenes, timeseries
from geocrop import temporal_quality as tq
from geocrop.tsfeatures import build_timeseries_stack, support_path

START, END = date(2023, 4, 1), date(2023, 10, 31)


def _curve(doy):
    return 0.2 + 0.6 * np.exp(-((doy - 200) / 40.0) ** 2)


def signatures(cloudy_peak=("B",), sparse=("C",)):
    """Parcels A (fully observed), B (clouds +-25 days around the peak), C (three clear dates only)."""
    rows = []
    d = START
    while d <= END:
        doy = d.timetuple().tm_yday
        for pid in ("A", "B", "C"):
            clear = True
            if pid in cloudy_peak and abs(doy - 200) <= 25:
                clear = False
            if pid in sparse and d not in (date(2023, 4, 11), date(2023, 7, 20), date(2023, 10, 18)):
                clear = False
            rows.append({"parcel_id": pid, "date": d.isoformat(), "index": "NDVI", "mean": _curve(doy),
                         "std": 0.02, "n_valid": 50 if clear else 3, "n_pixels": 50})
        d += timedelta(days=5)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- metrics

def test_gap_and_distance_metrics_use_actual_observations():
    obs = np.array([[100, 110, np.nan, 150], [np.nan] * 4, [100, 100, 200, np.nan]])
    np.testing.assert_array_equal(tq.clear_observations(obs), [3, 0, 2])        # duplicates counted once
    np.testing.assert_array_equal(tq.distance_to_observation(obs, [105, 105, 200]), [5, np.inf, 0])
    assert np.isnan(tq.distance_to_observation(obs, [np.nan, 1, 1])[0])
    # gap containing the target: observed day -> 0; edges bound missing sides
    np.testing.assert_array_equal(tq.gap_containing(obs, [130, 130, 200], 90, 250), [40, 160, 0])
    np.testing.assert_array_equal(tq.gap_containing(obs, [95, 95, 95], 90, 250), [10, 160, 10])
    np.testing.assert_array_equal(tq.longest_gap(obs, 90, 250), [100, 160, 100])
    cov = tq.season_coverage(obs, np.arange(90, 251, 10), 5)
    assert cov[1] == 0 and 0 < cov[0] < 1


def test_phenology_dependence_distinguishes_observed_from_interpolated_peak():
    grid = timeseries.make_grid(START, END, 5)
    days = np.array([g.timetuple().tm_yday for g in grid], dtype=float)
    full = days.copy()
    holed = np.where(np.abs(days - 200) <= 25, np.nan, days)
    obs = np.vstack([full, holed])
    y = _curve(np.vstack([full, holed]))
    w = np.isfinite(y).astype(float)
    z = timeseries.whittaker(np.nan_to_num(y), w, lam=20)
    phen = timeseries.phenology(z, days)
    m = tq.support_metrics(obs, days, 10, phen, days[::3])
    assert m["peak_obs_distance_days"][0] == 0
    assert m["peak_obs_distance_days"][1] > 10 and m["peak_gap_days"][1] >= 50
    assert m["interpolated_phenology"][0] == 0 and m["interpolated_phenology"][1] >= 1
    assert m["interpolated_feature_fraction"][1] > m["interpolated_feature_fraction"][0] == 0
    assert m["clear_observations"][0] > m["clear_observations"][1]


def test_parcel_support_per_parcel():
    sup = tq.parcel_support(signatures(), START, END).set_index("parcel_id")
    a, b, c = sup.loc["A"], sup.loc["B"], sup.loc["C"]
    assert a.tq_clear_observations > b.tq_clear_observations > c.tq_clear_observations == 3
    assert a.tq_peak_gap_days == 0 and a.tq_interpolated_phenology == 0
    assert b.tq_peak_gap_days > 50 and "peak" in b.tq_interpolated_metrics
    assert b.tq_longest_gap_days > a.tq_longest_gap_days
    # cloudy acquisitions are not listed as observed
    assert "2023-07-20" not in b.tq_observed_dates.split(";") and "2023-07-20" in a.tq_observed_dates
    assert np.isnan(c.tq_peak_doy)        # <4 clear bins: no smoothing, no phenology


def test_resolve_validates_and_defaults_off():
    assert tq.resolve()["enabled"] is False
    for bad in ({"outcome": "red"}, {"min_clear_observations": -1}, {"tolerance_days": 0},
                {"max_interpolated_feature_fraction": 2}, {"unknown": 1}, {"output_step_days": 7}):
        with pytest.raises(ValueError):
            tq.resolve(bad)


def test_config_section_defaults_and_validation(tmp_path):
    assert config.DEFAULTS["cbm"]["temporal_support"]["enabled"] is False
    assert config.DEFAULTS["timeseries"]["temporal_support"] is False
    path = tmp_path / "c.yaml"
    path.write_text("cbm:\n  temporal_support:\n    enabled: true\n    max_gap_days: 0\n", encoding="utf8")
    with pytest.raises(config.ConfigError):
        config.load(path)


# --------------------------------------------------------------------------- routing

def _frame():
    sup = tq.parcel_support(signatures(), START, END)
    base = pd.DataFrame({"parcel_id": ["A", "B", "C", "D"],
                         "declaration_status": ["agreement", "agreement", "discrepancy", "agreement"],
                         "cp_status": ["agreement", "agreement", "discrepancy", "agreement"],
                         "decision": ["accepted"] * 4})
    return tq.attach(base, sup)


def test_default_traffic_light_unchanged_without_or_with_disabled_rules():
    f = _frame()
    before = cbm.traffic_light(f)
    assert list(before) == ["green", "green", "red", "green"]
    assert list(cbm.traffic_light(f, temporal_rules={"enabled": False})) == list(before)
    d0 = cbm.traffic_light_details(f)
    d1 = cbm.traffic_light_details(f, temporal_rules=tq.RULE_DEFAULTS)
    pd.testing.assert_frame_equal(d0, d1)


def test_enabled_rules_route_weak_support_to_review_with_reason_codes():
    f = _frame()
    d = cbm.traffic_light_details(f, temporal_rules={"enabled": True})
    assert d.traffic_light.tolist() == ["green", "yellow", "red", "green"]
    assert "LONG_GAP_AT_PEAK" in d.traffic_light_reasons[1] and "INTERPOLATED_PHENOLOGY" in d.traffic_light_reasons[1]
    assert "LOW_OBS" in d.traffic_light_reasons[2] and "conformal_discrepancy" in d.traffic_light_reasons[2]
    assert d.traffic_light_reasons[3] == ""      # D has no support row; missing_support_outcome=ignore
    d = cbm.traffic_light_details(f, temporal_rules={"enabled": True, "demote_red": True,
                                                     "missing_support_outcome": "yellow"})
    assert d.traffic_light.tolist() == ["green", "yellow", "yellow", "yellow"]
    assert d.traffic_light_reasons[3] == "NO_TEMPORAL_SUPPORT"
    assert list(cbm.traffic_light(f, temporal_rules={"enabled": True})) == ["green", "yellow", "red", "green"]


def test_grey_and_review_are_never_promoted():
    f = _frame()
    f.loc[1, "decision"] = "insufficient_resolution"
    f.loc[1, "declaration_status"] = "insufficient_resolution"
    f.loc[0, "decision"] = "review"
    f.loc[0, "declaration_status"] = "uncertain"
    f.loc[0, "tq_clear_observations"] = 1
    d = cbm.traffic_light_details(f, temporal_rules={"enabled": True})
    assert d.traffic_light[1] == "grey" and d.traffic_light[0] == "yellow"
    assert "LOW_OBS" in d.traffic_light_reasons[0]


def test_temporal_rules_combine_with_marker_rules():
    f = _frame()
    f["declared_class"] = "maize"
    f["harvest_evidence"] = "none"
    f["harvest_detected"] = False
    f["harvest_confidence"] = 0.9
    f["data_sufficiency"] = "s2"
    d = cbm.traffic_light_details(f, {"enabled": True, "harvest_expected_classes": ["maize"]}, {"enabled": True})
    assert d.traffic_light_reasons[1].split(";")[:1] == ["no_harvest_evidence"]
    assert "LONG_GAP_AT_PEAK" in d.traffic_light_reasons[1]


def test_evaluate_reports_every_check():
    row = tq.attach(pd.DataFrame({"parcel_id": ["B"]}), tq.parcel_support(signatures(), START, END)).iloc[0]
    checks = tq.evaluate(row.to_dict(), tq.resolve({"enabled": True, "max_interpolated_feature_fraction": 0.5}))
    assert [c["code"] for c in checks] == ["LOW_OBS", "LONG_GAP", "LONG_GAP_AT_PEAK", "INTERPOLATED_PHENOLOGY",
                                           "INTERPOLATED_FEATURES"]
    assert all({"rule", "inputs", "threshold", "triggered", "outcome"} <= set(c) for c in checks)


# --------------------------------------------------------------------------- per-pixel rasters

def _scenes(tmp_path):
    root = tmp_path / "scenes"
    start = date(2023, 4, 1)
    reference = None
    offsets = list(range(0, 180, 5))
    for i, offset in enumerate(offsets):
        folder = root / str(i)
        folder.mkdir(parents=True)
        doy = (start + timedelta(days=offset)).timetuple().tm_yday
        nir = 0.1 + 0.5 * np.exp(-((doy - 170) / 30.0) ** 2)
        for band, value in (("B04", .05), ("B08", nir), ("SCL", 4)):
            data = np.full((1, 3), value, dtype="float32")
            if band == "SCL":
                data[0, 2] = 9                                   # pixel 2 always cloudy
                if abs(doy - 170) <= 20:
                    data[0, 1] = 9                               # pixel 1 cloudy around the peak
            with rasterio.open(folder / f"{band}.tif", "w", driver="GTiff", width=3, height=1, count=1,
                               dtype="float32", crs="EPSG:32634", nodata=-9999,
                               transform=from_origin(500000, 5000000, 10, 10)) as ds:
                ds.write(data, 1)
        scenes.write(folder, sensor="S2", datetime=f"{start + timedelta(days=offset)}T10:00:00")
        reference = folder / "B04.tif"
    return root, reference, start, start + timedelta(days=offsets[-1])


def test_pixel_support_raster_written_next_to_stack(tmp_path):
    root, ref, start, end = _scenes(tmp_path)
    out = build_timeseries_stack(root, ref, tmp_path / "ts.tif", start, end, ("NDVI",), (), step_days=5,
                                 output_step_days=15, lam=10, temporal_support=True)
    path = support_path(out)
    assert path.exists()
    manifest = json.loads(path.with_suffix(".json").read_text())
    assert contracts.check("temporal_support", manifest)["schema_version"] == 1
    with rasterio.open(path) as ds:
        names = list(ds.descriptions)
        assert names == [f"tq_NDVI_{m}" for m in tq.PIXEL_METRICS]
        assert [b["name"] for b in manifest["bands"]] == names
        data = {n: ds.read(i + 1)[0] for i, n in enumerate(names)}
        nodata = ds.nodata
    clear = data["tq_NDVI_clear_observations"]
    assert clear[0] == 36 and clear[1] < 36 and clear[2] == 0
    assert data["tq_NDVI_peak_obs_distance_days"][0] == 0
    assert data["tq_NDVI_peak_obs_distance_days"][1] > 10
    assert data["tq_NDVI_peak_gap_days"][1] > 30
    assert data["tq_NDVI_longest_gap_days"][2] == (end - start).days
    assert data["tq_NDVI_peak_obs_distance_days"][2] == nodata     # no series -> no phenology


def test_pixel_support_is_optional(tmp_path):
    root, ref, start, end = _scenes(tmp_path)
    out = build_timeseries_stack(root, ref, tmp_path / "ts.tif", start, end, ("NDVI",), (), lam=10)
    assert not support_path(out).exists()
