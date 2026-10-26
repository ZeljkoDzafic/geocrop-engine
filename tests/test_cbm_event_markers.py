"""Event, homogeneity and data-sufficiency markers (G5) on synthetic parcel series with planted events.

All series are synthetic: they prove that planted events are detected and that stable parcels stay
silent under the default thresholds; they say nothing about accuracy on real Sentinel data.
"""
from datetime import date, timedelta

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

from geocrop import cbm, cbm_markers, config, parcels, scenes, timeseries

START, END = date(2023, 3, 15), date(2023, 10, 31)
YEAR0 = date(2023, 1, 1)


def _doy(d):
    return d.timetuple().tm_yday


def _bell(doy, peak=190, width=40, low=0.2, high=0.8):
    return low + (high - low) * np.exp(-((doy - peak) / width) ** 2)


def ndvi_harvest(doy):          # crop peaking ~day 190, cut at day 215
    v = _bell(doy, 190, 45)
    return np.where(doy >= 215, 0.18, v)


def ndvi_senescence(doy):       # same crop, slow decline to 0.3 without a cut (drought senescence / standing crop)
    rise = _bell(doy, 190, 45)
    return np.where(doy > 190, 0.8 - 0.5 * np.clip((doy - 190) / 70, 0, 1), rise)


def ndvi_harvest_after_senescence(doy):
    return ndvi_senescence(doy)


def ndvi_stable_grass(doy):
    return np.full_like(doy, 0.7, dtype=float)


def ndvi_mowing(doy, cuts=(150, 200, 250)):
    v = np.full_like(doy, 0.75, dtype=float)
    for c in cuts:
        after = doy - c
        dip = np.where((after >= 0) & (after < 30), 0.35 * (1 - after / 30), 0.0)
        v = v - dip
    return v


def ndvi_ploughed(doy):
    return np.where(doy >= 200, 0.15, _bell(doy, 170, 40))


S2_PARCELS = {
    "harvest": ndvi_harvest, "senescence": ndvi_senescence, "harvest_s1_only": ndvi_harvest_after_senescence,
    "stable": ndvi_stable_grass, "mowing": ndvi_mowing, "ploughed": ndvi_ploughed, "cloudy_stable": ndvi_stable_grass,
}


def s1_series(pid, doy, orbit):
    """VV, VH (dB) for one orbit group; planted steps per parcel, geometry offset per orbit."""
    off = 0.0 if orbit == "ASCENDING" else -1.0          # different incidence -> different level
    vv = np.full_like(doy, -11.0 + off, dtype=float)
    vh = np.full_like(doy, -17.0 + off, dtype=float)
    if pid == "harvest":
        vh = np.where(doy >= 215, vh - 3.0, vh)
    if pid == "harvest_s1_only":
        vh = np.where(doy >= 245, vh - 3.0, vh)
    if pid == "ploughed":
        vh = np.where(doy >= 200, vh - 3.0, vh)
        vv = np.where(doy >= 240, vv + 3.0, vv)
    if pid == "mowing":
        for c in (150, 200, 250):
            vh = np.where((doy >= c) & (doy < c + 20), vh - 2.5, vh)
    return vv, vh


def make_s2(seed=0, every=5, cloud_fraction=0.25):
    rng = np.random.default_rng(seed)
    rows = []
    d = START
    while d <= END:
        doy = np.array([_doy(d)], dtype=float)
        for pid, f in S2_PARCELS.items():
            # the mowing parcel is cloud-free: with 25 % cloud loss a short cut-and-regrowth cycle can fall into
            # a gap and be missed (a documented limitation, not something to tune away)
            cloudy = rng.random() < cloud_fraction and pid != "mowing"
            value = float(f(doy)[0] + rng.normal(0, 0.02))
            n_valid = 0 if cloudy else 100
            if pid == "cloudy_stable" and d == date(2023, 7, 9):      # one residual cloud/shadow outlier
                value, n_valid = 0.3, 100
            rows.append({"parcel_id": pid, "date": d.isoformat(), "index": "NDVI",
                         "mean": value if n_valid else np.nan, "std": 0.04 if n_valid else np.nan,
                         "n_valid": n_valid, "n_pixels": 100, "skewness": 0.1, "bimodality": 0.35})
        d += timedelta(days=every)
    return pd.DataFrame(rows)


def make_s1(seed=1, revisit=12):
    rng = np.random.default_rng(seed)
    rows = []
    for orbit, first in (("ASCENDING", START), ("DESCENDING", START + timedelta(days=5))):
        d = first
        while d <= END:
            doy = np.array([_doy(d)], dtype=float)
            for pid in S2_PARCELS:
                vv, vh = s1_series(pid, doy, orbit)
                vv, vh = vv[0] + rng.normal(0, 0.3), vh[0] + rng.normal(0, 0.3)
                for name, value in (("VV_dB", vv), ("VH_dB", vh), ("VHVV_dB", vh - vv)):
                    rows.append({"parcel_id": pid, "date": d.isoformat(), "index": name, "mean": value, "std": 1.0,
                                 "n_valid": 100, "n_pixels": 100, "sensor": "S1", "orbit_direction": orbit,
                                 "relative_orbit": 51 if orbit == "ASCENDING" else 124})
            d += timedelta(days=revisit)
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def markers():
    return cbm.event_markers(make_s2(), START, END, make_s1()).set_index("parcel_id")


def test_harvest_detected_with_s1_confirmation(markers):
    h = markers.loc["harvest"]
    assert h.harvest_detected and h.harvest_evidence == "harvest_s2_s1"
    assert 205 <= h.harvest_event_doy <= 225 and h.harvest_s1_step_db < -2
    assert set(h.harvest_s1_groups.split(";")) == {"ASCENDING", "DESCENDING"}
    assert h.harvest_confidence >= 0.8


def test_senescence_is_not_harvest(markers):
    s = markers.loc["senescence"]
    assert not s.harvest_detected and s.harvest_evidence == "senescence"


def test_harvest_after_senescence_found_by_s1(markers):
    s = markers.loc["harvest_s1_only"]
    assert s.harvest_detected and s.harvest_evidence == "harvest_s1"
    assert 235 <= s.harvest_event_doy <= 255


def test_ploughing_after_harvest(markers):
    p = markers.loc["ploughed"]
    assert p.harvest_detected and p.tillage_detected and p.tillage_evidence == "tillage"
    assert 230 <= p.tillage_doy <= 250 and p.tillage_s1_step_db > 2
    assert not markers.loc["harvest"].tillage_detected       # bare soil without roughness change
    assert markers.loc["harvest"].tillage_evidence == "bare_soil"


def test_mowing_events_counted_with_dates(markers):
    m = markers.loc["mowing"]
    assert m.mowing_events == 3 and m.mowing_s1_supported >= 2
    doys = [int(v) for v in m.mowing_doys.split(";")]
    assert all(abs(a - b) <= 12 for a, b in zip(doys, (150, 200, 250)))


@pytest.mark.parametrize("pid", ["stable", "cloudy_stable"])
def test_no_false_positives_on_stable_parcels(markers, pid):
    s = markers.loc[pid]
    assert not s.harvest_detected and not s.tillage_detected and s.mowing_events == 0
    assert s.harvest_evidence == "none" and s.homogeneity == "homogeneous"


def test_detection_and_false_positives_across_noise_seeds():
    """20 synthetic noise/cloud realisations: every planted event found, nothing on stable parcels."""
    for seed in range(20):
        m = cbm.event_markers(make_s2(seed=seed), START, END, make_s1(seed=100 + seed)).set_index("parcel_id")
        for pid in ("stable", "cloudy_stable"):
            assert not m.loc[pid].harvest_detected and m.loc[pid].mowing_events == 0
            assert not m.loc[pid].tillage_detected
        assert not m.loc["senescence"].harvest_detected and not m.loc["harvest"].tillage_detected
        assert m.loc["harvest"].harvest_evidence == "harvest_s2_s1"
        assert m.loc["harvest_s1_only"].harvest_evidence == "harvest_s1"
        assert m.loc["ploughed"].tillage_detected and m.loc["mowing"].mowing_events == 3


def test_data_sufficiency_fields(markers):
    s = markers.loc["stable"]
    assert s.data_sufficiency == "s2+s1" and s.s2_clear_observations >= 8
    assert s.s1_acquisitions > 30 and s.s1_orbit_groups.startswith("ASCENDING:")
    only_s2 = cbm.event_markers(make_s2(), START, END).set_index("parcel_id")
    assert (only_s2.data_sufficiency == "s2").all()
    assert only_s2.loc["harvest"].harvest_evidence == "harvest_s2"
    assert only_s2.loc["senescence"].harvest_evidence == "senescence_or_harvest"
    assert only_s2.loc["harvest"].harvest_confidence < markers.loc["harvest"].harvest_confidence
    sparse = make_s2()
    sparse.loc[sparse.date > "2023-05-01", "n_valid"] = 0
    m = cbm.event_markers(sparse, START, END).set_index("parcel_id")
    assert (m.data_sufficiency == "insufficient").all() and (m.harvest_evidence == "insufficient_data").all()


def test_s1_orbit_groups_are_not_mixed():
    """A pure geometry offset between orbits must not look like a change when acquisitions interleave."""
    s1 = make_s1()
    s1.loc[s1.orbit_direction == "DESCENDING", "mean"] -= 4.0
    m = cbm.event_markers(make_s2(), START, END, s1).set_index("parcel_id")
    assert not m.loc["stable"].harvest_detected and m.loc["stable"].mowing_s1_supported == 0
    by_orbit = cbm.event_markers(make_s2(), START, END, s1, {"s1_group_by": "relative_orbit"})
    assert by_orbit.set_index("parcel_id").loc["harvest"].harvest_evidence == "harvest_s2_s1"


def test_homogeneity_flags_mixed_parcel():
    s2 = make_s2()
    mixed = s2[s2.parcel_id == "stable"].copy()
    mixed["parcel_id"] = "mixed"
    odd = np.arange(len(mixed)) % 2 == 0
    mixed.loc[odd, "std"] = 0.28
    mixed.loc[odd, "bimodality"] = 0.85
    # wide but unimodal dispersion (e.g. a gradient) is not bimodal
    wide = s2[s2.parcel_id == "stable"].copy()
    wide["parcel_id"] = "wide"
    wide["std"], wide["bimodality"] = 0.15, 0.4
    h = cbm.homogeneity(pd.concat([s2, mixed, wide])).set_index("parcel_id")
    assert h.loc["mixed"].homogeneity == "heterogeneous" and h.loc["mixed"].homogeneity_confidence >= 0.5
    assert h.loc["wide"].homogeneity == "homogeneous"
    assert (h.drop(index=["mixed", "wide"]).homogeneity == "homogeneous").all()
    std_only = cbm.homogeneity(pd.concat([s2, mixed]).drop(columns=["bimodality"])).set_index("parcel_id")
    assert std_only.loc["mixed"].homogeneity_basis == "std_only"
    assert std_only.loc["mixed"].homogeneity_confidence < h.loc["mixed"].homogeneity_confidence


def test_drop_events_basics():
    days = np.arange(60, 300, 5.0)
    z = np.vstack([ndvi_mowing(days), ndvi_stable_grass(days), ndvi_senescence(days)])
    ev = timeseries.drop_events(z, days, 0.15, 20, 0.5, 40)
    assert len(ev[0]) == 3 and all(e["recovered"] for e in ev[0])
    assert ev[1] == [] and ev[2] == []
    with pytest.raises(ValueError):
        timeseries.drop_events(z, days, 0, 20)


def test_s1_steps_detects_step_and_ignores_noise():
    rng = np.random.default_rng(0)
    doy = np.arange(80, 300, 12.0)
    flat = cbm_markers.s1_steps(doy, -15 + rng.normal(0, 0.3, len(doy)), 24, 2)
    assert (flat.step.abs() < 1.5).all()
    step = cbm_markers.s1_steps(doy, np.where(doy >= 200, -18, -15) + rng.normal(0, 0.3, len(doy)), 24, 2)
    best = step.loc[step.z.abs().idxmax()]
    assert 195 <= best.doy <= 210 and best.step < -2


def test_event_params_validation():
    with pytest.raises(ValueError):
        cbm_markers.resolve_event_params({"nope": 1})
    with pytest.raises(ValueError):
        cbm_markers.resolve_event_params({"abrupt_drop": -1})
    with pytest.raises(ValueError):
        cbm_markers.resolve_event_params({"s1_group_by": "platform"})
    with pytest.raises(ValueError):
        cbm_markers.resolve_rules({"no_harvest_outcome": "orange"})
    with pytest.raises(ValueError):
        cbm_markers.resolve_rules({"mixed_class_share": 0.9})
    with pytest.raises(config.ConfigError):
        config.load(overrides={"cbm": {"rules": {"tillage_outcome": "purple"}}})
    with pytest.raises(config.ConfigError):
        config.load(overrides={"cbm": {"events": {"unknown_key": 1}}})
    cfg = config.load(overrides={"cbm": {"rules": {"enabled": True, "harvest_expected_classes": ["maize"]}}})
    assert cfg["cbm"]["rules"]["no_harvest_outcome"] == "yellow"
    assert cfg["cbm"]["events"]["abrupt_drop"] == cbm_markers.EVENT_DEFAULTS["abrupt_drop"]


# --------------------------------------------------------------------------- traffic-light rules

def _frame(markers):
    ids = ["harvest", "senescence", "mowing", "stable", "ploughed", "stable"]
    declared = ["maize", "maize", "grassland", "grassland", "permanent_grassland", "maize"]
    f = markers.loc[ids].reset_index()
    f["declared_class"] = declared
    f["declaration_status"] = ["agreement"] * 5 + ["uncertain"]
    f["decision"] = ["accepted"] * 5 + ["review"]
    f["cp_status"] = "agreement"
    return f


def test_traffic_light_unchanged_without_rules(markers):
    f = _frame(markers)
    base = cbm.traffic_light(f)
    assert base.tolist() == ["green"] * 5 + ["yellow"]
    assert cbm.traffic_light(f, {"enabled": False}).tolist() == base.tolist()
    d = cbm.traffic_light_details(f)
    assert d.traffic_light.tolist() == base.tolist() and d.traffic_light_reasons.iloc[-1] == "review_decision"


def test_traffic_light_marker_rules(markers):
    f = _frame(markers)
    rules = {"enabled": True, "harvest_expected_classes": ["maize"], "mowing_expected_classes": ["grassland"],
             "tillage_forbidden_classes": ["permanent_grassland"], "tillage_outcome": "red"}
    d = cbm.traffic_light_details(f, rules)
    assert d.traffic_light.tolist() == ["green", "yellow", "green", "yellow", "red", "yellow"]
    reasons = d.traffic_light_reasons.tolist()
    assert reasons[0] == "" and reasons[1] == "no_harvest_evidence"
    assert reasons[3] == "insufficient_mowing_events" and reasons[4] == "tillage_detected"
    # a review decision stays yellow but keeps the marker reason for the analyst
    assert reasons[5] == "review_decision;no_harvest_evidence"
    ignore = cbm.traffic_light(f, {**rules, "no_harvest_outcome": "ignore"})
    assert ignore.iloc[1] == "green"


def test_traffic_light_insufficient_and_heterogeneity_rules():
    f = pd.DataFrame({"declared_class": ["maize", "maize", "maize"], "declaration_status": ["agreement"] * 3,
                      "cp_status": [None] * 3, "data_sufficiency": ["insufficient", "s2", "s2"],
                      "harvest_evidence": ["insufficient_data", "harvest_s2", "harvest_s2"],
                      "harvest_detected": [False, True, True], "harvest_confidence": [0.0, 0.6, 0.6],
                      "homogeneity": ["insufficient_data", "heterogeneous", "homogeneous"],
                      "homogeneity_confidence": [0.0, 0.9, 0.9],
                      "f_maize": [1.0, 0.55, 0.9], "f_soybean": [0.0, 0.45, 0.1]})
    rules = {"enabled": True, "harvest_expected_classes": ["maize"], "insufficient_data_outcome": "yellow",
             "mixed_class_share": 0.3}
    d = cbm.traffic_light_details(f, rules)
    assert d.traffic_light.tolist() == ["yellow", "yellow", "green"]
    assert d.traffic_light_reasons.tolist() == ["insufficient_marker_data_harvest",
                                                "heterogeneous_parcel;mixed_class_pixels", ""]


# --------------------------------------------------------------------------- raster extraction

def _write(path, array, transform, crs="EPSG:32634", dtype="float32", nodata=None):
    with rasterio.open(path, "w", driver="GTiff", width=array.shape[1], height=array.shape[0], count=1,
                       dtype=dtype, crs=crs, transform=transform, nodata=nodata) as ds:
        ds.write(array.astype(dtype), 1)


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    """20x20 grid with a two-crop parcel (left NDVI 0.8 / right 0.2) and a uniform parcel; S1 asc + desc."""
    root = tmp_path_factory.mktemp("tiny")
    t = from_origin(500000, 4900000, 10, 10)
    _write(root / "ref.tif", np.ones((20, 20)), t, dtype="uint8")
    gdf = gpd.GeoDataFrame({"parcel_id": ["split", "uniform"]},
                           geometry=[box(500000, 4899900, 500100, 4900000), box(500100, 4899800, 500200, 4899900)],
                           crs="EPSG:32634")
    for d in (date(2023, 6, 1), date(2023, 6, 6)):
        folder = root / "scenes" / "s2" / d.isoformat()
        folder.mkdir(parents=True)
        red = np.full((20, 20), 1000.0)
        nir = np.full((20, 20), 4000.0)
        nir[:10, 5:10] = 1500.0        # right half of the split parcel: low NDVI
        for name, arr in (("B04", red), ("B08", nir)):
            _write(folder / f"{name}.tif", arr, t)
        _write(folder / "SCL.tif", np.full((20, 20), 4), t, dtype="uint8")
        scenes.write(folder, sensor="S2", datetime=f"{d.isoformat()}T10:00:00")
    for orbit, d, level in (("ASCENDING", date(2023, 6, 2), 0.1), ("DESCENDING", date(2023, 6, 4), 0.05)):
        folder = root / "scenes" / "s1" / d.isoformat()
        folder.mkdir(parents=True)
        _write(folder / "VV.tif", np.full((20, 20), level), t, nodata=-9999.0)
        _write(folder / "VH.tif", np.full((20, 20), level / 10), t, nodata=-9999.0)
        scenes.write(folder, sensor="S1", datetime=f"{d.isoformat()}T16:50:00", orbit_direction=orbit,
                     relative_orbit=51, backscatter="sigma0")
    return {"root": root, "ref": root / "ref.tif", "parcels": gdf}


def test_parcel_signature_moments(tiny, tmp_path):
    plain = cbm.parcel_signatures(tiny["root"] / "scenes", tiny["parcels"], tiny["ref"], tmp_path / "a.csv",
                                  START, END, ("NDVI",), interior_buffer_px=0)
    assert "bimodality" not in plain.columns                      # default output unchanged
    sig = cbm.parcel_signatures(tiny["root"] / "scenes", tiny["parcels"], tiny["ref"], tmp_path / "b.csv",
                                START, END, ("NDVI",), interior_buffer_px=0, moments=True)
    split = sig[sig.parcel_id == "split"].iloc[0]
    uniform = sig[sig.parcel_id == "uniform"].iloc[0]
    assert split["std"] > 0.19 and split["bimodality"] > 0.9
    assert uniform["std"] < 1e-6 and np.isnan(uniform["bimodality"])
    assert np.allclose(plain[["mean", "std", "n_valid"]], sig[["mean", "std", "n_valid"]], equal_nan=True)


def test_s1_signatures_per_orbit(tiny, tmp_path):
    t = cbm.s1_signatures(tiny["root"] / "scenes", tiny["parcels"], tiny["ref"], tmp_path / "s1.parquet",
                          START, END, interior_buffer_px=0)
    assert set(t["index"]) == {"VV_dB", "VH_dB", "VHVV_dB"} and set(t.orbit_direction) == {"ASCENDING", "DESCENDING"}
    asc = t[(t.orbit_direction == "ASCENDING") & (t.parcel_id == "split")].set_index("index")["mean"]
    assert asc["VV_dB"] == pytest.approx(-10, abs=1e-4) and asc["VH_dB"] == pytest.approx(-20, abs=1e-4)
    assert asc["VHVV_dB"] == pytest.approx(-10, abs=1e-4)
    assert (tmp_path / "s1.parquet").exists()


def test_aggregate_with_marker_rules(tiny, tmp_path, markers):
    """parcels.aggregate keeps its output without rules and adds reason codes with enabled rules."""
    t = from_origin(500000, 4900000, 10, 10)
    prob = np.zeros((2, 20, 20), dtype="float32")
    prob[0] = 0.9
    prob[1] = 0.1
    with rasterio.open(tmp_path / "p.tif", "w", driver="GTiff", width=20, height=20, count=2, dtype="float32",
                       crs="EPSG:32634", transform=t) as ds:
        ds.write(prob)
        ds.descriptions = ("maize", "soybean")
    gdf = tiny["parcels"].assign(declared_crop="maize")
    mk = markers.loc[["senescence", "harvest"]].reset_index()
    mk["parcel_id"] = ["split", "uniform"]
    mk = mk.assign(peak_doy=200.0, harvest_doy=np.nan, bare_soil_days=0, clear_observations=30)
    base = parcels.aggregate(tmp_path / "p.tif", gdf, tmp_path / "a.gpkg", markers=mk)
    a = gpd.read_file(tmp_path / "a.gpkg")
    assert "traffic_light_reasons" not in a.columns and "harvest_evidence" not in a.columns
    rules = {"enabled": True, "harvest_expected_classes": ["maize"]}
    summary = parcels.aggregate(tmp_path / "p.tif", gdf, tmp_path / "b.gpkg", markers=mk, marker_rules=rules)
    b = gpd.read_file(tmp_path / "b.gpkg").set_index("parcel_id")
    assert b.loc["split", "traffic_light"] == "yellow" and b.loc["split", "traffic_light_reasons"] == "no_harvest_evidence"
    assert b.loc["uniform", "traffic_light"] == "green"
    assert base["traffic_light_counts"] == {"green": 2}
    assert summary["traffic_light_reason_counts"] == {"no_harvest_evidence": 1}


def test_cli_event_markers_and_traffic_light(tiny, tmp_path, capsys):
    import json

    from geocrop.cli import main

    s2, s1 = make_s2(), make_s1()
    s2.to_csv(tmp_path / "s2.csv", index=False)
    s1.to_parquet(tmp_path / "s1.parquet", index=False)
    main(["event-markers", "--signatures", str(tmp_path / "s2.csv"), "--s1-signatures", str(tmp_path / "s1.parquet"),
          "--start", START.isoformat(), "--end", END.isoformat(), "--out", str(tmp_path / "events.csv")])
    report = json.loads(capsys.readouterr().out)
    assert report["parcels"] == len(S2_PARCELS) and report["tillage_detected"] == 1
    main(["s1-signatures", "--scenes", str(tiny["root"] / "scenes"), "--parcels", str(_save(tiny, tmp_path)),
          "--reference", str(tiny["ref"]), "--start", START.isoformat(), "--end", END.isoformat(),
          "--out", str(tmp_path / "s1sig.csv")])
    assert json.loads(capsys.readouterr().out)["acquisitions"] == 2
    events = pd.read_csv(tmp_path / "events.csv")
    events.loc[events.parcel_id == "senescence", "parcel_id"] = "split"
    events.to_csv(tmp_path / "events2.csv", index=False)
    gdf = tiny["parcels"].assign(declared_class="maize", declaration_status="agreement", decision="accepted")
    gdf.to_file(tmp_path / "parcels.gpkg")
    main(["traffic-light", "--parcels", str(tmp_path / "parcels.gpkg"), "--markers", str(tmp_path / "events2.csv"),
          "--rules", '{"harvest_expected_classes": ["maize"]}', "--out", str(tmp_path / "tl.gpkg")])
    report = json.loads(capsys.readouterr().out)
    assert report["reason_counts"] == {"no_harvest_evidence": 1}
    out = gpd.read_file(tmp_path / "tl.gpkg").set_index("parcel_id")
    assert out.loc["split", "traffic_light"] == "yellow" and out.loc["uniform", "traffic_light"] == "green"


def _save(tiny, tmp_path):
    path = tmp_path / "tiny_parcels.gpkg"
    tiny["parcels"].to_file(path)
    return path
