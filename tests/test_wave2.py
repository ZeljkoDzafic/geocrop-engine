"""Time series, uncertainty, applicability, CbM, field planning and export."""
import json
from datetime import date
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio

from geocrop import (
    applicability,
    cbm,
    estimators,
    export,
    fieldwork,
    model,
    parcels,
    spatial,
    timeseries,
    uncertainty,
)
from geocrop.demo import generate_scenes
from geocrop.features import load_manifest
from geocrop.tsfeatures import build_timeseries_stack, merge_stacks

START, END = date(2023, 3, 15), date(2023, 10, 31)


@pytest.fixture(scope="module")
def scenes(tmp_path_factory):
    return generate_scenes(tmp_path_factory.mktemp("scenes"))


@pytest.fixture(scope="module")
def ts_stack(scenes, tmp_path_factory):
    return build_timeseries_stack(scenes["scene_root"], scenes["reference"],
                                  tmp_path_factory.mktemp("ts") / "ts.tif", START, END, s2_indices=("NDVI",),
                                  s1_indices=("VHVV_dB",), output_step_days=20, block_size=96)


def test_whittaker_fills_gaps():
    days = np.arange(0, 200, 5.0)
    truth = np.sin(days / 200 * np.pi)
    y = np.tile(truth, (2, 1))
    w = np.ones_like(y)
    w[:, 10:20] = 0
    y[:, 10:20] = np.nan
    z = timeseries.whittaker(y, w, lam=5)
    assert np.nanmax(np.abs(z - truth)) < 0.05


def test_whittaker_envelope_resists_clouds():
    days = np.arange(0, 200, 5.0)
    truth = 0.2 + 0.6 * np.exp(-((days - 100) / 30) ** 2)
    y = np.tile(truth, (1, 1)).copy()
    y[0, [18, 19, 21]] -= 0.4
    plain = timeseries.whittaker(y, np.ones_like(y), lam=20)
    env = timeseries.whittaker(y, np.ones_like(y), lam=20, envelope_iterations=5)
    assert abs(env[0, 20] - truth[20]) < abs(plain[0, 20] - truth[20])


def test_phenology_and_harmonics():
    days = np.arange(60, 330, 5.0)
    z = 0.2 + 0.6 * np.exp(-((days - 200) / 30) ** 2)
    ph = timeseries.phenology(z[None], days)
    assert ph["peak_doy"][0] == 200 and 170 < ph["sos"][0] < 180 and 220 < ph["eos"][0] < 230
    flat = timeseries.phenology(np.full((1, len(days)), 0.3), days)
    assert np.isnan(flat["sos"][0])
    h = timeseries.harmonics(z[None], np.ones((1, len(days))), days, 2)
    assert h["rmse"][0] < 0.1 and h["a0"][0] > 0.2


def test_timeseries_stack(ts_stack, scenes, tmp_path):
    groups = {f.group for f in load_manifest(ts_stack)}
    assert groups == {"ts", "phenology", "harmonic"}
    with rasterio.open(ts_stack) as ds:
        names = list(ds.descriptions)
        peak = ds.read(names.index("phen_NDVI_peak_doy") + 1)
    lab = gpd.read_file(scenes["parcels"])
    # winter crops peak early, soybean late (synthetic trajectories)
    with rasterio.open(ts_stack) as ds:
        def peak_of(cls):
            row = lab[lab.true_class == cls].iloc[0]
            r, c = ds.index(row.geometry.centroid.x, row.geometry.centroid.y)
            return peak[r, c]
        assert peak_of("winter_crops") < peak_of("soybean")
    merged = merge_stacks([ts_stack], tmp_path / "m.tif")
    assert len(load_manifest(merged)) == len(load_manifest(ts_stack))


def test_temperature_and_conformal():
    rng = np.random.default_rng(0)
    y = rng.choice(["a", "b", "c"], 600)
    idx = np.array([["a", "b", "c"].index(v) for v in y])
    raw = rng.dirichlet(np.ones(3), 600) * 0.4
    raw[np.arange(600), idx] += 0.6 * (rng.random(600) < 0.85)
    raw /= raw.sum(axis=1, keepdims=True)
    t = uncertainty.fit_temperature(raw, y, ["a", "b", "c"])
    assert 0.05 < t < 20
    cal, test = slice(0, 300), slice(300, 600)
    q = uncertainty.conformal_thresholds(raw[cal], y[cal], ["a", "b", "c"], alphas=(0.1,))
    sets = uncertainty.prediction_sets(raw[test], ["a", "b", "c"], q["thresholds"]["0.1"])
    cov = uncertainty.coverage(sets, y[test])
    assert cov["coverage"] >= 0.85          # nominal 0.9, finite-sample tolerance


def test_estimators_factory(samples):
    X = samples.X[:400]
    y = samples.y[:400]
    for algo in ("rf", "extra_trees", "hist_gb"):
        m = estimators.make_estimator({"algorithm": algo, "n_estimators": 20, "class_weight": "balanced_subsample"}, 0)
        m.fit(X, y)
        assert m.predict_proba(X[:5]).shape[1] == len(np.unique(y))
    with pytest.raises(ValueError):
        estimators.make_estimator({"algorithm": "nope"}, 0)


def test_tempcnn(ts_stack, scenes):
    pytest.importorskip("torch")
    from geocrop.features import select
    from geocrop.samples import extract
    names = select(load_manifest(ts_stack), groups=["ts"])
    # only the `ts` features must be valid: phenology is NaN for winter crops (season starts before the series)
    s = extract(ts_stack, scenes["training"], max_per_polygon=20, features=names)
    assert not s.missing_classes()
    rep = model.train(s, Path(scenes["root"]) / "tcnn", features=names,
                      params={"algorithm": "tempcnn", "epochs": 3, "batch_size": 64},
                      n_splits=2, spatial_block_m=500, bootstrap=0)
    assert rep["params"]["n_channels"] == 2 and 0 <= rep["cross_validation"]["macro_f1"] <= 1


def test_train_outputs_uncertainty_artifacts(samples, stack, tmp_path):
    rep = model.train(samples, tmp_path / "m", params={"n_estimators": 40, "n_jobs": 1}, n_splits=3,
                      spatial_block_m=500, bootstrap=0)
    assert rep["cross_validation"]["temperature"] > 0
    conf = json.loads((tmp_path / "m" / "conformal.json").read_text())
    assert set(conf["thresholds"]) == {"0.05", "0.1", "0.2"}
    assert (tmp_path / "m" / "suspect_labels.csv").exists() and (tmp_path / "m" / "aoa.json").exists()
    out = model.predict(stack, tmp_path / "m" / "model.joblib", tmp_path / "p")
    with rasterio.open(out["aoa"]) as ds:
        flag = ds.read(2)
        assert 0.5 < (flag[flag != -9999] == 1).mean() <= 1.0     # demo scene is mostly inside the AOA


def test_parcels_with_conformal_aoa_markers(samples, stack, demo, tmp_path):
    model.train(samples, tmp_path / "m", params={"n_estimators": 40, "n_jobs": 1}, n_splits=3, spatial_block_m=500,
                bootstrap=0)
    model.predict(stack, tmp_path / "m" / "model.joblib", tmp_path / "p")
    ids = gpd.read_file(demo["parcels"]).parcel_id
    mk = pd.DataFrame({"parcel_id": ids, "peak_doy": 200.0, "harvest_doy": np.nan, "bare_soil_days": 0,
                       "clear_observations": 30})
    summary = parcels.aggregate(tmp_path / "p" / "probabilities.tif", demo["parcels"], tmp_path / "parcels.gpkg",
                                conformal=tmp_path / "m" / "conformal.json", alpha=0.1, aoa=tmp_path / "p" / "aoa.tif",
                                markers=mk, calendar={"winter_crops": {"peak": [100, 150]}})
    out = gpd.read_file(tmp_path / "parcels.gpkg")
    assert {"prediction_set", "set_size", "cp_status", "aoa_fraction", "traffic_light", "calendar_consistent"} <= set(out.columns)
    assert set(out.traffic_light) <= {"green", "yellow", "red", "grey"}
    # winter crops declared with a summer peak contradict the calendar -> red
    wc = out[(out.declared_class == "winter_crops")]
    assert (wc.traffic_light == "red").all()
    assert summary["traffic_light_counts"]


def test_traffic_light_rules():
    f = pd.DataFrame({"declaration_status": ["agreement", "agreement", "discrepancy", "uncertain", "not_declared"],
                      "cp_status": ["agreement", "uncertain", "discrepancy", "uncertain", None],
                      "calendar_consistent": [True, True, None, None, None]})
    assert cbm.traffic_light(f).tolist() == ["green", "yellow", "red", "yellow", "grey"]


def test_parcel_signatures_and_markers(scenes, tmp_path):
    sig = cbm.parcel_signatures(scenes["scene_root"], scenes["parcels"], scenes["reference"], tmp_path / "s.csv",
                                START, END, ("NDVI",), block_size=64)
    assert {"parcel_id", "date", "mean", "n_valid", "n_pixels"} <= set(sig.columns)
    m = cbm.markers(sig, START, END)
    truth = gpd.read_file(scenes["parcels"])[["parcel_id", "true_class"]]
    m = m.merge(truth, on="parcel_id")
    med = m.groupby("true_class").peak_doy.median()
    assert med["winter_crops"] < med["sunflower"] < med["soybean"]


def test_applicability_flags_extrapolation():
    rng = np.random.default_rng(1)
    X = rng.normal(0, 1, (300, 4))
    folds = np.arange(300) % 3
    aoa = applicability.fit(X, folds, ["a", "b", "c", "d"])
    di_in = aoa.dissimilarity(rng.normal(0, 1, (50, 4)))
    di_out = aoa.dissimilarity(rng.normal(8, 1, (50, 4)))
    assert (di_in <= aoa.threshold).mean() > 0.8 and (di_out > aoa.threshold).all()


def test_variogram_recovers_range():
    rng = np.random.default_rng(3)
    x, y = rng.uniform(0, 30000, 800), rng.uniform(0, 30000, 800)
    cx, cy, a = rng.uniform(0, 30000, 300), rng.uniform(0, 30000, 300), rng.normal(0, 1, 300)
    d2 = (x[:, None] - cx) ** 2 + (y[:, None] - cy) ** 2
    v = (np.exp(-d2 / (2 * 1500.0**2)) * a).sum(axis=1) + rng.normal(0, 0.05, 800)
    r = spatial.variogram(x, y, v, max_distance=15000)
    assert r["range_m"] is not None and 1500 < r["range_m"] < 12000


def test_fieldwork_spacing(demo, tmp_path):
    gdf = gpd.read_file(demo["parcels"])
    gdf["n_valid"] = 10
    gdf["entropy"] = np.linspace(0, 1, len(gdf))
    gdf["predicted_class"] = gdf.true_class
    picked = fieldwork.rank(gdf, tmp_path / "visit.gpkg", n=10, min_spacing_m=400)
    c = np.column_stack([picked.geometry.centroid.x, picked.geometry.centroid.y])
    d = np.hypot(*(c[:, None] - c[None]).transpose(2, 0, 1))
    assert len(picked) == 10 and d[np.triu_indices(len(c), 1)].min() >= 400


def test_export_cog_and_stac(samples, stack, tmp_path):
    model.train(samples, tmp_path / "m", params={"n_estimators": 20, "n_jobs": 1}, n_splits=2, spatial_block_m=500,
                bootstrap=0)
    model.predict(stack, tmp_path / "m" / "model.joblib", tmp_path / "p")
    cog = export.to_cog(tmp_path / "p" / "classes.tif", tmp_path / "pub" / "classes.tif")
    with rasterio.open(cog) as ds:
        assert ds.driver == "GTiff" and ds.count == 1
    item = export.stac_item(tmp_path / "p", "demo-2023", "2023-03-15", "2023-10-31", tmp_path / "item.json")
    assert item["assets"]["classes"]["classification:classes"] and item["bbox"][0] < item["bbox"][2]


def test_pipeline_from_scenes_with_timeseries_and_cbm(scenes, tmp_path):
    import yaml

    from geocrop import pipeline
    cal = tmp_path / "cal.yaml"
    cal.write_text(yaml.safe_dump({"winter_crops": {"peak": [100, 160]}, "soybean": {"peak": [190, 240]}}))
    cfg = {
        "grid": {"reference": scenes["reference"]},
        "windows": {"P1": {"start": "2023-04-01", "end": "2023-05-31"}, "P2": {"start": "2023-06-01", "end": "2023-07-15"},
                    "P3": {"start": "2023-07-16", "end": "2023-08-31"}, "P4": {"start": "2023-09-01", "end": "2023-10-15"}},
        "paths": {"s2_scenes": str(Path(scenes["scene_root"]) / "s2"), "s1_scenes": str(Path(scenes["scene_root"]) / "s1"),
                  "composites": str(tmp_path / "comp"), "features": str(tmp_path / "stack.tif"),
                  "training": scenes["training"], "parcels": scenes["parcels"], "observations": scenes["observations"],
                  "output": str(tmp_path / "run")},
        "compositing": {"mask_dilation_px": 1, "temporal_screening": True},
        "timeseries": {"enabled": True, "start": "2023-03-15", "end": "2023-10-31", "s2_indices": ["NDVI"],
                       "s1_indices": [], "output_step_days": 30},
        "cbm": {"enabled": True, "calendar": str(cal)},
        # Phenology metrics are NaN for winter crops (their season starts before the series): train on every
        # other group, otherwise the samples stage would lose the class and train refuses to run.
        "model": {"n_estimators": 30, "cv_folds": 2, "spatial_block_m": 500, "max_samples_per_polygon": 30,
                  "feature_groups": ["s2_band", "s2_index", "s1", "delta", "seasonal", "ts", "harmonic"]},
        "validation": {"year": 2023, "bootstrap": 0},
        "experiments": {"windows": {"groups": ["s2_band", "s2_index"]}, "timeseries": {"groups": ["ts", "harmonic"]}},
    }
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg))
    res = pipeline.run(path, log=lambda *_: None)
    assert res["composite"]["s2"][0]["scenes"] > 0
    out = gpd.read_file(tmp_path / "run" / "parcels" / "parcels.gpkg")
    assert {"traffic_light", "cp_status", "peak_doy", "calendar_consistent"} <= set(out.columns)
    table = pd.read_csv(tmp_path / "run" / "experiments" / "ablation.csv")
    assert set(table.experiment) == {"windows", "timeseries"}
