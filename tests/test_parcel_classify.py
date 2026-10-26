"""Parcel-level classification on signatures (G4) and temporal deep models in the registry (G13). SYNTHETIC data."""
import json
import pickle

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest

from geocrop import benchmark, estimators, model
from geocrop import parcel_classify as pc
from geocrop.demo import synthetic_parcel_series

FAST_RF = {"algorithm": "rf", "n_estimators": 60, "max_features": "sqrt", "class_weight": "balanced_subsample",
           "n_jobs": 1}


@pytest.fixture(scope="module")
def series():
    return synthetic_parcel_series(n_side=10, seed=3)


@pytest.fixture(scope="module")
def frame(series):
    return pc.parcel_features(series["signatures"], series["parcels"], series["start"], series["end"])


@pytest.fixture(scope="module")
def trained(series, frame, tmp_path_factory):
    out = tmp_path_factory.mktemp("parcel_model")
    rep = pc.train(frame, series["parcels"], "true_class", out, FAST_RF, n_splits=4, spatial_block_m=300,
                   bootstrap=0)
    return out, rep


def test_feature_layout(series, frame):
    cols = pc.feature_columns(frame)
    assert set(pc.feature_columns(frame, ["ts"])) == {c for c in cols if c.startswith("ts_")}
    ts_cols = pc.feature_columns(frame, ["ts"])
    # channel-major: all dates of NDVI mean, then all dates of NDVI std
    assert ts_cols[0].startswith("ts_NDVI_mean_d") and ts_cols[-1].startswith("ts_NDVI_std_d")
    assert estimators.channels_from_names(ts_cols) == 2
    assert any(c.startswith("stat_NDVI_mean_p") for c in cols) and any(c.startswith("win_") for c in cols)
    assert {"shape_area_m2", "shape_index", "shape_elongation", "shape_solidity", "shape_n_pixels"} <= set(cols)
    assert not any(c.startswith("qa_") for c in cols)
    assert np.isfinite(frame[cols].to_numpy()).all()
    area = series["parcels"].set_index("parcel_id").geometry.area
    assert np.allclose(frame["shape_area_m2"], area.reindex(frame.index))


def test_cloud_screening_and_missing_parcels(series):
    sig = series["signatures"].copy()
    first = sig.parcel_id.iloc[0]
    sig.loc[sig.parcel_id == first, "n_valid"] = 0            # never clear
    f = pc.parcel_features(sig, series["parcels"], series["start"], series["end"], groups=["ts"])
    assert f.loc[first].filter(like="ts_").isna().all() and f.loc[first, "qa_clear_observations"] == 0
    assert np.isfinite(f.drop(index=first).filter(like="ts_").to_numpy()).all()


def test_parcel_classifier_beats_chance(trained, series):
    _, rep = trained
    chance = 1 / len(series["classes"])
    assert rep["level"] == "parcel" and rep["samples"] == len(series["parcels"])
    assert rep["cross_validation"]["macro_f1"] > chance + 0.4
    assert rep["polygons"] == rep["samples"]                   # one sample per parcel


def test_predict_parcels_schema_and_traffic_light(trained, series, frame, tmp_path):
    out, _ = trained
    parcels = series["parcels"].copy()
    extra = parcels.iloc[[0]].copy()
    extra["parcel_id"], extra["true_class"] = "NO_SIGNATURE", None
    parcels = gpd.GeoDataFrame(pd.concat([parcels, extra], ignore_index=True), crs=parcels.crs)
    summary = pc.predict(out, frame, parcels, tmp_path / "parcels.gpkg", alpha=0.1)
    table = gpd.read_file(tmp_path / "parcels.gpkg")
    needed = {"predicted_class", "confidence", "margin", "entropy", "prediction_set", "set_size", "decision",
              "decision_reasons", "declaration_status", "cp_status", "traffic_light", "aoa_fraction", "in_training"}
    assert needed <= set(table.columns)
    assert {f"p_{c}" for c in series["classes"]} <= set(table.columns)
    row = table.set_index("parcel_id").loc["NO_SIGNATURE"]
    assert row["decision"] == "insufficient_resolution" and row["traffic_light"] == "grey"
    assert set(table.traffic_light) <= {"green", "yellow", "red", "grey"}
    assert summary["parcels"] == len(parcels) and summary["calibration_model_sha256"] == summary["model_sha256"]
    p = table[[f"p_{c}" for c in series["classes"]]].dropna().to_numpy()
    assert np.allclose(p.sum(axis=1), 1, atol=1e-6)


def test_predict_rejects_foreign_conformal(trained, series, frame, tmp_path):
    out, _ = trained
    conf = json.loads((out / "conformal.json").read_text())
    try:
        (out / "conformal.json").write_text(json.dumps({**conf, "model_sha256": "0" * 64}))
        with pytest.raises(ValueError, match="different model"):
            pc.predict(out, frame, series["parcels"], tmp_path / "p.gpkg")
    finally:
        (out / "conformal.json").write_text(json.dumps(conf))


def test_comparison_same_folds_no_leakage(series, frame, tmp_path):
    cols = pc.feature_columns(frame)
    parcels, _ = pc.to_samples(frame, series["parcels"], "true_class", cols)
    rep = pc.compare_pixel_aggregation(parcels, series["pixels"], tmp_path, FAST_RF, FAST_RF, n_splits=4,
                                       spatial_block_m=300, seed=5, bootstrap=0)
    chance = 1 / len(series["classes"])
    for method in ("parcel_classifier", "pixel_mean_probability", "pixel_majority"):
        assert rep["cross_validation"][method]["macro_f1"] > chance + 0.4
    assert sum(f["test_parcels"] for f in rep["folds"]) == rep["development_parcels"] == len(parcels.y)
    # The parcel arm reproduces geocrop.model.cross_validate exactly: same folds, same per-fold seeds.
    groups = model.spatial_blocks(parcels, 300)
    cv = model.cross_validate(parcels.X, parcels.y, parcels.polygon, groups, FAST_RF, 4, 5)
    oof = pd.read_csv(tmp_path / "comparison_oof.csv")
    assert (oof.parcel_classifier.to_numpy() == cv["_oof"]).all()
    # No parcel (hence none of its pixels) is in training and test of one fold; blocks are disjoint too.
    for tr, te, _ in model._fold_plan(parcels.y, groups, 4, 5, polygon=parcels.polygon):
        assert not set(parcels.polygon[tr]) & set(parcels.polygon[te])
        assert not set(groups[tr]) & set(groups[te])


def test_comparison_rejects_label_mismatch(series, frame, tmp_path):
    parcels, _ = pc.to_samples(frame, series["parcels"], "true_class", pc.feature_columns(frame))
    pixels = series["pixels"]
    wrong = pixels.y.copy()
    wrong[0] = "forest" if wrong[0] != "forest" else "maize"
    bad = type(pixels)(pixels.X, wrong, pixels.polygon, pixels.x, pixels.yy, pixels.features, pixels.crs)
    with pytest.raises(ValueError, match="disagree"):
        pc.compare_pixel_aggregation(parcels, bad, tmp_path, FAST_RF, FAST_RF, spatial_block_m=300)


def test_holdout_is_locked_before_cv(series, frame, tmp_path):
    rep = pc.train(frame, series["parcels"], "true_class", tmp_path, FAST_RF, n_splits=3, spatial_block_m=300,
                   holdout_fraction=0.25, bootstrap=0)
    held = set(pd.read_csv(tmp_path / "holdout_predictions.csv").polygon.astype(str))
    oof = set(pd.read_csv(tmp_path / "oof_polygons.csv").polygon.astype(str))
    assert held and not held & oof and 0 <= rep["holdout"]["macro_f1"] <= 1


# --------------------------------------------------------------------------- temporal deep models (G13)

def _toy(n=300, t_len=10, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 3, n)
    t = np.linspace(0, 1, t_len)
    a = np.stack([np.sin(2 * np.pi * (t + 0.25 * k)) for k in y]) + rng.normal(0, 0.4, (n, t_len))
    X = np.concatenate([a, np.abs(a)], axis=1).astype("float32")
    names = [f"ts_{s}_d{100 + 10 * k}" for s in ("A", "B") for k in range(t_len)]
    return X, y.astype(str), names


def test_series_layout_helpers():
    from geocrop.temporal_models import series_names, to_series
    X, _, names = _toy(4)
    assert series_names(names) == (["ts_A", "ts_B"], [100 + 10 * k for k in range(10)])
    assert to_series(X, 2).shape == (4, 10, 2) and to_series(X, 2)[0, 3, 1] == X[0, 13]
    with pytest.raises(ValueError):
        series_names(["ts_A_d1", "ts_B_d1", "ts_A_d2"])        # not channel-major
    with pytest.raises(ValueError):
        series_names(["ts_A_d1", "ndvi_P1"])
    assert estimators.channels_from_names(["ts_VHVV_dB_d100", "ts_VHVV_dB_d110", "ts_NDVI_d100", "ts_NDVI_d110"]) == 2


def test_registry_lists_temporal_models():
    assert {"tempcnn", "ltae"} <= set(estimators.available_algorithms())
    assert set(estimators.SEQUENCE_ALGORITHMS) == {"tempcnn", "ltae"}
    _, _, names = _toy(2)
    p = estimators.sequence_params({"algorithm": "ltae"}, names)
    assert p["n_channels"] == 2 and p["positions"][:2] == [100, 110]
    assert estimators.sequence_params({"algorithm": "rf"}, ["x"]) == {"algorithm": "rf"}
    assert benchmark.candidate("ltae")["algorithm"] == "ltae" and benchmark.candidate("rf")["n_estimators"]


@pytest.mark.parametrize("algo", ["ltae", "tempcnn"])
def test_temporal_model_seeded_and_picklable(algo):
    pytest.importorskip("torch")
    from sklearn.base import clone
    X, y, names = _toy()
    params = estimators.sequence_params({"algorithm": algo, "epochs": 15, "patience": 4, "batch_size": 64}, names)
    a = estimators.make_estimator(params, 7).fit(X[:200], y[:200])
    b = estimators.make_estimator(params, 7).fit(X[:200], y[:200])
    pa = a.predict_proba(X[200:])
    assert np.array_equal(pa, b.predict_proba(X[200:]))                 # determinism with a fixed seed
    assert pa.shape == (100, 3) and np.allclose(pa.sum(axis=1), 1, atol=1e-5)
    assert list(a.classes_) == ["0", "1", "2"] and (a.predict(X[200:]) == y[200:]).mean() > 0.6
    assert 1 <= a.best_epoch_ <= a.n_epochs_ <= 15
    c = estimators.make_estimator(params, 8).fit(X[:200], y[:200])
    assert not np.array_equal(pa, c.predict_proba(X[200:]))
    restored = pickle.loads(pickle.dumps(a))
    assert np.allclose(restored.predict_proba(X[200:]), pa, atol=1e-6)
    assert clone(a).get_params()["seed"] == 7


def test_ltae_attention_weights():
    pytest.importorskip("torch")
    X, y, names = _toy(120)
    m = estimators.make_estimator(estimators.sequence_params({"algorithm": "ltae", "epochs": 3}, names), 0)
    att = m.fit(X, y).attention(X[:5])
    assert att.shape == (5, m.n_head, 10) and np.allclose(att.sum(axis=-1), 1, atol=1e-5)


def test_parcel_ltae_uses_ts_block(series, frame, tmp_path):
    pytest.importorskip("torch")
    rep = pc.train(frame, series["parcels"], "true_class", tmp_path,
                   {"algorithm": "ltae", "epochs": 40, "patience": 8}, n_splits=3, spatial_block_m=300, bootstrap=0)
    assert rep["feature_groups"] == ["ts"] and rep["params"]["n_channels"] == 2
    assert rep["cross_validation"]["macro_f1"] > 1 / len(series["classes"])


def test_benchmark_models_accepts_temporal_names(series, tmp_path):
    pytest.importorskip("torch")
    px = series["pixels"]
    keep = np.isin(px.polygon, np.unique(px.polygon)[:40])
    s = type(px)(px.X[keep], px.y[keep], px.polygon[keep], px.x[keep], px.yy[keep], px.features, px.crs)
    cands = {"ltae": {**benchmark.candidate("ltae"), "epochs": 3}, "tempcnn": {**benchmark.candidate("tempcnn"),
                                                                              "epochs": 3}}
    table = benchmark.models(s, tmp_path, candidates=cands, n_splits=2, spatial_block_m=300, repeats=1)
    assert list(table.status) == ["ok", "ok"]
    s2 = type(px)(np.c_[s.X, s.X[:, :1]], s.y, s.polygon, s.x, s.yy, s.features + ["ndvi_P1"], s.crs)
    table = benchmark.models(s2, tmp_path / "b", candidates={"ltae": cands["ltae"]}, n_splits=2,
                             spatial_block_m=300, repeats=1)
    assert table.status.iloc[0].startswith("skipped")


def test_cli_parcel_classify(tmp_path, capsys):
    from geocrop.cli import main
    main(["parcel-series-demo", "--out", str(tmp_path / "d"), "--n-side", "8"])
    capsys.readouterr()
    main(["parcel-classify", "--signatures", str(tmp_path / "d" / "signatures.csv"),
          "--parcels", str(tmp_path / "d" / "parcels.gpkg"), "--start", "2023-03-15", "--end", "2023-10-31",
          "--label-field", "true_class", "--out", str(tmp_path / "o"), "--block-m", "300", "--folds", "3",
          "--params", '{"n_estimators": 40, "n_jobs": 1}', "--pixel-params", '{"n_estimators": 40, "n_jobs": 1}',
          "--bootstrap", "0", "--compare-pixels", str(tmp_path / "d" / "pixel_samples.npz")])
    out = json.loads(capsys.readouterr().out)
    assert out["prediction"]["parcels"] == 64 and "pixel_mean_probability" in out["comparison"]
    assert (tmp_path / "o" / "comparison" / "comparison.json").exists()
    # prediction only with an existing model
    main(["parcel-classify", "--signatures", str(tmp_path / "d" / "signatures.csv"),
          "--parcels", str(tmp_path / "d" / "parcels.gpkg"), "--start", "2023-03-15", "--end", "2023-10-31",
          "--model", str(tmp_path / "o" / "model"), "--out", str(tmp_path / "p")])
    assert json.loads(capsys.readouterr().out)["prediction"]["parcels"] == 64
