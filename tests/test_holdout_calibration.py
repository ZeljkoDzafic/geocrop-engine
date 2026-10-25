"""Split calibration must use an immutable predictor that never learned held-out objects."""
import hashlib
import json

import numpy as np
import pandas as pd
import pytest
import rasterio
from joblib import load
from rasterio.transform import from_origin
from scipy.spatial.distance import cdist
from sklearn.ensemble import RandomForestClassifier

from geocrop import model, uncertainty
from geocrop.samples import Samples


def _samples(n_polygons=40):
    rng = np.random.default_rng(36)
    n = n_polygons * 3
    pid = np.repeat(np.arange(n_polygons).astype(str), 3)
    y = np.repeat(np.where(np.arange(n_polygons) % 2, "a", "b"), 3)
    X = np.column_stack([np.arange(n), (y == "a") + rng.normal(0, 0.3, n)]).astype("float32")
    x = np.repeat(np.arange(n_polygons) * 1000, 3) + np.tile([0, 10, 20], n_polygons)
    return Samples(X, y, pid, x, np.zeros(n), ["identity", "signal"], "EPSG:32634")


def test_holdout_is_excluded_from_tuning_temperature_fitting_and_final_model(tmp_path, monkeypatch):
    s = _samples()
    groups = model.spatial_blocks(s, None)
    proper, calibration, split = model._calibration_split(s, groups, .25, 9, 0)
    observed_fit, observed_temperature, observed_aoa = [], [], []
    original_fit = RandomForestClassifier.fit
    original_temperature = uncertainty.fit_temperature
    original_aoa = model.applicability.fit

    def record_fit(self, X, y, *args, **kwargs):
        observed_fit.append(np.asarray(X)[:, 0].copy())
        return original_fit(self, X, y, *args, **kwargs)

    def record_temperature(proba, y, classes):
        observed_temperature.append((len(proba), np.asarray(y).copy()))
        return original_temperature(proba, y, classes)

    def record_aoa(X, *args, **kwargs):
        observed_aoa.append(np.asarray(X)[:, 0].copy())
        return original_aoa(X, *args, **kwargs)

    monkeypatch.setattr(RandomForestClassifier, "fit", record_fit)
    monkeypatch.setattr(uncertainty, "fit_temperature", record_temperature)
    monkeypatch.setattr(model.applicability, "fit", record_aoa)
    report = model.train(s, tmp_path, params={"n_estimators": 5, "n_jobs": 1},
                         n_splits=3, spatial_block_m=None, seed=9, bootstrap=0,
                         tune_grid={"max_features": [1, 2]}, calibration_mode="holdout", calibration_fraction=.25)
    assert observed_fit
    for fit in observed_fit:
        assert set(fit).issubset(s.X[proper, 0])
        assert not set(fit) & set(s.X[calibration, 0])
    np.testing.assert_array_equal(observed_fit[-1], s.X[proper, 0])
    np.testing.assert_array_equal(observed_aoa[0], s.X[proper, 0])
    assert len(observed_temperature) == 1 and observed_temperature[0][0] == len(proper)
    np.testing.assert_array_equal(observed_temperature[0][1], s.y[proper])
    assert report["samples"] == len(proper) and report["input_samples"] == len(s.y)
    assert sum(f["n_test"] for f in report["cross_validation"]["folds"]) == len(proper)
    recorded = json.loads((tmp_path / "calibration_split.json").read_text())
    assert recorded == split == report["calibration_split"]
    assert not set(recorded["fit_polygon_ids"]) & set(recorded["calibration_polygon_ids"])
    assert not set(recorded["fit_groups"]) & set(recorded["calibration_groups"])
    assert recorded["fit_samples"] + recorded["calibration_samples"] == len(s.y)

    # The exported calibration probabilities must match the actual reloaded deployment predictor.
    payload = load(tmp_path / "model.joblib")
    proba = uncertainty.apply_temperature(payload["model"].predict_proba(s.X[calibration]), payload["temperature"])
    actual = uncertainty.polygon_oof(proba, s.y[calibration], s.polygon[calibration], payload["classes"])
    saved = pd.read_csv(tmp_path / "calibration_polygons.csv", dtype={"polygon": str})
    pd.testing.assert_frame_equal(saved, actual, check_exact=False, atol=1e-14)
    conf = json.loads((tmp_path / "conformal.json").read_text())
    assert conf["predictor_refit_after_calibration"] is False
    assert conf["exchangeability_verified"] is False and conf["coverage_guarantee"] is False
    assert conf["sparse_class_policy"] == "include"


def test_holdout_buffer_purges_whole_proper_training_polygons():
    s = _samples()
    groups = model.spatial_blocks(s, None)
    fit, cal, _ = model._calibration_split(s, groups, .2, 7, 0)
    neighbour = fit[0]
    s.x[neighbour] = s.x[cal[0]] + 5
    kept, held_out, split = model._calibration_split(s, groups, .2, 7, 10)
    np.testing.assert_array_equal(cal, held_out)
    assert not np.isin(s.polygon[kept], s.polygon[neighbour]).any()
    assert cdist(np.column_stack([s.x[kept], s.yy[kept]]),
                 np.column_stack([s.x[cal], s.yy[cal]])).min() > 10
    excluded = split["buffer_excluded_polygon_ids"]
    assert excluded == [str(s.polygon[neighbour])]
    assert split["fit_samples"] + split["calibration_samples"] + split["buffer_excluded_samples"] == len(s.y)


def test_holdout_fails_on_insufficient_group_or_class_support(tmp_path):
    s = _samples(2)
    with pytest.raises(ValueError, match="at least three spatial groups"):
        model.train(s, tmp_path, calibration_mode="holdout", spatial_block_m=None, bootstrap=0)
    s = _samples()
    groups = model.spatial_blocks(s, None)
    _, cal, _ = model._calibration_split(s, groups, .2, 7, 0)
    s.y = s.y.astype("<U10")
    s.y[s.polygon == s.polygon[cal[0]]] = "rare"
    with pytest.raises(ValueError, match="Proper-training split lacks classes"):
        model.train(s, tmp_path, calibration_mode="holdout", spatial_block_m=None, seed=7, bootstrap=0)
    with pytest.raises(ValueError, match="fewer than two proper-training groups"):
        model._calibration_split(s, groups, .98, 7, 0)


@pytest.mark.parametrize("fraction", [-.1, 0, 1, float("nan"), float("inf")])
def test_bad_calibration_fraction_rejected(fraction):
    s = _samples()
    with pytest.raises(ValueError, match="calibration_fraction"):
        model._calibration_split(s, s.polygon, fraction, 7, 0)


def test_sparse_or_unobserved_classes_are_always_included():
    classes = ["common", "rare", "absent"]
    y = ["common"] * 30 + ["rare"]
    p = np.tile([.8, .15, .05], (len(y), 1))
    out = uncertainty.conformal_thresholds(p, y, classes, alphas=(.1,), sparse_class_policy="include")
    assert out["thresholds"]["0.1"]["rare"] == out["thresholds"]["0.1"]["absent"] == 1
    assert out["always_include_classes"]["0.1"] == ["absent", "rare"]
    assert out["pooled_fallback_classes"] == {}
    sets = uncertainty.prediction_sets(np.array([[1, 0, 0], [0, 0, 1]]), classes, out["thresholds"]["0.1"])
    assert all({"rare", "absent"} <= set(s) for s in sets)


@pytest.mark.parametrize("calibration_mode", ["oof", "holdout"])
def test_training_rejects_mixed_label_polygons_before_splitting_or_fitting(tmp_path, monkeypatch, calibration_mode):
    s = _samples()
    s.y[1] = "a" if s.y[0] == "b" else "b"

    def should_not_split(*args, **kwargs):
        raise AssertionError("Malformed objects reached calibration splitting")

    monkeypatch.setattr(model, "_calibration_split", should_not_split)
    with pytest.raises(ValueError, match="mixed labels"):
        model.train(s, tmp_path, calibration_mode=calibration_mode, spatial_block_m=None)
    assert not (tmp_path / "model.joblib").exists()


def test_standalone_polygon_aggregation_rejects_multiple_labels_per_object():
    with pytest.raises(ValueError, match="mixed labels"):
        uncertainty.polygon_oof([[.8, .2], [.3, .7]], ["a", "b"], ["one", "one"], ["a", "b"])


def test_selection_helper_matches_training_partition_without_mutating_input(tmp_path):
    s = _samples()
    before = s.X.copy()
    proper, split = model.selection_samples(s, spatial_block_m=None, seed=9,
                                            calibration_mode="holdout", calibration_fraction=.25)
    report = model.train(s, tmp_path, params={"n_estimators": 5, "n_jobs": 1},
                         n_splits=3, spatial_block_m=None, seed=9, bootstrap=0,
                         calibration_mode="holdout", calibration_fraction=.25)
    assert split == report["calibration_split"]
    assert sorted(set(proper.polygon)) == split["fit_polygon_ids"]
    assert not set(proper.polygon) & set(split["calibration_polygon_ids"])
    np.testing.assert_array_equal(before, s.X)
    unchanged, no_split = model.selection_samples(s, calibration_mode="oof")
    assert unchanged is s and no_split is None


def test_default_oof_and_holdout_modes_remain_distinct(tmp_path):
    s = _samples()
    common = {"params": {"n_estimators": 5, "n_jobs": 1}, "spatial_block_m": None,
              "bootstrap": 0, "calibrate": False}
    report = model.train(s, tmp_path / "oof", **common)
    conf = json.loads((tmp_path / "oof" / "conformal.json").read_text())
    assert report["samples"] == len(s.y) and report["calibration_mode"] == "oof"
    assert conf["coverage_guarantee"] is False and conf["sparse_class_policy"] == "pooled"
    report = model.train(s, tmp_path / "holdout", calibration_mode="holdout", **common)
    payload = load(tmp_path / "holdout" / "model.joblib")
    assert payload["temperature"] == 1 and report["samples"] < len(s.y)


@pytest.mark.parametrize("calibration_mode", ["oof", "holdout"])
def test_prediction_manifest_and_rasters_bind_to_actual_saved_predictor(tmp_path, calibration_mode):
    s = _samples()
    trained = tmp_path / "model"
    report = model.train(s, trained, params={"n_estimators": 5, "n_jobs": 1},
                         n_splits=3, spatial_block_m=None, bootstrap=0, calibration_mode=calibration_mode)
    digest = hashlib.sha256((trained / "model.joblib").read_bytes()).hexdigest()
    conformal = json.loads((trained / "conformal.json").read_text())
    saved_report = json.loads((trained / "train_report.json").read_text())
    assert report["model_sha256"] == saved_report["model_sha256"] == conformal["model_sha256"] == digest
    stack = tmp_path / "stack.tif"
    with rasterio.open(stack, "w", driver="GTiff", dtype="float32", count=2,
                       width=len(s.y), height=1, crs=s.crs,
                       transform=from_origin(500000, 5000000, 10, 10), nodata=-9999) as dst:
        dst.write(s.X.T[:, None, :])
        for i, name in enumerate(s.features, 1):
            dst.set_band_description(i, name)
    outputs = model.predict(stack, trained / "model.joblib", tmp_path / "prediction", block_size=16)
    manifest = json.loads((tmp_path / "prediction/prediction_manifest.json").read_text())
    assert outputs["manifest"] == str(tmp_path / "prediction/prediction_manifest.json")
    assert manifest["model_sha256"] == digest
    assert manifest["calibration_mode"] == calibration_mode
    assert manifest["classes"] == ["a", "b"] and manifest["features"] == s.features
    payload = load(trained / "model.joblib")
    assert manifest["temperature"] == payload["temperature"]
    expected = uncertainty.apply_temperature(payload["model"].predict_proba(s.X), payload["temperature"])
    for key in ("classes", "probabilities", "uncertainty"):
        with rasterio.open(outputs[key]) as ds:
            assert ds.tags()["model_sha256"] == digest
            if key == "probabilities":
                np.testing.assert_allclose(ds.read()[:, 0, :].T, expected.astype("float32"), atol=1e-7)
