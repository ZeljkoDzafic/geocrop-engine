import json

import numpy as np
import pytest
import rasterio

from geocrop import features, model
from geocrop.samples import Samples


def test_stack_bands_and_manifest(stack):
    manifest = features.load_manifest(stack)
    with rasterio.open(stack) as ds:
        assert list(ds.descriptions) == [f.name for f in manifest]
        assert ds.nodata == -9999
    groups = {f.group for f in manifest}
    assert groups == {"s2_band", "s2_index", "s1", "delta", "seasonal"}
    names = [f.name for f in manifest]
    assert "P1_NDVI" in names and "P4_VHVV_dB" in names and "delta_NDVI_P1_P2" in names
    assert "season_NDVI_argmax" in names


def test_select_periods_and_groups(stack):
    manifest = features.load_manifest(stack)
    sel = features.select(manifest, groups=["s2_index", "delta"], periods=["P1", "P2"])
    assert all(("P3" not in n and "P4" not in n) for n in sel)
    assert "delta_NDVI_P1_P2" in sel and not any(n.startswith("season_") for n in sel)
    with pytest.raises(ValueError):
        features.select(manifest, names=["nope"])


def test_samples_roundtrip(samples, tmp_path):
    path = samples.save(tmp_path / "s.npz")
    again = Samples.load(path)
    assert again.X.shape == samples.X.shape and again.features == samples.features
    assert len(set(samples.polygon)) > 20


def test_spatial_blocks_contain_whole_polygons(samples):
    blocks = model.spatial_blocks(samples, 500)
    for pid in set(samples.polygon):
        assert len(set(blocks[samples.polygon == pid])) == 1


def test_folds_no_leakage(samples):
    groups = model.spatial_blocks(samples, 500)
    for tr, te in model.folds(samples.y, groups, 3, 0):
        assert not set(groups[tr]) & set(groups[te])


def test_train_and_predict(samples, stack, tmp_path):
    rep = model.train(samples, tmp_path / "m", params={"n_estimators": 30, "n_jobs": 1}, n_splits=3,
                      spatial_block_m=500, bootstrap=50)
    cv = rep["cross_validation"]
    assert 0 <= cv["macro_f1"] <= 1 and len(cv["folds"]) == 3
    assert cv["bootstrap"]["overall_accuracy_ci"][0] <= cv["overall_accuracy"] + 1e-9
    assert (tmp_path / "m" / "feature_importance.csv").exists()
    out = model.predict(stack, tmp_path / "m" / "model.joblib", tmp_path / "pred", block_size=64)
    with rasterio.open(out["uncertainty"]) as ds:
        u = ds.read()
        assert np.nanmin(u[0]) >= 0 and np.nanmax(u[0]) <= 1 + 1e-6
        assert np.nanmax(u[2]) <= 1 + 1e-6
    with rasterio.open(out["probabilities"]) as ds:
        assert np.allclose(ds.read().sum(axis=0), 1, atol=1e-4)
    with rasterio.open(out["classes"]) as ds:
        assert set(np.unique(ds.read(1))) <= set(range(len(out["legend"])))
    rep_json = json.loads((tmp_path / "m" / "train_report.json").read_text())
    assert "_oof" not in rep_json["cross_validation"]


def test_tuning_grid(samples, tmp_path):
    rep = model.train(samples, tmp_path / "t", params={"n_estimators": 10, "n_jobs": 1}, n_splits=2,
                      spatial_block_m=500, tune_grid={"max_features": ["sqrt", 0.5]}, bootstrap=0)
    assert len(rep["tuning"]["trials"]) == 2 and rep["params"]["max_features"] in ("sqrt", 0.5)


def test_predict_rejects_missing_features(samples, stack, tmp_path):
    model.train(samples, tmp_path / "m2", features=samples.features[:5], params={"n_estimators": 5},
                n_splits=2, spatial_block_m=500, bootstrap=0)
    from joblib import dump, load
    payload = load(tmp_path / "m2" / "model.joblib")
    payload["features"] = ["missing_feature"] + payload["features"][1:]
    dump(payload, tmp_path / "m2" / "bad.joblib")
    with pytest.raises(ValueError, match="lacks"):
        model.predict(stack, tmp_path / "m2" / "bad.joblib", tmp_path / "p2")
