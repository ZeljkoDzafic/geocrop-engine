"""Buffered CV must preserve test objects while excluding complete nearby training polygons."""
import json

import numpy as np
import pytest
from scipy.spatial.distance import cdist

from geocrop import experiments, model
from geocrop.features import Feature
from geocrop.samples import Samples


def _reference():
    polygon = np.repeat(np.arange(24).astype(str), 2)
    y = np.repeat(np.where(np.arange(24) % 2, "crop_a", "crop_b"), 2)
    groups = polygon.copy()
    xy = np.column_stack([1000.0 + np.arange(len(y)) * 1000, np.zeros(len(y))])
    original = model.folds(y, groups, 3, 7)
    tr, te = original[0]
    near, held_out = tr[0], te[0]
    xy[held_out] = [0, 0]
    xy[near] = [5, 0]
    # A second pixel of the same training polygon is far beyond the exclusion radius.
    sibling = np.flatnonzero(polygon == polygon[near])[-1]
    xy[sibling] = [500_000, 0]
    return y, polygon, groups, xy, original, near, sibling


def test_buffer_purges_entire_polygon_across_adjacent_blocks():
    y, polygon, groups, xy, original, near, sibling = _reference()
    buffered = model.folds(y, groups, 3, 7, polygon=polygon, coordinates=xy,
                           crs="EPSG:32634", spatial_buffer_m=10)
    assert near not in buffered[0][0] and sibling not in buffered[0][0]
    assert sibling in original[0][0]
    for (before, test_before), (train, test) in zip(original, buffered):
        np.testing.assert_array_equal(test, test_before)
        assert set(train).issubset(before)
        assert not set(polygon[train]) & set(polygon[test])
        assert cdist(xy[train], xy[test]).min() > 10
        for pid in np.unique(polygon[before]):
            kept = np.isin(np.flatnonzero(polygon == pid), train)
            assert kept.all() or not kept.any()
    np.testing.assert_array_equal(np.sort(np.concatenate([te for _, te in buffered])), np.arange(len(y)))


def test_zero_buffer_keeps_original_folds_and_probabilities():
    y, polygon, groups, xy, original, _, _ = _reference()
    kw = {"polygon": polygon, "coordinates": xy, "crs": "EPSG:32634", "spatial_buffer_m": 0}
    for expected, actual in zip(original, model.folds(y, groups, 3, 7, **kw)):
        for left, right in zip(expected, actual):
            np.testing.assert_array_equal(left, right)
    X = np.arange(len(y) * 2).reshape(-1, 2)
    params = {"n_estimators": 5, "n_jobs": 1}
    before = model.cross_validate(X, y, polygon, groups, params, 3, 7)
    after = model.cross_validate(X, y, polygon, groups, params, 3, 7, coordinates=xy, crs="EPSG:32634")
    np.testing.assert_array_equal(before["_oof_proba"], after["_oof_proba"])
    assert all(f["buffer_excluded_samples"] == 0 for f in after["folds"])
    assert all(f["min_train_test_distance_m"] is not None for f in after["folds"])


@pytest.mark.parametrize("crs", [None, "EPSG:4326", "EPSG:2277"])
def test_buffer_requires_metric_coordinates(crs):
    y, polygon, groups, xy, _, _, _ = _reference()
    with pytest.raises(ValueError, match="projected CRS in metres"):
        model.folds(y, groups, 3, 7, polygon=polygon, coordinates=xy, crs=crs, spatial_buffer_m=10)
    with pytest.raises(ValueError, match="coordinates and polygon"):
        model.folds(y, groups, 3, 7, polygon=polygon, spatial_buffer_m=10)


@pytest.mark.parametrize("radius", [-1, np.nan, np.inf])
def test_invalid_buffer_is_rejected(radius):
    y, _, groups, _, _, _, _ = _reference()
    with pytest.raises(ValueError, match="finite and non-negative"):
        model.folds(y, groups, 3, 7, spatial_buffer_m=radius)


def test_impossible_buffer_fails_instead_of_silently_changing_test_set():
    y, polygon, groups, xy, _, _, _ = _reference()
    with pytest.raises(ValueError, match="no training samples in fold 0"):
        model.folds(y, groups, 3, 7, polygon=polygon, coordinates=xy,
                    crs="EPSG:32634", spatial_buffer_m=1e9)


def test_buffer_cannot_remove_a_training_class():
    y, polygon, groups, xy, original, _, _ = _reference()
    tr, te = original[0]
    xy[tr[y[tr] == "crop_a"]] = [5, 0]
    xy[te[0]] = [0, 0]
    with pytest.raises(ValueError, match="removes training classes in fold 0"):
        model.folds(y, groups, 3, 7, polygon=polygon, coordinates=xy,
                    crs="EPSG:32634", spatial_buffer_m=10)


def test_training_and_ablation_report_the_same_buffered_folds(tmp_path):
    y, polygon, groups, xy, _, _, _ = _reference()
    rng = np.random.default_rng(8)
    samples = Samples(rng.normal(size=(len(y), 2)), y, polygon, xy[:, 0], xy[:, 1],
                      ["one", "two"], "EPSG:32634")
    params = {"n_estimators": 5, "n_jobs": 1}
    report = model.train(samples, tmp_path / "model", params=params, n_splits=3,
                         spatial_block_m=None, seed=7, bootstrap=0,
                         tune_grid={"max_features": [1, 2]}, spatial_buffer_m=10)
    cv = report["cross_validation"]
    assert report["spatial_buffer_m"] == cv["spatial_buffer_m"] == 10
    assert cv["folds"][0]["buffer_excluded_polygons"] >= 1
    assert all(f["min_train_test_distance_m"] > 10 for f in cv["folds"])
    assert len(report["tuning"]["trials"]) == 2
    manifest = [Feature("one", "s2_band", "S2"), Feature("two", "s2_index", "S2")]
    experiments.run(samples, manifest, tmp_path / "experiments",
                    {"both": {"features": ["one", "two"]}, "one": {"features": ["one"]}},
                    params=params, n_splits=3, spatial_block_m=None, seed=7, spatial_buffer_m=10)
    saved = json.loads((tmp_path / "experiments" / "ablation.json").read_text())
    assert saved["spatial_buffer_m"] == 10
    for exp in saved["experiments"].values():
        for observed, expected in zip(exp["cv"]["folds"], cv["folds"]):
            for key in ("n_train", "n_test", "buffer_excluded_samples", "buffer_excluded_polygons",
                        "min_train_test_distance_m"):
                assert observed[key] == expected[key]
    conformal = json.loads((tmp_path / "model" / "conformal.json").read_text())
    assert conformal["coverage_guarantee"] is False
    assert "not independent validation" in conformal["coverage_evaluation"]
