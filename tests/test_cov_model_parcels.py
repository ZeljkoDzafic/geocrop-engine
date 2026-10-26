"""Error and edge paths of geocrop.model and geocrop.parcels (synthetic data, small models)."""
from __future__ import annotations

import json

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
from joblib import dump
from rasterio.transform import from_origin
from shapely.geometry import box

from geocrop import contracts, model, parcels, plugins
from geocrop.raster import NODATA, RasterInputError
from geocrop.samples import Samples

UTM = "EPSG:32634"


def _samples(n_poly=12, per=8, classes=("a", "b", "c"), crs=UTM, spacing=6000.0, reference=None, seed=0):
    """`n_poly` polygons on a line, each in its own 5 km block, with separable classes."""
    rng = np.random.default_rng(seed)
    X, y, poly, xs, ys = [], [], [], [], []
    for p in range(n_poly):
        c = classes[p % len(classes)]
        k = classes.index(c)
        X.append(rng.normal(k * 3.0, 0.3, size=(per, 3)))
        y += [c] * per
        poly += [f"p{p}"] * per
        xs.append(500000 + p * spacing + rng.uniform(0, 50, per))
        ys.append(np.full(per, 4900000.0) + rng.uniform(0, 50, per))
    return Samples(np.vstack(X).astype("float32"), np.array(y), np.array(poly), np.concatenate(xs),
                   np.concatenate(ys), ["f1", "f2", "f3"], crs, reference_classes=reference)


# --------------------------------------------------------------------------- check_params / estimator plugins

class _Plain:
    """Estimator-like object without usable get_params, so attributes are compared instead."""

    def __init__(self, foo=1, bar=None):
        self.foo, self.bar = foo, bar

    def get_params(self, deep=False):
        raise RuntimeError("needs fitted state")

    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        return np.ones((len(X), 1))


class _NoEq:
    def __eq__(self, other):
        raise TypeError("not comparable")

    __hash__ = object.__hash__


@pytest.fixture
def plain_estimator():
    plugins.estimators.register("cov_plain", lambda params, seed: _Plain(foo=params.get("foo", 1)))
    yield "cov_plain"
    plugins.estimators.unregister("cov_plain")


def test_check_params_compares_lists_and_uncomparable_values(plain_estimator):
    model.check_params({"algorithm": plain_estimator, "foo": [1, (2, 3)]}, {"foo": [1, (2, 3)]})
    odd = _NoEq()
    model.check_params({"algorithm": plain_estimator, "foo": odd}, {"foo": odd})     # identity fallback


def test_check_params_rejects_dropped_and_unknown_keys(plain_estimator):
    with pytest.raises(ValueError, match="does not pass"):
        model.check_params({"algorithm": plain_estimator, "bar": 3}, {"bar": 3})
    with pytest.raises(ValueError, match="does not pass"):           # extra key not merged into params
        model.check_params({"algorithm": plain_estimator}, {"foo": [1, 2]})
    with pytest.raises(ValueError, match="not a parameter"):
        model.check_params({"algorithm": plain_estimator}, {"baz": 1})


# --------------------------------------------------------------------------- spatial validation errors

def test_spatial_blocks_need_metric_crs():
    s = _samples(crs="EPSG:4326")
    with pytest.raises(ValueError, match="projected CRS in metres"):
        model.spatial_blocks(s, 5000)
    assert (model.spatial_blocks(s, None) == s.polygon).all()


def test_fold_plan_rejects_bad_inputs():
    y = np.array(["a", "b"] * 5)
    groups = np.arange(10) % 5
    with pytest.raises(ValueError, match="finite"):
        model._fold_plan(y, groups, 3, 0, spatial_buffer_m=float("nan"))
    with pytest.raises(ValueError, match="coordinates"):
        model._fold_plan(y, groups, 3, 0, coordinates=np.zeros((10, 3)))
    with pytest.raises(ValueError, match="coordinates"):
        model._fold_plan(y, groups, 3, 0, coordinates=np.full((10, 2), np.nan))
    with pytest.raises(ValueError, match="one value per sample"):
        model._fold_plan(y, groups, 3, 0, polygon=np.arange(9))
    with pytest.raises(ValueError, match="two spatial groups"):
        model._fold_plan(y, np.zeros(10), 3, 0)
    with pytest.raises(ValueError, match="coordinates and polygon"):
        model._fold_plan(y, groups, 3, 0, spatial_buffer_m=10)
    with pytest.raises(ValueError, match="projected CRS"):
        model._fold_plan(y, groups, 3, 0, spatial_buffer_m=10, coordinates=np.zeros((10, 2)),
                         polygon=np.arange(10), crs="EPSG:4326")


def test_fold_plan_buffer_purges_whole_polygons():
    """Training polygons with any sampled pixel within the buffer of a test pixel are dropped entirely."""
    s = _samples(spacing=100.0)
    groups = s.polygon
    plain = model.folds(s.y, groups, 4, 0, polygon=s.polygon, coordinates=np.column_stack([s.x, s.yy]), crs=s.crs)
    buffered = model.folds(s.y, groups, 4, 0, polygon=s.polygon, coordinates=np.column_stack([s.x, s.yy]),
                           crs=s.crs, spatial_buffer_m=150)
    for (tr0, te0), (tr1, te1) in zip(plain, buffered):
        assert set(te0) == set(te1) and set(tr1) <= set(tr0)
        assert not set(s.polygon[tr1]) & set(s.polygon[te1])
        kept = set(s.polygon[tr1])
        assert all(set(np.flatnonzero(s.polygon == p)) <= set(tr1) for p in kept)   # whole polygons only


def test_holdout_mask_fraction_bounds():
    for bad in (0, 0.5, -0.1):
        with pytest.raises(ValueError, match="holdout fraction"):
            model.holdout_mask(np.array(["a"] * 4), np.arange(4), bad)


def test_calibration_split_errors():
    s = _samples()
    groups = model.spatial_blocks(s, 5000)
    with pytest.raises(ValueError, match="calibration_fraction"):
        model._calibration_split(s, groups, 1.5, 0, 0)
    with pytest.raises(ValueError, match="spatial_buffer_m"):
        model._calibration_split(s, groups, 0.2, 0, -1)
    with pytest.raises(ValueError, match="three spatial groups"):
        model._calibration_split(s, np.array(["g1", "g2"] * (len(s.y) // 2)), 0.2, 0, 0)
    geo = _samples(crs="EPSG:4326")
    with pytest.raises(ValueError, match="projected CRS"):
        model._calibration_split(geo, groups, 0.2, 0, 100)
    bad = _samples()
    bad.x[0] = np.nan
    with pytest.raises(ValueError, match="finite sample coordinates"):
        model._calibration_split(bad, groups, 0.2, 0, 100)
    with pytest.raises(ValueError, match="fewer than two proper-training groups"):
        model._calibration_split(s, groups, 0.2, 0, 1e7)                # the buffer purges every training polygon


def test_selection_samples_modes():
    s = _samples()
    with pytest.raises(ValueError, match="calibration_mode"):
        model.selection_samples(s, calibration_mode="bootstrap")
    same, split = model.selection_samples(s)
    assert same is s and split is None
    fit, split = model.selection_samples(s, calibration_mode="holdout", calibration_fraction=0.25)
    assert not set(split["fit_polygon_ids"]) & set(split["calibration_polygon_ids"])
    assert set(fit.polygon) == set(split["fit_polygon_ids"])


def test_train_with_missing_class_and_tuning(tmp_path):
    s = _samples(reference=["a", "b", "c", "d"])
    with pytest.raises(ValueError):
        model.train(s, tmp_path / "refused", params={"n_estimators": 5}, bootstrap=0)
    report = model.train(s, tmp_path / "m", params={"n_estimators": 5, "n_jobs": 1}, n_splits=3, bootstrap=0,
                         tune_grid={"min_samples_leaf": [1, 2]}, inner_splits=2, allow_missing_classes=True)
    assert report["reference_classes_without_samples"] == ["d"]
    assert report["params"]["min_samples_leaf"] in (1, 2)
    assert "nested_cv" in report and len(report["tuning"]["trials"]) == 2
    assert (tmp_path / "m" / "model.joblib").exists()


# --------------------------------------------------------------------------- smoothing / prediction edges

def test_smooth_probabilities_without_valid_pixels_is_identity():
    proba = np.full((2, 4, 4), 0.5)
    out = model.smooth_probabilities(proba, np.zeros((4, 4), dtype=bool))
    assert out.dtype == np.float32 and np.allclose(out, 0.5)


class _ManyClasses:
    classes_ = np.arange(255)


def _stack(path, nan_rows=0, size=16):
    data = np.random.default_rng(1).normal(0, 1, (2, size, size)).astype("float32")
    data[:, :nan_rows] = np.nan
    with rasterio.open(path, "w", driver="GTiff", width=size, height=size, count=2, dtype="float32",
                       crs=UTM, transform=from_origin(500000, 4900000, 10, 10), nodata=np.nan) as ds:
        ds.write(data)
        ds.descriptions = ("f1", "f2")
    return path


def test_predict_rejects_more_than_254_classes(tmp_path):
    path = tmp_path / "model.joblib"
    dump(contracts.stamp("model", {"model": _ManyClasses(), "features": ["f1", "f2"]}), path)
    with pytest.raises(ValueError, match="254"):
        model.predict(_stack(tmp_path / "s.tif"), path, tmp_path / "out")


def test_predict_smoothing_with_empty_block(tmp_path):
    from sklearn.ensemble import RandomForestClassifier

    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (60, 2))
    clf = RandomForestClassifier(n_estimators=5, random_state=0).fit(X, np.where(X[:, 0] > 0, "a", "b"))
    path = tmp_path / "model.joblib"
    dump(contracts.stamp("model", {"model": clf, "features": ["f1", "f2"]}), path)
    stack = _stack(tmp_path / "s.tif", nan_rows=8)
    res = model.predict(stack, path, tmp_path / "out", block_size=8, smoothing={"enabled": True, "window": 3})
    with rasterio.open(res["classes"]) as c, rasterio.open(res["probabilities"]) as p:
        cls, prob = c.read(1), p.read()
    assert (cls[:8] == 255).all() and (cls[8:] != 255).all()
    assert (prob[:, :8] == NODATA).all()
    assert np.allclose(prob[:, 8:].sum(axis=0), 1, atol=1e-5)
    assert json.loads((tmp_path / "out" / "prediction_manifest.json").read_text())["smoothing"]["window"] == 3


def test_predict_all_nodata_stack_fails(tmp_path):
    from sklearn.ensemble import RandomForestClassifier

    clf = RandomForestClassifier(n_estimators=2, random_state=0).fit(np.eye(2), ["a", "b"])
    path = tmp_path / "model.joblib"
    dump(contracts.stamp("model", {"model": clf, "features": ["f1", "f2"]}), path)
    with pytest.raises(RasterInputError, match="no pixel has valid values"):
        model.predict(_stack(tmp_path / "s.tif", nan_rows=16), path, tmp_path / "out")
    assert not (tmp_path / "out" / "classes.tif").exists()


# --------------------------------------------------------------------------- parcels.decision_details

ROW = {"n_valid": 10, "majority_class": "a", "mean_class": "a", "strategies_agree": True, "majority_share": 0.8,
       "p_a": 0.7, "p_b": 0.3}


@pytest.mark.parametrize("kwargs", [{"strategy": "vote"}, {"min_valid_pixels": 0}, {"min_valid_pixels": True},
                                    {"min_valid_pixels": 2.5}, {"min_confidence": 1.5},
                                    {"min_margin": -2}, {"min_majority_share": float("nan")}])
def test_decision_details_rejects_invalid_settings(kwargs):
    args = {"strategy": "mean_probability", "min_valid_pixels": 4, **kwargs}
    with pytest.raises(ValueError):
        parcels.decision_details(ROW, **args)


def test_decision_details_combined_disagreement():
    d = parcels.decision_details({**ROW, "strategies_agree": False}, "combined", 4, min_confidence=0.5)
    assert d["predicted_class"] is None and d["decision"] == "review"
    assert "strategies_disagree" in d["reasons"] and "missing_confidence" in d["reasons"]
    ok = parcels.decision_details({**ROW, "confidence": 0.7, "margin": 0.4}, "combined", 4, min_margin=0.5)
    assert ok["decision"] == "review" and ok["reasons"] == ["low_margin"]


# --------------------------------------------------------------------------- parcels.aggregate

TRANSFORM = from_origin(500000, 4900000, 10, 10)


@pytest.fixture
def prob_raster(tmp_path):
    prob = np.zeros((3, 20, 20), dtype="float32")
    prob[0, :, :10], prob[1, :, :10], prob[2, :, :10] = 0.8, 0.15, 0.05
    prob[0, :, 10:], prob[1, :, 10:], prob[2, :, 10:] = 0.1, 0.7, 0.2
    prob[:, 15:, :] = NODATA                                         # bottom rows: no valid probabilities
    path = tmp_path / "probabilities.tif"
    with rasterio.open(path, "w", driver="GTiff", width=20, height=20, count=3, dtype="float32", crs=UTM,
                       transform=TRANSFORM, nodata=NODATA) as ds:
        ds.write(prob)
        ds.descriptions = ("maize", "soy", "wheat")
    return path


def _parcels(**extra):
    geoms = [box(500000, 4899900, 500100, 4900000),     # left half, rows 0-9: maize
             box(500100, 4899900, 500200, 4900000),     # right half: soy
             box(500000, 4899800, 500200, 4899850),     # rows 15-19: no valid probabilities
             box(600000, 5000000, 600100, 5000100)]     # outside the raster
    data = {"parcel_id": ["L", "R", "N", "O"], "declared_crop": ["1", "2", "1", None], **extra}
    return gpd.GeoDataFrame(data, geometry=geoms, crs=UTM)


def test_aggregate_input_validation(prob_raster, tmp_path):
    out = tmp_path / "p.gpkg"
    with pytest.raises(ValueError, match="min_aoa_fraction"):
        parcels.aggregate(prob_raster, _parcels(), out, min_aoa_fraction=2)
    with pytest.raises(ValueError, match="not calibrated"):
        parcels.aggregate(prob_raster, _parcels(), out, conformal={"thresholds": {"0.2": {}}}, alpha=0.1)
    with pytest.raises(ValueError, match="Parcels need 'pid'"):
        parcels.aggregate(prob_raster, _parcels(), out, id_field="pid")
    with pytest.raises(ValueError, match="CRS"):
        parcels.aggregate(prob_raster, _parcels().set_crs(None, allow_override=True), out)
    dup = _parcels()
    dup.loc[1, "parcel_id"] = "L"
    with pytest.raises(ValueError, match="unique"):
        parcels.aggregate(prob_raster, dup, out)
    with rasterio.open(prob_raster, "r+") as ds:
        ds.descriptions = ("maize", "maize", "wheat")
    with pytest.raises(ValueError, match="unique class-name"):
        parcels.aggregate(prob_raster, _parcels(), out)


def test_aggregate_edge_parcels_mapping_and_aoa(prob_raster, tmp_path):
    aoa = tmp_path / "aoa.tif"
    with rasterio.open(aoa, "w", driver="GTiff", width=20, height=20, count=2, dtype="float32", crs=UTM,
                       transform=TRANSFORM) as ds:                     # no nodata value declared
        ds.write(np.stack([np.zeros((20, 20)), np.ones((20, 20))]).astype("float32"))
    markers = pd.DataFrame({"parcel_id": ["L", "R"], "peak_doy": [180.0, 190.0], "harvest_doy": [np.nan, 250.0]})
    summary = parcels.aggregate(prob_raster, _parcels(), tmp_path / "p.gpkg", min_valid_pixels=4,
                                crop_mapping={"1": "maize", "2": "soy"}, aoa=aoa, markers=markers,
                                marker_rules={"enabled": True, "harvest_expected_classes": ["soy"]})
    table = pd.read_csv(tmp_path / "p.csv").set_index("parcel_id")
    assert table.loc["L", "declared_class"] == "maize" and table.loc["L", "declaration_status"] == "agreement"
    assert table.loc["R", "predicted_class"] == "soy" and table.loc["R", "aoa_fraction"] == 1
    for pid in ("N", "O"):                              # no valid pixel / outside the raster
        assert table.loc[pid, "decision"] == "insufficient_resolution" and table.loc[pid, "n_valid"] == 0
        assert table.loc[pid, "traffic_light"] == "grey"
    assert "bare_soil_days" not in table                # rules on: missing marker columns are not invented
    assert summary["parcels"] == 4 and "traffic_light_reason_counts" in summary


def test_train_parcel_model_edges(tmp_path):
    empty = gpd.GeoDataFrame({"ref": [None], "n_valid": [5]}, geometry=[box(0, 0, 1, 1)], crs=UTM)
    empty.to_file(tmp_path / "e.gpkg")
    with pytest.raises(ValueError, match="No labelled parcels"):
        parcels.train_parcel_model(tmp_path / "e.gpkg", "ref", tmp_path / "m")
    rng = np.random.default_rng(0)
    n = 24
    labels = np.array(["maize", "soy"] * (n // 2))
    p = np.where(labels == "maize", 0.8, 0.2) + rng.normal(0, 0.05, n)
    gdf = gpd.GeoDataFrame({"ref": labels, "n_valid": 10, "p_maize": p, "p_soy": 1 - p,
                            "majority_class": labels, "mean_class": labels},
                           geometry=[box(i * 100, 0, i * 100 + 50, 50) for i in range(n)], crs=UTM)
    gdf.to_file(tmp_path / "p.gpkg")
    report = parcels.train_parcel_model(tmp_path / "p.gpkg", "ref", tmp_path / "m", n_splits=3,
                                        spatial_block_m=None, params={"n_estimators": 5, "n_jobs": 1})
    assert report["n_parcels"] == n and report["baselines"]["majority"]["overall_accuracy"] == 1.0
    assert (tmp_path / "m" / "parcel_model.joblib").exists()
