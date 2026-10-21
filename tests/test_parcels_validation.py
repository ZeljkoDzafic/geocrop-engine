import geopandas as gpd
import pytest

from geocrop import model, parcels, validation


@pytest.fixture(scope="module")
def prediction(samples, stack, tmp_path_factory):
    root = tmp_path_factory.mktemp("pred")
    model.train(samples, root / "m", params={"n_estimators": 40, "n_jobs": 1}, n_splits=3, spatial_block_m=500,
                bootstrap=0)
    model.predict(stack, root / "m" / "model.joblib", root / "p")
    return root / "p"


def test_decide_rules():
    row = {"n_valid": 10, "majority_class": "a", "mean_class": "b", "strategies_agree": False,
           "confidence": 0.6, "majority_share": 0.5, "margin": 0.1}
    assert parcels.decide(row, "majority", 4) == ("a", "accepted")
    assert parcels.decide(row, "combined", 4) == (None, "review")
    assert parcels.decide(row, "mean_probability", 4, min_confidence=0.7) == ("b", "review")
    assert parcels.decide({**row, "n_valid": 2}, "majority", 4)[1] == "insufficient_resolution"
    assert parcels.declaration_status("a", "a", "accepted") == "agreement"
    assert parcels.declaration_status(None, "a", "accepted") == "not_declared"


def test_aggregate_finds_wrong_declarations(prediction, demo, tmp_path):
    summary = parcels.aggregate(prediction / "probabilities.tif", demo["parcels"], tmp_path / "parcels.gpkg")
    out = gpd.read_file(tmp_path / "parcels.gpkg")
    assert summary["parcels"] == len(out) and (tmp_path / "parcels.csv").exists()
    wrong = out[out.declared_crop != out.true_class]
    flagged = out[out.declaration_status == "discrepancy"]
    # most injected declaration errors must be flagged, and few correct declarations
    assert len(set(wrong.parcel_id) & set(flagged.parcel_id)) >= 0.7 * len(wrong)
    assert len(set(flagged.parcel_id) - set(wrong.parcel_id)) <= 0.1 * len(out)
    assert {"p_soybean", "f_maize", "shape_index", "prob_std"} <= set(out.columns)


def test_parcel_model(prediction, demo, tmp_path):
    parcels.aggregate(prediction / "probabilities.tif", demo["parcels"], tmp_path / "parcels.gpkg")
    rep = parcels.train_parcel_model(tmp_path / "parcels.gpkg", "true_class", tmp_path / "pm", n_splits=3,
                                     spatial_block_m=500, params={"n_estimators": 30, "n_jobs": 1})
    assert 0 <= rep["cross_validation"]["macro_f1"] <= 1
    assert set(rep["baselines"]) == {"majority", "mean_probability"}


def test_validation_independence(prediction, demo, tmp_path):
    import pandas as pd

    obs = gpd.read_file(demo["observations"])
    train = gpd.read_file(demo["training"])
    leak = gpd.GeoDataFrame({"obs_id": ["leak"], "class_name": [train.class_name.iloc[0]],
                             "observed_at": ["2023-07-01T00:00:00Z"]},
                            geometry=[train.geometry.iloc[0].centroid], crs=obs.crs)
    obs = gpd.GeoDataFrame(pd.concat([obs, leak], ignore_index=True), crs=obs.crs)
    r = validation.evaluate_points(prediction, obs, tmp_path / "v", year=2023, training=demo["training"],
                                   training_exclusion_m=20, window_px=1, bootstrap=50)
    assert r["selection"]["excluded_near_training"] == 1
    assert r["accuracy"]["n"] == r["selection"]["evaluated"]
    assert "reject_curve" in r and "calibration" in r
    with pytest.raises(ValueError):
        validation.evaluate_points(prediction, demo["observations"], tmp_path / "v2", year=2019)


def test_mapped_area(prediction):
    areas = validation.mapped_area(prediction / "classes.tif")
    assert sum(areas.values()) > 0 and set(areas) >= {"soybean", "maize"}
