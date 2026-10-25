import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

from geocrop import cbm, parcels


def _disagreeing_row():
    # Two pixels vote for A, but one very confident B pixel changes the mean winner.
    return parcels._summarise(np.array([[0.6, 0.6, 0.0], [0.4, 0.4, 1.0]]), ["a", "b"])


def test_majority_thresholds_use_selected_class_probability():
    row = _disagreeing_row()
    assert row["majority_class"] == "a" and row["mean_class"] == "b"
    assert row["confidence"] == pytest.approx(0.6)
    assert parcels.decide(row, "majority", 1) == ("a", "accepted")
    assert parcels.decide(row, "majority", 1, min_confidence=0.55) == ("a", "review")
    assert parcels.decide(row, "majority", 1, min_margin=0.1) == ("a", "review")
    assert parcels.decide(row, "majority", 1, min_confidence=0.35) == ("a", "accepted")
    assert parcels.decide(row, "mean_probability", 1, min_confidence=0.55, min_margin=0.1) == ("b", "accepted")
    assert parcels.decide(row, "combined", 1) == (None, "review")


def test_thresholds_without_selected_class_evidence_require_review():
    row = _disagreeing_row()
    del row["p_a"]
    assert parcels.decide(row, "majority", 1, min_confidence=0.1) == ("a", "review")
    assert parcels.decide(row, "majority", 1) == ("a", "accepted")


def test_agreeing_strategies_share_confidence_thresholds():
    row = parcels._summarise(np.array([[0.8, 0.7], [0.2, 0.3]]), ["a", "b"])
    for strategy in parcels.STRATEGIES:
        assert parcels.decide(row, strategy, 1, min_confidence=0.7, min_margin=0.4) == ("a", "accepted")
        assert parcels.decide(row, strategy, 1, min_confidence=0.8) == ("a", "review")


def test_review_and_insufficient_decisions_override_discrepancy_evidence():
    frame = pd.DataFrame([
        {"declaration_status": "uncertain", "decision": "review", "cp_status": "discrepancy", "calendar_consistent": False},
        {"declaration_status": "discrepancy", "decision": "review", "cp_status": "discrepancy", "calendar_consistent": True},
        {"declaration_status": "agreement", "decision": "insufficient_resolution", "cp_status": "agreement", "calendar_consistent": True},
        {"declaration_status": "discrepancy", "decision": "accepted", "cp_status": "discrepancy", "calendar_consistent": True},
        {"declaration_status": "agreement", "decision": "accepted", "cp_status": "agreement", "calendar_consistent": True},
    ])
    assert cbm.traffic_light(frame).tolist() == ["yellow", "yellow", "grey", "red", "green"]


def _write_probabilities(path):
    profile = {"driver": "GTiff", "width": 3, "height": 1, "count": 2, "dtype": "float32",
               "crs": "EPSG:32634", "transform": from_origin(500000, 5000000, 10, 10), "nodata": -9999}
    with rasterio.open(path, "w", **profile) as ds:
        ds.write(np.array([[[0.6, 0.6, 0.0]], [[0.4, 0.4, 1.0]]], dtype="float32"))
        ds.set_band_description(1, "a")
        ds.set_band_description(2, "b")
    return profile


def test_aggregate_exports_selected_class_evidence_and_preserves_descriptors(tmp_path):
    probabilities = tmp_path / "probabilities.tif"
    _write_probabilities(probabilities)
    frame = gpd.GeoDataFrame({"parcel_id": ["p1"], "declared_crop": ["a"]},
                             geometry=[box(500000, 4999990, 500030, 5000000)], crs="EPSG:32634")
    output = tmp_path / "parcels.gpkg"
    parcels.aggregate(probabilities, frame, output, strategy="majority", interior_buffer_px=0,
                      min_valid_pixels=1, min_confidence=0.55)
    row = gpd.read_file(output).iloc[0]
    assert row["confidence"] == pytest.approx(0.6)
    assert row["margin"] == pytest.approx(0.2)
    assert row["decision_confidence"] == pytest.approx(0.4)
    assert row["decision_margin"] == pytest.approx(-0.2)
    assert row["decision"] == "review" and row["traffic_light"] == "yellow"


def test_outside_aoa_parcel_stays_yellow_despite_prediction_set_discrepancy(tmp_path):
    probabilities = tmp_path / "probabilities.tif"
    profile = _write_probabilities(probabilities)
    aoa = tmp_path / "aoa.tif"
    with rasterio.open(aoa, "w", **profile) as ds:
        ds.write(np.zeros((2, 1, 3), dtype="float32"))
    frame = gpd.GeoDataFrame({"parcel_id": ["p1"], "declared_crop": ["a"]},
                             geometry=[box(500000, 4999990, 500030, 5000000)], crs="EPSG:32634")
    output = tmp_path / "parcels.gpkg"
    parcels.aggregate(probabilities, frame, output, interior_buffer_px=0, min_valid_pixels=1,
                      conformal={"thresholds": {"0.1": {"a": 0.1, "b": 0.9}}}, aoa=aoa)
    row = gpd.read_file(output).iloc[0]
    assert row["aoa_fraction"] == 0
    assert row["cp_status"] == "discrepancy"
    assert row["decision"] == "review" and row["declaration_status"] == "uncertain"
    assert row["traffic_light"] == "yellow"
