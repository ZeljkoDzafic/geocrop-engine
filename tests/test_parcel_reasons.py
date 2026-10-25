import json

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

from geocrop import parcels


def raster_inputs(tmp_path, flags, transform=None, crs="EPSG:32634"):
    profile = {"driver": "GTiff", "width": 4, "height": 1, "count": 2, "dtype": "float32",
               "crs": crs, "transform": from_origin(500000, 5000000, 10, 10), "nodata": -9999}
    prob = tmp_path / "probabilities.tif"
    with rasterio.open(prob, "w", **profile) as dst:
        dst.write(np.array([[[.8] * 4], [[.2] * 4]], dtype="float32"))
        dst.set_band_description(1, "a")
        dst.set_band_description(2, "b")
    aoa = tmp_path / "aoa.tif"
    with rasterio.open(aoa, "w", **{**profile, "transform": transform or profile["transform"]}) as dst:
        dst.write(np.zeros((1, 4), "float32"), 1)
        dst.write(np.array([flags], "float32"), 2)
    vector = gpd.GeoDataFrame({"parcel_id": ["p1"], "declared_crop": ["a"]},
                              geometry=[box(500000, 4999990, 500040, 5000000)], crs=crs)
    return prob, aoa, vector


def test_details_preserve_all_failed_thresholds():
    row = parcels._summarise(np.array([[.6, .6, 0], [.4, .4, 1]]), ["a", "b"])
    detail = parcels.decision_details(row, "majority", 1, .5, .8, 0)
    assert detail["predicted_class"] == "a" and detail["decision"] == "review"
    assert detail["reasons"] == ["low_confidence", "low_majority_share", "low_margin"]
    assert parcels.decide(row, "majority", 1, .5, .8, 0) == ("a", "review")
    assert parcels.decision_details(row, "combined", 1)["reasons"] == ["strategies_disagree"]
    assert parcels.decision_details(row, "majority", 4)["reasons"] == ["insufficient_valid_pixels"]


@pytest.mark.parametrize("flags", [[-9999] * 4, [np.nan] * 4, [2] * 4])
def test_missing_aoa_never_becomes_accepted(tmp_path, flags):
    prob, aoa, vector = raster_inputs(tmp_path, flags)
    summary = parcels.aggregate(prob, vector, tmp_path / "parcels.gpkg", aoa=aoa,
                                interior_buffer_px=0, min_valid_pixels=1)
    row = gpd.read_file(tmp_path / "parcels.gpkg").iloc[0]
    assert row.decision == "review" and row.traffic_light == "yellow"
    assert row.decision_reasons == "aoa_missing_support"
    assert row.n_aoa_valid == 0 and row.aoa_coverage == 0
    assert summary["decision_reason_counts"] == {"aoa_missing_support": 1}
    assert summary["evidence_warning_counts"] == {"aoa_partial_support": 1}


def test_partial_aoa_denominator_and_exported_reasons(tmp_path):
    prob, aoa, vector = raster_inputs(tmp_path, [1, -9999, -9999, -9999])
    output = tmp_path / "parcels.gpkg"
    summary = parcels.aggregate(prob, vector, output, aoa=aoa, interior_buffer_px=0,
                                min_valid_pixels=1, min_confidence=.9)
    row = gpd.read_file(output).iloc[0]
    assert row.aoa_fraction == row.aoa_coverage == .25
    assert row.decision_reasons == "low_confidence;aoa_below_threshold"
    assert pd.read_csv(output.with_suffix(".csv")).decision_reasons.iloc[0] == row.decision_reasons
    assert json.loads(output.with_suffix(".summary.json").read_text())["decision_reason_counts"] == {
        "aoa_below_threshold": 1, "low_confidence": 1}
    assert summary["decision_counts"] == {"review": 1}


def test_aoa_counts_only_valid_probability_support(tmp_path):
    prob, aoa, vector = raster_inputs(tmp_path, [1, 0, 0, 0])
    with rasterio.open(prob, "r+") as ds:
        cube = ds.read()
        cube[:, :, 1:] = ds.nodata
        ds.write(cube)
    parcels.aggregate(prob, vector, tmp_path / "parcels.gpkg", aoa=aoa, interior_buffer_px=0, min_valid_pixels=1)
    row = gpd.read_file(tmp_path / "parcels.gpkg").iloc[0]
    assert row.n_valid == row.n_aoa_valid == 1
    assert row.aoa_fraction == row.aoa_coverage == 1
    assert row.decision == "accepted"


def test_rejects_misaligned_aoa_without_writing_outputs(tmp_path):
    prob, aoa, vector = raster_inputs(tmp_path, [1] * 4, from_origin(500010, 5000000, 10, 10))
    output = tmp_path / "bad.gpkg"
    with pytest.raises(ValueError, match="same CRS, shape and transform"):
        parcels.aggregate(prob, vector, output, aoa=aoa)
    assert not output.exists()


@pytest.mark.parametrize("values", [[.8, .8], [1.2, -.2]])
def test_rejects_invalid_probabilities(tmp_path, values):
    prob, _, vector = raster_inputs(tmp_path, [1] * 4)
    with rasterio.open(prob, "r+") as ds:
        ds.write(np.tile(np.asarray(values, "float32")[:, None, None], (1, 1, 4)))
    with pytest.raises(ValueError, match="sum to one"):
        parcels.aggregate(prob, vector, tmp_path / "bad.gpkg", interior_buffer_px=0, min_valid_pixels=1)


def test_geographic_crs_does_not_produce_square_metre_areas(tmp_path):
    prob, _, vector = raster_inputs(tmp_path, [1] * 4, crs="EPSG:4326")
    with pytest.raises(ValueError, match="projected CRS in metres"):
        parcels.aggregate(prob, vector, tmp_path / "bad.gpkg")


def test_fallback_warning_does_not_change_acceptance(tmp_path):
    prob, _, vector = raster_inputs(tmp_path, [1] * 4)
    summary = parcels.aggregate(prob, vector, tmp_path / "parcels.gpkg", interior_buffer_px=1, min_valid_pixels=1)
    assert summary["decision_counts"] == {"accepted": 1}
    assert summary["decision_reason_counts"] == {}
    assert summary["evidence_warning_counts"] == {"interior_buffer_fallback": 1}
