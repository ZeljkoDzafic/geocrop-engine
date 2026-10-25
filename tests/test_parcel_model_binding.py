import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

from geocrop.parcels import aggregate


def probability_input(tmp_path, model_hash):
    source = tmp_path / "probabilities.tif"
    with rasterio.open(source, "w", driver="GTiff", width=2, height=2, count=2, dtype="float32",
                       crs="EPSG:32634", transform=from_origin(500000, 5000000, 10, 10), nodata=-9999) as ds:
        ds.write(np.array([[[.8] * 2] * 2, [[.2] * 2] * 2], dtype="float32"))
        ds.set_band_description(1, "a")
        ds.set_band_description(2, "b")
        if model_hash:
            ds.update_tags(model_sha256=model_hash)
    parcels = gpd.GeoDataFrame({"parcel_id": ["one"], "declared_crop": ["a"]},
                              geometry=[box(500000, 4999980, 500020, 5000000)], crs="EPSG:32634")
    return source, parcels


@pytest.mark.parametrize("model_hash", ["different-model", None])
def test_bound_thresholds_reject_mismatched_or_unbound_predictions(tmp_path, model_hash):
    source, parcels = probability_input(tmp_path, model_hash)
    thresholds = {"model_sha256": "expected-model", "thresholds": {"0.1": {"a": .3, "b": .3}}}
    with pytest.raises(ValueError, match="does not match the probability raster"):
        aggregate(source, parcels, tmp_path / "parcels.gpkg", conformal=thresholds)
    assert not (tmp_path / "parcels.gpkg").exists()


def test_matching_binding_and_legacy_thresholds_work(tmp_path):
    source, parcels = probability_input(tmp_path, "same-model")
    thresholds = {"model_sha256": "same-model", "thresholds": {"0.1": {"a": .3, "b": .3}}}
    bound = aggregate(source, parcels, tmp_path / "bound.gpkg", conformal=thresholds,
                      interior_buffer_px=0, min_valid_pixels=1)
    assert bound["calibration_model_sha256"] == "same-model"
    thresholds.pop("model_sha256")
    legacy = aggregate(source, parcels, tmp_path / "legacy.gpkg", conformal=thresholds,
                       interior_buffer_px=0, min_valid_pixels=1)
    assert legacy["decision_counts"] == bound["decision_counts"]


def test_threshold_class_mismatch_is_not_silently_included(tmp_path):
    source, parcels = probability_input(tmp_path, None)
    with pytest.raises(ValueError, match="exactly match"):
        aggregate(source, parcels, tmp_path / "bad.gpkg", conformal={"thresholds": {"0.1": {"a": .3}}})
