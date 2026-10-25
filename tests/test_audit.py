import json

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Point, Polygon, box

from geocrop.audit import audit_inputs


def training(geoms=None, labels=None, ids=None, crs="EPSG:32634"):
    geoms = geoms if geoms is not None else [box(500000, 4900000, 500020, 4900020),
                                            box(500040, 4900000, 500060, 4900020)]
    n = len(geoms)
    return gpd.GeoDataFrame({"polygon_id": ids if ids is not None else list(range(n)),
                             "class_name": labels if labels is not None else ["maize"] * n},
                            geometry=geoms, crs=crs)


def failures(report, code):
    return [c for c in report["checks"] if c["code"] == code and c["status"] == "fail"]


def make_stack(path, descriptions=("P1_NDVI", "P2_NDVI"), crs="EPSG:32634"):
    with rasterio.open(path, "w", driver="GTiff", height=10, width=10, count=2,
                       dtype="float32", crs=crs, transform=from_origin(499990, 4900070, 10, 10),
                       nodata=-9999) as ds:
        ds.write(np.ones((2, 10, 10), dtype="float32"))
        for i, name in enumerate(descriptions, 1):
            if name is not None:
                ds.set_band_description(i, name)
    return path


def test_clean_inputs_are_json_serializable_and_unchanged(tmp_path):
    frame = training()
    before = frame.copy(deep=True)
    report = audit_inputs(training=frame, stack=make_stack(tmp_path / "features.tif"))
    assert report["ok"]
    assert report["summary"]["errors"] == report["summary"]["warnings"] == 0
    assert report["inputs"]["training"]["class_counts"] == {"maize": 2}
    assert json.loads(json.dumps(report))["schema"] == "geocrop-input-audit/1"
    assert frame.equals(before)


def test_missing_null_blank_and_duplicate_fields():
    frame = training([box(0, 0, 2, 2), box(5, 0, 7, 2), box(10, 0, 12, 2)],
                     labels=["maize", None, " "], ids=[1, "1", None])
    report = audit_inputs(training=frame)
    assert not report["ok"]
    assert failures(report, "schema.unique")[0]["count"] == 2
    assert {c["count"] for c in failures(report, "schema.values")} == {1, 2}
    missing = audit_inputs(training=frame.drop(columns="class_name"))
    assert failures(missing, "schema.column")


def test_invalid_empty_wrong_geometry_and_missing_crs():
    frame = training([Polygon([(0, 0), (2, 2), (2, 0), (0, 2), (0, 0)]), None, Point(4, 5)], crs=None)
    report = audit_inputs(training=frame)
    assert not report["ok"]
    for code in ("vector.crs", "geometry.valid", "geometry.present", "geometry.type"):
        assert failures(report, code)


def test_conflicting_overlaps_are_errors_but_touching_boundaries_are_safe():
    conflict = training([box(0, 0, 2, 2), box(1, 1, 3, 3)], labels=["maize", "soybean"])
    report = audit_inputs(training=conflict)
    assert not report["ok"]
    assert failures(report, "training.conflicting_overlap")[0]["count"] == 1
    same = audit_inputs(training=conflict.assign(class_name="maize"))
    assert same["ok"]
    assert failures(same, "training.duplicate_support")[0]["count"] == 1
    touching = training([box(0, 0, 2, 2), box(2, 0, 4, 2)], labels=["maize", "soybean"])
    assert not failures(audit_inputs(training=touching), "training.conflicting_overlap")


def test_field_leakage_includes_boundaries_and_metric_buffer_across_crs():
    train = training()
    # First is inside, second is on the boundary, third is 5 m outside, fourth safely outside.
    metric_obs = gpd.GeoDataFrame({"class_name": ["maize"] * 4},
                           geometry=[Point(500010, 4900010), Point(500020, 4900010),
                                     Point(500025, 4900010), Point(500100, 4900010)],
                           crs=train.crs)
    obs = metric_obs.to_crs(4326)
    report = audit_inputs(training=train, observations=obs, training_exclusion_m=6)
    assert report["ok"]  # Leakage is a warning: validation can exclude these points.
    assert failures(report, "validation.training_leakage")[0]["count"] == 3
    assert report["inputs"]["observations"]["leakage_crs"] == "EPSG:32634"
    no_buffer = audit_inputs(training=train, observations=metric_obs)
    assert failures(no_buffer, "validation.training_leakage")[0]["count"] == 2


def test_geographic_inputs_use_metric_projection_for_buffer():
    train = training().to_crs(4326)
    obs = gpd.GeoDataFrame({"class_name": ["maize"]}, geometry=[Point(500025, 4900010)], crs=32634).to_crs(4326)
    report = audit_inputs(training=train, observations=obs, training_exclusion_m=6)
    assert report["ok"]
    assert failures(report, "leakage.local_projection")
    assert report["inputs"]["observations"]["training_leakage_count"] == 1
    assert report["inputs"]["observations"]["leakage_crs"] != "EPSG:4326"
    invalid_crs = audit_inputs(training=train, observations=obs, training_exclusion_m=6, distance_crs=4326)
    assert not invalid_crs["ok"]
    assert failures(invalid_crs, "validation.leakage_check")


def test_stack_extent_reprojects_vectors_and_checks_band_identity(tmp_path):
    stack = make_stack(tmp_path / "features.tif")
    report = audit_inputs(training=training().to_crs(4326), stack=stack)
    assert report["ok"]
    assert report["inputs"]["training"]["outside_stack_count"] == 0
    far = training([box(0, 0, 2, 2), box(5, 0, 7, 2)])
    assert not audit_inputs(training=far, stack=stack)["ok"]
    bad_names = make_stack(tmp_path / "bad_names.tif", descriptions=("NDVI", "NDVI"))
    assert failures(audit_inputs(stack=bad_names), "stack.band_names")
    missing_names = make_stack(tmp_path / "missing_names.tif", descriptions=(None, "NDVI"))
    assert failures(audit_inputs(stack=missing_names), "stack.band_names")


def test_unreadable_inputs_empty_inputs_and_invalid_buffer_are_reports(tmp_path):
    assert not audit_inputs()["ok"]
    missing = audit_inputs(training=tmp_path / "missing.gpkg", stack=tmp_path / "missing.tif")
    assert not missing["ok"]
    assert failures(missing, "vector.readable")
    assert failures(missing, "stack.readable")
    assert not audit_inputs(training=training(), training_exclusion_m=float("nan"))["ok"]
    assert not audit_inputs(training=training(), training_exclusion_m=-1)["ok"]
    empty = training(geoms=[], labels=[], ids=[])
    assert failures(audit_inputs(training=empty), "vector.nonempty")


def test_parcels_schema_and_custom_observation_labels():
    parcel = training().rename(columns={"polygon_id": "parcel_id"})
    obs = gpd.GeoDataFrame({"crop": ["maize"]}, geometry=[Point(500100, 4900010)], crs=32634)
    report = audit_inputs(parcels=parcel, observations=obs, observation_label_field="crop")
    assert report["ok"]
    assert not audit_inputs(parcels=parcel.drop(columns="parcel_id"))["ok"]
