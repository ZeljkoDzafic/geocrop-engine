"""STAC assets must describe actual files and retain grid, class and temporal semantics."""
import json
from pathlib import Path

import numpy as np
import pytest
import rasterio
from affine import Affine
from rasterio.crs import CRS

from geocrop import export


def _raster(path, values, descriptions=(), transform=None, crs="EPSG:32634", nodata=-9999):
    values = np.asarray(values)
    if values.ndim == 2:
        values = values[None]
    profile = {"driver": "GTiff", "width": values.shape[2], "height": values.shape[1],
               "count": len(values), "dtype": values.dtype, "crs": crs,
               "transform": transform or Affine(10, 0, 500000, 0, -10, 5000000), "nodata": nodata}
    with rasterio.open(path, "w", **profile) as ds:
        ds.write(values)
        for i, name in enumerate(descriptions, 1):
            ds.set_band_description(i, name)


@pytest.fixture
def prediction(tmp_path):
    pred = tmp_path / "prediction"
    pred.mkdir()
    (pred / "class_legend.json").write_text(json.dumps({"0": "maize", "1": "winter crops"}))
    _raster(pred / "classes.tif", np.arange(20, dtype="uint8").reshape(4, 5) % 2, nodata=255)
    probabilities = np.stack([np.full((4, 5), 0.3), np.full((4, 5), 0.7)]).astype("float32")
    _raster(pred / "probabilities.tif", probabilities, ("maize", "winter crops"))
    _raster(pred / "uncertainty.tif", np.ones((3, 4, 5), "float32"), ("max_probability", "margin", "entropy"))
    return pred


def test_export_roundtrip_assets_resolve_from_external_item_location(prediction, tmp_path):
    pystac = pytest.importorskip("pystac")
    item_path = tmp_path / "catalog" / "item.json"
    item = export.stac_item(prediction, "study", "2023-03-15", "2023-10-31", item_path)
    parsed = pystac.Item.from_file(str(item_path))
    assert parsed.id == "study"
    for asset in parsed.assets.values():
        assert Path(asset.get_absolute_href()).is_file()
    classes = item["assets"]["classes"]["classification:classes"]
    assert classes[1] == {"value": 1, "name": "winter_crops", "title": "winter crops"}
    assert item["properties"]["proj:transform"] == [10, 0, 500000, 0, -10, 5000000]
    assert item["properties"]["proj:shape"] == [4, 5]
    assert export.RASTER_SCHEMA in item["stac_extensions"]
    band = item["assets"]["probabilities"]["raster:bands"][1]
    assert band["data_type"] == "float32" and band["nodata"] == -9999
    assert band["name"] == "winter crops"
    assert "aoa" not in item["assets"]


def test_only_existing_optional_assets_and_explicit_metadata_are_added(prediction, tmp_path):
    (prediction / "probabilities.tif").unlink()
    (prediction / "uncertainty.tif").unlink()
    (prediction / "run_manifest.json").write_text('{"schema":"geocrop-run/1"}')
    report = tmp_path / "model-report.json"
    report.write_text('{"samples":42}')
    item = export.stac_item(prediction, "minimal", "2023-01-01", "2023-12-31",
                            extra_assets={"model_report": {"href": str(report), "roles": ["metadata"],
                                                           "type": "application/json"}})
    assert set(item["assets"]) == {"classes", "class_legend", "run_manifest", "model_report"}
    assert (prediction / item["assets"]["model_report"]["href"]).resolve() == report
    with pytest.raises(ValueError, match="does not exist"):
        export.stac_item(prediction, "minimal", "2023-01-01", "2023-12-31",
                         extra_assets={"missing": {"href": "missing.json"}})


@pytest.mark.parametrize("start,end", [("2023-12-31", "2023-01-01"), ("2023-02-30", "2023-03-01"),
                                       ("invalid", "2023-03-01")])
def test_invalid_time_interval_is_rejected(prediction, start, end):
    with pytest.raises(ValueError):
        export.stac_item(prediction, "study", start, end)


def test_rotated_grid_and_non_epsg_crs_have_actual_footprint(prediction):
    for filename in ("probabilities.tif", "uncertainty.tif"):
        (prediction / filename).unlink()
    c, s = np.cos(np.deg2rad(17)), np.sin(np.deg2rad(17))
    transform = Affine(10 * c, 10 * s, 0, 10 * s, -10 * c, 100)
    crs = CRS.from_string("+proj=aeqd +lat_0=44.8 +lon_0=18.5 +datum=WGS84 +units=m +no_defs")
    assert crs.to_epsg() is None
    _raster(prediction / "classes.tif", np.zeros((4, 5), "uint8"), transform=transform, crs=crs, nodata=255)
    item = export.stac_item(prediction, "rotated", "2023-01-01", "2023-12-31")
    assert item["properties"]["proj:epsg"] is None
    assert "PROJCRS" in item["properties"]["proj:wkt2"]
    assert item["properties"]["proj:transform"] == list(transform)[:6]
    ring = np.asarray(item["geometry"]["coordinates"][0])
    assert len(ring) > 4  # Densified actual pixel perimeter, not a WGS84 bounding rectangle.
    np.testing.assert_allclose(item["bbox"], [*ring.min(axis=0), *ring.max(axis=0)])


def test_incompatible_prediction_grid_is_rejected(prediction):
    _raster(prediction / "probabilities.tif", np.ones((2, 4, 5), "float32"), ("maize", "winter crops"),
            transform=Affine(10, 0, 500010, 0, -10, 5000000))
    with pytest.raises(ValueError, match="not on the classes grid"):
        export.stac_item(prediction, "study", "2023-01-01", "2023-12-31")


def test_probability_band_semantics_must_match_legend(prediction):
    _raster(prediction / "probabilities.tif", np.ones((2, 4, 5), "float32"), ("winter crops", "maize"))
    with pytest.raises(ValueError, match="band descriptions"):
        export.stac_item(prediction, "study", "2023-01-01", "2023-12-31")


def test_hcat_is_optional_but_requested_mapping_must_cover_classes(prediction):
    with pytest.raises(ValueError, match="lacks a code"):
        export.stac_item(prediction, "study", "2023-01-01", "2023-12-31", hcat={"maize": {"code": "arbitrary"}})
    hcat = {name: {"code": code} for name, code in (("maize", "1"), ("winter crops", "2"))}
    item = export.stac_item(prediction, "study", "2023-01-01", "2023-12-31", hcat=hcat)
    assert item["assets"]["classes"]["classification:classes"][0]["description"] == "HCAT 1"


def test_item_cannot_overwrite_source_or_hardlink(prediction, tmp_path):
    legend = prediction / "class_legend.json"
    before = legend.read_bytes()
    alias = tmp_path / "item.json"
    alias.hardlink_to(legend)
    for output in (legend, alias):
        with pytest.raises(ValueError, match="must not overwrite"):
            export.stac_item(prediction, "study", "2023-01-01", "2023-12-31", output)
    assert legend.read_bytes() == before


def test_to_cog_preserves_class_values(prediction, tmp_path):
    output = export.to_cog(prediction / "classes.tif", tmp_path / "classes.tif")
    with rasterio.open(prediction / "classes.tif") as before, rasterio.open(output) as after:
        np.testing.assert_array_equal(before.read(), after.read())
        assert after.nodata == before.nodata and after.crs == before.crs
        assert after.tags(ns="IMAGE_STRUCTURE").get("LAYOUT") == "COG"
