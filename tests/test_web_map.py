import base64
from pathlib import Path

import pytest

pytest.importorskip("fastapi")


@pytest.fixture(scope="module")
def demo_run(tmp_path_factory):
    from geocrop.demo import run
    root = tmp_path_factory.mktemp("mapdemo") / "demo"
    run(root, quick=True)
    return root


def test_job_layers_from_demo(demo_run):
    from geocrop.web_map import job_layers
    files = [p for p in Path(demo_run).rglob("*") if p.is_file()]
    payload = job_layers(files, max_px=256)
    names = {layer["name"] for layer in payload["layers"]}
    assert {"Crop classes", "Max class probability", "Inside area of applicability", "Parcels — traffic light"} <= names
    classes = next(layer for layer in payload["layers"] if layer["name"] == "Crop classes")
    png = base64.b64decode(classes["url"].split(",", 1)[1])
    assert png.startswith(b"\x89PNG") and {e["label"] for e in classes["legend"]} >= {"soybean", "maize"}
    parcels = next(layer for layer in payload["layers"] if layer["name"] == "Parcels — traffic light")
    assert parcels["kind"] == "parcels" and parcels["data"]["features"]
    assert {f["properties"]["traffic_light"] for f in parcels["data"]["features"]} <= {"green", "yellow", "red", "grey"}
    (s, w), (n, e) = payload["bounds"]
    assert 44 < s < n < 46 and 21 < w < e < 23           # demo grid: UTM 34N near 44.9 N, 22.3 E
    assert all(layer["type"] == "image" for layer in payload["layers"][:3])  # rasters below vectors


def test_job_layers_reports_broken_files(tmp_path):
    from geocrop.web_map import job_layers
    bad = tmp_path / "broken.gpkg"
    bad.write_text("not a geopackage")
    payload = job_layers([bad])
    assert payload["layers"] == [] and payload["skipped"]
