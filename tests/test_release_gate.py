"""Release-gate tooling (tools/release_gate.py) and regressions found by running the gate."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from geocrop import benchmark, estimators

ROOT = Path(__file__).resolve().parents[1]


def _broken_native(monkeypatch, tmp_path, name: str):
    """A package that is installed but whose native library cannot be loaded (e.g. lightgbm without libomp)."""
    pkg = tmp_path / name
    pkg.mkdir()
    (pkg / "__init__.py").write_text("raise OSError('dlopen: Library not loaded: @rpath/libomp.dylib')\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, name, raising=False)


@pytest.mark.parametrize("algorithm", ["lightgbm", "catboost"])
def test_unloadable_native_backend_is_reported_as_missing(monkeypatch, tmp_path, algorithm):
    """Regression: an OSError from a broken native library crashed `geocrop benchmark-models` instead of
    marking that model as skipped."""
    _broken_native(monkeypatch, tmp_path, algorithm)
    with pytest.raises(RuntimeError, match="cannot load its native library"):
        estimators.make_estimator({"algorithm": algorithm}, 0)
    assert "native library" in benchmark._available({"algorithm": algorithm})


def test_benchmark_models_skips_unloadable_backend(monkeypatch, tmp_path, samples):
    _broken_native(monkeypatch, tmp_path, "lightgbm")
    table = benchmark.models(samples, tmp_path / "bm", candidates={
        "rf": {"algorithm": "rf", "n_estimators": 10, "n_jobs": 1},
        "lightgbm": {"algorithm": "lightgbm"}}, n_splits=3, spatial_block_m=500, repeats=1)
    status = dict(zip(table.model, table.status))
    assert status["rf"] == "ok" and status["lightgbm"].startswith("skipped:")


def _load_gate():
    spec = importlib.util.spec_from_file_location("release_gate", ROOT / "tools" / "release_gate.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["release_gate"] = module      # dataclasses resolve their module through sys.modules
    spec.loader.exec_module(module)
    return module


def test_release_gate_report_and_verdict(tmp_path):
    gate = _load_gate()
    results = [gate.CheckResult("a.ruff", "P0", "pass", 1.0, {"exit_code": 0}),
               gate.CheckResult("c.bench", "P1", "skip", 0.0, {}, notes="no network"),
               gate.CheckResult("e.contracts", "P1", "fail", 2.0, {"exit_code": 1}, notes="snapshot differs")]
    report = gate.build_report(results, {"commit": "abc"}, {"python": "3.12"})
    assert report["verdict"] == "FAIL" and report["counts"] == {"pass": 1, "fail": 1, "skip": 1}
    paths = gate.write_report(report, tmp_path)
    assert json.loads(Path(paths["json"]).read_text())["verdict"] == "FAIL"
    md = Path(paths["md"]).read_text()
    assert "| e.contracts | P1 | FAIL |" in md and "snapshot differs" in md
    assert gate.build_report(results[:2], {}, {})["verdict"] == "PASS (with skips)"


def test_pytest_summary_parser():
    gate = _load_gate()
    assert gate.pytest_counts("12 passed, 3 skipped, 1 deselected in 4.2s") == {
        "passed": 12, "skipped": 3, "deselected": 1}
    assert gate.pytest_counts("1 failed, 2 passed, 1 error in 1s")["failed"] == 1


def test_samples_warn_when_a_class_loses_every_polygon(tmp_path):
    """Regression: polygons whose pixels are all nodata in some feature (e.g. NaN phenology of a low-amplitude
    class in `ts-features`) were dropped silently, so a whole class could vanish from training."""
    import geopandas as gpd
    import numpy as np
    import rasterio
    from rasterio.transform import from_origin
    from shapely.geometry import box

    from geocrop.samples import SampleCoverageWarning, extract

    data = np.ones((2, 20, 20), "float32")
    data[1, :, 10:] = np.nan                       # second feature invalid over the east half
    path = tmp_path / "stack.tif"
    with rasterio.open(path, "w", driver="GTiff", width=20, height=20, count=2, dtype="float32", crs="EPSG:32634",
                       transform=from_origin(0, 200, 10, 10), nodata=np.nan) as ds:
        ds.write(data)
        ds.descriptions = ("P1_B04", "P1_PHENO")
    polys = gpd.GeoDataFrame({"class_name": ["maize", "wheat"], "polygon_id": ["a", "b"]},
                             geometry=[box(10, 10, 90, 190), box(110, 10, 190, 190)], crs="EPSG:32634")
    with pytest.warns(SampleCoverageWarning, match=r"WITHOUT any sample: \['wheat'\]"):
        s = extract(path, polys)
    assert set(s.y) == {"maize"}
