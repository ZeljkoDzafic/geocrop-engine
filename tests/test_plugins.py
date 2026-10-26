"""Plugin registries: a classifier, an index and a segmenter added without touching GeoCrop's code."""
from __future__ import annotations

from importlib import metadata
from pathlib import Path

import numpy as np
import pytest
import rasterio

from geocrop import benchmark, estimators, indices, model, plugins, segmenters

EXAMPLE_SRC = Path(__file__).resolve().parents[1] / "examples" / "plugin_example" / "src"


@pytest.fixture
def example(monkeypatch):
    """The example plugin, registered programmatically (as an installed entry point would be)."""
    monkeypatch.syspath_prepend(str(EXAMPLE_SRC))
    import geocrop_plugin_example as ex

    plugins.estimators.register("nearest_centroid", ex.make_nearest_centroid)
    plugins.indices.register("NDTI", ex.NDTI)
    plugins.segmenters.register("grid", ex.GridSegmenter)
    yield ex
    plugins.estimators.unregister("nearest_centroid")
    plugins.indices.unregister("NDTI")
    plugins.segmenters.unregister("grid")


def test_plugin_classifier_trains_and_predicts(example, samples, stack, tmp_path):
    rep = model.train(samples, tmp_path / "m", params={"algorithm": "nearest_centroid"}, n_splits=3,
                      spatial_block_m=500, bootstrap=20)
    assert 0 <= rep["cross_validation"]["macro_f1"] <= 1
    out = model.predict(stack, tmp_path / "m" / "model.joblib", tmp_path / "pred", block_size=64)
    with rasterio.open(out["probabilities"]) as ds:
        assert np.allclose(ds.read().sum(axis=0), 1, atol=1e-4)
    assert "nearest_centroid" in estimators.available_algorithms()
    assert isinstance(estimators.make_estimator({"algorithm": "nearest_centroid"}, 0), example.NearestCentroidProba)


def test_plugin_index_is_usable_like_a_builtin(example, demo):
    from geocrop import features
    bands = {"B11": np.array([0.3], "float32"), "B12": np.array([0.1], "float32")}
    assert np.isclose(indices.get("NDTI").compute(bands)[0], 0.5)
    assert "NDTI" in {row["name"] for row in indices.table()}
    _, planned, _ = features.plan(demo["composites"], s2_indices=("NDVI", "NDTI"))
    names = [f.name for f in planned]
    assert any(n.endswith("_NDTI") for n in names)


def test_plugin_segmenter_in_benchmark(example, demo, tmp_path):
    raster = Path(demo["composites"]) / "P3" / "B08.tif"
    table = benchmark.segmentation(raster, demo["parcels"], tmp_path, tile_size=96, segments_per_km2=40,
                                   methods={"grid_16": {"segmenter": "grid", "cell_px": 16}})
    assert set(table.method) == {"obia_tiles", "obia_seam_merged", "grid_16"}
    assert (table.status == "ok").all()


def test_builtin_segmenters_follow_the_protocol(demo, tmp_path):
    from geocrop import segmentation
    assert {"obia", "unet_watershed"} <= set(segmenters.available())
    obia = segmenters.get("obia")
    assert isinstance(obia, segmenters.Segmenter) and obia.name == "obia"
    raster = Path(demo["composites"]) / "P3" / "B08.tif"
    via_registry = obia.segment(raster, tmp_path / "a.gpkg", tile_size=64, segments_per_km2=60)
    direct = segmentation.obia(raster, tmp_path / "b.gpkg", tile_size=64, segments_per_km2=60)
    assert via_registry["segments"] == direct["segments"]
    with pytest.raises(ValueError, match="model="):
        segmenters.get("unet_watershed").segment(raster, tmp_path / "u.gpkg")


def _ep(name, value, group="geocrop.estimators"):
    return metadata.EntryPoint(name, value, group)


def test_entry_point_discovery_is_lazy_and_tolerates_broken_plugins(monkeypatch):
    monkeypatch.syspath_prepend(str(EXAMPLE_SRC))
    calls = []

    def fake(group):
        calls.append(group)
        return [_ep("nearest_centroid", "geocrop_plugin_example:make_nearest_centroid"),
                _ep("broken", "geocrop_no_such_module:factory"),
                _ep("not_callable", "geocrop_plugin_example:__doc__"),
                _ep("rf", "geocrop_plugin_example:make_nearest_centroid")]

    monkeypatch.setattr(plugins, "_entry_points", fake)
    registry = plugins.Registry("estimator", "geocrop.estimators", plugins._callable("estimator factory"))
    registry.register("builtin_one", lambda p, s: None, source=plugins.BUILTIN)
    registry.get("builtin_one")
    assert calls == []                                     # built-ins never trigger discovery
    with pytest.warns(plugins.PluginWarning) as record:
        assert registry.source("nearest_centroid").startswith("entry point")
    messages = " ".join(str(w.message) for w in record)
    assert "broken" in messages and "ModuleNotFoundError" in messages and "not_callable" in messages
    assert "broken" not in registry.names() and calls == ["geocrop.estimators"]


def test_entry_points_cannot_shadow_builtins(monkeypatch):
    monkeypatch.setattr(plugins, "_entry_points", lambda group: [_ep("rf", "builtins:dict")])
    monkeypatch.setattr(plugins.estimators, "_discovered", False)
    with pytest.warns(plugins.PluginWarning, match="already registered"):
        plugins.estimators.discover()
    assert plugins.estimators.source("rf") == plugins.BUILTIN


def test_make_estimator_finds_installed_plugin(monkeypatch):
    monkeypatch.syspath_prepend(str(EXAMPLE_SRC))
    monkeypatch.setattr(plugins, "_entry_points",
                        lambda group: [_ep("nearest_centroid", "geocrop_plugin_example:make_nearest_centroid")])
    monkeypatch.setattr(plugins.estimators, "_discovered", False)
    try:
        est = estimators.make_estimator({"algorithm": "nearest_centroid"}, 0)
        X = np.r_[np.zeros((5, 2)), np.ones((5, 2))]
        assert est.fit(X, ["a"] * 5 + ["b"] * 5).predict_proba(X).shape == (10, 2)
    finally:
        plugins.estimators.unregister("nearest_centroid")


def test_contract_violations_are_clear():
    with pytest.raises(ValueError, match="Unknown algorithm 'nope'.*rf"):
        estimators.make_estimator({"algorithm": "nope"}, 0)
    with pytest.raises(TypeError, match="must be callable"):
        plugins.estimators.register("bad", object())
    plugins.estimators.register("no_proba", lambda params, seed: object())
    try:
        with pytest.raises(TypeError, match="predict_proba"):
            estimators.make_estimator({"algorithm": "no_proba"}, 0)
    finally:
        plugins.estimators.unregister("no_proba")
    with pytest.raises(ValueError, match="already registered"):
        plugins.estimators.register("rf", lambda params, seed: None)
    with pytest.raises(ValueError, match="names must match"):
        plugins.indices.register("OTHER", indices.get("NDVI"))
    with pytest.raises(TypeError, match="segment"):
        plugins.segmenters.register("bad", object())
    with pytest.raises(KeyError, match="Unknown index 'NOPE'"):
        indices.get("NOPE")


def test_lazy_builtins_for_exporters_and_validators(demo):
    from geocrop import export, provenance
    assert plugins.exporters.get("cog") is export.to_cog
    assert plugins.validators.get("raster") is provenance.check_raster
    report = plugins.validators.get("raster")(Path(demo["composites"]) / "P1" / "B02.tif")
    assert isinstance(report["problems"], list)



def test_installed_distribution_is_discovered(tmp_path, monkeypatch):
    """Real importlib.metadata discovery: a dist-info with the example's entry points on sys.path."""
    dist = tmp_path / "geocrop_plugin_example-0.1.0.dist-info"
    dist.mkdir()
    (dist / "METADATA").write_text("Metadata-Version: 2.1\nName: geocrop-plugin-example\nVersion: 0.1.0\n")
    (dist / "entry_points.txt").write_text(
        "[geocrop.estimators]\nnearest_centroid = geocrop_plugin_example:make_nearest_centroid\n\n"
        "[geocrop.indices]\nNDTI = geocrop_plugin_example:NDTI\n\n"
        "[geocrop.segmenters]\ngrid = geocrop_plugin_example:GridSegmenter\n")
    monkeypatch.syspath_prepend(str(EXAMPLE_SRC))
    monkeypatch.syspath_prepend(str(tmp_path))
    for registry in (plugins.estimators, plugins.indices, plugins.segmenters):
        monkeypatch.setattr(registry, "_discovered", False)
    try:
        assert hasattr(estimators.make_estimator({"algorithm": "nearest_centroid"}, 1), "predict_proba")
        assert indices.get("NDTI").inputs == ("B11", "B12")
        assert segmenters.get("grid").name == "grid"
        assert plugins.describe()["segmenters"]["grid"].startswith("entry point")
    finally:
        plugins.estimators.unregister("nearest_centroid")
        plugins.indices.unregister("NDTI")
        plugins.segmenters.unregister("grid")
