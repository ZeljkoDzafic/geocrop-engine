"""`geocrop run` wiring of the optional capabilities (cv strategies, smoothing, embeddings, event markers and
marker rules, parcel classifier, model.params, feature-manifest schema versions).

Every option is off by default; enabling one re-runs only the stages that depend on it, and a configuration
that enables none keeps the stage fingerprints of GeoCrop before the options were wired in (synthetic data).
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import rasterio
import yaml

from geocrop import config, contracts, features, model, pipeline, plugins

EXAMPLE_SRC = Path(__file__).resolve().parents[1] / "examples" / "plugin_example" / "src"
RUN_STAGES = [s for s in pipeline.STAGES if s not in ("composite", "validate")]

# Inner (configuration + input) fingerprints of the stages for a default configuration with relative, absent
# paths, computed with GeoCrop main at c091962 before the optional stages were wired in. They must not change
# when no option is enabled; the engine-code part of each stage fingerprint changes with every release anyway.
MAIN_FINGERPRINTS = {
    "default": {"features": "8eea68506ac0bef4", "samples": "bf78fb7c492afd1f", "train": "a5d23e8b9e64fdc3",
                "predict": "95327f9ab42beefe", "experiments": "c0fc6adef2fd61bd"},
    "timeseries": {"features": "307464e7bab1804d", "samples": "bf78fb7c492afd1f", "train": "5864b208fa32c226",
                   "predict": "95327f9ab42beefe", "experiments": "25b757c3e0199e47"},
}
MAIN_DEPENDS = {"composite": (), "features": ("composite",), "samples": ("features",), "train": ("samples",),
                "predict": ("train", "features"), "cbm": (), "parcels": ("predict", "train", "cbm"),
                "validate": ("predict",), "experiments": ("samples", "features")}


def _inner_fingerprints(overrides):
    cfg = config.load(overrides={"paths": {"output": "run"}, **overrides})
    runner = pipeline.Runner(cfg, log=lambda *_: None)
    runner.stage = lambda name, fp, outputs, fn: fp
    return runner, {s: getattr(runner, s)() for s in MAIN_FINGERPRINTS["default"]}


def test_default_fingerprints_unchanged_from_main(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner, fps = _inner_fingerprints({})
    assert fps == MAIN_FINGERPRINTS["default"]
    _, fps = _inner_fingerprints({"timeseries": {"enabled": True, "start": "2023-03-01", "end": "2023-10-31"},
                                  "model": {"holdout_fraction": 0.2, "n_estimators": 50}})
    assert fps == MAIN_FINGERPRINTS["timeseries"]
    assert {s: runner.depends(s) for s in MAIN_DEPENDS} == MAIN_DEPENDS
    assert runner.embeddings() is None and runner.parcel_classify() is None
    assert config.fingerprint(config.load()) == "09ff103f5880447d"          # main c091962


def test_new_options_change_fingerprints_only_when_enabled(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _, base = _inner_fingerprints({})
    for overrides, changed in (
            ({"predict": {"smoothing": {"enabled": True}}}, {"predict"}),
            ({"cv": {"strategy": "region", "region_field": "region"}}, {"samples", "train", "experiments"}),
            ({"model": {"params": {"min_samples_split": 4}}}, {"train", "experiments"}),
            ({"features": {"embeddings": {"source": "alphaearth", "year": 2023, "attach": False}}}, set())):
        _, fps = _inner_fingerprints(overrides)
        assert {s for s in fps if fps[s] != base[s]} == changed, overrides


# --------------------------------------------------------------------------- end-to-end on synthetic scenes

@pytest.fixture(scope="module")
def study(tmp_path_factory):
    from geocrop.demo import generate_scenes
    root = tmp_path_factory.mktemp("options")
    d = generate_scenes(root / "data")
    training = gpd.read_file(d["training"])
    x = training.geometry.centroid.x
    training["region"] = np.where(x > x.median(), "east", "west")
    training.to_file(root / "training_regions.gpkg")
    cfg = {"windows": {"P1": {"start": "2023-04-01", "end": "2023-05-31"},
                       "P2": {"start": "2023-06-01", "end": "2023-07-15"},
                       "P3": {"start": "2023-07-16", "end": "2023-08-31"},
                       "P4": {"start": "2023-09-01", "end": "2023-10-15"}},
           "grid": {"reference": d["reference"]},
           "paths": {"composites": d["composites"], "features": str(root / "stack.tif"),
                     "s2_scenes": str(Path(d["scene_root"]) / "s2"), "s1_scenes": str(Path(d["scene_root"]) / "s1"),
                     "training": str(root / "training_regions.gpkg"), "parcels": d["parcels"],
                     "observations": d["observations"]},
           "timeseries": {"start": "2023-03-15", "end": "2023-10-31"},
           "cbm": {"enabled": True},
           "model": {"n_estimators": 15, "n_jobs": 1, "cv_folds": 2, "spatial_block_m": 500,
                     "max_samples_per_polygon": 20},
           "experiments": {"S2": {"groups": ["s2_index"]}}}
    first = copy.deepcopy(cfg)
    first["paths"]["output"] = str(root / "features_run")
    path = root / "features.yaml"
    path.write_text(yaml.safe_dump(first))
    pipeline.run(path, ["features"], log=lambda *_: None)
    return {"root": root, "cfg": cfg, "data": d}


def _merge(base: dict, override: dict) -> dict:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge(base[key], value)
        else:
            base[key] = value
    return base


def _run(study, name, options=None, stages=RUN_STAGES):
    cfg = _merge(copy.deepcopy(study["cfg"]), options or {})
    cfg["paths"].setdefault("output", str(study["root"] / name))
    path = study["root"] / f"{name}.yaml"
    path.write_text(yaml.safe_dump(cfg))
    logs = []
    result = pipeline.run(path, stages, log=logs.append)
    ran = {line.split()[-1] for line in logs if line.startswith("[run ]")}
    return result, ran, Path(cfg["paths"]["output"])


def _enable(study, name, options, stages=RUN_STAGES):
    """Run the default configuration, then the same output with `options`: returns (result, re-run stages)."""
    _, first, out = _run(study, name, stages=stages)
    assert {"samples", "train", "predict"} <= first
    result, ran, _ = _run(study, name, options, stages)
    _, again, _ = _run(study, name, options, stages)
    assert again == set(), f"{name}: an unchanged rerun must skip every stage, re-ran {again}"
    return result, ran, out


def test_default_run_has_no_optional_products_and_needs_no_adapters(study, monkeypatch):
    monkeypatch.setitem(sys.modules, "geocrop.embedding_sources", None)    # importing it would fail
    result, ran, out = _run(study, "default")
    assert result["embeddings"] is None and result["parcel_classify"] is None
    state = json.loads((out / "stages.json").read_text())
    assert "embeddings" not in state and "parcel_classify" not in state
    for absent in ("model/transfer_report.json", "model/transfer_per_group.csv",
                   "prediction/probabilities_unsmoothed.tif", "cbm/event_markers.csv", "cbm/s1_signatures.csv",
                   "parcel_classify"):
        assert not (out / absent).exists(), absent
    assert "traffic_light_reasons" not in gpd.read_file(out / "parcels" / "parcels.gpkg")
    body = json.loads(features.manifest_path(study["cfg"]["paths"]["features"]).read_text())
    assert body[contracts.KEY] == contracts.SCHEMA_VERSIONS["feature_manifest"]


def test_transfer_validation_strategy(study):
    result, ran, out = _enable(study, "transfer", {"cv": {"strategy": "region", "region_field": "region"}})
    assert "features" not in ran and {"samples", "train", "experiments"} <= ran and "cbm" not in ran
    report = json.loads((out / "model" / "transfer_report.json").read_text())
    assert report["cv_strategy"] == "region" and report["n_groups"] == 2
    assert {"in_domain", "out_of_domain", "transfer_gap", "per_group"} <= set(report)
    assert (out / "model" / "transfer_per_group.csv").exists()
    assert "transfer_strategy" in (out / "experiments" / "ablation.csv").read_text()
    assert result["train"]["transfer_gap_macro_f1"] is not None


def test_transfer_strategy_without_group_field_fails_clearly(study):
    with pytest.raises(config.ConfigError, match="region_field"):
        _run(study, "transfer_bad", {"cv": {"strategy": "region"}}, ["samples", "train"])


def test_probability_smoothing(study):
    _, ran, out = _enable(study, "smoothing", {"predict": {"smoothing": {"enabled": True, "window": 5}}})
    assert ran == {"predict", "parcels"}
    manifest = json.loads((out / "prediction" / "prediction_manifest.json").read_text())
    assert manifest["smoothing"]["window"] == 5
    assert (out / "prediction" / "probabilities_unsmoothed.tif").exists()
    _, ran, _ = _run(study, "smoothing")                                     # off again: raw copy removed
    assert "predict" in ran and not (out / "prediction" / "probabilities_unsmoothed.tif").exists()


def test_event_markers_and_marker_rules(study):
    _, ran, out = _enable(study, "events", {"cbm": {"event_markers": True}})
    assert ran == {"cbm", "parcels"}
    events = (out / "cbm" / "event_markers.csv").read_text()
    assert "harvest_evidence" in events and (out / "cbm" / "s1_signatures.csv").exists()
    assert "traffic_light_reasons" not in gpd.read_file(out / "parcels" / "parcels.gpkg")   # rules still off
    _, ran, _ = _run(study, "events", {"cbm": {"event_markers": True, "rules": {
        "enabled": True, "harvest_expected_classes": ["maize", "soybean", "sunflower"]}}})
    assert ran == {"cbm", "parcels"}
    parcels = gpd.read_file(out / "parcels" / "parcels.gpkg")
    assert "traffic_light_reasons" in parcels and "harvest_evidence" in parcels
    summary = json.loads((out / "parcels" / "parcels.summary.json").read_text())
    assert sum(summary["traffic_light_counts"].values()) == len(parcels)


def test_temporal_support_routing(study):
    _, ran, out = _enable(study, "temporal", {"cbm": {"temporal_support": {"enabled": True,
                                                                          "min_clear_observations": 40}}})
    assert ran == {"cbm", "parcels"}
    support = (out / "cbm" / "temporal_support.csv").read_text()
    assert "tq_clear_observations" in support
    parcels = gpd.read_file(out / "parcels" / "parcels.gpkg")
    assert "tq_clear_observations" in parcels and "traffic_light_reasons" in parcels
    assert parcels.traffic_light_reasons.fillna("").str.contains("LOW_OBS").any()


def test_temporal_support_raster_changes_only_features(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ts = {"enabled": True, "start": "2023-03-01", "end": "2023-10-31"}
    _, base = _inner_fingerprints({"timeseries": ts})
    _, fps = _inner_fingerprints({"timeseries": {**ts, "temporal_support": True}})
    assert {s for s in fps if fps[s] != base[s]} == {"features"}


@pytest.mark.parametrize("decision", ["additional", "replace"])
def test_parcel_classifier(study, decision):
    options = {"parcel_classify": {"enabled": True, "label_field": "true_class", "decision": decision,
                                   "params": {"n_estimators": 30, "n_jobs": 1}},
               "cbm": {"rules": {"enabled": True}}}
    _, ran, out = _enable(study, f"pclass_{decision}", options)
    assert "parcel_classify" in ran and "train" not in ran
    pc = gpd.read_file(out / "parcel_classify" / "parcels.gpkg")
    assert set(pc.strategy) == {"parcel_classifier"} and "traffic_light_reasons" in pc
    assert (out / "parcel_classify" / "model" / "model.joblib").exists()
    parcels = gpd.read_file(out / "parcels" / "parcels.gpkg")
    assert set(parcels.strategy) == ({"parcel_classifier"} if decision == "replace" else {"mean_probability"})


def test_parcel_classifier_config_is_validated():
    with pytest.raises(config.ConfigError, match="cbm.enabled"):
        config.load(overrides={"parcel_classify": {"enabled": True, "label_field": "verified"}})
    with pytest.raises(config.ConfigError, match="declaration"):
        config.load(overrides={"cbm": {"enabled": True},
                               "parcel_classify": {"enabled": True, "label_field": "declared_crop"}})
    with pytest.raises(config.ConfigError, match="decision"):
        config.load(overrides={"parcel_classify": {"decision": "both"}})


def test_plugin_estimator_receives_model_params(study, monkeypatch):
    monkeypatch.syspath_prepend(str(EXAMPLE_SRC))
    import geocrop_plugin_example as ex
    plugins.estimators.register("nearest_centroid", ex.make_nearest_centroid, replace=True)
    try:
        _, ran, out = _enable(study, "plugin", {"model": {"algorithm": "nearest_centroid",
                                                          "params": {"temperature": 2.5}}},
                              ["samples", "train", "predict"])
        assert ran == {"train", "predict"}
        from joblib import load
        assert load(out / "model" / "model.joblib")["model"].temperature == 2.5
        with pytest.raises(ValueError, match="not a parameter"):
            model.config_params({"algorithm": "nearest_centroid", "params": {"n_neighbours": 3}})
    finally:
        plugins.estimators.unregister("nearest_centroid")


def test_model_params_are_validated():
    assert model.config_params({"algorithm": "rf", "n_estimators": 10, "max_depth": None}) == \
        {"algorithm": "rf", "n_estimators": 10}
    assert model.config_params({"algorithm": "rf", "max_depth": None}, keep_none=True) == \
        {"algorithm": "rf", "max_depth": None}
    assert model.config_params({"algorithm": "rf", "params": {"min_samples_split": 4}})["min_samples_split"] == 4
    with pytest.raises(ValueError, match="not a parameter"):
        model.config_params({"algorithm": "rf", "params": {"min_sample_split": 4}})
    with pytest.raises(config.ConfigError, match="model.algorithm"):
        config.load(overrides={"model": {"params": {"algorithm": "rf"}}})
    with pytest.raises(config.ConfigError, match="seed"):
        config.load(overrides={"model": {"params": {"random_state": 1}}})


def _fake_extract(calls):
    def extract(source, reference, out_dir, year, monthly=None, **kwargs):
        from geocrop import embeddings as emb
        calls.append({"source": source, "monthly": monthly, **kwargs})
        target = emb.output_path(out_dir, source, year)
        with rasterio.open(reference) as ref:
            shape = ref.shape
        rng = np.random.default_rng(0)
        with emb.EmbeddingWriter(target, reference, source, year, dims=4, extra={"tiles": "synthetic"}) as sink:
            sink.write(rng.normal(size=(4, *shape)).astype("float32"))
        return {"source": source, "year": year, "raster": str(target)}
    return extract


@pytest.mark.parametrize("source", ["alphaearth", "presto"])
def test_embeddings_stage_attaches_emb_group(study, monkeypatch, source):
    from geocrop import embedding_sources
    calls = []
    monkeypatch.setattr(embedding_sources, "extract", _fake_extract(calls))
    stages = ["embeddings", *RUN_STAGES]
    name = f"emb_{source}"
    base = {"paths": {"features": str(study["root"] / name / "stack.tif")}}
    _run(study, name, base, stages)
    opts = {"source": source, "year": 2023}
    if source == "presto":
        opts.update(start="2023-06", months=2)
    _, ran, out = _run(study, name, _merge(base, {"features": {"embeddings": opts}}), stages)
    assert {"embeddings", "features", "samples", "train"} <= ran and len(calls) == 1
    stack = study["root"] / name / "stack.tif"
    groups = {f.group for f in features.load_manifest(stack)}
    assert "emb" in groups and "s2_index" in groups
    if source == "presto":   # monthly composites built locally from the scene folders
        assert sorted(p.name for p in (stack.parent / "embeddings" / "monthly").iterdir()) == ["2023-06", "2023-07"]
    _, again, _ = _run(study, name, _merge(base, {"features": {"embeddings": opts}}), stages)
    assert again == set() and len(calls) == 1


def test_feature_manifest_schema_version_is_checked(tmp_path):
    stack = tmp_path / "s.tif"
    features.write_manifest(stack, {"features": []})
    assert features.read_manifest(stack)[contracts.KEY] == 1
    features.manifest_path(stack).write_text(json.dumps({"features": []}))       # unversioned: accepted
    assert features.load_manifest(stack) == []
    features.manifest_path(stack).write_text(json.dumps({"features": [], contracts.KEY: 99}))
    with pytest.raises(contracts.SchemaVersionError, match="feature_manifest"):
        features.load_manifest(stack)


def test_cli_algorithm_choices_include_plugins(monkeypatch):
    from geocrop import cli, estimators
    assert cli.BUILTIN_ALGORITHMS == estimators.ALGORITHMS
    plugins.estimators.register("my_forest", lambda params, seed: None)
    try:
        args = cli.build_parser().parse_args(["benchmark", "--samples", "s.npz", "--out", "o",
                                              "--algorithms", "rf", "my_forest"])
        assert args.algorithms == ["rf", "my_forest"]
        args = cli.build_parser().parse_args(["train", "--samples", "s.npz", "--out", "o",
                                              "--algorithm", "my_forest"])
        assert args.algorithm == "my_forest"
    finally:
        plugins.estimators.unregister("my_forest")
