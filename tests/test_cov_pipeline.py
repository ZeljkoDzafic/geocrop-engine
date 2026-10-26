"""Pipeline guard rails that fire before any heavy computation: corrupt state, missing inputs, bad options."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from geocrop import config, pipeline


def _cfg(tmp_path: Path, **sections) -> dict:
    root = tmp_path / "study"
    paths = {"s2_scenes": root / "s2", "s1_scenes": root / "s1", "composites": root / "comp",
             "features": root / "feat" / "stack.tif", "training": root / "ref" / "training.gpkg",
             "parcels": root / "ref" / "parcels.gpkg", "observations": root / "ref" / "obs.gpkg",
             "output": root / "run"}
    return config.load(overrides={"paths": {k: str(v) for k, v in paths.items()}, **sections})


def _runner(cfg, messages=None):
    return pipeline.Runner(cfg, log=(messages.append if messages is not None else (lambda *_: None)))


def test_runner_rejects_corrupt_stage_state(tmp_path):
    cfg = _cfg(tmp_path)
    out = Path(cfg["paths"]["output"])
    out.mkdir(parents=True)
    (out / "stages.json").write_text(json.dumps(["not", "a", "dict"]))
    with pytest.raises(ValueError, match="Invalid stage state"):
        _runner(cfg)
    (out / "stages.json").write_text("{truncated")
    with pytest.raises(json.JSONDecodeError):
        _runner(cfg)


def test_optional_stages_skip_without_inputs(tmp_path):
    messages = []
    cfg = _cfg(tmp_path, cbm={"enabled": True})
    cfg["parcel_classify"]["enabled"] = True       # set after load: validation would require a label field
    runner = _runner(cfg, messages)
    assert runner.parcels() is None and runner.validate() is None and runner.cbm() is None
    assert runner.parcel_classify() is None
    assert {m.split(" (")[0] for m in messages} == {"[skip] parcels", "[skip] validate", "[skip] cbm",
                                                    "[skip] parcel_classify"}
    disabled = _runner(_cfg(tmp_path / "b"))
    assert disabled.cbm() is None and disabled.parcel_classify() is None and disabled.embeddings() is None


def test_cbm_needs_a_season(tmp_path):
    cfg = _cfg(tmp_path, cbm={"enabled": True})
    parcels = Path(cfg["paths"]["parcels"])
    parcels.parent.mkdir(parents=True)
    parcels.write_text("placeholder")
    with pytest.raises(config.ConfigError, match="timeseries.start"):
        _runner(cfg).cbm()


def test_composite_needs_windows_and_scenes(tmp_path):
    cfg = _cfg(tmp_path)
    cfg["windows"] = {}
    with pytest.raises(config.ConfigError, match="windows"):
        _runner(cfg).composite()
    cfg = _cfg(tmp_path / "b", windows={"P1": {"start": "2023-04-01", "end": "2023-04-30"}})
    with pytest.raises(RuntimeError, match="No scenes for windows"):
        _runner(cfg).composite()


def test_grid_reference_resolution(tmp_path):
    cfg = _cfg(tmp_path)
    runner = _runner(cfg)
    with pytest.raises(config.ConfigError, match="study grid"):
        runner._grid_reference("embeddings")
    b02 = Path(cfg["paths"]["composites"]) / "P1" / "B02.tif"
    b02.parent.mkdir(parents=True)
    b02.write_bytes(b"")
    assert runner._grid_reference("x") == b02
    cfg["grid"]["reference"] = str(tmp_path / "grid.tif")
    assert runner._grid_reference("x") == tmp_path / "grid.tif"


@pytest.mark.parametrize("strategy,field,match", [("region", "region_field", "region_field"),
                                                  ("year", "year_field", "year_field")])
def test_cv_strategy_needs_metadata(tmp_path, strategy, field, match):
    runner = _runner(_cfg(tmp_path))
    runner.cfg["cv"].update({"strategy": strategy, "region_field": None, "year_field": None, "year": None})
    with pytest.raises(config.ConfigError, match=match):
        runner._cv_strategy()
    runner.cfg["cv"].update({"region_field": "region", "year": 2023})
    assert runner._cv_strategy() == strategy


def test_run_rejects_unknown_stages(tmp_path):
    import yaml
    cfg = _cfg(tmp_path)
    path = tmp_path / "study.yaml"
    path.write_text(yaml.safe_dump({"paths": cfg["paths"]}))
    with pytest.raises(ValueError, match="Unknown stages"):
        pipeline.run(path, stages=["composite", "teleport"], log=lambda *_: None)


def test_file_identity_without_hash_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "_HASHER", None)
    small = tmp_path / "a.json"
    small.write_text("{}")
    big = tmp_path / "b.tif"
    big.write_bytes(b"0" * 10)
    assert "sha256" in pipeline._file_identity(small)
    ident = pipeline._file_identity(big)
    assert set(ident) == {"bytes", "mtime_ns"} and ident["bytes"] == 10
