"""Interrupted runs and stale products must never be silently accepted as complete stages."""
import json
import os
from pathlib import Path

import pytest

from geocrop import config, pipeline, s1, s2, scenes


def _cfg(tmp_path, **sections):
    return config.load(overrides={"paths": {"output": str(tmp_path / "run")}, **sections})


def test_failed_forced_rerun_invalidates_previous_success(tmp_path):
    cfg = _cfg(tmp_path)
    target = tmp_path / "product.txt"
    runner = pipeline.Runner(cfg, log=lambda *_: None)

    def good():
        target.write_text("complete")
        return {"complete": True}

    runner.stage("predict", "unchanged", [target], good)

    def interrupted():
        state = json.loads(runner.state_path.read_text())
        assert state["predict"]["status"] == "running"
        target.write_text("partial")
        raise KeyboardInterrupt("worker interrupted")

    with pytest.raises(KeyboardInterrupt):
        pipeline.Runner(cfg, force=True, log=lambda *_: None).stage("predict", "unchanged", [target], interrupted)
    state = json.loads(runner.state_path.read_text())
    assert state["predict"]["status"] == "failed"
    assert state["predict"]["error"]["type"] == "KeyboardInterrupt"
    logs = []
    pipeline.Runner(cfg, log=logs.append).stage("predict", "unchanged", [target], good)
    assert logs == ["[run ] predict"] and target.read_text() == "complete"


def test_deleted_or_changed_sidecar_rebuilds_complete_output_set(tmp_path):
    cfg = _cfg(tmp_path)
    product, sidecar = tmp_path / "classes.tif", tmp_path / "legend.json"
    calls = []

    def produce():
        calls.append(1)
        product.write_bytes(b"raster")
        sidecar.write_text('{"0":"a"}')

    def run():
        pipeline.Runner(cfg, log=lambda *_: None).stage("predict", "same", [product, sidecar], produce)

    run()
    run()
    assert len(calls) == 1
    sidecar.unlink()
    run()
    assert len(calls) == 2
    stat = sidecar.stat()
    sidecar.write_text('{"0":"b"}')
    os.utime(sidecar, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    run()
    assert len(calls) == 3 and json.loads(sidecar.read_text()) == {"0": "a"}
    product.write_bytes(b"corrupt raster")
    run()
    assert len(calls) == 4


@pytest.mark.parametrize("make_empty", [False, True])
def test_success_requires_nonempty_expected_outputs(tmp_path, make_empty):
    runner = pipeline.Runner(_cfg(tmp_path), log=lambda *_: None)
    target = tmp_path / "expected.json"

    def incomplete():
        if make_empty:
            target.touch()
        return {"claimed_success": True}

    with pytest.raises(RuntimeError, match="complete nonempty output set"):
        runner.stage("train", "x", [target], incomplete)
    assert json.loads(runner.state_path.read_text())["train"]["status"] == "failed"


def test_legacy_state_recomputes_instead_of_trusting_single_sentinel(tmp_path):
    runner = pipeline.Runner(_cfg(tmp_path), log=lambda *_: None)
    target = tmp_path / "result.json"
    target.write_text("old")
    runner.state_path.write_text(json.dumps({"train": {"fingerprint": "same", "seconds": 0}}))
    runner = pipeline.Runner(_cfg(tmp_path), log=lambda *_: None)
    runner.stage("train", "same", [target], lambda: target.write_text("new"))
    saved = json.loads(runner.state_path.read_text())["train"]
    assert saved["cache_schema"] == pipeline.CACHE_SCHEMA and saved["status"] == "success"
    assert target.read_text() == "new"


def test_state_replacement_failure_preserves_readable_previous_record(tmp_path, monkeypatch):
    runner = pipeline.Runner(_cfg(tmp_path), log=lambda *_: None)
    runner._record("first", "x", 0, "success", outputs={"product": "identity"})
    before = runner.state_path.read_bytes()

    def fail_replace(*_):
        raise OSError("disk unavailable")

    monkeypatch.setattr(pipeline.os, "replace", fail_replace)
    with pytest.raises(OSError, match="disk unavailable"):
        runner._record("next", "y", 0, "running")
    assert runner.state_path.read_bytes() == before
    assert not list(runner.out.glob(".stages-*.tmp"))


def test_scene_content_and_missing_composite_band_invalidate_resume(tmp_path, monkeypatch):
    scene = tmp_path / "s2" / "scene"
    scene.mkdir(parents=True)
    scenes.write(scene, sensor="S2", datetime="2023-05-01T10:00:00")
    source = scene / "B02.tif"
    source.write_text("first")
    cfg = _cfg(tmp_path, windows={"P1": {"start": "2023-04-01", "end": "2023-05-31"}})
    cfg["paths"].update(s2_scenes=str(scene.parent), s1_scenes=str(tmp_path / "no-s1"),
                        composites=str(tmp_path / "composites"))
    calls = []

    def composite(_scenes, out, **_):
        calls.append(1)
        out.mkdir(parents=True, exist_ok=True)
        for name in (*[f"{b}.tif" for b in s2.S2_BANDS], "valid_count.tif", "composite.json"):
            (out / name).write_text(source.read_text())

    monkeypatch.setattr(s2, "composite", composite)
    pipeline.Runner(cfg, log=lambda *_: None).composite()
    pipeline.Runner(cfg, log=lambda *_: None).composite()
    assert len(calls) == 1
    source.write_text("replacement scene")
    pipeline.Runner(cfg, log=lambda *_: None).composite()
    assert len(calls) == 2
    (tmp_path / "composites" / "P1" / "B12.tif").unlink()
    pipeline.Runner(cfg, log=lambda *_: None).composite()
    assert len(calls) == 3


def test_s1_only_composites_have_required_outputs(tmp_path, monkeypatch):
    scene = tmp_path / "s1" / "scene"
    scene.mkdir(parents=True)
    scenes.write(scene, sensor="S1", datetime="2023-05-01T10:00:00", orbit_direction="ASCENDING")
    (scene / "VV.tif").write_text("source")
    cfg = _cfg(tmp_path, windows={"P1": {"start": "2023-04-01", "end": "2023-05-31"}},
               s1={"separate_orbits": True})
    cfg["paths"].update(s1_scenes=str(scene.parent), s2_scenes=str(tmp_path / "no-s2"),
                        composites=str(tmp_path / "composites"))
    calls = []

    def composite(_scenes, out, **_):
        calls.append(1)
        out.mkdir(parents=True, exist_ok=True)
        for name in ("ASC_VV.tif", "ASC_VH.tif", "composite_s1.json"):
            (out / name).write_text("product")

    monkeypatch.setattr(s1, "composite", composite)
    pipeline.Runner(cfg, log=lambda *_: None).composite()
    pipeline.Runner(cfg, log=lambda *_: None).composite()
    assert len(calls) == 1
    (tmp_path / "composites" / "P1" / "ASC_VH.tif").unlink()
    pipeline.Runner(cfg, log=lambda *_: None).composite()
    assert len(calls) == 2


@pytest.mark.parametrize("sensor,stale_name", [("S1", "ASC_VV.tif"), ("S2", "VV.tif"), ("S1", "B02.tif")])
def test_stale_sensor_or_orbit_products_fail_without_deleting_files(tmp_path, sensor, stale_name):
    scene = tmp_path / "scenes" / "scene"
    scene.mkdir(parents=True)
    scenes.write(scene, sensor=sensor, datetime="2023-05-01T10:00:00", orbit_direction="ASCENDING")
    cfg = _cfg(tmp_path, windows={"P1": {"start": "2023-04-01", "end": "2023-05-31"}})
    cfg["paths"].update(s1_scenes=str(scene.parent if sensor == "S1" else tmp_path / "no-s1"),
                        s2_scenes=str(scene.parent if sensor == "S2" else tmp_path / "no-s2"),
                        composites=str(tmp_path / "composites"))
    window = tmp_path / "composites" / "P1"
    window.mkdir(parents=True)
    stale = window / stale_name
    stale.write_text("preserve until deliberately cleaned")
    unknown = window / "user-notes.txt"
    unknown.write_text("user notes")
    with pytest.raises(RuntimeError, match="new paths.composites directory") as exc:
        pipeline.Runner(cfg, log=lambda *_: None).composite()
    assert stale_name in str(exc.value)
    assert stale.read_text() == "preserve until deliberately cleaned"
    assert unknown.read_text() == "user notes"


def test_calibration_and_validation_dependencies_change_fingerprints(tmp_path):
    cfg = _cfg(tmp_path)
    cfg["paths"].update(observations=str(tmp_path / "observations.gpkg"),
                        training=str(tmp_path / "training.gpkg"))
    Path(cfg["paths"]["observations"]).write_text("observations")
    pred = Path(cfg["paths"]["output"]) / "prediction"
    pred.mkdir(parents=True)
    (pred / "classes.tif").write_text("classes")
    (pred / "probabilities.tif").write_text("probabilities")
    (pred / "class_legend.json").write_text('{"0":"crop"}')

    def fingerprint():
        runner = pipeline.Runner(cfg)
        runner.stage = lambda name, fp, outputs, fn: fp
        return runner.validate()

    before = fingerprint()
    (pred / "probabilities.tif").write_text("different probabilities")
    assert fingerprint() != before
    before = fingerprint()
    cfg["seed"] += 1
    assert fingerprint() != before


def test_experiment_seed_targets_and_manifest_affect_fingerprint(tmp_path):
    cfg = _cfg(tmp_path)
    cfg["paths"]["features"] = str(tmp_path / "stack.tif")
    manifest = tmp_path / "stack.features.json"
    manifest.write_text('{"feature":"first"}')

    def fingerprint():
        runner = pipeline.Runner(cfg)
        runner.stage = lambda name, fp, outputs, fn: fp
        return runner.experiments()

    before = fingerprint()
    cfg["seed"] += 1
    assert fingerprint() != before
    before = fingerprint()
    cfg["classes"]["targets"] = ["soybean"]
    assert fingerprint() != before
    before = fingerprint()
    manifest.write_text('{"feature":"changed"}')
    assert fingerprint() != before


def test_crop_mapping_contents_affect_parcel_fingerprint(tmp_path):
    cfg = _cfg(tmp_path)
    parcel = tmp_path / "parcels.gpkg"
    parcel.write_text("parcels")
    cfg["paths"]["parcels"] = str(parcel)
    mapping = tmp_path / "mapping.csv"
    mapping.write_text("source,class_name\n01,soybean\n")
    cfg["parcels"]["crop_mapping"] = str(mapping)

    def fingerprint():
        runner = pipeline.Runner(cfg)
        runner.stage = lambda name, fp, outputs, fn: fp
        return runner.parcels()

    before = fingerprint()
    mapping.write_text("source,class_name\n01,sunflower\n")
    assert fingerprint() != before
