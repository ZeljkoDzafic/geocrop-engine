"""Exercise study audit/report exits and routing of the actual training/ablation CV policy."""
import json

import pytest
import yaml

from geocrop import config, pipeline
from geocrop.cli import main


@pytest.mark.parametrize("value", [-1, float("nan"), float("inf"), True, "100", None])
def test_invalid_buffer_config(value):
    with pytest.raises(config.ConfigError, match="spatial_buffer_m"):
        config.load(overrides={"model": {"spatial_buffer_m": value}})


def test_audit_cli_persists_failed_report(tmp_path, capsys):
    out = tmp_path / "reports" / "audit.json"
    with pytest.raises(SystemExit) as exc:
        main(["audit", "--training", str(tmp_path / "missing.gpkg"), "--out", str(out)])
    assert exc.value.code == 2
    report = json.loads(out.read_text())
    assert report == json.loads(capsys.readouterr().out)
    assert not report["ok"] and report["summary"]["errors"] > 0


def test_audit_cli_preserves_input(tmp_path):
    source = tmp_path / "source.json"
    source.write_text("user input")
    with pytest.raises(ValueError, match="differ from every input"):
        main(["audit", "--training", str(source), "--out", str(source)])
    assert source.read_text() == "user input"


def test_audit_cli_preserves_hardlinked_input(tmp_path):
    source = tmp_path / "source.json"
    source.write_text("user input")
    alias = tmp_path / "report.json"
    alias.hardlink_to(source)
    with pytest.raises(ValueError, match="differ from every input"):
        main(["audit", "--training", str(source), "--out", str(alias)])
    assert source.read_text() == "user input"


def test_audit_cli_real_layers(demo, stack, tmp_path, capsys):
    report_path = tmp_path / "audit.json"
    main(["audit", "--training", demo["training"], "--parcels", demo["parcels"],
          "--observations", demo["observations"], "--features", str(stack), "--out", str(report_path)])
    report = json.loads(capsys.readouterr().out)
    assert report["ok"]
    assert report["inputs"]["training"]["features"] > 0
    assert report["inputs"]["stack"]["bands"] > 0
    assert json.loads(report_path.read_text()) == report


@pytest.mark.parametrize("calibration_mode", ["oof", "holdout"])
def test_pipeline_and_ablation_cli_share_configured_policy(samples, stack, tmp_path, capsys, calibration_mode):
    out = tmp_path / "run"
    out.mkdir()
    samples.save(out / "samples.npz")
    raw = {"seed": 17, "paths": {"output": str(out)},
           "model": {"algorithm": "extra_trees", "n_estimators": 5, "n_jobs": 1,
                     "cv_folds": 3, "spatial_block_m": 500, "spatial_buffer_m": 10,
                     "calibration_mode": calibration_mode, "calibration_fraction": .25},
           "experiments": {"first": {"features": [samples.features[0]]}}}
    cfg_path = tmp_path / "study.yaml"
    cfg_path.write_text(yaml.safe_dump(raw))
    runner = pipeline.Runner(config.load(cfg_path), log=lambda *_: None)
    runner.train()
    train = json.loads((out / "model" / "train_report.json").read_text())
    main(["ablation", "--samples", str(out / "samples.npz"), "--features", str(stack),
          "--out", str(out / "ablation"), "--config", str(cfg_path)])
    capsys.readouterr()
    exp = json.loads((out / "ablation" / "ablation.json").read_text())
    assert exp["params"]["algorithm"] == "extra_trees"
    assert exp["seed"] == 17 and exp["n_splits"] == 3
    assert exp["spatial_block_m"] == 500 and exp["spatial_buffer_m"] == 10
    assert exp["n_samples"] == train["samples"]
    assert exp["calibration_split"] == train.get("calibration_split")
    for actual, expected in zip(exp["experiments"]["first"]["cv"]["folds"], train["cross_validation"]["folds"]):
        for key in ("n_train", "n_test", "buffer_excluded_polygons", "min_train_test_distance_m"):
            assert actual[key] == expected[key]
