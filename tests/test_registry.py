"""Local model and benchmark registry (geocrop.registry) on synthetic demo data."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from geocrop import benchmark, experiments, model, registry
from geocrop.cli import main


@pytest.fixture(scope="module")
def trained(samples, tmp_path_factory):
    """Two models on the same samples and folds (different settings) and one on different folds."""
    root = tmp_path_factory.mktemp("trained")
    samples.save(root / "samples.npz")
    out = {}
    for name, params, block in (("a", {"n_estimators": 10, "n_jobs": 1}, 500),
                                ("b", {"n_estimators": 25, "max_features": 0.5, "n_jobs": 1}, 500),
                                ("c", {"n_estimators": 10, "n_jobs": 1}, 300)):
        model.train(samples, root / name, params=params, n_splits=3, spatial_block_m=block, bootstrap=20,
                    holdout_fraction=0.25 if name != "c" else 0.0)
        out[name] = root / name / "model.joblib"
    out["samples"] = root / "samples.npz"
    return out


def test_register_list_show_and_immutability(trained, stack, tmp_path):
    reg = tmp_path / "reg"
    e = registry.register(trained["a"], "crops", root=reg, tags=["demo"], samples=trained["samples"], features=stack,
                          licences={"presto": "MIT"}, description="synthetic")
    assert e["version"] == 1 and e["artifact"]["sha256"] == registry.sha256(trained["a"])
    assert e["fingerprints"]["verified"], e["fingerprints"]
    assert e["fingerprints"]["cv"] and e["fingerprints"]["holdout"]
    assert e["input_contract"]["crs"] and e["input_contract"]["resolution"] == [10.0, 10.0]
    assert e["feature_manifest"]["groups"] and e["feature_manifest"]["features"]
    assert e["training_data"]["class_counts"] and e["validation"]["cv"]["macro_f1_ci"]
    assert e["validation"]["holdout"]["macro_f1"] is not None and e["validation"]["conformal"]["coverage_check"]
    assert {"component": "presto", "licence": "MIT", "attribution": None} in e["licences"]
    assert e["engine"]["version"]
    folder = reg / "models" / "crops" / "1"
    assert (folder / "aoa.json").exists() and (folder / "train_report.json").exists()
    with pytest.raises(FileExistsError):
        registry._write_once(folder / "entry.json", {})
    with pytest.raises(registry.RegistryError, match="already holds"):
        registry.register(trained["a"], "crops", root=reg)
    with pytest.raises(registry.RegistryError, match="Invalid model name"):
        registry.register(trained["a"], "../x", root=reg)
    registry.register(trained["b"], "crops", root=reg, samples=trained["samples"])
    rows = registry.list_models(reg, "model")
    assert [(r["version"], r["stage"]) for r in rows] == [(1, "none"), (2, "none")]
    assert registry.get("registry:crops@latest", reg)["version"] == 2
    with pytest.raises(registry.RegistryError, match="no version in stage 'production'"):
        registry.get("crops", reg)


def test_promote_stage_log_and_verify(trained, tmp_path):
    reg = tmp_path / "reg"
    registry.register(trained["a"], "crops", root=reg)
    registry.register(trained["b"], "crops", root=reg)
    registry.promote("crops@1", "production", reg)
    r = registry.promote("crops@2", "production", reg, note="better")
    assert {"version": 1, "stage": "archived", "reason": "superseded by version 2"} in r["events"]
    assert registry.stages("crops", reg) == {1: "archived", 2: "production"}
    assert registry.get("crops", reg)["version"] == 2
    assert len((reg / "models" / "crops" / "stages.jsonl").read_text().splitlines()) == 3
    assert all(v["ok"] for v in registry.verify(root=reg, include_pretrained=False))
    model_file = reg / "models" / "crops" / "1" / "model.joblib"
    os.chmod(model_file, 0o644)
    model_file.write_bytes(model_file.read_bytes() + b"x")
    bad = registry.verify("crops@1", reg)[0]
    assert not bad["ok"] and "model artefact sha256 mismatch" in bad["problems"]
    with pytest.raises(registry.RegistryError, match="failed verification"):
        registry.promote("crops@1", "staging", reg)
    entry = reg / "models" / "crops" / "2" / "entry.json"
    os.chmod(entry, 0o644)
    entry.write_text(entry.read_text().replace('"algorithm": "rf"', '"algorithm": "xx"'))
    assert "entry.json was modified after registration" in registry.verify("crops@2", reg)[0]["problems"]


def test_compare_only_on_identical_splits(trained, tmp_path):
    reg = tmp_path / "reg"
    for name in ("a", "b", "c"):
        registry.register(trained[name], name, root=reg, samples=trained["samples"])
    r = registry.compare("a@1", "b@1", reg)
    assert set(r["sections"]) == {"cv", "holdout"}
    d = r["sections"]["cv"]["metrics"]["macro_f1"]
    assert d["delta"] == pytest.approx(d["b"] - d["a"])
    assert "| macro_f1 |" in registry.compare_markdown(r)
    with pytest.raises(registry.RegistryError, match="Refusing to compare"):
        registry.compare("a@1", "c@1", reg)              # different spatial blocks -> different folds
    import shutil
    other = tmp_path / "other"
    for name in ("a", "b"):          # model folders without a samples.npz nearby: fingerprints unverifiable
        shutil.copytree(trained[name].parent, tmp_path / "copies" / name)
        e = registry.register(tmp_path / "copies" / name / "model.joblib", "plain", root=other)
        assert not e["fingerprints"]["verified"]
    with pytest.raises(registry.RegistryError, match="unverified"):
        registry.compare("plain@1", "plain@2", other)


def test_fingerprint_verification_detects_wrong_samples(trained, samples, tmp_path):
    other = samples.take(slice(0, len(samples.y) // 2))
    other.save(tmp_path / "half.npz")
    e = registry.register(trained["a"], "x", root=tmp_path / "reg", samples=tmp_path / "half.npz")
    assert not e["fingerprints"]["verified"]


def test_predict_with_registry_reference_and_contract(trained, stack, tmp_path, capsys):
    import rasterio

    reg = tmp_path / "reg"
    registry.register(trained["a"], "crops", root=reg, samples=trained["samples"], features=stack)
    registry.promote("crops@1", "production", reg)
    main(["predict", "--features", str(stack), "--model", "registry:crops@production", "--registry", str(reg),
          "--out", str(tmp_path / "pred")])
    out = json.loads(capsys.readouterr().out)
    assert out["registry"]["version"] == 1
    assert json.loads((tmp_path / "pred" / "prediction_manifest.json").read_text())["registry"]["name"] == "crops"
    # A stack on another grid resolution violates the contract.
    with rasterio.open(stack) as src:
        profile = src.profile
        data = src.read()
        descriptions = src.descriptions
    profile["transform"] = profile["transform"] * profile["transform"].scale(2, 2)
    other = tmp_path / "coarse.tif"
    with rasterio.open(other, "w", **profile) as dst:
        dst.write(data)
        for i, d in enumerate(descriptions, 1):
            dst.set_band_description(i, d)
    with pytest.raises(SystemExit, match="grid resolution"):
        main(["predict", "--features", str(other), "--model", "registry:crops@1", "--registry", str(reg),
              "--out", str(tmp_path / "pred2")])
    with pytest.raises(SystemExit, match="unknown version"):
        main(["predict", "--features", str(stack), "--model", "registry:crops@7", "--registry", str(reg),
              "--out", str(tmp_path / "pred3")])


def test_pretrained_entries_reuse_pins():
    from geocrop.ftw import MODELS
    entries = {e["name"]: e for e in registry.pretrained_entries()}
    m = MODELS["FTW_PRUE_EFNET_B3_CCBY"]
    e = entries["ftw/FTW_PRUE_EFNET_B3_CCBY"]
    assert e["kind"] == "pretrained" and e["artifact"]["sha256"] == m.sha256 and e["licence"] == "CC-BY-4.0"
    assert entries["presto/default"]["licence"] == "MIT"
    assert registry.get("ftw/FTW_PRUE_EFNET_B3_CCBY")["artifact"]["url"] == m.url


def test_benchmark_hooks_and_leaderboard(samples, stack, tmp_path, monkeypatch):
    from geocrop.features import load_manifest

    reg = tmp_path / "reg"
    monkeypatch.setenv("GEOCROP_REGISTRY", str(reg))
    cands = {"rf": {"algorithm": "rf", "n_estimators": 5, "n_jobs": 1},
             "et": {"algorithm": "extra_trees", "n_estimators": 5, "n_jobs": 1}}
    benchmark.run(samples, tmp_path / "b1", cands, n_splits=3, spatial_block_m=500)
    benchmark.models(samples, tmp_path / "b2", candidates={"rf": cands["rf"]}, n_splits=3, spatial_block_m=500,
                     repeats=1)
    experiments.run(samples, load_manifest(stack), tmp_path / "abl", {"s2": {"groups": ["s2_band"]}},
                    params={"n_estimators": 5, "n_jobs": 1}, n_splits=3, spatial_block_m=500)
    benchmark.run(samples, tmp_path / "b3", {"rf": cands["rf"]}, n_splits=3, spatial_block_m=300)
    runs = registry.list_benchmarks(reg)
    assert sorted(r["kind"] for r in runs) == ["ablation", "benchmark", "benchmark", "benchmark-models"]
    assert all(r["task"] == registry.CLASSIFICATION_TASK and r["engine"]["version"] for r in runs)
    groups = registry.leaderboard(registry.CLASSIFICATION_TASK, reg)
    assert len(groups) == 2                                    # block 300 m run is never ranked with the others
    main_group = groups[0]
    assert main_group["runs"] == 3
    assert {r["candidate"] for r in main_group["rows"]} == {"rf", "et", "s2"}
    values = [r["macro_f1"] for r in main_group["rows"]]
    assert values == sorted(values, reverse=True) and [r["rank"] for r in main_group["rows"]] == list(range(1, 5))
    md = registry.leaderboard_markdown(groups)
    assert md.count("## crop-classification") == 2
    csv = registry.leaderboard_csv(groups)
    assert csv.splitlines()[0].startswith("split_fingerprint,metric,rank")
    only = registry.leaderboard(registry.CLASSIFICATION_TASK, reg, split=main_group["split_fingerprint"][:10])
    assert len(only) == 1
    run = registry.get_benchmark(runs[0]["id"][-10:], reg)
    assert run["entry_sha256"] and run["split"]["cv"]
    view = registry.overview(reg)
    json.dumps(view)
    assert len(view["benchmarks"]) == 4 and view["pretrained"]


def test_transfer_records_per_strategy(samples, tmp_path, monkeypatch):
    import numpy as np

    from geocrop.samples import Samples

    reg = tmp_path / "reg"
    monkeypatch.setenv("GEOCROP_REGISTRY", str(reg))
    region = np.where(samples.x < np.median(samples.x), "west", "east")
    s = Samples(samples.X, samples.y, samples.polygon, samples.x, samples.yy, samples.features, samples.crs,
                region=region)
    experiments.transfer(s, tmp_path / "t", params={"n_estimators": 5, "n_jobs": 1}, strategies=["region", "year"],
                         n_splits=3, spatial_block_m=500)
    runs = registry.list_benchmarks(reg)
    assert [r["task"] for r in runs] == ["crop-transfer/region"]      # year metadata missing: skipped, not recorded
    assert runs[0]["primary_metric"] == "out_of_domain_macro_f1"


def test_recording_can_be_disabled_and_never_breaks(samples, tmp_path, monkeypatch):
    monkeypatch.setenv("GEOCROP_REGISTRY", "off")
    monkeypatch.chdir(tmp_path)
    benchmark.run(samples, tmp_path / "b", {"rf": {"n_estimators": 3, "n_jobs": 1}}, n_splits=2, spatial_block_m=500)
    assert not (tmp_path / "registry").exists()
    monkeypatch.setenv("GEOCROP_REGISTRY", str(tmp_path / "reg"))
    with pytest.warns(UserWarning, match="not recorded"):
        assert registry.record_classification("benchmark", tmp_path, [], samples, n_splits=0, seeds=[0],
                                              spatial_block_m=500) is None


def test_cli_models_and_benchmarks(trained, tmp_path, capsys):
    reg = str(tmp_path / "reg")
    main(["models", "register", str(trained["a"]), "--name", "crops", "--registry", reg, "--tags", "demo",
          "--samples", str(trained["samples"]), "--licence", "presto=MIT"])
    assert json.loads(capsys.readouterr().out)["registered"] == "crops@1"
    main(["models", "register", str(trained["b"]), "--name", "crops", "--registry", reg,
          "--samples", str(trained["samples"])])
    capsys.readouterr()
    main(["models", "list", "--registry", reg, "--kind", "model"])
    assert len(json.loads(capsys.readouterr().out)) == 2
    main(["models", "compare", "crops@1", "crops@2", "--registry", reg])
    assert "delta = b - a" in capsys.readouterr().out
    main(["models", "promote", "crops@2", "--stage", "staging", "--registry", reg])
    assert json.loads(capsys.readouterr().out)["stage"] == "staging"
    main(["models", "show", "crops@staging", "--registry", reg])
    assert json.loads(capsys.readouterr().out)["version"] == 2
    main(["models", "verify", "--registry", reg])
    capsys.readouterr()
    with pytest.raises(SystemExit) as exc:
        main(["models", "show", "missing@1", "--registry", reg])
    assert exc.value.code == 2 and "No model 'missing'" in capsys.readouterr().err
    with pytest.raises(SystemExit) as exc:
        main(["benchmarks", "leaderboard", "--task", "crop-classification", "--registry", reg])
    assert exc.value.code == 2


def test_cli_subprocess_leaderboard_csv(samples, tmp_path):
    samples.save(tmp_path / "s.npz")
    env = {**os.environ, "GEOCROP_REGISTRY": str(tmp_path / "reg"),
           "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")}
    subprocess.run([sys.executable, "-m", "geocrop.cli", "benchmark-models", "--samples", str(tmp_path / "s.npz"),
                    "--out", str(tmp_path / "bm"), "--models", "rf", "--folds", "2", "--block-m", "500",
                    "--repeats", "1"], check=True, env=env, capture_output=True)
    out = tmp_path / "lb.csv"
    subprocess.run([sys.executable, "-m", "geocrop.cli", "benchmarks", "leaderboard", "--task",
                    "crop-classification", "--format", "csv", "--out", str(out)], check=True, env=env,
                   capture_output=True)
    lines = out.read_text().splitlines()
    assert len(lines) == 2 and ",macro_f1,1,rf," in lines[1]
