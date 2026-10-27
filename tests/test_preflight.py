"""Preflight verdicts, disk guard and safe cleanup on synthetic demo data."""
import json
import shutil
from collections import namedtuple
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from shapely.geometry import box

from geocrop import cleanup, cli, config, features, pipeline, preflight, scenes
from geocrop.demo import generate_scenes

WINDOWS = {"P1": {"start": "2023-04-01", "end": "2023-05-31"}, "P2": {"start": "2023-06-01", "end": "2023-07-15"},
           "P3": {"start": "2023-07-16", "end": "2023-08-31"}, "P4": {"start": "2023-09-01", "end": "2023-10-15"}}


@pytest.fixture(scope="module")
def study_src(tmp_path_factory):
    root = tmp_path_factory.mktemp("preflight-study")
    data = generate_scenes(root, s2_every=10, s1_every=20)
    return root, data


@pytest.fixture
def study(study_src, tmp_path):
    """Fresh copy per test (tests modify scenes and reference layers)."""
    root = tmp_path / "study"
    shutil.copytree(study_src[0], root)
    return root


def _cfg(root, **over):
    cfg = {"grid": {"crs": "EPSG:32634", "reference": str(root / "composites" / "P1" / "B02.tif")},
           "windows": WINDOWS,
           "paths": {"s2_scenes": str(root / "scenes" / "s2"), "s1_scenes": str(root / "scenes" / "s1"),
                     "composites": str(root / "run" / "composites"), "features": str(root / "run" / "features" / "stack.tif"),
                     "training": str(root / "reference" / "training.gpkg"), "parcels": str(root / "reference" / "parcels.gpkg"),
                     "observations": str(root / "reference" / "observations.gpkg"), "output": str(root / "run" / "out")},
           "model": {"spatial_block_m": 500, "cv_folds": 3, "n_estimators": 20}}
    for key, value in over.items():
        cfg[key] = {**cfg.get(key, {}), **value} if isinstance(value, dict) and key != "windows" else value
    return config.load(overrides=cfg)


def codes(report, status):
    return {c["code"] for c in report["checks"] if c["status"] == status}


def test_clean_study_is_ready_and_report_is_serialisable(study):
    report = preflight.run(_cfg(study))
    assert report["verdict"] in ("ready", "ready-with-warnings"), codes(report, "error")
    assert not codes(report, "error")
    assert [w["window"] for w in report["windows"]] == list(WINDOWS)
    assert all(w["s2_imported"] > 0 for w in report["windows"])
    assert report["windows"][0]["s1_by_orbit"] == {"ASCENDING": report["windows"][0]["s1_imported"]}
    assert {r["window"] for r in report["coverage"]} == set(WINDOWS)
    res = report["resources"]
    assert res["grid_pixels"] == 192 * 192 and res["features"]["total"] > 0
    assert {"composite", "features", "predict"} <= set(res["disk"]) and res["runtime_hours_rough"]["total"] >= 0
    assert set(report["classes"]["per_class"]) == {"winter_crops", "maize", "soybean", "sunflower", "grassland", "forest"}
    json.loads(json.dumps(report, default=str))
    md = preflight.to_markdown(report)
    assert md.startswith("# GeoCrop preflight: READY") and "ROUGH" in md and "Assumptions" in md


def test_feature_count_matches_feature_plan(study):
    cfg = _cfg(study)
    periods, planned, layout = features.plan(study / "composites", periods=list(WINDOWS))
    assert layout["s2"] and layout["s1_prefixes"] == [""]
    assert preflight.feature_count(cfg, len(periods), True, [""]) == len(planned)
    assert preflight.feature_count(cfg, 1, True, []) == len(cfg["features"]["s2_bands"]) + len(cfg["features"]["s2_indices"])


def test_missing_window_blocks_the_run(study):
    windows = {**WINDOWS, "P5": {"start": "2023-12-01", "end": "2023-12-31"}}
    report = preflight.run(_cfg(study, windows=windows))
    assert report["verdict"] == "blocked"
    assert "scenes.window_empty" in codes(report, "error")
    md = preflight.to_markdown(report)
    assert "BLOCKED" in md and "P5" in md


def test_sensor_missing_from_one_window_and_scene_minimum(study):
    for meta in scenes.discover(study / "scenes" / "s2"):
        if "2023-06-01" <= meta["date"].isoformat() <= "2023-07-15":
            shutil.rmtree(meta["path"])
    report = preflight.run(_cfg(study, preflight={"min_s1_scenes_per_window": 50}))
    assert "scenes.s2_every_window" in codes(report, "error")
    assert "scenes.window_minimum" in codes(report, "warn")


def test_fully_clouded_window_is_an_error(study):
    for meta in scenes.discover(study / "scenes" / "s2"):
        if "2023-07-16" <= meta["date"].isoformat() <= "2023-08-31":
            with rasterio.open(meta["path"] / "SCL.tif", "r+") as ds:
                ds.write(np.full((ds.height, ds.width), 9, dtype="uint8"), 1)
    report = preflight.run(_cfg(study))
    bad = [c for c in report["checks"] if c["code"] == "coverage.clear_observations" and c["status"] == "error"]
    assert [c["details"]["window"] for c in bad] == ["P3"]
    assert bad[0]["details"]["aoi_fraction_zero_clear"] == 1.0


def test_insufficient_class_blocks(study):
    one_block = preflight.run(_cfg(study, model={"spatial_block_m": 100000}))
    assert "classes.spatial_groups" in codes(one_block, "error") and one_block["verdict"] == "blocked"
    training = gpd.read_file(study / "reference" / "training.gpkg")
    extra = gpd.GeoDataFrame({"polygon_id": ["RARE1"], "class_name": ["rice"]},
                             geometry=[box(600010, 4978100, 600100, 4978190)], crs=training.crs)
    gpd.GeoDataFrame(pd_concat(training, extra), crs=training.crs).to_file(study / "reference" / "training.gpkg")
    report = preflight.run(_cfg(study, model={"cv_folds": 5, "holdout_fraction": 0.2}))
    assert "classes.blocks" in codes(report, "warn")
    rare = report["classes"]["per_class"]["rice"]
    assert rare == {"polygons": 1, "blocks": 1}
    assert "classes.fold_coverage" in codes(report, "warn") or "classes.development" in codes(report, "error")


def pd_concat(a, b):
    import pandas as pd
    return pd.concat([a, b], ignore_index=True)


def test_crs_mismatch_is_reported(study):
    report = preflight.run(_cfg(study, grid={"crs": "EPSG:32633"}))
    assert "grid.declared_crs" in codes(report, "error")
    training = gpd.read_file(study / "reference" / "training.gpkg")
    training.set_crs("EPSG:3857", allow_override=True).to_file(study / "reference" / "training.gpkg")
    report = preflight.run(_cfg(study))
    assert codes(report, "error") & {"training.overlap", "training.reprojection"}
    assert report["verdict"] == "blocked"


def test_inventory_counts_without_imported_scenes(study, tmp_path):
    inventory = tmp_path / "s2.csv"
    rows = ["id,name,collection,product_type,sensing_start,tile,relative_orbit,orbit_direction,cloud_cover,bytes,online"]
    rows += [f"id{m}{i},S2B_MSIL2A_x,SENTINEL-2,S2MSI2A,2023-0{m}-1{i}T09:40:00Z,T34TCQ,79,DESCENDING,10,1,true"
             for m in (4, 6, 8, 9) for i in range(2)]
    inventory.write_text("\n".join(rows) + "\n")
    cfg = _cfg(study, paths={"s2_scenes": str(tmp_path / "none2"), "s1_scenes": str(tmp_path / "none1")})
    report = preflight.run(cfg, inventories=[str(inventory)])
    assert [w["s2_inventory"] for w in report["windows"]] == [2, 2, 2, 2]
    assert "scenes.not_imported" in codes(report, "warn") and "scenes.window_minimum" in codes(report, "warn")
    assert "scenes.window_empty" not in codes(report, "error")


Usage = namedtuple("Usage", "total used free")


def test_disk_shortage_blocks_preflight_and_guards_pipeline(study, monkeypatch):
    monkeypatch.setattr(preflight.shutil, "disk_usage", lambda _: Usage(10**12, 10**12 - 10**6, 10**6))
    cfg = _cfg(study)
    report = preflight.run(cfg)
    assert "resources.disk" in codes(report, "error") and report["verdict"] == "blocked"
    runner = pipeline.Runner(cfg, log=lambda *_: None)
    with pytest.raises(preflight.DiskSpaceError, match="Not enough free disk for stage 'composite'"):
        runner.composite()
    assert "composite" not in json.loads(runner.state_path.read_text() if runner.state_path.exists() else "{}")
    assert not Path(cfg["paths"]["composites"]).exists()
    assert "Not enough free disk" in (Path(cfg["paths"]["output"]) / "run.log").read_text()
    # Disabled guard: the same stage starts (and here succeeds) regardless of the patched free space.
    off = _cfg(study, resources={"disk_guard": False})
    pipeline.Runner(off, log=lambda *_: None).composite()
    assert (Path(cfg["paths"]["composites"]) / "P1" / "B02.tif").is_file()


def test_disk_guard_credits_outputs_that_will_be_overwritten(study, monkeypatch):
    cfg = _cfg(study, resources={"min_free_gb": 0})
    need = preflight.stage_bytes(cfg, preflight.plan(cfg), "composite") * 0.8 * 1.25
    monkeypatch.setattr(preflight.shutil, "disk_usage", lambda _: Usage(10**12, 0, int(need * 0.5)))
    with pytest.raises(preflight.DiskSpaceError):
        preflight.check_disk(cfg, "composite")
    assert preflight.check_disk(cfg, "composite", existing_bytes=int(need))["need_gb"] == 0


@pytest.mark.parametrize("override", [{"safety_factor": 0.5}, {"min_free_gb": -1}, {"compression_ratio": 0},
                                      {"disk_guard": "yes"}])
def test_resource_config_validation(override):
    with pytest.raises(config.ConfigError, match="resources"):
        config.load(overrides={"resources": override})
    with pytest.raises(config.ConfigError, match="preflight"):
        config.load(overrides={"preflight": {"max_unobserved_fraction": 2}})


def test_failed_stage_outputs_are_quarantined(tmp_path):
    cfg = config.load(overrides={"paths": {"output": str(tmp_path / "run")}})
    old, partial = tmp_path / "old.json", tmp_path / "partial.tif"
    old.write_text("previous run")
    import os
    os.utime(old, (1_000_000_000, 1_000_000_000))

    def broken():
        partial.write_bytes(b"half a raster")
        raise RuntimeError("disk vanished")

    runner = pipeline.Runner(cfg, log=lambda *_: None)
    with pytest.raises(RuntimeError, match="disk vanished"):
        runner.stage("train", "x", [partial, old], broken)
    assert not partial.exists() and old.read_text() == "previous run"
    state = json.loads(runner.state_path.read_text())["train"]
    assert state["status"] == "failed" and state["info"]["quarantined"][0]["path"] == str(partial)
    moved = list((tmp_path / "run" / ".quarantine").rglob("*partial.tif"))
    assert len(moved) == 1 and moved[0].read_bytes() == b"half a raster"


def test_clean_dry_run_then_real_never_touches_inputs(study, tmp_path):
    cfg = _cfg(study, resources={"disk_guard": False})
    pipeline.Runner(cfg, log=lambda *_: None).composite()
    out, comp = Path(cfg["paths"]["output"]), Path(cfg["paths"]["composites"])
    (out / ".stages-abc.tmp").write_text("x")
    (comp / "P1" / "B02.tif.part").write_text("x")
    (comp / "P1" / "notes.txt").write_text("user notes")
    cleanup.quarantine_partial([comp / "P2" / "B03.tif"], 0, out, "composite")
    scene_tmp = study / "scenes" / "s2" / "leftover.tmp"     # inside a protected input: never removed
    scene_tmp.write_text("x")
    inputs = sorted(p for p in study.rglob("*") if p.is_file() and "run" not in p.relative_to(study).parts)
    before = {p: p.read_bytes() for p in inputs}

    dry = cleanup.clean(cfg, dry_run=True, all_intermediates=True)
    listed = {Path(i["path"]) for i in dry["would_remove"]}
    assert out / ".stages-abc.tmp" in listed and out / ".quarantine" in listed
    assert comp / "P1" / "B02.tif" in listed and comp / "P1" / "notes.txt" not in listed
    assert all(i["bytes"] >= 0 for i in dry["would_remove"]) and dry["bytes"] > 0
    assert (out / ".stages-abc.tmp").exists() and (comp / "P1" / "B02.tif").exists()

    light = cleanup.clean(cfg, dry_run=False)
    assert not (out / ".stages-abc.tmp").exists() and not (comp / "P1" / "B02.tif.part").exists()
    assert not (out / ".quarantine").exists() and (comp / "P1" / "B02.tif").exists()
    assert {Path(i["path"]) for i in light["removed"]}.isdisjoint(inputs)

    full = cleanup.clean(cfg, dry_run=False, all_intermediates=True)
    assert not (comp / "P1" / "B02.tif").exists() and (comp / "P1" / "notes.txt").read_text() == "user notes"
    assert not (comp / "P2").exists()
    assert full["errors"] == []
    assert {p: p.read_bytes() for p in inputs} == before and scene_tmp.exists()


def test_clean_skips_protected_paths_inside_output(tmp_path):
    out = tmp_path / "work"
    scenes_dir = out / "s2"
    scenes_dir.mkdir(parents=True)
    (scenes_dir / "download.part").write_text("resumable download")
    (out / "stray.tmp").write_text("x")
    cfg = config.load(overrides={"paths": {"output": str(out), "s2_scenes": str(scenes_dir)}})
    report = cleanup.clean(cfg, dry_run=False)
    assert (scenes_dir / "download.part").exists() and not (out / "stray.tmp").exists()
    assert any(i["path"].endswith("download.part") for i in report["skipped_protected"])


def test_cli_preflight_and_clean(study, tmp_path, capsys):
    cfg_path = tmp_path / "study.yaml"
    import yaml
    raw = _cfg(study)
    cfg_path.write_text(yaml.safe_dump({k: raw[k] for k in ("grid", "windows", "paths", "model")}, default_flow_style=False))
    cli.main(["preflight", "--config", str(cfg_path), "--out", str(tmp_path / "pf.json"),
              "--markdown", str(tmp_path / "pf.md")])
    report = json.loads((tmp_path / "pf.json").read_text())
    assert report["schema"] == "geocrop-preflight/1" and report["verdict"] != "blocked"
    assert (tmp_path / "pf.md").read_text().startswith("# GeoCrop preflight")
    blocked = yaml.safe_load(cfg_path.read_text())
    blocked["windows"]["P5"] = {"start": "2023-12-01", "end": "2023-12-31"}
    cfg_path.write_text(yaml.safe_dump(blocked))
    with pytest.raises(SystemExit) as exc:
        cli.main(["preflight", "--config", str(cfg_path)])
    assert exc.value.code == 2
    capsys.readouterr()
    cli.main(["clean", "--config", str(cfg_path), "--dry-run"])
    out = json.loads(capsys.readouterr().out)
    assert out["dry_run"] is True and "would_remove" in out


def test_unloadable_estimator_backend_blocks_preflight(study, monkeypatch):
    """Same rule as train/benchmark: a missing or unloadable optional backend is reported, never a crash."""
    from geocrop import estimators
    monkeypatch.setattr(estimators, "backend_error",
                        lambda algo: "lightgbm is installed but cannot load its native library (dlopen)"
                        if algo == "lightgbm" else None)
    report = preflight.run(_cfg(study, model={"algorithm": "lightgbm"}))
    assert "model.backend" in codes(report, "error") and report["verdict"] == "blocked"
    assert "model.backend" in codes(preflight.run(_cfg(study)), "pass")


def test_estimate_is_calibrated_against_the_measured_500_km2_run(study):
    """Coefficients recalibrated from tools/scale_benchmark.py (docs/performance.md): a 5 Mpx grid with 97
    features, 100 trees and 551 450 training samples measured 741 s prediction (AOA map dominates), 181 s
    training and 6.1 GB peak RSS. The estimate must stay within a factor of two and not under-estimate RAM."""
    cfg = _cfg(study, model={"n_estimators": 100, "cv_folds": 3})
    layout = preflight.plan(cfg)
    layout = {**layout, "pixels": 5_000_000, "window_features": 97,
              "grid": {**layout["grid"], "width": 2237, "height": 2237, "res": (10.0, 10.0)}}
    est = preflight.estimate(cfg, layout, n_samples=551_450, n_classes=4)
    predict_s = est["runtime_hours_rough"]["predict"] * 3600
    assert 741 / 2 <= predict_s <= 741 * 2
    assert est["peak_ram_gb"]["max"] >= 6.1
    small = preflight.estimate(cfg, {**layout, "pixels": 250_000}, n_samples=28_600, n_classes=4)
    assert small["peak_ram_gb"]["max"] >= 1.0          # interpreter + GDAL overhead is always included


def test_memory_limit_honours_cgroup_limits(tmp_path):
    installed, source = preflight.memory_limit_bytes(cgroup_files=())
    assert source == "installed"
    v2 = tmp_path / "memory.max"
    v2.write_text("max\n")                                    # cgroup v2 without a limit
    assert preflight.memory_limit_bytes(cgroup_files=(str(v2),)) == (installed, "installed")
    v2.write_text(f"{2**30}\n")                               # docker run --memory 1g
    assert preflight.memory_limit_bytes(cgroup_files=(str(v2),)) == (2**30, str(v2))
    v1 = tmp_path / "memory.limit_in_bytes"
    v1.write_text("9223372036854771712\n")                    # cgroup v1 "unlimited" sentinel
    assert preflight.memory_limit_bytes(cgroup_files=(str(tmp_path / "absent"), str(v1))) == (installed, "installed")


def test_ram_check_uses_the_container_limit(study, monkeypatch):
    monkeypatch.setattr(preflight, "memory_limit_bytes", lambda *a, **k: (10**6, "/sys/fs/cgroup/memory.max"))
    rep = preflight.run(_cfg(study))
    ram = next(c for c in rep["checks"] if c["code"] == "resources.ram")
    assert ram["status"] == "error" and "cgroup" in ram["message"]
    assert ram["details"]["memory_source"].endswith("memory.max") and rep["verdict"] == "blocked"


def test_available_cpus_honours_cgroup_quota(tmp_path):
    base, _ = preflight.available_cpus(cgroup_files=())
    v2 = tmp_path / "cpu.max"
    v2.write_text("max 100000\n")
    assert preflight.available_cpus(cgroup_files=(str(v2),))[0] == base
    v2.write_text("50000 100000\n")                              # docker run --cpus 0.5
    assert preflight.available_cpus(cgroup_files=(str(v2),)) == (0.5, str(v2))
    quota, period = tmp_path / "cfs_quota_us", tmp_path / "cfs_period_us"
    quota.write_text("-1\n")                                      # cgroup v1 without quota
    period.write_text("100000\n")
    assert preflight.available_cpus(cgroup_files=((str(quota), str(period)),))[0] == base


def test_training_estimate_scales_with_usable_cpus(study):
    cfg = _cfg(study, model={"n_estimators": 100, "cv_folds": 3})
    layout = preflight.plan(cfg)
    train = {c: preflight.estimate(cfg, layout, n_samples=5_000_000, cpus=c)["runtime_hours_rough"]["train"]
             for c in (2, 8, 32)}
    assert train[8] == train[32] and train[2] == pytest.approx(4 * train[8], rel=0.01)
