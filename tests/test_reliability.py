"""Release gate: reliability, recovery, reproducibility, corrupt data and scale.

* Fault injection: a real ``geocrop run`` subprocess is SIGKILLed inside the features, train and predict
  stages (via the inert-by-default hook in `geocrop.faultinject`), then rerun. Completed stages must be
  skipped, the interrupted stage and its dependants rerun, the stale lock recovered and every product must
  equal an uninterrupted reference run.
* Corrupt inputs must fail with a precise error naming the file and the problem, never yield a silently
  wrong classification.
* Two independent runs from fresh directories must give identical products.
* (slow) Peak memory of features + predict is bounded by the block size, not the raster size.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import tracemalloc
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import rasterio
import yaml
from rasterio.transform import from_origin

import geocrop
from geocrop import cache, config, faultinject, features, model, pipeline, s2, samples, scenes
from geocrop.demo import generate
from geocrop.indices import S2_BANDS
from geocrop.raster import NODATA, RasterInputError

SRC = Path(geocrop.__file__).resolve().parents[1]
STAGES = ("features", "samples", "train", "predict")
FAULT_POINTS = ("features", "train", "predict")
PRODUCTS = ("features/stack.tif", "features/stack.features.json", "samples.npz", "model/model.joblib",
            "model/train_report.json", "model/conformal.json", "prediction/classes.tif",
            "prediction/probabilities.tif", "prediction/uncertainty.tif", "prediction/aoa.tif",
            "prediction/class_legend.json")


# --------------------------------------------------------------------------- helpers

def _config(root: Path) -> Path:
    """Identical config text for every run: inputs under root/data, products under root/run (relative)."""
    cfg = {"seed": 11,
           "windows": {"P1": {"start": "2023-04-01", "end": "2023-05-15"},
                       "P2": {"start": "2023-06-01", "end": "2023-06-30"},
                       "P3": {"start": "2023-07-15", "end": "2023-08-15"},
                       "P4": {"start": "2023-09-01", "end": "2023-09-30"}},
           "paths": {"composites": "data/composites", "features": "run/features/stack.tif",
                     "training": "data/reference/training.gpkg", "parcels": "data/reference/parcels.gpkg",
                     "observations": "data/reference/observations.gpkg", "output": "run"},
           "model": {"n_estimators": 15, "n_jobs": 1, "cv_folds": 3, "spatial_block_m": 500},
           "validation": {"bootstrap": 20, "year": 2023}}
    path = root / "study.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=True), encoding="utf8")
    return path


def _workspace(base: Path, data: Path) -> Path:
    base.mkdir(parents=True)
    (base / "data").symlink_to(data, target_is_directory=True)
    _config(base)
    return base


def _env(**extra) -> dict:
    env = {k: v for k, v in os.environ.items() if not k.startswith("GEOCROP_FAULT_")}
    env["PYTHONPATH"] = os.pathsep.join([str(SRC), *filter(None, [env.get("PYTHONPATH")])])
    env.update(extra)
    return env


def _cli(root: Path, *stages, env=None, check=True) -> subprocess.CompletedProcess:
    proc = subprocess.run([sys.executable, "-m", "geocrop.cli", "run", "--config", "study.yaml",
                           "--stages", *stages], cwd=root, env=env or _env(), capture_output=True, text=True,
                          timeout=600)
    if check and proc.returncode:
        raise AssertionError(f"geocrop run failed ({proc.returncode}):\n{proc.stdout[-2000:]}\n{proc.stderr[-4000:]}")
    return proc


def _kill_inside(root: Path, point: str) -> int:
    """Start a full run that pauses at `point`, wait until it is there, SIGKILL it and reap it."""
    sig = root / f"{point}.signal"
    log = (root / "killed.out").open("w")
    proc = subprocess.Popen([sys.executable, "-m", "geocrop.cli", "run", "--config", "study.yaml",
                             "--stages", *STAGES], cwd=root, stdout=log, stderr=subprocess.STDOUT,
                            env=_env(**{faultinject.PAUSE_ENV: point, faultinject.SIGNAL_ENV: str(sig)}))
    try:
        deadline = time.monotonic() + 300
        while not sig.exists():
            if proc.poll() is not None:
                raise AssertionError(f"run ended before reaching {point}:\n{(root / 'killed.out').read_text()[-4000:]}")
            if time.monotonic() > deadline:
                raise AssertionError(f"run never reached fault point {point}")
            time.sleep(0.05)
        assert sig.read_text().split() == [point, str(proc.pid)]
    finally:
        proc.send_signal(signal.SIGKILL)
        proc.wait(timeout=60)
        log.close()
    sig.unlink()
    return proc.pid


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read(path: Path) -> np.ndarray:
    with rasterio.open(path) as ds:
        return ds.read()


def _metrics(report: dict) -> dict:
    cv = report["cross_validation"]
    return {"cv": {k: cv[k] for k in ("macro_f1", "overall_accuracy", "kappa", "confusion_matrix", "labels",
                                     "folds", "temperature") if k in cv},
            **{k: report[k] for k in ("samples", "polygons", "class_counts", "params", "aoa_threshold",
                                      "suspect_labels", "spatial_groups")}}


def _assert_same_products(run: Path, reference: Path):
    """Products of `run` equal those of `reference` (rasters numerically, reports by their metrics)."""
    for name in PRODUCTS:
        assert (run / name).is_file() and (run / name).stat().st_size, f"missing product {name}"
    stack, ref_stack = _read(run / "features/stack.tif"), _read(reference / "features/stack.tif")
    assert np.array_equal(stack, ref_stack)
    assert (json.loads((run / "features/stack.features.json").read_text())
            == json.loads((reference / "features/stack.features.json").read_text()))
    a, b = np.load(run / "samples.npz"), np.load(reference / "samples.npz")
    assert set(a.files) == set(b.files) and all(np.array_equal(a[k], b[k]) for k in a.files)
    assert np.array_equal(_read(run / "prediction/classes.tif"), _read(reference / "prediction/classes.tif"))
    for name in ("probabilities.tif", "uncertainty.tif", "aoa.tif"):
        np.testing.assert_allclose(_read(run / "prediction" / name), _read(reference / "prediction" / name),
                                   rtol=0, atol=1e-6, err_msg=name)
    report, ref_report = (json.loads((d / "model/train_report.json").read_text()) for d in (run, reference))
    assert _metrics(report) == _metrics(ref_report)
    assert (json.loads((run / "prediction/class_legend.json").read_text())
            == json.loads((reference / "prediction/class_legend.json").read_text()))


def _assert_consistent_state(run: Path):
    """Every stage succeeded, the recorded output identities match the files on disk, nothing stale is left."""
    state = json.loads((run / "stages.json").read_text())
    for name in STAGES:
        entry = state[name]
        assert entry["status"] == "success", (name, entry)
        for path, identity in entry["outputs"].items():
            assert identity["sha256"] == _sha(Path(path)), f"{path} differs from its recorded identity"
    assert not (run / ".geocrop.lock").exists()
    assert not [p for p in run.rglob("*") if p.name.endswith(".tmp")]
    manifest = json.loads((run / "prediction/prediction_manifest.json").read_text())
    assert manifest["model_sha256"] == _sha(run / "model/model.joblib")
    with rasterio.open(run / "prediction/classes.tif") as ds:
        assert ds.tags()["model_sha256"] == manifest["model_sha256"]


# --------------------------------------------------------------------------- fixtures

@pytest.fixture(scope="module")
def study_data(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("study") / "data"
    generate(root)
    return root


def _independent_run(base: Path) -> Path:
    """Fresh directory with freshly generated inputs and the same config text."""
    base.mkdir(parents=True)
    generate(base / "data")
    _config(base)
    _cli(base, *STAGES)
    return base / "run"


def _interrupted_run(base: Path, data: Path, point: str) -> dict:
    """Kill a run inside `point`, try a downstream stage, recover, rerun once more; record every step."""
    root = _workspace(base, data)
    run = root / "run"
    pid = _kill_inside(root, point)
    after_kill = {"pid": pid, "lock": (run / ".geocrop.lock").read_text().strip(),
                  "state": json.loads((run / "stages.json").read_text()),
                  "files": sorted(str(p.relative_to(run)) for p in run.rglob("*") if p.is_file())}
    downstream = {"features": "samples", "train": "predict", "predict": "validate"}[point]
    refused = _cli(root, downstream, check=False)
    state_after_refusal = json.loads((run / "stages.json").read_text())
    rerun = _cli(root, *STAGES)
    again = _cli(root, *STAGES)

    def stage_lines(proc):
        return [ln for ln in proc.stdout.splitlines() if ln.startswith("[")]
    return {"run": run, "after_kill": after_kill, "refused": refused, "state_after_refusal": state_after_refusal,
            "rerun": stage_lines(rerun), "again": stage_lines(again)}


@pytest.fixture(scope="module")
def runs(study_data, tmp_path_factory) -> dict:
    """All subprocess runs of this module, executed concurrently to keep the default tier fast:
    an uninterrupted reference, an independent reproduction and one SIGKILLed run per fault point."""
    from concurrent.futures import ThreadPoolExecutor
    base = tmp_path_factory.mktemp("runs")
    with ThreadPoolExecutor(max_workers=5) as pool:
        jobs = {"reference": pool.submit(lambda: _cli(_workspace(base / "reference", study_data), *STAGES)),
                "independent": pool.submit(_independent_run, base / "independent")}
        for point in FAULT_POINTS:
            jobs[point] = pool.submit(_interrupted_run, base / point, study_data, point)
        results = {name: job.result() for name, job in jobs.items()}
    results["reference"] = base / "reference" / "run"
    return results


# --------------------------------------------------------------------------- 1. fault injection

def test_fault_hook_is_inert_without_environment(monkeypatch):
    monkeypatch.delenv(faultinject.PAUSE_ENV, raising=False)
    started = time.perf_counter()
    for _ in range(10000):
        faultinject.point("predict")
    assert time.perf_counter() - started < 0.5
    monkeypatch.setenv(faultinject.PAUSE_ENV, "train")
    monkeypatch.setenv(faultinject.SECONDS_ENV, "0")
    faultinject.point("predict")                 # other names stay inert


@pytest.mark.parametrize("point", FAULT_POINTS)
def test_sigkill_inside_stage_recovers_to_reference(point, runs):
    result, reference = runs[point], runs["reference"]
    run, killed = result["run"], result["after_kill"]
    done = STAGES[:STAGES.index(point)]

    # The kill left a stale lock, a "running" record and partial products on disk.
    assert killed["lock"] == str(killed["pid"])
    assert killed["state"][point]["status"] == "running"
    assert all(killed["state"][s]["status"] == "success" for s in done)
    assert not any(s in killed["state"] for s in STAGES[len(done) + 1:])
    partial = {"features": ("features/stack.tif", "features/stack.features.json"),
               "train": ("model/model.joblib", "model/train_report.json"),
               "predict": ("prediction/classes.tif", "prediction/class_legend.json")}[point]
    assert partial[0] in killed["files"] and partial[1] not in killed["files"]

    # Downstream stages refuse the partial products of the interrupted stage instead of consuming them.
    refused = result["refused"]
    assert refused.returncode != 0
    assert f"depends on {{'{point}': 'running'}}" in refused.stderr and "rerun them first" in refused.stderr
    assert result["state_after_refusal"] == killed["state"]

    # Recovery: completed stages skipped, the interrupted one and its dependants rerun, stale lock taken over.
    assert result["rerun"] == [f"[skip] {s} (unchanged)" for s in done] + [f"[run ] {s}" for s in STAGES[len(done):]]
    _assert_consistent_state(run)
    _assert_same_products(run, reference)
    assert _sha(run / "model/model.joblib") == _sha(reference / "model/model.joblib")
    # A third run has nothing left to do.
    assert result["again"] == [f"[skip] {s} (unchanged)" for s in STAGES]


def test_interrupted_upstream_blocks_downstream_in_process(tmp_path):
    cfg = config.load(overrides={"paths": {"output": str(tmp_path / "run")}})
    runner = pipeline.Runner(cfg, log=lambda *_: None)
    runner.state["train"] = {"status": "failed", "fingerprint": "x"}
    calls = []
    with pytest.raises(RuntimeError, match=r"depends on \{'train': 'failed'\}"):
        runner.stage("predict", "fp", [tmp_path / "out.txt"], lambda: calls.append(1))
    assert not calls and "predict" not in runner.state
    # A disabled optional stage (cbm) with an old failure does not block parcels.
    runner.state = {"cbm": {"status": "failed"}}
    target = tmp_path / "parcels.txt"
    runner.stage("parcels", "fp", [target], lambda: target.write_text("ok"))
    assert runner.state["parcels"]["status"] == "success"


def test_stale_and_live_run_locks(tmp_path):
    lock = tmp_path / ".geocrop.lock"
    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait()
    for stale in (str(dead.pid), "", "garbage"):
        lock.write_text(stale)
        with cache.RunLock(tmp_path):
            assert lock.read_text() == str(os.getpid())
        assert not lock.exists()
    alive = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        lock.write_text(str(alive.pid))
        with pytest.raises(RuntimeError, match=f"pid {alive.pid}"):
            cache.RunLock(tmp_path).__enter__()
        assert lock.read_text() == str(alive.pid)
    finally:
        alive.kill()
        alive.wait()
    assert not list(tmp_path.glob("*.tmp"))


def test_corrupt_hash_cache_is_rebuilt(tmp_path):
    (tmp_path / ".hashcache.json").write_text('{"truncated": ')
    hc = cache.HashCache(tmp_path / ".hashcache.json")
    assert hc.entries == {}
    big = tmp_path / "big.bin"
    big.write_bytes(b"x" * (cache.SMALL_FILE_BYTES + 1))
    (tmp_path / ".hashcache.json").write_text(json.dumps({str(big.resolve()): {"size": "wrong"}}))
    assert cache.HashCache(tmp_path / ".hashcache.json").file(big) == _sha(big)


# --------------------------------------------------------------------------- 3. reproducibility

def test_independent_runs_are_identical(runs):
    """Fresh directory, freshly generated inputs, same config: identical products and metrics."""
    run, reference = runs["independent"], runs["reference"]
    _assert_consistent_state(run)
    _assert_same_products(run, reference)
    for name in ("features/stack.tif", "samples.npz", "model/model.joblib", "prediction/classes.tif",
                 "prediction/probabilities.tif", "prediction/aoa.tif"):
        assert _sha(run / name) == _sha(reference / name), name
    assert (json.loads((run / "model/train_report.json").read_text())
            == json.loads((reference / "model/train_report.json").read_text()))


# --------------------------------------------------------------------------- 2. corrupt data

@pytest.fixture
def composites(demo, tmp_path) -> Path:
    target = tmp_path / "composites"
    shutil.copytree(demo["composites"], target)
    return target


def samples_mod_extract(stack, polygons, **kw):
    return samples.extract(stack, polygons, max_per_polygon=5, **kw)


def _rewrite(path: Path, transform=None, crs=None, fill=None, edit=None, descriptions=True):
    with rasterio.open(path) as ds:
        profile, data, names = ds.profile, ds.read(), ds.descriptions
    profile.update({k: v for k, v in (("transform", transform), ("crs", crs)) if v is not None})
    if fill is not None:
        data[:] = fill
    if edit is not None:
        edit(data)
    path.unlink()
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)
        if descriptions:
            for i, n in enumerate(names, 1):
                if n:
                    dst.set_band_description(i, n)


@pytest.mark.parametrize("cut", ["header", "half", "tail"])
def test_truncated_composite_names_file(composites, tmp_path, cut):
    target = composites / "P2" / "B04.tif"
    data = target.read_bytes()
    keep = {"header": 100, "half": len(data) // 2, "tail": len(data) - 300}[cut]
    target.write_bytes(data[:keep])
    with pytest.raises(RasterInputError, match="truncated") as exc:
        features.build_stack(composites, tmp_path / "stack.tif")
    assert str(target) in str(exc.value)


@pytest.mark.parametrize("change,needle", [
    ({"crs": "EPSG:32633"}, "CRS"),
    ({"transform": from_origin(600005.0, 4980000.0, 10, 10)}, "transform"),
])
def test_composite_on_wrong_grid_names_file(composites, tmp_path, change, needle):
    target = composites / "P3" / "B11.tif"
    _rewrite(target, **change)
    with pytest.raises(RasterInputError, match="Grid mismatch") as exc:
        features.build_stack(composites, tmp_path / "stack.tif")
    assert str(target) in str(exc.value) and needle in str(exc.value)


def test_composite_with_different_shape_names_file(composites, tmp_path):
    target = composites / "P1" / "VH.tif"
    with rasterio.open(target) as ds:
        profile, data = ds.profile, ds.read(1)
    profile.update(width=profile["width"] - 8)
    with rasterio.open(target, "w", **profile) as dst:
        dst.write(data[:, :-8], 1)
    with pytest.raises(RasterInputError, match="shape") as exc:
        features.build_stack(composites, tmp_path / "stack.tif")
    assert str(target) in str(exc.value)


@pytest.mark.parametrize("fill", [NODATA, np.nan, np.inf])
def test_all_invalid_composite_band_is_rejected(composites, tmp_path, fill):
    target = composites / "P2" / "B08.tif"
    _rewrite(target, fill=fill)
    out = tmp_path / "stack.tif"
    with pytest.raises(RasterInputError, match="without a single valid pixel") as exc:
        features.build_stack(composites, out)
    assert str(target) in str(exc.value)
    assert not out.exists() and not features.manifest_path(out).exists()


def test_nan_and_inf_pixels_become_nodata_not_classes(composites, samples, tmp_path):
    """Isolated NaN/inf input pixels must end up unclassified (255), never as a class or a bogus probability."""
    def poison(data):
        data[0, 10, 10] = np.nan
        data[0, 20, 30] = np.inf
        data[0, 40, 50] = -np.inf
    _rewrite(composites / "P3" / "B04.tif", edit=poison)
    stack = features.build_stack(composites, tmp_path / "stack.tif")
    with rasterio.open(stack) as ds:
        cube = ds.read()
    assert np.isfinite(cube).all()
    bad = [(10, 10), (20, 30), (40, 50)]
    for r, c in bad:
        assert (cube[:, r, c] == NODATA).any()
    report = model.train(samples, tmp_path / "model", params={"n_estimators": 10, "n_jobs": 1}, n_splits=3,
                         spatial_block_m=500, bootstrap=0)
    assert report["samples"] > 0
    model.predict(stack, tmp_path / "model" / "model.joblib", tmp_path / "pred")
    classes, proba = _read(tmp_path / "pred" / "classes.tif")[0], _read(tmp_path / "pred" / "probabilities.tif")
    for r, c in bad:
        assert classes[r, c] == 255 and (proba[:, r, c] == NODATA).all()
    valid = classes != 255
    assert valid.sum() == classes.size - len(bad)
    assert np.isfinite(proba).all()
    np.testing.assert_allclose(proba[:, valid].sum(axis=0), 1, atol=1e-5)


def test_stack_without_band_descriptions_names_file(stack, samples, demo, tmp_path):
    plain = tmp_path / "plain.tif"
    shutil.copy(stack, plain)
    _rewrite(plain, descriptions=False)
    with pytest.raises(RasterInputError, match="no description") as exc:
        samples_mod_extract(plain, demo["training"])
    assert str(plain) in str(exc.value)
    model.train(samples, tmp_path / "model", params={"n_estimators": 5, "n_jobs": 1}, n_splits=3,
                spatial_block_m=500, bootstrap=0)
    with pytest.raises(RasterInputError, match="no description") as exc:
        model.predict(plain, tmp_path / "model" / "model.joblib", tmp_path / "pred")
    assert str(plain) in str(exc.value)


def test_truncated_or_all_nodata_stack_fails_predict(stack, samples, tmp_path):
    model.train(samples, tmp_path / "model", params={"n_estimators": 5, "n_jobs": 1}, n_splits=3,
                spatial_block_m=500, bootstrap=0)
    model_path = tmp_path / "model" / "model.joblib"
    broken = tmp_path / "broken.tif"
    broken.write_bytes(Path(stack).read_bytes()[: Path(stack).stat().st_size // 2])
    with pytest.raises(RasterInputError, match="truncated") as exc:
        model.predict(broken, model_path, tmp_path / "pred")
    assert str(broken) in str(exc.value)
    empty = tmp_path / "empty.tif"
    shutil.copy(stack, empty)
    _rewrite(empty, fill=NODATA)
    with pytest.raises(RasterInputError, match="no pixel has valid values") as exc:
        model.predict(empty, model_path, tmp_path / "pred2")
    assert str(empty) in str(exc.value)
    assert not (tmp_path / "pred2" / "classes.tif").exists()


@pytest.mark.parametrize("body,needle", [
    ('{"sensor": "S2", "datetime": "2023-05-', "not valid JSON"),
    ("[1, 2]", "expected a JSON object"),
    ('{"sensor": "S2"}', "lacks required keys ['datetime']"),
    ('{"sensor": "S2", "datetime": "yesterday"}', "invalid ISO-8601 datetime"),
    ('{"sensor": "L8", "datetime": "2023-05-01"}', "unknown sensor"),
    (b"\xff\xfe\x00garbage", "not valid JSON"),
])
def test_corrupt_scene_json_names_file(tmp_path, body, needle):
    folder = tmp_path / "s2" / "2023-05-01"
    folder.mkdir(parents=True)
    (tmp_path / "s2" / "2023-04-01").mkdir()
    scenes.write(tmp_path / "s2" / "2023-04-01", sensor="S2", datetime="2023-04-01T10:00:00")
    sidecar = folder / scenes.SIDE_CAR
    sidecar.write_bytes(body if isinstance(body, bytes) else body.encode())
    with pytest.raises(ValueError) as exc:
        scenes.discover(tmp_path / "s2")
    assert str(sidecar) in str(exc.value) and needle in str(exc.value)


def _scene(folder: Path, scl: np.ndarray, date="2023-05-01"):
    folder.mkdir(parents=True)
    prof = {"driver": "GTiff", "height": scl.shape[0], "width": scl.shape[1], "count": 1, "crs": "EPSG:32634",
            "transform": from_origin(600000, 4980000, 10, 10)}
    for i, b in enumerate(S2_BANDS):
        with rasterio.open(folder / f"{b}.tif", "w", dtype="float32", nodata=NODATA, **prof) as dst:
            dst.write(np.full(scl.shape, 0.05 + 0.02 * i, "float32"), 1)
    with rasterio.open(folder / "SCL.tif", "w", dtype=scl.dtype, **prof) as dst:
        dst.write(scl, 1)
    scenes.write(folder, sensor="S2", datetime=f"{date}T10:00:00")
    return folder


def test_scene_mask_with_unexpected_values_is_rejected(tmp_path):
    good = _scene(tmp_path / "a", np.full((8, 8), 4, "uint8"))
    scl = np.full((8, 8), 4, "uint8")
    scl[2:4, 2:4] = 200
    odd = _scene(tmp_path / "b", scl, "2023-05-06")
    s2.composite([good], tmp_path / "ok")                     # a valid mask composites fine
    with pytest.raises(RasterInputError, match=r"values \[200\] outside the ESA SCL classes") as exc:
        s2.composite([good, odd], tmp_path / "bad")
    assert str(odd / "SCL.tif") in str(exc.value)
    floaty = _scene(tmp_path / "c", np.full((8, 8), 4.5, "float32"), "2023-05-11")
    with pytest.raises(RasterInputError, match="must hold integer classes") as exc:
        s2.composite([floaty], tmp_path / "float")
    assert str(floaty / "SCL.tif") in str(exc.value)


def _training(demo, tmp_path, mutate) -> Path:
    gdf = gpd.read_file(demo["training"])
    gdf = mutate(gdf)
    path = tmp_path / "training.gpkg"
    gdf.to_file(path, driver="GPKG")
    return path


@pytest.mark.parametrize("mutate,needle", [
    (lambda g: g.iloc[0:0], "contains no (non-empty) polygons"),
    (lambda g: g.assign(geometry=g.geometry.buffer(-1e6)), "contains no (non-empty) polygons"),
    (lambda g: g.assign(polygon_id=["T1", "T1", *g.polygon_id[2:]]), "duplicated ids ['T1']"),
    (lambda g: g.assign(class_name=[None, *g.class_name[1:]]), "1 rows have an empty 'class_name'"),
    (lambda g: g.assign(polygon_id=["  ", *g.polygon_id[1:]]), "1 rows have an empty 'polygon_id'"),
    (lambda g: g.drop(columns="class_name"), "missing column 'class_name'"),
    (lambda g: g.set_crs("EPSG:4326", allow_override=True), "intersect feature stack"),
])
def test_bad_training_polygons_name_file_and_problem(demo, stack, tmp_path, mutate, needle):
    path = _training(demo, tmp_path, mutate)
    with pytest.raises(ValueError) as exc:
        samples_mod_extract(stack, path)
    assert str(path) in str(exc.value) and needle in str(exc.value)


def test_training_polygons_over_nodata_explain_why(demo, stack, tmp_path):
    blank = tmp_path / "blank.tif"
    shutil.copy(stack, blank)
    _rewrite(blank, fill=NODATA)
    with pytest.raises(ValueError, match="none contains a pixel with valid values") as exc:
        samples_mod_extract(blank, demo["training"])
    assert str(blank) in str(exc.value)


# --------------------------------------------------------------------------- 4. scale

def _large_composites(root: Path, size: int, periods=("P1", "P2")) -> Path:
    yy, xx = np.mgrid[0:256, 0:size].astype("float32")
    prof = {"driver": "GTiff", "height": size, "width": size, "count": 1, "dtype": "float32", "crs": "EPSG:32634",
            "transform": from_origin(500000, 5000000, 10, 10), "nodata": NODATA, "tiled": True,
            "blockxsize": 256, "blockysize": 256, "compress": "deflate"}
    for k, p in enumerate(periods):
        (root / p).mkdir(parents=True)
        for i, b in enumerate(S2_BANDS):
            with rasterio.open(root / p / f"{b}.tif", "w", **prof) as dst:
                for r in range(0, size, 256):
                    field = ((xx // 64 + (yy + r) // 64) % 3).astype("float32")      # three "crops"
                    arr = 0.05 + 0.01 * i + 0.04 * k + 0.02 * field
                    if b in ("B08", "B8A"):
                        arr = arr + 0.15 * field + 0.2
                    dst.write(arr[: min(256, size - r)], 1, window=rasterio.windows.Window(0, r, size, min(256, size - r)))
    return root


def _toy_model(tmp_path: Path) -> Path:
    """Small RF over the feature names that `_large_composites` produces (same names at every size)."""
    from joblib import dump
    from sklearn.ensemble import RandomForestClassifier
    stack = features.build_stack(_large_composites(tmp_path / "comp_model", 256), tmp_path / "model_stack.tif", s1=())
    with rasterio.open(stack) as ds:
        names = list(ds.descriptions)
        cube = ds.read().reshape(len(names), -1).T
    ndvi = cube[:, names.index("P1_NDVI")]
    y = np.digitize(ndvi, np.quantile(ndvi, [1 / 3, 2 / 3])).astype(str)
    rf = RandomForestClassifier(n_estimators=8, max_depth=8, random_state=0, n_jobs=1).fit(cube, y)
    path = tmp_path / "model" / "model.joblib"
    path.parent.mkdir()
    dump({"model": rf, "features": names, "classes": list(rf.classes_), "temperature": 1.0}, path)
    return path


def _scale_run(tmp_path: Path, size: int, model_path: Path) -> dict:
    comp = _large_composites(tmp_path / f"comp{size}", size)
    tracemalloc.start()
    started = time.perf_counter()
    stack = features.build_stack(comp, tmp_path / f"stack{size}.tif", block_size=512, s1=())
    model.predict(stack, model_path, tmp_path / f"pred{size}", block_size=512)
    seconds = time.perf_counter() - started
    peak = tracemalloc.get_traced_memory()[1] / 1e6
    tracemalloc.stop()
    with rasterio.open(stack) as ds:
        n_features = ds.count
    classes = _read(tmp_path / f"pred{size}" / "classes.tif")[0]
    assert classes.shape == (size, size) and (classes != 255).all()
    for path in (comp, stack):                               # keep the disk footprint small
        shutil.rmtree(path) if Path(path).is_dir() else Path(path).unlink()
    return {"peak_mb": round(peak, 1), "seconds": round(seconds, 2), "features": n_features}


@pytest.mark.slow
def test_features_and_predict_memory_bounded_by_block_and_runtime_linear(tmp_path):
    toy = _toy_model(tmp_path)
    small = _scale_run(tmp_path, 1024, toy)
    large = _scale_run(tmp_path, 2048, toy)
    full_stack_mb = 2048 * 2048 * 4 * large["features"] / 1e6
    # 4x the pixels: memory must not follow the raster (would be ~4x), runtime should (~4x).
    assert large["peak_mb"] < 0.25 * full_stack_mb, (large, full_stack_mb)
    assert large["peak_mb"] < 1.5 * small["peak_mb"] + 50, (small, large)
    ratio = large["seconds"] / small["seconds"]
    assert 1.5 < ratio < 10, (small, large)
