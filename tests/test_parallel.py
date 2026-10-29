"""Parallel imports, blockwise processes and tile workers: bounded, ordered and identical to one process.

Everything runs on synthetic data and local files; spawned worker processes import the package fresh, so
these tests also cover the minimal CI install (no optional extras beyond what the modules themselves need).
"""
from __future__ import annotations

import hashlib
import json
import math
import threading
import time
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.errors import RasterioIOError
from rasterio.transform import from_origin
from test_cov_adapters import SHAPE, _items_client, _PassSigner, _reference, _s1_item, _s2_item

from geocrop import applicability, model, parallel, preflight, s1_stac, stac_import
from geocrop import config as cfgmod

# --------------------------------------------------------------------------- ordered_map


def test_ordered_map_keeps_input_order_whatever_finishes_first():
    def slow_first(i):
        time.sleep(0.05 * (5 - i))          # item 0 is the slowest
        return i * i
    assert list(parallel.ordered_map(slow_first, range(5), workers=5)) == [0, 1, 4, 9, 16]


def test_ordered_map_is_bounded_and_lazy():
    pulled, seen = [], []

    def items():
        for i in range(30):
            pulled.append(i)
            yield i
    for k, value in enumerate(parallel.ordered_map(lambda i: i, items(), workers=2, max_pending=3)):
        seen.append(value)
        assert len(pulled) - (k + 1) <= 3           # never more than max_pending submitted but not yielded
    assert seen == list(range(30))


def test_ordered_map_single_worker_runs_in_the_calling_thread():
    threads = set(parallel.ordered_map(lambda i: threading.get_ident(), range(3), workers=1))
    assert threads == {threading.get_ident()}


def test_ordered_map_reraises_at_the_failing_item_and_stops():
    done = []

    def fn(i):
        if i == 3:
            raise ValueError("boom")
        done.append(i)
        return i
    got = []
    with pytest.raises(ValueError, match="boom"):
        for v in parallel.ordered_map(fn, range(50), workers=2, max_pending=4):
            got.append(v)
    assert got == [0, 1, 2]
    assert len(done) < 50                              # remaining work was cancelled, not run to the end


def test_ordered_map_processes_spawn_and_order():
    assert list(parallel.ordered_map(math.sqrt, [4.0, 9.0, 16.0, 25.0], workers=2, kind="process")) == [2, 3, 4, 5]
    with pytest.raises(ValueError):
        list(parallel.ordered_map(abs, [1], kind="gpu"))


def test_resolve_workers_precedence(monkeypatch):
    assert parallel.resolve_workers(3, n_jobs=8) == 3          # explicit wins, even above the CPU count
    assert parallel.resolve_workers(None, n_jobs=2, cpus=16) == 2
    assert parallel.resolve_workers(None, n_jobs=-1, cpus=16) == parallel.AUTO_WORKER_CAP
    assert parallel.resolve_workers(None, None, cpus=2.5) == 2
    assert parallel.resolve_workers(None, None, cpus=0.5) == 1
    assert parallel.import_workers(None) == parallel.DEFAULT_IMPORT_WORKERS
    assert parallel.import_workers(99) == parallel.MAX_IMPORT_WORKERS and parallel.import_workers(0) == 1


def test_with_backoff_retries_throttling_only_with_growing_delays():
    delays, calls = [], []

    def flaky(x):
        calls.append(x)
        if len(calls) < 3:
            raise RasterioIOError("HTTP response code: 503 Service Unavailable")
        return x
    assert parallel.with_backoff(flaky, base_delay=1.0, sleep=delays.append)(7) == 7
    assert len(calls) == 3 and 1.0 <= delays[0] <= 1.25 and 2.0 <= delays[1] <= 2.5
    boom = []

    def bad(x):
        boom.append(x)
        raise ValueError("not throttling")
    with pytest.raises(ValueError):
        parallel.with_backoff(bad, sleep=delays.append)(1)
    assert len(boom) == 1                                  # other errors are not retried
    always = []

    def throttled(x):
        always.append(x)
        raise RasterioIOError("429 Too Many Requests")
    with pytest.raises(RasterioIOError):
        parallel.with_backoff(throttled, attempts=3, sleep=lambda s: None)(1)
    assert len(always) == 3


def test_available_cpus_moved_to_parallel_and_still_exported_by_preflight():
    assert preflight.available_cpus is parallel.available_cpus
    assert parallel.available_cpus(cgroup_files=())[0] >= 1


# --------------------------------------------------------------------------- config


def test_config_resources_parallel_keys_default_and_validation():
    res = cfgmod.load(None)["resources"]
    assert res["import_workers"] == 4 and res["workers"] is None
    for bad in ({"import_workers": 0}, {"import_workers": 9}, {"import_workers": True}, {"workers": 0},
                {"workers": 1.5}, {"workers": True}):
        with pytest.raises(cfgmod.ConfigError, match="resources"):
            cfgmod._validate_resources({**res, **bad}, cfgmod.DEFAULTS["preflight"])
    cfgmod._validate_resources({**res, "import_workers": 8, "workers": 3}, cfgmod.DEFAULTS["preflight"])


# --------------------------------------------------------------------------- imports


def _tree_hashes(root: Path) -> dict:
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()}


def _s2_items(tmp_path):
    items = [_s2_item(tmp_path / f"i{k}", f"S{k}", dn=1500 + 100 * k, when=f"2023-07-{3 + 2 * k:02d}T09:48:18")
             for k in range(5)]
    items.append(_s2_item(tmp_path / "bad", "BAD", red=500, boa_applied=False, when="2023-07-04T09:48:18"))
    items.append(_s2_item(tmp_path / "cloud", "CLOUD", scl=9, when="2023-07-06T09:48:18"))
    return items


def test_s2_import_is_identical_for_one_and_many_workers_and_isolates_failures(tmp_path, monkeypatch):
    ref = _reference(tmp_path / "grid.tif")
    _items_client(monkeypatch, _s2_items(tmp_path))
    seq = stac_import.import_season(ref, tmp_path / "seq", "2023-07-01", "2023-07-31", min_clear_fraction=0.5,
                                    workers=1)
    par = stac_import.import_season(ref, tmp_path / "par", "2023-07-01", "2023-07-31", min_clear_fraction=0.5,
                                    workers=4)
    strip = lambda rows, root: [{k: (str(Path(v).relative_to(root)) if k == "path" else v) for k, v in r.items()}  # noqa: E731
                                for r in rows]
    assert strip(seq, tmp_path / "seq") == strip(par, tmp_path / "par")
    assert [r["item"] for r in par] == ["S0", "BAD", "S1", "CLOUD", "S2", "S3", "S4"]      # acquisition order
    status = {r["item"]: r["status"].split(":")[0] for r in par}
    assert status["BAD"] == "failed" and status["CLOUD"] == "skipped" and status["S4"] == "imported"
    assert _tree_hashes(tmp_path / "seq") == _tree_hashes(tmp_path / "par")                 # byte-identical files
    again = stac_import.import_season(ref, tmp_path / "par", "2023-07-01", "2023-07-31", min_clear_fraction=0.5)
    assert [r["status"].split(":")[0] for r in again].count("cached") == 5


def test_s2_import_retries_throttled_reads_only_when_parallel(tmp_path, monkeypatch):
    ref = _reference(tmp_path / "grid.tif")
    _items_client(monkeypatch, [_s2_item(tmp_path / "a", "A")])
    real, calls = stac_import.import_item, []

    def throttled(item, *args, **kwargs):
        calls.append(item.id)
        if len(calls) == 1:
            raise RasterioIOError("HTTP error 503 (Slow Down)")
        return real(item, *args, **kwargs)
    monkeypatch.setattr(stac_import, "import_item", throttled)
    monkeypatch.setattr(parallel.time, "sleep", lambda s: None)
    res = stac_import.import_season(ref, tmp_path / "out", "2023-07-01", "2023-07-31", workers=2)
    assert res[0]["status"] == "imported" and calls == ["A", "A"]
    calls.clear()
    res = stac_import.import_season(ref, tmp_path / "out2", "2023-07-01", "2023-07-31", workers=1)
    assert res[0]["status"].startswith("failed:") and calls == ["A"]        # sequential behaviour: no retry


def test_s1_import_is_identical_for_one_and_many_workers(tmp_path, monkeypatch):
    ref = _reference(tmp_path / "grid.tif")
    left, right = np.zeros(SHAPE, bool), np.zeros(SHAPE, bool)
    left[:, :4], right[:, 4:] = True, True
    items = [_s1_item(tmp_path / "l", "L", cover=left, vv=0.1),
             _s1_item(tmp_path / "r", "R", cover=right, vv=0.2, when="2023-07-03T16:30:25"),
             _s1_item(tmp_path / "f", "F", vv=0.3, when="2023-07-09T16:30:00", absolute_orbit=48100),
             _s1_item(tmp_path / "g", "G", vv=0.4, when="2023-07-15T16:30:00", absolute_orbit=48200),
             _s1_item(tmp_path / "m", "M", when="2023-07-21T16:30:00", absolute_orbit=48300, pols=("VV",))]
    monkeypatch.setattr(s1_stac, "search", lambda *a, **k: items)
    seq = s1_stac.import_season(ref, tmp_path / "seq", "2023-07-01", "2023-07-31", signer=_PassSigner(), workers=1)
    par = s1_stac.import_season(ref, tmp_path / "par", "2023-07-01", "2023-07-31", signer=_PassSigner(), workers=4)
    assert [r["item"] for r in par] == [["L", "R"], ["F"], ["G"], ["M"]]
    assert [r["status"].split(":")[0] for r in par] == ["imported", "imported", "imported", "failed"]
    assert [r["status"] for r in seq] == [r["status"] for r in par]
    assert _tree_hashes(tmp_path / "seq") == _tree_hashes(tmp_path / "par")


def test_s1_import_service_outage_propagates_from_the_pool(tmp_path, monkeypatch):
    ref = _reference(tmp_path / "grid.tif")

    class Down:
        def sign(self, href):
            raise s1_stac.ServiceUnavailable("token service down")
    monkeypatch.setattr(s1_stac, "search", lambda *a, **k: [_s1_item(tmp_path / f"x{k}", f"X{k}", absolute_orbit=1 + k)
                                                              for k in range(4)])
    with pytest.raises(s1_stac.ServiceUnavailable):
        s1_stac.import_season(ref, tmp_path / "s1", "2023-07-01", "2023-07-31", signer=Down(), workers=3)


def test_signer_fetches_one_token_for_concurrent_threads(monkeypatch):
    import requests
    calls = []

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"token": "sig=abc"}

    def get(*a, **k):
        calls.append(1)
        time.sleep(0.05)
        return Response()
    monkeypatch.setattr(requests, "get", get)
    monkeypatch.setitem(__import__("sys").modules, "planetary_computer", None)
    signer = s1_stac.Signer()
    hrefs = list(parallel.ordered_map(lambda i: signer.sign("https://a.blob.core.windows.net/x.tif"), range(8), 4))
    assert len(calls) == 1 and all(h.endswith("sig=abc") for h in hrefs)


# --------------------------------------------------------------------------- AOA and prediction


def _aoa_model(n_features, n_train=300, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_train, n_features))
    return applicability.fit(X, rng.integers(0, 3, n_train), [f"f{i}" for i in range(n_features)])


def test_blas_nearest_neighbour_equals_the_kdtree(monkeypatch):
    m = _aoa_model(24)
    rng = np.random.default_rng(1)
    Q = np.vstack([rng.normal(size=(500, 24)), 6 * rng.normal(size=(50, 24)), m.train_points[:20] * m.std + m.mean])
    Z = m.transform(Q)
    exact = m._tree.query(Z, k=1)[0]
    monkeypatch.delenv("GEOCROP_AOA_SEARCH", raising=False)
    np.testing.assert_allclose(m.nearest_distance(Z), exact, rtol=1e-12, atol=1e-12)
    assert (m.nearest_distance(Z[-20:]) < 1e-9).all()                  # training points are at distance 0
    monkeypatch.setenv("GEOCROP_AOA_SEARCH", "kdtree")
    np.testing.assert_array_equal(m.nearest_distance(Z), exact)
    assert m.nearest_distance(Z[:0]).shape == (0,)


def test_blas_search_is_exact_for_duplicated_and_nearly_identical_training_points():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(120, 20))
    X = np.vstack([X, X[:30], X[:30] + 1e-9, X[40:50] + 1e-6])       # duplicates: rounding cannot rank these
    m = applicability.fit(X, rng.integers(0, 3, len(X)), [f"f{i}" for i in range(20)])
    Z = np.vstack([m.train_points[:40] + 1e-7, rng.normal(size=(200, 20)) * 0.3])
    exact = m._tree.query(Z, k=1)[0]
    np.testing.assert_allclose(m.nearest_distance(Z), exact, rtol=1e-12, atol=1e-12)


def _feature_stack(path, size, n_features, seed=0, nan_rows=0):
    rng = np.random.default_rng(seed)
    data = rng.normal(size=(n_features, size, size)).astype("float32")
    data[:, :nan_rows] = -9999.0
    with rasterio.open(path, "w", driver="GTiff", width=size, height=size, count=n_features, dtype="float32",
                       crs="EPSG:32634", transform=from_origin(500000, 4900000, 10, 10), nodata=-9999.0) as ds:
        ds.write(data)
        ds.descriptions = tuple(f"f{i}" for i in range(n_features))
    return path


def test_map_aoa_processes_write_exactly_the_single_process_raster(tmp_path, monkeypatch):
    monkeypatch.setattr(applicability, "POOL_MIN_PIXELS", 1)
    stack = _feature_stack(tmp_path / "s.tif", 96, 18, nan_rows=5)
    m = _aoa_model(18)
    applicability.map_aoa(stack, m, tmp_path / "one.tif", block_size=32, workers=1)
    applicability.map_aoa(stack, m, tmp_path / "three.tif", block_size=32, workers=3)
    with rasterio.open(tmp_path / "one.tif") as a, rasterio.open(tmp_path / "three.tif") as b:
        one, three = a.read(), b.read()
        assert a.descriptions == b.descriptions == ("dissimilarity_index", "inside_aoa")
    np.testing.assert_array_equal(one, three)                           # max abs diff 0
    assert (one[:, :5] == -9999.0).all() and (one[0, 5:] > 0).all()
    assert (tmp_path / "one.tif").read_bytes() == (tmp_path / "three.tif").read_bytes()


def test_map_aoa_small_rasters_stay_in_process(tmp_path, monkeypatch):
    stack = _feature_stack(tmp_path / "s.tif", 40, 18)
    started = []
    monkeypatch.setattr(applicability, "ordered_map", lambda *a, **k: started.append(1))
    applicability.map_aoa(stack, _aoa_model(18), tmp_path / "o.tif", block_size=16, workers=4)
    assert started == []


def _trained_model(path, n_features=3):
    from joblib import dump
    from sklearn.ensemble import RandomForestClassifier

    from geocrop import contracts
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, n_features))
    y = np.where(X[:, 0] + X[:, 1] > 0, "a", np.where(X[:, 2] > 0, "b", "c"))
    clf = RandomForestClassifier(n_estimators=12, random_state=0, n_jobs=2).fit(X, y)
    dump(contracts.stamp("model", {"model": clf, "features": [f"f{i}" for i in range(n_features)]}), path)
    applicability.fit(X, rng.integers(0, 3, 200), [f"f{i}" for i in range(n_features)]).to_json(
        Path(path).with_name("aoa.json"))


@pytest.mark.parametrize("smoothing", [None, {"enabled": True, "window": 5}])
def test_predict_processes_equal_single_process(tmp_path, monkeypatch, smoothing):
    monkeypatch.setattr(applicability, "POOL_MIN_PIXELS", 1)
    stack = _feature_stack(tmp_path / "s.tif", 80, 3, nan_rows=3)
    _trained_model(tmp_path / "model.joblib")
    r1 = model.predict(stack, tmp_path / "model.joblib", tmp_path / "p1", block_size=32, smoothing=smoothing,
                       workers=1)
    r3 = model.predict(stack, tmp_path / "model.joblib", tmp_path / "p3", block_size=32, smoothing=smoothing,
                       workers=3)
    for name in ("classes", "probabilities", "uncertainty", "aoa"):
        with rasterio.open(r1[name]) as a, rasterio.open(r3[name]) as b:
            x, y = a.read(), b.read()
        assert np.abs(x.astype("float64") - y).max() == 0 if name in ("classes", "aoa") else \
            np.abs(x.astype("float64") - y).max() <= 1e-6, name
    assert json.loads(Path(r3["manifest"]).read_text())["classes"] == ["a", "b", "c"]


def test_predict_worker_resolution_follows_config_and_model_n_jobs():
    from geocrop import pipeline
    pl = pipeline.Runner.__new__(pipeline.Runner)
    pl.cfg = {"resources": {"workers": None}, "model": {"n_jobs": -1}}
    assert pl._block_workers() == parallel.auto_workers()
    pl.cfg = {"resources": {"workers": 2}, "model": {"n_jobs": 8}}
    assert pl._block_workers() == 2
    pl.cfg = {"resources": {"workers": None}, "model": {"n_jobs": 3}}
    assert pl._block_workers() == 3


# --------------------------------------------------------------------------- segmentation tiles


def test_obia_tile_workers_give_the_same_segment_table(tmp_path):
    pytest.importorskip("skimage")
    gpd = pytest.importorskip("geopandas")
    from geocrop import segmentation
    rng = np.random.default_rng(4)
    coarse = rng.uniform(0, 1, (11, 11))
    field = np.kron(coarse, np.ones((20, 20)))[:200, :200]
    data = np.stack([field + rng.normal(0, 0.01, field.shape) for _ in range(3)]).astype("float32")
    path = tmp_path / "s.tif"
    with rasterio.open(path, "w", driver="GTiff", width=200, height=200, count=3, dtype="float32", crs="EPSG:32634",
                       transform=from_origin(500000, 4900000, 10, 10), nodata=-9999.0) as ds:
        ds.write(data)
    one = segmentation.obia(path, tmp_path / "one.gpkg", tile_size=70, segments_per_km2=2000, workers=1)
    three = segmentation.obia(path, tmp_path / "three.gpkg", tile_size=70, segments_per_km2=2000, workers=3)
    assert one["seam_merges"] > 0 and one == {**three, "path": one["path"]}
    a, b = gpd.read_file(tmp_path / "one.gpkg"), gpd.read_file(tmp_path / "three.gpkg")
    assert a.drop(columns="geometry").equals(b.drop(columns="geometry"))
    assert all(g.equals_exact(h, 0) for g, h in zip(a.geometry, b.geometry))


# --------------------------------------------------------------------------- preflight


def test_preflight_credits_block_workers_for_the_aoa_map():
    cfg = cfgmod.load(None)
    layout = {"pixels": 5_000_000, "grid": {"width": 2236, "height": 2236, "res": (10, 10)},
              "window_features": 97, "timeseries": {"features": 0, "scenes": 0, "series": 0},
              "s2_by_window": {}, "s1_by_window": {}}
    one = preflight.estimate(cfg, layout, n_samples=1000, n_classes=4, cpus=1)["runtime_hours_rough"]["predict"]
    four = preflight.estimate(cfg, layout, n_samples=1000, n_classes=4, cpus=8)["runtime_hours_rough"]["predict"]
    assert four < one
