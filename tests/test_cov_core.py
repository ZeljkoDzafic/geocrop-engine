"""Edge and error paths of the result-critical core modules (validation, uncertainty, applicability,
splits, metrics, cleanup, plugins, contracts) on small synthetic inputs."""
from __future__ import annotations

import json
import os
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Point, box

from geocrop import (
    applicability,
    cleanup,
    config,
    contracts,
    metrics,
    plugins,
    splits,
    uncertainty,
    validation,
)

CRS = "EPSG:32634"
X0, Y0, PX = 500_000.0, 5_000_000.0, 10.0


# ----------------------------------------------------------------------------------------------- validation
def _prediction(root: Path, classes: np.ndarray, probabilities: bool = True, prob_names=("a", "b")) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    h, w = classes.shape
    prof = {"driver": "GTiff", "height": h, "width": w, "crs": CRS, "transform": from_origin(X0, Y0, PX, PX)}
    with rasterio.open(root / "classes.tif", "w", count=1, dtype="uint8", nodata=255, **prof) as ds:
        ds.write(classes.astype("uint8"), 1)
    (root / "class_legend.json").write_text(json.dumps({"0": "a", "1": "b"}), encoding="utf8")
    if probabilities:
        p = np.stack([(classes == 0).astype("float32") * 0.8 + 0.1, (classes == 1).astype("float32") * 0.8 + 0.1])
        with rasterio.open(root / "probabilities.tif", "w", count=2, dtype="float32", **prof) as ds:
            ds.write(p)
            for i, name in enumerate(prob_names, 1):
                ds.set_band_description(i, name)
    return root


def _pt(row: int, col: int) -> Point:
    return Point(X0 + (col + 0.5) * PX, Y0 - (row + 0.5) * PX)


def _obs(points, labels, dates=None, crs=CRS):
    data = {"class_name": labels}
    if dates is not None:
        data["observed_at"] = dates
    return gpd.GeoDataFrame(data, geometry=points, crs=crs)


@pytest.fixture
def classes():
    c = np.zeros((10, 10), dtype="uint8")
    c[:, 5:] = 1
    c[0:3, 0:3] = 255     # nodata corner
    c[9, 9] = 7           # class missing from the legend
    return c


def test_evaluate_points_input_errors(tmp_path, classes):
    pred = _prediction(tmp_path / "p", classes)
    obs = _obs([_pt(5, 1)], ["a"])
    with pytest.raises(ValueError, match="crop"):
        validation.evaluate_points(pred, obs, tmp_path / "v", label_field="crop")
    with pytest.raises(ValueError, match="CRS"):
        validation.evaluate_points(pred, obs.set_crs(None, allow_override=True), tmp_path / "v")
    with pytest.raises(ValueError, match="year"):
        validation.evaluate_points(pred, obs, tmp_path / "v", year=2023, date_field=None)
    with pytest.raises(ValueError, match="year"):
        validation.evaluate_points(pred, obs, tmp_path / "v", year=2023)          # no observed_at column
    # every point is outside the raster, on nodata or on a class missing from the legend
    bad = _obs([Point(X0 - 1000, Y0 + 1000), _pt(1, 1), _pt(9, 9)], ["a", "a", "b"])
    with pytest.raises(ValueError, match="No valid observations"):
        validation.evaluate_points(pred, bad, tmp_path / "v", bootstrap=0)
    # a GNSS window entirely on nodata yields no vote either
    with pytest.raises(ValueError, match="No valid observations"):
        validation.evaluate_points(pred, _obs([_pt(1, 1)], ["a"]), tmp_path / "v", window_px=1, bootstrap=0)


def test_evaluate_points_skips_invalid_points_and_maps_labels(tmp_path, classes):
    pred = _prediction(tmp_path / "p", classes)
    pts = [_pt(5, 1), _pt(6, 2), _pt(5, 7), _pt(6, 8), Point(X0 - 50, Y0), _pt(1, 1), _pt(9, 9), _pt(7, 7)]
    labels = ["A", "A", "B", "B", "A", "A", "B", "A"]
    dates = ["2023-06-01"] * 7 + ["2022-06-01"]            # last point is from another season
    # observations in another CRS are reprojected to the prediction grid
    obs = _obs(pts, labels, dates).to_crs("EPSG:4326")
    r = validation.evaluate_points(pred, obs, tmp_path / "v", year=2023, label_mapping={"A": "a", "B": "b"},
                                   bootstrap=20)
    sel = r["selection"]
    assert sel["input"] == 8 and sel["after_year_filter"] == 7
    assert sel["evaluated"] == 4 == r["accuracy"]["n"]
    assert r["accuracy"]["overall_accuracy"] == 1.0 and sel["observed_classes_not_in_model"] == []
    assert 0 <= r["calibration"]["ece"] <= 1 and r["reject_curve"]
    lo, hi = r["bootstrap"]["overall_accuracy_ci"]
    assert lo <= hi
    for name in ("observations_evaluated.csv", "confusion_matrix.csv", "validation_report.json"):
        assert (tmp_path / "v" / name).is_file()


def test_evaluate_points_without_probabilities_or_known_classes(tmp_path, classes):
    obs = _obs([_pt(5, 1), _pt(5, 7)], ["x", "y"])          # labels the model never saw
    no_prob = _prediction(tmp_path / "p1", classes, probabilities=False)
    r = validation.evaluate_points(no_prob, obs, tmp_path / "v1", bootstrap=0)
    assert "calibration" not in r and "bootstrap" not in r
    assert r["selection"]["observed_classes_not_in_model"] == ["x", "y"]
    assert r["accuracy"]["overall_accuracy"] == 0.0
    with_prob = _prediction(tmp_path / "p2", classes)
    r = validation.evaluate_points(with_prob, obs, tmp_path / "v2", bootstrap=0)
    assert "calibration" not in r and "reject_curve" not in r
    import pandas as pd
    table = pd.read_csv(tmp_path / "v2" / "observations_evaluated.csv")
    assert table.max_probability.between(0, 1).all()


def test_window_vote_majority_ignores_nodata(tmp_path, classes):
    pred = _prediction(tmp_path / "p", classes, probabilities=False)
    with rasterio.open(pred / "classes.tif") as ds:
        assert validation._window_vote(ds, 1, 1, 1) is None
        assert validation._window_vote(ds, 2, 2, 1) == 0     # one valid pixel among nodata
        assert validation._window_vote(ds, 5, 8, 2) == 1
        assert validation._window_vote(ds, 0, 0, 0) is None  # boundless corner


def test_mapped_area_is_pixel_count_times_area(tmp_path, classes):
    pred = _prediction(tmp_path / "p", classes, probabilities=False)
    areas = validation.mapped_area(pred / "classes.tif")
    assert areas["a"] == pytest.approx(((classes == 0).sum()) * PX * PX)
    assert areas["b"] == pytest.approx(((classes == 1).sum()) * PX * PX)


# ---------------------------------------------------------------------------------------------- uncertainty
def _proba(n=60, k=3, seed=0):
    rng = np.random.default_rng(seed)
    p = rng.dirichlet(np.ones(k), size=n)
    y = [str(c) for c in p.argmax(axis=1)]
    return p, y, [str(c) for c in range(k)]


def test_temperature_is_identity_at_one_and_preserves_ranking():
    p, y, classes = _proba()
    assert np.allclose(uncertainty.apply_temperature(p, 1.0), p / p.sum(axis=1, keepdims=True))
    for t in (0.5, 2.0):
        q = uncertainty.apply_temperature(p, t)
        assert np.allclose(q.sum(axis=1), 1) and (q.argmax(axis=1) == p.argmax(axis=1)).all()
    # labels equal to the argmax reward sharpening: T < 1, within the search bounds exp(+-3)
    t = uncertainty.fit_temperature(p, y, classes)
    assert np.exp(-3) - 1e-3 <= t < 1


def test_conformal_argument_errors_and_marginal_thresholds():
    p, y, classes = _proba()
    with pytest.raises(ValueError, match="sparse_class_policy"):
        uncertainty.conformal_thresholds(p, y, classes, sparse_class_policy="drop")
    for bad in ((0.0,), (1.0,), (float("nan"),)):
        with pytest.raises(ValueError, match="alphas"):
            uncertainty.conformal_thresholds(p, y, classes, alphas=bad)
    res = uncertainty.conformal_thresholds(p, y, classes, alphas=(0.1, 0.2), mondrian=False)
    for a in ("0.1", "0.2"):
        assert len(set(res["thresholds"][a].values())) == 1           # one pooled threshold for all classes
    assert res["thresholds"]["0.1"]["0"] >= res["thresholds"]["0.2"]["0"]   # lower alpha, wider sets
    assert res["pooled_fallback_classes"] == {}


def test_conformal_sparse_classes_and_prediction_sets():
    p, y, classes = _proba(n=40)
    y = ["0"] * 39 + ["1"]                                       # class 2 absent, class 1 scarce
    pooled = uncertainty.conformal_thresholds(p, y, classes, alphas=(0.1,))
    include = uncertainty.conformal_thresholds(p, y, classes, alphas=(0.1,), sparse_class_policy="include")
    assert set(pooled["pooled_fallback_classes"]["0.1"]) == {"1", "2"}
    assert include["thresholds"]["0.1"]["2"] == 1.0 and include["always_include_classes"]["0.1"] == ["1", "2"]
    sets = uncertainty.prediction_sets(p, classes, include["thresholds"]["0.1"])
    assert all({"1", "2"} <= set(s) for s in sets)               # threshold 1 always includes the class
    cov = uncertainty.coverage(sets, y)
    assert 0 <= cov["coverage"] <= 1 and cov["empty_rate"] == 0 and cov["mean_set_size"] >= 2
    tiny = uncertainty.conformal_thresholds(p[:1], ["0"], classes, alphas=(0.05,))
    assert tiny["thresholds"]["0.05"]["0"] == 1.0                # n too small for 95 %: conservative


def test_polygon_label_checks_and_suspects():
    with pytest.raises(ValueError, match="one polygon ID"):
        uncertainty.validate_polygon_labels(["a", "b"], [1])
    with pytest.raises(ValueError, match="mixed labels"):
        uncertainty.validate_polygon_labels(["a", "b"], [1, "1"])
    proba = np.array([[0.9, 0.1], [0.8, 0.2], [0.05, 0.95], [0.1, 0.9]])
    poly = uncertainty.polygon_oof(proba, ["a", "a", "a", "b"], [1, 1, 2, 3], ["a", "b"])
    assert len(poly) == 3 and set(poly.predicted) <= {"a", "b"}
    sus = uncertainty.suspect_labels(poly)
    assert list(sus.polygon) == ["2"] and sus.p_own.iloc[0] < 0.2


# -------------------------------------------------------------------------------------------- applicability
def test_aoa_fit_subsamples_and_handles_zero_importance():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(300, 3))
    folds = np.arange(300) % 3
    a = applicability.fit(X, folds, ["f1", "f2", "f3"], importance=[0, 0, np.nan], max_points=100)
    assert len(a.train_points) == 100 and np.allclose(a.weights, 1)
    assert a.threshold > 0
    # dissimilarity is zero at a training point and grows with distance
    centre = a.dissimilarity(X.mean(axis=0, keepdims=True))[0]
    far = a.dissimilarity(X.mean(axis=0, keepdims=True) + 50)[0]
    assert far > a.threshold >= 0 and far > centre


@pytest.mark.parametrize("X,folds,match", [
    (np.random.default_rng(0).normal(size=(20, 2)), np.zeros(20), "two cross-validation folds"),
    (np.ones((20, 2)), np.arange(20) % 2, "identical"),
    (np.r_[np.zeros((1, 2)) * np.nan, np.ones((5, 2))], np.arange(6) % 2, "finite"),
    (np.ones((1, 2)), np.zeros(1), "at least two rows"),
    (np.ones((4, 2)), np.zeros(3), "one fold id per row"),
])
def test_aoa_fit_rejects_degenerate_training_data(X, folds, match):
    with pytest.raises(ValueError, match=match):
        applicability.fit(X, folds, ["f1", "f2"])


def test_map_aoa_nodata_block_and_missing_feature(tmp_path):
    rng = np.random.default_rng(2)
    X = rng.normal(size=(60, 2))
    a = applicability.fit(X, np.arange(60) % 3, ["f1", "f2"])
    a2 = applicability.Applicability.from_json(a.to_json(tmp_path / "aoa.json"))
    assert a2.threshold == a.threshold and a2.features == ["f1", "f2"]
    data = np.full((2, 8, 8), -9999.0, dtype="float32")         # left 4 columns nodata (one whole block)
    data[:, :, 4:] = rng.normal(size=(2, 8, 4))
    prof = {"driver": "GTiff", "height": 8, "width": 8, "count": 2, "dtype": "float32", "crs": CRS,
            "transform": from_origin(X0, Y0, PX, PX), "nodata": -9999.0}
    with rasterio.open(tmp_path / "stack.tif", "w", **prof) as ds:
        ds.write(data)
        ds.set_band_description(1, "f1")
        ds.set_band_description(2, "f2")
    out = applicability.map_aoa(tmp_path / "stack.tif", a2, tmp_path / "o" / "aoa.tif", block_size=4)
    with rasterio.open(out) as ds:
        di, flag = ds.read(1), ds.read(2)
        nd = ds.nodata
    assert (di[:, :4] == nd).all() and (flag[:, :4] == nd).all()
    assert (di[:, 4:] >= 0).all() and set(np.unique(flag[:, 4:])) <= {0.0, 1.0}
    other = applicability.Applicability(a.mean, a.std, a.weights, a.train_points, a.mean_distance, a.threshold,
                                        ["f1", "missing"])
    with pytest.raises(ValueError, match="lacks AOA features"):
        applicability.map_aoa(tmp_path / "stack.tif", other, tmp_path / "o" / "x.tif")


# --------------------------------------------------------------------------------------------------- splits
def _polys(n=40, crs=CRS, step=3000.0):
    geoms = [box(X0 + i * step, Y0, X0 + i * step + 100, Y0 + 100) for i in range(n)]
    return gpd.GeoDataFrame({"class_name": ["a", "b"] * (n // 2)}, geometry=geoms, crs=crs)


def test_block_ids_and_fold_errors(tmp_path):
    with pytest.raises(ValueError, match="projected"):
        splits.block_ids(_polys().to_crs("EPSG:4326"), 1000)
    with pytest.raises(ValueError, match="projected"):
        splits.block_ids(_polys().set_crs(None, allow_override=True), 1000)
    with pytest.raises(ValueError, match="positive"):
        splits.block_ids(_polys(), 0)
    with pytest.raises(ValueError, match="Missing"):
        splits.assign_folds(_polys(), label_field="crop")
    with pytest.raises(ValueError, match="two spatial blocks"):
        splits.assign_folds(_polys(), block_m=1e9)
    res = splits.assign_folds(_polys(), block_m=5000, n_splits=4)    # no output file written
    assert res["n_folds"] == 4 and sum(res["polygons_per_fold"].values()) == 40
    path = tmp_path / "f" / "folds.geojson"
    splits.assign_folds(_polys(), path, block_m=5000, n_splits=4)
    assert set(gpd.read_file(path).fold) == {0, 1, 2, 3}


def test_strategy_groups_errors_and_values():
    with pytest.raises(ValueError, match="cv strategy"):
        splits.check_strategy("random")
    with pytest.raises(ValueError, match="block ids"):
        splits.strategy_groups("spatial_blocks")
    assert list(splits.strategy_groups("spatial_blocks", blocks=["1:1", "2:2"])) == ["1:1", "2:2"]
    with pytest.raises(ValueError, match="year metadata"):
        splits.strategy_groups("region_year", region=["r1"])
    with pytest.raises(ValueError, match="more than one region"):
        splits.strategy_groups("region", region=["r1", "r2"], polygon=[1, 1])
    g = splits.strategy_groups("region_year", region=["r1", "r2"], year=[2021, 2022], polygon=[1, 2])
    assert list(g) == ["r1|2021", "r2|2022"]
    assert list(splits.strategy_groups("year", year=[2020, 2021])) == ["2020", "2021"]


def test_leave_one_group_out_properties():
    with pytest.raises(ValueError, match="at least 2 distinct"):
        list(splits.leave_one_group_out(["r1", "r1"]))
    with pytest.raises(ValueError, match="two regions and two years"):
        list(splits.leave_one_group_out(["r1|2020", "r1|2021"], "region_year"))
    groups = np.array(["r1|2020", "r1|2021", "r2|2020", "r2|2021", "r3|2022"] * 2)
    folds = list(splits.leave_one_group_out(groups, "region_year"))
    assert {name for name, _, _ in folds} == set(groups)
    for name, tr, te in folds:
        region, year = name.split("|")
        assert all(g == name for g in groups[te])
        assert not any(g.startswith(region + "|") or g.endswith("|" + year) for g in groups[tr])
    # a cell that shares its region or year with every other cell has no training data and is skipped
    tight = np.array(["r1|2020", "r1|2021", "r2|2020"])
    assert [n for n, _, _ in splits.leave_one_group_out(tight, "region_year")] == ["r1|2021", "r2|2020"]
    plain = list(splits.leave_one_group_out(["a", "b", "a"], "region"))
    assert [(n, list(tr), list(te)) for n, tr, te in plain] == [("a", [1], [0, 2]), ("b", [0, 2], [1])]


# -------------------------------------------------------------------------------------------------- metrics
def test_metrics_errors_and_degenerate_inputs():
    with pytest.raises(ValueError, match="Empty"):
        metrics.from_confusion([[0, 0], [0, 0]], ["a", "b"])
    single = metrics.summary(["a"] * 5, ["a"] * 5)
    assert single["overall_accuracy"] == 1.0 and single["kappa"] == 1.0
    absent = metrics.summary(["a", "a"], ["a", "b"], labels=["a", "b", "c"])
    assert absent["per_class"]["c"]["producers_accuracy"] is None and absent["per_class"]["b"]["users_accuracy"] == 0
    with pytest.raises(ValueError, match="every label"):
        metrics.area_adjusted([[5, 1], [1, 5]], ["a", "b"], {"a": 1.0})
    with pytest.raises(ValueError, match="two sample units"):
        metrics.area_adjusted([[1, 0], [0, 5]], ["a", "b"], {"a": 1.0, "b": 1.0})


def test_area_adjusted_properties():
    cm = [[40, 5, 0], [3, 30, 0], [0, 0, 0]]       # class c is mapped (area>0) but never sampled as reference
    res = metrics.area_adjusted(np.array(cm), ["a", "b", "c"], {"a": 600.0, "b": 400.0, "c": 0.0})
    adj = sum(v["adjusted_area"] for v in res["per_class"].values())
    assert adj == pytest.approx(res["total_area"]) and 0 <= res["overall_accuracy"] <= 1
    assert res["per_class"]["c"]["producers_accuracy"] is None
    for v in res["per_class"].values():
        lo, hi = v["adjusted_area_ci95"]
        assert lo <= v["adjusted_area"] <= hi


def test_bootstrap_groups_calibration_reject_and_mcnemar():
    y = np.array(["a", "b"] * 20)
    pred = y.copy()
    pred[:6] = "b"
    b = metrics.bootstrap(y, pred, groups=np.repeat(np.arange(8), 5), n_boot=30)
    assert b["units"] == 8 and b["overall_accuracy_ci"][0] <= b["overall_accuracy_ci"][1] <= 1
    probs = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
    cal = metrics.calibration(probs, ["a", "b", "b"], ["a", "b"], bins=5)
    assert 0 <= cal["ece"] <= 1 and sum(r["count"] for r in cal["bins"]) == 3
    rows = metrics.reject_curve([0.2, 0.9], [False, True], thresholds=[0.0, 0.5, 0.99])
    assert [r["coverage"] for r in rows] == [1.0, 0.5, 0.0] and rows[-1]["accuracy"] is None
    assert metrics.mcnemar(y, y, y)["test"] == "none"
    exact = metrics.mcnemar(y, y, pred)
    assert exact["test"] == "exact binomial" and exact["a_only_correct"] == int((pred != y).sum())
    big = np.array(["a"] * 100)
    chi = metrics.mcnemar(big, big, np.array(["b"] * 30 + ["a"] * 70))
    assert chi["test"].startswith("chi-square") and chi["p_value"] < 0.001


# -------------------------------------------------------------------------------------------------- cleanup
def _clean_cfg(tmp_path: Path) -> dict:
    root = tmp_path / "study"
    paths = {"s2_scenes": root / "s2", "s1_scenes": root / "s1", "composites": root / "comp",
             "features": root / "feat" / "stack.tif", "training": root / "ref" / "training.gpkg",
             "parcels": root / "ref" / "parcels.gpkg", "observations": root / "ref" / "obs.gpkg",
             "output": root / "run"}
    return config.load(overrides={"paths": {k: str(v) for k, v in paths.items()}})


def test_quarantine_partial_edge_cases(tmp_path, monkeypatch):
    out = tmp_path / "run"
    old, new = tmp_path / "old.tif", tmp_path / "new.tif"
    old.write_text("old")
    os.utime(old, ns=(1, 1))
    new.write_text("new")
    assert cleanup.quarantine_partial([old, tmp_path / "missing.tif"], 10, out, "s") == []
    assert not (out / cleanup.QUARANTINE).exists()

    def refuse(src, dst):
        raise OSError("cross-device link")
    monkeypatch.setattr(cleanup.os, "replace", refuse)
    moved = cleanup.quarantine_partial([new], 10, out, "s")
    assert moved == [{"path": str(new), "quarantined": None, "deleted": True}] and not new.exists()

    import errno
    locked = tmp_path / "locked.tif"
    locked.write_text("x")
    real_stat = type(locked).stat

    def stat(self, *a, **k):
        if self.name == "locked.tif":
            raise OSError(errno.EACCES, "permission denied")
        return real_stat(self, *a, **k)
    monkeypatch.setattr(type(locked), "stat", stat)
    moved = cleanup.quarantine_partial([locked], 0, out, "s")
    assert "permission denied" in moved[0]["error"]


def test_clean_plan_results_intermediates_and_errors(tmp_path, monkeypatch):
    cfg = _clean_cfg(tmp_path)
    out, stack = Path(cfg["paths"]["output"]), Path(cfg["paths"]["features"])
    assert cleanup.plan(cfg, True, True) == []                   # nothing exists yet
    for d in ("model", "prediction"):
        (out / d).mkdir(parents=True)
        (out / d / "x.bin").write_bytes(b"1234")
    (out / "samples.npz").write_bytes(b"s")
    stack.parent.mkdir(parents=True)
    for f in (stack, stack.with_name("stack_quality.tif"), stack.with_name("stack_quality.json")):
        f.write_bytes(b"q")
    (stack.parent / "leftover.tmp").write_text("x")
    items = {Path(i["path"]).name: i for i in cleanup.plan(cfg, all_intermediates=True, include_results=True)}
    assert items["model"]["category"] == "result" and items["model"]["bytes"] == 4
    assert items["samples.npz"]["category"] == items["stack_quality.tif"]["category"] == "intermediate"
    assert items["leftover.tmp"]["category"] == "temporary"
    assert "model" not in {Path(i["path"]).name for i in cleanup.plan(cfg, all_intermediates=True)}

    real_rmtree = cleanup.shutil.rmtree

    def flaky(path, *a, **k):
        if Path(path).name == "model":
            raise OSError("busy")
        return real_rmtree(path, *a, **k)
    monkeypatch.setattr(cleanup.shutil, "rmtree", flaky)
    report = cleanup.clean(cfg, dry_run=False, all_intermediates=True, include_results=True)
    assert [Path(e["path"]).name for e in report["errors"]] == ["model"]
    assert not (out / "prediction").exists() and (out / "model").exists() and not stack.exists()
    assert report["bytes"] == sum(i["bytes"] for i in report["removed"])


def test_size_tolerates_unreadable_paths(tmp_path, monkeypatch):
    f = tmp_path / "a"
    f.write_bytes(b"12345")
    assert cleanup._size(f) == 5 and cleanup._size(tmp_path) == 5

    def boom(self, *a, **k):
        raise OSError("gone")
    monkeypatch.setattr(type(f), "stat", boom)
    assert cleanup._size(f) == 0


# -------------------------------------------------------------------------------------------------- plugins
def test_registry_registration_errors_and_decorator(monkeypatch):
    monkeypatch.setattr(plugins, "_entry_points", lambda group: [])
    reg = plugins.Registry("thing", "geocrop.things", plugins._callable("thing"))

    @reg.register("deco")
    def factory():
        return 1
    assert reg.get("deco") is factory and reg.source("deco") == "code" and "deco" in reg
    for bad in ("", "  ", 3):
        with pytest.raises(ValueError, match="non-empty"):
            reg.register(bad, factory)
    with pytest.raises(ValueError, match="already registered"):
        reg.register("deco", factory)
    reg.register_lazy("lazy", "json:dumps")
    with pytest.raises(ValueError, match="already registered"):
        reg.register_lazy("lazy", "json:loads")
    assert reg.get("lazy") is json.dumps and reg.declared_names() == ["deco", "lazy"]
    with pytest.raises(TypeError, match="callable"):
        reg.register("x", 42)
    with pytest.raises(KeyError, match="Unknown thing 'nope'"):
        reg.get("nope")
    reg.unregister("deco")
    assert "deco" not in reg and repr(reg).startswith("Registry('thing'")


def test_entry_point_metadata_failure_only_warns(monkeypatch):
    def broken(group):
        raise RuntimeError("corrupt METADATA")
    monkeypatch.setattr(plugins.metadata, "entry_points", broken)
    with pytest.warns(plugins.PluginWarning, match="corrupt METADATA"):
        assert plugins._entry_points("geocrop.anything") == []


def test_rediscovery_keeps_entry_point_registrations(monkeypatch):
    from importlib import metadata as md
    ep = md.EntryPoint("dumps", "json:dumps", "geocrop.things")
    monkeypatch.setattr(plugins, "_entry_points", lambda group: [ep])
    reg = plugins.Registry("thing", "geocrop.things", plugins._callable("thing"))
    assert reg.get("dumps") is json.dumps
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")                          # a re-discovered entry point is not a conflict
        reg.discover(force=True)
    assert reg.source("dumps").startswith("entry point")


def test_contract_checks_for_indices_segmenters_and_providers():
    from geocrop import indices
    ndvi = indices.get("NDVI")
    assert plugins._index("NDVI", lambda: ndvi) is ndvi                 # zero-argument factory
    with pytest.raises(TypeError, match="missing"):
        plugins._index("bad", object())
    with pytest.raises(ValueError, match="names must match"):
        plugins._index("OTHER", ndvi)

    class Seg:
        def segment(self, raster, output, **params):
            return {"path": output, "segments": 0}
    assert isinstance(plugins._segmenter("seg", Seg), Seg)
    with pytest.raises(TypeError, match="segment"):
        plugins._segmenter("seg", object())

    class Provider:
        name, dims, licence, weights, contract = "p", 4, "MIT", {}, {}

        def load_model(self):
            return None
        embed_pixels = embed_parcels = load_model
    assert isinstance(plugins._feature_provider("p", Provider), Provider)
    with pytest.raises(ValueError, match="names must match"):
        plugins._feature_provider("q", Provider)
    with pytest.raises(TypeError, match="missing"):
        plugins._feature_provider("p", object())


def test_describe_lists_every_registry():
    desc = plugins.describe()
    assert set(desc) == set(plugins.REGISTRIES)
    assert desc["exporters"]["cog"] == plugins.BUILTIN and "rf" in desc["estimators"]


# ------------------------------------------------------------------------------------------------ contracts
@pytest.mark.parametrize("version", [-1, "1", 1.0, True])
def test_contract_rejects_invalid_versions(version):
    with pytest.raises(contracts.SchemaVersionError, match="Invalid"):
        contracts.check("scene", {"schema_version": version})


def test_contract_upgrade_newer_and_missing_migration(monkeypatch):
    assert contracts.check("scene", {"a": 1}) == {"a": 1, "schema_version": 1}
    assert contracts.stamp("scene", {"schema_version": 99})["schema_version"] == 1
    with pytest.raises(contracts.SchemaVersionError, match="Upgrade geocrop-engine"):
        contracts.check("scene", {"schema_version": 2}, source="x.json")
    monkeypatch.setitem(contracts.SCHEMA_VERSIONS, "scene", 3)
    monkeypatch.setitem(contracts.MIGRATIONS, ("scene", 1), lambda b: {**b, "upgraded": True})
    with pytest.raises(contracts.SchemaVersionError, match="no migration to 3"):
        contracts.check("scene", {"schema_version": 0})
