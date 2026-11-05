"""Training-only controls, skipped-polygon report, partition stability, parcel aggregation, profiles, cloud scores."""
import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin

from geocrop import aggregation, cloudscore, model, profiles, stability
from geocrop.samples import extract


def test_train_mask_keeps_test_folds_and_drops_polygons(samples):
    pid = str(samples.polygon[0])
    keep = model.train_mask(samples.polygon, [pid], 5, seed=1)
    assert not keep[samples.polygon.astype(str) == pid].any()
    for other in set(samples.polygon.astype(str)) - {pid}:
        assert keep[samples.polygon.astype(str) == other].sum() <= 5
    groups = model.spatial_blocks(samples, 500)
    X, y = samples.X, samples.y
    params = {"n_estimators": 10, "n_jobs": 1}
    base = model.cross_validate(X, y, samples.polygon, groups, params, 3, 42)
    ctl = model.cross_validate(X, y, samples.polygon, groups, params, 3, 42, train_ok=keep)
    assert [f["n_test"] for f in base["folds"]] == [f["n_test"] for f in ctl["folds"]]
    assert sum(f["n_train"] for f in ctl["folds"]) < sum(f["n_train"] for f in base["folds"])


def test_train_controls_reported_and_refuse_tuning(samples, tmp_path):
    rep = model.train(samples, tmp_path / "m", params={"n_estimators": 10, "n_jobs": 1}, n_splits=3,
                      spatial_block_m=500, bootstrap=0, train_exclude=[str(samples.polygon[0])],
                      train_max_samples_per_polygon=10)
    assert rep["train_controls"]["training_samples_kept"] < rep["samples"]
    with pytest.raises(ValueError, match="tuning grid"):
        model.train(samples, tmp_path / "t", params={"n_estimators": 5, "n_jobs": 1}, n_splits=2, spatial_block_m=500,
                    tune_grid={"min_samples_leaf": [1, 2]}, train_exclude=[str(samples.polygon[0])])


def test_spatial_block_offset_changes_partition(samples):
    a = model.spatial_blocks(samples, 500)
    b = model.spatial_blocks(samples, 500, (250.0, 250.0))
    assert len(a) == len(b) and (a != b).any()
    assert (model.spatial_blocks(samples, 500, (0.0, 0.0)) == a).all()


def test_skipped_polygon_report(stack, demo, tmp_path):
    import geopandas as gpd
    from shapely.geometry import box
    gdf = gpd.read_file(demo["training"])
    with rasterio.open(stack) as ds:
        x0, y1 = ds.bounds.left, ds.bounds.top
        crs = ds.crs
    tiny = gpd.GeoDataFrame({gdf.columns[0]: ["tiny"], "polygon_id": ["tiny"], "class_name": [gdf["class_name"].iloc[0]]},
                            geometry=[box(x0 + 3.1, y1 - 3.9, x0 + 3.9, y1 - 3.1)], crs=crs)
    far = gpd.GeoDataFrame({"polygon_id": ["far"], "class_name": [gdf["class_name"].iloc[0]]},
                           geometry=[box(x0 - 5000, y1 + 5000, x0 - 4900, y1 + 5100)], crs=crs)
    mixed = pd.concat([gdf.to_crs(crs), tiny, far], ignore_index=True)
    mixed = gpd.GeoDataFrame(mixed, geometry="geometry", crs=crs)
    report = tmp_path / "skipped.csv"
    extract(stack, mixed, max_per_polygon=10, skipped_report=report)
    table = pd.read_csv(report)
    reasons = dict(zip(table["polygon"], table["reason"]))
    assert "smaller than one pixel" in reasons["tiny"]
    assert "outside" in reasons["far"]


def test_stability_reports_spread(samples, tmp_path):
    r = stability.run(samples, {"n_estimators": 8, "n_jobs": 1}, block_sizes_m=(500,), n_offsets=2, n_splits=2,
                      targets=[str(samples.y[0])], out=tmp_path)
    assert r["partitions"] >= 1 and "overall_accuracy" in r["summary"]
    assert (tmp_path / "cv_stability.csv").exists()
    with pytest.raises(ValueError):
        stability.partitions([500], 0)


def test_stability_polygon_cv_and_watch(samples, tmp_path):
    pid = str(samples.polygon[0])
    r = stability.run(samples, {"n_estimators": 8, "n_jobs": 1}, block_sizes_m=(500,), n_offsets=1, n_splits=2,
                      targets=[str(samples.y[0])], out=tmp_path, polygon_repeats=2, watch=[pid, "missing"],
                      radii_m=(100, 1000))
    kinds = set(r["table"]["kind"])
    assert "polygon_stratified" in kinds
    w = r["watch_table"]
    assert set(w["polygon_id"]) == {pid} and "same_class_training_within_1000m" in w
    assert (tmp_path / "watch_polygons.csv").exists() and list((tmp_path / "confusion").glob("*.csv"))


def _pixels():
    rng = np.random.default_rng(0)
    rows = []
    for g in range(4):
        for k in range(6):
            true = "a" if k % 2 == 0 else "b"
            conf = 0.9 if k < 4 else 0.52
            for _ in range(10):
                p = {"a": 0.0, "b": 0.0}
                right = rng.random() < conf
                top = true if right else ("b" if true == "a" else "a")
                p[top], p["b" if top == "a" else "a"] = conf, 1 - conf
                rows.append({"polygon": f"g{g}p{k}", "spatial_group": f"G{g}", "label": true, "oof_done": True,
                             "p_a": p["a"], "p_b": p["b"]})
    return pd.DataFrame(rows)


def test_parcel_aggregation_compares_methods(tmp_path):
    r = aggregation.compare(_pixels(), targets=["a", "b"], target_accuracy=0.9, out=tmp_path)
    assert set(r["methods"]) == {"majority", "mean_prob", "combined"}
    comb = r["methods"]["combined"]
    assert 0 <= comb["rejected_share"] <= 1 and comb["accepted_share"] + comb["rejected_share"] == pytest.approx(1)
    assert (tmp_path / "parcel_confusion_majority.csv").exists()
    fixed = aggregation.compare(_pixels(), threshold=0.99)
    assert fixed["methods"]["combined"]["accepted_share"] == 0
    with pytest.raises(ValueError):
        aggregation.polygon_table(pd.DataFrame({"polygon": ["x"]}))


def test_choose_threshold_prefers_lowest_sufficient():
    conf = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.97])
    ok = np.array([0, 0, 1, 1, 1, 1, 1], dtype=bool)
    assert aggregation.choose_threshold(conf, ok, 1.0, min_accepted=3) == 0.65


def test_profiles_distances_name_closest_class(tmp_path, monkeypatch):
    class F:
        def __init__(self, name):
            self.name = name
    names = [f"ts_NDVI_d{d}" for d in (100, 120, 140)]
    monkeypatch.setattr(profiles.featmod, "load_manifest", lambda stack: [F(n) for n in names])
    rows = []
    for i in range(5):
        rows.append({"polygon_id": f"a{i}", "class_name": "a", **dict(zip(names, [0.2, 0.5, 0.8]))})
        rows.append({"polygon_id": f"b{i}", "class_name": "b", **dict(zip(names, [0.6, 0.6, 0.2]))})
    rows.append({"polygon_id": "odd", "class_name": "a", **dict(zip(names, [0.6, 0.6, 0.2]))})
    monkeypatch.setattr(profiles.separability, "parcel_means", lambda *a, **k: pd.DataFrame(rows))
    r = profiles.run("stack.tif", "p.gpkg", "class_name", ["odd"], ["a", "b"], tmp_path)
    assert r["summary"]["odd"]["own_class"] == "a" and r["summary"]["odd"]["closest_class_by_mean_rmse"] == "b"
    assert (tmp_path / "profiles.csv").exists()
    with pytest.raises(ValueError, match="without data"):
        profiles.run("stack.tif", "p.gpkg", "class_name", ["missing"], ["a", "b"], tmp_path)


def test_cloud_score_attach(tmp_path):
    scene = tmp_path / "scenes" / "S2A_X"
    scene.mkdir(parents=True)
    tr = from_origin(0, 40, 10, 10)
    prof = dict(driver="GTiff", height=4, width=4, count=1, crs="EPSG:32633", transform=tr)
    with rasterio.open(scene / "SCL.tif", "w", dtype="uint8", **prof) as d:
        d.write(np.full((4, 4), 4, "uint8"), 1)
    scores = tmp_path / "scores"
    scores.mkdir()
    score = np.full((4, 4), 0.9, "float32")
    score[0, 0] = 0.5
    with rasterio.open(scores / "S2A_X.tif", "w", dtype="float32", **prof) as d:
        d.write(score, 1)
    r = cloudscore.attach(tmp_path / "scenes", scores)
    assert r["attached"] == 1
    with rasterio.open(scene / "CLDPRB.tif") as d:
        prob = d.read(1)
    assert prob[3, 3] == 10 and prob[0, 0] == 50
    assert cloudscore.attach(tmp_path / "scenes", scores)["already_present"] == 1


def test_train_labels_and_relabel_in_stability(samples, tmp_path):
    pid = str(samples.polygon[0])
    other = next(c for c in np.unique(samples.y) if c != samples.y[0])
    y2 = model.train_labels(samples.y, samples.polygon, {pid: other})
    assert (y2[samples.polygon == pid] == other).all() and (y2[samples.polygon != pid] == samples.y[samples.polygon != pid]).all()
    with pytest.raises(ValueError):
        model.train_labels(samples.y, samples.polygon, {"nope": other})
    with pytest.raises(ValueError):
        model.train_labels(samples.y, samples.polygon, {pid: "not_a_class"})
    r = stability.run(samples, {"n_estimators": 8, "n_jobs": 1}, block_sizes_m=(500,), n_offsets=1, n_splits=2,
                      targets=[str(samples.y[0])], train_relabel={pid: other}, watch=[pid], out=tmp_path)
    assert r["train_controls"]["relabelled_for_training_only"] == {pid: other}
    assert any(c.endswith("_unrelabelled") for c in r["table"].columns)
    assert any(c.startswith("mean_p_") for c in r["watch_table"].columns)


def test_phenology_timing_and_alignment(tmp_path):
    from geocrop import phenotiming
    days = list(range(91, 280, 15))
    rows = []
    for pid, cls, role, shift in [("a", "x", "reference", 0), ("b", "x", "reference", 3),
                                  ("c", "x", "highlight", -30), ("d", "y", "reference", 40)]:
        for d in days:
            v = np.exp(-((d - 190 - shift) / 35.0) ** 2)
            rows.append({"polygon": pid, "class": cls, "value": v, "family": "ts_NDVI", "doy": d, "role": role})
    pd.DataFrame(rows).to_csv(tmp_path / "p.csv", index=False)
    r = phenotiming.run(tmp_path / "p.csv", ["x", "y"], tmp_path / "o")
    rel = pd.read_csv(tmp_path / "o" / "phenology_highlight_vs_classes.csv").iloc[0]
    assert r["polygons"] == 4
    assert rel["peak_doy_minus_x_median"] < -20
    assert abs(rel["best_shift_days_x"] - 30) <= 3 and rel["rmse_aligned_x"] < rel["rmse_x"]
    with pytest.raises(ValueError):
        phenotiming.run(tmp_path / "p.csv", ["x"], tmp_path / "o2", fraction=1.5)
