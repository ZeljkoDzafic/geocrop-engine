"""Leave-one-region-out / leave-one-year-out validation, multi-year sample stores and probability smoothing."""
import json

import geopandas as gpd
import numpy as np
import pytest
import rasterio

from geocrop import cli, config, experiments, model, splits
from geocrop.features import Feature
from geocrop.samples import Samples, combine

PARAMS = {"n_estimators": 15, "n_jobs": 1}


def _multi(seed=0, regions=("north", "south", "east"), years=("2022", "2023", "2024"), per_cell=8, px=4):
    """Synthetic samples: each (region, year) cell holds `per_cell` polygons of `px` pixels; two classes whose
    signal is shifted per region and per year (domain shift), polygons spread over separate 1 km tiles."""
    rng = np.random.default_rng(seed)
    X, y, pid, xs, ys, reg, yr = [], [], [], [], [], [], []
    for ri, r in enumerate(regions):
        for yi, t in enumerate(years):
            for k in range(per_cell):
                label = "crop_a" if k % 2 else "crop_b"
                centre = (1.0 if label == "crop_a" else -1.0) + 0.6 * ri - 0.4 * yi
                X.append(np.column_stack([rng.normal(centre, 0.7, px), rng.normal(0, 1, px)]))
                y += [label] * px
                pid += [f"{r}-{t}-{k}"] * px
                xs.append(np.full(px, 100_000.0 * ri + 3000.0 * k + 500))
                ys.append(np.full(px, 20_000.0 * yi + 500))
                reg += [r] * px
                yr += [t] * px
    return Samples(np.concatenate(X).astype("float32"), np.array(y), np.array(pid), np.concatenate(xs),
                   np.concatenate(ys), ["signal", "noise"], "EPSG:32634", region=np.array(reg), year=np.array(yr))


@pytest.mark.parametrize("strategy", ["region", "year", "region_year"])
def test_group_folds_never_share_a_group(strategy):
    s = _multi()
    groups = model.cv_groups(s, strategy)
    plan = model._fold_plan(s.y, groups, 5, 1, polygon=s.polygon, strategy=strategy)
    tested = []
    for tr, te, diag in plan:
        assert not set(groups[tr]) & set(groups[te])
        assert len(set(groups[te])) == 1 and diag["held_out_group"] == groups[te][0]
        assert not set(s.polygon[tr]) & set(s.polygon[te])
        if strategy == "region":
            assert not set(s.region[tr]) & set(s.region[te])
        if strategy == "year":
            assert not set(s.year[tr]) & set(s.year[te])
        if strategy == "region_year":   # strict: neither the place nor the season was seen in training
            assert not set(s.region[tr]) & set(s.region[te])
            assert not set(s.year[tr]) & set(s.year[te])
        tested.append(te)
    allte = np.sort(np.concatenate(tested))
    np.testing.assert_array_equal(allte, np.arange(len(s.y)))   # every sample is tested exactly once
    assert len(plan) == len(np.unique(groups))


def test_default_strategy_is_unchanged():
    s = _multi()
    groups = model.spatial_blocks(s, 5000)
    a = model.folds(s.y, groups, 4, 3)
    b = model.folds(s.y, groups, 4, 3, strategy="spatial_blocks")
    from sklearn.model_selection import StratifiedGroupKFold
    ref = list(StratifiedGroupKFold(4, shuffle=True, random_state=3).split(np.zeros(len(s.y)), s.y, groups))
    for (tr1, te1), (tr2, te2), (tr3, te3) in zip(a, b, ref):
        np.testing.assert_array_equal(te1, te3)
        np.testing.assert_array_equal(te2, te3)
        np.testing.assert_array_equal(tr1, tr3)
        np.testing.assert_array_equal(tr2, tr3)
    cv = model.cross_validate(s.X, s.y, s.polygon, groups, PARAMS, 4, 3)
    assert cv["strategy"] == "spatial_blocks" and "_evaluated" not in cv
    assert "held_out_group" not in cv["folds"][0]
    np.testing.assert_array_equal(model.cv_groups(s, "spatial_blocks", 5000), groups)


def test_missing_metadata_and_inconsistent_polygons_fail_clearly():
    s = _multi()
    bare = Samples(s.X, s.y, s.polygon, s.x, s.yy, s.features, s.crs)
    with pytest.raises(ValueError, match="needs region metadata"):
        model.cv_groups(bare, "region")
    with pytest.raises(ValueError, match="needs year metadata"):
        model.cv_groups(bare, "year")
    bad = s.take(np.arange(len(s.y)))
    bad.region[1] = "elsewhere"   # one pixel of polygon 0 in another region
    with pytest.raises(ValueError, match="more than one region"):
        model.cv_groups(bad, "region")
    with pytest.raises(ValueError, match="cv strategy must be one of"):
        model.cv_groups(s, "random")
    single = s.take(s.year == "2022")
    with pytest.raises(ValueError):
        model.cross_validate(single.X, single.y, single.polygon, model.cv_groups(single, "year"), PARAMS,
                             strategy="year")


def test_transfer_validation_reports_groups_and_gap():
    s = _multi()
    blocks = model.spatial_blocks(s, 5000)
    tv = model.transfer_validation(s.X, s.y, s.polygon, model.cv_groups(s, "region"), blocks, PARAMS, "region",
                                   4, 0)
    assert tv["strategy"] == "region" and tv["n_groups"] == 3
    assert {r["held_out_group"] for r in tv["per_group"]} == {"north", "south", "east"}
    for k in ("overall_accuracy", "kappa", "macro_f1"):
        assert tv["transfer_gap"][k] == pytest.approx(tv["in_domain"][k] - tv["out_of_domain"][k])
    for r in tv["per_group"]:
        assert r["gap_macro_f1"] == pytest.approx(r["in_domain_macro_f1"] - r["out_of_domain_macro_f1"])
        assert r["n_test"] == int((s.region == r["held_out_group"]).sum())
    assert tv["worst_group"] in {"north", "south", "east"}
    with pytest.raises(ValueError, match="group strategy"):
        model.transfer_validation(s.X, s.y, s.polygon, blocks, blocks, PARAMS, "spatial_blocks")


def test_region_year_skips_unevaluable_cells():
    s = _multi(regions=("a", "b"), years=("2022", "2023"))
    s = s.take(~((s.region == "b") & (s.year == "2023")))   # b-2023 missing -> a-2022 has no fully foreign data
    groups = model.cv_groups(s, "region_year")
    cv = model.cross_validate(s.X, s.y, s.polygon, groups, PARAMS, strategy="region_year")
    assert cv["unevaluated_samples"] == int(((s.region == "a") & (s.year == "2022")).sum())
    assert {f["held_out_group"] for f in cv["folds"]} == {"a|2023", "b|2022"}
    tv = model.transfer_validation(s.X, s.y, s.polygon, groups, model.spatial_blocks(s, 5000), PARAMS,
                                   "region_year", 3, 0)
    assert tv["groups_not_evaluable"] == ["a|2022"]


def test_nested_cv_with_group_strategy_tunes_inside_training_groups():
    s = _multi()
    groups = model.cv_groups(s, "year")
    r = model.nested_cross_validate(s.X, s.y, s.polygon, groups, PARAMS, {"max_depth": [2, None]}, 5, 2, 0,
                                    strategy="year", inner_groups=model.spatial_blocks(s, 5000))
    assert [p["held_out_group"] for p in r["selected_params_per_fold"]] == ["2022", "2023", "2024"]
    assert r["strategy"] == "year" and 0 <= r["macro_f1"] <= 1


def test_train_with_transfer_strategy_and_backwards_compatible_default(tmp_path):
    s = _multi()
    default = model.train(s, tmp_path / "d", params=PARAMS, n_splits=3, bootstrap=0)
    assert "transfer_validation" not in default and default["cv_strategy"] == "spatial_blocks"
    assert not (tmp_path / "d" / "transfer_per_group.csv").exists()
    r = model.train(s, tmp_path / "t", params=PARAMS, n_splits=3, bootstrap=0, cv_strategy="year")
    assert r["cross_validation"]["macro_f1"] == default["cross_validation"]["macro_f1"]   # spatial CV untouched
    tv = r["transfer_validation"]
    assert [g["held_out_group"] for g in tv["per_group"]] == ["2022", "2023", "2024"]
    saved = json.loads((tmp_path / "t" / "train_report.json").read_text())
    assert saved["transfer_validation"]["strategy"] == "year"
    assert (tmp_path / "t" / "transfer_per_group.csv").exists()
    # metadata survives the locked holdout and the holdout calibration split
    r2 = model.train(s, tmp_path / "h", params=PARAMS, n_splits=3, bootstrap=0, cv_strategy="region",
                     holdout_fraction=0.2, calibration_mode="holdout", calibration_fraction=0.3)
    assert r2["transfer_validation"]["n_groups"] >= 2


def test_samples_metadata_roundtrip_and_old_files(tmp_path):
    s = _multi()
    loaded = Samples.load(s.save(tmp_path / "s.npz"))
    np.testing.assert_array_equal(loaded.region, s.region)
    np.testing.assert_array_equal(loaded.year, s.year)
    old = tmp_path / "old.npz"
    np.savez_compressed(old, X=s.X, y=s.y, polygon=s.polygon, x=s.x, yy=s.yy, features=np.array(s.features),
                        crs=np.array(s.crs))
    o = Samples.load(old)
    assert o.region is None and o.year is None
    sub = s.take(s.region == "north")
    assert set(sub.region) == {"north"} and len(sub.year) == len(sub.y)
    with pytest.raises(ValueError, match="one value per sample"):
        Samples(s.X, s.y, s.polygon, s.x, s.yy, s.features, s.crs, region=np.array(["a"]))


def test_combine_builds_multi_year_store_and_refuses_mismatches():
    s = _multi(years=("2022",))
    a = Samples(s.X, s.y, s.polygon, s.x, s.yy, s.features, s.crs, region=s.region)
    b = Samples(s.X + 1, s.y, s.polygon, s.x, s.yy, s.features, s.crs, region=s.region)
    m = combine([a, b], years=[2022, 2023])
    assert len(m.y) == 2 * len(s.y) and set(m.year) == {"2022", "2023"}
    assert len(np.unique(m.polygon)) == 2 * len(np.unique(s.polygon))   # same field, two seasons, two objects
    assert m.polygon[0].startswith("2022:")
    # same location in both seasons -> same spatial block, so spatial CV keeps the two seasons together
    blocks = model.spatial_blocks(m, 5000)
    assert blocks[0] == blocks[len(s.y)]
    with pytest.raises(ValueError, match="no year metadata"):
        combine([a, b])
    with pytest.raises(ValueError, match="year ids"):
        combine([a, b], years=[2022])
    c = Samples(s.X, s.y, s.polygon, s.x, s.yy, ["noise", "signal"], s.crs)
    with pytest.raises(ValueError, match="different order"):
        combine([a, c], years=[1, 2])
    d = Samples(s.X, s.y, s.polygon, s.x, s.yy, ["signal", "ndvi_2023"], s.crs)
    with pytest.raises(ValueError, match="missing .*noise"):
        combine([a, d], years=[1, 2])
    e = Samples(s.X, s.y, s.polygon, s.x, s.yy, s.features, "EPSG:32633")
    with pytest.raises(ValueError, match="CRS"):
        combine([a, e], years=[1, 2])
    with pytest.raises(ValueError, match="all or none"):
        combine([a, Samples(s.X, s.y, s.polygon, s.x, s.yy, s.features, s.crs)], years=[1, 2])
    m1 = [Feature("signal", "s2_index", "s2", "early"), Feature("noise", "s2_band", "s2", "early")]
    m2 = [Feature("signal", "s2_index", "s2", "late"), Feature("noise", "s2_band", "s2", "early")]
    with pytest.raises(ValueError, match="feature definitions .*signal"):
        combine([a, b], years=[1, 2], manifests=[m1, m2])
    assert len(combine([a, b], years=[1, 2], manifests=[m1, list(m1)]).y) == 2 * len(s.y)
    with pytest.raises(ValueError, match="prefix_polygon_ids"):
        combine([a, b], years=[1, 2], prefix_polygon_ids=False)


def test_extract_copies_region_and_year_attributes(stack, demo):
    from geocrop.samples import extract
    gdf = gpd.read_file(demo["training"])
    centre = gdf.geometry.centroid.x
    gdf["zone"] = np.where(centre < centre.median(), "west", "east")
    gdf["season"] = 2023.0
    s = extract(stack, gdf, max_per_polygon=5, region_field="zone", year_field="season")
    assert set(s.region) == {"west", "east"} and set(s.year) == {"2023"}
    s2 = extract(stack, gdf, max_per_polygon=5, year=2021)
    assert set(s2.year) == {"2021"} and s2.region is None
    with pytest.raises(ValueError, match="not both"):
        extract(stack, gdf, year_field="season", year=2021)
    gdf.loc[0, "zone"] = None
    with pytest.raises(ValueError, match="empty"):
        extract(stack, gdf, region_field="zone")


def test_experiments_transfer_and_ablation_columns(tmp_path):
    s = _multi()
    t = experiments.transfer(s, tmp_path / "tr", params=PARAMS, strategies=("region", "year", "region_year"),
                             n_splits=3)
    assert set(t.strategy) == {"region", "year", "region_year"}
    assert (t.held_out_group == "ALL").sum() == 3
    assert json.loads((tmp_path / "tr" / "transfer.json").read_text())["strategies"]["year"]["n_groups"] == 3
    bare = Samples(s.X, s.y, s.polygon, s.x, s.yy, s.features, s.crs)
    t2 = experiments.transfer(bare, tmp_path / "tr2", params=PARAMS, strategies=("region",), n_splits=3)
    assert t2.status.iloc[0].startswith("skipped")
    manifest = [Feature("signal", "s2_index", "s2"), Feature("noise", "s2_band", "s2")]
    table = experiments.run(s, manifest, tmp_path / "abl", {"A": {"groups": ["s2_index"]},
                                                            "B": {"groups": ["s2_band", "s2_index"]}},
                            PARAMS, 3, 5000, cv_strategy="region")
    assert {"transfer_macro_f1", "transfer_gap_macro_f1", "transfer_worst_group"} <= set(table.columns)
    plain = experiments.run(s, manifest, tmp_path / "abl0", {"A": {"groups": ["s2_index"]}}, PARAMS, 3, 5000)
    assert "transfer_macro_f1" not in plain.columns


def test_config_keys_and_validation():
    cfg = config.load()
    assert cfg["cv"]["strategy"] == "spatial_blocks"
    assert cfg["predict"]["smoothing"]["enabled"] is False
    assert config.load(overrides={"cv": {"strategy": "year"}})["cv"]["strategy"] == "year"
    with pytest.raises(config.ConfigError, match="cv.strategy"):
        config.load(overrides={"cv": {"strategy": "loro"}})
    with pytest.raises(config.ConfigError, match="window"):
        config.load(overrides={"predict": {"smoothing": {"window": 4}}})
    with pytest.raises(config.ConfigError, match="smoothness"):
        config.load(overrides={"predict": {"smoothing": {"smoothness": 0}}})
    assert splits.STRATEGIES[0] == "spatial_blocks"


# --- G15: probability smoothing --------------------------------------------------------------------------------

def _two_field_cube(rng, size=30):
    truth = np.zeros((size, size), dtype=int)
    truth[:, size // 2:] = 1
    p1 = np.clip(np.where(truth == 1, 0.7, 0.3) + rng.normal(0, 0.25, truth.shape), 0.01, 0.99)
    return np.stack([1 - p1, p1]).astype("float32"), truth


def test_smoothing_reduces_noise_keeps_edges_and_sums_to_one():
    rng = np.random.default_rng(0)
    cube, truth = _two_field_cube(rng)
    valid = np.ones(truth.shape, dtype=bool)
    valid[0, 0] = False
    out = model.smooth_probabilities(cube, valid, window=5, smoothness=10)
    np.testing.assert_allclose(out[:, valid].sum(axis=0), 1, atol=1e-5)
    np.testing.assert_array_equal(out[:, 0, 0], cube[:, 0, 0])   # invalid pixel untouched
    raw_acc = (cube.argmax(0) == truth)[valid].mean()
    smooth_acc = (out.argmax(0) == truth)[valid].mean()
    assert smooth_acc > raw_acc
    edge = np.zeros_like(valid)
    edge[:, 14:16] = True
    assert (out.argmax(0) == truth)[edge & valid].mean() >= 0.8   # the field boundary is not blurred away
    weak = model.smooth_probabilities(cube, valid, window=5, smoothness=1e-6)
    np.testing.assert_allclose(weak[:, valid], cube[:, valid] / cube[:, valid].sum(0), atol=1e-3)


def test_smoothing_settings_validation():
    assert model.smoothing_settings(None) is None
    assert model.smoothing_settings({"enabled": False, "window": 7}) is None
    assert model.smoothing_settings(True) == {"enabled": True, "window": 7, "smoothness": 10.0}
    with pytest.raises(ValueError, match="odd"):
        model.smoothing_settings({"enabled": True, "window": 6})
    with pytest.raises(ValueError, match="positive"):
        model.smoothing_settings({"enabled": True, "smoothness": -1})


def test_predict_smoothing_is_block_size_independent(samples, stack, tmp_path):
    model.train(samples, tmp_path / "m", params=PARAMS, n_splits=3, spatial_block_m=500, bootstrap=0)
    plain = model.predict(stack, tmp_path / "m" / "model.joblib", tmp_path / "p0", block_size=64)
    assert "probabilities_unsmoothed" not in plain
    assert not (tmp_path / "p0" / "probabilities_unsmoothed.tif").exists()
    cfg = {"enabled": True, "window": 5, "smoothness": 10.0}
    a = model.predict(stack, tmp_path / "m" / "model.joblib", tmp_path / "pa", block_size=16, smoothing=cfg)
    b = model.predict(stack, tmp_path / "m" / "model.joblib", tmp_path / "pb", block_size=128, smoothing=cfg)
    with rasterio.open(a["probabilities"]) as da, rasterio.open(b["probabilities"]) as db, \
            rasterio.open(a["probabilities_unsmoothed"]) as raw, rasterio.open(plain["probabilities"]) as p0:
        pa, pb = da.read(), db.read()
        np.testing.assert_allclose(pa, pb, atol=1e-5)
        np.testing.assert_allclose(raw.read(), p0.read(), atol=1e-6)
        assert not np.allclose(pa, p0.read())
        assert json.loads(da.tags()["smoothing"])["window"] == 5
    with rasterio.open(a["classes"]) as ca, rasterio.open(b["classes"]) as cb:
        np.testing.assert_array_equal(ca.read(), cb.read())
    manifest = json.loads((tmp_path / "pa" / "prediction_manifest.json").read_text())
    assert manifest["smoothing"]["window"] == 5


def test_smoothing_comparison_is_marked_synthetic(tmp_path):
    r = experiments.smoothing_comparison(tmp_path, size=40, parcel=10, settings=({"window": 5, "smoothness": 10},),
                                         n_estimators=10)
    assert r["synthetic"] is True and [row["variant"] for row in r["rows"]] == ["unsmoothed", "smoothed"]
    for row in r["rows"]:
        assert 0 <= row["coverage_own_calibration"] <= 1 and 0 <= row["pixel_overall_accuracy"] <= 1
    assert (tmp_path / "smoothing_comparison.csv").exists()


def test_cli_samples_merge_and_transfer(tmp_path, capsys):
    s = _multi(years=("2022",))
    a = Samples(s.X, s.y, s.polygon, s.x, s.yy, s.features, s.crs, region=s.region).save(tmp_path / "a.npz")
    b = Samples(s.X + 0.5, s.y, s.polygon, s.x, s.yy, s.features, s.crs, region=s.region).save(tmp_path / "b.npz")
    cli.main(["samples-merge", "--inputs", str(a), str(b), "--years", "2022", "2023", "--out",
              str(tmp_path / "m.npz")])
    merged = Samples.load(tmp_path / "m.npz")
    assert set(merged.year) == {"2022", "2023"} and set(merged.region) == {"north", "south", "east"}
    cli.main(["transfer", "--samples", str(tmp_path / "m.npz"), "--out", str(tmp_path / "t"),
              "--strategies", "region", "year", "--folds", "3", "--params", '{"n_estimators": 10, "n_jobs": 1}'])
    assert (tmp_path / "t" / "transfer.csv").exists()
    capsys.readouterr()
