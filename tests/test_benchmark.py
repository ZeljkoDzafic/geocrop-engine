import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

from geocrop import benchmark, config, experiments, model
from geocrop.cli import main
from geocrop.features import Feature
from geocrop.samples import Samples


def reference():
    rng = np.random.default_rng(42)
    polygon = np.repeat(np.arange(24).astype(str), 4)
    y = np.repeat(np.where(np.arange(24) % 2, "a", "b"), 4)
    X = rng.normal(size=(len(y), 3)) + (y == "a")[:, None]
    x = np.repeat(np.arange(24) * 1000.0, 4) + np.tile(np.arange(4) * 10, 24)
    return Samples(X, y, polygon, x, np.zeros(len(y)), ["one", "two", "three"], "EPSG:32634")


def test_candidates_use_identical_folds_and_raw_cv_scores(tmp_path):
    samples = reference()
    candidates = {a: {"algorithm": a, "n_estimators": 5, "n_jobs": 1}
                  for a in ("rf", "extra_trees", "hist_gb")}
    result = benchmark.run(samples, tmp_path, candidates, 3, None, 11, 20)
    assert result["status"] == "complete" and result["internal_only"]
    groups = model.spatial_blocks(samples, None)
    for name, candidate in candidates.items():
        manual = model.cross_validate(samples.X, samples.y, samples.polygon, groups, candidate, 3, 11,
                                      coordinates=np.column_stack([samples.x, samples.yy]),
                                      crs=samples.crs, spatial_buffer_m=20)
        assert result["results"][name]["cv"]["macro_f1"] == manual["macro_f1"]
        assert [f["n_test"] for f in result["results"][name]["cv"]["folds"]] == [
            f["n_test"] for f in result["folds"]]
    saved = json.loads((tmp_path / "benchmark.json").read_text())
    assert saved == result and len(saved["fold_fingerprint"]) == 64
    assert (tmp_path / "benchmark.csv").read_text().count("\n") == 4


def test_failures_are_preserved_without_claiming_a_complete_benchmark(tmp_path):
    samples = reference()
    result = benchmark.run(samples, tmp_path,
                           {"good": {"n_estimators": 3}, "bad": {"n_estimators": 0},
                            "backend_type_error": {"n_estimators": 3, "random_state": 99}},
                           n_splits=2, spatial_block_m=None)
    assert result["status"] == "incomplete"
    assert result["results"]["bad"]["status"] == "failed"
    assert "error" in result["results"]["bad"] and "cv" not in result["results"]["bad"]
    assert result["results"]["good"]["status"] == "ok"
    assert result["results"]["backend_type_error"]["status"] == "failed"
    assert "TypeError" in result["results"]["backend_type_error"]["error"]


def test_calibration_polygons_never_enter_model_or_feature_selection(tmp_path, monkeypatch):
    samples = reference()
    samples.X[:, 0] = np.arange(len(samples.y))
    proper, split = model.selection_samples(samples, spatial_block_m=None, seed=11,
                                            calibration_mode="holdout", calibration_fraction=.25)
    fitted = []
    original = RandomForestClassifier.fit

    def record_fit(self, X, y, *args, **kwargs):
        fitted.append(np.asarray(X)[:, 0].copy())
        return original(self, X, y, *args, **kwargs)

    monkeypatch.setattr(RandomForestClassifier, "fit", record_fit)
    params = {"n_estimators": 3, "n_jobs": 1}
    opts = {"n_splits": 3, "spatial_block_m": None, "seed": 11,
            "calibration_mode": "holdout", "calibration_fraction": .25}
    report = benchmark.run(samples, tmp_path / "benchmark", {"rf": params}, **opts)
    assert report["calibration_split"] == split
    assert report["samples"] == len(proper.y) < report["input_samples"]
    manifest = [Feature(n, "s2_band", "S2") for n in samples.features]
    experiments.run(samples, manifest, tmp_path / "ablation", {"all": {"features": samples.features}},
                    params=params, **opts)
    recorded = json.loads((tmp_path / "ablation" / "ablation.json").read_text())
    assert recorded["calibration_split"] == split
    # Keep identity as the sole feature so every observed fit can be checked directly.
    experiments.feature_reduction(samples, ["one"], params, tmp_path / "reduction", min_features=1, **opts)
    recorded = json.loads((tmp_path / "reduction" / "feature_reduction.json").read_text())
    assert recorded["calibration_split"] == split
    assert fitted
    for observed in fitted:
        assert set(observed) <= set(proper.X[:, 0])



def test_sample_fingerprint_changes_when_features_change(tmp_path):
    samples = reference()
    candidate = {"rf": {"n_estimators": 2}}
    before = benchmark.run(samples, tmp_path / "before", candidate, 2, None)
    samples.X[0, 0] += 1
    after = benchmark.run(samples, tmp_path / "after", candidate, 2, None)
    assert before["sample_fingerprint"] != after["sample_fingerprint"]
    assert before["fold_fingerprint"] == after["fold_fingerprint"]


def test_cli_writes_failed_outcomes_and_exits_nonzero(tmp_path, capsys):
    path = reference().save(tmp_path / "samples.npz")
    with pytest.raises(SystemExit) as exc:
        main(["benchmark", "--samples", str(path), "--out", str(tmp_path / "benchmark"),
              "--algorithms", "rf", "--params", '{"n_estimators": 0}', "--folds", "2", "--block-m", "100"])
    assert exc.value.code == 2
    assert json.loads(capsys.readouterr().out)["status"] == "incomplete"
    assert (tmp_path / "benchmark" / "benchmark.json").exists()


def test_cli_default_algorithms_accept_inherited_forest_defaults(tmp_path, capsys):
    path = reference().save(tmp_path / "samples.npz")
    main(["benchmark", "--samples", str(path), "--out", str(tmp_path / "benchmark"),
          "--params", '{"n_estimators": 2, "n_jobs": 1}', "--folds", "2", "--block-m", "100"])
    assert json.loads(capsys.readouterr().out)["status"] == "complete"
    report = json.loads((tmp_path / "benchmark" / "benchmark.json").read_text())
    assert report["results"]["rf"]["params"]["max_features"] == "sqrt"
    assert report["results"]["hist_gb"]["params"]["max_features"] == "sqrt"


@pytest.mark.parametrize("max_features", ["sqrt", "log2", .5])
def test_histogram_boosting_fits_shared_configuration(max_features):
    from geocrop.estimators import make_estimator

    cfg = config.load(overrides={"model": {"algorithm": "hist_gb", "max_features": max_features}})
    params = {k: cfg["model"][k] for k in model.RF_KEYS if k in cfg["model"]}
    params.update(algorithm="hist_gb", n_estimators=2)
    estimator = make_estimator(params, 42).fit(reference().X, reference().y)
    assert estimator.max_features == (.5 if max_features == .5 else 1.0)


@pytest.mark.parametrize("values", [
    {"calibration_mode": "guess"}, {"calibration_fraction": 0}, {"calibration_fraction": 1},
    {"calibration_fraction": float("nan")}, {"calibration_fraction": True},
])
def test_calibration_configuration_is_validated(values):
    with pytest.raises(config.ConfigError, match="calibration_"):
        config.load(overrides={"model": values})


# --- extended benchmark (repeats, nested tuning, holdout) and segmentation benchmark

def test_model_benchmark_identical_folds(samples, tmp_path):
    cands = {"rf": {"algorithm": "rf", "n_estimators": 15, "n_jobs": 1},
             "extra_trees": {"algorithm": "extra_trees", "n_estimators": 15, "n_jobs": 1},
             "nope": {"algorithm": "catboost"}}
    t = benchmark.models(samples, tmp_path, candidates=cands, grids={"extra_trees": {"max_features": ["sqrt", 0.5]}},
                         n_splits=2, spatial_block_m=500, repeats=2, holdout_fraction=0.25, targets=["soybean"])
    ok = t[t.status == "ok"].set_index("model")
    assert {"rf", "extra_trees"} <= set(ok.index)
    assert ok.loc["rf", "repeats"] == 2 and "macro_f1_sd" in ok.columns and "holdout_macro_f1" in ok.columns
    assert bool(ok.loc["extra_trees", "tuned"])
    try:
        import catboost  # noqa: F401
    except ImportError:
        assert t.set_index("model").loc["nope", "status"].startswith("skipped")
    assert (tmp_path / "benchmark_models.json").exists()


def test_fold_signature_is_deterministic(samples):
    assert benchmark.fold_signature(samples, 500, 3, 1) == benchmark.fold_signature(samples, 500, 3, 1)


def test_segmentation_benchmark(demo, tmp_path):
    raster = Path(demo["composites"]) / "P3" / "B08.tif"
    ref = gpd.read_file(demo["parcels"])
    t = benchmark.segmentation(raster, demo["parcels"], tmp_path, tile_size=96, segments_per_km2=40)
    assert set(t.method) == {"obia_tiles", "obia_seam_merged"}
    assert (t.status == "ok").all() and ((t.object_f1 >= 0) & (t.object_f1 <= 1)).all()
    mask = benchmark.rasterize_boundaries(ref, raster)
    assert mask.dtype == bool and 0 < mask.mean() < 0.5


def test_segmentation_benchmark_with_unet(demo, tmp_path):
    pytest.importorskip("torch")
    import rasterio

    from geocrop import segmentation
    raster = Path(demo["composites"]) / "P3" / "B08.tif"
    with rasterio.open(raster) as ds:
        img = ds.read(1)[None, None].repeat(2, axis=0)
    np.save(tmp_path / "x.npy", img[:, :, :64, :64].astype("float32"))
    np.save(tmp_path / "y.npy", (np.random.default_rng(0).random((2, 64, 64)) > 0.9).astype("float32"))
    segmentation.unet_train(tmp_path / "x.npy", tmp_path / "y.npy", tmp_path / "m.pt", epochs=1, groups=["a", "b"])
    t = benchmark.segmentation(raster, demo["parcels"], tmp_path / "b", unet_model=tmp_path / "m.pt", tile_size=96)
    assert "unet_watershed" in set(t.method)


def test_strict_benchmark_excludes_final_holdout(samples, tmp_path):
    from geocrop.model import split_final_holdout
    dev, test = split_final_holdout(samples, 500, 0.25, 42)
    rep = benchmark.run(samples, tmp_path, {"rf": {"algorithm": "rf", "n_estimators": 10, "n_jobs": 1}},
                        n_splits=2, spatial_block_m=500, holdout_fraction=0.25)
    assert rep["holdout_samples_excluded"] == len(test.y) and rep["samples"] == len(dev.y)


def test_tuned_benchmark_fits_selected_parameters(samples, tmp_path):
    cands = {"rf": {"algorithm": "rf", "n_estimators": 10, "n_jobs": 1}}
    benchmark.models(samples, tmp_path, candidates=cands, grids={"rf": {"max_features": ["sqrt", 0.5]}},
                     n_splits=2, spatial_block_m=500, repeats=1, holdout_fraction=0.25)
    details = json.loads((tmp_path / "benchmark_models.json").read_text())["details"]["rf"]["runs"][0]
    assert details["selected_params"]["max_features"] in ("sqrt", 0.5) and "holdout_macro_f1" in details
