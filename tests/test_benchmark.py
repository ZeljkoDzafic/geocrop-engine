from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest

from geocrop import benchmark


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
