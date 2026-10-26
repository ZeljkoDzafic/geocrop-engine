import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import yaml

from geocrop import experiments, features, pipeline, segmentation, segmentation_metrics, splits
from geocrop.cli import build_parser, main


def test_obia_tiles(demo, tmp_path):
    raster = Path(demo["composites"]) / "P3" / "B08.tif"
    r = segmentation.obia(raster, tmp_path / "seg.gpkg", tile_size=64, segments_per_km2=60)
    assert r["segments"] > 10 and r["segments"] <= r["segments_before_merge"]


def test_object_scores_identity(demo):
    ref = gpd.read_file(demo["parcels"]).head(20)
    r = segmentation_metrics.object_scores(ref, ref)
    assert r["object_f1"] == 1 and r["D_mean"] == pytest.approx(0, abs=1e-9)


def test_boundary_scores():
    a = np.zeros((20, 20), bool)
    b = a.copy()
    a[5, 2:10] = True
    b[6, 2:10] = True
    assert segmentation_metrics.boundary_scores(a, b, 0)["boundary_f1"] == 0
    assert segmentation_metrics.boundary_scores(a, b, 1)["boundary_f1"] == 1


def test_folds_report(demo, tmp_path):
    r = splits.assign_folds(demo["training"], tmp_path / "folds.gpkg", block_m=500, n_splits=3)
    assert r["n_folds"] == 3 and gpd.read_file(tmp_path / "folds.gpkg")["fold"].min() == 0


def test_ablation_table(samples, stack, tmp_path):
    t = experiments.run(samples, features.load_manifest(stack), tmp_path,
                        {"A": {"groups": ["s2_band"]}, "B": {"groups": ["s1"]}, "C": {"features": ["nope"]}},
                        {"n_estimators": 15, "n_jobs": 1}, 2, 500, targets=["soybean"])
    assert list(t.experiment) == ["A", "B", "C"]
    assert t.loc[t.experiment == "C", "status"].iloc[0].startswith("skipped")
    assert "mcnemar_p" in t and "f1_soybean" in t


def test_feature_reduction(samples, tmp_path):
    t = experiments.feature_reduction(samples, samples.features[:20], {"n_estimators": 10, "n_jobs": 1}, tmp_path,
                                      500, n_splits=2, step=0.5, min_features=5)
    assert t.n_features.tolist() == [20, 10, 5]


def test_pipeline_resumes(demo, tmp_path):
    cfg = {
        "windows": {"P1": {"start": "2023-04-01", "end": "2023-05-31"}, "P2": {"start": "2023-06-01", "end": "2023-07-15"},
                    "P3": {"start": "2023-07-16", "end": "2023-08-31"}, "P4": {"start": "2023-09-01", "end": "2023-10-15"}},
        "paths": {"composites": demo["composites"], "features": str(tmp_path / "stack.tif"),
                  "training": demo["training"], "parcels": demo["parcels"], "observations": demo["observations"],
                  "output": str(tmp_path / "run")},
        "classes": {"targets": ["soybean", "sunflower"]},
        "model": {"n_estimators": 20, "cv_folds": 2, "spatial_block_m": 500, "max_samples_per_polygon": 30},
        "validation": {"year": 2023, "bootstrap": 20},
        "experiments": {"S2": {"groups": ["s2_band", "s2_index"]}, "S1": {"groups": ["s1"]}},
    }
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg))
    stages = ["features", "samples", "train", "predict", "parcels", "validate", "experiments"]
    first = pipeline.run(path, stages, log=lambda *_: None)
    assert first["validate"]["oa"] > 0.5
    logs = []
    pipeline.run(path, stages, log=logs.append)
    assert all(line.startswith("[skip]") for line in logs if line.startswith("["))
    # touching a file (new mtime, same content) must not invalidate anything
    import os
    import time
    os.utime(demo["training"], (time.time() + 5, time.time() + 5))
    logs = []
    pipeline.run(path, stages, log=logs.append)
    assert all(line.startswith("[skip]") for line in logs if line.startswith("["))
    # a content change in the model configuration re-runs train and everything downstream of it
    cfg["model"]["n_estimators"] = 25
    path.write_text(yaml.safe_dump(cfg))
    logs = []
    pipeline.run(path, stages, log=logs.append)
    ran = {line.split()[-1] for line in logs if line.startswith("[run ]")}
    assert {"train", "predict", "parcels", "validate", "experiments"} <= ran and "features" not in ran
    assert (tmp_path / "run" / "run.log").exists()
    manifest = json.loads((tmp_path / "run" / "run_manifest.json").read_text())
    assert manifest["config_fingerprint"] and manifest["packages"]["numpy"]


def test_cli(capsys, tmp_path):
    assert build_parser().parse_args(["indices"]).cmd == "indices"
    main(["indices"])
    assert "| NDVI |" in capsys.readouterr().out
    main(["snap-graph", "--out", str(tmp_path / "g.xml"), "--backscatter", "gamma0"])
    assert (tmp_path / "g.xml").exists()


def test_unet_roundtrip(tmp_path, demo):
    torch = pytest.importorskip("torch")
    del torch
    rng = np.random.default_rng(0)
    x = rng.random((4, 2, 32, 32)).astype("float32")
    y = (x[:, 0] > 0.7).astype("float32")
    np.save(tmp_path / "x.npy", x)
    np.save(tmp_path / "y.npy", y)
    with pytest.raises(ValueError, match="groups"):
        segmentation.unet_train(tmp_path / "x.npy", tmp_path / "y.npy", tmp_path / "b.pt", epochs=1)
    r = segmentation.unet_train(tmp_path / "x.npy", tmp_path / "y.npy", tmp_path / "b.pt", epochs=1, batch_size=2,
                                groups=np.array(["a", "a", "b", "b"]))
    assert r["epochs"] == 1 and r["split"]["kind"] == "grouped" and r["split"]["val_groups"] == 1
    raster = Path(demo["composites"]) / "P2"
    import rasterio
    with rasterio.open(raster / "B04.tif") as a, rasterio.open(raster / "B08.tif") as b:
        prof = a.profile
        prof.update(count=2)
        with rasterio.open(tmp_path / "two.tif", "w", **prof) as dst:
            dst.write(a.read(1), 1)
            dst.write(b.read(1), 2)
    out = segmentation.unet_infer(tmp_path / "two.tif", tmp_path / "b.pt", tmp_path / "bp.tif", patch=64, overlap=8)
    with rasterio.open(out) as ds:
        assert ds.read(1).shape == (prof["height"], prof["width"])


def test_patch_group_split_has_no_leakage():
    groups = np.repeat(["g1", "g2", "g3", "g4", "g5"], 4)
    tr, val = segmentation.split_patch_groups(groups, 0.2, seed=1)
    assert not set(groups[tr]) & set(groups[val]) and len(tr) and len(val)
    with pytest.raises(ValueError):
        segmentation.split_patch_groups(np.array(["a", "a"]))


def test_extract_patches(demo, tmp_path):
    import rasterio
    src = Path(demo["composites"]) / "P2" / "B08.tif"
    with rasterio.open(src) as ds:
        prof = ds.profile
        mask = np.zeros(ds.shape, "uint8")
        mask[::16, :] = 1
        prof.update(dtype="uint8", nodata=None)
    with rasterio.open(tmp_path / "m.tif", "w", **prof) as dst:
        dst.write(mask, 1)
    X, Y, G = segmentation.extract_patches(src, tmp_path / "m.tif", size=64, block_m=640)
    assert X.shape[1:] == (1, 64, 64) and Y.shape[0] == len(G) and len(set(G)) > 1


def test_ablation_with_holdout(samples, stack, tmp_path):
    t = experiments.run(samples, features.load_manifest(stack), tmp_path,
                        {"A": {"groups": ["s2_band"]}, "B": {"groups": ["s1"]}},
                        {"n_estimators": 15, "n_jobs": 1}, 2, 500, holdout_fraction=0.25)
    assert {"holdout_overall_accuracy", "holdout_macro_f1", "holdout_mcnemar_p"} <= set(t.columns)


def test_run_lock(tmp_path):
    from geocrop.cache import RunLock
    with RunLock(tmp_path):
        (tmp_path / ".geocrop.lock").write_text(str(1))          # pid 1 is always alive
        with pytest.raises(RuntimeError), RunLock(tmp_path):
            pass


def test_hash_cache_content_not_mtime(tmp_path):
    import os

    from geocrop.cache import HashCache
    f = tmp_path / "a.tif"
    f.write_bytes(b"abc")
    hc = HashCache(tmp_path / "h.json")
    d1 = hc.file(f)
    os.utime(f, (1, 1))
    assert hc.file(f) == d1
    f.write_bytes(b"abd")
    assert hc.file(f) != d1


def test_obia_merges_fields_across_tile_seams(tmp_path):
    import rasterio
    from rasterio.transform import from_origin
    arr = np.full((2, 128, 128), 0.1, "float32")
    arr[:, 20:100, 30:110] = 0.6               # one homogeneous field crossing the seams at 64 px
    arr[1, 20:100, 30:110] = 0.3
    arr += np.random.default_rng(0).normal(0, 0.005, arr.shape).astype("float32")
    prof = {"driver": "GTiff", "height": 128, "width": 128, "count": 2, "dtype": "float32", "crs": "EPSG:32634",
            "transform": from_origin(600000, 4980000, 10, 10), "nodata": -9999.0}
    with rasterio.open(tmp_path / "img.tif", "w", **prof) as dst:
        dst.write(arr)
    kept = segmentation.obia(tmp_path / "img.tif", tmp_path / "split.gpkg", tile_size=64, segments_per_km2=20,
                             merge_seams=False)
    merged = segmentation.obia(tmp_path / "img.tif", tmp_path / "merged.gpkg", tile_size=64, segments_per_km2=20)
    assert merged["seam_merges"] > 0 and merged["segments"] < kept["segments"]
    seg = gpd.read_file(tmp_path / "merged.gpkg")
    seam_x = 600000 + 64 * 10
    crossing = seg[(seg.geometry.bounds.minx < seam_x) & (seg.geometry.bounds.maxx > seam_x) & seg.seam_merged]
    assert len(crossing) >= 1
    assert list(seg.segment_id) == list(range(1, len(seg) + 1))


def test_missing_scikit_image_names_the_extra(monkeypatch, tmp_path):
    import sys

    monkeypatch.setitem(sys.modules, "skimage", None)
    monkeypatch.setitem(sys.modules, "skimage.segmentation", None)
    monkeypatch.setitem(sys.modules, "skimage.measure", None)
    with pytest.raises(RuntimeError, match=r"\.\[segmentation\]"):
        segmentation.obia(tmp_path / "missing.tif", tmp_path / "out.gpkg")
    with pytest.raises(RuntimeError, match=r"\.\[segmentation\]"):
        segmentation.boundaries_to_parcels(tmp_path / "missing.tif", tmp_path / "out.gpkg")
