"""Fixes for the open findings of release gate c091962 (docs/release-gates/release-gate-c091962.md)."""
from __future__ import annotations

import json
import sys

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

from geocrop import attribution, benchmark
from geocrop.samples import MissingClassError, Samples, check_classes, extract

# --------------------------------------------------------------------------- 1. classes lost by `samples`


def _ts_stack(tmp_path):
    """Stack with `ts` features valid everywhere and a `phenology` feature NaN over the east half (class wheat)."""
    from geocrop.features import Feature, write_manifest

    rng = np.random.default_rng(0)
    data = np.empty((3, 40, 40), "float32")
    data[0] = rng.normal(0.3, 0.02, (40, 40))
    data[1] = rng.normal(0.6, 0.02, (40, 40))
    data[1, :, 20:] = rng.normal(0.4, 0.02, (40, 20))
    data[2] = 100.0
    data[2, :, 20:] = np.nan                       # flat seasonal curve: no phenology metric
    path = tmp_path / "stack_ts.tif"
    with rasterio.open(path, "w", driver="GTiff", width=40, height=40, count=3, dtype="float32", crs="EPSG:32634",
                       transform=from_origin(0, 400, 10, 10), nodata=-9999) as ds:
        ds.write(np.where(np.isfinite(data), data, -9999))
        ds.descriptions = ("ts_NDVI_d100", "ts_NDVI_d200", "phen_NDVI_sos")
    feats = [Feature("ts_NDVI_d100", "ts", "S2"), Feature("ts_NDVI_d200", "ts", "S2"),
             Feature("phen_NDVI_sos", "phenology", "S2")]
    write_manifest(path, {"stack": path.name, "features": [vars(f) for f in feats]})
    rows = []
    for x0, cls in [(0, "maize"), (200, "wheat")]:
        for j in range(4):
            rows.append({"class_name": cls, "polygon_id": f"{cls}{j}",
                         "geometry": box(x0 + 20, 20 + j * 95, x0 + 180, 95 + j * 95)})
    polys = tmp_path / "polys.gpkg"
    gpd.GeoDataFrame(rows, crs="EPSG:32634").to_file(polys)
    return path, polys


def test_samples_restricted_to_feature_groups_keep_every_class(tmp_path):
    from geocrop.features import load_manifest, select

    stack, polys = _ts_stack(tmp_path)
    with pytest.warns(UserWarning, match=r"WITHOUT any sample: \['wheat'\]"):
        full = extract(stack, polys)
    assert full.missing_classes() == ["wheat"]
    ts = extract(stack, polys, features=select(load_manifest(stack), groups=["ts"]))
    assert ts.features == ["ts_NDVI_d100", "ts_NDVI_d200"] and set(ts.y) == {"maize", "wheat"}
    assert ts.missing_classes() == [] and list(ts.reference_classes) == ["maize", "wheat"]
    # metadata survives save/load, take and combine
    loaded = Samples.load(full.save(tmp_path / "full.npz"))
    assert loaded.missing_classes() == ["wheat"] and loaded.take(np.arange(3)).missing_classes() == ["wheat"]
    with pytest.raises(ValueError, match="Features not in stack"):
        extract(stack, polys, features=["nope"])


def test_train_and_benchmark_refuse_lost_classes_unless_allowed(tmp_path):
    from geocrop.model import train

    stack, polys = _ts_stack(tmp_path)
    with pytest.warns(UserWarning):
        full = extract(stack, polys)
    fast = {"algorithm": "rf", "n_estimators": 5, "n_jobs": 1}
    with pytest.raises(MissingClassError, match=r"without any sample: \['wheat'\]"):
        train(full, tmp_path / "m", params=fast, n_splits=2, spatial_block_m=100)
    with pytest.raises(MissingClassError, match="--allow-missing-classes"):
        benchmark.models(full, tmp_path / "bm", candidates={"rf": fast}, n_splits=2, spatial_block_m=100, repeats=1)
    with pytest.raises(MissingClassError):
        benchmark.run(full, tmp_path / "b", {"rf": fast}, n_splits=2, spatial_block_m=100)
    with pytest.warns(UserWarning, match="without any sample"):
        assert check_classes(full, allow_missing=True) == ["wheat"]
    # older sample files without class metadata cannot be checked and pass
    legacy = Samples(full.X, full.y, full.polygon, full.x, full.yy, full.features, full.crs)
    assert check_classes(legacy) == []


def test_cli_ts_path_samples_groups_then_benchmark(tmp_path):
    """The gate's step-3 path: ts-features stack -> samples -> benchmark-models --groups ts."""
    from geocrop.cli import main

    stack, polys = _ts_stack(tmp_path)
    common = ["--features", str(stack), "--groups", "ts", "--block-m", "100", "--folds", "2", "--repeats", "1",
              "--models", "rf", "--params", '{"rf": {"n_estimators": 5, "n_jobs": 1}}']
    main(["samples", "--features", str(stack), "--polygons", str(polys), "--out", str(tmp_path / "all.npz")])
    with pytest.raises(MissingClassError):            # all features: wheat lost -> non-zero exit
        main(["benchmark-models", "--samples", str(tmp_path / "all.npz"), "--out", str(tmp_path / "b0"), *common])
    main(["samples", "--features", str(stack), "--polygons", str(polys), "--out", str(tmp_path / "ts.npz"),
          "--groups", "ts"])
    main(["benchmark-models", "--samples", str(tmp_path / "ts.npz"), "--out", str(tmp_path / "b1"), *common])
    report = json.loads((tmp_path / "b1" / "benchmark_models.json").read_text())
    assert report["reference_classes_without_samples"] == []
    assert report["details"]["rf"]["params"]["n_estimators"] == 5
    main(["benchmark-models", "--samples", str(tmp_path / "all.npz"), "--out", str(tmp_path / "b2"), *common,
          "--allow-missing-classes"])
    assert json.loads((tmp_path / "b2" / "benchmark_models.json").read_text())[
        "reference_classes_without_samples"] == ["wheat"]


# --------------------------------------------------------------------------- 5. fold signatures, params


def test_per_model_fold_signatures_come_from_the_folds_used(samples, tmp_path):
    cands = {"rf": {"algorithm": "rf", "n_estimators": 5, "n_jobs": 1},
             "extra_trees": {"algorithm": "extra_trees", "n_estimators": 5, "n_jobs": 1}}
    benchmark.models(samples, tmp_path / "bm", candidates=cands, grids={"extra_trees": {"max_depth": [2, 4]}},
                     n_splits=3, spatial_block_m=500, repeats=2)
    report = json.loads((tmp_path / "bm" / "benchmark_models.json").read_text())
    per_model = report["fold_signatures_per_model"]
    assert set(per_model) == {"rf", "extra_trees"} and report["identical_folds_across_models"]
    assert per_model["rf"][0] == report["fold_signature"]          # repeat 0 uses the base seed
    assert per_model["rf"][0] != per_model["rf"][1]                # repeats use different seeds
    assert [r["fold_signature"] for r in report["details"]["rf"]["runs"]] == per_model["rf"]


def test_benchmark_models_epochs_and_params_overrides(monkeypatch, samples, tmp_path):
    from geocrop.cli import main

    seen = {}

    def fake_models(s, out, feats, cands, *a, **kw):
        seen.update(cands)
        import pandas as pd
        return pd.DataFrame([{"model": m, "status": "ok"} for m in cands])
    monkeypatch.setattr(benchmark, "models", fake_models)
    path = samples.save(tmp_path / "s.npz")
    main(["benchmark-models", "--samples", str(path), "--out", str(tmp_path / "o"), "--models", "ltae", "tempcnn",
          "rf", "--epochs", "7", "--params", '{"ltae": {"epochs": 3, "patience": 2}, "rf": {"n_estimators": 9}}'])
    assert seen["ltae"]["epochs"] == 3 and seen["ltae"]["patience"] == 2       # explicit params win
    assert seen["tempcnn"]["epochs"] == 7 and "epochs" not in seen["rf"] and seen["rf"]["n_estimators"] == 9
    with pytest.raises(ValueError, match="not benchmarked"):
        main(["benchmark-models", "--samples", str(path), "--out", str(tmp_path / "o"), "--models", "rf",
              "--params", '{"ltae": {"epochs": 3}}'])


def test_torch_import_failure_is_a_skip_not_a_crash(monkeypatch, tmp_path):
    """Native-library failures are reported consistently for every optional backend, torch included."""
    pkg = tmp_path / "torch"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("raise OSError('dlopen: libtorch_cpu.dylib not loaded')\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    from geocrop import estimators
    reason = estimators.backend_error("ltae")
    assert reason and "cannot load its native library" in reason
    assert "native library" in benchmark._available({"algorithm": "ltae"})


# --------------------------------------------------------------------------- 3. Copernicus attribution


def test_attribution_text_and_merge(tmp_path):
    from datetime import date

    assert attribution.copernicus([2023]) == "Contains modified Copernicus Sentinel data 2023"
    assert attribution.copernicus([date(2024, 5, 1), "2023-07-01T10:00:00Z", 2024]) == \
        "Contains modified Copernicus Sentinel data 2023, 2024"
    assert attribution.years_of(attribution.copernicus([2022, 2023])) == [2022, 2023]
    paths = []
    for i, year in enumerate((2023, 2024, None)):
        p = tmp_path / f"{i}.tif"
        with rasterio.open(p, "w", driver="GTiff", width=2, height=2, count=1, dtype="uint8") as ds:
            ds.write(np.zeros((1, 2, 2), "uint8"))
            if year:
                ds.update_tags(**attribution.tags([year]))
        paths.append(p)
    assert attribution.read(paths) == "Contains modified Copernicus Sentinel data 2023, 2024"
    assert attribution.read(paths[2:]) is None


def test_attribution_flows_from_composites_to_exported_cog_and_stac(demo, tmp_path):
    """Composites carrying the tag (as written by every Sentinel import) -> stack -> prediction -> COG + STAC."""
    import shutil

    from geocrop.cli import main
    from geocrop.export import stac_item, to_cog
    from geocrop.features import build_stack
    from geocrop.model import predict, train

    comp = tmp_path / "composites"
    shutil.copytree(demo["composites"], comp)
    for tif in comp.rglob("*.tif"):
        with rasterio.open(tif, "r+") as ds:
            ds.update_tags(**attribution.tags([2023]))
    stack = build_stack(comp, tmp_path / "stack.tif", block_size=64)
    with rasterio.open(stack) as ds:
        assert ds.tags()[attribution.TAG] == "Contains modified Copernicus Sentinel data 2023"
    s = extract(stack, demo["training"], max_per_polygon=20)
    train(s, tmp_path / "model", params={"algorithm": "rf", "n_estimators": 5, "n_jobs": 1}, n_splits=2,
          spatial_block_m=500, bootstrap=0)
    pred = tmp_path / "pred"
    predict(stack, tmp_path / "model" / "model.joblib", pred)
    for name in ("classes", "probabilities", "uncertainty"):
        with rasterio.open(pred / f"{name}.tif") as ds:
            assert ds.tags()[attribution.TAG].endswith("2023"), name
    cog = to_cog(pred / "classes.tif", tmp_path / "exp" / "classes.tif")
    with rasterio.open(cog) as ds:
        assert ds.tags()[attribution.TAG].endswith("2023")
    item = stac_item(pred, "it", "2023-03-01", "2023-10-31")
    assert item["properties"]["attribution"] == "Contains modified Copernicus Sentinel data 2023"
    assert item["properties"]["license"] == "other" and item["properties"]["providers"][0]["roles"][-1] == "licensor"
    assert any(link["rel"] == "license" for link in item["links"])
    main(["export", "--prediction", str(pred), "--out", str(tmp_path / "exp2"), "--id", "x",
          "--start", "2023-03-01", "--end", "2023-10-31"])
    exported = json.loads((tmp_path / "exp2" / "item.json").read_text())
    assert exported["properties"]["attribution"].endswith("2023")


def test_synthetic_outputs_claim_no_copernicus_data(stack, tmp_path):
    with rasterio.open(stack) as ds:
        assert attribution.TAG not in ds.tags()


def test_stac_import_tags_scene_bands(tmp_path, monkeypatch):
    """Every Sentinel entry point writes the tag; checked here for the Earth Search COG import."""
    from geocrop import stac_import

    src = inspect_source(stac_import.import_item)
    assert "attribution.tags" in src and "update_tags" in src
    from geocrop import openeo_adapter, s1, s1_stac, s2
    for fn in (s2.import_safe, s1.import_geotiff, s1.composite, s1_stac.write_scene, openeo_adapter.write_scenes, openeo_adapter.write_s2_composite,
               openeo_adapter.write_s1_composite, s2.composite):
        assert "attribution." in inspect_source(fn), fn.__name__


def inspect_source(fn) -> str:
    import inspect
    return inspect.getsource(fn)


def test_openeo_scene_conversion_writes_attribution(tmp_path):
    from geocrop import openeo_adapter
    from geocrop.indices import S2_BANDS

    ref = tmp_path / "grid.tif"
    with rasterio.open(ref, "w", driver="GTiff", width=8, height=8, count=1, dtype="uint8", crs="EPSG:32634",
                       transform=from_origin(0, 80, 10, 10)) as ds:
        ds.write(np.ones((1, 8, 8), "uint8"))
    order = [*S2_BANDS, "SCL"]
    res = tmp_path / "openEO_2023-06-15Z.tif"
    data = np.full((len(order), 8, 8), 1500, "float32")
    data[-1] = 4
    with rasterio.open(res, "w", driver="GTiff", width=8, height=8, count=len(order), dtype="float32",
                       crs="EPSG:32634", transform=from_origin(0, 80, 10, 10)) as ds:
        ds.write(data)
    written = openeo_adapter.write_scenes([res], tmp_path / "scenes", ref, order, {"type": "openeo"})
    with rasterio.open(written[0] / "B04.tif") as ds:
        assert ds.tags()[attribution.TAG] == "Contains modified Copernicus Sentinel data 2023"


# --------------------------------------------------------------------------- 3. OmniCloudMask pinning


def test_omnicloudmask_uses_and_records_pinned_weights(monkeypatch, tmp_path):
    import types

    from geocrop import clouds

    calls = {}

    def predict_from_array(arr, batch_size=1, destination_model_dir=None, model_version=None):
        calls.update(dir=destination_model_dir, version=model_version)
        return np.zeros((1, *arr.shape[1:]), "uint8")
    monkeypatch.setitem(sys.modules, "omnicloudmask", types.SimpleNamespace(predict_from_array=predict_from_array))
    scene = tmp_path / "scene"
    scene.mkdir()
    for b in ("B04", "B03", "B08"):
        with rasterio.open(scene / f"{b}.tif", "w", driver="GTiff", width=4, height=4, count=1, dtype="float32",
                           crs="EPSG:32634", transform=from_origin(0, 40, 10, 10), nodata=-9999) as ds:
            ds.write(np.full((1, 4, 4), 0.1, "float32"))
    weights = tmp_path / "w"
    weights.mkdir()
    with pytest.raises(FileNotFoundError, match="OmniCloudMask weights"):
        clouds.omnicloudmask_scene(scene, model_dir=weights)
    for f in clouds.omnicloudmask_files():
        (weights / f["name"]).write_bytes(b"x")
    with pytest.raises(ValueError, match="pinned SHA-256"):
        clouds.omnicloudmask_scene(scene, model_dir=weights)
    monkeypatch.setattr(clouds, "check_omnicloudmask_weights",
                        lambda d: [weights / f["name"] for f in clouds.omnicloudmask_files()])
    out = clouds.omnicloudmask_scene(scene, model_dir=weights)
    assert calls == {"dir": str(weights), "version": clouds.OMNICLOUDMASK_VERSION}
    with rasterio.open(out) as ds:
        tags = ds.tags()
    assert clouds.OMNICLOUDMASK_WEIGHTS[0][1] in tags["GEOCROP_CLOUDMASK_WEIGHTS"]
    assert tags["licence"] == "weights MIT" and "Wright" in tags["citation"]
    assert all(f["url"].startswith("https://huggingface.co/NickWright/OmniCloudMask/resolve/" +
                                   clouds.OMNICLOUDMASK_REVISION) for f in clouds.omnicloudmask_files())


def test_config_feature_groups_and_allow_missing_classes_validation():
    from geocrop import config

    cfg = config.load(overrides={"model": {"feature_groups": ["ts", "harmonic"], "allow_missing_classes": True}})
    assert cfg["model"]["feature_groups"] == ["ts", "harmonic"]
    with pytest.raises(config.ConfigError, match="feature_groups"):
        config.load(overrides={"model": {"feature_groups": ["nope"]}})
    with pytest.raises(config.ConfigError, match="allow_missing_classes"):
        config.load(overrides={"model": {"allow_missing_classes": "yes"}})
    # at their defaults the new keys leave existing configuration fingerprints unchanged
    assert config.fingerprint(config.load()) == config.fingerprint(
        {k: v for k, v in config.load().items()} | {"model": {k: v for k, v in config.load()["model"].items()
                                                              if k not in ("feature_groups", "allow_missing_classes")}})
