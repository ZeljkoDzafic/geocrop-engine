"""AOI-aware processing (docs/aoi-aware.md): the mask, the block index, and every stage that skips blocks outside
the AOI. SYNTHETIC data throughout. The central guarantee is tested end to end: a pipeline run with the AOI mask
equals the run without it at every pixel inside the AOI, and everything outside the AOI is no-data.

Minimal-venv safe: scikit-image (segmentation) is optional and skipped when absent.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import aoi_study as S
import geopandas as gpd
import numpy as np
import pystac
import pytest
import rasterio
import yaml
from rasterio.transform import from_origin
from rasterio.windows import Window
from shapely.geometry import Point, box

from geocrop import (
    aoi,
    config,
    demo,
    features,
    model,
    pipeline,
    preflight,
    raster,
    s1,
    s1_stac,
    s2,
    stac_import,
)
from geocrop.aoi import AoiMask
from geocrop.indices import S2_BANDS

CRS = "EPSG:32634"
STAGES = ["composite", "features", "samples", "train", "predict", "parcels"]


# --------------------------------------------------------------------------- helpers

def write_raster(path: Path, array: np.ndarray, x0=500000.0, y0=5000000.0, res=10.0, nodata=None, crs=CRS) -> Path:
    array = np.asarray(array)
    bands = array if array.ndim == 3 else array[None]
    with rasterio.open(path, "w", driver="GTiff", width=bands.shape[2], height=bands.shape[1], count=len(bands),
                       dtype=str(bands.dtype), crs=crs, transform=from_origin(x0, y0, res, res), nodata=nodata) as dst:
        dst.write(bands)
    return Path(path)


def mask_from(path: Path, inside: np.ndarray) -> Path:
    return write_raster(path, inside.astype("uint8"), nodata=0)


def inside_blocks(mask: AoiMask, size: int = 512, halo=None) -> np.ndarray:
    """Boolean raster of the pixels in active blocks."""
    out = np.zeros((mask.height, mask.width), bool)
    for w in mask.windows(size, halo):
        out[int(w.row_off):int(w.row_off + w.height), int(w.col_off):int(w.col_off + w.width)] = True
    return out


def read(path: Path):
    with rasterio.open(path) as ds:
        return ds.read(), ds.nodata


def assert_equal_where(full: Path, masked: Path, region: np.ndarray, outside_is_nodata_in: np.ndarray):
    """`masked` equals `full` on `region` (within 1e-6, same NaN pattern) and holds no-data on the rest of
    `outside_is_nodata_in`."""
    a, nodata = read(full)
    b, nodata_m = read(masked)
    assert nodata == nodata_m
    a, b = a.astype("float64"), b.astype("float64")
    assert np.array_equal(np.isnan(a[:, region]), np.isnan(b[:, region])), f"{masked.name}: NaN pattern differs"
    assert np.allclose(a[:, region], b[:, region], atol=1e-6, rtol=0, equal_nan=True), f"{masked.name}: differs"
    rest = ~region & outside_is_nodata_in
    fill = nodata if nodata is not None else 0
    assert np.all(b[:, rest] == fill), f"{masked.name}: pixels outside the region are not no-data"


# --------------------------------------------------------------------------- the mask

def test_blocks_halo_and_pixels(tmp_path):
    inside = np.zeros((300, 1100), bool)
    inside[20:100, 0:512] = True                       # ends exactly at the block boundary (column 511)
    inside[250:260, 1050:1060] = True
    mask = AoiMask(mask_from(tmp_path / "m.tif", inside))
    assert mask.windows(512) == [Window(0, 0, 512, 300), Window(1024, 0, 76, 300)]
    assert [(w.col_off, w.row_off) for w in mask.windows(512, halo=0)] == [(0, 0), (1024, 0)]
    # a 1 px halo pulls in the block to the right (its window grown by 1 px touches column 511)
    assert (512, 0) in [(int(w.col_off), int(w.row_off)) for w in mask.windows(512, halo=1)]
    assert mask.block_index(512, halo=1) == [[0, 0], [0, 1], [0, 2]]
    assert mask.block_index(512, halo=0) == [[0, 0], [0, 2]]
    s = mask.summary(512, halo=0)
    assert s["aoi_pixels"] == int(inside.sum()) and s["grid_pixels"] == 300 * 1100
    assert s["blocks_active"] == 2 and s["blocks_total"] == 3
    assert s["work_pixels"] == 512 * 300 + 76 * 300 and s["aoi_share_of_grid"] == round(inside.mean(), 4)
    # reads clip to the raster and are False outside it
    got = mask.read(Window(-5, 15, 20, 10))
    assert got.shape == (10, 20) and not got[:, :5].any() and got[5:, 5:].all() and not got[:5].any()
    assert not mask.read(Window(2000, 0, 10, 10)).any()
    assert mask.dense().shape == inside.shape and np.array_equal(mask.dense(), inside)
    assert mask.footprint(0).bounds == (500000.0, 5000000.0 - 3000, 500000.0 + 11000, 5000000.0)


def test_identity_depends_on_content_and_halo(tmp_path):
    a = np.zeros((40, 40), bool)
    a[5:20, 5:20] = True
    b = a.copy()
    b[30, 30] = True
    ma, mb = AoiMask(mask_from(tmp_path / "a.tif", a), 2), AoiMask(mask_from(tmp_path / "b.tif", b), 2)
    assert ma.identity() != mb.identity() and ma.identity() != ma.with_halo(3).identity()
    assert ma.identity() == AoiMask(tmp_path / "a.tif", 2).identity()
    assert stac_import.grid_identity(tmp_path / "a.tif") != stac_import.grid_identity(tmp_path / "a.tif", ma)


def test_mask_must_match_the_grid(tmp_path):
    mask = AoiMask(mask_from(tmp_path / "m.tif", np.ones((10, 10), bool)))
    other = write_raster(tmp_path / "o.tif", np.ones((10, 12), "uint8"))
    with rasterio.open(other) as ds, pytest.raises(raster.RasterInputError, match="not on the grid"):
        mask.check_grid(ds, "the composites")
    with pytest.raises(raster.RasterInputError, match="does not exist"):
        AoiMask(tmp_path / "missing.tif")
    with pytest.raises(ValueError, match="halo"):
        AoiMask(tmp_path / "m.tif", -1)


def test_grid_writes_mask_and_block_index_only_on_request(tmp_path):
    geom = S.aoi_geometry()
    frame = gpd.GeoDataFrame(geometry=[geom], crs=CRS)
    plain = raster.make_grid(frame, CRS, 10, tmp_path / "plain" / "grid.tif")
    assert not aoi.default_mask_path(plain).exists()                      # library default: unchanged outputs
    grid = raster.make_grid(frame, CRS, 10, tmp_path / "g" / "grid.tif", aoi_mask=True, halo_px=3)
    mask_path = aoi.default_mask_path(grid)
    assert mask_path.name == "grid_aoi.tif" and mask_path.exists()
    with rasterio.open(grid) as g, rasterio.open(mask_path) as m:
        assert g.shape == m.shape and m.dtypes[0] == "uint8" and m.nodata == 0
        assert np.array_equal(g.read(1), m.read(1)) and 0 < m.read(1).mean() < 1
    body = json.loads(aoi.index_path(mask_path).read_text())
    mask = AoiMask(mask_path, 3)
    assert body["schema"] == "geocrop-aoi/1" and body["halo_px"] == 3 and body["block_px"] == 512
    assert body["aoi_pixels"] == mask.counts()["aoi_pixels"] and body["blocks"] == mask.block_index(512, 3)
    assert 0 < body["aoi_share_of_grid"] < 1 and body["blocks_active"] <= body["blocks_total"]


def test_cli_grid_writes_the_mask(tmp_path, capsys):
    from geocrop.cli import main
    path = tmp_path / "aoi.geojson"
    path.write_text(gpd.GeoDataFrame(geometry=[S.aoi_geometry()], crs=CRS).to_json())
    main(["grid", "--aoi", str(path), "--crs", CRS, "--out", str(tmp_path / "a" / "grid.tif")])
    captured = capsys.readouterr()
    assert (tmp_path / "a" / "grid_aoi.tif").exists() and (tmp_path / "a" / "grid_aoi.json").exists()
    assert "of the grid is inside the AOI" in captured.err and captured.out.strip().endswith("grid.tif")
    main(["grid", "--aoi", str(path), "--crs", CRS, "--out", str(tmp_path / "b" / "grid.tif"), "--no-aoi-mask"])
    assert not (tmp_path / "b" / "grid_aoi.tif").exists()


# --------------------------------------------------------------------------- configuration

def test_config_keys_discovery_and_halo(tmp_path):
    grid = write_raster(tmp_path / "grid.tif", np.ones((20, 20), "uint8"), nodata=0)
    cfg = config.load(overrides={"grid": {"reference": str(grid)}, "paths": {"output": str(tmp_path / "run")}})
    assert aoi.from_config(cfg) is None                                      # nothing configured or found
    inside = np.zeros((20, 20), bool)
    inside[2:9, 2:9] = True
    found = mask_from(aoi.default_mask_path(grid), inside)
    mask = aoi.from_config(cfg)
    assert mask.path == found and mask.halo == 0                             # auto-discovery next to the grid
    cfg["grid"]["aoi_halo_px"] = 2
    cfg["predict"]["smoothing"] = {"enabled": True, "window": 9, "smoothness": 10.0}
    assert aoi.from_config(cfg).halo == 4                                    # the smoothing radius is added
    cfg["predict"]["smoothing"]["window"] = 3
    assert aoi.from_config(cfg).halo == 2                                    # configured halo larger
    cfg["grid"]["aoi_mask"] = False
    assert aoi.from_config(cfg) is None                                      # explicit off beats discovery
    cfg["grid"]["aoi_mask"] = str(tmp_path / "missing.tif")
    with pytest.raises(FileNotFoundError, match="grid.aoi_mask"):
        aoi.from_config(cfg)
    cfg["grid"]["aoi_mask"] = str(found)
    assert aoi.from_config(cfg).path == found
    for key, bad in (("aoi_halo_px", -1), ("aoi_halo_px", 1.5), ("aoi_mask", 5), ("aoi", 3)):
        with pytest.raises(config.ConfigError, match="grid"):
            config.load(overrides={"grid": {key: bad}})


def test_vector_aoi_is_rasterised_onto_the_grid(tmp_path):
    grid = write_raster(tmp_path / "grid.tif", np.ones((30, 30), "uint8"), x0=0, y0=300, nodata=0)
    vector = tmp_path / "aoi.geojson"
    vector.write_text(gpd.GeoDataFrame(geometry=[box(0, 150, 150, 300)], crs=CRS).to_json())
    cfg = config.load(overrides={"grid": {"reference": str(grid), "aoi": str(vector)}})
    mask = aoi.from_config(cfg, tmp_path / "run")
    assert mask.path == tmp_path / "run" / "aoi" / "aoi_mask.tif"
    inside = mask.dense()
    assert inside[:15, :15].all() and not inside[15:].any() and not inside[:, 15:].any()


def test_unused_options_keep_config_fingerprints():
    assert config.fingerprint(config.load()) == "09ff103f5880447d"          # as before the keys existed
    assert config.fingerprint(config.load(overrides={"grid": {"aoi_halo_px": 4}})) != "09ff103f5880447d"
    runner_cfg = config.load(overrides={"paths": {"output": "run"}})
    assert pipeline._fp_with(("x",), {}) == pipeline._fp("x")               # empty extras add nothing
    assert "aoi" not in json.dumps(config.without_neutral_defaults(runner_cfg)["grid"])


# --------------------------------------------------------------------------- the pipeline, with and without a mask

@pytest.fixture(scope="module")
def study(tmp_path_factory):
    return S.build(tmp_path_factory.mktemp("aoi_study"))


def run_pipeline(study, name: str, aoi_on: bool, **extra) -> dict:
    cfg = S.config(study, name, aoi_on, uncertainty={"shift_detection": False}, **extra)
    path = Path(study["root"]) / f"{name}.yaml"
    path.write_text(yaml.safe_dump(cfg))
    logs: list[str] = []
    pipeline.run(path, STAGES, log=logs.append)
    return {"cfg": cfg, "root": Path(study["root"]) / name, "logs": logs}


@pytest.fixture(scope="module")
def runs(study):
    return {"full": run_pipeline(study, "full", False), "masked": run_pipeline(study, "masked", True)}


def test_study_geometry_exercises_the_halo(study):
    """The block right of the AOI is active only through the 3 px smoothing halo; one block is far from the AOI."""
    mask = S.load_mask(study, 3)
    assert {tuple(b) for b in mask.block_index(512, 0)} == {(0, 0), (1, 0)}
    assert {tuple(b) for b in mask.block_index(512, 3)} == {(0, 0), (0, 1), (1, 0)}
    assert 0.3 < mask.counts()["aoi_pixels"] / mask.counts()["grid_pixels"] < 0.6


def test_composites_features_and_predictions_equal_inside_the_aoi(study, runs):
    mask = S.load_mask(study, 3)
    inside, active = mask.dense(), inside_blocks(mask, 512, 3)
    full, masked = runs["full"], runs["masked"]
    for p in sorted((full["root"] / "composites").rglob("*.tif")):          # intermediates: active blocks computed
        assert_equal_where(p, masked["root"] / p.relative_to(full["root"]), active, active | ~active)
    assert_equal_where(full["root"] / "stack.tif", masked["root"] / "stack.tif", active, np.ones_like(active))
    for name in ("classes", "probabilities", "probabilities_unsmoothed", "uncertainty", "aoa"):   # finals: AOI pixels
        f, m = (r["root"] / "run" / "prediction" / f"{name}.tif" for r in (full, masked))
        assert_equal_where(f, m, inside, np.ones_like(inside))
    for stage in ("composite", "features", "predict"):
        assert f"[run ] {stage}" in masked["logs"]


def test_nothing_outside_the_aoi_is_classified(study, runs):
    mask = S.load_mask(study, 3)
    inside = mask.dense()
    pred = runs["masked"]["root"] / "run" / "prediction"
    classes, _ = read(pred / "classes.tif")
    assert np.all(classes[0][~inside] == 255) and np.all(classes[0][inside] != 255)
    probabilities, _ = read(pred / "probabilities.tif")
    assert np.all(probabilities[:, ~inside] == raster.NODATA) and np.all(probabilities[:, inside] >= 0)
    unsmoothed, _ = read(pred / "probabilities_unsmoothed.tif")       # smoothing context is masked afterwards too
    assert np.all(unsmoothed[:, ~inside] == raster.NODATA)
    manifest = json.loads((pred / "prediction_manifest.json").read_text())
    assert manifest["aoi"]["halo_px"] == 3 and manifest["aoi"]["mask"] == "grid_aoi.tif"
    full_manifest = json.loads((runs["full"]["root"] / "run" / "prediction" / "prediction_manifest.json").read_text())
    assert "aoi" not in full_manifest


def test_blocks_outside_the_aoi_are_not_computed_or_stored(study, runs):
    """The far block is unwritten (sparse): it reads as no-data and takes no space."""
    mask = S.load_mask(study, 3)
    far = Window(512, 512, mask.width - 512, mask.height - 512)
    assert not mask.active(far)
    with rasterio.open(runs["masked"]["root"] / "stack.tif") as ds:
        assert np.all(ds.read(window=far) == raster.NODATA)
        assert ds.read(1, window=Window(0, 0, 512, 512)).max() > 0
    full_size = (runs["full"]["root"] / "stack.tif").stat().st_size
    assert (runs["masked"]["root"] / "stack.tif").stat().st_size < full_size
    composite = json.loads((runs["masked"]["root"] / "composites" / "P1" / "composite.json").read_text())
    assert composite["aoi_halo_px"] == 3 and composite["aoi_blocks"] == 3


def test_parcels_inside_the_aoi_are_unchanged(runs):
    a = gpd.read_file(runs["full"]["root"] / "run" / "parcels" / "parcels.gpkg").drop(columns="geometry")
    b = gpd.read_file(runs["masked"]["root"] / "run" / "parcels" / "parcels.gpkg").drop(columns="geometry")
    assert len(a) == len(b) > 10
    assert a.equals(b)


def test_stage_fingerprints_contain_the_mask_only_when_used(study, runs):
    states = {k: json.loads((v["root"] / "run" / "stages.json").read_text()) for k, v in runs.items()}
    for stage in ("composite", "features", "predict"):
        assert states["full"][stage]["fingerprint"] != states["masked"][stage]["fingerprint"]
    plain = pipeline.Runner(config.load(overrides={"paths": {"output": str(study["root"] / "fp1")}}), log=print)
    assert plain.aoi is None and plain._aoi_extra() == {}
    cfg = config.load(overrides={"grid": {"reference": str(study["grid"])}, "paths": {"output": str(study["root"] / "fp2")}})
    on = pipeline.Runner(cfg, log=print)
    assert on.aoi is not None and on._aoi_extra()["aoi"]["halo_px"] == 0
    fp_on = pipeline._fp(on._aoi_extra())
    other = np.zeros((10, 10), bool)
    other[2:4, 2:4] = True
    other_mask = mask_from(Path(study["root"]) / "other_aoi.tif", other)
    assert pipeline._fp({"aoi": {"mask": other_mask, "halo_px": 0}}) != fp_on


def test_rerun_with_a_mask_skips_unchanged_stages(study, runs):
    path = Path(study["root"]) / "masked.yaml"
    logs: list[str] = []
    pipeline.run(path, STAGES, log=logs.append)
    assert not [line for line in logs if line.startswith("[run ]")], logs


def test_unmasked_run_does_not_use_the_mask(study, runs):
    """aoi_mask: false turns off the auto-discovered mask: full grid, no sparse blocks."""
    far = Window(512, 512, 64, 64)
    with rasterio.open(runs["full"]["root"] / "run" / "prediction" / "classes.tif") as ds:
        assert (ds.read(1, window=far) != 255).any()
    assert "aoi_blocks" not in json.loads((runs["full"]["root"] / "composites" / "P1" / "composite.json").read_text())


# --------------------------------------------------------------------------- preflight and the disk guard

def test_preflight_counts_aoi_pixels_and_reports_the_share(study):
    cfg = S.config(study, "pf", True, uncertainty={"shift_detection": False})
    full_cfg = S.config(study, "pf_full", False)
    cfg, full_cfg = config.load(overrides=cfg), config.load(overrides=full_cfg)
    layout, plain = preflight.plan(cfg), preflight.plan(full_cfg)
    mask = S.load_mask(study, 3)
    assert plain["aoi"] is None and plain["work_pixels"] == plain["pixels"] == mask.width * mask.height
    assert layout["pixels"] == plain["pixels"] and layout["aoi"]["aoi_pixels"] == mask.counts()["aoi_pixels"]
    assert layout["work_pixels"] == layout["aoi"]["work_pixels"] == int(inside_blocks(mask, 512, 3).sum())
    est, est_full = preflight.estimate(cfg, layout, 100, 4), preflight.estimate(full_cfg, plain, 100, 4)
    assert est["grid_pixels"] == est_full["grid_pixels"] and est["aoi_pixels"] == layout["aoi"]["aoi_pixels"]
    assert est["work_pixels"] == layout["work_pixels"] and est_full["aoi_pixels"] is None
    for stage in preflight.HEAVY_STAGES:
        assert preflight.stage_bytes(cfg, layout, stage, 4) < preflight.stage_bytes(full_cfg, plain, stage, 4)
    report = preflight.run(cfg)
    share = next(c for c in report["checks"] if c["code"] == "grid.aoi_share")
    assert share["status"] == "pass" and "AOI share of grid" in share["message"] and "%" in share["message"]
    assert share["details"]["aoi_pixels"] == layout["aoi"]["aoi_pixels"]
    assert report["resources"]["aoi_pixels"] and report["resources"]["work_megapixels"]
    assert "AOI mask" in preflight.to_markdown(report)


def test_preflight_suggests_the_mask_when_the_grid_values_show_an_irregular_aoi(tmp_path):
    frame = gpd.GeoDataFrame(geometry=[S.aoi_geometry()], crs=CRS)
    grid = raster.make_grid(frame, CRS, 10, tmp_path / "grid.tif")        # grid values 1/0, but no mask file
    cfg = config.load(overrides={"grid": {"reference": str(grid)}, "paths": {"output": str(tmp_path / "run")}})
    layout = preflight.plan(cfg)
    assert layout["aoi"] is None
    rep = preflight._Report()
    out: dict = {}
    preflight._check_aoi({**layout, "grid": layout["grid"]}, rep, out)
    msg = rep.checks[0]["message"]
    assert rep.checks[0]["code"] == "grid.aoi_share" and "No AOI mask is configured" in msg and "geocrop grid" in msg
    bad = config.load(overrides={"grid": {"reference": str(grid), "aoi_mask": str(tmp_path / "gone.tif")}})
    assert "grid.aoi_mask" in preflight.plan(bad)["aoi_error"]


def test_disk_guard_estimates_the_mask_not_the_bounding_box(study):
    cfg = config.load(overrides=S.config(study, "guard", True))
    cfg_full = config.load(overrides=S.config(study, "guard_full", False))
    info, info_full = (preflight.check_disk(c, "features") for c in (cfg, cfg_full))
    assert info["estimate_gb"] <= info_full["estimate_gb"]


# --------------------------------------------------------------------------- stage-level tests on a small grid

@pytest.fixture(scope="module")
def small(tmp_path_factory):
    """192 px demo grid (3 x 3 blocks of 64 px) with an AOI whose right edge ends on a block boundary (column 63)
    plus a disk; scene folders, composites and a trained model."""
    root = tmp_path_factory.mktemp("aoi_small")
    data = demo.generate_scenes(root, s2_every=20, s1_every=40)
    ref = Path(data["reference"])
    inside = np.zeros((192, 192), bool)
    inside[10:101, 0:64] = True
    yy, xx = np.ogrid[:192, :192]
    inside |= (yy - 140) ** 2 + (xx - 130) ** 2 < 25 ** 2
    with rasterio.open(ref) as ds:
        transform = ds.transform
    mask = write_raster(root / "mask.tif", inside.astype("uint8"), x0=transform.c, y0=transform.f, nodata=0, crs=demo.CRS)
    return {"root": root, "data": data, "ref": ref, "mask": mask, "inside": inside}


def test_block_activity_of_the_small_grid(small):
    mask = AoiMask(small["mask"])
    assert {tuple(b) for b in mask.block_index(64, 0)} == {(0, 0), (1, 0), (1, 1), (1, 2), (2, 1), (2, 2)}
    assert {tuple(b) for b in AoiMask(small["mask"], 3).block_index(64)} == {*{tuple(b) for b in mask.block_index(64, 0)},
                                                                         (0, 1)}


def test_s2_and_s1_composites_skip_blocks_and_match(small, tmp_path):
    mask = AoiMask(small["mask"], 1)
    active = inside_blocks(mask, 64)
    s2_dirs = [Path(p) for p in small["data"]["s2_scenes"][:6]]
    s1_dirs = [Path(p) for p in small["data"]["s1_scenes"][:3]]
    for kind, func, dirs, kwargs, names in (
            ("s2", s2.composite, s2_dirs, {"reference": small["ref"], "mask_dilation_px": 2, "block_size": 64,
                                           "temporal_screening": True}, [*S2_BANDS, "valid_count"]),
            ("s1", s1.composite, s1_dirs, {"reference": small["ref"], "block_size": 64}, ["VV", "VH"])):
        func(dirs, tmp_path / f"{kind}_full", **kwargs)
        func(dirs, tmp_path / f"{kind}_masked", aoi=mask, **kwargs)
        for name in names:
            assert_equal_where(tmp_path / f"{kind}_full" / f"{name}.tif", tmp_path / f"{kind}_masked" / f"{name}.tif",
                               active, np.ones_like(active))
    # the masked composite really skipped the inactive blocks (no read of their scene data)
    calls = []
    real = s2.read_aligned
    s2.read_aligned = lambda src, transform, *a, **k: (calls.append(transform), real(src, transform, *a, **k))[1]
    try:
        s2.composite(s2_dirs[:2], tmp_path / "count_masked", aoi=mask, **{**kwargs_s2(small)})
        masked_calls = len(calls)
        calls.clear()
        s2.composite(s2_dirs[:2], tmp_path / "count_full", **kwargs_s2(small))
    finally:
        s2.read_aligned = real
    assert 0 < masked_calls < len(calls)


def kwargs_s2(small):
    return {"reference": small["ref"], "block_size": 64}


def test_composite_refuses_a_mask_of_another_grid(small, tmp_path):
    other = AoiMask(mask_from(tmp_path / "m.tif", np.ones((10, 10), bool)))
    with pytest.raises(raster.RasterInputError, match="not on the grid"):
        s2.composite([Path(small["data"]["s2_scenes"][0])], tmp_path / "c", reference=small["ref"], aoi=other)


@pytest.fixture(scope="module")
def stacks(small, tmp_path_factory):
    """Feature stacks of the small grid: unmasked, masked with the smoothing halo, and with halo 0."""
    out = tmp_path_factory.mktemp("aoi_stacks")
    comp = small["data"]["composites"]
    features.build_stack(comp, out / "full.tif", block_size=64)
    features.build_stack(comp, out / "halo3.tif", block_size=64, aoi=AoiMask(small["mask"], 3))
    features.build_stack(comp, out / "halo0.tif", block_size=64, aoi=AoiMask(small["mask"], 0))
    return out


def test_feature_stack_blocks_and_manifest(small, stacks):
    mask = AoiMask(small["mask"], 3)
    active = inside_blocks(mask, 64)
    assert_equal_where(stacks / "full.tif", stacks / "halo3.tif", active, np.ones_like(active))
    assert features.read_manifest(stacks / "halo3.tif")["aoi_halo_px"] == 3
    assert "aoi_mask" not in features.read_manifest(stacks / "full.tif")
    # halo 0 leaves the block right of the AOI empty; with halo 3 it holds the smoothing context
    ctx = Window(64, 0, 64, 64)
    with rasterio.open(stacks / "halo0.tif") as a, rasterio.open(stacks / "halo3.tif") as b:
        assert np.all(a.read(window=ctx) == raster.NODATA) and (b.read(window=ctx) != raster.NODATA).any()


@pytest.fixture(scope="module")
def trained(small, stacks, tmp_path_factory):
    from geocrop.samples import extract
    out = tmp_path_factory.mktemp("aoi_model")
    gdf = gpd.read_file(small["data"]["training"])
    gdf.to_file(out / "training.gpkg", driver="GPKG")
    samples = extract(stacks / "full.tif", out / "training.gpkg", max_per_polygon=30)
    model.train(samples, out / "model", params={"n_estimators": 12, "n_jobs": 1}, n_splits=3, spatial_block_m=500)
    return out / "model" / "model.joblib"


SMOOTH = {"enabled": True, "window": 7, "smoothness": 10.0}


def test_predict_smoothing_and_aoa_equal_inside_the_aoi(small, stacks, trained, tmp_path):
    mask = AoiMask(small["mask"], 3)
    inside = mask.dense()
    full = model.predict(stacks / "full.tif", trained, tmp_path / "full", block_size=64, smoothing=SMOOTH)
    masked = model.predict(stacks / "halo3.tif", trained, tmp_path / "masked", block_size=64, smoothing=SMOOTH, aoi=mask)
    assert full["aoa"] and masked["aoa"]
    for name in ("classes", "probabilities", "probabilities_unsmoothed", "uncertainty", "aoa"):
        assert_equal_where(tmp_path / "full" / f"{name}.tif", tmp_path / "masked" / f"{name}.tif", inside,
                           np.ones_like(inside))
    # at the AOI border pixels that smoothing reads from a neighbouring block: columns 61-63 of the AOI
    edge = np.zeros_like(inside)
    edge[10:64, 61:64] = True
    assert (inside & edge).sum() == 3 * 54
    a, _ = read(tmp_path / "full" / "probabilities.tif")
    b, _ = read(tmp_path / "masked" / "probabilities.tif")
    assert np.allclose(a[:, inside & edge], b[:, inside & edge], atol=1e-6)


def test_predict_without_smoothing_classifies_only_aoi_pixels(small, stacks, trained, tmp_path):
    mask = AoiMask(small["mask"], 0)
    inside = mask.dense()
    model.predict(stacks / "full.tif", trained, tmp_path / "full", block_size=64)
    model.predict(stacks / "halo0.tif", trained, tmp_path / "masked", block_size=64, aoi=mask)
    for name in ("classes", "probabilities", "uncertainty", "aoa"):
        assert_equal_where(tmp_path / "full" / f"{name}.tif", tmp_path / "masked" / f"{name}.tif", inside,
                           np.ones_like(inside))
    assert not (tmp_path / "masked" / "probabilities_unsmoothed.tif").exists()


def test_a_too_small_halo_is_refused_and_a_missing_halo_is_visible(small, stacks, trained, tmp_path):
    with pytest.raises(ValueError, match="smaller than the smoothing radius"):
        model.predict(stacks / "halo0.tif", trained, tmp_path / "x", block_size=64, smoothing=SMOOTH,
                      aoi=AoiMask(small["mask"], 0))
    # Negative control: a stack that was built WITHOUT the halo blocks changes the smoothed result at the AOI
    # border, so the equality tested above is not trivial.
    mask = AoiMask(small["mask"], 3)
    model.predict(stacks / "full.tif", trained, tmp_path / "full", block_size=64, smoothing=SMOOTH)
    model.predict(stacks / "halo0.tif", trained, tmp_path / "short", block_size=64, smoothing=SMOOTH, aoi=mask)
    a, _ = read(tmp_path / "full" / "probabilities.tif")
    b, _ = read(tmp_path / "short" / "probabilities.tif")
    inside = mask.dense()
    border = np.zeros_like(inside)
    border[:, 56:64] = True
    differs = ~np.isclose(a[:, inside & border], b[:, inside & border], atol=1e-6).all(axis=0)
    assert differs.any(), "missing halo blocks should change smoothing at the AOI border"
    far = inside & ~border & (np.arange(192)[None, :] < 56)           # further than the radius from block (0, 1)
    assert np.allclose(a[:, far], b[:, far], atol=1e-6)


def test_merge_stacks_copies_only_active_blocks(small, stacks, tmp_path):
    from geocrop.tsfeatures import merge_stacks
    mask = AoiMask(small["mask"], 3)
    first = stacks / "halo3.tif"
    renamed = tmp_path / "renamed.tif"
    data, _ = read(first)
    with rasterio.open(first) as ds:
        prof = ds.profile
        names = [f"{d}_2" for d in ds.descriptions]
    with rasterio.open(renamed, "w", **prof) as dst:
        dst.write(data)
        for i, n in enumerate(names, 1):
            dst.set_band_description(i, n)
    manifest = features.read_manifest(first)
    second = json.loads(json.dumps(manifest))
    for f in second["features"]:
        f["name"] += "_2"
    features.write_manifest(renamed, second)
    out = merge_stacks([first, renamed], tmp_path / "merged.tif", block_size=64, aoi=mask)
    merged, _ = read(out)
    active = inside_blocks(mask, 64)
    assert np.all(merged[:, ~active] == raster.NODATA) and np.array_equal(merged[: len(data), active], data[:, active])


def test_timeseries_features_skip_blocks_and_match(small, tmp_path):
    from datetime import date

    from geocrop.tsfeatures import build_timeseries_stack
    mask = AoiMask(small["mask"], 2)
    roots = [Path(small["data"]["scene_root"]) / "s2", Path(small["data"]["scene_root"]) / "s1"]
    args = (roots, small["ref"])
    kwargs = {"start": date(2023, 3, 15), "end": date(2023, 10, 31)}
    common = dict(s2_indices=("NDVI",), s1_indices=("VH_dB",), block_size=64, temporal_support=True)
    full = build_timeseries_stack(*args, tmp_path / "full.tif", kwargs["start"], kwargs["end"], **common)
    masked = build_timeseries_stack(*args, tmp_path / "masked.tif", kwargs["start"], kwargs["end"], aoi=mask, **common)
    active = inside_blocks(mask, 64)
    assert active.any() and not active.all()
    for suffix in ("", "_quality", "_temporal_support"):
        f, m = (p.with_name(p.stem + suffix + ".tif") for p in (full, masked))
        assert_equal_where(f, m, active, np.ones_like(active))
    body = json.loads(features.manifest_path(masked).read_text())
    assert body["aoi_halo_px"] == 2
    quality = json.loads((tmp_path / "masked_quality.json").read_text())
    assert quality["series"]["NDVI"]["pixels"] == int(active.sum())            # totals count the computed blocks


def test_segmentation_matches_the_premasked_stack_and_skips_empty_tiles(small, stacks, tmp_path):
    pytest.importorskip("skimage")
    from geocrop.segmentation import obia
    mask = AoiMask(small["mask"], 0)
    inside = mask.dense()
    data, nodata = read(stacks / "full.tif")
    with rasterio.open(stacks / "full.tif") as ds:
        prof, names = ds.profile, ds.descriptions
    premasked = tmp_path / "premasked.tif"
    with rasterio.open(premasked, "w", **prof) as dst:
        dst.write(np.where(inside[None], data, nodata).astype(data.dtype))
        for i, n in enumerate(names, 1):
            dst.set_band_description(i, n)
    bands = [1, 2, 3]
    ref = obia(premasked, tmp_path / "ref.gpkg", bands, tile_size=64, segments_per_km2=2000)
    got = obia(stacks / "full.tif", tmp_path / "got.gpkg", bands, tile_size=64, segments_per_km2=2000, aoi=mask)
    assert got["tiles_skipped_outside_aoi"] == 3 and got["segments"] == ref["segments"]
    a, b = gpd.read_file(tmp_path / "ref.gpkg"), gpd.read_file(tmp_path / "got.gpkg")
    assert list(a.tile) == list(b.tile) and list(a.parts) == list(b.parts)
    assert all(x.equals_exact(y, 1e-6) for x, y in zip(a.geometry, b.geometry, strict=True))
    union = b.geometry.union_all() if hasattr(b.geometry, "union_all") else b.geometry.unary_union
    with rasterio.open(stacks / "full.tif") as ds:
        xs, ys = rasterio.transform.xy(ds.transform, *np.nonzero(~inside))
    outside = [Point(x, y) for x, y in list(zip(xs, ys, strict=True))[::397]]
    assert not any(union.contains(p) for p in outside), "segments must not cover pixels outside the AOI"


# --------------------------------------------------------------------------- imports: SAFE, STAC S2, STAC S1

def _safe(tmp_path: Path, size: int = 16) -> Path:
    """Mini SAFE with GeoTIFF payloads named like JP2 files (GDAL opens by content); 10 m bands `size` px wide."""
    safe = tmp_path / "S2A_MSIL2A_20230705T093041_N0509_R136_T34TCQ_20230705T140000.SAFE"
    img = safe / "GRANULE" / "L2A_T34TCQ" / "IMG_DATA"
    for res in ("R10m", "R20m"):
        (img / res).mkdir(parents=True)
    offsets = "".join(f"<BOA_ADD_OFFSET band_id='{i}'>-1000</BOA_ADD_OFFSET>" for i in range(13))
    (safe / "MTD_MSIL2A.xml").write_text(f"<r><BOA_QUANTIFICATION_VALUE>10000</BOA_QUANTIFICATION_VALUE>{offsets}</r>")
    rng = np.random.default_rng(3)
    for b in S2_BANDS:
        ten = b in ("B02", "B03", "B04", "B08")
        res, n, px = ("R10m", size, 10.0) if ten else ("R20m", size // 2, 20.0)
        write_raster(img / res / f"T34TCQ_20230705T093041_{b}_{res[1:]}.jp2",
                     rng.integers(1500, 4000, (n, n)).astype("uint16"), res=px)
    write_raster(img / "R20m" / "T34TCQ_20230705T093041_SCL_20m.jp2", np.full((size // 2, size // 2), 4, "uint8"),
                 res=20.0)
    return safe


def test_safe_import_reads_and_writes_only_aoi_blocks(tmp_path, monkeypatch):
    safe = _safe(tmp_path)
    grid = write_raster(tmp_path / "grid.tif", np.ones((16, 16), "uint8"), nodata=0)
    inside = np.zeros((16, 16), bool)
    inside[1:6, 1:6] = True                                          # top-left 8 px block only
    mask = AoiMask(mask_from(tmp_path / "m.tif", inside))
    full = s2.import_safe(safe, tmp_path / "full", block_size=8, reference=grid, shadow_projection=False)
    calls = []
    real = s2.read_aligned
    monkeypatch.setattr(s2, "read_aligned", lambda *a, **k: (calls.append(a[1]), real(*a, **k))[1])
    part = s2.import_safe(safe, tmp_path / "part", block_size=8, reference=grid, shadow_projection=False, aoi=mask)
    assert len(calls) == 11                                          # 1 block x (10 bands + SCL): not 4 blocks
    active = inside_blocks(mask, 8)
    for name in (*S2_BANDS, "SCL"):
        assert_equal_where(full / f"{name}.tif", part / f"{name}.tif", active, np.ones_like(active))
    assert json.loads((part / "scene.json").read_text())["aoi_halo_px"] == 0
    with pytest.raises(ValueError, match="needs the study grid"):
        s2.import_safe(safe, tmp_path / "x", aoi=mask)


def _stac_fixture(tmp_path: Path, shape=(300, 1100), seed=5):
    """Grid (EPSG:32634, 10 m), an AOI inside its first 512 px block and a STAC-like item with local COG assets:
    10 m and 20 m bands on a larger raster, SCL with a cloud patch (bilinear resampling of the 20 m bands matters)."""
    rng = np.random.default_rng(seed)
    h, w = shape
    inside = np.zeros(shape, bool)
    yy, xx = np.ogrid[:h, :w]
    inside |= (yy - 120) ** 2 + (xx - 200) ** 2 < 90 ** 2
    inside[240:290, 20:120] = True
    grid = write_raster(tmp_path / "grid.tif", np.ones(shape, "uint8"), nodata=0)
    mask = AoiMask(mask_from(tmp_path / "grid_aoi.tif", inside))
    return grid, mask, inside, rng


def _item(tmp_path: Path, rng, item_id="A", when="2023-07-03T09:48:18", shift_m=0.0) -> pystac.Item:
    folder = tmp_path / item_id
    folder.mkdir(parents=True, exist_ok=True)
    with rasterio.open(tmp_path / "grid.tif") as g:
        left, bottom, right, top = g.bounds
    from rasterio.warp import transform_bounds
    w, s, e, n = transform_bounds(CRS, "EPSG:4326", left - 500, bottom - 500, right + 500, top + 500)
    item = pystac.Item(id=item_id, geometry={"type": "Polygon", "coordinates": [[[w, s], [e, s], [e, n], [w, n], [w, s]]]},
                       bbox=[w, s, e, n], datetime=datetime.fromisoformat(when).replace(tzinfo=timezone.utc),
                       properties={"earthsearch:boa_offset_applied": True, "s2:mgrs_tile": "34TDQ", "platform": "S2A",
                                   "datetime": when + "Z"})
    x0, y0 = left - 500, top + 500
    for band in (*S2_BANDS, "SCL"):
        res = 10.0 if band in ("B02", "B03", "B04", "B08") else 20.0
        cols, rows = int((right - left + 1000) / res), int((top - bottom + 1000) / res)
        if band == "SCL":
            arr = np.full((rows, cols), 4, "uint8")
            arr[: rows // 3, : cols // 4] = 9
        else:
            arr = rng.integers(1200, 4500, (rows, cols)).astype("uint16")
        path = write_raster(folder / f"{band}.tif", arr, x0=x0, y0=y0, res=res, nodata=None)
        asset = pystac.Asset(href=str(path), media_type="image/tiff")
        if band != "SCL":
            asset.extra_fields["raster:bands"] = [{"scale": 1e-4, "offset": -0.1}]
        item.add_asset(stac_import.ASSET_KEYS[band], asset)
    return item


def test_stac_import_reads_only_windows_of_active_blocks_and_matches(tmp_path, monkeypatch):
    grid, mask, inside, rng = _stac_fixture(tmp_path)
    item = _item(tmp_path, rng)
    windows = []
    real = stac_import.read_window_aligned
    monkeypatch.setattr(stac_import, "read_window_aligned",
                        lambda src, transform, crs, shape, *a, **k: (windows.append((transform.c, shape)),
                                                                      real(src, transform, crs, shape, *a, **k))[1])
    full = stac_import.import_item(item, grid, tmp_path / "out" / "full")
    n_full = len(windows)
    windows.clear()
    part = stac_import.import_item(item, grid, tmp_path / "out" / "part", aoi=mask)
    assert full["status"] == part["status"] == "imported"
    # whole-grid reads (11 assets) against one 512 px block per asset: nothing of the other blocks is requested
    assert n_full == 11 and len(windows) == 11
    assert {shape for _, shape in windows} == {(300, 512)} and {c for c, _ in windows} == {500000.0}
    assert part["blocks_read"] == part["blocks_active"] == 1
    active = inside_blocks(mask, 512)
    for name in (*S2_BANDS, "SCL"):
        assert_equal_where(tmp_path / "out" / "full" / f"{name}.tif", tmp_path / "out" / "part" / f"{name}.tif",
                           active, np.ones_like(active))
    meta = json.loads((tmp_path / "out" / "part" / "scene.json").read_text())
    assert meta["grid_id"] == stac_import.grid_identity(grid, mask) != stac_import.grid_identity(grid)
    assert meta["aoi_halo_px"] == 0
    # the clear fraction is measured over the AOI pixels
    scl, _ = read(tmp_path / "out" / "full" / "SCL.tif")
    clear_aoi = float(np.isin(scl[0][inside], [4]).mean())
    assert part["clear_fraction"] == pytest.approx(clear_aoi)


def test_stac_import_skips_items_outside_the_aoi_blocks_before_any_read(tmp_path, monkeypatch):
    import pystac_client
    grid, mask, inside, rng = _stac_fixture(tmp_path)
    near = _item(tmp_path, rng, "NEAR", "2023-07-03T09:48:18")
    far = _item(tmp_path, rng, "FAR", "2023-07-08T09:48:18")
    far.geometry = {"type": "Polygon", "coordinates": [[[5.0, 40.0], [5.1, 40.0], [5.1, 40.1], [5.0, 40.1], [5.0, 40.0]]]}
    for asset in far.assets.values():
        asset.href = str(tmp_path / "does_not_exist.tif")           # a read would fail: it must not happen

    class Search:
        def items(self):
            return iter([near, far])

    class Fake:
        def search(self, **kw):
            return Search()

    monkeypatch.setattr(pystac_client.Client, "open", staticmethod(lambda *a, **k: Fake()))
    res = stac_import.import_season(grid, tmp_path / "scenes", "2023-07-01", "2023-07-31", aoi=mask)
    assert [r["status"] for r in res] == ["imported", "skipped: outside AOI blocks"]
    res = stac_import.import_season(grid, tmp_path / "scenes", "2023-07-01", "2023-07-31", aoi=mask)
    assert res[0]["status"] == "cached"                              # the mask-aware grid id is cached
    cached_plain = stac_import.import_season(grid, tmp_path / "scenes", "2023-07-01", "2023-07-31")
    assert cached_plain[0]["status"] == "imported"                   # an unmasked run does not reuse masked scenes


def _s1_item(tmp_path: Path, rng, item_id="S1A_1", when="2023-07-04T05:00:00") -> pystac.Item:
    folder = tmp_path / item_id
    folder.mkdir(parents=True, exist_ok=True)
    with rasterio.open(tmp_path / "grid.tif") as g:
        left, bottom, right, top = g.bounds
    from rasterio.warp import transform_bounds
    w, s, e, n = transform_bounds(CRS, "EPSG:4326", left - 500, bottom - 500, right + 500, top + 500)
    item = pystac.Item(id=item_id, geometry={"type": "Polygon", "coordinates": [[[w, s], [e, s], [e, n], [w, n], [w, s]]]},
                       bbox=[w, s, e, n], datetime=datetime.fromisoformat(when).replace(tzinfo=timezone.utc),
                       properties={"platform": "SENTINEL-1A", "sat:orbit_state": "ascending", "sat:relative_orbit": 7,
                                   "sat:absolute_orbit": 12345, "datetime": when + "Z"})
    rows, cols = int((top - bottom + 1000) / 10), int((right - left + 1000) / 10)
    for pol in ("vv", "vh"):
        arr = rng.uniform(0.001, 0.3, (rows, cols)).astype("float32")
        arr[: rows // 5] = -1                                          # non-positive: no data
        item.add_asset(pol, pystac.Asset(href=str(write_raster(folder / f"{pol}.tif", arr, x0=left - 500,
                                                               y0=top + 500, nodata=None)), media_type="image/tiff"))
    return item


def test_s1_stac_import_matches_inside_the_aoi_and_skips_blocks(tmp_path, monkeypatch):
    grid, mask, inside, rng = _stac_fixture(tmp_path)
    item = _s1_item(tmp_path, rng)
    calls = []
    real = s1_stac.read_window_aligned_active
    monkeypatch.setattr(s1_stac, "read_window_aligned_active",
                        lambda *a, **k: (calls.append(len(a[-1])), real(*a, **k))[1])
    full = s1_stac.import_item(item, grid, tmp_path / "s1" / "full", min_coverage=0.0)
    part = s1_stac.import_item(item, grid, tmp_path / "s1" / "part", min_coverage=0.0, aoi=mask)
    assert full["status"] == part["status"] == "imported" and calls == [1, 1]       # one active block per polarisation
    active = inside_blocks(mask, 512)
    for pol in ("VV", "VH"):
        assert_equal_where(tmp_path / "s1" / "full" / f"{pol}.tif", tmp_path / "s1" / "part" / f"{pol}.tif", active,
                           np.ones_like(active))
    meta = json.loads((tmp_path / "s1" / "part" / "scene.json").read_text())
    assert meta["grid_id"] == stac_import.grid_identity(grid, mask) and meta["aoi_halo_px"] == 0
    assert part["coverage"] == pytest.approx(
        float((np.isfinite(read(tmp_path / "s1" / "full" / "VV.tif")[0][0]) &
               (read(tmp_path / "s1" / "full" / "VV.tif")[0][0] != raster.NODATA))[inside].mean()))


def test_quicklook_keeps_the_bbox_path_and_takes_an_aoi_geometry(tmp_path, monkeypatch):
    """quicklook(bbox) is unchanged (no mask); quicklook(aoi=...) writes grid + mask and passes the mask on."""
    from geocrop import quicklook as ql
    seen = {}

    def fake_import(grid, out, start, end, endpoint, **kw):
        seen["aoi"] = kw.get("aoi")
        return []

    monkeypatch.setattr(ql, "import_season", fake_import)
    ql.quicklook([19.10, 44.72, 19.12, 44.74], "2023-07-01", "2023-07-31", tmp_path / "bbox")
    assert seen["aoi"] is None and not (tmp_path / "bbox" / "grid_aoi.tif").exists()
    triangle = gpd.GeoDataFrame(geometry=[box(19.10, 44.72, 19.12, 44.74).intersection(
        Point(19.10, 44.72).buffer(0.03))], crs="EPSG:4326")
    ql.quicklook(None, "2023-07-01", "2023-07-31", tmp_path / "aoi", aoi=triangle)
    assert isinstance(seen["aoi"], AoiMask) and (tmp_path / "aoi" / "grid_aoi.tif").exists()


def test_adjacent_blocks_are_read_as_one_run():
    from rasterio.windows import Window

    from geocrop.stac_import import merge_runs
    blocks_ = [Window(1024, 0, 512, 512), Window(0, 0, 512, 512), Window(512, 0, 512, 512),
               Window(512, 512, 512, 512), Window(1536, 512, 512, 512)]
    runs = [(int(w.col_off), int(w.row_off), int(w.width), int(w.height)) for w in merge_runs(blocks_)]
    assert runs == [(0, 0, 1536, 512), (512, 512, 512, 512), (1536, 512, 512, 512)]
