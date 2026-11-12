"""Segment-density sweep on synthetic patchwork rasters (no real data)."""
import json

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

from geocrop import segment_sweep
from geocrop.cli import build_parser, main

X0, Y0, N = 500000, 5000800, 80          # 80 x 80 px of 10 m = 64 ha
CLASSES = ["forest", "maize", "soybean"]


def _layout():
    cls = np.zeros((N, N), dtype=int)
    cls[:, 40:] = 1
    cls[40:, :40] = 2
    cls[40:, 40:] = 0
    return cls


def _raster(tmp_path):
    cls = _layout()
    rng = np.random.default_rng(3)
    level = np.array([0.2, 0.5, 0.8])[cls]
    stack = np.stack([level + rng.normal(0, 0.03, cls.shape), 1 - level + rng.normal(0, 0.03, cls.shape)]).astype("float32")
    path = tmp_path / "stack.tif"
    with rasterio.open(path, "w", driver="GTiff", height=N, width=N, count=2, dtype="float32", crs="EPSG:32633",
                       transform=from_origin(X0, Y0, 10, 10)) as dst:
        dst.write(stack)
    return path, cls


def _probs(tmp_path, cls):
    z = np.full((3, N, N), 0.03, dtype="float32")
    for i in range(3):
        z[i][cls == i] = 0.94
    path = tmp_path / "prob.tif"
    with rasterio.open(path, "w", driver="GTiff", height=N, width=N, count=3, dtype="float32", crs="EPSG:32633",
                       transform=from_origin(X0, Y0, 10, 10)) as dst:
        dst.write(z)
        dst.descriptions = CLASSES
    return path


def _box(c0, r0, c1, r1):
    return box(X0 + c0 * 10, Y0 - r1 * 10, X0 + c1 * 10, Y0 - r0 * 10)


def _parcels(tmp_path):
    g = gpd.GeoDataFrame({"parcel_id": ["a", "b", "c", "d"], "declared_crop": ["forest", "maize", "soybean", "maize"],
                          "geometry": [_box(0, 0, 40, 40), _box(40, 0, 80, 40), _box(0, 40, 40, 80), _box(40, 40, 80, 80)]},
                         crs="EPSG:32633")
    path = tmp_path / "parcels.gpkg"
    g.to_file(path, driver="GPKG")
    return path


def test_finer_density_gives_more_smaller_segments(tmp_path):
    raster, _ = _raster(tmp_path)
    r = segment_sweep.sweep(raster, tmp_path / "o", densities=(20, 80, 300), tile_size=64, workers=1)
    t = pd.read_csv(r["csv"])
    assert t.segments.is_monotonic_increasing and t.segments.iloc[-1] > 2 * t.segments.iloc[0]
    assert t.area_median_ha.is_monotonic_decreasing
    assert (t.area_min_ha <= t.area_median_ha).all() and (t.area_median_ha <= t.area_max_ha).all()
    assert (t.realised_per_km2 > 0).all()
    man = json.loads((tmp_path / "o" / "segment_sweep_manifest.json").read_text())
    assert man["inputs"]["raster_sha256"] and man["config"]["densities"] == [20, 80, 300] and man["seed"] is None
    assert "not an independent GEOBIA" in (tmp_path / "o" / "segment_sweep.md").read_text()


def test_parcel_comparison_and_purity(tmp_path):
    raster, _ = _raster(tmp_path)
    parcels = _parcels(tmp_path)
    r = segment_sweep.sweep(raster, tmp_path / "o", densities=(5, 400), tile_size=80, workers=1, parcels=parcels,
                            reference=parcels, label_field="declared_crop")
    t = pd.read_csv(r["csv"])
    coarse, fine = t.iloc[0], t.iloc[1]
    assert fine.segments_per_parcel_mean > coarse.segments_per_parcel_mean
    assert fine.parcels_with_min_segments_share >= coarse.parcels_with_min_segments_share
    assert coarse.parcel_median_ha == pytest.approx(16.0)
    assert fine.median_parcel_to_median_segment > coarse.median_parcel_to_median_segment
    assert 0 <= fine.boundary_mismatch_area_share <= 1 and 0 <= coarse.boundary_mismatch_area_share <= 1
    assert fine.purity_area_weighted >= coarse.purity_area_weighted - 1e-9
    assert fine.purity_area_weighted > 0.9


def test_purity_and_mismatch_on_known_geometry():
    crs = "EPSG:32633"
    ref = gpd.GeoDataFrame({"label": ["a", "b"], "geometry": [box(0, 0, 100, 100), box(100, 0, 200, 100)]}, crs=crs)
    pure = gpd.GeoDataFrame(geometry=[box(0, 0, 100, 100), box(100, 0, 200, 100)], crs=crs)
    mixed = gpd.GeoDataFrame(geometry=[box(0, 0, 200, 100)], crs=crs)
    assert segment_sweep.purity(pure, ref, "label")["purity_area_weighted"] == pytest.approx(1)
    assert segment_sweep.purity(mixed, ref, "label")["purity_area_weighted"] == pytest.approx(0.5)
    assert segment_sweep.parcel_comparison(pure, ref)["boundary_mismatch_area_share"] == pytest.approx(0)
    m = segment_sweep.parcel_comparison(mixed, ref)
    assert m["boundary_mismatch_area_share"] == pytest.approx(1) and m["parcels_with_min_segments_share"] == 0
    with pytest.raises(ValueError):
        segment_sweep.purity(pure, ref, "nope")


def test_presence_comparison_and_cli(tmp_path, capsys):
    raster, cls = _raster(tmp_path)
    probs = _probs(tmp_path, cls)
    out = tmp_path / "cli"
    main(["segmentation-sweep", "--raster", str(raster), "--out", str(out), "--densities", "10", "200", "--tile-size", "80",
          "--workers", "1", "--parcels", str(_parcels(tmp_path)), "--probabilities", str(probs),
          "--discard-segments"])
    t = pd.read_csv(out / "segment_sweep.csv")
    assert len(t) == 2 and (t.presence_parcels == 4).all()
    assert (t.status_changed >= 0).all() and ((t.supported_share >= 0) & (t.supported_share <= 1)).all()
    assert not list(out.glob("segments_d*.gpkg")) and (out / "presence_baseline" / "crop_presence.csv").exists()
    assert "segmentation-sweep" in build_parser().format_help()


def test_invalid_densities(tmp_path):
    raster, _ = _raster(tmp_path)
    for bad in ([], [0], [10, 10], [-1]):
        with pytest.raises(ValueError):
            segment_sweep.sweep(raster, tmp_path / "o", densities=bad)
    with pytest.raises(ValueError):
        segment_sweep.sweep(raster, tmp_path / "o", densities=(10,), reference="x.gpkg")


def test_parcel_coverage_detects_gaps():
    crs = "EPSG:32633"
    parcels = gpd.GeoDataFrame(geometry=[box(0, 0, 100, 100)], crs=crs)
    full = gpd.GeoDataFrame(geometry=[box(0, 0, 50, 100), box(50, 0, 100, 100)], crs=crs)
    gap = gpd.GeoDataFrame(geometry=[box(0, 0, 50, 100), box(60, 0, 100, 100)], crs=crs)
    assert segment_sweep.parcel_coverage(full, parcels) == pytest.approx(1)
    assert segment_sweep.parcel_coverage(gap, parcels) == pytest.approx(0.9)


def test_segments_per_tile_estimate_and_memory_warning(tmp_path):
    # 1000/km2 at 10 m in a 1024-pixel tile is the setting that exhausted 48 GB; 512-pixel tiles are a quarter of it
    assert segment_sweep.segments_per_tile(1000, 1024, 10) > segment_sweep.MAX_SEGMENTS_PER_TILE
    assert segment_sweep.segments_per_tile(1000, 512, 10) < segment_sweep.MAX_SEGMENTS_PER_TILE
    raster, _ = _raster(tmp_path)
    r = segment_sweep.sweep(raster, tmp_path / "o", densities=(20,), tile_size=64, workers=1)
    assert r["warnings"] == []
    assert json.loads((tmp_path / "o" / "segment_sweep_manifest.json").read_text())["warnings"] == []
