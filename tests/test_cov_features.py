"""Error paths of feature planning, time-series stacks and the time-series helpers (synthetic, fast)."""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from geocrop import features, indices, scenes, timeseries, tsfeatures
from geocrop.features import Feature
from geocrop.raster import RasterInputError


def _touch(folder: Path, names) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for n in names:
        (folder / f"{n}.tif").write_bytes(b"")


def test_plan_rejects_inconsistent_composite_layouts(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="No window folders"):
        features.plan(empty)
    partial = tmp_path / "partial"
    _touch(partial / "P1", indices.S2_BANDS)
    _touch(partial / "P2", indices.S2_BANDS[:3])
    with pytest.raises(ValueError, match="every window"):
        features.plan(partial)
    orbits = tmp_path / "orbits"
    _touch(orbits / "P1", ["VV", "VH"])
    _touch(orbits / "P2", ["ASC_VV", "ASC_VH"])
    with pytest.raises(ValueError, match="same orbit layout"):
        features.plan(orbits)
    nothing = tmp_path / "nothing"
    _touch(nothing / "P1", ["notes"])
    with pytest.raises(ValueError, match="No Sentinel-2 or Sentinel-1"):
        features.plan(nothing)


def test_plan_s1_only_single_window_has_no_deltas_or_seasonal(tmp_path):
    root = tmp_path / "s1"
    _touch(root / "P1", ["DESC_VV", "DESC_VH"])
    periods, feats, layout = features.plan(root)
    assert periods == ["P1"] and layout == {"s2": False, "s1_prefixes": ["DESC_"]}
    assert feats and all(f.sensor == "S1" and f.name.startswith("P1_DESC_") for f in feats)
    assert not any(f.group in ("delta", "seasonal") for f in feats)


def test_build_stack_rejects_composites_without_crs(tmp_path):
    folder = tmp_path / "comp" / "P1"
    folder.mkdir(parents=True)
    prof = {"driver": "GTiff", "height": 4, "width": 4, "count": 1, "dtype": "float32",
            "transform": from_origin(0, 40, 10, 10)}
    for b in indices.S2_BANDS:
        with rasterio.open(folder / f"{b}.tif", "w", **prof) as ds:
            ds.write(np.full((1, 4, 4), 0.2, dtype="float32"))
    with pytest.raises(RasterInputError, match="no CRS"):
        features.build_stack(tmp_path / "comp", tmp_path / "stack.tif", s1=())


def test_manifest_missing_and_selection_rules(tmp_path):
    with pytest.raises(FileNotFoundError, match="manifest"):
        features.read_manifest(tmp_path / "stack.tif")
    feats = [Feature("P1_NDVI", "s2_index", "S2", "P1", ("P1",)),
             Feature("P2_NDVI", "s2_index", "S2", "P2", ("P2",)),
             Feature("delta_NDVI_P1_P2", "delta", "S2", None, ("P1", "P2")),
             Feature("season_NDVI_max", "seasonal", "S2", None, ("P1", "P2")),
             Feature("P1_B04", "s2_band", "S2", "P1", ("P1",))]
    assert features.select(feats, periods=["P1"]) == ["P1_NDVI", "P1_B04"]
    assert features.select(feats, periods=["P1", "P2"], groups=["delta"]) == ["delta_NDVI_P1_P2"]
    assert features.select(feats, exclude=["NDVI"]) == ["P1_B04"]
    assert features.select(feats, names=["P2_NDVI", "unknown"]) == ["P2_NDVI"]
    with pytest.raises(ValueError, match="empty"):
        features.select(feats, groups=["s1"])
    features.write_manifest(tmp_path / "stack.tif", {"features": [{"name": "a", "group": "ts", "sensor": "S2"}]})
    assert features.load_manifest(tmp_path / "stack.tif")[0].periods == ()


# ------------------------------------------------------------------------------------------- tsfeatures
@pytest.mark.parametrize("kwargs,match", [
    ({"min_observed_fraction": 1.5}, "min_observed_fraction"),
    ({"min_observed_fraction": float("nan")}, "min_observed_fraction"),
    ({"max_gap_days": -1}, "max_gap_days"),
    ({"max_gap_days": float("inf")}, "max_gap_days"),
    ({"step_days": 5, "output_step_days": 7}, "multiple"),
    ({"step_days": 0}, "multiple"),
    ({"end": date(2024, 1, 10)}, "single calendar year"),
])
def test_timeseries_stack_argument_errors(tmp_path, kwargs, match):
    args = {"start": date(2023, 4, 1), "end": date(2023, 9, 30), **kwargs}
    with pytest.raises(ValueError, match=match):
        tsfeatures.build_timeseries_stack(tmp_path, tmp_path / "ref.tif", tmp_path / "out.tif", **args)


def test_timeseries_stack_scene_errors(tmp_path):
    with pytest.raises(ValueError, match="No scenes"):
        tsfeatures.build_timeseries_stack([tmp_path / "absent"], tmp_path / "ref.tif", tmp_path / "o.tif",
                                          date(2023, 4, 1), date(2023, 9, 30))
    folder = tmp_path / "scenes" / "2023-06-01"
    folder.mkdir(parents=True)
    scenes.write(folder, sensor="S2", datetime="2023-06-01T10:00:00")
    (folder / "SCL.tif").write_bytes(b"")
    with pytest.raises(FileNotFoundError, match="lacks"):
        tsfeatures.build_timeseries_stack(tmp_path / "scenes", tmp_path / "ref.tif", tmp_path / "o.tif",
                                          date(2023, 4, 1), date(2023, 9, 30), s1_indices=())
    _touch(folder, {b for b in indices.get("NDVI").inputs})
    with pytest.raises(ValueError, match="support_tolerance_days"):
        tsfeatures.build_timeseries_stack(tmp_path / "scenes", tmp_path / "ref.tif", tmp_path / "o.tif",
                                          date(2023, 4, 1), date(2023, 9, 30), s2_indices=("NDVI",), s1_indices=(),
                                          temporal_support=True, support_tolerance_days=0)


def _stack(path: Path, names, shift=0.0):
    prof = {"driver": "GTiff", "height": 4, "width": 4, "count": len(names), "dtype": "float32",
            "crs": "EPSG:32634", "transform": from_origin(500_000 + shift, 5_000_000, 10, 10)}
    with rasterio.open(path, "w", **prof) as ds:
        ds.write(np.zeros((len(names), 4, 4), dtype="float32"))
    features.write_manifest(path, {"features": [{"name": n, "group": "ts", "sensor": "S2"} for n in names]})
    return path


def test_merge_stacks_rejects_grid_mismatch_and_duplicates(tmp_path):
    a = _stack(tmp_path / "a.tif", ["x", "y"])
    b = _stack(tmp_path / "b.tif", ["z"])
    out = tsfeatures.merge_stacks([a, b], tmp_path / "m.tif", block_size=2)
    with rasterio.open(out) as ds:
        assert list(ds.descriptions) == ["x", "y", "z"]
    shifted = _stack(tmp_path / "c.tif", ["w"], shift=5.0)
    with pytest.raises(ValueError, match="not on the grid"):
        tsfeatures.merge_stacks([a, shifted], tmp_path / "m2.tif")
    dup = _stack(tmp_path / "d.tif", ["x"])
    with pytest.raises(ValueError, match="Duplicate"):
        tsfeatures.merge_stacks([a, dup], tmp_path / "m3.tif")
    assert json.loads(features.manifest_path(a).read_text())["schema_version"] == 1


# ------------------------------------------------------------------------------------------- timeseries
def test_time_grid_and_support_errors():
    with pytest.raises(ValueError, match="step_days"):
        timeseries.make_grid(date(2023, 1, 1), date(2023, 2, 1), 0)
    with pytest.raises(ValueError, match="precede"):
        timeseries.make_grid(date(2023, 2, 1), date(2023, 1, 1))
    assert timeseries.make_grid(date(2023, 1, 1), date(2023, 1, 1)) == [date(2023, 1, 1)]
    for bad in (0, -5, float("nan")):
        with pytest.raises(ValueError, match="step_days"):
            timeseries.temporal_support(np.ones((2, 3)), step_days=bad)


def test_harmonics_need_enough_observations():
    days = np.arange(10, 360, 10, dtype=float)
    y = np.vstack([np.sin(2 * np.pi * days / 365), np.full(len(days), np.nan)])
    out = timeseries.harmonics(y, np.ones_like(y), days, order=2)
    assert np.isfinite(out["a0"][0]) and np.isnan(out["a0"][1]) and np.isnan(out["rmse"][1])
    assert out["amp1"][0] == pytest.approx(1.0, abs=0.05)


def test_drop_events_degenerate_inputs():
    days = np.array([100.0, 110.0, 120.0])
    with pytest.raises(ValueError, match="matching days"):
        timeseries.drop_events(np.zeros((1, 2)), days, 0.2, 20)
    with pytest.raises(ValueError, match="positive"):
        timeseries.drop_events(np.zeros((1, 3)), days, 0.0, 20)
    z = np.array([[0.8, 0.3, 0.3], [0.8, np.nan, 0.2]])
    events = timeseries.drop_events(z, days, 0.2, 20)
    assert len(events[0]) == 1 and events[0][0]["drop"] == pytest.approx(0.5) and events[1] == []
    assert timeseries.drop_events(z[:, :2], days[:2], 0.2, 20) == [[], []]       # fewer than three steps
    # a window shorter than the step reaches no later observation: no event
    assert timeseries.drop_events(z[:1], days, 0.2, 5) == [[]]
