import json
from datetime import date, timedelta

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from geocrop import config, pipeline, scenes, timeseries
from geocrop.features import load_manifest
from geocrop.tsfeatures import build_timeseries_stack, quality_path


def scene_inputs(tmp_path):
    root = tmp_path / "scenes"
    start = date(2023, 4, 1)
    reference = None
    for i, offset in enumerate([0, 5, 10, 20, 25, 35]):
        folder = root / str(i)
        folder.mkdir(parents=True)
        for band, value in (("B04", .1), ("B08", .3 + .1 * np.sin(i)) , ("SCL", 4)):
            data = np.full((1, 3), value, dtype="float32")
            if band == "SCL":
                data[0, 2] = 9
                if i not in (0, 1, 3, 4):
                    data[0, 1] = 9
            with rasterio.open(folder / f"{band}.tif", "w", driver="GTiff", width=3, height=1,
                               count=1, dtype="float32", crs="EPSG:32634", nodata=-9999,
                               transform=from_origin(500000, 5000000, 10, 10)) as ds:
                ds.write(data, 1)
        scenes.write(folder, sensor="S2", datetime=f"{start + timedelta(days=offset)}T10:00:00")
        reference = folder / "B04.tif"
    return root, reference, start, start + timedelta(days=35)


def test_support_counts_bins_not_duplicate_acquisitions():
    weights = np.array([[2, 0, 1, 0, 0], [0, 1, 1, 1, 0], [0, 0, 0, 0, 0]])
    support = timeseries.temporal_support(weights, 5)
    np.testing.assert_array_equal(support["observation_count"], [3, 3, 0])
    np.testing.assert_allclose(support["observed_bin_fraction"], [.4, .6, 0])
    np.testing.assert_array_equal(support["longest_gap_days"], [10, 5, 25])


@pytest.mark.parametrize("weights", [np.array([1, 0]), np.empty((2, 0)), [[-1]], [[np.nan]]])
def test_invalid_support_weights(weights):
    with pytest.raises(ValueError, match="weights"):
        timeseries.temporal_support(weights)


def test_quality_outputs_and_opt_in_feature_mask(tmp_path):
    root, ref, start, end = scene_inputs(tmp_path)
    default = build_timeseries_stack(root, ref, tmp_path / "default.tif", start, end,
                                     s2_indices=["NDVI"], s1_indices=[], output_step_days=10)
    strict = build_timeseries_stack(root, ref, tmp_path / "strict.tif", start, end,
                                    s2_indices=["NDVI"], s1_indices=[], output_step_days=10,
                                    min_observed_fraction=.7, max_gap_days=5)
    with rasterio.open(quality_path(strict)) as qa:
        np.testing.assert_array_equal(qa.read(1)[0], [6, 4, 0])
        np.testing.assert_allclose(qa.read(2)[0], [.75, .5, 0])
        np.testing.assert_array_equal(qa.read(3)[0], [5, 10, 40])
        np.testing.assert_array_equal(qa.read(4)[0], [1, 0, 0])
    with rasterio.open(default) as before, rasterio.open(strict) as after:
        assert before.read(1)[0, 1] != before.nodata
        assert np.all(after.read()[:, 0, 1:] == after.nodata)
        np.testing.assert_allclose(before.read()[:, 0, 0], after.read()[:, 0, 0])
    assert not any(f.name.startswith("qa_") for f in load_manifest(strict))
    summary = json.loads(quality_path(strict).with_suffix(".json").read_text())
    assert summary["series"]["NDVI"]["passes_policy"] == 1
    assert summary["series"]["NDVI"]["without_observations"] == 1


def test_pipeline_repairs_missing_quality_companion(tmp_path):
    root, ref, start, end = scene_inputs(tmp_path)
    cfg = config.load(overrides={"grid": {"reference": str(ref)}, "timeseries": {
        "enabled": True, "start": str(start), "end": str(end), "s2_indices": ["NDVI"],
        "s1_indices": [], "output_step_days": 10}, "paths": {"s2_scenes": str(root),
        "s1_scenes": str(tmp_path / "no-radar"), "composites": str(tmp_path / "no-composites"),
        "features": str(tmp_path / "stack.tif"), "output": str(tmp_path / "run")}})
    runner = pipeline.Runner(cfg)
    runner.features()
    messages = []
    runner.log = messages.append
    runner.features()
    assert messages[-1].startswith("[skip]")
    qa = quality_path(tmp_path / "stack_ts.tif")
    qa.unlink()
    runner.features()
    assert messages[-1].startswith("[run") and qa.exists()


@pytest.mark.parametrize("overrides", [{"min_observed_fraction": 2}, {"max_gap_days": -1},
                                       {"step_days": 0}, {"output_step_days": 7}])
def test_config_rejects_invalid_temporal_policy(overrides):
    with pytest.raises(config.ConfigError):
        config.load(overrides={"timeseries": overrides})


def test_rejects_cross_year_instead_of_nonmonotonic_doy(tmp_path):
    with pytest.raises(ValueError, match="single calendar year"):
        build_timeseries_stack(tmp_path, tmp_path / "ref.tif", tmp_path / "out.tif",
                               date(2023, 12, 1), date(2024, 3, 1))
