from datetime import date

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

from geocrop import config, features, scenes, separability


def test_jm_bounds_and_symmetry():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 40)
    assert separability.jeffries_matusita(a, a + 0.0) == pytest.approx(0.0, abs=1e-9)
    far = separability.jeffries_matusita(a, rng.normal(50, 1, 40))
    assert 1.99 < far <= 2.0
    b = rng.normal(1, 1, 40)
    assert separability.jeffries_matusita(a, b) == pytest.approx(separability.jeffries_matusita(b, a))


def test_jm_needs_enough_polygons_and_variance():
    assert np.isnan(separability.jeffries_matusita([1, 2], [3, 4, 5]))
    assert np.isnan(separability.jeffries_matusita([1, 1, 1], [3, 4, 5]))


def _stack(path):
    rng = np.random.default_rng(1)
    data = rng.normal(0, 0.05, (2, 40, 40)).astype("float32")
    data[0, :, 20:] += 1.0          # P1_A separates left (class a) from right (class b)
    with rasterio.open(path, "w", driver="GTiff", height=40, width=40, count=2, dtype="float32", crs="EPSG:32633",
                       transform=from_origin(500000, 4900400, 10, 10), nodata=-9999) as ds:
        ds.write(data)
        ds.set_band_description(1, "P1_A")
        ds.set_band_description(2, "P1_B")
    features.write_manifest(path, {"stack": path.name, "nodata": -9999.0, "features": [
        {"name": "P1_A", "group": "s2_band", "sensor": "S2", "period": "P1", "periods": ["P1"], "source": "A"},
        {"name": "P1_B", "group": "s2_band", "sensor": "S2", "period": "P1", "periods": ["P1"], "source": "B"}]})


def test_run_ranks_the_separating_feature(tmp_path):
    stack = tmp_path / "s.tif"
    _stack(stack)
    polys = []
    for i in range(4):
        x0 = 500000 + i * 40
        polys.append({"polygon_id": f"a{i}", "class_name": "a", "geometry": box(x0, 4900300, x0 + 30, 4900390)})
        polys.append({"polygon_id": f"b{i}", "class_name": "b",
                      "geometry": box(x0 + 200, 4900300, x0 + 230, 4900390)})
    path = tmp_path / "p.gpkg"
    gpd.GeoDataFrame(polys, crs="EPSG:32633").to_file(path)
    res = separability.run(stack, path, "class_name", ["a", "b"], tmp_path / "out")
    best = (tmp_path / "out" / "jm_best_per_pair.csv").read_text()
    assert "P1_A" in best and res["polygons"] == {"a": 4, "b": 4}
    assert res["mean_best_pairwise_jm"]["P1"] > 1.9


def test_overlapping_windows_need_opt_in_and_scenes_enter_every_window():
    raw = {"P2": {"start": "2023-06-05", "end": "2023-07-05"}, "P3": {"start": "2023-06-25", "end": "2023-07-25"}}
    with pytest.raises(config.ConfigError, match="allow_window_overlap"):
        config.parse_windows(raw)
    windows = config.parse_windows(raw, allow_overlap=True)
    got = scenes.assign([{"date": date(2023, 6, 28)}, {"date": date(2023, 6, 10)}], windows)
    assert [len(got["P2"]), len(got["P3"])] == [2, 1]


def test_polygon_majority_and_target_macro_f1():
    from geocrop import metrics
    y = ["a"] * 4 + ["b"] * 4 + ["c"] * 2
    p = ["a", "a", "a", "b", "b", "b", "a", "b", "c", "c"]
    poly = ["p1"] * 4 + ["p2"] * 4 + ["p3"] * 2
    yt, yp = metrics.polygon_majority(y, p, poly)
    assert list(yt) == ["a", "b", "c"] and list(yp) == ["a", "b", "c"]
    s = metrics.summary(["a", "a", "b"], ["a", "b", "b"])
    assert metrics.target_macro_f1(s, ["a", "b"]) == pytest.approx((s["per_class"]["a"]["f1"] + s["per_class"]["b"]["f1"]) / 2)
    assert metrics.target_macro_f1(s, ["zzz"]) is None


def test_parcel_means_use_each_features_own_valid_pixels(tmp_path):
    path = tmp_path / "s.tif"
    data = np.array([[[1, 9]], [[2, -9999]]], dtype="float32")   # P1 valid 2 px, P2 valid 1 px
    with rasterio.open(path, "w", driver="GTiff", height=1, width=2, count=2, dtype="float32", crs="EPSG:32633",
                       transform=from_origin(500000, 4900010, 10, 10), nodata=-9999) as ds:
        ds.write(data)
        ds.set_band_description(1, "P1_A")
        ds.set_band_description(2, "P2_A")
    poly = gpd.GeoDataFrame([{"polygon_id": "x", "class_name": "a", "geometry": box(500000, 4900000, 500020, 4900010)}],
                            crs="EPSG:32633")
    m = separability.parcel_means(path, poly, "class_name", ["a"], features=["P1_A", "P2_A"])
    assert m.loc[0, "P1_A"] == pytest.approx(5.0) and m.loc[0, "P2_A"] == pytest.approx(2.0)


def test_empty_group_selection_is_rejected(tmp_path):
    stack = tmp_path / "s.tif"
    _stack(stack)
    with pytest.raises(ValueError, match="No period features match"):
        separability.run(stack, tmp_path / "none.gpkg", "class_name", ["a", "b"], tmp_path / "o", groups=["seasonal"])
    with pytest.raises(ValueError, match="No features selected"):
        separability.parcel_means(stack, tmp_path / "none.gpkg", "class_name", ["a"], features=[])
