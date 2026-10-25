"""Regression (golden values) and bounded-memory tests."""
import tracemalloc

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from geocrop.features import build_stack
from geocrop.indices import S2_BANDS


def _composites(root, size, periods=("P1", "P2"), seed=0):
    rng = np.random.default_rng(seed)
    prof = {"driver": "GTiff", "height": size, "width": size, "count": 1, "dtype": "float32", "crs": "EPSG:32634",
            "transform": from_origin(500000, 5000000, 10, 10), "nodata": -9999.0, "tiled": True,
            "blockxsize": 256, "blockysize": 256, "compress": "deflate"}
    for k, p in enumerate(periods):
        (root / p).mkdir(parents=True)
        for i, b in enumerate(S2_BANDS):
            with rasterio.open(root / p / f"{b}.tif", "w", **prof) as dst:
                for r in range(0, size, 256):
                    arr = (0.05 + 0.01 * i + 0.05 * k + rng.random((min(256, size - r), size)) * 0.02).astype("float32")
                    if b == "B08":
                        arr += 0.3
                    dst.write(arr, 1, window=rasterio.windows.Window(0, r, size, arr.shape[0]))
    return root


def test_feature_formulas_golden_values(tmp_path):
    """Constant reflectances -> exactly known index values (guards silent formula changes)."""
    for p, (red, nir, b5, swir) in {"P1": (0.05, 0.35, 0.1, 0.2), "P2": (0.04, 0.45, 0.12, 0.18)}.items():
        (tmp_path / p).mkdir()
        vals = {b: 0.06 for b in S2_BANDS} | {"B04": red, "B08": nir, "B8A": nir, "B05": b5, "B11": swir, "B03": 0.07}
        for b, v in vals.items():
            with rasterio.open(tmp_path / p / f"{b}.tif", "w", driver="GTiff", height=4, width=4, count=1,
                               dtype="float32", crs="EPSG:32634", transform=from_origin(0, 40, 10, 10),
                               nodata=-9999.0) as dst:
                dst.write(np.full((4, 4), v, "float32"), 1)
    stack = build_stack(tmp_path, tmp_path / "s.tif", s1=())
    with rasterio.open(stack) as ds:
        v = {n: float(ds.read(i + 1)[0, 0]) for i, n in enumerate(ds.descriptions)}
    golden = {
        "P1_NDVI": 0.30 / 0.40, "P2_NDVI": 0.41 / 0.49, "P1_GCVI": 0.35 / 0.07 - 1, "P1_RECI": 0.35 / 0.1 - 1,
        "P1_NDMI": 0.15 / 0.55, "P1_NDRE": 0.25 / 0.45, "delta_NDVI_P1_P2": 0.41 / 0.49 - 0.30 / 0.40,
        "season_NDVI_max": 0.41 / 0.49, "season_NDVI_argmax": 2.0,
    }
    for name, expected in golden.items():
        assert v[name] == pytest.approx(expected, abs=1e-5), name


@pytest.mark.slow
def test_feature_stack_memory_is_bounded_by_block_size(tmp_path):
    size = 1024
    comp = _composites(tmp_path / "comp", size)
    full_stack_mb = size * size * 4 * 70 / 1e6          # ~70 float32 features over the whole raster
    tracemalloc.start()
    stack = build_stack(comp, tmp_path / "stack.tif", block_size=256, s1=())
    peak_mb = tracemalloc.get_traced_memory()[1] / 1e6
    tracemalloc.stop()
    with rasterio.open(stack) as ds:
        assert ds.shape == (size, size)
    assert peak_mb < 0.5 * full_stack_mb, f"peak {peak_mb:.0f} MB vs whole stack {full_stack_mb:.0f} MB"
