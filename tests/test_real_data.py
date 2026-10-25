"""Integration tests on REAL Sentinel-2 L2A data (public COGs, Element 84 Earth Search, no credentials).

Run with:  pytest -m network      (skipped by default; needs internet access)
Area: ~2 x 2 km of arable land in Semberija (northern Bosnia and Herzegovina), July 2023.
"""
import json
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

pytestmark = pytest.mark.network

# 2.0 x 2.0 km grid in UTM 34N (EPSG:32634) east of Bijeljina; 10 m pixels
ORIGIN = (390000.0, 4960000.0)


@pytest.fixture(scope="module")
def grid(tmp_path_factory):
    path = tmp_path_factory.mktemp("grid") / "grid.tif"
    prof = {"driver": "GTiff", "height": 200, "width": 200, "count": 1, "dtype": "uint8", "crs": "EPSG:32634",
            "transform": from_origin(*ORIGIN, 10, 10), "nodata": 0}
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(np.ones((200, 200), "uint8"), 1)
    return path


@pytest.fixture(scope="module")
def season(grid, tmp_path_factory):
    pytest.importorskip("pystac_client")
    from geocrop.stac_import import import_season
    root = tmp_path_factory.mktemp("scenes")
    try:
        res = import_season(grid, root, "2023-07-01", "2023-08-15", max_cloud=30, min_clear_fraction=0.5, max_items=6)
    except Exception as exc:  # network / service outage must not masquerade as an engine failure
        pytest.skip(f"STAC service unavailable: {exc}")
    imported = [r for r in res if r["status"] in ("imported", "cached")]
    if len(imported) < 2:
        pytest.skip(f"not enough clear acquisitions: {res}")
    return root, imported


def test_real_reflectance_is_physically_plausible(season):
    root, imported = season
    scene = Path(imported[0]["path"])
    meta = json.loads((scene / "scene.json").read_text())
    assert meta["sensor"] == "S2" and meta["datetime"].startswith("2023-0")
    with rasterio.open(scene / "B04.tif") as r, rasterio.open(scene / "B08.tif") as n:
        red, nir = r.read(1), n.read(1)
    ok = (red != -9999) & (nir != -9999)
    assert ok.mean() > 0.5
    assert (red[ok] < 0).mean() < 0.02, "radiometric offset applied twice?"
    assert 0.0 < np.median(red[ok]) < 0.2 and 0.1 < np.median(nir[ok]) < 0.6


def test_real_composite_and_ndvi(season, grid, tmp_path):
    from geocrop import s2
    from geocrop.features import build_stack
    root, imported = season
    comp = tmp_path / "comp"
    s2.composite([p["path"] for p in imported], comp / "P3", reference=grid, mask_dilation_px=1,
                 temporal_screening=True)
    s2.composite([p["path"] for p in imported[:1]], comp / "P2", reference=grid)
    stack = build_stack(comp, tmp_path / "stack.tif", s2_indices=("NDVI", "NDMI"), s1=())
    with rasterio.open(stack) as ds:
        names = list(ds.descriptions)
        ndvi = ds.read(names.index("P3_NDVI") + 1)
    ok = ndvi != -9999
    # July arable land: mostly vegetated, NDVI in the physical range
    assert ok.mean() > 0.5 and np.all((ndvi[ok] >= -1) & (ndvi[ok] <= 1))
    assert 0.3 < np.median(ndvi[ok]) < 0.95
