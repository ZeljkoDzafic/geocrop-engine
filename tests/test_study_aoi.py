"""tools/make_study_aoi.py and tools/study_data_check.py: AOI logic on synthetic rasters and polygons (no network)."""
import importlib.util
import json
from datetime import date
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
from rasterio.transform import from_origin
from shapely.geometry import box

ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


aoi_tool = _load("make_study_aoi")
check_tool = _load("study_data_check")


def test_suggest_threshold_finds_break_between_plain_and_slopes():
    rng = np.random.default_rng(0)
    plain = rng.normal(95, 4, 60_000)                 # floodplain population (mode ~95 m)
    slopes = rng.uniform(130, 400, 40_000)            # flat continuum of hillslope elevations
    t = aoi_tool.suggest_threshold(np.concatenate([plain, slopes]))
    assert 105 <= t <= 135
    assert aoi_tool.suggest_threshold(np.concatenate([plain, slopes, [np.nan]])) == t


def test_suggest_threshold_rejects_empty():
    with pytest.raises(ValueError):
        aoi_tool.suggest_threshold(np.array([np.nan]))


def test_clean_mask_opening_islands_and_holes():
    m = np.zeros((60, 60), dtype=bool)
    m[10:50, 10:50] = True                 # main plain, 1600 px
    m[25:27, 25:27] = False                # small hole (4 px) -> filled
    m[30:40, 30:40] = False                # large hole (100 px) -> kept
    m[2:4, 55:58] = True                   # tiny island -> removed
    m[20, 50:58] = True                    # 1-px-wide finger -> opened away
    out = aoi_tool.clean_mask(m, pixel_area_m2=1e4, opening_px=1, min_island_km2=0.5, max_hole_km2=0.5)
    assert out[25, 25] and out[26, 26]
    assert not out[35, 35]
    assert not out[2:4, 55:58].any()
    assert not out[20, 52:58].any()
    assert out[15, 15]


def test_geographic_pixel_area_shrinks_with_latitude():
    tr = from_origin(17.0, 46.0, 1 / 3600, 1 / 3600)
    col = aoi_tool.geographic_pixel_area(tr, 3600)
    assert col.shape == (3600, 1)
    assert col[0, 0] < col[-1, 0]                     # north rows are smaller
    assert 600 < col[1800, 0] < 700                   # ~30.9 m x 21.6 m at 45.5 N


def test_dem_tiles_and_utm_zone():
    assert aoi_tool.dem_tiles((16.9, 44.7, 17.8, 45.3)) == [
        "N44_00_E016_00", "N44_00_E017_00", "N45_00_E016_00", "N45_00_E017_00"]
    assert aoi_tool.utm_zone(16.99) == aoi_tool.utm_zone(17.80) == 33


def test_select_municipalities_reports_missing():
    g = gpd.GeoDataFrame({"shapeName": ["Gradiška", "Laktaši"]}, geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1)],
                         crs="EPSG:4326")
    with pytest.raises(ValueError, match="Srbac"):
        aoi_tool.select_municipalities(g)
    assert len(aoi_tool.select_municipalities(g, names=("Laktaši",))) == 1


def test_build_aoi_on_synthetic_valley():
    """A low plain in the west half of two 'municipalities', hills in the east; clipped to the country."""
    res = 1 / 1200                                    # ~90 m pixels keep the test fast
    tr = from_origin(17.0, 45.2, res, res)
    h, w = 240, 480                                   # 0.2 deg x 0.4 deg
    x = np.arange(w)[None, :].repeat(h, 0)
    dem = np.where(x < 240, 95.0, 95.0 + (x - 240) * 2.0).astype("float32")
    dem[100:104, 100:104] = 160.0                     # forest patch in a DSM -> hole to fill
    munis = gpd.GeoDataFrame({"shapeName": ["Gradiška", "Srbac"]},
                             geometry=[box(17.0, 45.0, 17.2, 45.2), box(17.2, 45.0, 17.4, 45.2)], crs="EPSG:4326")
    country = box(17.0, 45.0, 17.4, 45.15)            # northern strip is "abroad"
    out = aoi_tool.build_aoi(dem, tr, munis, country, threshold="auto", opening_px=1, min_island_km2=1.0,
                             max_hole_km2=1.0, simplify_m=10.0)
    assert out["threshold_rule"] == "auto"
    geom = out["geometry"]
    assert geom.bounds[3] <= 45.15 + 1e-6            # never leaves the country
    assert geom.bounds[2] < 17.25                     # hills excluded
    assert geom.contains(box(17.08, 45.06, 17.09, 45.07).centroid)   # filled forest hole
    expected = aoi_tool.equal_area_km2(box(17.0, 45.0, 17.2, 45.15))
    assert out["area_km2"] == pytest.approx(expected, rel=0.05)
    assert set(out["area_by_municipality_km2"]) == {"Gradiška", "Srbac"}
    fixed = aoi_tool.build_aoi(dem, tr, munis, country, threshold=100.0, opening_px=1, min_island_km2=1.0,
                               max_hole_km2=1.0)
    assert fixed["threshold_m"] == 100.0 and fixed["threshold_rule"] == "fixed"


def test_equal_area_km2_of_one_degree_cell():
    a = aoi_tool.equal_area_km2(box(17.0, 45.0, 18.0, 46.0))
    assert a == pytest.approx(111.2 * 111.2 * np.cos(np.radians(45.5)), rel=0.01)


def test_committed_aoi_matches_its_properties():
    path = ROOT / "configs/studies/lijevce_polje/aoi.geojson"
    fc = json.loads(path.read_text(encoding="utf8"))
    props = fc["features"][0]["properties"]
    g = gpd.read_file(path)
    assert g.crs.to_epsg() == 4326
    assert aoi_tool.equal_area_km2(g.geometry.iloc[0]) == pytest.approx(props["area_km2"], abs=0.1)
    assert props["utm_zones"] == [33]
    assert props["sources"]["boundaries"]["sha256"] == aoi_tool.GB_SHA256
    assert "WorldDEM-30" in props["sources"]["dem"]["notice"]


def test_data_check_summary_counts_clear_dates():
    inside = np.zeros((4, 4), dtype=bool)
    inside[:2, :2] = True
    scl_a = np.full((4, 4), 4, dtype="uint8")        # vegetation: clear
    scl_b = np.full((4, 4), 9, dtype="uint8")        # high cloud
    scl_b[0, 0] = 5
    clear = {date(2023, 6, 5): check_tool.clear_from_scl(scl_a)[1],
             date(2023, 6, 20): check_tool.clear_from_scl(scl_b)[1],
             date(2023, 8, 1): check_tool.clear_from_scl(scl_a)[1]}
    windows = {"P2": (date(2023, 6, 1), date(2023, 7, 15)), "P3": (date(2023, 7, 16), date(2023, 8, 31))}
    s = check_tool.summarise(clear, inside, windows)
    assert s["P2"]["dates"] == 2 and s["P2"]["dates_mostly_clear"] == 1
    assert s["P2"]["clear_dates_per_pixel"] == {"min": 1, "median": 1.0, "max": 2, "share_ge1": 1.0,
                                                "share_ge2": 0.25, "share_ge3": 0.0}
    assert s["P3"]["dates"] == 1
    assert check_tool.window_of(date(2023, 7, 16), windows) == "P3"
    assert check_tool.window_of(date(2023, 1, 1), windows) is None
    observed, _ = check_tool.clear_from_scl(np.zeros((2, 2), dtype="uint8"))
    assert not observed.any()
