"""Parcel geometry suitability on synthetic geometries and rasters of known shape (no real data)."""
import json

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Polygon, box

from geocrop import suitability

X0, Y0 = 500000, 5000000


def _parcels():
    shapes = {
        "square": box(X0, Y0, X0 + 100, Y0 + 100),                              # 1 ha, width 100
        "strip": box(X0 + 200, Y0, X0 + 230, Y0 + 300),                         # 0.9 ha, 30 m wide
        "sliver": box(X0 + 300, Y0, X0 + 303, Y0 + 400),                        # 0.12 ha, 3 m wide
        "small": box(X0 + 400, Y0, X0 + 440, Y0 + 40),                          # 0.16 ha
        "L": Polygon([(X0 + 500, Y0), (X0 + 700, Y0), (X0 + 700, Y0 + 40), (X0 + 540, Y0 + 40),
                      (X0 + 540, Y0 + 200), (X0 + 500, Y0 + 200)]),             # 40 m arms, area 1.28 ha
    }
    crops = {"square": "maize", "strip": "maize", "sliver": "soybean", "small": "soybean", "L": "maize"}
    return gpd.GeoDataFrame({"parcel_id": list(shapes), "declared_crop": [crops[k] for k in shapes]},
                            geometry=list(shapes.values()), crs="EPSG:32633")


def _table(**kw):
    return suitability.classify(suitability.assess(_parcels()), **kw).set_index("parcel_id")


def test_inscribed_radius_known_shapes():
    assert suitability.max_inscribed_radius(box(0, 0, 100, 100), tol=0.05) == pytest.approx(50, abs=0.1)
    assert suitability.max_inscribed_radius(box(0, 0, 30, 300), tol=0.05) == pytest.approx(15, abs=0.1)
    assert suitability.max_inscribed_radius(Polygon()) == 0.0


def test_shape_metrics():
    sq = suitability.shape_metrics(box(0, 0, 100, 100))
    assert sq["area_ha"] == pytest.approx(1.0)
    assert sq["width_m"] == pytest.approx(100, abs=0.5)
    assert sq["compactness"] == pytest.approx(np.pi / 4, rel=1e-6)
    assert sq["elongation"] == pytest.approx(1.0)
    strip = suitability.shape_metrics(box(0, 0, 30, 300))
    assert strip["width_m"] == pytest.approx(30, abs=0.5)
    assert strip["mean_width_m"] == pytest.approx(2 * 9000 / 660)
    assert strip["elongation"] == pytest.approx(10.0)
    assert strip["compactness"] < sq["compactness"]


def test_known_geometries_rule():
    t = _table(min_area_ha=0.2, min_width_m=50)
    assert t.loc["square", "suitability"] == "suitable"
    assert t.loc["strip", "suitability"] == "technically_unsuitable"
    assert t.loc["strip", "suitability_reasons"] == "width_below_min"          # area-OK but too narrow
    assert t.loc["strip", "area_ha"] == pytest.approx(0.9)
    assert "area_below_min" in t.loc["sliver", "suitability_reasons"]
    assert "width_below_min" in t.loc["sliver", "suitability_reasons"]
    assert "too_few_inner_pixels" in t.loc["sliver", "suitability_reasons"]    # 10 m inner buffer removes everything
    assert t.loc["sliver", "n_inner_pixels"] == 0
    assert t.loc["small", "suitability_reasons"].startswith("area_below_min")
    assert 40 <= t.loc["L", "width_m"] < 50                                    # about the arm width (corner adds a little), not the bounding box


def test_virtual_grid_pixel_counts():
    t = _table()
    assert t.loc["square", "n_pixels"] == 100                                  # 10 x 10 centres
    assert t.loc["square", "n_inner_pixels"] == 64                             # 80 x 80 m core
    assert t.loc["square", "share_reliable_inner"] == 1.0


def test_width_method_mean_is_stricter_for_compact_shapes():
    assert _table(min_width_m=60, width_method="inscribed").loc["square", "suitability"] == "suitable"
    assert _table(min_width_m=60, width_method="mean").loc["square", "suitability"] == "technically_unsuitable"   # 2A/P = 50
    with pytest.raises(ValueError, match="width_method"):
        _table(width_method="other")


def _raster(tmp_path, nodata_block=True):
    """80 x 80 pixels of 10 m; the right half of the square parcel (rows 70-79, cols 5-9) is NaN."""
    z = np.ones((2, 80, 80), dtype="float32")
    if nodata_block:
        z[:, 70:80, 5:10] = np.nan
    path = tmp_path / "grid.tif"
    with rasterio.open(path, "w", driver="GTiff", height=80, width=80, count=2, dtype="float32", crs="EPSG:32633",
                       transform=from_origin(X0, Y0 + 800, 10, 10)) as dst:
        dst.write(z)
    return path


def test_raster_reliable_pixels_and_mask(tmp_path):
    r = _raster(tmp_path)
    t = suitability.assess(_parcels(), raster=r).set_index("parcel_id")
    sq = t.loc["square"]
    assert sq.n_pixels == 100 and sq.n_inner_pixels == 64
    assert sq.n_reliable_inner == 32                                           # inner cols 1-8 x 8 rows, cols 5-8 are NaN
    assert sq.share_reliable_inner == pytest.approx(0.5)
    c = suitability.classify(t.reset_index(), min_inner_pixels=4, min_reliable_inner_pixels=40).set_index("parcel_id")
    assert c.loc["square", "suitability_reasons"] == "too_few_reliable_inner_pixels"
    assert suitability.classify(t.reset_index(), min_reliable_inner_pixels=10).set_index("parcel_id").loc["square", "suitability"] == "suitable"
    # optional mask band on the same grid: 0 = unreliable
    m = np.ones((80, 80), dtype="uint8")
    m[70:80, 0:3] = 0
    mp = tmp_path / "mask.tif"
    with rasterio.open(mp, "w", driver="GTiff", height=80, width=80, count=1, dtype="uint8", crs="EPSG:32633",
                       transform=from_origin(X0, Y0 + 800, 10, 10)) as dst:
        dst.write(m, 1)
    t2 = suitability.assess(_parcels(), raster=_raster(tmp_path, nodata_block=False), mask=mp).set_index("parcel_id")
    assert t2.loc["square", "n_reliable_inner"] == 64 - 2 * 8                  # inner cols 1-2 masked
    with pytest.raises(ValueError, match="same grid"):
        bad = tmp_path / "bad.tif"
        with rasterio.open(bad, "w", driver="GTiff", height=10, width=10, count=1, dtype="uint8", crs="EPSG:32633",
                           transform=from_origin(X0, Y0 + 800, 10, 10)) as dst:
            dst.write(np.ones((10, 10), dtype="uint8"), 1)
        suitability.assess(_parcels(), raster=_raster(tmp_path), mask=bad)


def test_geographic_crs_refused():
    with pytest.raises(ValueError, match="projected"):
        suitability.assess(_parcels().to_crs(4326))


def test_sensitivity_grid_monotone():
    table = suitability.assess(_parcels())
    grid = suitability.sensitivity_grid(table, areas=(0.1, 0.2, 0.5), widths=(20, 40, 50, 60), inner_pixels=(0, 1, 4, 9, 16))
    allg = grid[grid.level == "all"]
    assert len(allg) == 3 * 4 * 5
    key = ["min_area_ha", "min_width_m", "min_inner_pixels"]
    rows = {tuple(r[k] for k in key): r for r in allg.to_dict("records")}
    for a, b in ((x, y) for x in rows for y in rows if all(p <= q for p, q in zip(x, y))):
        assert rows[a]["n_excluded"] <= rows[b]["n_excluded"]
        assert rows[a]["excluded_area_ha"] <= rows[b]["excluded_area_ha"] + 1e-9
    loosest = rows[(0.1, 20, 0)]
    assert loosest["n_excluded"] == 1                                          # only the 3 m sliver is narrower than 20 m
    # by crop and by size class partition the parcels
    for level in ("crop", "size_class"):
        sub = grid[(grid.level == level) & (grid.min_width_m == 50) & (grid.min_inner_pixels == 4) & (grid.min_area_ha == 0.2)]
        assert sub.n_parcels.sum() == 5 and sub.n_excluded.sum() == int(allg[(allg.min_width_m == 50) & (allg.min_inner_pixels == 4) & (allg.min_area_ha == 0.2)].n_excluded.iloc[0])


def test_write_sensitivity(tmp_path):
    grid = suitability.sensitivity_grid(suitability.assess(_parcels()))
    paths = suitability.write_sensitivity(grid, tmp_path)
    md = open(paths["md"], encoding="utf8").read()
    assert "not an unsupported declaration" in md and "not validated" in md
    assert len(pd.read_csv(paths["csv"])) == len(grid)


def _presence():
    return pd.DataFrame({
        "parcel_id": ["square", "strip", "sliver", "small", "L", "orphan"],
        "declared_crop": ["maize", "maize", "soybean", "soybean", "maize", "maize"],
        "parcel_area_ha": [1.0, 0.9, 0.12, 0.16, 1.28, 0.5],
        "p1_match_whole": [True, True, False, False, True, True],
        "p1_match_agricultural": [True, True, True, False, True, False],
        "soft_vote_agricultural": [0.9, 0.8, 0.2, 0.1, 0.7, 0.4],
        "preliminary_status": ["supported", "supported", "not_supported", "disputed_review", "supported", "disputed_review"],
    })


def test_compare_subsets():
    su = suitability.classify(suitability.assess(_parcels()), min_width_m=50)
    res = suitability.compare_subsets(_presence(), su)
    cmpr = res["comparison"].set_index(["subset", "method"])
    assert cmpr.loc[("all_declarations", "p1_whole"), "n_supported"] == 4
    assert cmpr.loc[("all_declarations", "p2_soft_vote"), "n_supported"] == 3
    assert cmpr.loc[("all_declarations", "p3_status"), "share_supported"] == pytest.approx(3 / 6)
    # suitable: only the square (the strip is too narrow, the L arms are 40 m, small/sliver too small, orphan not assessed)
    assert cmpr.loc[("suitable_only", "p3_status"), "n_evaluated"] == 1
    assert cmpr.loc[("suitable_only", "p3_status"), "share_supported"] == 1.0
    ex = res["excluded"].iloc[0]
    assert ex.n_excluded == 5 and ex.excluded_area_ha == pytest.approx(0.9 + 0.12 + 0.16 + 1.28 + 0.5)
    by = res["excluded_by_crop"].set_index("declared_crop")
    assert by.loc["maize", "n_excluded"] == 3 and by.loc["soybean", "n_excluded"] == 2


def test_compare_subsets_validates_input():
    with pytest.raises(ValueError, match="needs columns"):
        suitability.compare_subsets(_presence().drop(columns="parcel_area_ha"), pd.DataFrame({"parcel_id": [1], "suitability": ["suitable"]}))


def test_analyse_end_to_end(tmp_path):
    parcels = tmp_path / "parcels.gpkg"
    _parcels().to_file(parcels, driver="GPKG")
    cp = tmp_path / "crop_presence.csv"
    _presence().to_csv(cp, index=False)
    out = tmp_path / "out"
    res = suitability.analyse(parcels, out, crop_presence=cp)
    assert res["thresholds_validated"] is False
    assert res["suitability_counts"] == {"suitable": 1, "technically_unsuitable": 4}
    for name in ("parcel_suitability.csv", "suitability_sensitivity.csv", "suitability_sensitivity.md",
                 "subset_comparison.md", "subset_comparison.csv", "parcel_suitability_manifest.json"):
        assert (out / name).exists()
    manifest = json.loads((out / "parcel_suitability_manifest.json").read_text())
    assert manifest["config"]["pixel_grid"] == "virtual" and "unsupported declaration" in manifest["note"]


def test_cli(tmp_path, capsys):
    from geocrop.cli import main
    parcels = tmp_path / "parcels.gpkg"
    _parcels().to_file(parcels, driver="GPKG")
    main(["parcel-suitability", "--parcels", str(parcels), "--out", str(tmp_path / "o"), "--min-width-m", "25"])
    assert json.loads(capsys.readouterr().out)["suitability_counts"]["suitable"] == 3          # square, strip, L
