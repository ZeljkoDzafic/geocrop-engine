import json
import xml.etree.ElementTree as ET

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from geocrop import s1, s2, scenes
from geocrop.indices import S2_BANDS

PROFILE = {"driver": "GTiff", "height": 6, "width": 6, "count": 1, "crs": "EPSG:32634",
           "transform": from_origin(500000, 5000000, 10, 10)}


def _scene(folder, value, scl, date):
    folder.mkdir(parents=True)
    for b in S2_BANDS:
        with rasterio.open(folder / f"{b}.tif", "w", dtype="float32", nodata=-9999, **PROFILE) as ds:
            ds.write(np.full((6, 6), value, "float32"), 1)
    with rasterio.open(folder / "SCL.tif", "w", dtype="uint8", **PROFILE) as ds:
        ds.write(scl.astype("uint8"), 1)
    scenes.write(folder, sensor="S2", datetime=date)


def test_s2_composite_masks_clouds_and_mosaics_same_day(tmp_path):
    clear = np.full((6, 6), 4)
    cloudy = clear.copy()
    cloudy[:3] = 9
    _scene(tmp_path / "a", 0.1, clear, "2023-06-01T10:00:00")
    _scene(tmp_path / "b", 0.3, cloudy, "2023-06-11T10:00:00")
    _scene(tmp_path / "b_tile2", 0.3, cloudy, "2023-06-11T10:00:00")   # same acquisition, other tile
    out = s2.composite([tmp_path / n for n in ("a", "b", "b_tile2")], tmp_path / "P1", statistic="median")
    with rasterio.open(out / "B04.tif") as ds:
        v = ds.read(1)
    with rasterio.open(out / "valid_count.tif") as ds:
        count = ds.read(1)
    assert np.allclose(v[:3], 0.1) and np.allclose(v[3:], 0.2)
    assert count.max() == 2, "same-day tiles must count once"
    assert json.loads((out / "composite.json").read_text())["acquisitions"]


def test_clear_mask_dilation():
    scl = np.full((7, 7), 4)
    scl[3, 3] = 9
    assert (~s2.clear_mask(scl, None, dilation_px=1)).sum() == 9


def test_safe_metadata_fails_closed(tmp_path):
    with pytest.raises(ValueError, match="MTD_MSIL2A"):
        s2.safe_metadata(tmp_path)
    (tmp_path / "MTD_MSIL2A.xml").write_text("<root><BOA_ADD_OFFSET band_id='1'>-1000</BOA_ADD_OFFSET></root>")
    with pytest.raises(ValueError, match="QUANTIFICATION"):
        s2.safe_metadata(tmp_path)


def test_snap_graph_order_and_gamma0(tmp_path):
    g = ET.parse(s1.write_snap_graph(tmp_path / "s.xml")).getroot()
    ids = [n.attrib["id"] for n in g.findall("node")]
    assert ids.index("ThermalNoiseRemoval") < ids.index("Remove-GRD-Border-Noise") < ids.index("Calibration")
    g0 = ET.parse(s1.write_snap_graph(tmp_path / "g.xml", backscatter="gamma0", speckle_filter="Refined Lee")).getroot()
    ids0 = [n.attrib["id"] for n in g0.findall("node")]
    assert "Terrain-Flattening" in ids0 and "Speckle-Filter" in ids0
    assert "EPSG:32634" in (tmp_path / "g.xml").read_text()


def test_s1_composite_linear_mean(tmp_path):
    for i, (val, orbit) in enumerate([(0.01, "ASCENDING"), (0.03, "ASCENDING"), (0.05, "DESCENDING")]):
        d = tmp_path / f"s{i}"
        d.mkdir()
        for pol in ("VV", "VH"):
            with rasterio.open(d / f"{pol}.tif", "w", dtype="float32", nodata=-9999, **PROFILE) as ds:
                ds.write(np.full((6, 6), val, "float32"), 1)
        scenes.write(d, sensor="S1", datetime=f"2023-06-0{i + 1}T17:00:00", orbit_direction=orbit)
    ref = tmp_path / "s0" / "VV.tif"
    out = s1.composite([tmp_path / f"s{i}" for i in range(3)], tmp_path / "P1", ref, separate_orbits=True)
    with rasterio.open(out / "ASC_VH.tif") as ds:
        assert np.allclose(ds.read(1), 0.02)
    with rasterio.open(out / "DESC_VV.tif") as ds:
        assert np.allclose(ds.read(1), 0.05)


def test_import_safe_scaling(tmp_path):
    """Mini SAFE with GeoTIFF payloads named like JP2 files (GDAL opens by content)."""
    name = "S2A_MSIL2A_20230705T093041_N0509_R136_T34TCQ_20230705T140000.SAFE"
    safe = tmp_path / name
    img = safe / "GRANULE" / "L2A_T34TCQ" / "IMG_DATA"
    for res in ("R10m", "R20m"):
        (img / res).mkdir(parents=True)
    offsets = "".join(f"<BOA_ADD_OFFSET band_id='{i}'>-1000</BOA_ADD_OFFSET>" for i in range(13))
    (safe / "MTD_MSIL2A.xml").write_text(f"<r><BOA_QUANTIFICATION_VALUE>10000</BOA_QUANTIFICATION_VALUE>{offsets}</r>")

    def write(path, size, px, value, dtype="uint16"):
        prof = {"driver": "GTiff", "height": size, "width": size, "count": 1, "dtype": dtype, "crs": "EPSG:32634",
                "transform": from_origin(500000, 5000000, px, px)}
        with rasterio.open(path, "w", **prof) as ds:
            ds.write(np.full((size, size), value, dtype), 1)

    for b in S2_BANDS:
        ten = b in ("B02", "B03", "B04", "B08")
        res, size, px = ("R10m", 4, 10) if ten else ("R20m", 2, 20)
        write(img / res / f"T34TCQ_20230705T093041_{b}_{res[1:]}.jp2", size, px, 3000)
    write(img / "R20m" / "T34TCQ_20230705T093041_SCL_20m.jp2", 2, 20, 4, "uint8")
    out = s2.import_safe(safe, tmp_path / "scene")
    with rasterio.open(out / "B11.tif") as ds:
        assert ds.shape == (4, 4) and np.allclose(ds.read(1), 0.2)   # (3000 - 1000) / 10000
    meta = scenes.read(out)
    assert meta["tile"] == "34TCQ" and str(meta["date"]) == "2023-07-05"


def test_shadow_projection_direction():
    from geocrop import clouds
    cloud = np.zeros((60, 60), bool)
    cloud[30, 30] = True
    # sun in the south-east (azimuth 135): shadow falls to the north-west (smaller row and col)
    sh = clouds.project_shadows(cloud, sun_zenith=45, sun_azimuth=135, pixel_m=20, heights=[400])
    r, c = np.argwhere(sh)[0]
    assert r < 30 and c < 30


def test_temporal_outliers():
    from geocrop import clouds
    blue = np.full((5, 2, 2), 0.04)
    nir = np.full((5, 2, 2), 0.35)
    swir = np.full((5, 2, 2), 0.2)
    blue[1] = 0.2          # residual cloud
    nir[3], swir[3] = 0.1, 0.05   # residual shadow
    drop = clouds.temporal_outliers(blue, nir, swir, np.ones((5, 2, 2), bool))
    assert drop[1].all() and drop[3].all() and not drop[[0, 2, 4]].any()


def test_external_mask_respected(tmp_path):
    clear = np.full((6, 6), 4)
    _scene(tmp_path / "a", 0.1, clear, "2023-06-01T10:00:00")
    _scene(tmp_path / "b", 0.3, clear, "2023-06-11T10:00:00")
    m = np.zeros((6, 6), "uint8")
    m[:2] = 1
    with rasterio.open(tmp_path / "b" / "CLOUDMASK.tif", "w", dtype="uint8", **PROFILE) as ds:
        ds.write(m, 1)
    out = s2.composite([tmp_path / "a", tmp_path / "b"], tmp_path / "P", statistic="mean")
    with rasterio.open(out / "B04.tif") as ds:
        v = ds.read(1)
    assert np.allclose(v[:2], 0.1) and np.allclose(v[2:], 0.2)
