"""Cloud and cloud-shadow screening beyond the Sentinel-2 Scene Classification (SCL).

SCL and the L2A cloud probability miss thin clouds, cloud edges and many shadows. Residual
contamination biases composites and time series (e.g. NDVI drops that mimic harvest). Four
complementary, optional layers are provided; every one of them only *adds* invalid pixels:

1. Geometric shadow projection (as in Fmask; Zhu & Woodcock, 2012): cloud pixels are shifted along the
   solar azimuth for a range of cloud heights, displacement = h · tan(sun zenith); projected pixels
   that are also dark in NIR are shadows. Computed per scene at import (`SHADOW.tif`), because shadows
   can fall kilometres from their clouds.
2. External masks: any `CLOUDMASK.tif` in a scene folder (0 = clear, >0 = cloud/thin cloud/shadow),
   e.g. from OmniCloudMask (Wright et al., 2025), s2cloudless or Fmask.
   `omnicloudmask_scene` runs OmniCloudMask when that optional package is installed.
3. Mask dilation (in `s2.clear_mask`) against cloud halos.
4. Multi-temporal outlier test (in the spirit of MAJA/Tmask; Hagolle et al., 2010; Zhu & Woodcock, 2014):
   an observation whose blue reflectance exceeds the pixel's clear-sky median by `blue_delta` is a
   residual cloud; one whose NIR and SWIR fall below the median by `dark_delta` is a residual shadow.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

CLOUD_SCL = (8, 9, 10)


def sun_angles(safe: str | Path) -> tuple[float, float]:
    """Mean sun zenith and azimuth (degrees) from the tile metadata (GRANULE/*/MTD_TL.xml)."""
    hits = list(Path(safe).glob("GRANULE/*/MTD_TL.xml"))
    if len(hits) != 1:
        raise ValueError("SAFE needs exactly one GRANULE/*/MTD_TL.xml for sun angles")
    root = ET.parse(hits[0]).getroot()
    for node in root.iter():
        if node.tag.split("}")[-1] == "Mean_Sun_Angle":
            vals = {c.tag.split("}")[-1]: float(c.text) for c in node}
            return vals["ZENITH_ANGLE"], vals["AZIMUTH_ANGLE"]
    raise ValueError("Mean_Sun_Angle not found in MTD_TL.xml")


def _shift(mask: np.ndarray, drow: int, dcol: int) -> np.ndarray:
    out = np.zeros_like(mask)
    h, w = mask.shape
    r0, r1 = max(0, drow), min(h, h + drow)
    c0, c1 = max(0, dcol), min(w, w + dcol)
    if r1 > r0 and c1 > c0:
        out[r0:r1, c0:c1] = mask[r0 - drow:r1 - drow, c0 - dcol:c1 - dcol]
    return out


def project_shadows(cloud: np.ndarray, sun_zenith: float, sun_azimuth: float, pixel_m: float,
                    heights=range(250, 6001, 250)) -> np.ndarray:
    """Union of cloud masks displaced away from the sun for each candidate cloud height (metres).

    Azimuth is clockwise from north; shadows fall at azimuth + 180°. Rows grow southwards."""
    cloud = np.asarray(cloud, dtype=bool)
    tz = np.tan(np.radians(sun_zenith))
    az = np.radians(sun_azimuth)
    out = np.zeros_like(cloud)
    for h in heights:
        d = h * tz / pixel_m
        drow, dcol = int(round(d * np.cos(az))), int(round(-d * np.sin(az)))
        out |= _shift(cloud, drow, dcol)
    return out & ~cloud


def shadow_mask(scl: np.ndarray, nir: np.ndarray, sun_zenith: float, sun_azimuth: float, pixel_m: float,
                cloud_prob: np.ndarray | None = None, prob_min: float = 50, nir_dark: float = 0.15,
                heights=range(250, 6001, 250)) -> np.ndarray:
    """Projected shadows confirmed by low NIR reflectance, plus SCL class 3."""
    cloud = np.isin(scl, CLOUD_SCL)
    if cloud_prob is not None:
        cloud |= cloud_prob >= prob_min
    projected = project_shadows(cloud, sun_zenith, sun_azimuth, pixel_m, heights)
    dark = np.isfinite(nir) & (nir < nir_dark)
    return (projected & dark) | (scl == 3)


def temporal_outliers(blue: np.ndarray, nir: np.ndarray, swir: np.ndarray, clear: np.ndarray,
                      blue_delta: float = 0.04, dark_delta: float = 0.04, min_obs: int = 3) -> np.ndarray:
    """Flag residual clouds/shadows in a (n_obs, rows, cols) stack. Returns a mask of observations to drop."""
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mb = np.nanmedian(np.where(clear, blue, np.nan), axis=0)
        mn = np.nanmedian(np.where(clear, nir, np.nan), axis=0)
        ms = np.nanmedian(np.where(clear, swir, np.nan), axis=0)
    enough = clear.sum(axis=0) >= min_obs
    cloudy = blue > mb + blue_delta
    shadow = (nir < mn - dark_delta) & (swir < ms - dark_delta)
    return clear & enough[None] & (cloudy | shadow)


def omnicloudmask_scene(scene_dir: str | Path, batch_size: int = 1) -> Path:
    """Write CLOUDMASK.tif for a scene folder with OmniCloudMask (optional dependency, MIT licence).

    Uses red (B04), green (B03) and NIR (B08) reflectance. Output classes follow OmniCloudMask:
    0 clear, 1 thick cloud, 2 thin cloud, 3 cloud shadow.
    """
    import rasterio

    try:
        from omnicloudmask import predict_from_array
    except ImportError as exc:
        raise RuntimeError("pip install omnicloudmask") from exc
    folder = Path(scene_dir)
    arrays = []
    for b in ("B04", "B03", "B08"):
        with rasterio.open(folder / f"{b}.tif") as ds:
            a = ds.read(1).astype("float32")
            profile = ds.profile
            arrays.append(np.where(a == ds.nodata, 0, a))
    mask = np.asarray(predict_from_array(np.stack(arrays), batch_size=batch_size))[0].astype("uint8")
    profile.update(dtype="uint8", nodata=None, count=1)
    with rasterio.open(folder / "CLOUDMASK.tif", "w", **profile) as dst:
        dst.write(mask, 1)
    return folder / "CLOUDMASK.tif"
