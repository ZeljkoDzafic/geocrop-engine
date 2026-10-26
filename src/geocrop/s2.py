"""Sentinel-2 L2A: SAFE import, cloud/shadow masking and per-window temporal compositing.

Scene folder contract (written by `import_safe`, or prepared by any other tool):
    B02.tif ... B12.tif   surface reflectance in [0, 1], float32, nodata -9999
    SCL.tif               scene classification (uint8)
    CLDPRB.tif            optional cloud probability 0-100 (uint8)
    scene.json            {"sensor": "S2", "datetime": ...}
"""
from __future__ import annotations

import json
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import Resampling
from scipy.ndimage import binary_dilation

from . import attribution, clouds
from . import scenes as scene_meta
from .indices import S2_BANDS
from .raster import (
    NODATA,
    RasterInputError,
    blocks,
    expand,
    float_profile,
    gtiff_profile,
    open_input,
    read_aligned,
)

# ESA SCL: 0 no data, 1 saturated/defective, 3 cloud shadow, 8/9 cloud medium/high, 10 thin cirrus, 11 snow/ice.
DEFAULT_SCL_INVALID = (0, 1, 3, 8, 9, 10, 11)
SCL_MAX = 11                       # ESA L2A scene classification has classes 0..11
_BAND_IDS = {"B02": 1, "B03": 2, "B04": 3, "B05": 4, "B06": 5, "B07": 6, "B08": 7, "B8A": 8, "B11": 11, "B12": 12}


# --------------------------------------------------------------------------- SAFE import

def safe_metadata(safe: str | Path) -> tuple[float, dict[str, float]]:
    """Quantification value and per-band BOA offsets (processing baseline >= 04.00 uses -1000).

    Fails closed: reflectance scaling is never inferred from pixel magnitudes.
    """
    candidates = list(Path(safe).glob("MTD_MSIL2A.xml"))
    if len(candidates) != 1:
        raise ValueError("SAFE must contain exactly one MTD_MSIL2A.xml")
    root = ET.parse(candidates[0]).getroot()

    def nodes(name):
        return [x for x in root.iter() if x.tag.split("}")[-1] == name]

    quant = nodes("BOA_QUANTIFICATION_VALUE")
    if len(quant) != 1:
        raise ValueError("Missing BOA_QUANTIFICATION_VALUE: cannot safely scale reflectance")
    scale = float(quant[0].text)
    if scale <= 0:
        raise ValueError("Invalid quantification value")
    offsets = {int(x.attrib["band_id"]): float(x.text) for x in nodes("BOA_ADD_OFFSET")
               if x.text is not None and "band_id" in x.attrib}
    if offsets and any(_BAND_IDS[b] not in offsets for b in S2_BANDS):
        raise ValueError("Incomplete BOA_ADD_OFFSET metadata")
    return scale, {b: offsets.get(_BAND_IDS[b], 0.0) for b in S2_BANDS}


def _locate(safe: Path, band: str) -> Path:
    if band == "CLDPRB":
        hits = [p for p in safe.rglob("MSK_CLDPRB_20m.jp2")]
        return hits[0] if len(hits) == 1 else None
    resolution = "10m" if band in ("B02", "B03", "B04", "B08") else "20m"
    hits = [p for p in safe.rglob(f"*_{band}_{resolution}.jp2")]
    if len(hits) != 1:
        raise ValueError(f"Expected one {band} {resolution} JP2 in {safe.name}; found {len(hits)}")
    return hits[0]


def import_safe(safe: str | Path, out: str | Path, block_size: int = 1024, shadow_projection: bool = True,
                reference: str | Path | None = None) -> Path:
    """Convert one L2A SAFE to the scene-folder contract.

    By default the output keeps the native 10 m grid of B02 (the whole 110 km tile). With `reference`
    (the study grid from `geocrop grid`) the bands are written directly on that grid, i.e. cropped to the
    AOI and already aligned for compositing, which saves most of the disk space on regional runs.
    20 m bands are bilinearly resampled to 10 m (no information gain; recorded in provenance);
    SCL and CLDPRB use nearest neighbour.
    """
    safe, out = Path(safe), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    quant, offsets = safe_metadata(safe)
    atags = attribution.tags([scene_meta.parse_product_name(safe.name).get("datetime")])
    with rasterio.open(reference or _locate(safe, "B02")) as ref:
        if ref.crs is None:
            raise ValueError("Missing raster CRS")
        profile = float_profile(ref)
        for band in S2_BANDS:
            method = Resampling.nearest if band in ("B02", "B03", "B04", "B08") else Resampling.bilinear
            with rasterio.open(_locate(safe, band)) as src, rasterio.open(out / f"{band}.tif", "w", **profile) as dst:
                dst.update_tags(**atags)
                for w in blocks(ref.width, ref.height, block_size):
                    dn = read_aligned(src, ref.window_transform(w), ref.crs, (int(w.height), int(w.width)),
                                      method, dtype="float32", fill=0, src_nodata=0)
                    ok = dn > 0
                    dst.write(np.where(ok, (dn + offsets[band]) / quant, NODATA).astype("float32"), 1, window=w)
        uint8 = gtiff_profile(ref, count=1, dtype="uint8", nodata=None)
        for name in ("SCL", "CLDPRB"):
            path = _locate(safe, name)
            if path is None:
                continue
            with rasterio.open(path) as src, rasterio.open(out / f"{name}.tif", "w", **uint8) as dst:
                dst.update_tags(**atags)
                for w in blocks(ref.width, ref.height, block_size):
                    a = read_aligned(src, ref.window_transform(w), ref.crs, (int(w.height), int(w.width)),
                                     Resampling.nearest, dtype="uint8", fill=0)
                    dst.write(a, 1, window=w)
    extra = {}
    if shadow_projection:
        try:
            extra = _write_shadow(safe, out, quant, offsets["B8A"])
        except ValueError as exc:            # e.g. MTD_TL.xml missing: keep SCL shadows only
            extra = {"shadow_projection": f"skipped: {exc}"}
    meta = scene_meta.parse_product_name(safe.name)
    scene_meta.write(out, **meta, **extra, source=safe.name, quantification=quant, boa_offsets=offsets,
                     note="20 m bands bilinearly resampled to 10 m; no spatial information gain")
    return out


def _write_shadow(safe: Path, out: Path, quant: float, nir_offset: float) -> dict:
    """Projected cloud shadows on the native 20 m SCL grid (whole scene: shadows travel kilometres)."""
    zenith, azimuth = clouds.sun_angles(safe)
    with rasterio.open(_locate(safe, "SCL")) as s, rasterio.open(_locate(safe, "B8A")) as n:
        scl = s.read(1)
        dn = n.read(1, out_shape=scl.shape).astype("float32")
        nir = np.where(dn > 0, (dn + nir_offset) / quant, np.nan)
        prob = None
        cp = _locate(safe, "CLDPRB")
        if cp is not None:
            with rasterio.open(cp) as c:
                prob = c.read(1, out_shape=scl.shape)
        mask = clouds.shadow_mask(scl, nir, zenith, azimuth, abs(s.transform.a), prob)
        profile = gtiff_profile(s, count=1, dtype="uint8", nodata=None)
    with rasterio.open(out / "SHADOW.tif", "w", **profile) as dst:
        dst.write(mask.astype("uint8"), 1)
    return {"sun_zenith": zenith, "sun_azimuth": azimuth, "shadow_projection": "SHADOW.tif",
            "shadow_fraction": float(mask.mean())}


# --------------------------------------------------------------------------- compositing

def _reduce(stack: np.ndarray, statistic: str) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        if statistic == "median":
            return np.nanmedian(stack, axis=0)
        if statistic == "mean":
            return np.nanmean(stack, axis=0)
        if statistic.startswith("p"):
            return np.nanpercentile(stack, float(statistic[1:]), axis=0)
    raise ValueError(f"Unsupported statistic {statistic}")


def clear_mask(scl: np.ndarray, cloud_prob: np.ndarray | None, scl_invalid=DEFAULT_SCL_INVALID,
               cloud_probability_max: float | None = None, dilation_px: int = 0,
               external: np.ndarray | None = None, shadow: np.ndarray | None = None) -> np.ndarray:
    """True where an observation is usable. Combines SCL, cloud probability, an external cloud mask
    (CLOUDMASK.tif, >0 = invalid) and projected shadows (SHADOW.tif); invalid areas are optionally
    dilated to suppress cloud halos."""
    bad = np.isin(scl, list(scl_invalid))
    if cloud_prob is not None and cloud_probability_max is not None:
        bad |= cloud_prob > cloud_probability_max
    if external is not None:
        bad |= external > 0
    if shadow is not None:
        bad |= shadow > 0
    if dilation_px > 0:
        # Only clouds/shadows are dilated, not no-data borders.
        dilated = binary_dilation(bad & (scl != 0), structure=np.ones((3, 3), bool), iterations=dilation_px)
        bad |= dilated
    return ~bad


def composite(scene_dirs, out_dir, reference: str | Path | None = None, statistic: str = "median",
              scl_invalid=DEFAULT_SCL_INVALID, cloud_probability_max: float | None = None,
              mask_dilation_px: int = 0, min_valid_observations: int = 1, block_size: int = 512,
              temporal_screening: bool = False, blue_delta: float = 0.04, dark_delta: float = 0.04) -> Path:
    """Per-pixel temporal composite of clear observations from several scene folders.

    Optional per-scene CLOUDMASK.tif / SHADOW.tif are honoured; `temporal_screening` additionally drops
    observations that are outliers against the pixel's clear-sky median (residual clouds/shadows).

    Writes one GeoTIFF per band, `valid_count.tif` (clear observations per pixel) and
    `composite.json` (scene list and parameters). Pixels with fewer than
    `min_valid_observations` clear observations are no-data, never zero-filled.
    """
    dirs = [Path(d) for d in scene_dirs]
    if not dirs:
        raise ValueError("At least one scene folder is needed")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ref_path = Path(reference) if reference else dirs[0] / "B02.tif"
    # Tiles acquired on the same day (same datatake, overlapping MGRS tiles) are mosaicked first so
    # that a single acquisition is never counted twice in the statistic or in valid_count.
    dates: dict[str, list[Path]] = {}
    for d in dirs:
        try:
            key = scene_meta.read(d)["date"].isoformat()
        except FileNotFoundError:
            key = str(d)
        dates.setdefault(key, []).append(d)
    handles: dict[str, list[dict]] = {}
    try:
        for key, members in dates.items():
            for d in members:
                scene = {}
                handles.setdefault(key, []).append(scene)        # registered first: closed even on errors
                for b in (*S2_BANDS, "SCL"):
                    scene[b] = open_input(d / f"{b}.tif", "Sentinel-2 scene band")
                for extra in ("CLDPRB", "CLOUDMASK", "SHADOW"):
                    if (d / f"{extra}.tif").exists():
                        scene[extra] = open_input(d / f"{extra}.tif", "Sentinel-2 scene mask")
                if not np.issubdtype(np.dtype(scene["SCL"].dtypes[0]), np.integer):
                    raise RasterInputError(f"Scene classification {scene['SCL'].name} has dtype "
                                           f"{scene['SCL'].dtypes[0]}; SCL must hold integer classes 0-{SCL_MAX}")
        with rasterio.open(ref_path) as ref:
            if ref.crs is None:
                raise ValueError("Reference grid needs a CRS")
            profile = float_profile(ref)
            count_profile = gtiff_profile(ref, count=1, dtype="uint16", nodata=None)
            sinks = {b: rasterio.open(out_dir / f"{b}.tif", "w", **profile) for b in S2_BANDS}
            counter = rasterio.open(out_dir / "valid_count.tif", "w", **count_profile)
            credit = attribution.read([d / "B02.tif" for d in dirs])   # Copernicus attribution of the scenes
            for sink in (*sinks.values(), counter) if credit else ():
                sink.update_tags(**{attribution.TAG: credit, attribution.LICENCE_TAG: attribution.COPERNICUS_LICENCE})
            halo = int(mask_dilation_px)
            try:
                for w in blocks(ref.width, ref.height, block_size):
                    big = expand(w, halo)
                    transform, shape = ref.window_transform(big), (int(big.height), int(big.width))
                    inner = (slice(halo, halo + int(w.height)), slice(halo, halo + int(w.width)))
                    layers = {b: [] for b in S2_BANDS}
                    masks = []
                    for tiles in handles.values():
                        day_ok = np.zeros((int(w.height), int(w.width)), dtype=bool)
                        day = {b: np.full(day_ok.shape, np.nan, dtype="float32") for b in S2_BANDS}
                        for scene in tiles:
                            scl = read_aligned(scene["SCL"], transform, ref.crs, shape, Resampling.nearest, dtype="uint16", fill=0)
                            if scl.max(initial=0) > SCL_MAX:
                                odd = sorted(int(v) for v in np.unique(scl[scl > SCL_MAX])[:5])
                                raise RasterInputError(f"Scene classification {scene['SCL'].name} has values {odd} "
                                                       f"outside the ESA SCL classes 0-{SCL_MAX}; refusing to "
                                                       "treat unknown mask values as clear sky")
                            scl = scl.astype("uint8")
                            # fill 255 (not 0): GDAL would rewrite valid zeros when 0 is the nodata value
                            opt = {k: read_aligned(scene[k], transform, ref.crs, shape, Resampling.nearest,
                                                   dtype="uint8", fill=255)
                                   for k in ("CLDPRB", "CLOUDMASK", "SHADOW") if k in scene}
                            ok = clear_mask(scl, opt.get("CLDPRB"), scl_invalid, cloud_probability_max,
                                            mask_dilation_px, opt.get("CLOUDMASK"), opt.get("SHADOW"))[inner]
                            values = {}
                            for b in S2_BANDS:
                                v = read_aligned(scene[b], transform, ref.crs, shape)[inner]
                                ok &= np.isfinite(v) & (v >= -0.1) & (v <= 1.5)
                                values[b] = v
                            take = ok & ~day_ok          # first clear tile wins
                            for b in S2_BANDS:
                                day[b][take] = values[b][take]
                            day_ok |= ok
                        masks.append(day_ok)
                        for b in S2_BANDS:
                            layers[b].append(day[b])
                    if temporal_screening and len(masks) >= 3:
                        drop = clouds.temporal_outliers(np.stack(layers["B02"]), np.stack(layers["B08"]),
                                                        np.stack(layers["B11"]), np.stack(masks),
                                                        blue_delta, dark_delta)
                        for i in range(len(masks)):
                            masks[i] = masks[i] & ~drop[i]
                            for b in S2_BANDS:
                                layers[b][i] = np.where(drop[i], np.nan, layers[b][i])
                    count = np.sum(masks, axis=0).astype("uint16")
                    counter.write(count, 1, window=w)
                    enough = count >= min_valid_observations
                    for b in S2_BANDS:
                        value = _reduce(np.stack(layers[b]), statistic)
                        value = np.where(enough & np.isfinite(value), value, NODATA).astype("float32")
                        sinks[b].write(value, 1, window=w)
            finally:
                for sink in sinks.values():
                    sink.close()
                counter.close()
    finally:
        for tiles in handles.values():
            for scene in tiles:
                for ds in scene.values():
                    ds.close()
    (out_dir / "composite.json").write_text(json.dumps({
        "sensor": "S2", "acquisitions": {k: [str(d) for d in v] for k, v in dates.items()}, "statistic": statistic,
        "scl_invalid": list(scl_invalid), "cloud_probability_max": cloud_probability_max,
        "mask_dilation_px": mask_dilation_px, "min_valid_observations": min_valid_observations,
        "temporal_screening": temporal_screening,
        "reference_grid": str(ref_path)}, indent=2), encoding="utf8")
    return out_dir
