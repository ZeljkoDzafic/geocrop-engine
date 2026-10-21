"""Sentinel-1 IW GRD: ESA SNAP graph generation/execution, scene import and window compositing.

Scene folder contract:
    VV.tif, VH.tif   geocoded linear backscatter (sigma0 or gamma0), float32
    scene.json       {"sensor": "S1", "datetime": ..., "orbit_direction": "ASCENDING"|"DESCENDING",
                      "relative_orbit": int, "backscatter": "sigma0"|"gamma0"}

Temporal statistics are computed in the linear power domain and converted to dB only at
feature-generation time (averaging dB values biases the mean).
"""
from __future__ import annotations

import json
import subprocess
import warnings
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
import rasterio
from rasterio.warp import Resampling

from . import scenes as scene_meta
from .indices import S1_POLARISATIONS
from .raster import NODATA, blocks, float_profile, read_aligned

ORBITS = {"ASCENDING": "ASC", "DESCENDING": "DESC"}


# --------------------------------------------------------------------------- SNAP

def write_snap_graph(path, backscatter: str = "sigma0", dem: str = "Copernicus 30m Global DEM", pixel_spacing: float = 10.0,
                     crs: str = "EPSG:32634", speckle_filter: str | None = None, speckle_size: int = 5) -> Path:
    """Write a parameterised SNAP GPT graph for IW GRD -> geocoded linear VV/VH.

    sigma0: Orbit -> Thermal noise -> Border noise -> Calibration(sigma0) -> [Speckle] -> Terrain-Correction
    gamma0: Orbit -> Thermal noise -> Border noise -> Calibration(beta0) -> Terrain-Flattening -> [Speckle]
            -> Terrain-Correction (radiometrically terrain-corrected gamma0, Small 2011)

    The output is projected directly to `crs` (use the same CRS as the Sentinel-2 grid) and stays in
    linear power; dB conversion happens at feature time, after temporal averaging.

    Operator parameters follow SNAP 9-11; validate with `gpt <graph> -h` for the installed version.
    """
    if backscatter not in ("sigma0", "gamma0"):
        raise ValueError("backscatter must be sigma0 or gamma0")
    if pixel_spacing <= 0:
        raise ValueError("pixel_spacing must be positive")
    pols = "<selectedPolarisations>VV,VH</selectedPolarisations>"
    calib = ("<outputSigmaBand>true</outputSigmaBand><outputBetaBand>false</outputBetaBand>"
             if backscatter == "sigma0" else "<outputSigmaBand>false</outputSigmaBand><outputBetaBand>true</outputBetaBand>")
    stages = [
        ("Read", None, "<file>$input</file>"),
        ("Apply-Orbit-File", "Read", "<orbitType>Sentinel Precise (Auto Download)</orbitType>"
                                     "<polyDegree>3</polyDegree><continueOnFail>true</continueOnFail>"),
        ("ThermalNoiseRemoval", "Apply-Orbit-File", pols + "<removeThermalNoise>true</removeThermalNoise>"),
        ("Remove-GRD-Border-Noise", "ThermalNoiseRemoval", pols + "<borderLimit>500</borderLimit><trimThreshold>0.5</trimThreshold>"),
        ("Calibration", "Remove-GRD-Border-Noise", pols + calib + "<outputImageScaleInDb>false</outputImageScaleInDb>"),
    ]
    previous = "Calibration"
    if backscatter == "gamma0":
        stages.append(("Terrain-Flattening", previous, f"<demName>{escape(dem)}</demName>"
                                                       "<demResamplingMethod>BILINEAR_INTERPOLATION</demResamplingMethod>"
                                                       "<additionalOverlap>0.1</additionalOverlap><oversamplingMultiple>1.0</oversamplingMultiple>"))
        previous = "Terrain-Flattening"
    if speckle_filter:
        stages.append(("Speckle-Filter", previous, f"<filter>{escape(speckle_filter)}</filter>"
                                                   f"<filterSizeX>{speckle_size}</filterSizeX><filterSizeY>{speckle_size}</filterSizeY>"))
        previous = "Speckle-Filter"
    stages.append(("Terrain-Correction", previous,
                   f"<demName>{escape(dem)}</demName><demResamplingMethod>BILINEAR_INTERPOLATION</demResamplingMethod>"
                   "<imgResamplingMethod>BILINEAR_INTERPOLATION</imgResamplingMethod>"
                   f"<pixelSpacingInMeter>{pixel_spacing}</pixelSpacingInMeter>"
                   f"<mapProjection>{escape(crs)}</mapProjection><nodataValueAtSea>false</nodataValueAtSea>"
                   "<saveSelectedSourceBand>true</saveSelectedSourceBand>"))
    stages.append(("Write", "Terrain-Correction", "<file>$output</file><formatName>GeoTIFF-BigTIFF</formatName>"))
    nodes = []
    for node, source, params in stages:
        refs = f'<sources><sourceProduct refid="{source}"/></sources>' if source else "<sources/>"
        nodes.append(f'  <node id="{node}">\n    <operator>{node}</operator>\n    {refs}\n'
                     f'    <parameters class="com.bc.ceres.binding.dom.XppDomElement">{params}</parameters>\n  </node>')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f'<?xml version="1.0" encoding="UTF-8"?>\n<graph id="geocrop_s1_grd_{backscatter}">\n'
                    "  <version>1.0</version>\n" + "\n".join(nodes) + "\n</graph>\n", encoding="utf8")
    return path


def run_snap(gpt: str, graph: str | Path, product: str | Path, output: str | Path, cache: str = "8G",
             threads: int | None = None) -> int:
    graph = Path(graph)
    if not graph.is_file():
        raise FileNotFoundError(graph)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    command = [str(gpt), str(graph), f"-Pinput={product}", f"-Poutput={output}", "-c", cache]
    if threads:
        command += ["-q", str(threads)]
    return subprocess.run(command, check=True).returncode


# --------------------------------------------------------------------------- import

def import_geotiff(geotiff: str | Path, out: str | Path, product_name: str, orbit_direction: str,
                   backscatter: str = "sigma0") -> Path:
    """Split a SNAP GeoTIFF into VV.tif / VH.tif (by band description) and write scene.json."""
    orbit_direction = orbit_direction.upper()
    if orbit_direction not in ORBITS:
        raise ValueError("orbit_direction must be ASCENDING or DESCENDING")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    with rasterio.open(geotiff) as src:
        names = [d or "" for d in src.descriptions]
        for pol in S1_POLARISATIONS:
            hits = [i for i, n in enumerate(names, 1) if pol in n.upper()]
            if len(hits) != 1:
                raise ValueError(f"Expected one {pol} band in {geotiff}; descriptions={names}")
            data = src.read(hits[0]).astype("float32")
            data = np.where(np.isfinite(data) & (data > 0), data, NODATA)
            with rasterio.open(out / f"{pol}.tif", "w", **float_profile(src)) as dst:
                dst.write(data, 1)
                dst.set_band_description(1, f"{backscatter}_{pol}_linear")
    meta = scene_meta.parse_product_name(product_name)
    scene_meta.write(out, **meta, orbit_direction=orbit_direction, backscatter=backscatter, source=product_name)
    return out


# --------------------------------------------------------------------------- compositing

def composite(scene_dirs, out_dir, reference: str | Path, separate_orbits: bool = False,
              statistic: str = "mean", block_size: int = 512) -> Path:
    """Window composite of linear VV/VH on the reference (Sentinel-2) grid.

    With separate_orbits=True, ascending and descending scenes produce ASC_VV.tif, DESC_VH.tif, ...
    otherwise VV.tif / VH.tif. Statistic: mean or median, in linear power.
    """
    metas = [scene_meta.read(d) for d in scene_dirs]
    if not metas:
        raise ValueError("No Sentinel-1 scenes for this window")
    groups: dict[str, list[dict]] = {}
    for m in metas:
        key = ORBITS.get(str(m.get("orbit_direction", "")).upper(), "") if separate_orbits else ""
        if separate_orbits and not key:
            raise ValueError(f"{m['path']} lacks orbit_direction")
        groups.setdefault(key, []).append(m)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    reducer = {"mean": np.nanmean, "median": np.nanmedian}[statistic]
    with rasterio.open(reference) as ref:
        profile = float_profile(ref)
        for key, members in groups.items():
            prefix = f"{key}_" if key else ""
            handles = [{p: rasterio.open(m["path"] / f"{p}.tif") for p in S1_POLARISATIONS} for m in members]
            try:
                for pol in S1_POLARISATIONS:
                    with rasterio.open(out_dir / f"{prefix}{pol}.tif", "w", **profile) as dst:
                        for w in blocks(ref.width, ref.height, block_size):
                            shape = (int(w.height), int(w.width))
                            layers = []
                            for h in handles:
                                v = read_aligned(h[pol], ref.window_transform(w), ref.crs, shape, Resampling.bilinear)
                                layers.append(np.where(np.isfinite(v) & (v > 0), v, np.nan))
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore", category=RuntimeWarning)
                                value = reducer(np.stack(layers), axis=0)
                            dst.write(np.where(np.isfinite(value), value, NODATA).astype("float32"), 1, window=w)
            finally:
                for h in handles:
                    for ds in h.values():
                        ds.close()
    (out_dir / "composite_s1.json").write_text(json.dumps({
        "sensor": "S1", "statistic": statistic, "domain": "linear", "separate_orbits": separate_orbits,
        "groups": {k or "ALL": [str(m["path"]) for m in v] for k, v in groups.items()}}, indent=2), encoding="utf8")
    return out_dir
