"""Attach external cloud-clearness rasters (e.g. Cloud Score+ ``cs_cdf``) to imported Sentinel-2 scenes.

`geocrop composite` already masks a scene by an optional ``CLDPRB.tif`` (cloud probability 0-100) when
``compositing.cloud_probability_max`` is set. This module writes that file from a clearness score in 0-1
(1 = clear), warped bilinearly to the grid of the scene's SCL: ``CLDPRB = round((1 - score) * 100)``. A pixel with
no score becomes 100 (masked). A Cloud Score+ rule ``cs_cdf >= 0.60`` is then ``cloud_probability_max: 40``.

Score rasters are matched to scene folders by name: ``<scene folder name>.tif`` in the scores directory. The engine
does not download Cloud Score+; export it (e.g. from Google Earth Engine) for the same scenes first.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject


def to_cloud_probability(score: np.ndarray, valid: np.ndarray) -> np.ndarray:
    prob = np.rint((1.0 - np.clip(score, 0.0, 1.0)) * 100.0)
    prob[~valid | ~np.isfinite(score)] = 100
    return prob.astype("uint8")


def attach(scenes: str | Path, scores: str | Path, overwrite: bool = False) -> dict:
    scenes, scores = Path(scenes), Path(scores)
    attached, unmatched, kept = [], [], []
    for scene in sorted(p for p in scenes.iterdir() if p.is_dir() and (p / "SCL.tif").exists()):
        src = scores / f"{scene.name}.tif"
        target = scene / "CLDPRB.tif"
        if not src.exists():
            unmatched.append(scene.name)
            continue
        if target.exists() and not overwrite:
            kept.append(scene.name)
            continue
        with rasterio.open(scene / "SCL.tif") as ref, rasterio.open(src) as ds:
            score = np.full((ref.height, ref.width), np.nan, dtype="float32")
            reproject(rasterio.band(ds, 1), score, src_transform=ds.transform, src_crs=ds.crs,
                      src_nodata=ds.nodata, dst_transform=ref.transform, dst_crs=ref.crs, dst_nodata=np.nan,
                      resampling=Resampling.bilinear)
            prob = to_cloud_probability(score, np.isfinite(score))
            profile = ref.profile.copy()
            profile.update(dtype="uint8", count=1, nodata=None, compress="deflate")
            with rasterio.open(target, "w", **profile) as dst:
                dst.write(prob, 1)
        attached.append(scene.name)
    return {"attached": len(attached), "already_present": len(kept), "without_score": len(unmatched),
            "unmatched_scenes": unmatched[:20]}
