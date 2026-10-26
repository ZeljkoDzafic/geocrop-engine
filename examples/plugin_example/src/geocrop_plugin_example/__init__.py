"""Minimal GeoCrop plugin: one classifier, one spectral index and one segmenter.

Each object only has to satisfy the contract documented in `geocrop.plugins`; nothing in GeoCrop changes.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from geocrop.indices import Index, normalized_difference

# ---------------------------------------------------------------------------- classifier (geocrop.estimators)


class NearestCentroidProba:
    """Nearest class centroid on standardised features, with softmax(-distance) probabilities."""

    def __init__(self, temperature: float = 1.0):
        self.temperature = temperature

    def fit(self, X, y):
        X = np.nan_to_num(np.asarray(X, dtype="float64"))
        self.mean_, self.scale_ = X.mean(axis=0), X.std(axis=0) + 1e-9
        Z = (X - self.mean_) / self.scale_
        self.classes_, yi = np.unique(np.asarray(y), return_inverse=True)
        self.centroids_ = np.stack([Z[yi == k].mean(axis=0) for k in range(len(self.classes_))])
        return self

    def predict_proba(self, X):
        Z = (np.nan_to_num(np.asarray(X, dtype="float64")) - self.mean_) / self.scale_
        d = np.sqrt(((Z[:, None, :] - self.centroids_[None]) ** 2).mean(axis=2))
        logits = -d / self.temperature
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


def make_nearest_centroid(params: dict, seed: int) -> NearestCentroidProba:
    """Factory contract: (model params from YAML, seed) -> unfitted estimator. Unknown keys are ignored."""
    return NearestCentroidProba(temperature=float(params.get("temperature", 1.0)))


# ---------------------------------------------------------------------------- index (geocrop.indices)

NDTI = Index("NDTI", "S2", ("B11", "B12"), "(B11 - B12) / (B11 + B12)",
             "Van Deventer et al. (1997)", normalized_difference)


# ---------------------------------------------------------------------------- segmenter (geocrop.segmenters)


class GridSegmenter:
    """Baseline segmenter: square cells of `cell_px` pixels over valid data (a floor for benchmarks)."""

    name = "grid"

    def segment(self, raster, output, cell_px: int = 16, **params) -> dict:
        import geopandas as gpd
        import rasterio
        from shapely.geometry import box

        with rasterio.open(raster) as src:
            valid = src.read_masks(1) > 0
            cells = []
            for r in range(0, src.height, cell_px):
                for c in range(0, src.width, cell_px):
                    if valid[r:r + cell_px, c:c + cell_px].any():
                        x0, y0 = src.transform * (c, r)
                        x1, y1 = src.transform * (min(c + cell_px, src.width), min(r + cell_px, src.height))
                        cells.append({"segment_id": len(cells) + 1, "geometry": box(x0, y1, x1, y0)})
            crs = src.crs
        if not cells:
            raise ValueError("No valid pixels")
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        gpd.GeoDataFrame(cells, geometry="geometry", crs=crs).to_file(path, driver="GPKG")
        return {"path": str(path), "segments": len(cells)}
