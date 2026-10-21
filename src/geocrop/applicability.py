"""Area of Applicability (AOA) and Dissimilarity Index (DI) after Meyer & Pebesma (2021).

Predictions for pixels whose feature vectors are far from every training sample are extrapolations;
their accuracy is unknown even when the cross-validation looks good. The DI of a pixel is its
distance (in standardised, importance-weighted feature space) to the nearest training sample, divided
by the mean pairwise distance within the training data. The
AOA threshold is the outlier-trimmed maximum (upper whisker, Q3 + 1.5 IQR) of the training DIs computed
across cross-validation folds, i.e. each training sample measured against samples of *other* folds.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import rasterio
from scipy.spatial import cKDTree

from .raster import NODATA, blocks, float_profile, valid_mask


class Applicability:
    def __init__(self, mean, std, weights, train_points, mean_distance, threshold, features):
        self.mean, self.std, self.weights = np.asarray(mean), np.asarray(std), np.asarray(weights)
        self.train_points = np.asarray(train_points)
        self.mean_distance, self.threshold, self.features = float(mean_distance), float(threshold), list(features)
        self._tree = cKDTree(self.train_points)

    def transform(self, X) -> np.ndarray:
        return (np.asarray(X, dtype="float64") - self.mean) / self.std * self.weights

    def dissimilarity(self, X) -> np.ndarray:
        d, _ = self._tree.query(self.transform(X), k=1)
        return d / self.mean_distance

    def to_json(self, path: str | Path) -> Path:
        body = {"mean": self.mean.tolist(), "std": self.std.tolist(), "weights": self.weights.tolist(),
                "train_points": self.train_points.tolist(), "mean_distance": self.mean_distance,
                "threshold": self.threshold, "features": self.features, "method": "Meyer & Pebesma (2021)"}
        Path(path).write_text(json.dumps(body), encoding="utf8")
        return Path(path)

    @classmethod
    def from_json(cls, path: str | Path) -> Applicability:
        b = json.loads(Path(path).read_text(encoding="utf8"))
        return cls(b["mean"], b["std"], b["weights"], b["train_points"], b["mean_distance"], b["threshold"], b["features"])


def fit(X, folds_index, features, importance=None, max_points: int = 20000, seed: int = 42) -> Applicability:
    """X: training samples; folds_index: fold id per sample (spatial CV folds); importance: weights per feature."""
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype="float64")
    folds_index = np.asarray(folds_index)
    if len(X) > max_points:
        keep = rng.choice(len(X), max_points, replace=False)
        X, folds_index = X[keep], folds_index[keep]
    mean, std = X.mean(axis=0), X.std(axis=0)
    std[std == 0] = 1.0
    w = np.ones(X.shape[1]) if importance is None else np.nan_to_num(np.asarray(importance, dtype=float))
    if w.sum() == 0:
        w = np.ones(X.shape[1])
    Z = (X - mean) / std * w
    pair = rng.choice(len(Z), size=(min(len(Z) * 5, 200000), 2))
    pair = pair[pair[:, 0] != pair[:, 1]]
    mean_distance = float(np.linalg.norm(Z[pair[:, 0]] - Z[pair[:, 1]], axis=1).mean())
    di = np.empty(len(Z))
    for f in np.unique(folds_index):
        inside = folds_index == f
        tree = cKDTree(Z[~inside])
        di[inside] = tree.query(Z[inside], k=1)[0] / mean_distance
    q1, q3 = np.percentile(di, [25, 75])
    upper = q3 + 1.5 * (q3 - q1)
    threshold = float(di[di <= upper].max())
    return Applicability(mean, std, w, Z, mean_distance, threshold, features)


def map_aoa(stack: str | Path, model: Applicability, output: str | Path, block_size: int = 512) -> Path:
    """Write a two-band raster: DI and AOA flag (1 inside, 0 outside)."""
    with rasterio.open(stack) as src:
        names = list(src.descriptions)
        bands = [names.index(f) + 1 for f in model.features]
        out = Path(output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(out, "w", **float_profile(src, count=2)) as dst:
            dst.set_band_description(1, "dissimilarity_index")
            dst.set_band_description(2, "inside_aoa")
            for w in blocks(src.width, src.height, block_size):
                cube = src.read(bands, window=w)
                ok = valid_mask(cube, src.nodata)
                di = np.full(ok.shape, NODATA, dtype="float32")
                flag = np.full(ok.shape, NODATA, dtype="float32")
                if ok.any():
                    d = model.dissimilarity(cube[:, ok].T)
                    di[ok] = d
                    flag[ok] = (d <= model.threshold).astype("float32")
                dst.write(di, 1, window=w)
                dst.write(flag, 2, window=w)
    return out
