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
import os
from pathlib import Path

import numpy as np
import rasterio
from scipy.spatial import cKDTree

from .parallel import limit_threads, ordered_map, resolve_workers
from .raster import NODATA, blocks, float_profile, valid_mask

#: Below this many pixels the process pool is not worth its start-up time (~1-2 s per spawned worker).
POOL_MIN_PIXELS = 250_000
ROW_CHUNK = 16384          # rows standardised and searched at a time (bounds the float64 temporaries)
BLAS_MIN_FEATURES = 16     # below this a k-d tree is faster than the matrix-product search


class Applicability:
    def __init__(self, mean, std, weights, train_points, mean_distance, threshold, features):
        self.mean, self.std, self.weights = np.asarray(mean), np.asarray(std), np.asarray(weights)
        self.train_points = np.asarray(train_points)
        self.mean_distance, self.threshold, self.features = float(mean_distance), float(threshold), list(features)
        self._tree_cache = None
        self._f32 = None
        self.train_di = None

    @property
    def _tree(self) -> cKDTree:
        if self._tree_cache is None:
            self._tree_cache = cKDTree(self.train_points)
        return self._tree_cache

    def params(self) -> tuple:
        """Constructor arguments (picklable; the k-d tree is rebuilt in each worker process)."""
        return (self.mean, self.std, self.weights, self.train_points, self.mean_distance, self.threshold,
                self.features)

    def transform(self, X) -> np.ndarray:
        return (np.asarray(X, dtype="float64") - self.mean) / self.std * self.weights

    def dissimilarity(self, X) -> np.ndarray:
        X = np.asarray(X)
        out = np.empty(len(X))
        for a in range(0, len(X), ROW_CHUNK):          # float64 working copies stay ~ROW_CHUNK x features, not block-sized
            out[a:a + ROW_CHUNK] = self.nearest_distance(self.transform(X[a:a + ROW_CHUNK]))
        return out / self.mean_distance

    def nearest_distance(self, Z) -> np.ndarray:
        """Distance of each row of `Z` (already standardised and weighted) to its nearest training point.

        In high dimension (>= `BLAS_MIN_FEATURES`) a k-d tree degenerates to a scan, so the nearest point is
        found with a float32 matrix product (BLAS) and its distance is recomputed exactly in float64, over every
        point that rounding error could have made the nearest. The result equals the k-d tree's up to float64
        summation order (~1e-15 relative); about 4x faster on one core for 97 features and 20 000 training points.
        """
        Z = np.asarray(Z, dtype="float64")
        train = self.train_points
        search = os.environ.get("GEOCROP_AOA_SEARCH", "auto")      # "kdtree" = the k-d tree search of earlier releases
        if search == "kdtree" or Z.shape[1] < BLAS_MIN_FEATURES or len(train) < 2 or not len(Z):
            return self._tree.query(Z, k=1)[0]
        if self._f32 is None:
            zz = (train * train).sum(axis=1)
            self._f32 = (train.astype("float32"), zz.astype("float32"), float(zz.max()))
        t32, zz32, zz_max = self._f32
        n = Z.shape[1]
        out = np.empty(len(Z))
        step = max(32, min(2048, 8_000_000 // len(train)))     # keeps the (rows x train) float32 matrix ~32 MB
        for a in range(0, len(Z), step):
            q = Z[a:a + step]
            qq = (q * q).sum(axis=1)
            d2 = q.astype("float32") @ t32.T
            d2 *= -2
            d2 += zz32
            d2 += qq.astype("float32")[:, None]
            first = d2.argmin(axis=1)
            best = np.sqrt(((train[first] - q) ** 2).sum(axis=1))
            # float32 rounding error of any d2 is <= n * eps32 * (|q|^2 + |z|^2): the argmin is the nearest point
            # unless other points lie within twice that error of it (duplicated or nearly identical training
            # samples); for those rows the distance is recomputed exactly over all such candidates
            slack = (2 * n * 6e-8 * (qq + zz_max)).astype("float32")
            close = d2 <= (d2[np.arange(len(q)), first] + slack)[:, None]
            risky = np.flatnonzero(close.sum(axis=1) > 1)
            if len(risky):
                rows, cols = np.nonzero(close[risky])
                dist2 = ((train[cols] - q[risky][rows]) ** 2).sum(axis=1)
                starts = np.r_[0, np.flatnonzero(np.diff(rows)) + 1]
                best[risky] = np.sqrt(np.minimum.reduceat(dist2, starts))
            out[a:a + step] = best
        return out

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
    if X.ndim != 2 or len(X) < 2 or len(folds_index) != len(X):
        raise ValueError("AOA needs a 2-D sample matrix with at least two rows and one fold id per row")
    if not np.isfinite(X).all():
        raise ValueError("AOA training features must be finite (no NaN or infinity)")
    if len(np.unique(folds_index)) < 2:
        raise ValueError("AOA needs at least two cross-validation folds (each sample is measured against other folds)")
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
    mean_distance = float(np.linalg.norm(Z[pair[:, 0]] - Z[pair[:, 1]], axis=1).mean()) if len(pair) else 0.0
    if not mean_distance > 0:
        raise ValueError("AOA is undefined when all training feature vectors are identical")
    di = np.empty(len(Z))
    for f in np.unique(folds_index):
        inside = folds_index == f
        tree = cKDTree(Z[~inside])
        di[inside] = tree.query(Z[inside], k=1)[0] / mean_distance
    q1, q3 = np.percentile(di, [25, 75])
    upper = q3 + 1.5 * (q3 - q1)
    threshold = float(di[di <= upper].max())
    model = Applicability(mean, std, w, Z, mean_distance, threshold, features)
    model.train_di = di      # cross-validated training DIs (reference for domain-shift checks; not serialised)
    return model


def _aoa_block(src, bands, model: Applicability, w):
    """DI and AOA flag of one window (float32 arrays; NODATA where any feature is invalid)."""
    cube = src.read(bands, window=w)
    ok = valid_mask(cube, src.nodata)
    di = np.full(ok.shape, NODATA, dtype="float32")
    flag = np.full(ok.shape, NODATA, dtype="float32")
    if ok.any():
        d = model.dissimilarity(cube[:, ok].T)
        di[ok] = d
        flag[ok] = (d <= model.threshold).astype("float32")
    return di, flag


_WORKER: dict = {}


def _aoa_init(stack, bands, params):       # runs once in every worker process
    limit_threads(1)
    from multiprocessing.util import Finalize
    _WORKER["src"], _WORKER["bands"] = rasterio.open(stack), bands
    Finalize(None, _WORKER["src"].close, exitpriority=10)      # close the GDAL dataset before the worker exits
    _WORKER["model"] = Applicability(*params)


def _aoa_worker(w):
    return _aoa_block(_WORKER["src"], _WORKER["bands"], _WORKER["model"], w)


def map_aoa(stack: str | Path, model: Applicability, output: str | Path, block_size: int = 512,
            workers: int | None = None) -> Path:
    """Write a two-band raster: DI and AOA flag (1 inside, 0 outside).

    The nearest-neighbour search is the most expensive step of a prediction (single core: ~100-150 s per
    megapixel with 97 features). With `workers` > 1 (default: min(usable CPUs, 4)) blocks are computed by that many
    processes, each opening the stack and building the k-d tree itself; the main process writes the blocks in
    order, so the output is identical to ``workers=1`` and at most ``2 x workers`` blocks are in memory.
    """
    with rasterio.open(stack) as src:
        names = list(src.descriptions)
        missing = [f for f in model.features if f not in names]
        if missing:
            raise ValueError(f"Stack {stack} lacks AOA features {missing[:5]}")
        bands = [names.index(f) + 1 for f in model.features]
        out = Path(output)
        out.parent.mkdir(parents=True, exist_ok=True)
        wins = list(blocks(src.width, src.height, block_size))
        workers = resolve_workers(workers)
        if workers > 1 and len(wins) > 1 and src.width * src.height >= POOL_MIN_PIXELS:
            results = ordered_map(_aoa_worker, wins, min(workers, len(wins)), "process", initializer=_aoa_init,
                                  initargs=(str(stack), bands, model.params()))
        else:
            results = (_aoa_block(src, bands, model, w) for w in wins)
        with rasterio.open(out, "w", **float_profile(src, count=2)) as dst:
            dst.set_band_description(1, "dissimilarity_index")
            dst.set_band_description(2, "inside_aoa")
            for w, (di, flag) in zip(wins, results):
                dst.write(di, 1, window=w)
                dst.write(flag, 2, window=w)
    return out
