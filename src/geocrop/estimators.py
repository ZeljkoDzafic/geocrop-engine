"""Classifier factory. All estimators share the same spatial CV, tuning, ablation and prediction code.

`algorithm` (in model params):
    rf            RandomForestClassifier (default; Breiman, 2001)
    extra_trees   ExtraTreesClassifier
    hist_gb       HistGradientBoostingClassifier (scikit-learn, no extra dependency)
    lightgbm      LGBMClassifier            (pip install lightgbm)
    catboost      CatBoostClassifier        (pip install catboost; as in ESA WorldCereal)
    tempcnn       Temporal CNN (Pelletier et al., 2019) on `ts` features (pip install torch)

Unknown keys for the chosen algorithm are ignored, so one YAML `model:` section can hold settings for
several algorithms. Further algorithms come from plugins (entry-point group `geocrop.estimators`, see
`geocrop.plugins`); `algorithm: <plugin name>` selects them like a built-in.
"""
from __future__ import annotations

import inspect

import numpy as np

from . import plugins

#: Built-in algorithm names. Plugins add more through `geocrop.plugins.estimators`; see `available_algorithms`.
ALGORITHMS = ("rf", "extra_trees", "hist_gb", "lightgbm", "catboost", "tempcnn")


def _filtered(cls, params: dict, drop=("algorithm",)) -> dict:
    accepted = set(inspect.signature(cls.__init__).parameters)
    return {k: v for k, v in params.items() if k in accepted and k not in drop and v is not None}


def accepted_params(cls, params: dict) -> dict:
    """Subset of `params` that `cls.__init__` accepts (drops `algorithm` and None values); for plugin factories."""
    return _filtered(cls, params)


def _rf(p: dict, seed: int):
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(random_state=seed, **_filtered(RandomForestClassifier, p))


def _extra_trees(p: dict, seed: int):
    from sklearn.ensemble import ExtraTreesClassifier
    return ExtraTreesClassifier(random_state=seed, **_filtered(ExtraTreesClassifier, p))


def _hist_gb(p: dict, seed: int):
    from sklearn.ensemble import HistGradientBoostingClassifier
    p.setdefault("class_weight", "balanced")
    if p.get("class_weight") == "balanced_subsample":
        p["class_weight"] = "balanced"
    # Shared model configuration includes forest-only feature-count defaults.
    # Histogram boosting uses a fraction instead; keep numeric overrides.
    if p.get("max_features") in ("sqrt", "log2"):
        p.pop("max_features")
    n_iter = p.pop("n_estimators", 300)
    p.setdefault("max_iter", n_iter)
    return HistGradientBoostingClassifier(random_state=seed, **_filtered(HistGradientBoostingClassifier, p))


def _lightgbm(p: dict, seed: int):
    try:
        from lightgbm import LGBMClassifier
    except ImportError as exc:
        raise RuntimeError("pip install lightgbm") from exc
    if p.get("class_weight") == "balanced_subsample":
        p["class_weight"] = "balanced"
    p.setdefault("verbosity", -1)
    p.pop("max_features", None)
    return LGBMClassifier(random_state=seed, **{k: v for k, v in p.items() if k != "algorithm" and v is not None
                                                and k not in ("min_samples_leaf", "max_samples")})


def _catboost(p: dict, seed: int):
    try:
        from catboost import CatBoostClassifier
    except ImportError as exc:
        raise RuntimeError("pip install catboost") from exc
    kw = {"iterations": p.get("n_estimators", 500), "depth": p.get("max_depth") or 6,
          "learning_rate": p.get("learning_rate"), "auto_class_weights": "Balanced" if p.get("class_weight") else None,
          "thread_count": p.get("n_jobs", -1), "verbose": False, "random_seed": seed}
    return CatBoostClassifier(**{k: v for k, v in kw.items() if v is not None})


def _tempcnn(p: dict, seed: int):
    return TempCNNClassifier(seed=seed, **_filtered(TempCNNClassifier, p))


for _name, _factory in zip(ALGORITHMS, (_rf, _extra_trees, _hist_gb, _lightgbm, _catboost, _tempcnn)):
    plugins.estimators.register(_name, _factory, source=plugins.BUILTIN)


def available_algorithms() -> tuple[str, ...]:
    """Built-in algorithms followed by registered plugins (entry-point group `geocrop.estimators`)."""
    return tuple(plugins.estimators.names())


def make_estimator(params: dict, seed: int):
    """Unfitted classifier for `params["algorithm"]` (default `rf`), built-in or plugin."""
    algo = (params or {}).get("algorithm", "rf")
    p = dict(params or {})
    try:
        factory = plugins.estimators.get(algo)
    except KeyError:
        raise ValueError(f"Unknown algorithm {algo!r}; choose from {available_algorithms()}") from None
    estimator = factory(p, seed)
    missing = [m for m in ("fit", "predict_proba") if not callable(getattr(estimator, m, None))]
    if missing:
        raise TypeError(f"Estimator plugin {algo!r} returned {type(estimator).__name__} without {missing}; "
                        "estimators need fit(X, y), predict_proba(X) and classes_ after fitting")
    return estimator


def predict_labels(estimator, X) -> np.ndarray:
    """Class labels as a 1-D array (CatBoost returns a column vector)."""
    return np.asarray(estimator.predict(X)).ravel()


def importances(model, n_features: int) -> np.ndarray:
    imp = getattr(model, "feature_importances_", None)
    if imp is None:
        return np.full(n_features, np.nan)
    imp = np.asarray(imp, dtype=float)
    return imp / imp.sum() if imp.sum() > 0 else imp


def channels_from_names(names: list[str]) -> int:
    """Number of time series in an ordered list of `ts_<INDEX>_d<doy>` features."""
    series = [n.split("_d")[0] for n in names if n.startswith("ts_")]
    if len(series) != len(names):
        raise ValueError("tempcnn uses only `ts` features; select groups: [ts]")
    return len(dict.fromkeys(series))


class TempCNNClassifier:
    """Temporal CNN (Pelletier et al., 2019): 3 x [Conv1d(64, k=5) + BN + ReLU + Dropout] -> dense -> softmax.

    Expects features ordered channel-major (all dates of series 1, then series 2, ...), as produced by
    `geocrop.tsfeatures` and `features.select(groups=["ts"])`. Per-channel standardisation is learned in fit.
    """

    def __init__(self, n_channels: int = 1, epochs: int = 60, lr: float = 1e-3, batch_size: int = 256,
                 dropout: float = 0.3, width: int = 64, class_weight: str | None = "balanced", seed: int = 42,
                 device: str | None = None):
        self.n_channels, self.epochs, self.lr, self.batch_size = n_channels, epochs, lr, batch_size
        self.dropout, self.width, self.class_weight, self.seed, self.device = dropout, width, class_weight, seed, device

    def _reshape(self, X):
        X = np.asarray(X, dtype="float32")
        if X.shape[1] % self.n_channels:
            raise ValueError("Feature count is not a multiple of n_channels")
        return X.reshape(len(X), self.n_channels, -1)

    def _net(self, t_len, n_classes):
        from torch import nn
        w, c = self.width, self.n_channels

        def block(a, b):
            return [nn.Conv1d(a, b, 5, padding=2), nn.BatchNorm1d(b), nn.ReLU(), nn.Dropout(self.dropout)]
        return nn.Sequential(*block(c, w), *block(w, w), *block(w, w), nn.Flatten(),
                             nn.Linear(w * t_len, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(self.dropout),
                             nn.Linear(256, n_classes))

    def fit(self, X, y):
        import torch
        from torch.nn import functional as F

        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        X = self._reshape(X)
        self.mean_ = np.nanmean(X, axis=(0, 2), keepdims=True)
        self.std_ = np.nanstd(X, axis=(0, 2), keepdims=True) + 1e-6
        X = np.nan_to_num((X - self.mean_) / self.std_)
        self.classes_, yi = np.unique(np.asarray(y), return_inverse=True)
        counts = np.bincount(yi, minlength=len(self.classes_))
        weight = torch.tensor(len(yi) / (len(counts) * np.maximum(counts, 1)), dtype=torch.float32) \
            if self.class_weight else None
        dev = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.net_ = self._net(X.shape[2], len(self.classes_)).to(dev)
        opt = torch.optim.AdamW(self.net_.parameters(), lr=self.lr, weight_decay=1e-4)
        Xt, yt = torch.from_numpy(X), torch.from_numpy(yi.astype("int64"))
        self.net_.train()
        for _ in range(self.epochs):
            order = rng.permutation(len(X))
            for i in range(0, len(X), self.batch_size):
                b = order[i:i + self.batch_size]
                if len(b) < 2:
                    continue
                opt.zero_grad()
                loss = F.cross_entropy(self.net_(Xt[b].to(dev)), yt[b].to(dev),
                                       weight=None if weight is None else weight.to(dev))
                loss.backward()
                opt.step()
        self.net_.eval().cpu()
        return self

    def predict_proba(self, X):
        import torch
        X = np.nan_to_num((self._reshape(X) - self.mean_) / self.std_)
        with torch.no_grad():
            return torch.softmax(self.net_(torch.from_numpy(X.astype("float32"))), dim=1).numpy()

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]
