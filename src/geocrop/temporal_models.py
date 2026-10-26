"""Temporal deep-learning classifiers on (n, T, C) series, usable through the estimator registry.

Both models take the flat, channel-major time-series layout produced by `geocrop.tsfeatures`
(``ts_<SERIES>_d<doy>``: all dates of series 1, then series 2, ...) and by
`geocrop.parcel_classify` (``ts_<INDEX>_<stat>_d<doy>``), reshape it to ``(n, T, C)`` internally and
behave like scikit-learn classifiers (``fit``, ``predict_proba``, ``predict``, ``classes_``,
``get_params``). PyTorch is optional: it is imported only inside ``fit``/``predict_proba``.

* `LightTAEClassifier` - Lightweight Temporal Attention Encoder (Garnot & Landrieu, 2020,
  "Lightweight Temporal Self-Attention for Classifying Satellite Images Time Series", arXiv:2007.00586),
  re-implemented from the paper: a shared linear projection of every date, layer normalisation over
  (time, channel), sinusoidal encoding of the acquisition day of year, one *learned* master query per
  head attending over the dates, values formed by splitting the projected channels between heads (no value
  projection), concatenated head outputs -> MLP -> class logits.
* `geocrop.estimators.TempCNNClassifier` (Pelletier et al., 2019) shares the training loop below.

Training is CPU-friendly (small widths, mini-batches), seeded (``torch.manual_seed`` and a NumPy
generator for batch order and the validation split) and uses early stopping on a stratified validation
fraction of the *training* rows only, restoring the best weights. Inputs are standardised per channel
with statistics learned in ``fit``; missing values become the channel mean after standardisation.
The early-stopping split is random within the training rows; outer spatial cross-validation remains the
only accuracy estimate.
"""
from __future__ import annotations

import copy
import math

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


def _torch():
    try:
        import torch
    except ImportError as exc:  # optional dependency
        raise RuntimeError("Temporal deep-learning classifiers need PyTorch: pip install torch "
                           "(or the geocrop 'dl' extra)") from exc
    return torch


def series_names(names: list[str]) -> tuple[list[str], list[int]]:
    """(series, day-of-year grid) of a channel-major ``ts_<SERIES>_d<doy>`` feature list.

    Raises ValueError when a name is not a time-series feature, the layout is not channel-major or
    the series do not share one date grid.
    """
    series, days = [], {}
    for n in names:
        head, sep, tail = n.rpartition("_d")
        if not n.startswith("ts_") or not sep or not tail.isdigit():
            raise ValueError(f"Temporal models use only `ts_<series>_d<doy>` features; got {n!r} "
                             "(select groups: [ts])")
        if not series or series[-1] != head:
            if head in days:
                raise ValueError("Time-series features must be channel-major (all dates of one series together)")
            series.append(head)
            days[head] = []
        days[head].append(int(tail))
    grids = {tuple(v) for v in days.values()}
    if len(grids) != 1:
        raise ValueError("All time series must share the same day-of-year grid")
    return series, list(grids.pop())


def to_series(X, n_channels: int) -> np.ndarray:
    """Flat channel-major (n, C*T) -> (n, T, C) float32."""
    X = np.asarray(X, dtype="float32")
    if X.ndim != 2 or n_channels < 1 or X.shape[1] % n_channels:
        raise ValueError("Feature count is not a multiple of n_channels")
    return X.reshape(len(X), n_channels, -1).transpose(0, 2, 1)


def _validation_split(yi: np.ndarray, fraction: float, rng: np.random.Generator):
    """Stratified random split of training rows; classes with < 2 rows stay in training."""
    val = []
    for k in np.unique(yi):
        rows = np.flatnonzero(yi == k)
        n = int(round(fraction * len(rows)))
        if len(rows) >= 2 and n >= 1:
            val.append(rng.choice(rows, size=min(n, len(rows) - 1), replace=False))
    val = np.sort(np.concatenate(val)) if val else np.array([], dtype=int)
    return np.setdiff1d(np.arange(len(yi)), val), val


def fit_network(est, build, X: np.ndarray, y) -> None:
    """Shared seeded training loop (AdamW, class-weighted cross-entropy, optional early stopping).

    `est` provides seed, epochs, lr, batch_size, class_weight, device, patience, validation_fraction;
    `build(n_classes)` returns the torch module for prepared inputs `X` (already standardised).
    With ``patience=None`` no validation rows are held out and all epochs run.
    Sets ``classes_``, ``net_``, ``n_epochs_`` and ``best_epoch_`` on `est`.
    """
    torch = _torch()
    from torch.nn import functional as F

    torch.manual_seed(est.seed)
    rng = np.random.default_rng(est.seed)
    est.classes_, yi = np.unique(np.asarray(y), return_inverse=True)
    counts = np.bincount(yi, minlength=len(est.classes_))
    weight = torch.tensor(len(yi) / (len(counts) * np.maximum(counts, 1)), dtype=torch.float32) \
        if est.class_weight else None
    patience = getattr(est, "patience", None)
    train_idx, val_idx = np.arange(len(yi)), np.array([], dtype=int)
    if patience:
        train_idx, val_idx = _validation_split(yi, getattr(est, "validation_fraction", 0.1), rng)
    dev = est.device or ("cuda" if torch.cuda.is_available() else "cpu")
    net = build(len(est.classes_)).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=est.lr, weight_decay=1e-4)
    Xt, yt = torch.from_numpy(np.ascontiguousarray(X)), torch.from_numpy(yi.astype("int64"))
    w = None if weight is None else weight.to(dev)
    best, best_state, best_epoch, wait, epoch = math.inf, None, 0, 0, 0
    for epoch in range(1, est.epochs + 1):
        net.train()
        order = train_idx[rng.permutation(len(train_idx))]
        for i in range(0, len(order), est.batch_size):
            b = order[i:i + est.batch_size]
            if len(b) < 2:      # batch normalisation needs two rows
                continue
            opt.zero_grad()
            loss = F.cross_entropy(net(Xt[b].to(dev)), yt[b].to(dev), weight=w)
            loss.backward()
            opt.step()
        if len(val_idx):
            net.eval()
            with torch.no_grad():
                v = float(F.cross_entropy(net(Xt[val_idx].to(dev)), yt[val_idx].to(dev), weight=w))
            if v < best - 1e-4:
                best, best_state, best_epoch, wait = v, copy.deepcopy(net.state_dict()), epoch, 0
            else:
                wait += 1
                if wait >= patience:
                    break
    if best_state is not None:
        net.load_state_dict(best_state)
    est.net_ = net.eval().cpu()
    est.n_epochs_ = epoch
    est.best_epoch_ = best_epoch or epoch


def predict_network(net, X: np.ndarray, batch_size: int = 8192) -> np.ndarray:
    torch = _torch()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            out.append(torch.softmax(net(torch.from_numpy(np.ascontiguousarray(X[i:i + batch_size]))), dim=1).numpy())
    return np.concatenate(out) if out else np.zeros((0, 0), dtype="float32")


def sinusoid_table(positions, d: int, period: float = 1000.0) -> np.ndarray:
    """Transformer positional encoding evaluated at arbitrary (day-of-year) positions: (T, d)."""
    pos = np.asarray(positions, dtype="float64")[:, None]
    i = np.arange(d)[None, :]
    angle = pos / np.power(period, 2 * (i // 2) / d)
    return np.where(i % 2 == 0, np.sin(angle), np.cos(angle)).astype("float32")


def _ltae_net(n_channels, t_len, n_classes, positions, d_model, n_head, d_k, mlp, dropout, period):
    torch = _torch()
    from torch import nn

    if d_model % n_head:
        raise ValueError("d_model must be divisible by n_head")

    class LTAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.project = nn.Linear(n_channels, d_model)                 # per-date shared projection
            self.norm = nn.LayerNorm([t_len, d_model])                    # over (time, channel) as in L-TAE
            table = sinusoid_table(positions, d_model // n_head, period)
            self.register_buffer("position", torch.from_numpy(np.tile(table, (1, n_head))))
            self.query = nn.Parameter(torch.randn(n_head, d_k) * math.sqrt(2.0 / d_k))   # master queries
            self.key = nn.Linear(d_model, n_head * d_k)
            self.attention_dropout = nn.Dropout(0.1)
            layers, width = [], d_model
            for m in mlp:
                layers += [nn.Linear(width, m), nn.BatchNorm1d(m), nn.ReLU()]
                width = m
            self.mlp = nn.Sequential(*layers, nn.Dropout(dropout), nn.LayerNorm(width))
            self.decoder = nn.Sequential(nn.Linear(width, max(width // 2, n_classes)), nn.ReLU(),
                                         nn.Linear(max(width // 2, n_classes), n_classes))

        def attend(self, x):
            b = x.shape[0]
            h = self.norm(self.project(x)) + self.position                     # (B, T, d)
            k = self.key(h).view(b, t_len, n_head, d_k).permute(0, 2, 1, 3)    # (B, H, T, dk)
            score = torch.einsum("bhtk,hk->bht", k, self.query) / math.sqrt(d_k)
            att = torch.softmax(score, dim=-1)                                 # (B, H, T)
            v = h.view(b, t_len, n_head, d_model // n_head).permute(0, 2, 1, 3)
            return torch.einsum("bht,bhtc->bhc", self.attention_dropout(att), v).reshape(b, d_model), att

        def forward(self, x):
            return self.decoder(self.mlp(self.attend(x)[0]))

    return LTAE()


class LightTAEClassifier(ClassifierMixin, BaseEstimator):
    """Lightweight Temporal Attention Encoder (Garnot & Landrieu, 2020) as a scikit-learn classifier.

    n_channels   number of series in the channel-major feature layout (filled from feature names by
                 `geocrop.estimators.sequence_params`)
    positions    acquisition day of year of every time step (filled from ``_d<doy>`` names); default 1..T
    d_model, n_head, d_k, mlp   encoder sizes; CPU-friendly defaults (the paper uses 256/16/8/(256, 128))
    epochs, patience, validation_fraction   maximum epochs and early stopping on a stratified validation
                 fraction of the training rows (``patience=None`` disables it)
    """

    def __init__(self, n_channels: int = 1, positions=None, d_model: int = 64, n_head: int = 8, d_k: int = 8,
                 mlp=(64,), dropout: float = 0.2, period: float = 1000.0, epochs: int = 100, lr: float = 1e-3,
                 batch_size: int = 128, patience: int | None = 10, validation_fraction: float = 0.1,
                 class_weight: str | None = "balanced", seed: int = 42, device: str | None = None):
        self.n_channels, self.positions, self.d_model, self.n_head, self.d_k = n_channels, positions, d_model, n_head, d_k
        self.mlp, self.dropout, self.period, self.epochs, self.lr = mlp, dropout, period, epochs, lr
        self.batch_size, self.patience, self.validation_fraction = batch_size, patience, validation_fraction
        self.class_weight, self.seed, self.device = class_weight, seed, device

    def _prepare(self, X):
        Z = to_series(X, self.n_channels)
        return np.nan_to_num((Z - self.mean_) / self.std_).astype("float32")

    def fit(self, X, y):
        Z = to_series(X, self.n_channels)
        t_len = Z.shape[1]
        positions = list(range(1, t_len + 1)) if self.positions is None else list(self.positions)
        if len(positions) != t_len:
            raise ValueError(f"positions has {len(positions)} entries for {t_len} time steps")
        self.mean_ = np.nanmean(Z, axis=(0, 1), keepdims=True)
        self.std_ = np.nanstd(Z, axis=(0, 1), keepdims=True) + 1e-6
        self.mean_, self.std_ = np.nan_to_num(self.mean_), np.nan_to_num(self.std_, nan=1.0)
        self.t_len_, self.positions_ = t_len, positions
        fit_network(self, self._build, self._prepare(X), y)
        return self

    def _build(self, n_classes: int):
        return _ltae_net(self.n_channels, self.t_len_, n_classes, self.positions_, self.d_model, self.n_head,
                         self.d_k, tuple(self.mlp), self.dropout, self.period)

    # The network class is created lazily (torch is optional), so pickles store its weights instead.
    def __getstate__(self):
        state = dict(super().__getstate__())
        net = state.pop("net_", None)
        if net is not None:
            state["_net_weights"] = {k: v.detach().cpu().numpy() for k, v in net.state_dict().items()}
        return state

    def __setstate__(self, state):
        state = dict(state)
        weights = state.pop("_net_weights", None)
        super().__setstate__(state)
        if weights is not None:
            torch = _torch()
            self.net_ = self._build(len(self.classes_))
            self.net_.load_state_dict({k: torch.from_numpy(np.array(v)) for k, v in weights.items()})
            self.net_.eval()

    def predict_proba(self, X):
        return predict_network(self.net_, self._prepare(X))

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]

    def attention(self, X) -> np.ndarray:
        """Attention weights (n, n_head, T) of the master queries: which dates the model relies on."""
        torch = _torch()
        with torch.no_grad():
            return self.net_.attend(torch.from_numpy(self._prepare(X)))[1].numpy()
