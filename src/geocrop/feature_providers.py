"""Foundation-model feature providers: one adapter interface for every embedding source of the ``emb`` group.

A *feature provider* turns imagery into per-pixel (and per-parcel) embedding features on the study grid. It
declares what it is (name, licence, pinned weights) and what it needs (input contract), and implements
``embed_pixels`` / ``embed_parcels``. Providers live in the plugin registry
`geocrop.plugins.feature_providers` (entry-point group ``geocrop.feature_providers``), so third-party
encoders plug in exactly like the built-ins:

==========  ===========  ====  ==================================================  ==========================
name        kind         dims  computed from                                       licence (weights / data)
==========  ===========  ====  ==================================================  ==========================
alphaearth  precomputed  64    Google's annual COGs (windowed reads)               CC-BY-4.0
tessera     precomputed  128   Cambridge's annual ``.npy`` tiles (row ranges)      CC0-1.0 (see docs)
presto      monthly      128   GeoCrop monthly S2 (+S1) composites, on the CPU     MIT
galileo     monthly      128   GeoCrop monthly S2 (+S1) composites, on the CPU     MIT (nano model)
anysat      monthly      768   GeoCrop monthly S2 composites, on the CPU           MIT (base model)
==========  ===========  ====  ==================================================  ==========================

The pre-computed providers need the network and are implemented in the adapter `geocrop.embedding_sources`
(registered here by name only, so this core module never imports it). The monthly providers run locally;
their weights are downloaded by `geocrop.embedding_sources.fetch_weights` (pinned URL + SHA-256) and loaded
by ``load_model``. Feature names are ``emb_<provider>_NN`` (namespaced, so several providers fit in one stack);
AlphaEarth, TESSERA and Presto keep the legacy ``emb_NN`` by default for backward compatibility
(``namespace=True`` switches them too). See docs/embeddings.md.

Common keyword options of ``embed_pixels(reference, output, **options)`` and
``embed_parcels(reference, parcels, output, **options)``: ``year``, ``names`` (feature names), ``model``
(a loaded model, monthly providers), ``monthly``/``start``/``months`` (monthly providers), ``block_size``,
``batch_size``, ``extra`` (provenance tags); providers ignore options they do not use.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

from . import embeddings as emb
from . import plugins


@dataclass(frozen=True)
class WeightFile:
    """One pinned file of a model checkpoint, downloaded at runtime into the cache, never committed."""
    name: str            # file name in the provider's cache directory
    url: str             # immutable URL (commit- or revision-pinned)
    sha256: str
    size: int | None = None


@dataclass(frozen=True)
class InputContract:
    """What a provider consumes; documented and recorded in provenance, enforced by the provider."""
    kind: str                         # "precomputed" (read published embeddings) | "monthly" (encode composites)
    sensors: tuple[str, ...]
    bands: tuple[str, ...]
    temporal: str
    normalisation: str
    max_timesteps: int | None = None
    notes: str = ""


@runtime_checkable
class FeatureProvider(Protocol):
    """Contract of a registered feature provider (see the module docstring for the options)."""
    name: str
    dims: int
    sensor: str
    product: str
    licence: str                      # licence id of the weights / data (what ``--allow-license`` names)
    code_licence: str
    attribution: str
    dequantisation: str
    weights: tuple[WeightFile, ...]   # empty for pre-computed sources
    contract: InputContract
    requires_licence_ack: bool        # True: refuse to run unless the licence id is acknowledged
    legacy_names: bool                # True: default feature names are the legacy ``emb_NN``

    def load_model(self, weights: Path | None = None, **options) -> Any: ...

    def embed_pixels(self, reference, output, **options) -> Path: ...

    def embed_parcels(self, reference, parcels, output, **options): ...


class LicenceNotAcknowledged(PermissionError):
    """A provider's weights or data licence must be acknowledged explicitly (``--allow-license <id>``)."""


def get(name: str) -> FeatureProvider:
    return plugins.feature_providers.get(name)


def names() -> list[str]:
    return plugins.feature_providers.names()


def check_licence(provider: FeatureProvider, allow: list[str] | tuple[str, ...] | None = None) -> None:
    """Raise unless the provider is permissive or its licence id is among the acknowledged ones."""
    if getattr(provider, "requires_licence_ack", False) and provider.licence not in set(allow or ()):
        raise LicenceNotAcknowledged(
            f"Feature provider {provider.name!r} is distributed under {provider.licence!r}, which GeoCrop does not "
            f"treat as permissive. Read the licence, then acknowledge it with --allow-license {provider.licence} "
            f"(config: features.embeddings.allow_license: [{provider.licence}]).")


def feature_names(provider: FeatureProvider, dims: int | None = None, namespace: bool | None = None) -> list[str]:
    """Feature names a provider writes: namespaced unless it is a legacy provider and `namespace` is not set."""
    dims = int(dims or provider.dims)
    legacy = getattr(provider, "legacy_names", False) if namespace is None else not namespace
    return emb.band_names(dims, None if legacy else provider.name)


def table() -> list[dict]:
    """One row per registered provider (for listings and docs)."""
    rows = []
    for n in names():
        p = get(n)
        c = p.contract
        rows.append({"name": n, "kind": c.kind, "dims": p.dims, "licence": p.licence, "code_licence": p.code_licence,
                     "sensors": ", ".join(c.sensors), "temporal": c.temporal, "normalisation": c.normalisation,
                     "weights": [w.url for w in p.weights], "requires_licence_ack": p.requires_licence_ack,
                     "source": plugins.feature_providers.source(n)})
    return rows


# --------------------------------------------------------------------------- monthly-composite providers


class MonthlyProvider:
    """Base class of encoders that run on GeoCrop's calendar-month S2/S1 composites (see
    `geocrop.embeddings.timeseries_pixels`). Subclasses implement ``load_model``, ``model_dims`` and
    ``encode(model, s2, s1, latlons, month0, batch_size, state)``; ``fit`` is optional."""

    name = ""
    dims = 0
    sensor = ""
    product = ""
    licence = ""
    code_licence = ""
    attribution = ""
    dequantisation = "none (float32 encoder output)"
    weights: tuple[WeightFile, ...] = ()
    contract: InputContract
    requires_licence_ack = False
    legacy_names = False
    default_batch_size = 512

    def model_dims(self, model) -> int:
        return int(getattr(model, "embedding_size", self.dims))

    def _model(self, options: dict):
        model = options.get("model")
        return model if model is not None else self.load_model(options.get("weights"))

    def embed_pixels(self, reference, output, *, monthly, start: str, months: int = 12, model=None, weights=None,
                     block_size: int = 256, batch_size: int | None = None, extra: dict | None = None,
                     names: list[str] | None = None, state: dict | None = None, **_) -> Path:
        model = self._model({"model": model, "weights": weights})
        return emb.timeseries_pixels(self, monthly, reference, output, start, months, model, block_size,
                                     batch_size or self.default_batch_size, extra, names, state)

    def embed_parcels(self, reference, parcels, output, *, monthly, start: str, months: int = 12, model=None,
                      weights=None, id_field: str = "parcel_id", block_size: int = 512, min_pixels: int = 1,
                      batch_size: int | None = None, names: list[str] | None = None, state: dict | None = None, **_):
        model = self._model({"model": model, "weights": weights})
        return emb.timeseries_parcels(self, monthly, reference, parcels, output, start, months, id_field, model,
                                      block_size, min_pixels, batch_size or self.default_batch_size, names, state)

    def __repr__(self) -> str:
        return f"<FeatureProvider {self.name}>"


def _stack_s2(s2: dict[str, np.ndarray]) -> np.ndarray:
    """GeoCrop S2 series {B02: (T, N), ...} -> (N, T, 10) in Presto/Galileo/AnySat band order."""
    return np.stack([np.asarray(s2[emb.GEOCROP_S2[b]], dtype="float32").T for b in emb.PRESTO_S2], axis=-1)


def _s1_db(s1: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Linear VV/VH (T, N) -> dB (N, T, 2) and validity (N, T) (both polarisations finite and > 0)."""
    vv, vh = (np.asarray(s1[q], dtype="float64") for q in ("VV", "VH"))
    ok = (np.isfinite(vv) & (vv > 0) & np.isfinite(vh) & (vh > 0)).T
    with np.errstate(divide="ignore", invalid="ignore"):
        db = np.stack([10 * np.log10(np.where(np.isfinite(a) & (a > 0), a, 1.0)).T for a in (vv, vh)], axis=-1)
    return np.where(ok[..., None], db, 0).astype("float32"), ok


class PrestoProvider(MonthlyProvider):
    name = "presto"
    dims = 128
    sensor = "Presto"
    product = "Presto encoder embeddings computed from GeoCrop monthly S1/S2 composites"
    licence = "MIT"
    code_licence = "MIT"
    attribution = "Presto: Tseng et al. (2023), arXiv:2304.14065, NASA Harvest (MIT)."
    legacy_names = True
    weights = (WeightFile("presto_11e207a668a3_default_model.pt",
                          "https://raw.githubusercontent.com/nasaharvest/presto/"
                          "11e207a668a34336ced1d8e492a1bd5849b96c4a/data/default_model.pt",
                          "559c9d76f525bdf2dd3774da8b1e7f1510769b73baacac12e9e4d8d5c67fbe65"),)
    contract = InputContract(
        kind="monthly", sensors=("Sentinel-2 L2A (monthly median)", "Sentinel-1 (monthly linear mean, optional)"),
        bands=emb.PRESTO_BANDS, temporal="calendar months, 1..24 consecutive (pre-trained on 12)",
        normalisation="S1 (dB + 25) / 25; S2 reflectance (= DN / 10000); NDVI from B8/B4; ERA5, SRTM and "
                      "Dynamic World masked",
        max_timesteps=emb.PRESTO_MAX_TIMESTEPS)

    def load_model(self, weights=None, **_):
        return emb.load_presto(weights)

    def encode(self, model, s2, s1, latlons, month0, batch_size, state=None):
        x, mask = emb.presto_inputs(s2, s1)
        return emb.presto_embed(model, x, mask, latlons, month0, batch_size)


# --------------------------------------------------------------------------- Galileo

GALILEO_COMMIT = "0f0b5b95ac81acef4b74cf4686dc877202f4541b"
_GALILEO_RAW = f"https://raw.githubusercontent.com/nasaharvest/galileo/{GALILEO_COMMIT}/data/models/nano"
#: Galileo space-time group order (official ``SPACE_TIME_BANDS_GROUPS_IDX``) over the 13 space-time bands.
GALILEO_GROUPS = {"S1": [0, 1], "S2_RGB": [2, 3, 4], "S2_Red_Edge": [5, 6, 7], "S2_NIR_10m": [8],
                  "S2_NIR_20m": [9], "S2_SWIR": [10, 11], "NDVI": [12]}


def galileo_inputs(s2: dict | None, s1: dict | None) -> tuple[np.ndarray, np.ndarray]:
    """Galileo's raw space-time input (N, T, 13) and group mask (N, T, 7), 1 = missing.

    Bands VV, VH (dB), B2 ... B12 (DN = reflectance * 10000), NDVI ((B8 - B4) / (B8 + B4), 0 where the sum is
    not positive). A timestep's S2 groups and NDVI are masked together when any S2 band is missing; S1 when
    either polarisation is missing. Masked values are 0. Normalise with
    `geocrop.galileo_encoder.normalise_space_time` before encoding."""
    ref = next(iter((s2 or s1 or {}).values()), None)
    if ref is None:
        raise ValueError("Galileo needs Sentinel-2 and/or Sentinel-1 series")
    T, N = np.asarray(ref).shape
    x = np.zeros((N, T, 13), dtype="float32")
    mask = np.ones((N, T, len(GALILEO_GROUPS)), dtype="float32")
    if s1:
        db, ok = _s1_db(s1)
        x[:, :, :2] = db
        mask[:, :, 0] = ~ok
    if s2:
        v = _stack_s2(s2)
        ok = np.isfinite(v).all(axis=-1)
        dn = np.where(ok[..., None], v * 10000.0, 0).astype("float32")
        x[:, :, 2:12] = dn
        nir, red = dn[:, :, 6], dn[:, :, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            x[:, :, 12] = np.where((nir + red) > 0, (nir - red) / (nir + red), 0)
        mask[:, :, 1:] = (~ok)[..., None]
    return x, mask


class GalileoProvider(MonthlyProvider):
    name = "galileo"
    dims = 128
    sensor = "Galileo"
    product = "Galileo-nano encoder embeddings computed from GeoCrop monthly S1/S2 composites"
    licence = "MIT"
    code_licence = "MIT"
    attribution = "Galileo: Tseng et al. (2025), arXiv:2502.09356, NASA Harvest (MIT)."
    weights = (WeightFile("galileo_nano_0f0b5b95ac81_config.json", f"{_GALILEO_RAW}/config.json",
                          "8ec17cb83cb78d987b4b920b9c1c35aa245ec0f80e1abf0c22d6f77d828f431c"),
               WeightFile("galileo_nano_0f0b5b95ac81_encoder.pt", f"{_GALILEO_RAW}/encoder.pt",
                          "eab6acacf527886a508a555bc24efe4b4a2cf91b162e84cd1c4a1807e942180c", 4189406))
    contract = InputContract(
        kind="monthly", sensors=("Sentinel-2 (monthly median)", "Sentinel-1 (monthly linear mean, optional)"),
        bands=("VV", "VH", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12", "NDVI"),
        temporal="calendar months, 1..24 consecutive (month of year encoded); pixel time series (h = w = 1, "
                 "patch size 1), as in the official CropHarvest evaluation",
        normalisation="S1 dB, S2 DN; (x - (mean - 2 std)) / (4 std) with the pre-training statistics "
                      "(config/normalization.json); NDVI raw; location from WGS84 lat/lon (unit sphere); "
                      "ERA5, TerraClimate, VIIRS, SRTM, Dynamic World, WorldCereal, LandScan masked",
        max_timesteps=24)
    default_batch_size = 1024

    def load_model(self, weights=None, **_):
        try:
            from . import galileo_encoder
        except ImportError as exc:     # pragma: no cover - depends on the environment
            raise RuntimeError('Galileo embeddings need torch: pip install -e ".[deep]"') from exc
        if weights is None:
            return galileo_encoder.load(None)
        if isinstance(weights, (list, tuple)):
            config, encoder = (Path(p) for p in weights)
            return galileo_encoder.load(config=config, weights=encoder)
        return galileo_encoder.load(Path(weights))

    def encode(self, model, s2, s1, latlons, month0, batch_size, state=None):
        from . import galileo_encoder

        x, mask = galileo_inputs(s2, s1)
        if x.shape[1] > self.contract.max_timesteps:
            raise ValueError(f"Galileo accepts at most {self.contract.max_timesteps} monthly timesteps")
        x = galileo_encoder.normalise_space_time(x)
        x = np.where(np.repeat(mask, [len(v) for v in GALILEO_GROUPS.values()], axis=-1) > 0, 0, x)
        months = np.fmod(np.arange(month0, month0 + x.shape[1]), 12)
        return galileo_encoder.encode_pixels(model, x.astype("float32"), mask, months, latlons, batch_size)


# --------------------------------------------------------------------------- AnySat

ANYSAT_REVISION = "63f252119babfe896d200f88631496f86eddacd3"


def month_day_of_year(month0: int, months: int, day: int = 15) -> np.ndarray:
    """Day of year (``tm_yday``: 1 January = 1, non-leap calendar) of day `day` of each consecutive month from
    `month0` (0 = January): the convention of most AnySat pre-training loaders (TreeSatAI-TS, S2NAIP, PLANTED)."""
    return np.array([date(2023, (month0 + t) % 12 + 1, day).timetuple().tm_yday for t in range(months)])


class AnySatProvider(MonthlyProvider):
    name = "anysat"
    dims = 768
    sensor = "AnySat"
    product = "AnySat (base) encoder embeddings computed from GeoCrop monthly S2 composites"
    licence = "MIT"
    code_licence = "MIT"
    attribution = "AnySat: Astruc et al. (2025), CVPR, arXiv:2412.14123 (MIT)."
    weights = (WeightFile("anysat_63f252119bab_AnySat.pth",
                          f"https://huggingface.co/g-astruc/AnySat/resolve/{ANYSAT_REVISION}/models/AnySat.pth",
                          "ce1246527a28bade624048c448a1fa473ca85791b998f562ce2bcc3dcac93d4f", 503756453),)
    contract = InputContract(
        kind="monthly", sensors=("Sentinel-2 (monthly median)",),
        bands=("B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"),
        temporal="calendar months with a valid observation (per pixel), dated by the day of year (1 January = "
                 "1) of the 15th; pixel = 1 x 1 tile at 10 m, patch size 10 m, 'tile' output",
        normalisation="per-channel standardisation (x - mean) / std with statistics of the study's own "
                      "monthly composites (AnySat convention: per-dataset statistics)",
        notes="Sentinel-1 is not used (units of the pre-training S1 data not verified).")
    default_batch_size = 1024

    def load_model(self, weights=None, **_):
        try:
            from . import anysat_encoder
        except ImportError as exc:     # pragma: no cover - depends on the environment
            raise RuntimeError('AnySat embeddings need torch: pip install -e ".[deep]"') from exc
        return anysat_encoder.load(None if weights is None else Path(weights))

    def fit(self, blocks) -> dict:
        """Per-band mean and standard deviation of all valid monthly S2 observations (one pass)."""
        n, s, ss = 0, np.zeros(10), np.zeros(10)
        for s2, _ in blocks:
            if not s2:
                continue
            v = _stack_s2(s2).reshape(-1, 10).astype("float64")
            v = v[np.isfinite(v).all(axis=1)]
            n += len(v)
            s += v.sum(axis=0)
            ss += (v * v).sum(axis=0)
        if not n:
            raise ValueError("AnySat needs Sentinel-2 composites: no valid observation found")
        mean = s / n
        std = np.sqrt(np.maximum(ss / n - mean * mean, 0))
        return {"s2_mean": mean.round(8).tolist(), "s2_std": np.where(std > 0, std, 1).round(8).tolist(),
                "s2_observations": int(n)}

    def encode(self, model, s2, s1, latlons, month0, batch_size, state=None):
        from . import anysat_encoder

        if not s2:
            raise ValueError("AnySat needs Sentinel-2 composites")
        if not state:
            raise ValueError("AnySat needs per-channel statistics (provider.fit) before encoding")
        v = _stack_s2(s2)
        valid = np.isfinite(v).all(axis=-1)
        x = anysat_encoder.standardise(np.where(valid[..., None], v, 0), np.asarray(state["s2_mean"]),
                                       np.asarray(state["s2_std"]))
        doy = month_day_of_year(month0, v.shape[1])
        return anysat_encoder.encode_pixels(model, x.astype("float32"), valid, doy, batch_size)


PRESTO = PrestoProvider()
GALILEO = GalileoProvider()
ANYSAT = AnySatProvider()

for _p in (PRESTO, GALILEO, ANYSAT):
    plugins.feature_providers.register(_p.name, _p, source=plugins.BUILTIN)
# Pre-computed sources read remote tiles: implemented in the adapter, named here, imported on first use.
plugins.feature_providers.register_lazy("alphaearth", "geocrop.embedding_sources:ALPHAEARTH")
plugins.feature_providers.register_lazy("tessera", "geocrop.embedding_sources:TESSERA")

__all__ = ["ANYSAT", "GALILEO", "PRESTO", "AnySatProvider", "FeatureProvider", "GalileoProvider", "InputContract",
           "LicenceNotAcknowledged", "MonthlyProvider", "PrestoProvider", "WeightFile", "check_licence",
           "feature_names", "galileo_inputs", "get", "month_day_of_year", "names", "table"]
