"""Spectral and radar index registry.

Every index is defined once, with its exact Sentinel band formula and a literature
reference, so that the feature table in a manuscript can be generated from code
(`geocrop indices`) instead of being transcribed by hand.

Optical inputs are Sentinel-2 L2A surface reflectance in [0, 1]; radar inputs are
linear backscatter (sigma0 or gamma0), never dB.

Further indices come from plugins (entry-point group `geocrop.indices`, see `geocrop.plugins`).
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from . import plugins

S2_BANDS = ("B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12")
S1_POLARISATIONS = ("VV", "VH")


def safe_divide(a, b):
    """Elementwise a / b with NaN where the denominator is (near) zero."""
    a = np.asarray(a, dtype="float32")
    b = np.asarray(b, dtype="float32")
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(np.abs(b) > 1e-8, a / b, np.nan).astype("float32")


def normalized_difference(a, b):
    return safe_divide(a - b, a + b)


def to_db(linear):
    linear = np.asarray(linear, dtype="float32")
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(linear > 0, 10.0 * np.log10(linear), np.nan).astype("float32")


@dataclass(frozen=True)
class Index:
    name: str
    sensor: str
    inputs: tuple[str, ...]
    formula: str
    reference: str
    function: Callable[..., np.ndarray]

    def compute(self, bands: dict[str, np.ndarray]) -> np.ndarray:
        missing = [b for b in self.inputs if b not in bands]
        if missing:
            raise KeyError(f"{self.name} needs bands {missing}")
        return self.function(*(bands[b] for b in self.inputs)).astype("float32")


def _msavi2(nir, red):
    term = (2 * nir + 1) ** 2 - 8 * (nir - red)
    with np.errstate(invalid="ignore"):
        return ((2 * nir + 1 - np.sqrt(np.where(term >= 0, term, np.nan))) / 2).astype("float32")


INDICES: dict[str, Index] = {
    idx.name: idx
    for idx in (
        Index("NDVI", "S2", ("B08", "B04"), "(B08 - B04) / (B08 + B04)",
              "Rouse et al. (1974)", normalized_difference),
        Index("EVI", "S2", ("B08", "B04", "B02"), "2.5 (B08 - B04) / (B08 + 6 B04 - 7.5 B02 + 1)",
              "Huete et al. (2002)", lambda n, r, b: 2.5 * safe_divide(n - r, n + 6 * r - 7.5 * b + 1)),
        Index("EVI2", "S2", ("B08", "B04"), "2.5 (B08 - B04) / (B08 + 2.4 B04 + 1)",
              "Jiang et al. (2008)", lambda n, r: 2.5 * safe_divide(n - r, n + 2.4 * r + 1)),
        Index("MSAVI2", "S2", ("B08", "B04"), "(2 B08 + 1 - sqrt((2 B08 + 1)^2 - 8 (B08 - B04))) / 2",
              "Qi et al. (1994)", _msavi2),
        Index("NDRE", "S2", ("B8A", "B05"), "(B8A - B05) / (B8A + B05)",
              "Gitelson & Merzlyak (1994)", normalized_difference),
        Index("RECI", "S2", ("B08", "B05"), "B08 / B05 - 1",
              "Gitelson et al. (2003)", lambda n, re: safe_divide(n, re) - 1),
        Index("GCVI", "S2", ("B08", "B03"), "B08 / B03 - 1",
              "Gitelson et al. (2003)", lambda n, g: safe_divide(n, g) - 1),
        Index("NDMI", "S2", ("B08", "B11"), "(B08 - B11) / (B08 + B11)",
              "Gao (1996)", normalized_difference),
        Index("NDWI", "S2", ("B03", "B08"), "(B03 - B08) / (B03 + B08)",
              "McFeeters (1996)", normalized_difference),
        Index("BSI", "S2", ("B11", "B04", "B08", "B02"), "((B11 + B04) - (B08 + B02)) / ((B11 + B04) + (B08 + B02))",
              "Rikimaru et al. (2002)", lambda s, r, n, b: normalized_difference(s + r, n + b)),
        Index("PSRI", "S2", ("B04", "B02", "B06"), "(B04 - B02) / B06",
              "Merzlyak et al. (1999)", lambda r, b, re: safe_divide(r - b, re)),
        Index("VV_dB", "S1", ("VV",), "10 log10(VV)", "Torres et al. (2012)", to_db),
        Index("VH_dB", "S1", ("VH",), "10 log10(VH)", "Torres et al. (2012)", to_db),
        Index("VHVV_dB", "S1", ("VH", "VV"), "10 log10(VH / VV)",
              "Veloso et al. (2017)", lambda vh, vv: to_db(safe_divide(vh, vv))),
        Index("RVI", "S1", ("VH", "VV"), "4 VH / (VV + VH)",
              "Kim & van Zyl (2009); dual-pol form", lambda vh, vv: 4 * safe_divide(vh, vv + vh)),
    )
}

for _idx in INDICES.values():
    plugins.indices.register(_idx.name, _idx, source=plugins.BUILTIN)

DEFAULT_S2_INDICES = ("NDVI", "EVI", "MSAVI2", "NDRE", "RECI", "GCVI", "NDMI", "BSI")
DEFAULT_S1_FEATURES = ("VV_dB", "VH_dB", "VHVV_dB")


def available() -> list[str]:
    """Built-in index names followed by plugin indices (entry-point group `geocrop.indices`)."""
    return list(dict.fromkeys([*INDICES, *plugins.indices.names()]))


def get(name: str) -> Index:
    """Index by name: `INDICES` first (built-ins and in-code additions), then registered plugins."""
    if name in INDICES:
        return INDICES[name]
    try:
        return plugins.indices.get(name)
    except KeyError:
        raise KeyError(f"Unknown index {name!r}; available: {', '.join(available())}") from None


def table() -> list[dict[str, str]]:
    """Machine-readable index table for documentation and manuscripts (plugins included)."""
    return [{"name": i.name, "sensor": i.sensor, "formula": i.formula, "reference": i.reference}
            for i in map(get, available())]
