"""Segmentation interface: one small protocol for every parcel-delineation algorithm.

A segmenter turns a raster (path) into a polygon layer written to `output` and returns a summary dict with at
least ``path`` and ``segments``. Built-ins wrap the existing functions in `geocrop.segmentation`, which keep
working unchanged; new algorithms register in `geocrop.plugins.segmenters` (entry-point group
``geocrop.segmenters``) and are then available to `geocrop.benchmark.segmentation` without core changes.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from . import plugins


@runtime_checkable
class Segmenter(Protocol):
    name: str

    def segment(self, raster: str | Path, output: str | Path, **params: Any) -> dict:
        """Write candidate parcel polygons for `raster` to `output`; return ``{"path", "segments", ...}``."""
        ...


class OBIASegmenter:
    """SLIC superpixels with seam reconciliation (`geocrop.segmentation.obia`); CPU only."""

    name = "obia"

    def segment(self, raster: str | Path, output: str | Path, **params: Any) -> dict:
        from .segmentation import obia
        return obia(raster, output, **params)


class UNetWatershedSegmenter:
    """U-Net boundary probability followed by watershed objects (needs a trained model and torch).

    Parameters: ``model`` (checkpoint from `unet_train`, required), ``bands``, ``patch``, ``overlap``, ``device``
    for inference; ``threshold``, ``min_area_pixels``, ``seed_percentile`` for the watershed; and
    ``boundary_output`` for the intermediate probability raster (default: next to `output`).
    """

    name = "unet_watershed"
    _INFER = ("bands", "patch", "overlap", "device")

    def segment(self, raster: str | Path, output: str | Path, model: str | Path | None = None,
                boundary_output: str | Path | None = None, **params: Any) -> dict:
        from .segmentation import boundaries_to_parcels, unet_infer

        if model is None:
            raise ValueError("unet_watershed needs model=<U-Net checkpoint from unet_train>")
        output = Path(output)
        probability = Path(boundary_output) if boundary_output else output.with_name(output.stem + "_boundary.tif")
        infer = {k: params.pop(k) for k in self._INFER if k in params}
        unet_infer(raster, model, probability, **infer)
        info = boundaries_to_parcels(probability, output, **params)
        return {**info, "boundary_probability": str(probability)}


class FTWSegmenter:
    """Pretrained Fields of The World boundary model (`geocrop.ftw`) followed by watershed objects.

    `raster` is an 8-band FTW stack, or any raster on the study grid together with ``window_a``/``window_b``
    (GeoCrop composite or scene folders). Parameters: ``model`` (registry name, default CC-BY B3) or
    ``checkpoint`` (local file), ``allow_noncommercial``, ``cache_dir``, ``patch``, ``overlap``, ``device``,
    ``units``, ``threads``, ``threshold``, ``min_area_pixels``, ``seed_percentile``, ``min_seed_distance``,
    ``work_dir``. The
    checkpoint must already be cached (`geocrop ftw-boundaries --download-only`); this segmenter never
    downloads. Needs torch and segmentation-models-pytorch.
    """

    name = "ftw"

    def segment(self, raster: str | Path, output: str | Path, **params: Any) -> dict:
        from .ftw import segment
        return segment(raster, output, **params)


for _segmenter in (OBIASegmenter, UNetWatershedSegmenter, FTWSegmenter):
    plugins.segmenters.register(_segmenter.name, _segmenter, source=plugins.BUILTIN)


def get(name: str) -> Segmenter:
    """Registered segmenter by name (built-in or plugin)."""
    return plugins.segmenters.get(name)


def available() -> list[str]:
    return plugins.segmenters.names()
