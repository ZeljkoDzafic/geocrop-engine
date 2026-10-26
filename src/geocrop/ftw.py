"""Pretrained field-boundary models from Fields of The World (FTW), run on GeoCrop composites.

FTW (https://fieldsofthe.world, code MIT: github.com/fieldsoftheworld/ftw-baselines) publishes semantic
segmentation checkpoints that predict, per 10 m Sentinel-2 pixel, the classes background / field interior /
field boundary from a TWO-DATE stack. This module is the local (no network) half of the adapter:

* `MODELS` pins every supported checkpoint by URL, size, SHA-256 and licence; only CC-BY-4.0 checkpoints are
  allowed by default (`check_licence`). The download lives in the `geocrop.ftw_weights` adapter.
* `build_stack` writes the FTW input contract from two GeoCrop window folders (composites or scene folders,
  ``B02.tif ... B12.tif`` reflectance in [0, 1]) on the study grid: 8 bands, window A then window B, each
  ``B04 B03 B02 B08`` (red, green, blue, NIR).
* `predict` runs the network blockwise on CPU with an overlap halo (only tile centres are written, so there
  are no seams) and writes class probabilities plus a single-band "not field" raster for
  `geocrop.segmentation.boundaries_to_parcels`.

Input scaling follows FTW inference: Sentinel-2 L2A digital numbers of processing baseline < 04.00
(reflectance x 10000, i.e. the +1000 offset of newer baselines removed) divided by 3000. GeoCrop reflectance
is therefore multiplied by 10000 / 3000; invalid pixels are filled with 0 as in FTW. torch and
segmentation-models-pytorch are optional and imported lazily.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import transform as window_transform

from .raster import NODATA, blocks, expand, float_profile, read_aligned, same_grid, valid_mask

RELEASE_URL = "https://github.com/fieldsoftheworld/ftw-baselines/releases/download/"
WINDOW_BANDS = ("B04", "B03", "B02", "B08")          # red, green, blue, NIR (FTW band order per window)
STACK_BANDS = tuple(f"{w}_{b}" for w in ("A", "B") for b in WINDOW_BANDS)
REFLECTANCE_TO_DN = 10000.0
NORMALISATION = 3000.0                                 # FTW `DEFAULT_NORMALIZATION` for Sentinel-2 checkpoints
CLASS_NAMES = {2: ("background", "field"), 3: ("background", "field", "boundary")}
UNITS_TAG = "GEOCROP_FTW_UNITS"
CC_BY = "CC-BY-4.0"
CITATION = ("Kerner, H. et al. (2025). Fields of The World: A Machine Learning Benchmark Dataset for Global "
            "Agricultural Field Boundary Segmentation. AAAI 2025. arXiv:2409.16252.")


@dataclass(frozen=True)
class FTWModel:
    name: str
    url: str
    sha256: str
    size: int
    licence: str
    version: str
    classes: int
    description: str

    @property
    def commercial_use(self) -> bool:
        return self.licence == CC_BY

    @property
    def filename(self) -> str:
        return self.url.rsplit("/", 1)[-1]


def _m(name, path, sha256, size, licence, version, classes, description):
    return FTWModel(name, RELEASE_URL + path, sha256, size, licence, version, classes, description)


#: Two-window Sentinel-2 checkpoints of the official FTW releases. Names follow the `ftw-tools` model registry;
#: SHA-256 digests are the GitHub release-asset digests (v2, v3, v3.1) or were computed from the downloaded
#: files (v1, which has no published digest).
MODELS: dict[str, FTWModel] = {m.name: m for m in (
    _m("FTW_PRUE_EFNET_B3_CCBY", "v3.1/prue_efnetb3_ccby_checkpoint.ckpt",
       "cd51f020168e66ddd1eff501e81da347b39632af34bebf19a6a20d6c73c5a6d4", 53242827, CC_BY, "v3.1", 3,
       "U-Net EfficientNet-B3, trained on CC-BY/CC0 FTW countries only; fastest v3 model"),
    _m("FTW_PRUE_EFNET_B5_CCBY", "v3.1/prue_efnetb5_ccby_checkpoint.ckpt",
       "f87c041c1a98e1c7f5366012d2a07d4372eb392f36c4bccee4779e73189bcf97", 125921939, CC_BY, "v3.1", 3,
       "U-Net EfficientNet-B5, trained on CC-BY/CC0 FTW countries only"),
    _m("FTW_PRUE_EFNET_B7_CCBY", "v3.1/prue_efnetb7_ccby_checkpoint.ckpt",
       "3eb7624d22262915c82bc3e9bd56d27ea776a6b837df950da87a228c65c4228e", 270127109, CC_BY, "v3.1", 3,
       "U-Net EfficientNet-B7, trained on CC-BY/CC0 FTW countries only; slowest, most accurate v3 CC-BY"),
    _m("FTW_v1_3_Class_CCBY", "v1/3_Class_CCBY_FTW_Pretrained.ckpt",
       "d04671c498c181a2bca4b9c1a82fb8dcc93384eaf55841cb6672692f6fb21d3c", 204471122, CC_BY, "v1", 3,
       "legacy v1 three-class model, CC-BY/CC0 training data only"),
    _m("FTW_v1_2_Class_CCBY", "v1/2_Class_CCBY_FTW_Pretrained.ckpt",
       "805cc73e218195cdb32ccfdf5d04feb9d9f2098aa0180521de78357cc6319bfd", 204468754, CC_BY, "v1", 2,
       "legacy v1 two-class (field / not field) model, CC-BY/CC0 training data only"),
    # Non-commercial: trained on data including CC-BY-NC-SA and other licences ("FULL"). Opt-in only.
    _m("FTW_PRUE_EFNET_B3", "v3/prue_efnet3_checkpoint.ckpt",
       "9021b02e54ccf13d71b78b0ff16e6c75b6970007d6d2f74d3518beba36cd4e43", 53239115, "Mixed Open Licenses", "v3", 3,
       "FULL FTW training data (includes non-commercial licences)"),
    _m("FTW_PRUE_EFNET_B5", "v3/prue_efnet5_checkpoint.ckpt",
       "fcb6134b36ed0e20c00edb2bcd40c10825e01e9cfe150316527f58e9ca590d50", 125916483, "Mixed Open Licenses", "v3", 3,
       "FULL FTW training data (includes non-commercial licences); FTW's default model"),
    _m("FTW_PRUE_EFNET_B7", "v3/prue_efnet7_checkpoint.ckpt",
       "2b1b34a17b85b8f70da6ff737529743b6bc6049e987bce5a1fcdd7279eb3b120", 270119641, "Mixed Open Licenses", "v3", 3,
       "FULL FTW training data (includes non-commercial licences)"),
    _m("FTW_v2_3_Class_FULL_multiWindow", "v2/3_Class_FULL_FTW_Pretrained_v2.ckpt",
       "8d44562829ffc1d3d8a7cca29831ee214be8e0c592f9a35014bd30380f416342", 204472351, "Mixed Open Licenses", "v2", 3,
       "FULL FTW training data (includes non-commercial licences)"),
)}
DEFAULT_MODEL = "FTW_PRUE_EFNET_B3_CCBY"

NONCOMMERCIAL_WARNING = (
    "LICENCE WARNING: FTW model {name!r} was trained on the FULL Fields of The World dataset, which includes "
    "data under non-commercial licences (e.g. CC-BY-NC-SA). FTW recommends it only for non-commercial and "
    "academic use. Do not use its outputs for commercial or agency production work; use a CC-BY checkpoint "
    f"({', '.join(n for n, m in MODELS.items() if m.commercial_use)}) instead."
)


class LicenceError(PermissionError):
    """Raised when a non-commercial checkpoint is requested without explicit opt-in."""


def check_licence(name: str, allow_noncommercial: bool = False) -> FTWModel:
    """Registry entry for `name`; non-CC-BY models require ``allow_noncommercial=True``."""
    if name not in MODELS:
        raise ValueError(f"Unknown FTW model {name!r}; choose one of {sorted(MODELS)}")
    model = MODELS[name]
    if not model.commercial_use and not allow_noncommercial:
        raise LicenceError(f"FTW model {name!r} is licensed {model.licence!r} (non-commercial training data). "
                           "Use a CC-BY checkpoint or opt in explicitly (--allow-noncommercial).")
    return model


def default_cache_dir() -> Path:
    """``$GEOCROP_FTW_CACHE``, else ``$XDG_CACHE_HOME/geocrop/ftw`` (default ``~/.cache/geocrop/ftw``)."""
    if os.environ.get("GEOCROP_FTW_CACHE"):
        return Path(os.environ["GEOCROP_FTW_CACHE"])
    base = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    return base / "geocrop" / "ftw"


def checkpoint_path(name: str, cache_dir: str | Path | None = None) -> Path:
    model = MODELS[name]
    return Path(cache_dir or default_cache_dir()) / model.version / model.filename


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def identify(path: str | Path) -> FTWModel | None:
    """Registry entry whose SHA-256 matches the file at `path` (None for unknown checkpoints)."""
    path = Path(path)
    candidates = [m for m in MODELS.values() if m.size == path.stat().st_size]
    if not candidates:
        return None
    digest = sha256_file(path)
    return next((m for m in candidates if m.sha256 == digest), None)


def cached_checkpoint(name: str, cache_dir: str | Path | None = None, allow_noncommercial: bool = False) -> Path:
    """Verified local checkpoint for registry model `name` (no download; see `geocrop.ftw_weights.download`)."""
    model = check_licence(name, allow_noncommercial)
    path = checkpoint_path(name, cache_dir)
    if not path.exists():
        raise FileNotFoundError(f"FTW checkpoint {name} is not cached at {path}; download it with "
                                f"`geocrop ftw-boundaries --download-only --model {name}`")
    if path.stat().st_size != model.size or sha256_file(path) != model.sha256:
        raise ValueError(f"Checksum mismatch for cached FTW checkpoint {path}; delete it and download again")
    return path


# --------------------------------------------------------------------------- input stack

def _band_file(window: str | Path, band: str) -> Path:
    path = Path(window) / f"{band}.tif"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: a window must be a GeoCrop composite or scene folder "
                                f"with {', '.join(WINDOW_BANDS)}")
    return path


def build_stack(window_a: str | Path, window_b: str | Path, output: str | Path,
                reference: str | Path | None = None, block_size: int = 1024) -> dict:
    """Write the 8-band FTW input stack (reflectance, float32, nodata -9999) on the `reference` grid.

    `window_a` should be early in the growing season (green-up/planting) and `window_b` near its end
    (before harvest), as in FTW training. Bands are warped onto the reference grid only when their grid
    differs (bilinear); default reference is window A's B04.
    """
    files = [_band_file(w, b) for w in (window_a, window_b) for b in WINDOW_BANDS]
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(reference or files[0]) as ref:
        profile = float_profile(ref, count=len(files))
        crs, shape, transform = ref.crs, ref.shape, ref.transform
        sources = [rasterio.open(f) for f in files]
        try:
            with rasterio.open(target, "w", **profile) as dst:
                for i, name in enumerate(STACK_BANDS, start=1):
                    dst.set_band_description(i, name)
                dst.update_tags(**{UNITS_TAG: "reflectance", "GEOCROP_FTW_WINDOW_A": str(window_a),
                                   "GEOCROP_FTW_WINDOW_B": str(window_b), "GEOCROP_FTW_BANDS": " ".join(STACK_BANDS)})
                for w in blocks(shape[1], shape[0], block_size):
                    out = np.empty((len(files), int(w.height), int(w.width)), dtype="float32")
                    for i, src in enumerate(sources):
                        if same_grid(src, ref):
                            arr = src.read(1, window=w).astype("float32")
                            if src.nodata is not None:
                                arr[arr == src.nodata] = np.nan
                        else:
                            arr = read_aligned(src, window_transform(w, transform), crs, out.shape[1:])
                        out[i] = arr
                    ok = valid_mask(out, None)
                    dst.write(np.where(ok[None], out, NODATA).astype("float32"), window=w)
        finally:
            for src in sources:
                src.close()
    return {"path": str(target), "bands": list(STACK_BANDS), "window_a": str(window_a), "window_b": str(window_b),
            "shape": list(shape), "crs": str(crs)}


# --------------------------------------------------------------------------- model

def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise RuntimeError("FTW inference needs PyTorch: pip install torch segmentation-models-pytorch") from exc
    return torch


def load_checkpoint(path: str | Path):
    """Network from an FTW Lightning checkpoint, loaded with ``weights_only=True`` (no pickled code runs).

    Supports the FTW architectures built by `segmentation_models_pytorch` (unet, unet_r, upernet, deeplabv3+);
    the siamese torchgeo variants (fcsiam*) are not supported. Returns ``(model, hyper_parameters)``.
    """
    torch = _torch()
    try:
        import segmentation_models_pytorch as smp
    except ImportError as exc:
        raise RuntimeError("FTW checkpoints need segmentation-models-pytorch (MIT): "
                           "pip install segmentation-models-pytorch") from exc
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    hp = dict(ckpt["hyper_parameters"])
    kind = hp.get("model")
    common = {"encoder_name": hp["backbone"], "encoder_weights": None, "in_channels": hp["in_channels"],
              "classes": hp["num_classes"]}
    builders = {"unet": lambda: smp.Unet(**common),
                "unet_r": lambda: smp.Unet(decoder_channels=(16, 32, 64, 128, 256), **common),
                "upernet": lambda: smp.UPerNet(**common),
                "deeplabv3+": lambda: smp.DeepLabV3Plus(**common)}
    if kind not in builders:
        raise ValueError(f"FTW architecture {kind!r} is not supported by the GeoCrop adapter")
    net = builders[kind]()
    state = {k[len("model."):]: v for k, v in ckpt["state_dict"].items() if k.startswith("model.")}
    net.load_state_dict(state, strict=True)
    return net.eval(), hp


def _stack_units(src, units: str | None) -> str:
    units = units or src.tags().get(UNITS_TAG)
    if units not in ("reflectance", "dn"):
        raise ValueError("Unknown input scaling: pass units='reflectance' (GeoCrop, [0, 1]) or units='dn' "
                         "(FTW/`ftw inference download` stacks, baseline-03 digital numbers); "
                         "it is never inferred from pixel values")
    return units


def predict(stack: str | Path, output_dir: str | Path, model, patch: int = 256, overlap: int = 32,
            device: str = "cpu", units: str | None = None, threads: int | None = None) -> dict:
    """Blockwise class probabilities for an FTW input stack.

    `model` is a checkpoint path or an already-built ``torch.nn.Module`` mapping (N, 8, H, W) to class logits
    (N, C, H, W). Every `patch` x `patch` block is predicted with `overlap` context pixels on each side and
    only its centre is kept, so tiles agree wherever the network's receptive field fits inside the halo.

    Writes ``ftw_probabilities.tif`` (one band per class) and ``ftw_not_field.tif`` = 1 - P(field), the
    single-band "boundary probability" that `boundaries_to_parcels` turns into field objects.
    """
    torch = _torch()
    if threads:
        torch.set_num_threads(int(threads))
    t0 = time.perf_counter()
    hp: dict = {}
    if isinstance(model, (str, Path)):
        model, hp = load_checkpoint(model)
    model = model.to(device).eval()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    probs_path, not_field_path = out / "ftw_probabilities.tif", out / "ftw_not_field.tif"
    tiles, classes = 0, None
    with rasterio.open(stack) as src:
        if src.count != len(STACK_BANDS):
            raise ValueError(f"FTW stack must have {len(STACK_BANDS)} bands (A then B: {' '.join(WINDOW_BANDS)}), "
                             f"got {src.count}")
        scale = (REFLECTANCE_TO_DN if _stack_units(src, units) == "reflectance" else 1.0) / NORMALISATION
        fill = src.nodata if src.nodata is not None else 0
        dsts = {}
        try:
            with torch.inference_mode():
                for w in blocks(src.width, src.height, patch):
                    arr = src.read(window=expand(w, overlap), boundless=True, fill_value=fill).astype("float32")
                    ok = valid_mask(arr, src.nodata)
                    x = np.where(ok[None], arr * scale, 0.0).astype("float32")
                    h, wd = x.shape[1:]
                    ph, pw = (-h) % 32, (-wd) % 32          # encoders downsample 5 times
                    x = np.pad(x, ((0, 0), (0, ph), (0, pw)), mode="reflect" if min(h, wd) > max(ph, pw) else "constant")
                    logits = model(torch.from_numpy(x[None]).to(device))
                    p = torch.softmax(logits.float(), dim=1).cpu().numpy()[0, :, :h, :wd]
                    if classes is None:
                        classes = p.shape[0]
                        if classes not in CLASS_NAMES:
                            raise ValueError(f"Expected a 2- or 3-class FTW model, got {classes} classes")
                        dsts["p"] = rasterio.open(probs_path, "w", **float_profile(src, count=classes))
                        dsts["n"] = rasterio.open(not_field_path, "w", **float_profile(src, count=1))
                        for i, name in enumerate(CLASS_NAMES[classes], start=1):
                            dsts["p"].set_band_description(i, f"p_{name}")
                        dsts["n"].set_band_description(1, "not_field_probability")
                    core = (slice(overlap, overlap + int(w.height)), slice(overlap, overlap + int(w.width)))
                    keep = ok[core]
                    dsts["p"].write(np.where(keep[None], p[:, core[0], core[1]], NODATA).astype("float32"), window=w)
                    dsts["n"].write(np.where(keep, 1.0 - p[1][core], NODATA).astype("float32"), 1, window=w)
                    tiles += 1
        finally:
            for d in dsts.values():
                d.close()
    summary = {"probabilities": str(probs_path), "not_field": str(not_field_path), "classes": list(CLASS_NAMES[classes]),
               "tiles": tiles, "patch": patch, "overlap": overlap, "device": device,
               "seconds": round(time.perf_counter() - t0, 2),
               "architecture": {k: hp[k] for k in ("model", "backbone", "in_channels", "num_classes") if k in hp}}
    return summary


def segment(raster: str | Path, output: str | Path, model: str = DEFAULT_MODEL, checkpoint=None,
            window_a: str | Path | None = None, window_b: str | Path | None = None,
            allow_noncommercial: bool = False, cache_dir: str | Path | None = None, patch: int = 256,
            overlap: int = 32, device: str = "cpu", units: str | None = None, threads: int | None = None,
            threshold: float = 0.5, min_area_pixels: int = 16, seed_percentile: float = 0,
            min_seed_distance: float = 1.0,
            work_dir: str | Path | None = None) -> dict:
    """FTW probabilities -> field objects via `boundaries_to_parcels` (the segmenter used by the registry).

    `raster` is an FTW stack; with `window_a` and `window_b` the stack is built first on the grid of `raster`.
    The checkpoint is `checkpoint` (a local file) or the cached, checksum-verified registry `model`.
    FTW probabilities are nearly binary and boundaries about two pixels wide, so the watershed seeds every
    interior pixel more than `min_seed_distance` pixels from a boundary (``seed_percentile=0``); this keeps the
    narrow strip fields that the generic U-Net defaults (55th percentile, 2 px) drop.
    """
    from .segmentation import boundaries_to_parcels

    output = Path(output)
    work = Path(work_dir) if work_dir else output.with_name(output.stem + "_ftw")
    work.mkdir(parents=True, exist_ok=True)
    licence = None
    if checkpoint is None:
        checkpoint = cached_checkpoint(model, cache_dir, allow_noncommercial)
        licence = MODELS[model].licence
    elif not isinstance(checkpoint, (str, Path)):              # an in-memory torch module (tests, custom nets)
        model, licence = type(checkpoint).__name__, "unknown (in-memory model)"
    else:
        known = identify(checkpoint)
        if known is not None:
            check_licence(known.name, allow_noncommercial)
            model, licence = known.name, known.licence
        else:
            model, licence = Path(checkpoint).name, "unknown (not an FTW registry checkpoint)"
    stack = raster
    if window_a is not None or window_b is not None:
        if window_a is None or window_b is None:
            raise ValueError("Give both window_a and window_b, or an FTW stack as raster")
        stack = build_stack(window_a, window_b, work / "ftw_stack.tif", reference=raster)["path"]
    pred = predict(stack, work, checkpoint, patch=patch, overlap=overlap, device=device, units=units, threads=threads)
    info = boundaries_to_parcels(pred["not_field"], output, threshold=threshold, min_area_pixels=min_area_pixels,
                                 seed_percentile=seed_percentile, min_seed_distance=min_seed_distance)
    summary = {**info, "model": model, "licence": licence, "stack": str(stack),
               "boundary_probability": pred["not_field"], "probabilities": pred["probabilities"],
               "inference_seconds": pred["seconds"], "tiles": pred["tiles"]}
    (work / "ftw_summary.json").write_text(json.dumps({**summary, "citation": CITATION}, indent=2), encoding="utf8")
    return summary


def model_table() -> list[dict]:
    return [{**asdict(m), "commercial_use": m.commercial_use} for m in MODELS.values()]
