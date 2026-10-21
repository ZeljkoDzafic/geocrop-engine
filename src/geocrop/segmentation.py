"""Parcel delineation, for areas without reliable parcel geometries.

Path A (CPU): object-based image analysis with SLIC superpixels, tile-bounded for regional rasters.
Path B (optional GPU): a compact U-Net predicting field-boundary probability, followed by
watershed post-processing into candidate parcel objects.

Neither output is a cadastral parcel; both must be scored against independent reference
boundaries with `geocrop.segmentation_metrics`.
"""
from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape

from .raster import blocks, expand, float_profile, valid_mask


def _normalise(arr: np.ndarray, ok: np.ndarray) -> np.ndarray:
    out = arr.astype("float32").copy()
    for b in range(out.shape[0]):
        lo, hi = np.percentile(out[b][ok], [2, 98])
        out[b] = np.clip((out[b] - lo) / max(hi - lo, 1e-8), 0, 1)
    out[:, ~ok] = 0
    return out


def _write(frame: gpd.GeoDataFrame, output: str | Path) -> Path:
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.unlink(missing_ok=True)
    frame.to_file(target, driver="GPKG" if target.suffix.lower() == ".gpkg" else "GeoJSON")
    return target


def obia(raster: str | Path, output: str | Path, bands=None, tile_size: int = 1024, segments_per_km2: float = 50,
         compactness: float = 0.1, min_area_pixels: int = 9) -> dict:
    """SLIC superpixels per tile -> polygons. `bands` selects stack bands by name or 1-based index
    (e.g. peak-season red, NIR, SWIR and NDVI). Segments are never merged across tile seams; the
    `tile` column allows seam QA."""
    from skimage.segmentation import slic

    pieces = []
    with rasterio.open(raster) as src:
        names = list(src.descriptions)
        idx = [names.index(b) + 1 if isinstance(b, str) else int(b) for b in bands] if bands else list(range(1, src.count + 1))
        px_km2 = abs(src.transform.a * src.transform.e) / 1e6
        for w in blocks(src.width, src.height, tile_size):
            arr = src.read(idx, window=w).astype("float32")
            ok = valid_mask(arr, src.nodata)
            if ok.sum() < min_area_pixels:
                continue
            n_seg = max(2, int(ok.sum() * px_km2 * segments_per_km2))
            labels = slic(np.moveaxis(_normalise(arr, ok), 0, -1), n_segments=n_seg, compactness=compactness,
                          mask=ok, start_label=1, channel_axis=-1)
            for geom, value in shapes(labels.astype("int32"), mask=ok, transform=src.window_transform(w)):
                g = shape(geom)
                area_px = g.area / abs(src.transform.a * src.transform.e)
                if value > 0 and area_px >= min_area_pixels:
                    pieces.append({"segment_id": len(pieces) + 1, "tile": f"{int(w.row_off)}_{int(w.col_off)}",
                                   "area_px": round(area_px, 1), "geometry": g})
        crs = src.crs
    if not pieces:
        raise ValueError("No segments produced")
    path = _write(gpd.GeoDataFrame(pieces, geometry="geometry", crs=crs), output)
    return {"path": str(path), "segments": len(pieces), "seams_merged": False}


def boundaries_to_parcels(boundary_probability: str | Path, output: str | Path, threshold: float = 0.5,
                          min_area_pixels: int = 16, seed_percentile: float = 55) -> dict:
    """Watershed on the distance transform of field interiors (probability < threshold)."""
    from scipy.ndimage import distance_transform_edt
    from skimage.measure import label
    from skimage.segmentation import watershed

    with rasterio.open(boundary_probability) as src:
        if src.count != 1:
            raise ValueError("Expected a single-band boundary probability raster")
        prob = src.read(1)
        ok = valid_mask(prob[None], src.nodata)
        interior = ok & (prob < threshold)
        if not interior.any():
            raise ValueError("No field interiors below threshold")
        dist = distance_transform_edt(interior)
        seeds = label(dist > max(2.0, float(np.percentile(dist[interior], seed_percentile))))
        regions = watershed(-dist, seeds, mask=interior)
        polys = []
        px = abs(src.transform.a * src.transform.e)
        for geom, value in shapes(regions.astype("int32"), mask=interior, transform=src.transform):
            g = shape(geom)
            if value > 0 and g.area / px >= min_area_pixels:
                polys.append({"segment_id": len(polys) + 1, "geometry": g})
        crs = src.crs
    if not polys:
        raise ValueError("No segments after post-processing")
    path = _write(gpd.GeoDataFrame(polys, geometry="geometry", crs=crs), output)
    return {"path": str(path), "segments": len(polys), "validated": False}


# --------------------------------------------------------------------------- U-Net (optional torch)

def _dice_bce(logits, target, pos_weight: float):
    import torch
    import torch.nn.functional as F

    bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=torch.tensor(pos_weight))
    p = torch.sigmoid(logits)
    inter = (p * target).sum(dim=(1, 2, 3))
    dice = 1 - (2 * inter + 1) / (p.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + 1)
    return bce + dice.mean()


def unet_train(images: str | Path, masks: str | Path, model_path: str | Path, epochs: int = 30, batch_size: int = 8,
               lr: float = 1e-3, val_fraction: float = 0.2, seed: int = 42, device: str | None = None) -> dict:
    """Train the boundary U-Net. images: .npy [N,C,H,W]; masks: .npy [N,H,W] with 1 = boundary.

    Loss = class-weighted BCE + Dice (boundaries are a small minority of pixels). Per-channel
    mean/std are stored in the checkpoint and reapplied at inference (normalisation contract).
    Patches should come from spatially separate sites; the validation split here is by patch.
    """
    import torch

    from .unet import UNet

    X = np.load(images).astype("float32")
    Y = np.load(masks).astype("float32")
    if X.ndim != 4 or Y.shape != (X.shape[0], X.shape[2], X.shape[3]):
        raise ValueError("Expected images [N,C,H,W] and masks [N,H,W]")
    if X.shape[0] < 2:
        raise ValueError("Need at least two patches")
    mean = X.mean(axis=(0, 2, 3))
    std = X.std(axis=(0, 2, 3)) + 1e-6
    X = (X - mean[None, :, None, None]) / std[None, :, None, None]
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(X))
    n_val = max(1, int(round(len(X) * val_fraction)))
    val, tr = order[:n_val], order[n_val:] if len(order) > n_val else order
    torch.manual_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    net = UNet(X.shape[1]).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    pos = float(Y[tr].mean())
    pos_weight = (1 - pos) / max(pos, 1e-6)
    history, best, best_state = [], np.inf, None
    for epoch in range(epochs):
        net.train()
        losses = []
        for i in range(0, len(tr), batch_size):
            idx = tr[i:i + batch_size]
            xb = torch.from_numpy(X[idx]).to(device)
            yb = torch.from_numpy(Y[idx, None]).to(device)
            if rng.random() < 0.5:                       # flip augmentation
                xb, yb = xb.flip(-1), yb.flip(-1)
            opt.zero_grad()
            loss = _dice_bce(net(xb), yb, pos_weight)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        net.eval()
        with torch.no_grad():
            v = float(_dice_bce(net(torch.from_numpy(X[val]).to(device)),
                                torch.from_numpy(Y[val, None]).to(device), pos_weight).item())
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)), "val_loss": v})
        if v < best:
            best, best_state = v, {k: t.detach().cpu().clone() for k, t in net.state_dict().items()}
    target = Path(model_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": best_state, "channels": int(X.shape[1]), "mean": mean.tolist(), "std": std.tolist(),
                "history": history, "architecture": "geocrop.unet.UNet"}, target)
    target.with_suffix(".json").write_text(json.dumps({"history": history, "best_val_loss": best}, indent=2))
    return {"best_val_loss": best, "epochs": epochs, "history": history}


def unet_infer(raster: str | Path, model_path: str | Path, output: str | Path, patch: int = 256, overlap: int = 32,
               bands=None, device: str | None = None) -> Path:
    """Tiled boundary-probability inference; each tile is predicted with `overlap` context pixels on
    every side and only its centre is written, which removes tile-edge artefacts."""
    import torch

    from .unet import UNet

    ckpt = torch.load(model_path, map_location="cpu", weights_only=True)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    net = UNet(ckpt["channels"])
    net.load_state_dict(ckpt["state"])
    net.to(device).eval()
    mean = np.asarray(ckpt["mean"], dtype="float32")[:, None, None]
    std = np.asarray(ckpt["std"], dtype="float32")[:, None, None]
    with rasterio.open(raster) as src:
        names = list(src.descriptions)
        idx = [names.index(b) + 1 if isinstance(b, str) else int(b) for b in bands] if bands else list(range(1, src.count + 1))
        if len(idx) != ckpt["channels"]:
            raise ValueError("Band count does not match the checkpoint")
        profile = float_profile(src)
        target = Path(output)
        target.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(target, "w", **profile) as dst, torch.no_grad():
            dst.set_band_description(1, "boundary_probability")
            for w in blocks(src.width, src.height, patch):
                big = expand(w, overlap)
                arr = src.read(idx, window=big, boundless=True, fill_value=src.nodata if src.nodata is not None else 0)
                arr = arr.astype("float32")
                ok = valid_mask(arr, src.nodata)
                arr = np.where(ok[None], (arr - mean) / std, 0).astype("float32")
                h, wd = arr.shape[1:]
                ph, pw = (-h) % 8, (-wd) % 8            # U-Net needs sizes divisible by 2^depth
                arr = np.pad(arr, ((0, 0), (0, ph), (0, pw)))
                pred = torch.sigmoid(net(torch.from_numpy(arr[None]).to(device))).cpu().numpy()[0, 0, :h, :wd]
                core = (slice(overlap, overlap + int(w.height)), slice(overlap, overlap + int(w.width)))
                dst.write(np.where(ok[core], pred[core], -9999.0).astype("float32"), 1, window=w)
    return target
