"""Parcel delineation, for areas without reliable parcel geometries.

Path A (CPU): object-based image analysis with SLIC superpixels, tiled for regional rasters, with a global
stretch and seam reconciliation so that fields cut by tile borders become one object.
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
from rasterio.windows import Window
from shapely.geometry import shape

from .raster import blocks, expand, float_profile, valid_mask

_SKIMAGE_HINT = 'segmentation needs scikit-image: pip install -e ".[segmentation]"'


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


def global_stretch(src, idx, sample: int = 1024, low: float = 2, high: float = 98) -> np.ndarray:
    """Per-band percentiles of a regular subsample of the whole raster, so every tile is normalised identically.

    Every f-th row and column (f = longest side / `sample`, rounded up) is taken, reading window by window along
    the file's own blocks. A single decimated read (`out_shape`) of a pixel-interleaved multi-band stack makes GDAL
    decompress each row of tiles again for every output row once that row of tiles no longer fits the block cache
    (GDAL_CACHEMAX defaults to 5 % of RAM or of the container limit): on a 5 Mpx, 97-band stack in a 4 GB container
    it had not finished after 39 min; read block by block it takes ~10 s.
    """
    f = max(1, int(np.ceil(max(src.width, src.height) / sample)))
    bh, bw = src.block_shapes[0]
    wh, ww = -(-256 // bh) * bh, -(-256 // bw) * bw          # whole blocks, at least 256 px a side
    values = [[] for _ in idx]
    for top in range(0, src.height, wh):
        for left in range(0, src.width, ww):
            r0, c0 = (f // 2 - top) % f, (f // 2 - left) % f      # global rows/cols with index % f == f // 2
            h, w = min(wh, src.height - top), min(ww, src.width - left)
            if r0 >= h or c0 >= w:
                continue
            arr = src.read(idx, window=Window(left, top, w, h))[:, r0::f, c0::f].astype("float32")
            ok = valid_mask(arr, src.nodata)
            for b in range(len(idx)):
                values[b].append(arr[b][ok])
    if not any(v.size for v in values[0]):
        raise ValueError("No valid pixels")
    return np.array([np.percentile(np.concatenate(v), [low, high]) for v in values])


def _stretch(arr, ok, ranges):
    out = np.empty_like(arr, dtype="float32")
    for b, (lo, hi) in enumerate(ranges):
        out[b] = np.clip((arr[b] - lo) / max(hi - lo, 1e-8), 0, 1)
    out[:, ~ok] = 0
    return out


def obia(raster: str | Path, output: str | Path, bands=None, tile_size: int = 1024, segments_per_km2: float = 50,
         compactness: float = 0.1, min_area_pixels: int = 9, merge_seams: bool = True, merge_distance: float = 0.08,
         min_shared_px: float = 3.0) -> dict:
    """SLIC superpixels per tile -> polygons, with seam reconciliation across tiles.

    `bands` selects stack bands by name or 1-based index (e.g. peak-season red, NIR, SWIR and NDVI). All tiles
    use one global 2-98 % stretch. With `merge_seams`, two segments from different tiles are merged when they
    share at least `min_shared_px` pixels of border along a tile seam and their mean (stretched) spectra differ by
    less than `merge_distance` (Euclidean, RMS per band); merging is transitive (union-find). Output segments
    carry stable global ids ordered north-to-south, west-to-east, the tiles they span and whether they were merged.
    """
    try:
        from skimage.segmentation import slic
    except ImportError as exc:
        raise RuntimeError(_SKIMAGE_HINT) from exc

    pieces, means = [], []
    with rasterio.open(raster) as src:
        names = list(src.descriptions)
        idx = [names.index(b) + 1 if isinstance(b, str) else int(b) for b in bands] if bands else list(range(1, src.count + 1))
        px_area = abs(src.transform.a * src.transform.e)
        px_size = abs(src.transform.a)
        ranges = global_stretch(src, idx)
        for w in blocks(src.width, src.height, tile_size):
            arr = src.read(idx, window=w).astype("float32")
            ok = valid_mask(arr, src.nodata)
            if ok.sum() < min_area_pixels:
                continue
            norm = _stretch(arr, ok, ranges)
            n_seg = max(2, int(ok.sum() * px_area / 1e6 * segments_per_km2))
            labels = slic(np.moveaxis(norm, 0, -1), n_segments=n_seg, compactness=compactness, mask=ok,
                          start_label=1, channel_axis=-1)
            counts = np.bincount(labels.ravel())
            sums = np.stack([np.bincount(labels.ravel(), weights=norm[b].ravel(), minlength=len(counts))
                             for b in range(len(idx))], axis=1)
            tile = f"{int(w.row_off)}_{int(w.col_off)}"
            for geom, value in shapes(labels.astype("int32"), mask=ok, transform=src.window_transform(w)):
                g = shape(geom)
                area_px = g.area / px_area
                if value > 0 and area_px >= min_area_pixels:
                    pieces.append({"tile": tile, "area_px": area_px, "geometry": g})
                    means.append(sums[int(value)] / max(counts[int(value)], 1))
        crs = src.crs
        seams_x = {src.bounds.left + c * px_size for c in range(tile_size, src.width, tile_size)}
        seams_y = {src.bounds.top - r * px_size for r in range(tile_size, src.height, tile_size)}
    if not pieces:
        raise ValueError("No segments produced")
    frame = gpd.GeoDataFrame(pieces, geometry="geometry", crs=crs)
    means = np.asarray(means)
    parent = np.arange(len(frame))
    merges = 0
    if merge_seams and (seams_x or seams_y):
        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i
        b = frame.geometry.bounds
        on_seam = np.zeros(len(frame), bool)
        for x in seams_x:
            on_seam |= np.isclose(b.minx, x) | np.isclose(b.maxx, x)
        for y in seams_y:
            on_seam |= np.isclose(b.miny, y) | np.isclose(b.maxy, y)
        cand = np.flatnonzero(on_seam)
        sub = frame.iloc[cand]
        pairs = sub.sindex.query(sub.geometry, predicate="touches")
        tiles = frame.tile.to_numpy()
        for i, j in zip(cand[pairs[0]], cand[pairs[1]]):
            if i >= j or tiles[i] == tiles[j]:
                continue
            shared = frame.geometry.iloc[i].boundary.intersection(frame.geometry.iloc[j].boundary).length
            if shared < min_shared_px * px_size:
                continue
            if np.sqrt(np.mean((means[i] - means[j]) ** 2)) <= merge_distance:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[rj] = ri
                    merges += 1
        parent = np.array([find(i) for i in range(len(frame))])
    frame["group"] = parent
    frame["w_area"] = frame.area_px
    merged = frame.dissolve(by="group", aggfunc={"area_px": "sum", "tile": lambda t: ";".join(sorted(set(t)))})
    merged["parts"] = frame.groupby("group").size()
    merged["seam_merged"] = merged["parts"] > 1
    c = merged.geometry.centroid
    merged = merged.assign(_y=-c.y, _x=c.x).sort_values(["_y", "_x"]).drop(columns=["_y", "_x"]).reset_index(drop=True)
    merged.insert(0, "segment_id", np.arange(1, len(merged) + 1))
    merged["area_px"] = merged.area_px.round(1)
    path = _write(merged[["segment_id", "tile", "parts", "seam_merged", "area_px", "geometry"]], output)
    return {"path": str(path), "segments": int(len(merged)), "segments_before_merge": int(len(frame)),
            "seam_merges": int(merges), "seams_merged": bool(merge_seams)}


def boundaries_to_parcels(boundary_probability: str | Path, output: str | Path, threshold: float = 0.5,
                          min_area_pixels: int = 16, seed_percentile: float = 55,
                          min_seed_distance: float = 2.0) -> dict:
    """Watershed on the distance transform of field interiors (probability < threshold).

    Seeds are interior pixels farther than ``max(min_seed_distance, percentile(distance, seed_percentile))``
    pixels from the nearest boundary; lower both for narrow strip fields.
    """
    from scipy.ndimage import distance_transform_edt
    try:
        from skimage.measure import label
        from skimage.segmentation import watershed
    except ImportError as exc:
        raise RuntimeError(_SKIMAGE_HINT) from exc

    with rasterio.open(boundary_probability) as src:
        if src.count != 1:
            raise ValueError("Expected a single-band boundary probability raster")
        prob = src.read(1)
        ok = valid_mask(prob[None], src.nodata)
        interior = ok & (prob < threshold)
        if not interior.any():
            raise ValueError("No field interiors below threshold")
        dist = distance_transform_edt(interior)
        seeds = label(dist > max(min_seed_distance, float(np.percentile(dist[interior], seed_percentile))))
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


def split_patch_groups(groups, val_fraction: float = 0.2, seed: int = 42):
    """Hold out whole groups (spatial blocks, parcels, sites) so no location is in both train and validation."""
    groups = np.asarray(groups).astype(str)
    uniq = np.unique(groups)
    if len(uniq) < 2:
        raise ValueError("Need at least two patch groups for a grouped validation split")
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    target = max(1, int(round(len(groups) * val_fraction)))
    val_groups, n = [], 0
    for g in uniq[:-1]:                      # always leave at least one group for training
        if n >= target:
            break
        val_groups.append(g)
        n += int((groups == g).sum())
    val = np.flatnonzero(np.isin(groups, val_groups))
    tr = np.flatnonzero(~np.isin(groups, val_groups))
    assert not set(groups[tr]) & set(groups[val]), "patch group leakage"
    return tr, val


def extract_patches(raster: str | Path, boundary_mask: str | Path, size: int = 128, stride: int | None = None,
                    block_m: float = 2000.0, bands=None, min_valid: float = 0.9):
    """Cut aligned image / boundary-mask patches and give each the id of the spatial block containing its centre.

    Returns (X [N,C,H,W], Y [N,H,W], groups [N]) ready for `unet_train(..., groups=groups)`."""
    stride = stride or size
    with rasterio.open(raster) as src, rasterio.open(boundary_mask) as msk:
        if src.shape != msk.shape or not src.transform.almost_equals(msk.transform):
            raise ValueError("Image and boundary mask must share one grid")
        names = list(src.descriptions)
        idx = [names.index(b) + 1 if isinstance(b, str) else int(b) for b in bands] if bands else list(range(1, src.count + 1))
        X, Y, G = [], [], []
        for r in range(0, src.height - size + 1, stride):
            for c in range(0, src.width - size + 1, stride):
                w = rasterio.windows.Window(c, r, size, size)
                img = src.read(idx, window=w).astype("float32")
                ok = valid_mask(img, src.nodata)
                if ok.mean() < min_valid:
                    continue
                img[:, ~ok] = 0
                X.append(img)
                Y.append((msk.read(1, window=w) > 0).astype("float32"))
                cx, cy = rasterio.transform.xy(src.transform, r + size // 2, c + size // 2)
                G.append(f"{int(np.floor(cx / block_m))}:{int(np.floor(cy / block_m))}")
    if not X:
        raise ValueError("No valid patches")
    return np.stack(X), np.stack(Y), np.array(G)


def unet_train(images: str | Path, masks: str | Path, model_path: str | Path, epochs: int = 30, batch_size: int = 8,
               lr: float = 1e-3, val_fraction: float = 0.2, seed: int = 42, device: str | None = None,
               groups=None, allow_random_split: bool = False) -> dict:
    """Train the boundary U-Net. images: .npy [N,C,H,W]; masks: .npy [N,H,W] with 1 = boundary.

    Validation is split by `groups` (one id per patch: spatial block, parcel or site; a .npy/.txt path or
    an array), so that overlapping or neighbouring patches never appear in both training and validation.
    A random patch split is refused unless `allow_random_split=True` (it overestimates performance).
    Per-channel mean/std are computed on the TRAINING patches only, stored in the checkpoint and
    reapplied at inference. Loss = class-weighted BCE + Dice (boundaries are a small minority of pixels).
    """
    import torch

    from .unet import UNet

    X = np.load(images).astype("float32")
    Y = np.load(masks).astype("float32")
    if X.ndim != 4 or Y.shape != (X.shape[0], X.shape[2], X.shape[3]):
        raise ValueError("Expected images [N,C,H,W] and masks [N,H,W]")
    if X.shape[0] < 2:
        raise ValueError("Need at least two patches")
    if groups is None:
        if not allow_random_split:
            raise ValueError("Provide patch groups (spatial blocks/sites) for a leakage-free validation split, "
                             "or set allow_random_split=True explicitly")
        groups = np.arange(len(X)).astype(str)
        split_kind = "random_patches"
    else:
        if isinstance(groups, (str, Path)):
            groups = np.load(groups, allow_pickle=False) if str(groups).endswith(".npy") else \
                np.loadtxt(groups, dtype=str)
        groups = np.asarray(groups).astype(str)
        if len(groups) != len(X):
            raise ValueError("groups must have one entry per patch")
        split_kind = "grouped"
    tr, val = split_patch_groups(groups, val_fraction, seed)
    mean = X[tr].mean(axis=(0, 2, 3))
    std = X[tr].std(axis=(0, 2, 3)) + 1e-6
    X = (X - mean[None, :, None, None]) / std[None, :, None, None]
    rng = np.random.default_rng(seed)
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
    split = {"kind": split_kind, "train_patches": int(len(tr)), "val_patches": int(len(val)),
             "train_groups": int(len(np.unique(groups[tr]))), "val_groups": int(len(np.unique(groups[val]))),
             "normalisation": "train patches only"}
    torch.save({"state": best_state, "channels": int(X.shape[1]), "mean": mean.tolist(), "std": std.tolist(),
                "history": history, "split": split, "architecture": "geocrop.unet.UNet"}, target)
    target.with_suffix(".json").write_text(json.dumps({"history": history, "best_val_loss": best, "split": split},
                                                      indent=2))
    return {"best_val_loss": best, "epochs": epochs, "history": history, "split": split}


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
