# Parcel segmentation

When reliable parcel geometries (LPIS) exist, use them — segmentation is for areas without boundaries, for
checking boundary changes, and for research comparison.

## Path A — OBIA (CPU)
`geocrop segment-obia --raster stack.tif --bands P3_B04 P3_B08 P3_B11 P3_NDVI --out segments.gpkg`
* SLIC superpixels on 2–98 % stretched bands, per tile (`--tile-size`, default 1024 px).
* `--segments-per-km2` controls the expected object density (≈ 1 / mean field size); `--compactness` the shape regularity.
* Objects smaller than 9 px are dropped; objects are not merged across tile seams (the `tile` column supports seam QA).

## Path B — U-Net boundaries (optional GPU)
```bash
geocrop segment-unet-train --images patches_X.npy --masks patches_boundary.npy --model boundary.pt --epochs 50
geocrop segment-unet-infer --raster stack.tif --bands P2_B04 P2_B08 P3_B04 P3_B08 --model boundary.pt --out boundary.tif
geocrop boundaries-to-parcels --raster boundary.tif --out candidates.gpkg --threshold 0.5
```
* Compact three-level U-Net (Ronneberger et al., 2015) with batch normalisation.
* Loss: class-weighted BCE + Dice (boundary pixels are a small minority); flip augmentation; best epoch by validation loss.
* Per-channel mean/std are stored in the checkpoint and reapplied at inference (normalisation contract).
* Inference uses overlapping tiles (`--overlap` context pixels, centre written) to avoid edge artefacts.
* Watershed on the distance transform of field interiors converts boundaries into candidate objects.
* Training patches must come from areas spatially separate from the evaluation parcels. Pretrained global
  field-boundary models (e.g. Fields of The World) can be used upstream; check their weight licences separately.

## Evaluation
* `geocrop evaluate-segments --segments candidates.gpkg --reference parcels_ref.gpkg --out seg.json`:
  over-/under-segmentation and D (Clinton et al., 2010) per reference object, plus one-to-one matched-object
  precision/recall/F1 at IoU ≥ 0.5 and mean matched IoU.
* `geocrop evaluate-boundaries` compares boundary rasters on one grid: precision/recall/F1 within a tolerance
  (reported in pixels and metres) and boundary-pixel IoU (overlap of boundary masks, not parcel IoU).

Report both segmentation paths on the same withheld reference parcels; do not label candidate objects as parcels.

## References
Achanta et al. (2012) IEEE TPAMI 34:2274–2282 · Clinton et al. (2010) PE&RS 76:289–299 ·
Ronneberger et al. (2015) MICCAI · Waldner & Diakogiannis (2020) RSE 245:111741 · Kerner et al. (2025) AAAI (Fields of The World).
