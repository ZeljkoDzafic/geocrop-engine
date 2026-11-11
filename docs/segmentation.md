# Parcel segmentation

When reliable parcel geometries (LPIS) exist, use them — segmentation is for areas without boundaries, for
checking boundary changes, and for research comparison.

## Path A — OBIA (CPU)
`geocrop segment-obia --raster stack.tif --bands P3_B04 P3_B08 P3_B11 P3_NDVI --out segments.gpkg`
* SLIC superpixels on 2–98 % stretched bands, per tile (`--tile-size`, default 1024 px).
* `--segments-per-km2` controls the expected object density (≈ 1 / mean field size); `--compactness` the shape regularity.
* Objects smaller than 9 px are dropped. Segments cut by a tile border are merged across the seam by default
  (see [seam reconciliation](#hardening-in-09); `--no-merge-seams` disables it); `tile`, `parts` and
  `seam_merged` support seam QA.

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
  field-boundary models can be used instead: see [Fields of The World](field-boundaries.md) (`geocrop ftw-boundaries`).

## Segment-density sweep
`geocrop segmentation-sweep --raster stack.tif --out sweep/ [--densities 100 300 500 1000]` runs `segment-obia` at
several densities and writes `segment_sweep.csv`, `segment_sweep.md` (one column per density) and
`segment_sweep_manifest.json` (parameters, SHA-256 of every input, engine version; SLIC is deterministic, so there is
no seed). Segment files are kept as `segments_d<density>.gpkg` (`--discard-segments` removes them).

**What this is.** Segments are used for *segmentation-assisted post-classification refinement*: pixel classes or
probabilities of an existing classifier are aggregated per object (for example P3 of
[crop-presence](crop-presence.md)). It is not an independent GEOBIA classification: no object features are computed and
no classifier is trained on objects. An independent GEOBIA random forest is a possible future feature and is not
implemented here.

Per density the table reports the number of segments, the realised density, area quantiles in ha (min, p10, median, p90,
max), the share of segments below `--min-ha` (default 0.03 ha) and the runtime. Optional blocks:

* `--parcels`: median parcel area and its ratio to the median segment, share of parcels holding at least
  `--min-segments` segments (a segment belongs to the parcel containing its representative point), mean segments per
  parcel and *boundary mismatch*: area share of segments that overlap two or more parcels by at least 5 % of their
  area each (area-weighted, among segments touching any parcel).
* `--reference POLYGONS --label-field FIELD`: object purity, the share of the majority reference class in the covered
  part of each segment (area-weighted mean, median, share of segments below 0.8). Low purity indicates
  under-segmentation. Reference polygons must not overlap each other.
* `--probabilities probabilities.tif --parcels ... --declared-field ...`: `crop-presence` is run once on connected
  same-class pixels (baseline) and once per density with that density's segments. The table lists status changes,
  supported gained/lost, shares supported and the mean supported share of the agricultural part (P3), together with
  the P1 match share and mean P2 soft vote (neither depends on segments, so they are the fixed reference); read finer
  segments as beneficial only if they change P3 where the evidence supports it, not merely because more parcels become
  supported. `--declared-area-field`, `--crop-mapping` and `--support-prob` are passed to `crop-presence`.

**Caveats.** `segments_per_km2` is a target for SLIC initial centres per tile, not a count: segments below
`min_area_pixels` are dropped, SLIC does not use every centre on homogeneous areas and seam merging joins pieces, so the
realised density (reported) differs from the requested one. Runtime grows with density. All values are a recommendation
for human review, not a validated conclusion; purity and mismatch depend on the quality of the reference polygons.

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

## Hardening in 0.9
* **Leakage-free U-Net validation.** `unet_train` splits validation by patch *groups* (spatial blocks, sites or
  parcels); a random patch split is refused unless `allow_random_split=True`. Per-channel mean/std are computed on
  training patches only. `geocrop segment-patches --raster ... --mask ... --block-m 2000` cuts patches and writes
  the matching `_groups.npy`; pass it with `segment-unet-train --groups`.
* **Seam reconciliation.** OBIA normalises all tiles with one global 2–98 % stretch and merges segments from
  different tiles that share at least `min_shared_px` pixels of a tile seam and differ by less than
  `merge_distance` in mean stretched spectra (transitive union-find). Output ids are global and stable
  (north→south, west→east); `tile`, `parts` and `seam_merged` support QA. Disable with `--no-merge-seams`.
* **Benchmark.** `geocrop benchmark-segmentation --raster ... --reference parcels.gpkg [--unet-model ...]`
  scores OBIA with and without seam merging and U-Net + watershed against the same reference parcels.
