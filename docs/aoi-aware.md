# AOI-aware processing

The reference grid is the **bounding box** of the AOI. For a compact AOI that costs little; for an irregular one
(a river valley, a municipality group) most grid pixels lie outside the AOI. `geocrop preflight` on the Lijevče
polje study found a grid of 26.44 Mpx (~2644 km²) for an AOI of 719.4 km², 3.7 times more than needed. Without
special handling every stage imports, composites, featurises, classifies and writes those pixels, so network,
compute and disk all scale with the bounding box.

AOI-aware processing removes that cost **without changing any result inside the AOI**. Products keep the full grid
extent and transform, so everything stays aligned and existing consumers work; the work outside the AOI is simply
not done, and pixels outside the AOI are no-data (never silently classified).

```bash
geocrop grid --aoi configs/studies/lijevce_polje/aoi.geojson --crs EPSG:32633 --out data/reference/grid_10m.tif
#  writes grid_10m.tif, grid_10m_aoi.tif (the AOI mask) and grid_10m_aoi.json (the block index)
geocrop stac-import --reference data/reference/grid_10m.tif ...      # finds grid_10m_aoi.tif next to the grid
geocrop run --config configs/studies/lijevce_polje/study_2023.yaml   # idem, via grid.reference
```

With no mask configured or found, **nothing changes**: behaviour and outputs are identical to a build without this
feature (the pixel data of a full pipeline run were compared with the previous release, see below).

## The AOI mask and the block index

`geocrop grid` (unless `--no-aoi-mask`) writes next to the grid, for `--out data/grid_10m.tif`:

| File | Content |
|---|---|
| `grid_10m_aoi.tif` | the **AOI mask**: uint8, 1 inside the AOI, 0 outside, on exactly the grid (same CRS, transform, shape). With `--buffer-m` it is the buffered footprint, like the grid. |
| `grid_10m_aoi.json` | the **block index**: AOI pixels, share of the grid, area, and the `[row, col]` of the 512 px blocks that contain at least one AOI pixel (grown by `--halo-px`, default 0), with the pixels inside those blocks. |

The pipeline derives its blocks from the mask itself (a block is looked up in the compressed mask in
milliseconds) with the halo it needs, so the JSON is an inspection/QA record, not a second source of truth.
The mask is also created by `quicklook --aoi` and by `geocrop.raster.make_grid(..., aoi_mask=True)`; library callers
of `make_grid` get no mask unless they ask for it.

## Configuration

| Key | Default | Meaning |
|---|---|---|
| `grid.aoi_mask` | `null` | path of the mask raster. `null`: use `<grid.reference stem>_aoi.tif` next to the reference grid when it exists (auto-discovery), otherwise no mask. `false`: never use a mask (switches discovery off). |
| `grid.aoi` | `null` | optional AOI vector (GeoJSON/GPKG). Used only when no mask file is found: it is rasterised onto the reference grid into `<paths.output>/aoi/aoi_mask.tif`. |
| `grid.aoi_halo_px` | `0` | pixels around the AOI that intermediate products also hold. The probability-smoothing radius (`predict.smoothing.window // 2`) is added automatically: the effective halo is the larger of the two. |

All three keys are fingerprint-neutral while unset, so existing studies keep their configuration fingerprint. Stage
fingerprints (composite, features, predict) include the mask **content** (SHA-256) and the effective halo only
when a mask is used; changing the mask or the halo re-runs exactly those stages and everything downstream. Scene
folders imported with a mask carry the mask in their grid id (`scene.json`: `grid_id`, `aoi_mask`, `aoi_halo_px`),
so a masked import is never reused by an unmasked run and vice versa. Command-line options on the commands that
read or compute per block (`stac-import`, `s2-import`, `s2-composite`, `s1-composite`, `ts-features`, `features`,
`predict`, `segment-obia`, `segment-unet-infer`): `--aoi-mask PATH`, `--no-aoi-mask`, `--aoi-halo-px N`; where the
command takes a `--reference` grid the mask next to it is found automatically.

## What is skipped, stage by stage

An **active block** is a block of the grid (512 px; 256 px for time series; 64 px, 1024 px... for other block sizes)
whose window, grown by the halo, contains an AOI pixel. Stages compute active blocks completely and **write
nothing for the others**: the files are created with GDAL `SPARSE_OK`, so unwritten tiles take no disk space and
read back as no-data.

| Stage | With a mask |
|---|---|
| `stac-import` (S2 L2A, `stac_import`) | items whose footprint misses every active block are dropped before any request; for the others the SCL is read only for active blocks, then bands only for the blocks where the SCL shows data, one COG window per block (no request for tiles or blocks outside). Clear fraction is measured over the AOI pixels. |
| Sentinel-1 RTC import (`s1_stac`) | same: footprint pre-filter, COG windows of active blocks only; coverage over the AOI pixels. |
| SAFE import (`s2-import --reference`) | JP2 windows are decoded for active blocks only. (The whole-scene shadow projection is unchanged.) |
| S2/S1 composites | active blocks only; others stay no-data (`valid_count` 0). |
| feature stack, time-series stack, merge, embedding attach | active blocks only. |
| `predict` | AOI pixels only are classified; probabilities, uncertainty, classes, the AOA map and the unsmoothed probabilities are no-data outside the AOI. Blocks without AOI pixels are not read. The AOA map (nearest-neighbour search, the most expensive step) is computed for AOI pixels only. |
| `segment-obia`, `segment-unet-infer` | tiles without AOI pixels are skipped; see the segmentation note below. |
| samples, parcels | unchanged (they read rasters at polygons; see the limitations for parcels crossing the AOI border). |
| `quicklook` | the bounding-box quicklook is unchanged; `--aoi` uses the mask. |

### Intermediate and final products

Two granularities keep both the savings and the guarantee:

* **Intermediate products** (scenes, composites, feature stacks): every pixel of an active block is computed,
  including pixels outside the AOI. They are needed as context: the smoothed probability at an AOI pixel reads
  neighbours up to the smoothing radius, some of which lie outside the AOI and in another block. Blocks that are
  neither in the AOI nor within the halo are no-data.
* **Final products** (classes, probabilities, uncertainty, AOA map, unsmoothed probabilities): masked to the AOI
  pixels. `prediction_manifest.json` records `aoi` (mask name, halo) when a mask was used.

`predict` refuses an `AoiMask` whose halo is smaller than the smoothing radius, and the pipeline/CLI always build the
halo from the configuration, so stages agree. If you call stages yourself, build the feature stack with the same
halo that you give `predict`. A negative-control test shows that a stack built without the halo changes the smoothed
probabilities at the AOI border.

## Guarantees

* **No mask, no change.** Without a mask every code path, fingerprint and output is as before: a full pipeline run
  (composite → features → samples → train → predict → parcels) on synthetic data was compared with the preceding
  release; all rasters (composites, feature stack, classes, probabilities, uncertainty, AOA) have identical pixel
  data (only the embedded model checksum differs, because `model.joblib` is not byte-reproducible between any two
  training runs).
* **Inside the AOI the masked run equals the unmasked run** for scenes, composites, features, time series,
  probabilities (also with smoothing at AOI edges and block borders), classes, uncertainty and the AOA map, within
  1e-6 (observed: bit-identical on the synthetic study, `tests/test_aoi_aware.py`); parcels inside the AOI are
  identical. Intermediate products also equal the unmasked run in every active block.
* **Outside the AOI**: final products are no-data, intermediate products are no-data outside the active blocks.
* Blocks are skipped, never approximated: a pixel is either computed exactly as before or not at all.

## Resource estimate and disk guard

With a mask, `geocrop preflight` and the pipeline disk guard count the pixels of the **active blocks** ("work
pixels": what the stages read, compute and store) instead of the bounding box, and report all three numbers
(grid, AOI, work pixels) plus the check `grid.aoi_share` ("Info: AOI share of grid X %"). The area-of-applicability
runtime term uses the AOI pixels. When the reference grid was made by `geocrop grid` but no mask file is found,
the same info line reports the AOI share read from the grid values and tells you to create the mask.

## Limitations

* Scene admission with a mask: `min_clear_fraction` (S2) and `min_coverage` (S1) are evaluated over the **AOI
  pixels**, not the bounding box. This is the more meaningful measure for the study area, but a scene near the
  threshold can be admitted or skipped differently from a bounding-box run. Given the same imported scenes, results are equal.
* Segmentation (`segment-obia`) with a mask equals the unmasked run **on the stack with its non-AOI pixels set to
  no-data**, not the bounding-box run: the global percentile stretch and the SLIC superpixels depend on the pixels
  they see, so they are computed from AOI pixels only and no segment extends outside the AOI. Tiles are skipped
  without AOI pixels; seam reconciliation compares segments of adjacent processed tiles exactly as before.
* Parcels that cross the AOI border see no-data for their outside part, so their statistics use the inside part
  only; keep parcel layers inside the AOI (the Lijevče polje AOI is built from the municipalities).
* The domain-shift check (`uncertainty.shift_detection`) draws random 64 px windows from the whole grid when no
  parcel layer exists; with a mask many of them fall into no-data blocks, so its sample (and report) differs from
  an unmasked run. It samples inside the parcels when they exist.
* The foundation-model `embeddings` stage and the Sentinel-1 SNAP import are not AOI-aware (they run on the whole
  grid; attached embeddings are copied for active blocks only). The SAFE shadow projection still reads the whole
  scene.
* The quality JSON of time-series stacks counts the pixels of the computed blocks only.
* Reprojection: when the source COG is in another CRS than the grid, GDAL approximates the coordinate transformation
  per warp window; windows of different size can differ by a sub-pixel interpolation error. Items in the grid's
  own CRS and 10 m lattice (the usual case) are copied exactly.
* Sparse GeoTIFFs are standard (read by GDAL, QGIS, rasterio); tools that ignore the missing-tile convention
  would read zeros instead of no-data in the unwritten blocks.

## Measured effect

See [performance](performance.md#aoi-aware-import-measured-on-the-lijevče-polje-aoi) for the measurement of the
Sentinel-2 import on the real AOI (`tools/aoi_benchmark.py`).
