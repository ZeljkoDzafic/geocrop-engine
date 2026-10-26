# Foundation-model embeddings (optional feature group `emb`)

GeoCrop can add the per-pixel embeddings of a pretrained Earth-observation foundation model to its handcrafted
feature stack. They form the feature group `emb`, so you can compare handcrafted features, embeddings, and both
together under the same spatial cross-validation. Three sources are supported:

| source | what | dims | where it is computed | licence |
|---|---|---|---|---|
| `alphaearth` | AlphaEarth Foundations Satellite Embedding V1, annual | 64 | pre-computed by Google, read as COG windows | CC-BY-4.0 (attribution required) |
| `tessera` | TESSERA v1 annual embeddings | 128 | pre-computed by the University of Cambridge, read as `.npy` row ranges | see [TESSERA licence](#tessera) |
| `presto` | Presto encoder (Tseng et al., 2023) | 128 | **locally on the CPU**, from GeoCrop's own monthly S2/S1 composites | MIT (code and weights) |

Embeddings are optional. With `features.embeddings` unset, nothing changes: no network access and the same
stack, manifest, and default ablation table.

## Output contract

```
<features dir>/embeddings/<source>_<year>.tif            float32, nodata -9999, on the study grid
<features dir>/embeddings/<source>_<year>.features.json  feature manifest, group "emb"
<features dir>/embeddings/presto_<year>_parcels.csv      Presto only, with --parcels: parcel_id, n_pixels, emb_00..
```

* There is one band per embedding dimension, with descriptions `emb_00`, `emb_01`, ... (`emb_127` for 128-d). The
  band descriptions are the feature names, as in every GeoCrop stack.
* GeoTIFF tags record `embedding_source`, `year`, `licence`, `attribution` and `dequantisation`, plus the
  provenance: the tile URLs and mirror for pre-computed sources, or the checkpoint URL and SHA-256 and the start
  month for Presto.
* Missing values (masked AlphaEarth pixels, TESSERA gaps, pixels with no S1/S2 observation) are written as
  nodata. They are never zero-filled.
* The group `emb` counts as a season-wide group. Like `seasonal`, it is dropped when an experiment restricts
  `periods`.
* A stack can hold one embedding source at a time, because two sources would produce the same feature names.

## Configuration

```yaml
features:
  embeddings:
    source: alphaearth          # alphaearth | tessera | presto
    year: 2023
    # dir: data/features/embeddings   (default: <paths.features parent>/embeddings)
    # resampling: nearest              nearest | bilinear (pre-computed sources)
    # mirror: source.coop              AlphaEarth: source.coop | gcs
    # index: null                      AlphaEarth: local or remote aef_index.parquet
    # monthly: null                    Presto: monthly composites root (default <dir>/monthly)
    # start: "2023-01"                 Presto: first month (default <year>-01)
    # months: 12                       Presto: 1..24 monthly timesteps
    # checkpoint: null                 Presto: local checkpoint instead of the pinned download
    # parcels: false                   Presto: also write parcel-level embeddings for paths.parcels
    # attach: true                     geocrop features --config appends the raster to the stack
```

## Commands

```bash
# 1. embeddings on the study grid (grid.reference, or --reference)
geocrop embeddings --config study.yaml                         # source/year from the config
geocrop embeddings --config study.yaml --source tessera --year 2023
geocrop embeddings --config study.yaml --source presto --make-monthly --parcels data/reference/parcels.gpkg

# 2. handcrafted stack + emb group (features.embeddings.attach: true)
geocrop features --composites data/processed/composites --out data/features/stack.tif --config study.yaml
#    or, for any stack:  geocrop merge-stacks --stacks stack.tif embeddings/alphaearth_2023.tif --out stack_emb.tif

# 3. samples and ablation: handcrafted vs embeddings vs both on identical folds
geocrop samples ... && geocrop ablation --samples samples.npz --features data/features/stack.tif --out exp --config study.yaml
```

If the stack contains an `emb` group and the config defines no `experiments:`, `ablation` adds three rows to
the default design E0–E4:

* `E5_handcrafted_all`: every handcrafted group;
* `E6_embeddings_only`: the `emb` group only;
* `E7_handcrafted_plus_embeddings`: both.

McNemar's test compares each row with the first experiment, as before. To compare against another baseline,
define your own `experiments:` with that baseline listed first, for example `{groups: [emb]}`.

`geocrop run` (the resumable pipeline) does not attach embeddings yet, because this change does not modify
`pipeline.py`. Use `geocrop embeddings` followed by `geocrop features --config`, or `merge-stacks`, as shown
above.

## Sources in detail

### AlphaEarth Foundations Satellite Embedding V1

* **Access** (verified 2026-10-26). Public COGs, readable without credentials:
  * Source Cooperative (default): `https://data.source.coop/tge-labs/aef/v1/annual/<year>/<utm zone>/<id>.tiff`
  * Google Cloud Storage (`mirror: gcs`):
    `https://storage.googleapis.com/alphaearth_foundations/satellite_embedding/v1/annual/...`. The bucket is
    "provider pays" according to the official README (as of July 2026).

  Tiles are found through the official index `aef_index.parquet` (about 78 MB, 302,466 tiles for 2017–2025).
  It is downloaded once into `$GEOCROP_CACHE/alphaearth/` (default `~/.cache/geocrop`) and needs `pyarrow`
  (`pip install -e ".[parquet]"`).
* **Format.** Each tile is 8192 × 8192 px, 64 bands `A00`–`A63`, int8, 10 m, UTM. The COGs are **bottom-up**:
  the y pixel size is positive and the origin is the south-west corner. GeoCrop reads only the window that
  covers the grid, flips it to north-up, and reprojects with nearest neighbour by default. When grid and tile
  share the UTM zone and the 10 m lattice, this is an exact copy. If several tiles cover the grid, tiles in the
  grid's CRS are used first.
* **Dequantisation** (official README): `((q / 127.5) ** 2) * sign(q)`; `q = -128` is masked. The result
  lies in [-1, 1], and each pixel's 64-vector has unit length.
* **Licence and attribution.** CC-BY-4.0. Attribution required in any product or publication: *"The AlphaEarth
  Foundations Satellite Embedding dataset is produced by Google and Google DeepMind."* GeoCrop writes this
  text into the GeoTIFF tags and the manifest. Derived maps can be redistributed with attribution.

### TESSERA

* **Access** (verified 2026-10-26). Public HTTPS on Source Cooperative (`tessera/tessera`), dataset `v1`:
  * `npy/v1/<year>/grid_<lon>_<lat>/grid_<lon>_<lat>.npy`: int8 (H, W, 128)
  * `..._scales.npy`: float32 (H, W)
  * `landmasks/v1/grid_<lon>_<lat>.tiff`: CRS and transform of the tile (UTM)

  The paths and the dequantisation follow the official `geotessera` library (MIT), which GeoCrop does not need
  at runtime. Tiles sit on a 0.1° grid and are named by their centre (`grid_19.15_44.75`).
* **Windowed reads.** The `.npy` files are uncompressed and C-ordered. GeoCrop parses the header and fetches
  only the rows that cover the grid with HTTP range requests. Landmask GeoTIFFs (about 16 KB each) are cached.
  Neighbouring tiles overlap in UTM. Each pixel is taken from the tile whose 0.1° cell contains the pixel
  centre, and other tiles only fill gaps.
* **Dequantisation.** `q.astype(float32) * scales` (2-D scales are broadcast), with non-finite scales masked.
  This is identical to `geotessera.dequantize_embedding`.
* **Validation.** On the Semberija grid below, every pixel whose centre lies in tile `grid_19.15_44.75` is
  identical to `geotessera.GeoTessera(dataset_version="v1").fetch_embedding(19.15, 44.75, 2023)` (maximum
  absolute difference 0.0 over all 128 bands). The 2,445 pixels west of 19.10° E come from the neighbouring
  tile `grid_19.05_44.75`.
* <a id="tessera"></a>**Licence: conflicting statements, so GeoCrop treats it conservatively.** The producer's
  *Terms of Use* (`aup.md` in `ucam-eo/tessera`, last updated 2026-02-25) state that "TESSERA data and
  embeddings are made available under the Creative Commons 0 International License (CC-0)". The same file
  links to the CC-BY-4.0 legal text and later refers to "the CC-BY 4.0 license under which data is released".
  The AWS Open Data registry entry for the dClimate icechunk mirror lists "CC-BY-SA". Third-party summaries
  also mention the Copernicus terms. Every candidate licence allows reading the data at runtime as done here,
  and GeoCrop never bundles TESSERA data. Before redistributing derived products, credit TESSERA
  (Feng et al., 2025, arXiv:2506.20380, University of Cambridge) and follow the most restrictive reading
  (CC-BY-SA) until the producer resolves the inconsistency. The model weights are CC0 and the `geotessera`
  library is MIT. The ethical guidelines in the terms of use are advisory.

### Presto (computed on the VM)

* **Model.** The Presto encoder (Tseng et al., 2023, *Lightweight, Pre-trained Transformers for Remote Sensing
  Timeseries*, arXiv:2304.14065). The code is MIT. `geocrop/presto_encoder.py` adapts the official
  `single_file_presto.py` and keeps its licence notice.
* **Checkpoint.** The official `data/default_model.pt` is pinned by commit URL and SHA-256 and downloaded on
  first use into `$GEOCROP_CACHE/presto/`. It is never committed.
  * URL: `https://raw.githubusercontent.com/nasaharvest/presto/11e207a668a34336ced1d8e492a1bd5849b96c4a/data/default_model.pt`
  * SHA-256: `559c9d76f525bdf2dd3774da8b1e7f1510769b73baacac12e9e4d8d5c67fbe65` (3.3 MB)

  A file that fails the checksum is rejected.
* **Equivalence.** With this checkpoint, GeoCrop's encoder reproduces the official `Encoder(eval_task=True)`
  output exactly (maximum absolute difference 0.0 on 64 random pixels × 12 months with missing S2 and S1
  timesteps, start month April).
* **Input contract.** It follows the official `presto/dataops/pipelines/s1_s2_era5_srtm.py`:

  | Presto band (order) | GeoCrop source | normalisation |
  |---|---|---|
  | VV, VH | monthly S1 composite (linear) → dB | `(dB + 25) / 25` |
  | B2 B3 B4 B5 B6 B7 B8 B8A B11 B12 | monthly S2 composite (reflectance 0–1) | reflectance = DN / 10000 |
  | temperature_2m, total_precipitation | not available on-VM | **masked** |
  | elevation, slope | not available on-VM | **masked** |
  | NDVI | from normalised B8/B4 (Presto rule: 0 where B8+B4 ≤ 0) | none |
  | Dynamic World class | not used | class 9 = missing (masked) |

  B1, B9 and B10 are not inputs to the normalised model. Timesteps are calendar months: `month` is the 0-based
  month of the first timestep (0 = January), and later steps follow consecutively. Presto was pre-trained on
  12 months, and its positional table allows 24. Masking happens per token group, as in the official code: a
  month without a valid S2 composite masks all S2 groups and NDVI for that month, and a month without S1 masks
  the S1 token. Presto drops masked tokens, and the official encoder requires the same number of masked tokens
  per batch, so GeoCrop groups pixels by mask pattern. Pixels with no S1 or S2 observation are written as
  nodata. Lat/lon is the WGS84 pixel centre, or the parcel's representative point.
* **Monthly composites.** `geocrop embeddings --make-monthly` builds `<dir>/monthly/<YYYY-MM>/` from
  `paths.s2_scenes` and `paths.s1_scenes`, using the configured S2 compositing (cloud-masked median by
  default) and the S1 linear mean. You can also prepare any folders that follow the composite contract
  (`B02.tif` … `B12.tif`, `VV.tif`/`VH.tif`, or `ASC_`/`DESC_` pairs, whose orbits are averaged). Months
  without folders are masked.
* **Pixel level.** Processing runs block by block, using memory proportional to block² × months × 17. It uses
  torch on the CPU in inference mode.
* **Parcel level** (`--parcels`). Each parcel is reduced to its mean monthly series: the mean S2 reflectance
  and the mean linear S1 backscatter over the pixels whose centre lies inside the parcel. Presto then runs
  once per parcel. The output CSV can be joined to parcel descriptors for `parcel-model`. This is much cheaper
  than pixel level, but it cannot see heterogeneity within a parcel.
* **Optional dependency.** torch (`pip install -e ".[deep]"`). Without it, `presto` fails with an error that
  names the extra. AlphaEarth and TESSERA need no torch.

## CPU cost and data volume (measured)

The measurements come from the Semberija run below: a grid of 486 × 456 px (≈ 4.9 × 4.6 km, 221,616 pixels) on
a laptop CPU (Apple M5 Pro, 15 cores) over a home internet connection. Expect the cost of Presto at pixel level to
scale linearly: about 2 minutes per 100 km² at 10 m on this CPU.

| source | wall time | transferred | notes |
|---|---|---|---|
| AlphaEarth 2023 | 115 s | index 78 MB once + windows of 64 bands | 64 band-separate COG range reads per block; network-bound |
| TESSERA 2023 | 26 s | ≈ 2 × 456 rows × 817 × 128 B ≈ 95 MB + scales | 2 tiles, only the needed rows |
| Presto 2023 (pixel) | 23.5 s inference (+ 8 s monthly composites) | checkpoint 3.3 MB once | 12 months (11 with S2, April masked; no S1), batch 512, peak RSS 1.3 GB (6.1 GB at batch 4096), ≈ 9,400 px/s |
| Presto 2023 (parcel) | 0.6 s for 528 test cells of 200 × 200 m | – | one sequence per parcel |

## Real extraction: Semberija, 2023

AOI 19.10–19.16 °E, 44.72–44.76 °N; grid EPSG:32634, 10 m, 486 × 456 px. Reproduce with:

```bash
geocrop grid --aoi aoi.geojson --crs EPSG:32634 --resolution 10 --out grid.tif
geocrop embeddings --config cfg.yaml                       # features.embeddings: {source: alphaearth, year: 2023}
geocrop embeddings --config cfg.yaml --source tessera
geocrop stac-import --reference grid.tif --start 2023-01-01 --end 2023-12-31 --out s2_scenes --max-cloud 60 --min-clear 0.3
geocrop stac-import --reference grid.tif --start 2023-01-01 --end 2023-05-31 --out s2_scenes --max-cloud 95 --min-clear 0.1
geocrop embeddings --config cfg_presto.yaml --make-monthly   # {source: presto, year: 2023}; paths.s2_scenes: s2_scenes
```

The Presto run used 29 Sentinel-2 L2A scenes from Earth Search (tile 34TCQ), with no scene available for
April 2023, and no Sentinel-1 data, so the S1 token was masked.

| source | bands | valid fraction | min | max | mean | mean per-band std | vector norm (min / mean / max) |
|---|---|---|---|---|---|---|---|
| AlphaEarth 2023 | 64 | 1.000 | -0.434 | 0.404 | -0.004 | 0.060 | 0.992 / 1.000 / 1.008 |
| TESSERA 2023 | 128 | 1.000 | -10.78 | 12.73 | 0.041 | 1.561 | 23.8 / 28.7 / 41.0 |
| Presto 2023 (S2 only) | 128 | 1.000 | -5.57 | 5.32 | -0.043 | 0.320 | 11.5 / 12.9 / 13.4 |

The vector norms of about 1 confirm that the AlphaEarth dequantisation is correct, since its embeddings are
unit vectors. TESSERA embeddings are not normalised. The extraction step has no accuracy figures: an
embedding-versus-handcrafted comparison needs reference labels and the frozen protocol (gap G1), and has not
been run yet.

## Limitations

* **Annual and fixed.** AlphaEarth and TESSERA give one vector per pixel and year. They cannot be used for
  in-season monitoring of the current year until the producer publishes that year: AlphaEarth covers
  2017–2025, and TESSERA coverage varies by year and tile. Presto works for any period that has composites.
* **Black-box inputs.** The pre-computed embeddings come from the producers' own Sentinel-1/2 (and, for
  AlphaEarth, other) inputs and cloud handling. They do not reflect GeoCrop's masking, and their exact input
  periods are fixed by the producer.
* **Presto without ERA5, SRTM and Dynamic World.** These token groups are masked. Presto was pre-trained with
  random group masking, so this is in-distribution, but embeddings computed without these inputs are not
  identical to those of a full WorldCereal-style pipeline. S1 composites need GeoCrop's SNAP route (or another
  source of linear VV/VH). When S1 is absent, the S1 token is masked.
* **Resampling.** The default is nearest neighbour, which keeps AlphaEarth vectors unit-length. Grids that are
  not aligned to the source's 10 m lattice therefore shift by up to half a pixel.
* **Version drift.** AlphaEarth, the TESSERA v1 `.npy` tiles (deprecated upstream in favour of Zarr and
  Icechunk) and the Presto checkpoint can change upstream. GeoCrop pins the Presto weights by SHA-256 and
  records tile URLs in the output tags. Re-extract when you change versions.
* **Pipeline.** `geocrop run` does not yet attach embeddings (see [Commands](#commands)).
* **Licences.** You are responsible for attribution when publishing maps derived from AlphaEarth (CC-BY-4.0)
  and TESSERA (see the licence notes above).
