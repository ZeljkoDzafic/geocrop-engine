# Foundation-model embeddings (optional feature group `emb`)

GeoCrop can add the per-pixel embeddings of a pretrained Earth-observation foundation model to its handcrafted
feature stack. They form the feature group `emb`, so you can compare handcrafted features, embeddings, and both
together under the same spatial cross-validation. Every source is a **feature provider** behind one adapter
interface ([below](#feature-provider-interface)); five are built in:

| source | what | dims | where it is computed | licence (code / weights or data) |
|---|---|---|---|---|
| `alphaearth` | AlphaEarth Foundations Satellite Embedding V1, annual | 64 | pre-computed by Google, read as COG windows | – / CC-BY-4.0 (attribution required) |
| `tessera` | TESSERA v1 annual embeddings | 128 | pre-computed by the University of Cambridge, read as `.npy` row ranges | – / see [TESSERA licence](#tessera) |
| `presto` | Presto encoder (Tseng et al., 2023) | 128 | **locally on the CPU**, from GeoCrop's own monthly S2/S1 composites | MIT / MIT |
| `galileo` | Galileo-nano encoder (Tseng et al., 2025) | 128 | **locally on the CPU**, from GeoCrop's own monthly S2/S1 composites | MIT / MIT |
| `anysat` | AnySat base encoder (Astruc et al., 2025) | 768 | **locally on the CPU**, from GeoCrop's own monthly S2 composites | MIT / MIT |

`geocrop embeddings --list` prints every registered provider (built-ins and plugins) with its contract.

Embeddings are optional. With `features.embeddings` unset, nothing changes: no network access and the same
stack, manifest, and default ablation table.

## Output contract

```
<features dir>/embeddings/<source>_<year>.tif            float32, nodata -9999, on the study grid
<features dir>/embeddings/<source>_<year>.features.json  feature manifest, group "emb"
<features dir>/embeddings/<source>_<year>_parcels.csv   with --parcels: parcel_id, n_pixels, one column per feature
```

* There is one band per embedding dimension. The band descriptions are the feature names, as in every GeoCrop
  stack:
  * **namespaced** `emb_<source>_00`, `emb_<source>_01`, ... (`emb_anysat_767` for 768-d). This is the default
    for `galileo`, `anysat`, plugins, and whenever several sources are requested together, so any number of
    providers fit in one stack;
  * **legacy** `emb_00`, `emb_01`, ... (`emb_127` for 128-d). `alphaearth`, `tessera` and `presto` still write
    these by default, so existing configurations, stacks and trained models keep working unchanged.
    `namespace: true` (`--namespace`) switches them to namespaced names, `namespace: false` (`--no-namespace`)
    forces legacy names for a single source.
* Old stacks stay readable. Attaching a second legacy source to a stack that already holds `emb_00` ... renames
  the incoming features to `emb_<source>_NN` (the raster on disk is not changed), and
  `geocrop.embeddings.embedding_features(manifest, provider)` selects one provider's features in either naming
  scheme (legacy names are mapped back to their provider through the manifest's `sensor`).
* Parcel tables (`--parcels`): monthly providers encode one parcel-mean series per parcel; pre-computed
  providers average the pixel embeddings whose centre lies in the parcel.
* GeoTIFF tags record `embedding_source`, `year`, `licence`, `attribution` and `dequantisation`, plus the
  provenance: the tile URLs and mirror for pre-computed sources, or the checkpoint URL and SHA-256 and the start
  month for Presto.
* Missing values (masked AlphaEarth pixels, TESSERA gaps, pixels with no S1/S2 observation) are written as
  nodata. They are never zero-filled.
* The group `emb` counts as a season-wide group. Like `seasonal`, it is dropped when an experiment restricts
  `periods`.
* Every provider's output is the group `emb`; with several providers in a stack, `E6`/`E7` below use all of
  them. Compare providers with your own `experiments:` and `features:` lists (`embedding_features`).

## Configuration

```yaml
features:
  embeddings:
    source: alphaearth          # alphaearth | tessera | presto | galileo | anysat | <plugin>, or a list
    year: 2023
    # dir: data/features/embeddings   (default: <paths.features parent>/embeddings)
    # resampling: nearest              nearest | bilinear (pre-computed sources)
    # mirror: source.coop              AlphaEarth: source.coop | gcs
    # index: null                      AlphaEarth: local or remote aef_index.parquet
    # monthly: null                    Presto: monthly composites root (default <dir>/monthly)
    # start: "2023-01"                 Presto: first month (default <year>-01)
    # months: 12                       Presto: 1..24 monthly timesteps
    # checkpoint: null                 Presto: local checkpoint instead of the pinned download
    # parcels: false                   also write parcel-level embeddings for paths.parcels
    # attach: true                     geocrop features --config appends the raster(s) to the stack
    # namespace: null                  true: emb_<source>_NN, false: legacy emb_NN (null: see Output contract)
    # allow_license: []                licence ids acknowledged for non-permissive providers
```

`monthly`, `start`, `months` and `checkpoint` apply to every monthly provider (Presto, Galileo, AnySat). A list
of sources (`source: [presto, galileo, anysat]`) produces one raster per source with namespaced names; all are
attached to the stack. In `geocrop run` the embeddings stage currently supports a single AlphaEarth, TESSERA or
Presto source only (see [Limitations](#limitations)).

## Commands

```bash
# 1. embeddings on the study grid (grid.reference, or --reference)
geocrop embeddings --config study.yaml                         # source/year from the config
geocrop embeddings --config study.yaml --source tessera --year 2023
geocrop embeddings --config study.yaml --source presto --make-monthly --parcels data/reference/parcels.gpkg
geocrop embeddings --config study.yaml --source galileo anysat    # one raster each, namespaced names
geocrop embeddings --list                                         # registered providers and their contracts

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

`geocrop run` has an `embeddings` stage between `composite` and `features`. It runs only when
`features.embeddings` is configured; only then are the network adapters (AlphaEarth/TESSERA reads, the pinned
Presto checkpoint download) imported. For Presto without `monthly`, the stage first builds the monthly
composites from `paths.s2_scenes`/`paths.s1_scenes` into `<dir>/monthly`. With `attach: true` (default) the
features stage appends the raster to the stack, and the raster's content becomes part of the features
fingerprint, so a re-extracted embedding re-runs features and everything after it. Without
`features.embeddings`, neither the stage nor any fingerprint changes.

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
* **Optional dependency.** torch (`pip install -e ".[deep]"`). Without it, `presto`, `galileo` and `anysat` fail
  with an error that names the extra. AlphaEarth and TESSERA need no torch.

### Galileo (computed on the VM)

* **Model.** Galileo-nano (Tseng et al., 2025, *Galileo: Learning Global & Local Features of Many Remote
  Sensing Modalities*, arXiv:2502.09356), the smallest released model (1.04 M parameters, 128-d, depth 4),
  chosen for CPU use. Code MIT (`nasaharvest/galileo`, "Copyright (c) 2024 Presto Authors"); the weights are
  MIT according to the Hugging Face model card (`nasaharvest/galileo`, `license: mit`).
  `geocrop/galileo_encoder.py` ports the encoder of the official `single_file_galileo.py` (commit
  `0f0b5b95ac81acef4b74cf4686dc877202f4541b`) without einops and keeps the MIT notice; parameter names are
  unchanged, so the official `encoder.pt` loads with `strict=True`.
* **Weights.** Pinned by commit URL and SHA-256, downloaded on first use into `$GEOCROP_CACHE/galileo/`:
  * `https://raw.githubusercontent.com/nasaharvest/galileo/0f0b5b95ac81acef4b74cf4686dc877202f4541b/data/models/nano/config.json`,
    SHA-256 `8ec17cb83cb78d987b4b920b9c1c35aa245ec0f80e1abf0c22d6f77d828f431c`
  * `.../data/models/nano/encoder.pt`, SHA-256 `eab6acacf527886a508a555bc24efe4b4a2cf91b162e84cd1c4a1807e942180c`
    (4.2 MB; byte-identical to the Hugging Face copy at revision `f039dd5d`)
* **Equivalence.** Against the official `single_file_galileo.Encoder.load_from_folder(nano)`: the pixel
  embeddings differ by at most 1.2e-7 (64 random pixels × 12 months, start month April, different missing S1/S2
  months per pixel, S1-only and S2-only pixels); the spatial `forward` (4 × 4 px, patch size 2) and its pooled
  output differ by 0.0; GeoCrop's normaliser matches the official training normaliser to 6e-8 (float32
  rounding).
* **Input contract.** It follows the official training normaliser and the official CropHarvest evaluation
  (`src/eval/cropharvest/cropharvest_eval.py`), which feeds pixel time series:

  | Galileo band (order) | GeoCrop source | normalisation |
  |---|---|---|
  | VV, VH | monthly S1 composite (linear) → dB | `(x − (mean − 2·std)) / (4·std)` with the pre-training statistics (`config/normalization.json`) |
  | B2 B3 B4 B5 B6 B7 B8 B8A B11 B12 | monthly S2 composite, reflectance × 10000 (DN) | same |
  | NDVI | `(B8 − B4) / (B8 + B4)`, 0 where the sum is not positive | none |
  | location | WGS84 pixel centre (or parcel representative point) → unit-sphere x, y, z | none |
  | ERA5, TerraClimate, VIIRS, SRTM, Dynamic World, WorldCereal, LandScan | not available on-VM | **masked** |

  Each pixel is one 1 × 1 patch (`patch_size = 1`), months are encoded as month of year (0 = January) for up to
  24 consecutive timesteps, and the embedding is the mean of all unmasked output tokens (`average_tokens`), as
  in the official evaluation. Masking is per token group: a month without a valid S2 composite masks the S2
  groups and NDVI, a month without S1 masks the S1 token. Galileo handles different masks within one batch, so
  no grouping is needed. Note: the official helper `construct_galileo_input(normalize=True)` skips the
  standard-deviation normalisation (its statistics dictionary has string keys); GeoCrop follows the
  training/evaluation path, which applies it.

### AnySat (computed on the VM)

* **Model.** AnySat (Astruc et al., 2025, *AnySat: One Earth Observation Model for Many Resolutions, Scales,
  and Modalities*, CVPR, arXiv:2412.14123), base model (768-d; only "base" is released). Code MIT
  (`gastruc/AnySat`, commit `5f6f475e1a22ce5e3a56a5b18f4ed6d24eca2a4a`; its iRPE module derives from
  Microsoft's MIT-licensed iRPE); weights MIT according to the Hugging Face model card (`g-astruc/AnySat`,
  `license: mit`). Code and weights are permissive, so no licence acknowledgement is required.
  `geocrop/anysat_encoder.py` ports only the inference ("release") path for the `s2`, `s1-asc` and `s1`
  modalities (LTAE temporal projector, spatial transformer with iRPE, ViT blocks, cross-attention block) and
  keeps the MIT notices. GeoCrop uses `s2` only.
* **Weights.** `https://huggingface.co/g-astruc/AnySat/resolve/63f252119babfe896d200f88631496f86eddacd3/models/AnySat.pth`,
  SHA-256 `ce1246527a28bade624048c448a1fa473ca85791b998f562ce2bcc3dcac93d4f` (504 MB), downloaded on first use
  into `$GEOCROP_CACHE/anysat/`. The checkpoint holds every modality; GeoCrop loads the keys of the modules it
  builds with `strict=True` and skips the other modalities (aerial, SPOT, NAIP, Landsat, ALOS, MODIS).
* **Equivalence.** Against the official `hubconf.AnySat` (`flash_attn=False`, released weights): maximum
  absolute difference 0.0 for 32 pixels as 1 × 1 tiles ("tile" output, 11 dates) and for 6 × 6 px tiles
  ("patch" output at patch sizes 10 and 20 m, "tile" at 30 m); 1.8e-6 for GeoCrop's batched pixel encoding with
  per-pixel missing months against official per-pixel runs on the valid dates only.
* **Input contract.**

  | AnySat input | GeoCrop source | normalisation |
  |---|---|---|
  | `s2`: B2 B3 B4 B5 B6 B7 B8 B8A B11 B12, `(B, T, 10, 1, 1)` | monthly S2 composite (reflectance) | `(x − mean) / std` per band |
  | `s2_dates` | the 15th of each month as day of year (1 January = 1) | – |

  AnySat has no fixed normalisation statistics: every pre-training loader standardises with its own dataset's
  per-band mean and standard deviation (README: "(data − mean) / std"). GeoCrop therefore computes the
  statistics over all valid monthly S2 observations of the study grid in one pass and records them in the
  GeoTIFF tag `provider_state`; parcel embeddings use the same pixel statistics. Each pixel (or parcel series)
  is a 1 × 1 tile at 10 m encoded with patch size 10 m and the `tile` output, as in the AnySat baseline of the
  official Galileo evaluation for CropHarvest pixel time series. Months without a valid observation are left
  out of a pixel's sequence (variable-length sequences are native to AnySat; pixels are grouped by validity
  pattern). Dates use `tm_yday` (1 January = 1) like most pre-training loaders (TreeSatAI-TS, S2NAIP, PLANTED);
  the README says 1 January = 0 and PASTIS counts days from 1 September, so this convention is approximate by
  up to a day (or a season offset for PASTIS-style dating). Sentinel-1 is not used, because the units of the
  pre-training S1 data could not be verified.

## Feature-provider interface

All sources implement `geocrop.feature_providers.FeatureProvider` and are registered in
`geocrop.plugins.feature_providers` (entry-point group `geocrop.feature_providers`, see
[ADR 0002](adr/0002-plugin-entry-points.md)). A provider declares:

| attribute / method | meaning |
|---|---|
| `name`, `dims`, `sensor`, `product`, `attribution` | identity and provenance (GeoTIFF tags, manifest) |
| `licence`, `code_licence`, `requires_licence_ack` | licence id of the weights/data and the code; providers that are not clearly permissive set `requires_licence_ack = True` and run only with `--allow-license <licence id>` (`features.embeddings.allow_license`) |
| `weights` | pinned files: `WeightFile(name, url, sha256, size)`, downloaded at runtime by `geocrop.embedding_sources.fetch_weights` into `$GEOCROP_CACHE/<name>/`, never committed |
| `contract` | `InputContract(kind, sensors, bands, temporal, normalisation, max_timesteps, notes)`; `kind` is `precomputed` or `monthly` |
| `legacy_names` | write `emb_NN` instead of `emb_<name>_NN` by default (only the three original sources) |
| `load_model(weights)` | the model (random weights when `None`, for tests) |
| `embed_pixels(reference, output, **options)` | embeddings raster on the study grid |
| `embed_parcels(reference, parcels, output, **options)` | parcel table |

Monthly providers subclass `geocrop.feature_providers.MonthlyProvider` and only implement `load_model`,
`encode(model, s2, s1, latlons, month0, batch_size, state)` and optionally `fit(blocks)` (dataset-level state
such as AnySat's statistics); reading the monthly composites block by block, parcel means, the output contract
and provenance are shared. A third-party package registers its provider with an entry point:

```toml
[project.entry-points."geocrop.feature_providers"]
my_fm = "my_package.providers:MyProvider"
```

after which `geocrop embeddings --source my_fm` works like a built-in.

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
| Galileo-nano 2023 (pixel) | 118 s wall for the whole command (incl. weights download, statistics and summary) | weights 4.2 MB once | 12 months (11 with S2), no S1, batch 1024, torch with 5 threads, peak RSS 3.3 GB, ≈ 1,900 px/s |
| Galileo-nano 2023 (parcel) | 1.6 s for the same 528 cells | – | one sequence per parcel |
| AnySat base 2023 (pixel) | 199 s wall for the whole command | weights 504 MB once | as above, batch 1024, ≈ 1,250 px/s on one 256 × 256 block (16 validity patterns); peak RSS 6.1 GB, of which about 2.2 GB for inference and the rest for the summary of the 768-band raster; output 680 MB |
| AnySat base 2023 (parcel) | 1.6 s for the same 528 cells | – | statistics reused from the pixel run |

## Real extraction: Semberija, 2023

AOI 19.10–19.16 °E, 44.72–44.76 °N; grid EPSG:32634, 10 m, 486 × 456 px. Reproduce with:

```bash
geocrop grid --aoi aoi.geojson --crs EPSG:32634 --resolution 10 --out grid.tif
geocrop embeddings --config cfg.yaml                       # features.embeddings: {source: alphaearth, year: 2023}
geocrop embeddings --config cfg.yaml --source tessera
geocrop stac-import --reference grid.tif --start 2023-01-01 --end 2023-12-31 --out s2_scenes --max-cloud 60 --min-clear 0.3
geocrop stac-import --reference grid.tif --start 2023-01-01 --end 2023-05-31 --out s2_scenes --max-cloud 95 --min-clear 0.1
geocrop embeddings --config cfg_presto.yaml --make-monthly   # {source: presto, year: 2023}; paths.s2_scenes: s2_scenes
geocrop embeddings --config cfg_fm.yaml --source galileo     # {year: 2023, monthly: <dir>/monthly} (same composites)
geocrop embeddings --config cfg_fm.yaml --source anysat
```

The Presto run used 29 Sentinel-2 L2A scenes from Earth Search (tile 34TCQ), with no scene available for
April 2023, and no Sentinel-1 data, so the S1 token was masked.

| source | bands | valid fraction | min | max | mean | mean per-band std | vector norm (min / mean / max) |
|---|---|---|---|---|---|---|---|
| AlphaEarth 2023 | 64 | 1.000 | -0.434 | 0.404 | -0.004 | 0.060 | 0.992 / 1.000 / 1.008 |
| TESSERA 2023 | 128 | 1.000 | -10.78 | 12.73 | 0.041 | 1.561 | 23.8 / 28.7 / 41.0 |
| Presto 2023 (S2 only) | 128 | 1.000 | -5.57 | 5.32 | -0.043 | 0.320 | 11.5 / 12.9 / 13.4 |
| Galileo-nano 2023 (S2 only) | 128 | 1.000 | -0.678 | 0.419 | 0.001 | 0.016 | 1.10 / 1.20 / 1.28 |
| AnySat base 2023 (S2 only) | 768 | 1.000 | -1.064 | 1.025 | 0.009 | 0.128 | 5.06 / 5.56 / 6.27 |

Galileo and AnySat used the same 11 monthly composites as Presto (Jan–Dec 2023 without April). The AnySat
standardisation statistics of this run (reflectance, 1,850,662 valid pixel-months) are in the raster's
`provider_state` tag. Parcel embeddings of the 528 cells lie in [-0.652, 0.384] (Galileo) and [-0.941, 0.900]
(AnySat). The small per-band spread of Galileo (mean per-band standard deviation 0.016 against a vector norm
of 1.2) is a property of the mean-pooled nano model, not of missing data; whether it limits separability is an
open question for the frozen-protocol comparison.
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
* **Galileo and AnySat inputs.** Galileo was pre-trained on Sentinel-2 L1C top-of-atmosphere data
  (`COPERNICUS/S2_HARMONIZED`); GeoCrop composites are L2A surface reflectance, a domain shift (Presto has the
  same caveat). Galileo runs without its climate, topography, land-cover and population inputs (masked, which
  pre-training's random masking covers). AnySat's standardisation statistics come from the study area, so
  embeddings of two studies (or two years) are only comparable if they share the statistics; pass a fixed
  `state` to `embed_pixels` to reuse them. Both run per pixel without spatial context (Galileo also supports
  spatial patches and AnySat larger tiles, which GeoCrop does not use yet).
* **Presto without ERA5, SRTM and Dynamic World.** These token groups are masked. Presto was pre-trained with
  random group masking, so this is in-distribution, but embeddings computed without these inputs are not
  identical to those of a full WorldCereal-style pipeline. S1 composites need linear VV/VH: GeoCrop's SNAP
  route, the Planetary Computer RTC import (`s1_stac`) or openEO `s1_composites`. When S1 is absent, the S1 token is masked.
* **Resampling.** The default is nearest neighbour, which keeps AlphaEarth vectors unit-length. Grids that are
  not aligned to the source's 10 m lattice therefore shift by up to half a pixel.
* **Version drift.** AlphaEarth, the TESSERA v1 `.npy` tiles (deprecated upstream in favour of Zarr and
  Icechunk) and the model checkpoints can change upstream. GeoCrop pins the Presto, Galileo and AnySat weights
  by commit/revision URL and SHA-256 and records tile URLs and checkpoint hashes in the output tags. Re-extract when you change versions.
* **Pipeline.** `geocrop run` extracts and attaches embeddings when `features.embeddings` is set (see
  [Commands](#commands)), but only for a **single** AlphaEarth, TESSERA or Presto source: the embeddings stage
  does not yet pass monthly composites to Galileo/AnySat, does not accept a list of sources and ignores
  `namespace`/`allow_license`. For those, run `geocrop embeddings` and then `geocrop features --config` (or
  `geocrop merge-stacks`).
* **Licences.** You are responsible for attribution when publishing maps derived from AlphaEarth (CC-BY-4.0)
  and TESSERA (see the licence notes above). The MIT licences of the Galileo and AnySat weights cover the
  weights; they do not re-license the pre-training datasets, which GeoCrop never downloads or redistributes.
