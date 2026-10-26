# Public benchmarks

`geocrop benchmark-public` runs a **frozen, versioned protocol** on openly licensed reference data, so that
GeoCrop's accuracy, uncertainty and cost can be checked by anyone, on a fixed task, with one command. It is the
public part of gap G1; the private Semberija field-data study follows the same pattern.

| Protocol | Labels | Imagery | Result |
|---|---|---|---|
| [`eurocrops_si_2021_pomurje.v1.yaml`](../../configs/benchmarks/eurocrops_si_2021_pomurje.v1.yaml) | EuroCrops Slovenia 2021 (CC-BY-SA-4.0) | Sentinel-2 L2A, Earth Search STAC | [eurocrops-si-2021-pomurje.v1.md](eurocrops-si-2021-pomurje.v1.md) ([JSON](eurocrops-si-2021-pomurje.v1.json)) |

### v1 result in brief (2026-10-26)

Parcel-level overall accuracy 86.9 % (RF baseline) to 88.5 % (CatBoost), macro-F1 85.5-87.2 %, with block-bootstrap
intervals of roughly ±5 points; the four models overlap. Grassland, maize, oil pumpkin and winter rapeseed reach
F1 > 92 %; vegetables (mixed declarations) and soybean (29 test parcels) stay near 65-70 % with wide intervals.
Split-conformal sets at 90 % target covered 89-93 % of test parcels with 1.2-1.6 classes on average. Sending
8-20 % of parcels to review raised the accuracy of the accepted parcels to 91-94 %, short of the 95 % target set
on the calibration parcels. Imagery import and compositing took ~7.5 min for the four tiles; model fitting took
0.5-28 s per model on 19 407 training pixels; peak RSS 1.8 GB. LightGBM was skipped (its native library could not
be loaded on the benchmark machine). The same scores and sample fingerprint were obtained in three runs (two with a
fresh work directory). Full table: [eurocrops-si-2021-pomurje.v1.md](eurocrops-si-2021-pomurje.v1.md).

## Running

```bash
pip install -e ".[stac]"            # pystac-client; optional: ".[boost]" for lightgbm/catboost
geocrop benchmark-public --protocol configs/benchmarks/eurocrops_si_2021_pomurje.v1.yaml \
    --work outputs/benchmarks/eurocrops-si-2021-pomurje-v1 --publish docs/benchmarks
```

* EuroCrops `SI_2021.zip` (213 MB) is downloaded from Zenodo record 14094196 into `$GEOCROP_DATA_CACHE`
  (default `~/.cache/geocrop/public`) and verified against the md5 in the protocol; only the parcels of the tiles
  are read. Nothing derived from it is committed except aggregate scores.
* Sentinel-2 COGs are read window-wise for each 10 x 10 km tile (69 tile-acquisitions x 11 bands in the v1 run; ~4 GB
  of scenes, composites and stacks in `--work`). The run is resumable: imported scenes, composites and stacks are reused.
* A model whose optional backend is missing is reported as `skipped` with the install hint, never silently
  dropped. Results go to `<work>/results.json` (schema `geocrop-public-benchmark/1`) and `results.md`;
  `--publish` copies both here, named after the protocol.

## What the protocol freezes

| Section | Content |
|---|---|
| `dataset` | Zenodo record, archive name and md5, HCAT code field, parcel ID field, licence and citation |
| `year`, `grid`, `tiles` | season, CRS / resolution, tile origins (10 x 10 km) |
| `classes` | HCAT3 codes per benchmark class (`exact` or `subtree`, see `geocrop.hcat`) |
| `sampling` | minimum parcel area, parcels per class (stratified random, fixed seed), inner buffer, pixels per parcel |
| `split` | spatial blocks (2 km), test fraction (class-stratified block holdout), calibration fraction, train/test distance buffer, seed |
| `imagery` | STAC endpoint and collection, granule cloud filter, scenes per window, minimum clear fraction of the tile |
| `windows`, `compositing`, `features` | compositing windows, median compositing, bands, indices, deltas, seasonal statistics |
| `models` | estimator parameter sets (RF baseline and the other registry estimators) |
| `evaluation` | parcel unit, bootstrap size/seed/level, conformal alpha, target accuracy for the review burden |

The results record the **SHA-256 of the protocol file**; any edit (including comments) produces a new hash and
must be published as a new version file (`*.v2.yaml`), never by changing a released one.

## Metrics

* **Parcel OA, macro-F1 and per-class F1** on test parcels; a parcel's class is the argmax of the mean pixel
  probability. 95 % percentile intervals come from resampling the test parcels' **2 km spatial blocks**
  (1000 replicates), which is wider and more honest than resampling autocorrelated pixels. Pixel OA is shown for
  comparison with pixel-level studies.
* **Conformal coverage and mean set size**: Mondrian (class-conditional) split-conformal thresholds fitted on the
  calibration parcels (blocks disjoint from training and test), evaluated on test parcels; target 90 %.
* **Review burden**: share of test parcels sent to manual review when the confidence threshold is the smallest
  one that reaches the target accuracy (95 %) on the calibration parcels; the accuracy actually reached on the
  accepted test parcels is reported next to it. A value below the target shows that a threshold chosen on a
  small calibration set does not transfer exactly.
* **Wall time and peak RAM**: fit and predict seconds per model; seconds and process peak RSS per stage
  (labels, imagery + samples, models). Imagery time is dominated by network reads and depends on the connection.

## Reading the results honestly

* Labels are **farmers' declarations** harmonised to HCAT, not field observations; declaration errors are part of
  the "truth". The task is a transfer test in a neighbouring agro-climatic region, not an estimate of accuracy in
  Bosnia and Herzegovina.
* Spatial blocks of 2 km separate training and test parcels, and training/calibration parcels within 200 m of a
  test parcel are removed; neighbouring fields of the same farm can still share management.
* Seven classes and ~1000 parcels make per-class intervals wide (see soybean). Rare classes in the population
  (soybean: 96 eligible parcels) are used in full.
* One run per protocol version is reported; the seeds are fixed, but the remote imagery service may re-process
  scenes (the scene IDs used are recorded, including the processing version suffix).

## Licences

EuroCrops data: CC-BY-SA-4.0 (Schneider et al., 2023, *Scientific Data* 10, 612; Zenodo
doi:10.5281/zenodo.14094196). Labels derived from it during a run stay in the local work directory; if you
redistribute them, do so under CC-BY-SA-4.0 with attribution. Sentinel-2 data: Copernicus Sentinel data 2021,
accessed through Element 84 Earth Search (AWS Open Data). The protocol files and these reports contain no parcel
geometries or identifiers.
