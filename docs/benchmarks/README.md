# Public benchmarks

`geocrop benchmark-public` runs a **frozen, versioned protocol** on openly licensed reference data, so that
GeoCrop's accuracy, uncertainty and cost can be checked by anyone, on a fixed task, with one command. It is the
public part of gap G1; the private Semberija field-data study follows the same pattern.

| Protocol | Labels | Imagery | Result |
|---|---|---|---|
| [`eurocrops_si_2021_pomurje.v1.yaml`](../../configs/benchmarks/eurocrops_si_2021_pomurje.v1.yaml) | EuroCrops Slovenia 2021 (CC-BY-SA-4.0) | Sentinel-2 L2A, Earth Search STAC | [eurocrops-si-2021-pomurje.v1.md](eurocrops-si-2021-pomurje.v1.md) ([JSON](eurocrops-si-2021-pomurje.v1.json)) |
| [`si_transfer_pomurje.v1.yaml`](../../configs/benchmarks/si_transfer_pomurje.v1.yaml) (region/year **transfer**) | EuroCrops Slovenia 2021 (CC-BY-SA-4.0) + MKGP KMRS 2025 (open, citation required) | Sentinel-2 L2A, Earth Search STAC | [si-transfer-pomurje.v1.md](si-transfer-pomurje.v1.md) ([JSON](si-transfer-pomurje.v1.json)), [summary below](#transfer-protocol-si-transfer-pomurje-v1) |

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
accessed through Element 84 Earth Search (AWS Open Data). KMRS 2025: Ministrstvo za kmetijstvo, gozdarstvo in
prehrano (MKGP), open data with mandatory citation; downloaded at run time, never committed. The protocol files and these reports contain no parcel
geometries or identifiers.

## Transfer protocol `si-transfer-pomurje` v1

**A European public transfer task (Slovenia), not Republic of Srpska data.** It answers: how much does a
crop-type model trained in one region and year lose when applied, without retraining, to geographically separate
regions and to another year — and do its uncertainty outputs (conformal sets, area of applicability) notice?

```bash
geocrop benchmark-transfer --protocol configs/benchmarks/si_transfer_pomurje.v1.yaml \
    --work outputs/transfer-v1 --publish docs/benchmarks
```

| Domain | Role | Distance to source | Labels |
|---|---|---|---|
| Pomurje 2021 (3 tiles: Murska Sobota, Beltinci/Lendava, Ljutomer) | source (train / calibrate / in-domain test) | – | EuroCrops SI 2021 |
| Podravje 2021 (Ptujsko + Dravsko polje) | other region | ~35 km | EuroCrops SI 2021 |
| Posavje 2021 (Krško–Brežice plain) | other region | ~110 km | EuroCrops SI 2021 |
| Gorenjska 2021 (Kranj / Sorško polje, pre-alpine) | other region | ~150 km | EuroCrops SI 2021 |
| Pomurje 2025 (same 3 tiles) | other year (leave-one-year-out) | 4 seasons | MKGP KMRS 2025 |

Five HCAT classes (maize, winter cereals, grassland, oil pumpkin, winter rapeseed); a target class with fewer
than 20 eligible parcels is dropped from that target and listed (Posavje: oil pumpkin; Gorenjska: oil pumpkin
and rapeseed). *Common-class* macro-F1 uses only grassland, maize and winter cereals, which every domain has.

**Second year.** EuroCrops holds only Slovenia 2021 (checked against Zenodo record 14094196 on 2026-10-26). The
Slovenian Ministry of Agriculture publishes the crop declarations of the latest campaign as open data (KMRS,
metadata record `21ecba4a-1214-4617-8bf4-020d2f235a25`: *"Javno dostopni podatki, ni omejitve uporabe, citiranje
vira obvezno"* — public, no restriction of use, citation mandatory); only the current file (`KMRS_2025.zip`) is
downloadable, earlier years return HTTP 404. KMRS uses the national crop codes (`SIFRA_KMRS`); they are mapped to
HCAT3 with the code → HCAT table that EuroCrops used for 2021 (157 codes, every code maps to exactly one HCAT
class; 0.65 % of 2025 parcels carry codes absent in 2021 and are dropped). The file is pinned by SHA-256; when
the Ministry replaces it with the next campaign the protocol can no longer be re-run from the original URL —
keep the downloaded archive (the checksum identifies it).

### v1 result (2026-10-26, engine commit `42f65b0`, reproduced bit-identically from a clean checkout)

One Random Forest (300 trees) fitted on 303 source parcels; conformal thresholds from 147 source calibration
parcels; parcel-level scores with 95 % spatial-block bootstrap intervals.

| Domain | Parcels | OA % | Macro-F1 % | Common-class macro-F1 % | Conformal coverage % (target 90) | Parcels inside AOA % |
|---|---|---|---|---|---|---|
| Pomurje 2021, in-domain test | 232 | 93.1 [88.8, 96.9] | 93.1 [88.8, 96.7] | 93.8 [88.7, 98.1] | 89.7 | 98.7 |
| Podravje 2021 | 660 | 93.6 [91.6, 95.6] | 93.9 [91.8, 95.8] | 94.9 [92.6, 96.9] | 89.2 | 96.7 |
| Posavje 2021 | 561 | 88.8 [85.4, 92.2] | 89.3 [86.0, 92.6] | 91.7 [88.6, 94.6] | 84.7 | 99.6 |
| Gorenjska 2021 | 392 | 90.3 [87.0, 93.7] | 91.2 [88.0, 94.2] | 91.2 [88.0, 94.2] | 82.1 | 87.0 |
| Pomurje 2025 (other year) | 697 | 91.8 [89.5, 93.9] | 91.8 [89.4, 93.8] | 95.3 [93.5, 97.0] | 84.8 | 93.1 |

Same target test parcels, transferred model vs a model trained locally in the target with the same split rules:

| Target | Test parcels | Transferred macro-F1 % | Local macro-F1 % | Gap (pp) |
|---|---|---|---|---|
| Podravje 2021 | 222 | 94.0 [89.3, 97.6] | 97.5 [94.4, 99.5] | 3.6 |
| Posavje 2021 | 181 | 92.4 [85.7, 96.2] | 97.3 [94.5, 99.4] | 4.9 |
| Gorenjska 2021 | 125 | 90.3 [84.5, 95.9] | 99.2 [97.5, 100.0] | 8.9 |
| Pomurje 2025 | 234 | 94.0 [90.9, 96.7] | 95.7 [92.0, 98.7] | 1.7 |

Reading:

* **Accuracy drop is moderate and grows with distance**: none in the neighbouring Podravje, 2–9 points against
  a local model further away (largest in pre-alpine Gorenjska). The in-domain intervals overlap most target
  intervals, so single-domain differences of a few points are not significant; the local-vs-transferred
  comparison on identical parcels is the sharper measure.
* **Per-class**: in Posavje grassland, winter cereals and rapeseed lose 6–7 F1 points while maize gains; in
  2025 winter cereals, rapeseed and oil pumpkin lose 4–5 points (full per-class table in
  [si-transfer-pomurje.v1.md](si-transfer-pomurje.v1.md)).
* **Conformal coverage under shift falls below target** (82–85 % in Posavje, Gorenjska and 2025 vs 89.7 %
  in-domain): thresholds calibrated in the source do not guarantee coverage elsewhere. Winter cereals are the
  weakest class already in-domain (74 % class coverage, 50–66 % in the targets). Mean set sizes below 1 reflect
  empty sets (5–11 % of parcels), which Mondrian thresholds allow; an empty set should be read as "needs review".
* **AOA flags the far region**: 13 % of Gorenjska parcels fall outside the source area of applicability (vs 1 %
  in-domain); their OA is 62.7 % vs 94.4 % inside. In Posavje almost everything is "inside" although accuracy
  drops — the AOA detects feature-space novelty, not label/phenology shift, and is no substitute for local
  validation.
* The engine's pooled leave-one-region-out / leave-one-year-out CV (pixel level, all five domains pooled, so
  every held-out region still has other training regions) shows gaps of 1.7–5.7 points (region) and 2.1–4.6
  points (year) against spatial-block CV (96.2 %).

Limitations: declared, not field-verified labels; five classes; ~150 parcels per class and domain; one
season pair; one model family; imagery time of the first run 32 min for 13 tiles (re-runs reuse the stacks).
