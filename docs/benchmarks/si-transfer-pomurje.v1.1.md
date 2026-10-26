# Public transfer benchmark `si-transfer-pomurje` v1.1 (protocol v1 + domain-shift analysis)

Same frozen protocol as [v1](si-transfer-pomurje.v1.md): file and SHA-256 hash unchanged. The run also evaluates the
prediction-time shift check and shift-aware conformal thresholds of [`geocrop.shift`](../domain-shift.md). All
v1 scores below (accuracy, coverage, AOA, transfer CV) were reproduced exactly by this run. The new section is
*Domain-shift detection and shift-aware conformal sets*. Run on a shared workstation, so stage times are not
comparable with v1.

Protocol SHA-256 `5620feeaf651b96032cee1b328600ec605cafc065274ed8ac353b80f64e68ebe` · engine 0.9.0 (`eff2d0cf9ceeab82b5bb148061d2b57d030879fb`) · 2026-10-26T11:20:15+00:00 – 2026-10-26T12:02:17+00:00

> European public transfer task (Slovenia; declared, not field-verified labels). Not Republic of Srpska data and not evidence of accuracy there.

Reproduce: `geocrop benchmark-transfer --protocol configs/benchmarks/si_transfer_pomurje.v1.yaml --work outputs/transfer-v1`

## Domains

| Domain | Region | Year | Labels | Tiles | Sampled parcels per class | Dropped classes |
|---|---|---|---|---|---|---|
| pomurje_2021 (source) | pomurje | 2021 | eurocrops_si_2021 | SI-MS-S, SI-LE-W, SI-LJ | grassland 150, maize 150, oil_pumpkin 150, winter_cereals 150, winter_rapeseed 150 | - |
| podravje_2021 | podravje | 2021 | eurocrops_si_2021 | SI-PT-E, SI-DP | grassland 150, maize 150, oil_pumpkin 150, winter_cereals 150, winter_rapeseed 86 | - |
| posavje_2021 | posavje | 2021 | eurocrops_si_2021 | SI-KK-W, SI-KK-E | grassland 150, maize 150, winter_cereals 150, winter_rapeseed 123 | oil_pumpkin (10) |
| gorenjska_2021 | gorenjska | 2021 | eurocrops_si_2021 | SI-KR-S, SI-KR-N | grassland 150, maize 150, winter_cereals 150 | oil_pumpkin (1), winter_rapeseed (0) |
| pomurje_2025 | pomurje | 2025 | kmrs_si_2025 | SI-MS-S, SI-LE-W, SI-LJ | grassland 150, maize 150, oil_pumpkin 150, winter_cereals 150, winter_rapeseed 150 | - |

## In-domain vs out-of-domain (one model trained on the source)

Parcel level, 95 % spatial-block bootstrap intervals. Macro-F1 over the classes present in the domain; *common* = grassland, maize, winter cereals only (comparable across domains).

| Domain | Kind | Parcels | OA % | Macro-F1 % | Common-class macro-F1 % | Conformal coverage % (target) | Mean set size | Parcels inside AOA % | Pixels inside AOA % |
|---|---|---|---|---|---|---|---|---|---|
| pomurje_2021 | in-domain (source test parcels) | 232 | 93.1 [88.8, 96.9] | 93.1 [88.8, 96.7] | 93.8 [88.7, 98.1] | 89.7 (90.0) | 0.95 | 98.7 | 97.4 |
| podravje_2021 | out-of-domain (other region) | 660 | 93.6 [91.6, 95.6] | 93.9 [91.8, 95.8] | 94.9 [92.6, 96.9] | 89.2 (90.0) | 0.99 | 96.7 | 95.2 |
| posavje_2021 | out-of-domain (other region) | 561 | 88.8 [85.4, 92.2] | 89.3 [86.0, 92.6] | 91.7 [88.6, 94.6] | 84.7 (90.0) | 0.96 | 99.6 | 98.0 |
| gorenjska_2021 | out-of-domain (other region) | 392 | 90.3 [87.0, 93.7] | 91.2 [88.0, 94.2] | 91.2 [88.0, 94.2] | 82.1 (90.0) | 0.92 | 87.0 | 83.4 |
| pomurje_2025 | out-of-domain (other year, same region) | 697 | 91.8 [89.5, 93.9] | 91.8 [89.4, 93.8] | 95.3 [93.5, 97.0] | 84.8 (90.0) | 0.93 | 93.1 | 90.9 |

## Per-class F1 (%)

| Domain | grassland | maize | oil_pumpkin | winter_cereals | winter_rapeseed |
|---|---|---|---|---|---|
| pomurje_2021 | 92.5 | 88.6 | 95.5 | 94.4 | 94.6 |
| podravje_2021 | 93.6 | 90.7 | 93.9 | 95.1 | 96.3 |
| posavje_2021 | 86.2 | 95.5 | - | 87.0 | 88.6 |
| gorenjska_2021 | 89.7 | 89.8 | - | 94.0 | - |
| pomurje_2025 | 94.9 | 92.6 | 91.5 | 90.0 | 89.9 |

Drop vs source (percentage points, source F1 minus target F1):

| Domain | grassland | maize | oil_pumpkin | winter_cereals | winter_rapeseed |
|---|---|---|---|---|---|
| podravje_2021 | -1.2 | -2.0 | +1.6 | -0.7 | -1.7 |
| posavje_2021 | +6.2 | -6.9 | - | +7.4 | +6.0 |
| gorenjska_2021 | +2.7 | -1.2 | - | +0.4 | - |
| pomurje_2025 | -2.5 | -3.9 | +4.0 | +4.4 | +4.7 |

## Transferred vs local model on the same target test parcels

| Target | Test parcels | Transferred macro-F1 % | Local macro-F1 % | Gap (pp) | Transferred OA % | Local OA % | Coverage transferred / local % |
|---|---|---|---|---|---|---|---|
| podravje_2021 | 222 | 94.0 [89.3, 97.6] | 97.5 [94.4, 99.5] | +3.6 | 93.7 | 97.3 | 87.8 / 92.8 |
| posavje_2021 | 181 | 92.4 [85.7, 96.2] | 97.3 [94.5, 99.4] | +4.9 | 92.3 | 97.2 | 88.4 / 95.6 |
| gorenjska_2021 | 125 | 90.3 [84.5, 95.9] | 99.2 [97.5, 100.0] | +8.9 | 89.6 | 99.2 | 84.0 / 96.0 |
| pomurje_2025 | 234 | 94.0 [90.9, 96.7] | 95.7 [92.0, 98.7] | +1.7 | 94.0 | 95.7 | 87.6 / 92.3 |

## Domain-shift detection and shift-aware conformal sets

Shift report of each evaluated parcel set against the source training/calibration pixels (`geocrop.shift`, a-priori limits; docs/domain-shift.md). AUC = grouped-CV domain-classifier AUC after re-weighting the source to the target's predicted class mix (unadjusted in brackets); SMD = importance-weighted mean |standardised mean difference|; outside AOA = share of target pixels outside the AOA (source training CV in brackets); TVD = predicted class-mix distance.

| Domain | Verdict | AUC | SMD | Outside AOA % | DI median ratio | TVD | Metadata | Reasons |
|---|---|---|---|---|---|---|---|---|
| pomurje_2021 | **in-domain** | 0.65 (0.66) | 0.05 | 2.3 (2.1) | 1.02 | 0.05 | same | - |
| podravje_2021 | **severe** | 0.89 (0.89) | 0.16 | 5.0 (2.1) | 1.05 | 0.07 | differs | a classifier separates target from reference pixels (class-mix adjusted AUC 0.89); region podravje not among calibration regions ['pomurje'] |
| posavje_2021 | **severe** | 0.97 (0.97) | 0.12 | 1.9 (2.1) | 1.13 | 0.16 | differs | predicted class mix differs from the reference (total variation 0.16); a classifier separates target from reference pixels (class-mix adjusted AUC 0.97); region posavje not among calibration regions ['pomurje'] |
| gorenjska_2021 | **severe** | 0.99 (0.99) | 0.26 | 16.6 (2.1) | 1.37 | 0.35 | differs | predicted class mix differs from the reference (total variation 0.35); feature means differ (importance-weighted mean |SMD| 0.26); a classifier separates target from reference pixels (class-mix adjusted AUC 0.99); 16.6 % of target pixels lie outside the AOA (training CV: 2.1 %); median dissimilarity index is 1.37 x the training CV median; region gorenjska not among calibration regions ['pomurje'] |
| pomurje_2025 | **severe** | 0.99 (0.99) | 0.19 | 9.3 (2.1) | 1.34 | 0.06 | differs | a classifier separates target from reference pixels (class-mix adjusted AUC 0.99); 9.3 % of target pixels lie outside the AOA (training CV: 2.1 %); median dissimilarity index is 1.34 x the training CV median; year 2025 not among calibration years ['2021'] |

Conformal coverage % / mean set size by threshold mode (target 90.0 %):

| Domain | none | conservative | weighted | Effective calibration size (weighted) |
|---|---|---|---|---|
| pomurje_2021 | 89.7 / 0.95 | 96.1 / 1.39 | 100.0 / 4.29 | 54 of 147 |
| podravje_2021 | 89.2 / 0.99 | 97.4 / 1.48 | 100.0 / 5.00 | 17 of 147 |
| posavje_2021 | 84.7 / 0.96 | 93.8 / 1.58 | 100.0 / 5.00 | 6 of 147 |
| gorenjska_2021 | 82.1 / 0.92 | 96.7 / 1.53 | 100.0 / 5.00 | 5 of 147 |
| pomurje_2025 | 84.8 / 0.93 | 98.3 / 1.65 | 100.0 / 5.00 | 5 of 147 |

## Engine transfer CV (pixel level, pooled samples of all domains)

In-domain spatial-block CV macro-F1: 96.2 %.

| Strategy | Held-out group | Test pixels | In-domain macro-F1 % | Out-of-domain macro-F1 % | Gap (pp) | Classes missing in training |
|---|---|---|---|---|---|---|
| region | gorenjska | 15051 | 98.7 | 96.8 | +1.9 | - |
| region | podravje | 29303 | 96.7 | 95.0 | +1.7 | - |
| region | pomurje | 57285 | 95.6 | 89.9 | +5.7 | - |
| region | posavje | 22981 | 95.8 | 93.6 | +2.1 | - |
| year | 2021 | 95073 | 95.8 | 91.3 | +4.6 | - |
| year | 2025 | 29547 | 97.1 | 95.0 | +2.1 | - |

## Label sources and licences

- **eurocrops_si_2021** (eurocrops): CC-BY-SA-4.0. Schneider, M., Schelte, T., Schmitz, F., Koerner, M. (2023). EuroCrops: The Largest Harmonized Open Crop Dataset Across the European Union. Scientific Data 10, 612. Data: doi:10.5281/zenodo.14094196. Checksum `md5:b2acea403d8372ccb9c5f5276130f23b`.
- **kmrs_si_2025** (url): MKGP open data: public, no restriction of use, citation of the source mandatory. Ministrstvo za kmetijstvo, gozdarstvo in prehrano (MKGP), Sloj kmetijskih rastlin iz zbirnih vlog (KMRS) 2025, https://rkg.gov.si/ (metadata: eprostor.gov.si record 21ecba4a-1214-4617-8bf4-020d2f235a25). Checksum `sha256:2977a4163ce60d7b83e9cbc0a86bdfb5e437e5abde6f37c0e6a061335eb9fa14`.
- kmrs_si_2025: national codes mapped to HCAT with the EuroCrops table: 157 codes, 0 inconsistent (min. consistency 1.0), table SHA-256 `62053186a7a0d615…`.

Samples: 124620 pixels x 127 features, fingerprint `065e98f90c4f54c8…`; stages: labels 7.6 s, imagery_and_samples 27.3 s, source_model 4.4 s, transfer_scoring 2171.4 s, transfer_cv 311.2 s; peak RSS 1942.6 MB.
