# Public benchmark: eurocrops-si-2021-pomurje v1

* Protocol: `configs/benchmarks/eurocrops_si_2021_pomurje.v1.yaml`, SHA-256 `16106e5ec040a5e30facc23c518ffe2752f29988521493d93e29f774b72b6403`
* Command: `geocrop benchmark-public --protocol configs/benchmarks/eurocrops_si_2021_pomurje.v1.yaml --work outputs/benchmarks/eurocrops-si-2021-pomurje-v1`
* Engine: geocrop-engine 0.9.0, commit `b040c6e3d28d9b4fdb440d02aeec9b5a69b6d97a`
* Run: 2026-10-26T01:39:25+00:00 – 2026-10-26T01:47:35+00:00 on macOS-27.0.1-arm64-arm-64bit (15 logical CPUs), peak RSS 1795.8 MB
* Labels: EuroCrops `SI_2021.zip` (Zenodo 14094196, `md5:b2acea403d8372ccb9c5f5276130f23b`), licence CC-BY-SA-4.0. Schneider, M., Schelte, T., Schmitz, F., Koerner, M. (2023). EuroCrops: The Largest Harmonized Open Crop Dataset Across the European Union. Scientific Data 10, 612. Data: doi:10.5281/zenodo.14094196.
* Samples: 37402 pixels, 127 features, fingerprint `8a8a1569e9ab174b`

Scores are parcel-level (class = argmax of mean pixel probability) on spatial-block test parcels; intervals are 95 % block-bootstrap percentiles. Labels are farmers' declarations, not field observations.

## Models

| Model | OA % | Macro-F1 % | Conformal coverage % (target) | Mean set size | Review share % (accepted acc. %) | Fit s | Predict s | Pixel OA % |
|---|---|---|---|---|---|---|---|---|
| rf | 86.9 [81.1, 92.2] | 85.5 [78.2, 91.7] | 91.7 (90) | 1.27 | 16.0 (92.0) | 2.0 | 0.2 | 84.1 |
| extra_trees | 88.1 [82.8, 93.1] | 86.3 [78.6, 92.6] | 92.9 (90) | 1.37 | 19.9 (94.0) | 0.5 | 0.2 | 83.8 |
| hist_gb | 87.8 [83.0, 92.5] | 86.1 [79.4, 91.6] | 89.1 (90) | 1.22 | 16.7 (93.5) | 28.2 | 0.2 | 84.9 |
| lightgbm | skipped: pip install lightgbm | | | | | | | |
| catboost | 88.5 [83.5, 93.2] | 87.2 [80.9, 92.7] | 92.0 (90) | 1.57 | 8.3 (91.3) | 5.9 | 0.0 | 84.8 |

Review share: parcels whose maximum probability is below the threshold that reaches 95 % accuracy on the calibration parcels; the accepted accuracy is then measured on the test parcels.

## Per-class F1 % (test parcels, block-bootstrap CI)

| Class | Test parcels | rf | extra_trees | hist_gb | catboost |
|---|---|---|---|---|---|
| grassland | 50 | 99.0 [96.7, 100.0] | 99.0 [96.7, 100.0] | 100.0 [100.0, 100.0] | 99.0 [96.7, 100.0] |
| maize | 47 | 95.7 [89.1, 100.0] | 93.8 [85.7, 99.0] | 94.7 [88.1, 99.1] | 94.6 [87.5, 99.1] |
| oil_pumpkin | 47 | 92.8 [86.6, 97.2] | 94.0 [87.9, 98.3] | 92.9 [86.1, 97.7] | 96.8 [92.8, 100.0] |
| soybean | 29 | 68.2 [33.3, 95.0] | 65.1 [29.6, 92.3] | 65.1 [33.3, 91.7] | 70.8 [39.0, 95.0] |
| vegetables | 46 | 67.3 [52.4, 82.5] | 70.0 [54.1, 84.9] | 68.6 [53.5, 82.8] | 70.0 [56.8, 82.8] |
| winter_cereals | 46 | 78.7 [59.7, 90.6] | 84.1 [69.0, 93.3] | 84.1 [68.6, 93.8] | 84.1 [68.6, 93.8] |
| winter_rapeseed | 47 | 96.8 [91.9, 100.0] | 97.9 [94.6, 100.0] | 96.9 [92.3, 100.0] | 94.9 [85.2, 100.0] |

## Data and split

| Class | Population | Sampled | With pixels | Fit | Calibration | Test | Buffer-excluded |
|---|---|---|---|---|---|---|---|
| grassland | 907 | 150 | 150 | 71 | 27 | 50 | 2 |
| maize | 4052 | 150 | 139 | 68 | 22 | 47 | 2 |
| oil_pumpkin | 483 | 150 | 138 | 72 | 19 | 47 | 0 |
| soybean | 96 | 96 | 89 | 43 | 16 | 29 | 1 |
| vegetables | 221 | 150 | 140 | 78 | 12 | 46 | 4 |
| winter_cereals | 4442 | 150 | 138 | 68 | 23 | 46 | 1 |
| winter_rapeseed | 354 | 150 | 135 | 70 | 17 | 47 | 1 |

Spatial blocks per role: {'buffer_excluded': 7, 'calibration': 16, 'fit': 44, 'test': 34}; pixels per role: {'buffer_excluded': 346, 'calibration': 5324, 'fit': 19407, 'test': 12325}.

## Imagery (Sentinel-2 L2A, Earth Search)

| Tile | Window | Scenes used |
|---|---|---|
| SI-MS-S | P1 2021-03-15–2021-04-30 | S2A_33TWM_20210331_1_L2A, S2B_33TWM_20210326_1_L2A, S2B_33TWM_20210405_1_L2A |
| SI-MS-S | P2 2021-05-01–2021-05-31 | S2B_33TWM_20210505_1_L2A, S2B_33TWM_20210515_0_L2A |
| SI-MS-S | P3 2021-06-01–2021-06-30 | S2B_33TWM_20210621_1_L2A, S2A_33TWM_20210629_1_L2A, S2B_33TWM_20210624_1_L2A |
| SI-MS-S | P4 2021-07-01–2021-07-31 | S2A_33TWM_20210706_1_L2A, S2B_33TWM_20210724_1_L2A, S2A_33TWM_20210709_1_L2A |
| SI-MS-S | P5 2021-08-01–2021-08-31 | S2A_33TWM_20210815_1_L2A, S2B_33TWM_20210810_1_L2A, S2B_33TWM_20210820_1_L2A |
| SI-MS-S | P6 2021-09-01–2021-10-15 | S2B_33TWM_20210912_1_L2A, S2A_33TWM_20210904_1_L2A, S2A_33TWM_20211014_2_L2A |
| SI-LE-W | P1 2021-03-15–2021-04-30 | S2A_33TXM_20210331_1_L2A, S2A_33TXM_20210321_1_L2A, S2B_33TXM_20210405_2_L2A |
| SI-LE-W | P2 2021-05-01–2021-05-31 | S2B_33TXM_20210505_1_L2A |
| SI-LE-W | P3 2021-06-01–2021-06-30 | S2B_33TXM_20210624_1_L2A, S2A_33TXM_20210616_1_L2A, S2A_33TXM_20210619_1_L2A |
| SI-LE-W | P4 2021-07-01–2021-07-31 | S2B_33TXM_20210724_1_L2A, S2A_33TXM_20210729_1_L2A, S2A_33TXM_20210706_1_L2A |
| SI-LE-W | P5 2021-08-01–2021-08-31 | S2A_33TXM_20210815_1_L2A, S2B_33TXM_20210810_1_L2A, S2A_33TXM_20210808_1_L2A |
| SI-LE-W | P6 2021-09-01–2021-10-15 | S2B_33TXM_20210912_1_L2A, S2A_33TXM_20211004_1_L2A, S2A_33TXM_20211014_1_L2A |
| SI-LJ | P1 2021-03-15–2021-04-30 | S2A_33TWM_20210331_1_L2A, S2B_33TWM_20210326_1_L2A, S2B_33TWM_20210405_1_L2A |
| SI-LJ | P2 2021-05-01–2021-05-31 | S2A_33TWM_20210510_1_L2A, S2B_33TWM_20210505_1_L2A, S2A_33TWM_20210527_1_L2A |
| SI-LJ | P3 2021-06-01–2021-06-30 | S2B_33TWM_20210621_1_L2A, S2A_33TWM_20210629_1_L2A, S2B_33TWM_20210624_1_L2A |
| SI-LJ | P4 2021-07-01–2021-07-31 | S2A_33TWM_20210706_1_L2A, S2B_33TWM_20210724_1_L2A, S2A_33TWM_20210709_1_L2A |
| SI-LJ | P5 2021-08-01–2021-08-31 | S2A_33TWM_20210815_1_L2A, S2B_33TWM_20210810_1_L2A, S2B_33TWM_20210820_1_L2A |
| SI-LJ | P6 2021-09-01–2021-10-15 | S2B_33TWM_20210912_1_L2A, S2A_33TWM_20210904_1_L2A, S2A_33TWM_20211014_2_L2A |
| SI-PT-E | P1 2021-03-15–2021-04-30 | S2A_33TWM_20210331_1_L2A, S2B_33TWM_20210326_1_L2A, S2B_33TWM_20210405_1_L2A |
| SI-PT-E | P2 2021-05-01–2021-05-31 | S2A_33TWM_20210510_1_L2A, S2B_33TWM_20210505_1_L2A, S2A_33TWM_20210527_1_L2A |
| SI-PT-E | P3 2021-06-01–2021-06-30 | S2B_33TWM_20210621_1_L2A, S2A_33TWM_20210629_1_L2A, S2B_33TWM_20210624_1_L2A |
| SI-PT-E | P4 2021-07-01–2021-07-31 | S2A_33TWM_20210706_1_L2A, S2B_33TWM_20210724_1_L2A, S2A_33TWM_20210709_1_L2A |
| SI-PT-E | P5 2021-08-01–2021-08-31 | S2A_33TWM_20210815_1_L2A, S2B_33TWM_20210810_1_L2A, S2B_33TWM_20210820_1_L2A |
| SI-PT-E | P6 2021-09-01–2021-10-15 | S2B_33TWM_20210912_1_L2A, S2A_33TWM_20210904_1_L2A, S2A_33TWM_20211014_2_L2A |

## Stages

| Stage | Seconds | Peak RSS MB (process high-water mark) |
|---|---|---|
| labels | 1.3 | 273.5 |
| imagery_and_samples | 449.3 | 1625.0 |
| models | 39.4 | 1795.8 |

_Parcel-level scores on a spatial-block holdout of public EuroCrops declarations (declared, not field-verified, labels). Valid for this protocol only; not evidence of accuracy elsewhere._
