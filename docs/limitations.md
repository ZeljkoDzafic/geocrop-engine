# Scope, status and limitations

GeoCrop Engine 0.9 is research software. This page states, per part of the engine, what has been shown on
**synthetic data** (software correctness), what on **real or public data** (engineering and transfer evidence),
and what is still **open**. Every figure below comes from a run documented, with its command, on the linked page.

## In brief

* The software chain is tested end to end on synthetic data and runs on real Sentinel-2 L2A and Sentinel-1 RTC
  data from public STAC services, with physical plausibility checks ([acceptance](acceptance.md)).
* Crop-type accuracy has been measured only on **public European data with declared (not field-verified)
  labels** (Slovenia: EuroCrops 2021 and KMRS 2025; [benchmarks](benchmarks/README.md)).
* **Accuracy in the study region (Semberija, Republic of Srpska) against independent field observations has not
  been measured yet.** No accuracy figure for that region exists, and none should be inferred from the demo, the
  acceptance run or the scale benchmark: they use synthetic classes or k-means pseudo-labels.
* Uncertainty outputs are close to their target in-domain, but **conformal coverage falls below its target under
  domain shift** (other regions, another year).

## Evidence by component

| Component | Synthetic data / unit tests (default tier) | Real or public data | Status |
|---|---|---|---|
| Full pipeline `geocrop run`, resumption, interruption recovery | yes (`geocrop demo`, [reliability gate](testing.md#reliability-gate-teststest_reliabilitypy), offline acceptance) | whole chain in `geocrop acceptance` (Semberija 2023, ~4 km²) | software verified |
| Sentinel-2 L2A import (Earth Search STAC), cloud/shadow masking, compositing, features | yes | network tier and acceptance run (radiometry, negative reflectance, index ranges); [scale benchmark](performance.md) on 25–1000 km² | verified for one area and season |
| Sentinel-1 RTC import (Planetary Computer), linear compositing, S1/S2 co-registration | yes | acceptance run (dB ranges, VV > VH, residual shift within ±1 px) | verified for one area and season |
| CDSE OData/STAC search and download, SAFE import, SNAP graphs | yes (synthetic SAFE, mocked HTTP) | **not run** against the live service or real SAFE products in automated tests | to be verified per study (below) |
| openEO back-end | yes (process graphs) | graph validation against CDSE only when `CDSE_OPENEO_*` credentials are set | results differ from local compositing ([openEO](openeo.md)) |
| Crop-type classification accuracy | software check only | [EuroCrops SI 2021 Pomurje](benchmarks/eurocrops-si-2021-pomurje.v1.md): parcel OA 86.9–88.5 %, macro-F1 85.5–87.2 % (four models, intervals ≈ ±5 points), declared labels | **open for the study region** (independent field data) |
| Region / year transfer | transfer CV on synthetic groups | [si-transfer-pomurje v1](benchmarks/si-transfer-pomurje.v1.md): 1.7–8.9 points macro-F1 below a locally trained model on the same target parcels | measured on public data only |
| Calibration and conformal prediction sets | coverage on synthetic data | in-domain coverage 89.7 % (target 90 %); 82.1–84.8 % in Posavje, Gorenjska and Pomurje 2025 | **coverage not guaranteed under shift** |
| Area of Applicability (AOA) | yes | flags 13 % of Gorenjska parcels (OA 62.7 % outside vs 94.4 % inside), but not the Posavje shift | detects feature-space novelty only |
| OBIA segmentation with seam reconciliation | yes | runs on real composites in the acceptance run; [FTW vs OBIA on Semberija](field-boundaries.md#semberija-run-2023) (qualitative) | **no accuracy against reference parcels** yet |
| U-Net boundaries, Fields of The World | U-Net train/infer on synthetic patches | network tier downloads the FTW CC-BY checkpoint and runs inference; Semberija run qualitative only | no accuracy against local parcels |
| Foundation-model embeddings (AlphaEarth, TESSERA, Presto, Galileo, AnySat) | yes (mocked sources) | network tier reads real windows or downloads and loads the checkpoints | no accuracy evaluation of embedding features |
| Parcel aggregation, declaration status, CbM markers, traffic light | yes | acceptance run on segments with pseudo-labels | thresholds and crop calendars not yet checked against real declarations |
| Web workspace, HTTP API, QGIS plugin | yes (security gate, API, plugin runner) | — | single-owner local tool ([security model](security.md)) |

## Implemented following specifications, to be verified per study on real data
* CDSE OData/STAC queries, token refresh and download against the live service and your account quotas.
* SNAP graphs with your SNAP version (`gpt <graph> -h`) and visual QA of σ⁰/γ⁰ outputs.
* SAFE import on real products of different processing baselines (offset handling).
* Cloud/shadow masking quality (SCL has known omission errors for thin clouds and shadows; inspect composites).

## Methodological limits users must report
* Internal CV on interpreted polygons is optimistic; external accuracy requires independent field data.
* If training polygons were located using declarations, comparison with declarations is not fully independent;
  state this, and evaluate on independently verified parcels.
* Public benchmark labels are farmers' declarations harmonised to HCAT; declaration errors are part of the
  "truth", and the Slovenian results are not evidence of accuracy in Bosnia and Herzegovina.
* 20 m bands resampled to 10 m add no spatial detail; parcels with few interior pixels are flagged rather than decided.
* Temperature scaling and conformal sets are fitted on out-of-fold predictions or calibration parcels of the
  training domain. In the transfer benchmark, coverage of the 90 % sets fell to 82–85 % in two other regions and in
  another year. Recalibrate on independently verified parcels of the target region and season before relying on
  prediction sets operationally; treat an empty (Mondrian) set as "needs review".
* The AOA detects feature-space novelty, not label or phenology shift; "inside the AOA" is no substitute for
  local validation.
* Markers and crop calendars encode regional agronomic assumptions; harvest/rapid-senescence detection from NDVI
  alone cannot distinguish harvest from drought-induced senescence.
* Segmentation outputs are candidate objects, not cadastral parcels. OBIA reconciles tile seams with a spectral
  similarity rule ([segmentation](segmentation.md)); it can merge adjacent fields with similar spectra or leave a
  field split when its two halves differ across the seam. The `tile`, `parts` and `seam_merged` columns support QA.
* Spatial transfer to other regions or years requires new reference data (crop calendars and management differ).

## Not in scope
Legal compliance decisions, farmer-level reporting, real-time processing, or redistribution of satellite or
parcel data.
