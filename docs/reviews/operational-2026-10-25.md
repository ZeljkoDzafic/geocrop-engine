# Operational crop-monitoring systems: source review

Review date: 2026-10-25 (Europe/Sarajevo). Scope: primary public repository README/docs and selected source files, not a runtime benchmark. No claim that the survey covers every open-source engine, and no claim that GeoCrop outperforms the systems below. No external code was copied into GeoCrop.

## Verified repository snapshots

| System | Public repository | Reviewed commit | Commit date | Evidence-backed role |
|---|---|---|---|---|
| ESA WorldCereal classification | https://github.com/WorldCereal/worldcereal-classification | b739085ca786db3a39a802fb3bee6c5a0a5601d5 | 2026-10-20 | Python crop/cropland mapping using openEO/CDSE; default and custom heads, reference-data access, production-grid/job orchestration, pretrained multimodal features |
| Sen4CAP | https://github.com/Sen4CAP/Sen4CAP | 247f496fcf965b54b2c4b888f6ab5c7497ee3fb7 | 2025-12-11 | Operational crop type, mowing and agricultural-practice processors; marker storage/API, scheduler/executor, Docker and PostGIS-based infrastructure |
| Sen2Agri legacy source | https://github.com/Sen2Agri/Sen2Agri-System | 46b452fa4087246d26e9f02c1cccd3e52d517145 | 2020-11-12 | Related legacy source tree; README itself currently describes Sen4CAP. Treat as shared lineage rather than a separate current benchmark competitor |
| JRC CbM | https://github.com/ec-jrc/cbm | c4bc350bdc323118778a1456ecfbf730efcb08f9 | 2026-06-24 | Modular reference architecture and examples: parcel statistics, marker analysis, analyst notebooks, API endpoints and reports |

WorldCereal LICENSE is MIT; JRC CbM LICENSE is BSD-3-Clause. Sen4CAP/Sen2Agri include GPL-3 COPYING and many bundled third-party notices; inspect actual component licenses before integration. Do not label an entire multi-component stack with an inferred permissive license. Inspiration, external interoperability and independently implemented behavior do not require copying their source.

## Most useful gaps for GeoCrop

### 1. Durable execution state, complete-output checks and selective retries — immediate priority

WorldCereal persists a `CsvJobDatabase`, supports resetting error/start_failed/canceled jobs, surfaces errors from the background job thread and retries transient failures with capped exponential backoff and jitter. The distinction between bad parameters and a transient connection failure is explicit. Sen4CAP's own changelog documents both corrected step-failure propagation and the known hazard of success recorded when no input products exist.

GeoCrop's reviewed `pipeline.Runner._done` checks one recorded fingerprint and path existence; several stages list only one representative output. `_fp` hashes file metadata rather than contents, and nested input-path lists are JSON strings rather than recursively fingerprinted dependencies. A probability raster or JSON companion can disappear/corrupt while the class raster remains, leaving a supposedly reusable stage. It records no durable running/failed state and writes stage JSON in place.

Implement now: atomic stage-state writes, explicit running/completed/failed records, versioned cache schema, deterministic recursive dependency signatures, complete expected output manifests, and regression tests for interrupted runs, missing companions, changed scene contents and S1-only pipelines. Retry network errors selectively; do not retry invalid parameters or authentication denials indefinitely. No need to adopt distributed infrastructure to gain these guarantees locally.

Primary sources:
- https://github.com/WorldCereal/worldcereal-classification/blob/b739085ca786db3a39a802fb3bee6c5a0a5601d5/src/worldcereal/jobmanager.py (`_is_retryable_error`, `_run_jobs_with_retries`, `CsvJobDatabase`)
- https://github.com/Sen4CAP/Sen4CAP/blob/247f496fcf965b54b2c4b888f6ab5c7497ee3fb7/CHANGELOG_SEN4CAP.md (Known issues; version 1.2 failure propagation)

### 2. Parcel evidence and observation sufficiency — immediate or next priority

JRC CbM's `scenario_evidence` exports parcel geometry context, counts of primary markers, data gaps and other markers. Its extraction routines retain count/mean/std/quantiles rather than only a predicted class. Sen4CAP's time-series processor exposes minimum acquisition count (`minacqs`). These support explaining why an observation can or cannot support an administrative comparison.

GeoCrop already has signatures, valid pixel counts, AOA, exploratory prediction sets and traffic lights. Its aggregation consumes probabilities, however, without per-window observation sufficiency, and generic review/accepted states do not record every decision reason. The existing input audit checks geometry/metadata/footprint; it does not scan pixel observation availability.

Implement independently: per-parcel evidence JSON/CSV with stable reason codes (insufficient interior support, outside AOA, low confidence/margin, contradictory decision strategies, insufficient temporal observations), model/config/source identifiers and explicit exclusion of absent evidence. Optionally aggregate window `valid_count` rasters and enforce study-configured observation coverage before acceptance. Missing evidence should remain review, not become an inferred agreement. Add known synthetic sparse-observation cases and verify reason codes survive export.

Primary sources:
- https://github.com/ec-jrc/cbm/blob/c4bc350bdc323118778a1456ecfbf730efcb08f9/scripts/signal_marker_processing/scenario_evidence.py
- https://github.com/ec-jrc/cbm/blob/c4bc350bdc323118778a1456ecfbf730efcb08f9/scripts/extraction/postgisS2Extract.py
- https://github.com/Sen4CAP/Sen4CAP/blob/247f496fcf965b54b2c4b888f6ab5c7497ee3fb7/sen2agri-processors/AgriculturalPractices/TimeSeriesAnalysis/src/otbTimeSeriesAnalysis.cpp

### 3. Spatial/class sampling diagnostics and fair model comparison — immediate priority

WorldCereal's `SeasonalTaskBatchSampler` combines class balancing and spatial-density weights, with sparse-bin safeguards; its diagnostics export the actual sampled class/bin weights and counts. This is stronger transparency than claiming an estimator is balanced because it sets `class_weight`.

GeoCrop caps pixels per polygon and blocks CV, but does not directly report representation by spatial block/class or compare learner selection under nested validation. Useful independent additions are class×block support tables, class coverage checks before expensive fits, and a reproducible multi-estimator benchmark on identical held-out folds, reporting fitting time, memory/time resource settings and calibration. Model-selection results must remain separate from independent final evaluation. Do not promise that a new learner or weighting rule improves real accuracy without evidence.

Primary sources:
- https://github.com/WorldCereal/worldcereal-classification/blob/b739085ca786db3a39a802fb3bee6c5a0a5601d5/src/worldcereal/train/datasets.py (`_get_spatial_density_weights`, `SeasonalTaskBatchSampler`)
- https://github.com/WorldCereal/worldcereal-classification/blob/b739085ca786db3a39a802fb3bee6c5a0a5601d5/src/worldcereal/train/diagnostics.py

### 4. Standardized pretrained embeddings and covariates — later, with real-data benchmark

WorldCereal offers pretrained multimodal features plus custom crop heads. Current README calls the backbone Presto, and current source accesses that Presto model through `prometheo` predictor/inference wrappers. WorldCereal `train/backbone.py` confirms `prometheo.models.Presto` with strict weight loading; this is an interface/package distinction, not a backbone replacement. Implementation details should be tied to a specific tested model artifact. Input contracts include S1, S2, meteorology, DEM, dates and coordinates with explicit validity handling.

GeoCrop presently uses spectral/radar/time-series features and optional TempCNN. Add a versioned external embedding raster/feature adapter before hard-wiring a new heavyweight backbone. Record artifact checksum/license, required bands/timestep normalization, memory needs and training-data overlap limits. Validate on a spatial/year holdout against RF and ExtraTrees. This requires acquiring a trusted model artifact and suitable real observations; it is not responsibly demonstrated by relabeling synthetic benchmark gains.

Primary sources:
- https://github.com/WorldCereal/worldcereal-classification/blob/b739085ca786db3a39a802fb3bee6c5a0a5601d5/README.md
- https://github.com/WorldCereal/worldcereal-classification/blob/b739085ca786db3a39a802fb3bee6c5a0a5601d5/src/worldcereal/train/predictors.py
- https://github.com/WorldCereal/worldcereal-classification/blob/b739085ca786db3a39a802fb3bee6c5a0a5601d5/src/worldcereal/openeo/feature_extractor.py

### 5. Analyst-facing evidence retrieval — later, avoid premature service sprawl

JRC CbM separates data/statistics extraction from analyst interfaces through standard API endpoints and notebooks. Sen4CAP persists markers and exposes them via REST. GeoCrop's open files are a useful local interface but do not yet provide a compact run/parcel evidence index for a paying-agency analyst.

A bounded first step is a static JSON/HTML run report with links to its local COG/STAC/GeoPackage outputs, parcel decision reasons, data-availability charts and diagnostics. A REST service, PostGIS deployment or distributed worker fleet should follow a specific operational requirement, not be added merely to resemble a larger stack. No claim about legal compliance or administrative verdicts follows from software availability.

Primary sources:
- https://github.com/ec-jrc/cbm/blob/c4bc350bdc323118778a1456ecfbf730efcb08f9/README.md
- https://github.com/ec-jrc/cbm/tree/c4bc350bdc323118778a1456ecfbf730efcb08f9/api
- https://github.com/Sen4CAP/Sen4CAP/blob/247f496fcf965b54b2c4b888f6ab5c7497ee3fb7/CHANGELOG_SEN4CAP.md

## Evaluation boundary

Source inspection establishes implemented/documented functions, not measured accuracy, throughput, RAM use, legal suitability or production readiness. No external engine was installed or run for this review. A comparative performance claim needs harmonized data, preprocessing, labels, spatial/year splits and compute budgets. GeoCrop's current synthetic demonstration establishes software integration only. Keep observational study claims pending a real, independently evaluated run.

## Extension: pretrained representation and field-boundary baselines

The following additional official source trees were inspected on the same review date. Only source/documentation was inspected; no model weights, benchmark datasets or remote imagery were downloaded and no third-party model was executed.

| Project | Reviewed source commit and date | What it actually supplies | Code/data/artifact distinctions |
|---|---|---|---|
| NASA Harvest Presto | [`11e207a668a34336ced1d8e492a1bd5849b96c4a`](https://github.com/nasaharvest/presto/tree/11e207a668a34336ced1d8e492a1bd5849b96c4a), 2026-06-10 | Pretrained transformer for remote-sensing time series; pooled embeddings and finetuning; documented CropHarvest/other evaluation tasks | Repository LICENSE is MIT. This source license is not evidence that every upstream satellite dataset, downstream reference label or external checkpoint has identical terms. Record the exact checkpoint and its stated provenance/terms. |
| ESA WorldCereal PromethEO | [`e90f72996ccb2576e6b4d7254a967e1944fe08ad`](https://github.com/WorldCereal/prometheo/tree/e90f72996ccb2576e6b4d7254a967e1944fe08ad), 2026-10-20 | Typed multimodal `Predictors` interfaces, Presto wrapper, training/inference/pooling glue; current HEAD additionally documents optional OlmoEarth support | Repository LICENSE is MIT. PromethEO is a framework/wrapper, not a substitute name for a single pretrained artifact. WorldCereal source inspected above explicitly pins `prometheo@v0.1.6`, tag SHA `72cf6d75826dd57db6b9463ea45e9e93e318d4c9`, rather than moving PromethEO HEAD. Do not attribute HEAD-only features to that pin without verification. |
| NASA Harvest CropHarvest | [`aa3c94a667a737bd14329b53da3b5475607bda5f`](https://github.com/nasaharvest/cropharvest/tree/aa3c94a667a737bd14329b53da3b5475607bda5f), 2024-04-30 | Harmonized agricultural labels plus remote-sensing/climatology features, fixed benchmark tasks/test sets and reference RF/deep-learning methods | Root `LICENSE.txt` is **CC BY-SA 4.0**, and `datasets.md` lists varied constituent licenses (CC BY, CC BY-SA, Etalab, Open Government and others). Do not label this repository or every constituent source MIT/Apache. Keep dataset-specific attribution and distribution decisions explicit. |
| Fields of The World baseline / FTW CLI | [`3f7f1315be11a3665980f63f0ca67f1356876b06`](https://github.com/fieldsoftheworld/ftw-baselines/tree/3f7f1315be11a3665980f63f0ca67f1356876b06), 2026-10-24 | Field-boundary semantic/instance segmentation, training/testing CLI, released model registry, scene selection, tiled overlap inference and polygonization | Repository source is MIT. README explicitly distinguishes checkpoints trained only on CC-BY/equivalent sources from FULL checkpoints and recommends different usage categories; model/data terms must be checked separately. This is a **field delineation** baseline, not a crop-type/declaration classifier benchmark. |

### Concrete adapter requirements

**Presto:** The official README defines `x[batch,time,bands]`, Dynamic World labels `[batch,time]`, latitude/longitude `[batch,2]`, optional missing-value mask of the same shape as `x` (`1` means masked), and starting month. Documented sequence length is 1–24 timesteps. The source normalizes and orders bands through `construct_single_presto_input` / the S1/S2/ERA5/SRTM pipeline; missing Dynamic World can be encoded as class 9 and is ignored. A trustworthy GeoCrop adapter must preserve band ordering, physical units, normalization, timestep/month semantics, coordinate ordering and missing-modality masks. Filling unavailable covariates with zero and calling them observed would not reproduce the pretrained contract. The README distinguishes loading the packaged pretrained checkpoint from constructing a randomly initialized model. A future benchmark must record which was used.

Primary sources:
- https://github.com/nasaharvest/presto/blob/11e207a668a34336ced1d8e492a1bd5849b96c4a/README.md
- https://github.com/nasaharvest/presto/blob/11e207a668a34336ced1d8e492a1bd5849b96c4a/presto/dataops/pipelines/s1_s2_era5_srtm.py
- https://github.com/nasaharvest/presto/blob/11e207a668a34336ced1d8e492a1bd5849b96c4a/LICENSE

**PromethEO / WorldCereal:** Dataset adapters return typed `Predictors` modalities (`s1`, `s2`, `meteo`, `dem`, `latlon`, `timestamps`, optional labels). Missing modalities are `None`. Models return standardized embedding shapes; the README distinguishes pixel time series and spatial patches. The current optional OlmoEarth wrapper requires S2 plus timestamps, supports only a subset of modalities and returns patch-token grids rather than original input pixels. These are materially different contracts and should not share an unchecked raster grid assumption. WorldCereal's `train/backbone.py` specifically imports `prometheo.models.Presto` and strictly loads Presto weights; thus the earlier README/source discrepancy is a change in **packaging/interface**, not evidence that the current WorldCereal backbone ceased being Presto. For an initial GeoCrop experiment use the exact WorldCereal-pinned version and explicit checkpoint hash; evaluate later wrappers separately.

Primary sources:
- https://github.com/WorldCereal/prometheo/blob/e90f72996ccb2576e6b4d7254a967e1944fe08ad/README.md
- https://github.com/WorldCereal/prometheo/blob/e90f72996ccb2576e6b4d7254a967e1944fe08ad/LICENSE
- https://github.com/WorldCereal/worldcereal-classification/blob/b739085ca786db3a39a802fb3bee6c5a0a5601d5/pyproject.toml
- https://github.com/WorldCereal/worldcereal-classification/blob/b739085ca786db3a39a802fb3bee6c5a0a5601d5/src/worldcereal/train/backbone.py

**CropHarvest:** A candidate reproducible public baseline comprises pre-existing task definitions and held-out `test_features`, with task IDs and training-set-size budgets recorded. Its reference RF evaluates three task families (Kenya maize, Brazil coffee, Togo crop/noncrop), multiple shuffle seeds and fixed test data; these are not the same task as six-class crop mapping in Bosnia. Data retrieval uses versioned Zenodo releases; regenerating EO features requires Earth Engine/export infrastructure. Required reference metadata includes dataset identity, collection/export dates and `is_test`, while crop labels can be absent for binary-only observations. Preserve predefined test membership and temporal context. Do not claim independent generalization merely by randomly splitting an aggregated global source, and assess possible overlap with pretrained training/reference corpora before reporting transfer performance.

Primary sources:
- https://github.com/nasaharvest/cropharvest/blob/aa3c94a667a737bd14329b53da3b5475607bda5f/README.md
- https://github.com/nasaharvest/cropharvest/blob/aa3c94a667a737bd14329b53da3b5475607bda5f/data/README.md
- https://github.com/nasaharvest/cropharvest/blob/aa3c94a667a737bd14329b53da3b5475607bda5f/benchmarks/config.py
- https://github.com/nasaharvest/cropharvest/blob/aa3c94a667a737bd14329b53da3b5475607bda5f/benchmarks/random_forest.py
- https://github.com/nasaharvest/cropharvest/blob/aa3c94a667a737bd14329b53da3b5475607bda5f/datasets.md
- https://github.com/nasaharvest/cropharvest/blob/aa3c94a667a737bd14329b53da3b5475607bda5f/LICENSE.txt

**FTW:** The documented semantic baseline consumes eight bands from two aligned Sentinel-2 acquisitions in different seasons: red/green/blue/NIR at t1, followed by the same four at t2. It performs overlapping-patch inference and stitches GeoTIFF predictions; polygonization and optional land-cover filtering are distinct steps. The released registry/checkpoint choice matters. Current docs provide both CPU/GPU inference routes and separate country-based train/validation/test configurations. A fair GeoCrop segmentation comparison must match season selection, cloud/nodata coverage, input resolution, country/field separation, postprocessing and thresholds, and report object/boundary metrics separately from crop classification accuracy. Do not copy quoted README model scores into a GeoCrop comparison table as if measured on shared data.

Primary sources:
- https://github.com/fieldsoftheworld/ftw-baselines/blob/3f7f1315be11a3665980f63f0ca67f1356876b06/README.md (FTW semantic input ordering; overlapping inference; CC-BY versus FULL checkpoints)
- https://github.com/fieldsoftheworld/ftw-baselines/blob/3f7f1315be11a3665980f63f0ca67f1356876b06/EXPERIMENTS.md
- https://github.com/fieldsoftheworld/ftw-baselines/blob/3f7f1315be11a3665980f63f0ca67f1356876b06/LICENSE

### What would establish a defensible improvement claim

1. Freeze a task, data release, label taxonomy and spatial/year split before model selection. Use CropHarvest's official tasks for an external public benchmark and a separate local observation set for the Bosnia study; neither substitutes for the other.
2. Compare hand-crafted GeoCrop RF/ExtraTrees against pretrained-embedding-plus-identical-head and finetuned-backbone variants under matched reference-data budgets. Include a randomly initialized backbone where feasible to separate architecture from pretraining effects.
3. Keep selection/calibration observations separate from final evaluation. Record pretrained provenance and any known label/geographic/time overlap; do not call an opaque pretrained corpus demonstrably overlap-free.
4. Report per-class support/accuracy, uncertainty/calibration, abstention burden, fitting/inference time, peak memory, hardware, input availability and preprocessing costs—not just a single best F1.
5. For delineation, use FTW's country/field splits and boundary/object metrics with fixed postprocessing. Segmentation benchmarks do not establish declaration-verification accuracy or cadastral/legal boundary correctness.

These are source-grounded requirements for future benchmarks, not measured results. The present local synthetic suite cannot establish leadership over Presto, WorldCereal, FTW or any other real-data system.
