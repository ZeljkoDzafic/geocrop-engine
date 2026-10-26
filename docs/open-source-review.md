# Open-source comparison and development priorities

Reviewed 2026-10-25 from official source repositories at recorded commits. This is a broad,
selected ecosystem review, not a survey of every project or a measured accuracy/speed comparison.
Sources below describe different layers of the workflow; a raster loader is not a competing
declaration-verification system. No competitor workloads, pretrained weights or datasets were run.

| Area | Systems inspected | Evidence that informed this development round |
|---|---|---|
| Operational crop mapping | WorldCereal, Sen4CAP, legacy Sen2Agri, JRC CbM | Durable job states, failure recovery, acquisition sufficiency and parcel evidence |
| EO workflow and datasets | eo-learn, TorchGeo, openEO Python client | Explicit outcomes, spatial data contracts, reusable datasets and execution budgets |
| STAC raster loading | StackSTAC, odc-stac | Correct grids, categorical/continuous resampling, scale/offset and catalogue metadata |
| Time-series processing | sits, FORCE | Separate observed support from interpolation, documented uncertainty and memory budgets |
| Evaluation | blockCV, MAPIE | Deployment-aware spatial splits and fixed-predictor held-out calibration |
| Pretrained models and benchmarks | Presto, PromethEO, CropHarvest, Fields of The World | Versioned model/data contracts, fixed task splits and distinct crop-type/boundary tasks |

## Implemented in this development round

1. **Durable local execution.** Versioned stage state records running/success/failed outcomes and is
   replaced atomically. A failed forced rerun cannot reuse its earlier success record. Complete
   companion outputs and their identities are checked before reuse, and recursive scene dependencies,
   small metadata contents, seeds, legends, mappings and engine source changes invalidate caches.
   See [reproducibility](reproducibility.md) for the identity policy and its limits.
2. **Genuine held-out calibration.** `model.calibration_mode: holdout` reserves spatial groups before
   tuning, CV, temperature fitting and final predictor fitting. Set thresholds use a frozen predictor
   that never fits those calibration objects. Sparse classes use conservative inclusion. This is a
   separate calibration design, not proof of exchangeability or independent parcel coverage.
   See [uncertainty](uncertainty.md).
3. **Reproducible classifier comparison.** `geocrop benchmark` compares named estimator settings on
   identical samples and buffered spatial folds, saves sample/fold fingerprints, metrics, timings and
   explicit failed candidates. It supports internal selection; it does not rank GeoCrop against other
   engines or replace independent field evaluation.
4. **Stricter STAC output contracts.** Export checks actual files, grids, legends and dates, records
   per-band raster metadata and uses correctly resolved asset paths. This improves downstream
   interoperability; preprocessing quality still needs real-scene validation.

These changes are independently implemented. No third-party implementation was copied. Licences for
future dependencies, datasets and model weights must be checked individually; code, weights and data
can have different terms.

## What must be measured before a superiority claim

Use a versioned observational dataset with fixed spatial/year holdouts and declared input access.
Freeze preprocessing, crop taxonomy, calibration/tuning roles and compute budgets. Compare class and
parcel outcomes, confidence intervals, calibration, set coverage, review burden, wall time, peak memory
and recovery from interruption. Keep model selection separate from the final test. CropHarvest offers
defined crop-learning tasks; Fields of The World evaluates field boundaries and cannot alone establish
crop-type/declaration accuracy. The local synthetic demo establishes software behavior only.

## Priorities identified by the review: status

The review named four follow-up priorities. All four now exist in the engine; what is still missing is
real-data evidence, not code:

* **Temporal-support QA rasters** (observed counts, occupied bins, longest unsupported gaps, kept separate from
  classifier features unless a study opts in): implemented, off by default (`timeseries.temporal_support`,
  `cbm.temporal_support`, `geocrop temporal-support`; [temporal quality](temporal-quality.md)).
* **STAC-to-scene import with explicit band/radiometry mapping**: `geocrop stac-import` and
  `geocrop cube import --band-map` ([datacube](datacube.md)).
* **Pinned external embedding adapters**: Presto, Galileo and AnySat weights pinned by URL and SHA-256
  ([embeddings](embeddings.md)). **Open:** a real-data comparison of embeddings against RF/ExtraTrees on
  handcrafted features. WorldCereal's reviewed version uses Presto through a pinned PromethEO wrapper; current
  framework features must not be attributed automatically to older pinned versions.
* **Parcel evidence/reason-code reports, deployment-distance diagnostics and analyst interfaces**:
  `geocrop evidence` ([evidence](evidence.md)), the Area of Applicability ([uncertainty](uncertainty.md)) and
  the analyst review view of the web workspace ([review](web-interface.md#review-analyst-parcel-decisions)).

## Primary-source evidence

Full source links, exact commits, baseline observations and limitations are preserved in:

* [Operational systems, pretrained models and datasets](reviews/operational-2026-10-25.md)
* [Python EO workflows and STAC tooling](reviews/python-eo-2026-10-25.md)
* [Time-series quality, spatial validation and calibration](reviews/timeseries-2026-10-25.md)

The gap descriptions in those source-review notes refer to GeoCrop before this development round;
the implemented list above records the subsequent changes.
