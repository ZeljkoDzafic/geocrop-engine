# Classification and validation

## Two kinds of accuracy
| | Internal (spatial CV) | External (independent validation) |
|---|---|---|
| Data | pixels of interpreted reference polygons | field observations never used in training, delineation or tuning |
| Purpose | model selection, ablation, tuning | accuracy reported as the result of the study |
| Command | `geocrop train`, `geocrop ablation` | `geocrop validate`, `geocrop area` |

Never present internal CV scores as map accuracy; never tune on the external set.

## Audit study inputs
Before sampling, run a read-only audit of the actual inputs:

```bash
geocrop audit --training training.gpkg --parcels parcels.gpkg --observations observations.gpkg \
  --features stack.tif --exclusion-m 50 --out outputs/input-audit.json
```

Omit inputs that are not available yet. The JSON report separates errors from warnings and reports
schema/geometry problems, class support, overlapping training labels, field observations near training
polygons and vector/stack coverage. Errors exit with status 2; warnings need analyst review. Use
`--label-field`, `--id-field`, `--parcel-id-field` and `--observation-label-field` for custom schemas.
Distances are computed in a metric projection; `--distance-crs` can specify the study projection.
The audit does not repair inputs, prove observation independence, or evaluate pixel validity throughout
the raster. Inspect actual composites and the sampling/validation selection reports as well.

## Sampling
`geocrop samples` reads each polygon's bounding window, keeps pixels whose centres are inside the polygon
(optionally after `--inner-buffer-m` to drop mixed edge pixels) and caps large polygons at
`max_samples_per_polygon` (default 250) so they do not dominate a class (Schulthess et al., 2023). All
experiments reuse the same `samples.npz`.

## Spatial cross-validation
* Every pixel inherits the ID of its polygon; polygons are grouped into square **spatial blocks**
  (`spatial_block_m`, default 5 km) by the block containing their centroid, so no polygon and no block is
  split between training and test.
* Folds are built with `StratifiedGroupKFold`, which balances classes across folds while respecting groups;
  every fold reports classes missing from its training part. Use `geocrop folds` to inspect folds on a map first.
* Adjacent blocks can still contain nearby training and test pixels. Optional `model.spatial_buffer_m`
  (default 0, disabled) removes an entire training polygon from a fold if any of its sampled pixels is
  within that distance of a test sample. `geocrop train --buffer-m 100` enables a 100 m exclusion;
  use the same buffer for `ablation` and `reduce-features`. Distances require a projected CRS in metres.
  Reports record exclusions and the nearest retained training/test sample distance per fold. Test samples
  remain fixed, so every sample still receives an out-of-fold prediction. A buffer that removes all training
  samples or eliminates a previously represented training class fails with an actionable error.
  This controls separation of sampled support, not unsampled polygon boundaries. Choose and report the
  buffer based on the study design and autocorrelation diagnostics. The final mapping model still uses
  all training samples. `geocrop folds` is a polygon-level planning view; actual training folds use sampled
  pixel centroids and sample-level class stratification and may differ.
* Choose the block size larger than the range of spatial autocorrelation of the residuals where possible
  (Roberts et al., 2017; Ploton et al., 2020); report it.
* Metrics are pooled over out-of-fold predictions: OA, Cohen's κ, macro-F1, weighted F1, per-class
  producer's/user's accuracy and F1, fold mean ± std, polygon-level bootstrap CIs and ECE.

## Random Forest
Defaults: 500 trees, `max_features="sqrt"`, `class_weight="balanced_subsample"`, `min_samples_leaf=1`.
`tune_grid` (YAML) or `--tune` (CLI) runs an exhaustive grid scored by spatial-CV macro-F1; the winning
score is optimistic (selected on the same folds), which is why external validation remains mandatory.
The final model is refitted on all samples and stored with its ordered feature list; prediction refuses a stack
whose bands do not match.

With `model.calibration_mode: holdout`, the mapping model instead fits only proper-training samples;
reserved spatial groups never enter tuning/CV or the final fit. It remains frozen for calibration and
deployment. See [the calibration protocol](uncertainty.md).

## Compare classifiers reproducibly

```bash
geocrop benchmark --samples samples.npz --out outputs/benchmark --config study.yaml \
  --algorithms rf extra_trees hist_gb --params '{"n_estimators": 100, "n_jobs": 2}'
```

All candidates use identical samples, held-out folds, seeds and configured spatial buffer.
`benchmark.json` and `.csv` retain parameters, per-class/fold metrics, calibration diagnostics,
CV wall time, sample fingerprints and fold index hashes. This is raw internal CV comparison;
it does not fit mapping models or calibrate prediction sets. Missing optional dependencies or invalid
candidate settings are recorded as failed candidates; an incomplete CLI run exits 2. Inspect every
status rather than selecting a winner from only successful rows. Timing depends on this machine and
includes fitting/prediction; equal tree counts do not imply equal compute across algorithms.

Choosing an algorithm from these scores is model selection. Keep a separate untouched spatial/year
test set for final evaluation, especially when comparing many algorithms or hyperparameters. This
runner does not establish accuracy or speed superiority over other software. See the
[source comparison and benchmark requirements](open-source-review.md).

## Prediction
Block-wise (`512 × 512` default). Outputs: class index (`uint8`), per-class probabilities, and an
uncertainty raster with maximum probability, top-2 margin and normalised entropy (`H / ln K`). Raw Random Forest
probabilities are vote fractions, not calibrated confidences. With `model.calibrate: true` (default) `predict`
writes temperature-scaled probabilities ([uncertainty](uncertainty.md)); the scaling is fitted on the training
domain only, so check the reliability bins reported in training and validation before interpreting them as
probabilities.

## Experiments
`geocrop ablation` evaluates each feature subset on the *same* samples and folds and adds a McNemar test
against the first successful experiment. Each reference polygon contributes one majority-vote prediction,
with lexical tie-breaking, rather than one trial per pixel. The CSV/JSON record the unit count, raw p-values
and Holm-adjusted p-values (`mcnemar_p_holm`); the baseline self-comparison is excluded from the correction
family, and OOF and holdout comparisons form separate families. Distinct polygons may still be spatially
correlated: these internal diagnostics require independent-unit justification and do not replace external
field validation. Old pixel-level p-values are not directly comparable to these polygon-level results. Default ladder (`experiments.DEFAULT_EXPERIMENTS`,
used when the configuration has no `experiments:` section):

| ID | Features |
|---|---|
| E0 | Sentinel-2 bands |
| E1 | + spectral indices |
| E2 | + temporal deltas and seasonal statistics |
| E3 | Sentinel-1 only |
| E4 | Sentinel-1 + Sentinel-2 (all window groups) |
| E5–E7 | only when the stack has an `emb` group: handcrafted only, embeddings only, both ([embeddings](embeddings.md)) |

Study-specific rows go in the YAML `experiments:` section; for example [`configs/rs_north_2023.yaml`](../configs/rs_north_2023.yaml)
adds fusion restricted to early windows (P1–P2, P1–P3) for the earliest reliable identification.

`geocrop reduce-features` produces an accuracy-vs-number-of-features curve (recursive elimination by
impurity importance), and `geocrop.experiments.group_importance` gives permute-and-retrain importance per
feature group or window.

## Independent validation
`geocrop validate --prediction outputs/run/prediction --observations obs.gpkg --year 2023 --training training.gpkg --exclusion-m 50 --window-px 1`

1. keeps point observations from the classification season only (crop rotation);
2. removes observations inside, or within `--exclusion-m` of, any training polygon and reports how many;
3. optionally takes the majority class in a (2r+1)² window to absorb GNSS error near field edges;
4. maps field labels with `--label-mapping` and lists observed classes the model does not know;
5. reports the confusion matrix, OA, κ, macro-F1, per-class UA/PA/F1, bootstrap CIs, reliability bins and a
   **reject curve** (accuracy vs. share of points sent to review as the confidence threshold rises).

Use `--group-field parcel_id` (YAML `validation.group_field`) when multiple observations belong to the same
parcel, or provide a suitable spatial-block identifier. The bootstrap resamples whole groups and records the
field name and independent-unit count. At least two evaluated groups are required for cluster bootstrap;
`--bootstrap 0` (YAML `bootstrap: 0`) permits descriptive accuracy for smaller samples. The default resamples individual points and
therefore assumes independent observations. Grouping does not repair convenience sampling or establish that
neighbouring groups are independent. Accuracy remains weighted per evaluated observation; grouping changes
resampling, not the target metric to equal-weight parcel accuracy.

For a majority window, `decision_confidence` is the winning class's share of valid window votes; it is a spatial
agreement diagnostic, not a calibrated probability. The reject curve uses that confidence and that majority
decision. With `window_px: 0`, confidence is the probability of the predicted class. Central-pixel probability
calibration is explicitly reported as a separate diagnostic; invalid central probabilities are counted and
excluded from it without discarding a valid majority decision. Probability rasters must share the class grid
and have uniquely named bands matching the legend. The protocol and confidence support count are retained in
`validation_report.json`, with grouping identifiers in `observations_evaluated.csv` when configured.

`geocrop benchmark-models` uses the same polygon-majority comparisons and separate OOF/holdout Holm families.
Paired tests use the first repeat only; repeated seeds summarize accuracy variation rather than create additional
independent trials. Nested-tuned candidates without OOF predictions have no OOF paired test; the report records
the actual OOF and held-out reference model names separately.

## Area estimation
Pixel counting is biased by classification errors. When the validation sample is a probability sample
stratified by map class, `geocrop area --classes classes.tif --validation validation_report.json` reports
Olofsson et al. (2014) error-adjusted areas with standard errors and 95 % CIs, plus UA/PA/OA with SEs.
For convenience samples (e.g. opportunistic field visits) report only the confusion matrix.

## References
Olofsson et al. (2014) RSE 148:42–57 · Roberts et al. (2017) Ecography 40:913–929 ·
Ploton et al. (2020) Nat. Commun. 11:4540 · Schulthess et al. (2023) Remote Sens. 15:608 ·
Breiman (2001) Mach. Learn. 45:5–32 · McNemar (1947) Psychometrika 12:153–157 · Foody (2004) PE&RS 70:627–633.

## Nested CV and locked final holdout (0.9)
* **Nested spatial CV.** With a tuning grid, `train_report.json → nested_cv` reports outer-fold scores of the
  *whole tuning procedure*: hyperparameters are selected by an inner spatial CV that sees only the outer
  training blocks (Varma & Simon, 2006). The re-run `cross_validation` with the selected parameters is kept for
  calibration and diagnostics and is labelled as optimistic. The same spatial buffer applies in both loops.
* **Locked holdout.** `model.holdout_fraction` (CLI `--holdout`) removes whole, class-stratified spatial blocks
  *before* the calibration split, tuning, ablation or benchmarking. They are scored exactly once
  (`report["holdout"]`, `holdout_predictions.csv`, `holdout_*` columns in ablation and benchmark tables with a
  McNemar test on the holdout). In `oof` calibration mode the mapping model is then refitted on all samples;
  in `holdout` calibration mode the predictor stays frozen so that conformal thresholds remain bound to it.
* **Extended benchmark.** `geocrop benchmark-models` compares RF, ExtraTrees, HistGB, LightGBM and CatBoost on
  identical folds over repeated seeds (mean ± SD), with optional nested tuning per model, locked holdout,
  training time, prediction throughput and McNemar tests against the reference model; the fold signature is
  recorded. `geocrop benchmark` remains the strict single-run fingerprinted report.
