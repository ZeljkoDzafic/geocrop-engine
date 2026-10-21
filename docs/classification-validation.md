# Classification and validation

## Two kinds of accuracy
| | Internal (spatial CV) | External (independent validation) |
|---|---|---|
| Data | pixels of interpreted reference polygons | field observations never used in training, delineation or tuning |
| Purpose | model selection, ablation, tuning | accuracy reported as the result of the study |
| Command | `geocrop train`, `geocrop ablation` | `geocrop validate`, `geocrop area` |

Never present internal CV scores as map accuracy; never tune on the external set.

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

## Prediction
Block-wise (`512 × 512` default). Outputs: class index (`uint8`), per-class probabilities, and an
uncertainty raster with maximum probability, top-2 margin and normalised entropy (`H / ln K`). Random Forest
probabilities are vote fractions, **not calibrated** confidences; check the reliability bins reported in
training and validation before interpreting them as probabilities.

## Experiments
`geocrop ablation` evaluates each feature subset on the *same* samples and folds and adds a McNemar test
against the first experiment (out-of-fold predictions; pixels are autocorrelated, so treat p-values as
indicative and confirm with point- or parcel-level comparisons). Default ladder:

| ID | Features |
|---|---|
| E0 | Sentinel-2 bands |
| E1 | + spectral indices |
| E2 | + temporal deltas and seasonal statistics |
| E3 | Sentinel-1 only |
| E4 | Sentinel-1 + Sentinel-2 (all groups) |
| E5/E6 | fusion restricted to early windows (P1–P2, P1–P3) — earliest reliable identification |

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

## Area estimation
Pixel counting is biased by classification errors. When the validation sample is a probability sample
stratified by map class, `geocrop area --classes classes.tif --validation validation_report.json` reports
Olofsson et al. (2014) error-adjusted areas with standard errors and 95 % CIs, plus UA/PA/OA with SEs.
For convenience samples (e.g. opportunistic field visits) report only the confusion matrix.

## References
Olofsson et al. (2014) RSE 148:42–57 · Roberts et al. (2017) Ecography 40:913–929 ·
Ploton et al. (2020) Nat. Commun. 11:4540 · Schulthess et al. (2023) Remote Sens. 15:608 ·
Breiman (2001) Mach. Learn. 45:5–32 · McNemar (1947) Psychometrika 12:153–157 · Foody (2004) PE&RS 70:627–633.
