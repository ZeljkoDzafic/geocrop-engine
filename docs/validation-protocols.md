# Validation protocols: spatial CV, leave-one-region-out, leave-one-year-out

Different questions need different held-out data. A crop map that is reused next season, or in a
municipality without reference polygons, faces a *domain shift* (other weather, phenology, soils, management,
image acquisition dates) that ordinary spatial cross-validation does not measure. GeoCrop supports three
internal protocols on the same samples and one external one:

| Protocol | Held out per fold | Question it answers | CLI / API |
|---|---|---|---|
| Spatial-block CV (default) | groups of square blocks (default 5 km) | how well does the model map *unsampled places inside the sampled region and season*? | `geocrop train`, `model.cross_validate` |
| Leave-one-region-out (LORO) | one whole region | how well does it map a *region with no reference data* in a sampled season? | `--cv-strategy region`, `geocrop transfer --strategies region` |
| Leave-one-year-out (LOYO) | one whole season | how well does last season's model map *a new season* in the same regions? | `--cv-strategy year`, `geocrop transfer --strategies year` |
| Region-year transfer | one region-season cell; training excludes that region **and** that season | the hardest case: new place *and* new season | `--cv-strategy region_year` |
| Independent field validation | parcels never used for training | the accuracy statement for the map | `geocrop validate` ([classification](classification-validation.md)) |

All internal protocols use interpreted reference polygons. They are diagnostics for model design and
reuse decisions, not independent accuracy assessments.

## When to use which

* **Spatial-block CV** for feature/model selection within one study season and area. Choose the block
  size larger than the range of spatial autocorrelation of the residuals (field size and management units);
  `geocrop folds` exports the folds for inspection. Add `spatial_buffer_m` when adjacent blocks still
  share fields or field edges.
* **LOYO** whenever a model, its features or its thresholds are meant to be reused in a later season, which
  is the normal case for paying agencies. It needs at least two seasons of samples with an identical feature
  manifest (see below). With two seasons it gives two folds; report both, not just the mean.
* **LORO** when the map covers regions without reference polygons, or when reference data are clustered
  (e.g. one municipality per field campaign). Regions should be agronomically meaningful units
  (municipalities, agro-ecological zones), not arbitrary tiles; very small regions give noisy per-group scores.
* **Region-year** only with at least two regions and two seasons; cells without data from another region
  *and* another season cannot be evaluated and are listed in `groups_not_evaluable`.
* Hyperparameter tuning under LORO/LOYO (`--tune` with a transfer strategy) is nested: the outer loop holds
  out one region/season, the inner loop tunes by spatial-block CV on the remaining data only, so the held-out
  domain never influences the selected settings (`transfer_validation.nested` in the report).

## Preparing samples with group ids

Region and year ids are polygon attributes copied to every sampled pixel; all pixels of a polygon must
share one id (enforced).

```bash
# one season, region attribute "municipality", constant year
geocrop samples --features stack_2023.tif --polygons ref_2023.gpkg --out s2023.npz \
    --region-field municipality --year 2023
geocrop samples --features stack_2024.tif --polygons ref_2024.gpkg --out s2024.npz \
    --region-field municipality --year 2024
# multi-year sample store; --stacks makes the feature manifests part of the consistency check
geocrop samples-merge --inputs s2023.npz s2024.npz --stacks stack_2023.tif stack_2024.tif --out all.npz
```

Python: `samples.extract(..., region_field=..., year_field=... | year=...)` and
`samples.combine([s2023, s2024], years=[2023, 2024], manifests=[...])`.

`samples-merge` refuses (with the offending feature names) sample sets whose feature names, order, CRS or
manifest definitions (group, sensor, period) differ. Use **season-relative window names** (`early`, `peak`,
`late`, not `2023-05`) so the same feature means the same phenological stage every year. Polygon ids are
prefixed with the year (`2024:p17`) so a field observed in two seasons is two reference objects; because both
lie in the same spatial block, spatial-block CV still keeps the two seasons of a field together.

## Running

```bash
# transfer study next to spatial CV (writes transfer.csv and transfer.json)
geocrop transfer --samples all.npz --out out/transfer --strategies region year --config study.yaml
# add transfer validation to a training run (train_report.json -> transfer_validation, transfer_per_group.csv)
geocrop train --samples all.npz --out out/model --cv-strategy year
# score every ablation experiment out-of-domain as well (transfer_* columns in ablation.csv)
geocrop ablation --samples all.npz --features stack_2024.tif --out out/ablation --cv-strategy region
```

In `geocrop train` the default outputs (spatial `cross_validation`, temperature, conformal thresholds, AOA)
are computed exactly as before; the transfer strategy only adds `transfer_validation`. In YAML,
`cv.strategy` is read by `geocrop ablation --config`; the `geocrop run` pipeline does not read the `cv`
section yet.

## What is reported

For each strategy:

* `in_domain`: spatial-block CV metrics (overall accuracy, kappa, macro-F1) on the samples that were also
  evaluated out-of-domain;
* `out_of_domain`: pooled leave-one-group-out metrics with confusion matrix and per-class F1;
* `transfer_gap`: in-domain minus out-of-domain (positive = accuracy lost in an unseen domain);
* `per_group`: one row per held-out region/season with its own in-domain and out-of-domain scores and gap,
  test and training sizes, the classes present in the test group and the classes **missing from training**
  (a class that exists only in the held-out region can never be predicted there);
* `group_macro_f1_mean`/`_std` (every group weighted equally) and `worst_group`.

Both regimes are scored on identical samples, so the gap is not confounded by different test sets.

## Reporting guidance for papers

1. State the protocol in the abstract-level result: "macro-F1 0.xx (spatial-block CV, 5 km blocks)"
   is a different claim from "0.xx when transferred to the following season (LOYO)".
2. Report in-domain and out-of-domain side by side and the gap; give per-group results, not only the pooled
   value, and name the worst group. With few groups (two seasons, three regions) avoid standard errors across
   groups; show every group instead.
3. Report the number of polygons per group and per class, and classes missing from training per fold.
4. Report block size, buffer, number of folds and seed; the JSON outputs contain them.
5. Keep hyperparameter selection inside the training data of each fold (nested), or state that it was not.
6. Treat all of these as internal estimates. The accuracy of a published map comes from independent,
   probability-sampled field observations with area-adjusted estimators (`geocrop validate`).

## Probability smoothing before argmax (post-processing)

`geocrop predict --smooth [--smooth-window 7 --smoothness 10]` (or `predict.smoothing` in YAML with
`--config`) smooths the class probabilities before the class map, the uncertainty layers and any parcel
aggregation are derived. Each class probability is converted to log-odds; in a moving window the local mean and
variance over valid pixels act as a prior and the pixel's own value as the observation with variance
`smoothness`, giving `(v·x + s·m)/(v + s)`. Homogeneous field interiors (small local variance) are pulled
towards their neighbourhood; across field edges (large variance) pixels keep their own value. This is an
independent implementation of the idea behind the Bayesian smoothing of the sits package. Raw probabilities
are kept in `probabilities_unsmoothed.tif`; blocks are processed with a halo so results do not depend on the
block size.

Conformal thresholds in `conformal.json` are fitted on unsmoothed out-of-fold probabilities, so their
coverage on smoothed maps is **not** established; `prediction_manifest.json` records this.

Synthetic comparison (reproduce with `geocrop smoothing-experiment --out out/smoothing --seed 42`): three
independent 160 × 160 px synthetic maps (training, conformal calibration, test) with 10 × 10 px square parcels,
4 classes, 6 features with strong i.i.d. pixel noise; Random Forest with 100 trees; parcel-level split-conformal
sets at α = 0.1 (target coverage 0.90), calibrated either on probabilities processed the same way ("own") or on
unsmoothed probabilities.

| Variant | window | smoothness | pixel OA | edge-pixel OA | parcel OA | coverage (own) | coverage (unsmoothed thresholds) | mean set size (unsmoothed thresholds) |
|---|---|---|---|---|---|---|---|---|
| unsmoothed | – | – | 0.654 | 0.654 | 0.918 | 0.945 | 0.945 | 0.945 |
| smoothed | 5 | 10 | 0.850 | 0.786 | 0.898 | 0.922 | 0.992 | 1.004 |
| smoothed | 7 | 10 | 0.838 | 0.776 | 0.895 | 0.914 | 0.973 | 0.980 |
| smoothed | 7 | 1 | 0.724 | 0.699 | 0.910 | 0.922 | 0.957 | 0.965 |

**Synthetic data only.** On this design smoothing strongly improves pixel accuracy, also on parcel-edge
pixels, but slightly lowers parcel accuracy after mean-probability aggregation (parcel means already average
the noise, and smoothing mixes in probability from neighbouring parcels). Thresholds fitted on unsmoothed
probabilities gave higher, more conservative coverage with larger sets on smoothed probabilities; recalibrating
on smoothed probabilities restored coverage near the target. Whether smoothing helps on real Sentinel-2 maps
with irregular parcels must be tested on independent field data; for parcel-level decisions it is not
recommended by default.
