# Domain shift: detection, honest coverage statements and safeguards

A crop-type model trained and calibrated in one region and season can still classify well in another region
or year while its **uncertainty statements stop holding**. Split-conformal prediction sets reach their nominal
coverage (e.g. 90 %) only when calibration and target parcels are exchangeable. A new area or season breaks that
assumption. The public Slovenia transfer benchmark ([v1](benchmarks/si-transfer-pomurje.v1.md)) shows it: coverage
was 89.7 % in-domain and 82–85 % in Posavje, Gorenjska and Pomurje 2025. In Posavje the area of applicability (AOA)
flagged almost nothing, even though accuracy dropped.

`geocrop.shift` adds three things:

1. a **shift check** at prediction time (`prediction/shift_report.json`) with a verdict `in-domain | mild | severe`
   and the reasons for it;
2. **honest coverage statements**. Under any detected shift the conformal coverage is reported as *not established*
   in the prediction manifest, the parcel table (`conformal_coverage` column), the parcel summary, the evidence
   packages and the web review (which shows the parcel attributes and reason codes). A 90 % figure is never
   printed silently;
3. optional **safeguards**: routing parcels to review or withholding model decisions (`shift_policy`), and
   shift-adjusted conformal thresholds (`shift_conformal`).

## The shift check

Training (`geocrop train` / the train stage) writes `model/domain_reference.npz`. It holds a sample of up to 4,000
labelled training pixels, with the calibration objects' pixels first (at most 10 per object). It also holds the
calibration objects' class probabilities and labels, the cross-validated training dissimilarity indices, feature
importances and the training/calibration region and year ids (when `cv.region_field`/`year_field`/`year` is set).

At prediction, the pipeline samples up to `shift_sample_pixels` pixels of the feature stack. It samples inside the
parcels (`paths.parcels`) when they exist, because those are the objects that get prediction sets. Otherwise it
uses random windows of the whole map, which include non-agricultural land, so the signals are less specific. Six
signals are computed, and each is graded 0 (none), 1 (mild) or 2 (severe):

| Signal | What it measures | Mild ≥ | Severe ≥ |
|---|---|---|---|
| `aoa` | share of target pixels outside the AOA ([Meyer & Pebesma 2021](uncertainty.md)) minus the share of training samples outside it under spatial CV | 0.05 | 0.20 |
| `dissimilarity` | median dissimilarity index of the target / median cross-validated training DI | 1.25 | 2.0 |
| `domain_classifier` | ROC AUC of a classifier separating reference from target pixels (classifier two-sample test, Rabanser et al. 2019); out-of-fold, folds grouped by polygon/parcel; reference re-weighted to the target's *predicted* class mix | 0.70 | 0.85 |
| `feature_smd` | importance-weighted mean absolute standardised mean difference (Austin 2009), with the same class-mix re-weighting | 0.20 | 0.50 |
| `class_prior` | total-variation distance between the predicted class shares of target and reference (label shift; Lipton et al. 2018) | 0.15 | – |
| `metadata` | prediction `year`/`region` (config `uncertainty.year`/`region`) not among the calibration years/regions | differs | – |

The verdict is the highest grade. The limits were fixed before running the transfer benchmark. They are
conventions (for example, |SMD| 0.2/0.5 are Cohen's small/medium effects), not tuned values, and they can be
overridden with `uncertainty.shift_thresholds`.

Design choices:

- **Class-mix adjustment.** A region with more grassland differs in P(x) even when every class looks the same.
  The domain classifier and SMD therefore re-weight the reference pixels by target/reference predicted class
  share. The unadjusted AUC is reported too. Mondrian (class-conditional) sets are not affected by pure label
  shift, so `class_prior` alone gives at most *mild*.
- **Grouped folds.** Pixels of the same parcel are correlated. Random pixel folds would let the domain
  classifier recognise parcels and report shift between two samples of the same domain. The folds are therefore
  grouped by polygon/parcel.
- **Metadata counts.** Another year or region makes exchangeability with the calibration parcels implausible,
  whatever the feature statistics say. The coverage claim is then withdrawn (*mild*).

The signals are diagnostics with fixed limits. They are not hypothesis tests with error control. No detection is
**not** proof of exchangeability: a shift in P(y | x) (a crop that looks different under the same features) is
invisible to every feature-based check.

## Coverage statements

| Verdict | `conformal_coverage` |
|---|---|
| `in-domain` | `target 90 % assumes exchangeable calibration and target parcels (no domain shift detected)` |
| `mild` / `severe` | `not established (domain shift); sets: none\|conservative\|weighted (shift-adjusted)` |
| check unavailable (model without `domain_reference.npz`, too few pixels) | `target 90 % assumes exchangeable ... (not checked for domain shift)` |

Coverage stays *not established* even with adjusted thresholds (see below).

## Safeguards (`uncertainty` section)

```yaml
uncertainty:
  shift_detection: true          # write prediction/shift_report.json (needs model/domain_reference.npz)
  shift_policy: warn             # warn | review | abstain
  shift_conformal: none          # none | conservative | weighted
  shift_sample_pixels: 5000
  shift_test_weight_quantile: 0.9
  region: null                   # region id of the prediction domain
  year: null                     # year of the prediction domain
  shift_thresholds: {}           # e.g. {domain_auc: [0.75, 0.9]}
```

- `warn` (default): report and coverage statements only. Decisions and traffic lights do not change.
- `review`: under a mild or severe shift, every parcel whose evidence relies on the model (not grey) becomes
  **yellow** with reason code `DOMAIN_SHIFT`. A red light that also rests on non-model evidence (calendar or
  marker rules) stays red and gets the code. The existing AOA routing (`aoa_below_threshold`,
  `aoa_missing_support`) is unchanged.
- `abstain`: as `review`. Under a **severe** shift, parcels outside the AOA (`aoa_fraction` below
  `parcels.min_aoa_fraction` or unknown) also turn **grey** with `DOMAIN_SHIFT` and `model_decision_withheld = true`.
  The predicted class stays in the table for audit, but it is not a decision.

Adjusted thresholds are used for the parcel sets only when a shift is detected:

- `conservative`: for each class, the larger of the class-conditional and the pooled split-conformal quantile at
  level 1 − α/2. This is a heuristic safety margin with **no coverage guarantee** under shift.
- `weighted`: covariate-shift weighted split conformal (Tibshirani et al. 2019), applied per class (Mondrian).
  Each calibration score gets the estimated likelihood ratio w(x) = p/(1 − p) of the class-mix adjusted domain
  classifier (mean over the object's sampled pixels). The test point's weight is the 0.9 quantile of the target
  parcels' weights, instead of each parcel's own weight. That is more conservative for most parcels and gives one
  threshold per class. The guarantee of Tibshirani et al. needs (i) pure covariate shift, P(y | x) unchanged, and
  (ii) the true likelihood ratio. Neither holds here: the ratio is estimated, the test weight is shared, and a new
  season changes P(y | x) as well. **Coverage is therefore still reported as not established.** The report
  includes the effective calibration sample size, (Σw)² / Σw². When it is small, the weighted quantile is
  unstable and often 1, which means every class is included.

Barber et al. (2023) bound the coverage loss of (weighted) conformal prediction by the total-variation distance
between calibration and target distributions. This is the formal reason a large detected shift withdraws the
nominal level.

## Evaluation on the public transfer protocol

[Results v1.1](benchmarks/si-transfer-pomurje.v1.1.md) use the frozen protocol
`configs/benchmarks/si_transfer_pomurje.v1.yaml` with its hash unchanged. The page reports the shift verdicts for
the in-domain control and the four target domains, and coverage and mean set size for `none`, `conservative` and
`weighted`. Reproduce with:

```bash
geocrop benchmark-transfer --protocol configs/benchmarks/si_transfer_pomurje.v1.yaml --work outputs/transfer-v1
```

## Limitations

- The limits are a-priori conventions. They are not calibrated false-alarm rates. A domain classifier on about
  100 features separates almost any two regions or years, so in practice "severe" mostly means "do not trust the
  nominal coverage". It does not say how wrong the coverage is.
- The check samples pixels, not parcels: object weights are means of pixel likelihood ratios. That is a heuristic.
- Without parcels the whole-map sample mixes land covers and the signals lose specificity.
- Models trained before this feature have no `domain_reference.npz`. Their report says "unavailable" until they
  are retrained.
- The `geocrop predict` / `geocrop parcels` CLI commands do not run the check yet; it runs in `geocrop run`
  (predict and parcels stages) and in `geocrop benchmark-transfer`. The Python API is `model.predict(...,
  shift=...)` and `parcels.aggregate(..., domain_shift=report)`.

## References

- Meyer, H., Pebesma, E. (2021). Predicting into unknown space? Estimating the area of applicability of spatial
  prediction models. *Methods in Ecology and Evolution* 12, 1620–1633.
- Rabanser, S., Günnemann, S., Lipton, Z. C. (2019). Failing Loudly: An Empirical Study of Methods for Detecting
  Dataset Shift. *NeurIPS 32*.
- Lipton, Z. C., Wang, Y.-X., Smola, A. (2018). Detecting and Correcting for Label Shift with Black Box Predictors.
  *ICML*.
- Austin, P. C. (2009). Balance diagnostics for comparing the distribution of baseline covariates between treatment
  groups in propensity-score matched samples. *Statistics in Medicine* 28, 3083–3107.
- Tibshirani, R. J., Barber, R. F., Candès, E. J., Ramdas, A. (2019). Conformal Prediction Under Covariate Shift.
  *NeurIPS 32*. arXiv:1904.06019.
- Barber, R. F., Candès, E. J., Ramdas, A., Tibshirani, R. J. (2023). Conformal prediction beyond exchangeability.
  *Annals of Statistics* 51(2), 816–845.
- Podkopaev, A., Ramdas, A. (2021). Distribution-free uncertainty quantification for classification under label
  shift. *UAI 2021*, PMLR 161, 844–853. This is background for the label-shift argument; label-shift weighting is
  not implemented.
- Vovk, V., Gammerman, A., Shafer, G. (2005). *Algorithmic Learning in a Random World*. Springer. Mondrian conformal
  prediction.

Implementation is independent, numpy/scikit-learn only. No code was taken from the cited works.
