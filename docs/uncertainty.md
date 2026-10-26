# Uncertainty, calibration and applicability

A map is only useful for decisions if its confidence is meaningful. `geocrop train` produces four artefacts
from the out-of-fold (spatial CV) predictions, and `predict`/`parcels` apply them.

## Probability calibration
Random Forest vote fractions are typically under-confident. **Temperature scaling** (Guo et al., 2017) fits one
parameter T on the out-of-fold probabilities, `p_T ∝ p^(1/T)`, minimising log-loss and preserving
the ranking of classes. A one-parameter fit can still overfit a small reference set. ECE after fitting
on the same OOF labels is a calibration-fit diagnostic, not an independent calibration assessment. `train_report.json` reports ECE before and after
(`calibration`, `calibration_after_temperature`); `probabilities.tif` is written with calibrated values.
Disable with `model.calibrate: false` / `--no-calibrate`.

## Conformal prediction sets
For decisions about *parcels*, the engine produces sets of plausible classes using class-conditional
(Mondrian) conformal score quantiles (Vovk et al., 2005; Angelopoulos & Bates, 2023):

* reference objects = polygons, each represented by the mean of its out-of-fold pixel probabilities;
* per class c, threshold q_c = the ⌈(n_c + 1)(1 − α)⌉-th smallest score `1 − p(c)` of polygons of class c;
* prediction set of a parcel = {c : 1 − p̄(c) ≤ q_c}. Classes with too few objects for the requested
  quantile use the pooled threshold and are listed in `pooled_fallback_classes`.

The default training workflow is **OOF-based empirical calibration**, not a standard split-conformal
procedure: the probabilities come from several fitted models, temperature uses the same labels,
and the mapping predictor is then refitted on all samples. Consequently no finite-sample or
class-conditional coverage guarantee is established for its outputs. Spatial dependence and a shift
from interpreted reference polygons to LPIS parcels further limit transfer. A valid split-conformal
claim would require a fixed predictor and separately held-out, exchangeable calibration objects.

`conformal.json` stores thresholds for nominal α = 0.05, 0.1, 0.2 and descriptive coverage/set sizes
on the fitting polygons. Those figures are not independent validation. In `parcels`, `cp_status` is
*agreement* (set = {declared}), *discrepancy* (declared crop not in a non-empty set) or *uncertain*.
Measure coverage on independently verified parcels before operational use. Choose review thresholds
on separate development/calibration data and freeze them before evaluating the independent field set.

## Optional held-out calibration

```bash
geocrop train --samples samples.npz --out outputs/holdout-model --block-m 5000 \
  --calibration-mode holdout --calibration-fraction 0.2 --buffer-m 100
```

Equivalent YAML: `model.calibration_mode: holdout`, `model.calibration_fraction: 0.2`.
A deterministic, single group split reserves the requested fraction of spatial groups. It fails if
the remaining groups cannot support CV or lack a class; it does not search repeated splits for a
better score. An optional spatial buffer purges whole proper-training polygons near calibration
samples, using sampled pixel support. Tuning, internal CV, temperature fitting, AOA and suspect-label
diagnostics use only the proper-training subset. Temperature is fitted from its OOF probabilities.
The deployed predictor is then frozen; calibration polygons never enter fitting or a later refit.

`calibration_split.json` records group/polygon roles and exclusions; `calibration_polygons.csv` records
the held-out probabilities from that exact frozen predictor. `conformal.json` stores class thresholds;
when a class lacks enough calibration objects for the requested finite-sample quantile, its threshold
is 1 and the class is always included. Sparse support therefore produces larger sets rather than an
unsupported class-specific coverage claim. Training counts now describe proper training, while
`input_samples` and `calibration_split` provide full accounting.

The saved model SHA-256 binds calibration metadata to the deployed artifact. Prediction records that
identity in its manifest and raster tags; parcel aggregation rejects mismatched bound thresholds.
This guards against mixing artifacts; it does not certify data quality or accuracy.

Standard split-conformal theory still requires exchangeability between calibration and future objects.
A spatial holdout does not establish that assumption, especially across years, regions or polygon/LPIS
sampling designs. `coverage_check` describes the fitting calibration objects, not a third independent
evaluation. Holdout mode sacrifices training support and can reduce map accuracy; assess both coverage
and accuracy on untouched field data. Default `oof` retains the empirical workflow described above.

## Area of Applicability
Following Meyer & Pebesma (2021), the **dissimilarity index** DI of a pixel is its distance to the nearest
training sample in standardised, importance-weighted feature space, divided by the mean distance between
training samples. The **AOA threshold** is the outlier-trimmed maximum (Q3 + 1.5 IQR) of training DIs computed
across spatial folds. `predict` writes `aoa.tif` (band 1 DI, band 2 inside = 1); `parcels` records
`aoa_fraction` and sends parcels with < `min_aoa_fraction` (default 0.5) inside to review — the model would be
extrapolating there, e.g. on crops, soils or management absent from the reference data.

## Label-noise screening
`suspect_labels.csv` lists reference polygons whose out-of-fold probability of their own label is < 0.2 while
another class exceeds 0.6 (confident-learning principle; Northcutt et al., 2021). They are candidates for
re-interpretation by the analyst, not automatic corrections.

## Spatial autocorrelation and block size
`geocrop variogram --samples samples.npz --out variogram.json` fits a spherical semivariogram to the first
principal component of the polygon-mean features. Its range is an exploratory covariate diagnostic,
not a guaranteed lower bound for residual autocorrelation or a proof that a block size is adequate.
Consider response/error residuals, study geometry and deployment distances as well (Roberts et al.,
2017; Ploton et al., 2020). Report the analysed variable and range with the CV design.

## Domain shift
Conformal coverage assumes exchangeable calibration and target parcels. A new region or season breaks that.
`prediction/shift_report.json` checks for this at prediction time. Under any detected shift, coverage is reported as
*not established* everywhere it is shown. Optional routing (`review`/`abstain`) and shift-adjusted thresholds
(`conservative`/`weighted`) are available: see [domain shift](domain-shift.md).

## References
Guo et al. (2017) ICML · Vovk, Gammerman & Shafer (2005) Algorithmic Learning in a Random World ·
Angelopoulos & Bates (2023) Found. Trends ML 16:494–591 · Valle, Izbicki & Leite (2023) RSE 295:113682 ·
Meyer & Pebesma (2021) Methods Ecol. Evol. 12:1620–1633 · Northcutt et al. (2021) JAIR 70:1373–1411 ·
Roberts et al. (2017) Ecography 40:913–929 · Ploton et al. (2020) Nat. Commun. 11:4540.
