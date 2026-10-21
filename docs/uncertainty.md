# Uncertainty, calibration and applicability

A map is only useful for decisions if its confidence is meaningful. `geocrop train` produces four artefacts
from the out-of-fold (spatial CV) predictions, and `predict`/`parcels` apply them.

## Probability calibration
Random Forest vote fractions are typically under-confident. **Temperature scaling** (Guo et al., 2017) fits one
parameter T on the out-of-fold probabilities, `p_T ∝ p^(1/T)`, minimising log-loss; it cannot overfit a small
reference set and preserves the ranking of classes. `train_report.json` reports ECE before and after
(`calibration`, `calibration_after_temperature`); `probabilities.tif` is written with calibrated values.
Disable with `model.calibrate: false` / `--no-calibrate`.

## Conformal prediction sets
For decisions about *parcels*, a set of plausible classes with a coverage guarantee is more honest than a single
label with an uncalibrated score. The engine implements **split conformal prediction, class-conditional
(Mondrian)** (Vovk et al., 2005; Angelopoulos & Bates, 2023; for land cover: Valle et al., 2023):

* calibration objects = reference polygons, each represented by the mean of its *out-of-fold* pixel probabilities
  (so no polygon calibrates a model that saw it);
* per class c, threshold q_c = the ⌈(n_c + 1)(1 − α)⌉-th smallest score `1 − p(c)` of polygons of class c;
* prediction set of a parcel = {c : 1 − p̄(c) ≤ q_c}; it contains the true class with probability ≥ 1 − α for
  every class whose calibration set is large enough (n_c ≥ 1/α − 1). Smaller classes fall back to the pooled
  threshold and are listed in `conformal.json → pooled_fallback_classes`.

`conformal.json` stores thresholds for α = 0.05, 0.1, 0.2 and the empirical coverage / set sizes on the
calibration polygons. In `parcels`, `cp_status` is *agreement* (set = {declared}), *discrepancy* (declared crop
not in a non-empty set) or *uncertain*. Caveats to report: exchangeability is weakened by spatial clustering and
by the difference between interpreted training polygons and arbitrary LPIS parcels — check coverage on
independently verified parcels.

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
principal component of the polygon-mean features; use its range as a lower bound for `spatial_block_m`
(Roberts et al., 2017; Ploton et al., 2020). Report the range with the CV design.

## References
Guo et al. (2017) ICML · Vovk, Gammerman & Shafer (2005) Algorithmic Learning in a Random World ·
Angelopoulos & Bates (2023) Found. Trends ML 16:494–591 · Valle, Izbicki & Leite (2023) RSE 295:113682 ·
Meyer & Pebesma (2021) Methods Ecol. Evol. 12:1620–1633 · Northcutt et al. (2021) JAIR 70:1373–1411 ·
Roberts et al. (2017) Ecography 40:913–929 · Ploton et al. (2020) Nat. Commun. 11:4540.
