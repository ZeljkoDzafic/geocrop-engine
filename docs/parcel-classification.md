# Parcel-level classification on signatures

GeoCrop can classify parcels in two ways:

1. **Pixel → parcel aggregation** (`geocrop train` + `predict` + `parcels`): a pixel classifier, then mean
   probabilities / majority votes per parcel ([parcels.md](parcels.md)).
2. **Parcel classifier on signatures** (`geocrop parcel-classify`, module `geocrop.parcel_classify`): one
   feature vector per parcel, built from the parcel's time series (`geocrop signatures`), classified directly.
   This is how operational Checks-by-Monitoring crop-type markers usually work (for example the Sen4CAP L4A
   product or area-monitoring crop-type markers). Only the ideas are used here; no code from those (GPL)
   projects.

Both routes share the estimator registry, the parcel-grouped spatial cross-validation, the locked holdout,
temperature calibration, conformal sets and the area of applicability (AOA). Route 2 calls
`geocrop.model.train` with one sample per parcel, so it reuses the same safeguards.

## Features (`parcel_classify.parcel_features`)

Input: the long signature table from `geocrop signatures` (`parcel_id, date, index, mean, std, n_valid,
n_pixels`). For each index and signature statistic (`--stats`, default `mean std`; any other numeric column,
for example within-parcel percentiles, works too):

- observations covering less than `min_coverage` (0.5) of the parcel's interior pixels are dropped, as in
  `cbm.markers`;
- the rest are binned to a 5-day grid and smoothed with the weighted Whittaker smoother (`--lam`). Parcels
  with fewer than four clear bins get NaN features.

| Group | Columns | Meaning |
|---|---|---|
| `ts` | `ts_<INDEX>_<stat>_d<doy>` | smoothed series every `--output-step-days` (15), channel-major |
| `stats` | `stat_<INDEX>_<stat>_p10…p90`, `stat_<INDEX>_mean_peakdoy` | temporal percentiles, day of the peak |
| `windows` | `win_<INDEX>_<stat>_d<doy>` | mean per `--window-days` (30) window, keyed by its first day |
| `shape` | `shape_area_m2`, `shape_perimeter_m`, `shape_index`, `shape_elongation`, `shape_solidity`, `shape_n_pixels` | parcel geometry and interior pixel count |

The `qa_clear_observations` and `qa_n_pixels` columns record data quality. They are never model inputs.
Within-parcel percentiles per date need pixel values, and the default signatures only hold mean, std and
count. If your signature table has such columns, pass them with `--stats`.

## Training and outputs

```bash
geocrop parcel-classify --signatures signatures.csv --parcels parcels.gpkg \
  --start 2023-03-15 --end 2023-10-31 --label-field verified_crop --out outputs/parcel_cls \
  --algorithm rf --block-m 5000 --folds 5 --holdout 0.2 --alpha 0.1
```

- `--label-field` must hold **independently verified** classes, never the declarations being checked.
- Spatial CV groups parcels by spatial block (`--block-m`, optional `--buffer-m`). `--holdout` locks whole
  blocks away first and scores them once. `--calibration-mode holdout` gives a frozen predictor plus
  held-out calibration parcels. The semantics are those of `geocrop train` ([uncertainty.md](uncertainty.md)).
- `outputs/parcel_cls/model/` holds the usual training artifacts (`model.joblib`, `conformal.json`,
  `aoa.json`, `train_report.json`, `oof_polygons.csv` with one row per parcel) and `parcel_features.csv`.
- `outputs/parcel_cls/parcels.gpkg|.csv|.summary.json` uses the same decision schema as `geocrop parcels`:
  `predicted_class`, `confidence`, `margin`, `entropy`, `p_<class>`, `prediction_set`, `set_size`, `cp_status`,
  `decision`, `decision_reasons`, `declaration_status`, `aoa_fraction` (1 inside the AOA, 0 outside) and
  `traffic_light` (from `geocrop.cbm.traffic_light`). Parcels without enough clear observations are
  `insufficient_resolution`, shown grey. `in_training` marks parcels whose labels fitted the model; their
  predictions are in-sample. `--markers`/`--calendar` add `calendar_consistent` just as `geocrop parcels`
  does. `strategy` is `parcel_classifier`.
- `--model DIR` skips training and applies an existing parcel model. Build its features with the same
  settings.

Python API: `parcel_features`, `train`, `predict`, `compare_pixel_aggregation` in `geocrop.parcel_classify`.

## Temporal deep models (LightTAE, TempCNN)

The estimator registry has two sequence models. They read the channel-major `ts_…_d<doy>` layout,
reshape it to `(n, T, C)` and behave as scikit-learn classifiers (`fit`, `predict_proba`, `classes_`,
`get_params`, picklable):

| Name | Model | Defaults (CPU-friendly) |
|---|---|---|
| `tempcnn` | Temporal CNN (Pelletier et al., 2019) | 3×Conv1d(64), 60 epochs; early stopping off unless `patience` is set (unchanged behaviour) |
| `ltae` | Lightweight Temporal Attention Encoder (Garnot & Landrieu, 2020, arXiv:2007.00586) | d_model 64, 8 heads, d_k 8, MLP (64), 100 epochs, early stopping (`patience` 10, `validation_fraction` 0.1) |

L-TAE is re-implemented from the paper. Each date gets a shared linear projection, then layer normalisation
over time × channels and a sinusoidal encoding of the real **day of year** (`positions`, read from the
feature names). Each head has one learned master query. The values are the projected channels split across
the heads, and the concatenated head outputs pass through an MLP. TorchGeo's MIT `ltae` was a structural
reference. `LightTAEClassifier.attention(X)` returns the per-date attention weights.

Both models are seeded (`torch.manual_seed` plus a NumPy generator for batch order and the validation
split). Inputs are standardised per channel with training statistics. Early stopping uses a stratified random
fraction of the *training* rows only and restores the best weights. It never sees test folds, and outer
spatial CV stays the accuracy estimate. `estimators.sequence_params(params, feature_names)` fills
`n_channels` and `positions`. `train`, `benchmark` and `benchmark-models` call it automatically.

PyTorch is optional (`pip install torch`). Without it, the models fail with a message naming the package, and
`benchmark-models` reports them as skipped.

```bash
# pixel level, on a time-series stack (geocrop ts-features)
geocrop train --samples ts_samples.npz --manifest ts_stack.tif --groups ts --algorithm ltae --out m_ltae
geocrop benchmark-models --samples ts_samples.npz --features ts_stack.tif --groups ts \
  --models rf tempcnn ltae --out bench_ts
# parcel level (temporal models automatically use only the ts_ block)
geocrop parcel-classify ... --algorithm ltae --params '{"epochs": 150, "patience": 15}'
```

## Comparison with pixel → parcel aggregation

`--compare-pixels pixel_samples.npz` (or `compare_pixel_aggregation`) scores three arms:

- the parcel classifier;
- a pixel classifier aggregated by mean probability;
- the same pixel classifier aggregated by majority vote.

Pixel samples need `polygon` = parcel id, for example
`samples.extract(stack, parcels, label_field=..., id_field="parcel_id")`. All three arms use the
**same parcels, the same locked holdout and the same parcel-grouped spatial folds**. The holdout and folds
are computed once on the parcels with `model.split_final_holdout` / `model._fold_plan` and mapped to pixels
by parcel id. The run asserts that no parcel, and no spatial block, is in both training and test of one
fold. Fold k fits both models with seed + k, so the parcel arm reproduces `model.cross_validate` exactly.
Outputs: `comparison.json` (pooled and per-fold metrics, McNemar tests, fold fingerprint, holdout scores),
`comparison.csv` and `comparison_oof.csv`.

### Synthetic experiment (SYNTHETIC data only)

Reproduce with:

```bash
PYTHONPATH=src python tools/parcel_vs_pixel_experiment.py --out outputs/parcel_vs_pixel
```

**Setup**

- **Data.** `geocrop.demo.synthetic_parcel_series` with 400 parcels (`n_side` 20), six classes and NDVI only.
  Each parcel has its own sowing shift and amplitude. A fraction of each parcel's pixels is mixed 50/50 with
  another class, observations get pixel noise, and whole-parcel clouds remove dates. Two scenarios:
  - *moderate*: noise 0.08, clouds 30 %, mixed 15 %, shift sd 12 d;
  - *hard*: noise 0.15, clouds 50 %, mixed 30 %, shift sd 18 d.
- **Validation design.** Generator seeds 1, 2 and 3; 300 m spatial blocks; 5 folds; 20 % locked holdout
  (about 80 parcels).
- **Models.**
  - Pixel arm: RF with 300 trees on ~22,700 pixels × 16 smoothed `ts_NDVI` dates.
  - Parcel arms: RF on all feature groups (65 features), and L-TAE and TempCNN on the `ts` block (2 × 16).
- **Run.** Apple M5 Pro, torch 2.14.1, scikit-learn 1.9.1; about 4.5 minutes in total.

**Results** (mean ± sd over the three seeds):

| Scenario | Method | CV OA | CV macro-F1 | Holdout macro-F1 |
|---|---|---|---|---|
| moderate | parcel RF | 0.935 ± 0.005 | 0.938 ± 0.004 | 0.914 ± 0.017 |
| moderate | parcel TempCNN | 0.946 ± 0.009 | 0.949 ± 0.009 | 0.952 ± 0.021 |
| moderate | parcel L-TAE | 0.923 ± 0.011 | 0.927 ± 0.012 | 0.934 ± 0.072 |
| moderate | pixel RF → mean probability | 0.977 ± 0.010 | 0.979 ± 0.009 | 0.972 ± 0.007 |
| moderate | pixel RF → majority | 0.979 ± 0.006 | 0.981 ± 0.006 | 0.967 ± 0.008 |
| hard | parcel RF | 0.874 ± 0.004 | 0.876 ± 0.008 | 0.893 ± 0.017 |
| hard | parcel TempCNN | 0.880 ± 0.016 | 0.884 ± 0.014 | 0.891 ± 0.026 |
| hard | parcel L-TAE | 0.864 ± 0.019 | 0.869 ± 0.017 | 0.888 ± 0.031 |
| hard | pixel RF → mean probability | 0.912 ± 0.011 | 0.914 ± 0.013 | 0.891 ± 0.037 |
| hard | pixel RF → majority | 0.910 ± 0.010 | 0.912 ± 0.007 | 0.891 ± 0.037 |

**Reading.** On this generator, pixel → parcel aggregation scores higher in cross-validation than every
parcel classifier, by 3 to 4 macro-F1 points. On the locked holdout of about 80 parcels the arms are within
noise of each other in the hard scenario. All arms are far above chance (macro-F1 about 0.17 for six classes).

Two properties of the generator favour pixels:

- Mixed pixels move the parcel *mean* signature, but a pixel model that learns from mostly clean pixels
  outvotes them.
- The pixel model sees about 57 correlated replicates per parcel.

These numbers describe only the synthetic generator. They make no claim about real parcels, where
registration error, within-field management and the much lower cost of parcel-level training and inference
can change the balance. Run the same comparison on real signatures and pixel samples before choosing a
route.

## Limitations

- Signatures from `geocrop signatures` contain Sentinel-2 indices only. Sentinel-1 parcel series can be
  added as further `index` values in the same long table.
- With one sample per parcel, parcel training sets are small. The temporal deep models need several hundred
  labelled parcels to be competitive, and the early-stopping split is random rather than spatial.
- Shape descriptors can carry spurious signals, for example when class and parcel size are correlated in
  the reference data. Drop them with `--groups ts stats windows` when that is a concern.
- `parcels.gpkg` predictions for labelled parcels are in-sample (`in_training`). Use the CV/holdout reports
  or independent field data for accuracy.
