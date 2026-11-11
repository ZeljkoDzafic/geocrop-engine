# Temporal windows: separability, cumulative performance and the earliest reliable window

`geocrop windows` compares *when* and *with which sensors* classes become separable and classifiable. It does not
know an area, a season or a crop: windows are named date ranges, classes and variables are whatever your
observation table contains. It reuses `geocrop.separability` (JM), the spatial fold plan and estimator of
`geocrop.model`, `geocrop.metrics` and `geocrop.phenofeatures`; the new parts are window generation, the
single / cumulative / optimised comparison, the pre-registered earliest-window rule, exports and the in-season
leakage guard (module `geocrop.windows`).

Every number is an internal estimate on interpreted reference units. JM on few units is a ranking, not a test;
cross-validation is not independent validation; the earliest reliable window is a recommendation for human review,
not a definitive conclusion.

## Input

A long table (CSV or Parquet) of **raw** observations, one row per unit, date and variable:

| Column | Meaning |
|---|---|
| `unit_id`, `class_name` | reference polygon and its class (constant within a unit) |
| `x`, `y` | unit location in a projected CRS in metres (spatial blocks); optional `block` overrides them |
| `date`, `variable`, `value` | one observation, e.g. `NDVI` from S2, `VH` from S1 |

Units are parcels (parcel means), as in `geocrop separability`. The table is not produced by a new extractor:
write it from your scene folders or from any per-parcel extraction. Do not feed smoothed whole-season series into
it for in-season work (see the guard below).

## Configuration

YAML or JSON; every key has a default except `groups` and the windows.

```yaml
seed: 42
season: {start: "2001-04-01", end: "2001-09-28"}     # arbitrary; cumulative windows start here
windows:                                              # explicit, arbitrary names (may overlap)
  early: {start: "2001-04-15", end: "2001-05-31"}
generate: {length_days: 30, step_days: 15, prefix: W} # and/or generated; or overlap_days: 15; partial: drop|clip
groups: {S1: [VV, VH], S2: [NDVI, NDRE]}              # variable names of the table, grouped by sensor
stats: [mean, smoothed_mean]                          # mean std min max median smoothed_mean (causal)
classes: [a, b, c]                                    # default: all
pairs: [[a, b], [b, c]]                               # default: all pairs
kinds: [single, cumulative, optimised]
cv: {block_m: 5000, n_splits: 5, crs: "EPSG:32633"}   # blocks of block_m metres; crs only checks the units
model: {n_estimators: 200, max_features: sqrt, class_weight: balanced_subsample}
bootstrap: {n: 500, alpha: 0.05}
permutation_repeats: 3
optimise: {min_length_days: 10}
criteria:                                             # frozen in the manifest BEFORE evaluation
  configuration: combined                             # S1, S2, combined (drop_<g> with 3+ groups)
  kind: cumulative
  target_classes: [a, b]
  min_f1: 0.8
  min_recall: 0.8
  jm: {pairs: [[a, b]], min: 1.5}
  lower_bound: {metric: f1, min: 0.6, method: bootstrap}   # or fold_min
  stability: {mode: all_later}                        # or next_k with k, or none
```

Windows: both ends inclusive; `generate` uses length, step (default: length) or overlap (step = length - overlap)
and works for any year or season. A *cumulative* window runs from the season start to the window end; an
*optimised* window has a fixed end and a start chosen among the window starts and the season start (at least
`min_length_days` long).

## Commands

```bash
geocrop windows list  --config windows.yaml [--out windows_block.yaml]   # resolved windows + cumulative windows, as a `windows:` block
geocrop windows run   --config windows.yaml --observations obs.csv --out outputs/windows [--seed N]
geocrop windows guard --config windows.yaml --cutoff 2001-05-30 --names P1_NDVI ts_NDVI_d150   # exit 1 on leakage
```

## What `run` does

For each window and kind it builds features from raw observations (`variable__stat`), then, on the **same units
and the same spatial folds** (one fold plan, its signature is recorded and identical in every row):

1. **JM** per class pair and feature through `separability.pairwise` (single and cumulative), and the best
   feature of each configuration per pair (`jm_best`).
2. **Spatial CV** of a random forest for S1-only, S2-only and combined (and drop-one sets with 3+ groups):
   macro F1, class F1/recall/precision, bootstrap lower bounds (resampling units, paired across windows) and
   fold-wise minima.
3. **Optimised windows**: the start is chosen *inside each training fold* by the mean best-feature JM of the
   target pairs, so held-out units never influence it; chosen starts per fold are in `optimised_selection`.
4. **Contribution** of feature groups: macro F1 of combined minus the set without the group (`drop_group_gain`),
   the group-only score, and a group permutation importance (rows of the group's columns shuffled jointly in each
   test fold; `permutation_drop`).
5. **Earliest reliable window** under the criteria. A cumulative window qualifies if every enabled criterion holds
   (the weakest target class decides); stability then requires the same for all later windows (`all_later`) or the
   next `k` (`next_k`; fewer than `k` later windows left means not reliable). The result is a window name or
   `none`. The criteria are written to `manifest.json` (status `criteria_registered`, with their SHA-256) before
   any result is computed; the run fails if they differ at the end. Choosing criteria after looking at results
   defeats the purpose; keep the `criteria.json` of the first run.

Outputs in `--out`: `windows`, `jm_by_feature`, `jm_best`, `performance`, `performance_by_class`, `contribution`,
`optimised_selection`, `earliest_reliable`, `leakage_guard` (each as CSV and markdown), `jm_by_window.png`,
`performance_by_window.png`, `group_contribution.png` (need matplotlib; `pip install geocrop-engine[plots]`),
`resolved_config.json/.yaml`, `criteria.json` and `manifest.json` (seed, config and input hashes, package
versions, fold signature, output hashes).

## In-season leakage guard

An in-season analysis for classification date D may use only observations dated <= D, through every step:
smoothing, gap filling and feature derivation.

* **Per-feature provenance.** Each feature set built by `windows` records the latest observation date it used.
  `check_provenance(used, cutoff)` raises `LeakageError` naming every feature that depends on later data; `run`
  applies it to every (start, end) feature set with the window end as the cutoff and writes the audit trail to
  `leakage_guard.csv`.
* **Causal smoothing.** `causal_smooth` truncates the series at the cutoff *before* binning and Whittaker
  smoothing, and the grid ends at the cutoff; `causal_smooth_per_cutoff` refits for each cutoff. The `smoothed_mean`
  statistic uses it. `timeseries.whittaker` on a whole-season series is not causal: a value at D depends on
  observations after D (tested: it changes when only later observations are perturbed, the causal wrapper does not).
* **Existing columns.** `check_feature_names` / `geocrop windows guard` refuse columns that are not provably
  observable at the cutoff (windows ending later, later `ts_` days, `season_*`, `harm_*`, `phen_*`, unrecognised
  names), using the rules of `phenofeatures.feature_sets`.
* **Tests** (`tests/test_windows.py`) perturb all observations after D and assert bit-identical features and
  bit-identical results for every window ending on or before D.

Failure modes the guard cannot see, because they happen outside the table it receives:

* smoothed or gap-filled series, composites or `ts_` / `phen_` / `harm_` columns built from the whole season;
* cloud screening that compares each observation with a whole-season median (`temporal_screening`);
* normalisation or imputation statistics computed on all dates;
* labels or window choices tuned on the same units that are later scored (use the nested `optimised` kind);
* criteria edited after the results were seen.

Causal refitting is also honest about a cost: the smoothed value at the end of a short series is less constrained
than the same date in a full-season fit, so in-season scores are usually lower than whole-season ones.

## Classification maps for a chosen window

The module analyses parcel tables; it does not rasterise. To map a chosen window, write its dates to a
configuration (`geocrop windows list` prints a `windows:` block; a cumulative window is a window whose start is the
season start), build the feature stack with `geocrop features` (or `run`), train with `geocrop train` and predict with
`geocrop predict`, as in [classification-validation.md](classification-validation.md). For a real in-season map,
build the stack only from scenes dated up to the classification date (`geocrop inventory` shows the scenes per
window) and avoid `timeseries` features smoothed over the whole season.

## Limits

Units are parcel means; pixel-level CV is not offered here (use `geocrop train` / `cv-stability`). JM assumes
univariate Gaussian classes. Cumulative `mean` features dilute early signal as the window grows; add
`smoothed_mean`, `max` or `std`. Few units per class give wide bootstrap bounds and unstable JM.
