# Sensitivity and review tools

Tools for checking how much a result depends on the validation design, on single polygons and on the decision rule.
They describe the reference polygons you give them; none of them is independent validation, and a score from any
of them is a recommendation for human review, not a conclusion.

## Training-only controls

`model.train_exclude_polygons` (list of polygon ids) and `model.train_max_samples_per_polygon` (integer) change what
the model is *trained* on while the *test* folds stay exactly those of the full sample set. The fold plan is built
from all samples first; the controls only thin the training indices of each fold (and the final fit). The same
polygons and pixels are therefore scored with and without a control, so the scores are directly comparable. A pixel
is never removed from validation to raise accuracy.

Use `train_exclude_polygons` to ask "does the result depend on these polygons being in training?" (for example
polygons whose out-of-fold prediction disagrees with their label), without deleting them from the reference set.
Use `train_max_samples_per_polygon` to isolate the effect of the number of training pixels: `max_samples_per_polygon`
changes sampling at extraction and therefore also the number of validation pixels. Both are refused together with
`tune_grid`, because inner folds would not keep the fixed test polygons. `train_report.json` records
`train_controls`.

## Skipped polygons

`geocrop samples` (and the `samples` stage) writes `skipped_polygons.csv` next to `samples.npz`: every polygon that
yielded no sample, its class, area and one reason (empty geometry, outside the stack extent, no pixel centre inside
the polygon, or no pixel valid for every selected feature).

## `oof_pixels.csv`

`geocrop train` writes `model/oof_pixels.csv`: coordinates, spatial group, label, out-of-fold prediction and
temperature-scaled probabilities per training pixel (see [uncertainty.md](uncertainty.md)).

## Partition stability: `geocrop cv-stability`

One 5 km grid is one draw of blocks. `cv-stability` repeats the spatial-block cross-validation over several block
sizes and shifted grid origins and writes `cv_stability.csv` and `cv_stability.json` with the mean, spread, minimum
and maximum of overall accuracy, macro F1 and (with `--targets`) target macro F1. Whole polygons stay inside one
block in every partition. The spread is sensitivity to the partition, not a confidence interval.

    geocrop cv-stability --samples out/samples.npz --block-sizes 3000 5000 8000 --offsets 4 \
        --targets maize soybean sunflower --out out/stability

Further outputs: `confusion/<variant>.csv` (pixel confusion matrix of every variant) and per-class F1 columns for the
targets. `--polygon-repeats N` adds N repeats of stratified polygon-grouped k-fold CV (whole polygons held out,
neighbouring polygons stay in training, so it is less strict than block CV); it shows whether a result depends on
how blocks group the polygons. The table also lists, per variant, the smallest number of test polygons of any class
in a fold and how many folds lack a class completely.

`--watch ID ...` writes `watch_polygons.csv`: for each variant and watched polygon the out-of-fold majority
decision, the share of pixels predicted as the true class, the fold and group, how many other watched polygons were
held out in the same fold, how many same-class polygons share its group, and how many same-class polygons remain in
training within `--radii` metres of its centroid. This separates "misclassified whenever its neighbours are held
out too" from "misclassified in every configuration".

## Relabelling for training only: `--train-relabel ID=CLASS`

A sensitivity experiment for atypical samples: in `cv-stability` the listed polygons are treated as another class
*in training only*. Fold plans, test pixels and every score keep the true labels, so the run can be compared with the
untouched one fold by fold. The table adds `*_unrelabelled` columns (overall accuracy, precision, recall and F1 of the
target classes on the polygons that were not relabelled), which show whether the other polygons of those classes are
affected. It measures sensitivity of the model, not which label is correct, and it is not used to change reference
labels.

## Timing of seasonal curves: `geocrop phenology-timing`

From the `profiles.csv` of `geocrop profiles` it computes for every polygon and family the start (`sos`), peak,
end (`eos`) and length (`gsl`) of the seasonal curve (days where the interpolated curve crosses a fraction of its
amplitude, default 0.5), the offset of highlighted polygons from each class median and their rank within their own
class, and a time alignment: the shift (days, ±`--max-shift-days`) that minimises the RMSE to each class median curve
with the RMSE before and after it. Resolution is the composite step. The output is descriptive; it does not show that a
label is wrong.

## Parcel-level aggregation: `geocrop parcel-aggregate`

From `oof_pixels.csv` it compares, on the same polygons: `majority` (pixel votes), `mean_prob` (mean pixel
probability) and `combined` (mean-probability class accepted only if it equals the majority class and its confidence
reaches a threshold). For each method it writes the parcel confusion matrix, precision, recall and F1 per class and
the target macro F1; for `combined` also the share of polygons sent to review and the accuracy on the accepted ones.
The threshold is fitted without the polygons it is applied to: for every spatial group it is the smallest value on a
0.05 grid for which the accepted polygons of all *other* groups reach `--target-accuracy`; `--threshold` fixes it.
Those polygons' out-of-fold probabilities come from models that had the held-out group in training, so this is close
to, not identical with, a fully nested selection.

## Temporal profiles: `geocrop profiles`

For time-series features named `<family>_d<day>` (for example `ts_NDVI_d121`) it plots chosen polygons against the
median and inter-quartile band of each class (other polygons only) and writes `profiles.csv` and
`profile_distances.csv` (RMS distance of each chosen polygon to every class median curve). PNG figures need
`matplotlib` (`pip install geocrop[plots]`). The distance says which class a polygon resembles; it does not show that
a label is wrong.

## Cloud Score+ masks: `geocrop cloud-score`

`compositing.cloud_probability_max` masks a scene by an optional `CLDPRB.tif` (cloud probability 0-100) in the scene
folder. `geocrop cloud-score --scenes interim/s2 --scores <dir>` writes that file from clearness rasters
(0 = cloudy, 1 = clear, e.g. Cloud Score+ `cs_cdf`) named `<scene folder name>.tif`, warped to the grid of the
scene's `SCL.tif`: `CLDPRB = round((1 - score) * 100)`, and no score means masked. The rule `cs_cdf >= 0.60` is then
`cloud_probability_max: 40`. The engine does not download Cloud Score+; export it for the same scenes beforehand.
Comparing the SCL-only, SCL + Cloud Score+ and SCL + Cloud Score+ + shadow variants is three configurations over
identical windows, polygons, folds and parameters.

## Phenology features and time-aligned profiles: `geocrop phenology-features`

Question: does a model generalise better to crops whose development is shifted in time when it also sees the *timing*
of the season (start, peak, end, length) or the curve in *relative* phenological time? The command reads a samples file
with `ts_<index>_d<day>` columns and writes (a) an augmented samples file and (b) a JSON of named feature sets, so that
`geocrop cv-stability --feature-sets sets.json --feature-set post_season:B` compares them over identical polygons,
partitions, folds and Random Forest parameters.

- `ph_<idx>_*`: season start, peak day, peak value, amplitude, green-up rate, integral and, post-season only, end,
  length and senescence rate (`timeseries.phenology`, `--threshold`, `--min-amplitude`).
- `rel_<idx>_sos+N` / `rel_<idx>_pk+N`: curve normalised to the season amplitude, read N days after the start
  (`--sos-offsets`) or from the peak (`--peak-offsets`, post-season only).
- Sets per group: `A` reference (every original column), `B` = A + `ph_*`, `C` = A + `rel_*`.
- `--cutoffs 196 226 --windows NAME=END_DOY ...` adds in-season groups (the window names and end days of the
  configuration are required): curves are cut at that day of year before any feature is derived, and A keeps only
  columns recognised as observable by then: columns of windows ending on or before the cutoff (matched against the
  given window names), `ts_` columns up to it and deltas between such windows. Whole-season columns (`season_*`,
  `harm_*`, `phen_*`) and columns that match no window are dropped. End, length and senescence are not produced.
- `--indices` selects the series (default NDVI NDRE); undefined values are written as `--fill` (default -1).

Limits: the `ts_` curves were smoothed over the whole season, so in-season groups are an optimistic proxy; a strictly
causal test recomputes the curves with the season ended at the cutoff. Where the season amplitude is below
`--min-amplitude` (bare, forest, water) the metrics are undefined and filled. Internal cross-validation only; a
recommendation for human review, not a definitive conclusion.
