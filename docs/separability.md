# Class separability (Jeffries-Matusita) and window design

Before building a classifier it helps to know *when* during the season two crops become separable and *which
feature* carries the difference. `geocrop separability` answers that with the Jeffries-Matusita (JM) distance.

```
geocrop separability --stack features/stack.tif --polygons reference/training.gpkg \
    --classes maize soybean sunflower --out out/separability
```

Outputs (`out/separability/`): `parcel_means.csv`, `jm_all.csv` (feature x class pair), `jm_best_per_pair.csv`
(best feature per period and pair, with the mean over pairs of those maxima per period).

## Method
- **Unit = polygon.** Every feature is averaged over the valid pixels of a polygon, then the class distributions
  are compared over polygons. A large parcel gets no more weight than a small one.
- **Formula.** For univariate Gaussian classes, `B = (m1-m2)^2/(8v) + 0.5 ln(v/sqrt(v1 v2))` with
  `v = (v1+v2)/2`, and `JM = 2 (1 - exp(-B))`, in 0 (identical) to 2 (fully separable).
- Pairs with fewer than 3 polygons in a class, or zero variance, give NaN, not a number.

## How to read it
- JM is a diagnostic, not a model. It is per feature and per pair; a random forest uses interactions between
  features and all classes, so a high JM does not guarantee the best classifier. Read it together with the
  ablation (`geocrop run --stages experiments`, which reports `target_macro_f1` for `classes.targets`) and the
  cumulative-period experiments.
- With a few dozen polygons per class the variances are uncertain: use JM to **rank** periods and features,
  not to claim a threshold such as "good" or "excellent".
- Overlapping windows are not independent, so small differences between them are not independent contrasts.

## Overlapping candidate windows
By default windows must not overlap. For a window-design study (31-day windows shifted in small steps through
the expected phenophase) set `compositing.allow_window_overlap: true`; a scene then enters every window it
falls in. Evaluate every candidate on the same fixed folds and compare on the target-class macro F1
(`target_macro_f1`), and keep the final operational windows non-overlapping unless you can justify otherwise.

## Target macro F1 (configurable)
`classes.targets` lists the classes the study is about. The experiments table (`ablation.csv`) then reports, for
the same out-of-fold predictions on the identical folds:

| column | meaning |
|---|---|
| `f1_<class>` | F1 of each target class (pixels) |
| `target_macro_f1` | mean of those F1 values, pixels |
| `parcel_target_macro_f1` | the same on polygons: the pixels of a polygon vote, each polygon counts once |
| `parcel_overall_accuracy`, `parcel_kappa`, `parcel_n` | parcel-level overall accuracy, kappa and polygon count |

Overall accuracy over all classes can stay high because easy classes (forest, water, built-up) dominate; the
target macro F1 shows how well the crops of interest are separated from each other. For maize, soybean and
sunflower set `classes: {targets: [maize, soybean, sunflower]}`. Pixel-level and parcel-level values differ
(majority voting removes isolated pixel errors) and must not be compared with each other.
