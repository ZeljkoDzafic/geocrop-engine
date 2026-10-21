# Configuration

A study is one YAML file; `geocrop config-check --config file.yaml` prints the fully resolved configuration and
its fingerprint. Unknown keys are errors (typos never fall back silently). Annotated example:
[`configs/example.yaml`](../configs/example.yaml); case study: [`configs/rs_north_2023.yaml`](../configs/rs_north_2023.yaml).

| Section | Key | Default | Meaning |
|---|---|---|---|
| (top) | `project` | `geocrop-experiment` | name used in reports |
| | `seed` | 42 | seed for sampling, folds and models |
| `grid` | `crs`, `resolution_m`, `reference` | –, 10, – | study grid; `reference` = GeoTIFF from `geocrop grid` |
| `windows` | `<name>: {start, end}` | – | non-overlapping phenological windows (ISO dates, inclusive) |
| `paths` | `s2_scenes`, `s1_scenes`, `composites`, `features`, `training`, `parcels`, `observations`, `output` | `data/...` | file locations |
| `compositing` | `statistic` | `median` | `median`, `mean` or percentile `pNN` |
| | `scl_invalid` | `[0,1,3,8,9,10,11]` | SCL classes treated as unusable |
| | `cloud_probability_max` | null | additional MSK_CLDPRB threshold (%) |
| | `mask_dilation_px` | 0 | dilation of cloud/shadow masks |
| | `min_valid_observations` | 1 | minimum clear acquisitions per pixel and window |
| `s1` | `separate_orbits` | false | keep ascending/descending composites apart |
| | `statistic` | `mean` | linear-domain `mean` or `median` |
| `features` | `s2_bands` | all 10 | bands copied into the stack |
| | `s2_indices` | NDVI, EVI, MSAVI2, NDRE, RECI, GCVI, NDMI, BSI | see [features](features.md) |
| | `s1` | VV_dB, VH_dB, VHVV_dB | radar features |
| | `deltas` | NDVI, NDRE, NDMI | consecutive-window differences (S2 or S1 feature names) |
| | `seasonal` | NDVI | max/min/range/argmax over windows |
| `classes` | `label_field`, `id_field`, `targets` | `class_name`, `polygon_id`, [] | reference schema; target classes get extra columns in ablation tables |
| `model` | `n_estimators`, `max_features`, `min_samples_leaf`, `max_depth`, `class_weight`, `n_jobs` | 500, sqrt, 1, null, balanced_subsample, −1 | Random Forest |
| | `max_samples_per_polygon` | 250 | sampling cap |
| | `cv_folds`, `spatial_block_m` | 5, 5000 | spatial CV |
| | `tune_grid` | null | e.g. `{max_features: [sqrt, 0.3], min_samples_leaf: [1, 3]}` |
| `experiments` | `<id>: {groups, periods, features, exclude}` | E0–E4 ladder | ablation definitions |
| `parcels` | `id_field`, `declared_field`, `crop_mapping` | `parcel_id`, `declared_crop`, null | parcel schema |
| | `strategy` | `mean_probability` | `majority`, `mean_probability`, `combined` |
| | `interior_buffer_px`, `min_valid_pixels` | 1, 4 | edge exclusion, resolution limit |
| | `min_confidence`, `min_majority_share`, `min_margin` | null | review thresholds (choose on validation data) |
| `validation` | `label_field`, `date_field`, `year` | `class_name`, `observed_at`, null | observation schema and season |
| | `training_exclusion_m`, `window_px`, `bootstrap` | 0, 1, 1000 | leakage buffer, GNSS window, bootstrap replicates |
