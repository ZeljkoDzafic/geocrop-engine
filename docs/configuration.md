# Configuration

A study is one YAML file; `geocrop config-check --config file.yaml` prints the fully resolved configuration and
its fingerprint. Unknown keys are errors (typos never fall back silently). Annotated example:
[`configs/example.yaml`](../configs/example.yaml); case study: [`configs/rs_north_2023.yaml`](../configs/rs_north_2023.yaml).

| Section | Key | Default | Meaning |
|---|---|---|---|
| (top) | `project` | `geocrop-experiment` | name used in reports |
| | `seed` | 42 | seed for sampling, folds and models |
| `cache` | `hash` | `content` | stage-fingerprint hashing: `content` (SHA-256) or `metadata` (size + mtime, for raw archives too large to hash; see [reproducibility](reproducibility.md)) |
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
| | `embeddings` | null | optional `emb` group: `{source: <provider> or a list, year, ...}`, see [embeddings](embeddings.md); enables the `embeddings` stage of `geocrop run` |
| `compositing` | `temporal_screening` | false | drop multi-temporal outliers (residual clouds/shadows) |
| `timeseries` | `enabled`, `start`, `end`, `temporal_support` | false, –, –, false | build time-series features from scene folders and merge them into the stack; `temporal_support` also writes the per-pixel observed-vs-interpolated raster ([temporal-quality.md](temporal-quality.md)) |
| | `s2_indices`, `s1_indices` | NDVI, NDRE, NDMI; VHVV_dB | series to smooth |
| | `step_days`, `output_step_days`, `lambda`, `envelope_iterations`, `harmonic_order` | 5, 15, 50, 0, 2 | grid, output sampling, smoothness, upper envelope, harmonics |
| | `min_observed_fraction`, `max_gap_days` | 0, null | optional occupied-bin/gap policy for reconstructed features; separate QA products always emitted |
| `cbm` | `enabled`, `index`, `calendar`, `min_coverage` | false, NDVI, –, 0.5 | parcel signatures + markers; crop-calendar YAML |
| `cbm.events` | event-marker parameters (`abrupt_drop`, `s1_group_by`, ...) | see [markers.md](markers.md) | harvest/tillage/mowing/homogeneity markers |
| `cbm.rules` | `enabled`, class lists, outcomes, `min_marker_confidence`, `mixed_class_share` | off | marker rules for the traffic light, reason codes ([markers.md](markers.md)); applied by the parcels stage |
| `cbm` | `event_markers` | false | `geocrop run`: also Sentinel-1 signatures and `cbm/event_markers.csv` (implied by `rules.enabled`) |
| `cbm.temporal_support` | `enabled`, `tolerance_days`, `min_clear_observations`, `max_gap_days`, `max_gap_at_peak_days`, `max_interpolated_phenology`, `max_interpolated_feature_fraction`, `outcome`, `demote_red` | off | route weakly supported parcels to review (`LOW_OBS`, `LONG_GAP_AT_PEAK`, ...; [temporal-quality.md](temporal-quality.md)) |
| `classes` | `label_field`, `id_field`, `targets` | `class_name`, `polygon_id`, [] | reference schema; target classes get extra columns in ablation tables |
| `model` | `algorithm` | `rf` | `rf`, `extra_trees`, `hist_gb`, `lightgbm`, `catboost`, `tempcnn`, `ltae` or a registered plugin estimator |
| | `n_estimators`, `max_features`, `min_samples_leaf`, `max_depth`, `class_weight`, `n_jobs`, `learning_rate` | 500, sqrt, 1, null, balanced_subsample, −1, null | estimator settings (keys unknown to an algorithm are ignored) |
| | `calibrate` | true | temperature scaling of probabilities |
| | `calibration_mode`, `calibration_fraction` | oof, 0.2 | empirical OOF sets or frozen-predictor holdout; fraction of spatial groups reserved in holdout |
| | `max_samples_per_polygon` | 250 | sampling cap |
| | `cv_folds`, `spatial_block_m` | 5, 5000 | spatial CV |
| | `holdout_fraction` | 0 | share of spatial blocks locked away as a final test set before CV and tuning (`train_report.json` `holdout_design`); 0 disables |
| | `inner_splits` | 3 | inner spatial folds of nested tuning when `tune_grid` is set |
| | `spatial_buffer_m` | 0 | per-fold distance exclusion: purge entire training polygons near held-out sampled pixels; 0 disables |
| | `tune_grid` | null | e.g. `{max_features: [sqrt, 0.3], min_samples_leaf: [1, 3]}` |
| | `feature_groups` | null | train on these feature groups only (e.g. `[s2_band, s2_index, ts, harmonic]`); the samples stage then extracts only their features, so NaN in other groups does not remove polygons |
| | `allow_missing_classes` | false | train although a class of the reference polygons has no sample left; by default the run fails (`MissingClassError`) |
| | `params` | `{}` | estimator-specific hyperparameters passed to the estimator unchanged (e.g. a plugin's own settings); overrides the shared keys above; validated against the estimator's constructor (see below) |
| `cv` | `strategy` | `spatial_blocks` | `region`, `year`, `region_year` add leave-one-group-out transfer validation ([protocols](validation-protocols.md)); used by `geocrop run` (train, experiments) and `geocrop ablation --config` |
| | `region_field`, `year_field`, `year` | null | polygon attributes (or a constant year) to store as sample group ids (samples stage of `geocrop run`); CLI `geocrop samples --region-field/--year-field/--year` |
| `predict` | `smoothing: {enabled, window, smoothness}` | false, 7, 10 | edge-aware probability smoothing before argmax and parcel aggregation; used by `geocrop run` and `geocrop predict --config` |
| `parcel_classify` | `enabled`, `decision`, `label_field` | false, `additional`, null | optional parcel-level classifier on the cbm signatures ([parcel classification](parcel-classification.md#in-geocrop-run)); `replace` makes it the parcel product |
| | `indices`, `stats`, `groups` | [`cbm.index`], [mean, std], [ts, stats, windows, shape] | signature features |
| | `step_days`, `output_step_days`, `window_days`, `lambda`, `bootstrap` | 5, 15, 30, 20, 200 | series grid, Whittaker smoothing, bootstrap replicates |
| | `params` | `{}` | parcel estimator settings, e.g. `{algorithm: rf, n_estimators: 300}` (validated like `model.params`) |
| `experiments` | `<id>: {groups, periods, features, exclude}` | E0–E4 ladder | ablation definitions |
| `parcels` | `id_field`, `declared_field`, `crop_mapping` | `parcel_id`, `declared_crop`, null | parcel schema |
| | `strategy` | `mean_probability` | `majority`, `mean_probability`, `combined` |
| | `interior_buffer_px`, `min_valid_pixels` | 1, 4 | edge exclusion, resolution limit |
| | `min_confidence`, `min_majority_share`, `min_margin` | null | review thresholds (choose on validation data) |
| | `conformal_alpha`, `min_aoa_fraction` | 0.1, 0.5 | conformal level for parcel sets; AOA share required for acceptance |
| `validation` | `label_field`, `date_field`, `year` | `class_name`, `observed_at`, null | observation schema and season |
| | `training_exclusion_m`, `window_px`, `bootstrap` | 0, 1, 1000 | leakage buffer, GNSS window, bootstrap replicates |
| `openeo` | `url`, `products`, `s2_collection`, `s1_collection`, `s1_coefficient`, `s1_card4l`, `max_cloud_cover`, `signature_indices`, `start`, `end`, `job_options`, `work_dir` | CDSE openEO, `[s2_composites]`, … | optional openEO back-end of `geocrop openeo-graph` / `openeo-extract`; not read by `geocrop run`; defaults and meaning in [openEO](openeo.md#configuration) |
| `data` | `source` | `local` | `local` or `odc-stac`; read by `geocrop cube load --config`, not by `geocrop run` (the pipeline always reads `paths.s2_scenes`), see [datacube](datacube.md) |
| | `endpoint`, `collection`, `max_cloud`, `min_clear_fraction` | Earth Search v1, `sentinel-2-l2a`, 80, 0.2 | STAC defaults for `cube load` |
| `resources` | `disk_guard` | true | free-space check before `composite`, `features`, `predict` ([preflight](preflight.md)) |
| | `min_free_gb`, `safety_factor` | 5, 1.25 | space kept free per volume; multiplier (≥ 1) on the estimated stage size |
| | `compression_ratio` | 0.8 | expected compressed / uncompressed raster size, in (0, 1] |
| `preflight` | `min_s2_scenes_per_window`, `min_s1_scenes_per_window` | 3, 2 | warn below these scene counts |
| | `max_unobserved_fraction` | 0.05 | warn when more of the AOI lacks `min_valid_observations` clear observations |
| | `coverage_max_px`, `inventories` | 512, [] | decimated coverage grid size (≥ 16); inventory CSVs counted per window |

## Optional parts of `geocrop run`

All of the following are off by default. While they are off, `geocrop run` produces the same products and the
same stage fingerprints as before they existed (`tests/test_pipeline_options.py` pins the fingerprints of a
default configuration against GeoCrop main at c091962). Turning one on re-runs only the stages that depend on it.

| Option | Stage(s) | Products |
|---|---|---|
| `cv.strategy` (+ `region_field`/`year_field`/`year`) | samples (group ids), train, experiments | `model/transfer_report.json`, `model/transfer_per_group.csv`, transfer columns in `ablation.csv` |
| `predict.smoothing.enabled` | predict | smoothed `probabilities.tif`/`classes.tif`, `prediction/probabilities_unsmoothed.tif` |
| `features.embeddings` | embeddings, features | `<features dir>/embeddings/<source>_<year>.tif` per source (+ `monthly/` for on-VM providers), `emb` group in the stack |
| `cbm.event_markers` | cbm | `cbm/event_markers.csv`, `cbm/s1_signatures.csv` |
| `cbm.rules.enabled` | cbm, parcels | `traffic_light_reasons`, marker escalations of `traffic_light` |
| `parcel_classify.enabled` | parcel_classify (+ parcels with `decision: replace`) | `parcel_classify/model/`, `parcel_classify/parcels.*` |
| `model.params` | train, experiments | estimator built with these parameters |
| `timeseries.temporal_support` | features | `<stack>_ts_temporal_support.tif` ([temporal quality](temporal-quality.md)) |
| `cbm.temporal_support.enabled` | cbm, parcels (and parcel_classify) | `cbm/temporal_support.csv`, `tq_*` columns, `LOW_OBS`/`LONG_GAP*`/`INTERPOLATED_*` routing |

`model.params` is checked before any stage runs: the estimator selected by `model.algorithm` (built-in or plugin)
is constructed and every key must be one of its constructor parameters and must arrive there unchanged. A
misspelt key, a seed (`random_state`; the top-level `seed` is used) or a key the estimator factory does not
forward (the built-in CatBoost factory maps only the shared keys) is a configuration error. `geocrop ablation`,
`reduce-features` and `transfer --config` use the same parameters; `geocrop benchmark` compares several
algorithms and therefore uses only the shared keys.

Parallelism: there is no separate worker cap; model training follows `model.n_jobs`, raster stages are
block-wise single-process (see [preflight](preflight.md#disk-guard-during-runs)).
