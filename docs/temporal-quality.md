# Temporal quality: observed vs interpolated

Whittaker smoothing fills every gap of the regular time grid. A smoothed series, its phenology
metrics (start of season, peak, end of season), the CbM markers and the parcel-classifier inputs
therefore look equally complete whether they rest on 30 clear acquisitions or on four. `geocrop.temporal_quality`
keeps the two apart and lets the traffic light act on the difference. Operational Checks-by-Monitoring
systems (Sen4CAP, the JRC CbM notebooks) and time-series toolkits (sits, FORCE) have similar
diagnostics. This module borrows ideas from them but no code.

## Definitions

| Term | Meaning |
|---|---|
| observed | a clear acquisition of the pixel or parcel. Sentinel-2: the SCL/cloud-probability/external masks. Parcel signatures: `n_valid >= min_coverage * n_pixels` and a finite mean. Sentinel-1: finite backscatter. |
| interpolated | any value of the smoothed curve between observations |
| `clear_observations` | distinct observed days in the season |
| `longest_gap_days` | longest interval without an observation. The season start and end count as bounds, so a missing first or last month counts as a gap. The interval is computed from the exact acquisition dates, not from grid bins. |
| `season_coverage` | share of grid days within `tolerance_days` of an observation |
| `<metric>_obs_distance_days` | days from the date of SOS, peak or EOS (and harvest, when markers are supplied) to the nearest observation |
| `peak_gap_days` (also `sos_gap_days`, `eos_gap_days`) | length of the observation gap containing the date. It is 0 when that day was observed. |
| `interpolated_phenology` | number of SOS/peak/EOS dates farther than `tolerance_days` from an observation |
| `interpolated_feature_fraction` | share of the smoothed-series output days (`ts_*_d<doy>`, every `output_step_days`) plus the phenology dates that are farther than `tolerance_days` from an observation |

The existing `_quality.tif` QA raster of `ts-features` (bin occupancy, policy pass) is unchanged.

## Per-pixel rasters

```bash
geocrop ts-features --scenes scenes --reference grid.tif --start 2023-03-15 --end 2023-10-30 \
  --out features/ts.tif --s2-indices NDVI --s1-indices --temporal-support --support-tolerance-days 10
```

This writes `features/ts_temporal_support.tif` next to the stack. The raster has one band per index and
metric (`tq_<INDEX>_<metric>`, band descriptions set) and the same grid and nodata as the stack. Its
manifest `ts_temporal_support.json` (contract `temporal_support`, `schema_version` 1) lists the bands,
their meaning, the tolerance and the acquisition dates. Phenology distances are nodata where the
series fails the `min_observed_fraction`/`max_gap_days` policy or has no detectable season. The
feature stack itself is unchanged, so models and manifests do not move. In Python, call
`tsfeatures.build_timeseries_stack(..., temporal_support=True, support_tolerance_days=10)`. The config key is
`timeseries.temporal_support` (default `false`).

## Per-parcel support

```bash
geocrop temporal-support --signatures cbm/signatures.csv --start 2023-03-15 --end 2023-10-31 \
  --markers cbm/markers.csv --out cbm/temporal_support.csv
```

`temporal_quality.parcel_support` recomputes the parcel curve with the `cbm.markers` recipe (index,
`step_days`, `lambda`, `min_coverage`) so the phenology matches the markers. It returns `tq_*` columns:
the metrics above plus `tq_first_observation`, `tq_last_observation`, `tq_<sos|peak|eos>_doy`,
`tq_interpolated_metrics` (for example `peak;eos`) and `tq_observed_dates`.

## Routing weak support to review

The `cbm.temporal_support` section of the config holds the measurement settings and thresholds:

```yaml
cbm:
  temporal_support:
    enabled: false                 # default: traffic light unchanged
    index: NDVI
    step_days: 5
    lambda: 20.0
    min_coverage: 0.5
    output_step_days: 15
    tolerance_days: 10             # farther from a clear acquisition = interpolated
    min_clear_observations: 6      # LOW_OBS
    max_gap_days: 60               # LONG_GAP
    max_gap_at_peak_days: 30       # LONG_GAP_AT_PEAK
    max_interpolated_phenology: 0  # INTERPOLATED_PHENOLOGY (count of SOS/peak/EOS beyond tolerance)
    max_interpolated_feature_fraction: null   # INTERPOLATED_FEATURES (null = off)
    outcome: yellow                # or ignore (record the code, keep the colour)
    missing_support_outcome: ignore   # NO_TEMPORAL_SUPPORT when a parcel has no support row
    demote_red: false              # true: red resting on weak support -> yellow (review)
```

Each threshold can be set to `null` to switch that check off. When the section is enabled,
`cbm.traffic_light_details(frame, rules, temporal_rules)` applies the checks after the base rule and
the marker rules:

* green becomes yellow when a check triggers, and the reason code is appended to
  `traffic_light_reasons`
* grey stays grey, and review decisions stay yellow
* red stays red (the codes are still recorded). With `demote_red: true` it becomes yellow, because
  a discrepancy that rests on interpolated evidence is sent to review rather than reported as
  confirmed.

`parcels.aggregate(..., temporal_support=table, temporal_rules=cfg["cbm"]["temporal_support"])`
copies the `tq_*` columns and applies the rules. To re-evaluate any existing parcel layer (pixel
aggregation or `parcel-classify`), run:

```bash
geocrop temporal-support --signatures cbm/signatures.csv --start 2023-03-15 --end 2023-10-31 \
  --out cbm/temporal_support.csv --parcels parcels/parcels.gpkg --routed parcels/parcels_routed.gpkg
```

This command applies the rules even when `enabled` is false. It also applies `cbm.rules` when they
are enabled in `--config`.

### Pipeline wiring

`geocrop run` uses both settings; both are off by default and then change no output or fingerprint:

* `timeseries.temporal_support: true`: the features stage writes `<stack>_ts_temporal_support.tif` (+ `.json`)
  with `cbm.temporal_support.tolerance_days`.
* `cbm.temporal_support.enabled: true` (with `cbm.enabled`): the cbm stage writes `cbm/temporal_support.csv`
  (`temporal_quality.parcel_support` on the cbm signatures and markers), and the parcels stage (and the optional
  `parcel_classify` stage) copy the `tq_*` columns and route weakly supported parcels to review.

## Synthetic check (software verification, not accuracy)

The following synthetic run uses the demo parcel series with the default thresholds and
`enabled: true`. It covers 144 parcels with a median of 33 clear observations:

```bash
geocrop parcel-series-demo --out demo
geocrop parcel-classify --signatures demo/signatures.csv --parcels demo/parcels.gpkg --start 2023-03-15 \
  --end 2023-10-31 --out pc --label-field true_class --block-m 200 --params '{"n_estimators": 100}'
geocrop markers --signatures demo/signatures.csv --start 2023-03-15 --end 2023-10-31 --out markers.csv
geocrop temporal-support --signatures demo/signatures.csv --start 2023-03-15 --end 2023-10-31 \
  --markers markers.csv --out support.csv --parcels pc/parcels.gpkg --routed routed.gpkg
```

The traffic light moved from 127 green, 6 yellow and 11 red to 124 green, 9 yellow and 11 red. Three
green parcels were routed to review. One had `LONG_GAP_AT_PEAK` and `INTERPOLATED_PHENOLOGY`, one had
`LONG_GAP_AT_PEAK` only (a 35-day gap around the peak), and one had `INTERPOLATED_PHENOLOGY` only.
On synthetic scenes (`demo.generate_scenes`, 192 x 192 pixels, 46 Sentinel-2 dates, 35 % cloud
events), `ts-features --temporal-support` took 2.6 s. The rasters reported 35–46 clear observations
per pixel and peak distances of 0–10 days. These numbers describe the synthetic generator only.

## Limitations

* The day-of-year arithmetic covers one calendar year. Split seasons that cross the year boundary.
* "Observed" depends on the cloud mask. Residual clouds that pass the mask count as observations.
* The tolerance and thresholds are defaults for a 5-day Sentinel-2 revisit. Tune them on regional
  reference data before operational use.
