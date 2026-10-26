# Event and homogeneity markers (Sentinel-1 + Sentinel-2)

The phenology markers in [cbm.md](cbm.md) use one Sentinel-2 index. On its own, that index cannot tell a
harvest from drought senescence, and it cannot see ploughing, mowing or a parcel that is split between two crops.
`geocrop.cbm_markers` adds event markers that combine Sentinel-2 parcel series with Sentinel-1 backscatter. It
also adds a within-parcel homogeneity marker. Every marker has an explicit confidence and explicit
data-sufficiency fields, and optional traffic-light rules use them with reason codes.

The ideas follow the operational marker literature (Sen4CAP / Sen2-Agri, JRC CbM, Sinergise area-monitoring
markers, and the papers listed below). No code from those projects is used.

> **Status.** The thresholds are defaults chosen from the literature. They were checked only on *synthetic*
> parcel series (see [Tests](#tests)). They have not been validated on real Sentinel-1/2 data and must be tuned
> on regional reference parcels before any operational use. Markers support the prioritisation of inspections.
> They are never findings of non-compliance.

## Workflow

```bash
geocrop signatures    --scenes data/interim/s2 --parcels lpis.gpkg --reference grid.tif \
                      --start 2023-03-15 --end 2023-10-31 --indices NDVI --moments --out s2_sig.parquet
geocrop s1-signatures --scenes data/interim/s1 --parcels lpis.gpkg --reference grid.tif \
                      --start 2023-03-15 --end 2023-10-31 --out s1_sig.parquet
geocrop event-markers --signatures s2_sig.parquet --s1-signatures s1_sig.parquet \
                      --start 2023-03-15 --end 2023-10-31 [--config study.yaml] --out events.csv
geocrop traffic-light --parcels outputs/run/parcels/parcels.gpkg --markers events.csv \
                      --config study.yaml --out parcels_reviewed.gpkg
```

Python: `cbm.s1_signatures`, `cbm.parcel_signatures(..., moments=True)`, `cbm.event_markers`,
`cbm.homogeneity`, `cbm.traffic_light_details`, `parcels.aggregate(..., marker_rules=...)`.

### Parcel series

* **Sentinel-2** (`parcel_signatures`): mean, std, `n_valid`, `n_pixels` per parcel and acquisition over
  interior, cloud-free pixels. With `moments=True` (`--moments`) the table also gets `skewness` and Sarle's
  `bimodality` coefficient of the pixel values. The default output is unchanged.
* **Sentinel-1** (`s1_signatures`): `VV_dB`, `VH_dB` and `VHVV_dB` per parcel and acquisition from S1 scene
  folders (linear `VV.tif`/`VH.tif`, `scene.json` with `orbit_direction` and `relative_orbit`). `mean` is
  10·log10 of the parcel mean in *linear* power (the ratio uses the linear means), because averaging dB values
  biases the mean. `std` is the standard deviation of the pixel dB values. Each row carries `orbit_direction`,
  `relative_orbit` and `backscatter` (sigma0/gamma0).
* **Orbit geometry.** S1 series are analysed separately per orbit group (`s1_group_by: orbit_direction`
  (default) or `relative_orbit`). Ascending and descending acquisitions see the parcel at different incidence
  and aspect angles, so mixing them creates spurious "changes". Evidence from the groups is combined only after
  each group has been analysed on its own.

Sentinel-2 series are binned to a 5-day grid and Whittaker-smoothed (Eilers, 2003). Events use a lighter
smoothing (`lambda` 5) than the phenology markers (20), so that abrupt drops stay abrupt. Mowing uses
`mowing_lambda` 2, because a cut and regrowth cycle lasts only a few weeks. Days are day-of-year, so a season
must lie within one calendar year.

### Sentinel-1 step changes

For each orbit group and backscatter feature, the step at acquisition date *d* is
`median(values in [d, d+W)) − median(values in [d−W, d))`, with `W = s1_window_days` (24) and at least
`s1_min_obs` (2) acquisitions on each side. Its z-score uses a robust noise estimate: 1.4826·MAD of consecutive
differences / √2, floored at 0.1 dB. A step qualifies when |step| ≥ `s1_step_db` (1.5 dB) **and** |z| ≥
`s1_min_z` (3). Medians over several acquisitions suppress single-date effects such as rain or wet soil.

## Markers

| Marker | Columns | Definition |
|---|---|---|
| data sufficiency | `s2_clear_observations`, `s2_observed_fraction`, `s2_longest_gap_days`, `s1_acquisitions`, `s1_orbit_groups` (`group:n;...`), `s2_sufficient`, `s1_sufficient`, `data_sufficiency` (`s2+s1`/`s2`/`s1`/`insufficient`) | S2 is sufficient with ≥ `min_s2_observations` (8) occupied grid bins, no gap > `max_s2_gap_days` (45) and a complete smoothed series. S1 is sufficient with ≥ `min_s1_acquisitions` (6) dates in at least one orbit group. Observations covering < `min_coverage` (50 %) of the parcel are ignored. |
| harvest | `harvest_evidence`, `harvest_detected`, `harvest_event_doy`, `harvest_s2_drop`, `harvest_s1_step_db`, `harvest_s1_doy`, `harvest_s1_groups`, `harvest_confidence` | See below. |
| tillage / ploughing | `tillage_evidence`, `tillage_detected`, `tillage_doy`, `tillage_s1_step_db`, `bare_days_after_harvest`, `tillage_confidence` | Bare soil after harvest (smoothed NDVI < `bare_soil_ndvi` 0.25 for ≥ `min_bare_days` 15) together with a persistent **rise** of VV or VH ≥ `tillage_step_db` (1.5 dB) during the bare period. The rise must come at least `s1_match_days` after the harvest, so that the harvest step itself is not counted. |
| mowing | `mowing_events`, `mowing_doys` (`doy;doy;...`), `mowing_s1_supported`, `mowing_confidence` | A drop of ≥ `mowing_drop` (0.15) within `mowing_window_days` (30), followed within `recovery_days` (40) by regrowth of ≥ `recovery_fraction` (50 %) of the drop. Each event counts as S1-supported when a VH or VH/VV step of either sign occurs within ± `s1_match_days` (18). |
| homogeneity | `homogeneity` (`homogeneous`/`heterogeneous`/`insufficient_data`), `homogeneity_basis` (`std+bimodality`/`std_only`), `homogeneity_dates`, `heterogeneous_dates`, `heterogeneous_fraction`, `max_within_std`, `max_bimodality`, `homogeneity_confidence` | A date is heterogeneous when the within-parcel std ≥ `heterogeneity_std` (0.1 NDVI) **and** Sarle's bimodality coefficient ≥ 5/9. A parcel is heterogeneous when ≥ `heterogeneity_min_dates` (2) dates and ≥ `heterogeneity_min_fraction` (25 %) of its usable dates (≥ `min_pixels_homogeneity` 10 clear pixels) are heterogeneous. |

All drop events also need ≥ `min_event_observations` (2) clear observations in their low phase. This stops a
single residual cloud or shadow from creating an event.

### Harvest: Sentinel-2 and Sentinel-1 together

An event is *abrupt* when smoothed NDVI falls by ≥ `abrupt_drop` (0.2) within `abrupt_window_days` (20)
after the season peak. A *gradual* decline is a total post-peak decline ≥ `harvest_drop` (0.2) with no abrupt
event. The harvest step in Sentinel-1 is a **decrease** of VH or VH/VV, because removing the crop removes
volume scattering (`harvest_s1_sign: negative`; `any` accepts both signs).

| `harvest_evidence` | Condition | `harvest_detected` | confidence (before cap at 1) |
|---|---|---|---|
| `harvest_s2_s1` | abrupt S2 drop and an S1 step within ± `s1_match_days` | yes | 0.6 + 0.2·s2 + 0.2·s1 |
| `harvest_s2` | abrupt S2 drop, no S1 data around it | yes | 0.4 + 0.3·s2 |
| `harvest_s2_unconfirmed` | abrupt S2 drop, S1 data present but no step | yes | 0.3 + 0.3·s2 |
| `harvest_s1` | S1 step after the peak, with only a gradual S2 decline (≥ half `harvest_drop`), e.g. a harvest after maturation; or S1 only when S2 is insufficient (score × 0.7) | yes | 0.4 + 0.3·s1 + 0.1·min(1, decline / harvest_drop); S1 only: 0.7·(0.4 + 0.3·s1) |
| `senescence` | gradual S2 decline, S1 sufficient and present after the peak, no step: maturation or drought senescence of a standing crop | no | 0.5 + 0.3·decline |
| `senescence_or_harvest` | gradual decline without S1 (cannot be separated) | no | 0.2 |
| `none` | no qualifying decline or step | no | 0.7 with S1, 0.5 without; 0.3 when S2 is insufficient and S1 shows no step |
| `insufficient_data` | neither sensor sufficient | no | 0 |

`s2` = min(1, drop / (2·abrupt_drop)), `s1` = min(1, |z| / (2·s1_min_z)) and, in the `senescence` row,
`decline` = min(1, decline / (2·harvest_drop)). **Confidences are ordinal heuristic scores, not calibrated probabilities.** They rank
evidence within one configuration and are comparable only under the same parameters. Calibrating them needs
inspected reference parcels.

## Traffic-light rules

Marker rules live in `cbm.rules` (defaults in `geocrop.cbm_markers.RULE_DEFAULTS`). They are **off** by
default, and while they are off `traffic_light` and the parcel outputs are unchanged.

```yaml
cbm:
  events: {abrupt_drop: 0.2, s1_group_by: orbit_direction}   # any EVENT_DEFAULTS key
  rules:
    enabled: true
    min_marker_confidence: 0.5        # marker evidence below this is ignored
    harvest_expected_classes: [maize, soybean, sunflower, winter_crops]
    no_harvest_outcome: yellow         # ignore | yellow | red
    mowing_expected_classes: [grassland]
    min_mowing_events: 1
    no_mowing_outcome: yellow
    tillage_forbidden_classes: [permanent_grassland]
    tillage_outcome: red
    heterogeneity_outcome: yellow
    mixed_class_share: 0.3             # 2nd-largest pixel-class share (f_<class>), null = off
    mixed_class_outcome: yellow
    insufficient_data_outcome: ignore
```

Rules can only escalate a colour (green → yellow → red). Grey stays grey. A review decision or an uncertain
declaration stays yellow, but the marker reasons are still recorded for the analyst. "Absence" rules (no
harvest, too few mowing events) fire only when the absence has confidence ≥ `min_marker_confidence` and the
data are sufficient. Otherwise they produce `insufficient_marker_data_<marker>`, with the outcome
`insufficient_data_outcome`.

`traffic_light_reasons` (semicolon-separated) contains:
* base codes: `review_decision`, `calendar_inconsistent`, `conformal_discrepancy`, `declaration_discrepancy`,
  `ambiguous_prediction_set`, `mixed_evidence`, `not_declared`, `insufficient_data`
* marker codes: `no_harvest_evidence`, `insufficient_mowing_events`, `tillage_detected`,
  `heterogeneous_parcel`, `mixed_class_pixels`, `insufficient_marker_data_harvest`,
  `insufficient_marker_data_mowing`

`geocrop traffic-light` applies the rules from `--config` and/or `--rules` JSON to an existing `parcels.gpkg`,
with `enabled` forced to true. `parcels.aggregate(..., markers=events, marker_rules=cfg["cbm"]["rules"])` does
the same during aggregation.

In `geocrop run`, `cbm.event_markers: true` (or `cbm.rules.enabled: true`, which implies it) makes the cbm stage
compute the Sentinel-2 signatures with moments, Sentinel-1 signatures from `paths.s1_scenes` (when S1 scenes
fall in the season) and `cbm/event_markers.csv` with `cbm.events`. With `cbm.rules.enabled` the parcels stage
(and the optional parcel classifier) applies the rules and writes `traffic_light_reasons`. Both are off by
default, and then the cbm and parcels outputs and fingerprints are unchanged.

## Assumptions and limitations

* **Synthetic validation only.** The tests show that planted events are found and that stable parcels stay
  silent. They do not show accuracy on real parcels. No accuracy figure is claimed.
* **Clouds.** A cut and regrowth cycle (~3–4 weeks) or a harvest can fall into a Sentinel-2 gap. With 25 %
  random cloud loss in the synthetic series, a short mowing cycle could be missed, which is why the mowing test
  parcel is cloud-free. Sentinel-1 support per event is reported, but S1-only mowing events are not counted:
  backscatter responses to mowing are weak and variable (De Vroey et al., 2021; Lobert et al., 2021).
* **Soil moisture and rain** raise VV and VH on bare soil. Persistent medians over ≥ 2 acquisitions per side
  reduce the effect of single wet dates, but a wet spell lasting weeks can look like tillage. Precipitation data
  and interferometric coherence (Satalino et al., 2018; Mestre-Quereda et al., 2020) would make this more
  robust. Coherence needs SLC processing and is a later step.
* **Harvest sign.** The VH/VH-VV decrease after harvest is typical for broadleaf and maize canopies. Cereals can
  respond with a VV increase from the exposed soil and stubble (Veloso et al., 2017; Schlund & Erasmi, 2020).
  For such classes use `harvest_s1_sign: any` and tune `s1_step_db` per class.
* **Drought vs harvest.** The distinction holds only when Sentinel-1 is sufficient: a standing crop that
  senesces keeps its structure and shows no backscatter step. Without S1 the result is
  `senescence_or_harvest`. Very fast drought senescence (> 0.2 NDVI in 20 days) is indistinguishable from a
  harvest in S2 alone.
* **Tillage** is searched only after harvest (or after the peak). Spring tillage before sowing is not covered.
  Ploughing on the same date as the harvest is attributed to the harvest.
* **Homogeneity** needs `--moments` signatures for the bimodality test. Without them it falls back to std alone
  (`homogeneity_basis = std_only`, confidence × 0.7), which cannot tell a gradient (soil, slope) from two crops.
  Pixels at internal boundaries, partly harvested dates and small parcels (< 10 pixels) give unreliable
  statistics. The mixed-class rule uses the classifier's pixel-class shares (`f_<class>`) and inherits its errors.
* **One season, one calendar year.** Events are expressed as day-of-year.

## Tests

`tests/test_cbm_event_markers.py` (default test tier, synthetic):

* planted harvest with S1 confirmation, harvest after senescence found by S1, drought senescence without a
  step (not a harvest), ploughing after harvest, three mowing cycles with dates, a two-crop parcel
  (heterogeneous) and a wide unimodal parcel (homogeneous)
* stable parcels, one with a single cloud outlier: no harvest, tillage or mowing over 20 noise and cloud
  realisations (`pytest tests/test_cbm_event_markers.py -k seeds`)
* an S1 orbit offset of 4 dB between ascending and descending creates no change; S1 extraction checked
  against known linear values; signature moments on a half/half raster parcel
* rule outcomes, reason codes, unchanged behaviour without rules, config validation, CLI commands

## References

* Defourny, P. et al. (2019) Near real-time agriculture monitoring at national scale at parcel resolution:
  performance assessment of the Sen2-Agri automated system. *Remote Sens. Environ.* 221, 551–568.
  doi:10.1016/j.rse.2018.11.007
* De Vroey, M., Radoux, J., Defourny, P. (2021) Grassland mowing detection using Sentinel-1 time series:
  potential and limitations. *Remote Sens.* 13, 348. doi:10.3390/rs13030348
* Eilers, P. H. C. (2003) A perfect smoother. *Anal. Chem.* 75, 3631–3636. doi:10.1021/ac034173t
* Kanjir, U., Đurić, N., Veljanovski, T. (2018) Sentinel-2 based temporal detection of agricultural land use
  anomalies in support of common agricultural policy monitoring. *ISPRS Int. J. Geo-Inf.* 7, 405.
  doi:10.3390/ijgi7100405
* Kavats, O., Khramov, D., Sergieieva, K., Vasyliev, V. (2019) Monitoring harvesting by time series of
  Sentinel-1 SAR data. *Remote Sens.* 11, 2496. doi:10.3390/rs11212496
* Lobert, F., Holtgrave, A.-K., Schwieder, M., Pause, M. et al. (2021) Mowing event detection in permanent
  grasslands: systematic evaluation of input features from Sentinel-1, Sentinel-2, and Landsat 8 time series.
  *Remote Sens. Environ.* 267, 112751. doi:10.1016/j.rse.2021.112751
* Mestre-Quereda, A., Lopez-Sanchez, J. M. et al. (2020) Time-series of Sentinel-1 interferometric coherence
  and backscatter for crop-type mapping. *IEEE JSTARS* 13, 4070–4084. doi:10.1109/JSTARS.2020.3008096
* Pfister, R., Schwarz, K. A., Janczyk, M., Dale, R., Freeman, J. B. (2013) Good things peak in pairs: a note on
  the bimodality coefficient. *Front. Psychol.* 4, 700. doi:10.3389/fpsyg.2013.00700
* Satalino, G., Mattia, F., Balenzano, A., Lovergine, F. P., Rinaldi, M. et al. (2018) Sentinel-1 & Sentinel-2
  data for soil tillage change detection. *IGARSS 2018*. doi:10.1109/IGARSS.2018.8519103
* Schlund, M., Erasmi, S. (2020) Sentinel-1 time series data for monitoring the phenology of winter wheat.
  *Remote Sens. Environ.* 246, 111814. doi:10.1016/j.rse.2020.111814
* Veloso, A., Mermoz, S., Bouvet, A., Le Toan, T. et al. (2017) Understanding the temporal behavior of crops
  using Sentinel-1 and Sentinel-2-like data for agricultural applications. *Remote Sens. Environ.* 199,
  415–426. doi:10.1016/j.rse.2017.07.015

All DOIs were checked against the Crossref API on 2026-10-26.
