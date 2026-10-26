# Acceptance run

`geocrop acceptance` is one automatically executable, versioned acceptance test of the complete chain on a small
**real** area, with real Sentinel-2 **and** Sentinel-1 data. It produces a report that can be attached to a
public release.

```bash
geocrop acceptance --out outputs/acceptance                     # real data, Semberija 2023 (default)
geocrop acceptance --out outputs/acceptance --bbox 19.12 44.73 19.145 44.748 --start 2023-06-01 --end 2023-09-30
geocrop acceptance --out outputs/acceptance-ci --offline        # same chain and checks, synthetic, no network
```

The exit status is 0 only for the verdict `PASS`. Outputs: `acceptance_report.json` (schema
`geocrop-acceptance/1`), `acceptance_report.md`, `run_manifest.json` and every intermediate product (scenes,
composites, stack, model, prediction, parcels, validation, export). Imported scenes are cached per study grid,
so a re-run only repeats the processing.

## Chain

| Stage | What runs | Engine function |
|---|---|---|
| Grid | 10 m grid in the local UTM zone from the bounding box | `raster.make_grid` |
| Sentinel-2 | L2A COG windows from Element 84 Earth Search, cloud cover < 60 %, clear fraction in the grid ≥ 0.5 | `stac_import.import_season` |
| Sentinel-1 | RTC γ⁰ (linear) COG windows from Microsoft Planetary Computer, coverage of the grid ≥ 0.9 | `s1_stac.import_season` |
| QA | scene radiometry and backscatter plausibility | `acceptance.s2_radiometry`, `s1_stac.plausibility` |
| S2 composites | monthly windows P1… (median with cloud/shadow screening) | `s2.composite` |
| S1 co-registration | residual S1 offset per orbit geometry estimated against S2 NDVI on the first half of the windows, removed from copies of the scenes | `s1_stac.estimate_offsets`, `s1_stac.apply_offsets` |
| S1 composites | mean in linear power per window | `s1.composite` |
| Co-registration check | NDVI vs VH cross-correlation on the second half of the windows (independent of the estimation) | `acceptance.coregistration` |
| Features | full default stack (bands, indices, S1 dB/ratio, deltas, seasonal) | `features.build_stack` |
| Segmentation | OBIA (SLIC, global stretch) on the peak-NDVI window: red, NIR, SWIR1, NDVI | `segmentation.obia` |
| Pseudo-reference | k-means of per-segment NDVI trajectories, stratified 70/30 segment split | `acceptance.pseudo_reference` |
| Training | Random Forest, spatially blocked CV (500 m), calibration, conformal thresholds, AOA | `model.train` |
| Prediction | classes, probabilities, uncertainty, AOA | `model.predict` |
| Reproducibility | second training + prediction with the same seed | `model.train`, `model.predict` |
| Parcels | aggregation on all segments (mean probability, conformal sets, AOA, traffic light) | `parcels.aggregate` |
| Validation | held-out segments (parcel level) and one interior point per held-out segment | `metrics.summary`, `validation.evaluate_points` |
| Export | COG, STAC Item, GeoParquet (if `pyarrow` is installed) | `export.to_cog`, `export.stac_item`, `export.to_geoparquet` |
| Manifest | package versions, git commit, SHA-256 of key outputs | `provenance.run_manifest` |

Every stage records wall time, the peak of Python/NumPy allocations (tracemalloc; GDAL buffers are not
traced) and the process peak RSS.

## Checks

| Check | Threshold | Method |
|---|---|---|
| `grid_identity` | 0 mismatches | every scene band, composite, stack, prediction and exported COG has the grid's CRS, shape and transform |
| `s2_negative_reflectance` | ≤ 2 % | negative red reflectance in SCL-clear pixels (a double BOA offset gives ~70 %) |
| `s2_red_nir_medians` | 0 < red < 0.2, 0.1 < NIR < 0.6 | medians over clear pixels of all scenes |
| `s1_linear_range` | ≥ 95 % per scene | linear γ⁰ within 10⁻⁴…1 |
| `s1_db_ranges` | VV median −20…−3 dB, VH −28…−10 dB | per scene |
| `s1_vv_gt_vh` | ≥ 90 % per scene | pixelwise VV > VH |
| `window_acquisitions` | ≥ 1 S2 and ≥ 1 S1 per window | scene assignment to windows |
| `valid_coverage` | ≥ 90 % per window and sensor | composite pixels with a value |
| `s1_s2_coregistration` | residual shift within ±1 px, r(0,0) ≥ 0.2 | see below; evaluated only on windows not used for the offset estimation |
| `nodata_leakage` | 0 | a window's composite no-data must be no-data in every feature of that window; no value < −1000 other than the −9999 sentinel; no NaN/inf; declared nodata −9999 |
| `index_ranges` | [−1, 1] | NDVI, NDRE, NDMI, BSI per window |
| `stack_s1_s2` | S1 and S2 features | feature manifest |
| `segments_valid` | ≥ 20 valid, unique segments | OBIA output |
| `probabilities` | [0, 1], \|Σ − 1\| ≤ 10⁻⁴ | plus classes = argmax and class 255 outside valid pixels |
| `reproducibility` | bit-identical | classes and probabilities of two independent runs with the same seed |
| `parcel_schema` | all columns, one row per segment | required decision, evidence and `p_<class>` columns, grid CRS |
| `holdout_recovery` | OA ≥ 0.70, macro-F1 ≥ 0.60, ≥ 90 % decided | held-out pseudo-labelled segments |
| `export_cog` | `LAYOUT=COG` | GDAL metadata of the exported rasters |
| `export_stac` | required fields, resolvable assets | STAC 1.0.0, period, `proj:epsg`, classification classes; parsed with `pystac` when installed |
| `export_geoparquet` | rows equal, `geo` metadata | only when `pyarrow` is installed |
| `run_manifest` | all outputs hashed | core package versions present |

**Co-registration method.** Green vegetation (crops, tree lines, riparian woodland) raises cross-polarised
volume scattering, so S2 NDVI and S1 VH backscatter (dB) are positively correlated over arable land. Both fields
are smoothed (5 × 5, NaN-aware moving mean against speckle) and the Pearson correlation is computed with the
radar field shifted by whole pixels in both axes; the peak of the correlation surface is refined to sub-pixel
precision with a parabola per axis. Correct co-registration places the peak at (0, 0); an offset of k pixels
moves it to k (unit tests recover a 2-pixel and a (1.5, −3)-pixel shift).

**Sentinel-1 geolocation finding (2023, Semberija).** The first real run, without correction, failed this
check: the Planetary Computer RTC scenes are offset from Sentinel-2 L2A by a **systematic, orbit-dependent**
amount, stable through the season (estimated scene by scene): about 50 m for ascending relative orbit 73, 35 m
for ascending 175, ~10 m for descending 153 and < 10 m for descending 51, mostly in the east–west direction
(all scenes NRT-3h with predicted orbits; the one RESORB scene of orbit 73 shows the same offset). Our windowed
reprojection was cross-checked against GDAL's warper (r = 0.9999), and the descending orbit 51 agrees with
Sentinel-2, so the offset is in the RTC products, not in the engine or in Sentinel-2. The acceptance chain
therefore estimates one offset per acquisition geometry from the calibration windows (June–July), applies
offsets with a clear correlation peak and ≥ 0.5 px to copies of the scenes (bilinear, linear power; raw cached
scenes stay untouched; each `scene.json` records the shift), and verifies the result on the independent
windows (August–September): the residual offset drops from about (−0.8, −2.0) px to below 0.2 px.
`--no-s1-coregistration` keeps the delivered geolocation (the check then reports the raw offset).

**Verdicts.** `PASS` — every stage ran and every check passed. `FAIL` — a check failed or an engine stage
raised (the error and traceback are in the report). `INCOMPLETE` — the external data could not be obtained
(catalogue or token service unavailable, or no usable acquisitions); this is not an engine verdict, and the
network test skips on it.

## Pseudo-reference

No field labels for the area can be published. Training therefore uses a documented **pseudo-reference**:
the mean NDVI of every OBIA segment in every window forms its NDVI trajectory, and k-means (k = 4 by default,
reduced until every cluster has ≥ 4 segments; fixed seed) groups the trajectories. Clusters are named
`pseudo_A`, `pseudo_B`, … in order of mean seasonal NDVI. They are **pseudo-classes for software acceptance,
not crop types**. 70 % of the segments (stratified by pseudo-class) train the model; the other 30 % are held
out for validation and are never sampled for training.

## What the acceptance run proves

- The complete chain runs unattended on public data for a real area and season, from STAC search to
  exported products.
- Imported Sentinel-2 reflectance (BOA offset handling) and Sentinel-1 γ⁰ are physically plausible.
- All rasters share one grid; after per-geometry offset correction, Sentinel-1 and Sentinel-2 are
  co-registered within one 10 m pixel on windows independent of the correction.
- No no-data sentinel leaks into features; indices and probabilities respect their mathematical ranges.
- A model trained on the stack recovers a structure present in the data on segments it did not see.
- Training is reproducible with a fixed seed, outputs follow the documented schemas (parcel table, COG,
  STAC Item, GeoParquet), and the run is traceable through versions, git commit and output hashes.

## What it does NOT prove

- **Crop-type accuracy.** The pseudo-classes are derived from the same NDVI that the model sees, so their
  recovery is expected to be high; the scores are not comparable with crop-classification accuracy.
  Accuracy requires independent field data ([classification and validation](classification-validation.md)).
- That OBIA segments are agricultural parcels (no reference boundaries are used).
- Calibration or conformal coverage on independent data: held-out segments neighbour training segments
  (spatial autocorrelation is not controlled in this split).
- Transferability to other regions, seasons, processing baselines or data providers; in particular, the
  Sentinel-1 offsets are specific to this area and these products and must be re-estimated elsewhere.
- Absolute geolocation: both sensors are only compared with each other (Sentinel-2 is the reference).
- Performance at regional scale (see the `slow` test tier and [deployment](deployment.md) for sizing).

In `--offline` mode the same chain runs on the synthetic scene generator (`demo.generate_scenes`): it
verifies the software only, nothing about real data.

## Data sources, licences and attribution

| Data | Source | Licence | Attribution |
|---|---|---|---|
| Sentinel-2 L2A | Element 84 Earth Search (`sentinel-2-l2a`, AWS open data), no credentials | Copernicus Sentinel data terms (free, full and open) | "Contains modified Copernicus Sentinel data 2023" |
| Sentinel-1 RTC | Microsoft Planetary Computer (`sentinel-1-rtc`), SAS-signed URLs | CC BY 4.0 (Microsoft; RTC processing by Catalyst, PlanetDEM) | "Contains modified Copernicus Sentinel data 2023; Sentinel-1 RTC from Microsoft Planetary Computer (CC BY 4.0), processed by Catalyst" |

Sentinel-1 assets are signed with a short-lived token from
`https://planetarycomputer.microsoft.com/api/sas/v1/token/sentinel-1-rtc` (anonymous at the time of writing;
set `PC_SDK_SUBSCRIPTION_KEY` to send a subscription key). If the optional `planetary-computer` package is
installed it is used for signing. The report lists every STAC item id used, and `scene.json` of every
Sentinel-1 scene records its source items, licence and attribution.

## Published reports

| Report | Engine | Area | Verdict |
|---|---|---|---|
| [acceptance-0.10-dev-2026-10-26](acceptance/acceptance-0.10-dev-2026-10-26.md) | 0.10 development (`v0.9.0-5-g56d14bc`) | Semberija, 4.2 km², June–September 2023; 15 Sentinel-2 + 38 Sentinel-1 acquisitions | PASS, 21/21 checks |

Large data are never committed; re-run the command to reproduce a report (STAC items are listed by id).

## Continuous integration

- Default tier (`pytest`): `tests/test_acceptance.py` runs the offline acceptance (≈ 20–30 s) and unit tests of
  the verdict logic, the co-registration estimators (known synthetic shifts) and Sentinel-1 signing.
- Network tier (`pytest -m network`): real Sentinel-1 RTC import plausibility and the full real-data
  acceptance run (several minutes, mostly HTTP reads); it skips when a data service is unavailable.
- A release job can run `geocrop acceptance --out acceptance` and attach `acceptance/acceptance_report.md`
  and `.json` to the release.
