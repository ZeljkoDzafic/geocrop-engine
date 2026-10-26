# GeoCrop Engine acceptance report — PASS

Schema `geocrop-acceptance/1` · engine 0.9.0 (`v0.9.0-5-g56d14bc`) · started 2026-10-26T01:23:28Z · runtime 260.1 s · mode: network (real Sentinel-1/2)

**Verdict: PASS** — 21/21 checks passed.

## Area and period

| Item | Value |
|---|---|
| Bounding box (WGS 84) | 19.11938, 44.72958, 19.1456, 44.74844 |
| Grid | EPSG:32634, 205 x 203 px at 10 m, 4.162 km² |
| Period | 2023-06-01 to 2023-09-30 |
| Windows | P1 2023-06-01..2023-06-30, P2 2023-07-01..2023-07-31, P3 2023-08-01..2023-08-31, P4 2023-09-01..2023-09-30 |

## Data sources and attribution

- **sentinel-2**: `sentinel-2-l2a` at https://earth-search.aws.element84.com/v1 — Copernicus Sentinel data terms (free, full and open). *Contains modified Copernicus Sentinel data 2023.*
- **sentinel-1**: `sentinel-1-rtc` at https://planetarycomputer.microsoft.com/api/stac/v1 — CC-BY-4.0 (Microsoft Planetary Computer; RTC processing by Catalyst). *Contains modified Copernicus Sentinel data 2023; Sentinel-1 RTC from Microsoft Planetary Computer (CC BY 4.0), processed by Catalyst.*

<details><summary>sentinel-2: 15 STAC items used</summary>

- `S2A_34TCQ_20230613_0_L2A`
- `S2A_34TCQ_20230623_0_L2A`
- `S2A_34TCQ_20230703_0_L2A`
- `S2A_34TCQ_20230713_0_L2A`
- `S2A_34TCQ_20230812_0_L2A`
- `S2A_34TCQ_20230822_0_L2A`
- `S2A_34TCQ_20230901_0_L2A`
- `S2A_34TCQ_20230911_0_L2A`
- `S2A_34TCQ_20230921_0_L2A`
- `S2B_34TCQ_20230618_0_L2A`
- `S2B_34TCQ_20230708_0_L2A`
- `S2B_34TCQ_20230718_0_L2A`
- `S2B_34TCQ_20230728_0_L2A`
- `S2B_34TCQ_20230827_0_L2A`
- `S2B_34TCQ_20230906_0_L2A`

</details>

<details><summary>sentinel-1: 38 STAC items used</summary>

- `S1A_IW_GRDH_1SDV_20230601T164200_20230601T164225_048795_05DE38_rtc`
- `S1A_IW_GRDH_1SDV_20230607T044643_20230607T044708_048875_05E0A1_rtc`
- `S1A_IW_GRDH_1SDV_20230608T163358_20230608T163423_048897_05E150_rtc`
- `S1A_IW_GRDH_1SDV_20230612T045451_20230612T045516_048948_05E2D5_rtc`
- `S1A_IW_GRDH_1SDV_20230613T164200_20230613T164225_048970_05E391_rtc`
- `S1A_IW_GRDH_1SDV_20230619T044644_20230619T044709_049050_05E5F6_rtc`
- `S1A_IW_GRDH_1SDV_20230620T163359_20230620T163424_049072_05E6A9_rtc`
- `S1A_IW_GRDH_1SDV_20230624T045452_20230624T045517_049123_05E826_rtc`
- `S1A_IW_GRDH_1SDV_20230625T164201_20230625T164226_049145_05E8E0_rtc`
- `S1A_IW_GRDH_1SDV_20230701T044645_20230701T044710_049225_05EB41_rtc`
- `S1A_IW_GRDH_1SDV_20230702T163359_20230702T163424_049247_05EBF6_rtc`
- `S1A_IW_GRDH_1SDV_20230706T045452_20230706T045517_049298_05ED86_rtc`
- `S1A_IW_GRDH_1SDV_20230707T164202_20230707T164227_049320_05EE43_rtc`
- `S1A_IW_GRDH_1SDV_20230713T044646_20230713T044711_049400_05F0AE_rtc`
- `S1A_IW_GRDH_1SDV_20230714T163400_20230714T163425_049422_05F165_rtc`
- `S1A_IW_GRDH_1SDV_20230719T164202_20230719T164227_049495_05F3A0_rtc`
- `S1A_IW_GRDH_1SDV_20230725T044647_20230725T044712_049575_05F60A_rtc`
- `S1A_IW_GRDH_1SDV_20230726T163401_20230726T163426_049597_05F6BF_rtc`
- `S1A_IW_GRDH_1SDV_20230730T045454_20230730T045519_049648_05F853_rtc`
- `S1A_IW_GRDH_1SDV_20230806T044647_20230806T044712_049750_05FB6F_rtc`
- `S1A_IW_GRDH_1SDV_20230807T163402_20230807T163427_049772_05FC27_rtc`
- `S1A_IW_GRDH_1SDV_20230811T045454_20230811T045519_049823_05FDE0_rtc`
- `S1A_IW_GRDH_1SDV_20230812T164203_20230812T164228_049845_05FEAD_rtc`
- `S1A_IW_GRDH_1SDV_20230818T044648_20230818T044713_049925_06015A_rtc`
- `S1A_IW_GRDH_1SDV_20230819T163403_20230819T163428_049947_060223_rtc`
- `S1A_IW_GRDH_1SDV_20230823T045455_20230823T045520_049998_0603D4_rtc`
- `S1A_IW_GRDH_1SDV_20230824T164204_20230824T164229_050020_06049F_rtc`
- `S1A_IW_GRDH_1SDV_20230830T044648_20230830T044713_050100_06075F_rtc`
- `S1A_IW_GRDH_1SDV_20230904T045456_20230904T045521_050173_0609DA_rtc`
- `S1A_IW_GRDH_1SDV_20230905T164205_20230905T164230_050195_060AA8_rtc`
- `S1A_IW_GRDH_1SDV_20230911T044649_20230911T044714_050275_060D52_rtc`
- `S1A_IW_GRDH_1SDV_20230912T163404_20230912T163429_050297_060E1D_rtc`
- `S1A_IW_GRDH_1SDV_20230916T045456_20230916T045521_050348_060FC4_rtc`
- `S1A_IW_GRDH_1SDV_20230917T164206_20230917T164231_050370_061092_rtc`
- `S1A_IW_GRDH_1SDV_20230923T044649_20230923T044714_050450_061345_rtc`
- `S1A_IW_GRDH_1SDV_20230924T163404_20230924T163429_050472_061410_rtc`
- `S1A_IW_GRDH_1SDV_20230928T045457_20230928T045522_050523_0615C1_rtc`
- `S1A_IW_GRDH_1SDV_20230929T164206_20230929T164231_050545_06168E_rtc`

</details>

## Sentinel-1 geolocation relative to Sentinel-2

Offsets of the Sentinel-1 RTC scenes per acquisition geometry, estimated from the calibration windows by NDVI / VH cross-correlation (pixels of 10 m; reference[y, x] ~ S1[y + dy, x + dx]). Offsets with a clear correlation peak and >= 0.5 px are removed before compositing; the co-registration check is evaluated on the remaining (independent) windows only.

| Geometry | Scenes | dy | dx | Offset (m) | r at peak | r at (0, 0) | Applied |
|---|---|---|---|---|---|---|---|
| ASC_R073 | 5 | -1.88 | -4.96 | 53 | 0.48 | 0.29 | yes |
| ASC_R175 | 5 | -1.45 | -3.14 | 35 | 0.48 | 0.37 | yes |
| DES_R051 | 4 | +0.38 | +0.62 | 7 | 0.52 | 0.51 | yes |
| DES_R153 | 5 | -0.08 | -1.14 | 11 | 0.55 | 0.54 | yes |

## Pseudo-reference (pseudo-classes for software acceptance, not crop types)

k-means (k=4, n_init=10, seed=42) of per-segment mean NDVI per window (P1_NDVI, P2_NDVI, P3_NDVI, P4_NDVI); classes ordered by mean seasonal NDVI.

Segments used: 122 (85 training, 37 held out); class counts: pseudo_A: 26; pseudo_B: 53; pseudo_C: 25; pseudo_D: 18.

| Pseudo-class | Mean NDVI per window |
|---|---|
| pseudo_A | 0.55, 0.50, 0.43, 0.40 |
| pseudo_B | 0.61, 0.60, 0.55, 0.52 |
| pseudo_C | 0.74, 0.69, 0.67, 0.65 |
| pseudo_D | 0.85, 0.81, 0.78, 0.79 |

## Checks

| Check | Stage | Result | Value | Threshold | Method |
|---|---|---|---|---|---|
| `s2_negative_reflectance` — Share of negative red reflectance in clear pixels (double BOA offset would give ~70 %) | qa | PASS | 0 | <= 0.02 | SCL-clear pixels of all imported scenes, B04 < 0 |
| `s2_red_nir_medians` — Median red and NIR reflectance of clear pixels are physically plausible for summer arable land | qa | PASS | red_median: 0.0704; nir_median: 0.304 | 0 < red < 0.2 and 0.1 < NIR < 0.6 |  |
| `s1_linear_range` — Linear gamma0 values within 1e-4..1 (worst scene) | qa | PASS | 0.9988 | >= 0.95 of valid pixels per scene | VV and VH pooled, per scene |
| `s1_db_ranges` — Median backscatter of every scene in the expected dB range | qa | PASS | vv_median_db: [-11.04, -8.23]; vh_median_db: [-16.83, -14] | VV median in [-20, -3] dB, VH median in [-28, -10] dB | 10 log10 of linear gamma0, valid pixels |
| `s1_vv_gt_vh` — Co-polarised exceeds cross-polarised backscatter (worst scene) | qa | PASS | 0.9778 | >= 0.9 of pixels per scene | pixelwise VV > VH |
| `window_acquisitions` — Every window has Sentinel-2 and Sentinel-1 acquisitions | composites | PASS | s2: P1: 3; P2: 5; P3: 3; P4: 4; s1: P1: 9; P2: 10; P3: 9; P4: 10 | >= 1 S2 and >= 1 S1 per window |  |
| `valid_coverage` — Valid-pixel coverage of every window composite (S2 after cloud screening, S1 after co-registration) | composites | PASS | P1: s2: 1; s1: 0.9999; P2: s2: 1; s1: 0.9999; P3: s2: 1; s1: 0.9999; P4: s2: 1; s1: 0.9999 | >= 0.90 per window and sensor | fraction of grid pixels with a composite value |
| `s1_s2_coregistration` — Sentinel-1 and Sentinel-2 composites are co-registered (windows independent of the offset estimation) | coregistration | PASS | verification_windows: [P3, P4]; residual_shift_px: [-0.01, 0.16]; r_at_zero_shift: 0.6773; r_at_best_shift: 0.6773; correction_applied: True; shift_without_correction_px: [-0.77, -2.01]; offsets_per_geometry_px: ASC_R073: [-1.88, -4.96]; ASC_R175: [-1.45, -3.14]; DES_R051: [0.38, 0.62]; DES_R153:... | residual shift within +/-1 px and r(0,0) >= 0.2 | Pearson r between S2 NDVI and S1 VH (dB), 5x5 NaN-aware smoothing, radar shifted by -3..3 px, surface averaged over the verification windows, sub-pixel peak; S1 offsets per orbit geometry are estimated only from the calibration windows (P1, P2) |
| `nodata_leakage` — No no-data sentinel leaks into features | features | PASS | declared_nodata: -9999; window_nodata_leaks: 0; leaking_features: ; near_sentinel_values: 0; nonfinite_values: 0 | 0 leaks, 0 near-sentinel values, 0 NaN/inf, nodata declared -9999 | composite no-data of a sensor and window -> no-data in every feature derived from it (feature manifest; nan-aware seasonal max/min/range excepted); values < -1000 other than the sentinel |
| `index_ranges` — Normalised-difference indices within [-1, 1] | features | PASS | checked: 16; out_of_range: ; ndvi: P1_NDVI: [-0.1049, 0.9377]; P2_NDVI: [-0.1726, 0.9144]; P3_NDVI: [-0.1272, 0.8964]; P4_NDVI: [-0.2183, 0.9113] | all in [-1, 1] | min/max of NDVI, NDRE, NDMI, BSI per window |
| `stack_s1_s2` — Feature stack holds Sentinel-2 and Sentinel-1 features | features | PASS | [S1, S2] | S1 and S2 present |  |
| `segments_valid` — OBIA segments are valid, non-empty, unique polygons | segmentation | PASS | segments: 122; invalid: 0 | >= 20 valid segments |  |
| `probabilities` — Class probabilities lie in [0, 1] and sum to 1 | prediction | PASS | valid_pixels: 41612; min: 0; max: 1; max_abs_sum_error: 0; argmax_matches_classes: True; nodata_class_outside: True | min >= 0, max <= 1, |sum - 1| <= 1e-4, classes = argmax, 255 outside valid pixels |  |
| `reproducibility` — Training twice with the same seed gives identical predictions | reproducibility | PASS | classes.tif: True; probabilities.tif: True | bit-identical classes and probabilities | independent second training + prediction run |
| `parcel_schema` — Parcel output schema, one row per segment, CRS of the grid | parcels | PASS | rows: 122; expected_rows: 122; missing_columns: []; crs: EPSG:32634 | all required columns, row count equal, unique ids, grid CRS | required: parcel_id, declared_class, predicted_class, decision, decision_reasons, declaration_status, traffic_light, n_valid, confidence, margin, entropy, prediction_set, cp_status, aoa_fraction, area_m2, p_<class> |
| `holdout_recovery` — Held-out pseudo-labelled segments are recovered by the trained chain | validation | PASS | overall_accuracy: 0.7297; macro_f1: 0.6982; decided: 37; holdout: 37; point_overall_accuracy: 0.7027 | OA >= 0.70, macro-F1 >= 0.60, >= 90 % of held-out segments decided | parcel-level mean-probability class vs pseudo-class on segments excluded from training; point check via geocrop.validation at one interior point per held-out segment |
| `export_cog` — Exported rasters are readable Cloud-Optimised GeoTIFFs | export | PASS | classes.tif: layout: COG; tiled: True; crs: EPSG:32634; overviews: []; readable: True; probabilities.tif: layout: COG; tiled: True; crs: EPSG:32634; overviews: []; readable: True; uncertainty.tif: layout: COG; tiled: True; crs: EPSG:32634; overviews: []; readable: True | IMAGE_STRUCTURE LAYOUT=COG and readable | GDAL COG driver metadata |
| `export_stac` — STAC Item has required fields and resolvable assets | export | PASS | assets: [class_legend, classes, probabilities, uncertainty]; hrefs_resolve: True; pystac_parse: True; proj:epsg: 32634 | stac_version 1.0.0, period, proj:epsg, classification classes, all hrefs exist |  |
| `export_geoparquet` — GeoParquet parcels round-trip | export | PASS | rows: 122; geo_metadata: True | rows equal, geo metadata |  |
| `grid_identity` — Every raster (scenes, composites, stack, predictions, exports) shares the study grid | all | PASS | rasters_checked: 287; mismatches: []; crs: EPSG:32634 | 0 mismatches | CRS equality, identical shape, affine transform equal within 1e-9 |
| `run_manifest` — Run manifest records versions and output hashes | manifest | PASS | hashed_files: 9; packages: 13 | every listed output hashed; core package versions present |  |

## Stages (runtime and memory)

| Stage | Status | Seconds | Python peak MB | Process peak RSS MB |
|---|---|---|---|---|
| grid | ok | 0.04 | 0.3 | 190.9 |
| s2_import | ok | 176.76 | 9.4 | 291.4 |
| s1_import | ok | 39.8 | 1.9 | 291.4 |
| qa | ok | 1.64 | 11.6 | 291.4 |
| s2_composites | ok | 1.72 | 17.8 | 325.8 |
| s1_coregistration | ok | 2.25 | 5.1 | 326.3 |
| s1_composites | ok | 0.83 | 6.7 | 326.3 |
| coregistration_check | ok | 0.1 | 2.5 | 326.3 |
| features | ok | 0.78 | 23.2 | 361.5 |
| segmentation | ok | 0.67 | 6.2 | 388.0 |
| pseudo_reference | ok | 1.81 | 7.8 | 405.0 |
| training | ok | 8.76 | 44.4 | 499.5 |
| prediction | ok | 3.63 | 114.2 | 593.4 |
| reproducibility | ok | 9.76 | 114.3 | 630.1 |
| parcels | ok | 0.32 | 1.2 | 630.1 |
| validation | ok | 0.34 | 0.6 | 630.1 |
| export | ok | 0.7 | 1.4 | 644.8 |
| manifest | ok | 0.75 | 8.0 | 648.3 |

Python peak = peak of Python/NumPy allocations traced during the stage (tracemalloc; GDAL buffers excluded). Process peak RSS is the high-water mark of the process up to the end of the stage.

## Validation on held-out pseudo-labelled segments

Held-out PSEUDO-labelled segments (not used in training). Measures recovery of k-means pseudo-classes, NOT crop-type accuracy.

| Level | n | OA | kappa | macro-F1 |
|---|---|---|---|---|
| segment level | 37 | 0.730 | 0.592 | 0.698 |
| point level | 37 | 0.703 | 0.567 | 0.630 |

## What this acceptance run proves

- The complete chain runs unattended on public data for a real area and season, from STAC search to exports.
- Imported Sentinel-2 reflectance (BOA offset handling) and Sentinel-1 gamma0 are physically plausible.
- All rasters share one grid; after per-geometry offset correction, Sentinel-1 and Sentinel-2 are co-registered within one 10 m pixel on windows independent of the correction.
- No no-data sentinel leaks into features; indices and probabilities respect their mathematical ranges.
- A model trained on the stack recovers a structure present in the data on segments it did not see.
- Training is reproducible (same seed -> bit-identical predictions); outputs follow the documented schemas (parcel table, COG, STAC Item, GeoParquet) and are traceable through versions, git commit and hashes.

## What it does NOT prove

- Crop-type accuracy: the classes are k-means PSEUDO-classes of the same NDVI the model sees, not crops; their recovery is expected to be high and is not comparable with crop-classification accuracy.
- That OBIA segments are agricultural parcels (no reference boundaries are used).
- Calibration or conformal coverage on independent data (held-out segments neighbour training segments).
- Transferability to other regions, seasons, processing baselines or data providers (Sentinel-1 offsets must be re-estimated elsewhere), absolute geolocation, or regional-scale performance.

## Environment

Python 3.12.15 on macOS-27.0.1-arm64-arm-64bit; geocrop-engine 0.9.0, numpy 2.5.4, pandas 3.1.0, scikit-learn 1.9.1, rasterio 1.5.2, geopandas 1.2.0, shapely 2.2.0, pyproj 3.8.0, scipy 1.18.1, scikit-image 0.26.0, torch 2.14.1, pystac-client 0.9.1, gdal (rasterio) 3.12.2

Run manifest: `run_manifest.json` (git commit `56d14bce188b9806d9eb8655b2f298ea46a8348e`, dirty: False).

| Output | SHA-256 |
|---|---|
| `features/stack.tif` | `5816d60073617db0…` |
| `features/samples.npz` | `59d78474890c3bf7…` |
| `model/model.joblib` | `73da1ed439cf04ee…` |
| `prediction/classes.tif` | `7cd7f116e01bb8f7…` |
| `prediction/probabilities.tif` | `3942324e51777379…` |
| `parcels/parcels.gpkg` | `cf3efdc20ce6f101…` |
| `export/item.json` | `498f61f570b9f649…` |
| `reference/pseudo_training.gpkg` | `7a04cb20f358c52d…` |
| `grid.tif` | `9445550be8d1364f…` |
