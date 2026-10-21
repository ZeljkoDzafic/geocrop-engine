# Clouds, shadows and time series

## Cloud and shadow screening
Residual clouds and shadows are the largest single error source in optical crop mapping: a missed shadow
looks like a harvest, a missed thin cloud like senescence. The engine combines independent layers; each one
only *adds* invalid pixels.

| Layer | Where | How |
|---|---|---|
| Scene classification (SCL) | `s2-import`, compositing | classes 0, 1, 3, 8, 9, 10, 11 invalid by default (`scl_invalid`) |
| Cloud probability (MSK_CLDPRB) | compositing | `cloud_probability_max` (%) |
| **Projected shadows** (`SHADOW.tif`) | `s2-import` | clouds shifted away from the sun for heights 250–6000 m, displacement `h·tan(θ_sun)`, confirmed by NIR < 0.15 (Fmask principle; Zhu & Woodcock, 2012). Uses `Mean_Sun_Angle` from `MTD_TL.xml`; computed per scene because shadows travel kilometres |
| **External masks** (`CLOUDMASK.tif`) | any scene folder | 0 = clear, > 0 = invalid. `geocrop cloudmask --scenes ...` runs OmniCloudMask (Wright et al., 2025; `pip install omnicloudmask`); outputs of s2cloudless, Fmask or manual masks work the same way |
| Halo dilation | compositing | `mask_dilation_px` with a 3×3 element, halo-safe across blocks |
| **Multi-temporal screening** | compositing (`temporal_screening: true`) | per pixel, observations with blue > clear-sky median + 0.04 (residual cloud) or NIR and SWIR < median − 0.04 (residual shadow) are dropped; needs ≥ 3 clear observations (MAJA/Tmask principle) |
| Robust temporal statistic | compositing / time series | median or low percentile composites; upper-envelope Whittaker for NDVI |

Recommended for a study: SCL + shadow projection + dilation 2 px + temporal screening; evaluate OmniCloudMask on
a sample of manually checked scenes and keep it if it removes visible residuals. Always inspect `valid_count`.

## Regular-grid time series
`geocrop ts-features` (or `timeseries.enabled: true` in the configuration) builds features directly from the
individual scenes instead of window composites:

1. clear observations of each index are binned to a regular grid (`step_days`, default 5 days; several tiles
   on the same date are averaged);
2. the weighted **Whittaker smoother** (Eilers, 2003) fills gaps and removes noise by minimising
   `Σ w (y − z)² + λ Σ (Δ² z)²`; `lambda` (default 50 on a 5-day grid) sets smoothness;
   `envelope_iterations > 0` progressively down-weights observations below the curve, fitting the upper
   envelope (useful for NDVI when residual clouds remain);
3. three feature groups are written:

| Group | Names | Content |
|---|---|---|
| `ts` | `ts_NDVI_d150` | smoothed value every `output_step_days` (day of year in the name) |
| `phenology` | `phen_NDVI_sos`, `_eos`, `_los`, `_peak_doy`, `_peak_value`, `_amplitude`, `_greenup_rate`, `_senescence_rate`, `_integral` | season metrics (TIMESAT-style; Jönsson & Eklundh, 2004): SOS/EOS = crossings of base + 50 % amplitude; NaN when amplitude < 0.1 |
| `harmonic` | `harm_NDVI_a0`, `_amp1`, `_phase1`, `_amp2`, `_phase2`, `_rmse` | weighted harmonic regression (order 2 by default) |

Radar indices (e.g. `VHVV_dB`) are processed the same way from Sentinel-1 scene folders (orbits pooled unless
`orbit` is set). `geocrop merge-stacks` combines window and time-series stacks; ablation experiments compare
them (`groups: [ts, phenology]` vs `groups: [s2_band, s2_index]`).

Timing features are what separate crops with similar spectra but shifted calendars — in the study region
sunflower peaks and senesces weeks before soybean, and winter cereals are harvested in early summer.

## References
Eilers (2003) Anal. Chem. 75:3631–3636 · Jönsson & Eklundh (2004) Comput. Geosci. 30:833–845 ·
Zhu & Woodcock (2012) RSE 118:83–94 · Zhu & Woodcock (2014) RSE 152:217–234 ·
Hagolle et al. (2010) RSE 114:1747–1755 · Wright et al. (2025) RSE (OmniCloudMask) · Aybar et al. (2022) Sci. Data (CloudSEN12).
