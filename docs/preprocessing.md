# Preprocessing

## Reference grid
All products are resampled onto **one** grid created with `geocrop grid` (origin snapped to the resolution).
Study areas that span two UTM zones (e.g. northern Bosnia and Herzegovina, ~16.3–19.3° E across the 18° E
boundary) still use a single projected CRS; scenes from the other zone are reprojected during compositing.
Without a reference grid the first scene's grid is used, which is only acceptable for single-tile pilots.

## Sentinel-2 L2A

**Radiometry.** `s2-import` converts digital numbers with the product metadata:
`ρ = (DN + BOA_ADD_OFFSET_b) / BOA_QUANTIFICATION_VALUE` (offset −1000 from processing baseline 04.00, 2022).
DN = 0 is nodata. Import fails if the metadata are missing — reflectance is never guessed from magnitudes.

**Resolution.** B02, B03, B04, B08 are native 10 m; B05–B07, B8A, B11, B12 are bilinearly resampled from 20 m
(no spatial information is gained — this matters for small parcels and is recorded in `scene.json`). SCL and
cloud probability use nearest neighbour.

**Masking.** An observation is clear unless its SCL class is in `scl_invalid` (default 0 no data,
1 saturated/defective, 3 cloud shadow, 8/9 cloud medium/high probability, 10 thin cirrus, 11 snow/ice) or
its cloud probability exceeds `cloud_probability_max`. Invalid areas can be dilated by `mask_dilation_px`
(3 × 3 structuring element; computed with a halo so tiles have no seams) to suppress cloud halos and
undetected shadow edges. Classes 2 (dark area / topographic shadow) and 7 (unclassified) are kept by default;
add them to `scl_invalid` in mountainous or hazy areas after visual QA. Reflectance outside [−0.1, 1.5] is rejected.

**Compositing.** Scenes are assigned to windows by acquisition date (`scene.json`). Tiles acquired on the same
day are mosaicked first (first clear tile wins), so overlapping MGRS tiles never count one acquisition twice.
The per-pixel statistic (`median`, `mean` or a percentile `pNN`) is computed over clear observations only;
`valid_count.tif` stores the number of clear acquisitions and pixels with fewer than
`min_valid_observations` become nodata (never zero-filled). Report `valid_count` per window in a paper.

## Sentinel-1 GRD

`geocrop snap-graph` writes a SNAP GPT graph:

| Step | σ⁰ (default) | γ⁰ (terrain-flattened) |
|---|---|---|
| Apply-Orbit-File | precise orbits (auto download) | same |
| ThermalNoiseRemoval | VV, VH | same |
| Remove-GRD-Border-Noise | VV, VH | same |
| Calibration | σ⁰ | β⁰ |
| Terrain-Flattening | – | γ⁰ (Small, 2011) |
| Speckle-Filter | optional (e.g. Refined Lee) | optional |
| Terrain-Correction | DEM (default Copernicus 30 m), study CRS, 10 m | same |
| Write | GeoTIFF-BigTIFF, linear | same |

γ⁰ is preferable in hilly terrain; σ⁰ is simpler and adequate on plains. Choose one and keep it for the season.
Run `gpt <graph> -h` with your SNAP version to confirm operator parameters, and keep SNAP temp files on local NVMe.

`s1-import` splits the SNAP GeoTIFF into `VV.tif`/`VH.tif` and records date, absolute/relative orbit and
orbit direction. `s1-composite` averages scenes per window **in linear power** (averaging dB values biases
the mean) and, with `separate_orbits: true`, keeps ascending and descending geometries apart
(`ASC_VV.tif`, `DESC_VH.tif`, …). Conversion to dB happens in the feature stage.

Sentinel-1B stopped operating in December 2021 (mission end declared August 2022), so 2022+ seasons rely on
Sentinel-1A alone (12-day repeat per orbit) until Sentinel-1C data are available; expect fewer radar
observations per window.

### Sentinel-1 RTC from STAC (no SNAP)
`geocrop.s1_stac.import_season(grid, out, start, end)` reads radiometrically terrain-corrected Sentinel-1
backscatter (terrain-flattened γ⁰, linear power; Small 2011, processed by Catalyst with PlanetDEM) from the
Microsoft Planetary Computer collection `sentinel-1-rtc`. Only the grid window of each VV/VH COG is read and
reprojected (bilinear, in linear power) onto the study grid; slices of one overpass are mosaicked, overpasses
covering < 90 % of the grid are skipped, and each overpass becomes a standard scene folder (`VV.tif`, `VH.tif`
float32 linear, `scene.json` with `orbit_direction` from `sat:orbit_state`, relative/absolute orbit, platform,
`backscatter: gamma0`, source item ids, licence and attribution). Assets are signed with a short-lived SAS token
(the optional `planetary-computer` package when installed, otherwise the public token endpoint via `requests`;
`PC_SDK_SUBSCRIPTION_KEY` is sent when set). `s1_stac.plausibility(scene)` reports dB percentiles, the share of
linear values in 10⁻⁴…1 and the VV > VH share. Over Semberija (June 2023) 99.9 % of values lie in 10⁻⁴…1, VV
medians are −8 to −10 dB, VH −14 to −15 dB, and VV > VH in ≥ 98 % of pixels.

The RTC data are CC BY 4.0 (Microsoft; derived from Copernicus Sentinel-1). Credit products with "Contains
modified Copernicus Sentinel data <year>; Sentinel-1 RTC from Microsoft Planetary Computer (CC BY 4.0),
processed by Catalyst." RTC products use the provider's DEM and processing chain; for studies that must
control every step (orbit files, speckle filter, DEM), use the SNAP graph above.

**Geolocation.** Over Semberija (2023) the RTC scenes are offset from Sentinel-2 L2A by a systematic amount per
acquisition geometry (≈ 50 m for ascending relative orbit 73, ≈ 35 m for 175, ≈ 10 m for descending 153, < 10 m
for 51; see [acceptance](acceptance.md)). `s1_stac.estimate_offsets(scenes, s2_ndvi)` estimates one sub-pixel
offset per orbit direction and relative orbit by NDVI / VH cross-correlation, and `s1_stac.apply_offsets` writes
corrected copies of the scene folders (raw imports stay untouched; `scene.json` records the shift). Estimate the
offsets on part of the season and check the residual on the rest before composites from several geometries are
used for features.

See [timeseries.md](timeseries.md) for the full cloud/shadow screening chain (projected shadows, external
masks such as OmniCloudMask, multi-temporal outlier screening) and for time-series features from individual scenes.

## Quality control checklist
* Inventory per window: number of S2 acquisitions, S1 acquisitions per orbit direction.
* `valid_count` maps per window; gaps over target fields.
* Visual check of a few composites per window for residual clouds/shadows, especially over target crops.
* SAR: anomalies over water, built-up edges and steep slopes; ascending vs descending level differences.

## Cloud-native import from STAC (0.9)
`geocrop stac-import --reference grid.tif --start 2023-03-15 --end 2023-10-31 --out data/interim/s2` searches a
STAC API (default: Element 84 Earth Search, public Sentinel-2 L2A COGs, no credentials) over the grid footprint,
skips acquisitions whose clear fraction *inside the grid* is below `--min-clear`, and reads only the grid window of
each band over HTTP. The result is a normal scene folder per acquisition, so compositing, time series and CbM work
unchanged — without downloading full ~1 GB products.

**Radiometric offset.** For processing baseline ≥ 04.00 Earth Search already removes the −1000 DN offset
(`earthsearch:boa_offset_applied: true`) but still advertises `offset: −0.1` in `raster:bands`. Applying it would
shift reflectance by −0.1 (on 2023 scenes over Semberija, 71 % of red values became negative). The importer
therefore applies only the scale when the provider declares the offset as applied, and rejects any scene whose
valid red reflectance is more than 5 % negative.

`geocrop s2-import --safe ... --reference grid.tif` writes SAFE products directly on the study grid, which crops
each 110 km tile to the AOI and avoids a second resampling step.
