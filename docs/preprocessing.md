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

## Quality control checklist
* Inventory per window: number of S2 acquisitions, S1 acquisitions per orbit direction.
* `valid_count` maps per window; gaps over target fields.
* Visual check of a few composites per window for residual clouds/shadows, especially over target crops.
* SAR: anomalies over water, built-up edges and steep slopes; ascending vs descending level differences.
