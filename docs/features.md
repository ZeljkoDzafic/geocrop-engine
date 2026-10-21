# Features

`geocrop features` builds one multi-band GeoTIFF from the window composites and writes
`<stack>.features.json`, which records for every band its **group**, sensor, window(s) and formula. The
manifest drives feature selection in ablation experiments, so experiments are defined by meaning
("radar only", "first two windows") rather than by band numbers.

## Groups and naming

| Group | Example names | Meaning |
|---|---|---|
| `s2_band` | `P2_B8A` | composite surface reflectance in window P2 |
| `s2_index` | `P3_NDVI`, `P1_NDRE` | spectral index computed per window from the composite |
| `s1` | `P2_VH_dB`, `P2_ASC_VHVV_dB` | radar feature per window (orbit prefix when orbits are separate) |
| `delta` | `delta_NDVI_P1_P2` | change between consecutive windows, `X(P2) - X(P1)` |
| `seasonal` | `season_NDVI_max`, `season_NDVI_argmax` | statistic over all windows (`max`, `min`, `range`, `argmax` = 1-based window of the peak) |

Indices are computed from the composite reflectances of each window, which keeps band and index values
mutually consistent. Missing inputs propagate as nodata; samples and predictions use only pixels valid in every
selected feature.

## Index registry

Generated with `geocrop indices`; the code is the single source of truth for formulas.

| Name | Sensor | Formula | Reference |
|---|---|---|---|
| NDVI | S2 | `(B08 - B04) / (B08 + B04)` | Rouse et al. (1974) |
| EVI | S2 | `2.5 (B08 - B04) / (B08 + 6 B04 - 7.5 B02 + 1)` | Huete et al. (2002) |
| EVI2 | S2 | `2.5 (B08 - B04) / (B08 + 2.4 B04 + 1)` | Jiang et al. (2008) |
| MSAVI2 | S2 | `(2 B08 + 1 - sqrt((2 B08 + 1)^2 - 8 (B08 - B04))) / 2` | Qi et al. (1994) |
| NDRE | S2 | `(B8A - B05) / (B8A + B05)` | Gitelson & Merzlyak (1994) |
| RECI | S2 | `B08 / B05 - 1` | Gitelson et al. (2003) |
| GCVI | S2 | `B08 / B03 - 1` | Gitelson et al. (2003) |
| NDMI | S2 | `(B08 - B11) / (B08 + B11)` | Gao (1996) |
| NDWI | S2 | `(B03 - B08) / (B03 + B08)` | McFeeters (1996) |
| BSI | S2 | `((B11 + B04) - (B08 + B02)) / ((B11 + B04) + (B08 + B02))` | Rikimaru et al. (2002) |
| PSRI | S2 | `(B04 - B02) / B06` | Merzlyak et al. (1999) |
| VV_dB | S1 | `10 log10(VV)` | Torres et al. (2012) |
| VH_dB | S1 | `10 log10(VH)` | Torres et al. (2012) |
| VHVV_dB | S1 | `10 log10(VH / VV)` | Veloso et al. (2017) |
| RVI | S1 | `4 VH / (VV + VH)` | Kim & van Zyl (2009); dual-pol form |

Band roles (Sentinel-2 MSI): B02 blue, B03 green, B04 red, B05-B07 vegetation red edge, B08 NIR (broad),
B8A narrow NIR, B11/B12 SWIR. Radar inputs are linear backscatter; `VHVV_dB` equals `VH_dB - VV_dB`.

## Choosing features
The default configuration is a deliberately broad *candidate* space. Use

* `geocrop ablation` to compare groups and window subsets on identical spatial folds,
* `geocrop reduce-features` to trace accuracy as the least important features are removed,
* `geocrop.experiments.group_importance` (Python) for permute-and-retrain importance of whole groups or windows,

and report the reduced model next to the full one. Impurity importances of correlated multi-temporal features
are diluted across windows; interpret them per group, not per band.

## References
Gao (1996) RSE 58:257-266 · Gitelson & Merzlyak (1994) J. Plant Physiol. 143:286-292 ·
Gitelson et al. (2003) J. Plant Physiol. 160:271-282 · Huete et al. (2002) RSE 83:195-213 ·
Jiang et al. (2008) RSE 112:3833-3845 · Kim & van Zyl (2009) IEEE TGRS 47:2519-2527 ·
McFeeters (1996) IJRS 17:1425-1432 · Merzlyak et al. (1999) Physiol. Plant. 106:135-141 ·
Qi et al. (1994) RSE 48:119-126 · Rikimaru et al. (2002) Trop. Ecol. 43:39-47 ·
Rouse et al. (1974) NASA SP-351 · Torres et al. (2012) RSE 120:9-24 · Veloso et al. (2017) RSE 199:415-426.
