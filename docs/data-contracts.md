# Data contracts

All rasters must share the study reference grid (CRS, transform, size) unless stated otherwise. Rasters carry
explicit nodata (`-9999` for float, `255` for class indices) and band descriptions where noted.

## Inputs

| Input | Format | Required fields / content |
|---|---|---|
| AOI | any vector | polygon(s) with CRS |
| Reference grid | GeoTIFF uint8 | from `geocrop grid`; 1 inside AOI |
| S2 scene folder | `B02.tif … B12.tif`, `SCL.tif`, optional `CLDPRB.tif`, `scene.json` | surface reflectance in [0, 1]; written by `s2-import` |
| S1 scene folder | `VV.tif`, `VH.tif`, `scene.json` | geocoded **linear** backscatter; `orbit_direction` in `scene.json` |
| Composite folder | `<window>/<B..>.tif`, `<window>/[ASC_|DESC_]VV.tif`, `valid_count.tif` | one folder per window, same grid |
| Training polygons | GeoPackage/GeoJSON | `polygon_id` (unique), `class_name`; projected CRS recommended |
| Parcels | GeoPackage/GeoJSON | `parcel_id` (unique), optional `declared_crop`; map external fields with `normalize-parcels` |
| Field observations | GeoPackage/GeoJSON points | `class_name`, `observed_at` (ISO date/time); never used for training |
| Crop / label mapping | CSV | `source,class_name` (declaration or field codes → model classes) |
| U-Net patches | `.npy` | images `[N, C, H, W]`, boundary masks `[N, H, W]` (1 = boundary) |

`scene.json` example:
```json
{"sensor": "S2", "datetime": "2023-06-12T09:30:41", "tile": "34TCQ", "relative_orbit": 36}
```

## Outputs

| Output | Format | Content |
|---|---|---|
| `stack.tif` + `stack.features.json` | GeoTIFF float32 | one band per feature (description = name); manifest lists group, sensor, window(s), formula |
| `samples.npz` | NumPy | `X, y, polygon, x, yy, features, crs` |
| `model.joblib` | joblib | estimator, ordered feature list, classes, parameters, seed, scikit-learn version |
| `classes.tif` + `class_legend.json` | GeoTIFF uint8 | class index (255 = nodata) and index → name |
| `probabilities.tif` | GeoTIFF float32 | one band per class (description = class) |
| `uncertainty.tif` | GeoTIFF float32 | `max_probability`, `margin`, `entropy` (normalised to [0, 1]) |
| `parcels.gpkg` / `.csv` / `.summary.json` | vector + table | see [parcels.md](parcels.md) |
| `train_report.json`, `cv_confusion.csv` | JSON/CSV | spatial CV results |
| `validation_report.json`, `confusion_matrix.csv`, `observations_evaluated.csv` | JSON/CSV | independent validation |
| `ablation.csv` / `.json` | CSV/JSON | experiment comparison |
| `run_manifest.json`, `stages.json` | JSON | provenance and stage cache |

Confusion matrices use rows = reference, columns = prediction.
