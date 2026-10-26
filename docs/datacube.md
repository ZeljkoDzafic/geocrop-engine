# External data cubes (odc-stac, xarray, Zarr/NetCDF)

`geocrop.datacube` connects GeoCrop to the data-cube world in both directions:

* **In:** prepared spatio-temporal cubes replace GeoCrop's own preprocessing. A cube is converted into GeoCrop's
  scene-folder or composite contract ([data contracts](data-contracts.md)) on the study reference grid, so
  `composite`, `features`, `train`, `predict`, `parcels` and the CbM stages run unchanged.
* **Out:** scene folders (the per-acquisition time series), window composites and feature stacks are exported
  as `xarray` Datasets with CF-1.8 metadata, written as Zarr or NetCDF, for use in Open Data Cube, xarray/dask,
  QGIS/GDAL or other tools.

Three entry points:

| Command | Function | What it does |
|---|---|---|
| `geocrop cube load` | `load_stac_cube` + `cube_to_scenes` | Sentinel-2 L2A from any STAC API (default Earth Search) via **odc-stac**, loaded directly onto the study grid |
| `geocrop cube import` | `normalise_cube` + `cube_to_scenes` / `cube_to_composites` | validates a user / Open Data Cube cube and writes scene folders or window composites |
| `geocrop cube export` | `scenes_to_cube`, `composites_to_cube`, `stack_to_cube`, `write_cube` | GeoCrop outputs -> CF Zarr/NetCDF |

## Installation

All dependencies are optional and imported lazily; a missing package is reported with the install command.

```bash
pip install -e '.[cube]'           # all of the following (extra `cube`)
pip install xarray                 # every cube function (Apache-2.0)
pip install "odc-stac>=0.4"        # cube load only (Apache-2.0; pulls odc-geo, odc-loader)
pip install zarr                   # .zarr input/output (MIT)
pip install netCDF4                # .nc input/output (MIT)
```

## When to use it

Use `cube load` (odc-stac) when:

* you already work with odc-stac / xarray and want the same STAC source loaded exactly like `stac-import`, with
  parallel reads (`pool`) and, from Python, optional dask chunking (`chunks={"x": 2048, "y": 2048}`);
* you want the loaded season also as one Zarr/NetCDF cube (`--save-cube`).

Use `cube import` when:

* the imagery already exists as a cube: an Open Data Cube product loaded with `datacube.Datacube().load(...)`
  and saved, a cube built with odc-stac/stackstac by another team, or an analysis-ready NetCDF/Zarr delivery;
* preprocessing happens elsewhere (another processing chain, a national ARD archive) and GeoCrop should only do
  features, classification and parcel decisions.

Use `cube export` to hand GeoCrop composites, feature stacks or scene time series to xarray/ODC users.

Keep `geocrop stac-import` / `s2-import` when you have no reason to change: they are the reference path, have
no extra dependencies, and the pipeline (`geocrop run`) reads their scene folders.

## Configuration: `data.source`

```yaml
data:
  source: odc-stac          # local (default) | odc-stac
  endpoint: https://earth-search.aws.element84.com/v1
  collection: sentinel-2-l2a
  max_cloud: 80             # tile cloud-cover pre-filter (%)
  min_clear_fraction: 0.2   # clear share inside the grid (SCL screen)
```

`geocrop cube load --config study.yaml --start ... --end ...` takes the STAC settings from `data`, the grid from
`grid.reference`, the output from `paths.s2_scenes` and the SCL classes from `compositing.scl_invalid`.
**The pipeline itself does not branch on `data.source`:** `geocrop run` always reads scene folders from
`paths.s2_scenes`, whichever command filled them. `data.source` records how they were produced (it is part of the
configuration fingerprint) and documents the study; run `geocrop cube load --config` before `geocrop run`.

## `cube load`: odc-stac onto the study grid

```bash
geocrop grid --aoi aoi.geojson --crs EPSG:32634 --out grid.tif
geocrop cube load --reference grid.tif --start 2023-07-01 --end 2023-07-31 --out data/interim/s2 \
    [--save-cube july.zarr]
```

The odc-geo `GeoBox` is built from the reference raster (same CRS, affine transform and shape), so every pixel
is on the study grid; there is no second resampling step. The rules are those of `stac_import.import_item`,
and the radiometry function is shared, not re-implemented:

| Step | Rule (identical to `stac-import`) |
|---|---|
| Search | bbox of the grid, period, `eo:cloud_cover < max_cloud`, items sorted by time |
| Screening | SCL loaded first (nearest); items with no coverage (SCL all 0) or clear fraction `< min_clear_fraction` (SCL not in `scl_invalid` within the footprint) are dropped before reflectance is read |
| Radiometry | `reflectance = DN * scale + offset` from `raster:bands`; offset forced to 0 when the item declares `earthsearch:boa_offset_applied: true` (`stac_import.radiometry`); DN <= 0 = no data |
| Resampling | nearest for B02/B03/B04/B08 (10 m), bilinear for 20 m bands; each item is its own time step (no same-time fusing) |
| Rejection | more than 5 % negative clear red reflectance = failed item |
| Output | scene folders `B02.tif ... B12.tif`, `SCL.tif`, `scene.json` with `grid_id`, so a later `stac-import` reuses them as cached |

### Equivalence with `stac-import` (real data)

Semberija, northern Bosnia (19.10–19.16 E, 44.72–44.76 N), July 2023, Earth Search v1, grid EPSG:32634 10 m
(456 × 486 px): six items found, one skipped by the clear-fraction screen by both paths, five imported. Over
all five scenes and all ten bands plus SCL, the **maximum absolute difference between the `cube load` and
`stac-import` scene folders was 0** (bit-identical GeoTIFF values; the radiometry of all five items had
`boa_offset_applied: true`, processing baseline 05.09). Re-importing the `--save-cube` Zarr with `cube import`
also reproduced the same values (difference 0).

Reproduce (requires network):

```bash
pytest -m network tests/test_datacube.py::test_semberija_odc_stac_equals_stac_import
```

The default (offline) test suite checks the same equivalence on local STAC items with GeoTIFF assets at 10 and
20 m, one with and one without `boa_offset_applied`.

### Performance

Measured once on the same machine and network, same grid and period (wall-clock, network-bound, so expect
variation): `geocrop cube load` 22.4 s, `geocrop stac-import` 55.2 s (a first pair of runs: 21.2 s vs 56.4 s).
The difference comes from odc-stac reading bands concurrently (`pool=4` threads) where `stac-import` reads one
asset after another; the amount of data read is similar because both read only the windows covering the grid.
Memory: `load_stac_cube` without `chunks` holds the season in RAM (≈ 4 bytes × 10 bands × pixels × scenes; about
44 MB for this example). For large grids pass `chunks` from Python; reflectance stays lazy until written, and the
negative-red check then runs in `cube_to_scenes`.

## `cube import`: external cubes

```bash
geocrop cube import --cube product.zarr --reference grid.tif --out data/interim/s2                 # scenes
geocrop cube import --cube product.nc --config study.yaml --to composites                         # composites
```

Accepted layouts: a Dataset with one variable per band (Open Data Cube / odc-stac style), or a DataArray /
single-variable Dataset with a `band` dimension; dimension order is free (`time, y, x, band` or any other).
Validation, in order (each failure is an error naming what to fix, nothing is silently guessed):

| Check | Rule |
|---|---|
| Bands | GeoCrop names `B02 ... B12`, Earth Search names (`blue`, `rededge1`, `nir08`, `swir16` ...), ODC/DE Africa names (`red`, `nir_1`, `swir_1`, `nbart_*`), lower case; other names via `--band-map '{"my_red": "B04"}'`; all ten bands required |
| Dimensions | `x`/`y` (also `lon`/`lat`, `easting`/`northing`), `time` (also `t`, `date`); y may ascend (flipped) |
| Time | timezone-aware values are converted to UTC; naive values are read in `--timezone` (default UTC); cftime calendars are mapped when possible; sorted |
| Units | CF-decoded float reflectance (`units: "1"`); STAC-style `scale`/`offset` attributes; or `--units dn --scale 0.0001 [--offset -0.1]`. Integer DN without scale is refused; float data without unit metadata is accepted only if the red 99th percentile is <= 1.5 |
| Nodata | NaN, the `nodata` / `_FillValue` attribute or `--nodata`; for DN, values <= 0 |
| CRS | CF `grid_mapping` variable, `spatial_ref`/`crs` variable (`crs_wkt`, `spatial_ref`), `crs`/`epsg` attribute, or `--crs` |
| Grid | regular pixel-centre coordinates; same CRS and pixel size with an integer pixel offset = exact crop (partial coverage is padded with no data); anything else fails unless `--resample` (bilinear by default, nearest for SCL) |
| Values | > 5 % negative red (BOA offset applied twice) or > 50 % red above 1.5 (not reflectance) fails |
| SCL | integer ESA classes 0–11; without SCL, `--assume-masked` declares the cube cloud-masked: NaN = masked (SCL 0), other pixels SCL 7 "unclassified" (usable under the default rules), recorded as `scl_synthesised: true` in scene.json |

`--to composites` stages scene folders (temporary, or `--keep-scenes DIR`) and runs `s2.composite` with the
configuration's windows and compositing parameters, so masking, same-day mosaicking, statistics and
`valid_count` are those of the local pipeline. A cube with a `window` dimension (e.g. exported by
`cube export --kind composites`) is written directly; its `valid_count` variable is used if present, otherwise
`valid_count` is 1 where all bands are observed and `composite.json` says that the count is unknown.

## `cube export`: GeoCrop outputs as cubes

```bash
geocrop cube export --kind composites --config study.yaml --out composites.zarr
geocrop cube export --kind stack --src data/features/stack.tif --out stack.nc
geocrop cube export --kind scenes --src data/interim/s2 --start 2023-04-01 --end 2023-09-30 --out s2.zarr
```

| Kind | Dataset |
|---|---|
| `scenes` | dims (time, y, x); `B02 ... B12` (float32 reflectance, NaN = no data), `SCL` (uint8); per-time `scene_id`, `tile`, `platform`, `processing_baseline`, `boa_offset_applied`, `clear_fraction` |
| `composites` | dims (window, y, x); bands and `valid_count`; `window_start`/`window_end` when windows come from `--config` |
| `stack` | `features` (feature, y, x) with feature names, `feature_group`, `feature_sensor`, `feature_period` from the feature manifest |

CF metadata: `Conventions: CF-1.8`; a `spatial_ref` grid-mapping variable with the CF projection parameters
(`pyproj.CRS.to_cf`), `crs_wkt`, `spatial_ref` and GDAL `GeoTransform`; pixel-centre `x`/`y` with
`projection_x/y_coordinate` and units; reflectance with `standard_name: surface_bidirectional_reflectance`,
`units: "1"`, `grid_mapping`; time in UTC. GDAL reads the NetCDF output back onto the same grid
(`NETCDF:"stack.nc":B04`), and rioxarray / odc-geo recognise `spatial_ref`. The export → import round trip is
bit-identical for composites and scenes (tested).

## Differences and limitations

* `cube load` covers Sentinel-2 L2A only. Sentinel-1 cubes are not imported (use `s1-import`, the S1 STAC
  adapter or openEO `s1_composites`).
* Optional per-scene masks (`CLDPRB`, `CLOUDMASK`, `SHADOW`) are not part of the cube contract; cloud
  probability thresholds and external masks therefore do not apply to cube imports.
* odc-stac fuses nothing here (one time step per STAC item), exactly like `stac-import`; same-day tiles are
  mosaicked later by `s2.composite`.
* `--assume-masked` trusts the provider's mask; GeoCrop cannot check it.
* Resampled imports (`--resample`) are not bit-comparable with GeoCrop's own grid reads; prefer cubes produced
  on the reference grid (e.g. odc-stac with `geobox=reference_geobox("grid.tif")`).
* Whole time steps are read into memory during conversion (`cube_to_scenes` writes one step at a time).

## Licences

odc-stac, odc-geo and odc-loader (Open Data Cube project) and xarray are Apache-2.0; zarr-python and netCDF4 are
MIT. They are optional runtime dependencies, nothing is vendored. GeoCrop only borrows the cube layout and the
loading approach of Open Data Cube. Sentinel-2 data from Earth Search are Copernicus Sentinel data
(free, full and open; attribute "Contains modified Copernicus Sentinel data [year]"). Cubes supplied by users or
other providers keep their own licences (e.g. Digital Earth Africa products are CC-BY-4.0 and need attribution).
