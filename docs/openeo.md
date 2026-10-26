# openEO back-end (optional)

GeoCrop processes everything locally by default. The optional openEO adapter (`geocrop.openeo_adapter`) can move
the data-heavy steps (download, masking, compositing, parcel aggregation) to an openEO back-end, by default the
Copernicus Data Space Ecosystem (CDSE) at `https://openeo.dataspace.copernicus.eu`, and writes the downloaded
results in GeoCrop's own file contracts. Feature stacks, training, validation, parcel decisions and
Checks-by-Monitoring markers stay local and unchanged.

The same code exports a study's recipe as openEO process graphs, without credentials and without the openEO
client, so the preprocessing can be archived and re-run elsewhere.

## When to use it

Use it when:

* the AOI or season is large enough that downloading SAFE products or windowed COGs is the bottleneck;
* SNAP is not available on the VM and Sentinel-1 backscatter is needed (`s1_composites` uses the back-end's
  `sar_backscatter` or CARD4L processing);
* parcel time series are needed for many parcels and the scenes themselves are not (`signatures`);
* you want an archived, machine-readable description of the preprocessing (`geocrop openeo-graph`).

Keep local processing when exact reproduction of GeoCrop masking matters (projected shadows, external cloud
masks, temporal screening), when there is no network or no CDSE account, or for small AOIs where local
STAC import is already fast. A middle path is `s2_scenes`: the back-end only cuts and resamples the scenes,
masking and compositing then run locally with GeoCrop's exact rules.

## Installation and credentials

```bash
pip install "openeo>=0.30"     # Apache-2.0; only needed for real runs, not for graph export
```

Credentials are read from environment variables only. GeoCrop never writes them anywhere.

| Variable | Meaning |
|---|---|
| `CDSE_OPENEO_CLIENT_ID`, `CDSE_OPENEO_CLIENT_SECRET` | OIDC client-credentials flow (service accounts, CI) |
| `CDSE_OPENEO_PROVIDER_ID` | optional OIDC provider id |
| `CDSE_OPENEO_URL` | optional back-end URL (overrides `openeo.url`) |

Without client credentials, `--auth auto` (the default) and `--auth device` start the interactive OIDC device
flow (open the printed URL and log in with your CDSE account). The adapter calls it with
`store_refresh_token=False`, so the openEO client does not cache a refresh token either.

## Configuration

All keys are optional; the defaults are:

```yaml
openeo:
  url: https://openeo.dataspace.copernicus.eu
  products: [s2_composites]        # s2_composites, s1_composites, s2_scenes, signatures
  s2_collection: SENTINEL2_L2A
  s1_collection: SENTINEL1_GRD
  s1_coefficient: sigma0-ellipsoid # or gamma0-terrain (sar_backscatter)
  s1_card4l: false                 # true: ard_normalized_radar_backscatter (CARD4L NRB) instead
  max_cloud_cover: null            # optional eo:cloud_cover filter (scene-level, percent)
  signature_indices: [NDVI, BSI]
  start: null                      # period for s2_scenes/signatures; default timeseries.start/end, else the windows' span
  end: null
  job_options: null                # passed to the back-end batch job (e.g. memory settings)
  work_dir: data/openeo            # graphs, downloads, signatures.csv
```

The study reference grid (`grid.reference`, from `geocrop grid`) is required: its EPSG code, resolution and
bounds define the graphs, and every result is aligned to it. Windows, `compositing.*` and `s1.*` come from the
same configuration as local runs.

## Commands

```bash
# export the recipe (no credentials, no openeo client needed)
geocrop openeo-graph --config study.yaml --out recipe/openeo
geocrop openeo-graph --config study.yaml --products s2_composites --stdout > graphs.json

# dry run: write the graphs that a real run would submit
geocrop openeo-extract --config study.yaml --dry-run

# real run (creates batch jobs and consumes credits)
geocrop openeo-extract --config study.yaml --products s2_composites s1_composites
geocrop run --config study.yaml --stages features samples train predict parcels validate
```

`openeo-graph` and `--dry-run` write one openEO process object per graph (`<name>.json`: `id`, `summary`,
`description`, `process_graph`) and `recipe.json` (reference grid, collections, compositing parameters, band
order of every result, target paths, SHA-256 of every graph, and the list of differences below).

A real run submits each graph as a batch job, waits for it, downloads the result into
`<work_dir>/downloads/<name>/` and converts it:

| Product | Graph | Written to (GeoCrop contract) |
|---|---|---|
| `s2_composites` | `load_collection` -> `resample_spatial` -> mask (`reduce_dimension` over bands, optional `apply_kernel` dilation) -> `reduce_dimension` over time (median/mean/`quantiles`) + `count` | `<paths.composites>/<window>/B02.tif … B12.tif`, `valid_count.tif`, `composite.json` |
| `s1_composites` | `load_collection` (orbit filter when `s1.separate_orbits`) -> `sar_backscatter` or `ard_normalized_radar_backscatter` -> `resample_spatial` -> values > 0 -> mean/median | `<paths.composites>/<window>/[ASC_|DESC_]VV.tif`, `VH.tif`, `composite_s1.json` |
| `s2_scenes` | `load_collection` -> `resample_spatial` (one GeoTIFF per date) | `<paths.s2_scenes>/openeo_<date>/` scene folders (reflectance, `SCL.tif`, optional `CLDPRB.tif`, `scene.json`) |
| `signatures` | per pixel: index sum and sum of squares over clear pixels, clear count, pixel count (`apply_dimension`) -> `aggregate_spatial` with `sum` | `<work_dir>/signatures.csv` in the `geocrop signatures` table layout (`parcel_id, date, index, mean, std, n_valid, n_pixels`) |

Conversion checks before writing anything: expected band count, CRS present, the result covers the whole
reference grid, and (for Sentinel-2) that at most 5 % of the red reflectance is negative (the same guard as
`stac-import`). Results lying on the reference lattice (same CRS and resolution, integer pixel offset) are
copied pixel-for-pixel (`grid_alignment: exact` in the provenance); otherwise they are resampled with
`geocrop.raster.read_aligned` (bilinear for reflectance/backscatter, nearest for counts and SCL) and marked
`grid_alignment: resampled`. `composite.json`, `composite_s1.json` and `scene.json` record the back-end URL,
job id, graph SHA-256, client version and the differences list. Values are reflectance in [0, 1] (DN × 1e-4)
and linear backscatter; nodata is -9999, as for local products. `compositing.min_valid_observations` is
applied locally from the downloaded `valid_count`.

## Differences from local masking and compositing

The graphs reproduce GeoCrop's rules as far as standard openEO processes allow:

* **Same rules:** SCL classes in `compositing.scl_invalid` are invalid; an observation needs every band with
  0 < DN ≤ 15000 (local import requires DN > 0, local compositing requires reflectance ≤ 1.5);
  `cloud_probability_max` compares the back-end `CLD` band; dilation covers only the cloud/shadow classes, not
  SCL 0, with a (2k+1)×(2k+1) box kernel, which equals k iterations of the local 3×3 binary dilation
  (both checked by tests); the statistic is median, mean or the linear-interpolation percentile (`quantiles`,
  same definition as NumPy's default). Index formulas for signatures are checked against the index registry.
* **Resampling:** every band is resampled on the back-end to the study CRS/resolution with nearest neighbour;
  local import uses bilinear for 20 m bands.
* **Same-day tiles:** the back-end merges observations of one day with its own rule; locally the first clear
  tile per pixel wins.
* **Not available in `s2_composites`:** projected cloud shadows (`SHADOW.tif`), external cloud masks
  (`CLOUDMASK.tif`, OmniCloudMask) and `compositing.temporal_screening` (recorded as
  `temporal_screening_requested`). Use `s2_scenes` and composite locally when these matter.
* **CLD band:** availability of `CLD` in the CDSE `SENTINEL2_L2A` collection is to be verified for your
  back-end; omit `cloud_probability_max` otherwise.
* **Radiometry:** DN are scaled by 1e-4 on the assumption that the back-end harmonises the processing-baseline
  04.00 BOA offset (to verify; the negative-red guard catches a wrong assumption).
* **Kernel edges:** `apply_kernel` border handling at the extent edge may differ; the graphs request a 2-pixel
  margin around the reference grid.
* **Signatures:** pixel inclusion in `aggregate_spatial` follows the back-end rule (local code rasterises pixel
  centres after the same interior buffer); one clear-pixel count is shared by all indices (locally, non-finite
  index values are also dropped per index); the result layout (`JSON` by date or `CSV`) is parsed in both forms.
* **Sentinel-1:** calibration, noise removal, terrain correction and the DEM are those of the back-end
  (`sar_backscatter`, or CARD4L normalised radar backscatter), not the SNAP graph of `geocrop snap-graph`.
  Compare against a local SNAP run before mixing both in one study.

## Costs and credits

openEO batch jobs on CDSE consume processing credits from the account's quota; the cost depends on area, period,
number of bands and processes. GeoCrop does not estimate credits. Before a large run, export the recipe, submit
one window over a small AOI, and check the job's reported usage in the CDSE dashboard. Current quotas and
pricing are published by CDSE and change over time.

## Reproducibility

* `geocrop openeo-graph` produces a deterministic recipe: node ids are fixed, the JSON is written with sorted
  keys, and `recipe.json` lists the SHA-256 of each graph. Golden snapshots in `tests/golden/openeo/` pin the
  graph structure; regenerate them deliberately with `GEOCROP_UPDATE_GOLDEN=1 pytest tests/test_openeo_adapter.py`.
* The graphs are standard openEO 1.x process graphs and round-trip unchanged through the openEO Python client
  (tested when the client is installed). They contain no credentials.
* Back-end collections are reprocessed and the back-end software changes, so running the same graph again may
  not give bit-identical results. Archive the downloaded results with the recipe and the run manifest.

## Limitations

* No real CDSE run has been made with this adapter in the test suite; network tests (`pytest -m network`)
  only validate graphs on the back-end and are skipped without `CDSE_OPENEO_CLIENT_ID`/`SECRET`. Treat the first
  real run as a validation run and compare it with a local composite of the same window.
* Supported signature indices: NDVI, NDRE, NDMI, NDWI, BSI, EVI, EVI2, MSAVI2, RECI, GCVI, PSRI. Plugin indices
  have no openEO expression.
* Parcel geometries are embedded in the signatures graph; very large parcel layers make large graphs and may
  need splitting into several runs.
* Feature stacks, time-series features (`timeseries.enabled`) and model steps are not computed on the
  back-end; they run locally on the downloaded composites or scenes.
* Composites written by openEO list no individual acquisitions in `composite.json` (`acquisitions: {}`), because
  the back-end does not report them per pixel.
