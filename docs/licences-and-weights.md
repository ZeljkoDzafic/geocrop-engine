# Licences, weights and external data

Release-gate audit of every external model checkpoint, dataset, service and vendored asset that `geocrop`
uses at run time, plus the licences of the Python dependencies. Audited at commit `c091962`, all checks run
on **2026-10-26**; updated the same day after the gate findings were fixed (Galileo, AnySat and OmniCloudMask
pins added and verified, Copernicus attribution, NOTICE). Nothing listed here is committed to the repository
unless stated ("vendored"): checkpoints and datasets are downloaded at run time into a cache and verified
against the pins in the engine.

**Downloads and cache.** Every run-time download goes through `geocrop.downloads.fetch`: retry with exponential
backoff on connection errors, timeouts, truncated transfers, HTTP 429/5xx (honouring `Retry-After`; tape-staged
PANGAEA files are polled), resume of an interrupted transfer with an HTTP `Range` request, size + checksum
verification before the file is moved into place (a resumed file that fails verification is downloaded once
more from scratch). Everything is cached under one root, `$GEOCROP_CACHE` (default `$XDG_CACHE_HOME/geocrop`,
i.e. `~/.cache/geocrop`), with the subfolders `ftw/`, `presto/`, `galileo/`, `anysat/`, `<provider>/`,
`alphaearth/`, `omnicloudmask/` and `public/` (EuroCrops, WorldCereal). `$GEOCROP_FTW_CACHE` and
`$GEOCROP_DATA_CACHE` are deprecated aliases for `ftw/` and `public/` (they still work and print a warning).

`tools/check_weights.py` re-runs the checksum part of this audit (see [How to reproduce](#how-to-reproduce)).

## Models and checkpoints

All pinned files were downloaded and hashed on 2026-10-26; every one matched its pin (size and checksum).
Galileo, AnySat and OmniCloudMask were added to `tools/check_weights.py` after the first audit and checked the
same day (`--only galileo`, `--only anysat`, `--only omnicloudmask`): all match.

| Model | Pin (engine) | URL | Bytes | SHA-256 matches | Licence (source consulted) | Attribution recorded |
|---|---|---|---|---|---|---|
| FTW `FTW_PRUE_EFNET_B3_CCBY` (default) | `geocrop.ftw.MODELS` | `github.com/fieldsoftheworld/ftw-baselines/releases/download/v3.1/prue_efnetb3_ccby_checkpoint.ckpt` | 53,242,827 | yes | CC-BY-4.0 checkpoint (trained on CC-BY/CC0 data only); code MIT ([repo](https://github.com/fieldsoftheworld/ftw-baselines), README "CC-BY vs. the full model") | `ftw_summary.json` (`licence`, `citation`) written by `ftw.segment`; CLI prints licence + citation to stderr (`cli.py`, FTW block) |
| FTW `FTW_PRUE_EFNET_B5_CCBY` | same | `.../v3.1/prue_efnetb5_ccby_checkpoint.ckpt` | 125,921,939 | yes | CC-BY-4.0 | same |
| FTW `FTW_PRUE_EFNET_B7_CCBY` | same | `.../v3.1/prue_efnetb7_ccby_checkpoint.ckpt` | 270,127,109 | yes | CC-BY-4.0 | same |
| FTW `FTW_v1_3_Class_CCBY` | same | `.../v1/3_Class_CCBY_FTW_Pretrained.ckpt` | 204,471,122 | yes | CC-BY-4.0 | same |
| FTW `FTW_v1_2_Class_CCBY` | same | `.../v1/2_Class_CCBY_FTW_Pretrained.ckpt` | 204,468,754 | yes | CC-BY-4.0 | same |
| FTW `FTW_PRUE_EFNET_B3` | same | `.../v3/prue_efnet3_checkpoint.ckpt` | 53,239,115 | yes | "Mixed Open Licenses" (FULL data incl. CC-BY-NC-SA); FTW recommends non-commercial/academic use only | same; blocked unless `--allow-noncommercial` (`ftw.check_licence`), warning `ftw.NONCOMMERCIAL_WARNING` |
| FTW `FTW_PRUE_EFNET_B5` | same | `.../v3/prue_efnet5_checkpoint.ckpt` | 125,916,483 | yes | Mixed, non-commercial | same |
| FTW `FTW_PRUE_EFNET_B7` | same | `.../v3/prue_efnet7_checkpoint.ckpt` | 270,119,641 | yes | Mixed, non-commercial | same |
| FTW `FTW_v2_3_Class_FULL_multiWindow` | same | `.../v2/3_Class_FULL_FTW_Pretrained_v2.ckpt` | 204,472,351 | yes | Mixed, non-commercial | same |
| Presto `default_model.pt` | `geocrop.embedding_sources.PRESTO_CHECKPOINT` | `raw.githubusercontent.com/nasaharvest/presto/11e207a6.../data/default_model.pt` | 3,321,716 | yes | MIT (repository licence, GitHub API `nasaharvest/presto`; weights live in the same repository, no separate weights licence) | GeoTIFF tags `licence`, `attribution`, `checkpoint_url`, `checkpoint_sha256` (`embeddings.SOURCE_INFO`, `embedding_sources` Presto branch); adapted encoder code keeps the MIT notice in `presto_encoder.py` docstring |
| Galileo nano `config.json` + `encoder.pt` | `feature_providers.GalileoProvider.weights` | `raw.githubusercontent.com/nasaharvest/galileo/0f0b5b95.../data/models/nano/...` | 2,246 + 4,189,406 | yes | MIT (repository licence) | GeoTIFF tags `checkpoint_url`, `checkpoint_sha256`, `licence`; MIT notice in `galileo_encoder.py` and NOTICE |
| AnySat `AnySat.pth` | `feature_providers.AnySatProvider.weights` | Hugging Face `g-astruc/AnySat` (revision-pinned) | 503,756,453 | yes | MIT (model card) | same; MIT notice in `anysat_encoder.py` and NOTICE |
| OmniCloudMask v4 weights (2 `.safetensors`, optional `[cloud]` extra) | `clouds.OMNICLOUDMASK_WEIGHTS` (revision `c9a4fb88`), downloaded by `geocrop cloudmask` into `$GEOCROP_CACHE/omnicloudmask/` and passed to the package (`destination_model_dir`, `model_version=4`) | `huggingface.co/NickWright/OmniCloudMask/resolve/c9a4fb88.../PM_model_OCM_7.97_R_G_NIR_3_smp_{regnety_004,edgenext_small}...safetensors` | 27,094,520 + 30,711,472 | yes | MIT (HF model card tag `license:mit`; package MIT) | `CLOUDMASK.tif` tags `GEOCROP_CLOUDMASK_MODEL`, `GEOCROP_CLOUDMASK_WEIGHTS` (file names + SHA-256), `GEOCROP_CLOUDMASK_SOURCE`, `licence`, `citation`; `--unpinned-weights` restores the package's own download and records `unpinned` |

## Datasets and services

HTTP status = cheap request (STAC landing page, 1 KiB range request or one tile), following redirects.

| Item | Engine reference | Pin | Status 2026-10-26 | Licence (source consulted) | Attribution requirement | Where the engine records it |
|---|---|---|---|---|---|---|
| EuroCrops `SI_2021.zip` (Zenodo 14094196) | `public_data.eurocrops_*`, `configs/benchmarks/eurocrops_si_2021_pomurje.v1.yaml` | md5 in protocol | 200; downloaded 213,298,853 B, **md5 matches** | CC-BY-SA-4.0 (Zenodo API `metadata.license`) | cite Schneider et al. 2023; ShareAlike for derived labels | benchmark results JSON/Markdown `dataset.licence` + `citation` (`public_benchmark.py`); `public_data.EUROCROPS_CITATION` |
| MKGP KMRS 2025 crop declarations (`KMRS_2025.zip`, Slovenia) | `public_data.url_layer`, `configs/benchmarks/si_transfer_pomurje.v1.yaml` (`sources:`) | SHA-256 in protocol | downloaded and checksum-verified in the transfer run of 2026-10-26 ([result](benchmarks/si-transfer-pomurje.v1.md)); only the current campaign file is online, so the pin breaks when the Ministry replaces it | open data, "citation of the source mandatory" (MKGP metadata record) | cite MKGP KMRS 2025 | transfer results JSON/Markdown label sources (`public_transfer.py`) |
| HCAT3 mapping tables (`configs/hcat/*.csv`) | `hcat.py` | - | (vendored) | CC-BY-SA-4.0, derived from EuroCrops `HCAT3.csv` | attribution + ShareAlike | `configs/hcat/LICENSE-HCAT.md` (vendored data, see findings) |
| WorldCereal crop calendars (PANGAEA 946550, 4 GeoTIFFs) | `public_data.WORLDCEREAL_*`, `calendars.py` | SHA-256 | 200 (no tape staging needed); all 4 downloaded (826 KB each), **SHA-256 match** | CC-BY-SA-4.0 (PANGAEA JSON-LD `license`) | cite Cintas/Franch et al. 2022; ShareAlike | `source` field of `geocrop calendar worldcereal` output (`cli._calendar`, `WORLDCEREAL_CITATION`) |
| AlphaEarth Satellite Embedding V1 (Source Cooperative `tge-labs/aef`, mirror GCS) | `embedding_sources.AEF_MIRRORS` | **unpinned by design**: the index `aef_index.parquet` (77,829,744 B) is a mutable file that the provider extends with every new year, and tiles are read as COG windows (range requests), so no whole-file checksum applies; the download itself uses retry/resume | 206 both mirrors | CC-BY-4.0 (`data.source.coop/tge-labs/aef/README.md`) | fixed text "The AlphaEarth Foundations Satellite Embedding dataset is produced by Google and Google DeepMind." | GeoTIFF tags `licence`, `attribution` (`embeddings.SOURCE_INFO`, sink `update_tags`) |
| TESSERA v1 embeddings + landmasks (Source Cooperative `tessera/tessera`) | `embedding_sources.TESSERA_*` | **unpinned by design**: only the needed rows of each `.npy` tile and windows of the landmask are fetched with range requests, so the provider's whole-file `SHA256SUM` cannot be checked without downloading every tile in full | 200 listing; 206 for `npy/v1/2023/grid_19.15_44.75` and `landmasks/v1/grid_19.15_44.75.tiff` | CC0 per `aup.md` in `ucam-eo/tessera` (last updated 2026-02-25), which also links CC-BY-4.0 text: conflicting, treated as most restrictive (CC-BY-SA) in `docs/embeddings.md` | credit Feng et al. 2025 | GeoTIFF tags `licence`, `attribution` |
| Planetary Computer STAC + SAS token (`sentinel-1-rtc`) | `s1_stac.py`, `acceptance.DATA_SOURCES` | - | 200 STAC, 200 collection, 206 token | CC-BY-4.0 (collection `license` field and `license` link) | "Contains modified Copernicus Sentinel data {year}; Sentinel-1 RTC from Microsoft Planetary Computer ..., processed by Catalyst" | GeoTIFF tags `licence`, `attribution` (`s1_stac.LICENCE/ATTRIBUTION`); acceptance report "Data sources and attribution" |
| Earth Search STAC (`sentinel-2-l2a`) | `stac_import.EARTH_SEARCH`, `acceptance.py`, benchmark protocol | - | 200 STAC, 200 collection | `proprietary` in STAC, licence link = Sentinel Data Legal Notice (Copernicus free, full and open) | "Contains modified Copernicus Sentinel data {year}" | GeoTIFF tags `GEOCROP_ATTRIBUTION` + `GEOCROP_DATA_LICENCE` on imported scene bands (`stac_import`) and `scene.json`; carried into composites, feature/time-series stacks and predictions (`geocrop.attribution`); COG exports and STAC items (`license: other`, `attribution`, `providers`, `license` link); acceptance report and `quicklook.json` |
| CDSE OData catalogue, download, identity, STAC | `catalog.py` | - | 200 catalogue `$top=1`, 200 OIDC config, 200 STAC; download root 404 without product id (expected, needs auth) | Copernicus Sentinel data terms | as above | SAFE import (`s2.import_safe`, S1 `import_geotiff` after SNAP) writes the GeoTIFF tags with the acquisition year; propagated as above |
| CDSE openEO | `openeo_adapter.CDSE_URL`, `config.py` | - | 206 | Copernicus terms; CDSE service terms | as above | scene folders and composites converted from openEO results carry the tags (years of the job's temporal extent); propagated as above |
| OSM standard tiles | `web_static/map.js`, CSP in `web.py` | - | 206 (tile 0/0/0) | data ODbL; tile use under OSMF tile usage policy | "© OpenStreetMap contributors" | Leaflet attribution control (`map.js` `BASEMAPS.osm`) and fine print in `index.html`; off by default |
| EOX Sentinel-2 cloudless 2023 / 2024 | `web_static/map.js` | - | 200 (tile 0/0/0 each) | **CC-BY-NC-SA-4.0** (layer abstracts in `tiles.maps.eox.at/wmts/1.0.0/WMTSCapabilities.xml`); commercial use needs EOX licence | "EOxCloudless ... by EOX IT Services GmbH (Contains modified Copernicus Sentinel data YYYY)" with licence link | Leaflet attribution (`map.js`) + fine print in `index.html` (states CC BY-NC-SA 4.0, non-commercial); off by default |
| STAC extension schemas (classification v2.0.0, projection v1.1.0, raster v1.1.0) | `export.py` | - | 206 each | schema identifiers only; licence not checked (not downloaded at run time) | none | n/a |

## Vendored front-end assets

| Asset | Path | Licence | Integrity | Notice shipped |
|---|---|---|---|---|
| Leaflet 1.9.4 (`leaflet.js`, `leaflet.css`, 3 PNG icons) | `src/geocrop/web_static/vendor/leaflet/` | BSD-2-Clause | all 5 files and `LICENSE` byte-identical to `unpkg.com/leaflet@1.9.4/dist/` (SHA-256 compared 2026-10-26) | yes, `vendor/leaflet/LICENSE` is included in `package-data` |
| `website/`, QGIS plugin zip | `website/assets/` | project's own (Apache-2.0, `LICENSE.txt`/`NOTICE.txt`) | no third-party JS/CSS or CDN references found | yes |

No other third-party JS, CSS or fonts are vendored or loaded from CDNs.

## Python dependencies

Clean venv (Python 3.12, `pip install -c constraints/py312.txt -e .[<every declared extra>]` as in the release
gate, incl. torch, catboost, lightgbm, omnicloudmask, segmentation-models-pytorch), `pip-licenses 5.5.5`:
134 distributions. Declared licence strings: "Apache Software License" 34, MIT 31, BSD-3-Clause 21,
"MIT License" 13, Apache-2.0 10, "BSD License" 9, BSD-2-Clause 3, BSD 2, "Python Software Foundation License" 2,
PSF-2.0 1, MIT-CMU 1, "Apache License, Version 2.0" 1, "Apache Software License; BSD License" 1,
"Apache-2.0 OR BSD-2-Clause" 1, MPL-2.0 1, and three multi-licence expressions (numpy, torch, tqdm).

Re-run after the `cube`, `ftw`, `openeo` and `plots` extras were added (same date, clean CPython 3.12.15 venv with
every extra from the regenerated `constraints/py312.txt`): 156 distributions; the only new flags are
`odc-stac` (no `License` metadata field; the bundled `LICENSE` file is Apache-2.0, decision below) and the MIT
text of `planetary-computer` being printed in full.

**No distribution declares GPL, LGPL, AGPL or an unknown licence.** Entries worth noting:

| Package | Declared licence | Pulled in by | Assessment |
|---|---|---|---|
| certifi | MPL-2.0 | requests (core) | weak file-level copyleft; used unmodified as an installed dependency, not vendored: compatible with Apache-2.0 distribution |
| tqdm | MPL-2.0 AND MIT | omnicloudmask / huggingface-hub (`[cloud]`) | same as certifi |
| shapely (wheel) | BSD-3-Clause, but bundles **GEOS (LGPL-2.1**, `shapely/.dylibs/libgeos*`) | core | not visible to pip-licenses. Dynamically loaded shared library inside an unmodified third-party wheel; GeoCrop does not vendor or link it statically: acceptable. Relevant only if a binary bundle (Docker image) is distributed: then ship the LGPL notice/source offer for GEOS |
| scipy (wheel) | BSD-3-Clause, bundles libgfortran/libgcc_s/libquadmath | core | GPL-3 with GCC Runtime Library Exception; exception covers this use: acceptable |
| rasterio / pyogrio / pyproj (wheels) | BSD-3 / MIT / MIT, bundle GDAL (MIT), PROJ (MIT), libcurl, HDF5, netCDF etc. | core | permissive; no action |
| torch | Apache-2.0 AND BSD/MIT/BSL-1.0 (expression) | `[deep]` | permissive |
| numpy | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 | core | permissive |

Documented decisions read by the release gate (`d.licences_and_weights` fails on a copyleft licence without one):

<!-- licence-decision: certifi: MPL-2.0, unmodified installed dependency of requests, not vendored -->
<!-- licence-decision: tqdm: MPL-2.0 AND MIT, unmodified installed dependency (cloud extra), not vendored -->
<!-- licence-decision: odc-stac: Apache-2.0 (licence file in the wheel, no License metadata field), cube extra, not vendored -->

Raw output: `pip_licenses.json` (path in [How to reproduce](#how-to-reproduce)).

## Findings and gaps

Status after the gate-findings fixes (2026-10-26):

1. **No checksum mismatches and no unreachable pinned URLs** (20 pinned files: 9 FTW, 1 Presto, 2 Galileo,
   1 AnySat, 2 OmniCloudMask, 4 WorldCereal, 1 EuroCrops). The KMRS 2025 file of the transfer protocol is pinned
   by SHA-256 too, under the protocol's `sources:` key; it was verified by the transfer run itself.
   `tools/check_weights.py` now also reads every transfer protocol's `sources:` key, so the next gate run
   re-checks it (not yet run with downloads after this change).
2. Fixed: **Copernicus attribution** is now written where Sentinel data enter (Earth Search, CDSE SAFE, SNAP,
   Planetary Computer, openEO) and carried through composites, stacks and predictions into COG exports and STAC
   items (see the table). Synthetic demo data carry no such tag. Not covered: external cubes loaded through
   `geocrop cube` (the source is user-defined; add the attribution yourself when they hold Sentinel data).
3. Fixed: **OmniCloudMask weights are pinned** by revision + SHA-256 and recorded in the cloud mask.
4. Documented: **AlphaEarth index and TESSERA tiles are unpinned by design** (mutable index; partial range
   reads), see the table.
5. Fixed: **`NOTICE`** lists Leaflet (BSD-2), the MIT code adapted from Presto, Galileo and AnySat, the
   CC-BY-SA HCAT tables, the EOX non-commercial basemap and the run-time data terms.
6. Open (minor): FTW licence and citation are recorded in `ftw_summary.json`, not in the predicted GeoTIFF tags.
7. EOX s2cloudless basemaps are CC-BY-NC-SA-4.0; the UI says so and they are off by default, but commercial
   deployments must disable them or license them from EOX (also stated in `NOTICE`).
8. Fixed: FTW's `segmentation-models-pytorch` is declared by the new `ftw` extra (`pip install -e '.[ftw]'`);
   `constraints/py312.txt` now pins every extra.

## How to reproduce

```bash
# from the repository root, any venv with the engine's core deps + requests
PYTHONPATH=src python tools/check_weights.py --head-only               # resolution only, no downloads
PYTHONPATH=src python tools/check_weights.py --cache /tmp/geocrop-weights > check_weights.json   # ~1.7 GB
PYTHONPATH=src python tools/check_weights.py --cache /tmp/geocrop-weights --only presto          # one item

# service endpoints (status code of a cheap request)
curl -s -o /dev/null -L -r 0-1023 -w '%{http_code}\n' https://planetarycomputer.microsoft.com/api/stac/v1
curl -s -o /dev/null -L -r 0-1023 -w '%{http_code}\n' https://earth-search.aws.element84.com/v1
curl -s -o /dev/null -L -r 0-1023 -w '%{http_code}\n' https://stac.dataspace.copernicus.eu/v1
curl -s -o /dev/null -L -w '%{http_code}\n' https://tiles.maps.eox.at/wmts/1.0.0/s2cloudless-2024_3857/default/g/0/0/0.jpg

# licence facts
curl -s https://tiles.maps.eox.at/wmts/1.0.0/WMTSCapabilities.xml | grep -o 'Contains modified Copernicus Sentinel data 2024[^.]*'
curl -s https://zenodo.org/api/records/14094196 | python -c "import json,sys; print(json.load(sys.stdin)['metadata']['license'])"
curl -s https://data.source.coop/tge-labs/aef/README.md | grep -A2 '## License'

# Python dependency licences
# (the audited venv is the release gate's: every extra declared in pyproject.toml)
python3.12 -m venv venv && venv/bin/pip install -c constraints/py312.txt \
  -e '.[all,boost,cloud,cube,deep,dev,ftw,openeo,parquet,plots,segmentation,stac,web]' pip-licenses
venv/bin/pip-licenses --format=json --with-urls > pip_licenses.json
```

`check_weights.py` exits 1 on any mismatch or unreachable URL and prints
`[{name, url, status, sha256_expected, sha256_actual, match, bytes, ...}]`; EuroCrops is pinned by md5 in the
benchmark protocol, so its entry reports `checksum_expected`/`checksum_actual` with `sha256_expected: null`.
