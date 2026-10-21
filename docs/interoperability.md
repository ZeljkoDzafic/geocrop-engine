# Interoperability and related systems

## Publishing results
`geocrop export --prediction outputs/run/prediction --out publish/2023 --id rs-north-2023 --start 2023-03-15 --end 2023-10-31 --parcels outputs/run/parcels/parcels.gpkg --hcat configs/hcat_mapping.csv`

* **Cloud-Optimised GeoTIFF** for classes (nearest overviews), probabilities, uncertainty (average overviews) and AOA;
* **GeoParquet** parcel table (`pip install pyarrow`);
* a **STAC 1.0 Item** (`item.json`) with the projection and classification extensions; class values carry
  EuroCrops **HCAT** codes when a mapping is supplied (`configs/hcat_mapping.template.csv`; codes from
  <https://github.com/maja601/EuroCrops>).

Any STAC browser, QGIS, GDAL, openEO back-end or data catalogue can consume the output; platforms such as ETFarm
import the GeoPackage/GeoParquet parcel table and COG layers.

## Inputs from other systems
Data discovery via CDSE OData or any STAC API; Sentinel-1 via ESA SNAP; any external cloud mask through
`CLOUDMASK.tif`; parcels and declarations from any LPIS/IACS through `normalize-parcels` and crop mappings.

## How GeoCrop Engine relates to other open systems

| System | Focus | Relation |
|---|---|---|
| ESA WorldCereal (MIT) | global seasonal crop maps on openEO/CDSE, Presto + CatBoost | GeoCrop runs locally, adds parcel-level declaration verification; CatBoost available as `algorithm: catboost` |
| Sen2-Agri / Sen4CAP | national operational systems for paying agencies | shares signatures/markers/traffic-light concepts; GeoCrop is a light Python library with research-grade validation |
| sits (R, GPL-2) | complete SITS analysis in R | similar scope in Python under Apache-2.0; ideas (smoothing, uncertainty sampling) re-implemented, no code reused |
| eo-learn (MIT), JRC CbM (BSD-3) | EO processing framework, CbM notebooks | GeoCrop offers an end-to-end, config-driven pipeline with spatial CV, conformal sets and AOA |
| FORCE (GPL-3) | ARD and time-series analysis | complementary preprocessing alternative; GeoCrop reads any per-scene or composite rasters |
| Fields of The World (MIT) | field-boundary benchmark, pretrained models | can supply boundary probability to `boundaries-to-parcels` |

GPL- or share-alike-licensed tools are used only as ideas or external programs; nothing is vendored into this
Apache-2.0 code base.
