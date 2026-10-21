# Scope, status and limitations

## Verified
* Unit and integration tests (≈ 60) for every module, including SAFE radiometric import on a synthetic SAFE,
  shadow-projection geometry, multi-temporal cloud screening, Whittaker/phenology/harmonics, conformal coverage,
  AOA extrapolation flags, variogram range recovery, a full scene-to-traffic-light pipeline,
  same-day mosaicking, mask dilation, linear S1 compositing, feature manifest, leakage-free spatial folds,
  Olofsson estimators against hand-computed cases, parcel decisions, leakage exclusion in validation, the
  resumable pipeline and U-Net train/infer.
* End-to-end execution on synthetic data (`geocrop demo`).

## Implemented following specifications, to be verified per study on real data
* CDSE OData/STAC queries, token refresh and download against the live service and your account quotas.
* SNAP graphs with your SNAP version (`gpt <graph> -h`) and visual QA of σ⁰/γ⁰ outputs.
* SAFE import on real products of different processing baselines (offset handling).
* Cloud/shadow masking quality (SCL has known omission errors for thin clouds and shadows; inspect composites).

## Methodological limits users must report
* Internal CV on interpreted polygons is optimistic; external accuracy requires independent field data.
* If training polygons were located using declarations, comparison with declarations is not fully independent;
  state this, and evaluate on independently verified parcels.
* 20 m bands resampled to 10 m add no spatial detail; parcels with few interior pixels are flagged rather than decided.
* Temperature scaling and conformal sets are fitted on out-of-fold predictions of interpreted polygons; verify
  calibration and coverage on independently verified parcels before relying on them operationally.
* Markers and crop calendars encode regional agronomic assumptions; harvest/rapid-senescence detection from NDVI
  alone cannot distinguish harvest from drought-induced senescence.
* Segmentation outputs are candidate objects, not cadastral parcels; OBIA objects are not merged across tiles.
* Spatial transfer to other regions or years requires new reference data (crop calendars and management differ).

## Not in scope
Legal compliance decisions, farmer-level reporting, real-time processing, or redistribution of satellite or
parcel data.
