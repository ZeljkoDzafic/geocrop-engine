# Reproducibility and data protection

## What is recorded
* **Stage cache** (`<output>/stages.json`): schema-versioned running/success/failed states, configuration/input
  fingerprints, runtime, summary and complete output identities. State is flushed and atomically replaced;
  a failed/interrupted rerun cannot reuse its earlier success. Only successful stages with matching complete
  outputs are skipped. `--force` reruns. Legacy cache records recompute once under the new schema.
  Scene-directory dependencies are inspected recursively. Large binary products use size/mtime identity;
  small JSON/YAML/CSV/TXT/Python metadata up to 1 MiB also use SHA-256. This detects ordinary changes and
  missing/mutated companions; it is not a full content-integrity proof for large files whose size and mtime
  have both been restored. Engine source changes, seeds, class legends, mappings and feature manifests are
  included in relevant cache dependencies. Simultaneous writers to the same run directory are unsupported;
  use a separate output directory for each concurrent run.
* **Run manifest** (`run_manifest.json`, also `geocrop manifest`): UTC time, Python and platform, versions of
  all scientific packages and GDAL, git commit and dirty flag, `conda list --explicit` when conda is present,
  SNAP version (`--snap-gpt`), resolved configuration and its fingerprint, SHA-256 of input files.
* **Products**: `scene.json` per scene (source product, radiometric constants), `composite.json` per window
  (scenes, statistic, masks), feature manifest, model payload (features, classes, parameters, seed,
  scikit-learn version), download status with checksums.
* **Calibration identity**: SHA-256 of the actual saved model in training/conformal reports and prediction
  manifests/raster tags. Bound thresholds cannot be applied to predictions tagged with another model.
* **Benchmark identity**: actual sample-array/feature/CRS hashes plus training/test index hashes per fold;
  candidate metrics, parameter settings, explicit failures, runtimes and library versions are retained.
* **Determinism**: all random operations take the configured seed. Random Forest results are reproducible for a
  fixed seed, data and library version; changing `n_jobs` does not change results.

## Checklist for a publication
1. Archive the configuration file, `run_manifest.json`, scene inventories and `download_status.json`.
2. Tag the engine version used (`git tag`), cite it via `CITATION.cff` (consider a Zenodo DOI).
3. Report window dates, compositing statistic and masks, S1 backscatter convention and orbit handling, final
   feature list, RF parameters, block size and folds, parcel strategy and thresholds — all are in the outputs.
4. Publish aggregated metrics and confusion matrices; do not publish restricted parcel or field data.

When the input sensor/orbit layout changes, use a fresh composite output directory or deliberately clean
obsolete generated products after reviewing them. The pipeline rejects known stale sensor/orbit files
inside active windows, preserving unknown user files instead of mixing old features into the new run.

## Data protection
* Never commit LPIS/IACS data, farmer identities, field photographs, credentials or signed URLs; `.gitignore`
  excludes data, rasters, vectors and models by default.
* `geocrop normalize-parcels --pseudonymise` replaces parcel IDs by codes and writes the lookup separately; keep
  the lookup private.
* Credentials are read from environment variables only (`CDSE_USERNAME`, `CDSE_PASSWORD`, `CDSE_ACCESS_TOKEN`).
* Run manifests contain file hashes and paths but no data content.
