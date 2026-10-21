# Reproducibility and data protection

## What is recorded
* **Stage cache** (`<output>/stages.json`): per stage, a fingerprint of its configuration section and input files
  (size + modification time), runtime and summary. Unchanged stages are skipped; `--force` reruns.
* **Run manifest** (`run_manifest.json`, also `geocrop manifest`): UTC time, Python and platform, versions of
  all scientific packages and GDAL, git commit and dirty flag, `conda list --explicit` when conda is present,
  SNAP version (`--snap-gpt`), resolved configuration and its fingerprint, SHA-256 of input files.
* **Products**: `scene.json` per scene (source product, radiometric constants), `composite.json` per window
  (scenes, statistic, masks), feature manifest, model payload (features, classes, parameters, seed,
  scikit-learn version), download status with checksums.
* **Determinism**: all random operations take the configured seed. Random Forest results are reproducible for a
  fixed seed, data and library version; changing `n_jobs` does not change results.

## Checklist for a publication
1. Archive the configuration file, `run_manifest.json`, scene inventories and `download_status.json`.
2. Tag the engine version used (`git tag`), cite it via `CITATION.cff` (consider a Zenodo DOI).
3. Report window dates, compositing statistic and masks, S1 backscatter convention and orbit handling, final
   feature list, RF parameters, block size and folds, parcel strategy and thresholds — all are in the outputs.
4. Publish aggregated metrics and confusion matrices; do not publish restricted parcel or field data.

## Data protection
* Never commit LPIS/IACS data, farmer identities, field photographs, credentials or signed URLs; `.gitignore`
  excludes data, rasters, vectors and models by default.
* `geocrop normalize-parcels --pseudonymise` replaces parcel IDs by codes and writes the lookup separately; keep
  the lookup private.
* Credentials are read from environment variables only (`CDSE_USERNAME`, `CDSE_PASSWORD`, `CDSE_ACCESS_TOKEN`).
* Run manifests contain file hashes and paths but no data content.
