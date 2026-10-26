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
  fixed seed, data and library version; changing `n_jobs` does not change results. `tests/test_reliability.py`
  checks that two runs from fresh directories, with the same inputs and configuration, give byte-identical stack,
  samples, model and prediction rasters and an identical `train_report.json`. Caveats:
  - `model.joblib`, and with it `model_sha256` in reports, manifests and raster tags, embeds the configuration
    fingerprint, and that fingerprint includes the configured *paths*. The same study configured with other
    paths gives the same predictions and metrics but a different model hash. Use relative paths, or compare
    pixels and metrics rather than hashes.
  - Results are only bitwise reproducible with the same versions of NumPy, scikit-learn, GDAL/rasterio and
    the same platform. Other BLAS/CPU builds or library upgrades can change the last bits of float results
    and, rarely, an RF tie. The run manifest records the versions.
  - Estimators outside scikit-learn (LightGBM, CatBoost, TempCNN/PyTorch) can be nondeterministic with
    several threads or on a GPU. Set `n_jobs: 1` and deterministic framework flags when bitwise
    reproducibility matters.
  - `run_manifest.json`, `run.log` and the timing fields in `stages.json` contain times, so they always
    differ between runs.
* **Interruption**: a stage killed at any point (SIGKILL, power loss) stays `running` in `stages.json`. The next
  run takes over the stale lock and recomputes that stage and its dependants. Until then, downstream stages
  refuse to start rather than read its possibly partial products. A lock whose pid is alive (also after pid
  reuse) is respected: delete `<output>/.geocrop.lock` if no geocrop run is actually active.

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

## Content-addressed resumption (0.9)
* Stage fingerprints use SHA-256 **content** digests of inputs (large files cached by path/size/mtime in
  `<output>/.hashcache.json`; small files such as sidecars and configs are always re-hashed), the engine source
  code and the fingerprints of upstream stages (`composite → features → samples → train → predict → parcels`).
  Optional stages and options (`embeddings`, `parcel_classify`, `cv.strategy`, `predict.smoothing`,
  `cbm.event_markers`, `cbm.rules`, `cbm.temporal_support`, `timeseries.temporal_support`, `model.params`) enter a fingerprint only while they are enabled, so a
  configuration that uses none of them keeps its fingerprints; the engine-code part still changes with
  every GeoCrop version, so an upgrade recomputes the stages once.
  Touching or copying a file does not trigger reprocessing; any content, code or upstream change does.
* `cache.hash: metadata` switches to size + mtime for raw archives too large to hash.
* A lock file prevents two runs from writing the same output directory; `run.log` records every stage with
  duration and peak resident memory.
