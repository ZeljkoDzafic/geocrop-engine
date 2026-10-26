# 0003. File contracts with schema versions

* Status: Accepted
* Date: 2026-10-26

## Context
Stages communicate only through files (GeoTIFF, GeoPackage, CSV, JSON; see [data-contracts.md](../data-contracts.md)).
This is what lets any stage run elsewhere or be replaced, and it makes long-lived artefacts (scene folders,
trained models) outlive code versions. Without a version marker, a reader cannot tell an old file from a
malformed one, and a future layout change would be misread silently.

## Decision
* JSON contracts carry an integer `schema_version`, managed in
  [`geocrop.contracts`](../../src/geocrop/contracts.py) (`SCHEMA_VERSIONS`, `stamp`, `check`, `MIGRATIONS`).
* Versioned now: `scene.json` (stamped by `scenes.write`, used by `s2-import`, `s1-import`, `stac-import`,
  validated by `scenes.read`), the `model.joblib` payload (stamped by `train`, validated by `predict`) and
  `run_manifest.json` (stamped by `run_manifest`, read with `provenance.read_run_manifest`). The existing
  `"schema": "geocrop-run/1"` string stays for compatibility.
* Reading rules: no `schema_version` = version 0, the layout before versioning (accepted unchanged); older
  versions are upgraded step by step through `MIGRATIONS`; newer versions raise `SchemaVersionError` naming
  the file and asking to upgrade GeoCrop or regenerate the file; non-integer values are rejected.
* A layout change bumps the version and adds a migration `(contract, old) -> old + 1` plus a test.

## Consequences
* All existing files keep working; files written by this version are readable by older GeoCrop releases
  (they ignore the extra key).
* The feature manifest (`<stack>.features.json`) is stamped by `features.write_manifest` (used for window,
  time-series, merged and embedding stacks) and validated by `features.read_manifest` / `load_manifest`
  (added after this ADR; unversioned manifests still load). Reports (`train_report.json`, `benchmark.json`)
  keep their `schema` strings.
* Version numbers are per contract, not per GeoCrop release.
