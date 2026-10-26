# Versioning, releases and rollback

## Versioning policy

GeoCrop Engine follows [Semantic Versioning 2.0](https://semver.org/) from 1.0.0 on: `MAJOR.MINOR.PATCH`.
Before 1.0.0, minor releases may still contain breaking changes; they are listed under "Changed" in the
CHANGELOG.

The **public contract** consists of:

| Surface | Breaking change (MAJOR) | Compatible (MINOR/PATCH) |
|---|---|---|
| CLI | removing or renaming a command or flag, changing a default that alters results, changing exit-code meaning | new commands, new optional flags, better messages |
| Configuration | removing/renaming keys, changing a key's meaning or units, rejecting configs that were valid | new optional keys with defaults that reproduce previous behaviour |
| File contracts ([data contracts](data-contracts.md)) | removing/renaming output files, columns, raster bands or JSON fields; changing a `schema` identifier's major (`geocrop-run/1` → `/2`), CRS/nodata/dtype conventions | new optional files, columns or fields |
| REST API `/api/v1` ([web interface](web-interface.md)) | removing an endpoint, field or parameter; changing status codes or types | new endpoints, optional parameters and response fields |
| Python API | the documented entry points in `geocrop.client` and the module functions used in the docs | internal helpers (underscore names, unlisted modules) may change in any release |

Unversioned `/api/...` paths are compatibility aliases of `/api/v1/...` and will be kept for the whole 1.x
series. A future `/api/v2` would be added alongside `/api/v1`, which then stays for at least one minor release.
Saved models (`joblib`) are bound to the scikit-learn version they were trained with; a release that requires a
newer scikit-learn says so in the CHANGELOG, and models should be retrained (or the old environment kept) for
exact reproduction.

PATCH releases contain only bug and security fixes. Security fixes may tighten the web API's input validation
(for example, rejecting a path that was wrongly accepted) without a MAJOR bump.

## Support matrix

| Item | Supported |
|---|---|
| Python | 3.10, 3.11, 3.12 (tested in CI); pinned constraints for 3.12 |
| OS | Linux x86_64 (CI, Docker, VM runbook); macOS arm64 (development); Windows: CLI best effort, web job cancellation uses process termination instead of process groups |
| Releases | the latest MINOR line receives bug and security fixes (see [SECURITY.md](../SECURITY.md)) |
| ESA SNAP | external, not bundled; Sentinel-1 preprocessing only (see the [deployment runbook](../deploy/README.md)) |
| Browsers (web workspace) | current Firefox, Chrome/Edge and Safari |

## Release checklist

1. Start from an up-to-date `main` with a green `tests` workflow (matrix, pinned, audit, docker) and a passing
   [release gate](release-gate.md) report for the release commit (`tools/release_gate.py`).
2. Run `pip-audit -r constraints/py312.txt`; resolve findings or document them in [security](security.md).
3. Refresh the pinned set if dependencies changed (see *Reproducible installs*), and run the full suite against it.
4. Decide the version from the CHANGELOG entries according to the policy above.
5. Update the version in `pyproject.toml`, `CITATION.cff` (`version` and `date-released`) and the fallback in
   `src/geocrop/__init__.py`; rename `## Unreleased` in `CHANGELOG.md` to `## X.Y.Z — YYYY-MM-DD` and add a new
   empty `## Unreleased` above it.
6. `python tools/check_versions.py` must report "versions consistent".
7. Regenerate CLI docs if commands changed (`python tools/generate_cli_docs.py`).
8. Merge the release pull request, then tag the merge commit: `git tag -a vX.Y.Z -m "GeoCrop Engine X.Y.Z" && git push origin vX.Y.Z`.
9. The `release` workflow checks the tag against all version sources, runs the tests, builds sdist + wheel,
   smoke-tests the wheel, and creates the GitHub Release with that version's CHANGELOG section and the
   distributions plus `SHA256SUMS` attached. Tags such as `v1.1.0rc1` become pre-releases.
10. Check the release page; announce; update the Zenodo/DOI record if one is configured.

**PyPI** is not published automatically. To enable it, configure a
[trusted publisher](https://docs.pypi.org/trusted-publishers/) for this repository and workflow `release.yml`
on pypi.org, create a GitHub environment `pypi` (optionally with required reviewers) and uncomment the `pypi`
job at the end of `.github/workflows/release.yml`. No API token is stored in the repository.

## Reproducible installs

`constraints/py312.txt` pins every runtime dependency of **every** optional extra (`all`, `boost`, `cloud`,
`cube`, `deep`, `dev`, `ftw`, `openeo`, `parquet`, `plots`, `segmentation`, `stac`, `web`) to the set that
passed the test suite and `pip-audit`; any subset of extras can be installed against it:

```bash
python3.12 -m venv .venv && . .venv/bin/activate
python -m pip install -e '.[all,stac,web]' -c constraints/py312.txt   # add dev for tests, deep/ftw for torch ...
```

Regenerate it in a clean environment (never from a development venv). Installing with the previous file as
constraints keeps the existing pins and only resolves packages that are new; drop `-c` to move every pin:

```bash
EXTRAS=all,boost,cloud,cube,deep,dev,ftw,openeo,parquet,plots,segmentation,stac,web
python3.12 -m venv /tmp/geocrop-pins && /tmp/geocrop-pins/bin/python -m pip install --upgrade pip
/tmp/geocrop-pins/bin/python -m pip install -c constraints/py312.txt -e ".[$EXTRAS]"
/tmp/geocrop-pins/bin/python -m pip freeze --exclude-editable   # replace the pins below the header
/tmp/geocrop-pins/bin/pip-audit -r constraints/py312.txt --no-deps --disable-pip
```

The Docker image installs the engine with CPython 3.12 from this file (`docker build --build-arg
EXTRAS=all,stac,web,deep .` selects the extras), so a rebuild reproduces the same Python packages; the base
image (`python:3.12-slim-bookworm`) and its Debian packages are not pinned by digest. The conda
`environment.yml` resolves current conda-forge packages; for an exact rebuild use an image built from a tag
(or its digest) rather than rebuilding later.

## Rollback

* **pip:** `python -m pip install 'geocrop-engine==X.Y.Z' -c constraints/py312.txt` from the previous release's
  wheel (attached to its GitHub Release), or `git checkout vX.Y.Z && python -m pip install -e '.[all,stac,web]'
  -c constraints/py312.txt` using the constraints file *of that tag*.
* **Docker:** run the image built from the previous tag (`docker build -t geocrop-engine:X.Y.Z .` at that tag, or
  the image digest you deployed). Data live in mounted volumes, so switching images does not touch them.
* **Data compatibility:** output files carry schema identifiers (`geocrop-run/1`, `geocrop-prediction/1`, …).
  Within a MAJOR version, an older release can read outputs of a newer one unless the CHANGELOG states
  otherwise; new optional fields are ignored. Saved models require the scikit-learn version recorded in their
  metadata; after a rollback that changes scikit-learn, retrain or keep the newer environment for prediction.
* **Stage cache:** pipeline stage fingerprints include a hash of the engine's source code and a cache schema
  number, so a different engine version (in either direction) never reuses cached stages: they are recomputed
  on the next `geocrop run`, which is safe but costs time. To keep results of both versions, use a separate
  output directory per version. Never edit `stages.json` by hand.
* **Web workspace:** the job database (`.geocrop-web/jobs.sqlite3`) has been schema-stable since 0.9.0; jobs that
  were running when the server stopped are marked `interrupted` after restart with either version.
* **Clients:** `EngineClient` uses `/api/v1` and falls back to the unversioned paths, so notebooks keep working
  against both newer and older servers.
