"""Versioned JSON file contracts (see docs/adr/0003-file-contracts-and-schema-versions.md).

Writers stamp ``"schema_version": <int>``; readers call `check`, which
* accepts files without the field (written before versioning; read as version 0 = the same layout as 1),
* upgrades older versions through `MIGRATIONS`,
* refuses files from a newer GeoCrop with an actionable message instead of misreading them.
"""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

KEY = "schema_version"

#: Current version written for each contract. Bump on an incompatible change and add a migration.
SCHEMA_VERSIONS: dict[str, int] = {
    "scene": 1,             # scene.json written by s2-import, s1-import, stac-import
    "run_manifest": 1,      # run_manifest.json (geocrop manifest)
    "model": 1,             # model.joblib payload metadata
    "feature_manifest": 1,  # <stack>.features.json (features.write_manifest / read_manifest)
    "temporal_support": 1,  # <stack>_temporal_support.json (per-pixel observed vs interpolated support)
    "evidence": 1,          # parcel evidence package <id>/evidence.json (geocrop evidence)
}

#: (contract, from_version) -> function upgrading a body to from_version + 1.
#: Version 0 (unversioned files) has the version-1 layout, so its migration only stamps the number.
MIGRATIONS: dict[tuple[str, int], Callable[[dict], dict]] = {
    (kind, 0): (lambda body: body) for kind in SCHEMA_VERSIONS
}


class SchemaVersionError(ValueError):
    """A file declares a schema version this GeoCrop cannot read."""


def stamp(kind: str, body: dict) -> dict:
    """`body` with the current schema version of `kind` (an existing value is overwritten)."""
    return {**body, KEY: SCHEMA_VERSIONS[kind]}


def check(kind: str, body: dict, source: str | Path = "") -> dict:
    """Validate and, when needed, migrate `body` to the current version of `kind`."""
    current = SCHEMA_VERSIONS[kind]
    where = f" in {source}" if source else ""
    version = body.get(KEY, 0)
    if isinstance(version, bool) or not isinstance(version, int) or version < 0:
        raise SchemaVersionError(f"Invalid {KEY} {version!r} for {kind}{where}; expected a non-negative integer")
    if version > current:
        raise SchemaVersionError(
            f"{kind}{where} has {KEY} {version}, but this GeoCrop {_version()} reads up to {current}. "
            "Upgrade geocrop-engine (pip install -U geocrop-engine) or regenerate the file with this version.")
    while version < current:
        migrate = MIGRATIONS.get((kind, version))
        if migrate is None:
            raise SchemaVersionError(f"{kind}{where} has {KEY} {version}; no migration to {version + 1} exists. "
                                     "Regenerate the file with the current GeoCrop.")
        body = migrate(dict(body))
        version += 1
    return {**body, KEY: current}


def _version() -> str:
    from . import __version__
    return __version__
