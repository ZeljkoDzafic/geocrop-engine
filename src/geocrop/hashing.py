"""File checksums shared by core (model binding) and application (provenance) code."""
from __future__ import annotations

import hashlib
from pathlib import Path


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()
