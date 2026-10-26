"""Download of pinned Fields of The World checkpoints (HTTPS, SHA-256 verified) into the local cache.

Checkpoints are never bundled or committed: they are fetched at runtime from the official FTW GitHub releases,
checked against the size and digest pinned in `geocrop.ftw.MODELS`, and moved into place atomically
(`geocrop.downloads.fetch`: retry with exponential backoff, HTTP Range resume, checksum verification).
Non-commercial ("FULL") checkpoints are refused unless ``allow_noncommercial=True``.
"""
from __future__ import annotations

import sys
from pathlib import Path

from .ftw import NONCOMMERCIAL_WARNING, check_licence, checkpoint_path, sha256_file


def download(name: str, cache_dir: str | Path | None = None, allow_noncommercial: bool = False,
             timeout: float = 60, progress=None, retries: int = 5) -> Path:
    """Verified local path of FTW checkpoint `name`, downloading it on first use."""
    model = check_licence(name, allow_noncommercial)
    if not model.commercial_use:
        print(NONCOMMERCIAL_WARNING.format(name=name), file=sys.stderr)
    target = checkpoint_path(name, cache_dir)
    if target.exists() and target.stat().st_size == model.size and sha256_file(target) == model.sha256:
        return target
    from .downloads import fetch
    return fetch(model.url, target, f"sha256:{model.sha256}", size=model.size, timeout=timeout,
                 progress=progress, retries=retries)
