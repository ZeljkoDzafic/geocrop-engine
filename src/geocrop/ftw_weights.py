"""Download of pinned Fields of The World checkpoints (HTTPS, SHA-256 verified) into the local cache.

Checkpoints are never bundled or committed: they are fetched at runtime from the official FTW GitHub releases,
checked against the size and digest pinned in `geocrop.ftw.MODELS`, and moved into place atomically.
Non-commercial ("FULL") checkpoints are refused unless ``allow_noncommercial=True``.
"""
from __future__ import annotations

import hashlib
import os
import sys
import urllib.request
from pathlib import Path

from .ftw import NONCOMMERCIAL_WARNING, check_licence, checkpoint_path, sha256_file


def download(name: str, cache_dir: str | Path | None = None, allow_noncommercial: bool = False,
             timeout: float = 60, progress=None) -> Path:
    """Verified local path of FTW checkpoint `name`, downloading it on first use."""
    model = check_licence(name, allow_noncommercial)
    if not model.commercial_use:
        print(NONCOMMERCIAL_WARNING.format(name=name), file=sys.stderr)
    target = checkpoint_path(name, cache_dir)
    if target.exists() and target.stat().st_size == model.size and sha256_file(target) == model.sha256:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + f".{os.getpid()}.part")
    digest, done = hashlib.sha256(), 0
    try:
        request = urllib.request.Request(model.url, headers={"User-Agent": "geocrop-engine"})
        with urllib.request.urlopen(request, timeout=timeout) as resp, open(part, "wb") as fh:  # noqa: S310 - pinned https URL
            while block := resp.read(1 << 20):
                fh.write(block)
                digest.update(block)
                done += len(block)
                if progress:
                    progress(done, model.size)
        if done != model.size or digest.hexdigest() != model.sha256:
            raise ValueError(f"Downloaded FTW checkpoint {name} failed verification "
                             f"(size {done} vs {model.size}, sha256 {digest.hexdigest()} vs {model.sha256})")
        os.replace(part, target)
    finally:
        part.unlink(missing_ok=True)
    return target
