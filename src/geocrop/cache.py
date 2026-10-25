"""Content-addressed fingerprints for resumable runs.

A stage is re-run when the *content* of any input changes, not when a file is merely touched or copied.
SHA-256 digests of large files are cached per (path, size, mtime) in `<output>/.hashcache.json`, so large
rasters are hashed once; small files are always re-hashed. Directories are fingerprinted from the digests
of all files they contain. `mode="metadata"` (size + mtime only) is available for very large raw archives
where hashing is impractical.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

SMALL_FILE_BYTES = 4 * 1024 * 1024


class HashCache:
    def __init__(self, path: str | Path, mode: str = "content"):
        if mode not in ("content", "metadata"):
            raise ValueError("mode must be content or metadata")
        self.path, self.mode = Path(path), mode
        self.entries = json.loads(self.path.read_text()) if self.path.exists() else {}
        self.dirty = False

    def file(self, path: str | Path) -> str:
        p = Path(path).resolve()
        st = p.stat()
        if self.mode == "metadata":
            return f"meta:{st.st_size}:{st.st_mtime_ns}"
        key = str(p)
        hit = self.entries.get(key)
        # Small files (metadata, sidecars, configs) are always re-hashed: an edit that preserves size and
        # timestamp must still be detected. Only large rasters reuse the cached digest.
        if hit and st.st_size > SMALL_FILE_BYTES and hit["size"] == st.st_size and hit["mtime_ns"] == st.st_mtime_ns:
            return hit["sha256"]
        h = hashlib.sha256()
        with p.open("rb") as fh:
            for block in iter(lambda: fh.read(8 * 1024 * 1024), b""):
                h.update(block)
        digest = h.hexdigest()
        if st.st_size > SMALL_FILE_BYTES:
            self.entries[key] = {"size": st.st_size, "mtime_ns": st.st_mtime_ns, "sha256": digest}
            self.dirty = True
        return digest

    def tree(self, path: str | Path, suffixes=(".tif", ".tiff", ".json", ".gpkg", ".npz", ".joblib", ".csv",
                                               ".jp2", ".xml", ".yaml", ".yml")) -> str:
        root = Path(path)
        if root.is_file():
            return self.file(root)
        if not root.exists():
            return "missing"
        h = hashlib.sha256()
        for dirpath, _, files in sorted(os.walk(root)):
            for name in sorted(files):
                if name.endswith(suffixes) and not name.startswith("."):
                    f = Path(dirpath) / name
                    h.update(f"{f.relative_to(root)}:{self.file(f)}".encode())
        return h.hexdigest()

    def save(self):
        if self.dirty:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.entries))
            tmp.replace(self.path)
            self.dirty = False


def fingerprint(*parts) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(json.dumps(p, sort_keys=True, default=str).encode())
    return h.hexdigest()[:20]


class RunLock:
    """Prevent two concurrent runs from writing the same output directory."""

    def __init__(self, directory: str | Path):
        self.path = Path(directory) / ".geocrop.lock"

    def __enter__(self):
        if self.path.exists():
            pid = self.path.read_text().strip()
            try:
                os.kill(int(pid), 0)
                raise RuntimeError(f"Another geocrop run (pid {pid}) is using {self.path.parent}")
            except (ProcessLookupError, ValueError):
                pass                                   # stale lock from a crashed run
            except PermissionError as exc:
                raise RuntimeError(f"Output directory locked by pid {pid}") from exc
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(str(os.getpid()))
        return self

    def __exit__(self, *exc):
        self.path.unlink(missing_ok=True)
        return False
