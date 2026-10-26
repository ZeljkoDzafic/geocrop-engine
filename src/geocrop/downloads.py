"""Robust run-time downloads: retry with exponential backoff, resume (HTTP Range) and checksum verification.

Every file the engine downloads at run time (FTW checkpoints, Presto/Galileo/AnySat and OmniCloudMask
weights, the AlphaEarth index, EuroCrops archives, WorldCereal calendars) goes through `fetch`:

* the transfer is written to ``<target>.part`` and moved into place atomically only after verification;
* an interrupted transfer is resumed from the bytes already in ``.part`` with a ``Range`` request (a server
  that ignores the range answers 200 and the file is restarted; a resumed file that then fails the checksum is
  discarded and downloaded once more from the start);
* connection errors, timeouts, truncated bodies, HTTP 5xx and 429 are retried with exponential backoff
  (``backoff * 2**attempt`` seconds, capped at ``max_backoff``); ``Retry-After`` is honoured;
* data repositories that stage files from tape (PANGAEA answers 503 + ``Retry-After``, or an HTML placeholder)
  are polled until `max_wait_s` without consuming the retry budget;
* `checksum` is ``"sha256:<hex>"``, ``"md5:<hex>"`` or a bare SHA-256; `size` is checked as well. A cached file
  that matches is reused without any request. Without a checksum the file is cached as is (only for mutable
  sources that publish none, e.g. the AlphaEarth index; documented in docs/licences-and-weights.md).
"""
from __future__ import annotations

import hashlib
import os
import random
import time
from collections.abc import Callable
from pathlib import Path

import requests

USER_AGENT = "geocrop-engine (https://github.com/ZeljkoDzafic/geocrop-engine)"
RETRY_STATUS = frozenset({429, 500, 502, 503, 504})
#: replaced in tests to avoid real waiting
_sleep = time.sleep


class DownloadError(RuntimeError):
    """A download failed after all retries (network) or could not be verified (checksum/size)."""


class ChecksumError(DownloadError, ValueError):
    """The downloaded file does not match its pinned checksum or size."""


def parse_checksum(checksum: str | None) -> tuple[str, str] | None:
    """``(algorithm, hex)`` from ``"sha256:<hex>"``/``"md5:<hex>"``/bare 64-char SHA-256, None when unpinned."""
    if not checksum:
        return None
    algorithm, sep, expected = str(checksum).partition(":")
    if not sep:
        algorithm, expected = "sha256", algorithm
    algorithm = algorithm.lower()
    if algorithm not in ("md5", "sha256") or not expected:
        raise ValueError(f"Checksum must be 'md5:<hex>' or 'sha256:<hex>', got {checksum!r}")
    return algorithm, expected.lower()


def digest(path: str | Path, algorithm: str = "sha256") -> str:
    h = hashlib.new(algorithm)
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(path: str | Path, checksum: str | None = None, size: int | None = None) -> bool:
    """True when `path` is a file matching `size` and `checksum` (each only when given)."""
    path = Path(path)
    if not path.is_file():
        return False
    if size is not None and path.stat().st_size != size:
        return False
    pin = parse_checksum(checksum)
    return pin is None or digest(path, pin[0]) == pin[1]


def _retry_after(response) -> float | None:
    value = response.headers.get("retry-after")
    try:
        return max(float(value), 0.0) if value is not None else None
    except ValueError:            # HTTP-date form: fall back to the backoff schedule
        return None


def _staged(response, url: str) -> bool:
    """Tape-staging placeholder: 503 with Retry-After, or HTML where a data file was requested."""
    if response.status_code == 503 and response.headers.get("retry-after"):
        return True
    return (response.status_code < 300 and "text/html" in response.headers.get("content-type", "")
            and not url.lower().split("?", 1)[0].endswith((".html", ".htm")))


def fetch(url: str, target: str | Path, checksum: str | None = None, *, size: int | None = None,
          retries: int = 5, backoff: float = 2.0, max_backoff: float = 60.0, timeout: float = 60,
          max_wait_s: float = 900, session: requests.Session | None = None,
          progress: Callable[[int, int | None], None] | None = None, headers: dict | None = None,
          chunk_size: int = 1 << 16) -> Path:
    """Verified local copy of `url` at `target` (see the module docstring); returns `target`.

    Raises `ChecksumError` when the complete file does not match `checksum`/`size` (twice, the second time
    downloaded from scratch) and `DownloadError` when the transfer still fails after `retries` retries.
    """
    target = Path(target)
    pin = parse_checksum(checksum)
    if target.is_file() and verify(target, checksum, size):
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + ".part")
    http = session or requests.Session()
    base_headers = {"User-Agent": USER_AGENT, **(headers or {})}
    attempt, fresh_restarts = 0, 0
    deadline = time.monotonic() + max_wait_s
    last_error: str | None = None
    while True:
        offset = part.stat().st_size if part.is_file() else 0
        if size is not None and offset > size:          # stale or foreign partial file
            part.unlink()
            offset = 0
        req_headers = dict(base_headers)
        if offset:
            req_headers["Range"] = f"bytes={offset}-"
        wait = None
        try:
            with http.get(url, stream=True, timeout=timeout, headers=req_headers, allow_redirects=True) as r:
                if _staged(r, url):
                    wait = _retry_after(r) or 30.0
                    if time.monotonic() + wait > deadline:
                        raise DownloadError(f"{url} not available after {max_wait_s:.0f} s (HTTP {r.status_code}); "
                                            "the provider may still be staging it, retry later")
                    _sleep(max(wait, 1.0))
                    continue
                if r.status_code == 416 and offset:     # range beyond the end: the partial file may be complete
                    if verify(part, checksum, size) and (pin or size is not None):
                        break
                    part.unlink(missing_ok=True)
                    continue
                if r.status_code in RETRY_STATUS:
                    wait = _retry_after(r)
                    wait = None if wait is None else min(wait, max_wait_s)
                    raise requests.HTTPError(f"HTTP {r.status_code}", response=r)
                r.raise_for_status()
                total = None
                if r.status_code == 206 and offset:
                    content_range = r.headers.get("content-range", "")
                    if not content_range.startswith(f"bytes {offset}-"):
                        part.unlink(missing_ok=True)    # unexpected range: restart cleanly
                        continue
                    mode, done = "ab", offset
                    if "/" in content_range and content_range.rsplit("/", 1)[1].isdigit():
                        total = int(content_range.rsplit("/", 1)[1])
                else:                                   # 200: full body (server ignored or no range)
                    mode, done = "wb", 0
                    length = r.headers.get("content-length")
                    total = int(length) if length and length.isdigit() else None
                total = total or size
                with open(part, mode) as fh:
                    for block in r.iter_content(chunk_size):
                        fh.write(block)
                        done += len(block)
                        if progress:
                            progress(done, total)
                if total is not None and done < total:
                    raise requests.exceptions.ChunkedEncodingError(f"truncated transfer: {done} of {total} bytes")
            # complete body received
            if verify(part, checksum, size):
                break
            actual = digest(part, pin[0]) if pin else None
            part.unlink(missing_ok=True)
            if offset and fresh_restarts == 0:          # a resumed file may be corrupt: one clean retry
                fresh_restarts += 1
                continue
            raise ChecksumError(f"Checksum mismatch: downloaded {url} failed verification (size {done} vs {size}, "
                                f"{pin[0] if pin else 'checksum'} {actual} vs {pin[1] if pin else None}); "
                                "refusing to use it")
        except (requests.ConnectionError, requests.Timeout, requests.exceptions.ChunkedEncodingError,
                requests.HTTPError) as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if isinstance(exc, requests.HTTPError) and status is not None and status not in RETRY_STATUS:
                raise DownloadError(f"Download of {url} failed: HTTP {status}") from exc
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt >= retries:
                raise DownloadError(f"Download of {url} failed after {attempt + 1} attempts ({last_error}); "
                                    f"partial data kept in {part} and resumed on the next call") from exc
            delay = wait if wait is not None else min(backoff * 2 ** attempt, max_backoff) * (0.8 + 0.4 * random.random())
            attempt += 1
            _sleep(delay)
    os.replace(part, target)
    return target
