#!/usr/bin/env python3
"""Verify every pinned external checkpoint / dataset file the engine declares (release gate, licence audit).

The pins are imported from the engine itself, never duplicated here:

* ``geocrop.ftw.MODELS``                        - Fields of The World checkpoints (URL, size, SHA-256)
* ``geocrop.embedding_sources.PRESTO_CHECKPOINT`` - Presto ``default_model.pt`` (URL, SHA-256)
* ``geocrop.public_data.WORLDCEREAL_CHECKSUMS``   - WorldCereal crop-calendar GeoTIFFs on PANGAEA (SHA-256)
* ``configs/benchmarks/*.yaml`` ``dataset``       - EuroCrops archives on Zenodo (checksum as pinned, e.g. md5)

Usage::

    PYTHONPATH=src python tools/check_weights.py --cache ~/.cache/geocrop-check > check_weights.json
    PYTHONPATH=src python tools/check_weights.py --head-only

Prints a JSON list ``[{name, url, status, sha256_expected, sha256_actual, match, bytes, ...}]``. Files are
downloaded into ``--cache`` (reused when already present and matching). Exit status 1 when any URL is
unreachable or any checksum differs; files larger than ``--max-bytes`` are reported as ``skipped``.
Needs ``requests`` and the engine importable (``PYTHONPATH=src`` from the repository root).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

USER_AGENT = "geocrop-engine release gate (https://github.com/ZeljkoDzafic/geocrop-engine)"


def pins() -> list[dict]:
    """Every pinned URL + checksum declared by the engine."""
    import yaml

    from geocrop import ftw, public_data
    from geocrop.embedding_sources import PRESTO_CHECKPOINT

    out = [{"name": f"ftw:{m.name}", "url": m.url, "checksum": f"sha256:{m.sha256}", "size": m.size,
            "licence": m.licence} for m in ftw.MODELS.values()]
    out.append({"name": "presto:default_model.pt", "url": PRESTO_CHECKPOINT["url"],
                "checksum": f"sha256:{PRESTO_CHECKPOINT['sha256']}", "size": None,
                "licence": PRESTO_CHECKPOINT.get("licence")})
    out += [{"name": f"worldcereal:{name}", "url": f"{public_data.PANGAEA_FILES}/{name}", "checksum": checksum,
             "size": None, "licence": "CC-BY-SA-4.0"}
            for name, checksum in public_data.WORLDCEREAL_CHECKSUMS.items()]
    for path in sorted((ROOT / "configs" / "benchmarks").glob("*.yaml")):
        ds = (yaml.safe_load(path.read_text(encoding="utf8")) or {}).get("dataset") or {}
        if ds.get("zenodo_record") and ds.get("archive"):
            out.append({"name": f"{ds.get('name', 'dataset')}:{ds['archive']} ({path.name})",
                        "url": public_data.zenodo_url(str(ds["zenodo_record"]), ds["archive"]),
                        "checksum": ds.get("checksum"), "size": None, "licence": ds.get("licence")})
    return out


def _digests(path: Path) -> dict[str, str]:
    h = {"sha256": hashlib.sha256(), "md5": hashlib.md5(usedforsecurity=False)}
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            for d in h.values():
                d.update(chunk)
    return {k: v.hexdigest() for k, v in h.items()}


def _get(http: requests.Session, url: str, *, stream: bool, max_wait: float, head: bool = False):
    """GET (or HEAD) following redirects; waits on 429/503 + Retry-After (PANGAEA stages files from tape)."""
    deadline = time.monotonic() + max_wait
    while True:
        r = (http.head(url, allow_redirects=True, timeout=60) if head
             else http.get(url, stream=stream, allow_redirects=True, timeout=120))
        if r.status_code in (429, 503) and time.monotonic() < deadline:
            wait = float(r.headers.get("retry-after") or 30)
            r.close()
            time.sleep(min(max(wait, 5), max(deadline - time.monotonic(), 1)))
            continue
        return r


def check(pin: dict, http: requests.Session, cache: Path | None, max_bytes: int, max_wait: float) -> dict:
    algorithm, _, expected = (pin.get("checksum") or "").partition(":")
    rec = {"name": pin["name"], "url": pin["url"], "licence": pin.get("licence"), "status": None,
           "http_status": None, "algorithm": algorithm or None,
           "sha256_expected": expected.lower() if algorithm == "sha256" else None, "sha256_actual": None,
           "checksum_expected": pin.get("checksum"), "checksum_actual": None, "match": None,
           "bytes": None, "size_expected": pin.get("size")}
    try:
        r = _get(http, pin["url"], stream=True, max_wait=max_wait, head=cache is None)
        rec["http_status"] = r.status_code
        length = r.headers.get("content-length")
        rec["bytes"] = int(length) if length and length.isdigit() else None
        if r.status_code >= 400:
            r.close()
            rec["status"] = "unreachable"
            return rec
        if cache is None:
            rec["status"] = "reachable"
            return rec
        if rec["bytes"] and rec["bytes"] > max_bytes:
            r.close()
            rec["status"] = "skipped (too large)"
            return rec
        target = cache / hashlib.sha256(pin["url"].encode()).hexdigest()[:16] / pin["url"].rsplit("/", 1)[-1]
        if target.is_file() and (rec["bytes"] is None or target.stat().st_size == rec["bytes"]):
            r.close()
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            part = target.with_name(target.name + ".part")
            with r, part.open("wb") as fh:
                for chunk in r.iter_content(1 << 20):
                    fh.write(chunk)
            part.replace(target)
        digests = _digests(target)
        rec["bytes"] = target.stat().st_size
        rec["sha256_actual"] = digests["sha256"]
        if algorithm in digests:
            rec["checksum_actual"] = f"{algorithm}:{digests[algorithm]}"
            rec["match"] = digests[algorithm] == expected.lower() and (
                pin.get("size") is None or pin["size"] == rec["bytes"])
        rec["status"] = "ok" if rec["match"] else ("unpinned" if not algorithm else "mismatch")
    except requests.RequestException as exc:
        rec["status"] = f"unreachable ({type(exc).__name__}: {exc})"
    return rec


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--cache", type=Path, help="download directory (required unless --head-only)")
    p.add_argument("--head-only", action="store_true", help="only check that every pinned URL resolves")
    p.add_argument("--max-bytes", type=int, default=3_000_000_000, help="skip larger downloads")
    p.add_argument("--max-wait", type=float, default=900, help="seconds to wait for staged (503) files")
    p.add_argument("--only", help="substring filter on the pin name")
    a = p.parse_args(argv)
    if not a.head_only and a.cache is None:
        p.error("--cache is required unless --head-only")
    http = requests.Session()
    http.headers["User-Agent"] = USER_AGENT
    selected = [x for x in pins() if not a.only or a.only in x["name"]]
    results = [check(x, http, None if a.head_only else a.cache, a.max_bytes, a.max_wait) for x in selected]
    print(json.dumps(results, indent=2))
    bad = [r for r in results if r["status"] not in ("ok", "reachable", "skipped (too large)")]
    for r in bad:
        print(f"FAIL {r['name']}: {r['status']}", file=sys.stderr)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
