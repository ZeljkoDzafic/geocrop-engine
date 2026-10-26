"""Run-time downloads: retry with backoff, HTTP Range resume, checksum verification, one cache root."""
from __future__ import annotations

import hashlib
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from geocrop import cachedirs, downloads

BODY = bytes(range(256)) * 4096         # 1 MiB
SHA = hashlib.sha256(BODY).hexdigest()


class _Server:
    """Local HTTP server whose behaviour per request is scripted: a list of actions consumed in order."""

    def __init__(self, script):
        self.script, self.requests = list(script), []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                rng = self.headers.get("Range")
                outer.requests.append(rng)
                action = outer.script.pop(0) if outer.script else "ok"
                if isinstance(action, int):
                    self.send_response(action)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                start = int(rng.split("=")[1].split("-")[0]) if rng and action != "ignore_range" else 0
                body = BODY[start:]
                self.send_response(206 if start else 200)
                if start:
                    self.send_header("Content-Range", f"bytes {start}-{len(BODY) - 1}/{len(BODY)}")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Content-Type", "application/octet-stream")
                self.end_headers()
                if action == "truncate":
                    self.wfile.write(body[: len(body) // 3])
                    self.wfile.flush()
                    self.close_connection = True
                    return
                self.wfile.write(b"x" * len(body) if action == "corrupt" else body)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}/weights.bin"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def server():
    made = []

    def make(*script):
        s = _Server(script)
        made.append(s)
        return s
    yield make
    for s in made:
        s.close()


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    waits = []
    monkeypatch.setattr(downloads, "_sleep", waits.append)
    return waits


def test_resumes_truncated_transfer_with_range_request(server, tmp_path, no_sleep):
    srv = server("truncate", "ok")
    out = downloads.fetch(srv.url, tmp_path / "w.bin", f"sha256:{SHA}", size=len(BODY), timeout=5)
    assert out.read_bytes() == BODY
    assert srv.requests[0] is None and srv.requests[1].startswith("bytes=") and srv.requests[1] != "bytes=0-"
    assert len(no_sleep) == 1 and not (tmp_path / "w.bin.part").exists()
    # verified cache hit: no further request
    assert downloads.fetch(srv.url, out, SHA) == out and len(srv.requests) == 2


def test_retries_server_errors_with_exponential_backoff(server, tmp_path, no_sleep):
    srv = server(500, 502, "ok")
    downloads.fetch(srv.url, tmp_path / "w.bin", SHA, backoff=1.0, timeout=5)
    assert len(srv.requests) == 3 and len(no_sleep) == 2
    assert 0.8 <= no_sleep[0] <= 1.2 and 1.6 <= no_sleep[1] <= 2.4      # 1 s, 2 s (+-20 % jitter)


def test_gives_up_after_retries_and_keeps_partial_data(server, tmp_path):
    srv = server("truncate", 503, 503)
    with pytest.raises(downloads.DownloadError, match="after 3 attempts"):
        downloads.fetch(srv.url, tmp_path / "w.bin", SHA, retries=2, timeout=5)
    part = tmp_path / "w.bin.part"
    assert part.exists() and 0 < part.stat().st_size < len(BODY) and not (tmp_path / "w.bin").exists()
    downloads.fetch(srv.url, tmp_path / "w.bin", SHA, timeout=5)        # next call resumes
    assert srv.requests[-1].startswith("bytes=") and (tmp_path / "w.bin").read_bytes() == BODY


def test_server_ignoring_range_restarts_cleanly(server, tmp_path):
    (tmp_path / "w.bin.part").write_bytes(BODY[:1000])
    srv = server("ignore_range")
    downloads.fetch(srv.url, tmp_path / "w.bin", SHA, timeout=5)
    assert (tmp_path / "w.bin").read_bytes() == BODY


def test_corrupt_resume_is_redownloaded_from_scratch_once(server, tmp_path):
    (tmp_path / "w.bin.part").write_bytes(b"\xff" * 1000)              # foreign bytes before resuming
    srv = server("ok", "ok")
    downloads.fetch(srv.url, tmp_path / "w.bin", SHA, timeout=5)
    assert srv.requests == ["bytes=1000-", None] and (tmp_path / "w.bin").read_bytes() == BODY


def test_checksum_mismatch_is_refused(server, tmp_path):
    srv = server("corrupt")
    with pytest.raises(downloads.ChecksumError, match="Checksum mismatch"):
        downloads.fetch(srv.url, tmp_path / "w.bin", SHA, timeout=5)
    assert not (tmp_path / "w.bin").exists() and not (tmp_path / "w.bin.part").exists()
    with pytest.raises(ValueError):                                     # ChecksumError is a ValueError
        downloads.fetch(server("ok").url, tmp_path / "w.bin", "0" * 64, timeout=5)


def test_client_errors_are_not_retried(server, tmp_path, no_sleep):
    srv = server(404)
    with pytest.raises(downloads.DownloadError, match="HTTP 404"):
        downloads.fetch(srv.url, tmp_path / "w.bin", SHA, timeout=5)
    assert len(srv.requests) == 1 and not no_sleep


def test_checksum_parsing():
    assert downloads.parse_checksum(None) is None
    assert downloads.parse_checksum("MD5:ABC") == ("md5", "abc")
    assert downloads.parse_checksum("ab" * 32) == ("sha256", "ab" * 32)
    with pytest.raises(ValueError):
        downloads.parse_checksum("crc32:1")


def test_ftw_download_goes_through_the_robust_fetch(server, tmp_path, monkeypatch):
    from dataclasses import replace

    from geocrop import ftw, ftw_weights
    srv = server("truncate", "ok")
    model = replace(ftw.MODELS[ftw.DEFAULT_MODEL], url=srv.url, sha256=SHA, size=len(BODY))
    monkeypatch.setitem(ftw.MODELS, ftw.DEFAULT_MODEL, model)
    path = ftw_weights.download(ftw.DEFAULT_MODEL, tmp_path / "cache", timeout=5)
    assert path.read_bytes() == BODY and path.parent.parent == tmp_path / "cache"


def test_presto_weights_go_through_the_robust_fetch(server, tmp_path):
    from geocrop import embedding_sources
    srv = server(503, "ok")
    out = embedding_sources.download(srv.url, tmp_path / "p.pt", SHA, size=len(BODY))
    assert out.read_bytes() == BODY and len(srv.requests) == 2


# --------------------------------------------------------------------------- cache root

def test_single_cache_root_with_subfolders(monkeypatch, tmp_path):
    for var in ("GEOCROP_CACHE", "GEOCROP_FTW_CACHE", "GEOCROP_DATA_CACHE"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert cachedirs.cache_root() == tmp_path / "xdg" / "geocrop"
    monkeypatch.setenv("GEOCROP_CACHE", str(tmp_path / "c"))
    from geocrop import embedding_sources, ftw, public_data
    assert ftw.default_cache_dir() == tmp_path / "c" / "ftw"
    assert public_data.cache_dir() == tmp_path / "c" / "public"
    assert embedding_sources.cache_dir() == tmp_path / "c"
    with pytest.raises(ValueError):
        cachedirs.cache_dir("../x")


def test_deprecated_cache_variables_still_work_with_a_warning(monkeypatch, tmp_path):
    monkeypatch.setenv("GEOCROP_CACHE", str(tmp_path / "c"))
    monkeypatch.setenv("GEOCROP_FTW_CACHE", str(tmp_path / "old_ftw"))
    monkeypatch.setenv("GEOCROP_DATA_CACHE", str(tmp_path / "old_public"))
    from geocrop import ftw, public_data
    with pytest.warns(FutureWarning, match="GEOCROP_FTW_CACHE is deprecated"):
        assert ftw.default_cache_dir() == tmp_path / "old_ftw"
    with pytest.warns(FutureWarning, match="GEOCROP_DATA_CACHE is deprecated"):
        assert public_data.cache_dir() == tmp_path / "old_public"
