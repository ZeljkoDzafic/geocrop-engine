"""Dependency-free transport for QGIS's own Python runtime. No engine imports."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path, PurePosixPath
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode, urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener


class ApiError(RuntimeError):
    pass


class Cancelled(RuntimeError):
    pass


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None  # Never forward the access token to a redirected destination.


class ApiClient:
    def __init__(self, base_url, token, timeout=30):
        parsed = urlsplit(base_url)
        if (parsed.scheme not in ("http", "https") or parsed.hostname not in ("127.0.0.1", "localhost", "::1")
                or parsed.username or parsed.password or parsed.query or parsed.fragment):
            raise ApiError("Use a loopback engine URL; remote engines need a locally forwarded port")
        if not token:
            raise ApiError("Configure a token through the selected environment variable or QGIS authentication config")
        self.base_url = base_url.rstrip("/")
        self._token = token
        self.timeout = timeout
        self._opener = build_opener(ProxyHandler({}), NoRedirect())

    def _open(self, method, path, payload=None):
        body = json.dumps(payload).encode() if payload is not None else None
        request = Request(self.base_url + path, data=body, method=method,
                          headers={"Authorization": f"Bearer {self._token}", "Content-Type": "application/json"})
        try:
            return self._opener.open(request, timeout=self.timeout)
        except HTTPError as exc:
            exc.close()
            if exc.code == 401:
                raise ApiError("Engine authentication failed; check the configured token") from None
            raise ApiError(f"Engine HTTP {exc.code}; check server logs and command parameters") from None
        except (URLError, OSError) as exc:
            raise ApiError("Cannot reach the engine; check its URL, port and running server") from exc

    def request(self, method, path, payload=None):
        with self._open(method, path, payload) as response:
            raw = response.read(5 * 1024 * 1024 + 1)
        if len(raw) > 5 * 1024 * 1024:
            raise ApiError("Engine metadata response exceeds the supported size")
        try:
            return json.loads(raw)
        except (ValueError, UnicodeError) as exc:
            raise ApiError("Engine returned invalid JSON") from exc

    def health(self):
        return self.request("GET", "/api/health")

    def submit(self, argv=None, command=None):
        if (argv is None) == (command is None):
            raise ApiError("Supply exactly one of argv or command")
        return self.request("POST", "/api/jobs", {"argv": argv, "command": command})

    def job(self, job_id):
        return self.request("GET", f"/api/jobs/{quote(job_id, safe='')}")

    def logs(self, job_id, offset=0):
        return self.request("GET", f"/api/jobs/{quote(job_id, safe='')}/logs?{urlencode({'offset': offset})}")

    def cancel(self, job_id):
        return self.request("POST", f"/api/jobs/{quote(job_id, safe='')}/cancel")

    def artifacts(self, job_id):
        return self.request("GET", f"/api/jobs/{quote(job_id, safe='')}/artifacts")["files"]

    def download(self, job_id, artifact, directory, cancelled=lambda: False):
        """Keep server-relative paths beneath a fresh local directory; never replace files."""
        path = PurePosixPath(artifact)
        if (not artifact or path.is_absolute() or ".." in path.parts or "\\" in artifact or ":" in artifact):
            raise ApiError("Invalid artifact path returned by the engine")
        root = Path(directory).resolve()
        destination = (root / Path(*path.parts)).resolve()
        if not destination.is_relative_to(root):
            raise ApiError("Artifact path escapes the local results directory")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            query = urlencode({"path": artifact})
            with self._open("GET", f"/api/jobs/{quote(job_id, safe='')}/artifact?{query}") as response:
                with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=".geocrop-", delete=False) as fh:
                    temporary = Path(fh.name)
                    while True:
                        if cancelled():
                            raise Cancelled("Result download cancelled")
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        fh.write(chunk)
            os.link(temporary, destination)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)  # Only this invocation's new temporary file.
        return destination
