"""Notebook-friendly HTTP client; requires only the engine's existing requests dependency."""
from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path
from urllib.parse import quote

import requests

API_V1 = "/api/v1"
API_LEGACY = "/api"


def _missing_route(response) -> bool:
    """A 404 from the router (path unknown), as opposed to an unknown job or artifact."""
    if getattr(response, "status_code", 200) != 404:
        return False
    try:
        return response.json().get("detail") == "Not Found"
    except (ValueError, AttributeError):
        return True


def _job(job_id) -> str:
    return quote(str(job_id), safe="")


class EngineClient:
    """HTTP client for the stable ``/api/v1`` contract; falls back to ``/api`` on servers before 1.0."""

    def __init__(self, base_url="http://127.0.0.1:8765", token=None, timeout=30, api_prefix=None):
        self.base_url = base_url.rstrip("/")
        # None: detect on first request (v1 first, unversioned fallback when the server has no v1 routes).
        self.api_prefix = api_prefix
        token = token or os.environ.get("GEOCROP_WEB_TOKEN")
        if not token:
            raise ValueError("Supply a token or set GEOCROP_WEB_TOKEN")
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers["Authorization"] = f"Bearer {token}"

    def _send(self, send, path):
        """Call ``send(url)`` under the versioned prefix, retrying once under the legacy prefix if needed."""
        prefix = self.api_prefix or API_V1
        response = send(self.base_url + prefix + path)
        if self.api_prefix is None:
            if _missing_route(response):
                if hasattr(response, "close"):
                    response.close()
                prefix = API_LEGACY
                response = send(self.base_url + prefix + path)
            if not _missing_route(response):
                self.api_prefix = prefix
        return response

    def _request(self, method, path, **kwargs):
        response = self._send(lambda url: self.session.request(method, url, timeout=self.timeout, **kwargs), path)
        response.raise_for_status()
        return response.json()

    def health(self):
        return self._request("GET", "/health")

    def commands(self):
        return self._request("GET", "/commands")["commands"]

    def submit(self, argv=None, *, command=None):
        if (argv is None) == (command is None):
            raise ValueError("Provide exactly one of argv or command")
        return self._request("POST", "/jobs", json={"argv": argv, "command": command})

    def jobs(self, limit=50):
        return self._request("GET", "/jobs", params={"limit": limit})["jobs"]

    def job(self, job_id):
        return self._request("GET", f"/jobs/{_job(job_id)}")

    def logs(self, job_id, offset=0):
        return self._request("GET", f"/jobs/{_job(job_id)}/logs", params={"offset": offset})

    def cancel(self, job_id):
        return self._request("POST", f"/jobs/{_job(job_id)}/cancel")

    def artifacts(self, job_id):
        return self._request("GET", f"/jobs/{_job(job_id)}/artifacts")["files"]

    def download(self, job_id, path, destination):
        """Stream an artifact into a new file, preserving any existing destination."""
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with self._send(lambda url: self.session.get(url, params={"path": path}, timeout=self.timeout, stream=True),
                            f"/jobs/{_job(job_id)}/artifact") as response:
                response.raise_for_status()
                with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=".geocrop-download-", delete=False) as fh:
                    temporary = Path(fh.name)
                    for chunk in response.iter_content(1024 * 1024):
                        fh.write(chunk)
            # Same-filesystem link fails instead of replacing a user's existing file.
            os.link(temporary, destination)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        return destination

    def wait(self, job_id, timeout=600, poll_interval=1, check=True):
        if timeout <= 0 or poll_interval <= 0:
            raise ValueError("timeout and poll_interval must be positive")
        deadline = time.monotonic() + timeout
        while True:
            job = self.job(job_id)
            if job["status"] in {"succeeded", "failed", "cancelled", "interrupted"}:
                if check and job["status"] != "succeeded":
                    raise RuntimeError(f"GeoCrop job {job_id} ended as {job['status']} (exit {job['returncode']})")
                return job
            if time.monotonic() >= deadline:
                raise TimeoutError(f"GeoCrop job {job_id} is still {job['status']}")
            time.sleep(min(poll_interval, max(0, deadline - time.monotonic())))

    def close(self):
        self.session.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
