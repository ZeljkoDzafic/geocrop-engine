"""Job worker edge paths: child-process bookkeeping, refused arguments and output listing limits."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from geocrop import web_jobs
from geocrop.web_jobs import JobManager, SpawnedProcess

posix_only = pytest.mark.skipif(os.name != "posix" or not hasattr(os, "posix_spawn"), reason="POSIX spawn only")


@posix_only
def test_spawned_process_timeout_kill_and_lost_status():
    pid = os.posix_spawn(sys.executable, [sys.executable, "-c", "import time; time.sleep(30)"], dict(os.environ))
    proc = SpawnedProcess(pid)
    with pytest.raises(subprocess.TimeoutExpired):
        proc.wait(timeout=0.1)
    proc.kill()
    assert proc.wait(timeout=10) != 0
    assert proc.poll() == proc.returncode                     # status is cached after reaping
    # a pid that is not our child (already reaped elsewhere) reports a lost exit status
    assert SpawnedProcess(1).poll() == -1


@posix_only
def test_spawn_refuses_non_engine_commands(tmp_path):
    with open(tmp_path / "log", "wb") as fh, pytest.raises(ValueError, match="geocrop.cli"):
        web_jobs.spawn([sys.executable, "-c", "print(1)"], tmp_path, dict(os.environ), fh)


def test_job_manager_requires_workspace_and_refuses_after_close(tmp_path):
    with pytest.raises(ValueError, match="existing directory"):
        JobManager(tmp_path / "missing")
    manager = JobManager(tmp_path)
    manager.close()
    with pytest.raises(ValueError, match="shutting down"):
        manager.submit(argv=["indices"])


def _insert(manager, job_id, outputs):
    with manager._db() as db:
        db.execute("INSERT INTO jobs(id,argv,status,created,outputs) VALUES(?,?,?,?,?)",
                   (job_id, json.dumps(["indices"]), "succeeded", "2026-01-01T00:00:00", json.dumps(outputs)))


def test_artifact_listings_are_capped_and_confined(tmp_path):
    out = tmp_path / "run"
    out.mkdir()
    for i in range(205):
        (out / f"f{i:03d}.txt").write_text("x")
    outside = tmp_path.parent / f"{tmp_path.name}-outside.txt"
    outside.write_text("secret")
    manager = JobManager(tmp_path)
    try:
        _insert(manager, "j1", [str(out), str(outside), str(tmp_path / "absent")])
        listed = manager.artifacts("j1")
        assert len(listed) == 200 and all(item["path"].startswith("run") for item in listed)
        assert len(manager.output_files("j1", limit=7)) == 7
        assert len(manager.output_files("j1")) == 205              # the outside file is never listed
        assert manager.artifact("j1", listed[0]["path"]) == (tmp_path / listed[0]["path"]).resolve()
        with pytest.raises(KeyError):
            manager.artifact("j1", "../" + outside.name)
        assert manager.logs("j1") == {"text": "", "next_offset": 0}  # no log written yet
        assert manager.cancel("j1")["status"] == "succeeded"         # terminal jobs are not changed
    finally:
        manager.close()
        outside.unlink()


# ------------------------------------------------------------------------------------------------- client
class _Resp:
    def __init__(self, status=200, body=None, json_error=False):
        self.status_code, self.body, self.json_error, self.closed = status, body or {}, json_error, False

    def json(self):
        if self.json_error:
            raise ValueError("not JSON")
        return self.body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)

    def close(self):
        self.closed = True


def test_client_needs_a_token_and_detects_the_api_prefix(monkeypatch):
    from geocrop import client as cl
    monkeypatch.delenv("GEOCROP_WEB_TOKEN", raising=False)
    with pytest.raises(ValueError, match="token"):
        cl.EngineClient()
    assert cl._missing_route(_Resp(404, json_error=True))           # non-JSON 404 is a router miss
    assert not cl._missing_route(_Resp(404, {"detail": "Unknown job"}))
    monkeypatch.setenv("GEOCROP_WEB_TOKEN", "t")
    with cl.EngineClient() as c:
        urls = []
        first = _Resp(404, {"detail": "Not Found"})

        def request(method, url, **kwargs):
            urls.append(url)
            return first if "/api/v1/" in url else _Resp(200, {"jobs": [{"id": "a"}], "files": [], "id": "a"})
        monkeypatch.setattr(c.session, "request", request)
        assert c.jobs() == [{"id": "a"}] and first.closed and c.api_prefix == cl.API_LEGACY
        assert c.job("a/b")["id"] == "a" and urls[-1].endswith("/api/jobs/a%2Fb")
        c.logs("a"), c.cancel("a"), c.artifacts("a")
        assert all("/api/v1/" not in u for u in urls[1:])
    # both prefixes unknown: the prefix stays undetected for the next call
    c = cl.EngineClient(token="t")
    monkeypatch.setattr(c.session, "request", lambda method, url, **kw: _Resp(404, {"detail": "Not Found"}))
    with pytest.raises(RuntimeError):
        c.health()
    assert c.api_prefix is None


def test_client_download_failure_leaves_no_partial_file(tmp_path, monkeypatch):
    from geocrop import client as cl
    c = cl.EngineClient(token="t", api_prefix=cl.API_V1)

    class Broken(_Resp):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def iter_content(self, size):
            yield b"partial"
            raise ConnectionError("dropped")
    monkeypatch.setattr(c.session, "get", lambda url, **kw: Broken())
    with pytest.raises(ConnectionError):
        c.download("job", "run/x.tif", tmp_path / "d" / "x.tif")
    assert list((tmp_path / "d").iterdir()) == []
    (tmp_path / "d" / "x.tif").write_text("mine")
    monkeypatch.setattr(c.session, "get", lambda url, **kw: type("R", (Broken,), {
        "iter_content": lambda self, size: iter([b"new"])})())
    with pytest.raises(FileExistsError):
        c.download("job", "run/x.tif", tmp_path / "d" / "x.tif")
    assert (tmp_path / "d" / "x.tif").read_text() == "mine" and len(list((tmp_path / "d").iterdir())) == 1
    with pytest.raises(ValueError, match="positive"):
        c.wait("job", timeout=0)


@posix_only
def test_spawn_without_setsid_still_isolates_the_process_group(tmp_path, monkeypatch):
    original = os.posix_spawn
    calls = []

    def without_setsid(*args, **kwargs):
        calls.append(kwargs)
        if kwargs.get('setsid'):
            raise NotImplementedError('posix_spawn: setsid unavailable on this platform')
        return original(*args, **kwargs)

    def no_fork(*args, **kwargs):
        raise AssertionError('fallback must not fork')

    monkeypatch.setattr(os, 'posix_spawn', without_setsid)
    monkeypatch.setattr(subprocess, 'Popen', no_fork)
    with (tmp_path / 'job.log').open('wb') as out:
        proc = web_jobs.spawn([sys.executable, '-m', 'geocrop.cli', 'indices'], tmp_path,
                              {**os.environ, 'PYTHONPATH': str(Path(web_jobs.__file__).parents[1])}, out)
    assert os.getpgid(proc.pid) == proc.pid
    assert proc.wait(timeout=30) == 0
    assert calls[0]['setsid'] and calls[1]['setpgroup'] == 0
