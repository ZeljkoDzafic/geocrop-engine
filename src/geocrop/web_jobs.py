"""Durable, single-worker local engine jobs; never executes a shell command."""
from __future__ import annotations

import argparse
import codecs
import json
import os
import queue
import shlex
import signal
import sqlite3
import subprocess
import sys
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

TERMINAL = {"succeeded", "failed", "cancelled", "interrupted"}
BLOCKED_COMMANDS = {"serve", "snap"}
#: Parsed destinations that would run an external executable chosen by the request.
BLOCKED_OPTIONS = {"snap_gpt"}
MAX_ARGS = 256
MAX_ARG_BYTES = 65536
#: Total UTF-8 size of one job's arguments (well below the operating system's ARG_MAX).
MAX_ARGV_BYTES = 131072
#: Queued jobs waiting for the single worker.
MAX_QUEUED_JOBS = 32


class CommandParser(argparse.ArgumentParser):
    def error(self, message):
        raise ValueError(message)

    def exit(self, status=0, message=None):
        raise ValueError(message or "Use the command catalogue for help")

    def print_help(self, file=None):
        pass


def command_catalogue() -> list[dict]:
    from .cli import build_parser
    parser = build_parser(CommandParser)
    sub = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction))
    commands = []
    for name, command in sub.choices.items():
        if name in BLOCKED_COMMANDS:
            continue
        fields = []
        for action in command._actions:
            if not action.option_strings or isinstance(action, argparse._HelpAction):
                continue
            fields.append({"flag": action.option_strings[-1], "required": action.required,
                           "default": action.default, "choices": list(action.choices) if action.choices else None,
                           "boolean": isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction)),
                           "multiple": action.nargs in ("*", "+") or isinstance(action.nargs, int),
                           "help": action.help or ""})
        fields = [f for f in fields if f["flag"] != "--snap-gpt"]
        commands.append({"name": name, "description": command.description, "fields": fields})
    return commands


def validated_command(argv=None, command=None) -> list[str]:
    if (argv is None) == (command is None):
        raise ValueError("Provide exactly one of argv or command")
    args = shlex.split(command) if command is not None else list(argv)
    if args and args[0] == "geocrop":
        args = args[1:]
    if (not args or len(args) > MAX_ARGS
            or any(not isinstance(a, str) or "\x00" in a or len(a.encode("utf8", "surrogatepass")) > MAX_ARG_BYTES
                   for a in args)
            or sum(len(a.encode("utf8", "surrogatepass")) for a in args) > MAX_ARGV_BYTES):
        raise ValueError("Expected a bounded list of engine command arguments")
    allowed = {c["name"] for c in command_catalogue()}
    if args[0] not in allowed or any(a.split("=", 1)[0] == "--snap-gpt" for a in args):
        raise ValueError("This command is not available through the web API")
    from .cli import build_parser
    parsed = build_parser(CommandParser).parse_args(args)
    # argparse accepts unambiguous prefixes (``--snap-g``), so check the parsed value, not just the spelling.
    if any(getattr(parsed, dest, None) is not None for dest in BLOCKED_OPTIONS):
        raise ValueError("This command is not available through the web API")
    return args


def _now():
    return datetime.now(timezone.utc).isoformat()


#: Runs ``geocrop.cli`` in the workspace: ``python -c BOOTSTRAP <workspace> <args...>``. `os.posix_spawn` has no
#: working-directory action before Python 3.13, so the child changes directory itself; like Popen(close_fds=True)
#: it first closes descriptors above stderr that a native library may have left inheritable.
_BOOTSTRAP = ("import os, sys; os.closerange(3, min(os.sysconf('SC_OPEN_MAX'), 65536)); os.chdir(sys.argv[1]); "
              "from geocrop.cli import main; main(sys.argv[2:])")


class SpawnedProcess:
    """The part of `subprocess.Popen` the job worker uses, for a child started with `os.posix_spawn`.

    Both the worker (waiting) and `cancel` (terminating, then waiting) may reap the child, so waiting
    polls under a lock instead of blocking in `waitpid`.
    """

    def __init__(self, pid: int):
        self.pid, self.returncode = pid, None
        self._lock = threading.Lock()

    def poll(self):
        with self._lock:
            if self.returncode is None:
                try:
                    pid, status = os.waitpid(self.pid, os.WNOHANG)
                except ChildProcessError:      # reaped elsewhere; the exit status is lost
                    self.returncode = -1
                else:
                    if pid:
                        self.returncode = os.waitstatus_to_exitcode(status)
            return self.returncode

    def wait(self, timeout=None):
        deadline = None if timeout is None else time.monotonic() + timeout
        while self.poll() is None:
            if deadline is not None and time.monotonic() >= deadline:
                raise subprocess.TimeoutExpired([str(self.pid)], timeout)
            time.sleep(0.05)
        return self.returncode

    def terminate(self):
        os.kill(self.pid, signal.SIGTERM)

    def kill(self):
        os.kill(self.pid, signal.SIGKILL)


def spawn(args: list[str], cwd: Path, env: dict, stdout):
    """Start an engine job child in an isolated process group (and session where supported), with stdout and stderr appended to `stdout`.

    On POSIX the child is started with `os.posix_spawn`, never with fork(): this server process may hold
    libraries with fork handlers that are unsafe in a forked child. In particular PROJ (loaded by map
    previews through rasterio/pyproj) registers a `pthread_atfork` child handler that closes its cached
    SQLite handle of proj.db; when proj.db was replaced on disk (for example a reinstalled pyproj), SQLite logs
    from inside that handler and the forked child crashes with SIGSEGV before exec (every later job then
    exited with -11). posix_spawn runs no atfork handlers. `args` is ``[python, "-m", "geocrop.cli", ...]``.
    """
    if os.name != "posix" or not hasattr(os, "posix_spawn"):
        return subprocess.Popen(args, cwd=cwd, stdout=stdout, stderr=subprocess.STDOUT, env=env)
    if args[1:3] != ["-m", "geocrop.cli"]:
        raise ValueError("Only geocrop.cli jobs are spawned")
    fd = stdout.fileno()
    actions = [(os.POSIX_SPAWN_OPEN, 0, os.devnull, os.O_RDONLY, 0),
               (os.POSIX_SPAWN_DUP2, fd, 1), (os.POSIX_SPAWN_DUP2, fd, 2)]
    argv = [args[0], "-c", _BOOTSTRAP, str(cwd), *args[3:]]
    # Like Popen(restore_signals=True): do not inherit Python's ignored signals.
    options = {"file_actions": actions,
               "setsigdef": [getattr(signal, n) for n in ("SIGPIPE", "SIGXFSZ") if hasattr(signal, n)]}
    try:
        pid = os.posix_spawn(args[0], argv, env, setsid=True, **options)
    except NotImplementedError:
        # Some Python/libc builds lack POSIX_SPAWN_SETSID. A separate process group still makes
        # cancellation safe for the worker's group, without invoking unsafe fork handlers.
        pid = os.posix_spawn(args[0], argv, env, setpgroup=0, **options)
    return SpawnedProcess(pid)


class JobManager:
    """A trusted owner's local research workspace, not a multi-user execution sandbox.

    Metadata survives restarts. Interrupted/queued jobs are not silently replayed. One
    child process runs at a time so independent jobs do not oversubscribe the machine.
    """

    def __init__(self, workspace, state_dir=None, token_env="GEOCROP_WEB_TOKEN"):
        self.workspace = Path(workspace).resolve()
        if not self.workspace.is_dir():
            raise ValueError("Workspace must be an existing directory")
        self.state_dir = Path(state_dir or self.workspace / ".geocrop-web").resolve()
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self._state_lock = (self.state_dir / "worker.lock").open("a+b")
        if os.name == "posix":
            import fcntl
            try:
                fcntl.flock(self._state_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                self._state_lock.close()
                raise ValueError("Another web worker already uses this state directory") from exc
        self.db_path = self.state_dir / "jobs.sqlite3"
        self.token_env = token_env
        self._lock = threading.RLock()
        self._queue = queue.Queue(maxsize=MAX_QUEUED_JOBS)
        self._stopping = threading.Event()
        self._process = None
        self._active = None
        with self._db() as db:
            db.execute("CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, argv TEXT NOT NULL, status TEXT NOT NULL, "
                       "created TEXT NOT NULL, started TEXT, finished TEXT, returncode INTEGER, outputs TEXT NOT NULL)")
            db.execute("UPDATE jobs SET status='interrupted', finished=? WHERE status IN ('queued','running')", (_now(),))
        self._thread = threading.Thread(target=self._worker, name="geocrop-jobs", daemon=True)
        self._thread.start()

    @contextmanager
    def _db(self):
        db = sqlite3.connect(self.db_path, timeout=15)
        try:
            with db:
                yield db
        finally:
            db.close()

    def _outputs(self, args):
        from .cli import build_parser
        ns = build_parser(CommandParser).parse_args(args)
        roots = [getattr(ns, key, None) for key in ("out", "root")]
        if ns.cmd == "run":
            from .config import load
            cfg_path = Path(ns.config)
            try:
                cfg = load(cfg_path if cfg_path.is_absolute() else self.workspace / cfg_path)
            except Exception as exc:
                # The parser's message could quote the file's content; report only the failure type.
                raise ValueError(f"Could not read the run configuration ({type(exc).__name__})") from None
            roots += [cfg["paths"][key] for key in ("output", "features", "composites")]
        resolved = []
        for raw in roots:
            if raw is None:
                continue
            p = Path(raw)
            p = (p if p.is_absolute() else self.workspace / p).resolve()
            if not p.is_relative_to(self.workspace):
                raise ValueError("Web job outputs must remain inside the selected workspace")
            if p == self.workspace or p.is_relative_to(self.state_dir) or self.state_dir.is_relative_to(p):
                raise ValueError("Choose a specific output location outside the web job state directory")
            resolved.append(str(p))
        return list(dict.fromkeys(resolved))

    def submit(self, argv=None, command=None):
        args = validated_command(argv, command)
        outputs = self._outputs(args)
        job_id = uuid.uuid4().hex
        with self._lock:
            if self._stopping.is_set():
                raise ValueError("Job worker is shutting down")
            if self._queue.full():
                raise ValueError("Job queue is full; wait for a job to finish")
            with self._db() as db:
                db.execute("INSERT INTO jobs(id,argv,status,created,outputs) VALUES(?,?,?,?,?)",
                           (job_id, json.dumps(args), "queued", _now(), json.dumps(outputs)))
            self._queue.put_nowait(job_id)
        return self.get(job_id)

    def get(self, job_id):
        with self._db() as db:
            db.row_factory = sqlite3.Row
            row = db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        if row is None:
            raise KeyError(job_id)
        data = dict(row)
        data["argv"] = json.loads(data["argv"])
        data["outputs"] = json.loads(data["outputs"])
        return data

    def list(self, limit=50):
        with self._db() as db:
            ids = [r[0] for r in db.execute("SELECT id FROM jobs ORDER BY created DESC LIMIT ?", (limit,))]
        return [self.get(job_id) for job_id in ids]

    def _update(self, job_id, **fields):
        with self._db() as db:
            db.execute("UPDATE jobs SET " + ",".join(f"{k}=?" for k in fields) + " WHERE id=?",
                       (*fields.values(), job_id))

    def _worker(self):
        while not self._stopping.is_set():
            try:
                job_id = self._queue.get(timeout=.2)
            except queue.Empty:
                continue
            try:
                with self._lock:
                    job = self.get(job_id)
                    if job["status"] != "queued" or self._stopping.is_set():
                        continue
                    self._update(job_id, status="running", started=_now())
                    child_env = os.environ.copy()
                    child_env.pop(self.token_env, None)
                    child_env.pop("GEOCROP_WEB_TOKEN", None)
                    child_env["PYTHONUNBUFFERED"] = "1"
                    child_env["OMP_NUM_THREADS"] = os.environ.get("GEOCROP_WEB_THREADS", "2")
                    child_env["OPENBLAS_NUM_THREADS"] = child_env["OMP_NUM_THREADS"]
                    log = self.state_dir / f"{job_id}.log"
                    with log.open("ab") as output:
                        self._process = spawn([sys.executable, "-m", "geocrop.cli", *job["argv"]],
                                              self.workspace, child_env, output)
                    self._active = job_id
                    process = self._process
                code = process.wait()
                with self._lock:
                    if self.get(job_id)["status"] != "cancelled":
                        self._update(job_id, status="succeeded" if code == 0 else "failed", finished=_now(), returncode=code)
                    else:
                        self._update(job_id, returncode=code)
                    self._process = self._active = None
            except Exception as exc:
                with (self.state_dir / f"{job_id}.log").open("a", encoding="utf8") as output:
                    output.write(f"Worker error: {type(exc).__name__}: {exc}\n")
                self._update(job_id, status="failed", finished=_now())
            finally:
                self._queue.task_done()

    def cancel(self, job_id):
        with self._lock:
            job = self.get(job_id)
            if job["status"] in TERMINAL:
                return job
            self._update(job_id, status="cancelled", finished=_now())
            if self._active == job_id and self._process is not None:
                process = self._process
                try:
                    if os.name == "posix":
                        os.killpg(process.pid, signal.SIGTERM)
                    else:
                        process.terminate()
                    try:
                        process.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        if os.name == "posix":
                            os.killpg(process.pid, signal.SIGKILL)
                        else:
                            process.kill()
                except ProcessLookupError:
                    pass
        return self.get(job_id)

    def logs(self, job_id, offset=0, limit=65536):
        self.get(job_id)
        path = self.state_dir / f"{job_id}.log"
        data = b""
        if path.exists():
            with path.open("rb") as fh:
                fh.seek(offset)
                data = fh.read(limit)
                decoder = codecs.getincrementaldecoder("utf8")(errors="replace")
                decoder.decode(data, final=False)
                # Finish at most one UTF-8 character so incremental polling never splits it.
                for _ in range(3):
                    if not decoder.getstate()[0]:
                        break
                    extra = fh.read(1)
                    if not extra:
                        break
                    data += extra
                    decoder.decode(extra, final=False)
        return {"text": data.decode("utf8", errors="replace"), "next_offset": offset + len(data)}

    def artifacts(self, job_id):
        job = self.get(job_id)
        files = []
        for raw in job["outputs"]:
            root = Path(raw)
            candidates = [root] if root.is_file() else root.rglob("*") if root.is_dir() else []
            for path in candidates:
                resolved = path.resolve()
                if (path.is_file() and resolved.is_relative_to(self.workspace)
                        and not resolved.is_relative_to(self.state_dir)):
                    files.append({"path": str(path.relative_to(self.workspace)), "bytes": path.stat().st_size})
                if len(files) >= 200:
                    return files
        return files

    def output_files(self, job_id, limit: int = 5000) -> list[Path]:
        """All files under the job's declared outputs (workspace-confined), for server-side map previews."""
        job = self.get(job_id)
        files = []
        for raw in job["outputs"]:
            root = Path(raw)
            for path in ([root] if root.is_file() else sorted(root.rglob("*")) if root.is_dir() else []):
                resolved = path.resolve()
                if (path.is_file() and resolved.is_relative_to(self.workspace)
                        and not resolved.is_relative_to(self.state_dir)):
                    files.append(resolved)
                    if len(files) >= limit:
                        return files
        return files

    def artifact(self, job_id, relative):
        allowed = {item["path"] for item in self.artifacts(job_id)}
        if relative not in allowed:
            raise KeyError(relative)
        path = (self.workspace / relative).resolve()
        if not path.is_relative_to(self.workspace) or path.is_relative_to(self.state_dir):
            raise KeyError(relative)
        return path

    def close(self):
        self._stopping.set()
        with self._lock:
            if self._active:
                self.cancel(self._active)
        self._thread.join(timeout=5)
        with self._db() as db:
            db.execute("UPDATE jobs SET status='interrupted', finished=? WHERE status='queued'", (_now(),))
        self._state_lock.close()
