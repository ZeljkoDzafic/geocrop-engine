"""Evidence-based release gate: VERIFY a commit (it builds nothing that is shipped).

    python tools/release_gate.py --commit HEAD --scratch /tmp/gate            # every check
    python tools/release_gate.py --commit c091962 --scratch /tmp/gate --only a.ruff a.pytest_default
    python tools/release_gate.py ... --publish docs/release-gates             # also copy the Markdown report

The commit is cloned into ``<scratch>/src`` (a pristine checkout; the working tree that runs the gate is never
used as the code under test), a fresh virtual environment is created in ``<scratch>/venv`` and installed with
``pip install -c constraints/py312.txt -e .[<every extra declared in pyproject.toml>]``. The gate's own helper
scripts (tools/gate_integration.py, tools/recovery_check.py, tools/check_weights.py) and its contract tests
(tests/test_plugin_contracts.py) come from the tree that runs the gate, so a later gate can verify an older
commit. Every check records its command(s), exit codes, duration and evidence; ``release_gate_report.json`` and
``release_gate_report.md`` are written to ``--out`` (default ``<scratch>``), and the process exits 1 when any
check fails. Checks that cannot run here (no Docker, no network) are recorded as ``skip`` with the reason - never
as ``pass``. See docs/release-gate.md.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

GATE_ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "geocrop-release-gate/1"
DOCKER_TAG = "geocrop-release-gate:{commit}"
PROTOCOL = "configs/benchmarks/eurocrops_si_2021_pomurje.v1.yaml"
PUBLISHED = "docs/benchmarks/eurocrops-si-2021-pomurje.v1.json"
# Licence identifiers that must not appear among installed runtime dependencies without a documented decision.
COPYLEFT = re.compile(r"\b(A?GPL|GNU (Affero )?General Public|SSPL)", re.I)
WEAK_COPYLEFT = re.compile(r"\b(LGPL|Lesser General Public|MPL|Mozilla Public|EPL|Eclipse Public)", re.I)


@dataclass
class CheckResult:
    id: str
    priority: str
    status: str                      # pass | fail | skip
    seconds: float
    evidence: dict = field(default_factory=dict)
    notes: str = ""
    commands: list = field(default_factory=list)


class Gate:
    def __init__(self, args):
        self.args = args
        self.scratch = Path(args.scratch).resolve()
        self.src = self.scratch / "src"
        self.venv = self.scratch / "venv"
        self.logs = self.scratch / "logs"
        self.results: list[CheckResult] = []
        self.commit = ""
        self.facts: dict = {}

    # ------------------------------------------------------------------ helpers
    @property
    def py(self) -> str:
        return str(self.venv / "bin" / "python")

    @property
    def geocrop(self) -> str:
        return str(self.venv / "bin" / "geocrop")

    def sh(self, name: str, cmd: list[str], cwd: Path | None = None, timeout: float = 3600, env: dict | None = None,
           commands: list | None = None) -> subprocess.CompletedProcess:
        """Run `cmd`, tee the output into logs/<name>.log and return the completed process (never raises)."""
        self.logs.mkdir(parents=True, exist_ok=True)
        log = self.logs / f"{name}.log"
        full_env = {**os.environ, **(env or {})}
        full_env.pop("PYTHONPATH", None)      # the code under test comes from the clean venv only
        for key, value in (env or {}).items():
            full_env[key] = value
        t0 = time.monotonic()
        try:
            proc = subprocess.run(cmd, cwd=cwd or self.src, env=full_env, capture_output=True, text=True,
                                  timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            proc = subprocess.CompletedProcess(cmd, 124, exc.stdout or "", f"{exc.stderr or ''}\nTIMEOUT after {timeout}s")
        except FileNotFoundError as exc:
            proc = subprocess.CompletedProcess(cmd, 127, "", str(exc))
        stdout = proc.stdout if isinstance(proc.stdout, str) else (proc.stdout or b"").decode(errors="replace")
        stderr = proc.stderr if isinstance(proc.stderr, str) else (proc.stderr or b"").decode(errors="replace")
        proc.stdout, proc.stderr = stdout, stderr
        log.write_text(f"$ {' '.join(map(str, cmd))}\n# cwd={cwd or self.src} exit={proc.returncode} "
                       f"seconds={time.monotonic() - t0:.1f}\n{stdout}\n--- stderr ---\n{stderr}", encoding="utf8")
        if commands is not None:
            commands.append({"cmd": " ".join(map(str, cmd)), "exit": proc.returncode,
                             "seconds": round(time.monotonic() - t0, 1), "log": str(log)})
        return proc

    def record(self, id_: str, priority: str, fn) -> CheckResult:
        if self.args.only and not any(id_ == o or id_.startswith(o + ".") or id_.startswith(o)
                                      for o in self.args.only):
            return None
        if any(id_.startswith(s) for s in self.args.skip):
            res = CheckResult(id_, priority, "skip", 0.0, notes="skipped on request (--skip)")
            self.results.append(res)
            return res
        print(f"[gate] {id_} ...", flush=True)
        commands: list = []
        t0 = time.monotonic()
        try:
            status, evidence, notes = fn(commands)
        except Exception as exc:  # a crashing check is a failed check with its traceback as evidence
            import traceback
            status, evidence, notes = "fail", {"exception": traceback.format_exc()[-3000:]}, f"{type(exc).__name__}: {exc}"
        res = CheckResult(id_, priority, status, round(time.monotonic() - t0, 1), evidence, notes, commands)
        self.results.append(res)
        self._partial()
        print(f"[gate] {id_}: {status.upper()} ({res.seconds} s) {notes}", flush=True)
        return res

    def _partial(self) -> None:
        """Write the results so far (``release_gate_report.partial.json``): an interrupted gate keeps its evidence
        and can be completed with ``--reuse --merge``."""
        report = build_report(self.results, self.facts, {"partial": True, "utc": datetime.now(timezone.utc).isoformat(timespec="seconds")})
        out = Path(self.args.out or self.scratch)
        out.mkdir(parents=True, exist_ok=True)
        (out / "release_gate_report.partial.json").write_text(json.dumps(report, indent=2, default=str),
                                                              encoding="utf8")

    def network_ok(self) -> bool:
        if self.args.no_network:
            return False
        import urllib.request
        try:
            urllib.request.urlopen("https://earth-search.aws.element84.com/v1", timeout=20).read(100)
            return True
        except Exception:
            return False

    def docker_ok(self) -> tuple[bool, str]:
        if self.args.no_docker:
            return False, "disabled with --no-docker"
        if shutil.which("docker") is None:
            return False, "docker executable not available"
        proc = subprocess.run(["docker", "info", "--format", "{{.ServerVersion}}"], capture_output=True, text=True)
        if proc.returncode:
            return False, f"docker daemon not available: {proc.stderr.strip()[:200]}"
        return True, proc.stdout.strip()

    # ------------------------------------------------------------------ preparation
    def prepare(self) -> None:
        self.scratch.mkdir(parents=True, exist_ok=True)
        repo = Path(self.args.repo).resolve()
        sha = subprocess.run(["git", "-C", str(repo), "rev-parse", self.args.commit], capture_output=True,
                             text=True, check=True).stdout.strip()
        self.commit = sha
        if self.src.exists() and not self.args.reuse:
            shutil.rmtree(self.src)
        if not self.src.exists():
            subprocess.run(["git", "clone", "--quiet", "--no-local", str(repo), str(self.src)], check=True)
        subprocess.run(["git", "-C", str(self.src), "checkout", "--quiet", "--detach", sha], check=True)
        status = subprocess.run(["git", "-C", str(self.src), "status", "--porcelain"], capture_output=True,
                                text=True).stdout
        gate_sha = subprocess.run(["git", "-C", str(GATE_ROOT), "rev-parse", "HEAD"], capture_output=True,
                                  text=True).stdout.strip()
        self.facts = {"commit": sha, "short": sha[:7], "requested": self.args.commit, "checkout_clean": not status,
                      "subject": subprocess.run(["git", "-C", str(self.src), "log", "-1", "--format=%s"],
                                                capture_output=True, text=True).stdout.strip(),
                      "gate_tools_commit": gate_sha,
                      "gate_tools_dirty": bool(subprocess.run(["git", "-C", str(GATE_ROOT), "status", "--porcelain",
                                                               "tools", "tests"], capture_output=True,
                                                              text=True).stdout.strip())}

    def extras(self) -> list[str]:
        # read with the base interpreter (3.11+ has tomllib), so the gate itself runs on any Python >= 3.9
        code = ("import json, sys, tomllib; m = tomllib.load(open(sys.argv[1], 'rb'));"
                "print(json.dumps(sorted(m['project'].get('optional-dependencies', {}))))")
        out = subprocess.run([self.args.python, "-c", code, str(self.src / "pyproject.toml")], capture_output=True,
                             text=True, check=True).stdout
        return json.loads(out)

    def environment(self) -> dict:
        env = {"utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "host_platform": platform.platform(),
               "machine": platform.machine(), "cpu_count": os.cpu_count(), "gate_python": sys.version.split()[0],
               "base_python": self.args.python}
        try:
            env["base_python_version"] = subprocess.run([self.args.python, "--version"], capture_output=True,
                                                        text=True).stdout.strip()
        except OSError:
            pass
        if sys.platform == "darwin":
            mem = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True).stdout.strip()
            env["memory_gb"] = round(int(mem) / 2**30, 1) if mem.isdigit() else None
        elif Path("/proc/meminfo").exists():
            kb = int(Path("/proc/meminfo").read_text().split()[1])
            env["memory_gb"] = round(kb / 2**20, 1)
        ok, info = self.docker_ok()
        env["docker"] = info if ok else f"not available ({info})"
        if self.venv.exists():
            freeze = subprocess.run([self.py, "-m", "pip", "freeze", "--exclude-editable"], capture_output=True,
                                    text=True).stdout
            env["venv_packages"] = len(freeze.splitlines())
            env["venv_freeze_sha256"] = hashlib.sha256(freeze.encode()).hexdigest()
            (self.scratch / "pip_freeze.txt").write_text(freeze, encoding="utf8")
            versions = subprocess.run([self.py, "-c", (
                "import importlib.metadata as m, json\nout={}\n"
                "for d in ['numpy','pandas','scikit-learn','rasterio','geopandas','torch','catboost','lightgbm',"
                "'segmentation-models-pytorch','pystac-client','fastapi','pyarrow']:\n"
                "  try: out[d]=m.version(d)\n  except Exception: out[d]=None\nprint(json.dumps(out))")],
                capture_output=True, text=True).stdout
            env["key_versions"] = json.loads(versions or "{}")
        return env

    # ------------------------------------------------------------------ (a) clean install + suites
    def check_install(self, commands):
        if self.venv.exists() and not self.args.reuse:
            shutil.rmtree(self.venv)
        if not self.venv.exists():
            self.sh("venv", [self.args.python, "-m", "venv", str(self.venv)], commands=commands)
            self.sh("pip-upgrade", [self.py, "-m", "pip", "install", "-q", "--upgrade", "pip"], commands=commands)
        extras = self.extras()
        spec = f".[{','.join(extras)}]"
        proc = self.sh("pip-install", [self.py, "-m", "pip", "install", "-c", "constraints/py312.txt", "-e", spec],
                       commands=commands)
        if proc.returncode:
            return "fail", {"extras": extras, "tail": proc.stderr[-2000:]}, "pip install failed"
        imports = self.sh("imports", [self.py, "-c", (
            "import importlib, json\nres={}\n"
            "for m in ['geocrop','skimage','pystac_client','torch','fastapi','uvicorn','httpx','lightgbm','catboost',"
            "'omnicloudmask','pyarrow','segmentation_models_pytorch']:\n"
            "  try: importlib.import_module(m); res[m]='ok'\n"
            "  except Exception as e: res[m]=f'{type(e).__name__}: {str(e).splitlines()[0][:160]}'\n"
            "print(json.dumps(res))")], commands=commands)
        imported = json.loads(imports.stdout or "{}")
        version = self.sh("version", [self.geocrop, "--version"], commands=commands).stdout.strip()
        broken = {k: v for k, v in imported.items() if v != "ok"}
        notes = ""
        if broken:
            notes = "installed but not importable: " + "; ".join(f"{k} ({v})" for k, v in broken.items())
        status = "fail" if "geocrop" in broken or not version else "pass"
        return status, {"extras": extras, "install_spec": f"pip install -c constraints/py312.txt -e '{spec}'",
                        "geocrop_version": version, "imports": imported}, notes

    def _pytest(self, name, marker_args, commands, timeout=5400):
        proc = self.sh(name, [self.py, "-m", "pytest", "-q", "-p", "no:warnings", "-p", "no:cacheprovider",
                              *marker_args], commands=commands, timeout=timeout)
        summary = _last_summary(proc.stdout)
        counts = pytest_counts(summary)
        failures = [ln for ln in proc.stdout.splitlines() if ln.startswith(("FAILED", "ERROR"))][:30]
        ok = proc.returncode == 0 or (proc.returncode == 5 and not failures)   # 5 = nothing collected
        return ("pass" if ok else "fail"), {"summary": summary, "counts": counts, "failures": failures}, \
            ("" if ok else f"pytest exit {proc.returncode}")

    def check_ruff(self, commands):
        proc = self.sh("ruff", [str(self.venv / "bin" / "ruff"), "check", "src", "tests"], commands=commands)
        return ("pass" if proc.returncode == 0 else "fail"), {"output": proc.stdout.strip()[-1500:]}, ""

    def check_acceptance(self, commands, offline: bool):
        out = self.scratch / ("acceptance_offline" if offline else "acceptance_real")
        if out.exists():
            shutil.rmtree(out)
        if not offline and not self.network_ok():
            return "skip", {}, "network unavailable"
        cmd = [self.geocrop, "acceptance", "--out", str(out)] + (["--offline"] if offline else [])
        proc = self.sh(out.name, cmd, commands=commands, timeout=3600)
        report = out / "acceptance_report.json"
        if not report.exists():
            return "fail", {"tail": (proc.stdout + proc.stderr)[-2000:]}, "no acceptance_report.json"
        rep = json.loads(report.read_text())
        ev = {k: rep.get(k) for k in ("verdict", "summary", "runtime_s", "error", "mode", "data_sources")
              if k in rep}
        ev["report"] = str(report)
        ok = proc.returncode == 0 and str(rep.get("verdict", "")).upper().startswith("PASS")
        return ("pass" if ok else "fail"), ev, "" if ok else f"verdict {rep.get('verdict')}"

    # ------------------------------------------------------------------ (a) Docker as clean-machine proxy
    def _docker_image(self) -> str:
        return DOCKER_TAG.format(commit=self.commit[:7])

    def check_docker_build(self, commands):
        ok, info = self.docker_ok()
        if not ok:
            return "skip", {"docker": info}, f"Docker not available: {info}"
        cmd = ["docker", "build", "-t", self._docker_image()] + (["--no-cache"] if self.args.docker_no_cache else []) + ["."]
        proc = self.sh("docker-build", cmd, commands=commands, timeout=5400)
        inspect = subprocess.run(["docker", "image", "inspect", self._docker_image(), "--format",
                                  "{{.Id}} {{.Architecture}} {{.Size}}"], capture_output=True, text=True).stdout.split()
        cached = len(re.findall(r"\bCACHED\b", proc.stdout + proc.stderr))
        ev = {"image": self._docker_image(), "docker_server": info, "no_cache": self.args.docker_no_cache,
              "cached_layers": cached}
        if inspect:
            ev.update(image_id=inspect[0], architecture=inspect[1], size_mb=round(int(inspect[2]) / 2**20))
        return ("pass" if proc.returncode == 0 else "fail"), ev, "" if proc.returncode == 0 else "docker build failed"

    def _docker_run(self, name, args, commands, entry=None, mounts=(), timeout=1800):
        cmd = ["docker", "run", "--rm"]
        for host, guest in mounts:
            cmd += ["-v", f"{host}:{guest}"]
        if entry:
            cmd += ["--entrypoint", "/usr/local/bin/_entrypoint.sh"]
        cmd += [self._docker_image(), *args]
        return self.sh(name, cmd, commands=commands, timeout=timeout)

    def _docker_ready(self):
        built = [r for r in self.results if r.id == "a.docker_build"]
        if not built or built[0].status != "pass":
            return "Docker image not built (see a.docker_build)"
        return None

    def check_docker_help(self, commands):
        if reason := self._docker_ready():
            return "skip", {}, reason
        proc = self._docker_run("docker-help", ["--help"], commands)
        version = self._docker_run("docker-version", ["--version"], commands).stdout.strip()
        ok = proc.returncode == 0 and "acceptance" in proc.stdout
        return ("pass" if ok else "fail"), {"version": version, "help_lines": len(proc.stdout.splitlines())}, ""

    def check_docker_acceptance(self, commands):
        if reason := self._docker_ready():
            return "skip", {}, reason
        proc = self._docker_run("docker-acceptance-offline", ["acceptance", "--offline", "--out", "/tmp/acc"], commands)
        try:
            rep = json.loads(proc.stdout[proc.stdout.index("{"):])
        except ValueError:
            rep = {}
        ok = proc.returncode == 0 and str(rep.get("verdict", "")).upper().startswith("PASS")
        return ("pass" if ok else "fail"), {k: rep.get(k) for k in ("verdict", "summary", "runtime_s", "error")}, ""

    def check_docker_demo(self, commands):
        if reason := self._docker_ready():
            return "skip", {}, reason
        proc = self._docker_run("docker-demo", ["demo", "--quick", "--root", "/tmp/demo"], commands)
        try:
            summary = json.loads(proc.stdout[proc.stdout.index("{"):])
        except ValueError:
            summary = {}
        ok = proc.returncode == 0 and "cv_macro_f1" in summary
        return ("pass" if ok else "fail"), {"mode": summary.get("mode"), "cv_macro_f1": summary.get("cv_macro_f1"),
                                            "field_points_oa": summary.get("field_points_oa")}, \
            "SYNTHETIC demo: software check, not accuracy"

    def check_docker_recovery(self, commands):
        if reason := self._docker_ready():
            return "skip", {}, reason
        proc = self._docker_run("docker-recovery", ["python", "/gate/recovery_check.py", "--work", "/tmp/rec",
                                                    "--point", "train"], commands, entry=True,
                                mounts=[(GATE_ROOT / "tools", "/gate:ro")])
        try:
            res = json.loads(proc.stdout[proc.stdout.index("{"):])
        except ValueError:
            res = {"problems": [proc.stderr[-1500:]]}
        ok = proc.returncode == 0 and res.get("recovered")
        return ("pass" if ok else "fail"), res, "SIGKILL inside `train`, then rerun"

    def check_host_recovery(self, commands):
        proc = self.sh("recovery-host", [self.py, str(GATE_ROOT / "tools" / "recovery_check.py"), "--work",
                                         str(self.scratch / "recovery"), "--point", "predict"], commands=commands)
        try:
            res = json.loads(proc.stdout[proc.stdout.index("{"):])
        except ValueError:
            res = {"problems": [proc.stderr[-1500:]]}
        ok = proc.returncode == 0 and res.get("recovered")
        return ("pass" if ok else "fail"), res, "SIGKILL inside `predict`, then rerun (clean venv)"

    # ------------------------------------------------------------------ (b) combined integration
    def check_integration(self, commands):
        script = GATE_ROOT / "tools" / "gate_integration.py"
        if not script.exists():
            return "skip", {}, "tools/gate_integration.py not present"
        work = self.scratch / "integration"
        if work.exists():
            shutil.rmtree(work)
        out = self.scratch / "integration.json"
        cmd = [self.py, str(script), "--work", str(work), "--json", str(out)]
        if self.network_ok():
            cmd.append("--network")
        real = self.scratch / "acceptance_real"
        if (real / "acceptance_report.json").exists():
            cmd += ["--real-composites", str(real)]
        proc = self.sh("integration", cmd, commands=commands, timeout=5400)
        res = json.loads(out.read_text()) if out.exists() else {}
        steps = res.get("steps", res if isinstance(res, list) else [])
        table = [{k: s.get(k) for k in ("name", "status", "seconds", "reason") if k in s} for s in steps] \
            if isinstance(steps, list) else steps
        ok = proc.returncode == 0 and bool(steps)
        return ("pass" if ok else "fail"), {"steps": table, "evidence": res.get("evidence", {}) if isinstance(res, dict)
                                            else {}}, "" if ok else f"exit {proc.returncode}"

    # ------------------------------------------------------------------ (c) benchmark reproducibility
    def check_benchmark(self, commands):
        if not self.network_ok():
            return "skip", {}, "network unavailable"
        work, cache = self.scratch / "bench_work", self.scratch / "bench_cache"
        for d in (work, cache):
            if d.exists():
                shutil.rmtree(d)
        proc = self.sh("benchmark-public", [self.geocrop, "benchmark-public", "--protocol", PROTOCOL, "--work",
                                            str(work), "--cache", str(cache)], commands=commands, timeout=7200)
        new = work / "results.json"
        if not new.exists():
            return "fail", {"tail": (proc.stdout + proc.stderr)[-2000:]}, "benchmark produced no results.json"
        cmp = compare_benchmarks(json.loads((self.src / PUBLISHED).read_text()), json.loads(new.read_text()))
        ok = proc.returncode == 0 and not cmp["unexplained"]
        return ("pass" if ok else "fail"), cmp, "" if ok else f"{len(cmp['unexplained'])} unexplained differences"

    # ------------------------------------------------------------------ (d) licences and weights
    def check_licences(self, commands):
        ev, problems = {}, []
        self.sh("pip-licenses-install", [self.py, "-m", "pip", "install", "-q", "pip-licenses"], commands=commands)
        proc = self.sh("pip-licenses", [str(self.venv / "bin" / "pip-licenses"), "--format=json", "--with-urls"],
                       commands=commands)
        try:
            rows = json.loads(proc.stdout)
        except ValueError:
            rows = []
            problems.append("pip-licenses produced no JSON")
        allowed = _licence_decisions()
        flagged = []
        for row in rows:
            lic = str(row.get("License", ""))
            kind = ("copyleft" if COPYLEFT.search(lic) and not WEAK_COPYLEFT.search(lic) else
                    "weak-copyleft" if WEAK_COPYLEFT.search(lic) else
                    "unknown" if lic.strip().upper() in ("", "UNKNOWN") else None)
            if kind:
                decision = allowed.get(row["Name"].lower())
                flagged.append({"name": row["Name"], "version": row.get("Version"), "licence": lic[:120],
                                "class": kind, "decision": decision or "UNDECIDED"})
                if kind == "copyleft" and not decision:
                    problems.append(f"{row['Name']} ({lic[:60]}) has no documented decision")
        ev["python_packages"] = len(rows)
        ev["flagged"] = flagged
        weights = GATE_ROOT / "tools" / "check_weights.py"
        if weights.exists():
            wproc = self.sh("check-weights", [self.py, str(weights), "--cache", str(self.scratch / "weights_cache")],
                            commands=commands, timeout=7200)
            try:
                items = json.loads(wproc.stdout[wproc.stdout.index("["):])
            except ValueError:
                items = []
            ev["pinned_artifacts"] = items
            if wproc.returncode:
                problems.append(f"check_weights exit {wproc.returncode}")
        else:
            problems.append("tools/check_weights.py missing")
        doc = GATE_ROOT / "docs" / "licences-and-weights.md"
        ev["licence_register"] = str(doc.relative_to(GATE_ROOT)) if doc.exists() else None
        if not doc.exists():
            problems.append("docs/licences-and-weights.md missing")
        ev["problems"] = problems
        return ("fail" if problems else "pass"), ev, "; ".join(problems)

    # ------------------------------------------------------------------ (e) plugin contracts + external adapter
    def check_contracts(self, commands):
        test = GATE_ROOT / "tests" / "test_plugin_contracts.py"
        proc = self.sh("plugin-contracts", [self.py, "-m", "pytest", "-q", "-p", "no:warnings", "-p",
                                            "no:cacheprovider", str(test)], cwd=GATE_ROOT, commands=commands)
        summary = _last_summary(proc.stdout)
        failures = [ln for ln in proc.stdout.splitlines() if ln.startswith(("FAILED", "ERROR", "E  "))][:20]
        return ("pass" if proc.returncode == 0 else "fail"), {"summary": summary, "failures": failures,
                                                               "snapshot": "tests/golden/plugin_contracts.json"}, ""

    def check_plugin_wheel(self, commands):
        wheels = self.scratch / "wheels"
        if wheels.exists():
            shutil.rmtree(wheels)
        example = self.src / "examples" / "plugin_example"
        build = self.sh("plugin-wheel", [self.py, "-m", "pip", "wheel", "--no-deps", "--no-build-isolation", "-w",
                                         str(wheels), str(example)], commands=commands)
        whl = sorted(wheels.glob("*.whl")) if wheels.exists() else []
        if build.returncode or not whl:
            return "fail", {"tail": build.stderr[-1500:]}, "wheel build failed"
        work = self.scratch / "plugin_run"
        if work.exists():
            shutil.rmtree(work)
        work.mkdir(parents=True)
        ev = {"wheel": whl[0].name, "wheel_sha256": hashlib.sha256(whl[0].read_bytes()).hexdigest()}
        steps, problems = {}, []
        try:
            inst = self.sh("plugin-install", [self.py, "-m", "pip", "install", "--no-deps", "--force-reinstall",
                                              str(whl[0])], commands=commands)
            if inst.returncode:
                return "fail", ev, "wheel install failed"
            desc = self.sh("plugin-describe", [self.py, "-m", "geocrop.plugins"], cwd=work, commands=commands)
            described = json.loads(desc.stdout or "{}")
            want = {"estimators": "nearest_centroid", "indices": "NDTI", "segmenters": "grid"}
            for kind, name in want.items():
                src = described.get(kind, {}).get(name, "")
                steps[f"registered:{kind}/{name}"] = src
                if not src.startswith("entry point"):
                    problems.append(f"{kind}/{name} not discovered via entry point ({src or 'missing'})")
            prep = self.sh("plugin-data", [self.py, "-c", (
                "from geocrop.demo import generate; from geocrop.features import build_stack\n"
                "from geocrop.samples import extract\n"
                "d = generate('data'); s = build_stack(d['composites'], 'stack.tif', s2_indices=('NDVI','NDTI'))\n"
                "extract(s, d['training'], max_per_polygon=30).save('samples.npz')\n"
                "import json; print(json.dumps({'parcels': str(d['parcels']), 'raster': str(d['composites']) + '/P3/B08.tif'}))")],
                cwd=work, commands=commands)
            if prep.returncode:
                problems.append(f"feature stack with plugin index NDTI failed: {prep.stderr[-400:]}")
                raise RuntimeError("no data")
            data = json.loads(prep.stdout.strip().splitlines()[-1])
            manifest = json.loads((work / "stack.features.json").read_text())
            names = [f.get("name") for f in manifest.get("features", manifest if isinstance(manifest, list) else [])
                     if isinstance(f, dict)]
            steps["features:NDTI_bands"] = sum(str(n).endswith("_NDTI") for n in names)
            if not steps["features:NDTI_bands"]:
                problems.append("plugin index NDTI produced no feature bands")
            cli = [self.geocrop]
            run = [
                ("train", cli + ["train", "--samples", "samples.npz", "--out", "model", "--folds", "3", "--block-m",
                                 "500", "--params", '{"algorithm": "nearest_centroid"}']),
                ("predict", cli + ["predict", "--features", "stack.tif", "--model", "model/model.joblib", "--out",
                                   "prediction"]),
                ("benchmark-models", cli + ["benchmark-models", "--samples", "samples.npz", "--out", "bm", "--models",
                                            "rf", "nearest_centroid", "--folds", "3", "--repeats", "1", "--block-m",
                                            "500"]),
                ("benchmark-segmentation", cli + ["benchmark-segmentation", "--raster", data["raster"], "--reference",
                                                  data["parcels"], "--out", "seg", "--tile-size", "96",
                                                  "--segments-per-km2", "40", "--methods",
                                                  '{"grid_16": {"segmenter": "grid", "cell_px": 16}}']),
            ]
            for name, cmd in run:
                proc = self.sh(f"plugin-{name}", cmd, cwd=work, commands=commands)
                steps[name] = "ok" if proc.returncode == 0 else f"exit {proc.returncode}: {proc.stderr.strip()[-300:]}"
                if proc.returncode:
                    problems.append(f"{name} failed")
            report = work / "model" / "train_report.json"
            if report.exists():
                rep = json.loads(report.read_text())
                steps["train:algorithm"] = (rep.get("params") or {}).get("algorithm")
                steps["train:cv_macro_f1_synthetic"] = round(rep["cross_validation"]["macro_f1"], 3)
                if steps["train:algorithm"] != "nearest_centroid":
                    problems.append("trained model is not the plugin estimator")
            for table in sorted((work / "bm").glob("*.csv")) + sorted((work / "seg").glob("*.csv")):
                text = table.read_text()
                if "nearest_centroid" in text or "grid_16" in text:
                    steps[f"table:{table.parent.name}/{table.name}"] = [
                        ln[:160] for ln in text.splitlines() if "nearest_centroid" in ln or "grid_16" in ln][:3]
        except RuntimeError:
            pass
        finally:
            self.sh("plugin-uninstall", [self.py, "-m", "pip", "uninstall", "-y", "geocrop-plugin-example"],
                    commands=commands)
        ev["steps"] = steps
        ev["problems"] = problems
        return ("fail" if problems else "pass"), ev, "; ".join(problems) or \
            "external wheel discovered via entry points; SYNTHETIC data"

    # ------------------------------------------------------------------ orchestration
    def run(self) -> dict:
        t0 = time.monotonic()
        self.prepare()
        R = self.record
        if self.args.reuse and self.args.merge and (self.venv / "bin" / "python").exists():
            install = None        # keep the evidence of the original clean installation from the merged report
        else:
            install = R("a.install", "P0", self.check_install)
        if install is not None and install.status == "fail":
            print("[gate] installation failed; remaining checks are recorded as skipped", flush=True)
        ready = self.venv.exists()

        def need_venv(fn):
            return (lambda commands: fn(commands)) if ready else (lambda commands: ("skip", {}, "no venv"))
        R("a.ruff", "P0", need_venv(self.check_ruff))
        R("a.pytest_default", "P0", need_venv(lambda c: self._pytest("pytest-default", [], c)))
        R("a.pytest_slow", "P0", need_venv(lambda c: self._pytest("pytest-slow", ["-m", "slow"], c)))
        R("a.pytest_network", "P0", need_venv(
            lambda c: self._pytest("pytest-network", ["-m", "network"], c) if self.network_ok()
            else ("skip", {}, "network unavailable")))
        R("a.acceptance_real", "P0", need_venv(lambda c: self.check_acceptance(c, offline=False)))
        R("a.acceptance_offline", "P0", need_venv(lambda c: self.check_acceptance(c, offline=True)))
        R("a.recovery_host", "P0", need_venv(self.check_host_recovery))
        R("a.docker_build", "P0", self.check_docker_build)
        R("a.docker_help", "P0", self.check_docker_help)
        R("a.docker_acceptance_offline", "P0", self.check_docker_acceptance)
        R("a.docker_demo", "P0", self.check_docker_demo)
        R("a.docker_recovery", "P0", self.check_docker_recovery)
        R("b.integration", "P1", need_venv(self.check_integration))
        R("c.benchmark_reproducibility", "P1", need_venv(self.check_benchmark))
        R("d.licences_and_weights", "P1", need_venv(self.check_licences))
        R("e.plugin_contracts", "P1", need_venv(self.check_contracts))
        R("e.plugin_wheel", "P1", need_venv(self.check_plugin_wheel))
        env = self.environment()
        env["gate_seconds"] = round(time.monotonic() - t0, 1)
        report = build_report(self.results, self.facts, env)
        report["limitations"] = LIMITATIONS
        return report


LIMITATIONS = [
    "Docker is a clean-machine proxy only: the image reuses its base layers and runs on the gate host's kernel/CPU "
    "architecture. A real fresh-VM run (deploy/vm_acceptance.sh on the target Ubuntu 24.04 machine, e.g. the "
    "Telelink VM provisioned with deploy/setup_ubuntu.sh) is still required before a production release.",
    "The Docker image installs from environment.yml (conda-forge, Python 3.11, unpinned) and not from "
    "constraints/py312.txt; the clean venv is the pinned path.",
    "Network checks depend on public services (Earth Search, Planetary Computer, Zenodo, model hosts); a service "
    "outage is recorded as a failure or skip with the error, not hidden.",
]


# ---------------------------------------------------------------------- pure helpers (unit-tested)

def _last_summary(stdout: str) -> str:
    for line in reversed(stdout.strip().splitlines()):
        if re.search(r"\b(passed|failed|error|errors|skipped|deselected|no tests ran)\b", line):
            return line.strip("= ").strip()
    return ""


def pytest_counts(summary: str) -> dict:
    return {word if word != "errors" else "error": int(n)
            for n, word in re.findall(r"(\d+) (passed|failed|skipped|deselected|errors?|xfailed|xpassed)", summary)}


def _licence_decisions() -> dict:
    """Documented decisions for flagged Python licences: lines `<!-- licence-decision: name: reason -->` in
    docs/licences-and-weights.md."""
    doc = GATE_ROOT / "docs" / "licences-and-weights.md"
    if not doc.exists():
        return {}
    return {m.group(1).strip().lower(): m.group(2).strip()
            for m in re.finditer(r"<!--\s*licence-decision:\s*([^:]+):\s*(.*?)\s*-->", doc.read_text(encoding="utf8"))}


# Keys whose differences are expected between two runs of the same protocol and do not affect results.
VOLATILE = ("started_utc", "finished_utc", "command", "engine", "runtime", "stages", "peak_rss_mb", "wall_s",
            "peak_rss_mb_after", "fit_s", "predict_s", "seconds", "work", "cache", "publish", "time")


def _flatten(obj, prefix=""):
    if isinstance(obj, dict):
        for key, value in obj.items():
            yield from _flatten(value, f"{prefix}/{key}")
    elif isinstance(obj, list) and obj and all(isinstance(v, (dict, list)) for v in obj):
        for i, value in enumerate(obj):
            yield from _flatten(value, f"{prefix}[{i}]")
    else:
        yield prefix, obj


def compare_benchmarks(published: dict, rerun: dict) -> dict:
    """Every difference between a published results.json and a rerun, classified as explained or unexplained.

    Explained: run metadata (times, paths, engine commit, package versions, memory) and models whose status differs
    because an optional backend is (un)available on this machine. Everything else - protocol hash, labels, split,
    sample fingerprint, metrics of models that ran in both - must be identical."""
    a, b = dict(_flatten(published)), dict(_flatten(rerun))
    explained, unexplained = [], []
    model_status = {}
    for name in set(published.get("models", {})) | set(rerun.get("models", {})):
        model_status[name] = (published.get("models", {}).get(name, {}).get("status"),
                              rerun.get("models", {}).get(name, {}).get("status"))
    for key in sorted(set(a) | set(b)):
        va, vb = a.get(key, "<absent>"), b.get(key, "<absent>")
        if va == vb:
            continue
        parts = [p for p in re.split(r"[/\[\]]", key) if p]
        entry = {"key": key, "published": _short(va), "rerun": _short(vb)}
        if any(p in VOLATILE for p in parts):
            entry["why"] = "run metadata (time, throughput, path, versions, memory)"
            explained.append(entry)
        elif len(parts) == 3 and parts[0] == "models" and parts[2] == "reason" and \
                model_status.get(parts[1], (0, 1))[0] == model_status.get(parts[1], (0, 1))[1]:
            entry["why"] = f"same status ({model_status[parts[1]][0]}), different skip message on this machine"
            explained.append(entry)
        elif len(parts) >= 2 and parts[0] == "models" and len(set(model_status.get(parts[1], (0, 0)))) == 2:
            entry["why"] = (f"model {parts[1]} status {model_status[parts[1]][0]} -> {model_status[parts[1]][1]} "
                            "(optional backend availability differs)")
            explained.append(entry)
        else:
            unexplained.append(entry)
    fp = (published.get("samples", {}).get("fingerprint"), rerun.get("samples", {}).get("fingerprint"))
    proto = (published.get("protocol", {}).get("sha256"), rerun.get("protocol", {}).get("sha256"))
    metrics = {}
    for name, (sa, sb) in sorted(model_status.items()):
        if sa == sb == "ok":
            ma, mb = published["models"][name], rerun["models"][name]
            keys = [k for k in ma if k not in VOLATILE and k != "status"]
            metrics[name] = "identical" if all(ma.get(k) == mb.get(k) for k in keys) else "DIFFERENT"
        else:
            metrics[name] = f"status {sa} -> {sb}"
    return {"protocol_sha256": {"published": proto[0], "rerun": proto[1], "identical": proto[0] == proto[1]},
            "sample_fingerprint": {"published": fp[0], "rerun": fp[1], "identical": fp[0] == fp[1]},
            "models": metrics, "explained": explained, "unexplained": unexplained,
            "published_engine": published.get("engine"), "rerun_engine": rerun.get("engine")}


def _short(value):
    text = json.dumps(value, default=str)
    return value if len(text) <= 160 else text[:157] + "..."


def build_report(results: list[CheckResult], facts: dict, env: dict) -> dict:
    counts = {s: sum(r.status == s for r in results) for s in ("pass", "fail", "skip")}
    verdict = "FAIL" if counts["fail"] else ("PASS (with skips)" if counts["skip"] else "PASS")
    return {"schema": SCHEMA, "verdict": verdict, "counts": counts, "commit": facts, "environment": env,
            "checks": [asdict(r) for r in results]}


def redact(text: str, scratch: Path) -> str:
    """Published copy: machine-specific scratch and home paths replaced by placeholders."""
    for path in {str(Path(scratch).resolve()), str(scratch)}:
        text = text.replace(path, "<scratch>")
    text = text.replace(str(GATE_ROOT), "<gate>").replace(str(Path.home()), "~")
    return re.sub(r"(/private)?/(tmp|var/folders)/[^\s\"'`|]*", "<tmp-path>", text)   # truncated leftovers


def merge_reports(previous: dict, current: dict) -> dict:
    """`current` plus the checks of `previous` (same commit) that were not rerun; check order is preserved."""
    if previous.get("commit", {}).get("commit") != current.get("commit", {}).get("commit"):
        raise SystemExit("--merge: the previous report is for a different commit")
    rerun = {c["id"]: c for c in current["checks"] if not c["notes"].startswith("skipped on request")}
    checks = [rerun.pop(c["id"], c) for c in previous["checks"]] + list(rerun.values())
    results = [CheckResult(**c) for c in checks]
    merged = build_report(results, current["commit"], current["environment"])
    merged["environment"]["merged_from_earlier_run"] = previous.get("environment", {}).get("utc")
    merged["limitations"] = current.get("limitations", previous.get("limitations"))
    return merged


def _md_value(value) -> str:
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    return text.replace("|", "\\|").replace("\n", " ")


def write_report(report: dict, out: Path) -> dict:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    jpath, mpath = out / "release_gate_report.json", out / "release_gate_report.md"
    jpath.write_text(json.dumps(report, indent=2, default=str), encoding="utf8")
    c = report.get("commit", {})
    lines = [f"# Release gate: {c.get('short', c.get('commit', '?'))}", "",
             f"**Verdict: {report['verdict']}** - {report['counts']['pass']} pass, {report['counts']['fail']} fail, "
             f"{report['counts']['skip']} skip.", "",
             "Generated by `tools/release_gate.py` (docs/release-gate.md). Every value below comes from the run "
             "recorded here; synthetic-data steps are software checks, not accuracy.", "",
             "## Commit", ""]
    lines += [f"- {k}: `{_md_value(v)}`" for k, v in c.items()]
    lines += ["", "## Environment", ""]
    lines += [f"- {k}: `{_md_value(v)}`" for k, v in report.get("environment", {}).items()]
    lines += ["", "## Checks", "", "| Check | Priority | Result | Seconds | Notes |", "|---|---|---|---|---|"]
    for r in report["checks"]:
        lines.append(f"| {r['id']} | {r['priority']} | {r['status'].upper()} | {r['seconds']} | "
                     f"{_md_value(r['notes'])} |")
    lines += ["", "## Evidence", ""]
    for r in report["checks"]:
        lines += [f"### {r['id']} - {r['status'].upper()}", ""]
        if r["notes"]:
            lines += [r["notes"], ""]
        for cmd in r.get("commands", []):
            lines.append(f"- `{_md_value(cmd['cmd'])[:300]}` -> exit {cmd['exit']} ({cmd['seconds']} s)")
        if r.get("commands"):
            lines.append("")
        if r["evidence"]:
            lines += ["```json", json.dumps(r["evidence"], indent=1, default=str)[:6000], "```", ""]
    if report.get("limitations"):
        lines += ["## Limitations", ""] + [f"- {x}" for x in report["limitations"]] + [""]
    mpath.write_text("\n".join(lines), encoding="utf8")
    return {"json": str(jpath), "md": str(mpath)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--commit", default="HEAD", help="commit to verify (default HEAD of --repo)")
    ap.add_argument("--repo", default=str(GATE_ROOT), help="git repository holding the commit")
    ap.add_argument("--scratch", required=True, help="work directory (clone, venv, outputs, logs)")
    ap.add_argument("--python", default=shutil.which("python3.12") or sys.executable,
                    help="base interpreter for the clean venv (constraints are for CPython 3.12)")
    ap.add_argument("--out", help="report directory (default: --scratch)")
    ap.add_argument("--only", nargs="*", default=[], help="check ids or prefixes to run (a.install always runs)")
    ap.add_argument("--skip", nargs="*", default=[], help="check id prefixes to record as skipped")
    ap.add_argument("--no-network", action="store_true")
    ap.add_argument("--no-docker", action="store_true")
    ap.add_argument("--docker-no-cache", action="store_true", help="docker build --no-cache (slower, cleaner)")
    ap.add_argument("--reuse", action="store_true", help="reuse an existing clone/venv in --scratch (development)")
    ap.add_argument("--merge", action="store_true",
                    help="keep earlier results from --out/release_gate_report.json for checks not run now "
                         "(rerun single checks of the same commit)")
    ap.add_argument("--publish", help="directory to copy the Markdown report to as release-gate-<short>.md")
    args = ap.parse_args(argv)
    if args.only:
        args.only = ["a.install", *args.only]
    gate = Gate(args)
    report = gate.run()
    out = Path(args.out or gate.scratch)
    previous = out / "release_gate_report.json"
    if args.merge and not previous.exists() and (out / "release_gate_report.partial.json").exists():
        previous = out / "release_gate_report.partial.json"
    if args.merge and previous.exists():
        report = merge_reports(json.loads(previous.read_text()), report)
    paths = write_report(report, out)
    if args.publish:
        dest = Path(args.publish) / f"release-gate-{report['commit']['short']}.md"
        dest.parent.mkdir(parents=True, exist_ok=True)
        public = json.loads(redact(json.dumps(report, default=str), gate.scratch))   # before any truncation
        tmp = gate.scratch / "publish"
        shutil.copyfile(write_report(public, tmp)["md"], dest)
        paths["published"] = str(dest)
    print(json.dumps({"verdict": report["verdict"], "counts": report["counts"], **paths}, indent=2))
    return 1 if report["verdict"] == "FAIL" else 0


if __name__ == "__main__":
    sys.exit(main())
