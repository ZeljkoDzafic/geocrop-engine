"""Crash-recovery scenario for the release gate (host, Docker image or a fresh VM).

    python tools/recovery_check.py --work /tmp/geocrop-recovery [--point train] [--json result.json]

1. Generates the SYNTHETIC demo study and a config for `geocrop run`.
2. Starts `geocrop run` with the inert fault hook (`geocrop.faultinject`) armed at ``--point``, waits until the
   process reports it is inside that stage (partial outputs on disk), and SIGKILLs it.
3. Reruns `geocrop run` and verifies: exit code 0, every stage ``success`` in stages.json, the recorded output
   hashes equal the files on disk, no stale lock or ``*.tmp`` files remain, and the prediction carries the
   hash of the model on disk.

Only the standard library, PyYAML and geocrop are needed; exit status 0 = recovered, 1 = not recovered.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

STAGES = ("features", "samples", "train", "predict")


def _config(root: Path) -> None:
    import yaml

    cfg = {"seed": 11,
           "windows": {"P1": {"start": "2023-04-01", "end": "2023-05-15"},
                       "P2": {"start": "2023-06-01", "end": "2023-06-30"},
                       "P3": {"start": "2023-07-15", "end": "2023-08-15"},
                       "P4": {"start": "2023-09-01", "end": "2023-09-30"}},
           "paths": {"composites": "data/composites", "features": "run/features/stack.tif",
                     "training": "data/reference/training.gpkg", "parcels": "data/reference/parcels.gpkg",
                     "observations": "data/reference/observations.gpkg", "output": "run"},
           "model": {"n_estimators": 15, "n_jobs": 1, "cv_folds": 3, "spatial_block_m": 500},
           "validation": {"bootstrap": 20, "year": 2023}}
    (root / "study.yaml").write_text(yaml.safe_dump(cfg, sort_keys=True), encoding="utf8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _env(**extra) -> dict:
    env = {k: v for k, v in os.environ.items() if not k.startswith("GEOCROP_FAULT_")}
    env.update(extra)
    return env


def run(work: Path, point: str = "train", timeout: float = 600) -> dict:
    from geocrop import faultinject
    from geocrop.demo import generate

    work.mkdir(parents=True, exist_ok=True)
    generate(work / "data")
    _config(work)
    cmd = [sys.executable, "-m", "geocrop.cli", "run", "--config", "study.yaml", "--stages", *STAGES]
    sig = work / f"{point}.signal"
    sig.unlink(missing_ok=True)
    t0 = time.monotonic()
    with (work / "killed.log").open("w") as log:
        proc = subprocess.Popen(cmd, cwd=work, stdout=log, stderr=subprocess.STDOUT,
                                env=_env(**{faultinject.PAUSE_ENV: point, faultinject.SIGNAL_ENV: str(sig)}))
        try:
            while not sig.exists():
                if proc.poll() is not None:
                    raise RuntimeError(f"run ended (exit {proc.returncode}) before reaching fault point {point}")
                if time.monotonic() - t0 > timeout:
                    raise RuntimeError(f"run never reached fault point {point}")
                time.sleep(0.05)
        finally:
            proc.send_signal(signal.SIGKILL)
            proc.wait(timeout=60)
    run_dir = work / "run"
    killed = {"pid": proc.pid, "returncode": proc.returncode,
              "lock_left": (run_dir / ".geocrop.lock").exists(),
              "stage_status": {k: v.get("status") for k, v in
                               json.loads((run_dir / "stages.json").read_text()).items()}
              if (run_dir / "stages.json").exists() else {}}
    rerun = subprocess.run(cmd, cwd=work, env=_env(), capture_output=True, text=True, timeout=timeout)
    (work / "rerun.log").write_text(rerun.stdout + rerun.stderr, encoding="utf8")
    problems: list[str] = []
    if rerun.returncode:
        problems.append(f"rerun exit code {rerun.returncode}: {rerun.stderr[-1500:]}")
    state = json.loads((run_dir / "stages.json").read_text()) if (run_dir / "stages.json").exists() else {}
    for name in STAGES:
        entry = state.get(name, {})
        if entry.get("status") != "success":
            problems.append(f"stage {name} status {entry.get('status')!r}")
        for path, identity in entry.get("outputs", {}).items():
            p = Path(path) if Path(path).is_absolute() else work / path
            if not p.is_file() or identity.get("sha256") != _sha(p):
                problems.append(f"{path} differs from its recorded identity")
    if (run_dir / ".geocrop.lock").exists():
        problems.append("stale lock left after rerun")
    tmp = [str(p) for p in run_dir.rglob("*.tmp")]
    if tmp:
        problems.append(f"temporary files left: {tmp[:5]}")
    manifest = run_dir / "prediction" / "prediction_manifest.json"
    if not manifest.is_file():
        problems.append("prediction manifest missing")
    elif json.loads(manifest.read_text()).get("model_sha256") != _sha(run_dir / "model" / "model.joblib"):
        problems.append("prediction manifest does not reference the model on disk")
    return {"point": point, "killed": killed,
            "rerun_stage_lines": [ln for ln in rerun.stdout.splitlines() if ln.startswith("[")],
            "problems": problems, "recovered": not problems, "seconds": round(time.monotonic() - t0, 1)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--work", required=True, type=Path)
    ap.add_argument("--point", default="train", choices=("features", "train", "predict"))
    ap.add_argument("--json", type=Path)
    args = ap.parse_args(argv)
    try:
        result = run(args.work, args.point)
    except Exception as exc:  # report, never hide
        result = {"point": args.point, "recovered": False, "problems": [f"{type(exc).__name__}: {exc}"]}
    text = json.dumps(result, indent=2)
    print(text)
    if args.json:
        args.json.write_text(text, encoding="utf8")
    return 0 if result["recovered"] else 1


if __name__ == "__main__":
    sys.exit(main())
