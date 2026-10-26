# Release gate

`tools/release_gate.py` **verifies** a commit before it is released: it installs the commit into a fresh virtual
environment and runs every test tier, the real-data acceptance run, a Docker clean-machine proxy, a combined
integration study, a rerun of the public benchmark, a licence and model-weights audit and the plugin-contract
checks. It builds nothing that is shipped and changes nothing in the repository. It runs locally (GitHub
Actions is not required) and can be pointed at any later commit or tag.

```bash
python3.12 tools/release_gate.py --commit HEAD --scratch /tmp/geocrop-gate --docker-no-cache \
    --publish docs/release-gates                 # also copies the Markdown report as release-gate-<sha>.md
```

| Option | Meaning |
|---|---|
| `--commit` | commit, tag or branch to verify (default `HEAD`); cloned into `<scratch>/src` from `--repo` |
| `--scratch` | work directory: clone, venv, outputs, caches and one log per command under `logs/` |
| `--python` | base interpreter for the clean venv (default `python3.12`; `constraints/py312.txt` targets CPython 3.12) |
| `--only` / `--skip` | check ids or prefixes (`a.`, `b.integration`, `e.` ...); skipped checks are reported as `skip` |
| `--merge` | rerun a few checks of the same commit and merge them into the existing report |
| `--no-network`, `--no-docker` | record the network/Docker checks as skipped instead of attempting them |
| `--docker-no-cache` | build the image without the layer cache (slower, closer to a clean machine) |
| `--reuse` | keep an existing clone/venv (development of the gate only; a release run starts clean) |

Outputs: `<out>/release_gate_report.json` (schema `geocrop-release-gate/1`) and `release_gate_report.md` with the
commit, the environment (platform, CPU, memory, Docker, hash of `pip freeze`, versions of key packages), and for
every check its result (`pass`, `fail` or `skip`), duration, the commands with exit codes and the evidence. The exit
status is 1 if any check fails. A skip is never counted as a pass: the verdict is then `PASS (with skips)`.

The code under test is always the pristine clone. The gate's helpers (`tools/gate_integration.py`,
`tools/recovery_check.py`, `tools/check_weights.py`) and `tests/test_plugin_contracts.py` are taken from the
tree that runs the gate, so a newer gate can verify an older commit.

## Checks

| Id | Priority | What is verified |
|---|---|---|
| `a.install` | P0 | fresh `python -m venv`; `pip install -c constraints/py312.txt -e .[<every extra declared in pyproject.toml>]`; every optional backend imports (an installed but unloadable backend, e.g. LightGBM without `libomp`, is recorded) |
| `a.ruff` | P0 | `ruff check src tests` |
| `a.pytest_default` | P0 | full default suite (`pytest`, everything except `network`) |
| `a.pytest_slow` | P0 | `pytest -m slow` |
| `a.pytest_network` | P0 | `pytest -m network` (real Sentinel data) |
| `a.acceptance_real` | P0 | `geocrop acceptance --out ...`: real Sentinel-2 (Earth Search) + Sentinel-1 RTC (Planetary Computer), Semberija 2023, verdict must be PASS |
| `a.acceptance_offline` | P0 | `geocrop acceptance --offline` |
| `a.recovery_host` | P0 | `tools/recovery_check.py`: SIGKILL inside the `predict` stage of a real `geocrop run`, rerun, verify completion, recorded output hashes, no stale lock/temporary files |
| `a.docker_build` ... `a.docker_recovery` | P0 | `docker build` of the commit, then inside the image: `geocrop --help`, offline acceptance, `geocrop demo --quick`, and the recovery scenario (SIGKILL inside `train`) |
| `b.integration` | P1 | `tools/gate_integration.py`: one study combining the synthetic demo, Presto embeddings from local monthly composites (and AlphaEarth on the real acceptance area when online), LightTAE + RF + CatBoost in `benchmark-models` on the same folds, an active-learning select → ingest → retrain round, transfer CV with the region strategy, the FTW segmenter inside `benchmark-segmentation`, S1 markers + traffic-light rules, `preflight` and `clean` |
| `c.benchmark_reproducibility` | P1 | `geocrop benchmark-public` with the committed protocol in a fresh work and cache directory; every difference to `docs/benchmarks/*.json` is listed and classified (run metadata and optional-backend availability are explained; protocol hash, labels, split, sample fingerprint and metrics must be identical) |
| `d.licences_and_weights` | P1 | `tools/check_weights.py` downloads every pinned checkpoint/dataset and compares its checksum; `pip-licenses` on the clean venv, copyleft/unknown licences must carry a documented decision in [licences-and-weights.md](licences-and-weights.md) (`<!-- licence-decision: name: reason -->`) |
| `e.plugin_contracts` | P1 | `tests/test_plugin_contracts.py` against the frozen snapshot `tests/golden/plugin_contracts.json` |
| `e.plugin_wheel` | P1 | `examples/plugin_example` built as a wheel, installed into the clean venv and used only through its entry points: feature stack with the plugin index, `train` + `predict` with the plugin estimator, `benchmark-models` and `benchmark-segmentation` with the plugin segmenter; uninstalled afterwards |

## Plugin contract snapshot

`tests/test_plugin_contracts.py` freezes what external adapters rely on: the registry names and entry-point
groups, built-in names (a plugin cannot shadow them), the `Registry` API, the `Segmenter` protocol, the `Index`
record, the exporter/validator signatures and the versions of the JSON file contracts. Any change, including an
addition, fails the default test suite. To accept a deliberate change:

```bash
GEOCROP_UPDATE_CONTRACTS=1 python -m pytest tests/test_plugin_contracts.py   # bumps "version" in the snapshot
```

then add `plugin-contracts/<new version>` with a short migration note to `CHANGELOG.md` (checked by the test).

## Clean machine: Docker proxy and the real VM

The Docker checks are a **proxy** for a clean machine: the image starts from `mambaorg/micromamba` and installs
the engine from `environment.yml`, so they prove that a machine with nothing but Docker can install and run the
engine and recover from a crash. They do not replace a real VM: the container shares the host kernel and CPU
architecture (arm64 on a Mac workstation), resolves unpinned conda-forge packages (Python 3.11) instead of
`constraints/py312.txt`, and does not exercise `deploy/setup_ubuntu.sh`, systemd units or the target network.

On the target Ubuntu 24.04 VM (for example the Telelink research VM provisioned with `deploy/setup_ubuntu.sh`):

```bash
sudo bash deploy/vm_acceptance.sh --system     # python3.12-venv, git, build tools
bash deploy/vm_acceptance.sh                   # the same gate on the VM, Docker checks skipped
CONDA=1 bash deploy/vm_acceptance.sh           # additionally exercise setup_ubuntu.sh --user (Miniforge path)
```

## Published gate reports

Reports of gate runs are committed as Markdown only under [release-gates/](release-gates/) (no data, no logs).
