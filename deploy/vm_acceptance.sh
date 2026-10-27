#!/usr/bin/env bash
# GeoCrop Engine - release acceptance on a FRESH Ubuntu 26.04 (or 24.04) VM (the real "clean machine" test).
#
# Runs the same release gate as tools/release_gate.py does on a workstation, but on the target machine, so that
# the installation, the test tiers, the real Sentinel-1/2 acceptance run, the demo, crash recovery, the combined
# integration study, the public benchmark rerun and the licence/weights audit are verified on the operating
# system, CPU and network that production will use. Docker is not needed (the VM itself is the clean machine).
#
#   git clone https://github.com/ZeljkoDzafic/geocrop-engine.git && cd geocrop-engine
#   sudo bash deploy/vm_acceptance.sh --system          # once: apt packages (git, build tools; Python 3.12 via apt or uv)
#   bash deploy/vm_acceptance.sh                        # as the run user; verifies HEAD
#   COMMIT=v0.9.0 bash deploy/vm_acceptance.sh          # verify a tag or commit
#   SKIP="c. d." bash deploy/vm_acceptance.sh           # skip slow/network-heavy checks by id prefix
#   CONDA=1 bash deploy/vm_acceptance.sh                # additionally exercise deploy/setup_ubuntu.sh --user
#
# Output: $SCRATCH/gate/release_gate_report.{json,md}; exit status 1 if any check fails.
set -euo pipefail

REPO_DIR=$(cd "$(dirname "$0")/.." && pwd)
SCRATCH=${GEOCROP_SCRATCH:-/scratch/geocrop}
COMMIT=${COMMIT:-HEAD}
PYTHON=${PYTHON:-python3.12}

if [[ "${1:-}" == "--system" ]]; then
  apt-get update
  # libexpat1: needed by the GDAL inside the rasterio wheel (uv's standalone Python does not pull it in; today
  # only git's dependencies do); libgomp1: OpenMP runtime of the LightGBM wheel.
  apt-get install -y --no-install-recommends git curl ca-certificates build-essential libgomp1 libexpat1
  # Ubuntu 24.04 ships Python 3.12; Ubuntu 26.04 ships a newer default, so 3.12 (the pinned constraints'
  # interpreter) comes from uv's standalone builds when the distribution has no python3.12 package.
  if apt-cache show python3.12-venv >/dev/null 2>&1; then
    apt-get install -y --no-install-recommends python3.12 python3.12-venv python3.12-dev
  else
    curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh
    UV_PYTHON_INSTALL_DIR=/opt/uv-python uv python install 3.12
    ln -sf "$(UV_PYTHON_INSTALL_DIR=/opt/uv-python uv python find 3.12)" /usr/local/bin/python3.12
  fi
  mkdir -p "$SCRATCH" && chmod 1777 "$SCRATCH"
  echo "System packages installed. Next, as the run user: bash deploy/vm_acceptance.sh"
  exit 0
fi

command -v "$PYTHON" >/dev/null || { echo "$PYTHON not found; run: sudo bash deploy/vm_acceptance.sh --system" >&2; exit 2; }
mkdir -p "$SCRATCH"
echo "== host: $(uname -a)"
echo "== python: $($PYTHON --version)  cpus: $(nproc)  memory: $(free -g | awk '/Mem:/{print $2}') GB"
echo "== disk free in $SCRATCH: $(df -h "$SCRATCH" | awk 'NR==2{print $4}')"

if [[ "${CONDA:-0}" == "1" ]]; then
  # The documented VM installation path (Miniforge + environment.yml); its own test run is part of the script.
  bash "$REPO_DIR/deploy/setup_ubuntu.sh" --user
fi

status=0
# shellcheck disable=SC2086
"$PYTHON" "$REPO_DIR/tools/release_gate.py" --repo "$REPO_DIR" --commit "$COMMIT" --scratch "$SCRATCH/gate" \
  --python "$PYTHON" --no-docker ${SKIP:+--skip $SKIP} || status=$?
echo "Report: $SCRATCH/gate/release_gate_report.md"
exit $status
