#!/usr/bin/env bash
# GeoCrop Engine — one-time setup of an Ubuntu 26.04/24.04 research VM (no GPU needed).
#   sudo bash deploy/setup_ubuntu.sh            # system packages, directories
#   bash deploy/setup_ubuntu.sh --user          # Miniforge, conda env, engine (as the run user)
#   SNAP_VERSION=11.0 bash deploy/setup_ubuntu.sh --snap   # optional ESA SNAP (Sentinel-1)
set -euo pipefail

DATA=${GEOCROP_DATA:-/data/geocrop}
SCRATCH=${GEOCROP_SCRATCH:-/scratch/geocrop}
REPO_DIR=$(cd "$(dirname "$0")/.." && pwd)

if [[ "${1:-}" == "--user" ]]; then
  if [[ ! -d "$HOME/miniforge3" ]]; then
    curl -fsSL -o /tmp/miniforge.sh \
      "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-$(uname -m).sh"
    bash /tmp/miniforge.sh -b -p "$HOME/miniforge3"
  fi
  # shellcheck disable=SC1091
  source "$HOME/miniforge3/etc/profile.d/conda.sh"
  conda env update -n geocrop -f "$REPO_DIR/environment.yml" || conda env create -n geocrop -f "$REPO_DIR/environment.yml"
  conda activate geocrop
  pip install -e "$REPO_DIR[all,stac,web]"
  geocrop --version
  python -m pytest -q "$REPO_DIR/tests" -x
  echo "Engine ready. Activate with: source ~/miniforge3/etc/profile.d/conda.sh && conda activate geocrop"
  exit 0
fi

if [[ "${1:-}" == "--snap" ]]; then
  # Verify the installer name on https://step.esa.int/main/download/snap-download/ for the chosen version.
  SNAP_VERSION=${SNAP_VERSION:-11.0}
  INSTALLER="esa-snap_all_linux-${SNAP_VERSION}.0.sh"
  curl -fSL -o "/tmp/$INSTALLER" "https://download.esa.int/step/snap/${SNAP_VERSION}/installers/${INSTALLER}"
  bash "/tmp/$INSTALLER" -q -dir "$HOME/esa-snap"
  # Java heap and temporary files on the fast scratch disk
  mkdir -p "$SCRATCH/snap-tmp"
  sed -i "s/^-Xmx.*/-Xmx${SNAP_XMX:-48G}/" "$HOME/esa-snap/bin/gpt.vmoptions" || true
  echo "-Djava.io.tmpdir=$SCRATCH/snap-tmp" >> "$HOME/esa-snap/bin/gpt.vmoptions"
  "$HOME/esa-snap/bin/gpt" -h | head -3
  echo "SNAP gpt: $HOME/esa-snap/bin/gpt"
  exit 0
fi

# ---- system part (root) ----
apt-get update
apt-get install -y --no-install-recommends git tmux unzip curl ca-certificates htop iotop
mkdir -p "$DATA"/{raw,interim,processed,reference,outputs} "$SCRATCH"
chmod 750 "$DATA" "$SCRATCH"
echo "Directories: $DATA (archive), $SCRATCH (fast local disk for SNAP and temporary files)"
echo "Next, as the run user: bash deploy/setup_ubuntu.sh --user"
