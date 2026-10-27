#!/usr/bin/env bash
# GeoCrop Engine - tools/scale_benchmark.py on Ubuntu inside a resource-limited container.
#
# Builds an image from a pristine Ubuntu base with the documented pip installation (Python 3.12 from apt or uv,
# `pip install -c constraints/py312.txt -e '.[all,stac,web]'`, plus psutil for the measurement tool), then runs
#   python tools/scale_benchmark.py run --sizes <sizes>
# with explicit `docker run --cpus/--memory` limits and the output on a Docker volume (the container's Linux
# file system, not a bind mount of the host). benchmark.json files, runs.json and the report tables are copied to
# --out. The numbers describe that container on that host: state the limits and the host with any figure quoted.
#
#   tools/linux_benchmark.sh --sizes 100 --cpus 2 --memory 4g --out outputs/scale-linux
#   tools/linux_benchmark.sh --sizes 100 500 --max-hours 2 --image ubuntu:24.04 --out outputs/scale-linux-2404
set -euo pipefail

REPO_DIR=$(cd "$(dirname "$0")/.." && pwd)
IMAGE=ubuntu:26.04
REF=HEAD
CPUS=2
MEMORY=4g
SIZES=(100)
MAX_HOURS=2
OUT=""
VOLUME=geocrop-linux-benchmark

while [[ $# -gt 0 ]]; do
  case "$1" in
    --image) IMAGE=$2; shift 2 ;;
    --ref) REF=$2; shift 2 ;;
    --cpus) CPUS=$2; shift 2 ;;
    --memory) MEMORY=$2; shift 2 ;;
    --max-hours) MAX_HOURS=$2; shift 2 ;;
    --volume) VOLUME=$2; shift 2 ;;
    --out) OUT=$2; shift 2 ;;
    --sizes) shift; SIZES=(); while [[ $# -gt 0 && $1 != --* ]]; do SIZES+=("$1"); shift; done ;;
    -h|--help) sed -n '2,13p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done
[[ -n $OUT ]] || { echo "--out is required" >&2; exit 2; }
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)

COMMIT=$(git -C "$REPO_DIR" rev-parse "$REF")
TAG="geocrop-linux-benchmark:${COMMIT:0:12}-${IMAGE//[:\/]/-}"
ctx=$(mktemp -d "${TMPDIR:-/tmp}/geocrop-bench-ctx.XXXXXX")
trap 'rm -rf "$ctx"' EXIT
git -C "$REPO_DIR" archive --format=tar "$COMMIT" | tar -x -C "$ctx"
cat > "$ctx/Dockerfile.bench" <<EOF
FROM $IMAGE
ENV DEBIAN_FRONTEND=noninteractive PIP_DISABLE_PIP_VERSION_CHECK=1 PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends git curl ca-certificates libgomp1 libexpat1 \\
    && if apt-cache show python3.12-venv >/dev/null 2>&1; then \\
         apt-get install -y --no-install-recommends python3.12 python3.12-venv; \\
       else \\
         curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh \\
         && UV_PYTHON_INSTALL_DIR=/opt/uv-python uv python install 3.12 \\
         && ln -sf "\$(UV_PYTHON_INSTALL_DIR=/opt/uv-python uv python find 3.12)" /usr/local/bin/python3.12; \\
       fi && rm -rf /var/lib/apt/lists/*
COPY . /opt/geocrop-engine
WORKDIR /opt/geocrop-engine
RUN python3.12 -m venv /opt/venv && /opt/venv/bin/python -m pip install -q -c constraints/py312.txt -e '.[all,stac,web]' \\
    && /opt/venv/bin/python -m pip install -q psutil && /opt/venv/bin/geocrop --version
ENV PATH=/opt/venv/bin:\$PATH
EOF
echo "== building $TAG from $IMAGE at $COMMIT"
docker build -q -t "$TAG" -f "$ctx/Dockerfile.bench" "$ctx"

docker volume create "$VOLUME" >/dev/null
echo "== running sizes ${SIZES[*]} km2 with --cpus $CPUS --memory $MEMORY --memory-swap $MEMORY (no swap), volume $VOLUME"
status=0
docker run --rm --cpus "$CPUS" --memory "$MEMORY" --memory-swap "$MEMORY" -v "$VOLUME:/bench" -v "$OUT:/export" "$TAG" \
  bash -c "set -e
    echo \"== \$(. /etc/os-release; echo \$PRETTY_NAME) \$(uname -m) kernel \$(uname -r); nproc \$(nproc);\" \
         \"cgroup cpu.max \$(cat /sys/fs/cgroup/cpu.max 2>/dev/null); memory.max \$(cat /sys/fs/cgroup/memory.max 2>/dev/null)\"
    df -h /bench | tail -1
    python tools/scale_benchmark.py run --out /bench/scale --sizes ${SIZES[*]} --max-hours $MAX_HOURS || true
    python tools/scale_benchmark.py report --out /bench/scale --images /bench/images || true
    cd /bench/scale && cp runs.json scale_tables.md scale_summary.json /export/ 2>/dev/null || true
    for d in area_*km2; do mkdir -p /export/\$d; cp \$d/benchmark.json /export/\$d/ 2>/dev/null || true; done
    du -sh /bench/scale/area_*km2" || status=$?
echo "== results in $OUT (data stay in docker volume $VOLUME; remove with: docker volume rm $VOLUME)"
exit $status
