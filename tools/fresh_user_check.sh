#!/usr/bin/env bash
# GeoCrop Engine - "independent user" installation check in a pristine Ubuntu container.
#
# Starts a fresh `ubuntu:26.04` (or any) container with no GeoCrop-specific preparation, creates an ordinary user
# with sudo (what a new VM or workstation gives you) and, as that user, follows the documented installation and
# quick start LITERALLY, one numbered step per documented command block:
#
#   README.md "Install" (Ubuntu commands)    apt prerequisites, Python 3.12 (apt or uv), clone, venv, pip -c
#   README.md "Five-minute quick start"      geocrop demo, geocrop acceptance --offline, a small real
#                                            geocrop quicklook over Semberija (Earth Search), geocrop serve
#                                            with a generated token + curl /api/v1/health (401 without token)
#
# If a step here has to differ from the documents, the documents are wrong: fix them, not this script.
# Requires Docker and network access (apt, PyPI, GitHub or the local repository, Earth Search STAC/COGs).
#
#   tools/fresh_user_check.sh                                  # local HEAD (git archive), ubuntu:26.04
#   tools/fresh_user_check.sh --ref v1.0.0rc1 --clone             # a tag or branch cloned from GitHub
#   tools/fresh_user_check.sh --image ubuntu:24.04 --cpus 2 --memory 4g --out /tmp/fresh
#   tools/fresh_user_check.sh --skip quicklook                 # skip steps by name (offline machines)
#
# Output: <out>/transcript.log (every command and its output), <out>/steps.tsv (step, exit status, seconds) and
# <out>/summary.md. Exit status 0 only if every step passed.
set -euo pipefail

REPO_DIR=$(cd "$(dirname "$0")/.." && pwd)
IMAGE=ubuntu:26.04
REF=HEAD
CLONE=0
REPO_URL=https://github.com/ZeljkoDzafic/geocrop-engine.git
CPUS=2
MEMORY=4g
OUT=""
SKIP=""
PLATFORM=""
CA_BUNDLE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --image) IMAGE=$2; shift 2 ;;
    --ref) REF=$2; shift 2 ;;
    --clone) CLONE=1; shift ;;
    --repo-url) REPO_URL=$2; CLONE=1; shift 2 ;;
    --cpus) CPUS=$2; shift 2 ;;
    --memory) MEMORY=$2; shift 2 ;;
    --platform) PLATFORM=$2; shift 2 ;;
    --ca-bundle) CA_BUNDLE=$2; shift 2 ;;
    --out) OUT=$2; shift 2 ;;
    --skip) SKIP="$SKIP $2"; shift 2 ;;
    -h|--help) sed -n '2,24p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

command -v docker >/dev/null || { echo "docker is required" >&2; exit 2; }
OUT=${OUT:-$(mktemp -d "${TMPDIR:-/tmp}/geocrop-fresh-user.XXXXXX")}
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)
: > "$OUT/transcript.log"

if [[ $CLONE == 1 ]]; then
  SOURCE="clone $REPO_URL @ $REF"
else
  # The committed tree only (no untracked or ignored files): what a clone of this commit contains.
  COMMIT=$(git -C "$REPO_DIR" rev-parse "$REF")
  git -C "$REPO_DIR" archive --format=tar --prefix=geocrop-engine/ "$COMMIT" > "$OUT/source.tar"
  SOURCE="git archive of local commit $COMMIT"
fi
echo "== image $IMAGE, --cpus $CPUS --memory $MEMORY, source: $SOURCE, output: $OUT"

docker_args=(run --rm -i --cpus "$CPUS" --memory "$MEMORY" -v "$OUT:/report" -e "SKIP=$SKIP" -e "REF=$REF"
             -e "CLONE=$CLONE" -e "REPO_URL=$REPO_URL" -e "SOURCE=$SOURCE" -e "IMAGE=$IMAGE")
[[ -n $PLATFORM ]] && docker_args+=(--platform "$PLATFORM")
if [[ -n $CA_BUNDLE ]]; then
  [[ -f $CA_BUNDLE ]] || { echo "CA bundle does not exist" >&2; exit 2; }
  docker_args+=(-v "$CA_BUNDLE:/run/geocrop-ca.pem:ro")
  for name in SSL_CERT_FILE PIP_CERT REQUESTS_CA_BUNDLE CURL_CA_BUNDLE GIT_SSL_CAINFO; do
    docker_args+=(-e "$name=/run/geocrop-ca.pem")
  done
fi

status=0
docker "${docker_args[@]}" "$IMAGE" bash -s <<'CONTAINER' || status=$?
set -uo pipefail
export DEBIAN_FRONTEND=noninteractive
report_owner=$(stat -c '%u:%g' /report)
trap 'chown -R "$report_owner" /report' EXIT
# --- container stand-in for "a fresh Ubuntu machine with a sudo user" (not a documented step) -------------------
{ apt-get update -qq && apt-get install -y -qq --no-install-recommends sudo >/dev/null; } >/report/transcript.log 2>&1
useradd -m -s /bin/bash alice && echo 'alice ALL=(ALL) NOPASSWD:ALL' > /etc/sudoers.d/alice
echo 'Defaults env_keep += "HTTP_PROXY HTTPS_PROXY NO_PROXY http_proxy https_proxy no_proxy SSL_CERT_FILE PIP_CERT REQUESTS_CA_BUNDLE CURL_CA_BUNDLE GIT_SSL_CAINFO"' >> /etc/sudoers.d/alice
[[ -f /report/source.tar ]] && cp /report/source.tar /home/alice/ && chown alice: /home/alice/source.tar
chown alice: /report /report/transcript.log
chmod 755 /report
chmod 644 /report/transcript.log

# --- the documented steps, run as alice in a login shell ---------------------------------------------------------
cat > /home/alice/steps.sh <<'STEPS'
set -uo pipefail
cd "$HOME"
LOG=/report/transcript.log
printf 'step\tstatus\tseconds\n' > /report/steps.tsv
failed=0
step() {  # step <name> <commands...>: one documented command block, run in this shell (keeps cd/env/activate)
  local name=$1; shift
  if [[ " $SKIP " == *" $name "* ]]; then printf '%s\tskipped\t0\n' "$name" >> /report/steps.tsv; return; fi
  if [[ $failed == 1 ]]; then printf '%s\tnot-run\t0\n' "$name" >> /report/steps.tsv; return; fi
  echo "### [$name] $*" | tee -a "$LOG"
  local t0=$SECONDS rc=0
  eval "$*" >> "$LOG" 2>&1 || rc=$?
  printf '%s\t%s\t%s\n' "$name" "$rc" "$((SECONDS - t0))" >> /report/steps.tsv
  echo "### [$name] exit $rc after $((SECONDS - t0)) s" | tee -a "$LOG"
  [[ $rc == 0 ]] || failed=1
}

{ echo "== $(. /etc/os-release; echo "$PRETTY_NAME") $(uname -m), $(nproc) CPUs, $(free -m | awk '/Mem:/{print $2}') MiB,"
  echo "== image $IMAGE, source: $SOURCE, user $(id -un)"; } | tee -a "$LOG"

# README.md, Install (Ubuntu commands)
step apt 'sudo apt-get update && sudo apt-get install -y --no-install-recommends git curl ca-certificates libgomp1 libexpat1'
step python312 'if apt-cache show python3.12-venv >/dev/null 2>&1; then
    sudo apt-get install -y --no-install-recommends python3.12 python3.12-venv
  else
    curl -LsSf https://astral.sh/uv/install.sh | sh && export PATH="$HOME/.local/bin:$PATH" && uv python install 3.12
  fi && python3.12 --version'
if [[ $CLONE == 1 ]]; then
  step source 'git clone $REPO_URL geocrop-engine && cd geocrop-engine && git checkout $REF'
else  # the same tree as a clone of this commit, without needing it to be pushed
  step source 'tar -xf ~/source.tar && cd geocrop-engine'
fi
step venv 'python3.12 -m venv .venv && source .venv/bin/activate && python --version'
step install "python -m pip install -c constraints/py312.txt -e '.[all,web]'"
# README.md, Five-minute quick start 1
step help 'geocrop --version && geocrop --help | head -5'
step demo 'geocrop demo --root outputs/demo && ls outputs/demo/prediction outputs/demo/parcels'
# README.md, Five-minute quick start 2
step acceptance_offline 'geocrop acceptance --offline --out outputs/acceptance'
# README.md, Five-minute quick start 3: real Sentinel-2 (Earth Search, no account)
step quicklook 'geocrop quicklook --bbox 19.10 44.72 19.16 44.76 --start 2023-06-01 --end 2023-08-31 --out outputs/semberija-2023 && ls outputs/semberija-2023'
# README.md, Five-minute quick start 4 (+ docs/web-interface.md HTTP API)
step serve 'export GEOCROP_WEB_TOKEN=$(python -c "import secrets; print(secrets.token_urlsafe(32))")
  geocrop serve --workspace . --port 8765 > /report/serve.log 2>&1 & SERVE_PID=$!
  for i in $(seq 60); do curl -s -o /dev/null http://127.0.0.1:8765/ && break; sleep 1; done
  code=$(curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8765/api/v1/health); echo "without token: HTTP $code"
  curl -fsS -H "Authorization: Bearer $GEOCROP_WEB_TOKEN" http://127.0.0.1:8765/api/v1/health; rc=$?; echo
  kill $SERVE_PID; wait $SERVE_PID 2>/dev/null; [[ $rc == 0 && $code == 401 ]]'
exit $failed
STEPS
chown alice: /home/alice/steps.sh
su --whitelist-environment=SKIP,REF,CLONE,REPO_URL,SOURCE,IMAGE,HTTP_PROXY,HTTPS_PROXY,NO_PROXY,http_proxy,https_proxy,no_proxy,SSL_CERT_FILE,PIP_CERT,REQUESTS_CA_BUNDLE,CURL_CA_BUNDLE,GIT_SSL_CAINFO \
  - alice -c 'bash ~/steps.sh'
CONTAINER

{
  echo "# Fresh-user check: $IMAGE, --cpus $CPUS --memory $MEMORY"
  echo
  result=FAIL
  if [[ $status == 0 ]]; then
    result=PASS
    [[ -z $SKIP ]] || result="PARTIAL (requested steps passed; skipped:$SKIP)"
  fi
  echo "Source: $SOURCE; $(date -u +%Y-%m-%dT%H:%MZ); result: $result"
  echo
  echo "| Step | Exit status | Seconds |"
  echo "|---|---|---|"
  tail -n +2 "$OUT/steps.tsv" 2>/dev/null | awk -F'\t' '{printf "| %s | %s | %s |\n", $1, $2, $3}'
} > "$OUT/summary.md"
cat "$OUT/summary.md"
echo "Transcript: $OUT/transcript.log"
exit $status
