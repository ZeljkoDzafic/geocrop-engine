# Fresh-user check on Ubuntu (independent installation without author intervention)

`tools/fresh_user_check.sh` starts a pristine Ubuntu container, creates an ordinary user with `sudo` and, as that
user, runs only what the README says: the Ubuntu installation commands of *Install* and the four steps of the
*Five-minute quick start* (plus `geocrop --version/--help` and a health request with and without the token).
If a step has to differ from the README, the README is wrong.

```bash
tools/fresh_user_check.sh                                   # committed HEAD (git archive), ubuntu:26.04, 2 CPUs, 4 GB
tools/fresh_user_check.sh --clone --ref <tag-or-branch>     # git clone from GitHub instead
tools/fresh_user_check.sh --image ubuntu:24.04              # the apt python3.12 path
```

Output: `transcript.log` (every command and its output), `steps.tsv`, `summary.md` in `--out` (default a temporary
directory). Container stand-in for "a fresh machine": `apt-get install sudo` and `useradd` only; everything else
is a README command.

## Recorded run

| Item | Value |
|---|---|
| Date | 2026-10-26 10:44 UTC |
| Source | `git archive` of commit 3a05477 (main before this branch), i.e. exactly the files of a clone |
| Image | `ubuntu:26.04` (26.04.1 LTS), aarch64, `docker run --cpus 2 --memory 4g` (Rancher Desktop VM on an Apple M5 Pro laptop) |
| Result | **PASS**, 10/10 steps, 3 min |

| Step | README command | Exit | Seconds |
|---|---|---|---|
| apt | `sudo apt-get install -y --no-install-recommends git curl ca-certificates libgomp1 libexpat1` | 0 | 5 |
| python312 | no `python3.12-venv` package on 26.04, so uv: `curl -LsSf https://astral.sh/uv/install.sh \| sh`, `uv python install 3.12` (3.12.15) | 0 | 3 |
| source | clone (here: the archive of the commit) | 0 | 0 |
| venv | `python3.12 -m venv .venv && source .venv/bin/activate` | 0 | 2 |
| install | `python -m pip install -c constraints/py312.txt -e '.[all,stac,web]'` (the merged README now says `.[all,web]`; all wheels, nothing compiled on aarch64) | 0 | 26 |
| help | `geocrop --version`, `geocrop --help` | 0 | 0 |
| demo | `geocrop demo --root outputs/demo` | 0 | 13 |
| acceptance_offline | `geocrop acceptance --out outputs/acceptance-ci --offline` (PASS, 21/21 checks) | 0 | 18 |
| quicklook | `geocrop quicklook --bbox 19.10 44.72 19.16 44.76 --start 2023-06-01 --end 2023-08-31 …` (Earth Search, no account) | 0 | 112 |
| serve | token from `secrets.token_urlsafe(32)`, `geocrop serve --port 8765`; `/api/v1/health` → HTTP 401 without the token, `{"status":"ok","version":"0.9.0",…}` with it | 0 | 1 |

That run used the commands the README now contains. **At that commit the README did not contain them**, which is
what the check found (below). After these changes were merged with main (README quick start rewritten by the
documentation audit), the script was aligned with the merged README (`.[all,web]`, `acceptance --offline --out
outputs/acceptance`), but **the merged commit could not be re-run**: the laptop's Docker daemon stopped during the
Linux benchmark and was not available again before this branch was submitted. Re-run
`tools/fresh_user_check.sh` (and `--image ubuntu:24.04`) before the release.

## Findings and what was changed

1. **No working Ubuntu path in the README.** It offered conda or `pip install -e '.[all]'` with "Python >= 3.10";
   on Ubuntu 26.04 the system Python is 3.14 and there is no `python3.12` package, while the pinned constraints
   are for 3.12. The only working recipe (uv) was inside `deploy/vm_acceptance.sh`. The README now lists the apt
   packages, Python 3.12 from apt (24.04) or uv (26.04) and the pinned pip install.
2. **`libexpat1` is required with uv's Python.** In a minimal `ubuntu:26.04` with uv's CPython 3.12 and the pinned
   install, `geocrop demo` fails with `ImportError: libexpat.so.1` (GDAL inside the rasterio wheel). With the
   distribution's `python3` the package comes in as a dependency, which hides the problem. Now in the README's apt
   line and installed explicitly by `deploy/vm_acceptance.sh --system` (before, only `git`'s dependencies pulled it in).
3. **The serve example gave no way to make a token or check the server**: the README now generates the token with
   `secrets.token_urlsafe(32)` and shows the authenticated `/api/v1/health` request.
4. **The official tag v0.9.0 cannot run today's quick start**: it predates `constraints/`, `geocrop quicklook`,
   `geocrop acceptance` and `deploy/vm_acceptance.sh` (109 commits behind main on the day of the check), and there
   is no GitHub Release with a wheel although docs/release.md describes one. An independent user following the
   current README needs a new tagged release; until then the check runs on main.
5. `docs/getting-started.md` claimed "~40 tests, < 1 min" (about 1000 tests now); corrected in main by the audit.
6. `deploy/setup_ubuntu.sh` left `/data/geocrop` and `/scratch/geocrop` root-owned with mode 750, so every pilot
   command of the VM runbook would fail for the run user; the directories now belong to `GEOCROP_USER` (default the
   sudo user).

Side observations (no change): the distribution's Python 3.14 on 26.04 also installed the pinned set and ran the
demo, but 3.14 is not in the CI matrix, so 3.12 stays the documented interpreter. The check ran on aarch64; x86_64
(emulated or on a VM) was not run.

## Follow-up in the managed Linux environment (27 October 2026)

The historical aarch64 result above is not a fresh validation of every later commit. The follow-up below
uses x86_64 Ubuntu containers, `--cpus 2 --memory 4g`, on a shared managed host, through an enforced outbound
proxy. Raw transcripts, per-step status/timing files and their SHA-256 index are retained in
[2026-10-27-linux/evidence.json](2026-10-27-linux/evidence.json). These are software checks on synthetic data,
not observational crop results; their timings are not a new scale benchmark.

| Image / source | Result | Details |
|---|---|---|
| Ubuntu 24.04.5 / `88ed027` | incomplete | installation, help, demo and offline acceptance passed; Earth Search quicklook failed with proxy CONNECT 403; serve was not run after that failure |
| Ubuntu 24.04.5 / `13118df` | **PARTIAL: 9 passed, 1 explicitly skipped** | fresh installation, demo, offline acceptance and serve passed; quicklook explicitly skipped after the confirmed network denial |
| Ubuntu 26.04.1 / `e815448` | blocked at interpreter setup | apt prerequisites passed; proxy returned HTTP 403 for the astral.sh uv installer; later steps were not run |

In the 24.04 subset, demo took 66 s and offline acceptance 45 s. The API returned HTTP 401 without the token
and `status: ok`, version 0.9.0, with the token. The earlier complete-network attempt passed offline acceptance
in 43 s before its quicklook denial. Do not describe the subset as a full 10/10 PASS, or the 26.04 attempt as a
completed installation. Repeat the denied external-data and interpreter-download steps in an environment that
permits the documented providers before making an unrestricted fresh-user release claim.

The harness now preserves proxy settings through both the login shell and sudo, restores report ownership to
the calling user and labels skips explicitly. A combined trust bundle can be mounted read-only with
`--ca-bundle /path/to/ca-bundle.pem`; this is infrastructure trust configuration, not a way to grant access to
blocked destinations. TLS verification stays enabled.
