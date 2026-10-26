# Security model

GeoCrop Engine is a research library and command-line tool. The only network-facing component is the
optional local web workspace (`geocrop serve`, extra `web`), which runs engine commands as jobs. This page
describes what that component protects, what it deliberately does not, and how its guarantees are tested.
To report a vulnerability, see [SECURITY.md](../SECURITY.md).

## Threat model: a single owner's local workspace

The web server is a **single-owner tool on the owner's own machine**. It runs every job as the operating
system account that started it, against one chosen workspace directory. It is not a multi-user service, not
an internet-facing application and not a sandbox for untrusted code or data.

| Actor | Assumption |
|---|---|
| Owner | Holds the access token; trusted. Anything the owner can do in a shell, the owner may also do through the API within its restrictions. |
| Web pages in the owner's browser | **Untrusted.** Any website the owner visits may try to reach `127.0.0.1:8765` (CSRF, DNS rebinding, clickjacking). |
| Other local accounts on a shared host | Untrusted; they can reach the loopback port but do not know the token. |
| Network attackers | Out of reach: the server binds to loopback only (`--host` accepts `127.0.0.1`, `localhost`, `::1`). |
| Inputs (configs, rasters, vectors, `joblib` models) | **Trusted.** `joblib`/pickle model files execute code when loaded; never load models from untrusted sources. |

## What is protected

| Control | Implementation | Tests (`tests/test_security.py`) |
|---|---|---|
| Authentication | Every `/api` and `/api/v1` path, including unknown paths and routes added later, requires `Authorization: Bearer <token>` (middleware plus a per-route dependency). The token is compared in constant time, must be ≥ 32 characters, and is never accepted in a URL. | every route × missing, wrong, truncated, extended, wrong scheme, query-string token |
| DNS rebinding | Host header allow-list: `127.0.0.1`, `localhost`, `[::1]` with optional numeric port; anything else is HTTP 400. | allow-list unit cases, rebinding host names |
| Cross-site requests (CSRF) | Requests with an `Origin` that differs from the server's own scheme and host, `Origin: null`, or `Sec-Fetch-Site: cross-site` on the API are HTTP 403, even with a valid token. Browsers cannot attach the token cross-site because it lives only in page memory. | foreign, `null`, wrong scheme/port origins; fetch metadata |
| Clickjacking, content injection | `Content-Security-Policy` (`default-src 'self'`, no inline script, `object-src 'none'`, `frame-ancestors 'none'`, `base-uri 'none'`), `X-Frame-Options: DENY`, `X-Content-Type-Options: nosniff`, `Referrer-Policy: no-referrer`, `Cross-Origin-Opener-Policy` and `Cross-Origin-Resource-Policy: same-origin`, `Permissions-Policy`, `Cache-Control: no-store` on **every** response, including errors and rejections. | UI, static, success, 400/401/403/404/413/422/500 |
| Command execution | Jobs run `python -m geocrop.cli` with an argument list, never a shell. Only parser-known subcommands are accepted; `serve`, `snap` and `manifest --snap-gpt` (which would launch an external executable chosen by the request) are refused — also in abbreviated spellings such as `--snap-g`, because the parsed value is checked. | shell operators, substitutions, redirections, newlines, unbalanced quotes, abbreviations; literal metacharacters reach the engine unchanged with `shell=False` |
| Output confinement | Output roots (`--out`, `--root`, and `paths.output/features/composites` of a `run` config) are resolved (symbolic links followed) and must lie inside the workspace, outside the job state directory, and not be the workspace itself. | `..`, absolute, `//`, abbreviated `--ou`, `.`/empty, state directory, symlinked directories to outside and to `/etc` |
| Artifact downloads | Only files in the job's current listing can be downloaded; the listing is rebuilt per request and resolves every file, so symbolic links leaving the workspace or pointing into the state directory are never listed or served, even if swapped in after a listing. Downloads are `application/octet-stream` attachments. | `../`, absolute, URL- and double-encoded, Windows-style, NUL, symlink to `/etc/passwd`, symlinked `/etc` directory, swapped file, swapped output root |
| Resource bounds | Request bodies ≤ 1 MiB (413; chunked bodies without length 411); ≤ 256 arguments, ≤ 64 KiB per argument, ≤ 128 KiB in total (UTF-8 bytes); console command ≤ 65,536 characters; log reads ≤ 256 KiB per request; job lists ≤ 200; at most 32 queued jobs and exactly one running engine process. | each bound |
| Error hygiene | Unexpected errors return a generic HTTP 500 without traceback; configuration parse errors report only the error type, never file content. | |
| Secrets | The server token is removed from child-job environments. The UI keeps it in page memory only (not URLs, cookies or local storage). | `tests/test_web.py` |

## What is not protected

* **Anything the owner can do.** A token holder can read any file the account can read by passing it as an
  *input* (inputs are deliberately unrestricted), and can run every engine command except those listed above.
  Treat the token like a shell password.
* **Malicious inputs.** Configuration files, rasters, vector files and especially `joblib` models are trusted.
  A crafted model file can execute arbitrary code.
* **Other processes of the same account** can read the job database and logs in `<workspace>/.geocrop-web`
  and the token from the server's environment.
* **Races by local attackers on the workspace.** Output checks happen at submission; someone with write access
  to the workspace can replace an output directory with a symbolic link before the job writes to it.
  Downloads are rechecked at request time, but writes by the engine itself are not sandboxed.
* **Disk and CPU exhaustion by the owner.** Logs are read in bounded chunks but not truncated on disk; jobs
  have no time or memory limit. Use OS quotas or a container when needed.
* **Transport encryption.** Loopback traffic is plain HTTP. For remote notebooks, use SSH port forwarding;
  never expose the port through a reverse proxy or `0.0.0.0` forwarding.
* **Brute force** is not rate-limited; a random 32+ character token (e.g. `python -c "import secrets;
  print(secrets.token_urlsafe(32))"`) makes guessing infeasible.

## Tokens

Generate a fresh random token per server start or per machine, export it only in the shell that starts the
server (`GEOCROP_WEB_TOKEN`, or the variable named by `--token-env`), and never commit it, paste it into
notebooks with saved outputs, or put it in URLs. Rotating the token is a restart with a new value; existing
browser tabs and notebook clients then receive HTTP 401 until given the new token.

## Container and deployment

The Docker image runs as the unprivileged user `geocrop` (uid 1000) and does not start the web server by
default. The `.dockerignore` keeps `data/`, `outputs/`, `.geocrop-web/` job state, `.env` files and virtual
environments out of the build context. `geocrop serve --host` accepts only loopback addresses (`127.0.0.1`,
`localhost`, `::1`), so inside a container the server is reachable only from that container's own network
namespace; Docker port publishing (`-p`) does not reach it. Run the web workspace on the host, or on Linux with
`--network host` (the container then binds the host's loopback interface); the Host allow-list also requires a
loopback host name.

## Dependency vulnerabilities

The `audit` CI job runs [pip-audit](https://pypi.org/project/pip-audit/) on the pinned set
`constraints/py312.txt` and on a fresh install of the latest versions of all runtime extras; Dependabot opens
weekly update pull requests for Python dependencies and GitHub Actions. A release requires no known critical
or high vulnerability without a documented resolution below.

| Date | Scope | Result |
|---|---|---|
| 2026-10-26 | `constraints/py312.txt` (then 73 packages, extras `all,stac,web`; the file now pins every extra), PyPI advisory database and OSV | no known vulnerabilities |
| 2026-10-26 | development environment (pytest, ruff, build, pip-audit, coverage) | no known vulnerabilities |

**Accepted risks:** none at present. When an advisory cannot be fixed by a constraint bump, record it here with
the advisory ID, affected package and version, why GeoCrop is or is not exposed, and the planned resolution,
and pass `--ignore-vuln <ID>` in the `audit` job with a comment pointing to this table.
