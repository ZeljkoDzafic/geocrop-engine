"""Optional local HTML application and authenticated HTTP API."""
from __future__ import annotations

import hmac
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlsplit

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.routing import APIRoute
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from . import __version__
from .web_jobs import JobManager, command_catalogue

logger = logging.getLogger(__name__)

#: Unversioned compatibility prefix and the stable, versioned REST contract.
API_PREFIX = "/api"
API_V1_PREFIX = "/api/v1"
#: Loopback names only: a DNS-rebinding page cannot reach the API under its own host name.
ALLOWED_HOSTS = frozenset({"127.0.0.1", "localhost", "[::1]"})
#: Request bodies are small JSON documents; anything larger is refused before parsing.
MAX_BODY_BYTES = 1024 * 1024
SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "Cache-Control": "no-store",
    "X-Frame-Options": "DENY",
    "Cross-Origin-Opener-Policy": "same-origin",
    "Cross-Origin-Resource-Policy": "same-origin",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=(), payment=(), usb=()",
    "Content-Security-Policy": (
        "default-src 'self'; script-src 'self'; style-src 'self'; "
        "img-src 'self' data: https://tile.openstreetmap.org https://tiles.maps.eox.at; "
        "connect-src 'self'; object-src 'none'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'"),
}


def host_allowed(header: str | None) -> bool:
    """True for a loopback Host header, optionally with a numeric port (IPv6 in brackets)."""
    if not header:
        return False
    header = header.strip().lower()
    if header.startswith("["):
        end = header.find("]") + 1
        host, port = header[:end], header[end:]
    else:
        host, colon, port = header.partition(":")
        port = colon + port
    if port and not (port.startswith(":") and port[1:].isdigit() and len(port) <= 6):
        return False
    return host in ALLOWED_HOSTS


def bearer_valid(authorization: str | None, token: str) -> bool:
    supplied = authorization[7:] if authorization and authorization.startswith("Bearer ") else ""
    return hmac.compare_digest(supplied.encode("utf8", "surrogateescape"), token.encode("utf8"))


def is_api_path(path: str) -> bool:
    return path == API_PREFIX or path.startswith(API_PREFIX + "/")


def add_versioned_aliases(app: FastAPI, prefix: str = API_V1_PREFIX) -> None:
    """Expose every unversioned ``/api/...`` route also under ``prefix`` with the same handler.

    Call once after all routes are registered. The versioned path is the documented stable contract
    and the only one in the OpenAPI schema; the unversioned path stays as a compatibility alias.
    """
    existing = {(r.path, m) for r in app.router.routes if isinstance(r, APIRoute) for m in r.methods}
    for route in list(app.router.routes):
        if (not isinstance(route, APIRoute) or not route.path.startswith(API_PREFIX + "/")
                or route.path.startswith(prefix + "/")):
            continue
        alias = prefix + route.path[len(API_PREFIX):]
        methods = sorted(m for m in route.methods if (alias, m) not in existing)
        if not methods:
            continue
        app.router.add_api_route(
            alias, route.endpoint, methods=methods, status_code=route.status_code,
            dependencies=list(route.dependencies), response_model=route.response_model,
            response_class=route.response_class, summary=route.summary, description=route.description,
            responses=route.responses, tags=route.tags, deprecated=route.deprecated,
            include_in_schema=route.include_in_schema, name=f"v1_{route.name}")
        route.include_in_schema = False


class JobRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    argv: list[str] | None = Field(default=None, max_length=256)
    command: str | None = Field(default=None, max_length=65536)


FILE_LIMIT = 200
_FILE_SCAN_LIMIT = 20000


def workspace_entries(root: Path, state_dir: Path, prefix: str) -> list[dict]:
    """Entries of a workspace directory whose names start with the last prefix component.

    Rejects absolute paths, ``..`` and hidden components; resolves symlinks and drops anything that
    leaves the workspace or touches the web state directory.
    """
    text = prefix.replace("\\", "/")
    if "\x00" in text or text.startswith("/") or (len(text) > 1 and text[1] == ":"):
        raise HTTPException(status_code=400, detail="Use a path relative to the workspace")
    *parents, partial = text.split("/")
    parts = [p for p in parents if p not in ("", ".")]
    if any(p == ".." or p.startswith(".") for p in parts) or partial.startswith(".."):
        raise HTTPException(status_code=400, detail="Use a path inside the workspace")
    state = state_dir.resolve()
    directory = root.joinpath(*parts).resolve()
    if not directory.is_relative_to(root) or directory.is_relative_to(state) or not directory.is_dir():
        return []
    found, needle = [], partial.casefold()
    try:
        with os.scandir(directory) as scan:
            for count, entry in enumerate(scan):
                if count >= _FILE_SCAN_LIMIT:
                    break
                if entry.name.startswith(".") or not entry.name.casefold().startswith(needle):
                    continue
                try:
                    target = Path(entry.path).resolve()
                    is_dir = target.is_dir()
                except OSError:
                    continue
                if not target.is_relative_to(root) or target.is_relative_to(state):
                    continue
                found.append({"name": entry.name, "path": "/".join([*parts, entry.name]), "is_dir": is_dir})
    except OSError:
        return []
    found.sort(key=lambda e: (not e["is_dir"], e["name"].casefold()))
    return found[:FILE_LIMIT]


def create_app(workspace, token: str, state_dir=None, token_env="GEOCROP_WEB_TOKEN") -> FastAPI:
    if not isinstance(token, str) or len(token) < 32:
        raise ValueError("A random access token of at least 32 characters is required")
    root = Path(workspace).resolve()
    if not root.is_dir():
        raise ValueError("Workspace must exist")
    static = Path(__file__).with_name("web_static")

    @asynccontextmanager
    async def lifespan(app):
        app.state.jobs = JobManager(root, state_dir, token_env)
        try:
            yield
        finally:
            app.state.jobs.close()

    app = FastAPI(title="GeoCrop Engine", version=__version__, lifespan=lifespan,
                  description="Local single-owner research workspace. Commands run as the server's user.",
                  docs_url=None, redoc_url=None, openapi_url=None)

    def reject(status: int, detail: str, plain: bool = False):
        response = (PlainTextResponse(detail, status_code=status) if plain
                    else JSONResponse({"detail": detail}, status_code=status))
        if status == 401:
            response.headers["WWW-Authenticate"] = "Bearer"
        return response

    async def guard(request: Request):
        """Host, origin, size and token checks for every request, before any route runs."""
        if not host_allowed(request.headers.get("host")):
            return reject(400, "Invalid host header", plain=True)
        origin = request.headers.get("origin")
        # Browser requests stay same-origin; Python/Jupyter clients do not send Origin.
        if origin and (urlsplit(origin).netloc.lower() != request.headers.get("host", "").lower() or
                       urlsplit(origin).scheme != request.url.scheme):
            return reject(403, "Cross-origin requests are not permitted")
        if not is_api_path(request.url.path):
            return None
        if request.headers.get("sec-fetch-site") == "cross-site":
            return reject(403, "Cross-origin requests are not permitted")
        # Defence in depth: every /api path (including routes added later and unknown paths) needs the token.
        if not bearer_valid(request.headers.get("authorization"), token):
            return reject(401, "A valid Bearer token is required")
        length = request.headers.get("content-length")
        if length is not None:
            if not length.isdigit():
                return reject(400, "Invalid Content-Length")
            if int(length) > MAX_BODY_BYTES:
                return reject(413, "Request body is too large")
        elif "chunked" in request.headers.get("transfer-encoding", "").lower():
            return reject(411, "Content-Length is required")
        return None

    @app.middleware("http")
    async def browser_protection(request: Request, call_next):
        response = await guard(request)
        if response is None:
            try:
                response = await call_next(request)
            except Exception:
                # Unexpected failures carry no traceback to the client and keep the security headers.
                logger.exception("Unhandled error for %s %s", request.method, request.url.path)
                response = reject(500, "Internal server error")
        for name, value in SECURITY_HEADERS.items():
            response.headers[name] = value
        return response

    def authenticate(authorization: str | None = Header(default=None)):
        if not bearer_valid(authorization, token):
            raise HTTPException(status_code=401, detail="A valid Bearer token is required")

    def jobs(request: Request):
        return request.app.state.jobs

    @app.exception_handler(KeyError)
    async def missing(request, exc):
        return JSONResponse({"detail": "Job or artifact not found"}, status_code=404)

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(static / "index.html")

    app.mount("/static", StaticFiles(directory=static), name="static")

    @app.get("/api/health", dependencies=[Depends(authenticate)])
    def health():
        return {"status": "ok", "version": __version__, "workspace": str(root), "workers": 1}

    @app.get("/api/commands", dependencies=[Depends(authenticate)])
    def commands():
        return {"commands": command_catalogue()}

    @app.get("/api/jobs", dependencies=[Depends(authenticate)])
    def list_jobs(request: Request, limit: int = Query(default=50, ge=1, le=200)):
        return {"jobs": jobs(request).list(limit)}

    @app.post("/api/jobs", status_code=202, dependencies=[Depends(authenticate)])
    def submit(body: JobRequest, request: Request):
        try:
            return jobs(request).submit(body.argv, body.command)
        except (ValueError, OSError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/jobs/{job_id}", dependencies=[Depends(authenticate)])
    def get_job(job_id: str, request: Request):
        return jobs(request).get(job_id)

    @app.post("/api/jobs/{job_id}/cancel", dependencies=[Depends(authenticate)])
    def cancel(job_id: str, request: Request):
        return jobs(request).cancel(job_id)

    @app.get("/api/jobs/{job_id}/logs", dependencies=[Depends(authenticate)])
    def logs(job_id: str, request: Request, offset: int = Query(default=0, ge=0),
             limit: int = Query(default=65536, ge=1, le=262144)):
        return jobs(request).logs(job_id, offset, limit)

    @app.get("/api/jobs/{job_id}/artifacts", dependencies=[Depends(authenticate)])
    def artifacts(job_id: str, request: Request):
        return {"files": jobs(request).artifacts(job_id), "limit": 200}

    @app.get("/api/jobs/{job_id}/artifact", dependencies=[Depends(authenticate)])
    def artifact(job_id: str, path: str, request: Request):
        file = jobs(request).artifact(job_id, path)
        # Downloads remain attachments; generated HTML never executes in the app's origin.
        return FileResponse(file, filename=file.name, media_type="application/octet-stream")

    @app.get("/api/jobs/{job_id}/map", dependencies=[Depends(authenticate)])
    def job_map(job_id: str, request: Request, max_px: int = Query(default=1024, ge=128, le=4096)):
        """Map-ready previews (WGS 84) of the job's rasters and vector outputs."""
        from .web_map import job_layers
        manager = jobs(request)
        files = manager.output_files(job_id)
        return job_layers(files, max_px)

    @app.get("/api/files", dependencies=[Depends(authenticate)])
    def files(request: Request, prefix: str = Query(default="", max_length=4096)):
        """Workspace path completion for the terminal: at most 200 entries, never outside the workspace."""
        return {"entries": workspace_entries(root, jobs(request).state_dir, prefix), "limit": FILE_LIMIT}

    @app.get("/api/openapi.json", dependencies=[Depends(authenticate)])
    def schema():
        return app.openapi()

    # Analyst parcel review (gap G7): queue, evidence, chips and append-only decisions of finished parcel jobs.
    from .web_review import register_review_routes
    register_review_routes(app, jobs, authenticate)

    # Keep last: aliases every /api route registered above under the stable /api/v1 prefix.
    add_versioned_aliases(app)
    return app
