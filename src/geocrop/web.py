"""Optional local HTML application and authenticated HTTP API."""
from __future__ import annotations

import hmac
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlsplit

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field
from starlette.middleware.trustedhost import TrustedHostMiddleware

from . import __version__
from .web_jobs import JobManager, command_catalogue


class JobRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    argv: list[str] | None = Field(default=None, max_length=256)
    command: str | None = Field(default=None, max_length=65536)


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
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "[::1]"])

    @app.middleware("http")
    async def browser_protection(request: Request, call_next):
        origin = request.headers.get("origin")
        # Browser requests stay same-origin; Python/Jupyter clients do not send Origin.
        if origin and (urlsplit(origin).netloc != request.headers.get("host") or
                       urlsplit(origin).scheme != request.url.scheme):
            from fastapi.responses import JSONResponse
            return JSONResponse({"detail": "Cross-origin requests are not permitted"}, status_code=403)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Cache-Control"] = "no-store"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
            "connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'")
        return response

    def authenticate(authorization: str | None = Header(default=None)):
        supplied = authorization.removeprefix("Bearer ") if authorization and authorization.startswith("Bearer ") else ""
        if not hmac.compare_digest(supplied.encode(), token.encode()):
            raise HTTPException(status_code=401, detail="A valid Bearer token is required")

    def jobs(request: Request):
        return request.app.state.jobs

    @app.exception_handler(KeyError)
    async def missing(request, exc):
        from fastapi.responses import JSONResponse
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

    @app.get("/api/openapi.json", dependencies=[Depends(authenticate)])
    def schema():
        return app.openapi()

    return app
