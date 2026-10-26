"""Analyst parcel review over HTTP (gap G7): queue, parcel evidence, image chips and decisions.

Every file the review reads comes from `JobManager.output_files` of a finished job (resolved, inside the
workspace, outside the web state directory). Clients never pass file paths: a parcel layer is chosen by its
workspace-relative name among the layers found in the job, extra evidence by the id of another finished job.
Decisions and label tables are written only below `<job output>/review/`, after the same confinement check;\nchip and series caches live in the web state directory.
"""
from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field

from . import review
from .web_map import _png

MAX_EVIDENCE_JOBS = 8
REVIEW_DIR = "review"
LABEL_FILE = r"^(decisions|labels_(training|validation)(_round[0-9]{1,6})?)\.csv$"
_EMPTY = ("This job has no parcel layer. Review needs a finished parcel job: geocrop demo, geocrop run with "
          "parcels, geocrop parcels or geocrop traffic-light. Quicklooks contain imagery only.")


class DecisionBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    parcel_id: str = Field(min_length=1, max_length=256)
    decision: str = Field(pattern="^(agree|disagree|unsure)$")
    purpose: str = Field(pattern="^(training|validation)$")
    label: str | None = Field(default=None, max_length=64)
    analyst: str = Field(default="", max_length=80)
    comment: str = Field(default="", max_length=2000)
    confidence: float | None = Field(default=None, ge=0, le=1)
    layer: str | None = Field(default=None, max_length=4096)


class ExportBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    purpose: str = Field(pattern="^(training|validation)$")
    round: int | None = Field(default=None, ge=0, le=100000)
    layer: str | None = Field(default=None, max_length=4096)


def _split(value: str | None, limit: int = 64) -> list[str]:
    return [v for v in (value or "").split(",") if v][:limit]


class Context:
    """One finished job's review sources, confined to the workspace."""

    def __init__(self, manager, job_id: str, layer: str | None = None, evidence: str | None = None):
        job = manager.get(job_id)
        if job["status"] != "succeeded":
            raise HTTPException(status_code=409, detail="Review needs a job that finished successfully")
        self.manager, self.workspace, self.job = manager, manager.workspace, job
        self.files = manager.output_files(job_id)
        found = review.discover(self.files)
        self.layers = found["layers"]
        self.layer = None
        if self.layers:
            names = {self.rel(p): p for p in self.layers}
            if layer:
                if layer not in names:          # a name, never a path: only layers found in this job's outputs
                    raise HTTPException(status_code=404, detail="Parcel layer not found in this job")
                self.layer = names[layer]
            else:
                self.layer = self.layers[0]
        self.review_dir = self._review_dir() if self.layer else None
        if self.review_dir is not None:
            found = review.discover(self.files, exclude=self.review_dir)
        self.signatures, self.markers, self.imagery = found["signatures"], found["markers"], found["imagery"]
        self.evidence_jobs = []
        for other in _split(evidence, MAX_EVIDENCE_JOBS):
            if other == job_id:
                continue
            try:
                extra = manager.get(other)
            except KeyError:
                raise HTTPException(status_code=404, detail="Evidence job not found") from None
            if extra["status"] != "succeeded":
                raise HTTPException(status_code=409, detail="Evidence jobs must have finished successfully")
            more = review.discover(manager.output_files(other))
            self.signatures += [p for p in more["signatures"] if p not in self.signatures]
            self.markers += [p for p in more["markers"] if p not in self.markers]
            self.evidence_jobs.append(other)

    def rel(self, path: Path) -> str:
        return str(path.relative_to(self.workspace))

    def _review_dir(self) -> Path | None:
        """`<output folder>/review` (or `<file stem>_review` beside a file output), confined like the outputs."""
        state = self.manager.state_dir
        for raw in self.job["outputs"]:
            out = Path(raw).resolve()
            if out.is_dir() and self.layer.is_relative_to(out):
                target = out / REVIEW_DIR
            elif out.is_file() and self.layer == out:
                target = out.with_name(f"{out.stem}_{REVIEW_DIR}")
            else:
                continue
            resolved = target.resolve()
            if (resolved.is_relative_to(self.workspace) and not resolved.is_relative_to(state)
                    and not target.is_symlink()):
                return resolved
            return None
        return None

    def require_layer(self):
        if self.layer is None:
            raise HTTPException(status_code=404, detail=_EMPTY)
        try:
            return review.load_layer(self.layer)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    def writable(self, name: str) -> Path:
        """A file directly inside the review folder; refuses symlinks and anything that resolves elsewhere."""
        if self.review_dir is None:
            raise HTTPException(status_code=409, detail="No writable review folder for this job's outputs")
        path = self.review_dir / name
        if path.is_symlink() or (path.exists() and not path.is_file()):
            raise HTTPException(status_code=409, detail="The review folder holds an unexpected entry")
        if not path.resolve().is_relative_to(self.review_dir):
            raise HTTPException(status_code=409, detail="The review folder holds an unexpected entry")
        return path

    def log(self) -> review.DecisionLog:
        return review.DecisionLog(self.writable("decisions.csv"))

    def latest(self) -> dict:
        return self.log().latest() if self.review_dir is not None else {}


_cache_lock = threading.Lock()


def _cached(ctx: Context, parts: tuple, compute):
    """JSON cache in the web state directory (`review-cache/<job id>/`), keyed by the request and the source
    files' sizes and modification times, so the job's outputs stay free of cache files."""
    stamp = []
    for f in [ctx.layer, *ctx.signatures, *[m["folder"] / "B04.tif" for m in ctx.imagery]]:
        try:
            st = f.stat()
            stamp.append((str(f), st.st_mtime_ns, st.st_size))
        except OSError:
            stamp.append((str(f), 0, 0))
    key = hashlib.sha256(json.dumps([parts, stamp], default=str).encode()).hexdigest()[:40]
    folder = ctx.manager.state_dir / "review-cache" / ctx.job["id"]
    target = folder / f"{key}.json"
    try:
        if target.is_file():
            return json.loads(target.read_text())
    except (OSError, ValueError):
        pass
    value = compute()
    try:
        with _cache_lock:
            folder.mkdir(parents=True, exist_ok=True)
            tmp = target.with_suffix(".tmp")
            tmp.write_text(json.dumps(value))
            tmp.replace(target)
    except OSError:
        pass
    return value


def register_review_routes(app: FastAPI, jobs, authenticate) -> None:
    auth = [Depends(authenticate)]

    def context(request: Request, job_id: str, layer=None, evidence=None) -> Context:
        return Context(jobs(request), job_id, layer, evidence)

    @app.get("/api/jobs/{job_id}/review", dependencies=auth)
    def review_summary(job_id: str, request: Request, layer: str | None = Query(default=None, max_length=4096)):
        """What can be reviewed in this job: parcel layers, facets, evidence sources and progress."""
        ctx = context(request, job_id, layer)
        if ctx.layer is None:
            return {"available": False, "reason": _EMPTY, "imagery": len(ctx.imagery)}
        gdf = ctx.require_layer()
        latest = ctx.latest()
        counts = {d: 0 for d in review.DECISIONS}
        for r in latest.values():
            if r.get("parcel_id") in gdf.index and r.get("decision") in counts:
                counts[r["decision"]] += 1
        return {
            "available": True, "layer": ctx.rel(ctx.layer), "layers": [ctx.rel(p) for p in ctx.layers],
            "parcels": len(gdf), "classes": review.classes(gdf), "facets": review.facets(gdf),
            "progress": {"reviewed": sum(counts.values()), "total": len(gdf), **counts},
            "model_sha256": review.model_sha256(ctx.files),
            "sources": {"signatures": [ctx.rel(p) for p in ctx.signatures], "markers": [ctx.rel(p) for p in ctx.markers],
                        "imagery": len(ctx.imagery), "true_colour": sum(m["true_colour"] for m in ctx.imagery)},
            "review_dir": ctx.rel(ctx.review_dir) if ctx.review_dir else None,
            "sort_keys": list(review.SORT_KEYS),
        }

    @app.get("/api/jobs/{job_id}/review/parcels", dependencies=auth)
    def review_parcels(job_id: str, request: Request, layer: str | None = Query(default=None, max_length=4096),
                       traffic_light: str | None = Query(default=None, max_length=256),
                       declared_class: str | None = Query(default=None, max_length=4096),
                       decision: str | None = Query(default=None, max_length=1024),
                       reason: str | None = Query(default=None, max_length=4096),
                       max_confidence: float | None = Query(default=None, ge=0, le=1),
                       min_set_size: int | None = Query(default=None, ge=0, le=1000),
                       review_state: str = Query(default="all", alias="review", pattern="^(all|pending|done)$"),
                       q: str = Query(default="", max_length=128),
                       sort: str = Query(default="priority", max_length=32),
                       order: str = Query(default="asc", pattern="^(asc|desc)$"),
                       offset: int = Query(default=0, ge=0), limit: int = Query(default=50, ge=1, le=500)):
        """Filtered, sorted page of the parcel queue (comma-separated values are OR-ed within one filter)."""
        ctx = context(request, job_id, layer)
        gdf = ctx.require_layer()
        try:
            return review.query(gdf, traffic_light=_split(traffic_light), declared_class=_split(declared_class),
                                decision=_split(decision), reason=_split(reason), max_confidence=max_confidence,
                                min_set_size=min_set_size, review=review_state, reviewed=ctx.latest(), search=q,
                                sort=sort, descending=order == "desc", offset=offset, limit=limit)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/jobs/{job_id}/review/parcel", dependencies=auth)
    def review_parcel(job_id: str, request: Request, id: str = Query(min_length=1, max_length=256),
                      layer: str | None = Query(default=None, max_length=4096),
                      evidence: str | None = Query(default=None, max_length=512)):
        """One parcel: attributes, classes, probabilities, conformal set, markers, reasons, series, outline."""
        ctx = context(request, job_id, layer, evidence)
        gdf = ctx.require_layer()
        if id not in gdf.index:
            raise HTTPException(status_code=404, detail="Parcel not found")
        body = review.detail(gdf, id, ctx.markers)
        body["geometry"] = review.geometry_wgs84(gdf, id)
        if ctx.signatures:
            series = review.series_from_signatures(ctx.signatures, id)
            series["files"] = [ctx.rel(Path(f)) for f in series["files"]]
        else:
            series = None
        if not series or not series["panels"]:
            series = _cached(ctx, ("series", ctx.rel(ctx.layer), id),
                             lambda: review.series_from_imagery(gdf, id, ctx.imagery))
            series["files"] = [ctx.rel(Path(f)) for f in series["files"]]
        body["series"] = series
        mine = [r for r in (ctx.log().rows() if ctx.review_dir else []) if r.get("parcel_id") == id]
        body["history"] = mine[-20:]
        body["model_sha256"] = review.model_sha256(ctx.files)
        return body

    @app.get("/api/jobs/{job_id}/review/chips", dependencies=auth)
    def review_chips(job_id: str, request: Request, id: str = Query(min_length=1, max_length=256),
                     kind: str = Query(default="true", pattern="^(true|ndvi)$"),
                     size: int = Query(default=96, ge=16, le=256),
                     layer: str | None = Query(default=None, max_length=4096)):
        """Image crops around the parcel per imagery date (PNG data URLs, cached in the web state directory)."""
        ctx = context(request, job_id, layer)
        gdf = ctx.require_layer()
        if id not in gdf.index:
            raise HTTPException(status_code=404, detail="Parcel not found")

        def compute():
            chips = review.chips(gdf, id, ctx.imagery, kind, size)
            return [{**{k: v for k, v in c.items() if k != "rgba"}, "url": _png(c["rgba"])} for c in chips]
        return {"kind": kind, "chips": _cached(ctx, ("chips", ctx.rel(ctx.layer), id, kind, size), compute),
                "note": "Downsampled crops for visual checks; one stretch is shared by all dates."}

    @app.get("/api/jobs/{job_id}/review/decisions", dependencies=auth)
    def review_decisions(job_id: str, request: Request, layer: str | None = Query(default=None, max_length=4096)):
        ctx = context(request, job_id, layer)
        ctx.require_layer()
        rows = ctx.log().rows() if ctx.review_dir else []
        return {"rows": len(rows), "latest": list(ctx.latest().values()),
                "path": ctx.rel(ctx.review_dir / "decisions.csv") if ctx.review_dir else None}

    @app.post("/api/jobs/{job_id}/review/decisions", status_code=201, dependencies=auth)
    def review_decide(job_id: str, body: DecisionBody, request: Request):
        """Append one analyst decision (never edits earlier rows; the latest row per parcel counts)."""
        ctx = context(request, job_id, body.layer)
        gdf = ctx.require_layer()
        try:
            row = review.validate_decision(body.model_dump(), gdf, review.classes(gdf))
        except KeyError:
            raise HTTPException(status_code=404, detail="Parcel not found") from None
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        record = ctx.log().append(row, review.model_sha256(ctx.files), ctx.rel(ctx.layer))
        return {"decision": record, "new_class": row["new_class"],
                "path": ctx.rel(ctx.review_dir / "decisions.csv")}

    @app.get("/api/jobs/{job_id}/review/file", dependencies=auth)
    def review_file(job_id: str, request: Request, name: str = Query(max_length=64, pattern=LABEL_FILE),
                    layer: str | None = Query(default=None, max_length=4096)):
        """Download the decision log or an exported label table (fixed names inside the review folder only)."""
        ctx = context(request, job_id, layer)
        ctx.require_layer()
        path = ctx.writable(name)
        if not path.is_file():
            raise HTTPException(status_code=404, detail="Not written yet")
        return FileResponse(path, filename=name, media_type="text/csv")

    @app.post("/api/jobs/{job_id}/review/export", dependencies=auth)
    def review_export(job_id: str, body: ExportBody, request: Request):
        """Write the decisions of one purpose as an `active-learning ingest` label table into the review folder."""
        ctx = context(request, job_id, body.layer)
        ctx.require_layer()
        table = review.export_labels(ctx.log().rows(), body.purpose, body.round)
        name = f"labels_{body.purpose}" + (f"_round{body.round}" if body.round is not None else "") + ".csv"
        target = ctx.writable(name)
        table.to_csv(target, index=False)
        labelled = int((table["label"].astype(str) != "").sum()) if len(table) else 0
        command = None
        if body.purpose == "training":
            command = ["active-learning", "ingest", "--labels", ctx.rel(target), "--parcels", ctx.rel(ctx.layer),
                       "--training", "<training.gpkg>", "--out", "<training_next.gpkg>",
                       "--round", str(body.round if body.round is not None else 1)]
        return {"path": ctx.rel(target), "rows": len(table), "labelled": labelled, "purpose": body.purpose,
                "columns": list(table.columns), "ingest_argv": command,
                "preview": table.head(5).fillna("").astype(str).to_dict(orient="records") if len(table) else []}

