"""FastAPI application.

Three product surfaces over one engine:
  /api/route      classify arbitrary text against an arbitrary schema
  /api/panel      the evidence panel: every system scored on one fixture
  /api/schema-swap the disjoint-taxonomy experiment

The model is loaded lazily on first inference rather than at import, so the UI
shell and /api/health respond while a 1.8 GB checkpoint is still being read.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from contextlib import asynccontextmanager
from typing import Any

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import fixtures as fx
from .engine import get_engine
from .evaluate import run_panel, run_schema_swap
from .llm import LLMClassifier, LLMConfig

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("api")

STATIC_DIR = __import__("pathlib").Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("starting; model loads on first inference")
    yield
    await LLMClassifier().aclose()


app = FastAPI(title="GLiNER2.5-Decide console", lifespan=lifespan)


# --- request models ------------------------------------------------------

class Head(BaseModel):
    name: str = Field(..., description="Head name, e.g. intent")
    labels: list[str]
    multi_label: bool = False
    cls_threshold: float = 0.4
    described: dict[str, str] | None = Field(
        None, description="Optional label -> description map"
    )


class RouteRequest(BaseModel):
    text: str
    heads: list[Head] = Field(default_factory=lambda: [Head(name="intent", labels=["other"])])


class PanelRequest(BaseModel):
    fixture: str = "banking77"
    limit: int | None = 500
    systems: list[str] | None = None
    described: bool = False
    batch_size: int = 16
    # The LLM arm costs seconds per item, so the ceiling is low by default to
    # keep a panel run interactive. Raise it deliberately.
    llm_concurrency: int = Field(4, ge=1, le=16)


class SwapRequest(BaseModel):
    fixture: str = "clinc150"
    limit: int = 300


def _schema_from_heads(heads: list[Head]) -> dict[str, Any]:
    schema: dict[str, Any] = {}
    for head in heads:
        if not head.labels:
            raise HTTPException(400, f"head {head.name!r} has no labels")
        if len(head.labels) > 512:
            raise HTTPException(400, "label set too large (max 512)")
        if head.multi_label:
            schema[head.name] = {
                "labels": head.labels,
                "multi_label": True,
                "cls_threshold": head.cls_threshold,
            }
        elif head.described:
            schema[head.name] = {"labels": head.described}
        else:
            schema[head.name] = head.labels
    return schema


# --- routes --------------------------------------------------------------

@app.get("/api/health")
async def health() -> dict:
    engine = get_engine()
    return {
        "ok": True,
        "runtime": engine.info(),
        "llm": LLMConfig.from_env().status(),
        "fixtures": fx.available(),
    }


@app.get("/api/fixtures/{name}")
async def get_fixture(name: str, limit: int = 50) -> dict:
    try:
        fixture = fx.load(name)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    n = min(limit, len(fixture.texts))
    return {
        "name": fixture.name,
        "n": len(fixture.texts),
        "n_labels": fixture.n_labels,
        "label_coverage": fixture.described_coverage(),
        "labels": fixture.label_names,
        "descriptions": fixture.descriptions,
        "source": fixture.manifest.get("source"),
        "note": fixture.manifest.get("note"),
        "sample": [
            {"text": fixture.texts[i], "label": fixture.labels[i]} for i in range(n)
        ],
    }


@app.post("/api/route")
async def route(req: RouteRequest) -> dict:
    if not req.text.strip():
        raise HTTPException(400, "text is empty")
    schema = _schema_from_heads(req.heads)
    # classify_one, not .model.classify_text directly. The third positional
    # argument of classify_text is `threshold`, not `include_confidence`, so
    # calling it positionally with True silently drops every confidence value
    # and the router had nothing to show. classify_one also runs _parse, keeping
    # it the single normalisation point for the shapes classify_text returns.
    try:
        result = await asyncio.to_thread(get_engine().classify_one, req.text, schema)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"inference failed: {type(exc).__name__}: {exc}") from exc
    return {
        "schema": schema,
        "result": {head: pred.to_public() for head, pred in result.items()},
    }


@app.post("/api/panel")
async def panel(req: PanelRequest) -> dict:
    try:
        return await run_panel(
            req.fixture, limit=req.limit, systems=req.systems,
            described=req.described, batch_size=req.batch_size,
            llm_concurrency=req.llm_concurrency,
        )
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc


@app.post("/api/schema-swap")
async def schema_swap(req: SwapRequest) -> dict:
    try:
        return await run_schema_swap(req.fixture, limit=req.limit)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc


@app.get("/api/llm/probe")
async def llm_probe() -> dict:
    return await LLMClassifier().probe()


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")