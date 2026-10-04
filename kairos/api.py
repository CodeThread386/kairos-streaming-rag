"""HTTP + WebSocket service.

    GET  /health                      readiness (index loaded, corpus stats)
    GET  /                            live demo console
    GET  /v1/corpora                  available corpora
    GET  /v1/scenarios                demo scripts for the console
    GET  /v1/chunks/{chunk_id}        source text behind a citation
    POST /v1/replay                   run a whole (multi-turn) transcript, return the structured records
    WS   /v1/stream?corpus=tessera    full-duplex stream: send chunks/partials/end, receive events + answers
"""
from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
import os
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from .config import Config
from .engine import Engine
from .replay import run_session

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"
DEMO = ROOT / "demo" / "scenarios.json"


def _corpora() -> dict[str, str]:
    raw = os.environ.get("KAIROS_CORPORA", f"tessera={ROOT / 'corpus' / 'tessera'},handbook18f={ROOT / 'corpus' / 'handbook18f'}")
    out = {}
    for part in raw.split(","):
        if "=" in part:
            k, v = part.split("=", 1)
            out[k.strip()] = v.strip()
    return out


class Engines:
    def __init__(self):
        self.paths = _corpora()
        self.loaded: dict[str, Engine] = {}

    def get(self, name: str | None) -> Engine:
        name = name or next(iter(self.paths))
        if name not in self.paths:
            raise HTTPException(404, f"unknown corpus '{name}'")
        if name not in self.loaded:
            self.loaded[name] = Engine(Config(corpus_path=self.paths[name]))
        return self.loaded[name]


engines = Engines()


@asynccontextmanager
async def lifespan(_app):
    engines.get(None)          # build / load the default index before accepting traffic
    yield


app = FastAPI(title="Kairos — Streaming Live RAG", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health():
    e = engines.get(None)
    return {"status": "ok", "corpus": e.cfg.corpus_path, "index": e.index.stats(), "loaded": list(engines.loaded)}


@app.get("/")
def index():
    return FileResponse(WEB / "index.html")


@app.get("/v1/corpora")
def corpora():
    return {"corpora": list(engines.paths)}


@app.get("/v1/scenarios")
def scenarios():
    if DEMO.exists():
        return json.loads(DEMO.read_text(encoding="utf-8"))
    return {"scenarios": []}


@app.get("/v1/chunks/{chunk_id:path}")
def chunk(chunk_id: str, corpus: str | None = Query(None)):
    c = engines.get(corpus).chunk(chunk_id)
    if c is None:
        raise HTTPException(404, f"no chunk {chunk_id}")
    return c


class ReplayTurn(BaseModel):
    utterance: str | None = None
    chunks: list[dict] | None = None
    end_t: float | None = None


class ReplayRequest(BaseModel):
    corpus: str | None = None
    turns: list[ReplayTurn] = Field(..., min_length=1)
    asr_style: bool = False


@app.post("/v1/replay")
def replay(req: ReplayRequest):
    eng = engines.get(req.corpus)
    scen = {"id": f"replay-{int(time.time() * 1000)}", "turns": []}
    for t in req.turns:
        d = {"utterance": t.utterance or " ".join(c.get("text", "") for c in (t.chunks or [])), "gold": None}
        if t.chunks:
            d["chunks"] = t.chunks
            d["end_t"] = t.end_t if t.end_t is not None else (t.chunks[-1].get("t", 0) + 0.5)
        scen["turns"].append(d)
    out = run_session(eng, scen, asr_style=req.asr_style)
    keep = ("turn_id", "turn_type", "retrieval_required", "reason", "retrieval_events", "sub_queries", "answer",
            "citations", "uncertainty", "answer_version", "telemetry")
    return {"session_id": scen["id"], "turns": [{k: t[k] for k in keep if k in t} for t in out["turns"]]}


@app.websocket("/v1/stream")
async def stream(ws: WebSocket, corpus: str | None = None):
    await ws.accept()
    eng = engines.get(corpus)
    session = eng.new_session()
    loop = asyncio.get_running_loop()
    await ws.send_json({"type": "session", "session_id": session.id, "corpus": eng.cfg.corpus_path,
                        "index": eng.index.stats()})
    try:
        while True:
            msg = await ws.receive_json()
            kind = msg.get("type")
            t = float(msg.get("t", 0.0))
            if kind == "reset":
                session = eng.new_session()
                await ws.send_json({"type": "session", "session_id": session.id})
                continue
            if kind in ("chunk", "partial"):
                if session.turn is None:
                    session.begin_turn()
                evs = await loop.run_in_executor(None, session.feed, t, msg.get("text", ""), kind == "partial")
                await ws.send_json({"type": "events", "events": evs})
            elif kind == "end":
                if session.turn is None or not session.turn.text.strip():
                    continue
                n0 = len(session.tracer.events)
                rec = await loop.run_in_executor(None, session.end, t)
                await ws.send_json({"type": "events", "events": session.tracer.events[n0:]})
                rec_out = {k: v for k, v in rec.items() if k != "gold"}
                await ws.send_json({"type": "answer_start", "turn_id": rec["turn_id"]})
                # stream the answer word by word, as a speech engine would consume it
                words = (rec["answer"] or "").split(" ")
                for i in range(0, len(words), 3):
                    await ws.send_json({"type": "answer_delta", "text": " ".join(words[i:i + 3]) + " "})
                    await asyncio.sleep(0.03)
                await ws.send_json({"type": "record", "record": rec_out})
    except WebSocketDisconnect:
        return


def main() -> None:
    import uvicorn
    uvicorn.run("kairos.api:app", host=os.environ.get("HOST", "0.0.0.0"), port=int(os.environ.get("PORT", "8000")))
