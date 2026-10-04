"""Structured, schema-validated telemetry. Every event carries the session, turn, a virtual stream
timestamp (seconds since the turn's first chunk, as spoken) and a wall-clock offset in ms, so a
trace can be replayed and audited. The JSON Schema lives in docs/telemetry.schema.json."""
from __future__ import annotations

import json
import time
from pathlib import Path

EVENT_TYPES = (
    "session_started", "chunk_received", "controller_decision", "retrieval_started", "retrieval_completed",
    "sub_queries", "synthesis_started", "first_token", "answer_version", "suppression", "uncertainty",
    "turn_summary",
)

REQUIRED_BY_TURN = {
    # turn_type -> event types that must be present for the turn to count as fully traced (G6)
    "new": {"chunk_received", "controller_decision", "retrieval_started", "retrieval_completed", "sub_queries",
            "first_token", "answer_version", "turn_summary"},
    "refinement": {"chunk_received", "controller_decision", "retrieval_started", "retrieval_completed",
                   "sub_queries", "first_token", "answer_version", "turn_summary"},
    "presentation": {"chunk_received", "controller_decision", "suppression", "first_token", "answer_version", "turn_summary"},
    "chitchat": {"chunk_received", "controller_decision", "suppression", "first_token", "turn_summary"},
}


class Tracer:
    def __init__(self, session_id: str, trace_dir: str | None = None, write: bool = True):
        self.session_id = session_id
        self.events: list[dict] = []
        self.t0 = time.perf_counter()
        self.seq = 0
        self.turn_id = 0
        self.path = None
        if write and trace_dir:
            Path(trace_dir).mkdir(parents=True, exist_ok=True)
            self.path = Path(trace_dir) / f"{session_id}.jsonl"
            self.path.write_text("")
        self.listeners = []

    def wall_ms(self) -> float:
        return round((time.perf_counter() - self.t0) * 1000, 3)

    def emit(self, event: str, t: float | None = None, **fields) -> dict:
        assert event in EVENT_TYPES, event
        self.seq += 1
        rec = {"event": event, "session_id": self.session_id, "turn_id": self.turn_id, "seq": self.seq,
               "t_stream_s": None if t is None else round(float(t), 3), "wall_ms": self.wall_ms()}
        rec.update(fields)
        self.events.append(rec)
        if self.path is not None:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        for cb in self.listeners:
            cb(rec)
        return rec

    def turn_events(self, turn_id: int) -> list[dict]:
        return [e for e in self.events if e["turn_id"] == turn_id]


def coverage_ok(events: list[dict], turn_type: str) -> tuple[bool, list[str]]:
    have = {e["event"] for e in events}
    need = REQUIRED_BY_TURN.get(turn_type, REQUIRED_BY_TURN["new"])
    missing = sorted(need - have)
    return not missing, missing
