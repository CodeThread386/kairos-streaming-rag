"""Command line interface.

    python -m kairos serve                       # API + live console on :8000
    python -m kairos ask "utterance" [--corpus corpus/tessera]
    python -m kairos replay transcript.jsonl     # timestamped chunks -> structured output records
    python -m kairos index [--corpus PATH]       # (re)build the index and print its cost
    python -m kairos bench [--quick]             # full benchmark -> reports/
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def _records_from_file(path: Path) -> dict:
    """Accepts (a) a scenario JSON/JSONL with "turns", or (b) a JSONL transcript of chunk events
    {"t": 0.8, "text": "..."} where {"type": "end", "t": 2.1} (or "utterance_end") closes a turn."""
    lines = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if len(lines) == 1 and "turns" in lines[0]:
        return lines[0]
    if lines and "turns" in lines[0]:
        return lines[0]
    turns, cur = [], []
    for ev in lines:
        kind = ev.get("type") or ev.get("event")
        if kind in ("end", "utterance_end", "eou") or ev.get("end") is True:
            turns.append({"utterance": " ".join(c["text"] for c in cur), "chunks": cur, "end_t": float(ev.get("t", ev.get("timestamp_s", 0)))})
            cur = []
        else:
            cur.append({"t": float(ev.get("t", ev.get("timestamp_s", 0.0))), "text": ev.get("text") or ev.get("chunk") or ""})
    if cur:
        turns.append({"utterance": " ".join(c["text"] for c in cur), "chunks": cur, "end_t": cur[-1]["t"] + 0.5})
    return {"id": path.stem, "turns": turns}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="kairos")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve")
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--port", type=int, default=8000)
    a = sub.add_parser("ask")
    a.add_argument("utterance", nargs="+")
    a.add_argument("--corpus", default=None)
    r = sub.add_parser("replay")
    r.add_argument("path")
    r.add_argument("--corpus", default=None)
    r.add_argument("--asr", action="store_true", help="lower-case and strip punctuation like raw ASR")
    i = sub.add_parser("index")
    i.add_argument("--corpus", default=None)
    b = sub.add_parser("bench")
    b.add_argument("--quick", action="store_true")
    args = ap.parse_args(argv)

    if args.cmd == "serve":
        import uvicorn
        uvicorn.run("kairos.api:app", host=args.host, port=args.port)
        return
    if args.cmd == "bench":
        from bench.run_bench import main as bench_main
        sys.argv = ["run_bench"] + (["--quick"] if args.quick else [])
        bench_main()
        return
    from .config import Config
    cfg = Config(**({"corpus_path": args.corpus} if getattr(args, "corpus", None) else {}))
    if args.cmd == "index":
        from .index import build_index
        t0 = time.perf_counter()
        ix = build_index(cfg.corpus_path, cfg, use_cache=False)
        print(json.dumps({"corpus": cfg.corpus_path, "seconds": round(time.perf_counter() - t0, 3), **ix.stats()}, indent=2))
        return
    from .engine import Engine
    from .replay import run_session
    eng = Engine(cfg)
    if args.cmd == "ask":
        scen = {"id": "cli", "turns": [{"utterance": " ".join(args.utterance), "gold": None}]}
        out = run_session(eng, scen)
    else:
        scen = _records_from_file(Path(args.path))
        for t in scen["turns"]:
            t.setdefault("gold", None)
        out = run_session(eng, scen, asr_style=args.asr)
    keep = ("turn_id", "turn_type", "retrieval_required", "reason", "retrieval_events", "sub_queries", "answer",
            "citations", "uncertainty", "telemetry")
    for t in out["turns"]:
        print(json.dumps({k: t[k] for k in keep if k in t}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
