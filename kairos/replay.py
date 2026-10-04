"""Deterministic streaming replay: turns an utterance into timestamped ASR-style chunks and drives a
session through them on a virtual clock (stream timestamps), so runs are reproducible."""
from __future__ import annotations

import json
import random
import re
from pathlib import Path

WORDS_PER_SECOND = 2.6   # conversational speech rate (~155 wpm)


def chunk_utterance(utterance: str, seed: str = "kairos", min_words: int = 2, max_words: int = 5,
                    asr_style: bool = False) -> tuple[list[dict], float]:
    """Split an utterance into ASR-like partial chunks with timestamps (seconds from turn start)."""
    rng = random.Random(seed)
    text = utterance
    if asr_style:
        text = re.sub(r"[^\w\s'%₹$-]", "", text.lower())
    words = text.split()
    chunks, i, t = [], 0, 0.0
    while i < len(words):
        n = rng.randint(min_words, max_words)
        piece = words[i:i + n]
        chunks.append({"t": round(t, 2), "text": " ".join(piece)})
        t += len(piece) / WORDS_PER_SECOND
        i += n
    end_t = round(t + 0.25, 2)     # end-of-utterance detected ~250 ms after the last word
    return chunks, end_t


def turn_chunks(turn: dict, seed: str, asr_style: bool = False) -> tuple[list[dict], float]:
    if turn.get("chunks") and not asr_style:
        return turn["chunks"], float(turn.get("end_t") or (turn["chunks"][-1]["t"] + 0.5))
    return chunk_utterance(turn["utterance"], seed=seed, asr_style=asr_style)


def run_session(engine, scenario: dict, cfg=None, asr_style: bool = False, write_trace: bool = False) -> dict:
    session = engine.new_session(session_id=scenario.get("id"), cfg=cfg, write_trace=write_trace)
    out = {"id": scenario.get("id"), "category": scenario.get("category"), "turns": []}
    for k, turn in enumerate(scenario["turns"]):
        chunks, end_t = turn_chunks(turn, seed=f"{scenario.get('id')}:{k}", asr_style=asr_style)
        session.begin_turn()
        for c in chunks:
            session.feed(c["t"], c["text"])
        rec = session.end(end_t)
        rec["utterance"] = turn.get("utterance")
        rec["chunks"] = chunks
        rec["end_t"] = end_t
        rec["gold"] = turn.get("gold")
        out["turns"].append(rec)
    out["trace"] = session.tracer.events
    return out


def load_scenarios(path: str | Path) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
