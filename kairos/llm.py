"""Optional LLM polish (OFF by default). Any OpenAI-compatible endpoint works: Ollama
(http://localhost:11434/v1), vLLM, LM Studio, OpenAI, Groq, or Gemini's OpenAI-compatible endpoint.

The LLM only *rewrites* the grounded extractive draft into fluent speech. It sees nothing but the
evidence sentences and their IDs, must cite after every sentence, and every sentence it produces is
re-verified against the cited chunk (numbers must match, >= 60% content-word support, citation must
be one of the provided IDs). Unsupported sentences are dropped and replaced by the extractive claim,
so the LLM can improve fluency but can never add an ungrounded fact or a fabricated citation.
"""
from __future__ import annotations

import json
import re
import time
from typing import Iterator

import httpx

from .synthesize import Draft, split_cited_sentences, verify_claim

SYSTEM = ("You are a voice assistant answering strictly from the evidence provided. Rewrite the evidence into a "
          "short, natural spoken answer. Rules: (1) use ONLY facts stated in the evidence; (2) end EVERY sentence "
          "with its citation in square brackets exactly as given, e.g. [Doc_12 §2]; (3) never invent citations; "
          "(4) keep numbers exactly as written; (5) at most one sentence per evidence item; (6) no preamble.")


def build_prompt(question: str, draft: Draft) -> list[dict]:
    ev = "\n".join(f"- [{c.chunk_id}] {c.context + ' — ' if c.context else ''}{c.text}" for c in draft.claims)
    unc = ("\nUnverified (say you could not confirm these): " + " ".join(draft.uncertainty)) if draft.uncertainty else ""
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Question: {question}\nEvidence:\n{ev}{unc}\nAnswer:"}]


def approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


class LLMClient:
    def __init__(self, base_url: str, model: str, api_key: str = "", timeout: float = 20.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"
        self.timeout = timeout

    def stream(self, messages: list[dict]) -> Iterator[str]:
        body = {"model": self.model, "messages": messages, "stream": True, "temperature": 0}
        with httpx.stream("POST", f"{self.base_url}/chat/completions", headers=self.headers, json=body,
                          timeout=self.timeout) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    delta = json.loads(data)["choices"][0]["delta"].get("content") or ""
                except (KeyError, IndexError, json.JSONDecodeError):
                    continue
                if delta:
                    yield delta


def polish(client: LLMClient, question: str, draft: Draft, index) -> dict:
    """Stream the LLM rewrite, verifying sentence by sentence. Returns text, timing and token counts."""
    messages = build_prompt(question, draft)
    allowed = {c.chunk_id for c in draft.claims}
    t0 = time.perf_counter()
    first_token_ms = None
    buf, out_sents, raw = "", [], ""
    dropped = 0
    try:
        for tok in client.stream(messages):
            if first_token_ms is None:
                first_token_ms = (time.perf_counter() - t0) * 1000
            raw += tok
            buf += tok
    except Exception as exc:   # network / provider failure -> extractive answer, never a broken turn
        return {"text": draft.text(), "fallback": f"llm_error: {exc.__class__.__name__}", "dropped": 0,
                "first_token_ms": None, "input_tokens": approx_tokens(json.dumps(messages)), "output_tokens": 0}
    for sent, cits in split_cited_sentences(raw):
        if not cits or any(c not in allowed for c in cits):
            dropped += 1
            continue
        ok = all(verify_claim(sent, index.chunks[index.by_id[c]].text, index.chunks[index.by_id[c]].heading)[0] for c in cits)
        if ok:
            out_sents.append(f"{sent} " + " ".join(f"[{c}]" for c in cits))
        else:
            dropped += 1
    cited = {c for s in out_sents for c in re.findall(r"\[([^\]]+§[^\]]+)\]", s)}
    # anything the LLM dropped or mangled is restored from the extractive draft
    for c in draft.claims:
        if c.chunk_id not in cited:
            out_sents.append(c.cited())
    text = " ".join(out_sents)
    if draft.uncertainty:
        text += " " + " ".join(draft.uncertainty)
    return {"text": text.strip(), "fallback": None, "dropped": dropped, "first_token_ms": first_token_ms,
            "input_tokens": approx_tokens(json.dumps(messages)), "output_tokens": approx_tokens(raw)}
