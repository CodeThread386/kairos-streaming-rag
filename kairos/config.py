"""Runtime configuration. Every knob used by the ablation study lives here so that the
benchmark can switch architecture variants without code changes."""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field


def _env(name: str, default):
    raw = os.environ.get(name)
    if raw is None:
        return default
    if isinstance(default, bool):
        return raw.strip().lower() in ("1", "true", "yes", "on")
    if isinstance(default, int):
        return int(raw)
    if isinstance(default, float):
        return float(raw)
    return raw


@dataclass
class Config:
    # --- corpus / index -------------------------------------------------------------
    corpus_path: str = field(default_factory=lambda: _env("KAIROS_CORPUS", "corpus/tessera"))
    index_cache_dir: str = field(default_factory=lambda: _env("KAIROS_INDEX_CACHE", ".kairos_cache"))
    max_passage_words: int = 170          # long sections are split into passages (§N.1, §N.2 ...)

    # --- retrieval --------------------------------------------------------------------
    retrieval_mode: str = "hybrid"        # hybrid | dense | sparse
    top_k: int = 4                        # hits kept per sub-query
    rrf_k: int = 20                       # reciprocal-rank-fusion constant
    bm25_k1: float = 1.2
    bm25_b: float = 0.6
    w_bm25: float = 1.0                   # RRF weights per ranker
    w_dense: float = 1.0
    w_heading: float = 0.6
    entity_boost: float = 0.35            # requested entity present in chunk
    scope_conflict_penalty: float = 0.45  # chunk's doc title scoped to a *different* entity
    numeric_boost: float = 0.30           # "N people" constraint satisfied by chunk

    # --- controller -------------------------------------------------------------------
    controller_mode: str = "streaming"    # streaming | eager | end_only
    min_informativeness: float = 3.2      # summed (capped) IDF of in-vocabulary content tokens
    provisional_min_informativeness: float = 4.5
    upgrade_min_new_tokens: int = 1
    reuse_jaccard: float = 0.6            # final sub-query reuses a speculative retrieval at/above this

    # --- decomposition / refinement --------------------------------------------------
    decomposition: bool = True
    merge_jaccard: float = 0.6            # over-fragmentation guard
    refinement: str = "refine"            # refine | restart

    # --- synthesis ----------------------------------------------------------------------
    max_sentences_per_intent: int = 2
    list_mode_max_items: int = 3
    min_evidence_score: float = 0.30      # below this a sub-intent is flagged as unverified
    min_aspect_coverage: float = 0.4
    min_dense_support: float = 0.55       # semantic fallback when some query words are out-of-vocabulary
    synthesis_mode: str = field(default_factory=lambda: _env("KAIROS_SYNTHESIS", "extractive"))  # extractive | llm

    # --- optional LLM (OpenAI-compatible: Ollama, vLLM, OpenAI, Gemini, Groq ...) -----
    llm_base_url: str = field(default_factory=lambda: _env("KAIROS_LLM_BASE_URL", ""))
    llm_model: str = field(default_factory=lambda: _env("KAIROS_LLM_MODEL", ""))
    llm_api_key: str = field(default_factory=lambda: _env("KAIROS_LLM_API_KEY", ""))
    llm_timeout_s: float = 20.0
    price_in_per_mtok: float = field(default_factory=lambda: _env("KAIROS_PRICE_IN_PER_MTOK", 0.0))
    price_out_per_mtok: float = field(default_factory=lambda: _env("KAIROS_PRICE_OUT_PER_MTOK", 0.0))

    # --- telemetry ----------------------------------------------------------------------
    trace_dir: str = field(default_factory=lambda: _env("KAIROS_TRACE_DIR", "traces"))
    write_traces: bool = field(default_factory=lambda: _env("KAIROS_WRITE_TRACES", True))

    def variant(self, **overrides) -> "Config":
        d = asdict(self)
        d.update(overrides)
        return Config(**d)

    def public(self) -> dict:
        d = asdict(self)
        d["llm_api_key"] = "***" if self.llm_api_key else ""
        return d

    @property
    def llm_enabled(self) -> bool:
        return self.synthesis_mode == "llm" and bool(self.llm_base_url and self.llm_model)
