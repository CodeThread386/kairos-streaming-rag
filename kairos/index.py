"""Hybrid index: BM25 (sparse) + WordLlama static embeddings (dense) over section-level chunks,
plus corpus-derived lexicons that the controller and decomposer use instead of hard-coded rules:

* ``idf``            - informativeness of every stemmed token in the corpus
* ``entities``       - multi-word proper names harvested from titles/headings/text (lower-cased)
* ``scope_entities`` - entities that appear in document titles (e.g. a city a catalog is scoped to)
* ``aspect_vocab``   - stems that occur in section headings (what a section is *about*)
* ``title_vocab``    - stems that occur in document titles (topic nouns)
"""
from __future__ import annotations

import math
import pickle
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .ingest import Chunk, corpus_fingerprint, load_corpus
from .text import STOPWORDS, content_tokens, numbers_in, raw_tokens, stem

_EMBEDDER = None


def get_embedder():
    """WordLlama ships its 256-d weights inside the pip wheel, so this works fully offline."""
    global _EMBEDDER
    if _EMBEDDER is None:
        import wordllama
        from wordllama import WordLlama
        pkg = Path(wordllama.__file__).parent
        try:
            _EMBEDDER = WordLlama.load(cache_dir=pkg, disable_download=True)
        except Exception:
            _EMBEDDER = WordLlama.load()
    return _EMBEDDER


def embed(texts: list[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, 256), dtype=np.float32)
    v = get_embedder().embed(texts, norm=True)
    return np.asarray(v, dtype=np.float32)


_CAP_PHRASE = re.compile(r"\b(?:[A-Z][a-zA-Z'\-]+|[A-Z]{2,})(?:\s+(?:of\s+|de\s+|&\s+)?(?:[A-Z][a-zA-Z'\-]+|[A-Z]{2,}))*")
_GENERIC_CAPS = {"the", "a", "an", "this", "these", "those", "every", "each", "all", "any", "if", "for", "when",
                 "where", "how", "what", "who", "employees", "employee", "claims", "claim", "receipts", "events",
                 "event", "approved", "internal", "external", "travel", "domestic", "international", "use",
                 "hotel", "hotels", "lunch", "venue", "venues", "tier", "day", "days", "per", "on", "in", "at",
                 "it", "is", "no", "not", "they", "there", "attendees", "organisers", "guests", "customer",
                 "customers", "both", "one", "two", "three", "you", "your", "we", "our", "note", "trips", "trip"}


_CAP_CUE = re.compile(r"\b(?:seat|seats|seating|hosts?|capacity|accommodates?|fits?|up to|maximum of)\b", re.I)
_CAP_UNIT = re.compile(r"(\d+)\s+(?:people|persons|guests|attendees|participants|pax|delegates|seats)\b", re.I)


def capacity_numbers(text: str) -> list[int]:
    """Numbers that describe how many people something holds ("seats 40 ... 60 in theatre layout")."""
    t = re.sub(r"(?<=\d),(?=\d)", "", text)
    # "a minimum of 25 guests" is an order minimum, not a capacity
    t = re.sub(r"\b(?:minimum|min\.?|at least)\s+(?:of\s+)?\d+\s*\w*", " ", t, flags=re.I)
    out = [int(m.group(1)) for m in _CAP_UNIT.finditer(t)]
    for m in _CAP_CUE.finditer(t):
        window = t[m.end(): m.end() + 90]
        window = re.split(r"[.;]", window)[0]
        out += [int(x) for x in re.findall(r"\b(\d{1,4})\b", window) if not re.search(x + r"\s*(?:%|days?|hours?|km|per cent)", window)]
    return [n for n in out if 0 < n < 5000]


@dataclass
class Hit:
    idx: int
    chunk_id: str
    score: float
    bm25: float = 0.0
    dense: float = 0.0
    heading_sim: float = 0.0
    boosts: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"chunk_id": self.chunk_id, "score": round(self.score, 4), "bm25": round(self.bm25, 3),
                "dense": round(self.dense, 3), "boosts": {k: round(v, 3) for k, v in self.boosts.items()}}


class HybridIndex:
    def __init__(self, chunks: list[Chunk], k1: float = 1.2, b: float = 0.6):
        t0 = time.perf_counter()
        self.chunks = chunks
        self.k1, self.b = k1, b
        self.by_id = {c.chunk_id: i for i, c in enumerate(chunks)}
        # ---- sparse --------------------------------------------------------------------
        self.tfs: list[Counter] = []
        self.lens = np.zeros(len(chunks), dtype=np.float32)
        df: Counter = Counter()
        for i, c in enumerate(chunks):
            toks = (content_tokens(c.doc_title) + content_tokens(" ".join(c.heading_path or [c.heading])) * 2
                    + content_tokens(c.text))
            tf = Counter(toks)
            self.tfs.append(tf)
            self.lens[i] = len(toks)
            df.update(tf.keys())
        n = max(1, len(chunks))
        self.avgdl = float(self.lens.mean()) if len(chunks) else 1.0
        self.idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}
        self.max_idf = max(self.idf.values()) if self.idf else 1.0
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for i, tf in enumerate(self.tfs):
            for t, f in tf.items():
                self.postings[t].append((i, f))
        # ---- dense ---------------------------------------------------------------------
        full = [f"{c.doc_title}. {' > '.join(c.heading_path or [c.heading])}. {c.text}" for c in chunks]
        heads = [f"{c.doc_title}: {c.heading}" for c in chunks]
        self.emb = embed(full)
        self.head_emb = embed(heads)
        # sentence store for grounded composition
        self.sent_owner: list[int] = []
        self.sent_text: list[str] = []
        for i, c in enumerate(chunks):
            for s in c.sentences:
                self.sent_owner.append(i)
                self.sent_text.append(s)
        self.sent_emb = embed(self.sent_text)
        self.sent_tokens = [set(content_tokens(s)) for s in self.sent_text]
        self.chunk_sent_ix: dict[int, list[int]] = defaultdict(list)
        for j, i in enumerate(self.sent_owner):
            self.chunk_sent_ix[i].append(j)
        self.chunk_tokens = [set(tf) for tf in self.tfs]
        self.chunk_numbers = [numbers_in(c.text) for c in chunks]
        self.chunk_capacity = [capacity_numbers(c.text) for c in chunks]
        cap_vocab: Counter = Counter()
        for j, sent in enumerate(self.sent_text):
            if capacity_numbers(sent):
                cap_vocab.update(t for t in self.sent_tokens[j] if not t.isdigit() and t not in STOPWORDS)
        # words that typically describe capacity in THIS corpus (e.g. seat, layout, hall, room)
        ranked_cap = sorted(((n * self.idf.get(w, 0), w) for w, n in cap_vocab.items() if n >= 2 and self.idf.get(w, 0) >= 1.5),
                            reverse=True)
        self.capacity_vocab = [w for _, w in ranked_cap[:5]]
        # ---- lexicons ------------------------------------------------------------------
        self._build_lexicons()
        self.build_seconds = time.perf_counter() - t0

    # ------------------------------------------------------------------ lexicons
    def _build_lexicons(self) -> None:
        """Proper-noun harvesting from section text and headings (titles are excluded so document
        names do not become slots). A phrase is an entity if it is multi-word capitalised, or a single
        capitalised word seen mid-sentence that is not used in lower case more often than capitalised.
        Scope entities are entities that also appear in the scoping part of a document title
        ("Venue Catalog — Pune" -> "pune")."""
        cap_mid: Counter = Counter()
        seen_any: Counter = Counter()
        lower_text = " " + " ".join(re.sub(r"[^a-z0-9& ]", " ", c.text.lower()) for c in self.chunks) + " "
        for c in self.chunks:
            for src in [c.heading] + list(c.sentences):
                for m in _CAP_PHRASE.finditer(src):
                    words = m.group(0).split()
                    start = m.start()
                    while words and words[0].lower() in _GENERIC_CAPS:
                        start += len(words[0]) + 1
                        words = words[1:]
                    while words and words[-1].lower() in ("of", "&"):
                        words = words[:-1]
                    if not words:
                        continue
                    p = " ".join(words).lower()
                    if len(p) < 3 or p in STOPWORDS or len(words) > 5:
                        continue
                    seen_any[p] += 1
                    initial = start == 0 or src[:start].rstrip().endswith((".", ":", "!", "?", "—"))
                    if len(words) >= 2 or not initial:
                        cap_mid[p] += 1
        ents = set()
        for p, n in cap_mid.items():
            if " " not in p:
                low = max(0, lower_text.count(" " + p + " ") - seen_any[p])
                if low > n:
                    continue
            ents.add(p)
        doc_titles_ = {c.doc_id: c.doc_title for c in self.chunks}
        tv = Counter()
        for t in doc_titles_.values():
            tv.update(set(content_tokens(t)))
        self.entities = ents
        doc_titles = {c.doc_id: c.doc_title.lower() for c in self.chunks}
        scope = set()
        for t in doc_titles.values():
            for seg in re.split(r"\s[—–-]\s|:\s", t)[1:]:
                for e in ents:
                    if re.search(r"\b" + re.escape(e) + r"\b", seg):
                        scope.add(e)
        n_docs = max(1, len(doc_titles))
        self.scope_entities = {e for e in scope if sum(1 for t in doc_titles.values() if e in t) <= max(3, n_docs // 6)}
        # document names made only of title words ("reimbursement rules", "expense claims") are not slots
        self.entities = {e for e in self.entities if e in self.scope_entities or
                         not all(tv.get(x, 0) > 0 for x in content_tokens(e))}
        self.entity_stems = {e: tuple(content_tokens(e)) for e in self.entities if content_tokens(e)}
        self.doc_scope: dict[str, set[str]] = {did: {e for e in self.scope_entities if re.search(r"\b" + re.escape(e) + r"\b", t)}
                                               for did, t in doc_titles.items()}
        self.aspect_vocab = Counter()
        self.title_vocab = Counter()
        for c in self.chunks:
            self.aspect_vocab.update(set(content_tokens(c.heading)))
        for c in {c.doc_id: c for c in self.chunks}.values():
            self.title_vocab.update(set(content_tokens(c.doc_title)))

    # ------------------------------------------------------------------ helpers
    def informativeness(self, stems: list[str], cap: float = 3.0) -> float:
        return sum(min(cap, self.idf[s]) for s in set(stems) if s in self.idf)

    def in_vocab(self, s: str) -> bool:
        return s in self.idf

    def find_entities(self, text: str) -> list[str]:
        """Corpus entities mentioned in text (case-insensitive, longest match first). A distinctive
        first word of a multi-word name also matches ("Lakeview" -> "lakeview convention centre")."""
        stems = content_tokens(text)
        found = []
        joined = " " + " ".join(stems) + " "
        for e, es in sorted(self.entity_stems.items(), key=lambda kv: -len(kv[1])):
            pat = " " + " ".join(es) + " "
            if pat in joined:
                if any(set(es) <= set(self.entity_stems.get(f, ())) for f in found):
                    continue
                found.append(e)
        if not hasattr(self, "_head_index"):
            heads = defaultdict(list)
            for e, es in self.entity_stems.items():
                if len(es) >= 2:
                    heads[es[0]].append(e)
            units = {stem(u) for u in ("people", "person", "guests", "attendees", "participants", "members",
                                         "employees", "days", "hours", "team", "staff", "users")}
            self._head_index = {h: v for h, v in heads.items() if self.idf.get(h, 0) >= 2.5 and h not in units}
        for st in set(stems):
            cands = [e for e in self._head_index.get(st, []) if e not in found]
            if cands and not any(st in self.entity_stems.get(f, ()) for f in found):
                # ambiguous heads ("deccan" -> courtyard / delights) resolve to the most frequent name
                best = max(cands, key=lambda e: (sum(1 for c in self.chunks if e in c.text.lower()), -len(e)))
                found.append(best)
        return found

    # ------------------------------------------------------------------ search
    def bm25(self, q_stems: list[str]) -> np.ndarray:
        scores = np.zeros(len(self.chunks), dtype=np.float32)
        for t in set(q_stems):
            idf = self.idf.get(t)
            if idf is None:
                continue
            qtf = q_stems.count(t)
            for i, f in self.postings[t]:
                denom = f + self.k1 * (1 - self.b + self.b * self.lens[i] / self.avgdl)
                scores[i] += idf * (f * (self.k1 + 1)) / denom * (1 + 0.25 * (qtf - 1))
        return scores

    def search(self, query: str, cfg, entities: list[str] | None = None, numbers: list[tuple[int, str]] | None = None,
               k: int | None = None, mode: str | None = None, context: str = "", context_weight: float = 0.35) -> list[Hit]:
        k = k or cfg.top_k
        mode = mode or cfg.retrieval_mode
        q_stems = content_tokens(query)
        n = len(self.chunks)
        if n == 0:
            return []
        ranks = []
        bm = self.bm25(q_stems) if mode in ("hybrid", "sparse") else np.zeros(n, dtype=np.float32)
        if context and mode in ("hybrid", "sparse"):
            bm = bm + context_weight * self.bm25(content_tokens(context))
        if any(u == "people" for _, u in (numbers or [])) and getattr(self, "capacity_vocab", None) and mode in ("hybrid", "sparse"):
            # corpus-derived capacity expansion: "for 30 people" implies a question about what can hold 30 people
            bm = bm + 0.8 * self.bm25(self.capacity_vocab)
            k = max(k, 6)
        qv = embed([query])[0] if mode in ("hybrid", "dense") else None
        dn = self.emb @ qv if qv is not None else np.zeros(n, dtype=np.float32)
        hd = self.head_emb @ qv if qv is not None else np.zeros(n, dtype=np.float32)
        if mode in ("hybrid", "sparse"):
            ranks.append((cfg.w_bm25, np.argsort(-bm)))
        if mode in ("hybrid", "dense"):
            ranks.append((cfg.w_dense, np.argsort(-dn)))
            ranks.append((cfg.w_heading, np.argsort(-hd)))
        fused = np.zeros(n, dtype=np.float32)
        depth = min(n, 60)
        for w, order in ranks:
            for r, i in enumerate(order[:depth]):
                if mode in ("hybrid", "sparse") and w == cfg.w_bm25 and bm[i] <= 0 and order is ranks[0][1]:
                    break
                fused[i] += w / (cfg.rrf_k + r + 1)
        # normalise so that a document ranked first by every ranker scores 1.0
        best_possible = sum(w / (cfg.rrf_k + 1) for w, _ in ranks) or 1.0
        fused /= best_possible
        # ---- constraint-aware boosts (entities and numbers come from the corpus lexicon) ----
        entities = entities or []
        boosts = defaultdict(dict)
        cand = np.argsort(-fused)[: max(k * 5, 20)]
        req_scope = {e for e in entities if e in self.scope_entities}
        for i in cand:
            c = self.chunks[i]
            if entities:
                hay = (c.doc_title + " " + c.heading + " " + c.text).lower()
                present = [e for e in entities if e in hay or all(s in self.chunk_tokens[i] for s in self.entity_stems.get(e, ("~",)))]
                if present:
                    b = cfg.entity_boost * len(present) / len(entities)
                    boosts[i]["entity"] = b
                scope = self.doc_scope.get(c.doc_id, set())
                if req_scope and scope and not (scope & req_scope) and not present:
                    boosts[i]["scope_conflict"] = -cfg.scope_conflict_penalty
            people = [v for v, u in (numbers or []) if u == "people"]
            if people:
                caps = self.chunk_capacity[i]
                if caps:
                    boosts[i]["numeric"] = cfg.numeric_boost if max(caps) >= max(people) else -cfg.numeric_boost
        final = fused.copy()
        for i, bs in boosts.items():
            final[i] = final[i] * (1 + sum(bs.values()))
        order = np.argsort(-final)[:k]
        return [Hit(idx=int(i), chunk_id=self.chunks[i].chunk_id, score=float(final[i]), bm25=float(bm[i]),
                    dense=float(dn[i]), heading_sim=float(hd[i]), boosts=dict(boosts.get(int(i), {})))
                for i in order if final[i] > 0]

    # ------------------------------------------------------------------ persistence
    def stats(self) -> dict:
        return {"chunks": len(self.chunks), "docs": len({c.doc_id for c in self.chunks}),
                "sentences": len(self.sent_text), "vocab": len(self.idf), "entities": len(self.entities),
                "scope_entities": sorted(self.scope_entities)[:50], "build_seconds": round(self.build_seconds, 3)}


INDEX_VERSION = 5   # bump when index-building code changes so stale caches are not reused


def build_index(corpus_path: str, cfg, use_cache: bool = True) -> HybridIndex:
    fp = corpus_fingerprint(corpus_path)
    cache = Path(cfg.index_cache_dir) / f"index_v{INDEX_VERSION}_{fp}_{cfg.max_passage_words}.pkl"
    if use_cache and cache.exists():
        try:
            with cache.open("rb") as f:
                idx = pickle.load(f)
            idx.loaded_from_cache = True
            return idx
        except Exception:
            pass
    chunks = load_corpus(corpus_path, cfg.max_passage_words)
    idx = HybridIndex(chunks, k1=cfg.bm25_k1, b=cfg.bm25_b)
    idx.fingerprint = fp
    idx.loaded_from_cache = False
    if use_cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        with cache.open("wb") as f:
            pickle.dump(idx, f)
    return idx
