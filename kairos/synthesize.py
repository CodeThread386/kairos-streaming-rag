"""Grounded answer composition.

Extractive-first: every sentence in an answer is a sentence (or a heading-contextualised sentence)
taken from a retrieved chunk, followed by its citation ``[Doc_ID §Section]``. Fabricated citations
are therefore structurally impossible in the default mode. A sub-intent whose best evidence does
not cover its key terms is reported as *unverified* instead of being answered.

The optional LLM mode (llm.py) rewrites the grounded draft into fluent speech; every LLM sentence
is re-verified against the cited chunks with ``verify_claim`` before it is emitted.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from .decompose import GENERIC_ASPECT, REQUEST_WORDS, SubIntent
from .index import HybridIndex, Hit, capacity_numbers, embed
from .text import STOPWORDS, content_tokens, numbers_in, stem

LIST_CUES = re.compile(r"\b(?:which|options|venues|vendors|caterers|rooms|choices|alternatives|places)\b", re.I)
QUANT = {stem(w) for w in """many much long soon quickly often far early late how big large small rupee rupees rs inr
dollar dollars usd euro euros amount""".split()}
POINTER = re.compile(r"\b(?:are|is) (?:covered|listed|described|explained|set out|given|defined|found) in\b|"
                     r"\b(?:are|is) in the [A-Z]|\bsee the\b|\brefer to\b|\bfollow(?:s)? the [A-Z]", re.I)
QTYPE_NUM = re.compile(r"\b(?:how (?:many|much|long|soon|far|early|quickly|often)|when|what(?:'s| is) the (?:limit|cap|fee|rate|cost|"
                       r"budget|minimum|maximum|allowance|deadline|charge|amount|per diem)|in advance|minimum|maximum|deadline|limit|cap|"
                       r"how much notice)\b", re.I)
QTYPE_WHO = re.compile(r"\bwho\b|\bapprov", re.I)
ROLE_SENT = re.compile(r"\b(?:approv\w*|decid\w*|responsib\w*|authori[sz]\w*|signs?|owner)\b", re.I)
UNITS = {stem(w) for w in """people persons person guests attendees participants pax delegates seats members
fit fits hold holds seat seating accommodate accommodates host hosts capacity""".split()}
DEFINITION = re.compile(r"^(?:an?|the)?\s*[\w\- ]{2,40}? (?:is|are|means|refers to) (?:any|a|an|one|the)\b|\bis treated as\b", re.I)
EXCEPTION_CUES = re.compile(r"\b(?:only|except|exception|requires?|required|must|mandatory|not allowed|need|needs)\b", re.I)


@dataclass
class Claim:
    text: str
    chunk_id: str
    sub_intent: int
    score: float
    status: str = "new"          # new | preserved | added | superseded
    context: str = ""            # section heading shown before the sentence when it names the subject

    def cited(self, with_context: bool = True) -> str:
        body = f"{self.context} — {self.text}" if (self.context and with_context) else self.text
        return f"{body} [{self.chunk_id}]"

    def to_dict(self) -> dict:
        return {"text": self.text, "context": self.context, "citation": self.chunk_id, "sub_intent": self.sub_intent,
                "score": round(self.score, 3), "status": self.status}


@dataclass
class Draft:
    claims: list[Claim] = field(default_factory=list)
    uncertainty: list[str] = field(default_factory=list)
    leads: dict = field(default_factory=dict)       # sub_intent -> spoken lead ("Cancellation policy")

    @property
    def citations(self) -> list[str]:
        return list(dict.fromkeys(c.chunk_id for c in self.claims))

    def text(self, with_leads: bool = True) -> str:
        parts = []
        last = None
        show = with_leads and len({c.sub_intent for c in self.claims}) > 1
        for c in self.claims:
            lead = ""
            if show and c.sub_intent != last and self.leads.get(c.sub_intent):
                lead = self.leads[c.sub_intent] + ": "
            sep = "\n" if (show and last is not None and c.sub_intent != last) else ""
            parts.append(sep + lead + c.cited(with_context=not (show and c.context == self.leads.get(c.sub_intent))))
            last = c.sub_intent
        out = " ".join(parts).replace(" \n", "\n")
        if self.uncertainty:
            out = (out + " " if out else "") + " ".join(self.uncertainty)
        return out.strip()


def _lead(si: SubIntent) -> str:
    words = [w for w in re.findall(r"[A-Za-z0-9&'\-]+", si.text)
             if w.lower() not in STOPWORDS and stem(w.lower()) not in REQUEST_WORDS]
    words = [w for w in words if w.lower() not in ("whats", "what's", "tell", "much", "many", "soon", "long")]
    lead = " ".join(words[:5])
    return lead[:1].upper() + lead[1:] if lead else ""


def _context_for(sentence: str, heading: str, index: HybridIndex) -> str:
    """Return the section heading when it *starts with* a named entity that the sentence does not
    mention ("The Lotus Hall seats 40 ..." under "Orchid Business Hotel, Hinjewadi"), else ""."""
    hl = heading.lower().strip()
    ent = next((e for e in sorted(index.entities, key=len, reverse=True) if hl.startswith(e) and len(e) > 3), None)
    if ent is None or ent in sentence.lower():
        return ""
    return heading.strip().rstrip(".")


class Composer:
    def __init__(self, index: HybridIndex, cfg):
        self.ix = index
        self.cfg = cfg
        self.last_diag: list[dict] = []

    # -------------------------------------------------------------------- scoring
    def _specific(self, si: SubIntent) -> list[str]:
        has_count = any(u == "people" for _, u in si.numbers)
        return [s for s in dict.fromkeys(si.aspect_stems) if s not in GENERIC_ASPECT and s not in REQUEST_WORDS
                and s not in QUANT and not (has_count and s in UNITS) and not s.isdigit()]

    def _idf(self, s: str) -> float:
        if not hasattr(self, "_median_idf"):
            vals = sorted(self.ix.idf.values())
            self._median_idf = vals[len(vals) // 2] if vals else 1.0
        return self.ix.idf.get(s, self._median_idf)

    def coverage(self, si: SubIntent, chunk_idx: int) -> tuple[float, int, float]:
        """IDF-weighted share of the sub-intent's specific terms found in the evidence chunk (text,
        heading and document title). Out-of-vocabulary terms carry maximal weight, so asking about
        something the corpus never mentions ("gym memberships", "pets") yields low coverage.
        Returns (coverage, n_covered_specific_terms, oov_fraction). Entities the user named in the
        clause itself must appear in the evidence; entities inherited from context need not."""
        spec = self._specific(si)
        c = self.ix.chunks[chunk_idx]
        hay = self.ix.chunk_tokens[chunk_idx] | set(content_tokens(c.doc_title))
        own = [e for e in si.entities if e not in set(si.inherited)] if getattr(si, "entity_scoped", True) else []
        ent_ok = True
        if own:
            low = (c.doc_title + " " + c.heading + " " + c.text).lower()
            names = own + list(getattr(si, "alt_entities", []))      # "Lotus Hall" is satisfied by its venue
            ent_ok = any(e in low or all(x in hay for x in self.ix.entity_stems.get(e, ("~",))) for e in names)
        if not spec:
            return (1.0 if ent_ok else 0.4), 1, 0.0
        tot = sum(self._idf(x) for x in spec)
        got = sum(self._idf(x) for x in spec if x in hay)
        n_cov = sum(1 for x in spec if x in hay)
        oov = sum(1 for x in spec if not self.ix.in_vocab(x)) / len(spec)
        cov = got / tot if tot else 1.0
        return (cov if ent_ok else cov * 0.5), (n_cov if ent_ok else 0), oov

    def score_sentences(self, si: SubIntent, hits: list[Hit], qv: np.ndarray) -> list[tuple[float, int, int]]:
        if not hits:
            return []
        top = hits[0].score or 1.0
        q_stems = set(content_tokens(si.query))
        c_stems = set(content_tokens(si.context)) - q_stems if si.context else set()
        q_idf = (sum(self._idf(s) for s in q_stems) + 0.5 * sum(self._idf(s) for s in c_stems)) or 1.0
        want_cap = max((v for v, u in si.numbers if u == "people"), default=None)
        spec = set(self._specific(si))
        wants_num = bool(QTYPE_NUM.search(si.text))
        wants_who = bool(QTYPE_WHO.search(si.text))
        out = []
        depth = 5 if any(u == "people" for _, u in si.numbers) else 3
        for h in hits[:depth]:
            if h.score < 0.35 * top:
                continue
            ch = self.ix.chunks[h.idx]
            ctx_toks = set(content_tokens(ch.heading + " " + ch.doc_title))
            head_toks = set(content_tokens(ch.heading))
            head_match = len(spec & head_toks) / len(spec) if spec else 0.0
            names = list(si.entities) + list(getattr(si, "alt_entities", []))
            chunk_names_entity = any(e in (ch.doc_title + " " + ch.heading).lower() for e in names)
            sent_ix = self.ix.chunk_sent_ix.get(h.idx, [])
            for pos, j in enumerate(sent_ix):
                toks = self.ix.sent_tokens[j]
                lex = (sum(self._idf(s) for s in q_stems & toks) +
                       0.5 * sum(self._idf(s) for s in (q_stems & ctx_toks) - toks) +
                       0.5 * sum(self._idf(s) for s in c_stems & toks)) / q_idf
                dense = float(self.ix.sent_emb[j] @ qv)
                sc = 0.42 * dense + 0.38 * lex + 0.20 * (h.score / top)
                sent = self.ix.sent_text[j]
                sc += 0.15 * head_match                     # the section heading names what was asked
                if si.entities:
                    if chunk_names_entity:
                        sc += 0.06
                    elif any(e in sent.lower() for e in si.entities):
                        sc += 0.08
                if POINTER.search(sent):
                    sc -= 0.15                              # "X is covered in the Y document" is not an answer
                if wants_num and re.search(r"\d", sent):
                    sc += 0.08
                if wants_who and ROLE_SENT.search(sent):
                    sc += 0.08
                if want_cap is not None:
                    caps = capacity_numbers(sent)
                    if caps:
                        sc += 0.1 if max(caps) >= want_cap else -0.1
                if len(sent.split()) < 5:
                    sc -= 0.1
                if pos == 0:
                    sc += 0.04          # sections usually state their main rule first
                out.append((sc, h.idx, j))
        out.sort(key=lambda x: -x[0])
        return out

    # -------------------------------------------------------------------- compose
    def compose(self, subs: list[SubIntent], results: list[list[Hit]], exclude: set[str] | None = None,
                index_offset: int = 0, avoid_sentences: set[str] | None = None, prefer_rules: bool = False) -> Draft:
        draft = Draft()
        self.last_diag = []
        exclude = exclude or set()
        qvs = embed([s.query for s in subs]) if subs else np.zeros((0, 256))
        used_sents: set[int] = set()
        for k, (si, hits) in enumerate(zip(subs, results)):
            sid = k + index_offset
            draft.leads[sid] = _lead(si)
            hits = [h for h in hits if h.chunk_id not in exclude] or hits
            ranked = [r for r in self.score_sentences(si, hits, qvs[k]) if r[2] not in used_sents
                      and self.ix.sent_text[r[2]] not in (avoid_sentences or set())]
            if prefer_rules:
                # a late detail is a *statement*; what the user needs is the rule it triggers, not a definition
                adj = []
                for sc, ci, j in ranked:
                    sent = self.ix.sent_text[j]
                    if DEFINITION.search(sent):
                        sc -= 0.12
                    if EXCEPTION_CUES.search(sent):
                        sc += 0.08
                    adj.append((sc, ci, j))
                ranked = sorted(adj, key=lambda r: -r[0])
            if not ranked:
                draft.uncertainty.append(f"I could not find anything in the documents about {si.text.rstrip('?.')}.")
                continue
            best_sc, best_chunk, best_j = ranked[0]
            cov, n_cov, oov = self.coverage(si, best_chunk)
            dense_best = float(self.ix.sent_emb[best_j] @ qvs[k])
            verified = (cov >= self.cfg.min_aspect_coverage and best_sc >= self.cfg.min_evidence_score) or \
                       (n_cov >= 1 and oov <= 0.5 and dense_best >= self.cfg.min_dense_support)
            self.last_diag.append({"sub": si.text, "coverage": round(cov, 3), "covered": n_cov, "oov": round(oov, 2),
                                   "dense": round(dense_best, 3), "score": round(best_sc, 3), "verified": verified,
                                   "evidence": self.ix.chunks[best_chunk].chunk_id})
            if not verified:
                missing = [w for w in re.findall(r"[A-Za-z][A-Za-z\-]+", si.text)
                           if stem(w.lower()) in set(self._specific(si)) and stem(w.lower()) not in self.ix.chunk_tokens[best_chunk]]
                what = " ".join(dict.fromkeys(missing)) or si.text.rstrip("?.")
                ent = f" for {', '.join(e.title() for e in si.entities)}" if si.entities else ""
                draft.uncertainty.append(f"I could not verify {what}{ent} from the documents.")
                continue
            picks = [(best_sc, best_chunk, best_j)]
            list_mode = bool(LIST_CUES.search(si.text)) or any(u == "people" for _, u in si.numbers)
            limit = self.cfg.list_mode_max_items if list_mode else self.cfg.max_sentences_per_intent
            want_people = max((v for v, u in si.numbers if u == "people"), default=None)
            if want_people is not None:
                fits = [r for r in ranked if (lambda caps: caps and max(caps) >= want_people)(capacity_numbers(self.ix.sent_text[r[2]]))]
                if fits:
                    ranked = fits + [r for r in ranked if r not in fits]
                    best_sc, best_chunk, best_j = ranked[0]
                    picks = [ranked[0]]
            for sc, ci, j in ranked[1:]:
                if len(picks) >= limit:
                    break
                if want_people is not None and not capacity_numbers(self.ix.sent_text[j]):
                    continue
                if list_mode:
                    if ci not in {p[1] for p in picks} and sc >= 0.72 * best_sc:
                        picks.append((sc, ci, j))
                else:
                    # a second sentence from the same section when it adds detail (e.g. a fee and a deadline)
                    if ci == best_chunk and sc >= 0.72 * best_sc and abs(j - best_j) == 1:
                        picks.append((sc, ci, j))
            picks.sort(key=lambda p: (p[1] != best_chunk, p[2]))
            head = self.ix.chunks[picks[0][1]].heading.strip().rstrip(".")
            if 0 < len(head.split()) <= 6 and not head.lower().startswith(("how ", "what ", "who ", "when ", "can ", "is ", "are ", "do ")):
                draft.leads[sid] = head
            for sc, ci, j in picks:
                used_sents.add(j)
                chunk = self.ix.chunks[ci]
                sent = self.ix.sent_text[j]
                draft.claims.append(Claim(text=sent, chunk_id=chunk.chunk_id, sub_intent=sid, score=sc,
                                          context=_context_for(sent, chunk.heading, self.ix)))
        return draft


# ------------------------------------------------------------------------ verification
def verify_claim(sentence: str, chunk_text: str, heading: str = "") -> tuple[bool, float]:
    """Is a (possibly paraphrased) sentence supported by the cited chunk? All numbers must appear in
    the chunk and most content words must be present (stem match)."""
    s_toks = [t for t in content_tokens(sentence) if not t.isdigit()]
    c_toks = set(content_tokens(chunk_text + " " + heading))
    if numbers_in(sentence) - numbers_in(chunk_text + " " + heading):
        return False, 0.0
    if not s_toks:
        return True, 1.0
    overlap = sum(1 for t in s_toks if t in c_toks) / len(s_toks)
    return overlap >= 0.6, overlap


_CIT = re.compile(r"\[([^\[\]]+?§[^\[\]]+?)\]")


def split_cited_sentences(text: str) -> list[tuple[str, list[str]]]:
    """Split an answer into (sentence, [citations]) pairs."""
    out = []
    for raw in re.split(r"(?<=\])\s+(?=\S)|(?<=[.!?])\s+(?=[A-Z])", text):
        cits = _CIT.findall(raw)
        sent = _CIT.sub("", raw).strip()
        sent = re.sub(r"^[A-Z][A-Za-z0-9 &'\-]{0,40}:\s+", "", sent) if cits else sent
        if sent:
            out.append((sent, [c.strip() for c in cits]))
    return out
