"""Multi-intent decomposition with context propagation.

Pipeline for one (partial or final) utterance:
  1. disfluency repair  - "per diem in the UK, sorry, I mean the United States" -> "... in the United States"
  2. filler stripping   - um / uh / you know / like (as filler)
  3. clause segmentation - sentence punctuation, discourse connectors ("also", "as well as", "plus",
                           "along with"), "and"/"," followed by a new clause or determiner-led noun phrase
  4. role tagging       - request clauses vs context-only clauses ("For international travel, ...",
                           "I'm flying to Berlin next month.")
  5. distributive split - "hotel limits in Pune and Bengaluru" -> one sub-intent per entity
  6. propagation        - entities / topic words shared across the utterance are attached to every
                           sub-query that lacks them; aspect-less clauses ("what about Tier A cities?")
                           inherit the previous aspect (ellipsis)
  7. over-fragmentation guard - near-duplicate sub-intents are merged
All lexical knowledge (entities, topic words, IDF) comes from the corpus index.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .text import DANGLING, STOPWORDS, content_tokens, jaccard, normalise_numbers, raw_tokens, stem, strip_fillers

REQUEST_WORDS = {stem(w) for w in """need want know tell give show find get plan wonder mention sure say explain
summarize summarise describe list check confirm look help understand ask"""
                 .split()}
GENERIC_ASPECT = {stem(w) for w in """rule rules policy policies option options term terms detail details information
info process procedure requirement requirements allowed allow available availability thing things situation case
question answer work happen happens apply applies guideline guidelines""".split()}
GENERIC_TITLE = {stem(w) for w in """policy policies guideline guidelines overview scope faq directory catalog
catalogue terms matrix support rules rule handbook manual guide playbook introduction general""".split()}
THIN = 6.0  # clauses below this informativeness borrow topic words from the rest of the utterance
WH = r"(?:what|what's|whats|how|when|where|who|which|whether|why|is there|are there)"
AUX = r"(?:do|does|did|is|are|was|were|can|could|should|will|would|may|must|have|has)"
REQ_START = re.compile(rf"^\s*(?:{WH}|{AUX}\s+(?:i|we|you|they|the|my|our|it|there|employees?)|tell|give|show|list|find|explain|summari[sz]e|describe|i\s+(?:need|want|would like|'d like)|we\s+(?:need|want)|please)\b", re.I)
REPAIR = re.compile(r",?\s*(?:sorry|oops)?\s*,?\s*\b(?:i mean|i meant|no wait|or rather|rather|correction)\b\s*,?\s*", re.I)
NUM_UNIT = re.compile(r"\b(\d+(?:\.\d+)?)\s*(?:-\s*)?(people|persons|person|guests|attendees|participants|pax|delegates|seats|members|employees|days|nights|hours|percent|%)\b", re.I)


@dataclass
class SubIntent:
    text: str                      # the clause as spoken (cleaned)
    query: str                     # retrieval query after context propagation
    aspect_stems: list[str]
    entities: list[str]
    numbers: list[tuple[float, str]]
    inherited: list[str] = field(default_factory=list)
    key: str = ""
    context: str = ""              # weakly-weighted topic words propagated from the rest of the utterance
    context_weight: float = 0.35
    entity_scoped: bool = True     # False when retrieval found no entity-specific evidence
    alt_entities: list = field(default_factory=list)   # parent entities ("Lotus Hall" -> its venue)

    @property
    def signature(self) -> frozenset:
        """Stems of the strong query only (context words are shared by every sub-query of a turn)."""
        return frozenset(content_tokens(self.query))

    def to_dict(self) -> dict:
        return {"text": self.text, "query": self.query, "context": self.context, "entities": self.entities,
                "numbers": [list(n) for n in self.numbers], "inherited_context": self.inherited}


# --------------------------------------------------------------------------- cleaning
def repair_disfluencies(text: str) -> str:
    """Remove the reparandum of mid-utterance self-repairs. The reparandum is the phrase that the
    repair replaces: we take the noun phrase after the last preposition before the repair marker."""
    for _ in range(3):
        m = REPAIR.search(text)
        if not m or m.start() == 0 or not text[:m.start()].strip():
            break
        left, right = text[:m.start()], text[m.end():]
        if not right.strip():
            break
        rep_m = re.match(r"\s*([^,;.?!]+?)(?=(?:\s*[,;.?!]|\s+and\s+|\s+or\s+|$))", right)
        repair = rep_m.group(1).strip() if rep_m else right.strip()
        rest = right[rep_m.end():] if rep_m else ""
        pm = list(re.finditer(r"\b(?:in|at|for|to|of|on|from|with|about|than)\b", left, re.I))
        cut = pm[-1].end() if pm else max(0, left.rstrip().rfind(" "))
        text = (left[:cut].rstrip() + " " + repair + rest).strip()
    return text


def clean_utterance(text: str) -> str:
    t = repair_disfluencies(text)
    t = strip_fillers(t)
    return re.sub(r"\s+", " ", t).strip()


# --------------------------------------------------------------------------- segmentation
_SPLITTERS = [
    r"\s*[?!;]\s+", r"(?<=[a-z0-9%)])\.\s+(?=[A-Za-z])",
    r",?\s+(?:and\s+)?also\b,?\s*", r",?\s+as well as\s+", r",?\s+along with\s+", r",?\s+plus\s+",
    r",?\s+oh,?\s+and\s+", r",?\s+and\s+(?=(?:what|what's|whats|how|when|where|who|which|whether|why)\b)",
    rf",?\s+and\s+(?={AUX}\s+(?:i|we|you|they|the|my|our|it|there|employees?)\b)",
    r",?\s+and\s+(?=(?:are|is|do|does|can|will|should)\s+[a-z]+\s+[a-z]+)",
    r",?\s+and\s+(?=(?:i|we)\s+(?:need|want|also|would)\b)",
    r",?\s+and\s+(?=(?:the|a|an|any|my|our|your|their)\s)",
    r",\s+(?=(?:what|what's|whats|how|when|where|who|which|whether|why|and what|and how)\b)",
    r",\s+(?=(?:the|any|my|our|your|their)\s+\w+\s+\w+)",
]
_SPLIT_RE = re.compile("|".join(f"(?:{s})" for s in _SPLITTERS), re.I)


_PREP_AFTER = r"(?:during|on|for|in|at|when|while|if|after|before|with|from|abroad|there|\?|$)"


def segment_clauses(text: str, aspect_ok=None) -> list[str]:
    parts = [p.strip(" ,.?!;") for p in _SPLIT_RE.split(text) if p and p.strip(" ,.?!;")]
    if aspect_ok is None:
        return parts
    out = []
    for p in parts:
        # bare-noun coordination "tips and laundry during trips": split only when the second conjunct is
        # NOT followed by a shared head noun ("cancellation and refund terms" stays one intent)
        m = re.search(r"\b([a-z][a-z\-]+)\s+and\s+([a-z][a-z\-]+)(?=\s+" + _PREP_AFTER + r"|\s*$)", p, re.I)
        if m and aspect_ok(m.group(1)) and aspect_ok(m.group(2)):
            left = p[:m.end(1)].strip()
            right = p[m.start(2):].strip()
            out += [left, right]
        else:
            out.append(p)
    return out


_MARKER = re.compile(r"^(?:(?:and|also|oh|plus|so|but|then|now|well|okay|ok|and also|oh and|as well as|along with)\b[\s,]*)+", re.I)


def strip_marker(clause: str) -> str:
    return _MARKER.sub("", clause.strip()).strip(" ,")


def is_request_clause(clause: str) -> bool:
    return bool(REQ_START.search(clause)) or clause.rstrip().endswith("?")


def ends_dangling(text: str) -> bool:
    toks = re.findall(r"[a-zA-Z0-9']+", text.lower())
    return not toks or toks[-1] in DANGLING


def _dedupe_words(base: str, extra: list[str]) -> str:
    """Append context words to a clause, skipping words whose stem the clause already has."""
    seen = {stem(w.lower().strip(",.?!")) for w in base.split()}
    out = []
    for w in " ".join(extra).split():
        k = stem(w.lower().strip(",.?!"))
        if k in seen and k not in STOPWORDS:
            continue
        seen.add(k)
        out.append(w)
    return (base + " " + " ".join(out)).strip()


# --------------------------------------------------------------------------- decomposition
class Decomposer:
    def __init__(self, index, cfg):
        self.ix = index
        self.cfg = cfg

    # slot extraction --------------------------------------------------------------------
    def numbers(self, text: str) -> list[tuple[float, str]]:
        t = normalise_numbers(text)
        out = []
        for m in NUM_UNIT.finditer(t):
            unit = m.group(2).lower()
            if unit in ("people", "persons", "person", "guests", "attendees", "participants", "pax", "delegates", "seats", "members", "employees"):
                unit = "people"
            out.append((float(m.group(1)), unit))
        return out

    def _aspect_word(self, w: str) -> bool:
        st = stem(w.lower())
        return (w.lower() not in STOPWORDS and st not in GENERIC_ASPECT and st not in REQUEST_WORDS and
                bool(self.ix.aspect_vocab.get(st)) and self.ix.idf.get(st, 0) > 1.5)

    def aspect_stems(self, clause: str, entities: list[str]) -> list[str]:
        ent_stems = {s for e in entities for s in self.ix.entity_stems.get(e, ())}
        out = []
        for s in content_tokens(clause):
            if s in ent_stems or s in REQUEST_WORDS or s.isdigit():
                continue
            out.append(s)
        return out

    def specific_stems(self, stems: list[str]) -> list[str]:
        return [s for s in stems if s not in GENERIC_ASPECT]

    def informativeness(self, text: str) -> float:
        return self.ix.informativeness([s for s in content_tokens(text) if s not in REQUEST_WORDS])

    def topic_words(self, clause: str) -> list[str]:
        """Surface words of a clause that name the topic (stem occurs in document titles or headings)."""
        words = []
        for w in raw_tokens(clause):
            if w in STOPWORDS or w.isdigit():
                continue
            s = stem(w)
            if s in REQUEST_WORDS:
                continue
            if self.ix.title_vocab.get(s) and s not in GENERIC_TITLE and s not in GENERIC_ASPECT:
                words.append(w)
        return list(dict.fromkeys(words))

    # main --------------------------------------------------------------------------------
    def roles(self, clauses: list[str]) -> tuple[list[str], list[str]]:
        """Clauses before the first request clause are context ("For international travel, ...",
        "I'm flying to Berlin next month."); everything from the first request on is a request,
        including coordinated noun-phrase fragments ("... and the catering options")."""
        first = next((i for i, c in enumerate(clauses) if is_request_clause(c)), None)
        if first is None:
            return [], clauses
        return clauses[:first], clauses[first:]

    def decompose(self, utterance: str, enable: bool | None = None) -> list[SubIntent]:
        enable = self.cfg.decomposition if enable is None else enable
        text = clean_utterance(utterance)
        if not text:
            return []
        if not enable:
            ents = self.ix.find_entities(text)
            return [SubIntent(text=text, query=text, aspect_stems=self.aspect_stems(text, ents), entities=ents,
                              numbers=self.numbers(text), key="0")]
        clauses = [strip_marker(c) for c in segment_clauses(text, self._aspect_word)]
        clauses = [c for c in clauses if c]
        context, requests = self.roles(clauses or [text])
        ctx_text = " ".join(context)
        ctx_entities = self.ix.find_entities(ctx_text) if ctx_text else []
        ctx_topic = self.topic_words(ctx_text) if ctx_text else []
        first_topic = self.topic_words(requests[0]) if requests else []
        subs: list[SubIntent] = []
        prev_entities: list[str] = list(ctx_entities)
        prev_aspect_words: list[str] = []
        for ci, clause in enumerate(requests):
            own = self.ix.find_entities(clause)
            for gi, (cl, g_ents) in enumerate(self._distribute(clause, own)):
                own_aspects = self.aspect_stems(cl, g_ents)
                own_spec = [x for x in self.specific_stems(own_aspects) if self.ix.in_vocab(x)]
                inherited: list[str] = []
                parts = [cl]
                ents = list(g_ents)
                inherited_ents: list[str] = []
                if not ents and prev_entities:
                    ents = list(prev_entities)
                    inherited_ents = list(prev_entities)     # weak context, not part of the strong query
                    inherited += ents
                cl_stems = set(content_tokens(cl))
                thin = self.informativeness(cl) < THIN
                topic = [w for w in dict.fromkeys(ctx_topic + (first_topic if ci > 0 and thin else [])) if stem(w) not in cl_stems]
                ellipsis = bool(re.match(r"^(?:and\s+)?(?:what|how)\s+about\b|^(?:and\s+)?(?:for|in)\s+", cl, re.I))
                if (not own_spec or ellipsis) and prev_aspect_words:   # ellipsis: "what about Tier A cities?" -> strong
                    parts.append(" ".join(prev_aspect_words))
                    inherited += prev_aspect_words
                nums = self.numbers(cl) or self.numbers(ctx_text)
                query = _dedupe_words(parts[0], parts[1:])
                context = " ".join([w for w in topic if stem(w) not in set(content_tokens(query))] + inherited_ents)
                inherited += topic
                subs.append(SubIntent(text=cl, query=query, aspect_stems=own_aspects, entities=ents, numbers=nums,
                                      inherited=inherited, key=f"{ci}.{gi}", context=context))
                if own_spec and not ellipsis:
                    prev_aspect_words = [w for w in raw_tokens(cl) if stem(w) in set(own_spec)]
                if g_ents and len(self._distribute(clause, own)) == 1:
                    prev_entities = list(g_ents)
        return self._merge(subs)

    def _distribute(self, clause: str, ents: list[str]) -> list[tuple[str, list[str]]]:
        if len(ents) < 2:
            return [(clause, ents)]
        low = clause.lower()
        for a in ents:
            for b in ents:
                if a == b:
                    continue
                m = re.search(rf"\b{re.escape(a)}\s*(?:,|and|or|&|as well as)\s*{re.escape(b)}\b", low)
                if m:
                    others = [e for e in ents if e not in (a, b)]
                    base = low[:m.start()] + "{E}" + low[m.end():]
                    return [(base.replace("{E}", a), [a] + others), (base.replace("{E}", b), [b] + others)]
        return [(clause, ents)]

    def _merge(self, subs: list[SubIntent]) -> list[SubIntent]:
        out: list[SubIntent] = []
        for s in subs:
            sig = set(self.specific_stems(s.aspect_stems))
            dup = None
            for o in out:
                osig = set(self.specific_stems(o.aspect_stems))
                if set(o.entities) == set(s.entities) and sig and osig and jaccard(sig, osig) >= self.cfg.merge_jaccard:
                    dup = o
                    break
            if dup is not None:
                dup.text = f"{dup.text}; {s.text}"
                dup.query = f"{dup.query} {s.query}"
                dup.aspect_stems = list(dict.fromkeys(dup.aspect_stems + s.aspect_stems))
                continue
            out.append(s)
        # drop aspect-free fragments when there is something better (e.g. "I need", "so")
        good = [s for s in out if self.informativeness(s.query) >= 1.0]
        return good or out[:1]
