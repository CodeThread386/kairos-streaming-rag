"""Kairos streaming engine: retrieval controller + session state + refinement + synthesis.

A session consumes timestamped transcript chunks (``feed``) and an end-of-utterance marker (``end``).
On every chunk the Retrieval Controller decides WAIT / PROVISIONAL / MULTI-INTENT retrieval, or
suppresses retrieval for presentation-only and chit-chat turns. Speculative retrievals fired while the
user is still talking are reused at utterance end when the final decomposition matches them, so the
answer can start (time-to-first-token) within milliseconds of the user finishing.

Late-arriving details (refinements, corrections) patch the previous answer version instead of
restarting: only delta queries are executed, unaffected claims are preserved, affected ones are
superseded, and the new version records its parent and the diff.
"""
from __future__ import annotations

import copy
import re
import time
import uuid
from dataclasses import dataclass, field

from .config import Config
from .decompose import (Decomposer, GENERIC_ASPECT, REQUEST_WORDS, SubIntent, clean_utterance, ends_dangling,
                        segment_clauses, strip_marker)
from .index import HybridIndex, Hit, build_index
from .synthesize import EXCEPTION_CUES, Claim, Composer, Draft
from .telemetry import Tracer
from .text import STOPWORDS, content_tokens, jaccard, raw_tokens, stem

CHITCHAT = re.compile(r"^\s*(?:hi|hello|hey|good (?:morning|afternoon|evening)|thanks?|thank you|thx|cheers|ok(?:ay)?|"
                      r"cool|great|got it|perfect|awesome|nice|bye|goodbye|see you|sounds good|alright|understood|"
                      r"that'?s (?:all|it|helpful|great|perfect)|no(?:pe)?|yes|yeah|sure|wow|hmm)\b", re.I)
CHITCHAT_WORDS = {stem(w) for w in """hi hello hey good morning afternoon evening thanks thank thx cheers okay ok cool great
got perfect awesome nice bye goodbye see sounds alright understood helpful really much lot that's all it no yes yeah
sure wow there everyone""".split()}
PRESENTATION = re.compile(r"\b(?:repeat|again|rephrase|reword|shorter|shorten|brief(?:ly|er)?|concise|simpler|simplify|"
                          r"summari[sz]e (?:that|it|this|your|the (?:above|last|previous))|"
                          r"(?:in|as|into) (?:\w+ )?(?:bullets?|bullet points|points|a list|list form|one line|a sentence|a table|short)|"
                          r"tl;?dr|more slowly|say (?:that|it) again|list (?:those|them|that)|bullet points?)\b", re.I)
PRESENTATION_WORDS = {stem(w) for w in """repeat again rephrase reword shorter shorten brief briefly concise simpler simplify
summarize summarise bullets bullet points point list form one line sentence table short tldr slowly say make give
last answer previous above please can could you that it this those them into as two three four five""".split()}
REFINE_START = re.compile(r"^\s*(?:actually|also|oh,? and|and|but|what about|how about|no,? wait|wait|no,? i meant|"
                          r"i meant|make it|it'?s|it is|it was|it will|we'?ll|we will|we are|we're|they'?re|they are|"
                          r"in that case|what if|instead|plus|oh|the \w+ (?:was|is|will be|were|are))\b", re.I)
QUESTION_START = re.compile(r"^\s*(?:what|what's|whats|how|when|where|who|which|whether|why|is|are|do|does|can|could|"
                            r"should|will|would|tell|give|show|list|find|explain|summari[sz]e|describe)\b", re.I)
NEGATED = re.compile(r"\b(?:not|instead of|rather than|no longer)\s+(?:the\s+|in\s+|at\s+)?([a-z][a-z \-&]+?)(?=[,.?!]|$)", re.I)


@dataclass
class Fired:
    qid: str
    t: float
    sub: SubIntent
    trigger: str
    hits: list[Hit]
    latency_ms: float
    used: bool = False


@dataclass
class AnswerVersion:
    version: int
    parent: int | None
    mode: str                     # answer | refinement | presentation
    claims: list[Claim]
    uncertainty: list[str]
    text: str
    preserved: list[str] = field(default_factory=list)
    added: list[str] = field(default_factory=list)
    superseded: list[str] = field(default_factory=list)

    @property
    def citations(self) -> list[str]:
        return list(dict.fromkeys(c.chunk_id for c in self.claims))

    def to_dict(self) -> dict:
        return {"version": self.version, "parent": self.parent, "mode": self.mode, "text": self.text,
                "citations": self.citations, "claims": [c.to_dict() for c in self.claims],
                "uncertainty": self.uncertainty, "preserved": self.preserved, "added": self.added,
                "superseded": self.superseded}


@dataclass
class TurnState:
    turn_id: int
    chunks: list[dict] = field(default_factory=list)
    text: str = ""
    fired: list[Fired] = field(default_factory=list)
    decisions: list[dict] = field(default_factory=list)
    t_first: float | None = None
    controller_us: list[float] = field(default_factory=list)
    act_stream: str = "new"


class Engine:
    def __init__(self, cfg: Config | None = None, index: HybridIndex | None = None):
        self.cfg = cfg or Config()
        t0 = time.perf_counter()
        self.index = index or build_index(self.cfg.corpus_path, self.cfg)
        self.load_seconds = time.perf_counter() - t0
        from .index import embed
        embed(["warm up the embedder"])                     # first call pays a one-off JIT/alloc cost
        self.index.search("warm up", self.cfg)
        self.decomposer = Decomposer(self.index, self.cfg)
        self.composer = Composer(self.index, self.cfg)
        self.sessions: dict[str, "Session"] = {}

    def new_session(self, session_id: str | None = None, cfg: Config | None = None, write_trace: bool | None = None) -> "Session":
        sid = session_id or uuid.uuid4().hex[:12]
        s = Session(self, sid, cfg or self.cfg, write_trace)
        self.sessions[sid] = s
        return s

    def chunk(self, chunk_id: str) -> dict | None:
        i = self.index.by_id.get(chunk_id)
        return None if i is None else self.index.chunks[i].to_dict()


class Session:
    def __init__(self, engine: Engine, session_id: str, cfg: Config, write_trace: bool | None = None):
        self.engine = engine
        self.ix = engine.index
        self.cfg = cfg
        self.dec = Decomposer(self.ix, cfg)
        self.comp = Composer(self.ix, cfg)
        self.id = session_id
        wt = cfg.write_traces if write_trace is None else write_trace
        self.tracer = Tracer(session_id, cfg.trace_dir, wt)
        self.versions: list[AnswerVersion] = []
        self.last_request: dict | None = None       # {"subs", "hits", "draft"}
        self.turn: TurnState | None = None
        self.turn_count = 0
        self.records: list[dict] = []
        self.tracer.emit("session_started", corpus=cfg.corpus_path, index=self.ix.stats(),
                         config={k: v for k, v in cfg.public().items() if k not in ("trace_dir",)})

    # ================================================================ classification
    def classify(self, text: str) -> str:
        clean = clean_utterance(text)
        if not clean:
            return "empty"
        stems = content_tokens(clean)
        has_prev = bool(self.versions)
        if CHITCHAT.match(clean) and self.ix.informativeness([s for s in stems if s not in CHITCHAT_WORDS]) < 2.5 and "?" not in text:
            return "chitchat"
        if has_prev and PRESENTATION.search(clean):
            rest = [s for s in stems if s not in PRESENTATION_WORDS and not s.isdigit()]
            if self.ix.informativeness(rest) < 3.0:
                return "presentation"
        if has_prev and self.last_request is not None:
            if REFINE_START.match(clean):
                if self._standalone_new(clean):
                    return "new"
                return "refinement"
            if not QUESTION_START.match(clean) and "?" not in text:
                return "refinement"
        return "new"

    def _standalone_new(self, clean: str) -> bool:
        """'Also, what's the per diem in Singapore?' after a venue question is a new request, not a refinement."""
        body = strip_marker(re.sub(r"^\s*(?:actually|also|oh,? and|and|but|plus|oh)\b[\s,]*", "", clean, flags=re.I))
        if not QUESTION_START.match(body) or re.match(r"^\s*(?:what|how) about\b", body, re.I):
            return False
        subs = self.dec.decompose(body)
        if not subs:
            return False
        prev_q = " ".join(s.query for s in self.last_request["subs"])
        from .index import embed
        v = embed([body, prev_q])
        spec = [s for s in self.dec.specific_stems(subs[0].aspect_stems) if self.ix.in_vocab(s)]
        return len(spec) >= 2 and float(v[0] @ v[1]) < 0.35

    # ================================================================ streaming
    def begin_turn(self) -> TurnState:
        self.turn_count += 1
        self.tracer.turn_id = self.turn_count
        self.turn = TurnState(turn_id=self.turn_count)
        return self.turn

    def feed(self, t: float, text: str, replace: bool = False) -> list[dict]:
        """Consume one transcript chunk. ``replace=True`` means ``text`` is the full revised hypothesis
        (browser ASR interim results); otherwise it is appended."""
        if self.turn is None:
            self.begin_turn()
        ts = self.turn
        if ts.t_first is None:
            ts.t_first = t
        ts.text = text if replace else (ts.text + " " + text).strip()
        ts.chunks.append({"t": t, "text": text})
        n0 = len(self.tracer.events)
        self.tracer.emit("chunk_received", t=t, text=text, chars=len(text), replace=replace)
        t0 = time.perf_counter()
        self._control(t)
        ts.controller_us.append((time.perf_counter() - t0) * 1e6)
        return self.tracer.events[n0:]

    def _decide(self, t: float, decision: str, reason: str, **kw) -> None:
        d = {"decision": decision, "reason": reason, **kw}
        self.turn.decisions.append({"t": t, **d})
        self.tracer.emit("controller_decision", t=t, **d)

    def _control(self, t: float) -> None:
        ts, cfg = self.turn, self.cfg
        act = self.classify(ts.text)
        ts.act_stream = act
        if act in ("chitchat", "presentation", "empty"):
            self._decide(t, "WAIT" if act == "empty" else "SUPPRESS", f"{act}: no retrieval needed")
            return
        if cfg.controller_mode == "end_only":
            self._decide(t, "WAIT", "end_only controller: retrieval deferred to utterance end")
            return
        if cfg.controller_mode == "eager":
            q = clean_utterance(ts.text)
            if self.dec.informativeness(q) >= 1.0:
                si = SubIntent(text=q, query=q, aspect_stems=content_tokens(q), entities=self.ix.find_entities(q),
                               numbers=self.dec.numbers(q), key="eager")
                self._fire(t, si, "eager")
                self._decide(t, "RETRIEVE", "eager controller: retrieve on every chunk")
            else:
                self._decide(t, "WAIT", "no informative token yet")
            return
        if act == "refinement":
            subs = self.refinement_plan(ts.text)["delta"]
        else:
            subs = self.dec.decompose(ts.text)
        clean = clean_utterance(ts.text)
        raw_clauses = [strip_marker(c) for c in segment_clauses(clean)] or [clean]
        open_tail = not re.search(r"[.?!]\s*$", ts.text.strip())
        last_clause = raw_clauses[-1].lower().strip(" ,") if raw_clauses else ""
        fired_any = False
        to_fire: list[tuple[SubIntent, bool]] = []          # (sub-intent, is_upgrade)
        for i, si in enumerate(subs):
            is_open = open_tail and i == len(subs) - 1 and si.text.lower().strip(" ,") == last_clause
            if is_open and ends_dangling(ts.text):
                self._decide(t, "WAIT", "intent incomplete: clause ends mid-phrase", clause=si.text)
                continue
            inf = self.dec.informativeness(si.query + (" " + si.context if act == "refinement" else ""))
            spec = [s for s in self.dec.specific_stems(si.aspect_stems) if self.ix.in_vocab(s)]
            if is_open:
                anchored = bool(si.entities or si.numbers or len(spec) >= 2)
                if inf < cfg.provisional_min_informativeness or not anchored:
                    self._decide(t, "WAIT", "intent not yet stable", clause=si.text, informativeness=round(inf, 2))
                    continue
            elif inf < cfg.min_informativeness or not spec:
                self._decide(t, "WAIT", "clause carries too little information", clause=si.text, informativeness=round(inf, 2))
                continue
            sig = si.signature
            prev = max(ts.fired, key=lambda f: jaccard(sig, f.sub.signature), default=None)
            sim = jaccard(sig, prev.sub.signature) if prev else 0.0
            if prev is not None and sim >= 0.85:
                continue
            new_inf = [s for s in sig - (prev.sub.signature if prev else frozenset()) if self.ix.idf.get(s, 0) >= 1.5]
            if prev is not None and sim >= 0.5 and len(new_inf) < cfg.upgrade_min_new_tokens:
                continue
            to_fire.append((si, prev is not None and sim >= 0.5))
        for si, is_upgrade in to_fire:
            if act == "refinement":
                trigger = "refinement"
            elif is_upgrade:
                trigger = "upgrade"
            elif len(to_fire) > 1 or len(ts.fired) > 0:
                trigger = "multi_intent"
            else:
                trigger = "provisional"
            self._fire(t, si, trigger)
            fired_any = True
        if fired_any:
            n = sum(1 for f in ts.fired if f.t == t)
            self._decide(t, "RETRIEVE" if n == 1 else "DECOMPOSE_AND_RETRIEVE",
                         f"{n} stable sub-intent(s) identified before utterance end", sub_queries=[f.sub.query for f in ts.fired if f.t == t])
        elif not ts.decisions or ts.decisions[-1]["t"] != t:
            self._decide(t, "WAIT", "no new stable intent")

    def _fire(self, t: float, si: SubIntent, trigger: str, exclude: set[str] | None = None) -> Fired:
        qid = f"q{self.turn.turn_id}.{len(self.turn.fired) + 1}"
        self.tracer.emit("retrieval_started", t=t, query_id=qid, query=si.query, trigger=trigger, sub_intent=si.to_dict())
        t0 = time.perf_counter()
        hits = entity_conditional_search(self.ix, self.cfg, si, self.comp)
        lat = (time.perf_counter() - t0) * 1000
        f = Fired(qid=qid, t=t, sub=si, trigger=trigger, hits=hits, latency_ms=lat)
        self.turn.fired.append(f)
        self.tracer.emit("retrieval_completed", t=t, query_id=qid, latency_ms=round(lat, 3),
                         hits=[h.to_dict() for h in hits], top_score=round(hits[0].score, 4) if hits else 0.0)
        return f

    # ================================================================ refinement planning
    def _aspect_words(self, si: SubIntent) -> list[str]:
        spec = set(self.dec.specific_stems(si.aspect_stems))
        return [w for w in raw_tokens(si.text) if stem(w) in spec and self.ix.in_vocab(stem(w)) and not w.isdigit()]

    def refinement_plan(self, text: str) -> dict:
        prev = self.last_request
        clean = clean_utterance(text)
        body = re.sub(r"^\s*(?:actually|also|oh,? and|and|but|no,? wait|wait|oh)\b[\s,]*", "", clean, flags=re.I)
        found = self.ix.find_entities(body)
        negated = set()
        for m in NEGATED.finditer(body):
            negated |= set(self.ix.find_entities(m.group(1)))
        positive = [e for e in found if e not in negated]
        nums = self.dec.numbers(body)
        prev_subs: list[SubIntent] = prev["subs"] if prev else []
        prev_ents = [e for s in prev_subs for e in s.entities]
        body_aspects = [s for s in self.dec.specific_stems(self.dec.aspect_stems(body, found)) if self.ix.in_vocab(s)]
        swap_cue = bool(re.search(r"\b(?:what|how) about\b|\bi meant\b|\binstead\b|\bnot\b", body, re.I))
        plan = {"kind": "constraint", "affected": [], "delta": [], "negated": sorted(negated), "new_entities": positive}
        new_pos = [e for e in positive if e not in prev_ents]
        if new_pos and (negated or swap_cue or len(body_aspects) <= 1):
            plan["kind"] = "slot_correction"
            for k, s in enumerate(prev_subs):
                if not s.entities and not negated:
                    continue
                old = [e for e in s.entities if e in negated] or list(s.entities)
                q = s.query
                replaced = False
                for e in old:
                    pat = re.compile(re.escape(e), re.I)
                    if pat.search(q):
                        q = pat.sub(new_pos[0] if not replaced else "", q)
                        replaced = True
                if not replaced:
                    q = " ".join(self._aspect_words(s)) + " " + " ".join(new_pos)
                ns = copy.deepcopy(s)
                ns.query = re.sub(r"\s+", " ", q).strip()
                ns.entities = [e for e in s.entities if e not in old] + new_pos
                ns.text = f"{s.text} [{', '.join(new_pos)}]"
                plan["affected"].append(k)
                plan["delta"].append(ns)
            return plan
        people = [n for n in nums if n[1] == "people"]
        if people and any(any(u == "people" for _, u in s.numbers) for s in prev_subs):
            plan["kind"] = "slot_correction"
            for k, s in enumerate(prev_subs):
                if not any(u == "people" for _, u in s.numbers):
                    continue
                old = next(v for v, u in s.numbers if u == "people")
                ns = copy.deepcopy(s)
                ns.query = re.sub(rf"\b{int(old)}\b", str(int(people[0][0])), s.query)
                ns.numbers = [(people[0][0], "people")]
                ns.text = f"{s.text} [{int(people[0][0])} people]"
                plan["affected"].append(k)
                plan["delta"].append(ns)
            return plan
        # constraint addition / aspect follow-up: each clause becomes a delta query that carries the
        # previous topic (entities + aspect words) so it is retrieved *in context*, not from scratch
        ctx_words: list[str] = []
        for s in prev_subs:
            ctx_words += self._aspect_words(s)
        ctx_ents = list(dict.fromkeys(prev_ents))
        clauses = [strip_marker(c) for c in segment_clauses(body, self.dec._aspect_word)] or [body]
        for ci, cl in enumerate(c for c in clauses if c):
            ents = self.ix.find_entities(cl) or ctx_ents
            cl_stems = set(content_tokens(cl))
            extra = [w for w in dict.fromkeys(ctx_words) if stem(w) not in cl_stems]
            own_ents = self.ix.find_entities(cl)
            inh_ents = [e for e in ctx_ents if e not in own_ents] if not own_ents else []
            si = SubIntent(text=cl, query=re.sub(r"\s+", " ", cl).strip(),
                           aspect_stems=self.dec.aspect_stems(cl, own_ents), entities=own_ents + inh_ents,
                           numbers=self.dec.numbers(cl), inherited=extra[:6] + inh_ents, key=f"r{ci}",
                           context=" ".join(extra[:6] + inh_ents), context_weight=0.6)
            if self.dec.informativeness(si.text) >= 1.0 or len(clauses) == 1:
                plan["delta"].append(si)
        return plan

    # ================================================================ end of utterance
    def end(self, t: float) -> dict:
        ts = self.turn or self.begin_turn()
        t_end_wall = time.perf_counter()
        act = self.classify(ts.text)
        if act == "presentation":
            rec = self._presentation(t, ts)
        elif act in ("chitchat", "empty"):
            rec = self._chitchat(t, ts)
        elif act == "refinement" and self.cfg.refinement == "refine":
            rec = self._answer(t, ts, t_end_wall, refinement=True)
        else:
            rec = self._answer(t, ts, t_end_wall, refinement=False)
        rec["turn_type"] = act if act != "empty" else "chitchat"
        self.records.append(rec)
        self.turn = None
        return rec

    def _match_or_fire(self, t: float, si: SubIntent, trigger: str) -> tuple[Fired, bool]:
        """Reuse a speculative retrieval only if it already covered every informative word of the final
        sub-query (otherwise the late words - "...made *after travel*" - would be ignored)."""
        sig = si.signature
        best = max(self.turn.fired, key=lambda f: jaccard(sig, f.sub.signature), default=None)
        if best is not None and best.t <= t and jaccard(sig, best.sub.signature) >= self.cfg.reuse_jaccard:
            missing = [x for x in sig - best.sub.signature if self.ix.idf.get(x, 0) >= 1.0 or not self.ix.in_vocab(x)]
            if not missing:
                best.used = True
                return best, True
        f = self._fire(t, si, trigger)
        f.used = True
        return f, False

    def _answer(self, t: float, ts: TurnState, t_end_wall: float, refinement: bool) -> dict:
        prev_version = self.versions[-1] if self.versions else None
        plan = None
        if refinement and self.last_request is not None:
            plan = self.refinement_plan(ts.text)
            subs = plan["delta"]
            self._decide(t, "REFINE", f"late-arriving detail ({plan['kind']}): patching answer v{prev_version.version}",
                         affected=plan["affected"], delta_queries=[s.query for s in subs])
        else:
            subs = self.dec.decompose(ts.text)
            self._decide(t, "SYNTHESIZE", "utterance complete", n_sub_intents=len(subs))
        results, reused, used_ids = [], [], []
        for si in subs:
            f, hit = self._match_or_fire(t, si, "refinement" if refinement else "final")
            results.append(f.hits)
            reused.append(hit)
            used_ids.append(f.qid)
        self.tracer.emit("sub_queries", t=t, sub_queries=[s.query for s in subs], speculative_reuse=reused,
                         used_query_ids=used_ids, sub_intents=[s.to_dict() for s in subs])
        self.tracer.emit("synthesis_started", t=t)
        if refinement and plan is not None and prev_version is not None:
            version = self._compose_refinement(plan, subs, results, prev_version)
        else:
            draft = self.comp.compose(subs, results)
            version = AnswerVersion(version=len(self.versions) + 1, parent=prev_version.version if prev_version else None,
                                    mode="answer", claims=draft.claims, uncertainty=draft.uncertainty, text=draft.text(),
                                    added=[c.chunk_id for c in draft.claims])
            self.last_request = {"subs": subs, "hits": results, "draft": draft}
        llm_info = None
        if self.cfg.llm_enabled and version.claims:
            from .llm import LLMClient, polish
            client = LLMClient(self.cfg.llm_base_url, self.cfg.llm_model, self.cfg.llm_api_key, self.cfg.llm_timeout_s)
            d = Draft(claims=version.claims, uncertainty=version.uncertainty)
            llm_info = polish(client, clean_utterance(ts.text), d, self.ix)
            version.text = llm_info["text"]
            self.llm_usage = (llm_info["input_tokens"], llm_info["output_tokens"])
        ttft_ms = (time.perf_counter() - t_end_wall) * 1000
        if llm_info and llm_info.get("first_token_ms") is not None:
            ttft_ms = (llm_info["first_token_ms"] or 0) + 0.0
        self.tracer.emit("first_token", t=t, ttft_ms=round(ttft_ms, 3), first_text=version.text[:60],
                         mode="llm" if llm_info else "extractive",
                         llm=({k: v for k, v in llm_info.items() if k != "text"} if llm_info else None))
        self.versions.append(version)
        self.tracer.emit("answer_version", t=t, **version.to_dict())
        if version.uncertainty:
            self.tracer.emit("uncertainty", t=t, messages=version.uncertainty)
        return self._finish_turn(t, ts, version, subs, reused, ttft_ms, retrieval_required=True)

    def _compose_refinement(self, plan: dict, subs: list[SubIntent], results: list[list[Hit]], prev: AnswerVersion) -> AnswerVersion:
        used_sent_texts = {c.text for c in prev.claims}
        base_offset = 100 + len(self.versions) * 10
        avoid = used_sent_texts if plan["kind"] != "slot_correction" else set()
        draft = self.comp.compose(subs, [[h for h in hits] for hits in results], index_offset=base_offset,
                                  avoid_sentences=avoid, prefer_rules=plan["kind"] != "slot_correction")
        new_claims = [c for c in draft.claims if c.text not in avoid]
        prev_subs_n = len(self.last_request["subs"]) if self.last_request else 0
        if plan["kind"] == "slot_correction":
            affected = set(plan["affected"])
            keep = [copy.copy(c) for c in prev.claims if c.sub_intent not in affected]
            superseded = [c.chunk_id for c in prev.claims if c.sub_intent in affected]
            for c in keep:
                c.status = "preserved"
            for c in new_claims:
                c.status = "added"
            claims = keep + new_claims
            lead = f"Updated for {', '.join(e.title() for e in plan['new_entities'])}: " if plan["new_entities"] else "Updated: "
            text = " ".join(c.cited() for c in keep)
            text = (text + " " if text else "") + lead + " ".join(c.cited() for c in new_claims)
            # the corrected request becomes the new context for later turns
            if self.last_request:
                lr_subs = list(self.last_request["subs"])
                lr_hits = list(self.last_request["hits"])
                for k, ns, hits in zip(plan["affected"], subs, results):
                    if k < len(lr_subs):
                        lr_subs[k], lr_hits[k] = ns, hits
                self.last_request = {"subs": lr_subs, "hits": lr_hits, "draft": self.last_request["draft"]}
        else:
            keep = [copy.copy(c) for c in prev.claims if c.status != "superseded"]
            for c in keep:
                c.status = "preserved"
            for c in new_claims:
                c.status = "added"
            superseded = []
            claims = keep + new_claims
            if not new_claims and not draft.uncertainty:
                draft.uncertainty.append("I found nothing in the documents that changes the earlier answer.")
            head = " ".join(c.cited() for c in keep[:2])
            parts = [f"The earlier answer still applies: {_lower_first(head, self.ix)}" if head else ""]
            for i, c in enumerate(new_claims):
                conn = "However, " if EXCEPTION_CUES.search(c.text) and i == 0 else ("In addition, " if i == 0 else "Also, ")
                parts.append(conn + _lower_first(c.cited(), self.ix))
            text = " ".join(p for p in parts if p)
            if self.last_request:
                self.last_request = {"subs": self.last_request["subs"] + subs, "hits": self.last_request["hits"] + results,
                                     "draft": self.last_request["draft"]}
        unc = draft.uncertainty
        if unc:
            text = text + " " + " ".join(unc)
        return AnswerVersion(version=len(self.versions) + 1, parent=prev.version, mode="refinement", claims=claims,
                             uncertainty=unc, text=text.strip(), preserved=[c.chunk_id for c in claims if c.status == "preserved"],
                             added=[c.chunk_id for c in new_claims], superseded=superseded)

    # ---------------------------------------------------------------- no-retrieval turns
    def _presentation(self, t: float, ts: TurnState) -> dict:
        t0 = time.perf_counter()
        self._decide(t, "SUPPRESS", "presentation_restructure: transform the previous answer, no corpus query")
        prev = self.versions[-1] if self.versions else None
        self.tracer.emit("suppression", t=t, reason="presentation_restructure", retrieval_required=False)
        if prev is None:
            version_text, claims = "There is no previous answer to reformat yet.", []
        else:
            version_text, claims = restructure(prev, clean_utterance(ts.text))
        ttft_ms = (time.perf_counter() - t0) * 1000
        self.tracer.emit("first_token", t=t, ttft_ms=round(ttft_ms, 3), first_text=version_text[:60])
        version = None
        if prev is not None:
            version = AnswerVersion(version=len(self.versions) + 1, parent=prev.version, mode="presentation",
                                    claims=claims, uncertainty=[], text=version_text,
                                    preserved=[c.chunk_id for c in claims])
            self.versions.append(version)
            self.tracer.emit("answer_version", t=t, **version.to_dict())
        return self._finish_turn(t, ts, version, [], [], ttft_ms, retrieval_required=False, text=version_text,
                                 reason="presentation_restructure")

    def _chitchat(self, t: float, ts: TurnState) -> dict:
        t0 = time.perf_counter()
        self._decide(t, "SUPPRESS", "conversational turn: no retrieval needed")
        self.tracer.emit("suppression", t=t, reason="conversational", retrieval_required=False)
        low = ts.text.lower()
        if re.search(r"\b(thank|thanks|thx|cheers|helpful)\b", low):
            reply = "You're welcome. Ask me anything else about the documents."
        elif re.search(r"\b(hi|hello|hey|morning|afternoon|evening)\b", low):
            reply = "Hello! Ask me anything about the documents, even several questions at once."
        else:
            reply = "Okay."
        ttft_ms = (time.perf_counter() - t0) * 1000
        self.tracer.emit("first_token", t=t, ttft_ms=round(ttft_ms, 3), first_text=reply[:60])
        return self._finish_turn(t, ts, None, [], [], ttft_ms, retrieval_required=False, text=reply, reason="conversational")

    # ---------------------------------------------------------------- record
    def _finish_turn(self, t: float, ts: TurnState, version: AnswerVersion | None, subs: list[SubIntent], reused: list[bool],
                     ttft_ms: float, retrieval_required: bool, text: str | None = None, reason: str | None = None) -> dict:
        fired = ts.fired
        early = [f for f in fired if ts.chunks and f.t < t]
        first_t = min((f.t for f in fired), default=None)
        lead = (t - first_t) if first_t is not None else 0.0
        llm_in, llm_out = getattr(self, "llm_usage", (0, 0))
        self.llm_usage = (0, 0)
        cost = llm_in / 1e6 * self.cfg.price_in_per_mtok + llm_out / 1e6 * self.cfg.price_out_per_mtok
        summary = {
            "turn_type": ts.act_stream if not retrieval_required else ("refinement" if version and version.mode == "refinement" else "new"),
            "retrieval_required": retrieval_required,
            "latency": {"ttft_ms": round(ttft_ms, 3),
                        "controller_us_p50": round(sorted(ts.controller_us)[len(ts.controller_us) // 2], 1) if ts.controller_us else 0,
                        "controller_us_max": round(max(ts.controller_us), 1) if ts.controller_us else 0,
                        "retrieval_ms_total": round(sum(f.latency_ms for f in fired), 3)},
            "counts": {"chunks": len(ts.chunks), "retrievals_fired": len(fired), "retrievals_before_end": len(early),
                       "retrievals_used": sum(1 for f in fired if f.used), "speculative_hits": sum(1 for r in reused if r),
                       "sub_intents": len(subs)},
            "early_retrieval": bool(early), "retrieval_lead_s": round(lead, 3),
            "cost": {"llm_input_tokens": llm_in, "llm_output_tokens": llm_out, "embedding_calls": len(fired) + len(subs),
                     "est_usd": round(cost, 6)},
            "answer_version": version.version if version else None,
        }
        self.tracer.emit("turn_summary", t=t, **summary)
        answer_text = version.text if version else (text or "")
        rec = {
            "turn_id": ts.turn_id,
            "retrieval_required": retrieval_required,
            "reason": reason,
            "retrieval_events": [{"timestamp_s": round(f.t - (ts.t_first or 0.0), 3), "query": f.sub.query, "trigger": f.trigger}
                                 for f in fired],
            "sub_queries": [s.query for s in subs],
            "answer": answer_text,
            "citations": version.citations if version else [],
            "uncertainty": " ".join(version.uncertainty) if version and version.uncertainty else None,
            "answer_version": version.to_dict() if version else None,
            "telemetry": summary,
        }
        return rec


# ------------------------------------------------------------------------ helpers
def entity_conditional_search(ix: HybridIndex, cfg: Config, si: SubIntent, comp: Composer) -> list[Hit]:
    """Entities are used as a hard scope only if the corpus has entity-specific evidence for this aspect:
    some top hit must contain both the entity and a specific aspect term. Otherwise the aspect-only
    ranking is used and the entity requirement is dropped (e.g. a per-diem rule defined by city tier
    rather than per city). Entities that never appear in a heading are expanded with the named entity of
    the section they occur in ("Lotus Hall" -> "Orchid Business Hotel")."""
    ents = list(si.entities)
    parents = parent_entities(ix, ents)
    si.alt_entities = parents
    ctx = (si.context + " " + " ".join(parents)).strip()
    own = [e for e in ents if e not in set(si.inherited)]
    base = ix.search(si.query, cfg, entities=own + parents, numbers=si.numbers, context=ctx, context_weight=si.context_weight)
    if not ents:
        return base
    scoped = ix.search(si.query + " " + " ".join(e for e in ents if e not in si.query.lower()), cfg, entities=ents + parents,
                       numbers=si.numbers, context=ctx, context_weight=si.context_weight)
    spec = [s for s in comp._specific(si) if ix.in_vocab(s) and ix.idf.get(s, 0) >= 1.5]
    names = [e for e in ents + parents]
    for h in scoped[:3]:
        c = ix.chunks[h.idx]
        low = (c.doc_title + " " + c.heading + " " + c.text).lower()
        if any(e in low for e in names) and (not spec or any(s in ix.chunk_tokens[h.idx] for s in spec)):
            si.entity_scoped = True
            return scoped
    si.entity_scoped = False
    return ix.search(si.query, cfg, entities=[], numbers=si.numbers, context=ctx, context_weight=si.context_weight) if own else base


def parent_entities(ix: HybridIndex, ents: list[str]) -> list[str]:
    out = []
    head_ents = getattr(ix, "_heading_entities", None)
    if head_ents is None:
        head_ents = {}
        for i, c in enumerate(ix.chunks):
            hl = c.heading.lower()
            e = next((e for e in sorted(ix.entities, key=len, reverse=True) if hl.startswith(e) and len(e) > 3), None)
            if e:
                head_ents[i] = e
        ix._heading_entities = head_ents
    for e in ents:
        if any(e in (c.heading + " " + c.doc_title).lower() for c in ix.chunks):
            continue
        for i, c in enumerate(ix.chunks):
            if e in c.text.lower() and i in head_ents and head_ents[i] != e and head_ents[i] not in out:
                out.append(head_ents[i])
    return out[:2]


def _lower_first(s: str, ix: HybridIndex) -> str:
    if not s:
        return s
    first = s.split()[0]
    low = s.lower()
    if (len(first) > 1 and first.isupper()) or any(ch.isdigit() for ch in first) or \
            any(low.startswith(e) for e in ix.entities if len(e) > 2):
        return s
    return s[0].lower() + s[1:]


_NUM = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6}


def _compress(sentence: str) -> str:
    """Keep the main clause of a sentence for one-line / shorter answers."""
    s = re.sub(r"\s*\([^)]*\)", "", sentence)
    parts = re.split(r";\s+|,\s+(?:and|but|which|while|unless|including)\s+", s)
    out = parts[0].strip().rstrip(".,") + "."
    return out if len(out.split()) >= 5 else s


def restructure(prev: AnswerVersion, request: str) -> tuple[str, list[Claim]]:
    low = request.lower()
    claims = [copy.copy(c) for c in prev.claims if c.status != "superseded"]
    for c in claims:
        c.status = "preserved"
    if not claims:
        return prev.text, []
    m = re.search(r"\b(\d+|one|two|three|four|five|six)\s+(?:bullets?|bullet points|points|lines)\b", low)
    if m or re.search(r"bullet|list|points", low):
        n = int(m.group(1)) if m and m.group(1).isdigit() else (_NUM.get(m.group(1), len(claims)) if m else len(claims))
        n = max(1, n)
        if n > len(claims) == 1:
            pieces = [p.strip(" ,") for p in re.split(r",\s+(?:with|and|but|which|unless|while)\s+|;\s+", claims[0].text) if p.strip(" ,")]
            if len(pieces) >= 2:
                head, tail = pieces[0], " ".join(pieces[1:])
                cid = claims[0].chunk_id
                return f"• {head.rstrip('.')}. [{cid}]\n• {tail[0].upper() + tail[1:].rstrip('.')}. [{cid}]", claims
        groups: list[list[Claim]] = [[] for _ in range(min(n, len(claims)))]
        for i, c in enumerate(claims):
            groups[min(i * len(groups) // len(claims), len(groups) - 1)].append(c)
        lines = []
        for g in groups:
            txt = " ".join(_compress(c.text) for c in g)
            cits = "".join(f" [{cid}]" for cid in dict.fromkeys(c.chunk_id for c in g))
            lines.append(f"• {txt}{cits}")
        return "\n".join(lines), claims
    if re.search(r"one line|a sentence|one sentence|tl;?dr|shorter|shorten|brief|concise|summari[sz]e|simpl", low):
        seen, keep = set(), []
        for c in claims:
            if c.sub_intent in seen:
                continue
            seen.add(c.sub_intent)
            keep.append(c)
        if re.search(r"one line|a sentence|one sentence|tl;?dr", low):
            keep = keep[:3]
        text = " ".join(f"{_compress(c.text)} [{c.chunk_id}]" for c in keep)
        return text, keep
    return prev.text, claims
