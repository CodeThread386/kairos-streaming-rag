"""Kairos benchmark: replays held-out streaming scenarios and scores the six Theme-4 acceptance gates
(G1-G6), retrieval recall, answer quality, latency and cost, plus a batch-RAG baseline and ablations.

    python -m bench.run_bench                 # full suite -> reports/results.json + reports/metrics.md
    python -m bench.run_bench --quick         # main system on the Tessera test split only
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import jsonschema

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kairos.config import Config                      # noqa: E402
from kairos.engine import Engine                      # noqa: E402
from kairos.index import build_index                  # noqa: E402
from kairos.replay import load_scenarios, run_session  # noqa: E402
from kairos.synthesize import verify_claim            # noqa: E402
from kairos.telemetry import coverage_ok              # noqa: E402
from kairos.text import content_tokens, jaccard       # noqa: E402

SCHEMA = json.loads((ROOT / "docs" / "telemetry.schema.json").read_text())
VALIDATOR = jsonschema.Draft202012Validator(SCHEMA)
K = 4


def pct(xs, p):
    if not xs:
        return 0.0
    xs = sorted(xs)
    i = min(len(xs) - 1, max(0, int(round(p / 100 * (len(xs) - 1)))))
    return xs[i]


def resolve_handbook_gold(scenarios: list[dict], index) -> None:
    """18F scenarios reference gold as (file, heading); map them to section IDs of this index."""
    for s in scenarios:
        for t in s["turns"]:
            for si in t["gold"]["sub_intents"]:
                if "relevant" in si:
                    continue
                rel = []
                for ref in si.get("relevant_ref", []):
                    for c in index.chunks:
                        if c.path.endswith(ref["file"]) and c.heading.strip().lower() == ref["heading"].strip().lower():
                            rel.append(c.chunk_id)
                si["relevant"] = rel


# ----------------------------------------------------------------------------- scoring
def score_run(sessions: list[dict], index) -> dict:
    corpus_ids = set(index.by_id)
    hide = {L: [] for L in (50, 150, 400)}
    m = {k: [] for k in ("eligible_early", "lead_s", "false_trigger", "compound_ok", "compound_all", "subintent_hit",
                         "subintent_rr", "evidence_hit", "keyfact", "claims_supported", "fabricated", "unans_ok",
                         "false_abstain", "g5", "g6", "ttft_ms", "controller_us", "retrieval_ms", "retrievals_per_turn",
                         "wasted_retrievals", "overfrag", "speculative_hit", "subq_count")}
    failures = []
    for sess in sessions:
        events = sess["trace"]
        prev_version = None
        prev_subqs: list[str] = []
        for t in sess["turns"]:
            g = t["gold"]
            tt = g["turn_type"]
            tel = t["telemetry"]
            tid = t["turn_id"]
            tev = [e for e in events if e["turn_id"] == tid]
            # ---- G6 trace coverage + schema validity
            ok_cov, missing = coverage_ok(tev, tt)
            schema_ok = all(not list(VALIDATOR.iter_errors(e)) for e in tev)
            m["g6"].append(ok_cov and schema_ok)
            if not (ok_cov and schema_ok):
                failures.append({"id": sess["id"], "turn": tid, "gate": "G6", "missing": missing, "schema_ok": schema_ok})
            # ---- latency / cost
            m["ttft_ms"].append(tel["latency"]["ttft_ms"])
            m["controller_us"] += [e2 for e2 in [tel["latency"]["controller_us_p50"]] if e2]
            m["retrieval_ms"] += [e["latency_ms"] for e in tev if e["event"] == "retrieval_completed"]
            m["retrievals_per_turn"].append(tel["counts"]["retrievals_fired"])
            m["wasted_retrievals"].append(tel["counts"]["retrievals_fired"] - tel["counts"]["retrievals_used"])
            # ---- latency hiding: what TTFT would be if every retrieval took L ms (remote vector DB + reranker)
            sq_ev = [e for e in tev if e["event"] == "sub_queries"]
            if sq_ev and sq_ev[-1].get("used_query_ids"):
                fired_at = {e["query_id"]: e["t_stream_s"] for e in tev if e["event"] == "retrieval_started"}
                ready = [fired_at[q] for q in sq_ev[-1]["used_query_ids"] if q in fired_at]
                if ready:
                    for L in hide:
                        wait = max(0.0, max(r + L / 1000 for r in ready) - t["end_t"])
                        hide[L].append(wait * 1000 + tel["latency"]["ttft_ms"])
            # ---- G2 early retrieval / false triggers
            fired = tel["counts"]["retrievals_fired"]
            if g["needs_retrieval"]:
                m["eligible_early"].append(bool(tel["early_retrieval"]))
                if tel["early_retrieval"]:
                    m["lead_s"].append(tel["retrieval_lead_s"])
                else:
                    failures.append({"id": sess["id"], "turn": tid, "gate": "G2", "utterance": t["utterance"]})
                sq = tel["counts"]["sub_intents"]
                if sq:
                    m["speculative_hit"].append(tel["counts"]["speculative_hits"] / sq)
            else:
                m["false_trigger"].append(fired > 0 or t["retrieval_required"])
                if fired > 0 or t["retrieval_required"]:
                    failures.append({"id": sess["id"], "turn": tid, "gate": "G2-false-trigger", "utterance": t["utterance"]})
            # ---- retrieval quality per gold sub-intent
            ret_hits = {}
            for e in tev:
                if e["event"] == "retrieval_completed":
                    ret_hits[e["query_id"]] = [h["chunk_id"] for h in e["hits"][:K]]
            q_by_id = {e["query_id"]: e["query"] for e in tev if e["event"] == "retrieval_started"}
            used_qids = []
            subq_texts = t["sub_queries"]
            for q in subq_texts:
                # the retrieval actually used for this sub-query (same query text, or most similar speculative one)
                best = max(q_by_id.items(), key=lambda kv: jaccard(set(content_tokens(kv[1])), set(content_tokens(q))), default=None)
                if best:
                    used_qids.append(best[0])
            answerable = [si for si in g["sub_intents"] if si["relevant"]]
            if g["needs_retrieval"]:
                m["subq_count"].append(len(subq_texts))
            for si in answerable:
                rel = set(si["relevant"])
                ranks = []
                for qid in used_qids:
                    hits = ret_hits.get(qid, [])
                    for r, cid in enumerate(hits, 1):
                        if cid in rel:
                            ranks.append(r)
                            break
                m["subintent_hit"].append(bool(ranks))
                m["subintent_rr"].append(1.0 / min(ranks) if ranks else 0.0)
                cited = set(t["citations"])
                m["evidence_hit"].append(bool(cited & rel))
                if not (cited & rel):
                    failures.append({"id": sess["id"], "turn": tid, "gate": "evidence", "sub_intent": si["name"],
                                     "gold": sorted(rel), "cited": sorted(cited)})
            # ---- G3 multi-intent identification (distinct sub-queries matched to distinct gold sub-intents)
            if g["needs_retrieval"] and tt == "new" and len(answerable) >= 2:
                unused = list(used_qids)
                matched = 0
                for si in answerable:
                    rel = set(si["relevant"])
                    for qid in list(unused):
                        if set(ret_hits.get(qid, [])) & rel:
                            unused.remove(qid)
                            matched += 1
                            break
                m["compound_ok"].append(len(subq_texts) >= 2 and matched >= 2)
                m["compound_all"].append(matched >= len(answerable))
                if not (len(subq_texts) >= 2 and matched >= 2):
                    failures.append({"id": sess["id"], "turn": tid, "gate": "G3", "utterance": t["utterance"],
                                     "sub_queries": subq_texts, "matched": matched})
            if g["needs_retrieval"] and tt == "new" and len(g["sub_intents"]) == 1:
                m["overfrag"].append(len(subq_texts) > 1)
            # ---- answer quality
            ans = (t["answer"] or "").lower()
            for grp in g.get("key_facts", []):
                hit = any(x.lower() in ans for x in grp)
                m["keyfact"].append(hit)
                if not hit:
                    failures.append({"id": sess["id"], "turn": tid, "gate": "key_fact", "expected_any": grp, "answer": t["answer"][:300]})
            if g.get("unanswerable_aspects"):
                ok = bool(t["uncertainty"])
                m["unans_ok"].append(ok)
                if not ok:
                    failures.append({"id": sess["id"], "turn": tid, "gate": "unanswerable", "answer": t["answer"][:200]})
            elif g["needs_retrieval"] and answerable and t["uncertainty"]:
                m["false_abstain"].append(True)
                failures.append({"id": sess["id"], "turn": tid, "gate": "false_abstain", "uncertainty": t["uncertainty"]})
            elif g["needs_retrieval"] and answerable:
                m["false_abstain"].append(False)
            # ---- G4 grounding (every new claim checked against its cited chunk; fabricated IDs)
            v = t.get("answer_version")
            if v:
                for c in v["claims"]:
                    if c["status"] == "preserved":
                        continue
                    if c["citation"] not in corpus_ids:
                        m["fabricated"].append(c["citation"])
                        m["claims_supported"].append(False)
                        continue
                    ch = index.chunks[index.by_id[c["citation"]]]
                    ok, _ = verify_claim(c["text"], ch.text, ch.heading)
                    m["claims_supported"].append(ok)
                for cid in v["citations"]:
                    if cid not in corpus_ids:
                        m["fabricated"].append(cid)
            # ---- G5 session refinement
            if tt == "refinement":
                ok_parent = bool(v and prev_version and v["parent"] == prev_version["version"])
                executed = [e["query"] for e in tev if e["event"] == "retrieval_started"]
                restarted = any(q.strip().lower() == p.strip().lower() for q in executed for p in prev_subqs)
                prev_cits = set(prev_version["citations"]) if prev_version else set()
                accounted = set(v["preserved"]) | set(v["superseded"]) if v else set()
                continuity = prev_cits <= accounted if prev_cits else True
                delta_ok = bool(v and (v["added"] or v["uncertainty"]))
                passed = ok_parent and not restarted and continuity and delta_ok
                m["g5"].append(passed)
                if not passed:
                    failures.append({"id": sess["id"], "turn": tid, "gate": "G5", "parent_ok": ok_parent, "restarted": restarted,
                                     "continuity": continuity, "delta_ok": delta_ok})
            if v:
                prev_version = v
            if t["sub_queries"]:
                prev_subqs = t["sub_queries"]

    def rate(xs):
        return round(sum(1 for x in xs if x) / len(xs), 4) if xs else None

    return {
        "turns": sum(len(s["turns"]) for s in sessions),
        "G2_early_retrieval_rate": rate(m["eligible_early"]),
        "G2_false_trigger_rate": rate(m["false_trigger"]),
        "G2_median_lead_s": round(statistics.median(m["lead_s"]), 3) if m["lead_s"] else None,
        "G3_multi_intent_rate": rate(m["compound_ok"]),
        "G3_all_sub_intents_rate": rate(m["compound_all"]),
        "G3_compound_turns": len(m["compound_ok"]),
        "over_fragmentation_rate": rate(m["overfrag"]),
        "subintent_recall@4": rate(m["subintent_hit"]),
        "subintent_mrr@4": round(statistics.mean(m["subintent_rr"]), 4) if m["subintent_rr"] else None,
        "answer_evidence_recall": rate(m["evidence_hit"]),
        "key_fact_recall": rate(m["keyfact"]),
        "unanswerable_abstention_rate": rate(m["unans_ok"]),
        "false_abstention_rate": rate(m["false_abstain"]),
        "G4_citation_support_rate": rate(m["claims_supported"]),
        "G4_claims_checked": len(m["claims_supported"]),
        "G4_fabricated_citations": len(m["fabricated"]),
        "G5_refinement_pass_rate": rate(m["g5"]),
        "G5_refinement_turns": len(m["g5"]),
        "G6_trace_coverage": rate(m["g6"]),
        "ttft_ms_p50": round(pct(m["ttft_ms"], 50), 3), "ttft_ms_p95": round(pct(m["ttft_ms"], 95), 3),
        "controller_us_p50": round(pct(m["controller_us"], 50), 1), "controller_us_p95": round(pct(m["controller_us"], 95), 1),
        "retrieval_ms_p50": round(pct(m["retrieval_ms"], 50), 3), "retrieval_ms_p95": round(pct(m["retrieval_ms"], 95), 3),
        "retrievals_per_turn": round(statistics.mean(m["retrievals_per_turn"]), 3) if m["retrievals_per_turn"] else 0,
        "wasted_retrievals_per_turn": round(statistics.mean(m["wasted_retrievals"]), 3) if m["wasted_retrievals"] else 0,
        "speculative_reuse_rate": round(statistics.mean(m["speculative_hit"]), 4) if m["speculative_hit"] else None,
        "llm_cost_usd_per_turn": 0.0,
        "simulated_ttft_ms": {str(L): {"p50": round(pct(v, 50), 1), "p95": round(pct(v, 95), 1)} for L, v in hide.items()},
        "_failures": failures,
    }


def run_suite(engine: Engine, scenarios: list[dict], cfg: Config, asr_style: bool = False) -> tuple[dict, list[dict]]:
    sessions = []
    for s in scenarios:
        sid = s["id"]
        sessions.append(run_session(engine, json.loads(json.dumps(s)) | {"id": sid}, cfg=cfg, asr_style=asr_style))
    return score_run(sessions, engine.index), sessions


# ----------------------------------------------------------------------------- report
GATES = [
    ("G2", "Early retrieval (≥ 80% of eligible turns)", "G2_early_retrieval_rate", 0.80),
    ("G3", "Multi-intent identification (≥ 70% of compound turns)", "G3_multi_intent_rate", 0.70),
    ("G4", "Citation support (≥ 85% of claims), 0 fabricated IDs", "G4_citation_support_rate", 0.85),
    ("G5", "Session refinement continuity", "G5_refinement_pass_rate", 1.0),
    ("G6", "Telemetry trace coverage (100%)", "G6_trace_coverage", 1.0),
]


def fmt(v, pct_=True):
    if v is None:
        return "–"
    if isinstance(v, float) and pct_ and v <= 1.0:
        return f"{v * 100:.1f}%"
    return str(v)


def write_report(results: dict, out_dir: Path) -> None:
    main = results["main"]
    lines = ["# Kairos — Benchmark & Evaluation Report", "",
             f"*Generated by `python -m bench.run_bench` on {results['env']['generated_at']} — "
             f"{results['env']['python']} on {results['env']['platform']} ({results['env']['cpu_count']} vCPU). "
             f"Models: WordLlama `l2_supercat` 256-d static embeddings + BM25 (CPU only). LLM calls: 0.*", "",
             "## 1. Acceptance gates (Tessera held-out test split, "
             f"{main['turns']} turns / {results['n_sessions']['tessera_test']} sessions)", "",
             "| Gate | Criterion | Target | Measured | Status |", "|---|---|---|---|---|",
             "| G1 | Reproducibility: one-command container + automated replay suite | Pass/Fail | "
             f"{results['g1']['detail']} | {'PASS' if results['g1']['pass'] else 'FAIL'} |"]
    for gid, crit, key, target in GATES:
        val = main[key]
        ok = val is not None and val >= target and (gid != "G4" or main["G4_fabricated_citations"] == 0)
        extra = f" ({main['G4_claims_checked']} claims, {main['G4_fabricated_citations']} fabricated)" if gid == "G4" else ""
        if gid == "G2":
            extra = f" (false-trigger rate {fmt(main['G2_false_trigger_rate'])}, median lead {main['G2_median_lead_s']} s)"
        if gid == "G3":
            extra = f" ({main['G3_compound_turns']} compound turns; all sub-intents found {fmt(main['G3_all_sub_intents_rate'])})"
        if gid == "G5":
            extra = f" ({main['G5_refinement_turns']} refinement turns)"
        lines.append(f"| {gid} | {crit} | {'≥ ' + format(target, '.0%') if target < 1 else '100%'} | {fmt(val)}{extra} | {'PASS' if ok else 'FAIL'} |")
    if results.get("test2"):
        t2 = results["test2"]
        lines += ["", f"**Clean held-out split (test-2: {results['n_sessions'].get('tessera_test2')} sessions / {t2['turns']} turns, "
                  "frozen before any error analysis and never used for tuning):** "
                  f"G2 {fmt(t2['G2_early_retrieval_rate'])}, G3 {fmt(t2['G3_multi_intent_rate'])}, "
                  f"G4 {fmt(t2['G4_citation_support_rate'])} ({t2['G4_fabricated_citations']} fabricated), "
                  f"G5 {fmt(t2['G5_refinement_pass_rate'])}, G6 {fmt(t2['G6_trace_coverage'])}; "
                  f"answer evidence recall {fmt(t2['answer_evidence_recall'])}, key-fact recall {fmt(t2['key_fact_recall'])}.", "",
                  "Test-1 was used for one round of error analysis after its first run (results of that first run are kept in "
                  "`reports/results_test1_before_error_analysis.json`); test-2 is the uncontaminated number."]
    lines += ["", "## 2. Retrieval, answer quality, latency and cost", "",
              "| Metric | Tessera test-1 | Tessera test-2 (clean) | Test-1 as raw ASR (lower-case, no punctuation) | 18F handbook (real corpus, never tuned) |",
              "|---|---|---|---|---|"]
    rows = [("Sub-intent recall@4", "subintent_recall@4"), ("Sub-intent MRR@4", "subintent_mrr@4"),
            ("Answer evidence recall (gold section cited)", "answer_evidence_recall"), ("Key-fact recall in answer", "key_fact_recall"),
            ("Early retrieval rate (G2)", "G2_early_retrieval_rate"), ("False-trigger rate", "G2_false_trigger_rate"),
            ("Multi-intent identification (G3)", "G3_multi_intent_rate"), ("Over-fragmentation rate (single-intent turns)", "over_fragmentation_rate"),
            ("Unanswerable → explicit uncertainty", "unanswerable_abstention_rate"), ("False abstention (answerable flagged)", "false_abstention_rate"),
            ("Citation support (G4)", "G4_citation_support_rate"), ("Refinement pass (G5)", "G5_refinement_pass_rate"),
            ("Trace coverage (G6)", "G6_trace_coverage"), ("Speculative-retrieval reuse at utterance end", "speculative_reuse_rate"),
            ("TTFT p50 / p95 (ms after utterance end)", ("ttft_ms_p50", "ttft_ms_p95")),
            ("Controller decision p50 / p95 (µs per chunk)", ("controller_us_p50", "controller_us_p95")),
            ("Retrieval latency p50 / p95 (ms per query)", ("retrieval_ms_p50", "retrieval_ms_p95")),
            ("Retrievals per turn (wasted)", ("retrievals_per_turn", "wasted_retrievals_per_turn")),
            ("LLM cost per turn", "llm_cost_usd_per_turn")]
    for label, key in rows:
        cells = []
        for run in ("main", "test2", "asr", "handbook"):
            r = results.get(run)
            if not r:
                cells.append("–")
            elif isinstance(key, tuple):
                cells.append(f"{r[key[0]]} / {r[key[1]]}")
            elif key == "llm_cost_usd_per_turn":
                cells.append("$0.00 (no LLM in the loop)")
            else:
                cells.append(fmt(r[key]))
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines += ["", "## 3. Baseline comparison (Tessera test split)", "",
              "Baseline = conventional batch RAG: waits for the end of the utterance, embeds the whole utterance as one query, "
              "same index and composer (`controller_mode=end_only`, `decomposition=False`).", "",
              "| Metric | Batch RAG baseline | Kairos | Δ |", "|---|---|---|---|"]
    base = results["ablations"]["baseline_batch_rag"]
    for label, key in [("Early retrieval rate", "G2_early_retrieval_rate"), ("Median retrieval lead (s before utterance end)", "G2_median_lead_s"),
                       ("Multi-intent identification", "G3_multi_intent_rate"), ("Sub-intent recall@4", "subintent_recall@4"),
                       ("Answer evidence recall", "answer_evidence_recall"), ("Key-fact recall", "key_fact_recall"),
                       ("TTFT p50 (ms)", "ttft_ms_p50"), ("TTFT p95 (ms)", "ttft_ms_p95")]:
        b, k = base.get(key), main.get(key)
        d = "–"
        if isinstance(b, (int, float)) and isinstance(k, (int, float)):
            d = f"{(k - b) * 100:+.1f} pts" if (b <= 1 and k <= 1 and "ms" not in key and "lead" not in key) else f"{k - b:+.3f}"
        lines.append(f"| {label} | {fmt(b)} | {fmt(k)} | {d} |")
    lines += ["", "## 4. Architectural ablations (Tessera test split)", "",
              "| Variant | Recall@4 | MRR@4 | Evidence recall | Key facts | Early rate | False triggers | G3 | Retrievals/turn | Wasted/turn |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for name, r in results["ablations"].items():
        lines.append(f"| {name} | {fmt(r['subintent_recall@4'])} | {r['subintent_mrr@4']} | {fmt(r['answer_evidence_recall'])} | "
                     f"{fmt(r['key_fact_recall'])} | {fmt(r['G2_early_retrieval_rate'])} | {fmt(r['G2_false_trigger_rate'])} | "
                     f"{fmt(r['G3_multi_intent_rate'])} | {r['retrievals_per_turn']} | {r['wasted_retrievals_per_turn']} |")
    if "refine_vs_restart" in results:
        rr = results["refine_vs_restart"]
        lines += ["", "### Refine vs restart on late-arriving details", "",
                  "| Strategy | Queries executed on refinement turns | Earlier citations kept | Key facts on refinement turns |",
                  "|---|---|---|---|"]
        for name, r in rr.items():
            lines.append(f"| {name} | {r['queries']} | {fmt(r['kept'])} | {fmt(r['keyfacts'])} |")
    lines += ["", "### Latency hiding: TTFT if each retrieval took L ms", "",
              "The measured retrieval here is ~1 ms (in-process index), so both systems start answering within a few ms. "
              "With a production retrieval stack (remote vector DB + reranker) every query costs tens to hundreds of ms. "
              "Replaying the same traces with a simulated per-query latency L shows how much of it the streaming controller hides "
              "(speculative queries fired while the user is still talking finish before they stop):", "",
              "| Per-query retrieval latency L | Batch RAG TTFT p50 / p95 (ms) | Kairos TTFT p50 / p95 (ms) |", "|---|---|---|"]
    for L in ("50", "150", "400"):
        b, k = base["simulated_ttft_ms"][L], main["simulated_ttft_ms"][L]
        lines.append(f"| {L} ms | {b['p50']} / {b['p95']} | {k['p50']} / {k['p95']} |")
    if "handbook_ablations" in results:
        lines += ["", "### Retrieval ablation on the real 18F corpus (never tuned)", "",
                  "| Variant | Recall@4 | MRR@4 | Evidence recall | Key facts |", "|---|---|---|---|---|"]
        for name, r in results["handbook_ablations"].items():
            lines.append(f"| {name} | {fmt(r['subintent_recall@4'])} | {r['subintent_mrr@4']} | {fmt(r['answer_evidence_recall'])} | {fmt(r['key_fact_recall'])} |")
    sc = results.get("scale")
    if sc:
        lines += ["", "## 5. Indexing cost and scale", "", "| Corpus | Docs | Chunks | Index build (s) | Retrieval p50 / p95 (ms) |", "|---|---|---|---|---|"]
        for r in sc:
            lines.append(f"| {r['corpus']} | {r['docs']} | {r['chunks']} | {r['build_s']} | {r['p50_ms']} / {r['p95_ms']} |")
    lines += ["", "## 6. Failure log", "", "Every gate or quality miss on the main run is listed in `reports/results.json` → `main._failures`. "
              "The analysed edge cases are in `docs/EVALUATION.md`.", ""]
    (out_dir / "metrics.md").write_text("\n".join(lines), encoding="utf-8")


def refine_vs_restart(engine: Engine, scenarios: list[dict], cfg: Config) -> dict:
    out = {}
    for name, mode in (("refine (Kairos)", "refine"), ("restart (re-answer from scratch)", "restart")):
        c = cfg.variant(refinement=mode)
        queries, kept, keyf = [], [], []
        for s in scenarios:
            if not any(t["gold"]["turn_type"] == "refinement" for t in s["turns"]):
                continue
            sess = run_session(engine, s, cfg=c)
            prev = None
            for t in sess["turns"]:
                if t["gold"]["turn_type"] == "refinement":
                    queries.append(t["telemetry"]["counts"]["retrievals_fired"])
                    if prev and t["answer_version"]:
                        kept.append(set(prev["citations"]) <= set(t["answer_version"]["citations"]) | set(t["answer_version"]["superseded"]))
                    ans = (t["answer"] or "").lower()
                    keyf += [any(x.lower() in ans for x in grp) for grp in t["gold"]["key_facts"]]
                if t["answer_version"]:
                    prev = t["answer_version"]
        out[name] = {"queries": round(statistics.mean(queries), 2) if queries else 0,
                     "kept": round(sum(kept) / len(kept), 4) if kept else None,
                     "keyfacts": round(sum(keyf) / len(keyf), 4) if keyf else None}
    return out


def scale_test(cfg: Config, extra: list[str]) -> list[dict]:
    rows = []
    for corpus in [str(ROOT / "corpus" / "tessera"), str(ROOT / "corpus" / "handbook18f")] + extra:
        if not Path(corpus).exists():
            continue
        t0 = time.perf_counter()
        ix = build_index(corpus, cfg, use_cache=False)
        build = time.perf_counter() - t0
        qs = ["what is the reimbursement rule for travel", "how many days of leave", "who approves international trips",
              "cancellation policy for venues", "receipts required for expenses", "internet service while traveling"]
        lat = []
        for _ in range(5):
            for q in qs:
                t1 = time.perf_counter()
                ix.search(q, cfg)
                lat.append((time.perf_counter() - t1) * 1000)
        rows.append({"corpus": Path(corpus).name, "docs": len({c.doc_id for c in ix.chunks}), "chunks": len(ix.chunks),
                     "build_s": round(build, 2), "p50_ms": round(pct(lat, 50), 3), "p95_ms": round(pct(lat, 95), 3)})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default=str(ROOT / "reports"))
    ap.add_argument("--scale-corpus", action="append", default=[])
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    t_all = time.perf_counter()
    cfg = Config(corpus_path=str(ROOT / "corpus" / "tessera"), write_traces=False)
    engine = Engine(cfg)
    scen = load_scenarios(ROOT / "bench" / "scenarios" / f"tessera_{args.split}.jsonl")
    results = {"env": {"generated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "python": platform.python_version(),
                       "platform": platform.platform(), "cpu_count": __import__("os").cpu_count()},
               "n_sessions": {"tessera_" + args.split: len(scen)}, "config": cfg.public()}
    main_res, main_sessions = run_suite(engine, scen, cfg)
    results["main"] = main_res
    (out_dir / "traces").mkdir(exist_ok=True)
    with (out_dir / "traces" / f"tessera_{args.split}_records.jsonl").open("w", encoding="utf-8") as f:
        for s in main_sessions:
            for t in s["turns"]:
                f.write(json.dumps({"session": s["id"], **{k: v for k, v in t.items() if k not in ("answer_version",)}}, ensure_ascii=False) + "\n")
    with (out_dir / "traces" / f"tessera_{args.split}_telemetry.jsonl").open("w", encoding="utf-8") as f:
        for s in main_sessions:
            for e in s["trace"]:
                f.write(json.dumps(e, ensure_ascii=False, default=str) + "\n")
    results["g1"] = {"pass": True, "detail": f"replay suite completed unattended in {time.perf_counter() - t_all:.1f} s"}
    if not args.quick:
        t2 = ROOT / "bench" / "scenarios" / "tessera_test2.jsonl"
        if t2.exists():
            scen2 = load_scenarios(t2)
            results["n_sessions"]["tessera_test2"] = len(scen2)
            results["test2"], _ = run_suite(engine, scen2, cfg)
        results["asr"], _ = run_suite(engine, scen, cfg, asr_style=True)
        hb_cfg = cfg.variant(corpus_path=str(ROOT / "corpus" / "handbook18f"))
        hb_engine = Engine(hb_cfg)
        hb = load_scenarios(ROOT / "bench" / "scenarios" / "handbook18f_test.jsonl")
        resolve_handbook_gold(hb, hb_engine.index)
        results["n_sessions"]["handbook18f_test"] = len(hb)
        results["handbook"], _ = run_suite(hb_engine, hb, hb_cfg)
        hab = {}
        for name, ov in {"hybrid (Kairos)": {}, "dense-only": {"retrieval_mode": "dense"},
                         "sparse-only (BM25)": {"retrieval_mode": "sparse"}, "batch RAG baseline": {"controller_mode": "end_only", "decomposition": False}}.items():
            hab[name], _ = run_suite(hb_engine, hb, hb_cfg.variant(**ov))
            hab[name].pop("_failures", None)
        results["handbook_ablations"] = hab
        abl = {}
        variants = {
            "Kairos (hybrid + streaming controller + decomposition)": {},
            "baseline_batch_rag": {"controller_mode": "end_only", "decomposition": False},
            "dense-only retrieval": {"retrieval_mode": "dense"},
            "sparse-only (BM25) retrieval": {"retrieval_mode": "sparse"},
            "eager controller (retrieve every chunk)": {"controller_mode": "eager"},
            "end-of-utterance controller": {"controller_mode": "end_only"},
            "no decomposition": {"decomposition": False},
        }
        for name, ov in variants.items():
            c = cfg.variant(**ov)
            abl[name], _ = run_suite(engine, scen, c)
            abl[name].pop("_failures", None)
        results["ablations"] = abl
        results["refine_vs_restart"] = refine_vs_restart(engine, scen, cfg)
        results["scale"] = scale_test(cfg, args.scale_corpus)
        results["g1"]["detail"] = f"full replay suite (all runs) completed unattended in {time.perf_counter() - t_all:.1f} s"
    name = "results_quick.json" if args.quick else "results.json"
    (out_dir / name).write_text(json.dumps(results, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    if not args.quick:
        write_report(results, out_dir)
    m = results["main"]
    print(json.dumps({k: v for k, v in m.items() if not k.startswith("_")}, indent=1))
    print(f"failures: {len(m['_failures'])}")


if __name__ == "__main__":
    main()
