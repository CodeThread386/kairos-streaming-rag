# Kairos — System Architecture Brief

*Theme 04, Streaming Live RAG. Six pages or fewer, as the brief asks: design rationale, retrieval trigger logic, query
decomposition strategy, data provenance, trade-offs and failure-mode mitigations.*

![architecture](architecture.png)

## 1. Design rationale

A voice user speaks one natural request over two to ten seconds. A batch RAG system idles for that whole window and
then has to answer a compound request with a single query. Kairos treats the transcript as an **event stream** and asks
three questions on every chunk: *is there a stable intent yet? does it need the corpus at all? has it already been
fetched?* The answer is assembled from evidence gathered while the user was still talking.

Three principles shaped every component:

* **Cheap enough to run on every chunk.** The controller and retriever together cost about 1 ms per chunk on one CPU
  core (controller p50 ≈ 1.1 ms, retrieval p50 ≈ 0.6 ms). That is what makes it safe to decide continuously instead of
  waiting. The guide's "architectural parsimony" rule is taken literally: no agent framework, no GPU, no LLM on the hot
  path. Each component in this brief earns its place in the ablation table of `reports/metrics.md`.
* **Grounding by construction.** The default composer only emits corpus sentences, each followed by its section ID.
  A fabricated citation is therefore impossible, not merely unlikely. The optional LLM can only rephrase, and a
  sentence-level verifier re-checks everything it writes.
* **No domain knowledge in code.** Entities, the cities or venues a document is scoped to, heading vocabulary,
  capacity vocabulary and IDF are all mined from the corpus at index time. The only rules in code are about English
  and dialogue: function words, connectives, self-repair markers, presentation verbs. The same binary runs unchanged on
  the 18F handbook.

## 2. Components

| Stage | Input → output | Cost |
|---|---|---|
| Ingest | folder of md/txt/html/pdf/docx/json → section chunks `Doc_ID §N` (long sections → `§N.k`, ≤ 170 words) | 0.2 s per 160 chunks |
| Index | BM25 (title + heading×2 + text), WordLlama 256-d embeddings of chunk, heading and every sentence, lexicons | 8 s per 2 215 chunks |
| Controller | chunk → WAIT / RETRIEVE / DECOMPOSE_AND_RETRIEVE / SUPPRESS / REFINE / SYNTHESIZE | ≈ 1 ms |
| Decomposer | utterance → sub-intents (strong query, weak context, entities, counts) | < 1 ms |
| Retriever | sub-intent → top-k sections (RRF of 3 rankers + constraint boosts) | 0.6 ms p50 |
| Composer | sub-intents + hits → cited claims, uncertainty, answer version | ≈ 2 ms (TTFT) |
| Telemetry | every step → schema-validated JSONL events | negligible |

**Why WordLlama?** It is a 16 MB static token-embedding model whose weights ship inside the pip wheel. It is fully
offline, deterministic, runs in microseconds on CPU, and is good enough as the dense half of a hybrid ranker. On the
real 18F corpus the hybrid beats both halves: MRR@4 is 0.72, against 0.68 dense-only and 0.71 BM25-only, and evidence
recall is 72% against 67% for each half. On the small, clean Tessera corpus BM25 alone is already strong, and the
hybrid mainly improves ranking (MRR 0.96 against 0.95). Both observations are reported, not hidden.

## 3. Retrieval trigger logic (the controller)

For every incoming chunk (append-mode or a revised ASR hypothesis):

1. **Turn act.** These are classified first:
   * **Chit-chat:** a greeting or thanks with no in-vocabulary content. Result: SUPPRESS.
   * **Presentation-only:** a formatting verb ("repeat", "shorter", "in two bullets", "one line") that refers to the
     previous answer and adds no new informative terms. Result: SUPPRESS, and the last version is transformed.
   * **Refinement:** a previous answer exists and the turn is a statement or starts with "actually", "also",
     "what about", "no wait", "make it", "it's"… Result: REFINE.
   * **New request:** everything else.

   A refinement cue followed by a complete, unrelated question ("Also, what's the per diem in Singapore?") is still
   treated as a new request; the test is low embedding similarity to the previous sub-queries.
2. **Clause stability.** The utterance so far is decomposed. A clause that is still open and *ends in a function word*
   ("…workshop **in**", "…and I **need**") waits. A clause is retrieved when:
   * its informativeness (the sum of capped corpus IDF of its in-vocabulary content terms) reaches 3.2 for closed
     clauses, or 4.5 for the still-open last clause; and
   * it has an anchor (a corpus entity, a head-count, or two specific terms).

   This is what keeps false triggers at 0%. The *eager* ablation, which retrieves on every chunk, triggers on 14% of
   no-retrieval turns and wastes 2.3 queries per turn.
3. **Deduplication and upgrades.** A clause whose signature is ≥ 85% covered by an earlier query is skipped. If it
   gained new informative words, it is re-fired as an `upgrade`.
4. **Utterance end.** The final decomposition is matched against the speculative queries. A speculative result is
   reused only if it already covered *every* informative word of the final sub-query: "the booking was made" is not
   reused for "the booking was made *after travel*". Reuse is 93–94% on both test splits, so the answer starts about
   2 ms after the user stops.

**Latency hiding.** Replaying the same traces with a simulated per-query retrieval cost L (remote vector database plus
reranker) gives:

| Per-query latency L | Batch RAG, median time to first token | Kairos, median time to first token |
|---|---|---|
| 150 ms | 152 ms | 2.4 ms |
| 400 ms | 402 ms | 2.4 ms |

Speculative retrieval removes the round trip from the critical path whenever the intent stabilises before the user
finishes, which is the median turn.

## 4. Query decomposition strategy

* **Disfluency repair.** Fillers (um, uh, you know, *like* + number) are removed. Mid-utterance self-repairs delete the
  reparandum: "per diem in the UK, sorry, I mean the United States" becomes "per diem in the United States".
* **Clause segmentation.** Splits happen at:
  * sentence punctuation;
  * discourse connectives (also, as well as, plus, along with, oh and);
  * "and" or "," followed by a new clause (wh-word, auxiliary + pronoun) or by a determiner-led noun phrase ("… and
    the catering options");
  * comma lists of noun phrases;
  * bare-noun coordination *only* when no shared head noun follows ("tips and laundry during trips" splits;
    "cancellation and refund terms" does not).
* **Roles.** Clauses before the first request ("For international travel, …", "I'm flying to Berlin next month.") are
  *context*, not intents.
* **Distributive split.** "hotel limits in Pune and Bengaluru" becomes one sub-query per entity.
* **Context propagation.** Each sub-query carries a *strong* part (its own words, plus inherited aspect words for
  ellipsis such as "what about Tier A cities?") and a *weak* part (topic words and inherited entities). The weak part
  is weighted 0.35 in BM25. Weak propagation was a deliberate fix: strong propagation made "the per diem *there*"
  (there = Hyderabad) retrieve Hyderabad catering pages instead of the per-diem table.
* **Entity-conditional scoping.** An entity becomes a hard scope (boost plus a coverage requirement) only if a top hit
  contains both the entity and a specific aspect term. Otherwise the aspect-only ranking is used, as for a per-diem
  rule defined by city tier rather than by city. An entity that never appears in a heading is linked to the named
  entity of the section it occurs in: "Lotus Hall" links to "Orchid Business Hotel".
* **Over-fragmentation guard.** Sub-intents with the same entities and aspect-overlap ≥ 0.6 are merged. Fragments with
  no informative content are dropped. Measured over-fragmentation on single-intent turns is 0% on all splits.

## 5. Session state and refinement

Each session keeps its answer versions (with parent links), the evidence ledger, and the previous request's
sub-intents (with their entities and counts). A refinement is planned as one of two kinds:

* **Slot correction**, covering:
  * a new entity with "not X", "I meant", "what about", "instead";
  * a changed head-count ("make it 50 people").

  Only the *affected* sub-intents are re-retrieved with the slot replaced, and their claims are marked *superseded*.
* **Constraint or follow-up** ("the trip was international and the booking was made after travel"). Each new clause
  becomes a delta query that carries the previous topic as weak context. Previous claims are *preserved*, and new ones
  are *added* with "However," for exception-type rules or "In addition," otherwise. In this mode the composer prefers
  normative sentences (must, requires, only) over definitions, because the user stated a fact and needs the rule it
  triggers.

The original sub-queries are never re-executed. In the *restart* ablation the system answers from scratch instead: it
runs more queries (2.1 against 1.75), keeps 0% of the earlier citations, and reaches 33% key facts on refinement
turns against 78% for refine.

## 6. Grounding and uncertainty

For each sub-intent the composer:

1. Scores candidate sentences from the top hits with a fixed blend:
   * 0.42 × dense similarity;
   * 0.38 × IDF-weighted lexical overlap (heading and title terms count at half weight);
   * 0.20 × chunk rank.
2. Applies bonuses for a heading that names the asked aspect, for entity match, for numbers when the question asks
   for a quantity or deadline, for approval or role words in "who" questions, and for capacity ≥ head-count.
3. Applies a penalty for "pointer" sentences ("…are covered in the X document").
4. Picks the best sentence, adds an adjacent supporting sentence or further list items for "which / options"
   questions, and prefixes the section heading when the sentence alone would not name its subject ("Orchid Business
   Hotel, Hinjewadi — The Lotus Hall seats 40…").

**Verification.** A sub-intent is answered only if the IDF-weighted coverage of its specific terms in the evidence is
≥ 0.4, or, when some query words are out of vocabulary, if dense support is ≥ 0.55. Otherwise the answer says *"I could
not verify X from the documents."*

Results: every unanswerable test question (parking, gym, pets, dress code, moonlighting) was flagged. False abstention
on answerable sub-intents is 4.5% on test-1 and 6.2% on test-2.

## 7. Data provenance

* **Tessera corpus.** A fictional company written for this project with AI assistance. Every fact is internally
  consistent (the approval matrix agrees with the travel, hotel and event policies). It is marked synthetic in the
  README.
* **18F Handbook travel-and-leave pages.** Public domain / CC0, copied verbatim with the licence file.
* **Gold labels.** Section IDs and key facts were written by the team before the engine was run on each split. File
  hashes are in `bench/scenarios/FROZEN_SHA256.txt`.

No external data, no web access and no user profile is used at run time. Memory is strictly per session.

## 8. Trade-offs and failure-mode mitigations

| Risk (guide pitfall) | Mitigation in Kairos | Evidence |
|---|---|---|
| Eager retrieval on noise | dangling-phrase rule, IDF stability threshold, anchors | false triggers 0%; eager ablation 14% |
| Context loss on late constraints | versioned answers, delta queries, preserved claims | G5 100%; restart keeps 0% |
| Citation hallucination | extractive claims; LLM output re-verified sentence by sentence | 0 fabricated IDs; mock-LLM test |
| Querying on presentation-only turns | presentation detector, last-version transformer | 0 queries on reformat turns |
| Over-fragmenting sub-queries | coordination rules, merge guard | over-fragmentation 0% |
| Stale speculative result reused | reuse only if every informative final word was covered | "…made after travel" re-queried |
| Wrong scope (Bengaluru page for a Pune question) | scope-conflict penalty, entity-conditional scoping | evidence recall 89.5% / 95.8% |
| Unanswerable sub-intent answered anyway | coverage check → explicit uncertainty | 100% of unanswerable turns flagged |
| Raw ASR (no punctuation, lower case) | segmentation also keys on connectives and question words | G2 97.9%, G3 94.4% on ASR-style replay |

**Trade-offs we accepted.** The answers are extractive: correct and cited, but less fluent than an LLM's (the optional
polish mode addresses this). Synonym gaps remain (see `EVALUATION.md`). Thresholds were tuned on a 16-session dev split
and may need re-tuning on a very different corpus.
