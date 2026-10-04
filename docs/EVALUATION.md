# Evaluation methodology and edge-case analysis

All numbers come from `python -m bench.run_bench`. It replays every scenario through the real engine on a virtual
stream clock and writes `reports/metrics.md` and `reports/results.json`. Per-turn records and the full telemetry of the
last run are in `reports/traces/`.

## 1. Protocol

* **Splits.**

  | Split | Sessions | Turns | Use |
  |---|---|---|---|
  | `tessera_dev` | 16 | 21 | tuning |
  | `tessera_test` (test-1) | 40 | 54 | held-out; one round of error analysis after its first run |
  | `tessera_test2` (test-2) | 16 | 21 | frozen *before* that error analysis, never used for anything but reporting |
  | `handbook18f_test` | 12 | 13 | real CC0 corpus, written before the engine ever ran on it |

  Hashes are in `bench/scenarios/FROZEN_SHA256.txt`. The first test-1 run (before error analysis) is preserved in
  `reports/results_test1_before_error_analysis.json`. In that run all gates already passed; evidence recall was 80.6%
  and key-fact recall 72.5%.
* **Streaming.** Utterances are cut into 2–5-word chunks at 2.6 words per second (≈ 155 wpm), with a seeded RNG.
  End-of-utterance is detected 250 ms after the last word. Guide Example 1 is replayed with its exact timestamps.
  An *ASR-style* replay lowercases everything and strips punctuation.
* **Gold.** Each turn has `needs_retrieval`, a turn type, sub-intents with their acceptable section IDs, key facts
  (alternative strings) and unanswerable aspects.
* **One ingestion fix after the first 18F run.** Hard-wrapped Markdown lines were being split into separate sentences.
  We found this while preparing the demo, not from the 18F gold labels, and fixed it in the parser. It does not
  change any Tessera result. On 18F it raised key-fact recall from 66.7% to 72.2%; evidence recall stayed at 72.2%.
  Nothing else was changed for 18F.

## 2. Metrics

| Metric | Definition |
|---|---|
| G2 early retrieval | share of retrieval-needing turns whose first retrieval started before the utterance ended |
| False triggers | share of no-retrieval turns (chit-chat, reformat) where any retrieval fired |
| G3 multi-intent | share of compound turns (≥ 2 answerable gold sub-intents) where ≥ 2 *distinct* sub-queries each hit a *distinct* gold sub-intent in their top-4 (greedy matching) |
| Sub-intent recall@4 / MRR@4 | per gold sub-intent: best rank of a gold section among the sub-queries actually used |
| Answer evidence recall | per gold sub-intent: a gold section is cited in the answer |
| Key-fact recall | per key-fact group: any accepted string appears in the answer |
| G4 citation support | every non-preserved claim is re-checked against its cited chunk: all numbers present, ≥ 60% content-stem overlap; plus a count of citations that do not exist in the corpus |
| G5 refinement | the new version's parent is the previous version; no previous sub-query re-executed; every previous citation is preserved or explicitly superseded; the delta added evidence or an explicit uncertainty |
| G6 trace coverage | every turn has all required event types for its turn type, and every event validates against `telemetry.schema.json` |
| TTFT | wall-clock ms from receiving end-of-utterance to the first answer token |

## 3. Ablations

All ablations run on test-1. Each row changes one switch in `kairos/config.py`.

* **Hybrid vs dense-only vs BM25-only.**
  * On Tessera, hybrid has the best MRR (0.964, against 0.878 dense-only and 0.945 BM25-only).
  * On the real 18F corpus, hybrid wins on every metric:

    | Variant | Evidence recall | Key facts | MRR@4 |
    |---|---|---|---|
    | Hybrid | 72.2% | 66.7% | 0.72 |
    | Dense-only | 66.7% | 61.1% | 0.68 |
    | BM25-only | 66.7% | 61.1% | 0.71 |

  * BM25 alone is very strong on a small, clean, consistently written corpus. Real documents, with headings, links and
    varied wording, are where the dense half pays off.
* **Rule-based streaming controller vs eager vs end-only.**

  | Controller | Early rate | False triggers | Queries per turn | Wasted per turn |
  |---|---|---|---|---|
  | Eager (retrieve every chunk) | 100% | 14.3% | 3.6 | 2.3 |
  | End-only | 0% | 0% | — | — |
  | Streaming (Kairos) | 100% | 0% | 1.9 | 0.6 |

  Only the streaming controller gets both early retrieval and zero false triggers.
* **Decomposition on/off.**

  | Decomposition | G3 | Evidence recall | Key facts |
  |---|---|---|---|
  | Off | 0% | 62.7% | 62.3% |
  | On | 100% | 89.5% | 85.5% |

  This is the single largest contributor.
* **Refine vs restart** (refinement turns only).

  | Strategy | Queries | Earlier citations kept | Key facts |
  |---|---|---|---|
  | Refine | 1.75 | 100% | 78% |
  | Restart | 2.12 | 0% | 33% |

## 4. Analysed edge-case failures

The remaining failures are listed in `results.json → main._failures`. Five representative ones follow.

**E1. Synonym gap: "cap" vs "limit" (T12, "What's the hotel cap in Germany…?").**
* Retrieval is correct: Doc_03 §3 ranks first.
* The answer says *"I could not verify cap for Germany"*. The section says "reimbursed up to USD 210 per night in
  Germany" and never uses "cap" or "limit". "cap" is a rare corpus term (it occurs once, as "capped"), so it carries a
  high IDF weight and coverage falls below 0.4. Dense support (static embeddings) does not reach 0.55.
* This is a deliberate trade-off: the same strictness flags "gym memberships" and "pets" correctly.
* Fix path: a small cross-encoder or the optional LLM as a *verifier* for borderline coverage, at roughly 20–50 ms per
  sub-intent.

**E2. Synonym gap in a late detail (dev D03, "And if a cyclone warning forces us to cancel?").**
* The rule lives in Doc_12 §5: "If a government advisory or a severe weather alert prevents the event…".
* Neither "cyclone" nor "warning" is in the corpus vocabulary, and WordLlama's similarity for cyclone/weather is only
  0.18.
* The refinement keeps v1 and reports that it could not verify the new condition. That is safe, but unhelpful.
* Fix path: corpus-side query expansion with pseudo-relevance feedback, or a contextual embedding model.

**E3. Lexical attraction to the wrong section (T18, "How far in advance should I book domestic flights?").**
* Doc_04 §1 ("Domestic trips should be booked at least 14 days before departure") ranks first in retrieval.
* The composer still picks the Doc_02 §2 sentence "Domestic flights are booked in economy class through TravelHub",
  which shares "domestic", "flights" and "booked".
* The answer-type bonus for number-seeking questions narrows the gap but does not close it, because "in advance" ↔
  "before departure" has no lexical overlap.
* Fix path: answer-type classification with a learned (not keyword) model.

**E4. Implicit relational question (T40, "Can my team do an overnight offsite…?").**
* The governing sentence is "Overnight offsites need VP approval" (Doc_36 §2). The question asks about permission, not
  approval, so the composer prefers an alcohol sentence that shares more terms.
* Similar: "two days before a customer workshop" (T30) is attracted to "within 2 days" in the *After the workshop*
  section.

**E5. Multi-hop facts (T37, "the per diem for Chennai").**
* The corpus never states a Chennai per diem. Chennai is a Tier B city (Doc_02 §3), and the Tier B per diem is
  ₹1,500 (Doc_06 §2).
* Entity-conditional scoping correctly drops the Chennai requirement and retrieves the tier table. The answer cites
  both sections, so the user can join them, but Kairos does not state "₹1,500 for Chennai" itself.
* Fix path: an entity-to-category table mined from list sentences ("Tier B cities (Pune, Hyderabad, Chennai…)").

**Real-world corpus (18F).**
* Multi-intent identification is 60% (3 of 5 compound turns).
* Both failures (H01, H08) have the same cause. The second clause is elliptical ("how long do I have to *use them*",
  "are baggage fees covered"), so it inherits a context entity. On this corpus the entity lexicon also contains
  capitalised *link text* mined from the handbook ("Comp Time Fact Sheet", "Incidental Expenses"). That entity scopes
  the second sub-query to the wrong section.
* False abstention is 23% for the same reason (*"I could not verify use for Comp Time Fact Sheet"*), and because 18F
  text is full of link-only sentences and jargon ("M&IE", "POV").
* Fix path: filter entities by salience (document frequency, heading position) before they can scope a query.
* We did not relax the gold or retune on 18F after seeing these results.

## 5. Known limitations of the evaluation itself

* The Tessera corpus and its scenarios were written by the same team that built the system. The 18F results, and the
  test-2 split frozen before error analysis, are there to limit that bias, but they are small: 12 and 16 sessions.
* Speech is simulated from transcripts, as the theme allows. The live console also accepts real browser speech
  recognition, but no ASR-error benchmark was run beyond the lower-case, no-punctuation replay.
* Latency figures come from a 2-vCPU cloud container. The latency-hiding table *simulates* remote retrieval cost from
  real stream timestamps rather than measuring a deployed vector database.
