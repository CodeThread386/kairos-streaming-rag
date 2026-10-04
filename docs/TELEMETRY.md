# Telemetry and observability

Every session writes one JSON event per line to `traces/<session_id>.jsonl`. The same events are pushed live over the
WebSocket, so the console's controller log and timeline are drawn from them. The benchmark validates every event
against [`telemetry.schema.json`](telemetry.schema.json) (JSON Schema 2020-12). **G6 trace coverage** is the share of
turns that have every event type their turn type requires, with every event valid. It is 100% on all splits.

## Common fields

| Field | Meaning |
|---|---|
| `event` | event type (below) |
| `session_id`, `turn_id`, `seq` | identity and strict ordering |
| `t_stream_s` | position on the speech timeline (seconds since the turn started, as spoken) |
| `wall_ms` | wall-clock milliseconds since the session started (real compute time) |

## Event types

| Event | Emitted when | Key fields |
|---|---|---|
| `session_started` | session opens | corpus, index stats, config (API key masked) |
| `chunk_received` | every transcript chunk | `text`, `chars`, `replace` (true = revised ASR hypothesis) |
| `controller_decision` | after every chunk and at the end | `decision` ∈ WAIT / RETRIEVE / DECOMPOSE_AND_RETRIEVE / SUPPRESS / REFINE / SYNTHESIZE, `reason`, clause, informativeness |
| `retrieval_started` | a query is fired | `query_id`, `query`, `trigger` ∈ provisional / upgrade / multi_intent / final / refinement / eager, `sub_intent` (query, weak context, entities, counts) |
| `retrieval_completed` | results are back | `latency_ms`, `hits` [{chunk_id, score, bm25, dense, boosts}], `top_score` |
| `sub_queries` | utterance end | final `sub_queries`, `speculative_reuse` [bool], `used_query_ids` |
| `synthesis_started` | composer starts | |
| `first_token` | first answer token is ready | `ttft_ms`, `mode` (extractive or llm), LLM info (tokens, dropped sentences) |
| `answer_version` | a version is created | `version`, `parent`, `mode` (answer / refinement / presentation), `text`, `citations`, `claims` [{text, citation, status}], `preserved`, `added`, `superseded`, `uncertainty` |
| `suppression` | no-retrieval turn | `reason` (presentation_restructure or conversational), `retrieval_required:false` |
| `uncertainty` | a sub-intent could not be verified | `messages` |
| `turn_summary` | end of turn | `latency` (ttft_ms, controller µs p50/max, retrieval ms total), `counts` (chunks, retrievals fired, before end, used, speculative hits, sub-intents), `early_retrieval`, `retrieval_lead_s`, `cost` (LLM input/output tokens, embedding calls, est_usd), `answer_version` |

## Example (guide Example 1, abridged)

```json
{"event":"chunk_received","turn_id":1,"t_stream_s":0.0,"text":"I need to plan a customer workshop in"}
{"event":"controller_decision","t_stream_s":0.0,"decision":"WAIT","reason":"intent incomplete: clause ends mid-phrase"}
{"event":"retrieval_started","t_stream_s":0.8,"query_id":"q1.1","trigger":"provisional","query":"I need to plan a customer workshop in Pune for 30 people"}
{"event":"retrieval_completed","t_stream_s":0.8,"query_id":"q1.1","latency_ms":1.9,"hits":[{"chunk_id":"Doc_11 §2","score":1.38}]}
{"event":"retrieval_started","t_stream_s":1.6,"query_id":"q1.2","trigger":"multi_intent","query":"I need the cancellation policy"}
{"event":"sub_queries","t_stream_s":2.1,"speculative_reuse":[true,true,true],"used_query_ids":["q1.1","q1.2","q1.3"]}
{"event":"first_token","t_stream_s":2.1,"ttft_ms":2.0,"mode":"extractive"}
{"event":"answer_version","version":1,"parent":null,"mode":"answer","citations":["Doc_11 §2","Doc_11 §3","Doc_11 §4","Doc_12 §2","Doc_09 §2","Doc_09 §5"]}
{"event":"turn_summary","early_retrieval":true,"retrieval_lead_s":1.3,"cost":{"llm_input_tokens":0,"llm_output_tokens":0,"est_usd":0.0}}
```

## Cost accounting

* **Default mode:** no LLM calls, so the cost is $0 per turn. The trace records the number of embedding and retrieval
  calls; both run in-process on CPU.
* **LLM polish mode:** input and output tokens are recorded per turn. The estimate is about 4 characters per token,
  unless the provider reports usage. Set `KAIROS_PRICE_IN_PER_MTOK` / `KAIROS_PRICE_OUT_PER_MTOK` to turn those counts
  into `est_usd`.
