import json
import time
from pathlib import Path

import jsonschema

from conftest import run_turn
from kairos.config import Config
from kairos.engine import Engine
from kairos.synthesize import verify_claim

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = json.loads((ROOT / "docs" / "telemetry.schema.json").read_text())


def test_every_claim_is_supported_and_every_event_is_valid(engine):
    s = engine.new_session(write_trace=False)
    run_turn(s, [(0.0, "What's the per diem in Singapore,"), (1.0, "and do I need to get my trip approved in advance?")], 2.4)
    run_turn(s, [(0.0, "Okay, got it.")], 0.6)
    v = s.versions[0]
    for c in v.claims:
        ch = engine.index.chunks[engine.index.by_id[c.chunk_id]]
        assert verify_claim(c.text, ch.text, ch.heading)[0]
    validator = jsonschema.Draft202012Validator(SCHEMA)
    for e in s.tracer.events:
        assert not list(validator.iter_errors(e)), e


def test_llm_polish_cannot_add_unsupported_claims():
    from mock_llm import serve
    serve(18766)
    time.sleep(0.8)
    cfg = Config(corpus_path=str(ROOT / "corpus" / "tessera"), write_traces=False, synthesis_mode="llm",
                 llm_base_url="http://127.0.0.1:18766/v1", llm_model="mock")
    eng = Engine(cfg)
    s = eng.new_session(write_trace=False)
    r = run_turn(s, [(0.0, "What's the cancellation"), (0.6, "policy at Orchid Business Hotel?")], 1.4)
    assert "Doc_99" not in r["answer"] and "123456" not in r["answer"] and "spa" not in r["answer"]
    assert r["citations"] == ["Doc_12 §2"]
    ft = [e for e in s.tracer.events if e["event"] == "first_token"][-1]
    assert ft["mode"] == "llm" and ft["llm"]["dropped"] == 2
