from conftest import run_turn


def test_guide_example_1_timeline(engine):
    s = engine.new_session(write_trace=False)
    rec = run_turn(s, [(0.0, "I need to plan a customer workshop in"), (0.8, "Pune for 30 people, and I need"),
                       (1.6, "the cancellation policy and the catering options.")], 2.1)
    ev = rec["retrieval_events"]
    assert ev[0]["timestamp_s"] == 0.8 and ev[0]["trigger"] == "provisional"     # early, before the end
    assert all(e["timestamp_s"] < 2.1 for e in ev)
    assert len(rec["sub_queries"]) == 3                                         # three intents isolated
    assert rec["telemetry"]["counts"]["speculative_hits"] == 3                   # nothing re-searched at the end
    cited = set(rec["citations"])
    assert cited & {"Doc_11 §3", "Doc_11 §4", "Doc_11 §2"}
    assert cited & {"Doc_12 §2", "Doc_12 §3"}
    assert cited & {"Doc_09 §2", "Doc_09 §3", "Doc_09 §4"}


def test_first_chunk_waits_on_dangling_phrase(engine):
    s = engine.new_session(write_trace=False)
    s.begin_turn()
    evs = s.feed(0.0, "I need to plan a customer workshop in")
    decs = [e for e in evs if e["event"] == "controller_decision"]
    assert decs and decs[0]["decision"] == "WAIT"
    assert not [e for e in evs if e["event"] == "retrieval_started"]


def test_over_fragmentation_guard(engine):
    subs = engine.decomposer.decompose("What are the cancellation and refund terms for The Deccan Courtyard?")
    assert len(subs) == 1


def test_self_repair(engine):
    subs = engine.decomposer.decompose("What's the per diem in the UK, sorry, I mean the United States, and the hotel limit there?")
    assert len(subs) == 2
    assert "United States" in subs[0].query and "UK" not in subs[0].query


def test_unanswerable_is_flagged_not_answered(engine):
    s = engine.new_session(write_trace=False)
    rec = run_turn(s, [(0.0, "Is there parking available"), (0.8, "at the Orchid Business Hotel?")], 1.6)
    assert rec["uncertainty"] and "parking" in rec["uncertainty"].lower()
    assert rec["citations"] == []
