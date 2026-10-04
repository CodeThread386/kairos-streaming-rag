from conftest import run_turn


def test_refinement_patches_instead_of_restarting(engine):
    s = engine.new_session(write_trace=False)
    r1 = run_turn(s, [(0.0, "Summarize the travel reimbursement rule"), (0.9, "for an employee trip.")], 1.6)
    q1 = set(r1["sub_queries"])
    r2 = run_turn(s, [(0.0, "The trip was international"), (0.8, "and the booking was made"), (1.5, "after travel.")], 2.0)
    v2 = r2["answer_version"]
    assert r2["turn_type"] == "refinement"
    assert v2["parent"] == r1["answer_version"]["version"]
    assert not (q1 & {e["query"] for e in r2["retrieval_events"]})            # original query not re-run
    assert set(r1["citations"]) <= set(v2["preserved"])                       # prior evidence kept
    assert "Doc_04 §3" in v2["added"] and "Senior Director" in r2["answer"]


def test_presentation_turn_suppresses_retrieval(engine):
    s = engine.new_session(write_trace=False)
    run_turn(s, [(0.0, "What's the hotel limit"), (0.7, "for a trip to Pune?")], 1.4)
    r = run_turn(s, [(0.0, "Please repeat your last"), (0.7, "answer in two bullets.")], 1.3)
    assert r["retrieval_required"] is False and r["reason"] == "presentation_restructure"
    assert r["retrieval_events"] == []
    assert r["answer"].count("•") == 2


def test_slot_correction_supersedes(engine):
    s = engine.new_session(write_trace=False)
    run_turn(s, [(0.0, "What venues do we have"), (0.8, "in Pune for a 30 person training?")], 1.8)
    r = run_turn(s, [(0.0, "No wait, I meant"), (0.7, "Hyderabad, not Pune.")], 1.4)
    v = r["answer_version"]
    assert v["superseded"] and any(c.startswith("Doc_14") for c in v["added"])


def test_chitchat_has_no_retrieval(engine):
    s = engine.new_session(write_trace=False)
    r = run_turn(s, [(0.0, "Hi there,"), (0.5, "good morning!")], 1.0)
    assert r["retrieval_required"] is False and not r["retrieval_events"]
