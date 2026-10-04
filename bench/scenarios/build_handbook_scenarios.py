"""Generality set on a REAL public-domain corpus (18F Handbook, travel-and-leave pages, CC0).
Written before the engine was ever run on this corpus; never used for tuning.
Gold sections are given as (file, heading) and resolved to section IDs at evaluation time."""
import json

def G(f, h):
    return {"file": f, "heading": h}

def T(utt, sub=None, kind="new", key=None):
    return {"utterance": utt, "gold": {"needs_retrieval": kind in ("new", "refinement"), "turn_type": kind,
            "sub_intents": [{"name": n, "relevant_ref": r} for n, r in (sub or [])], "key_facts": key or [],
            "unanswerable_aspects": []}}

def S(sid, cat, *turns):
    return {"id": sid, "category": cat, "turns": list(turns)}

TT = "travel-and-leave-policies/travel-guide-3-travel.md"
ROWS = [
 S("H01", "multi_intent", T("How are travel comp time hours calculated, and how long do I have to use them?",
   [("calculating travel comp time", [G("overtime.md", "Calculating travel comp time"), G("overtime.md", "Understanding travel comp time")]),
    ("using travel comp time", [G("overtime.md", "Using travel comp time")])], key=[["elapsed time"], ["one year", "26 pay periods"]])),
 S("H02", "multi_intent", T("Which receipts do I need to keep when traveling, and is laundry reimbursed?",
   [("receipts", [G(TT, "Which receipts must I keep?")]), ("laundry", [G(TT, "Laundry expenses")])], key=[["$75"], ["4 consecutive nights"]])),
 S("H03", "single_intent", T("What does use or lose mean for annual leave?",
   [("use or lose", [G("leave.md", "Use or Lose")])], key=[["240 hour"]])),
 S("H04", "multi_intent", T("Can I take leave to vote, and what about donating blood?",
   [("voting", [G("leave.md", "Voting and poll volunteering")]), ("blood donation", [G("leave.md", "Blood, bone marrow, and organ donation")])],
   key=[["own time", "annual leave"], ["4 hours"]])),
 S("H05", "single_intent", T("How do I request paid parental leave before the baby is born?",
   [("request PPL prior to birth", [G("paid-parental-leave.md", "Prior to birth/placement"), G("paid-parental-leave.md", "Requesting paid parental leave")])],
   key=[["Paid Parental Leave Request", "HR Specialist"]])),
 S("H06", "single_intent", T("What do I have to do when I come back from personal foreign travel?",
   [("return from foreign travel", [G("travel-and-leave-policies/unofficial-foreign-travel.md", "When You Return")])], key=[["five days", "De-briefing"]])),
 S("H07", "single_intent", T("How do I get advanced sick leave approved?",
   [("advanced sick leave approval", [G("advanced-sick-leave.md", "Getting approval for advanced sick leave"), G("leave.md", "Advanced sick leave")])],
   key=[["Advanced Sick Leave Request", "HR Specialist", "supervisor"]])),
 S("H08", "multi_intent", T("How do meals and incidental expenses work, and are baggage fees covered?",
   [("M&IE", [G(TT, "Meals and Incidental Expenses (M&IE)"), G(TT, "M&IE and use of the GSA travel card")]), ("baggage fees", [G(TT, "Baggage fees")])],
   key=[["per diem", "receipts"], ["one bag"]])),
 S("H09", "single_intent", T("Who is eligible for FMLA?", [("FMLA eligibility", [G("fmla.md", "Eligibility")])], key=[["one year"]])),
 S("H10", "late_detail",
   T("How do I claim mileage when I drive my own car on travel?", [("mileage", [G(TT, "Mileage, Parking, and Tolls")])], key=[["POV", "mileage"]]),
   T("And what about internet at the hotel?", [("internet", [G(TT, "Internet service")])], kind="refinement", key=[["justification"]])),
 S("H11", "single_intent", T("What is court leave for jury duty?", [("court leave", [G("leave.md", "Court leave (jury duty)")])], key=[["Court Leave"]])),
 S("H12", "multi_intent", T("If I move to a new state, what changes with my taxes and my benefits?",
   [("tax changes", [G("travel-and-leave-policies/moving.md", "Changes in tax")]), ("benefit changes", [G("travel-and-leave-policies/moving.md", "Changes in benefits")])],
   key=[["withholding"], ["health insurance", "Benefits Specialist"]])),
]
with open("handbook18f_test.jsonl", "w", encoding="utf-8") as f:
    for r in ROWS:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(len(ROWS), "sessions")
