"""Second held-out split (test2). Written AFTER the first test run but BEFORE any change motivated by
test-1 failures, and never used for tuning. Reported separately as the clean held-out number."""
import json
from build_tessera_scenarios import S, T

TEST2 = [
 S("U01", "multi_intent", T("Which Bengaluru venue works for a 100 person launch, and do outside caterers cost extra at the Indigo Loft?",
   [("Bengaluru venue for 100", ["Doc_13 §2", "Doc_13 §3"]), ("Indigo Loft external catering", ["Doc_13 §4"])],
   key=[["Nandi", "Banyan"], ["no extra fee", "at no extra"]])),
 S("U02", "late_detail",
   T("How do I claim an expense when I've lost the receipt?", [("missing receipt", ["Doc_05 §3"])], key=[["₹2,000"]]),
   T("It was a dinner abroad that I paid in euros.", [("foreign currency receipt", ["Doc_03 §4", "Doc_05 §2"])], kind="refinement",
     key=[["USD 25", "foreign currency"]])),
 S("U03", "multi_intent", T("What's the international roaming limit, and can I buy a local SIM instead?",
   [("roaming", ["Doc_30 §2"]), ("local SIM", ["Doc_30 §3"])], key=[["₹5,000"], ["receipt", "reimbursed"]])),
 S("U04", "single_intent", T("How much notice does Brew & Bite need for same-day lunch boxes?",
   [("Brew & Bite same-day", ["Doc_09 §4"])], key=[["10 AM"]])),
 S("U05", "presentation",
   T("What are the security rules for working remotely?", [("remote security", ["Doc_20 §4"])], key=[["VPN"]]),
   T("Say that again but shorter.", kind="presentation")),
 S("U06", "multi_intent", T("um what's the per diem in germany and uh how long does a schengen visa take",
   [("per diem Germany", ["Doc_06 §3"]), ("Schengen visa time", ["Doc_25 §2"])], key=[["USD 90"], ["15 working days"]])),
 S("U07", "unanswerable", T("Does the company reimburse spa treatments on business trips and what about parking fines?",
   [("spa", ["Doc_02 §5"]), ("parking fines", [])], key=[["spa"]], unans=["parking fines"])),
 S("U08", "late_detail",
   T("Who approves a hotel that's above the city limit?", [("hotel above limit approval", ["Doc_27 §2", "Doc_35 §3"])], key=[["Director"]]),
   T("It's about 30% above.", [("more than 20% above", ["Doc_27 §2", "Doc_35 §3"])], kind="refinement", key=[["VP"]])),
 S("U09", "chitchat", T("Okay, thank you so much!", kind="chitchat")),
 S("U10", "multi_intent", T("When do laptops get replaced and what's the rule for personal phones accessing company email?",
   [("laptop refresh", ["Doc_22 §1"]), ("personal phones", ["Doc_22 §3"])], key=[["3 years"], ["work profile"]])),
 S("U11", "late_detail",
   T("What's the cancellation policy at Lakeview Convention Centre?", [("Lakeview cancellation", ["Doc_15 §1"])], key=[["30 days"]]),
   T("What about Pearl Residency?", [("Pearl Residency cancellation", ["Doc_15 §3"])], kind="refinement", key=[["10 days"]])),
 S("U12", "multi_intent", T("Tell me the visitor registration rule, the escort requirement, and the guest Wi-Fi setup.",
   [("visitor registration", ["Doc_29 §1", "Doc_28 §2"]), ("escort", ["Doc_29 §2"]), ("guest wifi", ["Doc_29 §4"])],
   key=[["pre-registered", "photo ID"], ["escorted"], ["access code"]])),
 S("U13", "single_intent", T("How much are parking and tolls reimbursed when I drive for work?",
   [("parking and tolls", ["Doc_08 §4"])], key=[["actual cost"]])),
 S("U14", "unanswerable", T("Is there a company policy on moonlighting or side jobs?", [("moonlighting", [])], unans=["moonlighting"])),
 S("U15", "multi_intent", T("For a team offsite, what's the budget per person and who approves it?",
   [("offsite budget", ["Doc_36 §1"]), ("offsite approval", ["Doc_36 §2"])], key=[["₹3,000"], ["department head"]])),
 S("U16", "late_detail",
   T("Can I extend a business trip for a holiday?", [("personal extension", ["Doc_04 §4"])], key=[["fare difference", "paid by the employee"]]),
   T("Thanks, and summarize that in one line.", kind="presentation")),
]
with open("tessera_test2.jsonl", "w", encoding="utf-8") as f:
    for r in TEST2:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(len(TEST2), "sessions", sum(len(r["turns"]) for r in TEST2), "turns")
