"""Debug helper: show decomposition, retrieval hits and evidence diagnostics for one utterance."""
import sys
from kairos.config import Config
from kairos.engine import Engine

eng = Engine(Config(write_traces=False))
for utt in sys.argv[1:]:
    subs = eng.decomposer.decompose(utt)
    print("=" * 90, "\n", utt)
    for si in subs:
        own = [e for e in si.entities if e not in set(si.inherited)]
        hits = eng.index.search(si.query, eng.cfg, entities=own, numbers=si.numbers, context=si.context, context_weight=si.context_weight)
        print(f"  SUB  q={si.query!r} ctx={si.context!r} ents={si.entities} own={own} nums={si.numbers}")
        for h in hits:
            print(f"       {h.chunk_id:12s} {h.score:.3f} bm25={h.bm25:.2f} dense={h.dense:.2f} {h.boosts}")
    d = eng.composer.compose(subs, [eng.index.search(s.query, eng.cfg, entities=[e for e in s.entities if e not in set(s.inherited)], numbers=s.numbers, context=s.context, context_weight=s.context_weight) for s in subs])
    for x in eng.composer.last_diag:
        print("  DIAG", x)
    print("  ANS ", d.text())
