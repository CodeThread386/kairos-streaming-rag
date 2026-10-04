from pathlib import Path

from kairos.config import Config
from kairos.index import build_index
from kairos.ingest import load_corpus

ROOT = Path(__file__).resolve().parents[1]


def test_tessera_sections_have_stable_ids():
    chunks = load_corpus(ROOT / "corpus" / "tessera")
    ids = {c.chunk_id for c in chunks}
    assert "Doc_12 §2" in ids and "Doc_04 §3" in ids
    assert len(ids) == len(chunks)


def test_real_markdown_corpus_parses():
    chunks = load_corpus(ROOT / "corpus" / "handbook18f")
    assert len(chunks) > 150
    assert not any(c.path.lower().startswith("license") for c in chunks)
    assert any(c.heading == "Calculating travel comp time" for c in chunks)


def test_corpus_derived_lexicons():
    ix = build_index(str(ROOT / "corpus" / "tessera"), Config(), use_cache=False)
    assert {"pune", "bengaluru", "hyderabad"} <= ix.scope_entities
    assert "orchid business hotel" in ix.find_entities("can we use the orchid business hotel")
    assert "lakeview convention centre" in ix.find_entities("do we need a contract for Lakeview")


def test_hybrid_search_finds_section():
    cfg = Config()
    ix = build_index(str(ROOT / "corpus" / "tessera"), cfg)
    hits = ix.search("free cancellation window at Orchid Business Hotel", cfg, entities=["orchid business hotel"])
    assert hits[0].chunk_id == "Doc_12 §2"
