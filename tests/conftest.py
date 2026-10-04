import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from kairos.config import Config  # noqa: E402
from kairos.engine import Engine  # noqa: E402


@pytest.fixture(scope="session")
def engine():
    return Engine(Config(corpus_path=str(ROOT / "corpus" / "tessera"), write_traces=False))


def run_turn(session, chunks, end_t):
    session.begin_turn()
    for t, text in chunks:
        session.feed(t, text)
    return session.end(end_t)
