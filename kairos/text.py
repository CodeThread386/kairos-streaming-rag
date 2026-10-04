"""Text utilities: tokenisation, stemming, number normalisation, sentence splitting.

Everything here is corpus-agnostic. Domain knowledge (entities, aspect words, IDF) is
derived from the corpus at index time (see index.py), never hard-coded.
"""
from __future__ import annotations

import re
from functools import lru_cache

import snowballstemmer

_STEMMER = snowballstemmer.stemmer("english")

STOPWORDS = frozenset("""
a about again against all am an and any are aren't as at be because been being
between both but by can can't cannot could couldn't did didn't do does doesn't doing don't down during each
few for from further had hadn't has hasn't have haven't having he he'd he'll he's her here here's hers herself
him himself his how how's i i'd i'll i'm i've if in into is isn't it it's its itself let's me more most mustn't
my myself no nor not of off on once only or other ought our ours ourselves out own same shan't she she'd
she'll she's should shouldn't so some such than that that's the their theirs them themselves then there there's
these they they'd they'll they're they've this those through to too under until up very was wasn't we we'd we'll
we're we've were weren't what what's when when's where where's which while who who's whom why why's with won't
would wouldn't you you'd you'll you're you've your yours yourself yourselves also just get got like need needs
want wants please tell know let us thing things really sure okay ok um uh er ah hmm yeah yes hi hello hey thanks
thank whats wondering mentioned anyway basically actually kind sort maybe gonna wanna lot bit something anything
one ones will shall may might must
""".split())

# Words that, when they END an open clause, signal the speaker has not finished the thought.
DANGLING = frozenset("""
a an the to for in on at of with and or but about from by into onto my our your their its his her this that
these those is are was were be been being i we you they he she need want would like um uh so if when what
which who whom how do does did can could should will whether also plus as than then some any more most
please tell let me us get give show find where why there per around between during after before over under
including include includes regarding re versus vs via such each every both either neither just only even
""".split())

FILLER_PATTERNS = [
    r"\bum+\b", r"\buh+\b", r"\ber+\b", r"\bah+\b", r"\bhmm+\b", r"\byou know\b", r"\bsort of\b", r"\bkind of\b",
    r"\bbasically\b", r"\bi guess\b", r"\blike(?=\s+(?:\d|a\b|an\b|the\b|maybe\b|around\b|about\b))",
]
_FILLER_RE = re.compile("|".join(FILLER_PATTERNS), re.IGNORECASE)

_NUM_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
    "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40,
    "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90, "hundred": 100, "a hundred": 100,
}
_SCALE = {"thousand": 1_000, "lakh": 100_000, "lakhs": 100_000, "crore": 10_000_000, "crores": 10_000_000,
          "million": 1_000_000}

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")
_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[]?[A-Z0-9₹$])")


# irregular English forms the Snowball stemmer cannot conflate ("paid" vs "pay")
_IRREGULAR = {"paid": "pay", "told": "tell", "spent": "spend", "made": "make", "bought": "buy", "sent": "send",
              "held": "hold", "kept": "keep", "left": "leave", "met": "meet", "given": "give", "gave": "give",
              "taken": "take", "took": "take", "gone": "go", "went": "go", "done": "do", "did": "do", "brought": "bring",
              "lost": "lose", "won": "win", "chose": "choose", "chosen": "choose", "began": "begin", "begun": "begin",
              "known": "know", "knew": "know", "flew": "fly", "flown": "fly", "drove": "drive", "driven": "drive",
              "ate": "eat", "eaten": "eat", "wrote": "write", "written": "write", "stayed": "stay", "children": "child",
              "people": "people", "women": "woman", "men": "man", "fees": "fee"}


@lru_cache(maxsize=200_000)
def stem(token: str) -> str:
    return _STEMMER.stemWord(_IRREGULAR.get(token, token))


def normalise_numbers(text: str) -> str:
    """Rewrite spelled-out numbers and Indian scales into digits ("five lakh" -> "500000",
    "thirty people" -> "30 people") and strip thousands separators ("2,50,000" -> "250000")."""
    t = re.sub(r"(?<=\d),(?=\d)", "", text)

    def scale_sub(m: re.Match) -> str:
        num, scale = m.group(1).lower(), m.group(2).lower()
        base = float(num) if re.match(r"^\d+(\.\d+)?$", num) else _NUM_WORDS.get(num)
        if base is None:
            return m.group(0)
        return str(int(base * _SCALE[scale]))

    words = "|".join(sorted(map(re.escape, _NUM_WORDS), key=len, reverse=True))
    t = re.sub(rf"\b(\d+(?:\.\d+)?|{words})\s+({'|'.join(_SCALE)})\b", scale_sub, t, flags=re.IGNORECASE)

    def tens_units(m: re.Match) -> str:
        return str(_NUM_WORDS[m.group(1).lower()] + _NUM_WORDS[m.group(2).lower()])

    t = re.sub(r"\b(twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)[- ](one|two|three|four|five|six|seven|eight|nine)\b",
               tens_units, t, flags=re.IGNORECASE)

    def word_sub(m: re.Match) -> str:
        return str(_NUM_WORDS[m.group(0).lower()])

    simple = "|".join(k for k in _NUM_WORDS if " " not in k and k != "one")
    t = re.sub(rf"\b({simple})\b", word_sub, t, flags=re.IGNORECASE)
    return t


def strip_fillers(text: str) -> str:
    t = _FILLER_RE.sub(" ", text)
    t = re.sub(r"\s+,", ",", t)
    t = re.sub(r",\s*,+", ",", t)
    t = re.sub(r"^\s*(?:so|well|okay|ok|alright|right)\s*,?\s+", "", t, flags=re.IGNORECASE)
    return re.sub(r"\s{2,}", " ", t).strip(" ,")


def raw_tokens(text: str) -> list[str]:
    t = normalise_numbers(text.lower())
    t = t.replace("%", " percent ").replace("₹", " ").replace("$", " ")
    return _TOKEN_RE.findall(t)


def content_tokens(text: str) -> list[str]:
    """Lower-cased, number-normalised, stop-word-filtered, stemmed tokens."""
    out = []
    for tok in raw_tokens(text):
        if tok in STOPWORDS or len(tok) < 2 and not tok.isdigit():
            continue
        for part in tok.split("-") if "-" in tok else (tok,):
            if part and part not in STOPWORDS:
                out.append(stem(part))
    return out


def numbers_in(text: str) -> set[str]:
    return set(re.findall(r"\d+(?:\.\d+)?", re.sub(r"(?<=\d),(?=\d)", "", normalise_numbers(text))))


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    parts = _SENT_RE.split(text)
    return [p.strip() for p in parts if len(p.strip()) > 2]


def last_word(text: str) -> str:
    toks = re.findall(r"[a-zA-Z0-9']+", text.lower())
    return toks[-1] if toks else ""


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / max(1, len(a | b))
