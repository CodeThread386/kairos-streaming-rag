"""Corpus ingestion: any folder of .md/.txt/.html/.json/.jsonl/.pdf/.docx files becomes a list of
section-level chunks with stable citation IDs of the form ``<Doc_ID> §<section>``.

Section numbering rules
* A heading that already carries a section marker ("## §3 Fees", "## 3. Fees", "Section 3") keeps it.
* Otherwise headings are numbered sequentially per document (§1, §2 ...); text before the first
  heading becomes §0.
* Sections longer than ``max_passage_words`` are split into passages §N.1, §N.2 ... on paragraph
  / sentence boundaries, so citations stay precise on long real-world documents.
"""
from __future__ import annotations

import hashlib
import html
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from .text import split_sentences

SUPPORTED = {".md", ".markdown", ".txt", ".html", ".htm", ".json", ".jsonl", ".pdf", ".docx"}
IGNORED_PREFIXES = ("license", "readme", "changelog", "contributing", ".")


@dataclass
class Chunk:
    chunk_id: str            # "Doc_12 §2"
    doc_id: str
    section_id: str          # "§2" / "§2.1"
    doc_title: str
    heading: str
    heading_path: list[str]
    text: str
    path: str
    sentences: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"chunk_id": self.chunk_id, "doc_id": self.doc_id, "section_id": self.section_id,
                "doc_title": self.doc_title, "heading": self.heading, "heading_path": self.heading_path,
                "text": self.text, "path": self.path}


# --------------------------------------------------------------------------- cleaning
def _clean_markdown(text: str) -> str:
    t = re.sub(r"\{%.*?%\}", " ", text, flags=re.S)            # liquid / jinja tags
    t = re.sub(r"\{\{.*?\}\}", " ", t, flags=re.S)
    t = re.sub(r"<!--.*?-->", " ", t, flags=re.S)
    t = re.sub(r"```.*?```", " ", t, flags=re.S)                 # code fences
    t = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", t)                  # images
    t = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", t)               # links -> anchor text
    t = re.sub(r"\[([^\]]+)\]\[[^\]]*\]", r"\1", t)
    t = re.sub(r"https?://\S+", " ", t)
    t = re.sub(r"<[^>]+>", " ", t)                               # inline html
    t = html.unescape(t)
    t = re.sub(r"(\*\*|__|\*|_|`)(?=\S)(.+?)(?<=\S)\1", r"\2", t)  # emphasis
    t = re.sub(r"^\s*[-*+]\s+", "\n", t, flags=re.M)             # list items start a new block
    t = re.sub(r"^\s*\d+\.\s+", "\n", t, flags=re.M)             # numbered list markers
    t = re.sub(r"^\s*>\s?", "", t, flags=re.M)
    t = re.sub(r"\|", " ", t)
    t = re.sub(r"[ \t]+", " ", t)
    return t.strip()


def _as_sentenceish(text: str) -> str:
    """Re-join hard-wrapped lines inside a paragraph, keep blocks (paragraphs, list items) apart, and end every
    block with punctuation so sentence splitting works on list-heavy and 80-column-wrapped markdown."""
    blocks = [b for b in re.split(r"\n\s*\n", text) if b.strip()]
    out = []
    for b in blocks:
        joined = re.sub(r"\s*\n\s*", " ", b.strip())
        joined = re.sub(r"\s{2,}", " ", joined)
        out.append(joined if re.search(r"[.!?:;]$", joined) else joined + ".")
    return " ".join(out)


def _front_matter(text: str) -> tuple[dict, str]:
    m = re.match(r"^---\s*\n(.*?)\n---\s*\n", text, flags=re.S)
    if not m:
        return {}, text
    meta = {}
    for line in m.group(1).splitlines():
        mm = re.match(r"^([A-Za-z0-9_\-]+)\s*:\s*(.*)$", line)
        if mm and mm.group(2).strip() and not mm.group(2).strip().startswith(("[", "{")):
            meta[mm.group(1).strip().lower()] = mm.group(2).strip().strip("\"'")
    return meta, text[m.end():]


_SEC_MARK = re.compile(r"^(?:§\s*(\d+(?:\.\d+)*)|section\s+(\d+(?:\.\d+)*)[:.)]?|(\d+(?:\.\d+)*)[.)])\s*(.*)$", re.I)


# --------------------------------------------------------------------------- parsers
def _parse_markdown(raw: str, fallback_title: str) -> tuple[dict, str, list[tuple[int, str, str]]]:
    meta, body = _front_matter(raw)
    title = meta.get("title", "")
    sections: list[tuple[int, str, str]] = []   # (level, heading, text)
    cur_level, cur_head, buf = 0, "", []
    for line in body.splitlines():
        hm = re.match(r"^(#{1,6})\s+(.*?)\s*#*\s*$", line)
        if hm:
            if buf or cur_head:
                sections.append((cur_level, cur_head, "\n".join(buf)))
            cur_level, cur_head, buf = len(hm.group(1)), _clean_markdown(hm.group(2)), []
        else:
            buf.append(line)
    sections.append((cur_level, cur_head, "\n".join(buf)))
    # an H1 that just repeats the title is not a section
    if not title:
        h1 = next((h for lvl, h, _ in sections if lvl == 1), "")
        title = h1 or fallback_title
    cleaned = []
    for lvl, head, txt in sections:
        if lvl == 1 and head.strip().lower() == title.strip().lower():
            head, lvl = "", 0
        cleaned.append((lvl, head, _as_sentenceish(_clean_markdown(txt))))
    return meta, title, cleaned


def _parse_plain(raw: str, fallback_title: str) -> tuple[dict, str, list[tuple[int, str, str]]]:
    blocks = [b.strip() for b in re.split(r"\n\s*\n", raw) if b.strip()]
    title = blocks[0].splitlines()[0][:120] if blocks else fallback_title
    return {}, title, [(0, "", _as_sentenceish(b)) for b in blocks]


def _parse_html(raw: str, fallback_title: str):
    t = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", raw, flags=re.S | re.I)
    tm = re.search(r"<title>(.*?)</title>", t, flags=re.S | re.I)
    t = re.sub(r"<h([1-6])[^>]*>(.*?)</h\1>", lambda m: "\n" + "#" * int(m.group(1)) + " " + re.sub("<[^>]+>", "", m.group(2)) + "\n", t, flags=re.S | re.I)
    t = re.sub(r"<(p|li|br|div|tr)[^>]*>", "\n", t, flags=re.I)
    t = re.sub(r"<[^>]+>", " ", t)
    meta, title, secs = _parse_markdown(html.unescape(t), fallback_title)
    if tm:
        title = html.unescape(tm.group(1)).strip() or title
    return meta, title, secs


def _parse_pdf(path: Path, fallback_title: str):
    from pypdf import PdfReader
    reader = PdfReader(str(path))
    pages = [(p.extract_text() or "") for p in reader.pages]
    secs = [(0, f"Page {i + 1}", _as_sentenceish(txt)) for i, txt in enumerate(pages) if txt.strip()]
    return {}, fallback_title, secs


def _parse_docx(path: Path, fallback_title: str):
    import docx  # python-docx
    d = docx.Document(str(path))
    secs, head, buf, title = [], "", [], ""
    for p in d.paragraphs:
        style = (p.style.name or "").lower() if p.style is not None else ""
        if style.startswith("title") and not title:
            title = p.text.strip()
        elif style.startswith("heading"):
            if buf or head:
                secs.append((1, head, _as_sentenceish("\n".join(buf))))
            head, buf = p.text.strip(), []
        elif p.text.strip():
            buf.append(p.text.strip())
    secs.append((1, head, _as_sentenceish("\n".join(buf))))
    return {}, title or fallback_title, secs


def _parse_json_records(path: Path) -> list[tuple[dict, str, list[tuple[int, str, str]]]]:
    raw = path.read_text(encoding="utf-8")
    recs = [json.loads(l) for l in raw.splitlines() if l.strip()] if path.suffix == ".jsonl" else json.loads(raw)
    if isinstance(recs, dict):
        recs = recs.get("documents") or recs.get("docs") or [recs]
    out = []
    for r in recs:
        if not isinstance(r, dict):
            continue
        meta = {"doc_id": str(r.get("doc_id") or r.get("id") or "")}
        title = str(r.get("title") or r.get("name") or meta["doc_id"])
        if isinstance(r.get("sections"), list):
            secs = [(1, str(s.get("heading") or s.get("title") or ""), _as_sentenceish(str(s.get("text") or s.get("content") or "")))
                    for s in r["sections"] if isinstance(s, dict)]
        else:
            body = str(r.get("text") or r.get("content") or r.get("body") or "")
            secs = _parse_markdown(body, title)[2]
        out.append((meta, title, secs))
    return out


# --------------------------------------------------------------------------- chunking
def _split_passages(text: str, max_words: int) -> list[str]:
    words = text.split()
    if len(words) <= int(max_words * 1.3):
        return [text]
    sents, out, cur = split_sentences(text), [], []
    for s in sents:
        if cur and len(" ".join(cur + [s]).split()) > max_words:
            out.append(" ".join(cur))
            cur = []
        cur.append(s)
    if cur:
        out.append(" ".join(cur))
    return out


def _doc_id_from(path: Path, root: Path, seen: set[str]) -> str:
    base = re.sub(r"[^A-Za-z0-9_\-]+", "_", path.stem).strip("_") or "doc"
    did = base
    if did in seen:
        parent = re.sub(r"[^A-Za-z0-9_\-]+", "_", path.parent.name)
        did = f"{parent}_{base}"
    n = 2
    while did in seen:
        did, n = f"{base}_{n}", n + 1
    return did


def _build_chunks(doc_id: str, title: str, secs: list[tuple[int, str, str]], rel: str, max_words: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    counter = 0
    path_stack: list[tuple[int, str]] = []
    for level, head, text in secs:
        if not text.strip() and not head:
            continue
        mark = _SEC_MARK.match(head) if head else None
        if head and mark:
            sec_num = next(g for g in mark.groups()[:3] if g)
            head = mark.group(4).strip() or head
        elif head:
            counter += 1
            sec_num = str(counter)
        else:
            sec_num = "0"
        if head:
            while path_stack and path_stack[-1][0] >= level:
                path_stack.pop()
            path_stack.append((level, head))
        if not text.strip():
            continue
        passages = _split_passages(text, max_words)
        for i, p in enumerate(passages, 1):
            sid = f"§{sec_num}" if len(passages) == 1 else f"§{sec_num}.{i}"
            chunks.append(Chunk(chunk_id=f"{doc_id} {sid}", doc_id=doc_id, section_id=sid, doc_title=title,
                                heading=head or title, heading_path=[h for _, h in path_stack], text=p, path=rel,
                                sentences=split_sentences(p)))
    return chunks


def load_corpus(root: str | Path, max_passage_words: int = 170) -> list[Chunk]:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"corpus path not found: {root}")
    files = [root] if root.is_file() else sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED)
    seen: set[str] = set()
    chunks: list[Chunk] = []
    for path in files:
        if path.name.lower().startswith(IGNORED_PREFIXES):
            continue
        rel = str(path.relative_to(root)) if root.is_dir() else path.name
        try:
            if path.suffix.lower() in (".json", ".jsonl"):
                docs = _parse_json_records(path)
            else:
                suffix = path.suffix.lower()
                fallback = path.stem.replace("-", " ").replace("_", " ").strip().title()
                if suffix in (".md", ".markdown"):
                    docs = [_parse_markdown(path.read_text(encoding="utf-8", errors="ignore"), fallback)]
                elif suffix in (".html", ".htm"):
                    docs = [_parse_html(path.read_text(encoding="utf-8", errors="ignore"), fallback)]
                elif suffix == ".pdf":
                    docs = [_parse_pdf(path, fallback)]
                elif suffix == ".docx":
                    docs = [_parse_docx(path, fallback)]
                else:
                    docs = [_parse_plain(path.read_text(encoding="utf-8", errors="ignore"), fallback)]
        except Exception as exc:  # a broken file must not take the whole corpus down
            print(f"[kairos] skipped {rel}: {exc}")
            continue
        for meta, title, secs in docs:
            did = meta.get("doc_id") or _doc_id_from(path, root if root.is_dir() else path.parent, seen)
            if did in seen:
                did = _doc_id_from(path, root if root.is_dir() else path.parent, seen)
            seen.add(did)
            chunks.extend(_build_chunks(did, title, secs, rel, max_passage_words))
    return chunks


def corpus_fingerprint(root: str | Path) -> str:
    root = Path(root)
    h = hashlib.sha1()
    files = [root] if root.is_file() else sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED)
    for p in files:
        h.update(str(p.relative_to(root) if root.is_dir() else p.name).encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:16]
