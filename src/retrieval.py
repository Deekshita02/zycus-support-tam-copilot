"""
Lightweight retrieval over the markdown knowledge base.

Design choice: this uses a dependency-free BM25 implementation instead of
embeddings. Rationale (see DESIGN_NOTE for more):
  - The KB is ~9 small files -- an embedding index is overkill and adds an
    external API call (cost + latency + another PII surface) for no real
    recall gain over lexical search on this corpus, which is full of exact
    product names, module names, and error codes (BM25's sweet spot).
  - Zero extra runtime dependency, no network call, fully deterministic.

Chunking strategy follows DATA_SCHEMA.md's recommendation: split on `---`
horizontal rules, and keep the nearest heading hierarchy as chunk metadata.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from functools import lru_cache

from src.config import KB_DIR

_TOKEN_RE = re.compile(r"[a-z0-9_]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


@dataclass
class Chunk:
    source: str          # relative file path, e.g. "products/databridge-pro.md"
    heading_path: str    # e.g. "DataBridge Pro > Common Support Scenarios > Connector authentication failure"
    text: str
    tokens: list[str] = field(default_factory=list, repr=False)

    def __post_init__(self):
        if not self.tokens:
            self.tokens = _tokenize(self.text)


def _split_into_chunks(md_text: str, source: str) -> list[Chunk]:
    """Split on --- rules; track heading hierarchy (#, ##, ###) as we go so
    every chunk knows which section it came from."""
    chunks: list[Chunk] = []
    heading_stack: list[tuple[int, str]] = []  # (level, text)

    for block in md_text.split("\n---\n"):
        block = block.strip()
        if not block:
            continue

        # Update heading stack using headings found at the START of this
        # block (a block may open with one or more headings).
        lines = block.split("\n")
        for line in lines:
            m = re.match(r"^(#{1,6})\s+(.*)$", line.strip())
            if m:
                level = len(m.group(1))
                text = m.group(2).strip()
                heading_stack = [h for h in heading_stack if h[0] < level]
                heading_stack.append((level, text))

        heading_path = " > ".join(h[1] for h in heading_stack)
        if len(block) > 20:  # skip degenerate near-empty chunks
            chunks.append(Chunk(source=source, heading_path=heading_path, text=block))

    return chunks


@lru_cache(maxsize=1)
def load_kb_chunks() -> tuple[Chunk, ...]:
    chunks: list[Chunk] = []
    for path in sorted(KB_DIR.rglob("*.md")):
        rel = str(path.relative_to(KB_DIR))
        md_text = path.read_text(encoding="utf-8")
        chunks.extend(_split_into_chunks(md_text, rel))
    return tuple(chunks)


class BM25Index:
    """Minimal, dependency-free BM25 (Okapi) implementation."""

    def __init__(self, chunks: tuple[Chunk, ...], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.n = len(chunks)
        self.doc_lens = [len(c.tokens) for c in chunks]
        self.avgdl = sum(self.doc_lens) / self.n if self.n else 0.0

        df: dict[str, int] = {}
        for c in chunks:
            for term in set(c.tokens):
                df[term] = df.get(term, 0) + 1
        self.idf: dict[str, float] = {
            term: math.log(1 + (self.n - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }

        self.term_freqs: list[dict[str, int]] = []
        for c in chunks:
            tf: dict[str, int] = {}
            for term in c.tokens:
                tf[term] = tf.get(term, 0) + 1
            self.term_freqs.append(tf)

    def search(self, query: str, top_k: int = 3) -> list[tuple[Chunk, float]]:
        query_terms = _tokenize(query)
        scores = [0.0] * self.n
        for i in range(self.n):
            dl = self.doc_lens[i] or 1
            tf = self.term_freqs[i]
            score = 0.0
            for term in query_terms:
                if term not in tf:
                    continue
                idf = self.idf.get(term, 0.0)
                freq = tf[term]
                denom = freq + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                score += idf * (freq * (self.k1 + 1)) / denom
            scores[i] = score

        ranked = sorted(zip(self.chunks, scores), key=lambda x: x[1], reverse=True)
        return [(c, s) for c, s in ranked[:top_k] if s > 0]


@lru_cache(maxsize=1)
def get_index() -> BM25Index:
    return BM25Index(load_kb_chunks())


def search_kb(query: str, top_k: int = 3) -> list[dict]:
    """Public helper: returns a list of {source, heading_path, text, score}."""
    results = get_index().search(query, top_k=top_k)
    return [
        {"source": c.source, "heading_path": c.heading_path, "text": c.text, "score": round(score, 3)}
        for c, score in results
    ]
