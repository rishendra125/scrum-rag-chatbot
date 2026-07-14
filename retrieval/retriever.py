"""
retriever.py
------------
Hybrid retrieval over the embedded Scrum-guide chunks:

  1. Dense search   -- cosine similarity over the vectors built by
                        embeddings/embed.py (TF-IDF locally, or real
                        embeddings if EMBEDDING_BACKEND=api).
  2. Sparse search  -- a small dependency-free BM25 implementation, so
                        exact Scrum terminology ("Sprint Backlog",
                        "Definition of Done") ranks well even when dense
                        similarity is fuzzy.
  3. Fusion         -- Reciprocal Rank Fusion (RRF) combines both rankings
                        without needing to normalize incomparable score
                        scales.
  4. Metadata boost -- chunks matching the requested role/framework get a
                        small rank boost (soft filter, not a hard filter --
                        see docs/architecture.md for why hard filtering
                        risks losing a relevant cross-framework chunk).

No re-ranker (cross-encoder) is wired in yet since it needs a real model
download; see reranker.py for a stub + notes on adding one.
"""

import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
INDEX_DIR = BASE_DIR / "embeddings" / "index"


# ---------------------------------------------------------------------------
# BM25 (Okapi), implemented from scratch to avoid an extra dependency.
# ---------------------------------------------------------------------------
class BM25:
    def __init__(self, corpus_tokens: list, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus = corpus_tokens
        self.doc_lens = [len(doc) for doc in corpus_tokens]
        self.avgdl = sum(self.doc_lens) / len(self.doc_lens) if corpus_tokens else 0
        self.df = defaultdict(int)
        for doc in corpus_tokens:
            for term in set(doc):
                self.df[term] += 1
        self.n_docs = len(corpus_tokens)
        self.idf = {
            term: math.log(1 + (self.n_docs - freq + 0.5) / (freq + 0.5))
            for term, freq in self.df.items()
        }

    def score(self, query_tokens: list) -> np.ndarray:
        scores = np.zeros(self.n_docs, dtype="float32")
        for i, doc in enumerate(self.corpus):
            doc_len = self.doc_lens[i]
            term_freqs = defaultdict(int)
            for term in doc:
                term_freqs[term] += 1
            s = 0.0
            for term in query_tokens:
                if term not in term_freqs:
                    continue
                idf = self.idf.get(term, 0.0)
                tf = term_freqs[term]
                denom = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
                s += idf * (tf * (self.k1 + 1)) / denom
            scores[i] = s
        return scores


def tokenize(text: str) -> list:
    return re.findall(r"[a-z0-9']+", text.lower())


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------
class HybridRetriever:
    def __init__(self):
        self.vectors = np.load(INDEX_DIR / "vectors.npy")
        # normalize for cosine similarity via dot product
        norms = np.linalg.norm(self.vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1e-9
        self.unit_vectors = self.vectors / norms

        with open(INDEX_DIR / "chunk_meta.jsonl", "r", encoding="utf-8") as f:
            self.chunks = [json.loads(line) for line in f]

        self.bm25 = BM25([tokenize(c["embed_text"]) for c in self.chunks])

        backend = os.environ.get("EMBEDDING_BACKEND", "local")
        if backend == "local":
            from embeddings.embed import LocalTfidfEmbedder
            self.embedder = LocalTfidfEmbedder.load(INDEX_DIR / "embedder_local.pkl")
        else:
            from embeddings.embed import APIEmbedder
            self.embedder = APIEmbedder.load(INDEX_DIR / "embedder_api.json")

    def _dense_rank(self, query: str) -> list:
        qvec = self.embedder.transform([query])[0]
        qnorm = np.linalg.norm(qvec)
        if qnorm == 0:
            qnorm = 1e-9
        qunit = qvec / qnorm
        sims = self.unit_vectors @ qunit
        order = np.argsort(-sims)
        return list(order)

    def _sparse_rank(self, query: str) -> list:
        scores = self.bm25.score(tokenize(query))
        order = np.argsort(-scores)
        return list(order)

    @staticmethod
    def _rrf_fuse(rankings: list, k: int = 60) -> dict:
        """Reciprocal Rank Fusion: combines multiple rank lists (each a list
        of doc indices, best first) into a single score per doc index."""
        fused = defaultdict(float)
        for ranking in rankings:
            for rank, doc_idx in enumerate(ranking):
                fused[doc_idx] += 1.0 / (k + rank + 1)
        return fused

    def retrieve(self, query: str, top_k: int = 6, role: str = None,
                 framework: str = None, boost: float = 0.15) -> list:
        """Returns top_k chunk dicts (with a `_score` field attached),
        best first. `role`/`framework` apply a soft metadata boost rather
        than a hard filter, so a highly relevant chunk from an
        unrequested framework can still surface."""
        dense_rank = self._dense_rank(query)
        sparse_rank = self._sparse_rank(query)
        fused = self._rrf_fuse([dense_rank, sparse_rank])

        if role or framework:
            for idx in list(fused.keys()):
                chunk = self.chunks[idx]
                bonus = 0.0
                if role and role in chunk.get("roles_relevant", []):
                    bonus += boost
                if framework and framework.lower() == chunk.get("framework", "").lower():
                    bonus += boost
                fused[idx] *= (1.0 + bonus)

        ranked = sorted(fused.items(), key=lambda x: -x[1])[:top_k]
        results = []
        for idx, score in ranked:
            chunk = dict(self.chunks[idx])
            chunk["_score"] = round(float(score), 5)
            results.append(chunk)
        return results


def format_context(chunks: list) -> str:
    """Assemble retrieved chunks into an LLM-ready context block, grouped
    with citation-friendly headers."""
    blocks = []
    for c in chunks:
        pages = ", ".join(str(p) for p in c["pages"])
        header = f"[{c['source_doc']} ({c['version']}) - {c['section']} - p.{pages}]"
        blocks.append(f"{header}\n{c['text']}")
    return "\n\n---\n\n".join(blocks)


if __name__ == "__main__":
    import sys
    retriever = HybridRetriever()
    query = " ".join(sys.argv[1:]) or "What is the timebox for Sprint Retrospective?"
    results = retriever.retrieve(query, top_k=5)
    print(f"Query: {query}\n")
    for r in results:
        print(f"[{r['_score']}] {r['source_doc']} > {r['section']} (p.{r['pages']})")
        print(f"    {r['text'][:160]}...\n")
