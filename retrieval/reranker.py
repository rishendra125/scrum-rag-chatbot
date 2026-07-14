"""
reranker.py
-----------
Optional second-stage re-ranking, applied to the top ~20 candidates from
HybridRetriever before truncating to the final top_k passed to the LLM.

This file ships with a lightweight *lexical-overlap* reranker (no model
download required) so the pipeline runs fully offline. It's a reasonable
placeholder, not a substitute for a real cross-encoder in production.

To upgrade to a real cross-encoder once you have network access:

    pip install sentence-transformers
    from sentence_transformers import CrossEncoder
    model = CrossEncoder("BAAI/bge-reranker-base")
    scores = model.predict([(query, c["text"]) for c in candidates])

...then sort candidates by `scores` descending. Everything else in the
pipeline (retriever output shape, chat_chain.py) stays the same -- only
this module's internals change.
"""

import re


def _tokenize(text: str) -> set:
    return set(re.findall(r"[a-z0-9']+", text.lower()))


def lexical_overlap_score(query: str, chunk_text: str) -> float:
    q_tokens = _tokenize(query)
    c_tokens = _tokenize(chunk_text)
    if not q_tokens:
        return 0.0
    overlap = q_tokens & c_tokens
    return len(overlap) / len(q_tokens)


def rerank(query: str, candidates: list, top_k: int = 5) -> list:
    """candidates: list of chunk dicts (as returned by HybridRetriever).
    Returns the top_k candidates re-sorted by a secondary relevance score,
    stored in `_rerank_score`. Ties broken by the original fused score."""
    scored = []
    for c in candidates:
        overlap = lexical_overlap_score(query, c["text"])
        scored.append((overlap, c.get("_score", 0.0), c))

    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)

    results = []
    for overlap, _, c in scored[:top_k]:
        c = dict(c)
        c["_rerank_score"] = round(overlap, 4)
        results.append(c)
    return results


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "..")
    from retrieval.retriever import HybridRetriever

    retriever = HybridRetriever()
    query = " ".join(sys.argv[1:]) or "What is a WIP limit?"
    candidates = retriever.retrieve(query, top_k=20)
    top = rerank(query, candidates, top_k=5)
    for c in top:
        print(f"[fused={c['_score']} rerank={c['_rerank_score']}] "
              f"{c['source_doc']} > {c['section']}")
