"""
Basic regression tests. Run with: pytest tests/

These assume the pipeline has already been run once (data/processed/ and
embeddings/index/ exist) -- they test the *retrieval and chain* behavior,
not re-run ingestion from scratch. Add an ingestion-from-scratch fixture
if you want fully hermetic CI runs.
"""

import json
from pathlib import Path

import pytest

BASE_DIR = Path(__file__).resolve().parent.parent


def test_processed_chunks_exist_and_are_nonempty():
    path = BASE_DIR / "data" / "processed" / "all_chunks_tagged.jsonl"
    assert path.exists(), "Run ingestion (extract -> chunker -> metadata_tagger) first."
    with open(path) as f:
        chunks = [json.loads(line) for line in f]
    assert len(chunks) > 0
    for c in chunks[:5]:
        assert c["text"].strip()
        assert c["source_doc"]
        assert isinstance(c["roles_relevant"], list)


def test_no_chunk_is_pure_toc_noise():
    """Regression guard for the dot-leader ToC bug fixed during development
    (see docs/architecture.md) -- no chunk should be almost entirely
    '....' leader characters or under ~10 real words."""
    path = BASE_DIR / "data" / "processed" / "all_chunks_tagged.jsonl"
    with open(path) as f:
        chunks = [json.loads(line) for line in f]
    for c in chunks:
        assert len(c["text"].split()) >= 10, f"Suspiciously short chunk: {c['chunk_id']}"
        assert c["text"].count("....") < 3, f"Possible ToC leftover: {c['chunk_id']}"


def test_retriever_returns_results():
    from retrieval.retriever import HybridRetriever
    retriever = HybridRetriever()
    results = retriever.retrieve("What is the timebox for Sprint Planning?", top_k=5)
    assert len(results) == 5
    assert all("source_doc" in r for r in results)


def test_retriever_role_boost_changes_ranking_or_scores():
    from retrieval.retriever import HybridRetriever
    retriever = HybridRetriever()
    query = "How should backlog items be ordered?"
    unboosted = retriever.retrieve(query, top_k=5)
    boosted = retriever.retrieve(query, top_k=5, role="Product Owner")
    # boosting should at least change scores even if top result is the same
    assert unboosted[0]["_score"] != boosted[0]["_score"] or \
           [r["chunk_id"] for r in unboosted] != [r["chunk_id"] for r in boosted]


def test_query_router_infers_role_and_framework():
    from retrieval.query_router import route
    result = route("How should a Product Owner order the backlog?")
    assert result["role"] == "Product Owner"

    result2 = route("My team keeps breaking its WIP limit")
    assert result2["framework"] == "Kanban"
    assert result2["is_scenario"] is True


def test_chat_chain_end_to_end_dry_run():
    from generation.chat_chain import ScrumRAGChain
    chain = ScrumRAGChain()
    result = chain.answer("What is the timebox for Sprint Retrospective?")
    assert result["sources"]
    assert "DryRunLLM" in result["answer"] or len(result["answer"]) > 0


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
