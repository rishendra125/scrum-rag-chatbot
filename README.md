# Scrum RAG Chatbot

A retrieval-augmented chatbot that answers Scrum-aligned questions for
Scrum Teams, Scrum Masters, Stakeholders, Project Managers, and Product
Owners, grounded in four official Scrum.org guides:

- The Scrum Guide (2020)
- The Kanban Guide for Scrum Teams (2021)
- The Evidence-Based Management Guide (2024)
- The Agility Guide to Evidence-Based Change (v1.5)

See `docs/architecture.md` for design rationale and `docs/data_sources.md`
for licensing/provenance details (all sources are CC BY-SA 4.0).

## Status

This repo is a **working, end-to-end skeleton** built and tested against
the real source PDFs:

| Stage | Status |
|---|---|
| PDF extraction (`ingestion/extract.py`) | ✅ runs, tested on all 4 guides |
| Chunking (`ingestion/chunker.py`) | ✅ runs, 111 chunks produced |
| Metadata tagging (`ingestion/metadata_tagger.py`) | ✅ runs |
| Embedding (`embeddings/embed.py`) | ✅ runs with local TF-IDF fallback; API backend coded but untested (needs API key) |
| Retrieval (`retrieval/`) | ✅ runs, hybrid dense+BM25+RRF, role/framework boosting |
| Evaluation (`evaluation/retrieval_eval.py`) | ✅ runs, real report in `evaluation/reports/` |
| Generation (`generation/chat_chain.py`) | ✅ runs in dry-run mode; Anthropic backend coded but untested (needs API key) |
| API (`app/api/main.py`) | Coded, not run in this environment (no `fastapi` installed here) |

**Read `docs/architecture.md` section 7 ("Known gaps / next steps") before
treating this as production-ready** -- the honest summary is: the pipeline
architecture and every non-model-dependent stage is real and verified; the
embedding and generation *quality* depends on plugging in real API keys,
which this build environment didn't have.

## Quickstart

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 1. Ingest
python -m ingestion.extract
python -m ingestion.chunker
python -m ingestion.metadata_tagger

# 2. Embed (local/offline by default)
python -m embeddings.embed
# For real semantic embeddings instead:
#   export EMBEDDING_BACKEND=api
#   export OPENAI_API_KEY=sk-...
#   python -m embeddings.embed

# 3. Try retrieval directly
python -m retrieval.retriever "What is the timebox for Sprint Retrospective?"

# 4. Run the full chat chain (dry-run, no LLM call)
python -m generation.chat_chain "How should a Product Owner order the backlog?"
# For real generated answers:
#   export LLM_BACKEND=anthropic
#   export ANTHROPIC_API_KEY=sk-ant-...
#   python -m generation.chat_chain "..."

# 5. Evaluate retrieval quality
python -m evaluation.retrieval_eval

# 6. Serve the API
uvicorn app.api.main:app --reload --port 8000
# curl -X POST localhost:8000/chat -H "Content-Type: application/json" \
#   -d '{"question": "What is a WIP limit?"}'
```

## Repository layout

```
scrum-rag-chatbot/
├── data/               raw PDFs -> interim (per-page) -> processed (chunks)
├── ingestion/          extract.py, chunker.py, metadata_tagger.py, config
├── embeddings/         embed.py (pluggable local/API backend) + index/
├── retrieval/          retriever.py (hybrid), reranker.py, query_router.py
├── generation/          prompt_templates/, chat_chain.py (orchestration)
├── evaluation/          golden_qa_set.jsonl, retrieval_eval.py, reports/
├── app/api/             FastAPI app exposing /chat
├── docs/                architecture.md, data_sources.md
└── tests/                pytest suite for chunking/retrieval/e2e
```

## Current evaluation snapshot

From `evaluation/reports/retrieval_eval_report.json` (local TF-IDF backend,
top_k=5, 15-question golden set):

- **Recall@5:** 0.60
- **MRR:** 0.472
- **Snippet match rate:** 0.933

The gap between snippet match (93%) and section-label recall (60%) is a
known chunking artifact, explained in `docs/architecture.md` section 2 --
the right content is usually retrieved, occasionally under a mis-attributed
section label from a boundary-merge edge case. This is the top item in the
"next steps" list.

## License note

Source guides are CC BY-SA 4.0. If you redistribute this repo's processed
`data/` folder publicly (not just the pipeline code), keep attribution and
license notices intact per `docs/data_sources.md`.
