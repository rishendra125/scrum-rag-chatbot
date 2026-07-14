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

This repo is a **working, end-to-end chatbot** — built, tested against the
real source PDFs, and verified live with real generated answers through a
working chat interface.

| Stage | Status |
|---|---|
| PDF extraction (`ingestion/extract.py`) | ✅ verified on all 4 guides — uses font-size-based heading detection |
| Chunking (`ingestion/chunker.py`) | ✅ verified, 73 chunks (accurate section boundaries; see "Evaluation snapshot" below) |
| Metadata tagging (`ingestion/metadata_tagger.py`) | ✅ verified |
| Embedding (`embeddings/embed.py`) | ✅ verified with local TF-IDF fallback; API backend coded, not yet tested with a real key |
| Retrieval (`retrieval/`) | ✅ verified — hybrid dense+BM25+RRF, role/framework boosting confirmed working |
| Evaluation (`evaluation/retrieval_eval.py`) | ✅ verified, real report in `evaluation/reports/` |
| Generation (`generation/chat_chain.py`) | ✅ **verified with live Anthropic API calls** — real, cited answers confirmed working |
| API (`app/api/main.py`) | ✅ **verified running locally** (`uvicorn`, `/health` checked, CORS enabled for local frontend use) |
| Frontend (`app/ui/index.html`) | ✅ **verified working** — role selector, cited answer cards with per-guide color tabs, tested live against the running API |

Read `docs/architecture.md` section 7 ("Known gaps / next steps") for what's
still genuinely open — real semantic embeddings (swapping TF-IDF for an
API embedding model) is the next highest-leverage improvement, along with
expanding the evaluation question set.

## Try It Yourself

This project is free and open to run — but it is **not a live, hosted
chatbot**. There is no shared or public API key. To try it, you run it
on your own machine using **your own Anthropic API key**, which means:

- You control your own cost (a few dollars of credit covers extensive
  testing — see "Current evaluation snapshot" for typical usage).
- The maintainer's API key is never shared, embedded, or accessible
  through this repository in any way.

### Steps to run it with your own key

1. Clone this repo and follow the **Quickstart** section below (steps 1-6)
   to get the pipeline running locally in offline/dry-run mode first —
   no API key needed for this part, and it costs nothing.
2. Create a free account at [console.anthropic.com](https://console.anthropic.com)
   and generate your own API key under **API Keys**.
3. Add a small amount of credit under **Plans & Billing** (a few dollars
   is enough for extensive testing).
4. Copy `.env.example` to a new file named `.env` in the project root,
   and fill in your own key:
   ```
   LLM_BACKEND=anthropic
   ANTHROPIC_API_KEY=sk-ant-your-own-key-here
   ```
5. **Never commit your `.env` file.** It's already excluded via
   `.gitignore` — double-check it doesn't appear in `git status` before
   pushing any changes of your own.
6. Run the chat chain or start the API server (see Quickstart steps 4-7)
   — you're now using your own key, at your own cost, under your own
   control.

If you don't want to set up an API key at all, the pipeline still runs
fully in **dry-run mode** (no cost, no key required) — you'll see exactly
which guide passages would be retrieved and how the prompt gets
assembled, just without a generated answer at the end.

## Quickstart

```bash
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
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

# 4. Run the full chat chain from the command line
#    (dry-run by default, no LLM call, no cost):
python -m generation.chat_chain "How should a Product Owner order the backlog?"
# For real generated answers, set these first:
#   export LLM_BACKEND=anthropic
#   export ANTHROPIC_API_KEY=sk-ant-...
#   python -m generation.chat_chain "..."

# 5. Evaluate retrieval quality
python -m evaluation.retrieval_eval

# 6. Serve the API (needed for the chat window in step 7)
uvicorn app.api.main:app --reload --port 8000
# curl -X POST localhost:8000/chat -H "Content-Type: application/json" \
#   -d '{"question": "What is a WIP limit?"}'

# 7. Use the actual chat window
# With the API server from step 6 still running, open app/ui/index.html
# directly in your browser (double-click it, or open it via File > Open).
# It talks to the API at http://localhost:8000/chat -- pick a role on the
# left, ask a question, and you'll get a real cited answer rendered as a
# card, with a colored tab showing which guide(s) it drew from.
```

## Repository layout

```
scrum-rag-chatbot/
├── data/               raw PDFs -> interim (per-page) -> processed (chunks)
├── ingestion/          extract.py, chunker.py, metadata_tagger.py, config
├── embeddings/         embed.py (pluggable local/API backend) + index/
├── retrieval/          retriever.py (hybrid), reranker.py, query_router.py
├── generation/         prompt_templates/, chat_chain.py (orchestration)
├── evaluation/         golden_qa_set.jsonl, retrieval_eval.py, reports/
├── app/
│   ├── api/            FastAPI app exposing /chat (CORS-enabled for local use)
│   └── ui/              index.html -- the working chat frontend
├── docs/                architecture.md, data_sources.md
└── tests/                pytest suite for chunking/retrieval/e2e
```

## Current evaluation snapshot

From `evaluation/reports/retrieval_eval_report.json` (local TF-IDF backend,
top_k=5, 15-question golden set), **after fixing the section-heading
detection bug**:

- **Recall@5:** 0.867 (up from an earlier 0.60)
- **MRR:** 0.789 (up from an earlier 0.472)
- **Snippet match rate:** 1.0 (up from an earlier 0.933)

### What was fixed

Earlier versions of this pipeline had a real, now-resolved bug: section
headings were detected with plain text search (`text.find(heading)`),
which could match a heading's name mentioned in passing prose (e.g.
"Sprint Backlog" bolded inline within the Sprint Planning section) instead
of the actual section heading. This caused some citations to show the
correct retrieved *content* but an incorrect *section label*.

**Fix:** `ingestion/extract.py` now uses pdfplumber's character-level font
size metadata to detect real headings -- a line is only treated as a
section heading if it's rendered at a distinctly larger size (>=1.0pt above
body text) than the surrounding prose. This reliably separates true
headings (13-24pt across all 4 guides) from inline bolded terms that stay
at body size (~11pt). `ingestion/chunker.py` was updated to resolve each
heading to its correct occurrence on the page using this signal, instead
of blindly taking the first text match.

The improvement above was verified by re-running the full pipeline and
evaluation before/after the fix, and by re-testing live questions through
`generation/chat_chain.py` and the `app/ui/index.html` frontend to confirm
citations now show the correct section names.

## License note

Source guides are CC BY-SA 4.0. If you redistribute this repo's processed
`data/` folder publicly (not just the pipeline code), keep attribution and
license notices intact per `docs/data_sources.md`.
