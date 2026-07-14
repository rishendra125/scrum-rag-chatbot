# Architecture & Design Decisions

This document explains *why* the pipeline is built the way it is, so future
changes are made with the same reasoning rather than by guesswork.

## 1. Pipeline stages

```
raw PDFs --extract.py--> interim JSONL (per page, headings detected)
        --chunker.py--> processed chunks (section-aware, ~150-400 words)
        --metadata_tagger.py--> tagged chunks (roles, concepts, artifact_type)
        --embed.py--> vector index (TF-IDF locally, or real embeddings via API)

query --query_router.py--> {role, framework, is_scenario}
      --retriever.py (dense + BM25, RRF-fused, metadata-boosted)--> top ~20
      --reranker.py--> top 5
      --chat_chain.py (prompt assembly)--> LLM --> cited answer
```

## 2. Why structure-aware chunking, not fixed-size windows

These guides are short (9-18 pages) and extremely dense with self-contained
definitions (a role, an event, an artifact). A fixed 500-token sliding
window with no regard for section boundaries would regularly straddle two
unrelated concepts (e.g., end of "Sprint Review" bleeding into start of
"Sprint Retrospective"), diluting the embedding for both. Splitting first on
detected headings, then chunking within a section, avoids this at the cost
of slightly more ingestion code.

**Fixed (previously a known limitation):** section headings used to be
detected with plain-text search (`text.find(heading)`), which could match
a heading's name mentioned in passing prose or as a bolded inline term
(e.g. "Sprint Backlog" bolded within the Sprint Planning section) instead
of the actual heading, causing some chunks to carry the right *content*
under the wrong section *label*. This showed up in the evaluation report
as a gap between snippet-match rate (93%) and section-match recall (60%).

`ingestion/extract.py` now detects headings using pdfplumber's
character-level font-size metadata instead of plain-text search: a line is
only accepted as a real heading if it's rendered at a distinctly larger
size (>=1.0pt above body text) than surrounding prose, which reliably
separates true headings (13-24pt across all 4 guides) from same-size
bolded inline terms. `ingestion/chunker.py` resolves each heading to its
correct occurrence on the page using this signal, rather than blindly
taking the first text match. After the fix, snippet-match rate reached
1.0 and section-match recall rose to 0.867 (from 0.60) -- see the README's
"Current evaluation snapshot" for the full before/after numbers.

## 3. Why hybrid retrieval (dense + BM25) instead of dense-only

Scrum terminology is precise and repeated verbatim across guides
("Definition of Done," "Sprint Backlog," "WIP limit"). Pure dense retrieval
over a weak local embedding space (TF-IDF, see below) under-weights exact
term matches that BM25 captures well. Reciprocal Rank Fusion combines both
without needing to normalize incomparable score scales (cosine similarity
vs. BM25 score), which is simpler and more robust than trying to tune a
linear combination weight.

## 4. Why the local embedder is TF-IDF, not a real model

This repository was built and tested in an offline sandbox with no network
access and no local model weights available. Rather than leave the
embedding layer as pseudocode, `embeddings/embed.py` ships a real, runnable
TF-IDF backend so the *entire* pipeline -- ingestion through evaluation --
can be executed and its output inspected today, with `evaluation/reports/`
holding real (not simulated) numbers.

**This is explicitly a placeholder for retrieval quality, not for
architecture.** The `Embedder` interface, `APIEmbedder` class, and
`EMBEDDING_BACKEND` env var are already in place -- switching to real
semantic embeddings is a one-line environment change
(`EMBEDDING_BACKEND=api`) plus an API key, not a rewrite. Expect recall to
improve substantially once real embeddings replace TF-IDF, since TF-IDF
cannot match paraphrased questions (e.g., "how do I keep the team from
overcommitting" won't lexically match "Sprint Backlog" or "capacity") the
way a semantic model will.

## 5. Why soft metadata boosting, not hard filtering

Early designs considered hard-filtering retrieval by `roles_relevant` or
`framework` once the query router infers them. This was rejected: the
router is a cheap regex classifier and will sometimes guess wrong or guess
`None`. A hard filter on a wrong guess silently removes the correct answer
from consideration with no recovery path. A soft multiplicative boost
(`retrieval/retriever.py: boost=0.15`) nudges ranking in the likely-correct
direction while still allowing a strongly-matching chunk from an
unexpected role/framework to surface.

## 6. Why RRF instead of a learned fusion weight

Reciprocal Rank Fusion needs no calibration and no training data -- it only
needs rank positions, which are comparable across totally different scoring
systems (cosine similarity, BM25). Given the small corpus (currently ~111
chunks total), there isn't enough data to responsibly fit a learned fusion
weight without overfitting to the golden Q&A set.

## 7. Known gaps / next steps (in priority order)

1. Swap `LocalTfidfEmbedder` for `APIEmbedder` (or a locally-hosted
   sentence-transformers model) -- highest-leverage quality improvement.
2. Swap the lexical-overlap `reranker.py` stand-in for a real cross-encoder
   (`bge-reranker-base` or similar) once model downloads are available.
3. Expand `golden_qa_set.jsonl` well beyond 15 questions before treating
   `retrieval_eval.py` numbers as a reliable regression gate.
4. Add a real cross-encoder-based `generation_eval.py` (faithfulness /
   groundedness scoring) -- currently only retrieval is evaluated
   quantitatively; generation quality still needs a human or LLM-judge pass.

~~Fix section-boundary detection using PDF font-size/style metadata~~ --
**done**: `ingestion/extract.py`/`ingestion/chunker.py` now use font-size
metadata for heading detection (see Section 2 above).
