"""
retrieval_eval.py
-----------------
Runs the golden Q&A set against the live HybridRetriever and reports:
  - Recall@k        : was the expected document/section found in the top k?
  - MRR             : mean reciprocal rank of the first correct hit
  - Snippet check   : does any retrieved chunk contain the expected
                      answer snippet (a cheap proxy for "the right content,
                      not just the right section label")

This is a regression suite: run it after any change to chunking,
tagging, or the retriever, and diff the report against
evaluation/reports/ to catch quality regressions before they reach users.
"""

import json
from pathlib import Path

from retrieval.retriever import HybridRetriever

BASE_DIR = Path(__file__).resolve().parent.parent
GOLDEN_PATH = BASE_DIR / "evaluation" / "golden_qa_set.jsonl"
REPORT_PATH = BASE_DIR / "evaluation" / "reports" / "retrieval_eval_report.json"


def load_golden_set() -> list:
    with open(GOLDEN_PATH, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def evaluate(top_k: int = 5) -> dict:
    retriever = HybridRetriever()
    golden = load_golden_set()

    per_question = []
    hits_at_k = 0
    reciprocal_ranks = []
    snippet_hits = 0

    for item in golden:
        results = retriever.retrieve(item["question"], top_k=top_k)

        rank = None
        snippet_found = False
        for i, chunk in enumerate(results, start=1):
            doc_match = chunk["doc_id"] == item["expected_doc"]
            section_match = chunk["section"] == item["expected_section"]
            if doc_match and section_match and rank is None:
                rank = i
            if item["expected_answer_snippet"].lower() in chunk["text"].lower():
                snippet_found = True

        found = rank is not None
        hits_at_k += int(found)
        reciprocal_ranks.append(1.0 / rank if found else 0.0)
        snippet_hits += int(snippet_found)

        per_question.append({
            "id": item["id"],
            "question": item["question"],
            "expected": f"{item['expected_doc']} / {item['expected_section']}",
            "found_at_rank": rank,
            "snippet_found": snippet_found,
            "top_result": f"{results[0]['source_doc']} > {results[0]['section']}" if results else None,
        })

    n = len(golden)
    report = {
        "top_k": top_k,
        "n_questions": n,
        "recall_at_k": round(hits_at_k / n, 3),
        "mrr": round(sum(reciprocal_ranks) / n, 3),
        "snippet_match_rate": round(snippet_hits / n, 3),
        "per_question": per_question,
    }
    return report


def main():
    report = evaluate(top_k=5)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Recall@{report['top_k']}: {report['recall_at_k']}")
    print(f"MRR: {report['mrr']}")
    print(f"Snippet match rate: {report['snippet_match_rate']}")
    print(f"\nFull report -> {REPORT_PATH}")

    failures = [q for q in report["per_question"] if q["found_at_rank"] is None]
    if failures:
        print(f"\n{len(failures)} question(s) missed the expected section entirely:")
        for f_ in failures:
            print(f"  - {f_['id']}: {f_['question']}  (got: {f_['top_result']})")


if __name__ == "__main__":
    main()
