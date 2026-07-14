"""
extract.py
----------
Extracts clean, page-mapped text from the raw Scrum guide PDFs and writes
one JSONL record per page to data/interim/<doc_id>.jsonl.

Design notes:
- Uses pdfplumber (pure-Python, no external binary deps) so the pipeline
  runs anywhere without poppler/tesseract installed.
- We do NOT try to be clever about multi-column layouts here — these guides
  are single-column, so a straightforward text extraction is reliable.
- Repeated footer boilerplate (license text that appears on every page) is
  stripped via a regex so it doesn't pollute embeddings later.
- Heading detection matches known section titles per document (declared in
  SECTION_HINTS below), but a plain-text substring search isn't enough: a
  heading's name can also appear earlier on the same page as running prose
  or a bolded inline term (e.g. "Sprint Backlog" as a bolded term inside
  the "Sprint Planning" section, well before "Sprint Backlog" is its own
  heading later on). To tell a true heading from a same-text mention, we
  use pdfplumber's character-level font-size metadata: across all four
  guides, real section headings sit at a distinctly larger point size than
  body text, while inline bolded terms stay at body size. Only a line whose
  average font size clears body size by HEADING_SIZE_MARGIN is accepted as
  a heading match. See docs/architecture.md section 2 for the bug this
  fixes.
"""

import json
import re
from collections import Counter
from pathlib import Path

import pdfplumber
import yaml

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
INTERIM_DIR = BASE_DIR / "data" / "interim"
CONFIG_PATH = Path(__file__).resolve().parent / "pipeline_config.yaml"

# Boilerplate patterns repeated on every page of these guides — stripped
# so they don't get embedded as if they were content.
FOOTER_PATTERNS = [
    r"©\s*20\d\d\s*Scrum\.Org.*?Commons\.",
    r"Page \d+ of \d+\s*\|.*?Commons\.",
    r"The Kanban Guide for Scrum Teams \| Page \d+",
    r"© 2020 Ken Schwaber and Jeff Sutherland.*?Commons\.",
]

# Known top-level section headings per document, used to tag each page
# with the section it most likely belongs to (refined further in chunker.py
# using in-text heading matches, not just page-level guesses).
SECTION_HINTS = {
    "scrum_guide_2020": [
        "Purpose of the Scrum Guide", "Scrum Definition", "Scrum Theory",
        "Scrum Values", "Scrum Team", "Developers", "Product Owner",
        "Scrum Master", "Scrum Events", "The Sprint", "Sprint Planning",
        "Daily Scrum", "Sprint Review", "Sprint Retrospective",
        "Scrum Artifacts", "Product Backlog", "Sprint Backlog",
        "Increment", "End Note",
    ],
    "kanban_guide_2021": [
        "Purpose", "Relation to the Scrum Guide", "Definition of Kanban",
        "Kanban with Scrum Theory", "Flow and Empiricism",
        "The Basic Metrics of Flow", "Little\u2019s Law", "Kanban Practices",
        "Definition of Workflow", "Visualization of the Workflow",
        "Limiting Work in Progress", "Active Management of Work Items",
        "Inspect and Adapt the Definition of Workflow", "Flow-Based Events",
        "The Sprint", "Sprint Planning", "Daily Scrum", "Sprint Review",
        "Sprint Retrospective", "Increment", "Endnote",
        "History and Acknowledgments",
    ],
    "ebm_guide_2024": [
        "Purpose of the EBM Guide", "Definition of Evidence-Based Management",
        "EBM Helps Organizations Achieve Their Goals in a Complex World",
        "Setting Goals", "Understanding What is Valuable",
        "Making Progress Toward Goals in a Series of Small Steps",
        "Hypotheses, Experiments, Features, and Requirements",
        "EBM Uses Key Value Areas to Examine Improvement Opportunities",
        "Current Value", "Unrealized Value", "Ability to Innovate",
        "Time-to-Market", "Inspecting and Adapting Based on Experiment Results",
        "End Note", "Appendix: Example Key Value Measures",
    ],
    "agility_guide_v1_5": [
        "Purpose of the Agility Guide", "Evidence-Based Change Overview",
        "Framework", "Domains", "Evidence-Based Change Theory",
        "Transparency", "Inspection", "Adaptation", "Agility Team",
        "Enterprise Product Owner", "Enterprise Change Team",
        "Enterprise Scrum Master", "Evidence-Based Change Events",
        "Sprint", "Sprint Planning", "Weekly Scrum", "Evaluation",
        "Sprint Review", "Sprint Retrospective",
        "Evidence-Based Change Artifacts", "Practice Backlog",
        "Sprint Backlog", "Evaluation Backlog", "Increment of Change",
        "Evidence-Based Metrics", "Agility Index", "Agility Acceleration",
        "Conclusion",
    ],
}


TOC_LEADER_LINE = re.compile(r"\.{4,}\s*\d*\s*$")


def clean_page_text(text: str) -> str:
    if not text:
        return ""
    for pattern in FOOTER_PATTERNS:
        text = re.sub(pattern, "", text, flags=re.DOTALL)

    # Drop Table-of-Contents "dot leader" lines (e.g. "Sprint Planning ..... 8")
    # -- these match heading names but carry no content, and would otherwise
    # be mistaken by the chunker for real section text.
    lines = [ln for ln in text.split("\n") if not TOC_LEADER_LINE.search(ln)]
    text = "\n".join(lines)

    # collapse excess whitespace left behind by footer/TOC removal
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def is_toc_page(text: str) -> bool:
    """Heuristic: a page dominated by short heading-like lines with no
    real prose is almost certainly a Table of Contents page."""
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    if not lines:
        return False
    avg_len = sum(len(ln) for ln in lines) / len(lines)
    return avg_len < 40 and len(lines) > 5


# A line must clear body-text font size by at least this many points to be
# accepted as a real heading, rather than a same-size bolded inline term
# (e.g. "Sprint Backlog" mentioned in bold within the "Sprint Planning"
# section). Empirically, real headings across all 4 guides sit 1.0-3pt+
# above body size; bolded inline terms stay at body size.
HEADING_SIZE_MARGIN = 1.0


def get_page_lines(page) -> list:
    """Return this page's visual text lines with each line's average
    character font size, using pdfplumber's char-level metadata rather than
    a layout heuristic. Order matches reading order top-to-bottom."""
    lines = page.extract_text_lines(layout=False, return_chars=True)
    result = []
    for ln in lines:
        chars = ln.get("chars", [])
        sizes = [c["size"] for c in chars if "size" in c]
        avg_size = sum(sizes) / len(sizes) if sizes else 0.0
        result.append({"text": ln["text"], "size": avg_size})
    return result


def compute_body_size(pages_lines: list) -> float:
    """The single most common line font size across the document. Body
    paragraph text vastly outnumbers heading lines, so the mode is a
    reliable proxy for "normal" text size even without knowing the PDF
    exporter's exact style conventions."""
    sizes = Counter(
        round(ln["size"], 1)
        for lines in pages_lines
        for ln in lines
        if ln["text"].strip()
    )
    if not sizes:
        return 0.0
    return sizes.most_common(1)[0][0]


def detect_headings_on_page(lines: list, body_size: float, doc_id: str) -> list:
    """Find where known section headings truly start on this page, using
    font size to reject cases where a heading's name merely appears as
    running prose or a bolded inline term at body size. Returns a list of
    (char_offset, heading) tuples; char_offset is the position within the
    "\\n"-joined raw page text (see extract_document, which builds `text`
    from these same lines so offsets line up)."""
    hints = SECTION_HINTS.get(doc_id, [])
    found = []
    offset = 0
    for line in lines:
        stripped = line["text"].strip()
        if line["size"] >= body_size + HEADING_SIZE_MARGIN:
            for heading in hints:
                if stripped == heading or stripped.startswith(heading):
                    found.append((offset, heading))
                    break
        offset += len(line["text"]) + 1  # +1 for the "\n" joiner
    return found


def extract_document(doc_cfg: dict) -> Path:
    doc_id = doc_cfg["doc_id"]
    pdf_path = RAW_DIR / doc_cfg["file"]
    out_path = INTERIM_DIR / f"{doc_id}.jsonl"

    records = []
    with pdfplumber.open(pdf_path) as pdf:
        pages = list(pdf.pages)
        pages_lines = [get_page_lines(p) for p in pages]
        body_size = compute_body_size(pages_lines)

        for page_number, lines in enumerate(pages_lines, start=1):
            raw_text = "\n".join(ln["text"] for ln in lines)
            text = clean_page_text(raw_text)
            if not text or is_toc_page(text):
                continue

            # Resolve each font-verified heading hit to an *occurrence
            # rank* (how many times that exact substring appeared earlier
            # on the page) rather than a raw char offset -- clean_page_text
            # rewrites `text` (footer/TOC stripping, whitespace collapse),
            # which can shift offsets but not the count/order of a short
            # substring's occurrences. The chunker resolves this rank back
            # to a position within the cleaned `text`.
            heading_hits = detect_headings_on_page(lines, body_size, doc_id)
            headings = []
            for raw_offset, heading in heading_hits:
                occurrence = raw_text.count(heading, 0, raw_offset)
                headings.append({"heading": heading, "occurrence": occurrence})

            records.append(
                {
                    "doc_id": doc_id,
                    "title": doc_cfg["title"],
                    "version": doc_cfg["version"],
                    "org": doc_cfg["org"],
                    "framework": doc_cfg["framework"],
                    "license": doc_cfg["license"],
                    "page": page_number,
                    "headings_on_page": headings,
                    "text": text,
                }
            )

    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[extract] {doc_id}: {len(records)} pages -> {out_path}")
    return out_path


def main():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    for doc_cfg in config["documents"]:
        extract_document(doc_cfg)


if __name__ == "__main__":
    main()
