"""
chunker.py
----------
Turns the page-level interim JSONL into final chunks for embedding.

Strategy (see docs/architecture.md section 3.2 for the rationale):
1. Concatenate a document's pages into one continuous text stream, but keep
   track of which page each character range came from (for citations).
2. Split on detected section headings first, so a chunk never spans two
   major sections (e.g., "Sprint Planning" content never bleeds into
   "Daily Scrum" content).
3. Within a section, split into ~target_tokens-sized chunks on paragraph
   boundaries, with overlap_tokens of trailing context carried into the
   next chunk.
4. Small sections (e.g., short definitions) are kept as a single atomic
   chunk even if under min_tokens, rather than merged with a neighbor --
   splitting/merging a self-contained definition hurts retrieval more than
   an undersized chunk does.

Token counting here uses whitespace word count as a cheap proxy. Swap in a
real tokenizer (tiktoken, etc.) if you want exact model-token budgets.
"""

import json
import re
from pathlib import Path

import yaml

BASE_DIR = Path(__file__).resolve().parent.parent
INTERIM_DIR = BASE_DIR / "data" / "interim"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
CONFIG_PATH = Path(__file__).resolve().parent / "pipeline_config.yaml"


def word_count(text: str) -> int:
    return len(text.split())


def load_pages(doc_id: str) -> list:
    path = INTERIM_DIR / f"{doc_id}.jsonl"
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def find_nth_occurrence(text: str, sub: str, n: int) -> int:
    """Return the start index of the n-th (0-based) occurrence of `sub` in
    `text`, or -1 if there aren't that many."""
    start = 0
    idx = -1
    for _ in range(n + 1):
        idx = text.find(sub, start)
        if idx == -1:
            return -1
        start = idx + 1
    return idx


def split_into_sections(pages: list) -> list:
    """Group page text into sections using the headings detected during
    extraction. Returns a list of dicts: {heading, text, pages: [page_nums]}.
    Text before the first detected heading is grouped under 'Front Matter'.
    """
    sections = []
    current = {"heading": "Front Matter", "text": "", "pages": []}

    for page in pages:
        text = page["text"]
        headings = page["headings_on_page"]

        if not headings:
            current["text"] += "\n" + text
            current["pages"].append(page["page"])
            continue

        # Resolve each heading's occurrence rank (assigned during
        # extraction, based on font size -- see extract.py) to its actual
        # position in this page's cleaned text, rather than blindly taking
        # the first substring match (which is what caused headings to be
        # mis-attributed when a heading's name also appeared earlier as
        # running prose or a bolded inline term).
        indices = [
            (find_nth_occurrence(text, h["heading"], h["occurrence"]), h["heading"])
            for h in headings
        ]
        indices = [(i, h) for i, h in indices if i != -1]
        indices.sort()

        cursor = 0
        for idx, heading in indices:
            pre_text = text[cursor:idx]
            if pre_text.strip():
                current["text"] += "\n" + pre_text
                if page["page"] not in current["pages"]:
                    current["pages"].append(page["page"])
            # close out current section, start new one
            if current["text"].strip():
                sections.append(current)
            current = {"heading": heading, "text": "", "pages": [page["page"]]}
            cursor = idx

        # remainder of the page after the last heading
        remainder = text[cursor:]
        # strip the heading text itself from the start of remainder
        for _, heading in indices:
            if remainder.startswith(heading):
                remainder = remainder[len(heading):]
                break
        current["text"] += "\n" + remainder
        if page["page"] not in current["pages"]:
            current["pages"].append(page["page"])

    if current["text"].strip():
        sections.append(current)

    return sections


def chunk_section(section: dict, doc_id: str, target: int, min_tok: int,
                   overlap: int) -> list:
    """Split one section's text into chunks on paragraph boundaries."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", section["text"]) if p.strip()]
    if not paragraphs:
        return []

    chunks = []
    buf = []
    buf_words = 0

    def flush():
        if buf:
            chunks.append(" ".join(buf).strip())

    for para in paragraphs:
        pw = word_count(para)
        if buf_words + pw > target and buf_words >= min_tok:
            flush()
            # carry overlap: take trailing words of previous buffer
            overlap_text = " ".join(" ".join(buf).split()[-overlap:]) if overlap else ""
            buf = [overlap_text] if overlap_text else []
            buf_words = word_count(overlap_text) if overlap_text else 0
        buf.append(para)
        buf_words += pw

    flush()

    # if the whole section is tiny, we still want it as one atomic chunk
    if not chunks:
        chunks = [section["text"].strip()]

    return chunks


def build_chunks(doc_id: str, cfg: dict, chunk_cfg: dict) -> list:
    pages = load_pages(doc_id)
    sections = split_into_sections(pages)

    records = []
    seq = 0
    for section in sections:
        heading = section["heading"]
        text_chunks = chunk_section(
            section, doc_id,
            target=chunk_cfg["target_tokens"],
            min_tok=chunk_cfg["min_tokens"],
            overlap=chunk_cfg["overlap_tokens"],
        )
        for chunk_text in text_chunks:
            chunk_text = re.sub(r"\s+", " ", chunk_text).strip()
            if not chunk_text:
                continue
            seq += 1
            chunk_id = f"{doc_id}_{seq:03d}"
            # embed-friendly text: prefix with heading path for context,
            # per the "hybrid semantic + structural chunking" approach
            embed_text = f"{cfg['title']} > {heading}: {chunk_text}"
            records.append(
                {
                    "chunk_id": chunk_id,
                    "source_doc": cfg["title"],
                    "doc_id": doc_id,
                    "source_org": cfg["org"],
                    "license": cfg["license"],
                    "framework": cfg["framework"],
                    "version": cfg["version"],
                    "section": heading,
                    "pages": sorted(set(section["pages"])),
                    "text": chunk_text,
                    "embed_text": embed_text,
                }
            )
    return records


def merge_small_chunks(records: list, min_tok: int) -> list:
    """Post-process safety net: merge any chunk under min_tok words into the
    next chunk from the *same document*, so stray tiny fragments (e.g. a
    leftover heading with no body text) don't end up as standalone,
    low-signal embeddings. Re-sequences chunk_ids afterward."""
    if not records:
        return records

    merged = []
    buf = None
    for rec in records:
        if buf is None:
            buf = dict(rec)
            continue
        if word_count(buf["text"]) < min_tok and buf["doc_id"] == rec["doc_id"]:
            # fold buf into the next record's text (keep next's section/page
            # metadata since that's where most of the content lives)
            combined_text = (buf["text"] + " " + rec["text"]).strip()
            rec = dict(rec)
            rec["text"] = combined_text
            rec["pages"] = sorted(set(buf["pages"]) | set(rec["pages"]))
            rec["embed_text"] = f"{rec['source_doc']} > {rec['section']}: {combined_text}"
            buf = rec
        else:
            merged.append(buf)
            buf = dict(rec)
    if buf is not None:
        merged.append(buf)

    # re-sequence chunk_ids per doc
    counters = {}
    for rec in merged:
        counters.setdefault(rec["doc_id"], 0)
        counters[rec["doc_id"]] += 1
        rec["chunk_id"] = f"{rec['doc_id']}_{counters[rec['doc_id']]:03d}"

    return merged


def main():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    chunk_cfg = config["chunking"]
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    all_chunks = []
    for doc_cfg in config["documents"]:
        doc_id = doc_cfg["doc_id"]
        chunks = build_chunks(doc_id, doc_cfg, chunk_cfg)
        chunks = merge_small_chunks(chunks, chunk_cfg["min_tokens"])
        out_path = PROCESSED_DIR / f"{doc_id}_chunks.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for rec in chunks:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[chunk] {doc_id}: {len(chunks)} chunks -> {out_path}")
        all_chunks.extend(chunks)

    combined_path = PROCESSED_DIR / "all_chunks.jsonl"
    with open(combined_path, "w", encoding="utf-8") as f:
        for rec in all_chunks:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[chunk] TOTAL: {len(all_chunks)} chunks -> {combined_path}")


if __name__ == "__main__":
    main()
