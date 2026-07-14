"""
metadata_tagger.py
-------------------
Enriches each chunk in data/processed/all_chunks.jsonl with:
  - concept_tags:   controlled-vocabulary tags matched via keyword search
                    (see pipeline_config.yaml -> taxonomy.concept_keywords)
  - roles_relevant: which of the 5 target personas this chunk matters to
  - artifact_type:  role | event | artifact | metric | theory-pillar |
                    value | practice | definition | overview

This is intentionally rule-based (not an LLM call) so tagging is fast,
free, deterministic, and reviewable in a diff. If you outgrow the keyword
approach, swap `tag_chunk()` for an LLM-assisted tagger that proposes tags
which a human then approves -- keep the taxonomy closed either way, or your
metadata filters in retrieval/retriever.py will silently stop working.
"""

import json
import re
from pathlib import Path

import yaml

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
CONFIG_PATH = Path(__file__).resolve().parent / "pipeline_config.yaml"

# Section-heading -> artifact_type mapping. Falls back to "overview" for
# sections that don't clearly map to one Scrum concept type.
ARTIFACT_TYPE_RULES = {
    "role": ["Scrum Master", "Product Owner", "Developers", "Enterprise Product Owner",
             "Enterprise Scrum Master", "Enterprise Change Team", "Agility Team"],
    "event": ["Sprint Planning", "Daily Scrum", "Weekly Scrum", "Sprint Review",
              "Sprint Retrospective", "The Sprint", "Sprint", "Evaluation"],
    "artifact": ["Product Backlog", "Sprint Backlog", "Increment", "Practice Backlog",
                 "Evaluation Backlog", "Increment of Change", "Definition of Workflow"],
    "metric": ["The Basic Metrics of Flow", "Little\u2019s Law", "Current Value",
               "Unrealized Value", "Ability to Innovate", "Time-to-Market",
               "Agility Index", "Agility Acceleration", "Evidence-Based Metrics"],
    "theory-pillar": ["Transparency", "Inspection", "Adaptation", "Scrum Theory",
                      "Kanban with Scrum Theory", "Evidence-Based Change Theory",
                      "Flow and Empiricism"],
    "value": ["Scrum Values"],
    "practice": ["Kanban Practices", "Visualization of the Workflow",
                 "Limiting Work in Progress", "Active Management of Work Items"],
    "definition": ["Definition of Kanban", "Scrum Definition",
                   "Definition of Evidence-Based Management"],
}

# Role relevance rules: which personas care most about which sections.
# A chunk can (and usually should) be relevant to more than one role.
ROLE_RULES = {
    "Scrum Master": ["Scrum Master", "Daily Scrum", "Sprint Retrospective",
                     "Sprint Planning", "Kanban Practices", "Transparency",
                     "Inspection", "Adaptation", "Weekly Scrum",
                     "Enterprise Scrum Master"],
    "Product Owner": ["Product Owner", "Product Backlog", "Sprint Backlog",
                      "Setting Goals", "Unrealized Value", "Current Value",
                      "Enterprise Product Owner", "Practice Backlog"],
    "Developers": ["Developers", "Sprint Backlog", "Daily Scrum",
                   "Definition of Workflow", "Limiting Work in Progress",
                   "The Basic Metrics of Flow"],
    "Stakeholder": ["Sprint Review", "Purpose of the Scrum Guide",
                    "Scrum Definition", "EBM Helps Organizations",
                    "Current Value", "Unrealized Value", "Ability to Innovate",
                    "Time-to-Market"],
    "Project Manager/PMO": ["Evidence-Based Metrics", "Agility Index",
                            "Agility Acceleration", "Domains", "Framework",
                            "Setting Goals", "Key Value Area",
                            "Enterprise Product Owner", "Enterprise Change Team"],
}


def match_artifact_type(section: str) -> str:
    for artifact_type, sections in ARTIFACT_TYPE_RULES.items():
        if section in sections:
            return artifact_type
    return "overview"


def match_roles(section: str, text: str) -> list:
    matched = set()
    haystack = f"{section} {text}".lower()
    for role, sections in ROLE_RULES.items():
        if section in sections:
            matched.add(role)
    # fallback: if nothing matched by section, everyone benefits from
    # foundational/overview content (theory, purpose, definitions)
    if not matched:
        matched = {"Scrum Master", "Product Owner", "Developers",
                   "Stakeholder", "Project Manager/PMO"}
    return sorted(matched)


def match_concept_tags(text: str, keyword_map: dict) -> list:
    text_lower = text.lower()
    tags = set()
    for keyword, tag in keyword_map.items():
        if keyword in text_lower:
            tags.add(tag)
    return sorted(tags)


def tag_chunk(chunk: dict, keyword_map: dict) -> dict:
    chunk["artifact_type"] = match_artifact_type(chunk["section"])
    chunk["roles_relevant"] = match_roles(chunk["section"], chunk["text"])
    chunk["concept_tags"] = match_concept_tags(chunk["text"], keyword_map)
    return chunk


def main():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    keyword_map = config["taxonomy"]["concept_keywords"]

    in_path = PROCESSED_DIR / "all_chunks.jsonl"
    with open(in_path, "r", encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f]

    tagged = [tag_chunk(c, keyword_map) for c in chunks]

    out_path = PROCESSED_DIR / "all_chunks_tagged.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in tagged:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # quick coverage report
    untagged_concepts = sum(1 for c in tagged if not c["concept_tags"])
    print(f"[tag] {len(tagged)} chunks tagged -> {out_path}")
    print(f"[tag] {untagged_concepts} chunks matched no concept keyword "
          f"(still fine -- they rely on dense retrieval, not keyword filters)")


if __name__ == "__main__":
    main()
