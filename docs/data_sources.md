# Data Sources & Provenance

All four source guides are published by Scrum.org (and co-authors) under the
**Creative Commons Attribution Share-Alike 4.0** license
(https://creativecommons.org/licenses/by-sa/4.0/legalcode). By ingesting and
serving derived content (chunks, embeddings, generated answers citing them),
this project must retain attribution and, if the processed corpus itself is
redistributed, remain under a compatible share-alike license.

| doc_id | Title | Version / Date | Author(s) / Org | Pages | License |
|---|---|---|---|---|---|
| `scrum_guide_2020` | The Scrum Guide | November 2020 | Ken Schwaber & Jeff Sutherland | 14 | CC BY-SA 4.0 |
| `kanban_guide_2021` | The Kanban Guide for Scrum Teams | January 2021 | Scrum.org, Daniel Vacanti, Yuval Yeret | 9 | CC BY-SA 4.0 |
| `ebm_guide_2024` | The Evidence-Based Management Guide | May 2024 | Scrum.org | 16 | CC BY-SA 4.0 |
| `agility_guide_v1_5` | The Agility Guide to Evidence-Based Change | v1.5 (2014) | Ken Schwaber & Scrum.org | 18 | CC BY-SA 4.0 |

## Version watch

Scrum.org periodically revises these guides (the Scrum Guide has been
revised multiple times, e.g. 2017 -> 2020). When a new version is released:

1. Add the new PDF to `data/raw/`.
2. Add/replace the entry in `ingestion/pipeline_config.yaml -> documents`
   with a new `doc_id` that includes the version (don't reuse an old
   `doc_id` for a new version -- that breaks citation history and makes it
   impossible to tell which version an old cached answer cited).
3. Re-run the full pipeline: `extract.py -> chunker.py -> metadata_tagger.py
   -> embed.py`.
4. Re-run `evaluation/retrieval_eval.py` and diff the report against the
   previous run before deploying.
5. Decide whether to keep the old version's chunks retrievable (useful if
   users reference "the old Scrum Guide") or retire them -- either way, tag
   clearly by version so answers never blend two versions' wording silently.

## Known content overlaps across guides

Several concepts appear in more than one guide with different framing.
The system prompt (`generation/prompt_templates/system_prompt.md`) instructs
the model to surface these differences rather than silently pick one:

- **Sprint / Sprint events** -- defined in the Scrum Guide; extended with a
  flow-based lens in the Kanban Guide; renamed/adapted as "(Change) Sprint"
  at enterprise scale in the Agility Guide.
- **Transparency, Inspection, Adaptation** -- the three empirical pillars
  appear in the Scrum Guide, the Kanban Guide (via Scrum), and the Agility
  Guide's "Evidence-Based Change Theory" section, each applied to a
  different unit of work (product increment vs. workflow vs. organizational
  change).
- **Product Owner vs. Enterprise Product Owner** -- the Agility Guide's
  role is explicitly modeled on the Scrum Guide's Product Owner but scoped
  to enterprise-wide practice adoption rather than a single product.
