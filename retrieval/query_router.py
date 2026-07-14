"""
query_router.py
---------------
Lightweight, rule-based intent classifier that decides:
  - which role the retriever should soft-boost (if the user didn't
    explicitly pick one in the UI)
  - which framework(s) the question likely concerns
  - whether this looks like a "real situation / scenario" question
    (mapped to the scenario prompt template) vs. a definitional lookup

Kept rule-based (regex/keyword) rather than an LLM call so routing is
instant and free -- it only needs to be "good enough" to bias retrieval,
since the retriever's soft-boost (not hard filter) forgives a wrong guess.
"""

import re

ROLE_KEYWORDS = {
    "Scrum Master": [r"\bscrum master\b", r"\bfacilitat", r"\bimpediment", r"\bcoach"],
    "Product Owner": [r"\bproduct owner\b", r"\bbacklog order", r"\bproduct goal",
                      r"\bmaximize value\b", r"\bprioriti[sz]"],
    "Developers": [r"\bdeveloper", r"\bhow do we build\b", r"\bsprint backlog\b.*plan"],
    "Stakeholder": [r"\bstakeholder\b", r"\bwhy (are|is) (we|the team)\b", r"\binvestor"],
    "Project Manager/PMO": [r"\bpmo\b", r"\bproject manager\b", r"\broi\b",
                            r"\bportfolio\b", r"\benterprise\b"],
}

FRAMEWORK_KEYWORDS = {
    "Scrum": [r"\bscrum\b", r"\bsprint\b", r"\bproduct backlog\b"],
    "Kanban": [r"\bkanban\b", r"\bwip\b", r"\bwork in progress\b", r"\bflow\b",
              r"\bcycle time\b", r"\bthroughput\b"],
    "Evidence-Based Management": [r"\bebm\b", r"\bkey value area\b", r"\bkva\b",
                                  r"\bunrealized value\b", r"\bcurrent value\b"],
    "Evidence-Based Change": [r"\benterprise\b", r"\bagility (index|acceleration)\b",
                              r"\bdomain\b", r"\bpractice backlog\b"],
}

SCENARIO_PATTERNS = [
    r"\bmy team\b", r"\bwe (have|are|keep)\b", r"\bwhat should (i|we) do\b",
    r"\bhow (do|should) (i|we)\b", r"\bsituation\b", r"\bproblem\b",
    r"\bstruggling\b", r"\bkeeps? happening\b",
]


def _any_match(patterns: list, text: str) -> bool:
    return any(re.search(p, text, flags=re.IGNORECASE) for p in patterns)


def infer_role(question: str) -> str:
    for role, patterns in ROLE_KEYWORDS.items():
        if _any_match(patterns, question):
            return role
    return None


def infer_framework(question: str) -> str:
    matches = [fw for fw, patterns in FRAMEWORK_KEYWORDS.items()
               if _any_match(patterns, question)]
    # if multiple frameworks match, don't pick one -- let retrieval stay broad
    if len(matches) == 1:
        return matches[0]
    return None


def is_scenario_question(question: str) -> bool:
    return _any_match(SCENARIO_PATTERNS, question)


def route(question: str, explicit_role: str = None) -> dict:
    return {
        "role": explicit_role or infer_role(question),
        "framework": infer_framework(question),
        "is_scenario": is_scenario_question(question),
    }


if __name__ == "__main__":
    tests = [
        "How should a Product Owner order the backlog?",
        "My team keeps missing WIP limits, what should we do?",
        "What is the timebox for Sprint Retrospective?",
        "How does EBM measure unrealized value?",
    ]
    for t in tests:
        print(t, "->", route(t))
