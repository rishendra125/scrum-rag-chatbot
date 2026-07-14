"""
chat_chain.py
-------------
End-to-end orchestration: user question -> routed retrieval -> reranked
context -> assembled prompt -> LLM call -> cited answer.

The LLM call itself is behind a small `generate()` function so you can
plug in Anthropic, OpenAI, or a local model without touching the rest of
the chain. Ships with a `DryRunLLM` that requires no API key/network, so
you can validate the full retrieval + prompt-assembly pipeline offline
before wiring in a real model.
"""

import os
from pathlib import Path

from retrieval.retriever import HybridRetriever, format_context
from retrieval.reranker import rerank
from retrieval.query_router import route

BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "prompt_templates"

ROLE_TEMPLATE_FILES = {
    "Scrum Master": "role_scrum_master.md",
    "Product Owner": "role_product_owner.md",
    "Stakeholder": "role_stakeholder.md",
    "Project Manager/PMO": "role_project_manager.md",
    # "Developers" intentionally has no distinct overlay yet -- they get the
    # base system prompt. Add generation/prompt_templates/role_developers.md
    # and register it here if you want Developer-specific framing.
}


def load_template(filename: str) -> str:
    path = TEMPLATES_DIR / filename
    return path.read_text(encoding="utf-8") if path.exists() else ""


def build_system_prompt(role: str = None) -> str:
    base = load_template("system_prompt.md")
    overlay = ""
    if role and role in ROLE_TEMPLATE_FILES:
        overlay = "\n\n" + load_template(ROLE_TEMPLATE_FILES[role])
    return base + overlay


def build_user_message(question: str, context: str) -> str:
    return (
        f"## Retrieved context\n\n{context}\n\n"
        f"## User question\n\n{question}\n\n"
        f"Answer using only the retrieved context above, with citations "
        f"in the format (Guide Name, Section, p.X)."
    )


class DryRunLLM:
    """No-network stand-in so the chain is fully testable offline. Echoes
    back which chunks would have been sent, instead of a generated answer.
    Swap for AnthropicLLM/OpenAILLM below once you have API access."""

    def generate(self, system_prompt: str, user_message: str) -> str:
        return (
            "[DryRunLLM -- no model call made]\n\n"
            "This stand-in confirms the prompt assembled correctly. "
            "Replace chat_chain.get_llm() with a real backend "
            "(see AnthropicLLM in this file) to get generated answers.\n\n"
            f"--- system prompt ({len(system_prompt)} chars) preview ---\n"
            f"{system_prompt[:300]}...\n\n"
            f"--- user message ({len(user_message)} chars) preview ---\n"
            f"{user_message[:500]}..."
        )


class AnthropicLLM:
    """Real backend using the Anthropic Messages API. Requires
    ANTHROPIC_API_KEY in the environment and network access."""

    def __init__(self, model: str = "claude-sonnet-4-6"):
        self.model = model

    def generate(self, system_prompt: str, user_message: str) -> str:
        import anthropic
        client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
        resp = client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )
        return "".join(block.text for block in resp.content if block.type == "text")


def get_llm():
    backend = os.environ.get("LLM_BACKEND", "dry_run")
    if backend == "anthropic":
        return AnthropicLLM()
    return DryRunLLM()


class ScrumRAGChain:
    def __init__(self):
        self.retriever = HybridRetriever()
        self.llm = get_llm()

    def answer(self, question: str, explicit_role: str = None,
               top_k_retrieve: int = 20, top_k_final: int = 5) -> dict:
        routed = route(question, explicit_role=explicit_role)

        candidates = self.retriever.retrieve(
            question,
            top_k=top_k_retrieve,
            role=routed["role"],
            framework=routed["framework"],
        )
        final_chunks = rerank(question, candidates, top_k=top_k_final)

        context = format_context(final_chunks)
        system_prompt = build_system_prompt(role=routed["role"])
        user_message = build_user_message(question, context)

        answer_text = self.llm.generate(system_prompt, user_message)

        return {
            "question": question,
            "routing": routed,
            "sources": [
                {
                    "source_doc": c["source_doc"],
                    "section": c["section"],
                    "pages": c["pages"],
                    "chunk_id": c["chunk_id"],
                }
                for c in final_chunks
            ],
            "answer": answer_text,
        }


if __name__ == "__main__":
    import sys
    import json as _json

    chain = ScrumRAGChain()
    question = " ".join(sys.argv[1:]) or "What is the timebox for Sprint Retrospective?"
    result = chain.answer(question)
    print(_json.dumps(result, indent=2))
