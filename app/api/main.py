"""
main.py (FastAPI app)
----------------------
Thin HTTP layer over generation/chat_chain.py.

Run locally:
    uvicorn app.api.main:app --reload --port 8000

POST /chat
{
  "question": "How should a Scrum Master handle a team missing WIP limits?",
  "role": "Scrum Master"   // optional -- omit to let the router infer it
}
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from generation.chat_chain import ScrumRAGChain

app = FastAPI(title="Scrum RAG Chatbot API", version="0.1.0")

# Wide open for local testing with app/ui/index.html served from a
# different origin/port. Tighten this before deploying anywhere public.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Loaded once at startup -- HybridRetriever holds the vector index and BM25
# corpus in memory, so we don't want to rebuild it per-request.
chain = ScrumRAGChain()


class ChatRequest(BaseModel):
    question: str
    role: str | None = None


class ChatResponse(BaseModel):
    question: str
    routing: dict
    sources: list
    answer: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    result = chain.answer(req.question, explicit_role=req.role)
    return result
