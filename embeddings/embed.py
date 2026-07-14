"""
embed.py
--------
Pluggable embedding layer. Two backends are supported out of the box:

1. "api"   -- calls an embedding API (OpenAI text-embedding-3-large, or any
              OpenAI-compatible endpoint). Requires network + API key, so
              this is what you'd use in a real deployment.
2. "local" -- a dependency-light TF-IDF vectorizer (scikit-learn) used as a
              fallback so the whole pipeline (ingestion -> embed -> index ->
              retrieve -> eval) can be built, run, and tested completely
              offline, with zero API cost, before you wire in real
              embeddings for production quality.

Swap backends via EMBEDDING_BACKEND env var or the `backend=` argument.
The rest of the pipeline (index_builder.py, retriever.py) only depends on
the `Embedder` interface below, so switching backends never requires
touching retrieval code.
"""

import json
import os
import pickle
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
INDEX_DIR = BASE_DIR / "embeddings" / "index"


class Embedder:
    """Common interface both backends implement."""

    def fit(self, texts: list):
        raise NotImplementedError

    def transform(self, texts: list) -> np.ndarray:
        raise NotImplementedError

    def save(self, path: Path):
        raise NotImplementedError

    @classmethod
    def load(cls, path: Path):
        raise NotImplementedError


class LocalTfidfEmbedder(Embedder):
    """
    Offline fallback embedder. Not a semantic embedding model -- it's a
    TF-IDF vector space -- but it lets the full pipeline run without
    network access or an API key, which is useful for local dev, CI, and
    demoing the architecture. Retrieval quality with this backend will be
    noticeably weaker on paraphrased questions than a real embedding model;
    plan to switch to APIEmbedder (or a local sentence-transformers model)
    before shipping to real users.
    """

    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            max_features=20000,
        )
        self._fitted = False

    def fit(self, texts: list):
        self.vectorizer.fit(texts)
        self._fitted = True
        return self

    def transform(self, texts: list) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Embedder.fit() must be called before transform().")
        matrix = self.vectorizer.transform(texts)
        return matrix.toarray().astype("float32")

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.vectorizer, f)

    @classmethod
    def load(cls, path: Path):
        obj = cls.__new__(cls)
        with open(path, "rb") as f:
            obj.vectorizer = pickle.load(f)
        obj._fitted = True
        return obj


class APIEmbedder(Embedder):
    """
    Real semantic embeddings via an OpenAI-compatible API. Requires
    `OPENAI_API_KEY` in the environment and network access. This is a thin
    wrapper -- swap `model` for any embedding endpoint (Voyage, Cohere,
    local vLLM server, etc.) as long as it returns a list of float vectors.
    """

    def __init__(self, model: str = "text-embedding-3-large"):
        self.model = model
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        return self._client

    def fit(self, texts: list):
        # API embedders are stateless -- nothing to fit.
        return self

    def transform(self, texts: list) -> np.ndarray:
        client = self._get_client()
        vectors = []
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            resp = client.embeddings.create(model=self.model, input=batch)
            vectors.extend([d.embedding for d in resp.data])
        return np.array(vectors, dtype="float32")

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump({"model": self.model, "backend": "api"}, f)

    @classmethod
    def load(cls, path: Path):
        with open(path, "r") as f:
            cfg = json.load(f)
        return cls(model=cfg["model"])


def get_embedder(backend: str = None) -> Embedder:
    backend = backend or os.environ.get("EMBEDDING_BACKEND", "local")
    if backend == "local":
        return LocalTfidfEmbedder()
    elif backend == "api":
        return APIEmbedder()
    raise ValueError(f"Unknown embedding backend: {backend}")


def load_chunks() -> list:
    path = PROCESSED_DIR / "all_chunks_tagged.jsonl"
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def main():
    backend = os.environ.get("EMBEDDING_BACKEND", "local")
    print(f"[embed] backend={backend}")

    chunks = load_chunks()
    texts = [c["embed_text"] for c in chunks]

    embedder = get_embedder(backend)
    embedder.fit(texts)
    vectors = embedder.transform(texts)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    np.save(INDEX_DIR / "vectors.npy", vectors)
    embedder.save(INDEX_DIR / f"embedder_{backend}.pkl" if backend == "local"
                  else INDEX_DIR / f"embedder_{backend}.json")

    with open(INDEX_DIR / "chunk_meta.jsonl", "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    print(f"[embed] {vectors.shape[0]} vectors of dim {vectors.shape[1]} "
          f"-> {INDEX_DIR}")


if __name__ == "__main__":
    main()
