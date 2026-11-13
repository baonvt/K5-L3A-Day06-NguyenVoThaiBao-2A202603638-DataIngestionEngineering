"""Local, real dense embeddings backed by FastEmbed and ONNX Runtime."""
from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol

from fastembed import TextEmbedding


class Embedder(Protocol):
    model_name: str
    dimensions: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, question: str) -> list[float]: ...


class FastEmbedder:
    """CPU inference with a downloadable, versioned FastEmbed model."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        supported = {item["model"]: item for item in TextEmbedding.list_supported_models()}
        if model_name not in supported:
            raise ValueError(f"Unsupported FastEmbed model: {model_name}")
        self.model_name = model_name
        self.dimensions = int(supported[model_name]["dim"])
        self.model = TextEmbedding(model_name=model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return [vector.tolist() for vector in self.model.embed(texts, batch_size=64)]

    def embed_query(self, question: str) -> list[float]:
        # The chosen MiniLM encoder uses the same text representation for query and passage.
        return next(iter(self.model.embed([question], batch_size=1))).tolist()

    def embed_iter(self, texts: Iterable[str], batch_size: int = 64) -> Iterable[list[float]]:
        for vector in self.model.embed(list(texts), batch_size=batch_size):
            yield vector.tolist()
