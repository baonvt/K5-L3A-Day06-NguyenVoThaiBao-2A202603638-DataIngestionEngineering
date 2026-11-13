"""Qdrant server REST adapter: persistent vectors, cosine search, payload filters."""
from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

import httpx

from .models import Chunk
from .qdrant_service import QDRANT_URL, ensure_qdrant


class QdrantVectorStore:
    FILTER_FIELDS = {"chapter", "section", "language", "document_type", "page"}

    def __init__(self, path: str, dimensions: int) -> None:
        self.path = str(Path(path).resolve())
        self.dimensions = dimensions
        storage_path = Path(self.path) / "storage"
        ensure_qdrant(storage_path)
        self.client = httpx.Client(base_url=QDRANT_URL, timeout=60.0)

    @staticmethod
    def collection_name(strategy: str) -> str:
        return f"day06_{strategy.lower()}"

    def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        response = self.client.request(method, path, **kwargs)
        response.raise_for_status()
        return response.json() if response.content else {}

    def replace(self, strategy: str, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        """Rebuild a strategy collection so re-ingestion removes stale chunks."""
        if len(chunks) != len(vectors):
            raise ValueError("Every chunk must have exactly one embedding")
        for vector in vectors:
            if len(vector) != self.dimensions:
                raise ValueError(f"Expected {self.dimensions}-dimensional vector, got {len(vector)}")
        if not chunks:
            raise ValueError(f"No chunks produced for strategy {strategy}")

        collection = self.collection_name(strategy)
        listed = self._request("GET", "/collections").get("result", {}).get("collections", [])
        if any(item.get("name") == collection for item in listed):
            self._request("DELETE", f"/collections/{collection}")
        self._request(
            "PUT",
            f"/collections/{collection}",
            json={"vectors": {"size": self.dimensions, "distance": "Cosine"}},
        )
        batch_size = 128
        for start in range(0, len(chunks), batch_size):
            batch_chunks = chunks[start:start + batch_size]
            batch_vectors = vectors[start:start + batch_size]
            points = []
            for chunk, vector in zip(batch_chunks, batch_vectors, strict=True):
                points.append({
                    "id": str(uuid.uuid5(uuid.NAMESPACE_URL, chunk.chunk_id)),
                    "vector": vector,
                    "payload": {
                        "chunk_id": chunk.chunk_id,
                        "document_id": chunk.document_id,
                        "source": chunk.source,
                        "page": chunk.page,
                        "text": chunk.text,
                        **chunk.metadata,
                    },
                })
            self._request(
                "PUT",
                f"/collections/{collection}/points",
                params={"wait": "true"},
                json={"points": points},
            )

    def search(
        self,
        strategy: str,
        query_vector: list[float],
        top_k: int = 3,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        if len(query_vector) != self.dimensions:
            raise ValueError(f"Expected a {self.dimensions}-dimensional query vector")
        if top_k < 1 or top_k > 20:
            raise ValueError("top_k must be between 1 and 20")
        must = []
        for field, value in (filters or {}).items():
            if field not in self.FILTER_FIELDS:
                raise ValueError(f"Unsupported metadata filter field: {field}")
            must.append({"key": field, "match": {"value": value}})
        body: dict[str, Any] = {
            "query": query_vector,
            "limit": top_k,
            "with_payload": True,
            "with_vector": False,
        }
        if must:
            body["filter"] = {"must": must}
        response = self._request(
            "POST",
            f"/collections/{self.collection_name(strategy)}/points/query",
            json=body,
        )
        points = response.get("result", {}).get("points", [])
        return [
            {
                "chunk_id": point.get("payload", {}).get("chunk_id"),
                "text": point.get("payload", {}).get("text", ""),
                "score": float(point.get("score", 0.0)),
                "metadata": {key: value for key, value in point.get("payload", {}).items() if key != "text"},
            }
            for point in points
        ]

    def count(self, strategy: str) -> int:
        response = self._request(
            "POST",
            f"/collections/{self.collection_name(strategy)}/points/count",
            json={"exact": True},
        )
        return int(response.get("result", {}).get("count", 0))

    def collection_info(self, strategy: str) -> dict[str, Any]:
        return self._request("GET", f"/collections/{self.collection_name(strategy)}").get("result", {})

    def server_version(self) -> str:
        return str(self._request("GET", "/").get("version", "unknown"))

    def close(self) -> None:
        self.client.close()
