"""Small retrieval API for live demos and reproducible simulated benchmark queries."""
from __future__ import annotations

import json
from contextlib import asynccontextmanager
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .embeddings import FastEmbedder
from .run import BENCHMARK_PATH, INDEX_PATH, MODEL_NAME
from .store import QdrantVectorStore


runtime: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    embedder = FastEmbedder(MODEL_NAME)
    store = QdrantVectorStore(str(INDEX_PATH), embedder.dimensions)
    runtime.update({"embedder": embedder, "store": store})
    yield
    store.close()
    runtime.clear()


app = FastAPI(
    title="Day 06 Data Ingestion Demo",
    description="Real local embedding + persistent Qdrant retrieval. No mock vectors or remote API.",
    version="1.0.0",
    lifespan=lifespan,
)


class SearchRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    strategy: Literal["A", "B", "C"] = "C"
    filters: dict[str, str | int] = Field(default_factory=dict)
    top_k: int = Field(default=3, ge=1, le=20)


def _get_components() -> tuple[FastEmbedder, QdrantVectorStore]:
    if not runtime:
        raise HTTPException(status_code=503, detail="Retrieval runtime is not ready")
    return runtime["embedder"], runtime["store"]


@app.get("/healthz")
def health() -> dict[str, Any]:
    if not runtime:
        return {"status": "starting"}
    store: QdrantVectorStore = runtime["store"]
    counts = {strategy: store.count(strategy) for strategy in ("A", "B", "C")}
    ready = all(count > 0 for count in counts.values())
    return {
        "status": "ready" if ready else "indexes_missing",
        "embedding_model": runtime["embedder"].model_name,
        "dimensions": runtime["embedder"].dimensions,
        "vector_store": "Qdrant server (persistent, cosine)",
        "points": counts,
    }


@app.post("/search")
def search(request: SearchRequest) -> dict[str, Any]:
    embedder, store = _get_components()
    if request.strategy != "C" and request.filters:
        raise HTTPException(status_code=400, detail="A/B are the no-filter baselines; use strategy C for metadata filters")
    try:
        vector = embedder.embed_query(request.question)
        hits = store.search(request.strategy, vector, top_k=request.top_k, filters=request.filters)
    except Exception as exc:
        # Surface backend/model errors; do not switch to a mock or a different index.
        raise HTTPException(status_code=503, detail=f"Retrieval failed: {exc}") from exc
    return {
        "question": request.question,
        "strategy": request.strategy,
        "filter_applied": request.filters or None,
        "top_k": request.top_k,
        "hits": hits,
    }


@app.get("/simulate/{question_id}")
def simulate(question_id: str, strategy: Literal["A", "B", "C"] = "C") -> dict[str, Any]:
    """Run one pre-written benchmark query without passing its answer or gold page to search."""
    embedder, store = _get_components()
    try:
        questions = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Benchmark file unavailable: {exc}") from exc
    question = next((item for item in questions if item["id"] == question_id), None)
    if question is None:
        raise HTTPException(status_code=404, detail=f"Unknown benchmark question: {question_id}")
    active_filter = question.get("filter") if strategy == "C" else None
    try:
        vector = embedder.embed_query(question["question"])
        hits = store.search(strategy, vector, top_k=3, filters=active_filter)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Simulation failed: {exc}") from exc
    return {
        "question_id": question_id,
        "split": question["split"],
        "question": question["question"],
        "strategy": strategy,
        "filter_applied": active_filter,
        "hits": hits,
    }


if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="Run the local Day 06 retrieval API")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run("src.server:app", host=args.host, port=args.port, reload=False)
