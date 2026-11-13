"""Check the local runtime dependencies without downloading or mocking a model."""
from __future__ import annotations

import os
import sys
from pathlib import Path

from fastembed import TextEmbedding

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.qdrant_service import QDRANT_EXE, QDRANT_VERSION

model_name = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
models = {item["model"]: item for item in TextEmbedding.list_supported_models()}
if model_name not in models:
    raise SystemExit(f"FastEmbed does not list configured model {model_name!r}")
print(f"Embedding model: {model_name} | dimensions: {models[model_name]['dim']}")
print(f"Qdrant: pinned server version {QDRANT_VERSION}; binary present: {QDRANT_EXE.exists()}")
print("Setup OK. Run `python -m src.run` to download/start Qdrant and download the real model if needed.")
