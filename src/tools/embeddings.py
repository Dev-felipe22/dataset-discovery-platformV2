"""Dense embedding model wrapper for semantic search (all-MiniLM-L6-v2).

Kept separate from the storage layer so storage.py (and the Postgres stub)
never needs torch/sentence-transformers as a dependency — storage only ever
receives an already-computed vector. The model itself is loaded lazily, on
first actual use, so importing this module (or starting the app) doesn't pay
the torch/model-load cost unless the dense search path is exercised.

GPU: sentence-transformers will use CUDA automatically if torch reports it
available, no code change needed for that part. What this module adds on
top is making the choice visible (get_device()/a startup print) rather than
silent, and a tunable batch_size — the default of 32 is sized for CPU; a
modern GPU is badly underused at that batch size.
"""

from __future__ import annotations

from functools import lru_cache
from typing import List

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
DEFAULT_BATCH_SIZE = 64


@lru_cache(maxsize=1)
def get_device() -> str:
    """'cuda' if a GPU is visible to this process and torch can use it, else
    'cpu'. Cached — checked once per process, not per call."""
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


@lru_cache(maxsize=1)
def _get_model():
    # Deferred import: this is the expensive dependency (torch + transformers).
    from sentence_transformers import SentenceTransformer

    device = get_device()
    return SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)


def embed_texts(texts: List[str], *, batch_size: int = DEFAULT_BATCH_SIZE) -> List[List[float]]:
    """Batch-embed a list of texts. Vectors are L2-normalized, so downstream
    cosine similarity is just a dot product — storage.search_embedding relies
    on that."""
    if not texts:
        return []
    model = _get_model()
    vectors = model.encode(
        list(texts),
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return vectors.tolist()


def embed_query(text: str) -> List[float]:
    return embed_texts([text])[0]
