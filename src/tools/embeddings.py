"""Dense embedding model wrapper for semantic search (all-MiniLM-L6-v2).

Kept separate from the storage layer so storage.py (and the Postgres stub)
never needs torch/sentence-transformers as a dependency — storage only ever
receives an already-computed vector. The model itself is loaded lazily, on
first actual use, so importing this module (or starting the app) doesn't pay
the torch/model-load cost unless the dense search path is exercised.
"""

from __future__ import annotations

from functools import lru_cache
from typing import List

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384


@lru_cache(maxsize=1)
def _get_model():
    # Deferred import: this is the expensive dependency (torch + transformers).
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Batch-embed a list of texts. Vectors are L2-normalized, so downstream
    cosine similarity is just a dot product — storage.search_embedding relies
    on that."""
    if not texts:
        return []
    model = _get_model()
    vectors = model.encode(list(texts), normalize_embeddings=True, show_progress_bar=False)
    return vectors.tolist()


def embed_query(text: str) -> List[float]:
    return embed_texts([text])[0]
