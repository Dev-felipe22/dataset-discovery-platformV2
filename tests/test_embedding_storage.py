from __future__ import annotations

import os
import tempfile
from datetime import datetime
from pathlib import Path

from src.storage.duckdb_backend import DuckDBStorage
from src.storage.models import Dataset


def _make_storage(tmp_path: Path) -> DuckDBStorage:
    db_path = tmp_path / "embedding_storage.duckdb"
    storage = DuckDBStorage(str(db_path))
    storage.init()
    return storage


def _seed_datasets(storage: DuckDBStorage) -> None:
    now = datetime.utcnow()
    datasets = [
        Dataset(
            id=f"acme/ds-{i}",
            source="hf",
            ns_local_id=f"ds-{i}",
            title=f"Dataset {i}",
            description=f"Description for dataset {i}",
            fingerprint=f"fp{i}",
            indexed_at=now,
        )
        for i in range(3)
    ]
    storage.bulk_upsert_datasets(datasets)


def test_list_dataset_embedding_inputs_does_not_require_search_index(tmp_path: Path) -> None:
    storage = _make_storage(tmp_path)
    _seed_datasets(storage)
    inputs = storage.list_dataset_embedding_inputs()
    ids = {i["id"] for i in inputs}
    assert ids == {"acme/ds-0", "acme/ds-1", "acme/ds-2"}
    ds0 = next(i for i in inputs if i["id"] == "acme/ds-0")
    assert "Dataset 0" in ds0["text"]


def test_search_embedding_ranks_by_cosine_similarity(tmp_path: Path) -> None:
    storage = _make_storage(tmp_path)
    _seed_datasets(storage)

    # Hand-crafted, pre-normalized toy vectors (dimensionality doesn't matter
    # to the ranking math, only that query/stored vectors share one).
    storage.upsert_dataset_embeddings(
        [
            {"dataset_id": "acme/ds-0", "embedding": [1.0, 0.0, 0.0, 0.0]},   # identical to query
            {"dataset_id": "acme/ds-1", "embedding": [0.0, 1.0, 0.0, 0.0]},   # orthogonal
            {"dataset_id": "acme/ds-2", "embedding": [0.7071, 0.7071, 0.0, 0.0]},  # 45 degrees
        ],
        model="fake-test-model",
    )

    query_embedding = [1.0, 0.0, 0.0, 0.0]
    hits, total = storage.search_embedding(query_embedding, limit=10, offset=0)

    assert total == 3
    ranked_ids = [h["id"] for h in hits]
    assert ranked_ids == ["acme/ds-0", "acme/ds-2", "acme/ds-1"]
    assert hits[0]["score"] > hits[1]["score"] > hits[2]["score"]
    assert abs(hits[0]["score"] - 1.0) < 1e-4
    assert abs(hits[2]["score"] - 0.0) < 1e-4


def test_search_embedding_respects_filters_and_pagination(tmp_path: Path) -> None:
    storage = _make_storage(tmp_path)
    _seed_datasets(storage)
    storage.upsert_dataset_embeddings(
        [
            {"dataset_id": "acme/ds-0", "embedding": [1.0, 0.0]},
            {"dataset_id": "acme/ds-1", "embedding": [0.9, 0.1]},
            {"dataset_id": "acme/ds-2", "embedding": [0.1, 0.9]},
        ],
        model="fake-test-model",
    )
    hits, total = storage.search_embedding([1.0, 0.0], limit=1, offset=0)
    assert total == 3
    assert len(hits) == 1
    assert hits[0]["id"] == "acme/ds-0"

    hits_page2, _ = storage.search_embedding([1.0, 0.0], limit=1, offset=1)
    assert hits_page2[0]["id"] == "acme/ds-1"


def test_embedding_index_status_reports_counts(tmp_path: Path) -> None:
    storage = _make_storage(tmp_path)
    _seed_datasets(storage)

    status_before = storage.embedding_index_status()
    assert status_before["total_datasets"] == 3
    assert status_before["embedded_count"] == 0
    assert status_before["model"] is None

    storage.upsert_dataset_embeddings(
        [{"dataset_id": "acme/ds-0", "embedding": [1.0, 0.0]}],
        model="fake-test-model",
    )
    status_after = storage.embedding_index_status()
    assert status_after["embedded_count"] == 1
    assert status_after["model"] == "fake-test-model"
