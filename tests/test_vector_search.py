from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.storage.duckdb_backend import DuckDBStorage
from src.storage.models import Dataset


def _make_storage(tmp_path: Path, name: str) -> DuckDBStorage:
    storage = DuckDBStorage(str(tmp_path / name))
    storage.init()
    return storage


def _seed(storage: DuckDBStorage) -> None:
    now = datetime.utcnow()
    licenses = ["mit", "apache-2.0", "mit"]
    storage.bulk_upsert_datasets(
        [
            Dataset(
                id=f"acme/ds-{i}",
                source="hf",
                ns_local_id=f"ds-{i}",
                title=f"Dataset {i}",
                description=f"Description {i}",
                license_class=licenses[i],
                fingerprint=f"fp{i}",
                indexed_at=now,
            )
            for i in range(3)
        ]
    )


def test_cache_refreshes_when_vectors_change(tmp_path: Path) -> None:
    storage = _make_storage(tmp_path, "refresh.duckdb")
    _seed(storage)
    storage.upsert_dataset_embeddings(
        [
            {"dataset_id": "acme/ds-0", "embedding": [1.0, 0.0]},
            {"dataset_id": "acme/ds-1", "embedding": [0.0, 1.0]},
            {"dataset_id": "acme/ds-2", "embedding": [0.0, 1.0]},
        ],
        model="test",
    )
    hits, _ = storage.search_embedding([1.0, 0.0], limit=1)
    assert hits[0]["id"] == "acme/ds-0"

    # Change ds-2 to match the query exactly. The cached matrix must notice
    # and reload, or this would still rank ds-0 first.
    storage.upsert_dataset_embeddings(
        [{"dataset_id": "acme/ds-2", "embedding": [1.0, 0.0]}],
        model="test",
    )
    hits_after, _ = storage.search_embedding([1.0, 0.0], limit=2)
    assert {h["id"] for h in hits_after} == {"acme/ds-0", "acme/ds-2"}


def test_filters_restrict_results_to_matching_datasets(tmp_path: Path) -> None:
    storage = _make_storage(tmp_path, "filters.duckdb")
    _seed(storage)
    storage.upsert_dataset_embeddings(
        [
            {"dataset_id": "acme/ds-0", "embedding": [1.0, 0.0]},
            {"dataset_id": "acme/ds-1", "embedding": [0.9, 0.1]},
            {"dataset_id": "acme/ds-2", "embedding": [0.8, 0.2]},
        ],
        model="test",
    )
    hits, total = storage.search_embedding(
        [1.0, 0.0], filters={"license_class": "apache-2.0"}, limit=10
    )
    assert total == 1
    assert [h["id"] for h in hits] == ["acme/ds-1"]


def test_search_returns_scores_in_descending_order(tmp_path: Path) -> None:
    storage = _make_storage(tmp_path, "order.duckdb")
    _seed(storage)
    storage.upsert_dataset_embeddings(
        [
            {"dataset_id": "acme/ds-0", "embedding": [0.0, 1.0]},
            {"dataset_id": "acme/ds-1", "embedding": [1.0, 0.0]},
            {"dataset_id": "acme/ds-2", "embedding": [0.6, 0.8]},
        ],
        model="test",
    )
    hits, _ = storage.search_embedding([1.0, 0.0], limit=3)
    scores = [h["score"] for h in hits]
    assert scores == sorted(scores, reverse=True)
    assert hits[0]["id"] == "acme/ds-1"


def test_migration_converts_legacy_json_rows(tmp_path: Path) -> None:
    from scripts.migrate_embeddings import migrate

    storage = _make_storage(tmp_path, "legacy.duckdb")
    _seed(storage)
    # Recreate the legacy layout: vectors as JSON text in dataset_embeddings.
    storage.conn.execute(
        """
        CREATE TABLE dataset_embeddings(
            dataset_id TEXT PRIMARY KEY, model TEXT, embedding TEXT, updated_at TIMESTAMP
        )
        """
    )
    legacy = {
        "acme/ds-0": [1.0, 0.0],
        "acme/ds-1": [0.0, 1.0],
        "acme/ds-2": [0.6, 0.8],
    }
    for did, vec in legacy.items():
        storage.conn.execute(
            "INSERT INTO dataset_embeddings VALUES (?, ?, ?, ?)",
            [did, "test", json.dumps(vec), datetime.utcnow()],
        )

    migrated = migrate(str(tmp_path / "legacy.duckdb"))
    assert migrated == 3

    reopened = DuckDBStorage(str(tmp_path / "legacy.duckdb"))
    reopened.init()
    legacy_left = reopened.conn.execute(
        "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'dataset_embeddings'"
    ).fetchone()[0]
    assert legacy_left == 0
    hits, _ = reopened.search_embedding([1.0, 0.0], limit=1)
    assert hits[0]["id"] == "acme/ds-0"


def test_migration_is_a_noop_without_legacy_table(tmp_path: Path) -> None:
    from scripts.migrate_embeddings import migrate

    storage = _make_storage(tmp_path, "nolegacy.duckdb")
    _seed(storage)
    assert migrate(str(tmp_path / "nolegacy.duckdb")) == 0
