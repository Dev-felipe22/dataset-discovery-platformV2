from __future__ import annotations

import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.demo_data import build_demo_db
from scripts.rebuild_embeddings import rebuild
from src.storage.duckdb_backend import DuckDBStorage

FIXTURE_PATH = Path("tests/fixtures/demo_catalog.json")


def _fake_embed_texts(texts: List[str], *, batch_size: int = 64) -> List[List[float]]:
    return [[float(i), 0.0] for i in range(len(texts))]


def test_rebuild_embeds_every_dataset_and_saves_immediately(tmp_path, monkeypatch) -> None:
    db_path = tmp_path / "rebuild_script.duckdb"
    build_demo_db(db_path, FIXTURE_PATH)

    from src.tools import embeddings

    monkeypatch.setattr(embeddings, "embed_texts", _fake_embed_texts)

    embedded_count = rebuild(str(db_path))
    assert embedded_count >= 1

    # Re-open the DB fresh (new connection) to prove the write was actually
    # persisted to disk, not just held in the first connection's memory.
    storage = DuckDBStorage(str(db_path))
    storage.init()
    status = storage.embedding_index_status()
    assert status["embedded_count"] == embedded_count
    assert status["embedded_count"] == status["total_datasets"]
    assert status["model"] == embeddings.EMBEDDING_MODEL_NAME


def test_rebuild_is_a_no_op_report_on_empty_catalog(tmp_path, monkeypatch) -> None:
    db_path = tmp_path / "empty.duckdb"
    storage = DuckDBStorage(str(db_path))
    storage.init()

    from src.tools import embeddings

    monkeypatch.setattr(embeddings, "embed_texts", _fake_embed_texts)

    assert rebuild(str(db_path)) == 0


def test_limit_embeds_only_the_first_n_datasets(tmp_path, monkeypatch) -> None:
    db_path = tmp_path / "limited.duckdb"
    build_demo_db(db_path, FIXTURE_PATH)  # 3 datasets in this fixture

    from src.tools import embeddings

    monkeypatch.setattr(embeddings, "embed_texts", _fake_embed_texts)

    embedded_count = rebuild(str(db_path), limit=2)
    assert embedded_count == 2

    storage = DuckDBStorage(str(db_path))
    storage.init()
    status = storage.embedding_index_status()
    assert status["embedded_count"] == 2
    assert status["total_datasets"] == 3  # limit affects embedding, not the catalog itself


def test_small_chunk_size_still_embeds_every_dataset(tmp_path, monkeypatch) -> None:
    # chunk_size=1 forces 3 separate embed+upsert passes over the 3-dataset
    # fixture — proves the chunking loop doesn't drop or duplicate rows.
    db_path = tmp_path / "chunked.duckdb"
    build_demo_db(db_path, FIXTURE_PATH)

    from src.tools import embeddings

    monkeypatch.setattr(embeddings, "embed_texts", _fake_embed_texts)

    embedded_count = rebuild(str(db_path), chunk_size=1)
    assert embedded_count == 3

    storage = DuckDBStorage(str(db_path))
    storage.init()
    status = storage.embedding_index_status()
    assert status["embedded_count"] == 3


def test_rerun_resumes_and_skips_already_embedded(tmp_path, monkeypatch) -> None:
    # Simulates exactly what happened in practice: a run gets interrupted
    # partway through, and a second run should pick up where it left off
    # instead of redoing already-completed work.
    db_path = tmp_path / "resume.duckdb"
    build_demo_db(db_path, FIXTURE_PATH)  # 3 datasets

    from src.tools import embeddings

    calls = []

    def _counting_embed_texts(texts, *, batch_size=64):
        calls.append(len(texts))
        return _fake_embed_texts(texts, batch_size=batch_size)

    monkeypatch.setattr(embeddings, "embed_texts", _counting_embed_texts)

    # First run only gets through 1 of 3 (simulating an interruption via --limit).
    first = rebuild(str(db_path), limit=1)
    assert first == 1
    assert calls == [1]

    # Second run, no --force: should only embed the remaining 2, not redo the first.
    second = rebuild(str(db_path))
    assert second == 2
    assert calls == [1, 2]

    storage = DuckDBStorage(str(db_path))
    storage.init()
    status = storage.embedding_index_status()
    assert status["embedded_count"] == 3


def test_force_redoes_already_embedded_datasets(tmp_path, monkeypatch) -> None:
    db_path = tmp_path / "forced.duckdb"
    build_demo_db(db_path, FIXTURE_PATH)  # 3 datasets

    from src.tools import embeddings

    monkeypatch.setattr(embeddings, "embed_texts", _fake_embed_texts)

    assert rebuild(str(db_path)) == 3
    # Without --force, a rerun has nothing left to do.
    assert rebuild(str(db_path)) == 0
    # With --force, all 3 are recomputed even though already embedded.
    assert rebuild(str(db_path), force=True) == 3
