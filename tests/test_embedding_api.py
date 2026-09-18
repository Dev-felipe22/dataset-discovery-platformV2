from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.demo_data import build_demo_db

FIXTURE_PATH = Path("tests/fixtures/demo_catalog.json")


def _fake_embed_texts(texts: List[str]) -> List[List[float]]:
    """Deterministic stand-in for the real MiniLM model: hashes text into a
    small vector so tests don't need network access or torch installed.
    Keeps the same [text] -> [[float, ...]] contract as the real function."""
    import hashlib

    vectors = []
    for t in texts:
        h = hashlib.sha256((t or "").encode("utf-8")).digest()
        vec = [b / 255.0 for b in h[:8]]
        norm = sum(v * v for v in vec) ** 0.5 or 1.0
        vectors.append([v / norm for v in vec])
    return vectors


@pytest.fixture(autouse=True)
def _stub_embedding_model(monkeypatch):
    from src.tools import embeddings

    monkeypatch.setattr(embeddings, "embed_texts", _fake_embed_texts)
    monkeypatch.setattr(embeddings, "embed_query", lambda text: _fake_embed_texts([text])[0])
    yield


def _create_client(tmp_path: Path) -> TestClient:
    db_path = tmp_path / "embedding_api.duckdb"
    build_demo_db(db_path, FIXTURE_PATH)
    os.environ["DUCKDB_PATH"] = str(db_path)
    os.environ["STORAGE_BACKEND"] = "duckdb"
    os.environ["PREFETCH_ENABLED"] = "true"

    from src.api.deps import _get_settings_cached

    _get_settings_cached.cache_clear()

    from src.api.main import create_app

    return TestClient(create_app())


def test_embedding_status_reports_zero_before_rebuild(tmp_path: Path) -> None:
    with _create_client(tmp_path) as client:
        resp = client.get("/v2/embedding_status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["embedded_count"] == 0
        assert data["total_datasets"] >= 1


def test_rebuild_embeddings_then_search(tmp_path: Path) -> None:
    with _create_client(tmp_path) as client:
        rebuild = client.post("/v2/admin/rebuild_embeddings")
        assert rebuild.status_code == 200
        body = rebuild.json()
        assert body["status"] == "ok"
        assert body["embedded"] >= 1

        status = client.get("/v2/embedding_status").json()
        assert status["embedded_count"] == status["total_datasets"]

        search = client.post(
            "/v2/search_embedding",
            json={"query": "text classification", "limit": 5},
        )
        assert search.status_code == 200
        payload = search.json()
        assert payload["total"] >= 1
        assert payload["hits"][0]["score"] is not None


def test_search_index_now_also_reports_a_score(tmp_path: Path) -> None:
    # score was added to SearchHit so BM25 and dense results are comparable
    # side by side in the UI.
    with _create_client(tmp_path) as client:
        resp = client.post(
            "/v2/search_index",
            json={"query": "text classification", "limit": 5},
        )
        assert resp.status_code == 200
        assert resp.json()["hits"][0]["score"] is not None


def test_minilm_engine_is_registered_for_benchmarking(tmp_path: Path) -> None:
    with _create_client(tmp_path) as client:
        engines = client.get("/v2/eval/engines").json()["engines"]
        assert "bm25" in engines
        assert "minilm" in engines

        client.post("/v2/admin/rebuild_embeddings")
        client.post(
            "/v2/eval/queries",
            json={"id": "E1", "label": "embedding smoke", "query": "text classification", "total_relevant": 1},
        )
        run = client.post(
            "/v2/eval/run",
            json={"label": "minilm smoke", "engine": "minilm", "k": 5},
        )
        assert run.status_code == 200
        e1 = next(q for q in run.json()["queries"] if q["query_id"] == "E1")
        assert e1["results"], "expected the minilm engine to return results too"
