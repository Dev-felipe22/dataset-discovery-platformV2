from __future__ import annotations

import os
import sys
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.demo_data import build_demo_db

FIXTURE_PATH = Path("tests/fixtures/demo_catalog.json")


def _create_client(tmp_path: Path) -> TestClient:
    db_path = tmp_path / "eval_smoke.duckdb"
    build_demo_db(db_path, FIXTURE_PATH)
    os.environ["DUCKDB_PATH"] = str(db_path)
    os.environ["STORAGE_BACKEND"] = "duckdb"
    os.environ["PREFETCH_ENABLED"] = "true"

    from src.api.deps import _get_settings_cached

    _get_settings_cached.cache_clear()

    from src.api.main import create_app

    return TestClient(create_app())


def test_eval_queries_are_seeded(tmp_path: Path) -> None:
    with _create_client(tmp_path) as client:
        response = client.get("/v2/eval/queries")
        assert response.status_code == 200
        ids = {q["id"] for q in response.json()}
        assert {"Q1", "Q2", "Q3"}.issubset(ids)


def test_eval_engines_lists_bm25(tmp_path: Path) -> None:
    with _create_client(tmp_path) as client:
        response = client.get("/v2/eval/engines")
        assert response.status_code == 200
        assert "bm25" in response.json()["engines"]


def test_run_judge_and_export_round_trip(tmp_path: Path) -> None:
    with _create_client(tmp_path) as client:
        # A benchmark query matched against the seeded demo catalog rather
        # than the fictional seed queries (which target a dataset that
        # doesn't exist in this fixture).
        upsert = client.post(
            "/v2/eval/queries",
            json={
                "id": "T1",
                "label": "smoke test query",
                "query": "text classification",
                "total_relevant": 1,
            },
        )
        assert upsert.status_code == 200

        run_resp = client.post(
            "/v2/eval/run",
            json={"label": "smoke run", "engine": "bm25", "k": 10},
        )
        assert run_resp.status_code == 200
        run_payload = run_resp.json()
        run_id = run_payload["run"]["id"]

        t1 = next(q for q in run_payload["queries"] if q["query_id"] == "T1")
        assert t1["results"], "expected the seeded acme dataset to be retrieved"
        top_hit = t1["results"][0]
        assert top_hit["dataset_id"] == "acme/text-classification-demo"
        # Unjudged yet, so it should not count as relevant.
        assert t1["metrics"]["recall"] == 0.0

        judge_resp = client.post(
            "/v2/eval/judge",
            json={"query_id": "T1", "dataset_id": top_hit["dataset_id"], "relevant": True},
        )
        assert judge_resp.status_code == 204

        detail_resp = client.get(f"/v2/eval/runs/{run_id}")
        assert detail_resp.status_code == 200
        t1_after = next(q for q in detail_resp.json()["queries"] if q["query_id"] == "T1")
        assert t1_after["metrics"]["recall"] == 1.0
        assert t1_after["metrics"]["mrr"] == 1.0

        runs_resp = client.get("/v2/eval/runs")
        assert runs_resp.status_code == 200
        summaries = {r["id"]: r for r in runs_resp.json()["runs"]}
        assert summaries[run_id]["map"] > 0.0

        export_resp = client.get(f"/v2/eval/runs/{run_id}/export")
        assert export_resp.status_code == 200
        assert "spreadsheetml" in export_resp.headers["content-type"]
        assert len(export_resp.content) > 1000  # a real xlsx, not an empty stub
