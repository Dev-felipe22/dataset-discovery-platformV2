from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from .base import StorageAdapter
from .models import Artifact, Dataset, DatasetID, Job


class PostgresStorage(StorageAdapter):
    """Postgres adapter skeleton for Option C (not implemented in MVP)."""

    def __init__(self, dsn: str):
        self.dsn = dsn

    def init(self) -> None:
        raise NotImplementedError("Postgres adapter will be implemented post-MVP.")

    # Datasets
    def upsert_dataset(self, dataset: Dataset) -> None:
        raise NotImplementedError

    def bulk_upsert_datasets(self, datasets: List[Dataset]) -> None:
        raise NotImplementedError

    def mark_artifacts_stale_on_fingerprint_change(
        self, dataset_id: DatasetID, new_fingerprint: str
    ) -> None:
        raise NotImplementedError

    def refresh_dataset_state_view(self) -> None:
        raise NotImplementedError

    def rebuild_search_index(self) -> None:
        raise NotImplementedError

    def search_bm25(
        self,
        query: str,
        *,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 30,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def get_dataset(self, dataset_id: DatasetID) -> Optional[Dataset]:
        raise NotImplementedError

    # Artifacts
    def get_artifact(self, dataset_id: DatasetID, kind: str) -> Optional[Artifact]:
        raise NotImplementedError

    def put_artifact(self, artifact: Artifact) -> None:
        raise NotImplementedError

    # Jobs
    def enqueue_job(self, job: Job) -> None:
        raise NotImplementedError

    def claim_next_job(self) -> Optional[Job]:
        raise NotImplementedError

    def mark_job_done(self, job_id: str) -> None:
        raise NotImplementedError

    def mark_job_error(
        self, job_id: str, *, error_class: str, error_detail: str
    ) -> None:
        raise NotImplementedError

    def job_metrics(self) -> Dict[str, Any]:
        raise NotImplementedError

    def reclaim_stale_jobs(self, lease_timeout_sec: float, max_attempts: int = 3) -> int:
        raise NotImplementedError

    # Signals / events
    def log_event(self, event: Dict[str, Any]) -> None:
        raise NotImplementedError

    def log_signal(self, signal: Dict[str, Any]) -> None:
        raise NotImplementedError

    def admin_metrics(self) -> Dict[str, Any]:
        raise NotImplementedError

    def get_connector_cursor(self, source: str) -> Optional[Dict[str, Any]]:
        raise NotImplementedError

    def upsert_connector(
        self,
        source: str,
        *,
        cursor: Optional[Dict[str, Any]],
        qps_limit: Optional[int] = None,
        backoff_state: Optional[Dict[str, Any]] = None,
    ) -> None:
        raise NotImplementedError

    def prefetch_candidates(self, limit: int):
        raise NotImplementedError

    def eval_list_queries(self, *, active_only: bool = True) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def eval_upsert_query(self, query: Dict[str, Any]) -> None:
        raise NotImplementedError

    def eval_set_query_active(self, query_id: str, active: bool) -> None:
        raise NotImplementedError

    def eval_create_run(
        self, run_id: str, *, label: str, engine: str, k: int, created_at: datetime
    ) -> None:
        raise NotImplementedError

    def eval_add_run_results(self, run_id: str, query_id: str, rows: List[Dict[str, Any]]) -> None:
        raise NotImplementedError

    def eval_list_runs(self) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def eval_get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        raise NotImplementedError

    def eval_get_run_results(self, run_id: str) -> Dict[str, List[Dict[str, Any]]]:
        raise NotImplementedError

    def eval_upsert_judgment(
        self, query_id: str, dataset_id: str, relevant: bool, judged_by: Optional[str] = None
    ) -> None:
        raise NotImplementedError
