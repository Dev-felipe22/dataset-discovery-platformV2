from __future__ import annotations

import abc
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from .models import Artifact, Dataset, DatasetID, Job


class StorageAdapter(abc.ABC):
    """Abstract storage adapter; duckdb/postgres should implement."""

    @abc.abstractmethod
    def init(self) -> None:
        """Create tables/indexes/materialized views if missing."""

    # Datasets
    @abc.abstractmethod
    def upsert_dataset(self, dataset: Dataset) -> None:
        ...

    @abc.abstractmethod
    def bulk_upsert_datasets(self, datasets: List[Dataset]) -> None:
        ...

    @abc.abstractmethod
    def mark_artifacts_stale_on_fingerprint_change(
        self, dataset_id: DatasetID, new_fingerprint: str
    ) -> None:
        ...

    @abc.abstractmethod
    def refresh_dataset_state_view(self) -> None:
        ...

    @abc.abstractmethod
    def rebuild_search_index(self) -> None:
        ...

    @abc.abstractmethod
    def search_bm25(
        self,
        query: str,
        *,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 30,
        offset: int = 0,
    ) -> Tuple[List[Dict[str, Any]], int]:
        ...

    @abc.abstractmethod
    def get_dataset(self, dataset_id: DatasetID) -> Optional[Dataset]:
        ...

    # Artifacts
    @abc.abstractmethod
    def get_artifact(self, dataset_id: DatasetID, kind: str) -> Optional[Artifact]:
        ...

    @abc.abstractmethod
    def put_artifact(self, artifact: Artifact) -> None:
        ...

    # Jobs
    @abc.abstractmethod
    def enqueue_job(self, job: Job) -> None:
        ...

    @abc.abstractmethod
    def claim_next_job(self) -> Optional[Job]:
        ...

    @abc.abstractmethod
    def mark_job_done(self, job_id: str) -> None:
        ...

    @abc.abstractmethod
    def mark_job_error(
        self, job_id: str, *, error_class: str, error_detail: str
    ) -> None:
        ...

    @abc.abstractmethod
    def job_metrics(self) -> Dict[str, Any]:
        ...

    @abc.abstractmethod
    def reclaim_stale_jobs(self, lease_timeout_sec: float, max_attempts: int = 3) -> int:
        """Requeue jobs stuck in state='running' beyond the lease timeout."""

    # Signals / events
    @abc.abstractmethod
    def log_event(self, event: Dict[str, Any]) -> None:
        ...

    @abc.abstractmethod
    def log_signal(self, signal: Dict[str, Any]) -> None:
        ...

    @abc.abstractmethod
    def admin_metrics(self) -> Dict[str, Any]:
        ...

    # Connectors (discovery cursors)
    @abc.abstractmethod
    def get_connector_cursor(self, source: str) -> Optional[Dict[str, Any]]:
        ...

    @abc.abstractmethod
    def upsert_connector(
        self,
        source: str,
        *,
        cursor: Optional[Dict[str, Any]],
        qps_limit: Optional[int] = None,
        backoff_state: Optional[Dict[str, Any]] = None,
    ) -> None:
        ...

    # Prefetch helpers
    @abc.abstractmethod
    def prefetch_candidates(self, limit: int) -> List[DatasetID]:
        ...

    # Search-quality evaluation (benchmark queries, judgments, runs)
    @abc.abstractmethod
    def eval_list_queries(self, *, active_only: bool = True) -> List[Dict[str, Any]]:
        ...

    @abc.abstractmethod
    def eval_upsert_query(self, query: Dict[str, Any]) -> None:
        ...

    @abc.abstractmethod
    def eval_set_query_active(self, query_id: str, active: bool) -> None:
        ...

    @abc.abstractmethod
    def eval_create_run(
        self, run_id: str, *, label: str, engine: str, k: int, created_at: datetime
    ) -> None:
        ...

    @abc.abstractmethod
    def eval_add_run_results(
        self, run_id: str, query_id: str, rows: List[Dict[str, Any]]
    ) -> None:
        ...

    @abc.abstractmethod
    def eval_list_runs(self) -> List[Dict[str, Any]]:
        ...

    @abc.abstractmethod
    def eval_get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        ...

    @abc.abstractmethod
    def eval_get_run_results(self, run_id: str) -> Dict[str, List[Dict[str, Any]]]:
        ...

    @abc.abstractmethod
    def eval_upsert_judgment(
        self, query_id: str, dataset_id: str, relevant: bool, judged_by: Optional[str] = None
    ) -> None:
        ...
