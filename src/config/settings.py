import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv


@dataclass
class Settings:
    """Runtime configuration for the dataset discovery services."""

    storage_backend: str = "duckdb"  # duckdb | postgres
    duckdb_path: str = os.path.join("data", "discovery.duckdb")
    postgres_dsn: Optional[str] = None
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    worker_poll_interval_sec: float = 1.0
    worker_lease_timeout_sec: float = 30.0
    resolve_budget_ms_default: int = 1500
    resolve_row_cap_default: int = 200
    prefetch_enabled: bool = True
    resolve_p95_max_ms: int = 3000
    queue_depth_max_multiplier: float = 2.0  # relative to hourly arrivals
    queue_depth_absolute_max: int = 500
    license_allowlist: tuple[str, ...] = (
        "apache-2.0",
        "mit",
        "bsd-3-clause",
        "bsd-2-clause",
        "cc-by-4.0",
        "cc-by-sa-4.0",
        "cc0-1.0",
        "openrail",
        "cdla-permissive-2.0",
        "odc-by",
        "odc-odbl",
    )
    max_size_class: Optional[str] = None  # optional ceiling like "10M<n<100M"
    es_url: str = "http://localhost:9200"
    es_rows_index: str = "hf_dataset_rows"

    def duckdb_uri(self) -> str:
        return self.duckdb_path


def load_settings() -> Settings:
    """Load settings from environment (dotenv respected)."""

    load_dotenv()
    return Settings(
        storage_backend=os.getenv("STORAGE_BACKEND", "duckdb"),
        duckdb_path=os.getenv("DUCKDB_PATH", os.path.join("data", "discovery.duckdb")),
        postgres_dsn=os.getenv("POSTGRES_DSN"),
        api_host=os.getenv("API_HOST", "0.0.0.0"),
        api_port=int(os.getenv("API_PORT", "8000")),
        worker_poll_interval_sec=float(os.getenv("WORKER_POLL_INTERVAL_SEC", "1.0")),
        worker_lease_timeout_sec=float(os.getenv("WORKER_LEASE_TIMEOUT_SEC", "30.0")),
        resolve_budget_ms_default=int(os.getenv("RESOLVE_BUDGET_MS_DEFAULT", "1500")),
        resolve_row_cap_default=int(os.getenv("RESOLVE_ROW_CAP_DEFAULT", "200")),
        prefetch_enabled=os.getenv("PREFETCH_ENABLED", "true").lower() == "true",
        resolve_p95_max_ms=int(os.getenv("RESOLVE_P95_MAX_MS", "3000")),
        queue_depth_max_multiplier=float(
            os.getenv("QUEUE_DEPTH_MAX_MULTIPLIER", "2.0")
        ),
        queue_depth_absolute_max=int(os.getenv("QUEUE_DEPTH_ABSOLUTE_MAX", "500")),
        license_allowlist=tuple(
            os.getenv(
                "LICENSE_ALLOWLIST",
                ",".join(
                    [
                        "apache-2.0",
                        "mit",
                        "bsd-3-clause",
                        "bsd-2-clause",
                        "cc-by-4.0",
                        "cc-by-sa-4.0",
                        "cc0-1.0",
                        "openrail",
                        "cdla-permissive-2.0",
                        "odc-by",
                        "odc-odbl",
                    ]
                ),
            ).split(",")
        ),
        max_size_class=os.getenv("MAX_SIZE_CLASS"),
        es_url=os.getenv("ES_URL", "http://localhost:9200"),
        es_rows_index=os.getenv("ES_ROWS_INDEX", "hf_dataset_rows"),
    )
