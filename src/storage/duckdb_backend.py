from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import duckdb

from .base import StorageAdapter
from .models import Artifact, Dataset, DatasetID, Job


class DuckDBStorage(StorageAdapter):
    """DuckDB-backed storage adapter (Option A)."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = duckdb.connect(database=db_path, read_only=False)
        self._fts_available = False
        self._fts_warned = False
        self._search_index_dirty = False

    # Helpers
    @staticmethod
    def _json_dump(value: Any) -> Optional[str]:
        if value is None:
            return None
        return json.dumps(value, ensure_ascii=False)

    @staticmethod
    def _json_load(value: Optional[str]) -> Any:
        if value in (None, ""):
            return None
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return None

    @staticmethod
    def _dataset_columns() -> List[str]:
        return [
            "id",
            "source",
            "ns_local_id",
            "title",
            "description",
            "readme_text",
            "tags",
            "modalities",
            "license_class",
            "access_class",
            "size_class",
            "languages",
            "last_seen_lm",
            "readme_sha256",
            "card_sha256",
            "fingerprint",
            "indexed_at",
            "readme_tokens",
            "readme_score",
            "eligibility_flags",
            "quality_signals",
            "schema_hint",
        ]

    def init(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS datasets(
                id TEXT PRIMARY KEY,
                source TEXT,
                ns_local_id TEXT,
                title TEXT,
                description TEXT,
                readme_text TEXT,
                tags TEXT,
                modalities TEXT,
                license_class TEXT,
                access_class TEXT,
                size_class TEXT,
                languages TEXT,
                last_seen_lm TIMESTAMP,
                readme_sha256 TEXT,
                card_sha256 TEXT,
                fingerprint TEXT,
                indexed_at TIMESTAMP,
                readme_tokens INTEGER,
                readme_score DOUBLE,
                eligibility_flags TEXT,
                quality_signals TEXT,
                schema_hint TEXT
            )
            """
        )
        cols = {row[1] for row in self.conn.execute("PRAGMA table_info('datasets')").fetchall()}
        if "readme_text" not in cols:
            self.conn.execute("ALTER TABLE datasets ADD COLUMN readme_text TEXT")

        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS artifacts(
                dataset_id TEXT,
                kind TEXT,
                payload TEXT,
                created_at TIMESTAMP,
                stale BOOLEAN DEFAULT FALSE,
                PRIMARY KEY(dataset_id, kind)
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS jobs(
                id TEXT PRIMARY KEY,
                dataset_id TEXT,
                kind TEXT,
                state TEXT,
                attempts INT,
                error_class TEXT,
                error_detail TEXT,
                budget_ms INT,
                row_cap INT,
                enqueued_at TIMESTAMP,
                started_at TIMESTAMP,
                finished_at TIMESTAMP
            )
            """
        )
        cols_jobs = {row[1] for row in self.conn.execute("PRAGMA table_info('jobs')").fetchall()}
        if "budget_ms" not in cols_jobs:
            self.conn.execute("ALTER TABLE jobs ADD COLUMN budget_ms INT")
        if "row_cap" not in cols_jobs:
            self.conn.execute("ALTER TABLE jobs ADD COLUMN row_cap INT")

        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS events(
                id TEXT PRIMARY KEY,
                type TEXT,
                dataset_id TEXT,
                payload TEXT,
                created_at TIMESTAMP
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS signals(
                id TEXT PRIMARY KEY,
                session_id TEXT,
                user_id TEXT,
                event TEXT,
                dataset_id TEXT,
                query TEXT,
                rank INT,
                created_at TIMESTAMP
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS connectors(
                source TEXT PRIMARY KEY,
                cursor TEXT,
                qps_limit INT,
                backoff_state TEXT
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS dataset_state_mv(
                id TEXT PRIMARY KEY,
                has_schema BOOLEAN,
                has_snips BOOLEAN,
                recent_errors BOOLEAN,
                last_refreshed_at TIMESTAMP
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS dataset_search(
                docid INTEGER PRIMARY KEY,
                dataset_id TEXT UNIQUE,
                content TEXT
            )
            """
        )
        self._init_eval_tables()
        self._ensure_fts_loaded()

    def _init_eval_tables(self) -> None:
        """Search-quality benchmark harness tables: a reusable query set,
        persistent relevance judgments (survive across runs, keyed by
        query+dataset so re-judging isn't needed every run), and the
        run/result history used to trend metrics over time."""
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS eval_queries(
                id TEXT PRIMARY KEY,
                label TEXT,
                query TEXT,
                target_id TEXT,
                total_relevant INT DEFAULT 1,
                active BOOLEAN DEFAULT TRUE,
                created_at TIMESTAMP
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS eval_relevance_judgments(
                query_id TEXT,
                dataset_id TEXT,
                relevant BOOLEAN,
                judged_by TEXT,
                judged_at TIMESTAMP,
                PRIMARY KEY(query_id, dataset_id)
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS eval_runs(
                id TEXT PRIMARY KEY,
                label TEXT,
                engine TEXT,
                k INT,
                created_at TIMESTAMP
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS eval_run_results(
                run_id TEXT,
                query_id TEXT,
                result_rank INT,
                dataset_id TEXT,
                title TEXT,
                extra TEXT,
                PRIMARY KEY(run_id, query_id, result_rank)
            )
            """
        )
        seeded = self.conn.execute("SELECT COUNT(*) FROM eval_queries").fetchone()[0]
        if seeded == 0:
            now = datetime.utcnow()
            # Seeds the benchmark set that this app's search-eval harness was
            # ported from (hf-queries.py) so it's usable immediately.
            seed = [
                ("Q1", "direct entity match", "teleportation health survey",
                 "NeuralNebula/teleportation-health-survey", 1),
                ("Q2", "structural, fiction removed", "symptom survey frequency",
                 "NeuralNebula/teleportation-health-survey", 1),
                ("Q3", "column level", "nausea dizziness questionnaire respondents",
                 "NeuralNebula/teleportation-health-survey", 1),
            ]
            for qid, label, qtext, target, total_rel in seed:
                self.conn.execute(
                    """
                    INSERT INTO eval_queries (id, label, query, target_id, total_relevant, active, created_at)
                    VALUES (?, ?, ?, ?, ?, TRUE, ?)
                    """,
                    [qid, label, qtext, target, total_rel, now],
                )

    # Datasets
    def upsert_dataset(self, dataset: Dataset) -> None:
        self.bulk_upsert_datasets([dataset])

    def bulk_upsert_datasets(self, datasets: List[Dataset]) -> None:
        if not datasets:
            return
        cols = self._dataset_columns()
        existing: Dict[str, Optional[str]] = {}
        ids = [d.id for d in datasets]
        if ids:
            placeholders = ",".join(["?"] * len(ids))
            rows = self.conn.execute(
                f"SELECT id, fingerprint FROM datasets WHERE id IN ({placeholders})",
                ids,
            ).fetchall()
            existing = {row[0]: row[1] for row in rows}

        values = []
        events: List[Dict[str, Any]] = []
        for d in datasets:
            old_fp = existing.get(d.id)
            if old_fp is not None and old_fp != d.fingerprint:
                self.mark_artifacts_stale_on_fingerprint_change(d.id, d.fingerprint or "")
                events.append(
                    {
                        "type": "dataset.changed",
                        "dataset_id": d.id,
                        "fingerprint_old": old_fp,
                        "fingerprint_new": d.fingerprint,
                        "created_at": datetime.utcnow(),
                    }
                )
            if old_fp is None:
                events.append(
                    {
                        "type": "dataset.discovered",
                        "dataset_id": d.id,
                        "fingerprint_new": d.fingerprint,
                        "created_at": datetime.utcnow(),
                    }
                )
            values.append(
                (
                    d.id,
                    d.source,
                    d.ns_local_id,
                    d.title,
                    d.description,
                    d.readme_text,
                    self._json_dump(d.tags),
                    self._json_dump(d.modalities),
                    d.license_class,
                    d.access_class,
                    d.size_class,
                    self._json_dump(d.languages),
                    d.last_seen_lm,
                    d.readme_sha256,
                    d.card_sha256,
                    d.fingerprint,
                    d.indexed_at,
                    d.readme_tokens,
                    d.readme_score,
                    self._json_dump(d.eligibility_flags),
                    self._json_dump(d.quality_signals),
                    d.schema_hint,
                )
            )
        placeholders = ",".join(["?"] * len(cols))
        assignments = ",".join([f"{c}=excluded.{c}" for c in cols[1:]])
        sql = f"""
        INSERT INTO datasets ({",".join(cols)})
        VALUES ({placeholders})
        ON CONFLICT(id) DO UPDATE SET {assignments}
        """
        self.conn.executemany(sql, values)
        self._search_index_dirty = True
        for ev in events:
            self.log_event(ev)

    def mark_artifacts_stale_on_fingerprint_change(
        self, dataset_id: DatasetID, new_fingerprint: str
    ) -> None:
        row = self.conn.execute(
            "SELECT fingerprint FROM datasets WHERE id = ?", [dataset_id]
        ).fetchone()
        if row and row[0] == new_fingerprint:
            return
        self.conn.execute(
            "UPDATE artifacts SET stale = TRUE WHERE dataset_id = ?", [dataset_id]
        )
        self._search_index_dirty = True

    def refresh_dataset_state_view(self) -> None:
        self.conn.execute(
            """
            CREATE OR REPLACE TABLE dataset_state_mv AS
            SELECT d.id,
                   MAX(CASE WHEN a.kind='schema' AND NOT a.stale THEN 1 ELSE 0 END) AS has_schema,
                   MAX(CASE WHEN a.kind='snips'  AND NOT a.stale THEN 1 ELSE 0 END) AS has_snips,
                   MAX(CASE WHEN j.state='error' THEN 1 ELSE 0 END) AS recent_errors,
                   NOW() AS last_refreshed_at
            FROM datasets d
            LEFT JOIN artifacts a ON a.dataset_id = d.id
            LEFT JOIN jobs j       ON j.dataset_id = d.id AND j.enqueued_at > NOW() - INTERVAL 7 DAY
            GROUP BY 1
            """
        )
        self._search_index_dirty = True

    def _ensure_fts_loaded(self) -> None:
        if self._fts_available:
            return
        try:
            self.conn.execute("INSTALL fts")
            self.conn.execute("LOAD fts")
            self._fts_available = True
        except Exception:
            self._fts_available = False
            if not self._fts_warned:
                import logging
                logging.getLogger(__name__).warning(
                    "DuckDB FTS extension unavailable; falling back to LIKE-based search."
                )
                self._fts_warned = True

    def rebuild_search_index(self) -> None:
        self._ensure_fts_loaded()
        self.conn.execute("DELETE FROM dataset_search")
        self.conn.execute(
            """
            INSERT INTO dataset_search (docid, dataset_id, content)
            SELECT
                row_number() OVER () AS docid,
                id AS dataset_id,
                lower(concat_ws(
                    ' ',
                    coalesce(title, ''),
                    coalesce(description, ''),
                    coalesce(readme_text, ''),
                    coalesce(tags, ''),
                    coalesce(modalities, ''),
                    coalesce(languages, '')
                )) AS content
            FROM datasets
            """
        )
        if self._fts_available:
            try:
                self.conn.execute("DROP INDEX IF EXISTS dataset_search_fts")
                self.conn.execute(
                    "CREATE INDEX dataset_search_fts ON dataset_search USING fts(content)"
                )
            except Exception as exc:  # noqa: BLE001
                import logging
                logging.getLogger(__name__).warning(
                    "FTS index build failed (%s); falling back to LIKE matcher.", exc
                )
                self._fts_available = False
        self._search_index_dirty = False

    def search_bm25(
        self,
        query: str,
        *,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 30,
        offset: int = 0,
    ) -> Tuple[List[Dict[str, Any]], int]:
        if self._search_index_dirty:
            self.rebuild_search_index()
        else:
            # Defensive: if the search table is empty but datasets exist, rebuild
            # the index so queries still return results even after external writes.
            try:
                ds_count = self.conn.execute("SELECT COUNT(*) FROM dataset_search").fetchone()[0]
                total_datasets = self.conn.execute("SELECT COUNT(*) FROM datasets").fetchone()[0]
                if total_datasets and ds_count == 0:
                    self.rebuild_search_index()
            except Exception:
                # If anything goes wrong reading counts, try a rebuild once.
                self.rebuild_search_index()
        use_fts = self._fts_available
        tokens = [t for t in query.lower().split() if t]
        if not tokens:
            return [], 0

        filter_conditions = ""
        filter_params: List[Any] = []
        if filters:
            if filters.get("modality"):
                filter_conditions += " AND d.modalities LIKE '%' || ? || '%'"
                filter_params.append(filters["modality"].lower())
            if filters.get("license_class"):
                filter_conditions += " AND d.license_class = ?"
                filter_params.append(filters["license_class"])
            if filters.get("size_class"):
                filter_conditions += " AND d.size_class = ?"
                filter_params.append(filters["size_class"])
            if filters.get("language"):
                filter_conditions += " AND d.languages LIKE '%' || ? || '%'"
                filter_params.append(filters["language"].lower())

        rows = []
        total = 0

        if use_fts:
            try:
                sql = f"""
                WITH hits AS (
                    SELECT rowid, match_bm25 AS score
                    FROM fts_main_dataset_search(?)
                ),
                schema_state AS (
                    SELECT DISTINCT dataset_id
                    FROM artifacts
                    WHERE kind='schema' AND stale = FALSE
                )
                SELECT
                    d.id,
                    d.title,
                    d.description,
                    d.readme_text,
                    d.tags,
                    d.modalities,
                    d.license_class,
                    d.size_class,
                    d.languages,
                    d.access_class,
                    CASE WHEN ss.dataset_id IS NULL THEN 0 ELSE 1 END AS has_schema,
                    h.score,
                    d.quality_signals
                FROM hits h
                JOIN dataset_search ds ON ds.docid = h.rowid
                JOIN datasets d ON d.id = ds.dataset_id
                LEFT JOIN schema_state ss ON ss.dataset_id = d.id
                WHERE 1=1 {filter_conditions}
                ORDER BY h.score DESC, d.id ASC
                LIMIT ? OFFSET ?
                """
                params_with_paging = [query] + filter_params + [limit, offset]
                rows = self.conn.execute(sql, params_with_paging).fetchall()

                count_sql = f"""
                WITH hits AS (
                    SELECT rowid FROM fts_main_dataset_search(?)
                )
                SELECT COUNT(*)
                FROM hits h
                JOIN dataset_search ds ON ds.docid = h.rowid
                JOIN datasets d ON d.id = ds.dataset_id
                WHERE 1=1 {filter_conditions}
                """
                count_params = [query] + filter_params
                total = self.conn.execute(count_sql, count_params).fetchone()[0]
            except Exception:
                use_fts = False
                # fall through to LIKE fallback

        # If FTS is unavailable *or* unexpectedly returns no hits, fall back to
        # the LIKE-based matcher so the user still gets reasonable results.
        if not use_fts or (total == 0 and tokens):
            score_params: List[Any] = []
            match_params: List[Any] = []
            score_parts = []
            for tok in tokens:
                score_parts.append(
                    "(CASE WHEN lower(ds.content) LIKE '%' || ? || '%' THEN 1 ELSE 0 END)"
                )
                score_params.append(tok)
                match_params.append(tok)
            score_expr = " + ".join(score_parts) if score_parts else "0"

            match_clause = ""
            if tokens:
                # OR semantics: include rows that match ANY token, closer to BM25 behaviour.
                ors = ["lower(ds.content) LIKE '%' || ? || '%'" for _ in tokens]
                match_clause = " AND (" + " OR ".join(ors) + ")"

            sql = f"""
            WITH schema_state AS (
                SELECT DISTINCT dataset_id
                FROM artifacts
                WHERE kind='schema' AND stale = FALSE
            )
            SELECT
                d.id,
                d.title,
                d.description,
                d.readme_text,
                d.tags,
                d.modalities,
                d.license_class,
                d.size_class,
                d.languages,
                d.access_class,
                CASE WHEN ss.dataset_id IS NULL THEN 0 ELSE 1 END AS has_schema,
                {score_expr} AS score,
                d.quality_signals
            FROM dataset_search ds
            JOIN datasets d ON d.id = ds.dataset_id
            LEFT JOIN schema_state ss ON ss.dataset_id = d.id
            WHERE 1=1
            """

            sql += match_clause + filter_conditions + " ORDER BY score DESC, d.id ASC LIMIT ? OFFSET ?"
            params_with_paging = score_params + match_params + filter_params + [limit, offset]
            rows = self.conn.execute(sql, params_with_paging).fetchall()

            count_sql = "SELECT COUNT(*) FROM dataset_search ds JOIN datasets d ON d.id = ds.dataset_id WHERE 1=1"
            count_conditions = ""
            if tokens:
                ors = ["lower(ds.content) LIKE '%' || ? || '%'" for _ in tokens]
                count_conditions = " AND (" + " OR ".join(ors) + ")"
            count_sql += count_conditions + filter_conditions
            count_params = match_params[:] + filter_params
            total = self.conn.execute(count_sql, count_params).fetchone()[0]

        results: List[Dict[str, Any]] = []
        for row in rows:
            (
                did,
                title,
                desc,
                readme_text,
                tags,
                modalities,
                license_class,
                size_class,
                languages,
                access_class,
                has_schema,
                score,
                quality_signals_json,
            ) = row
            qs = self._json_load(quality_signals_json) or {}
            fields_text = " ".join(
                [
                    str(title or "").lower(),
                    str(desc or "").lower(),
                    str(readme_text or "").lower(),
                    str(tags or "").lower(),
                    str(modalities or "").lower(),
                    str(languages or "").lower(),
                ]
            )
            why = [t for t in tokens if t in fields_text]
            results.append(
                {
                    "id": did,
                    "title": title,
                    "description": desc,
                    "readme_text": readme_text,
                    "tags": self._json_load(tags) or [],
                    "modalities": self._json_load(modalities) or [],
                    "license_class": license_class,
                    "size_class": size_class,
                    "languages": self._json_load(languages) or [],
                    "has_schema": bool(has_schema),
                    "score": score,
                    "why": why,
                    "access_class": access_class,
                    "downloads": qs.get("downloads"),
                    "likes": qs.get("likes"),
                }
            )
        return results, total

    def get_dataset(self, dataset_id: DatasetID) -> Optional[Dataset]:
        row = self.conn.execute(
            f"SELECT {','.join(self._dataset_columns())} FROM datasets WHERE id = ?",
            [dataset_id],
        ).fetchone()
        if not row:
            return None
        col_to_val = dict(zip(self._dataset_columns(), row))
        return Dataset(
            id=col_to_val["id"],
            source=col_to_val["source"],
            ns_local_id=col_to_val["ns_local_id"],
            title=col_to_val.get("title"),
            description=col_to_val.get("description"),
            readme_text=col_to_val.get("readme_text"),
            tags=self._json_load(col_to_val.get("tags")),
            modalities=self._json_load(col_to_val.get("modalities")),
            license_class=col_to_val.get("license_class"),
            access_class=col_to_val.get("access_class"),
            size_class=col_to_val.get("size_class"),
            languages=self._json_load(col_to_val.get("languages")),
            last_seen_lm=col_to_val.get("last_seen_lm"),
            readme_sha256=col_to_val.get("readme_sha256"),
            card_sha256=col_to_val.get("card_sha256"),
            fingerprint=col_to_val.get("fingerprint"),
            indexed_at=col_to_val.get("indexed_at"),
            readme_tokens=col_to_val.get("readme_tokens"),
            readme_score=col_to_val.get("readme_score"),
            eligibility_flags=self._json_load(col_to_val.get("eligibility_flags")),
            quality_signals=self._json_load(col_to_val.get("quality_signals")),
            schema_hint=col_to_val.get("schema_hint"),
        )

    # Artifacts
    def get_artifact(self, dataset_id: DatasetID, kind: str) -> Optional[Artifact]:
        row = self.conn.execute(
            "SELECT dataset_id, kind, payload, created_at, stale FROM artifacts WHERE dataset_id = ? AND kind = ?",
            [dataset_id, kind],
        ).fetchone()
        if not row:
            return None
        payload = self._json_load(row[2]) or {}
        return Artifact(
            dataset_id=row[0],
            kind=row[1],
            payload=payload,
            created_at=row[3],
            stale=row[4],
        )

    def put_artifact(self, artifact: Artifact) -> None:
        self.conn.execute(
            """
            INSERT INTO artifacts (dataset_id, kind, payload, created_at, stale)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(dataset_id, kind) DO UPDATE
            SET payload = excluded.payload,
                created_at = excluded.created_at,
                stale = excluded.stale
            """,
            [
                artifact.dataset_id,
                artifact.kind,
                self._json_dump(artifact.payload),
                artifact.created_at,
                artifact.stale,
            ],
        )

    # Jobs
    def enqueue_job(self, job: Job) -> None:
        self.conn.execute(
            """
            INSERT INTO jobs (id, dataset_id, kind, state, attempts, error_class, error_detail, budget_ms, row_cap, enqueued_at, started_at, finished_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                dataset_id=excluded.dataset_id,
                kind=excluded.kind,
                state=excluded.state,
                attempts=excluded.attempts,
                error_class=excluded.error_class,
                error_detail=excluded.error_detail,
                budget_ms=excluded.budget_ms,
                row_cap=excluded.row_cap,
                enqueued_at=excluded.enqueued_at,
                started_at=excluded.started_at,
                finished_at=excluded.finished_at
            """,
            [
                job.id,
                job.dataset_id,
                job.kind,
                job.state,
                job.attempts,
                job.error_class,
                job.error_detail,
                job.budget_ms,
                job.row_cap,
                job.enqueued_at,
                job.started_at,
                job.finished_at,
            ],
        )

    def claim_next_job(self) -> Optional[Job]:
        row = self.conn.execute(
            """
            SELECT id FROM jobs
            WHERE state='queued'
            ORDER BY enqueued_at ASC
            LIMIT 1
            """
        ).fetchone()
        if not row:
            return None
        job_id = row[0]
        now = datetime.utcnow()
        self.conn.execute(
            """
            UPDATE jobs
            SET state='running', started_at=?
            WHERE id=? AND state='queued'
            """,
            [now, job_id],
        )
        row = self.conn.execute(
            """
            SELECT id, dataset_id, kind, state, attempts, error_class, error_detail, enqueued_at, started_at, finished_at, budget_ms, row_cap
            FROM jobs WHERE id=?
            """,
            [job_id],
        ).fetchone()
        if not row:
            return None
        return Job(
            id=row[0],
            dataset_id=row[1],
            kind=row[2],
            state=row[3],
            attempts=row[4],
            error_class=row[5],
            error_detail=row[6],
            enqueued_at=row[7],
            started_at=row[8],
            finished_at=row[9],
            budget_ms=row[10],
            row_cap=row[11],
        )

    def mark_job_done(self, job_id: str) -> None:
        now = datetime.utcnow()
        self.conn.execute(
            "UPDATE jobs SET state='done', finished_at=? WHERE id=?", [now, job_id]
        )

    def mark_job_error(
        self, job_id: str, *, error_class: str, error_detail: str
    ) -> None:
        now = datetime.utcnow()
        self.conn.execute(
            """
            UPDATE jobs
            SET state='error',
                attempts = attempts + 1,
                error_class=?,
                error_detail=?,
                finished_at=?
            WHERE id=?
            """,
            [error_class, error_detail[:2000], now, job_id],
        )

    def job_metrics(self) -> Dict[str, Any]:
        rows = self.conn.execute(
            """
            SELECT state, COUNT(*) FROM jobs GROUP BY state
            """
        ).fetchall()
        return {state: count for state, count in rows}

    def reclaim_stale_jobs(self, lease_timeout_sec: float, max_attempts: int = 3) -> int:
        """Requeue jobs that have been 'running' longer than the lease timeout.

        This keeps the queue self-healing after crashes: when a worker restarts,
        any old in-flight jobs are made eligible for processing again, up to a
        small attempts cap to avoid infinite retry loops.
        """
        if lease_timeout_sec <= 0:
            return 0
        cutoff = datetime.utcnow() - timedelta(seconds=lease_timeout_sec)
        self.conn.execute(
            """
            UPDATE jobs
            SET state='queued', started_at=NULL, finished_at=NULL
            WHERE state='running'
              AND started_at IS NOT NULL
              AND started_at < ?
              AND attempts < ?
            """,
            [cutoff, max_attempts],
        )
        # DuckDB does not expose a stable rowcount here; return 0 for now.
        return 0

    # Signals / events
    def log_event(self, event: Dict[str, Any]) -> None:
        event_id = event.get("id") or str(uuid.uuid4())
        payload = {
            k: v
            for k, v in event.items()
            if k not in {"id", "type", "dataset_id", "created_at"}
        }
        self.conn.execute(
            """
            INSERT INTO events (id, type, dataset_id, payload, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            [
                event_id,
                event.get("type", "unknown"),
                event.get("dataset_id"),
                self._json_dump(payload),
                event.get("created_at") or datetime.utcnow(),
            ],
        )

    def log_signal(self, signal: Dict[str, Any]) -> None:
        signal_id = signal.get("id") or str(uuid.uuid4())
        self.conn.execute(
            """
            INSERT INTO signals (id, session_id, user_id, event, dataset_id, query, rank, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                signal_id,
                signal.get("session_id"),
                signal.get("user_id"),
                signal.get("event"),
                signal.get("dataset_id"),
                signal.get("query"),
                signal.get("rank"),
                signal.get("created_at") or datetime.utcnow(),
            ],
        )

    def admin_metrics(self) -> Dict[str, Any]:
        now = datetime.utcnow()
        day_ago = now - timedelta(days=1)
        total = self.conn.execute("SELECT COUNT(*) FROM datasets").fetchone()[0]
        fresh = self.conn.execute(
            "SELECT COUNT(*) FROM datasets WHERE indexed_at >= ?", [day_ago]
        ).fetchone()[0]
        thin = self.conn.execute(
            "SELECT COUNT(*) FROM datasets WHERE readme_tokens IS NULL OR readme_tokens < 50 OR readme_score < 0.35"
        ).fetchone()[0]
        public_total = self.conn.execute(
            "SELECT COUNT(*) FROM datasets WHERE access_class = 'public'"
        ).fetchone()[0]
        public_with_schema = self.conn.execute(
            """
            SELECT COUNT(DISTINCT d.id)
            FROM datasets d
            JOIN artifacts a
              ON a.dataset_id = d.id
             AND a.kind = 'schema'
             AND a.stale = FALSE
            WHERE d.access_class = 'public'
            """
        ).fetchone()[0]
        public_no_schema = max(public_total - public_with_schema, 0)
        queue_depth = self.conn.execute(
            "SELECT COUNT(*) FROM jobs WHERE state='queued'"
        ).fetchone()[0]
        error_rows = self.conn.execute(
            "SELECT error_class, COUNT(*) FROM jobs WHERE error_class IS NOT NULL GROUP BY error_class"
        ).fetchall()
        durations = self.conn.execute(
            "SELECT EXTRACT(EPOCH FROM (finished_at - started_at))*1000 AS ms FROM jobs WHERE state='done' AND finished_at IS NOT NULL AND started_at IS NOT NULL ORDER BY ms"
        ).fetchall()
        resolve_p95 = 0.0
        if durations:
            idx = int(len(durations) * 0.95) - 1
            idx = max(idx, 0)
            resolve_p95 = durations[idx][0] if durations[idx][0] is not None else 0.0

        return {
            "freshness_pct": (fresh / total * 100) if total else 0.0,
            "thin_readme_pct": (thin / total * 100) if total else 0.0,
            "resolve_p95_ms": resolve_p95,
            "queue_depth": queue_depth,
            "error_taxonomy": {r[0]: r[1] for r in error_rows},
            "public_no_schema_count": public_no_schema,
            "public_with_schema_pct": (public_with_schema / public_total * 100) if public_total else 0.0,
        }

    def get_connector_cursor(self, source: str) -> Optional[Dict[str, Any]]:
        row = self.conn.execute(
            "SELECT cursor FROM connectors WHERE source = ?", [source]
        ).fetchone()
        if not row or not row[0]:
            return None
        return self._json_load(row[0]) or None

    def upsert_connector(
        self,
        source: str,
        *,
        cursor: Optional[Dict[str, Any]],
        qps_limit: Optional[int] = None,
        backoff_state: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.conn.execute(
            """
            INSERT INTO connectors (source, cursor, qps_limit, backoff_state)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(source) DO UPDATE SET
                cursor=excluded.cursor,
                qps_limit=COALESCE(excluded.qps_limit, connectors.qps_limit),
                backoff_state=COALESCE(excluded.backoff_state, connectors.backoff_state)
            """,
            [
                source,
                self._json_dump(cursor),
                qps_limit,
                self._json_dump(backoff_state),
            ],
        )

    def prefetch_candidates(self, limit: int) -> List[DatasetID]:
        # Thin/no-README public datasets without fresh schema artifact.
        rows = self.conn.execute(
            """
            WITH sig AS (
              SELECT dataset_id, COUNT(*) AS sig_score
              FROM signals
              WHERE created_at >= NOW() - INTERVAL 7 DAY
              GROUP BY dataset_id
            )
            SELECT d.id
            FROM datasets d
            LEFT JOIN artifacts a
              ON a.dataset_id = d.id AND a.kind='schema' AND a.stale = FALSE
            LEFT JOIN sig ON sig.dataset_id = d.id
            WHERE d.access_class = 'public'
              AND (d.readme_tokens IS NULL OR d.readme_tokens < 50 OR d.readme_score < 0.35)
              AND a.dataset_id IS NULL
            ORDER BY COALESCE(sig.sig_score, 0) DESC,
                     TRY_CAST(json_extract(d.quality_signals, '$.downloads') AS BIGINT) DESC NULLS LAST,
                     d.last_seen_lm DESC NULLS LAST
            LIMIT ?
            """,
            [limit],
        ).fetchall()
        return [r[0] for r in rows]

    # Search-quality evaluation ------------------------------------------------

    def eval_list_queries(self, *, active_only: bool = True) -> List[Dict[str, Any]]:
        sql = "SELECT id, label, query, target_id, total_relevant, active, created_at FROM eval_queries"
        if active_only:
            sql += " WHERE active = TRUE"
        sql += " ORDER BY id"
        rows = self.conn.execute(sql).fetchall()
        return [
            {
                "id": r[0],
                "label": r[1],
                "query": r[2],
                "target_id": r[3],
                "total_relevant": r[4],
                "active": bool(r[5]),
                "created_at": r[6],
            }
            for r in rows
        ]

    def eval_upsert_query(self, query: Dict[str, Any]) -> None:
        self.conn.execute(
            """
            INSERT INTO eval_queries (id, label, query, target_id, total_relevant, active, created_at)
            VALUES (?, ?, ?, ?, ?, TRUE, ?)
            ON CONFLICT(id) DO UPDATE SET
                label=excluded.label,
                query=excluded.query,
                target_id=excluded.target_id,
                total_relevant=excluded.total_relevant,
                active=TRUE
            """,
            [
                query["id"],
                query.get("label", ""),
                query["query"],
                query.get("target_id"),
                query.get("total_relevant", 1),
                datetime.utcnow(),
            ],
        )

    def eval_set_query_active(self, query_id: str, active: bool) -> None:
        self.conn.execute("UPDATE eval_queries SET active = ? WHERE id = ?", [active, query_id])

    def eval_create_run(
        self, run_id: str, *, label: str, engine: str, k: int, created_at: datetime
    ) -> None:
        self.conn.execute(
            "INSERT INTO eval_runs (id, label, engine, k, created_at) VALUES (?, ?, ?, ?, ?)",
            [run_id, label, engine, k, created_at],
        )

    def eval_add_run_results(self, run_id: str, query_id: str, rows: List[Dict[str, Any]]) -> None:
        for row in rows:
            self.conn.execute(
                """
                INSERT INTO eval_run_results (run_id, query_id, result_rank, dataset_id, title, extra)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id, query_id, result_rank) DO UPDATE SET
                    dataset_id=excluded.dataset_id,
                    title=excluded.title,
                    extra=excluded.extra
                """,
                [
                    run_id,
                    query_id,
                    row["rank"],
                    row["dataset_id"],
                    row.get("title"),
                    self._json_dump(row.get("extra")),
                ],
            )

    def eval_list_runs(self) -> List[Dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT id, label, engine, k, created_at FROM eval_runs ORDER BY created_at DESC"
        ).fetchall()
        return [{"id": r[0], "label": r[1], "engine": r[2], "k": r[3], "created_at": r[4]} for r in rows]

    def eval_get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        row = self.conn.execute(
            "SELECT id, label, engine, k, created_at FROM eval_runs WHERE id = ?", [run_id]
        ).fetchone()
        if not row:
            return None
        return {"id": row[0], "label": row[1], "engine": row[2], "k": row[3], "created_at": row[4]}

    def eval_get_run_results(self, run_id: str) -> Dict[str, List[Dict[str, Any]]]:
        rows = self.conn.execute(
            """
            SELECT rr.query_id, rr.result_rank, rr.dataset_id, rr.title, rr.extra, j.relevant
            FROM eval_run_results rr
            LEFT JOIN eval_relevance_judgments j
              ON j.query_id = rr.query_id AND j.dataset_id = rr.dataset_id
            WHERE rr.run_id = ?
            ORDER BY rr.query_id, rr.result_rank
            """,
            [run_id],
        ).fetchall()
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for query_id, rank, dataset_id, title, extra, relevant in rows:
            grouped.setdefault(query_id, []).append(
                {
                    "rank": rank,
                    "dataset_id": dataset_id,
                    "title": title,
                    "extra": self._json_load(extra) or {},
                    "relevant": bool(relevant) if relevant is not None else None,
                }
            )
        return grouped

    def eval_upsert_judgment(
        self, query_id: str, dataset_id: str, relevant: bool, judged_by: Optional[str] = None
    ) -> None:
        self.conn.execute(
            """
            INSERT INTO eval_relevance_judgments (query_id, dataset_id, relevant, judged_by, judged_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(query_id, dataset_id) DO UPDATE SET
                relevant=excluded.relevant,
                judged_by=excluded.judged_by,
                judged_at=excluded.judged_at
            """,
            [query_id, dataset_id, relevant, judged_by, datetime.utcnow()],
        )
