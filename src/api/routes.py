import io
from uuid import uuid4
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse

from src.api import schemas
from src.api.deps import get_es_rows_client, get_settings, get_storage
from src.api.policy import evaluate_policy
from src.config import Settings
from src.connectors.es_rows_client import EsRowsClient
from src.storage import Job, StorageAdapter
from src.storage.duckdb_backend import DuckDBStorage
from src.tools.prefetch import should_pause
from src.tools.discovery_hf import discover
from src.tools import eval_report, eval_service

router = APIRouter(prefix="/v2")


@router.get("/healthz")
def healthcheck() -> dict:
    return {"status": "ok"}


@router.post("/search_index", response_model=schemas.SearchResponse)
def search_index(
    request: schemas.SearchRequest,
    storage: StorageAdapter = Depends(get_storage),
    es: EsRowsClient = Depends(get_es_rows_client),
) -> schemas.SearchResponse:
    filters = request.filters.dict() if request.filters else {}
    hits_raw, total = storage.search_bm25(
        request.query, filters=filters, limit=request.limit, offset=request.offset
    )

    # Join point between the two systems: check which of this page's dataset
    # IDs also have sample rows in Elasticsearch, in one batched query. Never
    # let an unreachable ES take down metadata search — default to "no rows".
    ids_with_rows: set = set()
    if hits_raw:
        ids_with_rows = es.dataset_ids_with_rows([h["id"] for h in hits_raw])

    hits = [
        schemas.SearchHit(
            id=h["id"],
            title=h.get("title"),
            description=h.get("description"),
            readme_text=h.get("readme_text"),
            why=h.get("why", []),
            has_schema=h.get("has_schema", False),
            blocked_reason=None,
            license_class=h.get("license_class"),
            access_class=h.get("access_class"),
            size_class=h.get("size_class"),
            modalities=h.get("modalities", []),
            languages=h.get("languages", []),
            tags=h.get("tags", []),
            downloads=h.get("downloads"),
            likes=h.get("likes"),
            has_sample_rows=h["id"] in ids_with_rows,
        )
        for h in hits_raw
    ]
    return schemas.SearchResponse(hits=hits, total=total)


@router.get("/dataset", response_model=schemas.DatasetMetadataResponse)
def get_dataset_metadata(
    id: str,
    storage: StorageAdapter = Depends(get_storage),
) -> schemas.DatasetMetadataResponse:
    """DuckDB-backed metadata for one dataset — used by the sample-rows detail
    page to show title/license/etc. alongside its Elasticsearch row content."""
    dataset = storage.get_dataset(id)
    if not dataset:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset not found")
    return schemas.DatasetMetadataResponse(
        id=dataset.id,
        title=dataset.title,
        description=dataset.description,
        license_class=dataset.license_class,
        access_class=dataset.access_class,
        size_class=dataset.size_class,
        modalities=dataset.modalities or [],
        languages=dataset.languages or [],
        tags=dataset.tags or [],
    )


@router.get("/sample_datasets", response_model=schemas.SampleDatasetsResponse)
def sample_datasets(
    limit: int = 50,
    offset: int = 0,
    q: Optional[str] = None,
    es: EsRowsClient = Depends(get_es_rows_client),
) -> schemas.SampleDatasetsResponse:
    """The 'group' of datasets that actually have sampled rows in Elasticsearch.

    Backed by the V1 pipeline's samples.jsonl, uploaded via
    scripts/import_samples_to_es.py — separate from the DuckDB dataset catalog.
    """
    if not es.ping():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Elasticsearch is unreachable.",
        )
    items, total = es.list_sample_datasets(limit=limit, offset=offset, query=q)
    return schemas.SampleDatasetsResponse(
        items=[schemas.SampleDatasetItem(**i) for i in items],
        total=total,
    )


@router.get("/sample_rows", response_model=schemas.SampleRowsResponse)
def sample_rows(
    dataset_id: str,
    limit: int = 100,
    es: EsRowsClient = Depends(get_es_rows_client),
) -> schemas.SampleRowsResponse:
    """Actual sampled row content for one dataset, read from Elasticsearch."""
    if not es.ping():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Elasticsearch is unreachable.",
        )
    rows = es.get_sample_rows(dataset_id, limit=limit)
    if not rows:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No sample rows found for this dataset.",
        )
    return schemas.SampleRowsResponse(
        dataset_id=dataset_id,
        rows=[schemas.SampleRowItem(**r) for r in rows],
        total=len(rows),
    )


@router.post("/request_resolve", response_model=schemas.RequestResolveResponse)
def request_resolve(
    request: schemas.RequestResolve,
    storage: StorageAdapter = Depends(get_storage),
    settings: Settings = Depends(get_settings),
) -> schemas.RequestResolveResponse:
    existing = storage.get_artifact(request.id, request.kind)
    if existing and not existing.stale:
        return schemas.RequestResolveResponse(job_id=f"cached-{request.id}")

    dataset = storage.get_dataset(request.id)
    allow, reason = evaluate_policy(
        dataset,
        budget_ms=request.budget_ms or settings.resolve_budget_ms_default,
        row_cap=request.row_cap or settings.resolve_row_cap_default,
        settings=settings,
    )
    if not allow:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={"reason_code": reason},
        )
    job_id = str(uuid4())
    now = datetime.utcnow()
    job = Job(
        id=job_id,
        dataset_id=request.id,
        kind=request.kind,
        state="queued",
        attempts=0,
        enqueued_at=now,
        started_at=None,
        finished_at=None,
        error_class=None,
        error_detail=None,
        budget_ms=request.budget_ms or settings.resolve_budget_ms_default,
        row_cap=request.row_cap or settings.resolve_row_cap_default,
    )
    storage.enqueue_job(job)
    storage.log_event(
        {
            "type": "dataset.requested",
            "dataset_id": request.id,
            "job_id": job_id,
            "kind": request.kind,
            "budget_ms": job.budget_ms,
            "row_cap": job.row_cap,
            "created_at": now,
        }
    )
    return schemas.RequestResolveResponse(job_id=job_id)


@router.get("/get_artifact", response_model=schemas.ArtifactResponse)
def get_artifact(
    id: str,
    kind: str,
    storage: StorageAdapter = Depends(get_storage),
) -> schemas.ArtifactResponse:
    artifact = storage.get_artifact(id, kind)
    if not artifact or artifact.stale:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="not found")
    return schemas.ArtifactResponse(payload=artifact.payload, stale=artifact.stale)


@router.post("/policy_check", response_model=schemas.PolicyCheckResponse)
def policy_check(
    request: schemas.PolicyCheckRequest,
    storage: StorageAdapter = Depends(get_storage),
    settings: Settings = Depends(get_settings),
) -> schemas.PolicyCheckResponse:
    dataset = storage.get_dataset(request.id)
    allow, reason = evaluate_policy(
        dataset,
        budget_ms=request.budget_ms or settings.resolve_budget_ms_default,
        row_cap=request.row_cap or settings.resolve_row_cap_default,
        settings=settings,
    )
    return schemas.PolicyCheckResponse(allow=allow, reason_code=reason)


@router.post("/log_signal", status_code=status.HTTP_204_NO_CONTENT)
def log_signal(
    request: schemas.SignalRequest,
    storage: StorageAdapter = Depends(get_storage),
) -> None:
    storage.log_signal(request.model_dump())
    return None


@router.get("/admin", response_model=schemas.AdminMetricsResponse)
def admin(
    storage: StorageAdapter = Depends(get_storage),
    settings: Settings = Depends(get_settings),
) -> schemas.AdminMetricsResponse:
    metrics = storage.admin_metrics()
    job_stats = storage.job_metrics()
    paused = False
    reason = None
    if metrics.get("resolve_p95_ms", 0) > settings.resolve_p95_max_ms:
        paused = True
        reason = "resolve_slow"
    if metrics.get("queue_depth", 0) > settings.queue_depth_absolute_max:
        paused = True
        reason = reason or "queue_depth"
    if not settings.prefetch_enabled:
        paused = True
        reason = reason or "prefetch_disabled"
    return schemas.AdminMetricsResponse(
        freshness_pct=metrics.get("freshness_pct", 0.0),
        thin_readme_pct=metrics.get("thin_readme_pct", 0.0),
        resolve_p95_ms=metrics.get("resolve_p95_ms", 0.0),
        queue_depth=metrics.get("queue_depth", 0),
        error_taxonomy=metrics.get("error_taxonomy", {}),
        public_no_schema_count=metrics.get("public_no_schema_count", 0),
        public_with_schema_pct=metrics.get("public_with_schema_pct", 0.0),
        prefetch_paused=paused,
        prefetch_reason=reason,
        jobs_by_state=job_stats or {},
    )


@router.post("/admin/rebuild_index", response_model=schemas.AdminOpResponse)
def admin_rebuild_index(
    settings: Settings = Depends(get_settings),
) -> schemas.AdminOpResponse:
    storage = DuckDBStorage(settings.duckdb_uri())
    storage.init()
    storage.refresh_dataset_state_view()
    storage.rebuild_search_index()
    return schemas.AdminOpResponse(
        status="ok",
        detail="Search index rebuilt (state view + FTS/LIKE structures).",
    )


@router.post("/admin/run_discovery", response_model=schemas.AdminDiscoveryResponse)
def admin_run_discovery(
    request: schemas.AdminDiscoveryRequest,
    settings: Settings = Depends(get_settings),
) -> schemas.AdminDiscoveryResponse:
    limit = 500
    qps = 1.0
    backfill_until = None
    window_days: Optional[int] = None
    if request.mode == schemas.DiscoveryMode.backfill:
        if request.window_days not in (1, 3, 7):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="window_days must be one of 1, 3, or 7 for backfill.",
            )
        window_days = request.window_days
        backfill_until = datetime.now(timezone.utc) - timedelta(days=window_days)
    discover(settings.duckdb_uri(), limit, qps, backfill_until)
    detail = (
        "Discovery top-up completed; check Metrics for updated freshness/coverage."
        if request.mode == schemas.DiscoveryMode.topup
        else f"Discovery backfill ({window_days}d) completed; check Metrics for updated coverage."
    )
    return schemas.AdminDiscoveryResponse(
        status="ok",
        detail=detail,
        mode=request.mode,
        window_days=window_days,
    )


@router.post("/admin/run_prefetch", response_model=schemas.AdminPrefetchResponse)
def admin_run_prefetch(
    request: schemas.AdminPrefetchRequest,
    settings: Settings = Depends(get_settings),
) -> schemas.AdminPrefetchResponse:
    storage = DuckDBStorage(settings.duckdb_uri())
    storage.init()
    metrics = storage.admin_metrics()
    paused, reason = should_pause(metrics, settings)
    if paused:
        return schemas.AdminPrefetchResponse(
            status="skipped",
            detail=f"Prefetch skipped: paused due to {reason}.",
            enqueued=0,
        )
    limit = request.limit or 100
    candidates = storage.prefetch_candidates(limit)
    if not candidates:
        return schemas.AdminPrefetchResponse(
            status="ok",
            detail="No prefetch candidates (thin/no-README public datasets without schema).",
            enqueued=0,
        )
    now = datetime.utcnow()
    enqueued = 0
    for did in candidates:
        storage.enqueue_job(
            Job(
                id=f"prefetch-{did}",
                dataset_id=did,
                kind="schema",
                state="queued",
                attempts=0,
                enqueued_at=now,
                started_at=None,
                finished_at=None,
                error_class=None,
                error_detail=None,
                budget_ms=settings.resolve_budget_ms_default,
                row_cap=settings.resolve_row_cap_default,
            )
        )
        enqueued += 1
    return schemas.AdminPrefetchResponse(
        status="ok",
        detail=f"Prefetch enqueued {enqueued} schema jobs.",
        enqueued=enqueued,
    )


# ----------------------------------------------------------------------------
# Search-quality evaluation (benchmark harness)
#
# Run on demand, not on every search: a fixed set of benchmark queries with
# accumulated relevance judgments, replayed through a named "engine" (today
# only the live app search). Ported from hf-queries.py so this doesn't need
# to be recoded when the search implementation changes later — a future
# engine registers itself in src/tools/eval_service.ENGINES.
# ----------------------------------------------------------------------------

@router.get("/eval/engines", response_model=schemas.EvalEnginesResponse)
def eval_engines() -> schemas.EvalEnginesResponse:
    return schemas.EvalEnginesResponse(engines=sorted(eval_service.ENGINES.keys()))


@router.get("/eval/queries", response_model=list[schemas.EvalQueryOut])
def eval_list_queries(
    storage: StorageAdapter = Depends(get_storage),
) -> list[schemas.EvalQueryOut]:
    return [schemas.EvalQueryOut(**q) for q in storage.eval_list_queries(active_only=False)]


@router.post("/eval/queries", response_model=schemas.EvalQueryOut)
def eval_upsert_query(
    request: schemas.EvalQueryIn,
    storage: StorageAdapter = Depends(get_storage),
) -> schemas.EvalQueryOut:
    storage.eval_upsert_query(request.model_dump())
    match = next((q for q in storage.eval_list_queries(active_only=False) if q["id"] == request.id), None)
    if not match:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="query not saved")
    return schemas.EvalQueryOut(**match)


@router.delete("/eval/queries/{query_id}", status_code=status.HTTP_204_NO_CONTENT)
def eval_deactivate_query(
    query_id: str,
    storage: StorageAdapter = Depends(get_storage),
) -> None:
    storage.eval_set_query_active(query_id, False)
    return None


@router.post("/eval/run", response_model=schemas.EvalRunDetailResponse)
def eval_run(
    request: schemas.EvalRunRequest,
    storage: StorageAdapter = Depends(get_storage),
) -> schemas.EvalRunDetailResponse:
    try:
        run_id = eval_service.run_eval(storage, label=request.label, engine=request.engine, k=request.k)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    detail = eval_service.get_run_detail(storage, run_id)
    return schemas.EvalRunDetailResponse(**detail)


@router.get("/eval/runs", response_model=schemas.EvalRunListResponse)
def eval_list_runs(
    storage: StorageAdapter = Depends(get_storage),
) -> schemas.EvalRunListResponse:
    return schemas.EvalRunListResponse(runs=eval_service.list_run_summaries(storage))


@router.get("/eval/runs/{run_id}", response_model=schemas.EvalRunDetailResponse)
def eval_get_run(
    run_id: str,
    storage: StorageAdapter = Depends(get_storage),
) -> schemas.EvalRunDetailResponse:
    detail = eval_service.get_run_detail(storage, run_id)
    if not detail:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="run not found")
    return schemas.EvalRunDetailResponse(**detail)


@router.post("/eval/judge", status_code=status.HTTP_204_NO_CONTENT)
def eval_judge(
    request: schemas.EvalJudgeRequest,
    storage: StorageAdapter = Depends(get_storage),
) -> None:
    storage.eval_upsert_judgment(request.query_id, request.dataset_id, request.relevant)
    return None


@router.get("/eval/runs/{run_id}/export")
def eval_export_run(
    run_id: str,
    storage: StorageAdapter = Depends(get_storage),
) -> StreamingResponse:
    detail = eval_service.get_run_detail(storage, run_id)
    if not detail:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="run not found")
    wb = eval_report.build_workbook(detail["run"], detail["queries"])
    buf = io.BytesIO()
    wb.save(buf)
    buf.seek(0)
    safe_label = "".join(c if c.isalnum() else "_" for c in detail["run"]["label"])[:40] or "run"
    filename = f"eval_{safe_label}_{run_id[:8]}.xlsx"
    return StreamingResponse(
        buf,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
