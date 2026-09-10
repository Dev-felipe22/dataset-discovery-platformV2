from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional
from enum import Enum

from pydantic import BaseModel, Field


class SearchFilters(BaseModel):
    modality: Optional[str] = None
    license_class: Optional[str] = None
    size_class: Optional[str] = None
    language: Optional[str] = None


class SearchRequest(BaseModel):
    query: str
    filters: Optional[SearchFilters] = None
    limit: int = Field(default=30, ge=1, le=100)
    offset: int = Field(default=0, ge=0)


class SearchHit(BaseModel):
    id: str
    title: Optional[str] = None
    description: Optional[str] = None
    readme_text: Optional[str] = None
    why: List[str] = Field(default_factory=list)
    has_schema: bool = False
    blocked_reason: Optional[str] = None
    license_class: Optional[str] = None
    access_class: Optional[str] = None
    size_class: Optional[str] = None
    modalities: List[str] = Field(default_factory=list)
    languages: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    downloads: Optional[int] = None
    likes: Optional[int] = None
    has_sample_rows: bool = False


class SearchResponse(BaseModel):
    hits: List[SearchHit]
    total: int


class RequestResolve(BaseModel):
    id: str
    kind: str = Field(default="schema", pattern="^(schema|stats|snips)$")
    budget_ms: Optional[int] = None
    row_cap: Optional[int] = None
    ui_context: Optional[str] = None


class RequestResolveResponse(BaseModel):
    job_id: str


class PolicyCheckRequest(BaseModel):
    id: str
    action: str = Field(default="schema")
    budget_ms: Optional[int] = None
    row_cap: Optional[int] = None


class PolicyCheckResponse(BaseModel):
    allow: bool
    reason_code: Optional[str] = None


class ArtifactResponse(BaseModel):
    payload: dict
    stale: bool = False


class SignalRequest(BaseModel):
    session_id: str
    user_id: Optional[str] = None
    event: str
    dataset_id: Optional[str] = None
    query: Optional[str] = None
    rank: Optional[int] = None


class AdminMetricsResponse(BaseModel):
    freshness_pct: float
    thin_readme_pct: float
    resolve_p95_ms: float
    queue_depth: int
    error_taxonomy: dict
    public_no_schema_count: int
    public_with_schema_pct: float
    prefetch_paused: bool
    prefetch_reason: Optional[str] = None
    jobs_by_state: dict = Field(default_factory=dict)


class DiscoveryMode(str, Enum):
    topup = "topup"
    backfill = "backfill"


class AdminOpResponse(BaseModel):
    status: str
    detail: Optional[str] = None


class AdminDiscoveryRequest(BaseModel):
    mode: DiscoveryMode = Field(default=DiscoveryMode.topup)
    window_days: Optional[int] = None  # used when mode=backfill


class AdminDiscoveryResponse(AdminOpResponse):
    mode: DiscoveryMode
    window_days: Optional[int] = None


class AdminPrefetchRequest(BaseModel):
    limit: Optional[int] = None


class AdminPrefetchResponse(AdminOpResponse):
    enqueued: int = 0


class SampleDatasetItem(BaseModel):
    dataset_id: str
    row_count: int


class SampleDatasetsResponse(BaseModel):
    items: List[SampleDatasetItem]
    total: int


class SampleRowItem(BaseModel):
    config: Optional[str] = None
    split: Optional[str] = None
    row_index: Optional[int] = None
    row: dict


class SampleRowsResponse(BaseModel):
    dataset_id: str
    rows: List[SampleRowItem]
    total: int


class DatasetMetadataResponse(BaseModel):
    id: str
    title: Optional[str] = None
    description: Optional[str] = None
    license_class: Optional[str] = None
    access_class: Optional[str] = None
    size_class: Optional[str] = None
    modalities: List[str] = Field(default_factory=list)
    languages: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)


# ----------------------------------------------------------------------------
# Search-quality evaluation (benchmark harness)
# ----------------------------------------------------------------------------

class EvalQueryIn(BaseModel):
    id: str
    label: str = ""
    query: str
    target_id: Optional[str] = None
    total_relevant: int = Field(default=1, ge=1)


class EvalQueryOut(EvalQueryIn):
    active: bool = True
    created_at: Optional[datetime] = None


class EvalEnginesResponse(BaseModel):
    engines: List[str]


class EvalRunRequest(BaseModel):
    label: str = Field(default="run")
    engine: str = Field(default="bm25")
    k: int = Field(default=10, ge=1, le=50)


class EvalRunSummary(BaseModel):
    id: str
    label: str
    engine: str
    k: int
    created_at: datetime
    map: float = 0.0
    mean_mrr: float = 0.0
    mean_ndcg: float = 0.0
    mean_precision: float = 0.0
    mean_recall: float = 0.0
    mean_f1: float = 0.0


class EvalRunListResponse(BaseModel):
    runs: List[EvalRunSummary]


class EvalResultItem(BaseModel):
    rank: int
    dataset_id: str
    title: Optional[str] = None
    extra: Dict[str, Any] = Field(default_factory=dict)
    relevant: Optional[bool] = None


class EvalQueryResult(BaseModel):
    query_id: str
    label: str
    query: str
    total_relevant: int
    results: List[EvalResultItem]
    metrics: Dict[str, float]


class EvalRunDetailResponse(BaseModel):
    run: EvalRunSummary
    queries: List[EvalQueryResult]


class EvalJudgeRequest(BaseModel):
    query_id: str
    dataset_id: str
    relevant: bool
