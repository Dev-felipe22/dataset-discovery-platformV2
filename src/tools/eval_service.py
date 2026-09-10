"""Search-quality benchmark harness: run registered engines against a fixed
set of benchmark queries, persist results, and compute retrieval metrics
against accumulated relevance judgments.

The point of this module is that it should NOT need to change when the
search implementation changes. To benchmark a new retrieval strategy (e.g.
an agent-based search), register a new function in ENGINES — everything
else (storage, metrics, UI, export) keeps working unmodified.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from src.storage import StorageAdapter
from src.tools.eval_metrics import compute_metrics, mean_metric

EngineFn = Callable[[StorageAdapter, str, int], List[Dict[str, Any]]]


def _run_bm25(storage: StorageAdapter, query: str, k: int) -> List[Dict[str, Any]]:
    """Today's default: whatever storage.search_bm25 returns — the exact same
    code path /v2/search_index uses, so this benchmarks the live app search.
    """
    hits, _total = storage.search_bm25(query, limit=k, offset=0)
    out = []
    for h in hits:
        out.append(
            {
                "dataset_id": h["id"],
                "title": h.get("title"),
                "extra": {
                    "description": (h.get("description") or "")[:200],
                    "license_class": h.get("license_class"),
                    "size_class": h.get("size_class"),
                    "modalities": h.get("modalities") or [],
                    "has_schema": h.get("has_schema", False),
                    "downloads": h.get("downloads"),
                    "likes": h.get("likes"),
                },
            }
        )
    return out


# Registry of retrieval strategies that can be benchmarked against the same
# query set and judgments. Add an entry here when a new search
# implementation needs measuring, e.g.:
#
#   def _run_agent_search(storage, query, k):
#       ... call the new search path, return [{"dataset_id", "title", "extra"}]
#   ENGINES["agent_v1"] = _run_agent_search
#
# The run endpoint takes `engine` as a plain string key into this dict.
ENGINES: Dict[str, EngineFn] = {
    "bm25": _run_bm25,
}


def run_eval(storage: StorageAdapter, *, label: str, engine: str, k: int) -> str:
    """Execute every active benchmark query through `engine` and persist a
    new run. Returns the new run id."""
    if engine not in ENGINES:
        raise ValueError(f"Unknown engine '{engine}'. Available: {sorted(ENGINES)}")

    run_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc)
    storage.eval_create_run(run_id, label=label, engine=engine, k=k, created_at=created_at)

    fn = ENGINES[engine]
    for q in storage.eval_list_queries(active_only=True):
        hits = fn(storage, q["query"], k)
        rows = [
            {
                "rank": i,
                "dataset_id": h["dataset_id"],
                "title": h.get("title"),
                "extra": h.get("extra") or {},
            }
            for i, h in enumerate(hits, start=1)
        ]
        storage.eval_add_run_results(run_id, q["id"], rows)
    return run_id


def _query_detail(q_cfg: Dict[str, Any], results: List[Dict[str, Any]], k: int) -> Dict[str, Any]:
    flags = [bool(r.get("relevant")) for r in results]
    total_relevant = q_cfg.get("total_relevant") or 1
    metrics = compute_metrics(flags, k=k, total_relevant=total_relevant)
    return {
        "query_id": q_cfg["id"],
        "label": q_cfg.get("label", ""),
        "query": q_cfg.get("query", ""),
        "total_relevant": total_relevant,
        "results": results,
        "metrics": metrics,
    }


def get_run_detail(storage: StorageAdapter, run_id: str) -> Optional[Dict[str, Any]]:
    """Full run detail: per-query captured results (joined live against
    current relevance judgments) plus metrics, recomputed fresh every call —
    so judging a result immediately changes what this returns, with no
    separate 'recalculate' step."""
    run = storage.eval_get_run(run_id)
    if not run:
        return None

    all_queries = {q["id"]: q for q in storage.eval_list_queries(active_only=False)}
    grouped = storage.eval_get_run_results(run_id)

    query_details = [
        _query_detail(
            all_queries.get(query_id, {"id": query_id, "label": "", "query": "", "total_relevant": 1}),
            results,
            run["k"],
        )
        for query_id, results in grouped.items()
    ]
    query_details.sort(key=lambda d: d["query_id"])

    metrics_list = [qd["metrics"] for qd in query_details]
    summary = {
        "id": run["id"],
        "label": run["label"],
        "engine": run["engine"],
        "k": run["k"],
        "created_at": run["created_at"],
        "map": mean_metric(metrics_list, "ap"),
        "mean_mrr": mean_metric(metrics_list, "mrr"),
        "mean_ndcg": mean_metric(metrics_list, "ndcg"),
        "mean_precision": mean_metric(metrics_list, "precision"),
        "mean_recall": mean_metric(metrics_list, "recall"),
        "mean_f1": mean_metric(metrics_list, "f1"),
    }
    return {"run": summary, "queries": query_details}


def list_run_summaries(storage: StorageAdapter) -> List[Dict[str, Any]]:
    """Run history with live-recomputed aggregate metrics, newest first —
    what the Summary/trend view compares runs on."""
    out = []
    for r in storage.eval_list_runs():
        detail = get_run_detail(storage, r["id"])
        if detail:
            out.append(detail["run"])
    return out
