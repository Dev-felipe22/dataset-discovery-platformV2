"""
Read-only client for the `hf_dataset_rows` Elasticsearch index.

That index is populated separately by the V1 pipeline's
`scripts/import_samples_to_es.py` — one document per sampled row, with
`dataset_id`/`config`/`split`/`row_index` fields plus `row_text` (searchable)
and the raw `row` (stored, not indexed). This client only reads from it.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

from elasticsearch import Elasticsearch


class EsRowsClient:
    def __init__(self, es_url: str, index: str):
        self._es = Elasticsearch(es_url)
        self._index = index

    def ping(self) -> bool:
        try:
            return bool(self._es.ping())
        except Exception:
            return False

    def list_sample_datasets(
        self, limit: int = 50, offset: int = 0, query: Optional[str] = None
    ) -> Tuple[List[Dict[str, Any]], int]:
        """Distinct dataset_ids that have sample rows, with per-dataset row counts.

        This is the "group" of datasets that actually contain sampled rows —
        an aggregation over dataset_id, not a plain document search. Sorted by
        row count descending. Pagination is applied in-memory over the
        aggregation buckets since ES terms aggregations aren't natively
        offset-paginated the way document search is.
        """
        body: Dict[str, Any] = {
            "size": 0,
            "aggs": {
                "datasets": {
                    "terms": {
                        "field": "dataset_id",
                        "size": max(offset + limit, 1000),
                        "order": {"_key": "asc"},
                    }
                }
            },
        }
        if query:
            body["query"] = {"match": {"row_text": query}}

        resp = self._es.search(index=self._index, body=body)
        buckets = resp["aggregations"]["datasets"]["buckets"]
        total = len(buckets)
        page = buckets[offset : offset + limit]
        items = [{"dataset_id": b["key"], "row_count": b["doc_count"]} for b in page]
        return items, total

    def dataset_ids_with_rows(self, ids: List[str]) -> Set[str]:
        """Given a batch of dataset IDs (e.g. one page of DuckDB search hits),
        return the subset that actually have sample rows in Elasticsearch.

        This is the join point between the two systems: DuckDB and
        Elasticsearch both key on the same HF dataset_id string, so no
        separate mapping table is needed — just an existence check per batch.
        """
        if not ids:
            return set()
        try:
            body: Dict[str, Any] = {
                "size": 0,
                "query": {"terms": {"dataset_id": ids}},
                "aggs": {"ids": {"terms": {"field": "dataset_id", "size": len(ids)}}},
            }
            resp = self._es.search(index=self._index, body=body)
            buckets = resp["aggregations"]["ids"]["buckets"]
            return {b["key"] for b in buckets}
        except Exception:
            return set()

    def get_sample_rows(
        self, dataset_id: str, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """All indexed sample rows for one dataset, ordered by config/split/row_index."""
        body = {
            "size": limit,
            "query": {"term": {"dataset_id": dataset_id}},
            "sort": [
                {"config": "asc"},
                {"split": "asc"},
                {"row_index": "asc"},
            ],
        }
        resp = self._es.search(index=self._index, body=body)
        hits = resp["hits"]["hits"]
        return [
            {
                "config": h["_source"].get("config"),
                "split": h["_source"].get("split"),
                "row_index": h["_source"].get("row_index"),
                "row": h["_source"].get("row"),
            }
            for h in hits
        ]
