"""
Bulk-index sampled dataset rows from the V1 pipeline into Elasticsearch.

Each line in samples.jsonl is one (dataset, config, split) unit containing an
array of sampled rows. This script flattens it to one ES document per row, so
each individual row becomes independently searchable.

Row content is wildly heterogeneous across thousands of different HF dataset
schemas, so raw row fields are stored but NOT dynamically mapped field-by-field
(letting ES auto-map them would blow past its default per-index field limit
almost immediately). Instead, `row_text` concatenates every string/number
value in the row into one searchable blob, and the raw `row` is kept as
stored-but-unindexed source for display/retrieval.

Usage:
  python -m scripts.import_samples_to_es \
      --input   /v1_pipeline_data/sampler/samples.jsonl \
      --es-url  http://host.docker.internal:9200 \
      --index   hf_dataset_rows \
      [--limit 1000] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

INDEX_MAPPING = {
    "mappings": {
        "properties": {
            "dataset_id": {"type": "keyword"},
            "config": {"type": "keyword"},
            "split": {"type": "keyword"},
            "sampled_at": {"type": "date"},
            "row_index": {"type": "integer"},
            "row_text": {"type": "text"},
            "row": {"type": "object", "enabled": False},
        }
    }
}


def _row_text(row: Dict[str, Any], max_len: int = 20000) -> str:
    """Concatenate every string/number value in a row into one searchable blob."""
    parts = []
    for v in row.values():
        if isinstance(v, str):
            parts.append(v)
        elif isinstance(v, (int, float, bool)):
            parts.append(str(v))
    return " ".join(parts)[:max_len]


def iter_docs(input_path: Path, index: str, limit: Optional[int]) -> Iterator[Dict[str, Any]]:
    emitted = 0
    with open(input_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                doc = json.loads(line)
            except json.JSONDecodeError:
                continue

            dataset_id = doc.get("id")
            config = doc.get("config")
            split = doc.get("split")
            sampled_at = doc.get("sampled_at")
            rows = doc.get("rows") or []

            for i, row in enumerate(rows):
                if not isinstance(row, dict):
                    continue
                yield {
                    "_index": index,
                    "_id": f"{dataset_id}::{config}::{split}::{i}",
                    "_source": {
                        "dataset_id": dataset_id,
                        "config": config,
                        "split": split,
                        "sampled_at": sampled_at,
                        "row_index": i,
                        "row_text": _row_text(row),
                        "row": row,
                    },
                }
                emitted += 1
                if limit and emitted >= limit:
                    return


def run_import(input_path: str, es_url: str, index: str, limit: Optional[int], dry_run: bool) -> None:
    in_file = Path(input_path)
    if not in_file.exists():
        print(f"ERROR: file not found: {in_file}", file=sys.stderr)
        sys.exit(1)

    if dry_run:
        print(f"[import_samples_to_es] DRY RUN — first 5 docs from {in_file}:\n")
        for i, doc in enumerate(iter_docs(in_file, index, limit or 5)):
            src = doc["_source"]
            print(f"  _id={doc['_id']}")
            print(f"    dataset_id={src['dataset_id']}  config={src['config']}  split={src['split']}")
            print(f"    row_text[:120]={src['row_text'][:120]!r}")
            if i >= 4:
                break
        return

    from elasticsearch import Elasticsearch
    from elasticsearch.helpers import bulk

    es = Elasticsearch(es_url)
    if not es.ping():
        print(f"ERROR: cannot reach Elasticsearch at {es_url}", file=sys.stderr)
        sys.exit(1)

    if not es.indices.exists(index=index):
        es.indices.create(index=index, body=INDEX_MAPPING)
        print(f"[import_samples_to_es] Created index '{index}'")

    print(f"[import_samples_to_es] Reading {in_file.name} ...")
    # Row size varies wildly (wide datasets can have many large text columns),
    # so cap batches by BYTES, not just doc count — this sandbox ES only has a
    # 512MB heap, and its bulk circuit breaker rejects any single request over
    # roughly 51MB regardless of how few documents are in it.
    success, errors = bulk(
        es,
        iter_docs(in_file, index, limit),
        chunk_size=200,
        max_chunk_bytes=10 * 1024 * 1024,  # 10MB, well under the ~51MB breaker
        raise_on_error=False,
    )
    error_count = len(errors) if isinstance(errors, list) else errors
    print(f"[import_samples_to_es] Indexed {success:,} rows, {error_count} errors")

    es.indices.refresh(index=index)
    count = es.count(index=index)["count"]
    print(f"[import_samples_to_es] Index '{index}' now has {count:,} documents total")


def main() -> None:
    parser = argparse.ArgumentParser(description="Bulk-index V1 pipeline sampled rows into Elasticsearch.")
    parser.add_argument("--input", required=True, help="Path to samples.jsonl")
    parser.add_argument("--es-url", default=os.getenv("ES_URL", "http://localhost:9200"))
    parser.add_argument("--index", default="hf_dataset_rows")
    parser.add_argument("--limit", type=int, default=None, help="Max rows to index")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run_import(args.input, args.es_url, args.index, args.limit, args.dry_run)


if __name__ == "__main__":
    main()
