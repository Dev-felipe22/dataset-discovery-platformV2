"""
Import a datasets.jsonl catalog file from the V1 pipeline into the V2 DuckDB.

Streams the file line by line — safe for large files (900k+ rows).
Batches inserts at 1000 rows for efficiency.

Usage:
  python -m scripts.import_jsonl \
      --db-path /data/demo_discovery.duckdb \
      --input   /v1_pipeline_data/catalog/2026-07-24/datasets.jsonl \
      [--limit 1000] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

TASK_TO_MODALITY: Dict[str, List[str]] = {
    "text-classification":            ["text"],
    "text-generation":                ["text"],
    "question-answering":             ["text"],
    "summarization":                  ["text"],
    "translation":                    ["text"],
    "text2text-generation":           ["text"],
    "fill-mask":                      ["text"],
    "token-classification":           ["text"],
    "sentence-similarity":            ["text"],
    "feature-extraction":             ["text"],
    "image-classification":           ["image"],
    "object-detection":               ["image"],
    "image-segmentation":             ["image"],
    "depth-estimation":               ["image"],
    "image-to-image":                 ["image"],
    "unconditional-image-generation": ["image"],
    "zero-shot-image-classification": ["image"],
    "image-captioning":               ["image", "text"],
    "image-to-text":                  ["image", "text"],
    "visual-question-answering":      ["image", "text"],
    "document-question-answering":    ["image", "text"],
    "image-text-to-text":             ["image", "text"],
    "automatic-speech-recognition":   ["audio"],
    "audio-classification":           ["audio"],
    "audio-to-audio":                 ["audio"],
    "text-to-audio":                  ["audio", "text"],
    "text-to-speech":                 ["audio", "text"],
    "tabular-classification":         ["tabular"],
    "tabular-regression":             ["tabular"],
    "tabular-to-text":                ["tabular", "text"],
    "video-classification":           ["video"],
}


def _modalities(task_categories: Any) -> Optional[List[str]]:
    if not task_categories:
        return None
    seen, result = set(), []
    for task in (task_categories or []):
        for m in TASK_TO_MODALITY.get(str(task), []):
            if m not in seen:
                seen.add(m)
                result.append(m)
    return result or None


def _size_class(size_categories: Any) -> Optional[str]:
    if not size_categories:
        return None
    items = list(size_categories) if not isinstance(size_categories, list) else size_categories
    return str(items[0]) if items else None


def _clean_list(value: Any) -> Optional[List[str]]:
    if value is None:
        return None
    items = [str(x) for x in (value if isinstance(value, list) else [value]) if x is not None]
    return items or None


def _token_count(text: Optional[str]) -> int:
    return len((text or "").split())


def _readme_score(text: Optional[str]) -> float:
    tokens = _token_count(text)
    return min(tokens / 50.0, 1.0) if tokens > 0 else 0.0


def row_to_dataset(row: Dict[str, Any], indexed_at: datetime):
    from src.storage.models import Dataset

    dataset_id = row.get("id")
    if not dataset_id or not isinstance(dataset_id, str):
        return None

    is_private = bool(row.get("private", False))
    is_gated   = bool(row.get("gated", False))
    is_disabled = bool(row.get("disabled", False))
    if is_disabled:
        return None

    tasks      = row.get("task_categories") or []
    tags       = _clean_list(row.get("tags"))
    languages  = _clean_list(row.get("language"))
    name       = row.get("name") or dataset_id.split("/")[-1]
    title      = row.get("pretty_name") or name
    description = row.get("description")

    eligibility: List[str] = []
    if is_gated:
        eligibility.append("gated")

    qs: Dict[str, Any] = {}
    if row.get("downloads") is not None:
        qs["downloads"] = row["downloads"]
    if row.get("likes") is not None:
        qs["likes"] = row["likes"]

    # Extract schema hint from dataset_info.features if present
    schema_hint = None
    dataset_info = row.get("dataset_info")
    if isinstance(dataset_info, dict):
        features = dataset_info.get("features")
        if features:
            schema_hint = json.dumps(features, ensure_ascii=False)

    last_modified = row.get("last_modified")
    last_seen_lm = None
    if last_modified:
        try:
            last_seen_lm = datetime.fromisoformat(last_modified.replace("Z", "+00:00"))
        except Exception:
            pass

    return Dataset(
        id=dataset_id,
        source="hf",
        ns_local_id=name,
        title=title,
        description=description,
        readme_text=None,
        tags=tags,
        modalities=_modalities(tasks),
        license_class=row.get("license"),
        access_class="private" if is_private else "public",
        size_class=_size_class(row.get("size_categories")),
        languages=languages,
        last_seen_lm=last_seen_lm,
        fingerprint=f"v1jsonl:{dataset_id}:{row.get('sha', '')}",
        indexed_at=indexed_at,
        readme_tokens=_token_count(description),
        readme_score=_readme_score(description),
        eligibility_flags=eligibility or None,
        quality_signals=qs or None,
        schema_hint=schema_hint,
    )


def run_import(db_path: str, input_path: str, limit: Optional[int], dry_run: bool) -> None:
    input_file = Path(input_path)
    if not input_file.exists():
        print(f"ERROR: file not found: {input_file}", file=sys.stderr)
        sys.exit(1)

    print(f"[import_jsonl] Reading {input_file.name} ...")

    indexed_at = datetime.now(timezone.utc)
    batch: List[Any] = []
    total_loaded = 0
    total_skipped = 0
    BATCH_SIZE = 1000

    if dry_run:
        print("[import_jsonl] DRY RUN — first 10 valid entries:\n")

    storage = None
    if not dry_run:
        from src.storage import DuckDBStorage
        storage = DuckDBStorage(db_path)
        storage.init()

    with open(input_file, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                total_skipped += 1
                continue

            ds = row_to_dataset(row, indexed_at)
            if ds is None:
                total_skipped += 1
                continue

            if dry_run:
                if total_loaded < 10:
                    print(f"  {ds.id:55s} modalities={ds.modalities}  license={ds.license_class}  size={ds.size_class}")
                total_loaded += 1
                if limit and total_loaded >= limit:
                    break
                continue

            batch.append(ds)
            if len(batch) >= BATCH_SIZE:
                storage.bulk_upsert_datasets(batch)
                total_loaded += len(batch)
                batch.clear()
                if total_loaded % 10000 == 0:
                    print(f"[import_jsonl] {total_loaded:,} loaded, {total_skipped:,} skipped ...")

            if limit and (total_loaded + len(batch)) >= limit:
                break

    if not dry_run and batch:
        storage.bulk_upsert_datasets(batch)
        total_loaded += len(batch)

    print(f"\n[import_jsonl] {total_loaded:,} loaded / {total_skipped:,} skipped")

    if not dry_run:
        print("[import_jsonl] Refreshing state view and search index ...")
        storage.refresh_dataset_state_view()
        storage.rebuild_search_index()
        print(f"[import_jsonl] Done. DB: {db_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Import datasets.jsonl from V1 pipeline into V2 DuckDB.")
    parser.add_argument("--db-path", default="data/demo_discovery.duckdb")
    parser.add_argument("--input",   required=True, help="Path to datasets.jsonl")
    parser.add_argument("--limit",   type=int, default=None, help="Max rows to import")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run_import(args.db_path, args.input, args.limit, args.dry_run)


if __name__ == "__main__":
    main()
