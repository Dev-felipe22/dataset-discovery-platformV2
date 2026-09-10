"""
Import enricher output from the V1 pipeline into the V2 DuckDB.

Reads metadata.parquet produced by the pipeline's step3_enricher, maps fields
to the V2 Dataset model, optionally pulls README text from the cards/ directory,
and bulk-upserts everything into DuckDB without wiping existing data.

Usage:
  python -m scripts.import_pipeline \
      --db-path   /data/demo_discovery.duckdb \
      --parquet   /pipeline_data/enricher/2026-04-22/metadata.parquet \
      --cards-dir /pipeline_data/enricher/2026-04-22/cards \
      [--limit 500] [--dry-run]
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pyarrow.parquet as pq

# --------------------------------------------------------------------------- #
# Field mapping helpers                                                         #
# --------------------------------------------------------------------------- #

TASK_TO_MODALITY: Dict[str, List[str]] = {
    "text-classification":          ["text"],
    "text-generation":              ["text"],
    "question-answering":           ["text"],
    "summarization":                ["text"],
    "translation":                  ["text"],
    "text2text-generation":         ["text"],
    "fill-mask":                    ["text"],
    "token-classification":         ["text"],
    "sentence-similarity":          ["text"],
    "feature-extraction":           ["text"],
    "image-classification":         ["image"],
    "object-detection":             ["image"],
    "image-segmentation":           ["image"],
    "depth-estimation":             ["image"],
    "image-to-image":               ["image"],
    "unconditional-image-generation": ["image"],
    "zero-shot-image-classification": ["image"],
    "image-captioning":             ["image", "text"],
    "image-to-text":                ["image", "text"],
    "visual-question-answering":    ["image", "text"],
    "document-question-answering":  ["image", "text"],
    "image-text-to-text":           ["image", "text"],
    "automatic-speech-recognition": ["audio"],
    "audio-classification":         ["audio"],
    "audio-to-audio":               ["audio"],
    "text-to-audio":                ["audio", "text"],
    "text-to-speech":               ["audio", "text"],
    "tabular-classification":       ["tabular"],
    "tabular-regression":           ["tabular"],
    "tabular-to-text":              ["tabular", "text"],
    "video-classification":         ["video"],
}


def _modalities_from_tasks(task_categories: Any) -> Optional[List[str]]:
    if not task_categories:
        return None
    tasks = list(task_categories) if not isinstance(task_categories, list) else task_categories
    seen, result = set(), []
    for task in tasks:
        for m in TASK_TO_MODALITY.get(str(task), []):
            if m not in seen:
                seen.add(m)
                result.append(m)
    return result or None


def _size_class_from_tags(tags: Any) -> Optional[str]:
    for tag in (list(tags) if tags is not None else []):
        s = str(tag)
        if s.startswith("size_categories:"):
            return s.split(":", 1)[1]
    return None


def _clean_list(value: Any) -> Optional[List[str]]:
    if value is None:
        return None
    items = list(value) if not isinstance(value, list) else value
    return [str(x) for x in items if x is not None] or None


def _str(value: Any) -> Optional[str]:
    if value is None:
        return None
    s = str(value).strip()
    return s if s and s != "nan" else None


def _token_count(text: Optional[str]) -> int:
    return len((text or "").split())


def _readme_score(text: Optional[str]) -> float:
    tokens = _token_count(text)
    return min(tokens / 50.0, 1.0) if tokens > 0 else 0.0


def _readme_text(dataset_id: str, cards_dir: Optional[Path]) -> Optional[str]:
    if not cards_dir or not cards_dir.exists():
        return None
    fname = dataset_id.replace("/", "__") + ".README.md"
    card_file = cards_dir / fname
    if card_file.exists():
        try:
            return card_file.read_text(encoding="utf-8", errors="replace").strip() or None
        except Exception:
            return None
    return None


# --------------------------------------------------------------------------- #
# Row conversion                                                                #
# --------------------------------------------------------------------------- #

def row_to_dataset(row: Dict[str, Any], cards_dir: Optional[Path], indexed_at: datetime):
    from src.storage.models import Dataset

    dataset_id = _str(row.get("id"))
    if not dataset_id:
        return None

    name       = _str(row.get("name")) or dataset_id.split("/")[-1]
    is_private = bool(row.get("is_private"))
    is_gated   = bool(row.get("is_gated"))
    tags       = _clean_list(row.get("tags"))
    tasks      = _clean_list(row.get("task_categories"))
    languages  = _clean_list(row.get("languages"))
    downloads  = row.get("downloads")
    readme     = _readme_text(dataset_id, cards_dir)

    eligibility: List[str] = []
    if is_gated:
        eligibility.append("gated")

    qs: Dict[str, Any] = {}
    if downloads is not None:
        try:
            qs["downloads"] = int(float(downloads))
        except (ValueError, TypeError):
            pass

    return Dataset(
        id=dataset_id,
        source="hf",
        ns_local_id=name,
        title=name.replace("-", " ").replace("_", " ").title(),
        description=None,
        readme_text=readme,
        tags=tags,
        modalities=_modalities_from_tasks(tasks),
        license_class=_str(row.get("license")),
        access_class="private" if is_private else "public",
        size_class=_size_class_from_tags(tags),
        languages=languages,
        last_seen_lm=None,
        readme_sha256=_str(row.get("readme_sha256")),
        fingerprint=f"pipeline:{dataset_id}",
        indexed_at=indexed_at,
        readme_tokens=_token_count(readme),
        readme_score=_readme_score(readme),
        eligibility_flags=eligibility or None,
        quality_signals=qs or None,
        schema_hint=None,
    )


# --------------------------------------------------------------------------- #
# Main import logic                                                             #
# --------------------------------------------------------------------------- #

def run_import(
    db_path: str,
    parquet_path: str,
    cards_dir: Optional[str],
    limit: Optional[int],
    dry_run: bool,
) -> None:
    parquet_file = Path(parquet_path)
    if not parquet_file.exists():
        print(f"ERROR: parquet file not found: {parquet_file}", file=sys.stderr)
        sys.exit(1)

    cards_path = Path(cards_dir) if cards_dir else None

    print(f"[import_pipeline] Reading {parquet_file.name} ...")
    table = pq.read_table(parquet_file)
    total_rows = len(table)
    print(f"[import_pipeline] {total_rows} rows in parquet")

    if limit:
        table = table.slice(0, limit)
        print(f"[import_pipeline] Limiting to {limit} rows")

    indexed_at = datetime.now(timezone.utc)
    datasets, skipped = [], 0

    rows = table.to_pylist()
    for row in rows:
        if _str(row.get("enrich_error")):
            skipped += 1
            continue
        ds = row_to_dataset(row, cards_path, indexed_at)
        if ds:
            datasets.append(ds)
        else:
            skipped += 1

    print(f"[import_pipeline] {len(datasets)} valid / {skipped} skipped (errored or missing id)")

    if not datasets:
        print("[import_pipeline] Nothing to load.", file=sys.stderr)
        sys.exit(1)

    if dry_run:
        print("\n[import_pipeline] DRY RUN — first 10 entries:")
        for d in datasets[:10]:
            print(f"  {d.id:50s}  modalities={d.modalities}  license={d.license_class}")
        return

    from src.storage import DuckDBStorage
    db = DuckDBStorage(db_path)
    db.init()

    batch_size = 500
    for i in range(0, len(datasets), batch_size):
        batch = datasets[i : i + batch_size]
        db.bulk_upsert_datasets(batch)
        print(f"[import_pipeline] Inserted {min(i + batch_size, len(datasets))}/{len(datasets)} ...")

    db.refresh_dataset_state_view()
    db.rebuild_search_index()
    print(f"[import_pipeline] Done. Loaded {len(datasets)} datasets into {db_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Import V1 pipeline enricher output into V2 DuckDB.")
    parser.add_argument("--db-path",   default="data/demo_discovery.duckdb")
    parser.add_argument("--parquet",   required=True, help="Path to metadata.parquet")
    parser.add_argument("--cards-dir", default=None,  help="Path to enricher cards/ directory (README files)")
    parser.add_argument("--limit",     type=int, default=None, help="Max rows to import")
    parser.add_argument("--dry-run",   action="store_true")
    args = parser.parse_args()
    run_import(args.db_path, args.parquet, args.cards_dir, args.limit, args.dry_run)


if __name__ == "__main__":
    main()
