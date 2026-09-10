"""
Load datasets from a JSON catalog file into an existing DuckDB without wiping it.

The JSON file must have a top-level "datasets" list. Each entry must have:
  - id          (str, required) e.g. "owner/dataset-name"
  - source      (str, required) e.g. "hf"
  - ns_local_id (str, required) e.g. "dataset-name"

All other fields are optional. See README or tests/fixtures/sample_pipeline_output.json
for a full example.

Usage:
  python -m scripts.load_catalog --db-path data/demo_discovery.duckdb --input path/to/file.json
  python -m scripts.load_catalog --db-path data/demo_discovery.duckdb --input path/to/file.json --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

REQUIRED_FIELDS = {"id", "source", "ns_local_id"}


def _token_count(text: str | None) -> int:
    return len((text or "").split())


def _readme_score(text: str | None) -> float:
    tokens = _token_count(text)
    if tokens <= 0:
        return 0.0
    return min(tokens / 50.0, 1.0)


def validate_entries(entries: List[Dict[str, Any]]) -> Tuple[List[Dict], List[str]]:
    valid, errors = [], []
    seen_ids: set = set()
    for i, entry in enumerate(entries):
        label = entry.get("id") or f"entry[{i}]"
        missing = REQUIRED_FIELDS - entry.keys()
        if missing:
            errors.append(f"  [{label}] missing required fields: {sorted(missing)}")
            continue
        if not isinstance(entry["id"], str) or not entry["id"].strip():
            errors.append(f"  [{label}] 'id' must be a non-empty string")
            continue
        if entry["id"] in seen_ids:
            errors.append(f"  [{label}] duplicate id — skipped")
            continue
        seen_ids.add(entry["id"])
        valid.append(entry)
    return valid, errors


def entry_to_dataset(item: Dict[str, Any], indexed_at: datetime):
    from src.storage.models import Dataset

    readme_text = item.get("readme_text")
    return Dataset(
        id=item["id"],
        source=item["source"],
        ns_local_id=item["ns_local_id"],
        title=item.get("title"),
        description=item.get("description"),
        readme_text=readme_text,
        tags=item.get("tags"),
        modalities=item.get("modalities"),
        license_class=item.get("license_class"),
        access_class=item.get("access_class", "public"),
        size_class=item.get("size_class"),
        languages=item.get("languages"),
        fingerprint=item.get("fingerprint") or f"catalog:{item['id']}",
        indexed_at=item.get("indexed_at") or indexed_at,
        readme_tokens=_token_count(readme_text),
        readme_score=_readme_score(readme_text),
        quality_signals=item.get("quality_signals"),
        eligibility_flags=item.get("eligibility_flags"),
        schema_hint=item.get("schema_hint"),
    )


def load(db_path: str, input_path: str, dry_run: bool = False) -> None:
    input_file = Path(input_path)
    if not input_file.exists():
        print(f"ERROR: input file not found: {input_file}", file=sys.stderr)
        sys.exit(1)

    with open(input_file, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    if "datasets" not in data or not isinstance(data["datasets"], list):
        print('ERROR: JSON must have a top-level "datasets" list.', file=sys.stderr)
        sys.exit(1)

    entries = data["datasets"]
    print(f"[load_catalog] Found {len(entries)} entries in {input_file.name}")

    valid, errors = validate_entries(entries)

    if errors:
        print(f"\n[load_catalog] Validation errors ({len(errors)}):")
        for e in errors:
            print(e)

    print(f"[load_catalog] {len(valid)} valid / {len(errors)} skipped")

    if not valid:
        print("[load_catalog] Nothing to load.", file=sys.stderr)
        sys.exit(1)

    if dry_run:
        print("\n[load_catalog] DRY RUN — no changes written. Valid entries:")
        for v in valid:
            print(f"  {v['id']}")
        return

    from src.storage import DuckDBStorage

    db = DuckDBStorage(db_path)
    db.init()

    indexed_at = datetime.now(timezone.utc)
    datasets = [entry_to_dataset(e, indexed_at) for e in valid]

    db.bulk_upsert_datasets(datasets)
    db.refresh_dataset_state_view()
    db.rebuild_search_index()

    print(f"[load_catalog] Loaded {len(datasets)} datasets into {db_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Add datasets from a JSON file into an existing DuckDB.")
    parser.add_argument("--db-path", default="data/demo_discovery.duckdb", help="Path to the DuckDB file.")
    parser.add_argument("--input", required=True, help="Path to the JSON catalog file.")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print entries without writing.")
    args = parser.parse_args()
    load(args.db_path, args.input, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
