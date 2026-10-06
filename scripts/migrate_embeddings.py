"""
One-off migration: move embeddings from the legacy dataset_embeddings table
(vectors stored as JSON text) into dataset_vectors (raw float32 BLOBs).

Why: search used to decode every JSON vector on every query, which took
minutes per query at catalog scale. Binary vectors load in seconds and
are cached in memory by the storage layer afterward.

Safe to rerun: if there is no legacy table, it does nothing. Rows are
upserted, and the legacy table is dropped only after the row counts match.

Usage:
  python -m scripts.migrate_embeddings --db-path data/discovery_real.duckdb
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np


def migrate(db_path: str, *, chunk_size: int = 10000) -> int:
    from src.storage import DuckDBStorage

    db = DuckDBStorage(db_path)
    db.init()  # creates dataset_vectors if missing; leaves the legacy table alone
    con = db.conn

    has_legacy = con.execute(
        "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'dataset_embeddings'"
    ).fetchone()[0]
    if not has_legacy:
        print("[migrate_embeddings] No legacy dataset_embeddings table found. Nothing to migrate.")
        return 0

    legacy_count = con.execute("SELECT COUNT(*) FROM dataset_embeddings").fetchone()[0]
    print(f"[migrate_embeddings] Converting {legacy_count} legacy row(s) to binary vectors...")

    # A separate cursor reads the legacy rows while the main connection writes,
    # so the two don't interfere with each other.
    reader = con.cursor()
    reader.execute("SELECT dataset_id, model, embedding, updated_at FROM dataset_embeddings")
    migrated = 0
    while True:
        batch = reader.fetchmany(chunk_size)
        if not batch:
            break
        params = [
            (
                did,
                model,
                np.asarray(json.loads(emb), dtype=np.float32).tobytes(),
                updated_at,
            )
            for did, model, emb, updated_at in batch
        ]
        # Bulk path on purpose: per-row upserts are ~500x slower in DuckDB.
        db.upsert_vector_blobs(params)
        migrated += len(params)
        print(f"[migrate_embeddings] {migrated}/{legacy_count} converted")
    reader.close()

    new_count = con.execute("SELECT COUNT(*) FROM dataset_vectors").fetchone()[0]
    if new_count < legacy_count:
        print(
            f"[migrate_embeddings] ERROR: only {new_count} of {legacy_count} rows present in "
            "dataset_vectors. Legacy table left in place; nothing dropped.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    con.execute("DROP TABLE dataset_embeddings")
    con.execute("CHECKPOINT")
    print(f"[migrate_embeddings] Done. {new_count} vector(s) in dataset_vectors; legacy table dropped.")
    return new_count


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert legacy JSON embeddings to binary vectors.")
    parser.add_argument("--db-path", required=True, help="Path to the DuckDB file to migrate.")
    args = parser.parse_args()
    migrate(args.db_path)


if __name__ == "__main__":
    main()
