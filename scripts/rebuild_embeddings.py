"""
Recompute dense embeddings (all-MiniLM-L6-v2) for every dataset in a DuckDB
catalog, and persist them so /v2/search_embedding (and the "minilm" eval
engine) have something to rank against.

CLI-only, on purpose — same pattern as load_catalog.py and create_demo_db.py.
There is no UI button and no admin HTTP endpoint for this: it's an operator
step, not something an end user of the running app ever needs to trigger.

Run it once against a fresh catalog, and run it again any time the catalog's
content changes meaningfully (new datasets, or edited titles/descriptions/
READMEs/tags on existing ones) — there's no automatic staleness detection,
this doesn't run on a schedule, and it always re-embeds the full catalog
rather than just what changed. Results are written straight to the DuckDB
file, so they persist across app/container restarts; you don't need to rerun
this every time you start the app, only after the index itself changes.

At demo scale (a handful of datasets) this finishes instantly. At real-catalog
scale (hundreds of thousands of datasets) it can take a long time on CPU —
minutes on GPU if one is visible to the process, hours otherwise. It uses
CUDA automatically when torch reports it available (see src/tools/embeddings
.get_device()); it prints which device it picked at startup, so you can
confirm the GPU is actually being used before committing to a long run
rather than finding out afterward. Before committing to a full run against a
large catalog either way, measure real throughput first with --limit, e.g.:

  python -m scripts.rebuild_embeddings --db-path /data/demo_discovery.duckdb --limit 2000

...time that, then multiply out to estimate the full run before starting it
unattended. Processing happens in chunks (see --chunk-size) so progress
prints as it goes instead of the terminal going silent for hours, and each
chunk is committed to disk as it finishes rather than held until the very end.

Interrupted partway through a large run? Just rerun the same command — by
default it skips datasets that already have an embedding and only does the
rest, rather than redoing the whole catalog. Pass --force if you actually
want to recompute everything (e.g. after switching models).

Usage:
  python -m scripts.rebuild_embeddings
  python -m scripts.rebuild_embeddings --db-path data/discovery.duckdb
  python -m scripts.rebuild_embeddings --db-path data/discovery.duckdb --limit 2000
  python -m scripts.rebuild_embeddings --db-path data/discovery.duckdb --batch-size 256
  python -m scripts.rebuild_embeddings --db-path data/discovery.duckdb --force
"""

from __future__ import annotations

import argparse
import time
from typing import Optional


def rebuild(
    db_path: str,
    *,
    limit: Optional[int] = None,
    chunk_size: int = 2000,
    batch_size: Optional[int] = None,
    force: bool = False,
) -> int:
    from src.storage import DuckDBStorage
    from src.tools import embeddings

    db = DuckDBStorage(db_path)
    db.init()

    device = embeddings.get_device()
    effective_batch_size = batch_size or embeddings.DEFAULT_BATCH_SIZE
    print(f"[rebuild_embeddings] Device: {device} (batch_size={effective_batch_size})")
    if device == "cpu":
        print(
            "[rebuild_embeddings] No GPU visible to this process — running on CPU. "
            "If you expected a GPU, check `--gpus all` (Docker) or that torch.cuda.is_available() "
            "is True in this environment before running a large job."
        )

    inputs = db.list_dataset_embedding_inputs()
    if not force:
        already_done = db.get_embedded_dataset_ids()
        if already_done:
            before = len(inputs)
            inputs = [i for i in inputs if i["id"] not in already_done]
            skipped = before - len(inputs)
            if skipped:
                print(
                    f"[rebuild_embeddings] Resuming: {skipped} dataset(s) already embedded, "
                    f"{len(inputs)} left to go. (--force to redo everything.)"
                )
    if limit:
        inputs = inputs[:limit]
    if not inputs:
        print(f"[rebuild_embeddings] Nothing to embed in {db_path} (already up to date, or empty).")
        return 0

    total = len(inputs)
    print(
        f"[rebuild_embeddings] Embedding {total} dataset(s) with {embeddings.EMBEDDING_MODEL_NAME} "
        f"in chunks of {chunk_size}…"
    )
    t0 = time.time()
    done = 0
    for start in range(0, total, chunk_size):
        chunk = inputs[start : start + chunk_size]
        vectors = embeddings.embed_texts([i["text"] for i in chunk], batch_size=effective_batch_size)
        rows = [{"dataset_id": i["id"], "embedding": v} for i, v in zip(chunk, vectors)]
        db.upsert_dataset_embeddings(rows, model=embeddings.EMBEDDING_MODEL_NAME)
        done += len(chunk)

        elapsed = time.time() - t0
        rate = done / elapsed if elapsed > 0 else 0.0
        remaining_min = ((total - done) / rate / 60) if rate > 0 else float("nan")
        print(
            f"[rebuild_embeddings] {done}/{total} done "
            f"({rate:.1f}/s, ~{remaining_min:.1f} min remaining)"
        )

    elapsed = time.time() - t0
    status = db.embedding_index_status()
    print(
        f"[rebuild_embeddings] Saved {done} embedding(s) to {db_path} in {elapsed / 60:.1f} min "
        f"({status['embedded_count']}/{status['total_datasets']} datasets now covered)."
    )
    return done


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recompute dense (all-MiniLM-L6-v2) embeddings for every dataset in a DuckDB catalog."
    )
    parser.add_argument("--db-path", default="data/demo_discovery.duckdb", help="Path to the DuckDB file.")
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Only embed the first N datasets — use this to measure throughput before a full run.",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=2000,
        help="Datasets embedded and committed per batch (default 2000).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="Model encode() batch size (default 64). Larger values use a GPU more fully; "
        "if you're on CPU, leave this alone or lower it.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Recompute every dataset's embedding, including ones already done. "
        "Default behavior skips already-embedded datasets so an interrupted run can resume.",
    )
    args = parser.parse_args()
    rebuild(
        args.db_path,
        limit=args.limit,
        chunk_size=args.chunk_size,
        batch_size=args.batch_size,
        force=args.force,
    )


if __name__ == "__main__":
    main()
