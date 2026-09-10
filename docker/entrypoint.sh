#!/bin/sh
set -e

if [ ! -f "$DUCKDB_PATH" ]; then
    echo "[entrypoint] DB not found at $DUCKDB_PATH - seeding from fixture..."
    python -m scripts.create_demo_db --db-path "$DUCKDB_PATH"
    echo "[entrypoint] DB seeded."
else
    echo "[entrypoint] DB found at $DUCKDB_PATH - skipping seed."
fi

exec "$@"