"""Persistence layer — Supabase (Postgres) backend.

Every function here takes plain flat dicts (already shaped by
youtube_client.py's flatten_* functions) and a destination table name, so
nothing outside this file needs to know or care that storage is a database
rather than CSV. Connects via SUPABASE_DB_URL — use the Transaction-mode
pooler connection string (port 6543), not the direct connection (port 5432):
GitHub Actions jobs are short-lived and the pooler avoids connection-
exhaustion issues from many short-lived jobs each opening their own connection.

Identity tables (channels, videos, video_categories) and snapshot tables
(channel_snapshots, video_snapshots) both go through the same upsert path —
upserting is safe either way: identity rows never actually duplicate (callers
only pass already-unseen ids), and snapshot rows keyed on (id, captured_at)
would only conflict on an exact rerun with the same timestamp, which upsert
handles gracefully instead of erroring.

Previous CSV-backed version is in git history (see the "sabith" branch commits
before this one) if ever needed for reference.
"""
import os
from urllib.parse import urlparse, urlunparse

import psycopg2
from psycopg2 import sql
from psycopg2.extras import execute_batch
from dotenv import load_dotenv

load_dotenv()

DB_URL = os.environ["SUPABASE_DB_URL"]

PRIMARY_KEYS = {
    "channels": ["channel_id"],
    "channel_snapshots": ["channel_id", "captured_at"],
    "videos": ["video_id"],
    "video_snapshots": ["video_id", "captured_at"],
    "video_categories": ["category_id"],
}

# Columns needing an explicit cast — flatten_* functions hand back plain
# strings (including "" for missing fields), and Postgres won't implicitly
# cast text to bigint/boolean/timestamptz for a typed column.
COLUMN_CASTS = {
    "channel_published_at": "timestamptz",
    "captured_at": "timestamptz",
    "published_at": "timestamptz",
    "live_actual_start_time": "timestamptz",
    "live_actual_end_time": "timestamptz",
    "subscriber_count": "bigint",
    "view_count": "bigint",
    "video_count": "bigint",
    "like_count": "bigint",
    "comment_count": "bigint",
    "live_concurrent_viewers": "bigint",
    "hidden_subscriber_count": "boolean",
    "made_for_kids": "boolean",
}


def connect():
    """Shared connection helper — strips Supabase's ?pgbouncer=true query
    hint first, since plain psycopg2/libpq doesn't recognize it as a valid
    connection parameter (that hint is meant for ORMs like Prisma; it's a
    no-op for a raw psycopg2 connection)."""
    clean_url = urlunparse(urlparse(DB_URL)._replace(query=""))
    return psycopg2.connect(clean_url)


_connect = connect  # internal alias, kept for the calls below


def _clean(value):
    """Blank strings from flatten_* (a missing API field) become SQL NULL —
    an empty string is not a valid bigint/boolean/timestamptz literal."""
    return None if value == "" else value


def _upsert(rows: list[dict], table: str) -> None:
    if not rows:
        return
    columns = list(rows[0].keys())
    pk = PRIMARY_KEYS[table]
    update_columns = [c for c in columns if c not in pk]

    col_idents = sql.SQL(", ").join(sql.Identifier(c) for c in columns)
    val_placeholders = sql.SQL(", ").join(
        sql.SQL("%s::{}").format(sql.SQL(COLUMN_CASTS[c])) if c in COLUMN_CASTS else sql.SQL("%s")
        for c in columns
    )
    conflict_cols = sql.SQL(", ").join(sql.Identifier(c) for c in pk)

    if update_columns:
        conflict_action = sql.SQL("DO UPDATE SET {}").format(
            sql.SQL(", ").join(
                sql.SQL("{0} = EXCLUDED.{0}").format(sql.Identifier(c)) for c in update_columns
            )
        )
    else:
        conflict_action = sql.SQL("DO NOTHING")

    query = sql.SQL("INSERT INTO {} ({}) VALUES ({}) ON CONFLICT ({}) {}").format(
        sql.Identifier(table), col_idents, val_placeholders, conflict_cols, conflict_action
    )
    values = [[_clean(row.get(c)) for c in columns] for row in rows]

    conn = _connect()
    try:
        with conn.cursor() as cur:
            execute_batch(cur, query, values)
        conn.commit()
    finally:
        conn.close()


def append_rows(rows: list[dict], table: str) -> None:
    """Adds rows to a growing table."""
    _upsert(rows, table)


def write_rows(rows: list[dict], table: str) -> None:
    """One-time/occasional reference data (video categories) — upsert covers
    this fine since YouTube's category list rarely changes."""
    _upsert(rows, table)


def load_active_video_ids(table: str, since_date: str) -> set[str]:
    """Video ids published on/after since_date — so the daily poll knows which
    already-known videos still need a fresh snapshot, not just newly-discovered
    ones. Returns an empty set on the very first run against an empty table."""
    conn = _connect()
    try:
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL("SELECT video_id FROM {} WHERE published_at >= %s::timestamptz").format(
                    sql.Identifier(table)
                ),
                (since_date,),
            )
            return {row[0] for row in cur.fetchall()}
    finally:
        conn.close()


def load_known_ids(table: str, id_column: str) -> set[str]:
    """Every id already recorded in an identity table — used so identity rows
    are only ever written once per entity. Returns an empty set on the very
    first run against an empty table."""
    conn = _connect()
    try:
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL("SELECT {} FROM {}").format(sql.Identifier(id_column), sql.Identifier(table))
            )
            return {row[0] for row in cur.fetchall()}
    finally:
        conn.close()
