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
import time
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
    "video_metadata_changes": ["video_id", "observed_at"],
}

# Columns needing an explicit cast — flatten_* functions hand back plain
# strings (including "" for missing fields), and Postgres won't implicitly
# cast text to bigint/boolean/timestamptz for a typed column.
COLUMN_CASTS = {
    "channel_published_at": "timestamptz",
    "captured_at": "timestamptz",
    "observed_at": "timestamptz",
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


# The pooler resolves to several addresses. Without an explicit timeout libpq
# waits out the operating system's TCP timeout on each one in turn, so a
# transient network problem costs minutes of hanging before the job even
# reports failure. A short timeout with a few backed-off retries turns that
# into a brief pause that usually recovers on its own.
CONNECT_TIMEOUT = int(os.environ.get("PG_CONNECT_TIMEOUT", "15"))
CONNECT_RETRIES = int(os.environ.get("PG_CONNECT_RETRIES", "4"))


def connect():
    """Shared connection helper.

    Strips Supabase's ?pgbouncer=true query hint first, since plain
    psycopg2/libpq doesn't recognize it as a valid connection parameter (that
    hint is meant for ORMs like Prisma; it's a no-op for a raw psycopg2
    connection).

    Retries a connection that times out or is refused. Scheduled jobs reach the
    pooler across the public internet and an occasional unreachable moment is
    normal; without a retry a single blip fails a whole collection or archive
    run, and for collection that leaves a permanent hole in the history."""
    clean_url = urlunparse(urlparse(DB_URL)._replace(query=""))
    last = None
    for attempt in range(CONNECT_RETRIES):
        try:
            return psycopg2.connect(clean_url, connect_timeout=CONNECT_TIMEOUT)
        except psycopg2.OperationalError as e:
            last = e
            if attempt == CONNECT_RETRIES - 1:
                break
            wait = 5 * (2 ** attempt)
            print(f"  database unreachable (attempt {attempt + 1}/{CONNECT_RETRIES}); "
                  f"retrying in {wait}s")
            time.sleep(wait)
    raise last


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


def ensure_snapshot_partitions(days_ahead: int = 14) -> int:
    """Creates any missing daily partitions of video_snapshots, up to days_ahead.

    video_snapshots is partitioned by day on captured_at so that old data can be
    archived and dropped instantly instead of needing a VACUUM FULL. A write
    whose day has no partition would land in the default partition, which then
    blocks that day's partition from ever being created — so this runs at the
    start of every collection run rather than relying on a schedule.

    Returns the number created; normally zero, since the nightly archive keeps
    the window topped up. Never fails a run: the collector must keep collecting
    even if partition maintenance has a problem, because the default partition
    will catch the writes either way."""
    conn = _connect()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT ensure_snapshot_partitions(%s)", (days_ahead,))
            made = cur.fetchone()[0]
        conn.commit()
        return made
    except psycopg2.Error as e:
        print(f"  WARNING: could not ensure partitions ({type(e).__name__}); "
              f"writes will fall back to the default partition")
        return 0
    finally:
        conn.close()


def load_roster_mapping(table: str = "channels") -> dict[str, tuple[str, bool]]:
    """Maps every roster key we could be handed to (channel_id, active).

    A channel_handles.txt entry is either an @handle or a raw 'UC...' id, so
    both forms are keyed here. This is what lets a full run refresh known
    channels by id in batches of fifty instead of resolving each handle
    individually, and lets it skip inactive channels entirely rather than
    paying to discover they are dead again.

    Returns an empty mapping if the custom_url column has not been added yet,
    so the caller falls back to one-at-a-time resolution rather than failing."""
    conn = _connect()
    try:
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL("SELECT channel_id, custom_url, active FROM {}").format(
                    sql.Identifier(table)))
            mapping = {}
            for channel_id, custom_url, active in cur.fetchall():
                mapping[channel_id.lower()] = (channel_id, active)
                if custom_url:
                    mapping[custom_url] = (channel_id, active)
            return mapping
    except psycopg2.errors.UndefinedColumn:
        return {}
    finally:
        conn.close()


def load_metadata_shas(video_ids: list[str]) -> dict[str, str | None]:
    """video_id -> fingerprint of its last observed metadata, None if never set.

    Only the hash is loaded, not the fields themselves: at fifty thousand
    tracked videos the titles and descriptions would be tens of megabytes to
    move every run, where the hashes are about two."""
    if not video_ids:
        return {}
    conn = _connect()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT video_id, metadata_sha FROM videos WHERE video_id = ANY(%s)",
                (list(video_ids),))
            return {r[0]: r[1] for r in cur.fetchall()}
    finally:
        conn.close()


def save_metadata_shas(shas: dict[str, str]) -> None:
    """Record the latest fingerprint per video."""
    if not shas:
        return
    conn = _connect()
    try:
        with conn.cursor() as cur:
            execute_batch(cur, "UPDATE videos SET metadata_sha = %s WHERE video_id = %s",
                          [(v, k) for k, v in shas.items()], page_size=500)
        conn.commit()
    finally:
        conn.close()


def load_active_channels(table: str = "channels") -> list[tuple[str, str]]:
    """(channel_id, uploads_playlist_id) for every channel still worth polling.

    Discovery needs both: the channel id addresses the free RSS feed, the
    playlist id addresses the metered API call used when RSS cannot vouch for
    a channel. Same exclusions as load_channel_playlist_ids -- inactive
    channels and rows with no uploads playlist."""
    conn = _connect()
    try:
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL("SELECT channel_id, uploads_playlist_id FROM {} "
                        "WHERE uploads_playlist_id IS NOT NULL AND active").format(
                    sql.Identifier(table)))
            return [(r[0], r[1]) for r in cur.fetchall()]
    finally:
        conn.close()


def load_channel_playlist_ids(table: str = "channels") -> list[str]:
    """Uploads-playlist ids for every channel still worth polling — lets runs
    find new uploads without re-resolving each channel.

    Two exclusions. Rows with a null playlist id (channels that never had an
    uploads playlist) are skipped because there is nothing to discover from
    them. Rows flagged active = false are skipped because they have stopped
    uploading: an activity sweep found 679 of 1,282 rostered channels silent
    for more than 60 days, and each one still cost a call on every discovery
    run. They keep their rows and their collected videos; only the polling
    stops, and clearing the flag reinstates them."""
    conn = _connect()
    try:
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL("SELECT uploads_playlist_id FROM {} "
                        "WHERE uploads_playlist_id IS NOT NULL AND active").format(
                    sql.Identifier(table)
                )
            )
            return [row[0] for row in cur.fetchall()]
    finally:
        conn.close()
