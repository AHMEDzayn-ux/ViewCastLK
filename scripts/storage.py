"""Persistence layer — the ONLY file the Supabase/GitHub-Actions migration needs
to rewrite. Every function here takes plain flat dicts (already shaped by
youtube_client.py's flatten_* functions) and a destination name; swap the CSV
internals for supabase-py calls and nothing outside this file has to change.

CSV -> Supabase mapping for whoever does that migration:
    append_rows(rows, path)         -> supabase.table(name).upsert(rows) for identity
                                        tables, .insert(rows) for snapshot tables
    write_rows(rows, path)          -> supabase.table(name).upsert(rows) (categories
                                        only change if YouTube changes its taxonomy)
    load_active_video_ids(path, ..) -> supabase.table("videos").select("video_id")
                                        .gte("published_at", since_date)
"""
import csv
import os


def append_rows(rows: list[dict], path: str) -> None:
    """Adds rows to a growing table. Writes the header only if the file is new."""
    if not rows:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    file_exists = os.path.isfile(path)
    with open(path, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)


def write_rows(rows: list[dict], path: str) -> None:
    """Overwrites a table. Used for one-time reference data (video categories),
    where there's nothing to accumulate over time."""
    if not rows:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def load_active_video_ids(path: str, since_date: str) -> set[str]:
    """Reads back previously-discovered video identities and returns the ones
    still inside the tracking window (published_at >= since_date) — so the daily
    poll knows which already-known videos still need a fresh snapshot, not just
    newly-discovered ones. Returns an empty set on the very first run."""
    if not os.path.isfile(path):
        return set()
    with open(path, encoding="utf-8-sig") as f:
        return {row["video_id"] for row in csv.DictReader(f) if row["published_at"] >= since_date}
