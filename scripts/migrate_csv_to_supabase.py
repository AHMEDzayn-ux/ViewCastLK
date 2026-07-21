"""One-time migration: load the existing CSV-collected data into Supabase.

Run this once, right after applying sql/schema.sql to a fresh Supabase
project, to carry over everything collected before the Supabase migration.
Safe to re-run — storage.append_rows/write_rows upsert on each table's
primary key, so re-running this just no-ops on rows already migrated.

Order matters: channels before channel_snapshots and videos (foreign key),
video_categories before videos (foreign key), videos before video_snapshots.
"""
import csv

from storage import append_rows, write_rows

OUTPUT_DIR = "output"


def load_csv(name: str) -> list[dict]:
    with open(f"{OUTPUT_DIR}/{name}", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def migrate(name: str, table: str, loader=append_rows) -> None:
    rows = load_csv(name)
    loader(rows, table)
    print(f"Migrated {len(rows)} rows from {name} -> {table}")


def main():
    migrate("video_categories.csv", "video_categories", loader=write_rows)
    migrate("channels.csv", "channels")
    migrate("channel_snapshots.csv", "channel_snapshots")
    migrate("videos.csv", "videos")
    migrate("video_snapshots.csv", "video_snapshots")
    print("\nMigration complete.")


if __name__ == "__main__":
    main()
