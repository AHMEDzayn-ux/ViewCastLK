"""Materialise horizon labels from the Parquet archives.

WHY THIS EXISTS
Labels are normally written nightly by archive_snapshots.extract_labels, which
looks back a few days over what is still in Postgres. Partitions older than the
retention window are exported to Parquet and dropped, so once a day falls out
of Postgres its observations are only readable here. This script re-derives
labels from those exported files.

WHAT IT FOUND THE FIRST TIME IT RAN (12 August 2026)
Nothing. Across all 23 archived days -- 17 July to 9 August, 2,148,774 rows --
it produced 60,792 candidate labels and every single one was already in
video_horizon_labels, none closer to its mark than what was recorded.

That is the correct outcome, and worth writing down so the question is not
reopened later. Label extraction ran UNBOUNDED from 2 to 9 August, and the
first partition was not dropped until 10 August, so every one of those nights
scanned the full history back to 17 July. The labels were harvested while the
partitions were still in Postgres. The apparent gap in day-7 coverage for
mid-July publications is not a processing failure at all.

WHAT IT CANNOT RECOVER, AND WHY THE GAP IS REAL
Only observations that were actually taken. Most videos published 13-19 July
have no snapshot near their day-7 mark because they were not being watched that
week -- their channels joined in the August roster expansion, and the videos
arrived through backfill with their early history already missing. Those labels
do not exist in Postgres, in the archives, or anywhere else.

The script remains useful: it is the only path back to a label once a partition
is dropped, and a future gap -- a run of failed nights, a bounded catch-up that
misses a window -- would be recoverable through it precisely because it was not
needed this time.

SEMANTICS
Identical to extract_labels, deliberately: for each (video, horizon) keep the
observation nearest the mark, accept it only within TOLERANCE hours, and on
conflict keep whichever is closer. Running this can only improve a label, never
degrade one, so it is safe to re-run and safe to run alongside the nightly job.

Usage:
    python scripts/recover_labels_from_archive.py --archives DIR --dry-run
    python scripts/recover_labels_from_archive.py --archives DIR
"""
import argparse
import glob
import os
import sys

import pandas as pd
from psycopg2.extras import execute_batch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from storage import connect, read_df  # noqa: E402

HORIZONS = (7, 14, 21, 30)
TOLERANCE = 12.0

# An archived observation must beat the recorded one by at least this much to
# count as an improvement. Without it, a candidate that is the SAME observation
# already in the table -- recomputed here in float from the same two timestamps
# -- lands a few times 1e-13 hours either side of the stored value, and the
# sign is arbitrary. That reported 29,505 "improvements" out of 60,792
# candidates, almost exactly half, which is the signature of a coin flip rather
# than a finding. The largest real gain measured was 0.43 nanoseconds.
#
# One second is far below anything that matters (labels are accepted up to 12
# hours from the mark) and far above float noise.
EPSILON_HOURS = 1.0 / 3600.0
COLS = ["video_id", "captured_at", "view_count", "like_count", "comment_count"]

UPSERT = """
INSERT INTO video_horizon_labels
    (video_id, horizon_days, captured_at, hours_off,
     view_count, like_count, comment_count)
VALUES (%s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (video_id, horizon_days) DO UPDATE
   SET captured_at   = EXCLUDED.captured_at,
       hours_off     = EXCLUDED.hours_off,
       view_count    = EXCLUDED.view_count,
       like_count    = EXCLUDED.like_count,
       comment_count = EXCLUDED.comment_count
 WHERE abs(EXCLUDED.hours_off) < abs(video_horizon_labels.hours_off) - %s
"""


def best_from_file(path, published):
    """Nearest-to-mark candidates from one day's partition.

    Files are read one at a time rather than concatenated. The full archive is
    2.1 million rows and growing daily; holding it in memory works today and
    stops working at some point, whereas the running best is bounded by the
    number of videos.
    """
    df = pd.read_parquet(path, columns=COLS)
    df = df.merge(published, on="video_id", how="inner")   # FK: skip unknowns
    if df.empty:
        return None

    elapsed = (df.captured_at - df.published_at).dt.total_seconds() / 3600.0
    out = []
    for h in HORIZONS:
        off = elapsed - h * 24
        keep = off.abs() <= TOLERANCE
        if not keep.any():
            continue
        c = df.loc[keep, COLS].copy()
        c["horizon_days"] = h
        c["hours_off"] = off[keep]
        out.append(c)
    if not out:
        return None

    c = pd.concat(out, ignore_index=True)
    c["abs_off"] = c.hours_off.abs()
    c = c.sort_values("abs_off").drop_duplicates(["video_id", "horizon_days"])
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archives", required=True,
                    help="directory of video_snapshots_*.parquet files")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.archives, "**", "*.parquet"),
                             recursive=True))
    if not files:
        raise SystemExit(f"no parquet files under {args.archives}")
    print(f"{len(files)} archive file(s)")

    conn = connect(session_pooler=True)
    published = read_df(conn, "SELECT video_id, published_at FROM videos")
    published["published_at"] = pd.to_datetime(published.published_at, utc=True)
    print(f"{len(published):,} videos known")

    existing = read_df(
        conn, "SELECT video_id, horizon_days, hours_off FROM video_horizon_labels")
    # hours_off is numeric, so psycopg2 hands back Decimal in an object column.
    # Cast once, here, so every comparison below is float against float rather
    # than relying on pandas' object-dtype path. (This was not what produced
    # the spurious 29,505 improvements -- see EPSILON_HOURS for that -- but
    # mixed dtypes have no business in a numeric comparison either way.)
    existing["hours_off"] = existing.hours_off.astype(float)
    print(f"{len(existing):,} labels already recorded")

    best = None
    for i, p in enumerate(files, 1):
        c = best_from_file(p, published)
        if c is not None:
            best = c if best is None else pd.concat([best, c], ignore_index=True)
            best = best.sort_values("abs_off").drop_duplicates(
                ["video_id", "horizon_days"])
        print(f"  [{i:>2}/{len(files)}] {os.path.basename(p):<34} "
              f"running best: {0 if best is None else len(best):,}")

    if best is None or best.empty:
        print("nothing found in the archives")
        conn.close()
        return

    # Split for reporting only -- the upsert's WHERE clause is what actually
    # protects existing labels, so this cannot disagree with what gets written.
    merged = best.merge(existing, on=["video_id", "horizon_days"], how="left",
                        suffixes=("", "_old"))
    new = merged.hours_off_old.isna()
    better = (~new) & (merged.hours_off.abs()
                       < merged.hours_off_old.abs() - EPSILON_HOURS)

    print(f"\ncandidates nearest each mark: {len(best):,}")
    print(f"  new labels                 {int(new.sum()):,}")
    print(f"  improve an existing label  {int(better.sum()):,}")
    print(f"  no better than recorded    {int((~new & ~better).sum()):,}")
    print("\nby horizon (new only):")
    for h in HORIZONS:
        print(f"  day {h:>2}: {int((new & (merged.horizon_days == h)).sum()):>7,}")

    if args.dry_run:
        print("\ndry run — nothing written")
        conn.close()
        return

    # Only send rows that would actually change something. Sending all
    # 60,792 would make the database re-evaluate every one and, before the
    # epsilon existed, rewrite half of them for no reason.
    write = best.merge(merged.loc[new | better, ["video_id", "horizon_days"]],
                       on=["video_id", "horizon_days"], how="inner")
    if write.empty:
        print("\nnothing to write — the archives hold no label the database "
              "does not already have, and none closer to its mark")
        conn.close()
        return

    rows = [(r.video_id, int(r.horizon_days), r.captured_at.to_pydatetime(),
             float(r.hours_off),
             None if pd.isna(r.view_count) else int(r.view_count),
             None if pd.isna(r.like_count) else int(r.like_count),
             None if pd.isna(r.comment_count) else int(r.comment_count),
             EPSILON_HOURS)
            for r in write.itertuples(index=False)]

    with conn.cursor() as cur:
        cur.execute("SET LOCAL statement_timeout = '20min'")
        execute_batch(cur, UPSERT, rows, page_size=1000)
    conn.commit()

    after = read_df(conn, "SELECT count(*) n FROM video_horizon_labels").n[0]
    print(f"\nwrote {len(rows):,} candidates; label table now {after:,} rows "
          f"(was {len(existing):,})")
    conn.close()


if __name__ == "__main__":
    main()
