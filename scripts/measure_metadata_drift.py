"""How much of the stored metadata has already changed since we recorded it?

videos holds what a video looked like when first discovered, and 73% of the
corpus was first seen more than a day after publication. This fetches current
metadata and compares, which both answers "how often does this happen" and
initialises the fingerprints so ongoing detection has a baseline.

Rows written here are marked baseline = true: the edit happened somewhere
between first observation and now, and pretending otherwise would put a spike
of simultaneous changes into any timing analysis.

Cost is one unit per fifty videos.
"""
import argparse
import os
import sys
import warnings
from datetime import datetime, timezone

warnings.filterwarnings("ignore", message="pandas only supports SQLAlchemy")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd  # noqa: E402

import metadata_changes  # noqa: E402
from storage import append_rows, connect, save_metadata_shas  # noqa: E402
from youtube_client import get_video_details  # noqa: E402

BATCH = 50


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    conn = connect()
    stored = pd.read_sql_query("""
        SELECT video_id, title, description, tags, published_at,
               COALESCE(category_name, '(none)') AS category
          FROM videos ORDER BY video_id
    """, conn)
    first_seen = pd.read_sql_query("""
        SELECT video_id, min(captured_at) AS first_seen
          FROM video_snapshots GROUP BY 1
    """, conn)
    conn.close()

    stored = stored.merge(first_seen, on="video_id", how="left")
    if args.limit:
        # Sample across the whole age range, not the head of an ordering --
        # the youngest videos have had least time to be edited, so taking the
        # first N understates the rate.
        stored = stored.sample(min(args.limit, len(stored)), random_state=7)
    print(f"comparing {len(stored):,} videos "
          f"(~{-(-len(stored) // BATCH):,} units)")
    if args.dry_run:
        return

    by_id = stored.set_index("video_id")
    captured_at = datetime.now(timezone.utc).isoformat()
    rows, shas, checked, gone = [], {}, 0, 0

    ids = stored.video_id.tolist()
    for i in range(0, len(ids), BATCH):
        chunk = ids[i:i + BATCH]
        details = get_video_details(chunk)
        gone += len(chunk) - len(details)
        for item in details:
            vid = item["id"]
            if vid not in by_id.index:
                continue
            old = by_id.loc[vid]
            title, desc, tags, sha = metadata_changes.from_api(item)
            shas[vid] = sha
            checked += 1
            # pd.isna, not `or ""`. A NULL column arrives as NaN, NaN is
            # truthy, so `old.description or ""` returns NaN and the
            # comparison is always unequal. That marked every video with a
            # null description or null tags as edited -- 15,494 phantom rows
            # out of 15,867 before this was caught.
            def _same(new_value, stored):
                return new_value == ("" if pd.isna(stored) else str(stored))

            if not (_same(title, old.title) and _same(desc, old.description)
                    and _same(tags, old.tags)):
                rows.append({
                    "video_id": vid, "observed_at": captured_at, "title": title,
                    "description_len": len(desc),
                    "description_sha": metadata_changes._sha(desc),
                    "tags_sha": metadata_changes._sha(tags),
                    "baseline": True,
                    "prev_title": old.title, "category": old.category,
                    "first_seen": old.first_seen,
                })
        if (i // BATCH) % 100 == 0:
            print(f"  {checked:,}/{len(ids):,} checked, {len(rows):,} changed")

    print(f"\nchecked {checked:,}; {gone:,} no longer returned by the API")

    df = pd.DataFrame(rows)
    if df.empty:
        print("no metadata changes found")
        save_metadata_shas(shas)
        return

    df["title_changed"] = df.prev_title.fillna("") != df.title
    print(f"\nchanged in any field : {len(df):,}  ({100*len(df)/checked:.1f}%)")
    print(f"  title changed      : {int(df.title_changed.sum()):,}  "
          f"({100*df.title_changed.mean():.1f}% of changed, "
          f"{100*df.title_changed.sum()/checked:.1f}% of all)")
    print(f"  description/tags   : {int((~df.title_changed).sum()):,}")

    retitled = df[df.title_changed]
    if len(retitled):
        print("\nretitled by category:")
        cat = retitled.category.value_counts().head(8)
        tot = stored.category.value_counts()
        for k, v in cat.items():
            print(f"  {k:24}{v:>6,}  of {tot.get(k, 0):>6,}  ({100*v/tot.get(k, 1):.1f}%)")
        print("\nexamples:")
        for r in retitled.head(5).itertuples():
            print(f"  was: {str(r.prev_title)[:70]}")
            print(f"  now: {str(r.title)[:70]}")

    append_rows(df[["video_id", "observed_at", "title", "description_len",
                    "description_sha", "tags_sha", "baseline"]].to_dict("records"),
                "video_metadata_changes")
    save_metadata_shas(shas)
    print(f"recorded {len(df):,} baseline change rows; "
          f"{len(shas):,} fingerprints initialised")


if __name__ == "__main__":
    main()
