"""One-off: record the player shape of every video seen before video_shapes existed.

Costs 1 YouTube quota unit per 50 videos (videos.list, player part only). Run
with no arguments to see how many videos are missing and what it would cost;
nothing is fetched until --run is given. --max-units caps the spend so the
backfill can be split across days without starving the scheduled poll.

Videos that no longer come back (deleted or private) get a row with null
dimensions, so a rerun does not pay to ask about them again.

Usage:
    python scripts/backfill_video_shapes.py                   # count and cost only
    python scripts/backfill_video_shapes.py --run --max-units 1500
"""
import argparse
import math
from datetime import datetime, timezone

from googleapiclient.errors import HttpError

from storage import _connect, append_rows
from youtube_client import API_RETRIES, PLAYER_MAX_HEIGHT, flatten_video_shape, youtube

TABLE = "video_shapes"


def missing_ids() -> list[str]:
    conn = _connect()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT v.video_id FROM videos v
                LEFT JOIN video_shapes s USING (video_id)
                WHERE s.video_id IS NULL
                ORDER BY v.published_at DESC
            """)
            return [r[0] for r in cur.fetchall()]
    finally:
        conn.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true", help="actually call the API")
    ap.add_argument("--max-units", type=int, default=1500, help="quota units to spend at most")
    args = ap.parse_args()

    ids = missing_ids()
    units = math.ceil(len(ids) / 50)
    print(f"{len(ids):,} videos without a shape; full backfill costs {units:,} units")
    if not args.run:
        print("dry run: pass --run to fetch")
        return

    spent = written = 0
    for i in range(0, len(ids), 50):
        if spent >= args.max_units:
            print(f"stopped at the {args.max_units}-unit cap; rerun to continue")
            break
        chunk = ids[i:i + 50]
        try:
            resp = youtube.videos().list(part="player", id=",".join(chunk),
                                         maxHeight=PLAYER_MAX_HEIGHT).execute(num_retries=API_RETRIES)
        except HttpError as e:
            if e.resp.status == 403 and b"quotaExceeded" in e.content:
                print("quota exceeded; everything written so far is kept, rerun after the reset")
                break
            raise
        spent += 1
        now = datetime.now(timezone.utc).isoformat()
        got = {item["id"]: item for item in resp.get("items", [])}
        rows = [flatten_video_shape(got.get(v, {"id": v}), now) for v in chunk]
        append_rows(rows, TABLE)
        written += len(rows)
        if spent % 100 == 0:
            print(f"  {written:,} written, {spent} units")
    print(f"done: {written:,} rows written, {spent} units spent")


if __name__ == "__main__":
    main()
