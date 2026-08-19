"""Establish, per rostered channel, when it last uploaded anything.

Why this exists: 853 of the 1,282 channels produced no video in the 16 days
have been collecting, and at four discovery runs a day they cost about 3,400
quota units daily for nothing. But "silent for 16 days" is not "dead" — a
creator who posts monthly looks identical to one who has stopped. Pruning on
our own window alone would quietly delete exactly the mid-frequency channels
the project is aimed at. This asks YouTube instead.

Cost. One playlistItems.list call per channel, 1 unit, asking for a single item:
uploads playlists come back newest-first, so item one is the most recent upload.
Channels that already have a video in our warehouse are skipped — we know their
last upload date for free — so the sweep costs roughly 850 units, not 1,282.

Safe to interrupt. Results append to the CSV as they arrive and completed
channels are skipped on restart, so a quota cut-off or a dropped connection
costs only the call in flight.

Usage:
    python sweep_last_upload.py                 # silent channels only
    python sweep_last_upload.py --all           # every rostered channel
    python sweep_last_upload.py --max-units 400 # stop early
"""
import argparse
import csv
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "Project Code"))

from googleapiclient.errors import HttpError  # noqa: E402

from storage import connect  # noqa: E402
from youtube_client import API_RETRIES, youtube  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "channel_last_upload.csv")
FIELDS = ["channel_id", "title", "country", "source", "last_upload",
          "days_since", "status"]

DORMANT_AFTER = 30   # days without an upload before a channel stops counting as active
DEAD_AFTER = 60      # days without an upload before it is a pruning candidate


def classify(days):
    if days is None:
        return "no_uploads"
    if days > DEAD_AFTER:
        return "dead"
    if days > DORMANT_AFTER:
        return "dormant"
    return "active"


def newest_upload(playlist_id):
    """1 unit. Newest item of an uploads playlist, or None.

    A 404 is playlistNotFound, which is the legitimate answer for a channel that
    has never uploaded — not a failure. Items lacking videoPublishedAt are
    private, deleted or scheduled; YouTube still lists them, so walk past them
    rather than reading the first row blindly.
    """
    try:
        resp = youtube.playlistItems().list(
            part="contentDetails", playlistId=playlist_id, maxResults=5,
        ).execute(num_retries=API_RETRIES)
    except HttpError as e:
        if e.resp.status == 404:
            return None
        raise
    for item in resp.get("items", []):
        ts = item.get("contentDetails", {}).get("videoPublishedAt")
        if ts:
            return ts
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true",
                    help="sweep every channel, not only those silent in our window")
    ap.add_argument("--max-units", type=int, default=1500,
                    help="stop once this many API units have been spent")
    args = ap.parse_args()

    now = datetime.now(timezone.utc)

    conn = connect()
    cur = conn.cursor()
    cur.execute("""
        SELECT c.channel_id, c.title, c.country, c.uploads_playlist_id,
               MAX(v.published_at) AS newest_known
        FROM channels c
        LEFT JOIN videos v USING (channel_id)
        GROUP BY c.channel_id, c.title, c.country, c.uploads_playlist_id
    """)
    rows = cur.fetchall()
    conn.close()
    print(f"roster: {len(rows):,} channels")

    done = set()
    if os.path.exists(OUT):
        with open(OUT, encoding="utf-8-sig", newline="") as fh:
            done = {r["channel_id"] for r in csv.DictReader(fh)}
        print(f"resuming: {len(done):,} already swept")

    fresh = not os.path.exists(OUT)
    fh = open(OUT, "a", encoding="utf-8-sig", newline="")
    writer = csv.DictWriter(fh, fieldnames=FIELDS)
    if fresh:
        writer.writeheader()

    units = 0
    counts = {}
    todo = []
    for channel_id, title, country, playlist_id, newest_known in rows:
        if channel_id in done:
            continue
        if newest_known is not None and not args.all:
            # already alive on local evidence; costs nothing to record
            days = (now - newest_known).total_seconds() / 86400
            rec = dict(channel_id=channel_id, title=title, country=country,
                       source="warehouse", last_upload=newest_known.isoformat(),
                       days_since=round(days, 1), status=classify(days))
            writer.writerow(rec)
            counts[rec["status"]] = counts.get(rec["status"], 0) + 1
            continue
        todo.append((channel_id, title, country, playlist_id))

    fh.flush()
    print(f"resolved from warehouse: {sum(counts.values()):,}")
    print(f"needing an API call: {len(todo):,}  (~{len(todo):,} units)\n")

    failed = 0
    try:
        for i, (channel_id, title, country, playlist_id) in enumerate(todo, 1):
            if units >= args.max_units:
                print(f"\nreached --max-units {args.max_units}; stopping cleanly")
                break
            if not playlist_id:
                rec_status, ts, days = "no_uploads", "", None
            else:
                try:
                    ts = newest_upload(playlist_id)
                    units += 1
                except HttpError as e:
                    if e.resp.status == 403 and "quotaExceeded" in str(e):
                        print("\nQUOTA EXCEEDED — stopping cleanly. "
                              "Re-run after reset to continue where this left off.")
                        break
                    failed += 1
                    continue
                except Exception:
                    failed += 1
                    continue
                if ts:
                    days = (now - datetime.fromisoformat(
                        ts.replace("Z", "+00:00"))).total_seconds() / 86400
                else:
                    days = None
                rec_status = classify(days)
                ts = ts or ""

            writer.writerow(dict(
                channel_id=channel_id, title=title, country=country, source="api",
                last_upload=ts, days_since=(round(days, 1) if days is not None else ""),
                status=rec_status))
            counts[rec_status] = counts.get(rec_status, 0) + 1
            if i % 100 == 0:
                fh.flush()
                print(f"  {i:,}/{len(todo):,}   units {units:,}")
    finally:
        fh.close()

    print(f"\nunits spent: {units:,}")
    if failed:
        print(f"failed after retries: {failed:,} (re-run to retry them)")
    print(f"\nwrote {OUT}\n")
    print(f"{'status':12}{'channels':>10}")
    for k in ("active", "dormant", "dead", "no_uploads"):
        if k in counts:
            print(f"{k:12}{counts[k]:>10,}")
    prunable = counts.get("dead", 0) + counts.get("no_uploads", 0)
    if prunable:
        print(f"\npruning candidates (no upload in {DEAD_AFTER}+ days, or none ever): "
              f"{prunable:,}")
        print(f"discovery quota they consume at 4 runs/day: {prunable * 4:,} units/day")


if __name__ == "__main__":
    main()
