"""Check candidate channels against live data, on all three criteria.

Criteria, as specified:
  1. the channel declares LK          -- already established by querying
     channels.list over all 41,127 prior-batch channels; 8,599 verified-LK
     channels are untracked
  2. it is NOT in our top three categories (News & Politics, Entertainment,
     People & Blogs), which already hold 87% of what we collect
  3. it is not dead -- proven by a real upload in the last 60 days, from the
     API right now, not from the previous batch's Sept-Oct 2025 snapshot

Category is re-derived from a recent upload rather than trusted from the old
dataset. A channel's category was assigned there from videos published up to
ten months ago, and channels drift; more importantly the mode over one or two
old videos is a weak signal. Checking the newest upload's actual category costs
one unit per fifty channels, so there is no reason to guess.

An uploads playlist id is UU + the channel id minus its UC prefix, verified
against all 1,282 rostered channels, so no channels.list call is needed.

Resumable: results append as they arrive and completed channels are skipped on
restart, so a quota cut-off costs only the call in flight.
"""
import argparse
import csv
import os
import sys
from datetime import datetime, timezone

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "Project Code"))

from googleapiclient.errors import HttpError  # noqa: E402

from youtube_client import API_RETRIES, youtube  # noqa: E402

SRC = os.path.join(HERE, "candidates_nontop3.csv")
OUT = os.path.join(HERE, "candidates_live_checked.csv")
FIELDS = ["channel_id", "name", "subs", "old_category", "live_category",
          "last_upload", "days_since", "status"]
TOP3 = {"News & Politics", "Entertainment", "People & Blogs"}
ALIVE_WITHIN = 60

CATEGORY_NAMES = {
    1: "Film & Animation", 2: "Autos & Vehicles", 10: "Music", 15: "Pets & Animals",
    17: "Sports", 19: "Travel & Events", 20: "Gaming", 22: "People & Blogs",
    23: "Comedy", 24: "Entertainment", 25: "News & Politics", 26: "Howto & Style",
    27: "Education", 28: "Science & Technology", 29: "Nonprofits & Activism",
}


def newest(channel_id):
    """1 unit. Returns (published_at, video_id) of the newest real upload."""
    try:
        resp = youtube.playlistItems().list(
            part="contentDetails", playlistId="UU" + channel_id[2:], maxResults=5,
        ).execute(num_retries=API_RETRIES)
    except HttpError as e:
        if e.resp.status in (403, 404) and "quota" not in str(e).lower():
            return None, None
        raise
    for item in resp.get("items", []):
        cd = item.get("contentDetails", {})
        if cd.get("videoPublishedAt"):
            return cd["videoPublishedAt"], cd.get("videoId")
    return None, None


def categories_for(video_ids):
    """1 unit per 50. Live category of each video."""
    out = {}
    for i in range(0, len(video_ids), 50):
        resp = youtube.videos().list(
            part="snippet", id=",".join(video_ids[i:i + 50]), maxResults=50,
        ).execute(num_retries=API_RETRIES)
        for it in resp.get("items", []):
            cid = it.get("snippet", {}).get("categoryId")
            out[it["id"]] = CATEGORY_NAMES.get(int(cid), f"id:{cid}") if cid else None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-units", type=int, default=5000)
    args = ap.parse_args()

    cand = pd.read_csv(SRC)
    done = set()
    if os.path.exists(OUT):
        with open(OUT, encoding="utf-8-sig", newline="") as fh:
            done = {r["channel_id"] for r in csv.DictReader(fh)}
        print(f"resuming: {len(done):,} already checked")
    todo = cand[~cand.channel_id.isin(done)]
    print(f"to check: {len(todo):,} of {len(cand):,}")

    fresh = not os.path.exists(OUT)
    fh = open(OUT, "a", encoding="utf-8-sig", newline="")
    w = csv.DictWriter(fh, fieldnames=FIELDS)
    if fresh:
        w.writeheader()

    now = datetime.now(timezone.utc)
    units = 0
    pending, stopped = [], False

    def flush(force=False):
        """Resolve the live category of the queued batch, then write it out.

        Categories are looked up fifty videos at a time, so rows are held back
        until a full batch exists rather than costing a unit each."""
        nonlocal units
        if not pending or (len(pending) < 50 and not force):
            return
        vids = [p["video_id"] for p in pending if p["video_id"]]
        cats = categories_for(vids) if vids else {}
        units += -(-len(vids) // 50)
        for p in pending:
            p["live_category"] = cats.get(p["video_id"], "") or ""
            w.writerow({k: p[k] for k in FIELDS})
        pending.clear()
        fh.flush()

    try:
        for row in todo.itertuples():
            if units >= args.max_units:
                print(f"\nreached --max-units {args.max_units}; stopping cleanly")
                break
            try:
                ts, vid = newest(row.channel_id)
                units += 1
            except HttpError as e:
                if "quotaExceeded" in str(e):
                    print("\nQUOTA EXCEEDED — stopping cleanly; re-run tomorrow to continue")
                    stopped = True
                    break
                continue
            except Exception:
                continue

            if ts is None:
                days, status = None, "gone"
            else:
                days = (now - datetime.fromisoformat(
                    ts.replace("Z", "+00:00"))).total_seconds() / 86400
                status = "alive" if days <= ALIVE_WITHIN else "dead"

            pending.append(dict(channel_id=row.channel_id, name=row.name,
                                subs=row.subs, old_category=row.category,
                                live_category="", last_upload=ts or "",
                                days_since=round(days, 1) if days is not None else "",
                                status=status,
                                video_id=vid if status == "alive" else None))
            if len(pending) >= 50:
                flush()
                print(f"  {len(done)+units:,} checked, {units:,} units")
        flush(force=True)
    finally:
        fh.close()

    print(f"\nunits spent: {units:,}")
    d = pd.read_csv(OUT)
    print(f"\n{'status':8}{'channels':>10}")
    for k, v in d.status.value_counts().items():
        print(f"{k:8}{v:>10,}")
    alive = d[(d.status == "alive") & ~d.live_category.isin(TOP3)
              & d.live_category.notna() & (d.live_category != "")]
    print(f"\nMEETS ALL THREE CRITERIA: {len(alive):,}")
    print("\nby live category:")
    for k, v in alive.live_category.value_counts().items():
        print(f"    {k:24}{v:>7,}")
    moved = d[(d.status == "alive") & d.live_category.isin(TOP3)]
    if len(moved):
        print(f"\nreclassified INTO a top-3 category by live data, so dropped: {len(moved):,}")


if __name__ == "__main__":
    main()
