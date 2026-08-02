"""Add verified channels to the roster and recover what is still recoverable.

Two things happen here, and the second is the time-sensitive one.

ADD. Channel details are fetched fifty ids per call, written as identity and
snapshot rows, and the ids appended to channel_handles.txt so ordinary runs
pick them up. Storing custom_url at the same time is what lets every later
refresh batch them instead of resolving one handle at a time.

BACKFILL. Ordinary discovery looks back 26 hours, so a channel added today has
its entire back catalogue invisible -- and unlike a missed snapshot, that is
not recoverable later. A video published N days ago can still yield a label at
any horizon beyond N: at four days old all four horizons are still ahead of it,
at twenty-five only day 30 is. Every day of delay costs one day's worth of the
oldest still-recoverable horizon, which is why this runs at the moment channels
are added rather than being left for later.

Discovery here goes through RSS first, as elsewhere, but the fallback is
uncapped: this pass happens once and cannot be repeated, so completeness is
worth more than the units. Channels uploading slowly enough that 31 days fits
inside the feed's fifteen entries -- most of them -- cost nothing at all.

Usage:
    python scripts/add_channels.py --file ../Analysis/channels_to_add.csv --dry-run
    python scripts/add_channels.py --file ../Analysis/channels_to_add.csv --limit 500
"""
import argparse
import math
import os
import sys
from datetime import datetime, timedelta, timezone

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import rss_discovery  # noqa: E402
from googleapiclient.errors import HttpError  # noqa: E402
from storage import append_rows, load_known_ids  # noqa: E402
from youtube_client import (flatten_channel_identity,  # noqa: E402
                            flatten_channel_snapshot, flatten_video_identity,
                            flatten_video_snapshot, get_channel_videos_since_by_playlist,
                            get_channels_by_ids, get_video_categories,
                            get_video_details)

ROSTER = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "..", "channel_handles.txt")
BATCH = 50


def roster_path():
    """The roster file, which must already exist.

    Resolving it relative to this script is only correct while the script sits
    in scripts/. Run a copy from elsewhere and '..' points somewhere else
    entirely -- and because appending opens in 'a' mode, that silently creates
    a new file instead of failing. It happened: 2,178 ids went to a stray file
    while the real roster stayed untouched, so the channels were discovered but
    never had their statistics refreshed.

    A roster that does not exist is always a mistake, so refuse rather than
    create one."""
    if not os.path.isfile(ROSTER):
        raise SystemExit(
            f"roster not found at {ROSTER}\n"
            f"Run this from the repository's scripts/ directory, or pass "
            f"--roster explicitly. It will not be created.")
    return ROSTER


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True, help="csv with a channel_id column")
    ap.add_argument("--limit", type=int, default=0, help="add only the first N")
    ap.add_argument("--backfill-days", type=int, default=31,
                    help="how far back to look for still-labelable uploads")
    ap.add_argument("--roster", default=None,
                    help="roster file to append to; defaults to ../channel_handles.txt")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    roster = args.roster or roster_path()
    if args.roster and not os.path.isfile(roster):
        raise SystemExit(f"roster not found at {roster}; it will not be created.")

    want = pd.read_csv(args.file).channel_id.dropna().unique().tolist()
    known = load_known_ids("channels", "channel_id")
    new = [c for c in want if c not in known]
    if args.limit:
        new = new[:args.limit]
    print(f"{len(want):,} in file, {len(want) - len(new):,} already known, "
          f"{len(new):,} to add")
    if not new:
        return

    est_ch = math.ceil(len(new) / BATCH)
    print(f"\nestimated cost")
    print(f"  channel details, {BATCH} per call        ~{est_ch} units")
    print(f"  backfill discovery                     RSS free + API where a feed "
          f"cannot cover {args.backfill_days}d")
    print(f"  video details for what is recovered    ~1 unit per 50 videos")
    if args.dry_run:
        print("\ndry run — nothing fetched, written or appended")
        return

    now = datetime.now(timezone.utc)
    captured_at = now.isoformat()

    # ---------------------------------------------------------------- add
    added, units = [], 0
    for i in range(0, len(new), BATCH):
        chunk = new[i:i + BATCH]
        got = get_channels_by_ids(chunk)
        units += 1
        if got:
            append_rows([flatten_channel_identity(c) for c in got], "channels")
            append_rows([flatten_channel_snapshot(c, captured_at) for c in got],
                        "channel_snapshots")
            added.extend(got)
        print(f"  resolved {len(added)}/{len(new)}")

    missing = len(new) - len(added)
    print(f"added {len(added):,} channels ({units} units)"
          + (f"; {missing} not returned by the API" if missing else ""))

    # Append to the roster file so ordinary runs include them. Raw ids are a
    # valid roster entry -- get_channel_by_roster_entry accepts either form --
    # and every one of these now has a stored custom_url, so refreshes batch.
    with open(roster, "a", encoding="utf-8") as fh:
        for c in added:
            fh.write(c["id"] + "\n")
    print(f"appended {len(added):,} ids to {roster}")

    # ----------------------------------------------------------- backfill
    since = (now - timedelta(days=args.backfill_days)).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"\nbackfilling uploads since {since}")
    playlists = {c["id"]: c["contentDetails"]["relatedPlaylists"]["uploads"]
                 for c in added}

    rss = rss_discovery.discover(list(playlists), since)
    print(f"  {rss.summary(len(playlists))}")
    video_ids = set(rss.video_ids)

    fell_back = 0
    for cid in rss.needs_api:                     # uncapped: one-off, unrepeatable
        pl = playlists.get(cid)
        if not pl:
            continue
        try:
            video_ids.update(get_channel_videos_since_by_playlist(pl, since))
            units += 1
            fell_back += 1
        except HttpError as e:
            if e.resp.status == 403 and "quotaExceeded" in str(e):
                print("  QUOTA EXCEEDED during backfill — channels are added and "
                      "the rest will be picked up by ordinary runs, but their "
                      "older uploads are lost. Re-run to recover what remains.")
                break
            continue
        except Exception:
            continue
    print(f"  API fallback used for {fell_back} channel(s)")
    print(f"  {len(video_ids):,} videos found")

    if not video_ids:
        print("nothing to record")
        return

    categories = get_video_categories()
    units += 1
    ids = list(video_ids)
    known_videos = load_known_ids("videos", "video_id")
    recorded = 0
    for i in range(0, len(ids), BATCH):
        details = get_video_details(ids[i:i + BATCH])
        units += 1
        unseen = [v for v in details if v["id"] not in known_videos]
        append_rows([flatten_video_identity(v, categories) for v in unseen], "videos")
        known_videos.update(v["id"] for v in unseen)
        append_rows([flatten_video_snapshot(v, captured_at) for v in details],
                    "video_snapshots")
        recorded += len(details)
        if (i // BATCH) % 20 == 0:
            print(f"  recorded {recorded:,}/{len(ids):,}")

    print(f"\nrecorded {recorded:,} videos, {units:,} units spent")
    print("Their first snapshot is taken now; ordinary runs continue from here, "
          "so any horizon still ahead of a video will be captured normally.")


if __name__ == "__main__":
    main()
