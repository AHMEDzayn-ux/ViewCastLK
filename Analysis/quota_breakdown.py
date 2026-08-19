"""Itemised daily quota usage, derived from the code and the live warehouse.

Every YouTube Data API endpoint the collector calls, what it costs, how often a
run makes the call, and what that totals per day. Figures come from the actual
roster and video counts rather than from estimates.

Endpoint costs are fixed by the API, not by the caller:
    channels.list        1 unit per call, up to 50 ids per call
    playlistItems.list   1 unit per call, up to 50 items per page, ONE playlist
    videos.list          1 unit per call, up to 50 ids per call
    videoCategories.list 1 unit per call
    search.list          100 units per call

The asymmetry that shapes everything: channels.list and videos.list take fifty
ids per call, so their cost is per fifty entities. playlistItems.list takes one
playlistId -- there is no batching parameter -- so discovery costs one unit per
channel per run no matter what. That is why batching helped the refresh
(1,282 -> 19) and cannot help discovery.
"""
import math
import os
import sys
import warnings

warnings.filterwarnings("ignore", message="pandas only supports SQLAlchemy")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "Project Code"))

import pandas as pd  # noqa: E402
from storage import connect  # noqa: E402

FULL_RUNS = 2          # REFRESH_CHANNELS=true slots
DISCOVERY_RUNS = 2     # REFRESH_CHANNELS=false slots
LOOKBACK_H = 26
PAGE = 50


def main():
    conn = connect()
    q = lambda s: pd.read_sql_query(s, conn)                      # noqa: E731
    active = q("select count(*) n from channels where active").n[0]
    total_ch = q("select count(*) n from channels").n[0]
    videos = q("select count(*) n from videos").n[0]
    per_day = q("""select round(count(*)/10.0) n from videos
                    where published_at >= now() - interval '10 days'""").n[0]
    # channels prolific enough to need more than one discovery page
    prolific = q(f"""
        select count(*) n from (
          select channel_id, count(*)/16.0 vpd from videos group by 1
        ) t where vpd * {LOOKBACK_H}/24.0 > {PAGE}""").n[0]
    conn.close()

    runs = FULL_RUNS + DISCOVERY_RUNS
    ch_batches = math.ceil(active / PAGE)
    vid_batches = math.ceil(videos / PAGE)
    disc_pages = active + prolific          # one page each, plus extras

    rows = [
        ("channels.list (refresh, batched by id)", f"{ch_batches} calls",
         FULL_RUNS, ch_batches * FULL_RUNS,
         f"{active} active channels / {PAGE} per call"),
        ("channels.list (new handles, forHandle)", "7 calls", FULL_RUNS,
         7 * FULL_RUNS, "roster entries with no stored custom_url"),
        ("playlistItems.list (discovery)", f"{disc_pages} calls", runs,
         disc_pages * runs,
         f"ONE playlist per call -- cannot be batched"),
        ("videos.list (snapshots + new identities)", f"{vid_batches} calls",
         runs, vid_batches * runs,
         f"{videos:,} tracked videos / {PAGE} per call"),
        ("videoCategories.list", "1 call", runs, runs, "reference list"),
        ("search.list", "0 calls", 0, 0,
         "100 units each; reserved for roster expansion, never routine"),
    ]

    print(f"roster: {active} active of {total_ch} ({total_ch - active} inactive, "
          f"skipped)   videos tracked: {videos:,}   new videos/day: {per_day:,.0f}")
    print(f"runs/day: {FULL_RUNS} full + {DISCOVERY_RUNS} discovery-only = {runs}\n")

    print(f"{'endpoint':44}{'per run':>12}{'runs/day':>10}{'units/day':>11}")
    print("-" * 77)
    total = 0
    for name, per, n, units, _ in rows:
        total += units
        print(f"{name:44}{per:>12}{n:>10}{units:>11,}")
    print("-" * 77)
    print(f"{'TOTAL':44}{'':>12}{'':>10}{total:>11,}  of 10,000\n")

    for name, _, _, _, note in rows:
        print(f"  {name.split(' (')[0]:22} {note}")

    print(f"\nshare of the total:")
    for name, _, _, units, _ in sorted(rows, key=lambda r: -r[3]):
        if units:
            print(f"  {name.split(' (')[0]:22} {units:>6,}  {100*units/total:5.1f}%")

    print(f"\nbefore the recent work, for comparison:")
    old_refresh = total_ch * FULL_RUNS
    old_disc = total_ch * runs
    old = old_refresh + old_disc + vid_batches * runs + runs
    print(f"  refresh unbatched, all {total_ch} channels : {old_refresh:,}")
    print(f"  discovery over all {total_ch} channels     : {old_disc:,}")
    print(f"  total                                  : {old:,}  "
          f"-> now {total:,}  ({100*(old-total)/old:.0f}% less)")


if __name__ == "__main__":
    main()
