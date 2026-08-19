"""Turn the last-upload sweep into a roster decision.

Answers three things:
  1. how many channels are genuinely dead, and what pruning them frees
  2. whether pruning strips a topic area already under-represented
  3. how many channels the quota supports afterwards, per discovery cadence

Quota model, all figures per day:
    channel refresh  = 2 full runs x ceil(N/50)          [after batching by id]
    discovery        = R runs x N                        [1 unit per channel]
    video snapshots  = 4 runs x ceil(active_videos/50)   [50 ids per call]
    active_videos    = videos_per_day x TRACKING_WINDOW_DAYS

Snapshots stay at four runs a day in every scenario: that cadence is what puts
every horizon within three hours of its mark, and it is not the cost driver.
Discovery is the lever, and lowering it costs nothing but tolerance for a
missed run, because the 26-hour look-back still finds every upload.
"""
import math
import os
import sys
import warnings

import pandas as pd

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "Project Code"))

SWEEP = os.path.join(HERE, "channel_last_upload.csv")
DAILY_QUOTA = 10_000
BUDGET = 8_500          # leave headroom rather than run at the ceiling
VIDEOS_PER_DAY = 787    # measured over the last ten complete days
WINDOW_NOW, WINDOW_PROPOSED = 60, 40
NEW_CHANNEL_RATE = 0.214   # ~1.5 uploads/week, from the candidate set


def units(n_channels, discovery_runs, videos_per_day, window):
    refresh = 2 * math.ceil(n_channels / 50)
    discovery = discovery_runs * n_channels
    snaps = 4 * math.ceil(videos_per_day * window / 50)
    return refresh, discovery, snaps, refresh + discovery + snaps


def main():
    if not os.path.exists(SWEEP):
        raise SystemExit("run sweep_last_upload.py first")
    d = pd.read_csv(SWEEP)
    print(f"swept: {len(d):,} channels\n")

    print(f"{'status':12}{'channels':>10}{'share':>9}")
    for k in ("active", "dormant", "dead", "no_uploads"):
        n = int((d.status == k).sum())
        if n:
            print(f"{k:12}{n:>10,}{100*n/len(d):>8.1f}%")

    dead = d[d.status.isin(["dead", "no_uploads"])]
    keep = d[~d.status.isin(["dead", "no_uploads"])]
    print(f"\nprunable: {len(dead):,}   keeping: {len(keep):,}")
    print(f"discovery they consume at 4 runs/day: {len(dead)*4:,} units/day")

    if "days_since" in d and dead.days_since.notna().any():
        q = dead.days_since.dropna().quantile([.5, .9]).round(0)
        print(f"dead channels' silence: median {q.iloc[0]:.0f} d, "
              f"90th pct {q.iloc[1]:.0f} d")

    # --- does pruning strip a topic area already thin?
    from storage import connect
    conn = connect()
    topics = pd.read_sql_query(
        "select channel_id, topic_categories from channels", conn)
    conn.close()
    d = d.merge(topics, on="channel_id", how="left")

    def topics_of(frame):
        c = {}
        for s in frame.topic_categories.dropna():
            for u in str(s).split("|"):
                t = u.rsplit("/", 1)[-1].replace("_", " ")
                if t:
                    c[t] = c.get(t, 0) + 1
        return c

    t_dead = topics_of(d[d.status.isin(["dead", "no_uploads"])])
    t_keep = topics_of(d[~d.status.isin(["dead", "no_uploads"])])
    print(f"\n{'topic':26}{'kept':>7}{'pruned':>8}{'% lost':>9}")
    for k in sorted(set(t_dead) | set(t_keep),
                    key=lambda x: -(t_dead.get(x, 0) + t_keep.get(x, 0)))[:14]:
        a, b = t_keep.get(k, 0), t_dead.get(k, 0)
        print(f"{k:26}{a:>7,}{b:>8,}{100*b/(a+b):>8.0f}%")

    # --- what the quota supports afterwards
    n_keep = len(keep)
    print(f"\n\nquota budget {BUDGET:,} of {DAILY_QUOTA:,} units/day")
    print(f"{'scenario':44}{'refresh':>9}{'disc':>8}{'snaps':>8}{'total':>8}")
    # today's refresh is one call per channel, not one per fifty
    _, di, s, _ = units(len(d), 4, VIDEOS_PER_DAY, WINDOW_NOW)
    unbatched = 2 * len(d)
    print(f"{'today, unbatched refresh, 60d window':44}"
          f"{unbatched:>9}{di:>8}{s:>8}{unbatched+di+s:>8}")
    r, di, s, tot = units(len(d), 4, VIDEOS_PER_DAY, WINDOW_NOW)
    print(f"{'+ batched refresh':44}{r:>9}{di:>8}{s:>8}{tot:>8}")
    r, di, s, tot = units(len(d), 4, VIDEOS_PER_DAY, WINDOW_PROPOSED)
    print(f"{'+ 40-day tracking window':44}{r:>9}{di:>8}{s:>8}{tot:>8}")
    r, di, s, tot = units(n_keep, 4, VIDEOS_PER_DAY, WINDOW_PROPOSED)
    print(f"{'+ pruned dead':44}{r:>9}{di:>8}{s:>8}{tot:>8}")

    print(f"\n{'discovery':>12}{'base total':>12}{'new channels':>14}{'roster':>9}")
    for runs in (4, 3, 2, 1):
        _, _, _, base = units(n_keep, runs, VIDEOS_PER_DAY, WINDOW_PROPOSED)
        spare = BUDGET - base
        per_new = runs + 4 * (NEW_CHANNEL_RATE * WINDOW_PROPOSED / 50) + 2 / 50
        n_new = max(0, int(spare / per_new))
        print(f"{str(runs)+'x/day':>12}{base:>12,}{n_new:>14,}{n_keep+n_new:>9,}")

    print("\nnote: 1x/day needs the discovery look-back raised from 26h to ~50h,")
    print("      or one missed run leaves a permanent hole in the history.")


if __name__ == "__main__":
    main()
