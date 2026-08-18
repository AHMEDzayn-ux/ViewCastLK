"""Estimate how many prior-batch LK channels are still uploading.

The previous batch's dataset holds 8,888 channels the API confirms declare LK,
of which 8,599 are not on our roster. Their posting rates come from a snapshot
taken around September-October 2025, so a channel that looked busy then may
have stopped since -- our own roster turned out to be 53 per cent dead when
actually asked.

Checking all 8,599 costs 8,599 units, more than a day's remaining headroom, so
this samples instead and reports a proportion with its margin of error. The
population is split in two because the strata behave differently: channels the
snapshot caught five or more times are the practical candidates, while the
sparse tail may be mostly one-off uploaders.

An uploads playlist id is UU + the channel id minus its UC prefix -- verified
against all 1,282 rostered channels -- so no channels.list call is needed and
liveness costs exactly one unit per channel sampled.
"""
import argparse
import os
import random
import sys
from datetime import datetime, timezone

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "Project Code"))

from googleapiclient.errors import HttpError  # noqa: E402

from youtube_client import API_RETRIES, youtube  # noqa: E402

CANDIDATES = os.path.join(HERE, "candidate_channels.csv")
COUNTRIES = os.path.join(HERE, "prior_channel_countries.csv")
OUT = os.path.join(HERE, "candidate_liveness_sample.csv")
DEAD_AFTER = 60
SEED = 20260801


def newest_upload(channel_id):
    """1 unit, or None. 404 means the channel has no uploads playlist at all."""
    try:
        resp = youtube.playlistItems().list(
            part="contentDetails", playlistId="UU" + channel_id[2:], maxResults=5,
        ).execute(num_retries=API_RETRIES)
    except HttpError as e:
        if e.resp.status in (403, 404):
            return "gone" if e.resp.status == 404 else None
        raise
    for item in resp.get("items", []):
        ts = item.get("contentDetails", {}).get("videoPublishedAt")
        if ts:
            return ts
    return "gone"


def wilson(k, n):
    """95% confidence interval for a proportion. Normal approximation breaks
    down near 0 and 1 and on small samples, which is exactly where an estimate
    like this lands, so use the Wilson score interval instead."""
    if not n:
        return 0.0, 0.0, 0.0
    z, p = 1.96, k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    m = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return p, max(0.0, c - m), min(1.0, c + m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-stratum", type=int, default=250)
    args = ap.parse_args()

    solid = set(pd.read_csv(CANDIDATES).channel_id)
    cc = pd.read_csv(COUNTRIES)
    lk = set(cc.loc[cc.country == "LK", "channel_id"])

    from storage import connect
    conn = connect()
    ours = set(pd.read_sql_query("select channel_id from channels", conn).channel_id)
    conn.close()

    untracked = lk - ours
    sparse = untracked - solid
    print(f"verified-LK, untracked: {len(untracked):,}"
          f"   solid (>=5 videos): {len(solid):,}   sparse: {len(sparse):,}")

    rng = random.Random(SEED)
    now = datetime.now(timezone.utc)
    rows = []
    for name, pool in (("solid", sorted(solid)), ("sparse", sorted(sparse))):
        pick = rng.sample(pool, min(args.per_stratum, len(pool)))
        alive = checked = 0
        for cid in pick:
            try:
                ts = newest_upload(cid)
            except HttpError as e:
                if "quotaExceeded" in str(e):
                    print("QUOTA EXCEEDED — stopping; partial result below")
                    break
                continue
            except Exception:
                continue
            checked += 1
            if ts in (None, "gone"):
                days, status = None, "gone"
            else:
                days = (now - datetime.fromisoformat(
                    ts.replace("Z", "+00:00"))).total_seconds() / 86400
                status = "alive" if days <= DEAD_AFTER else "dead"
                alive += status == "alive"
            rows.append(dict(stratum=name, channel_id=cid,
                             last_upload=ts if isinstance(ts, str) and ts != "gone" else "",
                             days_since=round(days, 1) if days is not None else "",
                             status=status))
        p, lo, hi = wilson(alive, checked)
        pop = len(solid) if name == "solid" else len(sparse)
        print(f"\n{name}: {alive}/{checked} still uploading "
              f"= {100*p:.1f}% (95% CI {100*lo:.1f}-{100*hi:.1f}%)")
        print(f"  implies {int(p*pop):,} of {pop:,} live "
              f"(range {int(lo*pop):,}-{int(hi*pop):,})")

    pd.DataFrame(rows).to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"\nwrote {OUT}   units spent: {len(rows):,}")


if __name__ == "__main__":
    main()
