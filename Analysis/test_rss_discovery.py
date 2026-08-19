"""Does the RSS feed find what playlistItems.list finds, and where does it fail?

Discovery is 54% of daily quota and cannot be batched: playlistItems.list takes
one playlistId per call, so it costs one unit per channel per run. YouTube also
publishes a per-channel Atom feed that is not part of the Data API and is not
quota-metered:

    https://www.youtube.com/feeds/videos.xml?channel_id=UC...

If it returns the same recent uploads, discovery becomes free. The catch is that
it returns at most 15 entries, so a channel publishing faster than the poll
interval can overflow it. This measures that rather than assuming it.

Sampling is deliberately skewed towards the busiest channels, because those are
the only ones that can overflow fifteen entries -- a uniform sample would say
everything is fine and be useless.
"""
import argparse
import os
import random
import sys
import time
import urllib.request
import warnings
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone

warnings.filterwarnings("ignore", message="pandas only supports SQLAlchemy")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "Project Code"))

import pandas as pd  # noqa: E402
from storage import connect  # noqa: E402
from youtube_client import get_channel_videos_since_by_playlist  # noqa: E402

NS = {"a": "http://www.w3.org/2005/Atom",
      "yt": "http://www.youtube.com/xml/schemas/2015"}
FEED = "https://www.youtube.com/feeds/videos.xml?channel_id={}"


def rss_videos(channel_id, since, timeout=20):
    """Video ids published on/after `since`, from the free Atom feed.

    Returns (ids, entry_count, covers_window).
    """
    req = urllib.request.Request(FEED.format(channel_id),
                                 headers={"User-Agent": "ViewCastLK/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        root = ET.fromstring(r.read())
    ids, n, oldest = set(), 0, None
    for e in root.findall("a:entry", NS):
        n += 1
        published = datetime.fromisoformat(e.find("a:published", NS).text)
        oldest = published if oldest is None else min(oldest, published)
        if published >= since:
            ids.add(e.find("yt:videoId", NS).text)
    # The feed always returns its 15 newest videos, so "15 entries" says nothing
    # about truncation. What matters is whether the feed reaches back past the
    # window: if its OLDEST entry predates the cutoff, everything inside the
    # window was included. If the oldest entry is still inside the window, the
    # feed may have been cut off and only the API can say.
    covers = oldest is not None and oldest < since
    return ids, n, covers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lookback-hours", type=int, default=26)
    ap.add_argument("--busiest", type=int, default=15)
    ap.add_argument("--random", type=int, default=25)
    args = ap.parse_args()

    since = datetime.now(timezone.utc) - timedelta(hours=args.lookback_hours)
    conn = connect()
    rates = pd.read_sql_query("""
        select v.channel_id, count(*)/16.0 vpd, c.uploads_playlist_id pl
          from videos v join channels c using (channel_id)
         where c.active group by 1, 3 order by vpd desc""", conn)
    conn.close()

    busiest = rates.head(args.busiest)
    rest = rates.iloc[args.busiest:]
    sample = pd.concat([busiest,
                        rest.sample(min(args.random, len(rest)), random_state=7)])
    print(f"comparing RSS against playlistItems.list over {args.lookback_hours}h")
    print(f"sample: {len(busiest)} busiest + {len(sample)-len(busiest)} random\n")
    print(f"{'channel':26}{'uploads/day':>12}{'api':>6}{'rss':>6}{'covers':>7}{'':>3}{'verdict'}")

    rows, t0 = [], time.time()
    for r in sample.itertuples():
        try:
            api = set(get_channel_videos_since_by_playlist(r.pl, since.strftime("%Y-%m-%dT%H:%M:%SZ")))
        except Exception as e:
            print(f"{r.channel_id:26} api error {type(e).__name__}")
            continue
        try:
            rss, n, covers = rss_videos(r.channel_id, since)
        except Exception as e:
            print(f"{r.channel_id:26} rss error {type(e).__name__}")
            continue
        missed = api - rss
        verdict = ("MISSED %d" % len(missed)) if missed else "complete"
        if missed and not covers:
            verdict += " (feed does not reach the cutoff — flagged, API fallback)"
        elif missed:
            verdict += " (feed REACHED the cutoff yet missed — undetectable!)"
        print(f"{r.channel_id:26}{r.vpd:>12.1f}{len(api):>6}{len(rss):>6}"
              f"{'yes' if covers else 'NO':>7}   {verdict}")
        rows.append(dict(channel_id=r.channel_id, vpd=r.vpd, api=len(api),
                         rss=len(rss), entries=n, missed=len(missed),
                         covers=covers))
        time.sleep(0.3)                      # be polite to a non-API endpoint

    d = pd.DataFrame(rows)
    took = time.time() - t0
    print(f"\nchecked {len(d)} channels in {took:.0f}s "
          f"({took/max(len(d),1):.2f}s each)")
    print(f"total videos found by API: {d.api.sum():,}   by RSS: {d.rss.sum():,}")
    agree = (d.missed == 0).sum()
    print(f"channels where RSS found everything: {agree}/{len(d)}")

    bad = d[d.missed > 0]
    if len(bad):
        undetectable = bad[bad.covers]
        print(f"channels where RSS missed videos: {len(bad)}")
        print(f"  correctly flagged for API fallback: {len(bad) - len(undetectable)}")
        print(f"  missed WITHOUT being flagged: {len(undetectable)}"
              f"{'  <-- silent loss; do not rely on RSS' if len(undetectable) else '  (none — no silent loss)'}")
        print(f"\nlowest uploads/day that overflowed: {bad.vpd.min():.1f}")
    else:
        print("no channel lost a video to RSS at this lookback")

    need_api = int((~d.covers).sum())
    print(f"\nchannels needing an API fallback: {need_api}/{len(d)} "
          f"({100*need_api/len(d):.0f}%) — the remaining "
          f"{len(d)-need_api} are free")


if __name__ == "__main__":
    main()
