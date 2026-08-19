"""Can RSS discovery serve the whole roster, concurrently, reliably?

The single-channel test showed the feed is accurate and that any shortfall is
detectable. That says nothing about whether 2,781 requests to an undocumented,
unmetered endpoint succeed when issued together -- which is the question that
decides whether RSS can replace discovery.

Measures what matters for that decision:
  * success rate, and the shape of the failures (timeout? 429? 404?)
  * wall-clock time, since discovery has to fit inside a run
  * how many feeds cannot prove they cover the look-back, because those are the
    ones still needing an API call

404s are expected and are not failures: a channel deleted since it was last seen
has no feed. They are counted separately from errors that would make RSS
unreliable.

Deliberately polite: bounded concurrency, a short delay per worker, and any
429 is treated as a stop signal rather than something to retry through.
"""
import argparse
import os
import random
import sys
import threading
import time
import urllib.error
import urllib.request
import warnings
import xml.etree.ElementTree as ET
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone

warnings.filterwarnings("ignore", message="pandas only supports SQLAlchemy")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "Project Code"))

import pandas as pd  # noqa: E402
from storage import connect  # noqa: E402

NS = {"a": "http://www.w3.org/2005/Atom",
      "yt": "http://www.youtube.com/xml/schemas/2015"}
FEED = "https://www.youtube.com/feeds/videos.xml?channel_id={}"

_rate_limited = threading.Event()


def fetch(channel_id, since, timeout):
    """Returns (status, n_in_window, covers_window, seconds)."""
    t0 = time.time()
    if _rate_limited.is_set():
        return "skipped_rate_limit", 0, False, 0.0
    req = urllib.request.Request(FEED.format(channel_id),
                                 headers={"User-Agent": "ViewCastLK/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = r.read()
    except urllib.error.HTTPError as e:
        if e.code == 429:
            _rate_limited.set()
            return "rate_limited", 0, False, time.time() - t0
        return f"http_{e.code}", 0, False, time.time() - t0
    except Exception as e:
        return f"err_{type(e).__name__}", 0, False, time.time() - t0

    try:
        root = ET.fromstring(body)
    except ET.ParseError:
        return "parse_error", 0, False, time.time() - t0

    n, oldest = 0, None
    for e in root.findall("a:entry", NS):
        p = datetime.fromisoformat(e.find("a:published", NS).text)
        oldest = p if oldest is None else min(oldest, p)
        if p >= since:
            n += 1
    covers = oldest is not None and oldest < since
    return "ok", n, covers, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--timeout", type=int, default=20)
    ap.add_argument("--lookback-hours", type=int, default=26)
    ap.add_argument("--delay", type=float, default=0.05,
                    help="per-request jittered delay, to stay polite")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    conn = connect()
    active = pd.read_sql_query(
        "select channel_id from channels where active", conn).channel_id.tolist()
    conn.close()
    cand_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "channels_to_add.csv")
    cands = pd.read_csv(cand_path).channel_id.tolist() if os.path.exists(cand_path) else []
    ids = active + cands
    if args.limit:
        ids = ids[:args.limit]
    print(f"{len(active)} active + {len(cands)} candidates = {len(ids):,} channels")
    print(f"{args.workers} workers, {args.timeout}s timeout, "
          f"{args.lookback_hours}h look-back\n")

    since = datetime.now(timezone.utc) - timedelta(hours=args.lookback_hours)
    results, t0 = [], time.time()

    def job(cid):
        time.sleep(random.uniform(0, args.delay))
        return (cid,) + fetch(cid, since, args.timeout)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(job, c): c for c in ids}
        for i, f in enumerate(as_completed(futs), 1):
            results.append(f.result())
            if i % 500 == 0:
                el = time.time() - t0
                print(f"  {i:,}/{len(ids):,}  {el:.0f}s  "
                      f"({i/el:.0f}/s, eta {(len(ids)-i)/(i/el):.0f}s)")

    took = time.time() - t0
    d = pd.DataFrame(results, columns=["channel_id", "status", "found",
                                       "covers", "secs"])

    print(f"\ncompleted {len(d):,} in {took:.0f}s  ({len(d)/took:.1f} feeds/s)")
    print(f"median request {d.secs.median():.2f}s, p95 {d.secs.quantile(.95):.2f}s\n")

    print("outcomes:")
    for k, v in Counter(d.status).most_common():
        print(f"  {k:24}{v:>7,}  {100*v/len(d):5.1f}%")

    ok = d[d.status == "ok"]
    hard_fail = d[~d.status.isin(["ok", "http_404"])]
    print(f"\nusable feeds: {len(ok):,}/{len(d):,} ({100*len(ok)/len(d):.1f}%)")
    print(f"deleted channels (404, expected): {(d.status == 'http_404').sum():,}")
    print(f"genuine failures: {len(hard_fail):,} ({100*len(hard_fail)/len(d):.2f}%)")

    need_api = int((~ok.covers).sum()) + len(hard_fail)
    print(f"\nchannels needing an API discovery call this run: {need_api:,}"
          f"  ({100*need_api/len(d):.1f}%)")
    print(f"  feed could not prove it covered the window: {int((~ok.covers).sum()):,}")
    print(f"  feed failed outright: {len(hard_fail):,}")
    print(f"videos found inside the window: {int(ok.found.sum()):,}")

    if _rate_limited.is_set():
        print("\nRATE LIMITED — YouTube returned 429. Lower --workers and retry; "
              "this endpoint has no published limit so treat it as fragile.")


if __name__ == "__main__":
    main()
