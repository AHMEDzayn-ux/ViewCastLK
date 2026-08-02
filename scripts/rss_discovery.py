"""Free upload discovery through YouTube's per-channel Atom feed.

Discovery was 54% of the daily quota and is the one call that cannot be
batched: playlistItems.list takes a single playlistId, so it costs one unit per
channel per run however the roster grows. YouTube also publishes

    https://www.youtube.com/feeds/videos.xml?channel_id=UC...

which is not part of the Data API and is not metered. Measured across all 2,781
channels: 100% success, ~10 feeds/s at 12 workers, no rate limiting, and only
15 channels (0.5%) needed the API afterwards.

CORRECTNESS
The feed returns the 15 newest uploads and nothing more, so a channel
publishing faster than the look-back can overflow it. The count of entries is
useless as a signal -- it is always 15 for any established channel. What
matters is whether the feed reaches back past the cutoff:

    oldest entry older than the cutoff  ->  the window is fully covered
    oldest entry still inside it        ->  may be truncated, ask the API

Measured against playlistItems.list over 40 channels: every shortfall was
flagged by that rule and none was missed silently.

FAILURE
This endpoint is undocumented and could change or throttle without notice, so
it is never the only path. Any channel whose feed fails, or whose feed cannot
prove it covered the window, is handed to the API. The caller caps how much
quota that fallback may spend, which bounds the cost of RSS degrading -- in the
worst case discovery simply becomes what it is today.
"""
import random
import threading
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

NS = {"a": "http://www.w3.org/2005/Atom",
      "yt": "http://www.youtube.com/xml/schemas/2015"}
FEED = "https://www.youtube.com/feeds/videos.xml?channel_id={}"

WORKERS = 12          # ~10 feeds/s measured; no throttling observed at this rate
TIMEOUT = 20
JITTER = 0.05
RETRIES = 2


class RssResult:
    """What one run of RSS discovery found, and what it could not vouch for."""

    def __init__(self):
        self.video_ids: set[str] = set()
        self.needs_api: list[str] = []      # channel ids the API must re-check
        self.ok = 0
        self.failed = 0
        self.rate_limited = False

    def summary(self, total):
        covered = total - len(self.needs_api)
        return (f"RSS: {self.ok}/{total} feeds read, {len(self.video_ids)} videos, "
                f"{covered} channels fully covered, {len(self.needs_api)} need the API"
                + (" — RATE LIMITED" if self.rate_limited else ""))


def _fetch_one(channel_id, since, stop):
    """(channel_id, video_ids, covers_window, ok). Never raises."""
    for attempt in range(RETRIES + 1):
        if stop.is_set():
            return channel_id, set(), False, False
        try:
            time.sleep(random.uniform(0, JITTER))
            req = urllib.request.Request(
                FEED.format(channel_id),
                headers={"User-Agent": "ViewCastLK/1.0 (research collector)"})
            with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
                body = r.read()
            root = ET.fromstring(body)
            ids, oldest = set(), None
            for e in root.findall("a:entry", NS):
                published = datetime.fromisoformat(e.find("a:published", NS).text)
                oldest = published if oldest is None else min(oldest, published)
                if published >= since:
                    ids.add(e.find("yt:videoId", NS).text)
            # See CORRECTNESS above: coverage, not entry count, is the signal.
            return channel_id, ids, (oldest is not None and oldest < since), True
        except urllib.error.HTTPError as e:
            if e.code == 429:
                # Back off the whole run rather than hammering; the API fallback
                # will pick up everything this pass could not vouch for.
                stop.set()
                return channel_id, set(), False, False
            if e.code == 404:
                # No feed: a deleted channel, or one that never uploaded. There
                # is nothing for the API to find either, so this is covered.
                return channel_id, set(), True, True
            if attempt == RETRIES:
                return channel_id, set(), False, False
        except Exception:
            if attempt == RETRIES:
                return channel_id, set(), False, False
        time.sleep(0.5 * (attempt + 1))
    return channel_id, set(), False, False


def discover(channel_ids: list[str], since_iso: str,
             workers: int = WORKERS) -> RssResult:
    """Find uploads published on/after since_iso across many channels at once.

    A channel appears in .needs_api when its feed failed or could not prove it
    reached back past the cutoff. Callers must consult the API for those, or
    accept that the window may be incomplete for them."""
    since = datetime.fromisoformat(since_iso.replace("Z", "+00:00"))
    if since.tzinfo is None:
        since = since.replace(tzinfo=timezone.utc)

    out = RssResult()
    stop = threading.Event()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_fetch_one, c, since, stop) for c in channel_ids]
        for f in as_completed(futures):
            cid, ids, covers, ok = f.result()
            if ok:
                out.ok += 1
                out.video_ids |= ids
            else:
                out.failed += 1
            if not covers:
                out.needs_api.append(cid)
    out.rate_limited = stop.is_set()
    return out
