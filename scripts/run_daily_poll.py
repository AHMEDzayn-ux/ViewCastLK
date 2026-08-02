"""The recurring collection job — runs on a schedule via GitHub Actions
(.github/workflows/collect.yml). There are two kinds of run, selected by the
REFRESH_CHANNELS env var (the workflow sets it per cron slot):

  Full run (REFRESH_CHANNELS=true, twice a day):
    1. Refreshes identity + stats for every ACTIVE tracked channel, looking
       them up by id fifty at a time (~19 units for the whole roster; it was
       ~1,282 when every handle was resolved singly).
    2. Discovers videos published since the last poll.
    3. Snapshots every video still inside its 60-day tracking window.

  Discovery-only run (REFRESH_CHANNELS=false, the other two of the four runs):
    - Skips the channel refresh entirely. Subscriber and view counts barely move
      within a few hours, so refreshing them 4x/day rather than 2x buys almost
      no extra signal. Reads the active channel list from the DB instead, then
      does discovery + video snapshots exactly as a full run does.

Discovery reads YouTube's free per-channel Atom feed first and spends quota only
on channels whose feed failed or could not prove it reached back past the
cutoff. Measured across 2,781 channels that is about 0.5% of them. The fallback
is capped per run, so RSS degrading costs at most what discovery used to cost
rather than the day's whole allowance. See rss_discovery.py.

Channels flagged inactive are skipped by both kinds of run. An activity sweep
found 679 of 1,282 rostered channels had not uploaded in over 60 days (median
silence 414 days), and each was still costing a discovery call four times a
day. Their rows and collected videos are retained; only the polling stops.

Video snapshots — the day-by-day trajectory the model learns from — happen on
EVERY run regardless. That is the data that must stay dense, and it is what
puts each horizon within about three hours of its mark.

Persists to Supabase via storage.py.

Safety: channel resolution (full runs only) writes every 50 channels instead
of all at once, so a mid-run quotaExceeded loses at most one batch. If quota
runs out at any point, the script stops cleanly with a clear message instead
of a crash/traceback — whatever was already written stays in the database.
"""
import os
import time
from datetime import datetime, timedelta, timezone

from googleapiclient.errors import HttpError

import rss_discovery

from youtube_client import (
    get_channel_by_roster_entry,
    get_channels_by_ids,
    normalise_handle,
    get_channel_videos_since_by_playlist,
    get_video_categories,
    get_video_details,
    flatten_channel_identity,
    flatten_channel_snapshot,
    flatten_video_identity,
    flatten_video_snapshot,
)
from storage import (
    append_rows,
    load_active_channels,
    ensure_snapshot_partitions,
    load_roster_mapping,
    load_active_video_ids,
    load_known_ids,
)
from channel_roster import load_handles

TRACKING_WINDOW_DAYS = 60

# >24h buffer so a daily cron never drops a video to timing drift. Overridable
# because a one-off backfill is the same operation over a longer reach: uploads
# published before a channel entered the roster, or during a run that failed,
# were never discovered and no amount of future polling will find them. Setting
# this to e.g. 744 (31 days) walks each playlist back that far, inserts the
# identities it finds and lets the ordinary snapshot pass pick them up. Labels
# are only recoverable while the video is younger than the horizon, so the
# reach that pays shrinks by a day every day.
DISCOVERY_LOOKBACK_HOURS = int(os.environ.get("DISCOVERY_LOOKBACK_HOURS", "26"))

# The metered fallback reaches further back than RSS. A channel the cap pushed
# out of one run must be findable by the next, or the gap is permanent.
FALLBACK_LOOKBACK_HOURS = int(os.environ.get("FALLBACK_LOOKBACK_HOURS", "50"))

# Ceiling on how many channels one run may re-check through the API. Bounds the
# cost of RSS failing: at worst discovery costs what it did before RSS existed,
# rather than exhausting the day's allowance in a single run. Channels beyond
# the cap are found by the next run, whose longer look-back still covers them.
FALLBACK_CALL_CAP = int(os.environ.get("FALLBACK_CALL_CAP", "500"))

CHANNEL_BATCH_SIZE = 50

# Full run refreshes channel stats; discovery-only run skips that to save quota.
# Defaults to a full run when unset (e.g. a local manual invocation).
REFRESH_CHANNELS = os.environ.get("REFRESH_CHANNELS", "true").strip().lower() == "true"

CHANNELS_TABLE = "channels"
CHANNEL_SNAPSHOTS_TABLE = "channel_snapshots"
VIDEOS_TABLE = "videos"
VIDEO_SNAPSHOTS_TABLE = "video_snapshots"


class QuotaExceeded(Exception):
    pass


def is_quota_exceeded(e: HttpError) -> bool:
    return e.resp.status == 403 and "quotaExceeded" in str(e)


def _persist_channels(batch_channels: list[dict], captured_at: str,
                      known_channel_ids: set[str]) -> None:
    """Writes one batch: identity rows only for channels never seen before,
    snapshot rows for all of them. Writing per batch rather than at the end
    means a quota cutoff partway through loses at most the current batch."""
    if not batch_channels:
        return
    new_channels = [c for c in batch_channels if c["id"] not in known_channel_ids]
    append_rows([flatten_channel_identity(c) for c in new_channels], CHANNELS_TABLE)
    known_channel_ids.update(c["id"] for c in new_channels)
    append_rows([flatten_channel_snapshot(c, captured_at) for c in batch_channels],
                CHANNEL_SNAPSHOTS_TABLE)


def resolve_channels(handles: list[str], captured_at: str, known_channel_ids: set[str]) -> list[dict]:
    """Refreshes channel identity and statistics for the roster.

    Two paths, because the API prices them differently. A channel already in
    the warehouse is looked up by id, and channels.list takes fifty ids per
    call for one unit — so the whole known roster costs a handful of units
    instead of one per channel. A handle we have never resolved has no id yet,
    and forHandle accepts exactly one value per call, so those still go singly.
    That is the only reason the slow path survives.

    Channels flagged inactive are skipped outright: the activity sweep already
    established they have stopped uploading, and paying to rediscover that
    twice a day is the waste this split exists to remove.

    A channel that still errors after youtube_client's retries is skipped
    rather than aborting the run — one bad channel should not cost the cycle.
    Failures are counted and reported at the end."""
    mapping = load_roster_mapping(CHANNELS_TABLE)
    known_ids, unknown, skipped = [], [], 0
    for h in handles:
        entry = mapping.get(normalise_handle(h))
        if entry is None:
            unknown.append(h)
        elif not entry[1]:
            skipped += 1
        else:
            known_ids.append(entry[0])

    est = -(-len(known_ids) // CHANNEL_BATCH_SIZE) + len(unknown)
    print(f"  {len(known_ids)} known (batched), {len(unknown)} new (single), "
          f"{skipped} inactive skipped — about {est} units")

    channels, failed = [], []

    for i in range(0, len(known_ids), CHANNEL_BATCH_SIZE):
        batch = known_ids[i:i + CHANNEL_BATCH_SIZE]
        try:
            got = get_channels_by_ids(batch)
        except HttpError as e:
            if is_quota_exceeded(e):
                raise QuotaExceeded()
            failed.append((f"batch@{i}", f"HTTP {e.resp.status}"))
            continue
        except Exception as e:                          # transport/DNS/timeout
            failed.append((f"batch@{i}", type(e).__name__))
            continue
        # ids YouTube cannot return are absent rather than an error
        missing = len(batch) - len(got)
        if missing:
            failed.append((f"batch@{i}", f"{missing} id(s) not returned"))
        _persist_channels(got, captured_at, known_channel_ids)
        channels.extend(got)
        print(f"  refreshed {len(channels)}/{len(known_ids)} known channels...")

    for h in unknown:
        try:
            c = get_channel_by_roster_entry(h)
        except HttpError as e:
            if is_quota_exceeded(e):
                raise QuotaExceeded()
            failed.append((h, f"HTTP {e.resp.status}"))
            continue
        except Exception as e:                          # transport/DNS/timeout
            failed.append((h, type(e).__name__))
            continue
        if c:
            _persist_channels([c], captured_at, known_channel_ids)
            channels.append(c)
    if unknown:
        print(f"  resolved {len(channels) - len(known_ids)} new channel(s) from handles")

    if failed:
        print(f"  WARNING: skipped {len(failed)} channel(s) after retries: "
              + ", ".join(f"{h} ({why})" for h, why in failed[:8])
              + (" ..." if len(failed) > 8 else ""))
    return channels


def discover_new_videos(channels: list[tuple[str, str]], discovery_since: str,
                        fallback_since: str) -> set[str]:
    """Find uploads newer than the cutoff, free where possible.

    Two passes. The first reads YouTube's per-channel Atom feed, which costs no
    quota; measured across the whole roster it covers about 99.5% of channels.
    The second spends quota only on channels whose feed failed or could not
    prove it reached back past the cutoff, and is capped so that RSS degrading
    -- or disappearing -- cannot exhaust the day's allowance in one run. At the
    cap, discovery simply costs what it used to.

    The fallback uses a longer look-back than RSS. A channel that was capped
    out of one run needs the next one to reach further back, or the gap becomes
    permanent.

    A playlist that still errors after retries is skipped rather than aborting
    discovery for every other channel."""
    ids_by_channel = dict(channels)
    rss = rss_discovery.discover([c for c, _ in channels], discovery_since)
    print(f"  {rss.summary(len(channels))}")
    if rss.rate_limited:
        print("  WARNING: RSS rate-limited this run; the API fallback is capped, "
              "so some channels may be picked up next run instead.")

    new_video_ids = set(rss.video_ids)

    to_check = rss.needs_api[:FALLBACK_CALL_CAP]
    if len(rss.needs_api) > FALLBACK_CALL_CAP:
        print(f"  fallback capped: {len(rss.needs_api)} channels need the API, "
              f"checking {FALLBACK_CALL_CAP} this run; the rest are covered by "
              f"the next run's look-back")

    failed = 0
    for channel_id in to_check:
        playlist_id = ids_by_channel.get(channel_id)
        if not playlist_id:
            continue
        try:
            new_video_ids.update(
                get_channel_videos_since_by_playlist(playlist_id, fallback_since))
        except HttpError as e:
            if is_quota_exceeded(e):
                raise QuotaExceeded()
            failed += 1
        except Exception:                               # transport/DNS/timeout
            failed += 1
    if to_check:
        print(f"  API fallback: {len(to_check)} channel(s) checked"
              + (f", {failed} failed after retries" if failed else ""))
    return new_video_ids


def main():
    start = time.time()
    now = datetime.now(timezone.utc)
    captured_at = now.isoformat()
    discovery_since = (now - timedelta(hours=DISCOVERY_LOOKBACK_HOURS)).strftime("%Y-%m-%dT%H:%M:%SZ")
    fallback_since = (now - timedelta(hours=FALLBACK_LOOKBACK_HOURS)).strftime("%Y-%m-%dT%H:%M:%SZ")
    tracking_since = (now - timedelta(days=TRACKING_WINDOW_DAYS)).strftime("%Y-%m-%dT%H:%M:%SZ")

    # video_snapshots is partitioned by day; make sure this run has somewhere
    # to write before it starts spending quota gathering rows.
    made = ensure_snapshot_partitions(14)
    if made:
        print(f"Created {made} snapshot partition(s).")

    known_video_ids = load_known_ids(VIDEOS_TABLE, "video_id")

    try:
        if REFRESH_CHANNELS:
            handles = load_handles()
            known_channel_ids = load_known_ids(CHANNELS_TABLE, "channel_id")
            print(f"Full run: refreshing {len(handles)} tracked channels...")
            channels = resolve_channels(handles, captured_at, known_channel_ids)
            # Discovery reads the stored playlist ids rather than the ones just
            # resolved, so that channels flagged inactive are skipped here too.
            # Resolution still walks the whole roster — it works from handles and
            # cannot tell which are inactive until the call has been made — but
            # that waste is confined to the two full runs instead of all four.
            active_channels = load_active_channels(CHANNELS_TABLE)
            channel_count = len(channels)
        else:
            active_channels = load_active_channels(CHANNELS_TABLE)
            channel_count = len(active_channels)
            print(f"Discovery-only run: skipping channel refresh, "
                  f"discovering from {channel_count} stored channels...")

        print(f"Discovering videos published since {discovery_since}...")
        new_video_ids = discover_new_videos(active_channels, discovery_since,
                                            fallback_since)
        print(f"  {len(new_video_ids)} new videos found")

        active_video_ids = load_active_video_ids(VIDEOS_TABLE, tracking_since)
        video_ids_to_snapshot = active_video_ids | new_video_ids
        print(f"Snapshotting {len(video_ids_to_snapshot)} videos "
              f"({len(active_video_ids)} still-active + {len(new_video_ids)} new)...")

        category_names = get_video_categories()
        details = get_video_details(list(video_ids_to_snapshot))

        unseen_details = [v for v in details if v["id"] not in known_video_ids]
        append_rows([flatten_video_identity(v, category_names) for v in unseen_details], VIDEOS_TABLE)
        append_rows([flatten_video_snapshot(v, captured_at) for v in details], VIDEO_SNAPSHOTS_TABLE)

        elapsed = time.time() - start
        run_kind = "full" if REFRESH_CHANNELS else "discovery-only"
        print(f"Done ({run_kind}) in {elapsed:.1f}s. {channel_count} channels, "
              f"{len(unseen_details)} new videos, {len(details)} total snapshotted.")

    except QuotaExceeded:
        elapsed = time.time() - start
        print(f"\nQUOTA EXCEEDED after {elapsed:.1f}s — stopped cleanly. "
              f"Everything resolved/snapshotted so far is already committed to the database. "
              f"Re-run after quota resets to continue.")


if __name__ == "__main__":
    main()
