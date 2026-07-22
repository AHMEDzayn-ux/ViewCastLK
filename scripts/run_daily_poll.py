"""The recurring collection job — runs on a schedule via GitHub Actions
(.github/workflows/collect.yml). There are two kinds of run, selected by the
REFRESH_CHANNELS env var (the workflow sets it per cron slot):

  Full run (REFRESH_CHANNELS=true, twice a day):
    1. Refreshes identity + stats for every tracked channel (channels.list —
       the single most expensive part, ~1 unit per channel).
    2. Discovers videos published since the last poll.
    3. Snapshots every video still inside its 60-day tracking window.

  Discovery-only run (REFRESH_CHANNELS=false, the other two of the four runs):
    - Skips the channel refresh entirely. Channel subscriber/view counts barely
      move within a few hours, so refreshing them 4x/day instead of 2x wastes
      ~1,282 units per skipped run for almost no extra signal. Reads each
      channel's uploads-playlist id from the DB instead, then does discovery +
      video snapshots exactly as a full run does.

Video snapshots (the actual day-by-day trajectory the model needs) happen on
EVERY run, full or discovery-only — that's the data that must stay dense.

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

from youtube_client import (
    get_channel_by_roster_entry,
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
    load_active_video_ids,
    load_known_ids,
    load_channel_playlist_ids,
)
from channel_roster import load_handles

TRACKING_WINDOW_DAYS = 60
DISCOVERY_LOOKBACK_HOURS = 26  # >24h buffer so a daily cron never drops a video to timing drift
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


def resolve_channels(handles: list[str], captured_at: str, known_channel_ids: set[str]) -> list[dict]:
    """Resolves channels in batches, writing snapshot rows every run (subscriber
    count etc. genuinely changes) but identity rows only the first time a channel
    is ever seen — known_channel_ids is checked (and updated) as we go so identity
    data isn't re-appended every single run just because the channel was refreshed.
    Batched writes mean a quota cutoff partway through only loses the current
    batch's channels, not everything resolved so far."""
    channels = []
    for i in range(0, len(handles), CHANNEL_BATCH_SIZE):
        batch = handles[i:i + CHANNEL_BATCH_SIZE]
        batch_channels = []
        stopped = False
        for h in batch:
            try:
                c = get_channel_by_roster_entry(h)
            except HttpError as e:
                if is_quota_exceeded(e):
                    stopped = True
                    break
                raise
            if c:
                batch_channels.append(c)

        if batch_channels:
            new_channels = [c for c in batch_channels if c["id"] not in known_channel_ids]
            append_rows([flatten_channel_identity(c) for c in new_channels], CHANNELS_TABLE)
            known_channel_ids.update(c["id"] for c in new_channels)
            append_rows([flatten_channel_snapshot(c, captured_at) for c in batch_channels], CHANNEL_SNAPSHOTS_TABLE)
            channels.extend(batch_channels)
        print(f"  resolved {len(channels)}/{len(handles)} channels...")

        if stopped:
            raise QuotaExceeded()
    return channels


def discover_new_videos(playlist_ids: list[str], discovery_since: str) -> set[str]:
    """Walks each channel's uploads playlist for videos newer than the cutoff.
    Takes playlist ids directly, so it works the same whether they came from a
    just-completed channel refresh (full run) or straight from the DB
    (discovery-only run)."""
    new_video_ids = set()
    for playlist_id in playlist_ids:
        try:
            new_video_ids.update(get_channel_videos_since_by_playlist(playlist_id, discovery_since))
        except HttpError as e:
            if is_quota_exceeded(e):
                raise QuotaExceeded()
            raise
    return new_video_ids


def main():
    start = time.time()
    now = datetime.now(timezone.utc)
    captured_at = now.isoformat()
    discovery_since = (now - timedelta(hours=DISCOVERY_LOOKBACK_HOURS)).strftime("%Y-%m-%dT%H:%M:%SZ")
    tracking_since = (now - timedelta(days=TRACKING_WINDOW_DAYS)).strftime("%Y-%m-%dT%H:%M:%SZ")

    known_video_ids = load_known_ids(VIDEOS_TABLE, "video_id")

    try:
        if REFRESH_CHANNELS:
            handles = load_handles()
            known_channel_ids = load_known_ids(CHANNELS_TABLE, "channel_id")
            print(f"Full run: refreshing {len(handles)} tracked channels...")
            channels = resolve_channels(handles, captured_at, known_channel_ids)
            playlist_ids = [c["contentDetails"]["relatedPlaylists"]["uploads"] for c in channels]
            channel_count = len(channels)
        else:
            playlist_ids = load_channel_playlist_ids(CHANNELS_TABLE)
            channel_count = len(playlist_ids)
            print(f"Discovery-only run: skipping channel refresh, "
                  f"discovering from {channel_count} stored channels...")

        print(f"Discovering videos published since {discovery_since}...")
        new_video_ids = discover_new_videos(playlist_ids, discovery_since)
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
