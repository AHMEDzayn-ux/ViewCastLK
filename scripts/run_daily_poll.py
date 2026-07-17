"""The recurring collection job — this is the script that becomes a scheduled
GitHub Actions workflow (e.g. cron: daily). Each run:
  1. Refreshes identity + stats for every tracked channel.
  2. Discovers videos published since the last poll.
  3. Re-snapshots every video still inside its 60-day tracking window, not just
     newly-discovered ones — this is what actually builds the day-by-day
     view/like/comment trajectory the forecasting model needs.

Swap point for the Supabase migration: only storage.py needs to change. This
file's outputs are already shaped as the target Supabase tables' columns.
"""
import time
from datetime import datetime, timedelta, timezone

from youtube_client import (
    get_channel_by_roster_entry,
    get_channel_videos_since,
    get_video_categories,
    get_video_details,
    flatten_channel_identity,
    flatten_channel_snapshot,
    flatten_video_identity,
    flatten_video_snapshot,
)
from storage import append_rows, load_active_video_ids
from channel_roster import load_handles

TRACKING_WINDOW_DAYS = 60
DISCOVERY_LOOKBACK_HOURS = 26  # >24h buffer so a daily cron never drops a video to timing drift

CHANNELS_PATH = "output/channels.csv"
CHANNEL_SNAPSHOTS_PATH = "output/channel_snapshots.csv"
VIDEOS_PATH = "output/videos.csv"
VIDEO_SNAPSHOTS_PATH = "output/video_snapshots.csv"


def main():
    start = time.time()
    now = datetime.now(timezone.utc)
    captured_at = now.isoformat()
    discovery_since = (now - timedelta(hours=DISCOVERY_LOOKBACK_HOURS)).strftime("%Y-%m-%dT%H:%M:%SZ")
    tracking_since = (now - timedelta(days=TRACKING_WINDOW_DAYS)).strftime("%Y-%m-%dT%H:%M:%SZ")

    handles = load_handles()
    print(f"Refreshing {len(handles)} tracked channels...")
    channels = [c for c in (get_channel_by_roster_entry(h) for h in handles) if c]
    append_rows([flatten_channel_identity(c) for c in channels], CHANNELS_PATH)
    append_rows([flatten_channel_snapshot(c, captured_at) for c in channels], CHANNEL_SNAPSHOTS_PATH)

    print(f"Discovering videos published since {discovery_since}...")
    new_video_ids = set()
    for c in channels:
        new_video_ids.update(get_channel_videos_since(c["id"], discovery_since))
    print(f"  {len(new_video_ids)} new videos found")

    active_video_ids = load_active_video_ids(VIDEOS_PATH, tracking_since)
    video_ids_to_snapshot = active_video_ids | new_video_ids
    print(f"Snapshotting {len(video_ids_to_snapshot)} videos "
          f"({len(active_video_ids)} still-active + {len(new_video_ids)} new)...")

    category_names = get_video_categories()
    details = get_video_details(list(video_ids_to_snapshot))

    new_details = [v for v in details if v["id"] in new_video_ids]
    append_rows([flatten_video_identity(v, category_names) for v in new_details], VIDEOS_PATH)
    append_rows([flatten_video_snapshot(v, captured_at) for v in details], VIDEO_SNAPSHOTS_PATH)

    elapsed = time.time() - start
    print(f"Done in {elapsed:.1f}s. {len(channels)} channels, "
          f"{len(new_details)} new videos, {len(details)} total snapshotted.")


if __name__ == "__main__":
    main()
