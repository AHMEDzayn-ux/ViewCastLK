"""
ViewCastLK - Automated YouTube Data Collection Script
------------------------------------------------------
Designed to run automatically on a schedule (e.g. via GitHub Actions).
Reads the API key securely from an environment variable, and appends
new data to the CSV each run so you build up a day-by-day / hour-by-hour
history instead of overwriting previous data.
"""

import csv
import os
import time
from dotenv import load_dotenv


from datetime import datetime, timezone
from googleapiclient.discovery import build

load_dotenv(override=True)

# ---------------------------------------------------------
# 1. SETUP - API key comes from environment variable (safe for GitHub Actions)
# ---------------------------------------------------------
API_KEY = os.environ.get("YOUTUBE_API_KEY")

if not API_KEY:
    raise ValueError(
        "No API key found. Set the YOUTUBE_API_KEY environment variable "
        "(locally: export YOUTUBE_API_KEY=your_key_here)"
    )

youtube = build("youtube", "v3", developerKey=API_KEY)

OUTPUT_FILE = "youtube_data.csv"

# ---------------------------------------------------------
# 2. YOUR CURATED LIST OF SRI LANKAN CHANNELS
# ---------------------------------------------------------
CHANNEL_HANDLES = [
    "@AdaDerana",
    "@HiruNewsOfficial",
    # add more channels here, across your categories
]


# ---------------------------------------------------------
# 3. Resolve a channel handle -> channelId + uploads playlist ID
# ---------------------------------------------------------
def get_channel_info(handle):
    request = youtube.channels().list(
        part="id,statistics,contentDetails,snippet",
        forHandle=handle.lstrip("@")
    )
    response = request.execute()

    if not response.get("items"):
        print(f"  [!] Could not resolve channel: {handle}")
        return None

    item = response["items"][0]
    return {
        "channel_id": item["id"],
        "channel_title": item["snippet"]["title"],
        "subscriber_count": item["statistics"].get("subscriberCount"),
        "channel_video_count": item["statistics"].get("videoCount"),
        "channel_view_count": item["statistics"].get("viewCount"),
        "channel_created_at": item["snippet"]["publishedAt"],
        "uploads_playlist_id": item["contentDetails"]["relatedPlaylists"]["uploads"],
    }


# ---------------------------------------------------------
# 4. Get all video IDs from a channel's uploads playlist
# ---------------------------------------------------------
def get_all_video_ids(uploads_playlist_id, max_videos=5):
    video_ids = []
    next_page_token = None

    while len(video_ids) < max_videos:
        request = youtube.playlistItems().list(
            part="contentDetails",
            playlistId=uploads_playlist_id,
            maxResults=50,
            pageToken=next_page_token
        )
        response = request.execute()

        for item in response.get("items", []):
            video_ids.append(item["contentDetails"]["videoId"])

        next_page_token = response.get("nextPageToken")
        if not next_page_token:
            break

    return video_ids[:max_videos]


# ---------------------------------------------------------
# 5. Batch-fetch full video stats (up to 50 IDs per call)
# ---------------------------------------------------------
def get_video_details(video_ids):
    all_videos = []
    collected_at = datetime.now(timezone.utc).isoformat()

    for i in range(0, len(video_ids), 50):
        batch = video_ids[i:i + 50]
        request = youtube.videos().list(
            part="snippet,statistics,contentDetails",
            id=",".join(batch)
        )
        response = request.execute()

        for item in response.get("items", []):
            all_videos.append({
                "collected_at": collected_at,   # timestamp of THIS collection run
                "video_id": item["id"],
                "title": item["snippet"]["title"],
                "published_at": item["snippet"]["publishedAt"],
                "category_id": item["snippet"].get("categoryId"),
                "duration": item["contentDetails"]["duration"],
                "view_count": item["statistics"].get("viewCount"),
                "like_count": item["statistics"].get("likeCount"),
                "comment_count": item["statistics"].get("commentCount"),
            })

        time.sleep(0.2)

    return all_videos


# ---------------------------------------------------------
# 6. MAIN - runs the whole pipeline, APPENDS to CSV (builds history over time)
# ---------------------------------------------------------
def main():
    all_rows = []

    for handle in CHANNEL_HANDLES:
        print(f"Processing channel: {handle}")
        channel_info = get_channel_info(handle)
        if not channel_info:
            continue

        video_ids = get_all_video_ids(channel_info["uploads_playlist_id"])
        print(f"  Found {len(video_ids)} videos")

        videos = get_video_details(video_ids)

        for v in videos:
            v.update({
                "channel_id": channel_info["channel_id"],
                "channel_title": channel_info["channel_title"],
                "subscriber_count": channel_info["subscriber_count"],
                "channel_video_count": channel_info["channel_video_count"],
                "channel_view_count": channel_info["channel_view_count"],
                "channel_created_at": channel_info["channel_created_at"],
            })
            all_rows.append(v)

    if not all_rows:
        print("No data collected.")
        return

    file_exists = os.path.isfile(OUTPUT_FILE)
    keys = all_rows[0].keys()

    # append mode ("a") - each run ADDS new rows, doesn't erase old ones
    with open(OUTPUT_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        if not file_exists:
            writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nDone! Appended {len(all_rows)} rows to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
