import os
import sys
from dotenv import load_dotenv
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

sys.stdout.reconfigure(encoding="utf-8")
load_dotenv()

API_KEY = os.environ["YOUTUBE_API_KEY"]
youtube = build("youtube", "v3", developerKey=API_KEY)


CHANNEL_PARTS = "snippet,statistics,contentDetails,topicDetails"


def get_channel_info(channel_id: str) -> dict:
    """1 unit. Channel-level metadata: subscriber count, channel age, uploads playlist id."""
    response = youtube.channels().list(
        part=CHANNEL_PARTS,
        id=channel_id,
    ).execute()
    items = response.get("items", [])
    return items[0] if items else None


def get_channel_by_handle(handle: str) -> dict:
    """1 unit. Resolves a YouTube @handle directly to full channel info —
    used for onboarding a pre-curated list of handles (e.g. from Social Blade's
    top-creators list) without spending search.list quota on discovery."""
    response = youtube.channels().list(
        part=CHANNEL_PARTS,
        forHandle=handle,
    ).execute()
    items = response.get("items", [])
    return items[0] if items else None


def get_channel_by_roster_entry(value: str) -> dict:
    """1 unit. Accepts either a raw channel ID ('UC...', 24 chars) or an @handle
    and resolves it the right way — lets the roster mix both, since not every
    discovered channel has a clean custom handle to key off of."""
    if value.startswith("UC") and len(value) == 24:
        response = youtube.channels().list(part=CHANNEL_PARTS, id=value).execute()
    else:
        response = youtube.channels().list(part=CHANNEL_PARTS, forHandle=value).execute()
    items = response.get("items", [])
    return items[0] if items else None


def get_uploads_playlist_id(channel_id: str) -> str:
    channel = get_channel_info(channel_id)
    return channel["contentDetails"]["relatedPlaylists"]["uploads"]


def get_videos_from_playlist(playlist_id: str, max_results: int = 50) -> list[dict]:
    """1 unit per page (up to 50 items/page). Enumerates a channel's uploaded videos cheaply."""
    videos = []
    page_token = None
    while True:
        response = youtube.playlistItems().list(
            part="snippet,contentDetails",
            playlistId=playlist_id,
            maxResults=50,
            pageToken=page_token,
        ).execute()
        videos.extend(response.get("items", []))
        page_token = response.get("nextPageToken")
        if not page_token or len(videos) >= max_results:
            break
    return videos[:max_results]


def is_sri_lankan_channel(channel_id: str) -> bool:
    """Checks the channel's self-declared country. Many channels leave this blank,
    so a False here doesn't prove the channel isn't Sri Lankan — treat as one signal,
    not a guarantee, and expect to supplement with manual curation."""
    channel = get_channel_info(channel_id)
    return bool(channel) and channel["snippet"].get("country") == "LK"


def get_channel_videos_since(channel_id: str, since_date: str) -> list[dict]:
    """Convenience wrapper that resolves the uploads playlist id first (1 extra
    channels.list unit). Prefer get_channel_videos_since_by_playlist() whenever
    you already have the channel's resolved data (e.g. from a prior
    channels.list call this same run) — that extra lookup is pure waste when
    the playlist id is already sitting in memory."""
    playlist_id = get_uploads_playlist_id(channel_id)
    return get_channel_videos_since_by_playlist(playlist_id, since_date)


def get_channel_videos_since_by_playlist(playlist_id: str, since_date: str) -> list[dict]:
    """Cheap discovery: walks a channel's uploads playlist and keeps only videos
    published on/after since_date (ISO format, e.g. '2026-06-01').
    Uses contentDetails.videoPublishedAt, which playlistItems.list returns for free —
    no extra videos.list call needed just to check the date.
    Uploads playlists are returned newest-first (confirmed empirically), so we stop
    as soon as we hit a video older than the cutoff instead of paging full history —
    this is what makes daily polling cheap: a same-day check only costs 1 page for
    most channels, not a full re-walk of the channel's upload history each time.
    Some channels' uploads playlists 404 (playlistNotFound) — e.g. channels with
    zero uploads — which is a legitimate "no videos" outcome, not a failure."""
    matches = []
    page_token = None
    while True:
        try:
            response = youtube.playlistItems().list(
                part="contentDetails",
                playlistId=playlist_id,
                maxResults=50,
                pageToken=page_token,
            ).execute()
        except HttpError as e:
            if e.resp.status == 404:
                return matches
            raise
        hit_older_video = False
        for item in response.get("items", []):
            published_at = item["contentDetails"]["videoPublishedAt"]
            if published_at >= since_date:
                matches.append(item["contentDetails"]["videoId"])
            else:
                hit_older_video = True
                break
        if hit_older_video:
            break
        page_token = response.get("nextPageToken")
        if not page_token:
            break
    return matches


def get_channel_videos_since_search(channel_id: str, since_date: str, max_results: int = None) -> list[str]:
    """100 units PER PAGE (not per call) — expensive, but terminates immediately via a real
    server-side date filter instead of paging a channel's full history like
    get_channel_videos_since() does. Use this only for the first poll of a newly-added
    channel; use get_channel_videos_since() for the cheap ongoing daily poll.
    since_date must be RFC 3339, e.g. '2026-07-10T00:00:00Z'.
    No cap by default — pages until exhausted, so it captures every matching video;
    pass max_results only if you want to deliberately limit quota spend during testing."""
    video_ids = []
    page_token = None
    while True:
        response = youtube.search().list(
            part="snippet",
            channelId=channel_id,
            type="video",
            publishedAfter=since_date,
            order="date",
            maxResults=50,
            pageToken=page_token,
        ).execute()
        video_ids.extend(item["id"]["videoId"] for item in response.get("items", []))
        page_token = response.get("nextPageToken")
        if not page_token or (max_results and len(video_ids) >= max_results):
            break
    return video_ids[:max_results] if max_results else video_ids


def discover_channel_candidates(keyword: str, max_results: int = 25) -> list[str]:
    """100 units per call. Returns candidate channel IDs for a keyword search —
    run this occasionally to seed your channel list, then verify each result with
    is_sri_lankan_channel() before adding it. Not meant to run on every poll."""
    response = youtube.search().list(
        part="snippet",
        q=keyword,
        type="channel",
        regionCode="LK",
        maxResults=max_results,
    ).execute()
    return [item["snippet"]["channelId"] for item in response.get("items", [])]


VIDEO_PARTS = "snippet,statistics,contentDetails,status,liveStreamingDetails"


def get_video_details(video_ids: list[str]) -> list[dict]:
    """1 unit per call regardless of how many ids (up to 50 ids per call).
    Batch ids together instead of calling once per video."""
    details = []
    for i in range(0, len(video_ids), 50):
        chunk = video_ids[i:i + 50]
        response = youtube.videos().list(
            part=VIDEO_PARTS,
            id=",".join(chunk),
        ).execute()
        details.extend(response.get("items", []))
    return details


def get_video_categories(region_code: str = "LK") -> dict[str, str]:
    """1 unit. Maps numeric category_id -> human-readable name (e.g. '25' -> 'News & Politics').
    Fetch once per run/session and reuse — categories don't change day to day."""
    response = youtube.videoCategories().list(
        part="snippet",
        regionCode=region_code,
    ).execute()
    return {item["id"]: item["snippet"]["title"] for item in response.get("items", [])}


def search_videos(query: str, published_after: str = None, region_code: str = "LK",
                   video_category_id: str = None, max_results: int = 25) -> list[dict]:
    """100 units per call — expensive, use sparingly (only to fill category gaps)."""
    response = youtube.search().list(
        part="snippet",
        q=query,
        type="video",
        regionCode=region_code,
        publishedAfter=published_after,
        videoCategoryId=video_category_id,
        maxResults=max_results,
        order="date",
    ).execute()
    return response.get("items", [])


# --- Flatten functions: raw API responses -> plain flat dicts, ready for storage.py.
# No file I/O here on purpose — these are the columns that end up in Supabase tables
# once the persistence layer is swapped from CSV to a real database.

def flatten_channel_identity(c: dict) -> dict:
    """Fields that rarely change — one row per channel, not per poll."""
    return {
        "channel_id": c["id"],
        "title": c["snippet"]["title"],
        "description": c["snippet"].get("description", ""),
        "country": c["snippet"].get("country", ""),
        "channel_published_at": c["snippet"]["publishedAt"],
        "uploads_playlist_id": c["contentDetails"]["relatedPlaylists"]["uploads"],
        "topic_categories": "|".join(c.get("topicDetails", {}).get("topicCategories", [])),
    }


def flatten_channel_snapshot(c: dict, captured_at: str) -> dict:
    """Fields that change over time — one row per channel per poll."""
    stats = c["statistics"]
    return {
        "channel_id": c["id"],
        "captured_at": captured_at,
        "subscriber_count": stats.get("subscriberCount", ""),
        "hidden_subscriber_count": stats.get("hiddenSubscriberCount", ""),
        "view_count": stats.get("viewCount", ""),
        "video_count": stats.get("videoCount", ""),
    }


def flatten_video_identity(v: dict, category_names: dict[str, str] = None) -> dict:
    """Fields that rarely change — one row per video, written only when first discovered."""
    category_names = category_names or {}
    snippet = v.get("snippet", {})
    content_details = v.get("contentDetails", {})
    category_id = snippet.get("categoryId", "")
    thumbnails = snippet.get("thumbnails", {})
    thumbnail = thumbnails.get("high") or thumbnails.get("default") or {}
    return {
        "video_id": v["id"],
        "channel_id": snippet["channelId"],
        "title": snippet["title"],
        "description": snippet.get("description", ""),
        "tags": "|".join(snippet.get("tags", [])),
        "published_at": snippet["publishedAt"],
        "category_id": category_id,
        "category_name": category_names.get(category_id, ""),
        "duration": content_details.get("duration", ""),
        "definition": content_details.get("definition", ""),
        "caption": content_details.get("caption", ""),
        "default_audio_language": snippet.get("defaultAudioLanguage", ""),
        "default_language": snippet.get("defaultLanguage", ""),
        "thumbnail_url": thumbnail.get("url", ""),
        "made_for_kids": v.get("status", {}).get("madeForKids", ""),
    }


def flatten_video_snapshot(v: dict, captured_at: str) -> dict:
    """Fields that change over time — one row per video per poll, for as long as
    the video stays inside its tracking window."""
    stats = v.get("statistics", {})
    live_details = v.get("liveStreamingDetails", {})
    return {
        "video_id": v["id"],
        "captured_at": captured_at,
        "view_count": stats.get("viewCount", ""),
        "like_count": stats.get("likeCount", ""),
        "comment_count": stats.get("commentCount", ""),
        "live_broadcast_content": v.get("snippet", {}).get("liveBroadcastContent", ""),
        "live_actual_start_time": live_details.get("actualStartTime", ""),
        "live_actual_end_time": live_details.get("actualEndTime", ""),
        "live_concurrent_viewers": live_details.get("concurrentViewers", ""),
    }
