from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Iterable, Iterator
from zoneinfo import ZoneInfo

import httpx

from app import config


PACIFIC = ZoneInfo("America/Los_Angeles")
ANALYTICS_LAG_DAYS = 3
YOUTUBE_PLAYLIST_ITEMS_URL = "https://www.googleapis.com/youtube/v3/playlistItems"
YOUTUBE_VIDEOS_URL = "https://www.googleapis.com/youtube/v3/videos"
YOUTUBE_ANALYTICS_URL = "https://youtubeanalytics.googleapis.com/v2/reports"

CATEGORY_NAMES = {
    "1": "Film & Animation",
    "2": "Autos & Vehicles",
    "10": "Music",
    "15": "Pets & Animals",
    "17": "Sports",
    "18": "Short Movies",
    "19": "Travel & Events",
    "20": "Gaming",
    "21": "Videoblogging",
    "22": "People & Blogs",
    "23": "Comedy",
    "24": "Entertainment",
    "25": "News & Politics",
    "26": "Howto & Style",
    "27": "Education",
    "28": "Science & Technology",
    "29": "Nonprofits & Activism",
}


class CreatorSyncException(Exception):
    pass


@dataclass(frozen=True)
class CreatorVideo:
    video_id: str
    title: str
    category: str
    duration_seconds: int
    audio_language: str | None
    published_at: datetime
    is_short: bool


def batched_video_ids(video_ids: Iterable[str], batch_size: int = 10) -> Iterator[list[str]]:
    if batch_size < 1 or batch_size > 10:
        raise ValueError("YouTube Analytics batch size must be between 1 and 10")
    batch: list[str] = []
    for video_id in video_ids:
        batch.append(video_id)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def parse_iso8601_duration(value: str) -> int:
    match = re.fullmatch(
        r"P(?:(?P<days>\d+)D)?T(?:(?P<hours>\d+)H)?(?:(?P<minutes>\d+)M)?(?:(?P<seconds>\d+)S)?",
        value,
    )
    if not match:
        raise ValueError("Unsupported YouTube duration")
    values = {name: int(raw or 0) for name, raw in match.groupdict().items()}
    return (
        values["days"] * 86400
        + values["hours"] * 3600
        + values["minutes"] * 60
        + values["seconds"]
    )


def classify_short(*, duration_seconds: int, width: int | None, height: int | None) -> bool:
    if width and height:
        return height >= width and duration_seconds <= 180
    return duration_seconds <= 60


def cumulative_horizons(
    *,
    published_at: datetime,
    daily_views: dict[date, int],
    today_pacific: date | None = None,
    analytics_lag_days: int = ANALYTICS_LAG_DAYS,
) -> dict[int, int | None]:
    if published_at.tzinfo is None:
        published_at = published_at.replace(tzinfo=timezone.utc)
    published_pacific = published_at.astimezone(PACIFIC)
    publish_date = published_pacific.date()
    extra_day = 1 if published_pacific.hour >= 12 else 0
    current_date = today_pacific or datetime.now(PACIFIC).date()
    last_ready = current_date - timedelta(days=analytics_lag_days)

    results: dict[int, int | None] = {}
    for horizon in (7, 14, 21, 30):
        day_count = horizon + extra_day
        final_date = publish_date + timedelta(days=day_count - 1)
        if final_date > last_ready:
            results[horizon] = None
            continue
        # YouTube omits zero-view rows, so absent calendar dates are explicit zeros.
        results[horizon] = sum(
            max(0, int(daily_views.get(publish_date + timedelta(days=offset), 0)))
            for offset in range(day_count)
        )
    return results


async def _google_get(
    url: str, *, access_token: str, params: dict[str, str | int]
) -> dict:
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(
                url,
                params=params,
                headers={
                    "Authorization": f"Bearer {access_token}",
                    "Accept": "application/json",
                },
            )
    except httpx.HTTPError as exc:
        raise CreatorSyncException("YouTube data is temporarily unavailable") from exc
    if response.status_code != 200:
        raise CreatorSyncException("YouTube data is temporarily unavailable")
    try:
        payload = response.json()
    except ValueError as exc:
        raise CreatorSyncException("YouTube returned an invalid response") from exc
    if not isinstance(payload, dict):
        raise CreatorSyncException("YouTube returned an invalid response")
    return payload


async def fetch_upload_video_ids(
    *, access_token: str, uploads_playlist_id: str, limit: int
) -> list[str]:
    identifiers: list[str] = []
    page_token: str | None = None
    while len(identifiers) < limit:
        params: dict[str, str | int] = {
            "part": "contentDetails",
            "playlistId": uploads_playlist_id,
            "maxResults": min(50, limit - len(identifiers)),
        }
        if page_token:
            params["pageToken"] = page_token
        payload = await _google_get(
            YOUTUBE_PLAYLIST_ITEMS_URL, access_token=access_token, params=params
        )
        for item in payload.get("items", []):
            video_id = item.get("contentDetails", {}).get("videoId")
            if isinstance(video_id, str) and video_id:
                identifiers.append(video_id)
        page_token = payload.get("nextPageToken")
        if not page_token:
            break
    return identifiers[:limit]


async def fetch_video_metadata(
    *, access_token: str, video_ids: list[str]
) -> list[CreatorVideo]:
    videos: list[CreatorVideo] = []
    for batch in batched_video_ids(video_ids, batch_size=10):
        payload = await _google_get(
            YOUTUBE_VIDEOS_URL,
            access_token=access_token,
            params={
                "part": "snippet,contentDetails,player",
                "id": ",".join(batch),
                "maxHeight": 8192,
            },
        )
        for item in payload.get("items", []):
            try:
                snippet = item["snippet"]
                details = item["contentDetails"]
                player = item.get("player", {})
                duration = parse_iso8601_duration(details["duration"])
                published_at = datetime.fromisoformat(
                    snippet["publishedAt"].replace("Z", "+00:00")
                )
                width = player.get("embedWidth")
                height = player.get("embedHeight")
                videos.append(
                    CreatorVideo(
                        video_id=item["id"],
                        title=snippet.get("title", ""),
                        category=CATEGORY_NAMES.get(
                            str(snippet.get("categoryId", "")), "Unknown"
                        ),
                        duration_seconds=duration,
                        audio_language=snippet.get("defaultAudioLanguage")
                        or snippet.get("defaultLanguage"),
                        published_at=published_at,
                        is_short=classify_short(
                            duration_seconds=duration,
                            width=int(width) if width is not None else None,
                            height=int(height) if height is not None else None,
                        ),
                    )
                )
            except (KeyError, TypeError, ValueError):
                continue
    return videos


async def fetch_daily_analytics(
    *, access_token: str, videos: list[CreatorVideo], today_pacific: date | None = None
) -> dict[str, dict[date, int]]:
    results = {video.video_id: {} for video in videos}
    if not videos:
        return results
    current_date = today_pacific or datetime.now(PACIFIC).date()
    end_date = current_date - timedelta(days=ANALYTICS_LAG_DAYS)

    by_id = {video.video_id: video for video in videos}
    for batch in batched_video_ids(
        [video.video_id for video in videos], config.YOUTUBE_ANALYTICS_BATCH_SIZE
    ):
        start_date = min(by_id[video_id].published_at.astimezone(PACIFIC).date() for video_id in batch)
        if start_date > end_date:
            continue
        payload = await _google_get(
            YOUTUBE_ANALYTICS_URL,
            access_token=access_token,
            params={
                "ids": "channel==MINE",
                "startDate": start_date.isoformat(),
                "endDate": end_date.isoformat(),
                "metrics": "views",
                "dimensions": "day,video",
                "filters": f"video=={','.join(batch)}",
                "sort": "day,video",
            },
        )
        headers = [item.get("name") for item in payload.get("columnHeaders", [])]
        try:
            day_index = headers.index("day")
            video_index = headers.index("video")
            views_index = headers.index("views")
        except ValueError as exc:
            raise CreatorSyncException("YouTube Analytics columns were incomplete") from exc
        for row in payload.get("rows", []) or []:
            if not isinstance(row, list) or len(row) <= max(day_index, video_index, views_index):
                continue
            video_id = str(row[video_index])
            if video_id not in results:
                continue
            try:
                results[video_id][date.fromisoformat(str(row[day_index]))] = max(
                    0, int(row[views_index])
                )
            except (TypeError, ValueError):
                continue
    return results
