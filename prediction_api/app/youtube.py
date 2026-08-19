from datetime import datetime, timezone
import re
from typing import Optional, Tuple
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from app.schemas import ChannelStatsResponse


class ChannelLookupException(Exception):
    def __init__(self, message: str, code: str, status_code: int = 400):
        self.message = message
        self.code = code
        self.status_code = status_code
        super().__init__(message)


def parse_channel_identifier(raw_input: str) -> Tuple[str, str]:
    if not raw_input or not isinstance(raw_input, str):
        raise ChannelLookupException(
            message="Enter a valid YouTube channel URL, handle, or channel ID.",
            code="invalid_channel_identifier",
            status_code=400,
        )

    cleaned = raw_input.strip()
    if not cleaned:
        raise ChannelLookupException(
            message="Enter a valid YouTube channel URL, handle, or channel ID.",
            code="invalid_channel_identifier",
            status_code=400,
        )

    # Strip protocol
    cleaned = re.sub(r"^https?://", "", cleaned, flags=re.IGNORECASE)
    # Strip domain prefix (www.youtube.com, youtube.com, m.youtube.com, etc.)
    cleaned = re.sub(r"^(www\.|m\.)?youtube\.com/", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^(www\.|m\.)?youtu\.be/", "", cleaned, flags=re.IGNORECASE)

    # Strip query parameters and trailing slashes
    cleaned = cleaned.split("?")[0].split("#")[0].strip("/")

    if not cleaned:
        raise ChannelLookupException(
            message="Enter a valid YouTube channel URL, handle, or channel ID.",
            code="invalid_channel_identifier",
            status_code=400,
        )

    # Check for channel/UC...
    if cleaned.lower().startswith("channel/"):
        channel_id = cleaned[8:].strip("/")
        if channel_id.startswith("UC") and len(channel_id) == 24:
            return ("id", channel_id)
        if channel_id.startswith("UC"):
            return ("id", channel_id)
        raise ChannelLookupException(
            message="Enter a valid YouTube channel URL, handle, or channel ID.",
            code="invalid_channel_identifier",
            status_code=400,
        )

    # Check for c/handle or user/handle
    if cleaned.lower().startswith("c/"):
        cleaned = cleaned[2:].strip("/")
    elif cleaned.lower().startswith("user/"):
        cleaned = cleaned[5:].strip("/")

    # If starts with @
    if cleaned.startswith("@"):
        handle_body = cleaned[1:]
        if not handle_body or re.search(r"\s", handle_body):
            raise ChannelLookupException(
                message="The channel URL or identifier should not contain spaces.",
                code="invalid_channel_identifier",
                status_code=400,
            )
        return ("handle", cleaned)

    # Check for bare UC ID (24 characters starting with UC)
    if cleaned.startswith("UC") and len(cleaned) == 24:
        return ("id", cleaned)

    # Any whitespace inside is invalid
    if re.search(r"\s", cleaned):
        raise ChannelLookupException(
            message="The channel URL or identifier should not contain spaces.",
            code="invalid_channel_identifier",
            status_code=400,
        )

    # Fallback: treat as handle name (prepend @ if not present)
    return ("handle", f"@{cleaned}" if not cleaned.startswith("@") else cleaned)


def calculate_channel_age_days(published_at_iso: str) -> Optional[int]:
    if not published_at_iso:
        return None
    try:
        cleaned_iso = published_at_iso.replace("Z", "+00:00")
        created_dt = datetime.fromisoformat(cleaned_iso)
        if created_dt.tzinfo is None:
            created_dt = created_dt.replace(tzinfo=timezone.utc)

        now_dt = datetime.now(timezone.utc)
        delta = now_dt - created_dt
        return max(0, delta.days)
    except Exception:
        return None


def fetch_channel_stats(
    channel_identifier: str, api_key: str, youtube_client=None
) -> ChannelStatsResponse:
    lookup_type, normalized_value = parse_channel_identifier(channel_identifier)

    if not api_key and youtube_client is None:
        raise ChannelLookupException(
            message="YouTube API key is not configured.",
            code="api_not_configured",
            status_code=500,
        )

    try:
        client = youtube_client or build("youtube", "v3", developerKey=api_key)

        if lookup_type == "id":
            response = (
                client.channels()
                .list(
                    part="snippet,statistics",
                    id=normalized_value,
                )
                .execute(num_retries=3)
            )
        else:
            response = (
                client.channels()
                .list(
                    part="snippet,statistics",
                    forHandle=normalized_value,
                )
                .execute(num_retries=3)
            )

        items = response.get("items", [])
        if not items:
            raise ChannelLookupException(
                message="The channel could not be found.",
                code="channel_not_found",
                status_code=404,
            )

        channel_item = items[0]
        snippet = channel_item.get("snippet", {})
        statistics = channel_item.get("statistics", {})

        # Hidden subscriber count handling
        hidden_subscribers = statistics.get("hiddenSubscriberCount", False)
        raw_sub_count = statistics.get("subscriberCount")
        subscriber_count: Optional[int] = None
        if not hidden_subscribers and raw_sub_count is not None:
            try:
                subscriber_count = int(raw_sub_count)
            except (ValueError, TypeError):
                subscriber_count = None

        # Views count
        raw_view_count = statistics.get("viewCount")
        total_view_count: Optional[int] = None
        if raw_view_count is not None:
            try:
                total_view_count = int(raw_view_count)
            except (ValueError, TypeError):
                total_view_count = None

        # Video count
        raw_video_count = statistics.get("videoCount")
        video_count: Optional[int] = None
        if raw_video_count is not None:
            try:
                video_count = int(raw_video_count)
            except (ValueError, TypeError):
                video_count = None

        # Created at & age
        created_at = snippet.get("publishedAt")
        channel_age_days = (
            calculate_channel_age_days(created_at) if created_at else None
        )

        return ChannelStatsResponse(
            subscriberCount=subscriber_count,
            totalViewCount=total_view_count,
            videoCount=video_count,
            createdAt=created_at,
            channelAgeDays=channel_age_days,
        )
    except ChannelLookupException:
        raise
    except HttpError as err:
        if err.resp.status == 404:
            raise ChannelLookupException(
                message="The channel could not be found.",
                code="channel_not_found",
                status_code=404,
            )
        raise ChannelLookupException(
            message="Channel statistics are currently unavailable.",
            code="channel_stats_unavailable",
            status_code=502,
        )
    except Exception:
        raise ChannelLookupException(
            message="Channel statistics are currently unavailable.",
            code="channel_stats_unavailable",
            status_code=500,
        )
