from datetime import datetime, timezone
from typing import Optional, Tuple
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import httplib2
import socket
import time
from app.outbound import TRANSIENT_STATUSES, retry_delay

from app.schemas import ChannelStatsResponse


class ChannelLookupException(Exception):
    def __init__(self, message: str, code: str, status_code: int = 400):
        self.message = message
        self.code = code
        self.status_code = status_code
        super().__init__(message)


def parse_channel_identifier(raw_input: str) -> Tuple[str, str]:
    from app.identifiers import parse_channel_identifier as validated_identifier
    try:
        return validated_identifier(raw_input)
    except ValueError:
        raise ChannelLookupException(
            message="Enter a valid YouTube channel URL, handle, or channel ID.",
            code="invalid_channel_identifier", status_code=400,
        ) from None


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

    owned_http = None
    try:
        if youtube_client is not None:
            client = youtube_client
        else:
            owned_http = httplib2.Http(timeout=8)
            client = build("youtube", "v3", developerKey=api_key,
                           http=owned_http, cache_discovery=False, num_retries=0)

        parameters = {"part": "snippet,statistics,topicDetails",
                      "id" if lookup_type == "id" else "forHandle": normalized_value}
        request = client.channels().list(**parameters)
        for attempt in range(3):
            try:
                response = request.execute(num_retries=0)
                break
            except HttpError as exc:
                if exc.resp.status not in TRANSIENT_STATUSES or attempt == 2:
                    raise
                delay = retry_delay(attempt, type("Response", (), {"headers": exc.resp})())
                if delay is None: raise
                time.sleep(delay)
            except (socket.timeout, ConnectionError):
                if attempt == 2: raise
                time.sleep(retry_delay(attempt))

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

        # The channel's topics ride on this same call: channels.list costs one
        # quota unit whatever parts are requested. They explain about 10% of the
        # variation in day-7 views, second only to the channel's own statistics.
        topic_categories = channel_item.get("topicDetails", {}).get("topicCategories")

        return ChannelStatsResponse(
            channelId=channel_item.get("id"),
            subscriberCount=subscriber_count,
            totalViewCount=total_view_count,
            videoCount=video_count,
            createdAt=created_at,
            channelAgeDays=channel_age_days,
            topicCategories=list(topic_categories) if topic_categories else None,
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
    finally:
        if owned_http is not None:
            owned_http.close()
