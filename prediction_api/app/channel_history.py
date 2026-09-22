"""The channel's own recent record, rebuilt at prediction time.

The training table derives eighteen history features per video in
scripts/prepare_model_datasets.py::add_channel_history_features. Every one of
them is a summary of the channel's *earlier* videos, so all of it is knowable
before the creator publishes and all of it can be rebuilt here from the public
warehouse, given the channel and the moment the forecast is for.

Two rules carry over from training and both matter:

* a prior video's day-7 figure is not known the instant that video is
  published. It becomes known at published_at + 7 days + hours_off, where
  hours_off is how far the nearest observation fell from the mark. A label is
  only fed in once that moment has passed, which is what stops a future view
  count leaking backwards into an earlier forecast.
* a label counts only when it is usable: the observation landed within 12
  hours of the horizon and recorded a real view count.

A channel the warehouse has never seen produces the same values as a channel's
first video: zero counts and missing medians. That is a state the model met
during training, so it degrades to the cold-start case rather than failing.
"""

from __future__ import annotations

import asyncio
import heapq
import math
from collections import deque
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import urlparse, urlunparse

import psycopg2

from app import config
from app.creator_analytics import classify_short, parse_iso8601_duration

# Same tolerance the training table applies when it marks a label usable.
LABEL_TOLERANCE_HOURS = 12.0

HISTORY_COLUMNS: tuple[str, ...] = (
    "prior_channel_video_count",
    "uploads_previous_7d",
    "uploads_previous_30d",
    "days_since_previous_upload",
    "prior_d7_view_count",
    "prior_d7_median_views",
    "prior_d7_mean_log_views",
    "prior_d7_std_log_views",
    "prior_d7_last_views",
    "prior_d7_recent5_median_views",
    "prior_d7_recent_log_trend",
    "prior_same_category_d7_count",
    "prior_same_category_d7_median_views",
    "prior_same_format_d7_count",
    "prior_same_format_d7_median_views",
    "prior_d30_view_count",
    "prior_d30_median_views",
    "prior_d30_mean_log_views",
)


class ChannelHistoryUnavailable(Exception):
    """The warehouse could not be reached. Callers fall back to no history."""


class RunningViewStats:
    """Streaming median, log mean and log spread over a channel's labels.

    Ported deliberately from scripts/prepare_model_datasets.py rather than
    reimplemented, because the model learned the numbers this class produces.
    The two heaps keep the median without holding a sorted list, and the log
    mean and variance use Welford's method, so a long-running channel costs no
    more than a short one.
    """

    def __init__(self) -> None:
        self.lower: list[float] = []
        self.upper: list[float] = []
        self.count = 0
        self.log_mean = 0.0
        self.log_m2 = 0.0
        self.last_views = math.nan
        self.recent: deque[float] = deque(maxlen=5)

    def add(self, views: float) -> None:
        value = float(views)
        if not self.lower or value <= -self.lower[0]:
            heapq.heappush(self.lower, -value)
        else:
            heapq.heappush(self.upper, value)
        if len(self.lower) > len(self.upper) + 1:
            heapq.heappush(self.upper, -heapq.heappop(self.lower))
        elif len(self.upper) > len(self.lower):
            heapq.heappush(self.lower, -heapq.heappop(self.upper))

        self.count += 1
        logged = math.log1p(value)
        delta = logged - self.log_mean
        self.log_mean += delta / self.count
        self.log_m2 += delta * (logged - self.log_mean)
        self.last_views = value
        self.recent.append(value)

    @property
    def median(self) -> float:
        if not self.count:
            return math.nan
        if len(self.lower) == len(self.upper):
            return (-self.lower[0] + self.upper[0]) / 2.0
        return -self.lower[0]

    @property
    def log_std(self) -> float:
        if self.count <= 1:
            return math.nan
        return math.sqrt(self.log_m2 / (self.count - 1))

    @property
    def recent_median(self) -> float:
        if not self.recent:
            return math.nan
        values = sorted(self.recent)
        middle = len(values) // 2
        if len(values) % 2:
            return float(values[middle])
        return float((values[middle - 1] + values[middle]) / 2.0)

    @property
    def recent_log_trend(self) -> float:
        """Per-video slope across the five latest log view counts."""
        if len(self.recent) < 2:
            return math.nan
        values = [math.log1p(v) for v in self.recent]
        n = len(values)
        mean_x = (n - 1) / 2.0
        mean_y = sum(values) / n
        numerator = sum((i - mean_x) * (y - mean_y) for i, y in enumerate(values))
        denominator = sum((i - mean_x) ** 2 for i in range(n))
        if denominator == 0:
            return math.nan
        return float(numerator / denominator)


def empty_history_features() -> dict[str, float]:
    """What a channel with no observable history looks like to the model."""
    features: dict[str, float] = {column: math.nan for column in HISTORY_COLUMNS}
    features["prior_channel_video_count"] = 0.0
    features["uploads_previous_7d"] = 0.0
    features["uploads_previous_30d"] = 0.0
    features["prior_d7_view_count"] = 0.0
    features["prior_d30_view_count"] = 0.0
    features["prior_same_category_d7_count"] = 0.0
    features["prior_same_format_d7_count"] = 0.0
    return features


def _as_utc(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    if not isinstance(value, datetime):
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _label_event(
    video: Mapping[str, Any],
    horizon: int,
    published_at: datetime,
    as_of: datetime,
) -> tuple[datetime, float] | None:
    """When this video's horizon figure became known, and what it was.

    None when the label is unusable, absent, or not yet observable at as_of.
    """
    views = video.get(f"d{horizon}_views")
    hours_off = video.get(f"d{horizon}_hours_off")
    if views is None or hours_off is None:
        return None
    try:
        views = float(views)
        hours_off = float(hours_off)
    except (TypeError, ValueError):
        return None
    if math.isnan(views) or math.isnan(hours_off) or views < 0:
        return None
    if abs(hours_off) > LABEL_TOLERANCE_HOURS:
        return None
    available_at = published_at + timedelta(days=horizon, hours=hours_off)
    if available_at > as_of:
        return None
    return available_at, views


def compute_history_features(
    videos: Iterable[Mapping[str, Any]],
    *,
    as_of: datetime,
    category_name: Any = None,
    is_short: Any = None,
) -> dict[str, float]:
    """Summarise a channel's earlier videos as the model expects them.

    videos carries one mapping per video on the channel, with published_at,
    category_name, is_short and the day-7 and day-30 view counts and hour
    offsets. as_of is the moment the forecast is made for; anything published
    at or after it is ignored, as is any label not yet observable by then.
    """
    as_of = _as_utc(as_of) or datetime.now(timezone.utc)

    prior: list[tuple[datetime, Mapping[str, Any]]] = []
    for video in videos:
        published_at = _as_utc(video.get("published_at"))
        if published_at is None or published_at >= as_of:
            continue
        prior.append((published_at, video))

    features = empty_history_features()
    if not prior:
        return features

    prior.sort(key=lambda item: item[0])

    features["prior_channel_video_count"] = float(len(prior))
    features["uploads_previous_7d"] = float(
        sum(1 for published_at, _ in prior if published_at >= as_of - timedelta(days=7))
    )
    features["uploads_previous_30d"] = float(
        sum(1 for published_at, _ in prior if published_at >= as_of - timedelta(days=30))
    )
    features["days_since_previous_upload"] = (
        as_of - prior[-1][0]
    ).total_seconds() / 86_400.0

    stats = {7: RunningViewStats(), 30: RunningViewStats()}
    category_d7: dict[str, RunningViewStats] = {}
    format_d7: dict[bool, RunningViewStats] = {}

    for horizon in (7, 30):
        events = []
        for published_at, video in prior:
            event = _label_event(video, horizon, published_at, as_of)
            if event is not None:
                events.append((event[0], event[1], video))
        # Observation order, not publication order: that is the order in which
        # the channel's own numbers actually became visible.
        events.sort(key=lambda item: item[0])
        for _, views, video in events:
            stats[horizon].add(views)
            if horizon == 7:
                key = str(video.get("category_name"))
                category_d7.setdefault(key, RunningViewStats()).add(views)
                format_d7.setdefault(bool(video.get("is_short")), RunningViewStats()).add(views)

    features["prior_d7_view_count"] = float(stats[7].count)
    features["prior_d7_median_views"] = stats[7].median
    features["prior_d7_mean_log_views"] = stats[7].log_mean if stats[7].count else math.nan
    features["prior_d7_std_log_views"] = stats[7].log_std
    features["prior_d7_last_views"] = stats[7].last_views
    features["prior_d7_recent5_median_views"] = stats[7].recent_median
    features["prior_d7_recent_log_trend"] = stats[7].recent_log_trend
    features["prior_d30_view_count"] = float(stats[30].count)
    features["prior_d30_median_views"] = stats[30].median
    features["prior_d30_mean_log_views"] = stats[30].log_mean if stats[30].count else math.nan

    category_key = str(category_name)
    if category_key in category_d7:
        features["prior_same_category_d7_count"] = float(category_d7[category_key].count)
        features["prior_same_category_d7_median_views"] = category_d7[category_key].median

    format_key = bool(is_short)
    if format_key in format_d7:
        features["prior_same_format_d7_count"] = float(format_d7[format_key].count)
        features["prior_same_format_d7_median_views"] = format_d7[format_key].median

    return features


# The warehouse holds one row per video and one row per horizon label. Both
# horizons are pulled in a single pass so a forecast costs one round trip.
CHANNEL_HISTORY_SQL = """
SELECT v.published_at,
       v.category_name,
       v.duration,
       sh.embed_width,
       sh.embed_height,
       l7.view_count  AS d7_views,
       l7.hours_off   AS d7_hours_off,
       l30.view_count AS d30_views,
       l30.hours_off  AS d30_hours_off
FROM videos v
LEFT JOIN video_shapes sh ON sh.video_id = v.video_id
LEFT JOIN video_horizon_labels l7
       ON l7.video_id = v.video_id AND l7.horizon_days = 7
LEFT JOIN video_horizon_labels l30
       ON l30.video_id = v.video_id AND l30.horizon_days = 30
WHERE v.channel_id = %s
  AND v.published_at < %s
ORDER BY v.published_at
"""


class ChannelHistoryStore:
    """Reads a channel's public collection history from the warehouse."""

    @property
    def is_configured(self) -> bool:
        return bool(config.SUPABASE_WAREHOUSE_DB_URL)

    def _connect(self):
        if not config.SUPABASE_WAREHOUSE_DB_URL:
            raise ChannelHistoryUnavailable("Warehouse access is not configured.")
        try:
            clean_url = urlunparse(
                urlparse(config.SUPABASE_WAREHOUSE_DB_URL)._replace(query="")
            )
            return psycopg2.connect(clean_url, connect_timeout=5)
        except psycopg2.Error as exc:
            raise ChannelHistoryUnavailable("Warehouse is unavailable.") from exc

    async def fetch(self, *, channel_id: str, as_of: datetime) -> list[dict[str, Any]]:
        return await asyncio.to_thread(self._fetch, channel_id=channel_id, as_of=as_of)

    def _fetch(self, *, channel_id: str, as_of: datetime) -> list[dict[str, Any]]:
        try:
            with self._connect() as connection:
                with connection.cursor() as cursor:
                    cursor.execute(CHANNEL_HISTORY_SQL, (channel_id, as_of))
                    rows = cursor.fetchall()
        except psycopg2.Error as exc:
            raise ChannelHistoryUnavailable("Warehouse query failed.") from exc
        return [_row_to_video(row) for row in rows]


def _row_to_video(row: Sequence[Any]) -> dict[str, Any]:
    published_at, category_name, duration, width, height, d7, d7_off, d30, d30_off = row
    try:
        duration_seconds = parse_iso8601_duration(duration) if duration else 0
    except (TypeError, ValueError):
        duration_seconds = 0
    return {
        "published_at": published_at,
        "category_name": category_name,
        "is_short": classify_short(
            duration_seconds=duration_seconds, width=width, height=height
        ),
        "d7_views": d7,
        "d7_hours_off": d7_off,
        "d30_views": d30,
        "d30_hours_off": d30_off,
    }


async def history_features_for_channel(
    *,
    channel_id: str | None,
    as_of: datetime | None = None,
    category_name: Any = None,
    is_short: Any = None,
    store: ChannelHistoryStore | None = None,
) -> dict[str, float]:
    """History features for one channel, or the cold-start values on failure.

    A forecast is worth more than a perfect feature row, so an unreachable
    warehouse degrades to the no-history case rather than failing the request.
    """
    features = empty_history_features()
    if not channel_id:
        return features

    store = store or ChannelHistoryStore()
    if not store.is_configured:
        return features

    moment = _as_utc(as_of) or datetime.now(timezone.utc)
    try:
        videos = await store.fetch(channel_id=channel_id, as_of=moment)
    except ChannelHistoryUnavailable:
        return features

    return compute_history_features(
        videos, as_of=moment, category_name=category_name, is_short=is_short
    )
