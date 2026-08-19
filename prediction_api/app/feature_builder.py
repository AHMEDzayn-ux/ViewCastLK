"""Feature builder for ViewCastLK ML candidate model input generation."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import pandas as pd

# Safe resolution of model artifact directory and importing viewcastlk_ml
ARTIFACT_DIR = (
    Path(__file__).resolve().parent.parent
    / "model_artifacts"
    / "viewcastlk_mvp_candidate_v1"
)

if str(ARTIFACT_DIR) not in sys.path:
    sys.path.insert(0, str(ARTIFACT_DIR))

try:
    from viewcastlk_ml.horizon_preprocessing import subscriber_tier_from_count
except ImportError:
    # Fallback definition if viewcastlk_ml is missing
    SUBSCRIBER_TIER_ORDER = (
        "under_1k",
        "1k_to_10k",
        "10k_to_100k",
        "100k_to_250k",
        "250k_to_500k",
        "500k_to_1m",
        "1m_plus",
        "missing",
    )

    def subscriber_tier_from_count(series: pd.Series) -> pd.Series:
        subscriber_count = pd.to_numeric(series, errors="coerce")
        return (
            pd.cut(
                subscriber_count,
                bins=[
                    -np.inf,
                    999,
                    9_999,
                    99_999,
                    249_999,
                    499_999,
                    999_999,
                    np.inf,
                ],
                labels=SUBSCRIBER_TIER_ORDER[:-1],
                include_lowest=True,
            )
            .astype("string")
            .fillna("missing")
        )


MANIFEST_PATH = ARTIFACT_DIR / "manifest.json"
if MANIFEST_PATH.exists():
    manifest_data = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    EXPECTED_COLUMNS = manifest_data["input_schema"]["expected_columns"]
else:
    EXPECTED_COLUMNS = [
        "category_name",
        "duration_seconds",
        "ch_subs_at_publish",
        "ch_avg_views_per_video_at_publish",
        "ch_videos_at_publish",
        "channel_age_days_at_publish",
        "is_short",
        "publish_is_weekend",
        "topic_entertainment",
        "topic_fashion",
        "topic_food",
        "topic_gaming",
        "topic_health",
        "topic_hobby",
        "topic_humour",
        "topic_knowledge",
        "topic_lifestyle",
        "topic_music",
        "topic_pet",
        "topic_politics",
        "topic_religion",
        "topic_society",
        "topic_sports",
        "topic_technology",
        "topic_tourism",
        "topic_vehicle",
        "topic_missing",
        "default_language",
        "publish_time_bucket",
        "subscriber_tier",
    ]


def map_language(raw_lang: Optional[str]) -> Any:
    """Map user/frontend language input to canonical model language code."""
    if not raw_lang or not isinstance(raw_lang, str):
        return np.nan

    cleaned = raw_lang.strip()
    if not cleaned:
        return np.nan

    lower = cleaned.lower()
    if lower in ("english", "en"):
        return "en"
    if lower in ("sinhala", "si"):
        return "si"
    if lower in ("tamil", "ta"):
        return "ta"

    # Mixed / multilingual or unmapped categories are treated as missing/unknown
    return np.nan


def map_publish_is_weekend(raw_day: Optional[Union[str, int]]) -> Any:
    """Map planned publish day to publish_is_weekend boolean flag."""
    if raw_day is None or raw_day == "":
        return np.nan

    if isinstance(raw_day, int):
        # 0=Mon, 1=Tue, ..., 4=Fri, 5=Sat, 6=Sun (or 1=Mon .. 7=Sun)
        if raw_day in (5, 6, 7):
            return True
        if raw_day in (0, 1, 2, 3, 4):
            return False
        return np.nan

    if isinstance(raw_day, str):
        cleaned = raw_day.strip().lower()
        if not cleaned:
            return np.nan
        if cleaned in ("saturday", "sunday", "sat", "sun"):
            return True
        if cleaned in (
            "monday",
            "tuesday",
            "wednesday",
            "thursday",
            "friday",
            "mon",
            "tue",
            "wed",
            "thu",
            "fri",
        ):
            return False

    return np.nan


def map_publish_time_bucket(raw_hour: Optional[Union[int, float, str]]) -> Any:
    """Map planned publish hour (0-23) to publish_time_bucket category."""
    if raw_hour is None or raw_hour == "":
        return np.nan

    try:
        hour = int(raw_hour)
    except (ValueError, TypeError):
        return np.nan

    if not (0 <= hour <= 23):
        return np.nan

    if 0 <= hour <= 5:
        return "early_morning"
    if 6 <= hour <= 14:
        return "morning_afternoon"
    if 15 <= hour <= 20:
        return "evening"
    if 21 <= hour <= 23:
        return "late_night"

    return np.nan


def derive_is_short(
    duration_seconds: Optional[float] = None,
    raw_is_short: Optional[bool] = None,
) -> Any:
    """Derive is_short field for candidate model v1.

    Serving logic for is_short is unresolved; returns np.nan (missing) unless explicitly set.
    """
    if raw_is_short is not None:
        return bool(raw_is_short)
    return np.nan


def derive_topic_features() -> dict[str, Any]:
    """Derive topic_* candidate features.

    Serving logic for topic category extraction is unresolved; set all topic_* to False and topic_missing to True.
    """
    return {
        "topic_entertainment": False,
        "topic_fashion": False,
        "topic_food": False,
        "topic_gaming": False,
        "topic_health": False,
        "topic_hobby": False,
        "topic_humour": False,
        "topic_knowledge": False,
        "topic_lifestyle": False,
        "topic_music": False,
        "topic_pet": False,
        "topic_politics": False,
        "topic_religion": False,
        "topic_society": False,
        "topic_sports": False,
        "topic_technology": False,
        "topic_tourism": False,
        "topic_vehicle": False,
        "topic_missing": True,
    }


def _extract_val(obj: Any, keys: list[str]) -> Any:
    """Extract first found key or attribute from a dict or object."""
    if obj is None:
        return None
    for key in keys:
        if isinstance(obj, dict) and key in obj:
            return obj[key]
        if hasattr(obj, key):
            val = getattr(obj, key)
            if val is not None:
                return val
    return None


def build_candidate_feature_frame(
    request: Any,
    channel_stats: Any = None,
) -> pd.DataFrame:
    """Convert creator form request and channel stats into one model-ready raw DataFrame.

    Matches the authoritative expected_columns contract from manifest.json.
    """
    # 1. Direct request values
    category_name = _extract_val(request, ["category", "category_name"])
    raw_duration = _extract_val(request, ["durationSeconds", "duration_seconds"])
    duration_seconds = float(raw_duration) if raw_duration is not None else np.nan

    raw_language = _extract_val(request, ["audioLanguage", "default_language", "language"])
    default_language = map_language(raw_language)

    raw_day = _extract_val(request, ["plannedPublishDay", "publish_day"])
    publish_is_weekend = map_publish_is_weekend(raw_day)

    raw_hour = _extract_val(request, ["plannedPublishHour", "publish_hour"])
    publish_time_bucket = map_publish_time_bucket(raw_hour)

    # 2. Channel stats values
    raw_subs = _extract_val(channel_stats, ["subscriberCount", "ch_subs_at_publish"])
    ch_subs_at_publish = float(raw_subs) if raw_subs is not None else np.nan

    raw_views = _extract_val(channel_stats, ["totalViewCount", "ch_views_at_publish"])
    total_view_count = float(raw_views) if raw_views is not None else None

    raw_videos = _extract_val(channel_stats, ["videoCount", "ch_videos_at_publish"])
    ch_videos_at_publish = float(raw_videos) if raw_videos is not None else np.nan

    raw_age = _extract_val(channel_stats, ["channelAgeDays", "channel_age_days_at_publish"])
    channel_age_days_at_publish = float(raw_age) if raw_age is not None else np.nan

    # 3. Derived channel average views per video
    if (
        total_view_count is not None
        and not pd.isna(ch_videos_at_publish)
        and ch_videos_at_publish > 0
    ):
        ch_avg_views_per_video_at_publish = float(total_view_count) / float(ch_videos_at_publish)
    else:
        ch_avg_views_per_video_at_publish = np.nan

    # 4. Subscriber tier derivation using Ruzain's preprocessing function
    tier_series = subscriber_tier_from_count(pd.Series([ch_subs_at_publish]))
    subscriber_tier = str(tier_series.iloc[0])

    # 5. Is short derivation (unresolved candidate v1 -> missing)
    raw_is_short = _extract_val(request, ["is_short"])
    is_short = derive_is_short(duration_seconds=duration_seconds, raw_is_short=raw_is_short)

    # 6. Topic features (unresolved candidate v1 -> missing)
    topic_dict = derive_topic_features()

    # Construct complete dictionary of fields
    row_dict = {
        "category_name": category_name,
        "duration_seconds": duration_seconds,
        "ch_subs_at_publish": ch_subs_at_publish,
        "ch_avg_views_per_video_at_publish": ch_avg_views_per_video_at_publish,
        "ch_videos_at_publish": ch_videos_at_publish,
        "channel_age_days_at_publish": channel_age_days_at_publish,
        "is_short": is_short,
        "publish_is_weekend": publish_is_weekend,
        "default_language": default_language,
        "publish_time_bucket": publish_time_bucket,
        "subscriber_tier": subscriber_tier,
    }
    row_dict.update(topic_dict)

    # Create 1-row DataFrame and reindex explicitly to match authoritative manifest.json order
    df = pd.DataFrame([row_dict])
    df = df.reindex(columns=EXPECTED_COLUMNS)
    return df


def main() -> None:
    """Developer-only demo printing one generated candidate feature row."""
    sample_request = {
        "category": "Music",
        "durationSeconds": 480,
        "audioLanguage": "English",
        "plannedPublishDay": None,
        "plannedPublishHour": None,
    }

    sample_channel_stats = {
        "subscriberCount": 11000000,
        "totalViewCount": 5158454941,
        "videoCount": 763,
        "channelAgeDays": 2116,
    }

    df = build_candidate_feature_frame(sample_request, sample_channel_stats)

    print("Generated Feature Row (Shape:", df.shape, "):")
    print("=" * 60)
    for col in df.columns:
        val = df[col].iloc[0]
        print(f"  {col:35s}: {val!r}")
    print("=" * 60)


if __name__ == "__main__":
    main()
