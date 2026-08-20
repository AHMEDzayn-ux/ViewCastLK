"""Unit tests for ViewCastLK candidate model feature builder."""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from app.feature_builder import (
    build_candidate_feature_frame,
    derive_is_short,
    derive_topic_features,
    map_language,
    map_publish_is_weekend,
    map_publish_time_bucket,
    EXPECTED_COLUMNS,
)

MANIFEST_PATH = (
    Path(__file__).resolve().parent.parent
    / "model_artifacts"
    / "viewcastlk_monotonic_trajectory_experimental_v1"
    / "manifest.json"
)


def test_1_category_mapped_correctly():
    df = build_candidate_feature_frame({"category": "Entertainment"})
    assert df["category_name"].iloc[0] == "Entertainment"


def test_2_duration_mapped_correctly():
    df = build_candidate_feature_frame({"durationSeconds": 600.0})
    assert df["duration_seconds"].iloc[0] == 600.0


def test_3_subscriber_count_mapped_correctly():
    df = build_candidate_feature_frame({}, {"subscriberCount": 50000})
    assert df["ch_subs_at_publish"].iloc[0] == 50000.0


def test_4_video_count_mapped_correctly():
    df = build_candidate_feature_frame({}, {"videoCount": 120})
    assert df["ch_videos_at_publish"].iloc[0] == 120.0


def test_5_channel_age_mapped_correctly():
    df = build_candidate_feature_frame({}, {"channelAgeDays": 450})
    assert df["channel_age_days_at_publish"].iloc[0] == 450.0


def test_6_average_views_per_video_calculation():
    df = build_candidate_feature_frame({}, {"totalViewCount": 100000, "videoCount": 50})
    assert df["ch_avg_views_per_video_at_publish"].iloc[0] == 2000.0


def test_7_zero_video_count_average_missing():
    df = build_candidate_feature_frame({}, {"totalViewCount": 100000, "videoCount": 0})
    assert pd.isna(df["ch_avg_views_per_video_at_publish"].iloc[0])


def test_8_missing_total_views_average_missing():
    df = build_candidate_feature_frame({}, {"totalViewCount": None, "videoCount": 50})
    assert pd.isna(df["ch_avg_views_per_video_at_publish"].iloc[0])


def test_9_hidden_missing_subscribers_remain_missing():
    df = build_candidate_feature_frame({}, {"subscriberCount": None})
    assert pd.isna(df["ch_subs_at_publish"].iloc[0])
    assert df["subscriber_tier"].iloc[0] == "missing"


def test_10_subscriber_tier_boundary_behavior():
    # under_1k: <= 999
    assert build_candidate_feature_frame({}, {"subscriberCount": 500})["subscriber_tier"].iloc[0] == "under_1k"
    assert build_candidate_feature_frame({}, {"subscriberCount": 999})["subscriber_tier"].iloc[0] == "under_1k"
    # 1k_to_10k: 1000 to 9999
    assert build_candidate_feature_frame({}, {"subscriberCount": 1000})["subscriber_tier"].iloc[0] == "1k_to_10k"
    assert build_candidate_feature_frame({}, {"subscriberCount": 9999})["subscriber_tier"].iloc[0] == "1k_to_10k"
    # 10k_to_100k: 10000 to 99999
    assert build_candidate_feature_frame({}, {"subscriberCount": 10000})["subscriber_tier"].iloc[0] == "10k_to_100k"
    assert build_candidate_feature_frame({}, {"subscriberCount": 99999})["subscriber_tier"].iloc[0] == "10k_to_100k"
    # 100k_to_250k: 100000 to 249999
    assert build_candidate_feature_frame({}, {"subscriberCount": 100000})["subscriber_tier"].iloc[0] == "100k_to_250k"
    assert build_candidate_feature_frame({}, {"subscriberCount": 249999})["subscriber_tier"].iloc[0] == "100k_to_250k"
    # 250k_to_500k: 250000 to 499999
    assert build_candidate_feature_frame({}, {"subscriberCount": 250000})["subscriber_tier"].iloc[0] == "250k_to_500k"
    assert build_candidate_feature_frame({}, {"subscriberCount": 499999})["subscriber_tier"].iloc[0] == "250k_to_500k"
    # 500k_to_1m: 500000 to 999999
    assert build_candidate_feature_frame({}, {"subscriberCount": 500000})["subscriber_tier"].iloc[0] == "500k_to_1m"
    assert build_candidate_feature_frame({}, {"subscriberCount": 999999})["subscriber_tier"].iloc[0] == "500k_to_1m"
    # 1m_plus: >= 1000000
    assert build_candidate_feature_frame({}, {"subscriberCount": 1000000})["subscriber_tier"].iloc[0] == "1m_plus"


def test_11_english_to_en():
    assert map_language("English") == "en"
    assert map_language("en") == "en"


def test_12_sinhala_to_si():
    assert map_language("Sinhala") == "si"
    assert map_language("si") == "si"


def test_13_tamil_to_ta():
    assert map_language("Tamil") == "ta"
    assert map_language("ta") == "ta"


def test_14_mixed_multilingual_to_missing():
    assert pd.isna(map_language("Mixed / multilingual"))
    assert pd.isna(map_language("Other"))
    assert pd.isna(map_language(None))


def test_15_saturday_sunday_to_weekend_true():
    assert map_publish_is_weekend("Saturday") is True
    assert map_publish_is_weekend("Sunday") is True
    assert map_publish_is_weekend("Sat") is True
    assert map_publish_is_weekend("Sun") is True
    assert map_publish_is_weekend(5) is True  # Sat
    assert map_publish_is_weekend(6) is True  # Sun


def test_16_weekday_to_weekend_false():
    for day in ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Mon", "Fri"):
        assert map_publish_is_weekend(day) is False
    for idx in range(5):  # 0..4
        assert map_publish_is_weekend(idx) is False


def test_17_omitted_day_to_missing():
    assert pd.isna(map_publish_is_weekend(None))
    assert pd.isna(map_publish_is_weekend(""))


def test_18_hour_0_5_early_morning():
    for h in range(6):
        assert map_publish_time_bucket(h) == "early_morning"


def test_19_hour_6_14_morning_afternoon():
    for h in range(6, 15):
        assert map_publish_time_bucket(h) == "morning_afternoon"


def test_20_hour_15_20_evening():
    for h in range(15, 21):
        assert map_publish_time_bucket(h) == "evening"


def test_21_hour_21_23_late_night():
    for h in range(21, 24):
        assert map_publish_time_bucket(h) == "late_night"


def test_22_omitted_hour_to_missing():
    assert pd.isna(map_publish_time_bucket(None))
    assert pd.isna(map_publish_time_bucket(""))
    assert pd.isna(map_publish_time_bucket(-1))
    assert pd.isna(map_publish_time_bucket(24))


def test_23_topic_missing_candidate_behavior():
    topics = derive_topic_features()
    assert topics["topic_missing"] is True
    for key, val in topics.items():
        if key != "topic_missing":
            assert val is False


def test_24_unresolved_is_short_candidate_behavior():
    val = derive_is_short()
    assert pd.isna(val)


def test_25_output_column_names_exactly_match_manifest():
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    expected = manifest["input_schema"]["expected_columns"]
    df = build_candidate_feature_frame({"category": "Music"})
    assert set(df.columns) == set(expected)


def test_26_output_column_order_exactly_matches_manifest():
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    expected = manifest["input_schema"]["expected_columns"]
    df = build_candidate_feature_frame({"category": "Music"})
    assert list(df.columns) == expected
