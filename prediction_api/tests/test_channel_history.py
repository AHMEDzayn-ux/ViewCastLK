"""Serving must rebuild the channel history features the way training did.

The model learned these eighteen columns from
scripts/prepare_model_datasets.py::add_channel_history_features. If serving
computes them even slightly differently the feature means something else at
prediction time, and the error is invisible: the request still succeeds and
still returns a number. The first test here is therefore an equivalence test
against the training code itself rather than against hand-written expectations.
"""

from __future__ import annotations

import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from app.channel_history import (
    ChannelHistoryUnavailable,
    compute_history_features,
    empty_history_features,
    history_features_for_channel,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
UTC = timezone.utc


def _training_implementation():
    """The dataset builder, imported from the repository if it is reachable."""
    scripts_dir = REPO_ROOT / "scripts"
    if not (scripts_dir / "prepare_model_datasets.py").exists():
        return None
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    try:
        import prepare_model_datasets  # type: ignore
    except Exception:  # pragma: no cover - only when the pipeline deps are absent
        return None
    return prepare_model_datasets


def _channel_rows():
    """One channel, eight videos, deliberately awkward.

    Includes a late observation that must be discarded, a video whose day-7
    mark has not been reached, both formats, two categories, and a burst of
    uploads so the 7-day and 30-day upload counts differ.
    """
    start = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
    specs = [
        # days after start, category, is_short, d7 views, d7 off, d30 views, d30 off
        (0, "Music", False, 1200, 0.5, 4000, 1.0),
        (10, "Music", True, 30000, -2.0, 52000, 3.0),
        (20, "Entertainment", False, 800, 11.5, 2600, 0.0),
        (35, "Music", False, 1500, 30.0, 5200, 2.0),   # day-7 label unusable
        (60, "Entertainment", True, 9000, 1.0, 15000, -4.0),
        (100, "Music", False, 2100, 0.0, 6100, 6.0),
        (128, "Music", True, 700, 2.0, 2400, 0.0),
        (131, "Entertainment", False, 4500, 0.0, 9000, 0.0),  # day 7 not reached
    ]
    rows = []
    for offset, category, is_short, d7, d7_off, d30, d30_off in specs:
        rows.append(
            {
                "video_id": f"vid{offset:04d}",
                "channel_id": "UCtest",
                "published_at": start + timedelta(days=offset),
                "category_name": category,
                "is_short": is_short,
                "d7_views": d7,
                "d7_hours_off": d7_off,
                "d30_views": d30,
                "d30_off_placeholder": None,
                "d30_hours_off": d30_off,
            }
        )
    return rows


AS_OF = datetime(2026, 1, 1, 12, 0, tzinfo=UTC) + timedelta(days=136)


def test_matches_the_training_implementation_column_for_column():
    training = _training_implementation()
    if training is None:
        pytest.skip("dataset builder is not importable from this checkout")

    import numpy as np
    import pandas as pd

    rows = _channel_rows()
    # The candidate video the forecast is for, appended so the training code
    # computes its history exactly as it did for every training row.
    candidate = {
        "video_id": "vidcand",
        "channel_id": "UCtest",
        "published_at": AS_OF,
        "category_name": "Music",
        "is_short": False,
        "d7_views": np.nan,
        "d7_hours_off": np.nan,
        "d30_views": np.nan,
        "d30_hours_off": np.nan,
    }
    frame = pd.DataFrame(rows + [candidate])
    frame = frame.drop(columns=["d30_off_placeholder"], errors="ignore")
    for horizon in (7, 30):
        offsets = pd.to_numeric(frame[f"d{horizon}_hours_off"], errors="coerce")
        views = pd.to_numeric(frame[f"d{horizon}_views"], errors="coerce")
        frame[f"d{horizon}_usable"] = offsets.abs().le(12.0) & views.notna()

    expected_frame = training.add_channel_history_features(frame)
    expected = expected_frame.iloc[-1]

    actual = compute_history_features(
        rows, as_of=AS_OF, category_name="Music", is_short=False
    )

    from app.channel_history import HISTORY_COLUMNS

    mismatches = []
    for column in HISTORY_COLUMNS:
        want = float(expected[column])
        got = float(actual[column])
        if math.isnan(want) and math.isnan(got):
            continue
        if not math.isclose(want, got, rel_tol=1e-9, abs_tol=1e-9):
            mismatches.append(f"{column}: training={want!r} serving={got!r}")
    assert not mismatches, "\n".join(mismatches)


def test_a_channel_with_no_history_looks_like_a_first_video():
    features = compute_history_features([], as_of=AS_OF, category_name="Music")
    assert features["prior_channel_video_count"] == 0
    assert features["prior_d7_view_count"] == 0
    assert features["uploads_previous_7d"] == 0
    assert math.isnan(features["prior_d7_median_views"])
    assert math.isnan(features["days_since_previous_upload"])
    assert features == empty_history_features()


def test_a_label_observed_too_far_from_the_mark_is_not_counted():
    published = AS_OF - timedelta(days=40)
    usable = [{"published_at": published, "category_name": "Music", "is_short": False,
               "d7_views": 500, "d7_hours_off": 2.0, "d30_views": None, "d30_hours_off": None}]
    unusable = [dict(usable[0], d7_hours_off=13.0)]
    assert compute_history_features(usable, as_of=AS_OF)["prior_d7_view_count"] == 1
    assert compute_history_features(unusable, as_of=AS_OF)["prior_d7_view_count"] == 0


def test_a_day_seven_figure_is_invisible_until_its_moment_arrives():
    published = AS_OF - timedelta(days=5)
    video = [{"published_at": published, "category_name": "Music", "is_short": False,
              "d7_views": 900, "d7_hours_off": 0.0, "d30_views": None, "d30_hours_off": None}]
    features = compute_history_features(video, as_of=AS_OF)
    assert features["prior_channel_video_count"] == 1
    assert features["prior_d7_view_count"] == 0, "a label from the future leaked in"


def test_videos_published_after_the_forecast_moment_are_ignored():
    later = [{"published_at": AS_OF + timedelta(days=1), "category_name": "Music",
              "is_short": False, "d7_views": 10, "d7_hours_off": 0.0,
              "d30_views": None, "d30_hours_off": None}]
    assert compute_history_features(later, as_of=AS_OF)["prior_channel_video_count"] == 0


def test_category_and_format_statistics_follow_the_candidate():
    features = compute_history_features(
        _channel_rows(), as_of=AS_OF, category_name="Entertainment", is_short=True
    )
    assert features["prior_same_category_d7_count"] >= 1
    assert features["prior_same_format_d7_count"] >= 1
    other = compute_history_features(
        _channel_rows(), as_of=AS_OF, category_name="Music", is_short=False
    )
    assert other["prior_same_category_d7_count"] != features["prior_same_category_d7_count"]


def test_upload_cadence_counts_only_the_relevant_windows():
    features = compute_history_features(_channel_rows(), as_of=AS_OF)
    assert features["uploads_previous_7d"] <= features["uploads_previous_30d"]
    assert features["uploads_previous_30d"] <= features["prior_channel_video_count"]
    assert features["days_since_previous_upload"] == pytest.approx(5.0)


@pytest.mark.asyncio
async def test_an_unreachable_warehouse_degrades_to_no_history():
    class Failing:
        is_configured = True

        async def fetch(self, **_kwargs):
            raise ChannelHistoryUnavailable("down")

    features = await history_features_for_channel(
        channel_id="UCtest", as_of=AS_OF, store=Failing()
    )
    assert features == empty_history_features()


@pytest.mark.asyncio
async def test_no_channel_id_means_no_history_and_no_query():
    class Exploding:
        is_configured = True

        async def fetch(self, **_kwargs):  # pragma: no cover - must not run
            raise AssertionError("the warehouse was queried without a channel")

    features = await history_features_for_channel(channel_id=None, store=Exploding())
    assert features == empty_history_features()
