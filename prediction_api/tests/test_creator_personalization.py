import asyncio
import math
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo
from unittest.mock import AsyncMock

import pytest
import numpy as np

from app import creator_analytics, personalization
from app.creator_analytics import (
    CreatorVideo,
    batched_video_ids,
    cumulative_horizons,
    fetch_daily_analytics,
)
from app.personalization import compute_adjustments, load_current_training_video_ids
from app.youtube_oauth import YouTubeChannelIdentity


PACIFIC = ZoneInfo("America/Los_Angeles")


def test_analytics_batches_begin_at_ten_video_ids():
    batches = list(batched_video_ids([f"v{i}" for i in range(23)]))

    assert [len(batch) for batch in batches] == [10, 10, 3]
    with pytest.raises(ValueError):
        list(batched_video_ids(["v1"], batch_size=11))


def test_missing_analytics_dates_are_filled_with_zero_for_morning_upload():
    published = datetime(2026, 1, 1, 8, tzinfo=PACIFIC)
    views = {
        date(2026, 1, 1): 5,
        date(2026, 1, 3): 7,
        date(2026, 1, 7): 11,
    }

    result = cumulative_horizons(
        published_at=published,
        daily_views=views,
        today_pacific=date(2026, 1, 10),
    )

    assert result[7] == 23
    assert result[14] is None


def test_afternoon_upload_includes_an_extra_pacific_calendar_day():
    published = datetime(2026, 1, 1, 12, tzinfo=PACIFIC)
    views = {date(2026, 1, day): 1 for day in range(1, 10)}

    result = cumulative_horizons(
        published_at=published,
        daily_views=views,
        today_pacific=date(2026, 1, 11),
    )

    assert result[7] == 8


def test_horizon_maturity_uses_calendar_dates_across_dst_and_analytics_lag():
    published_utc = datetime(2026, 3, 8, 4, tzinfo=timezone.utc)
    views = {date(2026, 3, day): 2 for day in range(7, 16)}

    immature = cumulative_horizons(
        published_at=published_utc,
        daily_views=views,
        today_pacific=date(2026, 3, 16),
    )
    mature = cumulative_horizons(
        published_at=published_utc,
        daily_views=views,
        today_pacific=date(2026, 3, 18),
    )

    assert immature[7] is None
    assert mature[7] == 16


def test_analytics_request_uses_day_video_dimensions_and_batches(monkeypatch):
    calls = []

    async def fake_get(url, *, access_token, params):
        calls.append((url, access_token, params))
        return {
            "columnHeaders": [
                {"name": "day"},
                {"name": "video"},
                {"name": "views"},
            ],
            "rows": [["2026-01-01", params["filters"].split("==")[1].split(",")[0], 4]],
        }

    monkeypatch.setattr(creator_analytics, "_google_get", fake_get)
    videos = [
        CreatorVideo(
            video_id=f"v{i}",
            title="Video",
            category="Education",
            duration_seconds=60,
            audio_language="en",
            published_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            is_short=True,
        )
        for i in range(12)
    ]

    result = asyncio.run(
        fetch_daily_analytics(
            access_token="not-a-real-token",
            videos=videos,
            today_pacific=date(2026, 2, 1),
        )
    )

    assert len(calls) == 2
    assert calls[0][2]["dimensions"] == "day,video"
    assert calls[0][2]["metrics"] == "views"
    assert len(calls[0][2]["filters"].split("==")[1].split(",")) == 10
    assert result["v0"][date(2026, 1, 1)] == 4


def _record(
    index: int,
    *,
    actual: int = 199,
    predicted: int = 99,
    is_short: bool = False,
    model_version: str = "model-v1",
):
    return {
        "video_id": f"v{index}",
        "published_at": datetime(2026, 1, min(index + 1, 28), tzinfo=timezone.utc),
        "is_short": is_short,
        "model_version": model_version,
        "d7": actual,
        "d14": actual,
        "d21": actual,
        "d30": actual,
        "pred7": predicted,
        "pred14": predicted,
        "pred21": predicted,
        "pred30": predicted,
    }


def test_adjustment_uses_only_current_model_unseen_rows_and_shrinks():
    records = [_record(0), _record(1), _record(2, model_version="old-model")]
    adjustments = compute_adjustments(
        records=records,
        training_video_ids=frozenset({"v0"}),
        model_version="model-v1",
    )
    day7 = next(
        row for row in adjustments if row["horizon"] == 7 and row["format"] == "all"
    )

    assert day7["n_videos"] == 1
    assert day7["factor"] == pytest.approx(math.exp(math.log(2) / 6))


def test_adjustment_caps_at_40_recent_rows_and_requires_five_per_format():
    records = [
        _record(i, is_short=i < 4, actual=99 + i, predicted=99)
        for i in range(45)
    ]
    adjustments = compute_adjustments(
        records=records,
        training_video_ids=frozenset(),
        model_version="model-v1",
    )

    day7_all = next(
        row for row in adjustments if row["horizon"] == 7 and row["format"] == "all"
    )
    assert day7_all["n_videos"] == 40
    assert not any(row["format"] == "short" for row in adjustments)
    assert any(row["format"] == "long" for row in adjustments)


def test_no_usable_rows_produces_neutral_all_format_factor():
    adjustments = compute_adjustments(
        records=[_record(1, model_version="old-model")],
        training_video_ids=frozenset(),
        model_version="model-v1",
    )

    assert all(row["format"] == "all" for row in adjustments)
    assert all(row["factor"] == 1.0 and row["n_videos"] == 0 for row in adjustments)


def test_current_training_id_artifact_is_unique_and_nonempty():
    identifiers = load_current_training_video_ids()

    assert len(identifiers) == 40851
    assert all(identifier.strip() == identifier for identifier in identifiers)


def test_shared_training_builder_never_references_creator_private_sources():
    source = (
        Path(__file__).resolve().parents[2] / "scripts" / "build_training_table.py"
    ).read_text(encoding="utf-8").lower()

    forbidden = (
        "creator.video_history",
        "creator.adjustments",
        "creator.insights",
        "youtube analytics",
    )
    assert all(value not in source for value in forbidden)


def test_history_sync_persists_predictions_and_current_model_adjustments(monkeypatch):
    video = CreatorVideo(
        video_id="new-unseen-video",
        title="A new upload",
        category="Education",
        duration_seconds=240,
        audio_language="en",
        published_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        is_short=False,
    )
    monkeypatch.setattr(
        personalization,
        "fetch_upload_video_ids",
        AsyncMock(return_value=[video.video_id]),
    )
    monkeypatch.setattr(
        personalization,
        "fetch_video_metadata",
        AsyncMock(return_value=[video]),
    )
    monkeypatch.setattr(
        personalization,
        "fetch_daily_analytics",
        AsyncMock(
            return_value={
                video.video_id: {
                    date(2026, 1, 1) + timedelta(days=offset): 10
                    for offset in range(31)
                }
            }
        ),
    )
    monkeypatch.setattr(
        personalization,
        "load_current_training_video_ids",
        lambda: frozenset(),
    )
    store = AsyncMock()

    class Registry:
        def get_manifest(self):
            return {"artifact_version": "model-v1"}

        def predict_trajectory(self, frame):
            assert list(frame.columns)
            return np.asarray([[70.0, 140.0, 210.0, 300.0]])

    asyncio.run(
        personalization.synchronize_creator_history(
            user_id="user-a",
            access_token="not-a-real-token",
            channel=YouTubeChannelIdentity(
                channel_id="UC-test",
                title="Creator",
                uploads_playlist_id="UU-test",
                published_at=datetime(2020, 1, 1, tzinfo=timezone.utc),
            ),
            store=store,
            model_registry=Registry(),
        )
    )

    saved_rows = store.upsert_video_history.await_args.kwargs["rows"]
    assert saved_rows[0]["video_id"] == "new-unseen-video"
    assert saved_rows[0]["model_version"] == "model-v1"
    stored_adjustments = store.replace_adjustments.await_args.kwargs
    assert stored_adjustments["user_id"] == "user-a"
    assert stored_adjustments["model_version"] == "model-v1"
    store.set_youtube_connection_status.assert_awaited_once_with(
        user_id="user-a", status="active", refresh_ok=True
    )
