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
from app.personalization import (
    apply_forecast_adjustments,
    compute_adjustments,
    load_current_training_video_ids,
)
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


def test_unconnected_or_stale_adjustment_keeps_shared_forecast():
    shared = {7: 100, 14: 200, 21: 300, 30: 400}
    displayed, metadata = apply_forecast_adjustments(
        shared_predictions=shared,
        adjustment_rows=[
            {
                "horizon": 7,
                "format": "all",
                "factor": 2.0,
                "n_videos": 10,
                "model_version": "old-model",
            }
        ],
        requested_format="long",
        model_version="current-model",
    )

    assert displayed == shared
    assert metadata["applied"] is False
    assert metadata["adjustments"] == []


@pytest.mark.parametrize("factor", [float("nan"), float("inf"), "invalid", 0])
def test_invalid_adjustment_keeps_shared_forecast(factor):
    shared = {7: 100, 14: 200, 21: 300, 30: 400}
    displayed, metadata = apply_forecast_adjustments(
        shared_predictions=shared,
        adjustment_rows=[{
            "horizon": 7,
            "format": "all",
            "factor": factor,
            "n_videos": 10,
            "model_version": "model-v1",
        }],
        requested_format="long",
        model_version="model-v1",
    )

    assert displayed == shared
    assert metadata["applied"] is False


def test_format_adjustment_wins_then_falls_back_to_all():
    shared = {7: 100, 14: 200, 21: 300, 30: 400}
    rows = [
        {
            "horizon": horizon,
            "format": "all",
            "factor": 1.5,
            "n_videos": 8,
            "model_version": "model-v1",
        }
        for horizon in (7, 14, 21, 30)
    ]
    rows.append(
        {
            "horizon": 7,
            "format": "short",
            "factor": 2.0,
            "n_videos": 6,
            "model_version": "model-v1",
        }
    )

    short, short_metadata = apply_forecast_adjustments(
        shared_predictions=shared,
        adjustment_rows=rows,
        requested_format="short",
        model_version="model-v1",
    )
    long, _ = apply_forecast_adjustments(
        shared_predictions=shared,
        adjustment_rows=rows,
        requested_format="long",
        model_version="model-v1",
    )

    assert short[7] == 200
    assert short[14] == 300
    assert long[7] == 150
    assert short_metadata["applied"] is True
    assert short_metadata["sharedEstimates"][0]["cumulativeViews"] == 100


def test_long_adjustment_wins_for_long_form_forecast():
    shared = {7: 100, 14: 200, 21: 300, 30: 400}
    rows = []
    for horizon in (7, 14, 21, 30):
        rows.extend(
            [
                {
                    "horizon": horizon,
                    "format": "all",
                    "factor": 1.25,
                    "n_videos": 9,
                    "model_version": "model-v1",
                },
                {
                    "horizon": horizon,
                    "format": "long",
                    "factor": 1.5,
                    "n_videos": 7,
                    "model_version": "model-v1",
                },
            ]
        )

    displayed, metadata = apply_forecast_adjustments(
        shared_predictions=shared,
        adjustment_rows=rows,
        requested_format="long",
        model_version="model-v1",
    )

    assert displayed == {7: 150, 14: 300, 21: 450, 30: 600}
    assert all(row["format"] == "long" for row in metadata["adjustments"])


def test_connected_creator_without_adjustments_receives_shared_forecast():
    shared = {7: 100, 14: 200, 21: 300, 30: 400}
    displayed, metadata = apply_forecast_adjustments(
        shared_predictions=shared,
        adjustment_rows=[],
        requested_format="short",
        model_version="model-v1",
    )

    assert displayed == shared
    assert metadata["applied"] is False


def test_current_training_id_artifact_is_unique_and_nonempty():
    import json

    from app.artifact import ARTIFACT_DIR

    identifiers = load_current_training_video_ids()
    record = json.loads((ARTIFACT_DIR / "training_video_ids.json").read_text(encoding="utf-8"))

    assert len(identifiers) == record["count"]
    # A superset is only safe if it really covers what the model trained on.
    assert len(identifiers) >= record["largest_component_training_rows"]
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


def test_past_videos_are_re_predicted_with_the_inputs_a_live_forecast_gets():
    """A creator's correction is learned from re-predictions of their own past
    videos and applied to live forecasts, so both must see the same inputs.
    The served model reads the title and the channel's own record; leaving
    either out of the re-prediction would learn the correction against a
    different kind of forecast from the one it is applied to."""
    from datetime import datetime, timedelta, timezone

    from app.creator_analytics import CreatorVideo
    from app.feature_builder import EXPECTED_COLUMNS
    from app.personalization import _historical_feature_frame

    if not {"title_length", "prior_d7_view_count"} <= set(EXPECTED_COLUMNS):
        pytest.skip("the served model reads neither the title nor the channel's record")
    from app.youtube_oauth import YouTubeChannelIdentity

    start = datetime(2026, 8, 1, 12, tzinfo=timezone.utc)
    channel = YouTubeChannelIdentity(
        channel_id="UCtest", title="Test", published_at=start - timedelta(days=900)
    )
    collected = [
        {"published_at": start, "category_name": "Music", "is_short": False,
         "d7_views": 1500, "d7_hours_off": 0.0, "d30_views": None, "d30_hours_off": None},
    ]

    def frame_for(days_after_first):
        video = CreatorVideo(
            video_id=f"v{days_after_first}",
            title="Aluth Sindu 2026 | Best Hits!",
            category="Music",
            duration_seconds=240,
            audio_language="Sinhala",
            published_at=start + timedelta(days=days_after_first),
            is_short=False,
        )
        return _historical_feature_frame(
            video=video, position=2, channel=channel, history_rows=collected
        )

    later = frame_for(10)
    assert later.loc[0, "title_length"] == len("Aluth Sindu 2026 | Best Hits!")
    # Ten days on, the first video's day-7 figure had been observed.
    assert later.loc[0, "prior_d7_view_count"] == 1
    assert later.loc[0, "prior_d7_median_views"] == 1500

    sooner = frame_for(3)
    # Three days on it had been published but its day-7 figure did not exist yet.
    assert sooner.loc[0, "prior_channel_video_count"] == 1
    assert sooner.loc[0, "prior_d7_view_count"] == 0
