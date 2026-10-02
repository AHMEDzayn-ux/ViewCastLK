"""The connected creator's own pattern, computed from their synced history."""

import math
from datetime import datetime, timedelta, timezone

from fastapi.testclient import TestClient

import app.main as main_module
from app.creator_insights import MIN_MEASURED, compute_creator_insights
from app.main import app

NOW = datetime(2026, 10, 3, tzinfo=timezone.utc)
# 13:30 UTC is 19:00 in Sri Lanka; 03:30 UTC is 09:00.
EVENING, MORNING = timedelta(hours=13, minutes=30), timedelta(hours=3, minutes=30)


def _channel(pairs: int = 12):
    """Each day: a crowded pair (two uploads 30 minutes apart, weak) in the
    morning, then a lone evening upload (strong) a day before the next pair."""
    rows = []
    start = datetime(2026, 7, 1, tzinfo=timezone.utc)
    for day in range(pairs):
        base = start + timedelta(days=2 * day)
        rows.append(dict(published_at=base + MORNING, d7=1000, d30=1100,
                         is_short=False, category="Education"))
        rows.append(dict(published_at=base + MORNING + timedelta(minutes=30), d7=900,
                         d30=1000, is_short=True, category="Education"))
        rows.append(dict(published_at=base + EVENING, d7=4000, d30=4200,
                         is_short=False, category="Education"))
    return rows


def test_crowded_uploads_and_evening_are_measured_against_the_channel_normal():
    result = compute_creator_insights(_channel(), now=NOW)

    assert result["videosSynced"] == 36
    assert result["videosMeasured"] == 36
    assert result["mainCategory"] == "Education"
    spacing = result["spacing"]
    # 12 of 36 uploads were followed by another within the hour.
    assert spacing["shareWithinHour"] == round(100 * 12 / 36, 1)
    buckets = {b["key"]: b for b in spacing["buckets"]}
    assert buckets["under_1h"]["effectPct"] < 0 < buckets["6h_plus"]["effectPct"]
    assert buckets["under_1h"]["highPct"] < 0  # clearly below normal

    timing = {t["key"]: t for t in result["timing"]}
    assert timing["evening"]["effectPct"] > 0 > timing["morning"]["effectPct"]
    assert "night" not in timing  # no uploads there, so no claim

    assert result["format"]["shorts"] == 12
    assert result["format"]["shortsVsRegular"]["effectPct"] < 0
    assert result["growth"]["videos"] == 36
    assert 0 < result["growth"]["medianGrowthPct"] < 15


def test_every_effect_sits_inside_its_own_range():
    result = compute_creator_insights(_channel(), now=NOW)
    effects = result["spacing"]["buckets"] + result["timing"]
    effects.append(result["format"]["shortsVsRegular"])
    for effect in effects:
        assert effect["lowPct"] <= effect["effectPct"] <= effect["highPct"]


def test_too_few_measured_videos_returns_counts_but_no_claims():
    rows = _channel()[: MIN_MEASURED - 1]
    result = compute_creator_insights(rows, now=NOW)

    assert result["videosMeasured"] == MIN_MEASURED - 1
    assert result["spacing"]["shareWithinHour"] is not None
    assert result["spacing"]["buckets"] == []
    assert result["timing"] == []
    assert result["format"] is None and result["growth"] is None


def test_videos_without_day_7_views_count_for_spacing_but_not_effects():
    rows = _channel()
    for row in rows[-3:]:
        row["d7"] = None
    result = compute_creator_insights(rows, now=NOW)
    assert result["videosSynced"] == 36
    assert result["videosMeasured"] == 33


def test_the_newest_upload_counts_as_spaced_only_after_two_days():
    rows = [dict(published_at=NOW - timedelta(hours=10), d7=None, is_short=False)]
    assert compute_creator_insights(rows, now=NOW)["spacing"]["uploadsCounted"] == 0
    rows = [dict(published_at=NOW - timedelta(days=3), d7=None, is_short=False)]
    assert compute_creator_insights(rows, now=NOW)["spacing"]["uploadsCounted"] == 1


def test_empty_history():
    result = compute_creator_insights([], now=NOW)
    assert result["videosSynced"] == 0 and result["spacing"] is None


class FakeStore:
    def __init__(self, connection, rows):
        self.connection, self.rows = connection, rows

    async def get_youtube_connection_status(self, *, user_id):
        return self.connection

    async def get_video_history(self, *, user_id):
        return self.rows


def test_endpoint_requires_a_connected_channel(authenticated_api, monkeypatch):
    monkeypatch.setattr(main_module, "creator_store", FakeStore(None, []))
    response = TestClient(app).get("/creator/insights")
    assert response.status_code == 404
    assert response.json()["code"] == "youtube_not_connected"


def test_endpoint_returns_the_connected_channels_pattern(authenticated_api, monkeypatch):
    store = FakeStore({"channel_title": "Learn With Us"}, _channel())
    monkeypatch.setattr(main_module, "creator_store", store)
    body = TestClient(app).get("/creator/insights").json()

    assert body["channelTitle"] == "Learn With Us"
    assert body["videosMeasured"] == 36
    assert {b["key"] for b in body["spacing"]["buckets"]} >= {"under_1h", "6h_plus"}
    assert all(math.isfinite(t["effectPct"]) for t in body["timing"])


def test_endpoint_rejects_signed_out_requests():
    response = TestClient(app).get("/creator/insights")
    assert response.status_code == 401
