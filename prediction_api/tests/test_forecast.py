"""Automated tests for POST /forecast endpoint."""

from unittest.mock import AsyncMock, patch
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.artifact import (
    ACTIVE_ARTIFACT_VERSION,
    V8_ARTIFACT_VERSION,
    V9_BREAKOUT_MODEL_VERSION,
)
from app.main import app
import app.main as main_module
from app.schemas import ChannelStatsResponse
from app.youtube import ChannelLookupException

client = TestClient(app)
pytestmark = pytest.mark.usefixtures("authenticated_api")

MOCK_CHANNEL_STATS = ChannelStatsResponse(
    subscriberCount=125000,
    totalViewCount=6000000,
    videoCount=500,
    createdAt="2020-01-01T00:00:00Z",
    channelAgeDays=1200,
)

VALID_FORECAST_PAYLOAD = {
    "title": "Sri Lanka Travel Guide 2026",
    "category": "Travel & Events",
    "durationSeconds": 510.0,
    "audioLanguage": "English",
    "channelIdentifier": "@samplechannel",
    "plannedPublishDay": "Friday",
    "plannedPublishHour": 18,
}


@patch("app.main.fetch_channel_stats")
def test_1_valid_forecast_returns_200(mock_fetch):
    mock_fetch.return_value = MOCK_CHANNEL_STATS
    response = client.post("/forecast", json=VALID_FORECAST_PAYLOAD)
    assert response.status_code == 200
    assert mock_fetch.call_count == 1
    assert mock_fetch.call_args[0][0] == "@samplechannel"


@patch("app.main.fetch_channel_stats")
def test_2_and_3_and_4_response_contains_four_exact_horizons(mock_fetch):
    mock_fetch.return_value = MOCK_CHANNEL_STATS
    response = client.post("/forecast", json=VALID_FORECAST_PAYLOAD)
    assert response.status_code == 200
    data = response.json()
    estimates = data.get("estimates", [])
    assert len(estimates) == 4
    horizons = [e["horizonDays"] for e in estimates]
    assert horizons == [7, 14, 21, 30]


@patch("app.main.fetch_channel_stats")
def test_5_all_cumulative_views_numeric_non_negative(mock_fetch):
    mock_fetch.return_value = MOCK_CHANNEL_STATS
    response = client.post("/forecast", json=VALID_FORECAST_PAYLOAD)
    assert response.status_code == 200
    data = response.json()
    for est in data["estimates"]:
        val = est["cumulativeViews"]
        assert isinstance(val, int)
        assert val >= 0


@patch("app.main.fetch_channel_stats")
def test_6_model_metadata_identifies_monotonic_trajectory(mock_fetch):
    mock_fetch.return_value = MOCK_CHANNEL_STATS
    response = client.post("/forecast", json=VALID_FORECAST_PAYLOAD)
    assert response.status_code == 200
    data = response.json()
    model = data.get("model", {})
    assert model.get("artifactVersion") == ACTIVE_ARTIFACT_VERSION
    assert model.get("modelVersion") == V9_BREAKOUT_MODEL_VERSION
    assert model.get("dataSource") == "prediction_api"
    assert model.get("status") == "experimental"


@patch("app.main.fetch_channel_stats")
def test_7_channel_stats_returned_correctly(mock_fetch):
    mock_fetch.return_value = MOCK_CHANNEL_STATS
    response = client.post("/forecast", json=VALID_FORECAST_PAYLOAD)
    assert response.status_code == 200
    data = response.json()
    ch = data.get("channelStats", {})
    assert ch["subscriberCount"] == 125000
    assert ch["totalViewCount"] == 6000000
    assert ch["videoCount"] == 500
    assert ch["channelAgeDays"] == 1200


@patch("app.main.fetch_channel_stats")
def test_8_missing_optional_day_hour_works(mock_fetch):
    mock_fetch.return_value = MOCK_CHANNEL_STATS
    payload = dict(VALID_FORECAST_PAYLOAD)
    payload["plannedPublishDay"] = None
    payload["plannedPublishHour"] = None

    response = client.post("/forecast", json=payload)
    assert response.status_code == 200
    assert len(response.json()["estimates"]) == 4


@patch("app.main.fetch_channel_stats")
def test_9_supplied_day_hour_works(mock_fetch):
    mock_fetch.return_value = MOCK_CHANNEL_STATS
    payload = dict(VALID_FORECAST_PAYLOAD)
    payload["plannedPublishDay"] = "Saturday"
    payload["plannedPublishHour"] = 9

    response = client.post("/forecast", json=payload)
    assert response.status_code == 200
    assert len(response.json()["estimates"]) == 4


def test_10_invalid_duration_fails_validation():
    payload = dict(VALID_FORECAST_PAYLOAD)
    payload["durationSeconds"] = -10.0
    response = client.post("/forecast", json=payload)
    assert response.status_code == 400
    assert response.json()["code"] == "invalid_request"


def test_11_invalid_publish_hour_fails_validation():
    payload = dict(VALID_FORECAST_PAYLOAD)
    payload["plannedPublishHour"] = 25
    response = client.post("/forecast", json=payload)
    assert response.status_code == 400
    assert response.json()["code"] == "invalid_request"


@patch("app.main.fetch_channel_stats")
def test_12_channel_not_found_produces_clean_error(mock_fetch):
    mock_fetch.side_effect = ChannelLookupException(
        message="The channel could not be found.",
        code="channel_not_found",
        status_code=404,
    )
    response = client.post("/forecast", json=VALID_FORECAST_PAYLOAD)
    assert response.status_code == 404
    data = response.json()
    assert data["code"] == "channel_not_found"
    assert data["message"] == "The channel could not be found."


@patch("app.main.fetch_channel_stats")
def test_13_hidden_subscriber_count_reaches_inference_as_missing(mock_fetch):
    stats = ChannelStatsResponse(
        subscriberCount=None,
        totalViewCount=500000,
        videoCount=200,
        createdAt="2021-05-01T00:00:00Z",
        channelAgeDays=800,
    )
    mock_fetch.return_value = stats
    response = client.post("/forecast", json=VALID_FORECAST_PAYLOAD)
    assert response.status_code == 200
    data = response.json()
    assert data["completeness"]["status"] == "degraded"
    assert data["channelStats"]["subscriberCount"] is None


def test_14_and_15_gemini_and_supabase_never_required():
    """Verify endpoint runs without requiring Gemini API or Supabase connection."""
    with patch("app.main.fetch_channel_stats") as mock_fetch:
        mock_fetch.return_value = MOCK_CHANNEL_STATS
        response = client.post("/forecast", json=VALID_FORECAST_PAYLOAD)
        assert response.status_code == 200


@patch("app.main.fetch_channel_stats")
def test_16_forecast_trajectory_is_monotonic(mock_fetch):
    mock_fetch.return_value = MOCK_CHANNEL_STATS
    response = client.post("/forecast", json=VALID_FORECAST_PAYLOAD)
    assert response.status_code == 200
    data = response.json()
    cumulative_views = [estimate["cumulativeViews"] for estimate in data["estimates"]]
    assert cumulative_views == sorted(cumulative_views)
    assert data["model"]["trajectoryMonotonic"] is True


@patch("app.main.fetch_channel_stats")
def test_personalization_is_scoped_to_authenticated_user_and_current_model(mock_fetch):
    mock_fetch.return_value = MOCK_CHANNEL_STATS
    adjustment_rows = [
        {
            "horizon": horizon,
            "format": "all",
            "factor": 1.25,
            "n_videos": 12,
            "model_version": ACTIVE_ARTIFACT_VERSION,
        }
        for horizon in (7, 14, 21, 30)
    ]
    with patch.object(
        main_module.creator_store,
        "get_active_adjustments",
        new=AsyncMock(return_value=adjustment_rows),
    ) as get_adjustments:
        response = client.post("/forecast", json=VALID_FORECAST_PAYLOAD)

    assert response.status_code == 200
    body = response.json()
    assert body["personalization"]["applied"] is True
    assert body["personalization"]["format"] == "long"
    assert body["estimates"][0]["cumulativeViews"] == round(
        body["personalization"]["sharedEstimates"][0]["cumulativeViews"] * 1.25
    )
    for normal, upside in zip(body["estimates"], body["breakout"]["conditionalUpside"]):
        assert upside["cumulativeViews"] > normal["cumulativeViews"]
    get_adjustments.assert_awaited_once_with(
        user_id="test-authenticated-user",
        model_version=ACTIVE_ARTIFACT_VERSION,
    )


@patch("app.main.fetch_channel_stats", return_value=MOCK_CHANNEL_STATS)
def test_authenticated_user_without_adjustments_gets_shared_forecast(mock_fetch):
    with patch.object(
        main_module.creator_store,
        "get_active_adjustments",
        new=AsyncMock(return_value=[]),
    ) as get_adjustments:
        response = client.post("/forecast", json=VALID_FORECAST_PAYLOAD)

    assert response.status_code == 200
    assert response.json()["personalization"]["applied"] is False
    get_adjustments.assert_awaited_once_with(
        user_id="test-authenticated-user",
        model_version=ACTIVE_ARTIFACT_VERSION,
    )


@patch("app.main.fetch_channel_stats", return_value=MOCK_CHANNEL_STATS)
def test_wrong_model_adjustment_is_not_applied(mock_fetch):
    with patch.object(
        main_module.creator_store,
        "get_active_adjustments",
        new=AsyncMock(return_value=[{
            "horizon": 7, "format": "all", "factor": 2.0,
            "n_videos": 10, "model_version": "old-model",
        }]),
    ):
        response = client.post("/forecast", json=VALID_FORECAST_PAYLOAD)

    assert response.status_code == 200
    body = response.json()
    assert body["personalization"]["applied"] is False
    assert body["estimates"] == body["personalization"]["sharedEstimates"]


@patch("app.main.fetch_channel_stats")
def test_duration_format_fallback_is_used_when_is_short_is_absent(mock_fetch):
    mock_fetch.return_value = MOCK_CHANNEL_STATS
    payload = dict(VALID_FORECAST_PAYLOAD)
    payload["durationSeconds"] = 45.0

    response = client.post("/forecast", json=payload)

    assert response.status_code == 200
    assert response.json()["personalization"]["format"] == "short"


@patch("app.main.fetch_channel_stats")
def test_creator_format_selects_matching_personal_adjustment(mock_fetch):
    mock_fetch.return_value = MOCK_CHANNEL_STATS
    payload = dict(VALID_FORECAST_PAYLOAD)
    payload["durationSeconds"] = 30.0
    payload["isShort"] = False
    adjustment_rows = [
        {
            "horizon": horizon,
            "format": format_name,
            "factor": factor,
            "n_videos": 8,
            "model_version": ACTIVE_ARTIFACT_VERSION,
        }
        for horizon in (7, 14, 21, 30)
        for format_name, factor in (("short", 1.5), ("long", 1.25))
    ]
    with patch.object(
        main_module.creator_store,
        "get_active_adjustments",
        new=AsyncMock(return_value=adjustment_rows),
    ):
        response = client.post("/forecast", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["personalization"]["format"] == "long"
    assert {row["format"] for row in body["personalization"]["adjustments"]} == {"long"}
    assert body["estimates"][0]["cumulativeViews"] == round(
        body["personalization"]["sharedEstimates"][0]["cumulativeViews"] * 1.25
    )


def test_a_category_the_model_never_saw_is_reported_not_hidden():
    with patch("app.main.fetch_channel_stats") as mock_fetch:
        mock_fetch.return_value = MOCK_CHANNEL_STATS
        unseen = client.post(
            "/forecast", json={**VALID_FORECAST_PAYLOAD, "category": "Made Up Category"}
        )
        seen = client.post("/forecast", json={**VALID_FORECAST_PAYLOAD, "category": "Music"})

    assert unseen.status_code == 200
    issues = unseen.json()["completeness"]["issues"]
    assert any(issue["source"] == "category" for issue in issues)
    assert unseen.json()["completeness"]["status"] == "degraded"
    assert not any(issue["source"] == "category" for issue in seen.json()["completeness"]["issues"])


def test_nonprofits_is_not_reported_as_unknown_because_the_model_learned_it_blank():
    with patch("app.main.fetch_channel_stats") as mock_fetch:
        mock_fetch.return_value = MOCK_CHANNEL_STATS
        response = client.post(
            "/forecast", json={**VALID_FORECAST_PAYLOAD, "category": "Nonprofits & Activism"}
        )
    assert response.status_code == 200
    issues = response.json()["completeness"]["issues"]
    assert not any(issue["source"] == "category" for issue in issues)


@patch("app.main.fetch_channel_stats", return_value=MOCK_CHANNEL_STATS)
def test_v8_engine_returns_breakout_probability_and_conditional_upside(mock_fetch):
    response = client.post(
        "/forecast",
        json={**VALID_FORECAST_PAYLOAD, "modelEngine": "v8"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["model"]["modelVersion"] == V8_ARTIFACT_VERSION
    assert 0 <= body["breakout"]["probability"] <= 1
    assert [row["horizonDays"] for row in body["breakout"]["conditionalUpside"]] == [
        7,
        14,
        21,
        30,
    ]
    for normal, upside in zip(body["estimates"], body["breakout"]["conditionalUpside"]):
        assert upside["cumulativeViews"] > normal["cumulativeViews"]


@patch("app.main.fetch_channel_stats", return_value=MOCK_CHANNEL_STATS)
def test_v9_engine_uses_v9_normal_forecast_with_v8_breakout_scoring(mock_fetch):
    v8_scenario = pd.DataFrame(
        {
            "normal_day_7_views": [10.0],
            "normal_day_14_views": [20.0],
            "normal_day_21_views": [30.0],
            "normal_day_30_views": [40.0],
            "breakout_probability": [0.42],
            "viral_upside_day_7_views": [10_000.0],
            "viral_upside_day_14_views": [12_000.0],
            "viral_upside_day_21_views": [14_000.0],
            "viral_upside_day_30_views": [16_000.0],
        }
    )
    with (
        patch.object(
            main_module.model_registry,
            "predict_trajectory",
            return_value=np.array([[100.0, 200.0, 300.0, 400.0]]),
        ),
        patch.object(
            main_module.v8_model_registry,
            "predict_scenario",
            return_value=v8_scenario,
        ),
    ):
        response = client.post(
            "/forecast",
            json={**VALID_FORECAST_PAYLOAD, "modelEngine": "v9"},
        )

    assert response.status_code == 200
    body = response.json()
    assert [row["cumulativeViews"] for row in body["estimates"]] == [
        100,
        200,
        300,
        400,
    ]
    assert body["breakout"]["probability"] == pytest.approx(0.42)
    assert body["model"]["artifactVersion"] == ACTIVE_ARTIFACT_VERSION
    assert body["model"]["modelVersion"] == V9_BREAKOUT_MODEL_VERSION


def test_unknown_model_engine_fails_validation():
    response = client.post(
        "/forecast",
        json={**VALID_FORECAST_PAYLOAD, "modelEngine": "unknown"},
    )
    assert response.status_code == 400
    assert response.json()["code"] == "invalid_request"
