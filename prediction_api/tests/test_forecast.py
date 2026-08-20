"""Automated tests for POST /forecast endpoint."""

from unittest.mock import patch
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.schemas import ChannelStatsResponse
from app.youtube import ChannelLookupException

client = TestClient(app)

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
    assert model.get("artifactVersion") == "viewcastlk_monotonic_trajectory_experimental_v1"
    assert model.get("modelVersion") == "viewcastlk_monotonic_trajectory_experimental_v1"
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
