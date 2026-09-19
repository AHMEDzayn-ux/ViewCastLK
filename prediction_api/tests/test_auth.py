from unittest.mock import AsyncMock, patch

import httpx
from fastapi.testclient import TestClient

from app.auth import AuthenticatedUser, require_authenticated_user
from app.main import app
from app.schemas import ChannelStatsResponse


client = TestClient(app)

VALID_FORECAST_PAYLOAD = {
    "title": "Authenticated forecast",
    "category": "Education",
    "durationSeconds": 300,
    "audioLanguage": "English",
    "channelIdentifier": "@samplechannel",
    "plannedPublishDay": None,
    "plannedPublishHour": None,
}

MOCK_CHANNEL_STATS = ChannelStatsResponse(
    subscriberCount=1000,
    totalViewCount=100000,
    videoCount=50,
    createdAt="2020-01-01T00:00:00Z",
    channelAgeDays=1200,
)


def test_health_remains_public():
    response = client.get("/health")
    assert response.status_code == 200


def test_forecast_without_authorization_is_denied():
    response = client.post("/forecast", json=VALID_FORECAST_PAYLOAD)
    assert response.status_code == 401
    assert response.json() == {
        "message": "Authentication is required to use this endpoint.",
        "code": "authentication_required",
    }


def test_malformed_bearer_header_is_denied():
    response = client.post(
        "/forecast",
        json=VALID_FORECAST_PAYLOAD,
        headers={"Authorization": "Basic not-a-bearer-token"},
    )
    assert response.status_code == 401
    assert response.json()["code"] == "authentication_required"


def test_invalid_token_is_denied_without_echoing_token():
    token = "expired-private-test-token"
    auth_response = httpx.Response(
        401,
        json={"message": "provider-internal-detail"},
        request=httpx.Request("GET", "https://example.supabase.co/auth/v1/user"),
    )

    with patch(
        "app.auth._request_supabase_user",
        new=AsyncMock(return_value=auth_response),
    ):
        response = client.post(
            "/forecast",
            json=VALID_FORECAST_PAYLOAD,
            headers={"Authorization": f"Bearer {token}"},
        )

    assert response.status_code == 401
    assert response.json()["code"] == "invalid_session"
    assert token not in response.text
    assert "provider-internal-detail" not in response.text


def test_auth_upstream_failure_is_not_reported_as_invalid_token():
    with patch(
        "app.auth._request_supabase_user",
        new=AsyncMock(side_effect=httpx.ConnectError("private upstream detail")),
    ):
        response = client.post(
            "/forecast",
            json=VALID_FORECAST_PAYLOAD,
            headers={"Authorization": "Bearer test-token"},
        )

    assert response.status_code == 503
    assert response.json()["code"] == "auth_service_unavailable"
    assert "private upstream detail" not in response.text
    assert "test-token" not in response.text


def test_mocked_authenticated_user_reaches_existing_forecast_behavior():
    app.dependency_overrides[require_authenticated_user] = lambda: AuthenticatedUser(
        id="test-user"
    )
    try:
        with patch("app.main.fetch_channel_stats", return_value=MOCK_CHANNEL_STATS):
            response = client.post("/forecast", json=VALID_FORECAST_PAYLOAD)
    finally:
        app.dependency_overrides.pop(require_authenticated_user, None)

    assert response.status_code == 200
    assert [item["horizonDays"] for item in response.json()["estimates"]] == [
        7,
        14,
        21,
        30,
    ]


def test_channel_lookup_also_requires_authentication():
    response = client.post(
        "/channel-lookup", json={"channelIdentifier": "@samplechannel"}
    )
    assert response.status_code == 401
    assert response.json()["code"] == "authentication_required"
