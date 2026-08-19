from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.main import app
from app.youtube import fetch_channel_stats

client = TestClient(app)


def test_channel_lookup_success_by_handle():
    mock_youtube = MagicMock()
    mock_request = MagicMock()
    mock_request.execute.return_value = {
        "items": [
            {
                "snippet": {"publishedAt": "2018-03-14T12:00:00Z"},
                "statistics": {
                    "subscriberCount": "125000",
                    "viewCount": "50000000",
                    "videoCount": "500",
                    "hiddenSubscriberCount": False,
                },
            }
        ]
    }
    mock_youtube.channels().list.return_value = mock_request

    result = fetch_channel_stats(
        "@wasthi", api_key="test-api-key", youtube_client=mock_youtube
    )

    mock_youtube.channels().list.assert_called_once_with(
        part="snippet,statistics", forHandle="@wasthi"
    )

    assert result.subscriberCount == 125000
    assert result.totalViewCount == 50000000
    assert result.videoCount == 500
    assert result.createdAt == "2018-03-14T12:00:00Z"
    assert result.channelAgeDays is not None
    assert result.channelAgeDays > 0


def test_channel_lookup_success_by_id():
    channel_id = "UCX6OQ3DkcsbYNE6H8uQQuVA"
    mock_youtube = MagicMock()
    mock_request = MagicMock()
    mock_request.execute.return_value = {
        "items": [
            {
                "snippet": {"publishedAt": "2020-01-01T00:00:00Z"},
                "statistics": {
                    "subscriberCount": "50000",
                    "viewCount": "1000000",
                    "videoCount": "120",
                    "hiddenSubscriberCount": False,
                },
            }
        ]
    }
    mock_youtube.channels().list.return_value = mock_request

    result = fetch_channel_stats(
        channel_id, api_key="test-api-key", youtube_client=mock_youtube
    )

    mock_youtube.channels().list.assert_called_once_with(
        part="snippet,statistics", id=channel_id
    )

    assert result.subscriberCount == 50000
    assert result.totalViewCount == 1000000
    assert result.videoCount == 120


def test_channel_lookup_hidden_subscriber_count():
    mock_youtube = MagicMock()
    mock_request = MagicMock()
    mock_request.execute.return_value = {
        "items": [
            {
                "snippet": {"publishedAt": "2021-06-15T00:00:00Z"},
                "statistics": {
                    "subscriberCount": "10000",
                    "viewCount": "200000",
                    "videoCount": "45",
                    "hiddenSubscriberCount": True,
                },
            }
        ]
    }
    mock_youtube.channels().list.return_value = mock_request

    result = fetch_channel_stats(
        "@hiddensubs", api_key="test-api-key", youtube_client=mock_youtube
    )

    assert result.subscriberCount is None
    assert result.totalViewCount == 200000
    assert result.videoCount == 45


def test_channel_lookup_not_found():
    mock_youtube = MagicMock()
    mock_request = MagicMock()
    mock_request.execute.return_value = {"items": []}
    mock_youtube.channels().list.return_value = mock_request

    with patch("app.main.YOUTUBE_API_KEY", "dummy-key"):
        with patch("app.youtube.build", return_value=mock_youtube):
            response = client.post(
                "/channel-lookup", json={"channelIdentifier": "@nonexistent"}
            )
            assert response.status_code == 404
            assert response.json() == {
                "message": "The channel could not be found.",
                "code": "channel_not_found",
            }


def test_channel_lookup_invalid_identifier():
    response = client.post(
        "/channel-lookup", json={"channelIdentifier": "invalid handle name"}
    )
    assert response.status_code == 400
    assert response.json() == {
        "message": "The channel URL or identifier should not contain spaces.",
        "code": "invalid_channel_identifier",
    }


def test_channel_lookup_missing_api_key():
    with patch("app.main.YOUTUBE_API_KEY", ""):
        response = client.post(
            "/channel-lookup", json={"channelIdentifier": "@channel"}
        )
        assert response.status_code == 500
        assert response.json() == {
            "message": "YouTube API key is not configured.",
            "code": "api_not_configured",
        }
