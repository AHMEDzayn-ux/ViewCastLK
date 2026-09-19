import base64
from datetime import datetime, timezone
from urllib.parse import parse_qs, urlparse
from unittest.mock import AsyncMock

from fastapi.testclient import TestClient

from app import config
from app.auth import AuthenticatedUser, require_authenticated_user
from app.creator_store import OAuthStateRecord
from app.main import app
import app.main as main_module
from app.youtube_oauth import (
    GoogleTokenResponse,
    YouTubeChannelIdentity,
    YOUTUBE_OAUTH_SCOPES,
    decrypt_refresh_token,
    encrypt_refresh_token,
)


client = TestClient(app)
TEST_KEY = base64.urlsafe_b64encode(bytes(range(32))).decode("ascii")


def _configure_oauth(monkeypatch):
    monkeypatch.setattr(config, "GOOGLE_OAUTH_CLIENT_ID", "test-client-id")
    monkeypatch.setattr(config, "GOOGLE_OAUTH_CLIENT_SECRET", "test-client-secret")
    monkeypatch.setattr(
        config,
        "GOOGLE_OAUTH_REDIRECT_URI",
        "https://api.example.test/auth/youtube/callback",
    )
    monkeypatch.setattr(config, "TOKEN_ENCRYPTION_KEY", TEST_KEY)


def test_oauth_start_requires_authentication():
    response = client.get("/auth/youtube/start")

    assert response.status_code == 401
    assert response.json()["code"] == "authentication_required"


def test_oauth_start_binds_random_state_to_authenticated_user(monkeypatch):
    _configure_oauth(monkeypatch)
    store = AsyncMock()
    monkeypatch.setattr(main_module, "creator_store", store)
    app.dependency_overrides[require_authenticated_user] = lambda: AuthenticatedUser(
        id="user-a"
    )
    try:
        first = client.get("/auth/youtube/start")
        second = client.get("/auth/youtube/start")
    finally:
        app.dependency_overrides.pop(require_authenticated_user, None)

    assert first.status_code == 200
    first_url = urlparse(first.json()["authorizationUrl"])
    second_url = urlparse(second.json()["authorizationUrl"])
    first_query = parse_qs(first_url.query)
    second_query = parse_qs(second_url.query)
    assert first_url.netloc == "accounts.google.com"
    assert first_query["access_type"] == ["offline"]
    assert first_query["prompt"] == ["consent"]
    assert set(first_query["scope"][0].split()) == set(YOUTUBE_OAUTH_SCOPES)
    assert first_query["state"] != second_query["state"]
    assert store.create_oauth_state.await_count == 2
    first_call = store.create_oauth_state.await_args_list[0].kwargs
    assert first_call["user_id"] == "user-a"
    assert first_call["state_hash"] != first_query["state"][0]
    assert first_call["expires_at"] > datetime.now(timezone.utc)


def test_callback_rejects_invalid_expired_or_replayed_state(monkeypatch):
    store = AsyncMock()
    store.consume_oauth_state.return_value = None
    monkeypatch.setattr(main_module, "creator_store", store)

    response = client.get(
        "/auth/youtube/callback", params={"state": "invalid-state", "code": "code"}
    )

    assert response.status_code == 400
    assert response.json()["code"] == "invalid_oauth_state"


def test_callback_encrypts_refresh_token_and_redirects_without_tokens(monkeypatch):
    _configure_oauth(monkeypatch)
    store = AsyncMock()
    store.consume_oauth_state.return_value = OAuthStateRecord(user_id="user-a")
    monkeypatch.setattr(main_module, "creator_store", store)
    monkeypatch.setattr(
        main_module,
        "exchange_authorization_code",
        AsyncMock(
            return_value=GoogleTokenResponse(
                access_token="test-access-token",
                refresh_token="test-refresh-token",
                scopes=YOUTUBE_OAUTH_SCOPES,
            )
        ),
    )
    monkeypatch.setattr(
        main_module,
        "fetch_authenticated_channel",
        AsyncMock(
            return_value=YouTubeChannelIdentity(
                channel_id="UC-test-channel", title="Test creator"
            )
        ),
    )

    response = client.get(
        "/auth/youtube/callback",
        params={"state": "one-time-state", "code": "authorization-code"},
        follow_redirects=False,
    )

    assert response.status_code == 303
    assert response.headers["location"].endswith("/account?youtube=connected")
    assert "test-access-token" not in response.headers["location"]
    assert "test-refresh-token" not in response.headers["location"]
    saved = store.upsert_youtube_connection.await_args.kwargs
    assert saved["user_id"] == "user-a"
    assert saved["channel_id"] == "UC-test-channel"
    assert saved["encrypted_refresh_token"] != "test-refresh-token"
    assert decrypt_refresh_token(
        saved["encrypted_refresh_token"], user_id="user-a"
    ) == "test-refresh-token"


def test_callback_state_is_consumed_before_google_denial(monkeypatch):
    store = AsyncMock()
    store.consume_oauth_state.return_value = OAuthStateRecord(user_id="user-a")
    monkeypatch.setattr(main_module, "creator_store", store)

    response = client.get(
        "/auth/youtube/callback",
        params={"state": "one-time-state", "error": "access_denied"},
        follow_redirects=False,
    )

    assert response.status_code == 303
    assert response.headers["location"].endswith("/account?youtube=not_connected")
    store.consume_oauth_state.assert_awaited_once()
    store.upsert_youtube_connection.assert_not_awaited()


def test_refresh_token_ciphertext_is_user_bound(monkeypatch):
    _configure_oauth(monkeypatch)
    encrypted = encrypt_refresh_token("refresh-value", user_id="user-a")

    assert "refresh-value" not in encrypted
    assert decrypt_refresh_token(encrypted, user_id="user-a") == "refresh-value"
    try:
        decrypt_refresh_token(encrypted, user_id="user-b")
    except Exception:
        pass
    else:
        raise AssertionError("Ciphertext must not decrypt for another user")


def test_connection_status_returns_only_browser_safe_fields(monkeypatch):
    store = AsyncMock()
    store.get_youtube_connection_status.return_value = {
        "channel_id": "UC-test-channel",
        "channel_title": "Test creator",
        "connected_at": datetime(2026, 9, 20, tzinfo=timezone.utc),
        "last_refresh_ok_at": None,
        "status": "pending_sync",
        "encrypted_refresh_token": "must-not-appear",
    }
    monkeypatch.setattr(main_module, "creator_store", store)
    app.dependency_overrides[require_authenticated_user] = lambda: AuthenticatedUser(
        id="user-a"
    )
    try:
        response = client.get("/creator/youtube-connection")
    finally:
        app.dependency_overrides.pop(require_authenticated_user, None)

    assert response.status_code == 200
    assert response.json() == {
        "isConnected": True,
        "channelId": "UC-test-channel",
        "channelTitle": "Test creator",
        "status": "pending_sync",
        "connectedAt": "2026-09-20T00:00:00+00:00",
        "lastRefreshOkAt": None,
    }
    assert "must-not-appear" not in response.text
