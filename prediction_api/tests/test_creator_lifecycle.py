import inspect
from unittest.mock import AsyncMock

import pytest

import app.creator_lifecycle as lifecycle
from app.creator_store import CreatorStore
from app.youtube_oauth import (
    GoogleCredentialRevoked,
    GoogleRefreshResponse,
    YouTubeChannelIdentity,
    YouTubeOAuthException,
)


@pytest.mark.asyncio
async def test_successful_refresh_records_consent_and_recomputes(monkeypatch):
    store = AsyncMock()
    roster = AsyncMock()
    registry = object()
    monkeypatch.setattr(
        lifecycle, "decrypt_refresh_token", lambda *_args, **_kwargs: "refresh"
    )
    monkeypatch.setattr(
        lifecycle,
        "refresh_access_token",
        AsyncMock(return_value=GoogleRefreshResponse("access", ("scope",))),
    )
    channel = YouTubeChannelIdentity("UC-new", "Creator", "UU-uploads")
    monkeypatch.setattr(
        lifecycle, "fetch_authenticated_channel", AsyncMock(return_value=channel)
    )
    sync = AsyncMock()
    monkeypatch.setattr(lifecycle, "synchronize_creator_history", sync)

    result = await lifecycle.refresh_creator_connection(
        connection={"user_id": "user-a", "encrypted_refresh_token": "ciphertext"},
        store=store,
        model_registry=registry,
        roster_store=roster,
    )

    assert result == "refreshed"
    store.record_refresh_success.assert_awaited_once_with(user_id="user-a")
    sync.assert_awaited_once_with(
        user_id="user-a",
        access_token="access",
        channel=channel,
        store=store,
        model_registry=registry,
    )
    roster.request_channel_collection.assert_awaited_once_with(channel_id="UC-new")


@pytest.mark.asyncio
async def test_temporary_refresh_failure_preserves_creator_rows(monkeypatch):
    store = AsyncMock()
    monkeypatch.setattr(
        lifecycle, "decrypt_refresh_token", lambda *_args, **_kwargs: "refresh"
    )
    monkeypatch.setattr(
        lifecycle,
        "refresh_access_token",
        AsyncMock(
            side_effect=YouTubeOAuthException(
                status_code=502,
                message="temporary",
                code="youtube_refresh_temporary_failure",
            )
        ),
    )

    result = await lifecycle.refresh_creator_connection(
        connection={"user_id": "user-a", "encrypted_refresh_token": "ciphertext"},
        store=store,
        model_registry=object(),
        roster_store=AsyncMock(),
    )

    assert result == "temporary_failure"
    store.set_youtube_connection_status.assert_awaited_once_with(
        user_id="user-a", status="error"
    )
    store.delete_creator_data.assert_not_awaited()


@pytest.mark.asyncio
async def test_revoked_refresh_deletes_only_creator_private_rows(monkeypatch):
    store = AsyncMock()
    monkeypatch.setattr(
        lifecycle, "decrypt_refresh_token", lambda *_args, **_kwargs: "refresh"
    )
    monkeypatch.setattr(
        lifecycle,
        "refresh_access_token",
        AsyncMock(side_effect=GoogleCredentialRevoked()),
    )

    result = await lifecycle.refresh_creator_connection(
        connection={"user_id": "user-a", "encrypted_refresh_token": "ciphertext"},
        store=store,
        model_registry=object(),
        roster_store=AsyncMock(),
    )

    assert result == "revoked"
    store.delete_creator_data.assert_awaited_once_with(user_id="user-a")
    store.set_youtube_connection_status.assert_not_awaited()
    deletion_source = inspect.getsource(CreatorStore._delete_creator_data)
    assert "public." not in deletion_source
    for table in (
        "oauth_states",
        "insights",
        "adjustments",
        "video_history",
        "youtube_connections",
    ):
        assert f'"{table}"' in deletion_source


@pytest.mark.asyncio
async def test_disconnect_revokes_then_deletes(monkeypatch):
    store = AsyncMock()
    store.get_youtube_connection_secret.return_value = {
        "encrypted_refresh_token": "ciphertext"
    }
    monkeypatch.setattr(
        lifecycle, "decrypt_refresh_token", lambda *_args, **_kwargs: "refresh"
    )
    revoke = AsyncMock()
    monkeypatch.setattr(lifecycle, "revoke_google_token", revoke)

    await lifecycle.disconnect_creator_connection(user_id="user-a", store=store)

    revoke.assert_awaited_once_with("refresh")
    store.delete_creator_data.assert_awaited_once_with(user_id="user-a")


@pytest.mark.asyncio
async def test_disconnect_deletes_locally_when_google_is_unavailable(monkeypatch):
    store = AsyncMock()
    store.get_youtube_connection_secret.return_value = {
        "encrypted_refresh_token": "ciphertext"
    }
    monkeypatch.setattr(
        lifecycle, "decrypt_refresh_token", lambda *_args, **_kwargs: "refresh"
    )
    monkeypatch.setattr(
        lifecycle, "revoke_google_token", AsyncMock(side_effect=RuntimeError("offline"))
    )

    await lifecycle.disconnect_creator_connection(user_id="user-a", store=store)

    store.delete_creator_data.assert_awaited_once_with(user_id="user-a")


@pytest.mark.asyncio
async def test_refresh_job_processes_each_eligible_connection(monkeypatch):
    store = AsyncMock()
    store.list_refreshable_connections.return_value = [
        {"user_id": "a"},
        {"user_id": "b"},
        {"user_id": "c"},
    ]
    process = AsyncMock(side_effect=["refreshed", "revoked", "temporary_failure"])
    monkeypatch.setattr(lifecycle, "refresh_creator_connection", process)

    result = await lifecycle.run_creator_refresh_job(
        store=store,
        model_registry=object(),
        roster_store=AsyncMock(),
    )

    assert result == {"refreshed": 1, "revoked": 1, "temporary_failure": 1}
    assert process.await_count == 3


@pytest.mark.asyncio
async def test_refresh_job_continues_after_one_unexpected_failure(monkeypatch):
    store = AsyncMock()
    store.list_refreshable_connections.return_value = [
        {"user_id": "a"},
        {"user_id": "b"},
    ]
    process = AsyncMock(side_effect=[RuntimeError("bad row"), "refreshed"])
    monkeypatch.setattr(lifecycle, "refresh_creator_connection", process)

    result = await lifecycle.run_creator_refresh_job(
        store=store,
        model_registry=object(),
        roster_store=AsyncMock(),
    )

    assert result == {"refreshed": 1, "revoked": 0, "temporary_failure": 1}
    assert process.await_count == 2
