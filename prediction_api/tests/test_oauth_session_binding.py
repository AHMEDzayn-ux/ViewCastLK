"""Exercise production state SQL on a local transactional SQLite adapter.

Only psycopg's connection/placeholder interface and clock function are adapted.
All owner/session/nonce/expiry/replay predicates are executed by the database.
PostgreSQL/live auth.sessions permissions remain an integration verification.
"""

import base64
import json
import logging
import sqlite3
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest
from fastapi.testclient import TestClient

from app import auth, config
from app.auth import AuthenticatedUser, require_authenticated_user
from app.creator_store import CreatorStore
import app.main as main
from app.oauth_logging import OAuthCallbackAccessFilter
from app.youtube_oauth import (
    GoogleTokenResponse, YouTubeChannelIdentity, YouTubeOAuthException, YOUTUBE_OAUTH_SCOPES,
    hash_oauth_state, hash_session_binding,
)

SESSION_A = "00000000-0000-0000-0000-000000000001"
SESSION_B = "00000000-0000-0000-0000-000000000002"
client = TestClient(main.app)


class LocalCursor:
    def __init__(self, connection):
        self.cursor = connection.cursor()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.cursor.close()

    def execute(self, sql, params):
        converted = tuple(v.isoformat() if isinstance(v, datetime) else v for v in params)
        self.cursor.execute(sql.replace("%s", "?"), converted)

    def fetchone(self):
        return self.cursor.fetchone()


class LocalConnection:
    def __init__(self, root):
        self.connection = sqlite3.connect(":memory:")
        self.connection.row_factory = sqlite3.Row
        self.connection.create_function("now", 0, lambda: datetime.now(timezone.utc).isoformat())
        for schema in ("creator", "auth"):
            self.connection.execute(f"ATTACH DATABASE ? AS {schema}", (str(root / f"{schema}.db"),))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, *_):
        if exc_type:
            self.connection.rollback()
        else:
            self.connection.commit()
        self.connection.close()

    def cursor(self, **_):
        return LocalCursor(self.connection)


@pytest.fixture
def flow(monkeypatch, tmp_path):
    with LocalConnection(tmp_path) as db:
        db.connection.executescript("""
            CREATE TABLE auth.sessions (id TEXT PRIMARY KEY, user_id TEXT);
            CREATE TABLE creator.oauth_states (
                state_hash TEXT PRIMARY KEY, user_id TEXT, session_id TEXT,
                binding_hash TEXT, expires_at TEXT, consumed_at TEXT
            );
        """)
        db.connection.executemany("INSERT INTO auth.sessions VALUES (?, ?)", [(SESSION_A, "user-a"), (SESSION_B, "user-a")])
    store = CreatorStore()
    monkeypatch.setattr(store, "_connect", lambda: LocalConnection(tmp_path))
    monkeypatch.setattr(main, "creator_store", store)
    monkeypatch.setattr(config, "GOOGLE_OAUTH_CLIENT_ID", "test-client")
    monkeypatch.setattr(config, "GOOGLE_OAUTH_CLIENT_SECRET", "test-only")
    monkeypatch.setattr(config, "TOKEN_ENCRYPTION_KEY", base64.urlsafe_b64encode(bytes(range(32))).decode())
    exchange = AsyncMock(return_value=GoogleTokenResponse("test-access", "test-refresh", YOUTUBE_OAUTH_SCOPES))
    channel = AsyncMock(return_value=YouTubeChannelIdentity("UC-test", "Test channel"))
    save = AsyncMock()
    sync = AsyncMock()
    roster = AsyncMock()
    monkeypatch.setattr(main, "exchange_authorization_code", exchange)
    monkeypatch.setattr(main, "fetch_authenticated_channel", channel)
    monkeypatch.setattr(store, "upsert_youtube_connection", save)
    monkeypatch.setattr(main, "synchronize_creator_history", sync)
    monkeypatch.setattr(main, "request_channel_collection_if_available", roster)
    app_user = [AuthenticatedUser("user-a", SESSION_A)]
    main.app.dependency_overrides[require_authenticated_user] = lambda: app_user[0]
    started = client.get("/auth/youtube/start")
    assert started.status_code == 200
    start = started.json()
    body = {"state": start["state"], "bindingNonce": start["bindingNonce"], "code": "test-code"}
    yield {"start": start, "body": body, "user": app_user, "root": tmp_path,
           "effects": [exchange, channel, save, sync, roster], "store": store}
    main.app.dependency_overrides.pop(require_authenticated_user, None)


def assert_no_effects(flow):
    for effect in flow["effects"]:
        effect.assert_not_awaited()


def test_same_user_session_and_tab_proof_succeeds(flow):
    relay = client.get("/auth/youtube/callback", params={"state": flow["body"]["state"], "code": "test-code"}, follow_redirects=False)
    assert relay.status_code == 303
    target = urlsplit(relay.headers["location"])
    assert target.path == "/account/youtube-callback"
    assert target.query == ""
    assert parse_qs(target.fragment)["youtube_code"] == ["test-code"]
    assert relay.headers["cache-control"] == "no-store"
    assert relay.headers["referrer-policy"] == "no-referrer"
    assert_no_effects(flow)  # GET callback alone cannot link ANY channel.
    response = client.post("/auth/youtube/complete", json=flow["body"])
    assert response.status_code == 200
    assert response.json() == {"connected": True}
    for effect in flow["effects"]:
        effect.assert_awaited_once()
    saved = flow["effects"][2].await_args.kwargs
    assert saved["user_id"] == "user-a"
    assert saved["encrypted_refresh_token"] != "test-refresh"
    assert "test-refresh" not in response.text


@pytest.mark.parametrize("problem", [
    "missing_state", "unknown_state", "expired_state", "wrong_user", "wrong_session",
    "missing_binding", "other_browser_binding", "tampered_binding", "missing_session",
    "signed_out_session", "legacy_unbound_state", "missing_google_response",
])
def test_rejection_has_no_exchange_storage_roster_or_sync(flow, problem):
    body = dict(flow["body"])
    if problem == "missing_state":
        body.pop("state")
    elif problem == "unknown_state":
        body["state"] = "unknown-state"
    elif problem == "expired_state":
        with LocalConnection(flow["root"]) as db:
            db.connection.execute("UPDATE creator.oauth_states SET expires_at=?", ((datetime.now(timezone.utc)-timedelta(seconds=1)).isoformat(),))
    elif problem == "wrong_user":
        flow["user"][0] = AuthenticatedUser("user-b", SESSION_A)
    elif problem == "wrong_session":
        flow["user"][0] = AuthenticatedUser("user-a", SESSION_B)
    elif problem == "missing_session":
        flow["user"][0] = AuthenticatedUser("user-a")
    elif problem == "missing_binding":
        body.pop("bindingNonce")
    elif problem == "other_browser_binding":
        body["bindingNonce"] = "Z" * 43
    elif problem == "tampered_binding":
        body["bindingNonce"] = ("a" if body["bindingNonce"][0] != "a" else "b") + body["bindingNonce"][1:]
    elif problem == "signed_out_session":
        with LocalConnection(flow["root"]) as db:
            db.connection.execute("DELETE FROM auth.sessions WHERE id=?", (SESSION_A,))
    elif problem == "legacy_unbound_state":
        with LocalConnection(flow["root"]) as db:
            db.connection.execute("UPDATE creator.oauth_states SET binding_hash=NULL, session_id=NULL")
    elif problem == "missing_google_response":
        body.pop("code")
    response = client.post("/auth/youtube/complete", json=body)
    assert response.status_code == 400
    assert_no_effects(flow)
    for secret in [body.get("bindingNonce"), "test-code", "test-refresh"]:
        if secret:
            assert secret not in response.text


def test_wrong_proof_does_not_burn_the_legitimate_flow(flow):
    assert client.post("/auth/youtube/complete", json={**flow["body"], "bindingNonce": "Z"*43}).status_code == 400
    assert_no_effects(flow)
    assert client.post("/auth/youtube/complete", json=flow["body"]).status_code == 200


def test_nonce_from_a_different_started_flow_cannot_complete_this_state(flow):
    second = client.get("/auth/youtube/start").json()
    wrong_pair = {**flow["body"], "bindingNonce": second["bindingNonce"]}
    assert client.post("/auth/youtube/complete", json=wrong_pair).status_code == 400
    assert_no_effects(flow)
    assert client.post("/auth/youtube/complete", json=flow["body"]).status_code == 200


def test_active_session_row_must_belong_to_the_initiating_user(flow):
    with LocalConnection(flow["root"]) as db:
        db.connection.execute("UPDATE auth.sessions SET user_id='user-b' WHERE id=?", (SESSION_A,))
    assert client.post("/auth/youtube/complete", json=flow["body"]).status_code == 400
    assert_no_effects(flow)


def test_invalid_google_response_cannot_store_or_synchronize(flow):
    flow["effects"][0].side_effect = YouTubeOAuthException(
        status_code=502, message="Google could not complete the channel connection. Please try again.",
        code="youtube_oauth_exchange_failed",
    )
    response = client.post("/auth/youtube/complete", json=flow["body"])
    assert response.status_code == 502
    assert "test-code" not in response.text
    flow["effects"][0].assert_awaited_once()
    for effect in flow["effects"][1:]:
        effect.assert_not_awaited()
    # Preserve single-use state even when Google refuses the code.
    assert client.post("/auth/youtube/complete", json=flow["body"]).status_code == 400
    flow["effects"][0].assert_awaited_once()


def test_replay_cannot_repeat_google_exchange_or_persist(flow):
    assert client.post("/auth/youtube/complete", json=flow["body"]).status_code == 200
    for effect in flow["effects"]:
        effect.reset_mock()
    assert client.post("/auth/youtube/complete", json=flow["body"]).status_code == 400
    assert_no_effects(flow)
    assert client.get("/auth/youtube/callback", params={"state": flow["body"]["state"], "code": "test-code"}).status_code == 400


@pytest.mark.parametrize("problem", ["missing", "unknown", "expired"])
def test_relay_rejects_bad_state_without_side_effects(flow, problem):
    params = {"code": "test-code"}
    if problem != "missing":
        params["state"] = "unknown" if problem == "unknown" else flow["body"]["state"]
    if problem == "expired":
        with LocalConnection(flow["root"]) as db:
            db.connection.execute("UPDATE creator.oauth_states SET expires_at='2000-01-01T00:00:00+00:00'")
    assert client.get("/auth/youtube/callback", params=params).status_code == 400
    assert_no_effects(flow)


def test_reconnect_gets_independent_state_and_binding(flow):
    assert client.post("/auth/youtube/complete", json=flow["body"]).status_code == 200
    next_start = client.get("/auth/youtube/start").json()
    assert next_start["state"] != flow["start"]["state"]
    assert next_start["bindingNonce"] != flow["start"]["bindingNonce"]
    assert client.post("/auth/youtube/complete", json={"state": next_start["state"], "bindingNonce": next_start["bindingNonce"], "code": "next-code"}).status_code == 200
    assert flow["effects"][2].await_count == 2


def test_plaintext_binding_is_not_stored(flow):
    with LocalConnection(flow["root"]) as db:
        row = db.connection.execute("SELECT * FROM creator.oauth_states").fetchone()
        assert row["state_hash"] == hash_oauth_state(flow["start"]["state"])
        assert row["binding_hash"] == hash_session_binding(flow["start"]["bindingNonce"])
        assert flow["start"]["bindingNonce"] not in list(row)


def test_completion_requires_authentication(flow):
    main.app.dependency_overrides.pop(require_authenticated_user)
    assert client.post("/auth/youtube/complete", json=flow["body"]).status_code == 401
    assert_no_effects(flow)


def test_start_requires_a_validated_session_id(flow):
    flow["user"][0] = AuthenticatedUser("user-a")
    assert client.get("/auth/youtube/start").status_code == 401
    assert_no_effects(flow)


def test_oauth_callback_access_logs_redact_queries_only():
    record = logging.LogRecord("uvicorn.access", logging.INFO, "", 0, '%s - "%s %s HTTP/%s" %s',
        ("client", "GET", "/auth/youtube/callback?code=private-code&state=private-state", "1.1", 303), None)
    assert OAuthCallbackAccessFilter().filter(record)
    assert "private-code" not in record.getMessage()
    assert "private-state" not in record.getMessage()
    assert "/auth/youtube/callback" in record.getMessage()
    assert "303" in record.getMessage()
    assert any(isinstance(f, OAuthCallbackAccessFilter) for f in logging.getLogger("uvicorn.access").filters)


@pytest.mark.parametrize("session_claim", [SESSION_A, None, 42, "not-a-uuid"])
def test_session_claim_is_read_only_after_remote_validation(monkeypatch, session_claim):
    claims = {"sub": "user-a", "session_id": session_claim}
    encoded = base64.urlsafe_b64encode(json.dumps(claims).encode()).decode().rstrip("=")
    token = f"header.{encoded}.signature"
    remote = AsyncMock(return_value=httpx.Response(200, json={"id": "user-a"}))
    monkeypatch.setattr(auth, "_request_supabase_user", remote)
    import asyncio
    validated = asyncio.run(auth.validate_access_token(token))
    remote.assert_awaited_once_with(token)
    assert validated.session_id == (SESSION_A if session_claim == SESSION_A else None)
    remote.return_value = httpx.Response(401)
    with pytest.raises(auth.AuthenticationException):
        asyncio.run(auth.validate_access_token(token))


@pytest.mark.parametrize("claims", [{"sub": "user-b", "session_id": SESSION_A}, [SESSION_A]])
def test_validated_user_must_match_the_session_claim_subject(monkeypatch, claims):
    encoded = base64.urlsafe_b64encode(json.dumps(claims).encode()).decode().rstrip("=")
    token = f"header.{encoded}.signature"
    remote = AsyncMock(return_value=httpx.Response(200, json={"id": "user-a"}))
    monkeypatch.setattr(auth, "_request_supabase_user", remote)
    import asyncio
    validated = asyncio.run(auth.validate_access_token(token))
    assert validated.id == "user-a"
    assert validated.session_id is None
    remote.assert_awaited_once_with(token)
