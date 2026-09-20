import base64
import hashlib
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from urllib.parse import urlencode

import httpx
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from app import config


YOUTUBE_OAUTH_SCOPES = (
    "https://www.googleapis.com/auth/yt-analytics.readonly",
    "https://www.googleapis.com/auth/youtube.readonly",
)
GOOGLE_AUTHORIZATION_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_REVOCATION_URL = "https://oauth2.googleapis.com/revoke"
YOUTUBE_CHANNELS_URL = "https://www.googleapis.com/youtube/v3/channels"
OAUTH_STATE_LIFETIME = timedelta(minutes=10)


class YouTubeOAuthException(Exception):
    def __init__(self, *, status_code: int, message: str, code: str):
        super().__init__(message)
        self.status_code = status_code
        self.message = message
        self.code = code


class GoogleCredentialRevoked(Exception):
    """The refresh grant is no longer valid and private data must be removed."""


@dataclass(frozen=True)
class GoogleTokenResponse:
    access_token: str
    refresh_token: str
    scopes: tuple[str, ...]


@dataclass(frozen=True)
class GoogleRefreshResponse:
    access_token: str
    scopes: tuple[str, ...]


@dataclass(frozen=True)
class YouTubeChannelIdentity:
    channel_id: str
    title: str | None
    uploads_playlist_id: str | None = None
    published_at: datetime | None = None
    subscriber_count: int | None = None
    view_count: int | None = None
    video_count: int | None = None


def require_oauth_configuration() -> None:
    if not all(
        (
            config.GOOGLE_OAUTH_CLIENT_ID,
            config.GOOGLE_OAUTH_CLIENT_SECRET,
            config.GOOGLE_OAUTH_REDIRECT_URI,
            config.TOKEN_ENCRYPTION_KEY,
        )
    ):
        raise YouTubeOAuthException(
            status_code=503,
            message="YouTube channel connection is temporarily unavailable.",
            code="youtube_connection_unavailable",
        )


def generate_oauth_state() -> str:
    return secrets.token_urlsafe(32)


def hash_oauth_state(state: str) -> str:
    return hashlib.sha256(state.encode("utf-8")).hexdigest()


def oauth_state_expiry() -> datetime:
    return datetime.now(timezone.utc) + OAUTH_STATE_LIFETIME


def build_authorization_url(state: str) -> str:
    require_oauth_configuration()
    parameters = {
        "client_id": config.GOOGLE_OAUTH_CLIENT_ID,
        "redirect_uri": config.GOOGLE_OAUTH_REDIRECT_URI,
        "response_type": "code",
        "scope": " ".join(YOUTUBE_OAUTH_SCOPES),
        "state": state,
        "access_type": "offline",
        "prompt": "consent",
        "include_granted_scopes": "true",
    }
    return f"{GOOGLE_AUTHORIZATION_URL}?{urlencode(parameters)}"


async def exchange_authorization_code(code: str) -> GoogleTokenResponse:
    require_oauth_configuration()
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                GOOGLE_TOKEN_URL,
                data={
                    "client_id": config.GOOGLE_OAUTH_CLIENT_ID,
                    "client_secret": config.GOOGLE_OAUTH_CLIENT_SECRET,
                    "code": code,
                    "grant_type": "authorization_code",
                    "redirect_uri": config.GOOGLE_OAUTH_REDIRECT_URI,
                },
                headers={"Accept": "application/json"},
            )
    except httpx.HTTPError as exc:
        raise _oauth_exchange_failed() from exc

    if response.status_code != 200:
        raise _oauth_exchange_failed()

    try:
        payload = response.json()
    except ValueError as exc:
        raise _oauth_exchange_failed() from exc

    access_token = payload.get("access_token") if isinstance(payload, dict) else None
    refresh_token = payload.get("refresh_token") if isinstance(payload, dict) else None
    scope_value = payload.get("scope", "") if isinstance(payload, dict) else ""
    scopes = tuple(item for item in str(scope_value).split() if item)

    if not access_token or not refresh_token:
        raise YouTubeOAuthException(
            status_code=400,
            message="Google did not return offline access. Please reconnect and approve access.",
            code="youtube_offline_access_required",
        )
    if scopes and not set(YOUTUBE_OAUTH_SCOPES).issubset(scopes):
        raise YouTubeOAuthException(
            status_code=400,
            message="The required YouTube permissions were not granted.",
            code="youtube_scopes_missing",
        )

    return GoogleTokenResponse(
        access_token=str(access_token),
        refresh_token=str(refresh_token),
        scopes=tuple(sorted(scopes or YOUTUBE_OAUTH_SCOPES)),
    )


async def refresh_access_token(refresh_token: str) -> GoogleRefreshResponse:
    """Exchange a server-held refresh credential without exposing it to clients."""
    require_oauth_configuration()
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                GOOGLE_TOKEN_URL,
                data={
                    "client_id": config.GOOGLE_OAUTH_CLIENT_ID,
                    "client_secret": config.GOOGLE_OAUTH_CLIENT_SECRET,
                    "refresh_token": refresh_token,
                    "grant_type": "refresh_token",
                },
                headers={"Accept": "application/json"},
            )
    except httpx.HTTPError as exc:
        raise _refresh_failed() from exc

    try:
        payload = response.json()
    except ValueError as exc:
        raise _refresh_failed() from exc

    if response.status_code != 200:
        if isinstance(payload, dict) and payload.get("error") == "invalid_grant":
            raise GoogleCredentialRevoked()
        raise _refresh_failed()

    access_token = payload.get("access_token") if isinstance(payload, dict) else None
    scope_value = payload.get("scope", "") if isinstance(payload, dict) else ""
    scopes = tuple(item for item in str(scope_value).split() if item)
    if not access_token:
        raise _refresh_failed()
    if scopes and not set(YOUTUBE_OAUTH_SCOPES).issubset(scopes):
        raise GoogleCredentialRevoked()
    return GoogleRefreshResponse(
        access_token=str(access_token),
        scopes=tuple(sorted(scopes or YOUTUBE_OAUTH_SCOPES)),
    )


async def revoke_google_token(refresh_token: str) -> None:
    """Revoke a Google grant; an already-invalid token is treated idempotently."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                GOOGLE_REVOCATION_URL,
                data={"token": refresh_token},
                headers={
                    "Accept": "application/json",
                    "Content-Type": "application/x-www-form-urlencoded",
                },
            )
    except httpx.HTTPError as exc:
        raise YouTubeOAuthException(
            status_code=502,
            message="Google could not confirm revocation.",
            code="youtube_revocation_failed",
        ) from exc
    if response.status_code not in (200, 400):
        raise YouTubeOAuthException(
            status_code=502,
            message="Google could not confirm revocation.",
            code="youtube_revocation_failed",
        )


def _oauth_exchange_failed() -> YouTubeOAuthException:
    return YouTubeOAuthException(
        status_code=502,
        message="Google could not complete the channel connection. Please try again.",
        code="youtube_oauth_exchange_failed",
    )


def _refresh_failed() -> YouTubeOAuthException:
    return YouTubeOAuthException(
        status_code=502,
        message="Google access could not be refreshed. It will be retried later.",
        code="youtube_refresh_temporary_failure",
    )


async def fetch_authenticated_channel(access_token: str) -> YouTubeChannelIdentity:
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                YOUTUBE_CHANNELS_URL,
                params={
                    "part": "id,snippet,contentDetails,statistics",
                    "mine": "true",
                },
                headers={
                    "Authorization": f"Bearer {access_token}",
                    "Accept": "application/json",
                },
            )
    except httpx.HTTPError as exc:
        raise _channel_lookup_failed() from exc

    if response.status_code != 200:
        raise _channel_lookup_failed()
    try:
        payload = response.json()
        items = payload.get("items", [])
        first = items[0]
        channel_id = first["id"]
        snippet = first.get("snippet", {})
        title = snippet.get("title")
        uploads_playlist_id = first.get("contentDetails", {}).get(
            "relatedPlaylists", {}
        ).get("uploads")
        published_raw = snippet.get("publishedAt")
        published_at = (
            datetime.fromisoformat(published_raw.replace("Z", "+00:00"))
            if isinstance(published_raw, str)
            else None
        )
        statistics = first.get("statistics", {})
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise _channel_lookup_failed() from exc

    if not isinstance(channel_id, str) or not channel_id:
        raise _channel_lookup_failed()
    return YouTubeChannelIdentity(
        channel_id=channel_id,
        title=title if isinstance(title, str) and title else None,
        uploads_playlist_id=(
            uploads_playlist_id if isinstance(uploads_playlist_id, str) else None
        ),
        published_at=published_at,
        subscriber_count=_optional_int(statistics.get("subscriberCount")),
        view_count=_optional_int(statistics.get("viewCount")),
        video_count=_optional_int(statistics.get("videoCount")),
    )


def _optional_int(value) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _channel_lookup_failed() -> YouTubeOAuthException:
    return YouTubeOAuthException(
        status_code=502,
        message="The connected YouTube channel could not be identified.",
        code="youtube_channel_lookup_failed",
    )


def _encryption_key() -> bytes:
    value = config.TOKEN_ENCRYPTION_KEY
    try:
        padded = value + "=" * (-len(value) % 4)
        key = base64.urlsafe_b64decode(padded.encode("ascii"))
    except (ValueError, UnicodeEncodeError) as exc:
        raise YouTubeOAuthException(
            status_code=503,
            message="YouTube channel connection is temporarily unavailable.",
            code="youtube_connection_unavailable",
        ) from exc
    if len(key) != 32:
        raise YouTubeOAuthException(
            status_code=503,
            message="YouTube channel connection is temporarily unavailable.",
            code="youtube_connection_unavailable",
        )
    return key


def encrypt_refresh_token(refresh_token: str, *, user_id: str) -> str:
    require_oauth_configuration()
    nonce = secrets.token_bytes(12)
    ciphertext = AESGCM(_encryption_key()).encrypt(
        nonce,
        refresh_token.encode("utf-8"),
        user_id.encode("utf-8"),
    )
    encoded = base64.urlsafe_b64encode(nonce + ciphertext).decode("ascii")
    return f"v1:{encoded}"


def decrypt_refresh_token(encrypted_token: str, *, user_id: str) -> str:
    if not encrypted_token.startswith("v1:"):
        raise ValueError("Unsupported encrypted token version.")
    raw = base64.urlsafe_b64decode(encrypted_token[3:].encode("ascii"))
    plaintext = AESGCM(_encryption_key()).decrypt(
        raw[:12], raw[12:], user_id.encode("utf-8")
    )
    return plaintext.decode("utf-8")
