"""Server-only refresh, revocation, and disconnect lifecycle operations."""

from __future__ import annotations

import sys
from collections import Counter
from typing import Any

from app.service_limits import bounded_creator_sync as synchronize_creator_history
from app.public_roster import request_channel_collection_if_available
from app.rate_limits import enforce_user_limit, RateLimitExceeded
from app.log_safety import redact
from app.youtube_oauth import (
    GoogleCredentialRevoked,
    decrypt_refresh_token,
    fetch_authenticated_channel,
    refresh_access_token,
    revoke_google_token,
)


def _report(user_id: str, stage: str, exc: Exception) -> None:
    """Say why a creator was not refreshed, without echoing credentials.

    Keep the stage/class for diagnosis. Arbitrary exception strings may contain
    tokens, database URLs or provider payloads and must never be printed.
    """
    print(redact(f"creator {user_id}: {stage} failed: {type(exc).__name__}"), file=sys.stderr)


async def refresh_creator_connection(
    *, connection: dict[str, Any], store, model_registry, roster_store
) -> str:
    user_id = str(connection["user_id"])
    try:
        await enforce_user_limit("creator_sync", user_id)
    except RateLimitExceeded:
        return "cooldown"
    try:
        refresh_token = decrypt_refresh_token(
            str(connection["encrypted_refresh_token"]), user_id=user_id
        )
        refreshed = await refresh_access_token(refresh_token)
    except GoogleCredentialRevoked:
        await store.delete_creator_data(user_id=user_id)
        return "revoked"
    except Exception as exc:
        # Crypto/configuration and temporary provider/network failures are not
        # evidence of revocation. Preserve private rows so a later run can retry.
        _report(user_id, "token refresh", exc)
        await store.set_youtube_connection_status(user_id=user_id, status="error")
        return "temporary_failure"

    await store.record_refresh_success(user_id=user_id)
    try:
        channel = await fetch_authenticated_channel(refreshed.access_token)
        await synchronize_creator_history(
            user_id=user_id,
            access_token=refreshed.access_token,
            channel=channel,
            store=store,
            model_registry=model_registry,
        )
        await request_channel_collection_if_available(
            store=roster_store,
            channel_id=channel.channel_id,
        )
    except Exception as exc:
        _report(user_id, "history sync", exc)
        await store.set_youtube_connection_status(user_id=user_id, status="error")
        return "temporary_failure"
    return "refreshed"


async def disconnect_creator_connection(*, user_id: str, store) -> None:
    connection = await store.get_youtube_connection_secret(user_id=user_id)
    if connection is None:
        return
    try:
        refresh_token = decrypt_refresh_token(
            str(connection["encrypted_refresh_token"]), user_id=user_id
        )
        await revoke_google_token(refresh_token)
    except Exception:
        # Local deletion is immediate even if Google is temporarily unreachable
        # or has already invalidated the grant. No credential is retained to retry.
        pass
    finally:
        await store.delete_creator_data(user_id=user_id)


async def run_creator_refresh_job(*, store, model_registry, roster_store) -> dict[str, int]:
    results: Counter[str] = Counter()
    for connection in await store.list_refreshable_connections():
        try:
            result = await refresh_creator_connection(
                connection=connection,
                store=store,
                model_registry=model_registry,
                roster_store=roster_store,
            )
        except Exception as exc:
            # One malformed row or transient per-user failure must not prevent
            # other connected creators from being refreshed in the same run.
            _report(str(connection.get("user_id")), "refresh", exc)
            result = "temporary_failure"
        results[result] += 1
    summary = {
        "refreshed": results["refreshed"],
        "revoked": results["revoked"],
        "temporary_failure": results["temporary_failure"],
    }
    if results["cooldown"]:
        summary["cooldown"] = results["cooldown"]
    return summary
