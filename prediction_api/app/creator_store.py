import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import psycopg2
from psycopg2.extras import RealDictCursor

from app.config import SUPABASE_AUTH_DB_URL


class CreatorStoreUnavailable(Exception):
    pass


@dataclass(frozen=True)
class OAuthStateRecord:
    user_id: str


class CreatorStore:
    """Server-only persistence for the unexposed creator schema."""

    def _connect(self):
        if not SUPABASE_AUTH_DB_URL:
            raise CreatorStoreUnavailable("Creator storage is not configured.")
        try:
            return psycopg2.connect(SUPABASE_AUTH_DB_URL, connect_timeout=5)
        except psycopg2.Error as exc:
            raise CreatorStoreUnavailable("Creator storage is unavailable.") from exc

    async def create_oauth_state(
        self, *, state_hash: str, user_id: str, expires_at: datetime
    ) -> None:
        await asyncio.to_thread(
            self._create_oauth_state,
            state_hash=state_hash,
            user_id=user_id,
            expires_at=expires_at,
        )

    def _create_oauth_state(
        self, *, state_hash: str, user_id: str, expires_at: datetime
    ) -> None:
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    insert into creator.oauth_states (state_hash, user_id, expires_at)
                    values (%s, %s, %s)
                    """,
                    (state_hash, user_id, expires_at),
                )

    async def consume_oauth_state(self, *, state_hash: str) -> OAuthStateRecord | None:
        return await asyncio.to_thread(self._consume_oauth_state, state_hash=state_hash)

    def _consume_oauth_state(self, *, state_hash: str) -> OAuthStateRecord | None:
        with self._connect() as connection:
            with connection.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(
                    """
                    update creator.oauth_states
                    set consumed_at = now()
                    where state_hash = %s
                      and consumed_at is null
                      and expires_at > now()
                    returning user_id
                    """,
                    (state_hash,),
                )
                row = cursor.fetchone()
        if not row:
            return None
        return OAuthStateRecord(user_id=str(row["user_id"]))

    async def upsert_youtube_connection(
        self,
        *,
        user_id: str,
        channel_id: str,
        channel_title: str | None,
        encrypted_refresh_token: str,
        scopes: tuple[str, ...],
    ) -> None:
        await asyncio.to_thread(
            self._upsert_youtube_connection,
            user_id=user_id,
            channel_id=channel_id,
            channel_title=channel_title,
            encrypted_refresh_token=encrypted_refresh_token,
            scopes=scopes,
        )

    def _upsert_youtube_connection(
        self,
        *,
        user_id: str,
        channel_id: str,
        channel_title: str | None,
        encrypted_refresh_token: str,
        scopes: tuple[str, ...],
    ) -> None:
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    insert into creator.youtube_connections (
                      user_id, channel_id, channel_title,
                      encrypted_refresh_token, scopes, status
                    )
                    values (%s, %s, %s, %s, %s, 'pending_sync')
                    on conflict (user_id) do update set
                      channel_id = excluded.channel_id,
                      channel_title = excluded.channel_title,
                      encrypted_refresh_token = excluded.encrypted_refresh_token,
                      scopes = excluded.scopes,
                      connected_at = now(),
                      last_refresh_ok_at = null,
                      status = 'pending_sync'
                    """,
                    (
                        user_id,
                        channel_id,
                        channel_title,
                        encrypted_refresh_token,
                        list(scopes),
                    ),
                )

    async def get_youtube_connection_status(self, *, user_id: str) -> dict[str, Any] | None:
        return await asyncio.to_thread(
            self._get_youtube_connection_status, user_id=user_id
        )

    def _get_youtube_connection_status(self, *, user_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            with connection.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(
                    """
                    select channel_id, channel_title, connected_at,
                           last_refresh_ok_at, status
                    from creator.youtube_connections
                    where user_id = %s
                    """,
                    (user_id,),
                )
                row = cursor.fetchone()
        return dict(row) if row else None
