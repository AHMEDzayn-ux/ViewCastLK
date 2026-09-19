import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import psycopg2
from psycopg2.extras import RealDictCursor, execute_values

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

    async def get_youtube_connection_secret(self, *, user_id: str) -> dict[str, Any] | None:
        return await asyncio.to_thread(
            self._get_youtube_connection_secret, user_id=user_id
        )

    def _get_youtube_connection_secret(self, *, user_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            with connection.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(
                    """
                    select user_id, channel_id, encrypted_refresh_token, scopes, status
                    from creator.youtube_connections
                    where user_id = %s
                    """,
                    (user_id,),
                )
                row = cursor.fetchone()
        return dict(row) if row else None

    async def set_youtube_connection_status(
        self, *, user_id: str, status: str, refresh_ok: bool = False
    ) -> None:
        await asyncio.to_thread(
            self._set_youtube_connection_status,
            user_id=user_id,
            status=status,
            refresh_ok=refresh_ok,
        )

    def _set_youtube_connection_status(
        self, *, user_id: str, status: str, refresh_ok: bool
    ) -> None:
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    update creator.youtube_connections
                    set status = %s,
                        last_refresh_ok_at = case when %s then now() else last_refresh_ok_at end
                    where user_id = %s
                    """,
                    (status, refresh_ok, user_id),
                )

    async def upsert_video_history(self, *, user_id: str, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        await asyncio.to_thread(self._upsert_video_history, user_id=user_id, rows=rows)

    def _upsert_video_history(self, *, user_id: str, rows: list[dict[str, Any]]) -> None:
        values = [
            (
                user_id,
                row["video_id"],
                row.get("title"),
                row.get("category"),
                row.get("duration_seconds"),
                row["published_at"],
                row["is_short"],
                row.get("d7"),
                row.get("d14"),
                row.get("d21"),
                row.get("d30"),
                row.get("pred7"),
                row.get("pred14"),
                row.get("pred21"),
                row.get("pred30"),
                row.get("model_version"),
            )
            for row in rows
        ]
        with self._connect() as connection:
            with connection.cursor() as cursor:
                execute_values(
                    cursor,
                    """
                    insert into creator.video_history (
                      user_id, video_id, title, category, duration_seconds,
                      published_at, is_short, d7, d14, d21, d30,
                      pred7, pred14, pred21, pred30, model_version
                    ) values %s
                    on conflict (user_id, video_id) do update set
                      title = excluded.title,
                      category = excluded.category,
                      duration_seconds = excluded.duration_seconds,
                      published_at = excluded.published_at,
                      is_short = excluded.is_short,
                      d7 = excluded.d7,
                      d14 = excluded.d14,
                      d21 = excluded.d21,
                      d30 = excluded.d30,
                      pred7 = excluded.pred7,
                      pred14 = excluded.pred14,
                      pred21 = excluded.pred21,
                      pred30 = excluded.pred30,
                      model_version = excluded.model_version,
                      updated_at = now()
                    """,
                    values,
                )

    async def replace_adjustments(
        self, *, user_id: str, model_version: str, rows: list[dict[str, Any]]
    ) -> None:
        await asyncio.to_thread(
            self._replace_adjustments,
            user_id=user_id,
            model_version=model_version,
            rows=rows,
        )

    def _replace_adjustments(
        self, *, user_id: str, model_version: str, rows: list[dict[str, Any]]
    ) -> None:
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    "delete from creator.adjustments where user_id = %s and model_version = %s",
                    (user_id, model_version),
                )
                if rows:
                    execute_values(
                        cursor,
                        """
                        insert into creator.adjustments (
                          user_id, horizon, format, factor, n_videos, model_version
                        ) values %s
                        """,
                        [
                            (
                                user_id,
                                row["horizon"],
                                row["format"],
                                row["factor"],
                                row["n_videos"],
                                model_version,
                            )
                            for row in rows
                        ],
                    )
