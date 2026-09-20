"""Minimal boundary from creator connection to the public collection roster.

Only the public YouTube channel ID crosses this boundary. Private Analytics,
OAuth credentials, Auth user IDs, and personal adjustment data never do.
"""

from __future__ import annotations

import asyncio
from urllib.parse import urlparse, urlunparse

import psycopg2

from app import config


class PublicRosterUnavailable(Exception):
    pass


class PublicRosterStore:
    @property
    def is_configured(self) -> bool:
        return bool(config.SUPABASE_WAREHOUSE_DB_URL)

    def _connect(self):
        if not config.SUPABASE_WAREHOUSE_DB_URL:
            raise PublicRosterUnavailable("Public roster storage is not configured.")
        try:
            clean_url = urlunparse(
                urlparse(config.SUPABASE_WAREHOUSE_DB_URL)._replace(query="")
            )
            return psycopg2.connect(clean_url, connect_timeout=5)
        except psycopg2.Error as exc:
            raise PublicRosterUnavailable(
                "Public roster storage is unavailable."
            ) from exc

    async def request_channel_collection(self, *, channel_id: str) -> str:
        return await asyncio.to_thread(
            self._request_channel_collection, channel_id=channel_id
        )

    def _request_channel_collection(self, *, channel_id: str) -> str:
        try:
            with self._connect() as connection:
                with connection.cursor() as cursor:
                    cursor.execute(
                        "select 1 from public.channels where channel_id = %s",
                        (channel_id,),
                    )
                    if cursor.fetchone():
                        return "already_tracked"
                    cursor.execute(
                        """
                        insert into public.roster_requests (channel_id, source)
                        values (%s, 'creator_connection')
                        on conflict (channel_id) do nothing
                        """,
                        (channel_id,),
                    )
                    return "requested" if cursor.rowcount else "already_requested"
        except psycopg2.Error as exc:
            raise PublicRosterUnavailable(
                "Public roster storage is unavailable."
            ) from exc


async def request_channel_collection_if_available(
    *, store: PublicRosterStore, channel_id: str
) -> str:
    """Best-effort optional handoff; core personalization never depends on it."""
    if not store.is_configured:
        return "skipped"
    try:
        return await store.request_channel_collection(channel_id=channel_id)
    except PublicRosterUnavailable:
        return "skipped"
