"""Shared, fail-closed quotas; no in-memory production counters or raw IP storage."""
import asyncio
import hashlib
import hmac
import ipaddress
import json
import os
from dataclasses import dataclass
from contextlib import closing
import psycopg2
from fastapi import Request
from app import config

DEFAULT_LIMITS = {
    "forecast_guest": [(6, 60), (60, 86400)],
    "forecast_user": [(12, 60), (120, 86400)],
    "channel_lookup": [(20, 60), (300, 86400)],
    "oauth_start": [(3, 600), (20, 86400)],
    "oauth_complete": [(6, 600), (30, 86400)],
    "creator_read": [(30, 60)], "disconnect": [(6, 60)],
    "creator_sync": [(1, 1800)], "ingress": [(60, 60)],
    "forecast_global": [(120, 60), (1000, 86400)],
    "channel_lookup_global": [(120, 60), (3000, 86400)],
    "oauth_global": [(120, 60), (1000, 86400)], "creator_global": [(300, 60)],
}
PROTECTED_PATHS = {
    "/forecast": "forecast_global", "/channel-lookup": "channel_lookup_global",
    "/auth/youtube/start": "oauth_global", "/auth/youtube/callback": "oauth_global",
    "/auth/youtube/complete": "oauth_global", "/creator/youtube-connection": "creator_global",
    "/creator/insights": "creator_global",
}

class RateLimitExceeded(Exception):
    def __init__(self, retry_after: int):
        self.retry_after = max(1, retry_after)

class RateLimitUnavailable(Exception):
    pass

@dataclass(frozen=True)
class Limit:
    bucket: str
    maximum: int
    window: int

def configured_limits():
    limits = dict(DEFAULT_LIMITS)
    try:
        overrides = json.loads(os.getenv("RATE_LIMITS_JSON", "{}"))
        if not isinstance(overrides, dict) or set(overrides) - set(limits):
            raise ValueError()
        for name, entries in overrides.items():
            if not isinstance(entries, list) or not 1 <= len(entries) <= 3:
                raise ValueError()
            for entry in entries:
                if (not isinstance(entry, list) or len(entry) != 2 or
                    any(type(value) is not int for value in entry) or
                    not 1 <= entry[0] <= 100000 or not 1 <= entry[1] <= 86400):
                    raise ValueError()
            limits[name] = entries
    except (ValueError, TypeError):
        raise RateLimitUnavailable() from None
    return limits

def public_identity(request: Request) -> str:
    """Raw ASGI peer; XFF only from explicitly trusted peers, right-to-left.

    Unverified private proxy peers share a conservative bucket until the ingress
    topology is configured. Disable Uvicorn's automatic proxy rewriting.
    """
    try:
        peer = ipaddress.ip_address(request.client.host)
        networks = [ipaddress.ip_network(value.strip()) for value in
                    os.getenv("RATE_LIMIT_TRUSTED_PROXY_CIDRS", "").split(",") if value.strip()]
        if any(network.prefixlen == 0 for network in networks):
            return "unknown-proxy"
        def trusted(ip):
            return any(ip in network for network in networks)
        if trusted(peer):
            forwarded = request.headers.get("x-forwarded-for", "")
            parts = forwarded.split(",")
            if not forwarded or len(parts) > 16:
                return "unknown-proxy"
            for part in reversed(parts):
                candidate = ipaddress.ip_address(part.strip())
                if not trusted(candidate):
                    return str(candidate)
            return "unknown-proxy"
        return "unknown-proxy" if peer.is_private or peer.is_loopback else str(peer)
    except (AttributeError, ValueError):
        return "unknown-proxy"

class PostgresRateLimiter:
    async def check(self, scope: str, identity: str):
        secret = os.getenv("RATE_LIMIT_HASH_KEY", "")
        if len(secret) < 32 or not config.SUPABASE_AUTH_DB_URL:
            raise RateLimitUnavailable()
        entries = configured_limits().get(scope)
        if not entries:
            raise RateLimitUnavailable()
        hashed = hmac.new(secret.encode(), identity.encode(), hashlib.sha256).hexdigest()
        await asyncio.to_thread(self._check, [Limit(f"{scope}:{window}:{hashed}", maximum, window)
                                            for maximum, window in entries])

    def _connect(self):
        return psycopg2.connect(config.SUPABASE_AUTH_DB_URL, connect_timeout=5,
                               options="-c statement_timeout=3000 -c lock_timeout=1000")

    def _check(self, limits: list[Limit]):
        try:
            with closing(self._connect()) as connection, connection:
                with connection.cursor() as cursor:
                    cursor.execute("select pg_advisory_xact_lock(742193018)")
                    cursor.execute("select floor(extract(epoch from clock_timestamp()))::bigint")
                    now = int(cursor.fetchone()[0])
                    cursor.execute("delete from creator.rate_limits where bucket in "
                        "(select bucket from creator.rate_limits where expires_at <= %s "
                        "order by expires_at limit 1000)", (now,))
                    cursor.execute("select count(*) from creator.rate_limits")
                    count = int(cursor.fetchone()[0])
                    for limit in limits:
                        cursor.execute("select requests, expires_at from creator.rate_limits where bucket=%s",
                                       (limit.bucket,))
                        row = cursor.fetchone()
                        end = (now + limit.window if limit.bucket.startswith("creator_sync:")
                               else (now // limit.window + 1) * limit.window)
                        if row and row[1] > now and row[0] >= limit.maximum:
                            raise RateLimitExceeded(row[1] - now)
                        if row is None:
                            if count >= 10000:
                                raise RateLimitUnavailable()
                            count += 1
                        cursor.execute("insert into creator.rate_limits (bucket, requests, expires_at) "
                            "values (%s, 1, %s) on conflict (bucket) do update set "
                            "requests=case when creator.rate_limits.expires_at <= %s then 1 "
                            "else creator.rate_limits.requests+1 end, expires_at=%s",
                            (limit.bucket, end, now, end))
        except psycopg2.Error:
            raise RateLimitUnavailable() from None

limiter = PostgresRateLimiter()

async def enforce_user_limit(scope: str, user_id: str):
    await limiter.check(scope, f"user:{user_id}")

class RateLimitMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        path = scope.get("path", "").rstrip("/")
        if scope["type"] != "http" or scope["method"] == "OPTIONS" or path not in PROTECTED_PATHS:
            return await self.app(scope, receive, send)
        from starlette.responses import JSONResponse
        try:
            await limiter.check("ingress", public_identity(Request(scope)))
            await limiter.check(PROTECTED_PATHS[path], "global")
        except RateLimitExceeded as exc:
            response = JSONResponse({"message": "Too many requests. Please try again later.",
                "code": "rate_limited"}, status_code=429,
                headers={"Retry-After": str(exc.retry_after), "Cache-Control": "no-store"})
            return await response(scope, receive, send)
        except RateLimitUnavailable:
            response = JSONResponse({"message": "This service is temporarily unavailable. Please try again later.",
                "code": "rate_limit_unavailable"}, status_code=503)
            return await response(scope, receive, send)
        return await self.app(scope, receive, send)
