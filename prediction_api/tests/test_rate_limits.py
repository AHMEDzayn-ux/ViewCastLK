import asyncio
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import AsyncMock, MagicMock
import pytest
from fastapi.testclient import TestClient
from starlette.requests import Request
from app import main, rate_limits as rl
from app.auth import AuthenticatedUser, require_authenticated_user
from app.creator_lifecycle import refresh_creator_connection

class Cursor:
    def __init__(self, db, now):
        self.cursor, self.now = db.cursor(), now
    def __enter__(self): return self
    def __exit__(self, *_): self.cursor.close()
    def execute(self, sql, params=()):
        if "pg_advisory_xact_lock" in sql:
            return self.cursor.execute("BEGIN IMMEDIATE")
        if "extract(epoch" in sql:
            return self.cursor.execute("SELECT ?", (self.now[0],))
        return self.cursor.execute(sql.replace("%s", "?").replace("creator.rate_limits.", "rate_limits."), params)
    def fetchone(self): return self.cursor.fetchone()

class Connection:
    def __init__(self, path, now):
        self.db = sqlite3.connect(":memory:", timeout=5)
        self.db.execute("ATTACH DATABASE ? AS creator", (str(path),))
        self.now = now
    def __enter__(self): return self
    def __exit__(self, exc, *_):
        self.db.rollback() if exc else self.db.commit()
        self.db.close()
    def cursor(self): return Cursor(self.db, self.now)
    def close(self): self.db.close()

@pytest.fixture
def store(tmp_path, monkeypatch):
    path, now = tmp_path / "rate.db", [1000]
    with Connection(path, now) as conn:
        conn.db.execute("CREATE TABLE creator.rate_limits(bucket TEXT PRIMARY KEY, requests INTEGER, expires_at INTEGER)")
    instance = rl.PostgresRateLimiter()
    monkeypatch.setattr(instance, "_connect", lambda: Connection(path, now))
    return instance, path, now

def test_sql_limits_boundary_expiry_cleanup_and_identity_isolation(store):
    limiter, path, now = store
    quota = [rl.Limit("client-a", 2, 60)]
    limiter._check(quota); limiter._check(quota)
    with pytest.raises(rl.RateLimitExceeded) as exc: limiter._check(quota)
    assert exc.value.retry_after == 20
    limiter._check([rl.Limit("client-b", 2, 60)])
    now[0] = 1020
    limiter._check(quota)
    with Connection(path, now) as db:
        assert db.db.execute("SELECT requests FROM creator.rate_limits WHERE bucket='client-a'").fetchone()[0] == 1
        assert db.db.execute("SELECT count(*) FROM creator.rate_limits").fetchone()[0] == 1

def test_multiple_instances_concurrent_admission_share_database(store, monkeypatch):
    limiter, path, now = store
    second = rl.PostgresRateLimiter()
    monkeypatch.setattr(second, "_connect", lambda: Connection(path, now))
    def attempt(index):
        try:
            [limiter, second][index % 2]._check([rl.Limit("shared", 5, 60)])
            return True
        except rl.RateLimitExceeded: return False
    with ThreadPoolExecutor(max_workers=8) as pool:
        assert sum(pool.map(attempt, range(20))) == 5

def test_capacity_is_bounded_and_rejection_rolls_back(store):
    limiter, path, now = store
    with Connection(path, now) as db:
        db.db.executemany("INSERT INTO creator.rate_limits VALUES (?, 1, 2000)", [(str(i),) for i in range(10000)])
    with pytest.raises(rl.RateLimitUnavailable): limiter._check([rl.Limit("new", 1, 60)])
    with Connection(path, now) as db:
        assert db.db.execute("SELECT count(*) FROM creator.rate_limits").fetchone()[0] == 10000

def test_creator_cooldown_does_not_reset_at_wall_clock_boundary(store):
    limiter, _, now = store
    now[0] = 1799
    quota = [rl.Limit("creator_sync:user", 1, 1800)]
    limiter._check(quota)
    now[0] = 1801
    with pytest.raises(rl.RateLimitExceeded): limiter._check(quota)
    now[0] = 3599
    limiter._check(quota)

def request(peer, forwarded=None):
    return Request({"type": "http", "client": (peer, 1), "headers":
                    [] if forwarded is None else [(b"x-forwarded-for", forwarded.encode())]})

def test_public_ip_ignores_spoofed_forwarding_and_trusts_nearest_untrusted_hop(monkeypatch):
    monkeypatch.delenv("RATE_LIMIT_TRUSTED_PROXY_CIDRS", raising=False)
    assert rl.public_identity(request("8.8.8.8", "1.1.1.1")) == "8.8.8.8"
    assert rl.public_identity(request("1.1.1.1")) != rl.public_identity(request("8.8.8.8"))
    assert rl.public_identity(request("127.0.0.1", "spoof")) == "unknown-proxy"
    monkeypatch.setenv("RATE_LIMIT_TRUSTED_PROXY_CIDRS", "10.0.0.0/8")
    assert rl.public_identity(request("10.0.0.2", "spoofed, 8.8.8.8, 10.0.0.3")) == "8.8.8.8"
    assert rl.public_identity(request("10.0.0.2", "bad")) == "unknown-proxy"
    monkeypatch.setenv("RATE_LIMIT_TRUSTED_PROXY_CIDRS", "0.0.0.0/0")
    assert rl.public_identity(request("8.8.8.8", "1.1.1.1")) == "unknown-proxy"

def test_hashing_and_bad_configuration_fail_closed(monkeypatch):
    instance = rl.PostgresRateLimiter()
    monkeypatch.setattr(rl.config, "SUPABASE_AUTH_DB_URL", "test-only")
    monkeypatch.setenv("RATE_LIMIT_HASH_KEY", "a"*32)
    captured = []
    monkeypatch.setattr(instance, "_check", captured.extend)
    asyncio.run(instance.check("forecast_guest", "8.8.8.8"))
    assert all("8.8.8.8" not in q.bucket for q in captured)
    monkeypatch.setenv("RATE_LIMITS_JSON", '{"unknown": [[1,60]]}')
    with pytest.raises(rl.RateLimitUnavailable): asyncio.run(instance.check("forecast_guest", "8.8.8.8"))
    monkeypatch.delenv("RATE_LIMIT_HASH_KEY")
    with pytest.raises(rl.RateLimitUnavailable): asyncio.run(instance.check("forecast_guest", "8.8.8.8"))

@pytest.mark.parametrize("exceeded", [False, True])
def test_database_connection_closes_after_admission_or_rejection(monkeypatch, exceeded):
    instance = rl.PostgresRateLimiter()
    connection = MagicMock()
    connection.__enter__.return_value = connection
    cursor = connection.cursor.return_value.__enter__.return_value
    cursor.fetchone.side_effect = [(1000,), (1,), (1, 1020) if exceeded else None]
    monkeypatch.setattr(instance, "_connect", lambda: connection)
    if exceeded:
        with pytest.raises(rl.RateLimitExceeded):
            instance._check([rl.Limit("client", 1, 60)])
    else:
        instance._check([rl.Limit("client", 1, 60)])
    connection.close.assert_called_once()
    assert connection.__exit__.call_args.args[0] is (rl.RateLimitExceeded if exceeded else None)

def test_http_429_before_provider_call_and_safe_retry_header(monkeypatch):
    check = AsyncMock(side_effect=rl.RateLimitExceeded(30))
    monkeypatch.setattr(rl.limiter, "check", check)
    provider = AsyncMock()
    monkeypatch.setattr(main, "analyze_title_tone", provider)
    response = TestClient(main.app).post("/forecast", json={})
    assert response.status_code == 429
    assert response.headers["retry-after"] == "30"
    assert response.json()["code"] == "rate_limited"
    provider.assert_not_called()

def test_http_store_failure_fails_closed_health_stays_available(monkeypatch):
    monkeypatch.setattr(rl.limiter, "check", AsyncMock(side_effect=rl.RateLimitUnavailable()))
    client = TestClient(main.app)
    assert client.get("/auth/youtube/start").status_code == 503
    assert client.get("/health").status_code == 200

def test_authenticated_user_identity_and_oauth_cooldown(monkeypatch):
    calls = []
    async def check(scope, identity):
        calls.append((scope, identity))
        if scope == "oauth_start" and identity == "user:limited": raise rl.RateLimitExceeded(10)
    monkeypatch.setattr(rl.limiter, "check", check)
    main.app.dependency_overrides[require_authenticated_user] = lambda: AuthenticatedUser("limited", "sid")
    try:
        assert TestClient(main.app).get("/auth/youtube/start").status_code == 429
        main.app.dependency_overrides[require_authenticated_user] = lambda: AuthenticatedUser("other")
        assert TestClient(main.app).get("/auth/youtube/start").status_code == 401  # reaches session validation
        assert ("oauth_start", "user:other") in calls
    finally: main.app.dependency_overrides.pop(require_authenticated_user, None)

def test_worker_cooldown_before_refresh_or_sync(monkeypatch):
    monkeypatch.setattr(rl.limiter, "check", AsyncMock(side_effect=rl.RateLimitExceeded(10)))
    store = AsyncMock()
    assert asyncio.run(refresh_creator_connection(connection={"user_id": "user"}, store=store,
        model_registry=None, roster_store=None)) == "cooldown"
    store.record_refresh_success.assert_not_awaited()
