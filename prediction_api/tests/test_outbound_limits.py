import asyncio
from unittest.mock import AsyncMock, Mock, MagicMock
import httpx
import pytest
from app import outbound, service_limits, title_analysis
from app.youtube import fetch_channel_stats, ChannelLookupException
from app.youtube_oauth import exchange_authorization_code, YouTubeOAuthException
from app import config

def transport(monkeypatch, side_effect):
    client = AsyncMock()
    client.__aenter__.return_value = client
    client.get.side_effect = side_effect
    client.post.side_effect = side_effect
    factory = Mock(return_value=client)
    monkeypatch.setattr(outbound.httpx, "AsyncClient", factory)
    monkeypatch.setattr(outbound.asyncio, "sleep", AsyncMock())
    return client, factory

def test_read_transient_then_success_and_explicit_transport_bounds(monkeypatch):
    client, factory = transport(monkeypatch, [httpx.Response(503), httpx.Response(200)])
    result = asyncio.run(outbound.request("GET", "https://example.invalid"))
    assert result.status_code == 200 and client.get.await_count == 2
    outbound.asyncio.sleep.assert_awaited_once_with(0.25)
    timeout = factory.call_args.kwargs["timeout"]
    assert (timeout.connect, timeout.read, timeout.write, timeout.pool) == (5, 10, 5, 2)

@pytest.mark.parametrize("status", [400, 401, 403, 404])
def test_non_retryable_status_has_one_attempt(monkeypatch, status):
    client, _ = transport(monkeypatch, [httpx.Response(status)])
    assert asyncio.run(outbound.request("GET", "https://example.invalid")).status_code == status
    client.get.assert_awaited_once()

def test_retries_exhausted_are_bounded(monkeypatch):
    client, _ = transport(monkeypatch, [httpx.ReadTimeout("private")]*3)
    with pytest.raises(httpx.TimeoutException): asyncio.run(outbound.request("GET", "https://example.invalid"))
    assert client.get.await_count == 3
    assert outbound.asyncio.sleep.await_count == 2

def test_oauth_post_is_never_retried_or_repeated(monkeypatch):
    client, _ = transport(monkeypatch, [httpx.ReadTimeout("private-code private-secret")])
    for name in ("GOOGLE_OAUTH_CLIENT_ID", "GOOGLE_OAUTH_CLIENT_SECRET", "GOOGLE_OAUTH_REDIRECT_URI", "TOKEN_ENCRYPTION_KEY"):
        monkeypatch.setattr(config, name, "test-only")
    with pytest.raises(YouTubeOAuthException) as exc: asyncio.run(exchange_authorization_code("test-code"))
    assert "private-code" not in str(exc.value)
    client.post.assert_awaited_once()
    outbound.asyncio.sleep.assert_not_awaited()

def test_retry_after_long_delay_stops_instead_of_retrying_early(monkeypatch):
    client, _ = transport(monkeypatch, [httpx.Response(429, headers={"Retry-After": "60"})])
    assert asyncio.run(outbound.request("GET", "https://example.invalid")).status_code == 429
    client.get.assert_awaited_once()

def test_short_retry_after_is_respected(monkeypatch):
    client, _ = transport(monkeypatch, [httpx.Response(429, headers={"Retry-After": "1"}), httpx.Response(200)])
    assert asyncio.run(outbound.request("GET", "https://example.invalid")).status_code == 200
    outbound.asyncio.sleep.assert_awaited_once_with(1)

def test_whole_operation_deadline_terminates_slow_provider(monkeypatch):
    original_sleep = asyncio.sleep
    client, _ = transport(monkeypatch, [])
    async def slow(*_, **__): await original_sleep(1)
    client.get.side_effect = slow
    with pytest.raises(httpx.TimeoutException, match="External service deadline"):
        asyncio.run(outbound.request("GET", "https://example.invalid", budget=0.01))
    client.get.assert_awaited_once()

def test_data_api_transport_and_retry_count(monkeypatch):
    import app.youtube as yt
    client = MagicMock()
    execute = client.channels.return_value.list.return_value.execute
    execute.side_effect = [ConnectionError(), {"items": [{"id": "UCX6OQ3DkcsbYNE6H8uQQuVA", "snippet": {}, "statistics": {}}]}]
    builder = Mock(return_value=client)
    http_factory = Mock(return_value=Mock(timeout=8))
    monkeypatch.setattr(yt, "build", builder)
    monkeypatch.setattr(yt.httplib2, "Http", http_factory)
    monkeypatch.setattr(yt.time, "sleep", Mock())
    assert fetch_channel_stats("@creator", "test-only").channelId == "UCX6OQ3DkcsbYNE6H8uQQuVA"
    assert builder.call_args.kwargs["http"].timeout == 8
    http_factory.assert_called_once_with(timeout=8)
    http_factory.return_value.close.assert_called_once()
    assert execute.call_count == 2
    assert all(call.kwargs["num_retries"] == 0 for call in execute.call_args_list)

def test_lookup_deadline_is_safe(monkeypatch):
    # Close the submitted coroutine because this fake wait_for does not run it.
    async def timeout(coro, timeout): coro.close(); raise TimeoutError()
    monkeypatch.setattr(service_limits.asyncio, "wait_for", timeout)
    with pytest.raises(ChannelLookupException) as exc:
        asyncio.run(service_limits.bounded_channel_lookup(Mock(), "value"))
    assert exc.value.status_code == 502

def test_creator_sync_deadline_cancels_without_restarting_sync(monkeypatch):
    from app.creator_analytics import CreatorSyncException
    actual_timeout = asyncio.timeout
    configured_budgets, attempts, cancelled = [], [], []
    def short_timeout(budget):
        configured_budgets.append(budget)
        return actual_timeout(0.01)
    async def slow_sync(**kwargs):
        attempts.append(kwargs)
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.append(True)
    monkeypatch.setattr(service_limits.asyncio, "timeout", short_timeout)
    monkeypatch.setattr(service_limits, "synchronize_creator_history", slow_sync)
    with pytest.raises(CreatorSyncException, match="time budget"):
        asyncio.run(service_limits.bounded_creator_sync(user_id="test"))
    assert configured_budgets == [300] and attempts == [{"user_id": "test"}]
    assert cancelled == [True]

def test_database_timeout_options_are_bounded(monkeypatch):
    from app.database_limits import connection_options
    assert "statement_timeout=5000" in connection_options()
    monkeypatch.setenv("DB_STATEMENT_TIMEOUT_MS", "0")
    with pytest.raises(ValueError): connection_options()

def test_gemini_transport_is_closed_and_maximum_three_models(monkeypatch):
    from google import genai
    client = MagicMock()
    client.models.generate_content.return_value.text = "not-json"
    monkeypatch.setattr(genai, "Client", Mock(return_value=client))
    monkeypatch.setattr(title_analysis, "GEMINI_API_KEY", "test-only")
    monkeypatch.setattr(title_analysis, "_gemini_model_candidates", lambda: ("a", "b", "c", "d"))
    assert title_analysis.analyze_title_tone("Test") == (None, None)
    assert client.models.generate_content.call_count == 3
    client.close.assert_called_once()
    assert all(call.kwargs["config"].http_options.timeout <= 8000 for call in client.models.generate_content.call_args_list)

def test_repeated_playlist_token_cannot_loop_or_spend_unbounded_quota(monkeypatch):
    from app import creator_analytics
    get = AsyncMock(return_value={"items": [], "nextPageToken": "repeat"})
    monkeypatch.setattr(creator_analytics, "_google_get", get)
    assert asyncio.run(creator_analytics.fetch_upload_video_ids(access_token="test", uploads_playlist_id="test", limit=200)) == []
    assert get.await_count == 2
