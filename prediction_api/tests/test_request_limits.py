import asyncio
import json
from unittest.mock import Mock
import pytest
from fastapi.testclient import TestClient
from app import main
from app.request_limits import RequestLimitMiddleware
from app.schemas import ForecastRequest
from app.identifiers import parse_channel_identifier

NORMAL = {"title": "ශ්‍රී ලංකා பயணம் Travel", "category": "Travel & Events",
          "audioLanguage": "Mixed / multilingual", "durationSeconds": 510,
          "channelIdentifier": "@samplechannel", "plannedPublishDay": "Friday", "plannedPublishHour": 18}

@pytest.mark.parametrize("field,value", [
    ("durationSeconds", float("nan")), ("durationSeconds", float("inf")),
    ("durationSeconds", -float("inf")), ("durationSeconds", -1),
    ("durationSeconds", 0), ("durationSeconds", 1e100), ("title", "a"*101),
    ("title", "   "), ("category", "bad\u0000category"), ("audioLanguage", "invented"),
    ("title", "bad\u0000title"), ("title", "bad\ud800title"),
    ("plannedPublishDay", "Funday"), ("plannedPublishHour", 99),
    ("channelIdentifier", "UC-short"), ("channelIdentifier", "https://evil.invalid/@name"),
    ("channelIdentifier", "@bad/name"), ("channelIdentifier", "a"*513),
    ("unknownArray", ["a"]*100),
])
def test_invalid_input_rejected_before_providers(field, value, monkeypatch):
    provider = Mock()
    monkeypatch.setattr(main, "fetch_channel_stats", provider)
    response = TestClient(main.app).post("/forecast", content=json.dumps({**NORMAL, field: value}),
                                         headers={"Content-Type": "application/json"})
    assert response.status_code == 400
    provider.assert_not_called()

def test_normal_unicode_and_maximum_valid_duration():
    assert ForecastRequest(**NORMAL).title == NORMAL["title"]
    assert ForecastRequest(**{**NORMAL, "durationSeconds": 43200}).durationSeconds == 43200
    assert parse_channel_identifier("https://www.youtube.com/channel/UCX6OQ3DkcsbYNE6H8uQQuVA") == ("id", "UCX6OQ3DkcsbYNE6H8uQQuVA")
    assert parse_channel_identifier("https://youtube.com/@தமிழ்") == ("handle", "@தமிழ்")

def test_oversized_content_length_rejected_before_parser(monkeypatch):
    provider = Mock()
    monkeypatch.setattr(main, "fetch_channel_stats", provider)
    response = TestClient(main.app).post("/forecast", content=b"x"*65537)
    assert response.status_code == 413
    provider.assert_not_called()

def test_chunked_or_lying_content_length_cannot_bypass_size_limit():
    events, called = [], []
    async def endpoint(*_): called.append(True)
    messages = iter([{"type": "http.request", "body": b"x"*40000, "more_body": True},
                     {"type": "http.request", "body": b"x"*40000, "more_body": False}])
    async def receive(): return next(messages)
    async def send(message): events.append(message)
    asyncio.run(RequestLimitMiddleware(endpoint)({"type": "http", "method": "POST", "headers": []}, receive, send))
    assert events[0]["status"] == 413
    assert not called

def test_normal_body_at_limit_is_forwarded_exactly(monkeypatch):
    monkeypatch.setenv("MAX_REQUEST_BODY_BYTES", "1024")
    observed = []
    async def endpoint(scope, receive, send): observed.append((await receive())["body"])
    async def receive(): return {"type": "http.request", "body": b"x"*1024, "more_body": False}
    async def send(_): pass
    asyncio.run(RequestLimitMiddleware(endpoint)({"type": "http", "method": "POST", "headers": []}, receive, send))
    assert observed == [b"x"*1024]

def test_deep_json_is_rejected_without_recursion_error():
    response = TestClient(main.app).post("/forecast", content="["*1000+"0"+"]"*1000,
                                        headers={"Content-Type": "application/json"})
    assert response.status_code == 400

def test_quoted_braces_do_not_count_as_nesting():
    body = json.dumps({**NORMAL, "title": "{"*98+'"\\'}).encode()
    observed = []
    async def endpoint(scope, receive, send): observed.append((await receive())["body"])
    async def receive(): return {"type": "http.request", "body": body, "more_body": False}
    async def send(_): pass
    asyncio.run(RequestLimitMiddleware(endpoint)({"type": "http", "method": "POST", "headers": []}, receive, send))
    assert observed == [body]
