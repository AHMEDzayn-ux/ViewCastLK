"""A failed chunk must cost one chunk, not the whole run.

On 21 September 2026 a scheduled run died partway through snapshotting 93,710
videos. One videos.list call of fifty ids came back 403 with reason
"forbidden", blaming a myRating parameter the request never sent. Two things
turned that into a total loss:

* googleapiclient retries a 403 only when the reason is userRateLimitExceeded
  or rateLimitExceeded, so num_retries did nothing and the first attempt was
  the last;
* get_video_details gathered every chunk into one list and the caller wrote
  afterwards, so the single failure discarded roughly 1,875 chunks and the
  quota already spent on them.

These tests pin the behaviour that replaced it. They make no network calls.
"""

import os
import sys
import types
from pathlib import Path

import pytest

os.environ.setdefault("YOUTUBE_API_KEY", "offline-test-key")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import googleapiclient.discovery as discovery  # noqa: E402
from googleapiclient.errors import HttpError  # noqa: E402

discovery.build = lambda *args, **kwargs: types.SimpleNamespace()

import youtube_client as yc  # noqa: E402


class FakeResponse:
    def __init__(self, status):
        self.status = status
        self.reason = "Forbidden"


def http_error(status: int, reason: str) -> HttpError:
    """An HttpError shaped like a real one.

    error.message has to be present. Without it HttpError renders the short
    form and the reason never reaches str(e), which is what the quota check
    reads.
    """
    content = (
        '{"error": {"code": %d, "message": "%s happened",'
        ' "errors": [{"message": "%s happened", "reason": "%s"}]}}'
        % (status, reason, reason, reason)
    ).encode()
    return HttpError(FakeResponse(status), content)


class FakeRequest:
    def __init__(self, outcomes):
        self.outcomes = outcomes

    def execute(self, num_retries=None):
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class FakeVideos:
    """One entry per expected chunk, each a list of outcomes per attempt."""

    def __init__(self, script):
        self.script = script
        self.calls = 0

    def list(self, **kwargs):
        self.calls += 1
        return FakeRequest(self.script.pop(0))


def install(monkeypatch, script):
    videos = FakeVideos(script)
    monkeypatch.setattr(yc, "youtube", types.SimpleNamespace(videos=lambda: videos))
    monkeypatch.setattr(yc.time, "sleep", lambda _seconds: None)
    return videos


def items(count, start=0):
    return {"items": [{"id": f"v{start + i}"} for i in range(count)]}


def ids(count):
    return [f"id{i}" for i in range(count)]


def test_a_transient_403_is_retried_rather_than_ending_the_run(monkeypatch):
    install(monkeypatch, [[
        http_error(403, "forbidden"),
        http_error(403, "forbidden"),
        items(2),
    ]])
    assert list(yc.iter_video_details(ids(2))) == [[{"id": "v0"}, {"id": "v1"}]]


def test_a_chunk_that_never_recovers_is_skipped_and_the_rest_continue(monkeypatch):
    videos = install(monkeypatch, [
        [http_error(403, "forbidden")] * yc.TRANSIENT_403_ATTEMPTS,
        [items(3, start=10)],
    ])
    collected = list(yc.iter_video_details(ids(100)))
    assert collected == [[{"id": "v10"}, {"id": "v11"}, {"id": "v12"}]]
    assert videos.calls == 2, "the second chunk was never attempted"


def test_quota_exhaustion_stays_terminal(monkeypatch):
    videos = install(monkeypatch, [
        [http_error(403, "quotaExceeded")],
        [items(1)],
    ])
    with pytest.raises(HttpError, match="quotaExceeded"):
        list(yc.iter_video_details(ids(100)))
    assert videos.calls == 1, "kept spending after the allowance was gone"


def test_a_non_403_is_never_skipped(monkeypatch):
    # A 400 means the request itself is wrong. Skipping it would repeat for
    # every chunk and the run would report success having collected nothing.
    install(monkeypatch, [[http_error(400, "badRequest")]])
    with pytest.raises(HttpError) as raised:
        list(yc.iter_video_details(ids(1)))
    assert raised.value.resp.status == 400


def test_a_flood_of_403s_stops_the_run(monkeypatch):
    videos = install(monkeypatch, [
        [http_error(403, "forbidden")] * yc.TRANSIENT_403_ATTEMPTS
        for _ in range(30)
    ])
    with pytest.raises(RuntimeError, match="no longer transient"):
        list(yc.iter_video_details(ids(1500), max_skipped_chunks=5))
    assert videos.calls == 6, "should stop one chunk past the cap"


def test_get_video_details_still_returns_one_flat_list(monkeypatch):
    install(monkeypatch, [[items(2)], [items(1, start=5)]])
    details = yc.get_video_details(ids(60))
    assert [video["id"] for video in details] == ["v0", "v1", "v5"]
