import logging
from app.log_safety import redact, protect_sensitive_logs
from app.creator_lifecycle import _report

def test_redacts_sensitive_fields_urls_bearers_and_database_credentials(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "configured-test-key")
    text = redact("https://api.invalid/callback?code=private-code&state=private-state "
        "Authorization: Bearer private-bearer refresh_token=private-refresh "
        "{'bindingNonce': 'private-nonce', 'client_secret': 'private-client-secret'} "
        "postgresql://user:private-password@database/db configured-test-key")
    for secret in ["private-code", "private-state", "private-bearer", "private-refresh",
                   "private-nonce", "private-client-secret", "private-password", "configured-test-key"]:
        assert secret not in text
    assert "/callback" in text

def test_all_child_loggers_redact_formatted_arguments_and_exception_details(caplog):
    protect_sensitive_logs()
    logger = logging.getLogger("httpx.security-test")
    with caplog.at_level(logging.INFO):
        logger.info("Request %s status=%s", "https://api.invalid/token?key=private-key", 429)
        try: raise RuntimeError("unlabeled-private-refresh")
        except RuntimeError: logger.exception("Provider failure endpoint=/forecast")
    assert "private-key" not in caplog.text
    assert "unlabeled-private-refresh" not in caplog.text
    assert "429" in caplog.text and "RuntimeError" in caplog.text and "/forecast" in caplog.text

def test_worker_never_prints_arbitrary_exception_details(capsys):
    _report("test-user", "token refresh", RuntimeError("unlabeled-private-code private-password"))
    text = capsys.readouterr().err
    assert "private-code" not in text and "private-password" not in text
    assert "token refresh" in text and "RuntimeError" in text

def test_uvicorn_access_formatter_keeps_endpoint_status_and_argument_contract():
    from uvicorn.logging import AccessFormatter
    protect_sensitive_logs()
    record = logging.getLogger("uvicorn.access").makeRecord(
        "uvicorn.access", logging.INFO, __file__, 1, '%s - "%s %s HTTP/%s" %d',
        ("127.0.0.1:8000", "GET", "/auth/youtube/callback?code=private-code&state=private-state", "1.1", 200), None)
    formatted = AccessFormatter('%(request_line)s %(status_code)s', use_colors=False).format(record)
    assert "GET /auth/youtube/callback HTTP/1.1" in formatted and "200" in formatted
    assert "private-code" not in formatted and "private-state" not in formatted
