"""Redact formatted records centrally, including third-party child loggers."""
import logging
import os
import re

_KEY_VALUE = re.compile(
    r'''(?ix)(?<![A-Za-z0-9_])(["']?(?:authorization|access_token|refresh_token|client_secret|code|state|
        bindingNonce|binding_nonce|token_encryption_key|api_key|apikey|password|
        supabase_auth_db_url|supabase_warehouse_db_url|rate_limit_hash_key)
        ["']?\s*[:=]\s*)(?:"[^"]*"|'[^']*'|[^\s,;&}\]]+)''')
_BEARER = re.compile(r"(?i)\bBearer\s+[^\s,;\"']+")
_JWT = re.compile(r"\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\b")
_DATABASE = re.compile(r"(?i)postgres(?:ql)?://[^\s\"']+")
_QUERY_URL = re.compile(r"(https?://[^\s\"'?#]+|/[A-Za-z0-9_./-]+)\?[^\s\"']+")

def redact(value):
    text = str(value)
    # Current configured secrets can appear without a key/label in exceptions.
    for name, secret in os.environ.items():
        if len(secret) >= 8 and any(part in name.upper() for part in ("SECRET", "TOKEN", "PASSWORD", "KEY", "DB_URL")):
            text = text.replace(secret, "[REDACTED]")
    text = _DATABASE.sub("[REDACTED DATABASE URL]", text)
    text = _QUERY_URL.sub(r"\1?[REDACTED]", text)
    text = _BEARER.sub("Bearer [REDACTED]", text)
    text = _JWT.sub("[REDACTED JWT]", text)
    return _KEY_VALUE.sub(lambda match: match[1] + "[REDACTED]", text)

def protect_sensitive_logs():
    previous = logging.getLogRecordFactory()
    if getattr(previous, "_viewcastlk_redaction", False): return
    def factory(*args, **kwargs):
        record = previous(*args, **kwargs)
        access_record = record.name == "uvicorn.access" and isinstance(record.args, tuple) and len(record.args) == 5
        if access_record:
            items = list(record.args)
            if isinstance(items[2], str): items[2] = items[2].split("?", 1)[0]
            # Uvicorn's AccessFormatter unpacks these five arguments itself.
            record.args = tuple(redact(item) if isinstance(item, str) else item for item in items)
            record.msg = redact(record.msg)
        else:
            try: record.msg = redact(record.getMessage())
            except Exception: record.msg = "Log details unavailable"
            record.args = ()
        if record.exc_info:
            record.exc_text = f"{record.exc_info[0].__name__}: details redacted"
            record.exc_info = None
        if record.stack_info: record.stack_info = redact(record.stack_info)
        return record
    factory._viewcastlk_redaction = True
    logging.setLogRecordFactory(factory)
