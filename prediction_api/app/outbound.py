"""Finite transport deadlines and retries for idempotent HTTP reads only."""
import asyncio
import time
from email.utils import parsedate_to_datetime
import httpx

TRANSIENT_STATUSES = {408, 429, 500, 502, 503, 504}

def retry_delay(attempt, response=None):
    delay = min(2.0, 0.25 * 2**attempt)
    if response is not None:
        raw = response.headers.get("Retry-After", "")
        try:
            requested = float(raw)
        except ValueError:
            try: requested = parsedate_to_datetime(raw).timestamp() - time.time()
            except (ValueError, TypeError, OverflowError): requested = 0
        # Never retry earlier than a provider's long Retry-After. End this
        # request instead and let the caller return its normal safe failure.
        if requested > 2: return None
        delay = max(delay, max(0, requested))
    return delay

async def request(method, url, *, timeout=10.0, budget=20.0, **kwargs):
    attempts = 3 if method.upper() == "GET" else 1
    try:
        async with asyncio.timeout(budget):
            async with httpx.AsyncClient(timeout=httpx.Timeout(timeout, connect=min(5, timeout),
                                                               write=min(5, timeout), pool=2)) as client:
                for attempt in range(attempts):
                    response = None
                    try:
                        response = await getattr(client, method.lower())(url, **kwargs)
                        if response.status_code not in TRANSIENT_STATUSES or attempt == attempts-1:
                            return response
                    except httpx.TransportError:
                        if attempt == attempts-1: raise
                    delay = retry_delay(attempt, response)
                    if delay is None: return response
                    await asyncio.sleep(delay)
    except TimeoutError:
        raise httpx.TimeoutException("External service deadline exceeded") from None
