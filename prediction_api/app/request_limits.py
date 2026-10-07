"""Bound actual bytes before JSON parsing, even without Content-Length."""
import os
import asyncio
from starlette.responses import JSONResponse

class RequestLimitMiddleware:
    def __init__(self, app): self.app = app
    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] not in {"POST", "PUT", "PATCH", "DELETE"}:
            return await self.app(scope, receive, send)
        maximum = int(os.getenv("MAX_REQUEST_BODY_BYTES", "65536"))
        if not 1024 <= maximum <= 1048576: raise RuntimeError("Invalid request size configuration")
        headers = dict(scope.get("headers", []))
        try:
            length = int(headers.get(b"content-length", b"0"))
            if length < 0: raise ValueError()
        except ValueError:
            return await JSONResponse({"message": "Invalid request.", "code": "invalid_request"},
                                      status_code=400)(scope, receive, send)
        chunks, total = [], 0
        if length > maximum:
            return await self.too_large(scope, receive, send)
        try:
            async with asyncio.timeout(10):
                while True:
                    message = await receive()
                    if message["type"] == "http.disconnect": return
                    chunk = message.get("body", b"")
                    total += len(chunk)
                    if total > maximum: return await self.too_large(scope, receive, send)
                    chunks.append(chunk)
                    if not message.get("more_body", False): break
        except TimeoutError:
            return await JSONResponse({"message": "Request timed out.", "code": "request_timeout"},
                                      status_code=408)(scope, receive, send)
        body = b"".join(chunks)
        # Bound JSON nesting before Python's JSON parser can recurse. Respect
        # escaped quotes/brackets inside strings; normal JSON is still parsed
        # by FastAPI and validated by Pydantic.
        depth, in_string, escaped = 0, False, False
        for byte in body:
            if in_string:
                if escaped: escaped = False
                elif byte == 92: escaped = True
                elif byte == 34: in_string = False
            elif byte == 34: in_string = True
            elif byte in (91, 123):
                depth += 1
                if depth > 32:
                    return await JSONResponse({"message": "Invalid request.", "code": "invalid_request"},
                                              status_code=400)(scope, receive, send)
            elif byte in (93, 125): depth -= 1
        pending = True
        async def bounded_receive():
            nonlocal pending
            if pending:
                pending = False
                return {"type": "http.request", "body": body, "more_body": False}
            return await receive()
        return await self.app(scope, bounded_receive, send)
    async def too_large(self, scope, receive, send):
        return await JSONResponse({"message": "Request is too large.", "code": "request_too_large"},
                                  status_code=413)(scope, receive, send)
