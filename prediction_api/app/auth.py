from dataclasses import dataclass
from typing import Any

import httpx
from fastapi import Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.config import SUPABASE_PUBLISHABLE_KEY, SUPABASE_URL


@dataclass(frozen=True)
class AuthenticatedUser:
    id: str


class AuthenticationException(Exception):
    def __init__(self, *, status_code: int, message: str, code: str):
        super().__init__(message)
        self.status_code = status_code
        self.message = message
        self.code = code


bearer_scheme = HTTPBearer(auto_error=False)


def _authentication_required() -> AuthenticationException:
    return AuthenticationException(
        status_code=401,
        message="Authentication is required to use this endpoint.",
        code="authentication_required",
    )


def _invalid_session() -> AuthenticationException:
    return AuthenticationException(
        status_code=401,
        message="Your session is invalid or expired. Sign in again.",
        code="invalid_session",
    )


def _auth_service_unavailable() -> AuthenticationException:
    return AuthenticationException(
        status_code=503,
        message="Authentication is temporarily unavailable. Please try again.",
        code="auth_service_unavailable",
    )


async def _request_supabase_user(access_token: str) -> httpx.Response:
    if not SUPABASE_URL or not SUPABASE_PUBLISHABLE_KEY:
        raise _auth_service_unavailable()

    async with httpx.AsyncClient(timeout=5.0) as client:
        return await client.get(
            f"{SUPABASE_URL.rstrip('/')}/auth/v1/user",
            headers={
                "apikey": SUPABASE_PUBLISHABLE_KEY,
                "Authorization": f"Bearer {access_token}",
                "Accept": "application/json",
            },
        )


async def validate_access_token(access_token: str) -> AuthenticatedUser:
    try:
        response = await _request_supabase_user(access_token)
    except AuthenticationException:
        raise
    except httpx.HTTPError as exc:
        raise _auth_service_unavailable() from exc

    if response.status_code in {429} or response.status_code >= 500:
        raise _auth_service_unavailable()

    if response.status_code != 200:
        raise _invalid_session()

    try:
        payload = response.json()
    except ValueError as exc:
        raise _auth_service_unavailable() from exc

    if not isinstance(payload, dict):
        raise _auth_service_unavailable()

    user_id = payload.get("id")
    if not isinstance(user_id, str) or not user_id.strip():
        raise _auth_service_unavailable()

    if payload.get("is_anonymous") is True:
        raise _invalid_session()

    return AuthenticatedUser(id=user_id)


async def require_authenticated_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
) -> AuthenticatedUser:
    if (
        credentials is None
        or credentials.scheme.lower() != "bearer"
        or not credentials.credentials.strip()
    ):
        raise _authentication_required()

    return await validate_access_token(credentials.credentials)


async def optional_authenticated_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
) -> AuthenticatedUser | None:
    if request.headers.get("authorization") is None:
        return None
    if (
        credentials is None
        or credentials.scheme.lower() != "bearer"
        or not credentials.credentials.strip()
    ):
        raise _invalid_session()
    return await validate_access_token(credentials.credentials)
