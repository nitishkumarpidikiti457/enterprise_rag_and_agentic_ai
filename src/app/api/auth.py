"""JWT bearer-token auth + a simple per-user sliding-window rate limiter."""

from __future__ import annotations

import hmac
import time
from collections import defaultdict, deque
from datetime import UTC, datetime, timedelta

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.config import get_settings

_bearer = HTTPBearer(auto_error=False)


def authenticate(username: str, password: str) -> bool:
    s = get_settings()
    return hmac.compare_digest(username, s.demo_username) and hmac.compare_digest(password, s.demo_password)


def create_token(subject: str) -> tuple[str, int]:
    s = get_settings()
    expires = s.jwt_expire_minutes * 60
    payload = {"sub": subject, "iat": datetime.now(UTC),
               "exp": datetime.now(UTC) + timedelta(seconds=expires)}
    return jwt.encode(payload, s.jwt_secret, algorithm=s.jwt_algorithm), expires


def current_user(creds: HTTPAuthorizationCredentials | None = Depends(_bearer)) -> str:
    if creds is None or creds.scheme.lower() != "bearer":
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing bearer token",
                            headers={"WWW-Authenticate": "Bearer"})
    s = get_settings()
    try:
        payload = jwt.decode(creds.credentials, s.jwt_secret, algorithms=[s.jwt_algorithm])
    except jwt.ExpiredSignatureError as e:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Token expired") from e
    except jwt.InvalidTokenError as e:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid token") from e
    return payload["sub"]


_hits: dict[str, deque] = defaultdict(deque)


def rate_limited_user(user: str = Depends(current_user)) -> str:
    limit = get_settings().rate_limit_per_minute
    now = time.monotonic()
    q = _hits[user]
    while q and now - q[0] > 60:
        q.popleft()
    if len(q) >= limit:
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, "Rate limit exceeded, try again shortly")
    q.append(now)
    return user


def reset_rate_limits() -> None:
    _hits.clear()
