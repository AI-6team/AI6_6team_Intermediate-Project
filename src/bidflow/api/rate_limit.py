"""Simple in-memory API rate limiting middleware."""

from __future__ import annotations

import os
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from threading import Lock
from typing import DefaultDict, Deque, Tuple

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware


@dataclass(frozen=True)
class _LimitRule:
    requests: int
    window_seconds: int
    scope: str


def _bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name, "1" if default else "0").strip().lower()
    return raw in {"1", "true", "yes", "y", "on"}


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


class _InMemoryLimiter:
    def __init__(self) -> None:
        self._events: DefaultDict[str, Deque[float]] = defaultdict(deque)
        self._lock = Lock()

    def allow(self, key: str, limit: int, window_seconds: int) -> Tuple[bool, int]:
        now = time.monotonic()
        with self._lock:
            q = self._events[key]
            cutoff = now - window_seconds
            while q and q[0] <= cutoff:
                q.popleft()

            if len(q) >= limit:
                retry_after = max(1, int(window_seconds - (now - q[0])))
                return False, retry_after

            q.append(now)
            return True, 0


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate-limit API requests by client IP."""

    def __init__(self, app):
        super().__init__(app)
        self.enabled = _bool_env("BIDFLOW_RATE_LIMIT_ENABLED", default=True)
        self.auth_rule = _LimitRule(
            requests=_int_env("BIDFLOW_RATE_LIMIT_AUTH_REQUESTS", 20),
            window_seconds=_int_env("BIDFLOW_RATE_LIMIT_AUTH_WINDOW_SECONDS", 60),
            scope="auth",
        )
        self.general_rule = _LimitRule(
            requests=_int_env("BIDFLOW_RATE_LIMIT_REQUESTS", 240),
            window_seconds=_int_env("BIDFLOW_RATE_LIMIT_WINDOW_SECONDS", 60),
            scope="general",
        )
        self.limiter = _InMemoryLimiter()

    @staticmethod
    def _is_exempt_path(path: str) -> bool:
        if path in {"/", "/health", "/openapi.json", "/docs", "/redoc"}:
            return True
        return path.startswith(("/docs", "/redoc", "/openapi.json"))

    @staticmethod
    def _client_key(request: Request) -> str:
        xff = request.headers.get("x-forwarded-for", "")
        if xff:
            return xff.split(",")[0].strip() or "unknown"
        if request.client and request.client.host:
            return request.client.host
        return "unknown"

    async def dispatch(self, request: Request, call_next):
        if not self.enabled or request.method.upper() == "OPTIONS":
            return await call_next(request)

        path = request.url.path
        if self._is_exempt_path(path):
            return await call_next(request)

        rule = self.auth_rule if path.startswith("/auth/") else self.general_rule
        key = f"{self._client_key(request)}:{rule.scope}"
        allowed, retry_after = self.limiter.allow(key, rule.requests, rule.window_seconds)

        if not allowed:
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded"},
                headers={"Retry-After": str(retry_after)},
            )

        return await call_next(request)

