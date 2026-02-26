"""JWT token helpers for API authentication."""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Tuple

import jwt
from jwt import ExpiredSignatureError, InvalidTokenError

_LOGGER = logging.getLogger(__name__)
_DEV_FALLBACK_SECRET = "bidflow-dev-secret-change-me"
_warned_missing_secret = False


def _bool_env(name: str, default: bool = False) -> bool:
    raw = os.getenv(name, "1" if default else "0").strip().lower()
    return raw in {"1", "true", "yes", "y", "on"}


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def _get_jwt_secret() -> str:
    global _warned_missing_secret

    secret = os.getenv("BIDFLOW_JWT_SECRET", "").strip()
    if secret:
        return secret

    cookie_key = os.getenv("BIDFLOW_COOKIE_KEY", "").strip()
    if cookie_key:
        return cookie_key

    if not _warned_missing_secret:
        _LOGGER.warning(
            "BIDFLOW_JWT_SECRET is not configured. Falling back to a development-only secret."
        )
        _warned_missing_secret = True
    return _DEV_FALLBACK_SECRET


def _get_algorithm() -> str:
    return (os.getenv("BIDFLOW_JWT_ALGORITHM", "HS256") or "HS256").strip()


def get_access_token_ttl_seconds() -> int:
    return _int_env("BIDFLOW_JWT_EXPIRE_MINUTES", 480) * 60


def create_access_token(username: str, role: str) -> Tuple[str, int]:
    ttl_seconds = get_access_token_ttl_seconds()
    now = datetime.now(timezone.utc)
    exp = now + timedelta(seconds=ttl_seconds)

    payload: Dict[str, Any] = {
        "sub": username,
        "role": role or "member",
        "type": "access",
        "iat": int(now.timestamp()),
        "exp": int(exp.timestamp()),
    }
    token = jwt.encode(payload, _get_jwt_secret(), algorithm=_get_algorithm())
    return token, ttl_seconds


def decode_access_token(token: str) -> Dict[str, str]:
    token = (token or "").strip()
    if not token:
        raise ValueError("Missing access token")

    allow_legacy = _bool_env("BIDFLOW_ALLOW_LEGACY_TOKEN", default=False)
    if allow_legacy and token.startswith("user:"):
        username = token.split(":", 1)[1].strip()
        if not username:
            raise ValueError("Invalid token structure")
        return {"sub": username, "role": "member"}

    try:
        payload = jwt.decode(token, _get_jwt_secret(), algorithms=[_get_algorithm()])
    except ExpiredSignatureError as exc:
        raise ValueError("Token expired") from exc
    except InvalidTokenError as exc:
        raise ValueError("Invalid token") from exc

    token_type = str(payload.get("type", ""))
    if token_type != "access":
        raise ValueError("Invalid token type")

    username = str(payload.get("sub", "")).strip()
    if not username:
        raise ValueError("Token subject is missing")

    role = str(payload.get("role", "member") or "member").strip() or "member"
    return {"sub": username, "role": role}

