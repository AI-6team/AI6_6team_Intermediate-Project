"""Password hashing utilities.

API authentication uses bcrypt by default and can verify legacy SHA-256 hashes
for seamless migration of existing users.
"""

from __future__ import annotations

import hashlib
import re

import bcrypt

_LEGACY_SHA256_RE = re.compile(r"^[a-fA-F0-9]{64}$")
_BCRYPT_PREFIXES = ("$2a$", "$2b$", "$2y$")


def hash_password(password: str) -> str:
    """Hash a password using bcrypt."""
    if not password:
        raise ValueError("password must not be empty")
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt(rounds=12)).decode("utf-8")


def is_bcrypt_hash(password_hash: str | None) -> bool:
    if not password_hash:
        return False
    return str(password_hash).startswith(_BCRYPT_PREFIXES)


def is_legacy_sha256_hash(password_hash: str | None) -> bool:
    if not password_hash:
        return False
    return bool(_LEGACY_SHA256_RE.fullmatch(str(password_hash)))


def verify_password(password: str, password_hash: str | None) -> bool:
    """Verify password against bcrypt or legacy SHA-256 hash."""
    if not password_hash:
        return False

    stored = str(password_hash)
    if is_bcrypt_hash(stored):
        try:
            return bcrypt.checkpw(password.encode("utf-8"), stored.encode("utf-8"))
        except ValueError:
            return False

    if is_legacy_sha256_hash(stored):
        candidate = hashlib.sha256(password.encode("utf-8")).hexdigest()
        return candidate == stored.lower()

    return False


def needs_hash_upgrade(password_hash: str | None) -> bool:
    """Return True when the stored hash is not bcrypt and should be upgraded."""
    return not is_bcrypt_hash(password_hash)

