"""Encrypted browser state helpers."""

from __future__ import annotations

import base64
from hashlib import sha256
import json
from typing import Any

from cryptography.fernet import Fernet, InvalidToken


class EncryptedStateError(RuntimeError):
    """Raised when encrypted browser state cannot be decoded."""


def _fernet(secret: str) -> Fernet:
    key = base64.urlsafe_b64encode(sha256(secret.encode("utf-8")).digest())
    return Fernet(key)


def encrypt_json(payload: dict[str, Any], secret: str) -> str:
    raw = json.dumps(payload, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    return _fernet(secret).encrypt(raw).decode("ascii")


def decrypt_json(token: str, secret: str) -> dict[str, Any]:
    try:
        raw = _fernet(secret).decrypt(token.encode("ascii"))
        payload = json.loads(raw.decode("utf-8"))
    except (InvalidToken, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise EncryptedStateError("Encrypted browser state is invalid or expired.") from exc
    if not isinstance(payload, dict):
        raise EncryptedStateError("Encrypted browser state has an unexpected shape.")
    return payload
