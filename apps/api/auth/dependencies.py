"""Optional Bearer authentication & request context helpers for GigBid API."""

from __future__ import annotations

import os
import secrets

from fastapi import Header, HTTPException, status


def verify_optional_api_token(
    authorization: str | None = Header(default=None),
) -> str:
    """Validate Bearer token when `GIGBID_REQUIRE_API_AUTH=true`; permissive in local dev."""
    require_auth = os.getenv("GIGBID_REQUIRE_API_AUTH", "false").lower() in {"1", "true", "yes"}
    expected = os.getenv("GIGBID_API_TOKEN") or os.getenv(
        "GIGSIM_API_TOKEN", "gigsim-dev-token-2026"
    )
    if not require_auth:
        return "dev-user"

    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization Bearer token.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = authorization[7:].strip()
    if not secrets.compare_digest(token, expected):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API Bearer token.",
        )
    return token


__all__ = ["verify_optional_api_token"]
