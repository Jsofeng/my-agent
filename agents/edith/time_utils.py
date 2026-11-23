"""Timestamp normalization for stored memory events."""

from __future__ import annotations

from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from typing import Optional


def normalize_utc_timestamp(value: Optional[str], user_timezone: str) -> Optional[str]:
    """Normalize ISO datetimes to UTC; retain date-only values as calendar dates."""
    if value is None:
        return None
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if len(value) == 10 and parsed.time().isoformat() == "00:00:00":
        return parsed.date().isoformat()
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=ZoneInfo(user_timezone))
    return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
