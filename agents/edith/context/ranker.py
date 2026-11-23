"""Configurable memory relevance ranking."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Mapping

from ..config import EdithConfig, config


def _parse_time(value):
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed
    except ValueError:
        return None


def recency_score(memory: Mapping[str, object], now: datetime) -> float:
    field = "occurred_at" if memory.get("layer") == "episodic" else "updated_at"
    timestamp = _parse_time(memory.get(field) or memory.get("created_at"))
    if timestamp is None:
        return 0.0
    age_days = max(0.0, (now - timestamp).total_seconds() / 86400.0)
    return math.exp(-math.log(2.0) * age_days / 180.0)


def rank_memory(memory: Mapping[str, object], similarity: float,
                settings: EdithConfig = config, now: datetime | None = None) -> float:
    """Combine normalized similarity, recency, importance, and use signals."""
    now = now or datetime.now(timezone.utc)
    sim = max(0.0, min(1.0, (float(similarity) + 1.0) / 2.0))
    rec = recency_score(memory, now)
    importance = max(0.0, min(1.0, float(memory.get("importance") or 0.0)))
    count = max(0, int(memory.get("access_count") or 0))
    use = min(1.0, math.log1p(count) / math.log1p(100.0))
    return (settings.weight_similarity * sim + settings.weight_recency * rec
            + settings.weight_importance * importance + settings.weight_use * use)
