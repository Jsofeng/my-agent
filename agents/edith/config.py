"""Central configuration for Project Edith."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class EdithConfig:
    enabled: bool = False
    database_path: Path = Path("data/edith.db")
    index_path: Path = Path("data/edith.faiss")
    user_id: str = "default"
    user_timezone: str = "America/Toronto"
    min_similarity: float = 0.35
    dup_threshold: float = 0.92
    min_importance: float = 0.4
    min_confidence: float = 0.6
    resolve_threshold: float = 0.6
    ambiguity_margin: float = 0.15
    context_token_budget: int = 600
    top_k: int = 8
    weight_similarity: float = 0.55
    weight_recency: float = 0.20
    weight_importance: float = 0.15
    weight_use: float = 0.10
    extraction_idle_minutes: int = 10
    extraction_batch_messages: int = 8
    pending_confirmation_ttl_turns: int = 4
    conversation_idle_hours: int = 6
    busy_timeout_ms: int = 5000

    @classmethod
    def from_env(cls) -> "EdithConfig":
        """Load settings from EDITH_* environment variables."""
        defaults = cls()
        values: dict[str, object] = {}
        for name, default in defaults.__dict__.items():
            env_name = "EDITH_" + name.upper()
            raw = os.getenv(env_name)
            if raw is None:
                continue
            if isinstance(default, bool):
                values[name] = raw.strip().lower() in {"1", "true", "yes", "on"}
            elif isinstance(default, Path):
                values[name] = Path(raw)
            elif isinstance(default, int):
                values[name] = int(raw)
            elif isinstance(default, float):
                values[name] = float(raw)
            else:
                values[name] = raw
        return cls(**values)


config = EdithConfig.from_env()
