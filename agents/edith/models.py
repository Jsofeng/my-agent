"""Small typed records shared by Edith's persistence layer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


LAYERS = {"episodic", "semantic"}
KINDS = {
    "event", "decision", "achievement", "experience", "preference", "skill",
    "interest", "goal", "relationship", "project", "profile_fact",
}
PROVENANCE = {"user_stated", "inferred", "user_confirmed"}
MEMORY_STATUSES = {"active", "superseded", "archived", "deleted"}
ENTITY_TYPES = {"person", "company", "project", "technology", "event", "place", "other"}


@dataclass(frozen=True)
class MemoryInput:
    layer: str
    kind: str
    content: str
    subject: Optional[str] = None
    predicate: Optional[str] = None
    importance: float = 0.5
    confidence: float = 0.7
    provenance: str = "user_stated"
    occurred_at: Optional[str] = None
    valid_from: Optional[str] = None
    source_conversation_id: Optional[int] = None
    source_message_ids: str = "[]"
    embedding: Optional[bytes] = None
    embedding_model: Optional[str] = None


@dataclass(frozen=True)
class SecretScanResult:
    safe_text: str
    matches: tuple[str, ...]

    @property
    def has_secrets(self) -> bool:
        return bool(self.matches)
