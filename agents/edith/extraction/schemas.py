"""Pydantic schemas for extraction output validation."""

from __future__ import annotations

from datetime import datetime
from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ExtractedEntity(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1, max_length=200)
    type: Literal["person", "company", "project", "technology", "event", "place", "other"]


class ExtractedMemory(BaseModel):
    model_config = ConfigDict(extra="forbid")
    layer: Literal["episodic", "semantic"]
    kind: Literal[
        "event", "decision", "achievement", "experience", "preference", "skill",
        "interest", "goal", "relationship", "project", "profile_fact",
    ]
    content: str = Field(min_length=1, max_length=10000)
    subject: Optional[str] = Field(default=None, max_length=200)
    predicate: Optional[str] = Field(default=None, max_length=200)
    entities: List[ExtractedEntity] = Field(default_factory=list, max_length=20)
    importance: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    occurred_at: Optional[str] = None
    provenance: Literal["user_stated", "inferred", "user_confirmed"] = "user_stated"

    @field_validator("occurred_at")
    @classmethod
    def validate_occurred_at(cls, value):
        if value is None:
            return value
        try:
            datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("occurred_at must be an ISO-8601 timestamp or date") from exc
        return value
