"""Service for stable, current facts about the user."""

from ..db.repository import MemoryRepository
from ..models import MemoryInput


class SemanticMemoryService:
    def __init__(self, repository: MemoryRepository):
        self.repository = repository

    def add(self, kind: str, content: str, **fields: object) -> int:
        return self.repository.create_memory(MemoryInput(layer="semantic", kind=kind,
                                                         content=content, **fields))

    def profile(self) -> list[dict[str, object]]:
        return self.repository.list_memories(layer="semantic", status="active")
