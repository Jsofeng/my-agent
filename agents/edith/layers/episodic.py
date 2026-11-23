"""Service for immutable dated events and experiences."""

from ..db.repository import MemoryRepository
from ..models import MemoryInput


class EpisodicMemoryService:
    def __init__(self, repository: MemoryRepository):
        self.repository = repository

    def add(self, kind: str, content: str, **fields: object) -> int:
        return self.repository.create_memory(MemoryInput(layer="episodic", kind=kind,
                                                         content=content, **fields))

    def recent(self, limit: int = 20) -> list[dict[str, object]]:
        rows = self.repository.connections.get().execute(
            "SELECT * FROM memories WHERE user_id=? AND layer='episodic' AND status='active' "
            "ORDER BY occurred_at DESC,id DESC LIMIT ?", (self.repository.user_id, max(0, limit))
        ).fetchall()
        return [dict(row) for row in rows]
