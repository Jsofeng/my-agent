"""Exact and cosine near-duplicate checks for a single memory layer."""

from __future__ import annotations

import hashlib
import numpy as np

from ..db.repository import MemoryRepository, utc_now


class MemoryDeduplicator:
    def __init__(self, repository: MemoryRepository, threshold: float = 0.92):
        self.repository = repository
        self.threshold = threshold

    def find_duplicate(self, *, layer: str, kind: str, subject: str | None,
                       content: str, embedding: np.ndarray | None = None) -> dict | None:
        digest = hashlib.sha256(content.strip().encode("utf-8")).hexdigest()
        rows = self.repository.connections.get().execute(
            "SELECT * FROM memories WHERE user_id=? AND layer=? AND kind=? AND subject IS ? "
            "AND status='active' ORDER BY id",
            (self.repository.user_id, layer, kind, subject),
        ).fetchall()
        for row in rows:
            item = dict(row)
            if item["content_hash"] == digest:
                return item
            if embedding is not None and item.get("embedding") is not None:
                other = np.frombuffer(item["embedding"], dtype=np.float32)
                if other.size == embedding.size:
                    similarity = float(np.dot(embedding, other) /
                                       max(np.linalg.norm(embedding) * np.linalg.norm(other), 1e-12))
                    if similarity >= self.threshold:
                        return item
        return None

    def reinforce(self, memory_id: int, importance: float, confidence: float,
                  source_conversation_id: int | None = None,
                  source_message_ids: str | None = None) -> None:
        with self.repository._write_lock:
            db = self.repository.connections.get()
            db.execute(
                "UPDATE memories SET importance=?,confidence=?,updated_at=?, "
                "source_conversation_id=COALESCE(?,source_conversation_id), "
                "source_message_ids=CASE WHEN ? IS NULL OR ?='[]' THEN source_message_ids "
                "WHEN source_message_ids='[]' THEN ? ELSE source_message_ids END "
                "WHERE id=? AND user_id=?",
                (importance, confidence, utc_now(), source_conversation_id,
                 source_message_ids, source_message_ids, source_message_ids,
                 memory_id, self.repository.user_id),
            )
            self.repository._event(memory_id, "reinforce")
            db.commit()
