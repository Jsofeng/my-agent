"""Keep the disposable FAISS index synchronized from SQLite embeddings."""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Optional

import numpy as np

from ..db.repository import MemoryRepository, utc_now
from ..config import config
from ..embeddings.base import Embedder
from .base import VectorStore
from .faiss_store import FaissVectorStore


class VectorSynchronizer:
    def __init__(self, repository: MemoryRepository, embedder: Embedder,
                 store: Optional[VectorStore] = None,
                 index_path: Path | str = "data/edith.faiss"):
        self.repository = repository
        self.embedder = embedder
        self.store = store or FaissVectorStore(embedder.dimension)
        self.index_path = Path(index_path)

    def index_memory(self, memory_id: int) -> None:
        row = self.repository.get_memory(memory_id)
        if row is None or row["status"] != "active":
            raise ValueError("active memory does not exist")
        vector = self._stored_vector(row)
        if vector is None or row["embedding_model"] != self.embedder.model_name:
            vector = self.embedder.embed([row["content"]])[0]
        vector = self._normalized(vector)
        # Persist the vector and pending state before updating the derived index.
        db = self.repository.connections.get()
        with self.repository._write_lock:
            db.execute("UPDATE memories SET embedding=?,embedding_model=?,index_status='pending' WHERE id=?",
                       (vector.tobytes(), self.embedder.model_name, int(memory_id)))
            db.commit()
        self.store.add(memory_id, vector)
        self.store.save(self.index_path)
        with self.repository._write_lock:
            db.execute("UPDATE memories SET index_status='indexed' WHERE id=?", (int(memory_id),))
            db.commit()
            self._update_index_state(db)

    def delete_memory(self, memory_id: int, hard: bool = True) -> None:
        self.repository.delete_memory(memory_id, hard=hard)
        self.store.remove(memory_id)
        self.store.save(self.index_path)
        with self.repository._write_lock:
            self._update_index_state(self.repository.connections.get())

    def remove_from_index(self, memory_id: int) -> None:
        """Remove a superseded or archived ID from the derived active-memory index."""
        self.store.remove(int(memory_id))
        self.store.save(self.index_path)
        with self.repository._write_lock:
            self._update_index_state(self.repository.connections.get())

    def reconcile(self) -> dict[str, int | float | bool]:
        started = time.perf_counter()
        rebuilt = False
        try:
            state = self._index_state()
            if not self.index_path.is_file():
                rebuilt = True
            else:
                try:
                    if state and (state["embedding_model"] != self.embedder.model_name or
                                  state["dim"] != self.embedder.dimension):
                        rebuilt = True
                    else:
                        self.store.load(self.index_path)
                except Exception:
                    rebuilt = True
            if rebuilt:
                self.store = FaissVectorStore(self.embedder.dimension)

            rows = self.repository.connections.get().execute(
                "SELECT * FROM memories WHERE user_id=? AND status='active' ORDER BY id",
                (self.repository.user_id,),
            ).fetchall()
            active_ids = {int(row["id"]) for row in rows}
            indexed_ids = self.store.ids()
            orphan_ids = indexed_ids - active_ids
            for memory_id in orphan_ids:
                self.store.remove(memory_id)
            added = 0
            reembedded = 0
            for row in rows:
                memory_id = int(row["id"])
                vector = self._stored_vector(row)
                needs_model = row["embedding_model"] != self.embedder.model_name
                if vector is None or needs_model:
                    vector = self.embedder.embed([row["content"]])[0]
                    reembedded += 1
                    self._persist_pending_embedding(memory_id, vector)
                vector = self._normalized(vector)
                if rebuilt or memory_id not in indexed_ids or row["index_status"] == "pending" or needs_model:
                    self.store.add(memory_id, vector)
                    added += 1
                    with self.repository._write_lock:
                        db = self.repository.connections.get()
                        db.execute("UPDATE memories SET index_status='pending' WHERE id=?", (memory_id,))
                        db.commit()
            if rebuilt or added or orphan_ids:
                self.store.save(self.index_path)
            with self.repository._write_lock:
                db = self.repository.connections.get()
                db.execute("UPDATE memories SET index_status='indexed' WHERE user_id=? AND status='active'",
                           (self.repository.user_id,))
                db.commit()
                self._update_index_state(db)
            return {"rebuilt": rebuilt, "added": added, "removed_orphans": len(orphan_ids),
                    "reembedded": reembedded, "vector_count": len(self.store.ids()),
                    "duration_ms": (time.perf_counter() - started) * 1000}
        except Exception:
            raise

    def search(self, query: str, top_k: int = 8, layer: Optional[str] = None,
               kind: Optional[str] = None, subject: Optional[str] = None,
               min_similarity: Optional[float] = None) -> list[dict[str, object]]:
        if top_k < 1 or top_k > 100:
            raise ValueError("top_k must be between 1 and 100")
        if min_similarity is None:
            min_similarity = config.min_similarity
        query_vector = self._normalized(self.embedder.embed([query])[0])
        candidates = self.store.search(query_vector, min(1000, max(top_k * 8, top_k)))
        if not candidates:
            return []
        by_id = {memory_id: score for memory_id, score in candidates}
        ids = list(by_id)
        placeholders = ",".join("?" for _ in ids)
        sql = f"SELECT * FROM memories WHERE user_id=? AND status='active' AND id IN ({placeholders})"
        params: list[object] = [self.repository.user_id] + ids
        for field, value in (("layer", layer), ("kind", kind), ("subject", subject)):
            if value is not None:
                sql += f" AND {field}=?"
                params.append(value)
        rows = self.repository.connections.get().execute(sql, params).fetchall()
        result = []
        for row in rows:
            score = by_id[int(row["id"])]
            if score >= min_similarity:
                item = dict(row)
                item["similarity"] = score
                result.append(item)
        result.sort(key=lambda item: (-item["similarity"], item["id"]))
        return result[:top_k]

    def verify_integrity(self) -> dict[str, object]:
        db_ids = {int(row[0]) for row in self.repository.connections.get().execute(
            "SELECT id FROM memories WHERE user_id=? AND status='active' AND embedding IS NOT NULL",
            (self.repository.user_id,),
        ).fetchall()}
        index_ids = self.store.ids()
        return {"missing_ids": sorted(db_ids - index_ids),
                "orphan_ids": sorted(index_ids - db_ids),
                "sqlite_vector_count": len(db_ids), "index_vector_count": len(index_ids),
                "ok": db_ids == index_ids}

    def rebuild_index(self) -> dict[str, int | float | bool]:
        self.index_path.unlink(missing_ok=True)
        self.store = FaissVectorStore(self.embedder.dimension)
        return self.reconcile()

    def _persist_pending_embedding(self, memory_id: int, vector: np.ndarray) -> None:
        normalized = self._normalized(vector)
        with self.repository._write_lock:
            db = self.repository.connections.get()
            db.execute("UPDATE memories SET embedding=?,embedding_model=?,index_status='pending' WHERE id=?",
                       (normalized.tobytes(), self.embedder.model_name, int(memory_id)))
            db.commit()

    def _stored_vector(self, row) -> Optional[np.ndarray]:
        raw = row["embedding"]
        if raw is None:
            return None
        vector = np.frombuffer(raw, dtype=np.float32)
        return vector if vector.size == self.embedder.dimension else None

    @staticmethod
    def _normalized(vector: np.ndarray) -> np.ndarray:
        normalized = np.asarray(vector, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(normalized))
        if not np.isfinite(norm) or norm == 0:
            raise ValueError("embedding must have a finite, non-zero norm")
        return np.ascontiguousarray(normalized / norm)

    def _index_state(self):
        return self.repository.connections.get().execute(
            "SELECT * FROM index_state ORDER BY rowid DESC LIMIT 1"
        ).fetchone()

    def _update_index_state(self, db: sqlite3.Connection) -> None:
        db.execute("""INSERT INTO index_state(embedding_model,dim,vector_count,last_synced_at)
        VALUES(?,?,?,?) ON CONFLICT(embedding_model) DO UPDATE SET dim=excluded.dim,
        vector_count=excluded.vector_count,last_synced_at=excluded.last_synced_at""",
                   (self.embedder.model_name, self.embedder.dimension, len(self.store.ids()), utc_now()))
        db.commit()
