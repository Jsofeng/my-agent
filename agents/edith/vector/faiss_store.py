"""Exact cosine-similarity search using an ID-mapped FAISS flat index."""

from __future__ import annotations

import os
import threading
from pathlib import Path

import faiss
import numpy as np

from .base import VectorStore


class FaissVectorStore(VectorStore):
    def __init__(self, dimension: int = 384):
        self.dimension = int(dimension)
        self._lock = threading.RLock()
        self._index = self._new_index()

    def _new_index(self):
        return faiss.IndexIDMap2(faiss.IndexFlatIP(self.dimension))

    def add(self, memory_id: int, vector: np.ndarray) -> None:
        normalized = self._normalize(vector)
        with self._lock:
            self._index.remove_ids(np.asarray([int(memory_id)], dtype=np.int64))
            self._index.add_with_ids(normalized.reshape(1, -1), np.asarray([memory_id], dtype=np.int64))

    def remove(self, memory_id: int) -> None:
        with self._lock:
            self._index.remove_ids(np.asarray([int(memory_id)], dtype=np.int64))

    def search(self, vector: np.ndarray, top_k: int) -> list[tuple[int, float]]:
        if top_k <= 0:
            return []
        query = self._normalize(vector).reshape(1, -1)
        with self._lock:
            if self._index.ntotal == 0:
                return []
            scores, ids = self._index.search(query, min(int(top_k), int(self._index.ntotal)))
            return [(int(idx), float(score)) for idx, score in zip(ids[0], scores[0]) if idx >= 0]

    def ids(self) -> set[int]:
        with self._lock:
            return set(int(i) for i in faiss.vector_to_array(self._index.id_map))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_name(path.name + ".tmp")
        with self._lock:
            faiss.write_index(self._index, str(temp))
            os.replace(temp, path)

    def load(self, path: Path) -> None:
        loaded = faiss.read_index(str(path))
        if loaded.d != self.dimension or not isinstance(loaded, faiss.IndexIDMap2):
            raise ValueError("FAISS index dimension or type does not match configuration")
        with self._lock:
            self._index = loaded

    def _normalize(self, vector: np.ndarray) -> np.ndarray:
        result = np.asarray(vector, dtype=np.float32).reshape(-1)
        if result.size != self.dimension:
            raise ValueError("expected vector dimension %d, got %d" % (self.dimension, result.size))
        norm = float(np.linalg.norm(result))
        if not np.isfinite(norm) or norm == 0:
            raise ValueError("vector must have a finite, non-zero norm")
        return np.ascontiguousarray(result / norm)
