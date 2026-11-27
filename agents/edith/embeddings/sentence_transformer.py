"""Lazy singleton wrapper for the configured MiniLM sentence model."""

from __future__ import annotations

import threading
from typing import Sequence

import numpy as np

from .base import Embedder


class SentenceTransformerEmbedder(Embedder):
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    dimension = 384
    _model = None
    _lock = threading.Lock()

    def _get_model(self):
        if self.__class__._model is None:
            with self.__class__._lock:
                if self.__class__._model is None:
                    from sentence_transformers import SentenceTransformer
                    self.__class__._model = SentenceTransformer(self.model_name)
        return self.__class__._model

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        vectors = self._get_model().encode(
            list(texts), convert_to_numpy=True, normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)
