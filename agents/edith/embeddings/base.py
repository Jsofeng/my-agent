"""Embedding interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np


class Embedder(ABC):
    model_name: str
    dimension: int

    @abstractmethod
    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Return L2-normalized float32 vectors, one per input text."""
        raise NotImplementedError
