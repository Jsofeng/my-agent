"""Vector storage interface; SQLite remains authoritative."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import numpy as np


class VectorStore(ABC):
    @abstractmethod
    def add(self, memory_id: int, vector: np.ndarray) -> None:
        raise NotImplementedError

    @abstractmethod
    def remove(self, memory_id: int) -> None:
        raise NotImplementedError

    @abstractmethod
    def search(self, vector: np.ndarray, top_k: int) -> list[tuple[int, float]]:
        raise NotImplementedError

    @abstractmethod
    def ids(self) -> set[int]:
        raise NotImplementedError

    @abstractmethod
    def save(self, path: Path) -> None:
        raise NotImplementedError

    @abstractmethod
    def load(self, path: Path) -> None:
        raise NotImplementedError
