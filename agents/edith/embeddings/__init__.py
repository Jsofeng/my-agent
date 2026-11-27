"""Embedding model interfaces and implementations."""

from .base import Embedder
from .sentence_transformer import SentenceTransformerEmbedder

__all__ = ["Embedder", "SentenceTransformerEmbedder"]
