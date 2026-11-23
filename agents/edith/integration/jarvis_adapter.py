"""Fail-open read adapter between Jarvis and Edith."""

from __future__ import annotations

import logging
import threading
from typing import Optional

from ..context.context_manager import ContextManager
from ..db.repository import MemoryRepository
from ..vector.sync import VectorSynchronizer


logger = logging.getLogger(__name__)


class JarvisMemoryAdapter:
    def __init__(self, repository: MemoryRepository, synchronizer: VectorSynchronizer,
                 enabled: bool = True, context_token_budget: int = 600, top_k: int = 8):
        self.repository = repository
        self.synchronizer = synchronizer
        self.enabled = enabled
        self.conversation_id: Optional[int] = None
        self.context_manager = ContextManager(repository, synchronizer, context_token_budget, top_k)
        self._ready = False
        self._init_lock = threading.Lock()

    def before_response(self, user_message: str, conversation_id: Optional[int]) -> str:
        """Retrieve reference context; failures never prevent Jarvis from responding."""
        if not self.enabled or not user_message.strip():
            return ""
        try:
            count = self.repository.connections.get().execute(
                "SELECT count(*) FROM memories WHERE user_id=? AND status='active'",
                (self.repository.user_id,),
            ).fetchone()[0]
            if not count:
                return ""
            self._ensure_ready()
            return self.context_manager.build(user_message)
        except Exception as exc:
            # Exceptions can contain SQL parameters or memory content; log only the type.
            logger.error("Edith retrieval failed (error_type=%s)", type(exc).__name__)
            return ""

    def after_response(self, conversation_id: Optional[int], user_message: str,
                       assistant_message: str) -> None:
        """Reserved post-response hook; persistence and extraction arrive in later phases."""
        return None

    def search(self, query: str, top_k: int = 5) -> list[dict[str, object]]:
        self._ensure_ready()
        return self.synchronizer.search(query, top_k=top_k)

    def _ensure_ready(self) -> None:
        if self._ready:
            return
        with self._init_lock:
            if not self._ready:
                self.synchronizer.reconcile()
                self._ready = True


def build_agent_task(user_message: str, adapter: Optional[JarvisMemoryAdapter],
                     conversation_id: Optional[int]) -> str:
    """Add retrieved context to an internal task while leaving display text alone."""
    if adapter is None:
        return user_message
    context = adapter.before_response(user_message, conversation_id)
    return (context + "\n\nUser request:\n" + user_message) if context else user_message
