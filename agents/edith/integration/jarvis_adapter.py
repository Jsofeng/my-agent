"""Fail-open read adapter between Jarvis and Edith."""

from __future__ import annotations

import logging
import threading
from typing import Optional

from ..context.context_manager import ContextManager
from ..config import EdithConfig, config
from ..db.repository import MemoryRepository
from ..extraction.confirmations import PendingConfirmationService
from .background_worker import ExtractionWorker
from ..vector.sync import VectorSynchronizer


logger = logging.getLogger(__name__)


class JarvisMemoryAdapter:
    def __init__(self, repository: MemoryRepository, synchronizer: VectorSynchronizer,
                 enabled: bool = True, context_token_budget: int = 600, top_k: int = 8,
                 worker: Optional[ExtractionWorker] = None,
                 pending_confirmations: Optional[PendingConfirmationService] = None,
                 settings: EdithConfig = config):
        self.repository = repository
        self.synchronizer = synchronizer
        self.enabled = enabled
        self.conversation_id: Optional[int] = None
        self.context_manager = ContextManager(repository, synchronizer, context_token_budget, top_k, settings)
        self.settings = settings
        self.worker = worker
        self.pending_confirmations = pending_confirmations
        self._ready = False
        self._init_lock = threading.Lock()

    def before_response(self, user_message: str, conversation_id: Optional[int]) -> str:
        """Retrieve reference context; failures never prevent Jarvis from responding."""
        if not self.enabled or not user_message.strip():
            return ""
        try:
            confirmation_prompt = False
            if conversation_id is not None and self.pending_confirmations is not None:
                confirmation_prompt = self.pending_confirmations.claim_prompt(conversation_id)
            resolution = self.context_manager.resolve(user_message, conversation_id)
            if resolution.status == "ambiguous":
                blocks = ["<pending_question>Ask the user: %s Do not guess the reference. "
                          "Wait for the clarification before using historical details.</pending_question>" %
                          (resolution.question or "Which one do you mean?")]
                if confirmation_prompt:
                    blocks.append("<pending_confirmation>Ask the user once: "
                                  "Want me to remember that? Do not assume yes.</pending_confirmation>")
                return "\n".join(blocks)
            count = self.repository.connections.get().execute(
                "SELECT count(*) FROM memories WHERE user_id=? AND status='active'",
                (self.repository.user_id,),
            ).fetchone()[0]
            context = ""
            if count:
                self._ensure_ready()
                context = self.context_manager.build(user_message, conversation_id, resolution)
            if confirmation_prompt:
                context += ("\n<pending_confirmation>Ask the user once: "
                            "Want me to remember that? Do not assume yes.</pending_confirmation>")
            return context
        except Exception as exc:
            # Exceptions can contain SQL parameters or memory content; log only the type.
            logger.error("Edith retrieval failed (error_type=%s)", type(exc).__name__)
            if confirmation_prompt:
                return ("<pending_confirmation>Ask the user once: "
                        "Want me to remember that? Do not assume yes.</pending_confirmation>")
            return ""

    def record_user_message(self, conversation_id: Optional[int], user_message: str) -> None:
        """Persist a user turn before model generation so shutdown can recover it."""
        if conversation_id is None:
            return
        try:
            self.repository.add_message(conversation_id, "user", user_message)
        except Exception as exc:
            logger.error("Edith user-message persistence failed (error_type=%s)", type(exc).__name__)

    def after_response(self, conversation_id: Optional[int], user_message: str,
                       assistant_message: str) -> None:
        """Persist the assistant turn and enqueue extraction without waiting for the model."""
        if conversation_id is None:
            return
        try:
            self.repository.add_message(conversation_id, "assistant", assistant_message)
            if self.worker is not None:
                self.worker.after_response(conversation_id)
        except Exception as exc:
            logger.error("Edith post-response persistence failed (error_type=%s)", type(exc).__name__)

    def resolve_pending_response(self, conversation_id: Optional[int],
                                 user_message: str) -> Optional[str]:
        if conversation_id is None or self.pending_confirmations is None:
            return None
        try:
            return self.pending_confirmations.resolve_reply(conversation_id, user_message)
        except Exception as exc:
            logger.error("Pending confirmation resolution failed (error_type=%s)", type(exc).__name__)
            return None

    def end_conversation(self, conversation_id: Optional[int]) -> None:
        if conversation_id is None:
            return
        try:
            self.repository.end_conversation(conversation_id)
            if self.worker is not None:
                self.worker.end_conversation(conversation_id)
        except Exception as exc:
            logger.error("Edith conversation close failed (error_type=%s)", type(exc).__name__)

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
