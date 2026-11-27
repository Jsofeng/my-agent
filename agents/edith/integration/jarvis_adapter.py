"""Fail-open read adapter between Jarvis and Edith."""

from __future__ import annotations

import logging
import json
import threading
from html import escape
from typing import Optional

from ..context.context_manager import ContextManager
from ..config import EdithConfig, config
from ..db.repository import MemoryRepository
from ..extraction.confirmations import PendingConfirmationService
from ..layers.short_term import build_short_term_context
from ..models import MemoryInput
from .background_worker import ExtractionWorker
from ..vector.sync import VectorSynchronizer


logger = logging.getLogger(__name__)


class JarvisMemoryAdapter:
    def __init__(self, repository: MemoryRepository, synchronizer: VectorSynchronizer,
                 enabled: bool = True, context_token_budget: int = 600, top_k: int = 8,
                 worker: Optional[ExtractionWorker] = None,
                 pending_confirmations: Optional[PendingConfirmationService] = None,
                 conflict_service=None,
                 consolidator=None,
                 settings: EdithConfig = config):
        self.repository = repository
        self.synchronizer = synchronizer
        self.enabled = enabled
        self.conversation_id: Optional[int] = None
        self.context_manager = ContextManager(repository, synchronizer, context_token_budget, top_k, settings)
        self.settings = settings
        self.worker = worker
        self.pending_confirmations = pending_confirmations
        self.conflict_service = conflict_service
        self.consolidator = consolidator
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
            semantic_question = (self.repository.get_pending_question(conversation_id, "semantic_conflict")
                                 if conversation_id is not None else None)
            if semantic_question is not None:
                try:
                    payload = json.loads(semantic_question["payload_json"])
                    old = self.repository.get_memory(int(payload["memory_id"]))
                    old_content = old["content"] if old else "previous saved fact"
                    new_content = str(payload.get("new_content", "new fact"))
                    prompt = ("I have two conflicting saved facts: first, '%s'; second, '%s'. "
                              "Ask which one is current. Do not choose or use either as settled history." %
                              (escape(str(old_content), quote=True), escape(new_content, quote=True)))
                    blocks = ["<pending_question>" + prompt + "</pending_question>"]
                    if confirmation_prompt:
                        blocks.append("<pending_confirmation>Ask the user once: "
                                      "Want me to remember that? Do not assume yes.</pending_confirmation>")
                    return "\n".join(blocks)
                except Exception:
                    logger.error("Pending conflict question failed (error_type=RuntimeError)")
            resolution = self.context_manager.resolve(user_message, conversation_id)
            if resolution.status == "ambiguous":
                blocks = ["<pending_question>Ask the user: %s Do not guess the reference. "
                          "Wait for the clarification before using historical details.</pending_question>" %
                          (resolution.question or "Which one do you mean?")]
                if confirmation_prompt:
                    blocks.append("<pending_confirmation>Ask the user once: "
                                  "Want me to remember that? Do not assume yes.</pending_confirmation>")
                return "\n".join(blocks)
            short_term = build_short_term_context(
                self.repository, conversation_id, user_message, self.settings)
            count = self.repository.connections.get().execute(
                "SELECT count(*) FROM memories WHERE user_id=? AND status='active'",
                (self.repository.user_id,),
            ).fetchone()[0]
            context = short_term
            if count:
                self._ensure_ready()
                memory_context = self.context_manager.build(user_message, conversation_id, resolution)
                context = "\n".join(part for part in (short_term, memory_context) if part)
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
        if conversation_id is None:
            return None
        if self.pending_confirmations is not None:
            try:
                result = self.pending_confirmations.resolve_reply(conversation_id, user_message)
                if result is not None:
                    return result
            except Exception as exc:
                logger.error("Pending confirmation resolution failed (error_type=%s)", type(exc).__name__)
        try:
            pending = self.repository.get_pending_question(conversation_id, "semantic_conflict")
            if pending is None:
                return None
            payload = json.loads(pending["payload_json"])
            old_id = int(payload["memory_id"])
            old = self.repository.get_memory(old_id)
            if old is None:
                self.repository.resolve_pending_question(int(pending["id"]))
                return None
            normalized = " ".join(user_message.lower().strip().rstrip(".!?").split())
            choose_old = normalized in {"first", "first one", "the first", "the first one",
                                       "old", "old one", "keep the old one"}
            choose_new = normalized in {"second", "second one", "the second", "the second one",
                                       "new", "new one", "use the new one"}
            new_content = str(payload.get("new_content", ""))
            normalized_new = " ".join(new_content.lower().split())
            normalized_old = " ".join(str(old["content"]).lower().split())
            if not choose_new and normalized_new and normalized_new in normalized:
                choose_new = True
            if not choose_old and normalized_old and normalized_old in normalized:
                choose_old = True
            if not choose_old and not choose_new:
                return None
            if choose_new:
                self._ensure_ready()
                replacement = MemoryInput(
                    layer="semantic", kind=old["kind"], content=new_content,
                    subject=old["subject"], predicate=old["predicate"],
                    importance=float(old["importance"]), confidence=0.9,
                    provenance="user_stated", source_conversation_id=conversation_id,
                )
                new_id = self.repository.supersede_memory(old_id, replacement)
                self.synchronizer.remove_from_index(old_id)
                self.synchronizer.index_memory(new_id)
            self.repository.resolve_pending_question(int(pending["id"]))
            return "Okay, I’ll keep the existing fact." if choose_old else "Okay, I’ve updated the saved fact."
        except Exception as exc:
            logger.error("Pending conflict resolution failed (error_type=%s)", type(exc).__name__)
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
