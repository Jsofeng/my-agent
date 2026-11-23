"""Validated, dependency-injected extraction service."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Optional, Protocol, Sequence

from pydantic import TypeAdapter, ValidationError

from ..config import EdithConfig, config
from ..db.repository import MemoryRepository
from ..models import MemoryInput
from ..layers.short_term import RollingSummaryService
from ..time_utils import normalize_utc_timestamp
from .prefilter import filter_messages
from .prompts import SYSTEM_PROMPT, build_extraction_prompt, build_repair_prompt
from .schemas import ExtractedMemory
from .secrets import scan_secrets
from .sensitivity import sensitive_categories
from ..updating.conflicts import LLMConflictAdjudicator, SemanticConflictService


logger = logging.getLogger(__name__)
_BATCH_ADAPTER = TypeAdapter(list[ExtractedMemory])


class LLMClient(Protocol):
    def complete(self, system_prompt: str, user_prompt: str, temperature: float = 0.0) -> str:
        """Return raw model text without performing application-side parsing."""


class SmolagentsLLMClient:
    """Use the existing smolagents model abstraction with deterministic temperature."""

    def __init__(self, model):
        self.model = model

    def complete(self, system_prompt: str, user_prompt: str, temperature: float = 0.0) -> str:
        from smolagents import ChatMessage
        from smolagents.models import MessageRole
        result = self.model.generate(
            [ChatMessage(role=MessageRole.SYSTEM, content=system_prompt),
             ChatMessage(role=MessageRole.USER, content=user_prompt)],
            stop_sequences=[], temperature=temperature,
        )
        if not isinstance(result.content, str):
            raise ValueError("extraction model returned non-text output")
        return result.content


class ExtractionService:
    def __init__(self, repository: MemoryRepository, llm: LLMClient,
                 settings: EdithConfig = config, clock=None, synchronizer=None):
        self.repository = repository
        self.llm = llm
        self.settings = settings
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.synchronizer = synchronizer
        self.rolling_summary = RollingSummaryService(repository, llm, settings)
        self.conflicts = SemanticConflictService(
            repository, LLMConflictAdjudicator(llm), synchronizer, settings.dup_threshold)

    def process_conversation(self, conversation_id: int) -> dict[str, int]:
        messages = self.repository.unextracted_messages(
            conversation_id, batch_user_messages=self.settings.extraction_batch_messages)
        if not messages:
            return {"processed": 0, "saved": 0, "pending": 0, "skipped": 0}
        user_messages = [row for row in messages if row["role"] == "user"]
        useful = filter_messages([row["content"] for row in user_messages])
        saved = 0
        pending = 0
        skipped = 0
        try:
            if useful:
                raw = self.llm.complete(
                    SYSTEM_PROMPT,
                    build_extraction_prompt(useful, self.clock(), self.settings.user_timezone),
                    temperature=0.0,
                )
                candidates = self._parse_with_one_repair(raw, useful)
                for candidate in candidates:
                    outcome = self._store_candidate(candidate, conversation_id, messages)
                    if outcome == "saved":
                        saved += 1
                    elif outcome == "pending":
                        pending += 1
                    else:
                        skipped += 1
            else:
                skipped = len(user_messages)
            self.repository.mark_messages_extracted([row["id"] for row in messages])
            try:
                self.rolling_summary.maintain(conversation_id)
            except Exception as summary_error:
                logger.error("Rolling summary failed (conversation_id=%d,error_type=%s)",
                             conversation_id, type(summary_error).__name__)
        except Exception as exc:
            logger.error("Extraction attempt failed (conversation_id=%d,error_type=%s)",
                         conversation_id, type(exc).__name__)
            raise
        return {"processed": len(messages), "saved": saved, "pending": pending, "skipped": skipped}

    def _parse_with_one_repair(self, raw: str, messages: Sequence[str]) -> list[ExtractedMemory]:
        try:
            return self._parse(raw)
        except (ValueError, ValidationError, json.JSONDecodeError) as first_error:
            prompt = build_repair_prompt(raw, str(first_error))
            repaired = self.llm.complete(SYSTEM_PROMPT, prompt, temperature=0.0)
            try:
                return self._parse(repaired)
            except (ValueError, ValidationError, json.JSONDecodeError):
                logger.error("Extraction output rejected after repair (message_count=%d)", len(messages))
                return []

    @staticmethod
    def _parse(raw: str) -> list[ExtractedMemory]:
        text = raw.strip()
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        data = json.loads(text.strip())
        return _BATCH_ADAPTER.validate_python(data)

    def _store_candidate(self, candidate: ExtractedMemory, conversation_id: int,
                         messages: Sequence[dict]) -> str:
        # Automatic extraction may never promote a model-supplied confirmation.
        if candidate.provenance == "user_confirmed":
            return "skipped"
        if candidate.importance < self.settings.min_importance or candidate.confidence < self.settings.min_confidence:
            return "skipped"
        candidate_fields = [candidate.content, candidate.subject or "", candidate.predicate or ""]
        for entity in candidate.entities:
            candidate_fields.extend([entity.name, entity.type])
        candidate_text = " ".join(candidate_fields)
        if any(scan_secrets(value).has_secrets for value in candidate_fields):
            logger.info("Extraction candidate dropped by secret scan")
            return "skipped"
        if not candidate.content.strip():
            return "skipped"
        categories = sensitive_categories(candidate_text)
        if categories:
            encoded_candidate = candidate.model_dump()
            encoded_candidate["occurred_at"] = normalize_utc_timestamp(
                candidate.occurred_at, self.settings.user_timezone)
            encoded_candidate["source_message_ids"] = [
                int(row["id"]) for row in messages if row["role"] == "user"]
            existing_pending = self.repository.connections.get().execute(
                "SELECT id FROM pending_confirmations WHERE conversation_id=? "
                "AND status IN ('pending','asked') AND candidate_json=? LIMIT 1",
                (conversation_id, json.dumps(encoded_candidate, ensure_ascii=False)),
            ).fetchone()
            if existing_pending:
                return "pending"
            self.repository.add_pending_confirmation(
                conversation_id,
                encoded_candidate,
                "sensitive_category:" + ",".join(categories),
                self.settings.pending_confirmation_ttl_turns,
                asked=False,
            )
            return "pending"
        source_message_ids = [int(row["id"]) for row in messages if row["role"] == "user"]
        item = MemoryInput(
            layer=candidate.layer, kind=candidate.kind, content=candidate.content,
            subject=candidate.subject, predicate=candidate.predicate,
            importance=candidate.importance, confidence=candidate.confidence,
            provenance=candidate.provenance,
            occurred_at=normalize_utc_timestamp(candidate.occurred_at, self.settings.user_timezone),
            source_conversation_id=conversation_id,
            source_message_ids=json.dumps(source_message_ids),
        )
        relation, memory_id = self.conflicts.save(item)
        if relation == "pending":
            return "pending"
        if memory_id is not None:
            self._link_entities(memory_id, candidate.entities, conversation_id,
                                source_message_ids[-1] if source_message_ids else None)
        return "saved" if relation != "pending" else "skipped"

    def _link_entities(self, memory_id: int, entities, conversation_id: int,
                       message_id: Optional[int]) -> None:
        for entity in entities:
            entity_id = self.repository.add_entity(entity.name, entity.type)
            self.repository.link_entity(memory_id, entity_id)
            self.repository.link_conversation_entity(conversation_id, entity_id, message_id)
