"""Resolve explicit and implicit references from recent context and stored memories."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Optional

from ..config import EdithConfig, config
from ..db.repository import MemoryRepository
from ..vector.sync import VectorSynchronizer


_REFERENCE_TYPES = (
    (re.compile(r"(?i)\b(?:the interview|that interview|my interview|an interview|interview coming up)\b"), {"company", "project", "event"}, "application"),
    (re.compile(r"(?i)\b(?:my application|that application|the application)\b"), {"company", "project", "event"}, "application"),
    (re.compile(r"(?i)\b(?:that project|the project|my project)\b"), {"project", "company", "technology"}, "project"),
    (re.compile(r"(?i)\b(?:he|him|his|she|her|hers)\b"), {"person"}, "person"),
    (re.compile(r"(?i)\b(?:it|that|this)\b"), {"project", "technology", "event", "other"}, "reference"),
)
_CLOSED_EPISODE = re.compile(r"(?i)\b(?:accepted|rejected|declined|withdrew|completed|cancelled|canceled|ended)\b")
_OPEN_EPISODE = re.compile(r"(?i)\b(?:applying|application|interview|upcoming|in progress|working on|pending)\b")


@dataclass
class EntityResolution:
    status: str
    entity_ids: list[int] = field(default_factory=list)
    memory_ids: list[int] = field(default_factory=list)
    question: Optional[str] = None
    options: list[str] = field(default_factory=list)
    reference_kind: Optional[str] = None


class EntityResolver:
    def __init__(self, repository: MemoryRepository, synchronizer: VectorSynchronizer,
                 settings: EdithConfig = config):
        self.repository = repository
        self.synchronizer = synchronizer
        self.settings = settings

    def resolve(self, user_message: str, conversation_id: Optional[int]) -> EntityResolution:
        if conversation_id is not None:
            pending = self.repository.get_pending_question(conversation_id, "entity_ambiguity")
            if pending:
                return self._resolve_pending(pending, user_message, conversation_id)
        entities = self._all_entities()
        explicit = self._explicit_matches(user_message, entities)
        if explicit:
            if conversation_id is not None:
                latest = self.repository.get_messages(conversation_id, limit=100000)
                message_id = latest[-1]["id"] if latest else None
                for entity in explicit:
                    self.repository.link_conversation_entity(conversation_id, entity["id"], message_id)
            ids = [int(entity["id"]) for entity in explicit]
            return EntityResolution("resolved", ids, self._memory_ids_for_entities(ids),
                                    reference_kind="explicit")

        reference = self._reference_kind(user_message)
        if reference is None:
            return EntityResolution("none")
        expected_types, kind = reference
        candidate_scores: dict[int, tuple[float, dict, set[int]]] = {}
        if conversation_id is not None:
            recent = self.repository.recent_conversation_entities(conversation_id)
            for rank, entity in enumerate(recent):
                entity_id = int(entity["id"])
                compatibility = 1.0 if entity["type"] in expected_types else 0.45
                score = (0.20 + 0.65 * compatibility) / (rank + 1)
                candidate_scores[entity_id] = (score, entity, set(self._memory_ids_for_entities([entity_id])))

        try:
            retrieved = self.synchronizer.search(user_message, top_k=self.settings.top_k)
        except Exception:
            retrieved = []
        for item in retrieved:
            memory_id = int(item["id"])
            for entity in self.repository.memory_entities(memory_id):
                entity_id = int(entity["id"])
                old = candidate_scores.get(entity_id)
                linked_memories = old[2] if old else set()
                linked_memories.add(memory_id)
                recent_bonus = 0.16 if old else 0.0
                compat = 0.18 if entity["type"] in expected_types else 0.0
                similarity = max(0.0, min(1.0, float(item.get("similarity", 0.0))))
                open_bonus = 0.22 if self._has_open_episode(entity_id) else 0.0
                score = 0.12 + 0.24 * similarity + recent_bonus + compat + open_bonus
                if old:
                    score = max(score, old[0])
                candidate_scores[entity_id] = (score, entity, linked_memories)

        ranked = sorted(candidate_scores.values(), key=lambda row: (-row[0], row[1]["canonical_name"].lower()))
        if not ranked or ranked[0][0] < self.settings.resolve_threshold:
            return EntityResolution("unresolved", reference_kind=kind)
        close = [row for row in ranked if ranked[0][0] - row[0] < self.settings.ambiguity_margin]
        if len(close) > 1:
            options = [str(row[1]["canonical_name"]) for row in close]
            question = self._question(kind, options)
            if conversation_id is not None:
                existing = self.repository.get_pending_question(conversation_id, "entity_ambiguity")
                if not existing:
                    self.repository.add_pending_question(
                        conversation_id, "entity_ambiguity",
                        {"options": [{"id": int(row[1]["id"]), "name": str(row[1]["canonical_name"])}
                                     for row in close], "reference_kind": kind},
                    )
            return EntityResolution("ambiguous", question=question, options=options,
                                    reference_kind=kind)

        score, entity, memory_ids = ranked[0]
        entity_id = int(entity["id"])
        if conversation_id is not None:
            latest = self.repository.get_messages(conversation_id, limit=100000)
            message_id = latest[-1]["id"] if latest else None
            self.repository.link_conversation_entity(conversation_id, entity_id, message_id)
        return EntityResolution("resolved", [entity_id], sorted(memory_ids),
                                reference_kind=kind)

    def _resolve_pending(self, pending: dict, user_message: str,
                         conversation_id: int) -> EntityResolution:
        payload = json.loads(pending["payload_json"])
        options = payload.get("options", [])
        ordinal = re.search(r"(?i)\b(first|second|third|1st|2nd|3rd)\b", user_message)
        if ordinal:
            positions = {"first": 0, "1st": 0, "second": 1, "2nd": 1, "third": 2, "3rd": 2}
            position = positions[ordinal.group(1).lower()]
            matched = [options[position]] if position < len(options) else []
        else:
            matched = self._explicit_matches(user_message, options)
        if len(matched) == 1:
            entity_id = int(matched[0]["id"])
            self.repository.resolve_pending_question(pending["id"])
            messages = self.repository.get_messages(conversation_id, limit=100000)
            self.repository.link_conversation_entity(
                conversation_id, entity_id, messages[-1]["id"] if messages else None)
            return EntityResolution("resolved", [entity_id], self._memory_ids_for_entities([entity_id]),
                                    reference_kind=payload.get("reference_kind"))
        names = [str(option["name"]) for option in options]
        return EntityResolution("ambiguous", question=self._question(payload.get("reference_kind", "reference"), names),
                                options=names, reference_kind=payload.get("reference_kind"))

    def _explicit_matches(self, text: str, entities: list[dict]) -> list[dict]:
        normalized_text = self._normalize(text)
        exact: list[dict] = []
        fuzzy: list[tuple[float, dict]] = []
        for entity in entities:
            names = [str(entity.get("canonical_name") or entity.get("name") or "")]
            try:
                names.extend(json.loads(entity.get("aliases_json") or "[]"))
            except (TypeError, json.JSONDecodeError):
                pass
            for name in names:
                norm = self._normalize(str(name))
                if norm and re.search(r"(?<!\w)" + re.escape(norm) + r"(?!\w)", normalized_text):
                    exact.append(entity)
                    break
                ratio = self._fuzzy_phrase_ratio(norm, normalized_text) if norm else 0.0
                if ratio >= self.settings.entity_fuzzy_threshold:
                    fuzzy.append((ratio, entity))
        if exact:
            return self._unique_entities(exact)
        if fuzzy:
            best = max(score for score, _ in fuzzy)
            return self._unique_entities([entity for score, entity in fuzzy
                                          if best - score < self.settings.ambiguity_margin])
        return []

    def _all_entities(self) -> list[dict]:
        rows = self.repository.connections.get().execute(
            "SELECT * FROM entities WHERE user_id=? ORDER BY id", (self.repository.user_id,)
        ).fetchall()
        return [dict(row) for row in rows]

    @staticmethod
    def _unique_entities(entities: list[dict]) -> list[dict]:
        unique: dict[int, dict] = {int(entity["id"]): entity for entity in entities}
        return list(unique.values())

    def _memory_ids_for_entities(self, entity_ids: list[int]) -> list[int]:
        if not entity_ids:
            return []
        marks = ",".join("?" for _ in entity_ids)
        rows = self.repository.connections.get().execute(
            "SELECT DISTINCT m.id FROM memories m JOIN memory_entities me ON me.memory_id=m.id "
            "WHERE m.user_id=? AND m.status='active' AND me.entity_id IN (%s) ORDER BY m.id" % marks,
            [self.repository.user_id] + entity_ids,
        ).fetchall()
        return [int(row[0]) for row in rows]

    def _has_open_episode(self, entity_id: int) -> bool:
        rows = self.repository.connections.get().execute(
            """SELECT m.content FROM memories m JOIN memory_entities me ON me.memory_id=m.id
            WHERE m.user_id=? AND me.entity_id=? AND m.layer='episodic' AND m.status='active'""",
            (self.repository.user_id, entity_id),
        ).fetchall()
        for row in rows:
            content = str(row[0])
            if _OPEN_EPISODE.search(content) and not _CLOSED_EPISODE.search(content):
                return True
        return False

    @staticmethod
    def _normalize(text: str) -> str:
        return re.sub(r"[^\w\s]", " ", text.lower()).strip()

    @staticmethod
    def _fuzzy_phrase_ratio(name: str, text: str) -> float:
        tokens = text.split()
        name_tokens = name.split()
        if not tokens or not name_tokens:
            return 0.0
        widths = {max(1, len(name_tokens) - 1), len(name_tokens), len(name_tokens) + 1}
        best = 0.0
        for width in widths:
            for start in range(max(1, len(tokens) - width + 1)):
                phrase = " ".join(tokens[start:start + width])
                best = max(best, SequenceMatcher(None, name, phrase).ratio())
        return best

    @staticmethod
    def _reference_kind(text: str):
        for pattern, types, kind in _REFERENCE_TYPES:
            if pattern.search(text):
                return types, kind
        return None

    @staticmethod
    def _question(kind: str, options: list[str]) -> str:
        if kind == "application" or kind == "interview":
            lead = "Which application do you mean"
        elif kind == "project":
            lead = "Which project do you mean"
        elif kind == "person":
            lead = "Which person do you mean"
        else:
            lead = "Which one do you mean"
        return lead + ", " + " or ".join(options) + "?"
