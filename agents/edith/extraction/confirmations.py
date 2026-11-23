"""Explicit yes/no handling and expiry for sensitive candidate memories."""

from __future__ import annotations

import json
import logging
import re
from typing import Optional

from ..db.repository import MemoryRepository, utc_now
from ..models import MemoryInput
from ..config import config
from ..time_utils import normalize_utc_timestamp
from ..vector.sync import VectorSynchronizer


logger = logging.getLogger(__name__)
_YES = re.compile(r"(?i)^\s*(?:yes|yes,? remember(?: that)?|please remember(?: that)?|go ahead|confirm)\s*[!.]*\s*$")
_NO = re.compile(r"(?i)^\s*(?:no|no thanks|don't remember(?: that)?|do not remember(?: that)?|forget it|reject)\s*[!.]*\s*$")


class PendingConfirmationService:
    def __init__(self, repository: MemoryRepository, synchronizer: VectorSynchronizer,
                 ttl_turns: int = 4, user_timezone: str = config.user_timezone):
        self.repository = repository
        self.synchronizer = synchronizer
        self.ttl_turns = ttl_turns
        self.user_timezone = user_timezone

    def claim_prompt(self, conversation_id: int) -> bool:
        """Mark one unasked candidate as asked, so the UI asks only once."""
        turn = self.repository.count_user_turns(conversation_id)
        db = self.repository.connections.get()
        with self.repository._write_lock:
            self._expire(db, conversation_id, turn)
            row = db.execute(
                "SELECT id FROM pending_confirmations WHERE conversation_id=? AND status='pending' "
                "ORDER BY id LIMIT 1", (conversation_id,),
            ).fetchone()
            if not row:
                return False
            db.execute("UPDATE pending_confirmations SET status='asked' WHERE id=?", (int(row[0]),))
            db.commit()
            return True

    def resolve_reply(self, conversation_id: int, user_text: str) -> Optional[str]:
        """Resolve an explicit answer to an already asked confirmation prompt."""
        yes = bool(_YES.fullmatch(user_text))
        no = bool(_NO.fullmatch(user_text))
        if not yes and not no:
            upcoming_turn = self.repository.count_user_turns(conversation_id) + 1
            self._expire(self.repository.connections.get(), conversation_id, upcoming_turn)
            return None
        db = self.repository.connections.get()
        turn = self.repository.count_user_turns(conversation_id) + 1
        with self.repository._write_lock:
            self._expire(db, conversation_id, turn)
            row = db.execute(
                "SELECT id,candidate_json FROM pending_confirmations WHERE conversation_id=? "
                "AND status='asked' ORDER BY id LIMIT 1", (conversation_id,),
            ).fetchone()
            if not row:
                return None
            pending_id, candidate_json = int(row[0]), row[1]
            candidate = json.loads(candidate_json)
            if yes:
                item = MemoryInput(
                    layer=candidate["layer"], kind=candidate["kind"], content=candidate["content"],
                    subject=candidate.get("subject"), predicate=candidate.get("predicate"),
                    importance=float(candidate.get("importance", 0.7)),
                    confidence=float(candidate.get("confidence", 0.9)),
                    provenance="user_confirmed",
                    occurred_at=normalize_utc_timestamp(candidate.get("occurred_at"), self.user_timezone),
                    source_conversation_id=conversation_id,
                    source_message_ids=json.dumps(candidate.get("source_message_ids", [])),
                )
                supersedes_id = candidate.get("supersedes_id")
                if supersedes_id is not None:
                    old = self.repository.get_memory(int(supersedes_id))
                    if old is None or old["status"] != "active":
                        db.execute("UPDATE pending_confirmations SET candidate_json='{}',status='expired' WHERE id=?",
                                   (pending_id,))
                        db.commit()
                        return "That memory changed before confirmation, so the correction was not applied."
                    memory_id = self.repository.supersede_memory(int(supersedes_id), item)
                    try:
                        self.synchronizer.remove_from_index(int(supersedes_id))
                    except Exception as exc:
                        logger.error("Superseded memory index removal deferred (memory_id=%d,error_type=%s)",
                                     int(supersedes_id), type(exc).__name__)
                else:
                    memory_id = self.repository.create_memory(item)
                for entity in candidate.get("entities", []):
                    entity_id = self.repository.add_entity(entity["name"], entity["type"])
                    self.repository.link_entity(memory_id, entity_id)
                    self.repository.link_conversation_entity(
                        conversation_id, entity_id,
                        (candidate.get("source_message_ids") or [None])[-1])
                db.execute("UPDATE pending_confirmations SET candidate_json='{}',status='confirmed' WHERE id=?",
                           (pending_id,))
                db.commit()
                try:
                    self.synchronizer.index_memory(memory_id)
                except Exception as exc:
                    logger.error("Confirmed memory index deferred (memory_id=%d,error_type=%s)",
                                 memory_id, type(exc).__name__)
                return "Okay, I’ll remember that."
            db.execute("UPDATE pending_confirmations SET candidate_json='{}',status='rejected' WHERE id=?",
                       (pending_id,))
            db.commit()
            return "Okay, I won’t save it."

    def expire(self, conversation_id: int) -> int:
        return self._expire(self.repository.connections.get(), conversation_id,
                            self.repository.count_user_turns(conversation_id))

    def _expire(self, db, conversation_id: int, turn: int) -> int:
        with self.repository._write_lock:
            cursor = db.execute(
                "UPDATE pending_confirmations SET candidate_json='{}',status='expired' "
                "WHERE conversation_id=? AND status IN ('pending','asked') AND expires_after_turn<?",
                (conversation_id, turn),
            )
            db.commit()
            return cursor.rowcount
