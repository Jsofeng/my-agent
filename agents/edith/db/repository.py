"""Single write-coordinating repository for Edith's SQLite data."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Union

from .connection import ConnectionManager
from .migrations import apply_migrations
from ..extraction.secrets import scan_secrets
from ..models import ENTITY_TYPES, KINDS, LAYERS, MEMORY_STATUSES, PROVENANCE, MemoryInput


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class MemoryRepository:
    def __init__(self, path: Union[Path, str] = "data/edith.db", user_id: str = "default",
                 busy_timeout_ms: int = 5000):
        self.user_id = user_id
        self.connections = ConnectionManager(path, busy_timeout_ms)
        self._write_lock = threading.RLock()
        apply_migrations(self.connections.get())

    def close_thread(self) -> None:
        self.connections.close_thread()

    def close(self) -> None:
        self.connections.close_all()

    def create_conversation(self, title: Optional[str] = None) -> int:
        now = utc_now()
        with self._write_lock:
            cursor = self.connections.get().execute(
                "INSERT INTO conversations(user_id,started_at,title,status) VALUES(?,?,?,'active')",
                (self.user_id, now, title),
            )
            self.connections.get().commit()
            return int(cursor.lastrowid)

    def end_conversation(self, conversation_id: int) -> None:
        with self._write_lock:
            self.connections.get().execute(
                "UPDATE conversations SET ended_at=?,status='ended' WHERE id=? AND user_id=?",
                (utc_now(), conversation_id, self.user_id),
            )
            self.connections.get().commit()

    def add_message(self, conversation_id: int, role: str, content: str,
                    extracted: bool = False) -> int:
        if role not in {"user", "assistant", "system"}:
            raise ValueError("role must be user, assistant, or system")
        if not isinstance(content, str):
            raise TypeError("content must be text")
        # Short-term chat storage also must not retain secret-like tokens.
        content = scan_secrets(content).safe_text
        with self._write_lock:
            cursor = self.connections.get().execute(
                "INSERT INTO messages(conversation_id,role,content,created_at,extracted) "
                "SELECT id,?,?,?,? FROM conversations WHERE id=? AND user_id=?",
                (role, content, utc_now(), int(extracted), conversation_id, self.user_id),
            )
            if cursor.rowcount != 1:
                raise ValueError("conversation does not exist")
            self.connections.get().commit()
            return int(cursor.lastrowid)

    def get_messages(self, conversation_id: int, limit: Optional[int] = None) -> list[dict[str, Any]]:
        sql = "SELECT * FROM messages WHERE conversation_id=? ORDER BY id"
        args: tuple[Any, ...] = (conversation_id,)
        if limit is not None:
            sql += " LIMIT ?"
            args += (max(0, int(limit)),)
        rows = self.connections.get().execute(sql, args).fetchall()
        return [dict(row) for row in rows]

    def recent_messages(self, conversation_id: int, limit: int = 8) -> list[dict[str, Any]]:
        rows = self.connections.get().execute(
            "SELECT * FROM (SELECT * FROM messages WHERE conversation_id=? "
            "ORDER BY id DESC LIMIT ?) ORDER BY id",
            (int(conversation_id), max(0, int(limit))),
        ).fetchall()
        return [dict(row) for row in rows]

    def mark_messages_extracted(self, message_ids: list[int]) -> None:
        if not message_ids:
            return
        with self._write_lock:
            self.connections.get().executemany(
                "UPDATE messages SET extracted=1 WHERE id=?", [(int(i),) for i in message_ids]
            )
            self.connections.get().commit()

    def unextracted_messages(self, conversation_id: int,
                             batch_user_messages: int = 8) -> list[dict[str, Any]]:
        """Return through the Nth pending user turn with its intervening chat context."""
        db = self.connections.get()
        boundary = db.execute(
            "SELECT id FROM messages WHERE conversation_id=? AND extracted=0 AND role='user' "
            "ORDER BY id LIMIT 1 OFFSET ?",
            (conversation_id, max(0, int(batch_user_messages) - 1)),
        ).fetchone()
        if boundary:
            next_user = db.execute(
                "SELECT id FROM messages WHERE conversation_id=? AND extracted=0 AND role='user' "
                "AND id>? ORDER BY id LIMIT 1",
                (conversation_id, int(boundary[0])),
            ).fetchone()
            max_id = (int(next_user[0]) - 1) if next_user else int(db.execute(
                "SELECT max(id) FROM messages WHERE conversation_id=? AND extracted=0",
                (conversation_id,),
            ).fetchone()[0])
            rows = db.execute(
                "SELECT * FROM messages WHERE conversation_id=? AND extracted=0 AND id<=? ORDER BY id",
                (conversation_id, max_id),
            ).fetchall()
        else:
            rows = db.execute(
                "SELECT * FROM messages WHERE conversation_id=? AND extracted=0 ORDER BY id",
                (conversation_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def unextracted_conversation_ids(self) -> list[int]:
        rows = self.connections.get().execute(
            "SELECT DISTINCT conversation_id FROM messages WHERE extracted=0 ORDER BY conversation_id"
        ).fetchall()
        return [int(row[0]) for row in rows]

    def count_unextracted_user_messages(self, conversation_id: int) -> int:
        return int(self.connections.get().execute(
            "SELECT count(*) FROM messages WHERE conversation_id=? AND extracted=0 AND role='user'",
            (conversation_id,),
        ).fetchone()[0])

    def count_unextracted_messages(self, conversation_id: int) -> int:
        return int(self.connections.get().execute(
            "SELECT count(*) FROM messages WHERE conversation_id=? AND extracted=0",
            (conversation_id,),
        ).fetchone()[0])

    def count_user_turns(self, conversation_id: int) -> int:
        return int(self.connections.get().execute(
            "SELECT count(*) FROM messages WHERE conversation_id=? AND role='user'",
            (conversation_id,),
        ).fetchone()[0])

    def save_summary(self, conversation_id: int, summary: str,
                     covers_up_to_message_id: Optional[int]) -> int:
        with self._write_lock:
            cursor = self.connections.get().execute(
                "INSERT INTO conversation_summaries(conversation_id,summary,covers_up_to_message_id,created_at) "
                "SELECT id,?,?,? FROM conversations WHERE id=? AND user_id=?",
                (summary, covers_up_to_message_id, utc_now(), conversation_id, self.user_id),
            )
            if cursor.rowcount != 1:
                raise ValueError("conversation does not exist")
            self.connections.get().commit()
            return int(cursor.lastrowid)

    def latest_summary(self, conversation_id: int) -> Optional[dict[str, Any]]:
        row = self.connections.get().execute(
            "SELECT s.* FROM conversation_summaries s JOIN conversations c ON c.id=s.conversation_id "
            "WHERE s.conversation_id=? AND c.user_id=? ORDER BY s.id DESC LIMIT 1",
            (conversation_id, self.user_id),
        ).fetchone()
        return dict(row) if row else None

    def create_memory(self, item: MemoryInput) -> int:
        self._validate_memory(item)
        now = utc_now()
        digest = hashlib.sha256(item.content.strip().encode("utf-8")).hexdigest()
        with self._write_lock:
            cursor = self.connections.get().execute(
                """INSERT INTO memories(uuid,user_id,layer,kind,content,subject,predicate,importance,
                confidence,provenance,index_status,occurred_at,valid_from,source_conversation_id,
                source_message_ids,content_hash,embedding,embedding_model,created_at,updated_at)
                VALUES(?,?,?,?,?,?,?,?,?,?,'pending',?,?,?,?,?,?,?,?,?)""",
                (str(uuid.uuid4()), self.user_id, item.layer, item.kind, item.content, item.subject,
                 item.predicate, item.importance, item.confidence, item.provenance, item.occurred_at,
                 item.valid_from, item.source_conversation_id, item.source_message_ids, digest,
                 item.embedding, item.embedding_model, now, now),
            )
            memory_id = int(cursor.lastrowid)
            self._event(memory_id, "create")
            self.connections.get().commit()
            return memory_id

    def consolidate_memories(self, summary: MemoryInput, source_ids: list[int]) -> int:
        """Atomically add a summary and archive/link all its active source rows."""
        self._validate_memory(summary)
        unique_ids = sorted(set(int(value) for value in source_ids))
        if len(unique_ids) < 2:
            raise ValueError("consolidation needs at least two source memories")
        now = utc_now()
        digest = hashlib.sha256(summary.content.strip().encode("utf-8")).hexdigest()
        with self._write_lock:
            db = self.connections.get()
            try:
                marks = ",".join("?" for _ in unique_ids)
                rows = db.execute(
                    "SELECT id FROM memories WHERE user_id=? AND status='active' AND id IN (" + marks + ")",
                    [self.user_id] + unique_ids,
                ).fetchall()
                if len(rows) != len(unique_ids):
                    raise ValueError("all consolidation sources must be active memories")
                cursor = db.execute(
                    """INSERT INTO memories(uuid,user_id,layer,kind,content,subject,predicate,importance,
                    confidence,provenance,index_status,occurred_at,valid_from,source_conversation_id,
                    source_message_ids,content_hash,embedding,embedding_model,created_at,updated_at)
                    VALUES(?,?,?,?,?,?,?,?,?,?,'pending',?,?,?,?,?,?,?,?,?)""",
                    (str(uuid.uuid4()), self.user_id, summary.layer, summary.kind, summary.content,
                     summary.subject, summary.predicate, summary.importance, summary.confidence,
                     summary.provenance, summary.occurred_at, summary.valid_from,
                     summary.source_conversation_id, summary.source_message_ids, digest,
                     summary.embedding, summary.embedding_model, now, now),
                )
                summary_id = int(cursor.lastrowid)
                db.execute(
                    "INSERT INTO memory_entities(memory_id,entity_id,role) "
                    "SELECT ?,entity_id,role FROM memory_entities WHERE memory_id IN (" + marks + ")",
                    [summary_id] + unique_ids,
                )
                self._event(summary_id, "create")
                for source_id in unique_ids:
                    db.execute("UPDATE memories SET status='archived',index_status='pending',consolidated_into=?,updated_at=? WHERE id=?",
                               (summary_id, now, source_id))
                    self._event(source_id, "consolidate")
                db.commit()
                return summary_id
            except Exception:
                db.rollback()
                raise

    def merge_duplicate(self, source_id: int, target_id: int) -> None:
        """Archive a near-duplicate semantic row into its surviving fact atomically."""
        source_id, target_id = int(source_id), int(target_id)
        if source_id == target_id:
            return
        now = utc_now()
        with self._write_lock:
            db = self.connections.get()
            try:
                source = db.execute(
                    "SELECT importance,confidence FROM memories WHERE id=? AND user_id=? AND status='active'",
                    (source_id, self.user_id),
                ).fetchone()
                target = db.execute(
                    "SELECT importance,confidence FROM memories WHERE id=? AND user_id=? AND status='active'",
                    (target_id, self.user_id),
                ).fetchone()
                if source is None or target is None:
                    raise ValueError("both memories must be active")
                db.execute(
                    "INSERT OR IGNORE INTO memory_entities(memory_id,entity_id,role) "
                    "SELECT ?,entity_id,role FROM memory_entities WHERE memory_id=?",
                    (target_id, source_id),
                )
                db.execute("UPDATE memories SET importance=?,confidence=?,updated_at=? WHERE id=?",
                           (max(float(source["importance"]), float(target["importance"])),
                            max(float(source["confidence"]), float(target["confidence"])), now, target_id))
                db.execute("UPDATE memories SET status='archived',index_status='pending',consolidated_into=?,updated_at=? WHERE id=?",
                           (target_id, now, source_id))
                self._event(target_id, "reinforce")
                self._event(source_id, "merge")
                db.commit()
            except Exception:
                db.rollback()
                raise

    def supersede_memory(self, memory_id: int, item: MemoryInput) -> int:
        """Atomically preserve an old fact and insert its replacement."""
        self._validate_memory(item)
        now = utc_now()
        digest = hashlib.sha256(item.content.strip().encode("utf-8")).hexdigest()
        with self._write_lock:
            db = self.connections.get()
            try:
                old = db.execute(
                    "SELECT id FROM memories WHERE id=? AND user_id=? AND status='active'",
                    (int(memory_id), self.user_id),
                ).fetchone()
                if old is None:
                    raise ValueError("active memory does not exist")
                db.execute("UPDATE memories SET status='superseded',valid_to=?,updated_at=? WHERE id=?",
                           (now, now, int(memory_id)))
                db.execute("UPDATE memories SET index_status='pending' WHERE id=?", (int(memory_id),))
                self._event(int(memory_id), "supersede")
                cursor = db.execute(
                    """INSERT INTO memories(uuid,user_id,layer,kind,content,subject,predicate,importance,
                    confidence,provenance,index_status,occurred_at,valid_from,source_conversation_id,
                    source_message_ids,content_hash,embedding,embedding_model,created_at,updated_at)
                    VALUES(?,?,?,?,?,?,?,?,?,?,'pending',?,?,?,?,?,?,?,?,?)""",
                    (str(uuid.uuid4()), self.user_id, item.layer, item.kind, item.content, item.subject,
                     item.predicate, item.importance, item.confidence, item.provenance, item.occurred_at,
                     item.valid_from or now, item.source_conversation_id, item.source_message_ids, digest,
                     item.embedding, item.embedding_model, now, now),
                )
                new_id = int(cursor.lastrowid)
                db.execute(
                    "INSERT OR IGNORE INTO memory_entities(memory_id,entity_id,role) "
                    "SELECT ?,entity_id,role FROM memory_entities WHERE memory_id=?",
                    (new_id, int(memory_id)),
                )
                db.execute("UPDATE memories SET superseded_by=? WHERE id=?", (new_id, int(memory_id)))
                self._event(new_id, "create")
                db.commit()
                return new_id
            except Exception:
                db.rollback()
                raise

    @staticmethod
    def _validate_memory(item: MemoryInput) -> None:
        if item.layer not in LAYERS:
            raise ValueError("layer must be episodic or semantic")
        if item.kind not in KINDS:
            raise ValueError("unsupported memory kind")
        if item.provenance not in PROVENANCE:
            raise ValueError("unsupported memory provenance")
        if not item.content or len(item.content) > 10000:
            raise ValueError("memory content must be 1 to 10000 characters")
        for candidate_field in (item.content, item.subject or "", item.predicate or ""):
            if scan_secrets(candidate_field).has_secrets:
                raise ValueError("memory fields cannot contain secret-like data")
        if not (0 <= item.importance <= 1 and 0 <= item.confidence <= 1):
            raise ValueError("importance and confidence must be between 0 and 1")

    def get_memory(self, memory_id: int) -> Optional[dict[str, Any]]:
        row = self.connections.get().execute(
            "SELECT * FROM memories WHERE id=? AND user_id=?", (int(memory_id), self.user_id)
        ).fetchone()
        return dict(row) if row else None

    def list_memories(self, layer: Optional[str] = None,
                      status: str = "active") -> list[dict[str, Any]]:
        if status not in MEMORY_STATUSES:
            raise ValueError("unsupported memory status")
        sql = "SELECT * FROM memories WHERE user_id=? AND status=?"
        args: list[Any] = [self.user_id, status]
        if layer is not None:
            if layer not in LAYERS:
                raise ValueError("unsupported memory layer")
            sql += " AND layer=?"
            args.append(layer)
        sql += " ORDER BY id"
        return [dict(row) for row in self.connections.get().execute(sql, args).fetchall()]

    def update_memory(self, memory_id: int, **changes: Any) -> None:
        allowed = {"content", "subject", "predicate", "importance", "confidence", "occurred_at",
                   "valid_from", "valid_to", "status", "embedding", "embedding_model", "index_status"}
        if not changes or set(changes) - allowed:
            raise ValueError("unsupported or empty memory update")
        if "status" in changes and changes["status"] not in MEMORY_STATUSES:
            raise ValueError("unsupported memory status")
        if "index_status" in changes and changes["index_status"] not in {"pending", "indexed"}:
            raise ValueError("unsupported index status")
        for field in ("importance", "confidence"):
            if field in changes and not 0 <= changes[field] <= 1:
                raise ValueError(field + " must be between 0 and 1")
        changes["updated_at"] = utc_now()
        if changes.get("status") and changes["status"] != "active":
            changes["index_status"] = "pending"
        if "content" in changes:
            changes["content_hash"] = hashlib.sha256(changes["content"].strip().encode()).hexdigest()
            changes.setdefault("index_status", "pending")
        with self._write_lock:
            cursor = self.connections.get().execute(
                "UPDATE memories SET " + ",".join(f"{key}=?" for key in changes) +
                " WHERE id=? AND user_id=?",
                (*changes.values(), int(memory_id), self.user_id),
            )
            if cursor.rowcount != 1:
                raise ValueError("memory does not exist")
            self._event(memory_id, "update")
            self.connections.get().commit()

    def delete_memory(self, memory_id: int, hard: bool = False) -> None:
        with self._write_lock:
            db = self.connections.get()
            row = db.execute("SELECT id FROM memories WHERE id=? AND user_id=?",
                             (int(memory_id), self.user_id)).fetchone()
            if row is None:
                raise ValueError("memory does not exist")
            if hard:
                db.execute("DELETE FROM memory_entities WHERE memory_id=?", (int(memory_id),))
                db.execute("DELETE FROM memories WHERE id=? AND user_id=?", (int(memory_id), self.user_id))
                self._event(memory_id, "delete")
                db.commit()
                result = db.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
                if result and result[0] != 0:
                    raise RuntimeError("WAL checkpoint could not complete after hard delete")
            else:
                db.execute("UPDATE memories SET status='deleted',index_status='pending',updated_at=? WHERE id=? AND user_id=?",
                           (utc_now(), int(memory_id), self.user_id))
                self._event(memory_id, "delete")
                db.commit()

    def add_entity(self, canonical_name: str, entity_type: str,
                   aliases: Optional[list[str]] = None) -> int:
        if entity_type not in ENTITY_TYPES or not canonical_name.strip():
            raise ValueError("valid entity name and type are required")
        now = utc_now()
        name = canonical_name.strip()
        with self._write_lock:
            db = self.connections.get()
            existing = db.execute(
                "SELECT id,aliases_json FROM entities WHERE user_id=? AND canonical_name=?",
                (self.user_id, name),
            ).fetchone()
            merged_aliases = list(json.loads(existing["aliases_json"] or "[]")) if existing else []
            for alias in aliases or []:
                clean_alias = alias.strip()
                if clean_alias and clean_alias.casefold() != name.casefold() and all(
                    clean_alias.casefold() != old.casefold() for old in merged_aliases
                ):
                    merged_aliases.append(clean_alias)
            db.execute(
                """INSERT INTO entities(user_id,canonical_name,type,aliases_json,first_seen_at,last_seen_at)
                VALUES(?,?,?,?,?,?) ON CONFLICT(user_id,canonical_name) DO UPDATE SET
                type=excluded.type,aliases_json=excluded.aliases_json,last_seen_at=excluded.last_seen_at,
                mention_count=entities.mention_count+1""",
                (self.user_id, name, entity_type, json.dumps(merged_aliases, ensure_ascii=False), now, now),
            )
            row = self.connections.get().execute(
                "SELECT id FROM entities WHERE user_id=? AND canonical_name=?",
                (self.user_id, name),
            ).fetchone()
            self.connections.get().commit()
            return int(row[0])

    def link_entity(self, memory_id: int, entity_id: int, role: str = "mentioned") -> None:
        with self._write_lock:
            self.connections.get().execute(
                "INSERT OR IGNORE INTO memory_entities(memory_id,entity_id,role) VALUES(?,?,?)",
                (int(memory_id), int(entity_id), role),
            )
            self.connections.get().commit()

    def link_conversation_entity(self, conversation_id: int, entity_id: int,
                                 message_id: Optional[int] = None) -> None:
        with self._write_lock:
            self.connections.get().execute(
                """INSERT INTO conversation_entities(conversation_id,entity_id,last_referenced_at,
                last_referenced_message_id,reference_count) SELECT c.id,e.id,?,?,1 FROM conversations c,
                entities e WHERE c.id=? AND c.user_id=? AND e.id=? AND e.user_id=?
                ON CONFLICT(conversation_id,entity_id) DO UPDATE SET
                last_referenced_at=excluded.last_referenced_at,
                last_referenced_message_id=excluded.last_referenced_message_id,
                reference_count=conversation_entities.reference_count+1""",
                (utc_now(), message_id, conversation_id, self.user_id, entity_id, self.user_id),
            )
            self.connections.get().commit()

    def recent_conversation_entities(self, conversation_id: int,
                                     limit: int = 12) -> list[dict[str, Any]]:
        rows = self.connections.get().execute(
            """SELECT e.*,ce.last_referenced_at,ce.last_referenced_message_id,ce.reference_count
            FROM conversation_entities ce JOIN entities e ON e.id=ce.entity_id
            WHERE ce.conversation_id=? AND e.user_id=?
            ORDER BY ce.last_referenced_at DESC,ce.reference_count DESC LIMIT ?""",
            (conversation_id, self.user_id, max(0, int(limit))),
        ).fetchall()
        return [dict(row) for row in rows]

    def memory_entities(self, memory_id: int) -> list[dict[str, Any]]:
        rows = self.connections.get().execute(
            """SELECT e.*,me.role FROM memory_entities me JOIN entities e ON e.id=me.entity_id
            WHERE me.memory_id=? AND e.user_id=? ORDER BY e.id""",
            (int(memory_id), self.user_id),
        ).fetchall()
        return [dict(row) for row in rows]

    def get_pending_question(self, conversation_id: int,
                             kind: Optional[str] = None) -> Optional[dict[str, Any]]:
        sql = "SELECT * FROM pending_questions WHERE conversation_id=? AND status='pending'"
        params: list[Any] = [conversation_id]
        if kind:
            sql += " AND kind=?"
            params.append(kind)
        sql += " ORDER BY id LIMIT 1"
        row = self.connections.get().execute(sql, params).fetchone()
        return dict(row) if row else None

    def resolve_pending_question(self, question_id: int) -> None:
        with self._write_lock:
            self.connections.get().execute(
                "UPDATE pending_questions SET status='resolved' WHERE id=?", (int(question_id),))
            self.connections.get().commit()

    def add_pending_confirmation(self, conversation_id: int, candidate: dict[str, Any],
                                 reason: str, expires_after_turn: int,
                                 asked: bool = False, current_turn_offset: int = 0) -> int:
        current_turn = self.count_user_turns(conversation_id)
        expiry_turn = current_turn + max(1, int(expires_after_turn)) + max(0, int(current_turn_offset))
        with self._write_lock:
            cursor = self.connections.get().execute(
                """INSERT INTO pending_confirmations(conversation_id,candidate_json,reason,created_at,
                expires_after_turn,status) SELECT id,?,?,?,?,'pending' FROM conversations
                WHERE id=? AND user_id=?""",
                (json.dumps(candidate, ensure_ascii=False), reason, utc_now(), expiry_turn,
                 conversation_id, self.user_id),
            )
            if cursor.rowcount != 1:
                raise ValueError("conversation does not exist")
            if asked:
                self.connections.get().execute(
                    "UPDATE pending_confirmations SET status='asked' WHERE id=?", (cursor.lastrowid,))
            self.connections.get().commit()
            return int(cursor.lastrowid)

    def add_pending_question(self, conversation_id: int, kind: str,
                             payload: dict[str, Any]) -> int:
        with self._write_lock:
            cursor = self.connections.get().execute(
                """INSERT INTO pending_questions(conversation_id,kind,payload_json,status,created_at)
                SELECT id,?,?,'pending',? FROM conversations WHERE id=? AND user_id=?""",
                (kind, json.dumps(payload, ensure_ascii=False), utc_now(), conversation_id, self.user_id),
            )
            if cursor.rowcount != 1:
                raise ValueError("conversation does not exist")
            self.connections.get().commit()
            return int(cursor.lastrowid)

    def _event(self, memory_id: int, action: str) -> None:
        # Keep audit details content-free, especially for deletion.
        self.connections.get().execute(
            "INSERT INTO memory_events(memory_id,action,at,detail_json) VALUES(?,?,?,'{}')",
            (int(memory_id), action, utc_now()),
        )
