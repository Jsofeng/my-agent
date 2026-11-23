"""Validated, dependency-injected smolagents tools for Edith memory control."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

from ..db.repository import MemoryRepository
from ..extraction.secrets import scan_secrets
from ..extraction.sensitivity import sensitive_categories
from ..integration.jarvis_adapter import JarvisMemoryAdapter
from ..models import KINDS, LAYERS, MEMORY_STATUSES, MemoryInput
from ..updating.conflicts import SemanticConflictService
from ..vector.sync import VectorSynchronizer


def build_memory_tools(repository: MemoryRepository, synchronizer: VectorSynchronizer,
                       adapter: JarvisMemoryAdapter):
    """Build the complete user memory tool set with validation and fail-safe errors."""
    from smolagents import tool

    conflicts = adapter.conflict_service or SemanticConflictService(
        repository, synchronizer=synchronizer)

    @tool
    def save_memory(content: str, layer: str = "semantic", kind: str = "profile_fact",
                    subject: str = "", predicate: str = "") -> str:
        """Save one durable user fact through screening, confirmation, dedup, and conflict checks.

        Args:
            content: One concise fact or dated event that the user asked to save.
            layer: episodic or semantic.
            kind: A supported event or semantic memory kind.
            subject: Optional fact subject, such as user or a named project.
            predicate: Optional field name such as current_employer or skill.
        """
        try:
            if not isinstance(content, str) or not content.strip() or len(content) > 10000:
                return "Memory not saved: content must contain 1 to 10000 characters."
            if layer not in LAYERS or kind not in KINDS:
                return "Memory not saved: unsupported layer or kind."
            if len(subject) > 200 or len(predicate) > 200:
                return "Memory not saved: subject and predicate are limited to 200 characters."
            if scan_secrets(" ".join((content, subject, predicate))).has_secrets:
                return "Memory not saved because it contains secret-like data. Remove it and try again."
            categories = sensitive_categories(" ".join((content, subject, predicate)))
            if categories:
                conversation_id = adapter.conversation_id
                if conversation_id is None:
                    return "Sensitive memory not saved because no active conversation is available."
                pending_id = repository.add_pending_confirmation(
                    conversation_id,
                    {"layer": layer, "kind": kind, "content": content.strip(),
                     "subject": subject or None, "predicate": predicate or None},
                    "sensitive_category:" + ",".join(categories),
                    expires_after_turn=(adapter.pending_confirmations.ttl_turns
                                        if adapter.pending_confirmations else 4),
                    asked=True, current_turn_offset=1,
                )
                return "Sensitive candidate not stored. Ask once: Want me to remember that? (confirmation %d)" % pending_id
            item = MemoryInput(layer=layer, kind=kind, content=content.strip(),
                               subject=subject or None, predicate=predicate or None,
                               provenance="user_stated",
                               source_conversation_id=adapter.conversation_id)
            relation, memory_id = conflicts.save(item)
            if relation == "pending":
                return "Memory not changed yet; a clarification is pending for this conflicting fact."
            return "Memory saved (ID %d)." % memory_id
        except Exception as exc:
            return "Memory could not be saved (%s)." % type(exc).__name__

    @tool
    def search_memory(query: str, top_k: int = 5, layer: Optional[str] = None,
                      kind: Optional[str] = None) -> str:
        """Search active memories by meaning and return IDs for later edits or deletion.

        Args:
            query: A short natural-language search query.
            top_k: Maximum results, from 1 to 20.
            layer: Optional episodic or semantic filter.
            kind: Optional supported memory-kind filter.
        """
        try:
            if not isinstance(query, str) or not query.strip() or len(query) > 1000:
                return "Search failed: query must contain 1 to 1000 characters."
            if not isinstance(top_k, int) or top_k < 1 or top_k > 20:
                return "Search failed: top_k must be between 1 and 20."
            if layer is not None and layer not in LAYERS:
                return "Search failed: unsupported layer."
            if kind is not None and kind not in KINDS:
                return "Search failed: unsupported memory kind."
            adapter._ensure_ready()
            rows = synchronizer.search(query, top_k=top_k, layer=layer, kind=kind)
            return ("\n".join("ID %d | %s | %s" % (
                row["id"], row.get("occurred_at") or row.get("updated_at") or "date unknown",
                row["content"]) for row in rows) if rows else "No matching memories found.")
        except Exception as exc:
            return "Search failed (%s)." % type(exc).__name__

    @tool
    def update_memory(memory_id: int, new_content: str) -> str:
        """Correct a saved memory, preserving its old value in the superseded history chain.

        Args:
            memory_id: Existing active ID returned by search_memory or list_memories.
            new_content: Corrected content supplied or explicitly confirmed by the user.
        """
        try:
            if not isinstance(memory_id, int) or memory_id < 1:
                return "Update failed: memory_id must be a positive integer."
            if not isinstance(new_content, str) or not new_content.strip() or len(new_content) > 10000:
                return "Update failed: new_content must contain 1 to 10000 characters."
            if scan_secrets(new_content).has_secrets:
                return "Update failed because the new content contains secret-like data."
            old = repository.get_memory(memory_id)
            if old is None or old["status"] != "active":
                return "Update failed: active memory does not exist. Search for it again."
            categories = sensitive_categories(" ".join((new_content, old["subject"] or "",
                                                          old["predicate"] or "")))
            if categories:
                conversation_id = adapter.conversation_id
                if conversation_id is None:
                    return "Sensitive correction not saved; no active conversation is available."
                pending_id = repository.add_pending_confirmation(
                    conversation_id,
                    {"layer": old["layer"], "kind": old["kind"], "content": new_content.strip(),
                     "subject": old["subject"], "predicate": old["predicate"],
                     "supersedes_id": memory_id},
                    "sensitive_category:" + ",".join(categories), 4, asked=True,
                    current_turn_offset=1)
                return "Sensitive correction not stored. Ask once: Want me to remember that? (confirmation %d)" % pending_id
            item = MemoryInput(
                layer=old["layer"], kind=old["kind"], content=new_content.strip(),
                subject=old["subject"], predicate=old["predicate"],
                importance=float(old["importance"]), confidence=max(float(old["confidence"]), 0.8),
                provenance="user_stated", source_conversation_id=adapter.conversation_id,
                occurred_at=old["occurred_at"],
            )
            new_id = repository.supersede_memory(memory_id, item)
            synchronizer.remove_from_index(memory_id)
            synchronizer.index_memory(new_id)
            return "Memory %d corrected; history preserved as superseded ID %d." % (new_id, memory_id)
        except Exception as exc:
            return "Update failed (%s)." % type(exc).__name__

    @tool
    def delete_memory(memory_id: int) -> str:
        """Hard-delete a memory only when the user explicitly asked to forget or delete it.

        Args:
            memory_id: Existing memory ID returned by search_memory or list_memories.
        """
        try:
            if not isinstance(memory_id, int) or memory_id < 1:
                return "Delete failed: memory_id must be a positive integer."
            if repository.get_memory(memory_id) is None:
                return "Delete failed: memory does not exist. Search for it again."
            synchronizer.delete_memory(memory_id, hard=True)
            return "Memory %d was deleted." % memory_id
        except Exception as exc:
            return "Delete failed (%s)." % type(exc).__name__

    @tool
    def get_user_profile() -> str:
        """Return the current active semantic facts grouped by memory kind.

        Returns:
            Current facts with confidence and last-updated dates, or a clear empty-profile message.
        """
        try:
            rows = repository.connections.get().execute(
                "SELECT id,kind,content,confidence,updated_at FROM memories "
                "WHERE user_id=? AND layer='semantic' AND status='active' ORDER BY kind,updated_at DESC",
                (repository.user_id,),
            ).fetchall()
            if not rows:
                return "I don't have any saved profile facts yet."
            grouped: dict[str, list[str]] = {}
            for row in rows:
                grouped.setdefault(row["kind"], []).append(
                    "ID %d | %s | confidence %.2f | updated %s" %
                    (row["id"], row["content"], row["confidence"], row["updated_at"]))
            return "\n".join("%s:\n%s" % (kind, "\n".join(values))
                             for kind, values in grouped.items())
        except Exception as exc:
            return "Profile lookup failed (%s)." % type(exc).__name__

    @tool
    def get_recent_events(days: int = 30, top_k: int = 10) -> str:
        """List recent active episodic events in date order.

        Args:
            days: Look-back window from 1 to 3650 days.
            top_k: Maximum results from 1 to 50.
        """
        try:
            if not isinstance(days, int) or not 1 <= days <= 3650:
                return "Event lookup failed: days must be between 1 and 3650."
            if not isinstance(top_k, int) or not 1 <= top_k <= 50:
                return "Event lookup failed: top_k must be between 1 and 50."
            since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat().replace("+00:00", "Z")
            rows = repository.connections.get().execute(
                "SELECT id,content,occurred_at,updated_at FROM memories WHERE user_id=? "
                "AND layer='episodic' AND status='active' AND occurred_at>=? "
                "ORDER BY occurred_at DESC LIMIT ?", (repository.user_id, since, top_k),
            ).fetchall()
            return ("\n".join("ID %d | %s | %s" %
                               (r["id"], r["occurred_at"], r["content"]) for r in rows)
                    if rows else "No recent saved events found.")
        except Exception as exc:
            return "Event lookup failed (%s)." % type(exc).__name__

    @tool
    def get_project_context(project_name: str) -> str:
        """Return active facts and recent episodes linked to a named project or alias.

        Args:
            project_name: Canonical project name or a stored alias, up to 200 characters.
        """
        try:
            if not isinstance(project_name, str) or not project_name.strip() or len(project_name) > 200:
                return "Project lookup failed: provide a name of 1 to 200 characters."
            entities = repository.connections.get().execute(
                "SELECT * FROM entities WHERE user_id=? AND (canonical_name=? COLLATE NOCASE OR "
                "lower(aliases_json) LIKE ?) ORDER BY id",
                (repository.user_id, project_name.strip(), "%\"" + project_name.strip().lower() + "\"%"),
            ).fetchall()
            if not entities:
                return "No saved project context found for that name."
            ids = [int(row["id"]) for row in entities]
            marks = ",".join("?" for _ in ids)
            rows = repository.connections.get().execute(
                "SELECT DISTINCT m.id,m.layer,m.content,m.occurred_at,m.updated_at FROM memories m "
                "JOIN memory_entities me ON me.memory_id=m.id WHERE m.user_id=? AND m.status='active' "
                "AND me.entity_id IN (" + marks + ") ORDER BY COALESCE(m.occurred_at,m.updated_at) DESC LIMIT 30",
                [repository.user_id] + ids,
            ).fetchall()
            names = ", ".join(row["canonical_name"] for row in entities)
            return "Project: %s\n%s" % (names, "\n".join(
                "ID %d | %s | %s | %s" % (r["id"], r["layer"], r["occurred_at"] or r["updated_at"], r["content"])
                for r in rows) if rows else "No linked memories found.")
        except Exception as exc:
            return "Project lookup failed (%s)." % type(exc).__name__

    @tool
    def list_memories(page: int = 1, page_size: int = 20, layer: Optional[str] = None,
                      kind: Optional[str] = None, entity: str = "",
                      status: str = "active") -> str:
        """List memories in pages with optional layer, kind, entity, and status filters.

        Args:
            page: One-based page number.
            page_size: Page size from 1 to 100.
            layer: Optional episodic or semantic filter.
            kind: Optional supported memory kind.
            entity: Optional canonical entity name or alias, up to 200 characters.
            status: active, superseded, archived, or deleted.
        """
        try:
            if not isinstance(page, int) or page < 1 or not isinstance(page_size, int) or not 1 <= page_size <= 100:
                return "List failed: page must be positive and page_size must be 1 to 100."
            if layer is not None and layer not in LAYERS:
                return "List failed: unsupported layer."
            if kind is not None and kind not in KINDS:
                return "List failed: unsupported kind."
            if status not in MEMORY_STATUSES:
                return "List failed: unsupported status."
            if len(entity) > 200:
                return "List failed: entity filter is limited to 200 characters."
            sql = "SELECT DISTINCT m.* FROM memories m"
            params: list[object] = [repository.user_id, status]
            if entity.strip():
                sql += " JOIN memory_entities me ON me.memory_id=m.id JOIN entities e ON e.id=me.entity_id"
                sql += " AND (e.canonical_name=? COLLATE NOCASE OR lower(e.aliases_json) LIKE ?)"
                params.extend([entity.strip(), "%\"" + entity.strip().lower() + "\"%"])
            sql += " WHERE m.user_id=? AND m.status=?"
            # For entity join predicates are already appended before WHERE; reorder parameters.
            if entity.strip():
                entity_params = params[2:]
                params = [repository.user_id, status] + entity_params
                # Rewrite the join predicate aliases as WHERE constraints, avoiding parameter-order ambiguity.
                sql = "SELECT DISTINCT m.* FROM memories m JOIN memory_entities me ON me.memory_id=m.id " \
                      "JOIN entities e ON e.id=me.entity_id WHERE e.user_id=? " \
                      "AND (e.canonical_name=? COLLATE NOCASE OR lower(e.aliases_json) LIKE ?) " \
                      "AND m.user_id=? AND m.status=?"
                params = [repository.user_id, entity.strip(), "%\"" + entity.strip().lower() + "\"%",
                          repository.user_id, status]
            if layer is not None:
                sql += " AND m.layer=?"
                params.append(layer)
            if kind is not None:
                sql += " AND m.kind=?"
                params.append(kind)
            sql += " ORDER BY m.id DESC LIMIT ? OFFSET ?"
            params.extend([page_size, (page - 1) * page_size])
            rows = repository.connections.get().execute(sql, params).fetchall()
            return ("\n".join("ID %d | %s | %s | %s" %
                               (r["id"], r["status"], r["layer"], r["content"]) for r in rows)
                    if rows else "No memories found on that page.")
        except Exception as exc:
            return "List failed (%s)." % type(exc).__name__

    @tool
    def consolidate_memories() -> str:
        """Queue an on-demand consolidation in the background; it never blocks the chat response.

        Returns:
            Whether a consolidation was queued or one is already running.
        """
        try:
            if adapter.consolidator is None:
                from ..consolidation.consolidator import MemoryConsolidator
                adapter.consolidator = MemoryConsolidator(repository, synchronizer)
            queued = adapter.consolidator.enqueue()
            return ("Consolidation queued in the background." if queued else
                    "A consolidation is already running; this request was coalesced.")
        except Exception as exc:
            return "Consolidation could not be queued (%s)." % type(exc).__name__

    return [save_memory, search_memory, update_memory, delete_memory, get_user_profile,
            get_recent_events, get_project_context, list_memories, consolidate_memories]
