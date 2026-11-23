"""The early save/search tool set for the Phase 3 Jarvis slice."""

from __future__ import annotations

from typing import Optional

from ..db.repository import MemoryRepository
from ..extraction.secrets import scan_secrets
from ..integration.jarvis_adapter import JarvisMemoryAdapter
from ..models import KINDS, LAYERS, MemoryInput
from ..extraction.sensitivity import sensitive_categories
from ..vector.sync import VectorSynchronizer


def build_memory_tools(repository: MemoryRepository, synchronizer: VectorSynchronizer,
                       adapter: JarvisMemoryAdapter):
    """Create dependency-injected, validated smolagents memory tools."""
    from smolagents import tool

    @tool
    def save_memory(content: str, layer: str = "semantic", kind: str = "profile_fact",
                    subject: str = "", predicate: str = "") -> str:
        """Save a user-requested memory after scanning for secrets.

        Args:
            content: One concise fact or dated event about the user.
            layer: episodic or semantic.
            kind: One supported memory kind such as preference, skill, event, or project.
            subject: Optional subject for semantic facts.
            predicate: Optional semantic predicate such as current_employer.
        """
        try:
            if not isinstance(content, str) or not content.strip() or len(content) > 10000:
                return "Memory not saved: content must contain 1 to 10000 characters."
            if layer not in LAYERS or kind not in KINDS:
                return "Memory not saved: unsupported layer or kind."
            scan = scan_secrets(content)
            if scan.has_secrets:
                return "Memory not saved because it contains secret-like data. Remove it and try again."
            categories = sensitive_categories(content)
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
                                        if adapter.pending_confirmations is not None else 4),
                    asked=True,
                    current_turn_offset=1,
                )
                return ("This may be sensitive information, so I have not saved it as a memory. "
                        "Want me to remember that? (confirmation %d)" % pending_id)
            memory_id = repository.create_memory(MemoryInput(
                layer=layer, kind=kind, content=content.strip(), subject=subject or None,
                predicate=predicate or None, provenance="user_stated",
            ))
            synchronizer.index_memory(memory_id)
            return "Memory saved (ID %d)." % memory_id
        except Exception as exc:
            return "Memory could not be saved (%s)." % type(exc).__name__

    @tool
    def search_memory(query: str, top_k: int = 5, layer: Optional[str] = None,
                      kind: Optional[str] = None) -> str:
        """Search memories by meaning and return matching saved facts.

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
            if not rows:
                return "No matching memories found."
            return "\n".join("ID %d | %s | %s" % (
                row["id"], row.get("occurred_at") or row.get("updated_at") or "date unknown",
                row["content"]) for row in rows)
        except Exception as exc:
            return "Search failed (%s)." % type(exc).__name__

    return [save_memory, search_memory]
