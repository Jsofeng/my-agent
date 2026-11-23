"""User-controlled memory inspector actions and Gradio tab construction."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Optional

from .db.repository import MemoryRepository, utc_now
from .extraction.secrets import scan_secrets
from .extraction.sensitivity import sensitive_categories
from .models import MemoryInput
from .vector.sync import VectorSynchronizer


def _display_memory(row: dict) -> dict:
    return {key: value for key, value in row.items() if key not in {"embedding"}}


def browse_memories(repository: MemoryRepository, layer: str = "all", kind: str = "all",
                    entity: str = "", status: str = "active", page: int = 1,
                    page_size: int = 50) -> list[dict]:
    """Return a validated page of inspector rows without embedding blobs."""
    if layer not in {"all", "episodic", "semantic"}:
        raise ValueError("unsupported layer")
    if kind != "all" and kind not in {"event", "decision", "achievement", "experience", "preference",
                                     "skill", "interest", "goal", "relationship", "project", "profile_fact"}:
        raise ValueError("unsupported kind")
    if status not in {"active", "superseded", "archived", "deleted"}:
        raise ValueError("unsupported status")
    if len(entity) > 200 or page < 1 or not 1 <= page_size <= 100:
        raise ValueError("invalid entity, page, or page size")
    sql = "SELECT DISTINCT m.* FROM memories m"
    params: list[object] = []
    if entity.strip():
        sql += " JOIN memory_entities me ON me.memory_id=m.id JOIN entities e ON e.id=me.entity_id " \
               "AND e.user_id=? AND (e.canonical_name=? COLLATE NOCASE OR lower(e.aliases_json) LIKE ?)"
        params.extend([repository.user_id, entity.strip(), "%\"" + entity.strip().lower() + "\"%"])
    sql += " WHERE m.user_id=? AND m.status=?"
    params.extend([repository.user_id, status])
    if layer != "all":
        sql += " AND m.layer=?"
        params.append(layer)
    if kind != "all":
        sql += " AND m.kind=?"
        params.append(kind)
    sql += " ORDER BY m.id DESC LIMIT ? OFFSET ?"
    params.extend([page_size, (page - 1) * page_size])
    rows = repository.connections.get().execute(sql, params).fetchall()
    return [_display_memory(dict(row)) for row in rows]


def memory_history(repository: MemoryRepository, memory_id: int) -> list[dict]:
    """Return one supersede chain from its oldest active or historical root."""
    row = repository.get_memory(memory_id)
    if row is None:
        raise ValueError("memory does not exist")
    root = row
    while root.get("id"):
        previous = repository.connections.get().execute(
            "SELECT * FROM memories WHERE user_id=? AND superseded_by=? ORDER BY id LIMIT 1",
            (repository.user_id, root["id"]),
        ).fetchone()
        if previous is None:
            break
        root = dict(previous)
    chain: list[dict] = []
    current = root
    seen: set[int] = set()
    while current and int(current["id"]) not in seen:
        seen.add(int(current["id"]))
        chain.append(_display_memory(current))
        next_id = current.get("superseded_by")
        current = repository.get_memory(int(next_id)) if next_id is not None else None
    return chain


def export_memories(repository: MemoryRepository) -> str:
    """Write a fresh JSON export of current SQLite state to a temporary file."""
    db = repository.connections.get()
    data = {
        "exported_at": utc_now(),
        "memories": [_display_memory(dict(row)) for row in db.execute(
            "SELECT * FROM memories WHERE user_id=? ORDER BY id", (repository.user_id,)).fetchall()],
        "entities": [dict(row) for row in db.execute(
            "SELECT * FROM entities WHERE user_id=? ORDER BY id", (repository.user_id,)).fetchall()],
        "memory_entities": [dict(row) for row in db.execute(
            "SELECT me.* FROM memory_entities me JOIN memories m ON m.id=me.memory_id "
            "WHERE m.user_id=? ORDER BY me.memory_id,me.entity_id", (repository.user_id,)).fetchall()],
        "memory_events": [dict(row) for row in db.execute(
            "SELECT ev.* FROM memory_events ev JOIN memories m ON m.id=ev.memory_id "
            "WHERE m.user_id=? ORDER BY ev.id", (repository.user_id,)).fetchall()],
    }
    descriptor, filename = tempfile.mkstemp(prefix="edith-export-", suffix=".json")
    os.close(descriptor)
    path = Path(filename)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(path)


def _inspector_actions(repository: MemoryRepository, synchronizer: VectorSynchronizer,
                       adapter, consolidator):
    def browse(layer, kind, entity, status, page):
        try:
            return browse_memories(repository, layer, kind, entity, status, int(page or 1))
        except Exception as exc:
            return [{"error": type(exc).__name__}]

    def history(memory_id):
        try:
            return memory_history(repository, int(memory_id))
        except Exception as exc:
            return {"error": type(exc).__name__}

    def edit(memory_id, new_content):
        try:
            memory_id = int(memory_id)
            old = repository.get_memory(memory_id)
            if old is None or old["status"] != "active":
                return "Active memory not found. Refresh and select an active ID."
            if not isinstance(new_content, str) or not new_content.strip() or len(new_content) > 10000:
                return "Edited content must contain 1 to 10000 characters."
            if scan_secrets(new_content).has_secrets:
                return "Edit rejected because it contains secret-like data."
            if adapter is not None:
                adapter._ensure_ready()
            categories = sensitive_categories(" ".join((new_content, old["subject"] or "",
                                                          old["predicate"] or "")))
            if categories:
                conversation_id = getattr(adapter, "conversation_id", None)
                if conversation_id is None:
                    return "Sensitive edit was not saved; no active conversation is available for confirmation."
                repository.add_pending_confirmation(
                    conversation_id,
                    {"layer": old["layer"], "kind": old["kind"], "content": new_content.strip(),
                     "subject": old["subject"], "predicate": old["predicate"],
                     "supersedes_id": memory_id},
                    "sensitive_category:" + ",".join(categories), 4, asked=False)
                return "Sensitive edit was held for confirmation in the chat."
            item = MemoryInput(layer=old["layer"], kind=old["kind"], content=new_content.strip(),
                               subject=old["subject"], predicate=old["predicate"],
                               importance=float(old["importance"]), confidence=float(old["confidence"]),
                               provenance="user_stated", source_conversation_id=getattr(adapter, "conversation_id", None),
                               occurred_at=old["occurred_at"])
            new_id = repository.supersede_memory(memory_id, item)
            synchronizer.remove_from_index(memory_id)
            synchronizer.index_memory(new_id)
            return "Saved corrected memory %d. Previous fact %d remains in history." % (new_id, memory_id)
        except Exception as exc:
            return "Edit failed (%s)." % type(exc).__name__

    def delete(memory_id):
        try:
            memory_id = int(memory_id)
            if adapter is not None:
                adapter._ensure_ready()
            synchronizer.delete_memory(memory_id, hard=True)
            return "Memory %d deleted." % memory_id
        except Exception as exc:
            return "Delete failed (%s)." % type(exc).__name__

    def export():
        try:
            return export_memories(repository)
        except Exception as exc:
            return None

    def integrity():
        try:
            if adapter is not None:
                adapter._ensure_ready()
            return synchronizer.verify_integrity()
        except Exception as exc:
            return {"error": type(exc).__name__}

    def rebuild():
        try:
            result = synchronizer.rebuild_index()
            if adapter is not None:
                adapter._ready = True
            return result
        except Exception as exc:
            return {"error": type(exc).__name__}

    def vacuum():
        try:
            db = repository.connections.get()
            with repository._write_lock:
                db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                db.execute("VACUUM")
            return "SQLite VACUUM completed."
        except Exception as exc:
            return "VACUUM failed (%s)." % type(exc).__name__

    def consolidate():
        try:
            return {"queued": consolidator.enqueue(),
                    "message": "Consolidation queued or already running."}
        except Exception as exc:
            return {"error": type(exc).__name__}

    def review_flags():
        try:
            rows = repository.connections.get().execute(
                "SELECT m.id,m.kind,m.content,m.updated_at,m.access_count FROM memories m "
                "JOIN memory_events ev ON ev.memory_id=m.id WHERE m.user_id=? AND m.layer='semantic' "
                "AND m.status='active' AND ev.action='stale_review' ORDER BY ev.id DESC",
                (repository.user_id,),
            ).fetchall()
            unique = {int(row["id"]): dict(row) for row in rows}
            return list(unique.values()) or [{"message": "No stale semantic facts are flagged."}]
        except Exception as exc:
            return [{"error": type(exc).__name__}]

    return browse, history, edit, delete, export, integrity, rebuild, vacuum, consolidate, review_flags


def build_memory_inspector_tab(gr, repository: Optional[MemoryRepository],
                               synchronizer: Optional[VectorSynchronizer], adapter=None,
                               consolidator=None) -> None:
    """Render the Edith user-control tab; no Edith state is opened when disabled."""
    with gr.Tab("Memory Inspector"):
        gr.Markdown("### Jarvis memory controls\nBrowse, correct, delete, and export stored memories.")
        if repository is None or synchronizer is None:
            gr.Markdown("Memory storage is disabled.")
            return
        from .consolidation.consolidator import MemoryConsolidator
        actions = _inspector_actions(repository, synchronizer, adapter,
                                     consolidator or MemoryConsolidator(repository, synchronizer))
        browse, history, edit, delete, export, integrity, rebuild, vacuum, consolidate, review_flags = actions
        with gr.Row():
            layer = gr.Dropdown(["all", "semantic", "episodic"], value="all", label="Layer")
            kind = gr.Dropdown(["all", "event", "decision", "achievement", "experience", "preference",
                                "skill", "interest", "goal", "relationship", "project", "profile_fact"],
                               value="all", label="Kind")
            entity = gr.Textbox(label="Entity or alias filter")
            status = gr.Dropdown(["active", "superseded", "archived", "deleted"], value="active", label="Status")
            page = gr.Number(value=1, precision=0, label="Page")
        browse_button = gr.Button("Refresh memories")
        memories = gr.JSON(label="Memories")
        browse_button.click(browse, [layer, kind, entity, status, page], memories)
        with gr.Row():
            memory_id = gr.Number(value=None, precision=0, label="Memory ID")
            history_button = gr.Button("View history")
            history_output = gr.JSON(label="Superseded history")
            history_button.click(history, memory_id, history_output)
        with gr.Row():
            new_content = gr.Textbox(label="Correct memory content", lines=2)
            edit_button = gr.Button("Save correction")
            delete_button = gr.Button("Hard delete selected memory")
        action_status = gr.Textbox(label="Action result", interactive=False)
        edit_button.click(edit, [memory_id, new_content], action_status).then(
            browse, [layer, kind, entity, status, page], memories)
        delete_button.click(delete, memory_id, action_status).then(
            browse, [layer, kind, entity, status, page], memories)
        with gr.Row():
            export_button = gr.Button("Export current memory data as JSON")
            export_file = gr.File(label="JSON export")
            export_button.click(export, outputs=export_file)
        with gr.Row():
            check_button = gr.Button("Check index integrity")
            rebuild_button = gr.Button("Rebuild vector index")
            consolidate_button = gr.Button("Run consolidation")
            vacuum_button = gr.Button("Run optional VACUUM")
            review_button = gr.Button("Show stale fact review flags")
        maintenance_result = gr.JSON(label="Maintenance result")
        check_button.click(integrity, outputs=maintenance_result)
        rebuild_button.click(rebuild, outputs=maintenance_result)
        consolidate_button.click(consolidate, outputs=maintenance_result)
        review_button.click(review_flags, outputs=maintenance_result)
        vacuum_button.click(vacuum, outputs=action_status)
