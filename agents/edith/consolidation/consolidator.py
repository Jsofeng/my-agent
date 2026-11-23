"""Idempotent, reversible consolidation over SQLite source memories."""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone

import numpy as np

from ..config import EdithConfig, config
from ..db.repository import MemoryRepository, utc_now
from ..models import MemoryInput
from ..vector.sync import VectorSynchronizer

logger = logging.getLogger(__name__)

class MemoryConsolidator:
    def __init__(self, repository: MemoryRepository, synchronizer: VectorSynchronizer | None = None,
                 settings: EdithConfig = config):
        self.repository = repository
        self.synchronizer = synchronizer
        self.settings = settings
        self._background_lock = threading.Lock()
        self._background_active = False

    def run(self) -> dict[str, int | str]:
        started = utc_now()
        with self.repository._write_lock:
            cursor = self.repository.connections.get().execute(
                "INSERT INTO consolidation_runs(started_at,status,stats_json) VALUES(?,'running','{}')",
                (started,),
            )
            run_id = int(cursor.lastrowid)
            self.repository.connections.get().commit()
        try:
            archived = 0
            summaries = 0
            merged = 0
            db = self.repository.connections.get()
            # Merge only same-subject, same-kind semantic rows with stored embeddings.
            semantic_rows = db.execute(
                "SELECT id,kind,subject,importance,confidence,embedding FROM memories "
                "WHERE user_id=? AND layer='semantic' AND status='active' AND embedding IS NOT NULL "
                "ORDER BY id",
                (self.repository.user_id,),
            ).fetchall()
            surviving: dict[tuple[str, str], list[tuple[int, np.ndarray]]] = {}
            for row in semantic_rows:
                key = (str(row["kind"]), str(row["subject"] or ""))
                vector = np.frombuffer(row["embedding"], dtype=np.float32)
                target_id = None
                for survivor_id, survivor_vector in surviving.get(key, []):
                    if vector.size == survivor_vector.size and vector.size:
                        similarity = float(np.dot(vector, survivor_vector) /
                                           max(float(np.linalg.norm(vector) * np.linalg.norm(survivor_vector)), 1e-12))
                        if similarity >= self.settings.dup_threshold:
                            target_id = survivor_id
                            break
                if target_id is None:
                    surviving.setdefault(key, []).append((int(row["id"]), vector))
                    continue
                self.repository.merge_duplicate(int(row["id"]), target_id)
                if self.synchronizer:
                    self.synchronizer.remove_from_index(int(row["id"]))
                merged += 1

            # Group only unarchived episodic events; each source is retained and linked.
            rows = db.execute(
                "SELECT * FROM memories WHERE user_id=? AND layer='episodic' AND status='active' "
                "AND kind IN ('event','experience','achievement') ORDER BY subject,predicate,occurred_at,id",
                (self.repository.user_id,),
            ).fetchall()
            groups: dict[tuple[str, str], list[dict]] = {}
            for row in rows:
                item = dict(row)
                key = (item.get("subject") or "user", item.get("predicate") or item["kind"])
                groups.setdefault(key, []).append(item)
            for (subject, predicate), group in groups.items():
                if len(group) < 2:
                    continue
                # A prior source-to-summary link makes the operation idempotent.
                ungrouped = [item for item in group if item.get("consolidated_into") is None]
                if len(ungrouped) < 2:
                    continue
                source_ids = [int(item["id"]) for item in ungrouped]
                text = "Related events: " + " | ".join(
                    (item.get("occurred_at") or "date unknown") + " — " + item["content"]
                    for item in ungrouped
                )
                summary_id = self.repository.consolidate_memories(MemoryInput(
                    layer="episodic", kind="experience", content=text[:10000], subject=subject,
                    predicate=predicate, importance=max(float(item["importance"]) for item in ungrouped),
                    confidence=min(float(item["confidence"]) for item in ungrouped),
                    provenance="inferred", occurred_at=ungrouped[-1].get("occurred_at"),
                ), source_ids)
                if self.synchronizer:
                    for source_id in source_ids:
                        self.synchronizer.remove_from_index(source_id)
                    self.synchronizer.index_memory(summary_id)
                archived += len(source_ids)
                summaries += 1

            # Importance decays gently with age while preserving access reinforcement.
            now = datetime.now(timezone.utc)
            active = db.execute(
                "SELECT id,layer,kind,subject,predicate,importance,confidence,access_count,updated_at "
                "FROM memories WHERE user_id=? AND status='active'",
                (self.repository.user_id,),
            ).fetchall()
            stale_flagged = 0
            for row in active:
                try:
                    age_days = max(0.0, (now - datetime.fromisoformat(
                        str(row["updated_at"]).replace("Z", "+00:00")).astimezone(timezone.utc)).total_seconds() / 86400)
                except (TypeError, ValueError):
                    age_days = 0.0
                decay = self.settings.importance_annual_decay ** (age_days / 365.0)
                use_boost = min(0.15, 0.02 * int(row["access_count"]))
                score = min(1.0, max(0.0, float(row["importance"]) * decay + use_boost))
                stale = (row["layer"] == "semantic"
                         and age_days >= self.settings.stale_fact_days
                         and int(row["access_count"]) <= self.settings.stale_fact_max_accesses)
                with self.repository._write_lock:
                    db.execute("UPDATE memories SET importance=? WHERE id=?", (score, row["id"]))
                    if stale:
                        exists = db.execute(
                            "SELECT 1 FROM memory_events WHERE memory_id=? AND action='stale_review' LIMIT 1",
                            (row["id"],),
                        ).fetchone()
                        if not exists:
                            self.repository._event(int(row["id"]), "stale_review")
                            stale_flagged += 1
            with self.repository._write_lock:
                db.commit()

            themes = self._queue_recurring_themes(active)
            stats = {"archived": archived, "summaries": summaries, "merged_semantic": merged,
                     "stale_flagged": stale_flagged,
                     "inferred_themes_queued": themes,
                     "importance_recomputed": len(active)}
            with self.repository._write_lock:
                db.execute("UPDATE consolidation_runs SET finished_at=?,status='complete',stats_json=? WHERE id=?",
                           (utc_now(), json.dumps(stats), run_id))
                db.commit()
            return {"run_id": run_id, **stats}
        except Exception as exc:
            with self.repository._write_lock:
                self.repository.connections.get().execute(
                    "UPDATE consolidation_runs SET finished_at=?,status='failed',stats_json=? WHERE id=?",
                    (utc_now(), json.dumps({"error_type": type(exc).__name__}), run_id),
                )
                self.repository.connections.get().commit()
            raise

    def enqueue(self) -> bool:
        """Start one coalesced background consolidation without blocking a Gradio response."""
        with self._background_lock:
            if self._background_active:
                return False
            self._background_active = True

        def work() -> None:
            try:
                self.run()
            except Exception as exc:
                logger.error("Background consolidation failed (error_type=%s)", type(exc).__name__)
            finally:
                with self._background_lock:
                    self._background_active = False

        threading.Thread(target=work, name="edith-consolidation", daemon=True).start()
        return True

    def unarchive(self, summary_id: int) -> int:
        with self.repository._write_lock:
            db = self.repository.connections.get()
            sources = [int(row[0]) for row in db.execute(
                "SELECT id FROM memories WHERE user_id=? AND status='archived' AND consolidated_into=?",
                (self.repository.user_id, int(summary_id)),
            ).fetchall()]
            cursor = db.execute(
                "UPDATE memories SET status='active',consolidated_into=NULL,updated_at=? "
                "WHERE user_id=? AND status='archived' AND consolidated_into=?",
                (utc_now(), self.repository.user_id, int(summary_id)),
            )
            changed = int(cursor.rowcount)
            db.execute("UPDATE memories SET status='archived',index_status='pending',updated_at=? WHERE id=? AND user_id=? AND status='active'",
                       (utc_now(), int(summary_id), self.repository.user_id))
            self.repository._event(int(summary_id), "unarchive")
            db.commit()
        if self.synchronizer:
            try:
                self.synchronizer.remove_from_index(int(summary_id))
                for source_id in sources:
                    self.synchronizer.index_memory(source_id)
            except Exception:
                # Startup reconciliation will recover pending active vectors.
                pass
        return changed

    def _queue_recurring_themes(self, active_rows) -> int:
        counts: dict[tuple[str, str], int] = {}
        for row in active_rows:
            if row["layer"] == "semantic":
                key = (str(row["kind"]), str(row["predicate"] or ""))
                if key[1]:
                    counts[key] = counts.get(key, 0) + 1
        top = [(key, count) for key, count in counts.items() if count >= 3]
        if not top:
            return 0
        conversation = self.repository.connections.get().execute(
            "SELECT id FROM conversations WHERE user_id=? AND status='active' ORDER BY id DESC LIMIT 1",
            (self.repository.user_id,),
        ).fetchone()
        if conversation is None:
            return 0
        queued = 0
        for (kind, predicate), _ in top:
            content = "A recurring theme in saved %s memories is %s." % (kind, predicate.replace("_", " "))
            candidate = {"layer": "semantic", "kind": "interest", "content": content,
                         "subject": "user", "predicate": "inferred_theme", "importance": 0.4,
                         "confidence": 0.4, "provenance": "inferred", "entities": []}
            encoded = json.dumps(candidate, ensure_ascii=False)
            exists = self.repository.connections.get().execute(
                "SELECT 1 FROM pending_confirmations WHERE conversation_id=? AND reason='inferred_theme' "
                "AND candidate_json=? AND status IN ('pending','asked') LIMIT 1",
                (int(conversation["id"]), encoded),
            ).fetchone()
            already_saved = self.repository.connections.get().execute(
                "SELECT 1 FROM memories WHERE user_id=? AND layer='semantic' AND content=? "
                "AND status='active' LIMIT 1", (self.repository.user_id, content),
            ).fetchone()
            if not exists and not already_saved:
                self.repository.add_pending_confirmation(
                    int(conversation["id"]), candidate, "inferred_theme",
                    self.settings.pending_confirmation_ttl_turns)
                queued += 1
        return queued
