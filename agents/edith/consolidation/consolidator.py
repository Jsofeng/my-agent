"""Idempotent, reversible consolidation over SQLite source memories."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from ..db.repository import MemoryRepository, utc_now
from ..models import MemoryInput
from ..vector.sync import VectorSynchronizer


class MemoryConsolidator:
    def __init__(self, repository: MemoryRepository, synchronizer: VectorSynchronizer | None = None):
        self.repository = repository
        self.synchronizer = synchronizer

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
            db = self.repository.connections.get()
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
                summary_id = self.repository.create_memory(MemoryInput(
                    layer="episodic", kind="experience", content=text[:10000], subject=subject,
                    predicate=predicate, importance=max(float(item["importance"]) for item in ungrouped),
                    confidence=min(float(item["confidence"]) for item in ungrouped),
                    provenance="inferred", occurred_at=ungrouped[-1].get("occurred_at"),
                ))
                with self.repository._write_lock:
                    for source_id in source_ids:
                        db.execute("UPDATE memories SET status='archived',consolidated_into=? WHERE id=?",
                                   (summary_id, source_id))
                        self.repository._event(source_id, "consolidate")
                    db.commit()
                if self.synchronizer:
                    self.synchronizer.index_memory(summary_id)
                archived += len(source_ids)
                summaries += 1

            # Importance decays gently with age while preserving access reinforcement.
            now = datetime.now(timezone.utc)
            active = db.execute(
                "SELECT id,importance,access_count,updated_at FROM memories WHERE user_id=? AND status='active'",
                (self.repository.user_id,),
            ).fetchall()
            for row in active:
                try:
                    age_days = max(0.0, (now - datetime.fromisoformat(
                        str(row["updated_at"]).replace("Z", "+00:00")).astimezone(timezone.utc)).total_seconds() / 86400)
                except (TypeError, ValueError):
                    age_days = 0.0
                decay = 0.85 ** (age_days / 365.0)
                use_boost = min(0.15, 0.02 * int(row["access_count"]))
                score = min(1.0, max(0.0, float(row["importance"]) * decay + use_boost))
                db.execute("UPDATE memories SET importance=? WHERE id=?", (score, row["id"]))
            stats = {"archived": archived, "summaries": summaries,
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

    def unarchive(self, summary_id: int) -> int:
        with self.repository._write_lock:
            db = self.repository.connections.get()
            cursor = db.execute(
                "UPDATE memories SET status='active',consolidated_into=NULL,updated_at=? "
                "WHERE user_id=? AND status='archived' AND consolidated_into=?",
                (utc_now(), self.repository.user_id, int(summary_id)),
            )
            changed = int(cursor.rowcount)
            self.repository._event(int(summary_id), "unarchive")
            db.commit()
            return changed
