import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from edith.config import EdithConfig
from edith.consolidation.consolidator import MemoryConsolidator
from edith.db import MemoryRepository
from edith.embeddings.base import Embedder
from edith.models import MemoryInput
from edith.vector import FaissVectorStore, VectorSynchronizer
from edith.updating.conflicts import (ConflictDecision, LLMConflictAdjudicator,
                                      SemanticConflictService)
from edith.updating.dedup import MemoryDeduplicator
from edith.consolidation.scheduler import create_scheduler


class UpdatingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.repo = MemoryRepository(self.root / "edith.db")
        self.addCleanup(self.repo.close)

    def memory(self, content, *, layer="semantic", kind="profile_fact", subject="user",
                predicate="current_employer", provenance="user_stated", confidence=0.95):
        return MemoryInput(layer=layer, kind=kind, content=content, subject=subject,
                           predicate=predicate, importance=0.7, confidence=confidence,
                           provenance=provenance)

    def test_exact_and_near_duplicate_reinforcement(self):
        first_id = self.repo.create_memory(self.memory("Works at ExampleCo"))
        dedup = MemoryDeduplicator(self.repo, 0.92)
        exact = dedup.find_duplicate(layer="semantic", kind="profile_fact", subject="user",
                                     content="Works at ExampleCo")
        self.assertEqual(exact["id"], first_id)
        # Deterministic identical vectors exercise the cosine threshold path.
        with self.repo._write_lock:
            self.repo.connections.get().execute(
                "UPDATE memories SET embedding=? WHERE id=?",
                (np.asarray([1, 0, 0], dtype=np.float32).tobytes(), first_id))
            self.repo.connections.get().commit()
        near = dedup.find_duplicate(layer="semantic", kind="profile_fact", subject="user",
                                    content="Employed at ExampleCo", embedding=np.asarray([1, 0, 0]))
        self.assertEqual(near["id"], first_id)
        dedup.reinforce(first_id, 0.9, 0.99)
        self.assertEqual(self.repo.get_memory(first_id)["importance"], 0.9)
        dedup.reinforce(first_id, 0.5, 0.7, source_message_ids="[10,11]")
        reinforced = self.repo.get_memory(first_id)
        self.assertAlmostEqual(reinforced["importance"], 0.92)
        self.assertEqual(json.loads(reinforced["source_message_ids"]), [10, 11])
        actions = [row[0] for row in self.repo.connections.get().execute(
            "SELECT action FROM memory_events WHERE memory_id=?", (first_id,)).fetchall()]
        self.assertIn("reinforce", actions)

    def test_high_confidence_update_supersedes_without_overwriting_history(self):
        conversation_id = self.repo.create_conversation()
        old_id = self.repo.create_memory(self.memory("Works at OldCo"))
        class UpdateAdjudicator:
            def adjudicate(self, old_content, new_content):
                return ConflictDecision(relation="update", rationale="Changed employer")
        service = SemanticConflictService(self.repo, UpdateAdjudicator())
        relation, new_id = service.save(self.memory("Works at NewCo"))
        self.assertEqual(relation, "update")
        old = self.repo.get_memory(old_id)
        new = self.repo.get_memory(new_id)
        self.assertEqual(old["content"], "Works at OldCo")
        self.assertEqual(old["status"], "superseded")
        self.assertEqual(old["superseded_by"], new_id)
        self.assertEqual(new["content"], "Works at NewCo")
        self.assertIsNotNone(new["valid_from"])

    def test_contradiction_keeps_both_and_asks(self):
        conversation_id = self.repo.create_conversation()
        old_id = self.repo.create_memory(self.memory("Works at OldCo"))
        item = MemoryInput(layer="semantic", kind="profile_fact", content="Works at NewCo",
                           subject="user", predicate="current_employer", importance=0.7,
                           confidence=0.95, source_conversation_id=conversation_id)
        class ContradictionAdjudicator:
            def adjudicate(self, old_content, new_content):
                return {"relation": "contradiction", "rationale": "Needs clarification"}
        service = SemanticConflictService(self.repo, ContradictionAdjudicator())
        relation, _ = service.save(item)
        self.assertEqual(relation, "pending")
        self.assertEqual(self.repo.get_memory(old_id)["status"], "active")
        self.assertEqual(len(self.repo.list_memories()), 1)
        self.assertEqual(self.repo.get_pending_question(conversation_id, "semantic_conflict")["kind"],
                         "semantic_conflict")

    def test_llm_adjudicator_requests_temperature_zero_and_validates_json(self):
        class FakeLLM:
            def complete(self, system_prompt, user_prompt, temperature):
                self.temperature = temperature
                return '{"relation":"update","rationale":"Newer fact"}'
        llm = FakeLLM()
        decision = LLMConflictAdjudicator(llm).adjudicate("Old fact", "New fact")
        self.assertEqual(decision.relation, "update")
        self.assertEqual(llm.temperature, 0.0)

    def test_consolidation_archives_sources_links_summary_and_is_idempotent(self):
        first_id = self.repo.create_memory(self.memory(
            "Completed milestone A", layer="episodic", kind="achievement",
            predicate="project_alpha", provenance="user_stated"))
        second_id = self.repo.create_memory(self.memory(
            "Completed milestone B", layer="episodic", kind="achievement",
            predicate="project_alpha", provenance="user_stated"))
        consolidator = MemoryConsolidator(self.repo)
        result = consolidator.run()
        self.assertEqual(result["archived"], 2)
        first = self.repo.get_memory(first_id)
        second = self.repo.get_memory(second_id)
        self.assertEqual(first["status"], "archived")
        self.assertEqual(first["consolidated_into"], second["consolidated_into"])
        summary = self.repo.get_memory(first["consolidated_into"])
        self.assertIn("milestone A", summary["content"])
        self.assertIn("milestone B", summary["content"])
        again = consolidator.run()
        self.assertEqual(again["archived"], 0)
        self.assertEqual(consolidator.unarchive(summary["id"]), 2)
        self.assertEqual(self.repo.get_memory(first_id)["status"], "active")
        self.assertEqual(self.repo.get_memory(summary["id"])["status"], "archived")

    def test_interrupted_consolidation_is_recorded_and_safe_to_retry(self):
        self.repo.create_memory(self.memory("Event one", layer="episodic", kind="event",
                                            predicate="milestones"))
        self.repo.create_memory(self.memory("Event two", layer="episodic", kind="event",
                                            predicate="milestones"))
        class BrokenIndex:
            def remove_from_index(self, memory_id):
                return None
            def index_memory(self, memory_id):
                raise RuntimeError("simulated index interruption")
        consolidator = MemoryConsolidator(self.repo, BrokenIndex())
        with self.assertRaisesRegex(RuntimeError, "simulated"):
            consolidator.run()
        status = self.repo.connections.get().execute(
            "SELECT status FROM consolidation_runs ORDER BY id DESC LIMIT 1").fetchone()[0]
        self.assertEqual(status, "failed")
        count_before = len(self.repo.list_memories(status="active"))
        retry = MemoryConsolidator(self.repo).run()
        self.assertEqual(retry["archived"], 0)
        self.assertEqual(len(self.repo.list_memories(status="active")), count_before)

    def test_scheduler_is_single_instance_coalesced_and_does_not_start_implicitly(self):
        scheduler = create_scheduler(MemoryConsolidator(self.repo))
        job = scheduler.get_job("edith_consolidation")
        self.assertIsNotNone(job)
        self.assertEqual(job.max_instances, 1)
        self.assertTrue(job.coalesce)
        self.assertEqual(scheduler.state, 0)

    def test_consolidator_merges_semantic_duplicates_and_flags_stale_facts(self):
        class FlatEmbedder(Embedder):
            model_name = "flat-consolidation"
            dimension = 3
            def embed(self, texts):
                return np.tile(np.asarray([[1, 0, 0]], dtype=np.float32), (len(texts), 1))
        settings = EdithConfig(database_path=self.root / "edith.db", index_path=self.root / "edith.faiss",
                               stale_fact_days=0, stale_fact_max_accesses=0)
        sync = VectorSynchronizer(self.repo, FlatEmbedder(), FaissVectorStore(3), settings.index_path)
        first = self.repo.create_memory(self.memory("Can program in Python", kind="skill",
                                                    predicate="skill"))
        second = self.repo.create_memory(self.memory("Writes Python code", kind="skill",
                                                     predicate="skill"))
        sync.reconcile()
        result = MemoryConsolidator(self.repo, sync, settings).run()
        self.assertEqual(result["merged_semantic"], 1)
        self.assertEqual(self.repo.get_memory(second)["status"], "archived")
        self.assertEqual(self.repo.get_memory(second)["consolidated_into"], first)
        self.assertNotIn(second, sync.store.ids())
        stale_event = self.repo.connections.get().execute(
            "SELECT count(*) FROM memory_events WHERE memory_id=? AND action='stale_review'",
            (first,)).fetchone()[0]
        self.assertEqual(stale_event, 1)

    def test_recurring_theme_is_queued_for_confirmation_not_saved_as_fact(self):
        conversation_id = self.repo.create_conversation()
        for index, vector in enumerate(([1, 0, 0], [0, 1, 0], [0, 0, 1])):
            self.repo.create_memory(MemoryInput(
                layer="semantic", kind="skill", content="Synthetic skill fixture %d" % index,
                subject="user", predicate="skill", embedding=np.asarray(vector, dtype=np.float32).tobytes(),
                embedding_model="synthetic"))
        result = MemoryConsolidator(self.repo, settings=EdithConfig(
            database_path=self.root / "edith.db", index_path=self.root / "edith.faiss",
            dup_threshold=0.99, stale_fact_days=99999)).run()
        self.assertEqual(result["inferred_themes_queued"], 1)
        self.assertEqual(self.repo.connections.get().execute(
            "SELECT count(*) FROM memories WHERE predicate='inferred_theme' AND status='active'").fetchone()[0], 0)
        pending = self.repo.connections.get().execute(
            "SELECT reason,candidate_json FROM pending_confirmations WHERE conversation_id=?",
            (conversation_id,)).fetchone()
        self.assertEqual(pending["reason"], "inferred_theme")
        self.assertIn('"confidence": 0.4', pending["candidate_json"])

    def test_hard_delete_removes_entity_links_and_keeps_content_free_audit(self):
        memory_id = self.repo.create_memory(self.memory("Sensitive-ish deletion fixture"))
        entity_id = self.repo.add_entity("FictionalCompany", "company")
        self.repo.link_entity(memory_id, entity_id)
        self.repo.delete_memory(memory_id, hard=True)
        self.assertIsNone(self.repo.get_memory(memory_id))
        self.assertEqual(self.repo.memory_entities(memory_id), [])
        event = self.repo.connections.get().execute(
            "SELECT action,detail_json FROM memory_events WHERE memory_id=? ORDER BY id DESC LIMIT 1",
            (memory_id,)).fetchone()
        self.assertEqual(tuple(event), ("delete", "{}"))
        self.assertNotIn(b"Sensitive-ish deletion fixture", (self.root / "edith.db").read_bytes())
        wal_path = Path(str(self.root / "edith.db") + "-wal")
        if wal_path.exists():
            self.assertNotIn(b"Sensitive-ish deletion fixture", wal_path.read_bytes())


if __name__ == "__main__":
    unittest.main()
