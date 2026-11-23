"""Restart-style acceptance path using only synthetic fixtures and fake clients."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from edith.config import EdithConfig
from edith.context.context_manager import ContextManager
from edith.context.entity_resolver import EntityResolver
from edith.db import MemoryRepository
from edith.embeddings.base import Embedder
from edith.extraction.extractor import ExtractionService
from edith.models import MemoryInput
from edith.vector import FaissVectorStore, VectorSynchronizer


class AcceptanceEmbedder(Embedder):
    model_name = "acceptance-fake"
    dimension = 3

    def embed(self, texts):
        return np.tile(np.asarray([[1, 0, 0]], dtype=np.float32), (len(texts), 1))


class AcceptanceLLM:
    def complete(self, system_prompt, user_prompt, temperature=0.0):
        return json.dumps([{
            "layer": "episodic", "kind": "event",
            "content": "Synthetic fixture: applying for a Software Engineering internship at Snowflake",
            "subject": "user", "predicate": "application",
            "entities": [{"name": "Snowflake", "type": "company"}],
            "importance": 0.8, "confidence": 0.9,
            "occurred_at": "2026-03-02T00:00:00Z", "provenance": "user_stated",
        }])


class AcceptanceTests(unittest.TestCase):
    def test_extract_restart_rebuild_reference_recall_and_delete(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            settings = EdithConfig(database_path=root / "edith.db", index_path=root / "edith.faiss")
            repo = MemoryRepository(settings.database_path)
            conversation_id = repo.create_conversation()
            repo.add_message(conversation_id, "user",
                             "I'm applying for a Software Engineering internship at Snowflake.")
            embedder = AcceptanceEmbedder()
            sync = VectorSynchronizer(repo, embedder, FaissVectorStore(3), settings.index_path)
            extraction = ExtractionService(repo, AcceptanceLLM(), settings, synchronizer=sync)
            self.assertEqual(extraction.process_conversation(conversation_id)["saved"], 1)
            repo.close()

            # Simulate loss of the disposable index and a fresh app/repository instance.
            settings.index_path.unlink()
            restarted_repo = MemoryRepository(settings.database_path)
            self.addCleanup(restarted_repo.close)
            restarted_sync = VectorSynchronizer(
                restarted_repo, AcceptanceEmbedder(), FaissVectorStore(3), settings.index_path)
            report = restarted_sync.reconcile()
            self.assertTrue(report["rebuilt"])
            self.assertTrue(restarted_sync.verify_integrity()["ok"])
            next_conversation = restarted_repo.create_conversation()
            resolver = EntityResolver(restarted_repo, restarted_sync, settings)
            resolution = resolver.resolve("I have an interview coming up. Help me prepare.",
                                          next_conversation)
            self.assertEqual(resolution.status, "resolved")
            context = ContextManager(restarted_repo, restarted_sync, settings=settings).build(
                "I have an interview coming up. Help me prepare.", next_conversation, resolution)
            self.assertIn("Snowflake", context)
            self.assertIn("Software Engineering internship", context)
            self.assertNotIn("interviewer", context.lower())
            memory_id = resolution.memory_ids[0]
            restarted_sync.delete_memory(memory_id, hard=True)
            self.assertIsNone(restarted_repo.get_memory(memory_id))
            self.assertTrue(restarted_sync.verify_integrity()["ok"])


if __name__ == "__main__":
    unittest.main()
