import tempfile
import unittest
from pathlib import Path

import numpy as np

from edith.db import MemoryRepository
from edith.embeddings.base import Embedder
from edith.models import MemoryInput
from edith.vector import FaissVectorStore, VectorSynchronizer


class FakeEmbedder(Embedder):
    dimension = 3

    def __init__(self, model_name="fake-v1"):
        self.model_name = model_name
        self.calls = 0

    def embed(self, texts):
        self.calls += len(texts)
        result = []
        for text in texts:
            lowered = text.lower()
            if "likes tart fruit" in lowered or "citrus is a favorite" in lowered:
                result.append([1.0, 0.0, 0.0])
            elif "deadline" in lowered or "release date" in lowered:
                result.append([0.0, 1.0, 0.0])
            else:
                result.append([0.0, 0.0, 1.0])
        return np.asarray(result, dtype=np.float32)


class VectorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.repo = MemoryRepository(self.root / "edith.db")
        self.addCleanup(self.repo.close)
        self.embedder = FakeEmbedder()
        self.index_path = self.root / "edith.faiss"
        self.sync = VectorSynchronizer(self.repo, self.embedder,
                                       FaissVectorStore(3), self.index_path)

    def add(self, content, kind="preference", subject="user"):
        return self.repo.create_memory(MemoryInput(
            layer="semantic", kind=kind, content=content, subject=subject))

    def test_semantic_similarity_without_shared_keywords_and_filters(self):
        fruit = self.add("Likes tart fruit", kind="preference", subject="user")
        self.add("Release date is not known", kind="event", subject="Project Red")
        self.sync.reconcile()
        # Query's words differ from the stored memory, but the fake semantic vectors match.
        results = self.sync.search("Citrus is a favorite", top_k=2, layer="semantic",
                                   kind="preference", subject="user")
        self.assertEqual([row["id"] for row in results], [fruit])
        self.assertAlmostEqual(results[0]["similarity"], 1.0)

    def test_delete_removes_vector_and_sqlite_row(self):
        memory_id = self.add("Likes tart fruit")
        self.sync.reconcile()
        self.assertIn(memory_id, self.sync.store.ids())
        self.sync.delete_memory(memory_id, hard=True)
        self.assertNotIn(memory_id, self.sync.store.ids())
        self.assertIsNone(self.repo.get_memory(memory_id))
        self.assertTrue(self.sync.verify_integrity()["ok"])

    def test_missing_and_corrupt_indexes_rebuild_from_stored_embeddings(self):
        memory_id = self.add("Likes tart fruit")
        self.sync.reconcile()
        initial_embed_calls = self.embedder.calls
        self.index_path.unlink()
        restarted = VectorSynchronizer(self.repo, self.embedder, FaissVectorStore(3), self.index_path)
        result = restarted.reconcile()
        self.assertTrue(result["rebuilt"])
        self.assertEqual(self.embedder.calls, initial_embed_calls)
        self.assertIn(memory_id, restarted.store.ids())
        self.index_path.write_bytes(b"not a faiss index")
        restarted_again = VectorSynchronizer(self.repo, self.embedder, FaissVectorStore(3), self.index_path)
        result = restarted_again.reconcile()
        self.assertTrue(result["rebuilt"])
        self.assertEqual(self.embedder.calls, initial_embed_calls)
        self.assertTrue(restarted_again.verify_integrity()["ok"])

    def test_model_change_reembeds_stored_memories(self):
        memory_id = self.add("Likes tart fruit")
        self.sync.reconcile()
        replacement = FakeEmbedder("fake-v2")
        changed = VectorSynchronizer(self.repo, replacement, FaissVectorStore(3), self.index_path)
        report = changed.reconcile()
        self.assertGreaterEqual(report["reembedded"], 1)
        self.assertEqual(replacement.calls, 1)
        self.assertEqual(self.repo.get_memory(memory_id)["embedding_model"], "fake-v2")
        self.assertTrue(changed.verify_integrity()["ok"])

    def test_pending_rows_are_indexed_during_startup_reconciliation(self):
        memory_id = self.add("Likes tart fruit")
        self.assertEqual(self.repo.get_memory(memory_id)["index_status"], "pending")
        report = self.sync.reconcile()
        self.assertEqual(report["added"], 1)
        self.assertEqual(self.repo.get_memory(memory_id)["index_status"], "indexed")
        self.assertTrue(self.sync.verify_integrity()["ok"])

    def test_rebuild_and_invalid_query_bounds(self):
        memory_id = self.add("Likes tart fruit")
        self.sync.reconcile()
        self.sync.store = FaissVectorStore(3)
        self.assertFalse(self.sync.verify_integrity()["ok"])
        report = self.sync.rebuild_index()
        self.assertTrue(report["rebuilt"])
        self.assertIn(memory_id, self.sync.store.ids())
        with self.assertRaisesRegex(ValueError, "top_k"):
            self.sync.search("Citrus is a favorite", top_k=0)

    def test_integrity_reports_active_rows_without_embeddings(self):
        memory_id = self.add("Likes tart fruit")
        report = self.sync.verify_integrity()
        self.assertFalse(report["ok"])
        self.assertEqual(report["missing_ids"], [memory_id])
        self.assertEqual(report["missing_embeddings"], [memory_id])
        self.sync.reconcile()
        self.assertTrue(self.sync.verify_integrity()["ok"])


if __name__ == "__main__":
    unittest.main()
