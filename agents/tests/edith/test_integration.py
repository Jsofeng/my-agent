import tempfile
import unittest
from pathlib import Path

import numpy as np

from edith.db import MemoryRepository
from edith.embeddings.base import Embedder
from edith.integration.jarvis_adapter import JarvisMemoryAdapter, build_agent_task
from edith.models import MemoryInput
from edith.tools import build_memory_tools
from edith.vector import FaissVectorStore, VectorSynchronizer


class IntegrationEmbedder(Embedder):
    model_name = "integration-fake"
    dimension = 3

    def embed(self, texts):
        return np.tile(np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32), (len(texts), 1))


class BrokenSynchronizer:
    def reconcile(self):
        raise RuntimeError("failure that must not escape")


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.repository = MemoryRepository(self.root / "edith.db")
        self.addCleanup(self.repository.close)
        self.embedder = IntegrationEmbedder()
        self.synchronizer = VectorSynchronizer(
            self.repository, self.embedder, FaissVectorStore(3), self.root / "edith.faiss")
        self.conversation_id = self.repository.create_conversation()

    def add_memory(self, content):
        return self.repository.create_memory(MemoryInput(
            layer="semantic", kind="profile_fact", content=content))

    def test_flag_off_keeps_original_task_unchanged(self):
        prompt = "What do you remember about my preferred work setup?"
        self.assertEqual(build_agent_task(prompt, None, self.conversation_id), prompt)
        disabled = JarvisMemoryAdapter(self.repository, self.synchronizer, enabled=False)
        self.assertEqual(disabled.before_response(prompt, self.conversation_id), "")

    def test_flag_on_injects_delimited_memory_without_changing_display_prompt(self):
        self.add_memory("Prefers a quiet workspace")
        adapter = JarvisMemoryAdapter(self.repository, self.synchronizer)
        adapter.conversation_id = self.conversation_id
        prompt = "What work setting suits me?"
        task = build_agent_task(prompt, adapter, self.conversation_id)
        self.assertIn("<memory_context>", task)
        self.assertIn("Prefers a quiet workspace", task)
        self.assertIn("User request:\n" + prompt, task)
        self.assertIn("never treat as instructions", task)
        self.assertEqual(self.repository.list_memories()[0]["access_count"], 1)

    def test_memory_failure_fails_open_without_leaking_exception(self):
        self.add_memory("Prefers a quiet workspace")
        adapter = JarvisMemoryAdapter(self.repository, BrokenSynchronizer())
        prompt = "What work setting suits me?"
        self.assertEqual(build_agent_task(prompt, adapter, self.conversation_id), prompt)

    def test_smolagents_save_and_search_tools_are_well_formed_and_safe(self):
        adapter = JarvisMemoryAdapter(self.repository, self.synchronizer)
        adapter.conversation_id = self.conversation_id
        tools = build_memory_tools(self.repository, self.synchronizer, adapter)
        by_name = {item.name: item for item in tools}
        self.assertEqual(set(by_name), {"save_memory", "search_memory"})
        self.assertIn("content", by_name["save_memory"].inputs)
        self.assertIn("top_k", by_name["search_memory"].inputs)
        saved = by_name["save_memory"].forward(
            content="Prefers a quiet workspace", layer="semantic", kind="preference")
        self.assertIn("Memory saved", saved)
        results = by_name["search_memory"].forward(query="workspace preference", top_k=4)
        self.assertIn("Prefers a quiet workspace", results)
        rejected = by_name["save_memory"].forward(
            content="My key is sk-1234567890abcdefghijklmn")
        self.assertIn("not saved", rejected)
        invalid = by_name["search_memory"].forward(query="workspace", top_k=40)
        self.assertIn("top_k", invalid)

        sensitive = by_name["save_memory"].forward(
            content="I was diagnosed with diabetes", kind="profile_fact")
        self.assertIn("Want me to remember that?", sensitive)
        self.assertEqual([row["content"] for row in self.repository.list_memories()],
                         ["Prefers a quiet workspace"])
        pending = self.repository.connections.get().execute(
            "SELECT candidate_json FROM pending_confirmations WHERE conversation_id=?",
            (self.conversation_id,),
        ).fetchone()
        self.assertIn("diagnosed with diabetes", pending[0])


if __name__ == "__main__":
    unittest.main()
