import tempfile
import unittest
from pathlib import Path

import numpy as np

from edith.db import MemoryRepository
from edith.embeddings.base import Embedder
from edith.extraction.confirmations import PendingConfirmationService
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

    def test_after_response_persists_both_turns_and_queues_extraction(self):
        class Worker:
            def __init__(inner):
                inner.calls = []
            def after_response(inner, conversation_id):
                inner.calls.append(conversation_id)
        worker = Worker()
        adapter = JarvisMemoryAdapter(self.repository, self.synchronizer, worker=worker)
        adapter.record_user_message(self.conversation_id, "I prefer quiet workspaces.")
        self.assertEqual([row["role"] for row in self.repository.get_messages(self.conversation_id)],
                         ["user"])
        adapter.after_response(self.conversation_id, "I prefer quiet workspaces.", "Understood.")
        self.assertEqual([row["role"] for row in self.repository.get_messages(self.conversation_id)],
                         ["user", "assistant"])
        self.assertEqual(worker.calls, [self.conversation_id])

    def test_sensitive_confirmation_prompt_is_asked_once(self):
        confirmations = PendingConfirmationService(self.repository, self.synchronizer)
        self.repository.add_pending_confirmation(
            self.conversation_id, {"layer": "semantic", "kind": "profile_fact",
                                   "content": "A sensitive fact"},
            "sensitive_category:health", expires_after_turn=4)
        adapter = JarvisMemoryAdapter(
            self.repository, self.synchronizer, pending_confirmations=confirmations)
        first = adapter.before_response("Can you help me with a different question?", self.conversation_id)
        second = adapter.before_response("Please continue with the question.", self.conversation_id)
        self.assertIn("Want me to remember that?", first)
        self.assertEqual(second, "")

    def test_sensitive_correction_supersedes_only_after_explicit_yes(self):
        memory_id = self.add_memory("User works at OldCo")
        self.synchronizer.index_memory(memory_id)
        confirmations = PendingConfirmationService(self.repository, self.synchronizer)
        adapter = JarvisMemoryAdapter(self.repository, self.synchronizer,
                                      pending_confirmations=confirmations)
        adapter.conversation_id = self.conversation_id
        tools = {item.name: item for item in build_memory_tools(
            self.repository, self.synchronizer, adapter)}
        result = tools["update_memory"].forward(
            memory_id=memory_id, new_content="User has a medical diagnosis")
        self.assertIn("Want me to remember that?", result)
        self.assertEqual(self.repository.get_memory(memory_id)["status"], "active")
        answer = adapter.resolve_pending_response(self.conversation_id, "yes, remember that")
        self.assertIn("remember", answer)
        self.assertEqual(self.repository.get_memory(memory_id)["status"], "superseded")
        current = self.repository.connections.get().execute(
            "SELECT id,content FROM memories WHERE status='active' AND id<>?", (memory_id,)).fetchone()
        self.assertIn("medical diagnosis", current["content"])
        self.assertNotIn(memory_id, self.synchronizer.store.ids())
        self.assertIn(current["id"], self.synchronizer.store.ids())

    def test_smolagents_save_and_search_tools_are_well_formed_and_safe(self):
        adapter = JarvisMemoryAdapter(self.repository, self.synchronizer)
        adapter.conversation_id = self.conversation_id
        tools = build_memory_tools(self.repository, self.synchronizer, adapter)
        by_name = {item.name: item for item in tools}
        self.assertEqual(set(by_name), {"save_memory", "search_memory", "update_memory", "delete_memory",
                                        "get_user_profile", "get_recent_events", "get_project_context",
                                        "list_memories"})
        self.assertIn("content", by_name["save_memory"].inputs)
        self.assertIn("top_k", by_name["search_memory"].inputs)
        saved = by_name["save_memory"].forward(
            content="Prefers a quiet workspace", layer="semantic", kind="preference")
        self.assertIn("Memory saved", saved)
        results = by_name["search_memory"].forward(query="workspace preference", top_k=4)
        self.assertIn("Prefers a quiet workspace", results)
        profile = by_name["get_user_profile"].forward()
        self.assertIn("Prefers a quiet workspace", profile)
        project_id = self.repository.create_memory(MemoryInput(
            layer="semantic", kind="project", content="Synthetic TestProject tracks milestones",
            subject="TestProject"))
        project_entity_id = self.repository.add_entity("TestProject", "project", ["TP"])
        self.repository.link_entity(project_id, project_entity_id)
        self.synchronizer.index_memory(project_id)
        listed = by_name["list_memories"].forward(page=1, page_size=10, status="active")
        self.assertIn("ID 1", listed)
        filtered = by_name["list_memories"].forward(page=1, page_size=10, entity="TP")
        self.assertIn("TestProject tracks milestones", filtered)
        project = by_name["get_project_context"].forward(project_name="TestProject")
        self.assertIn("TestProject tracks milestones", project)
        events = by_name["get_recent_events"].forward(days=30, top_k=10)
        self.assertIn("No recent", events)
        rejected = by_name["save_memory"].forward(
            content="My key is sk-1234567890abcdefghijklmn")
        self.assertIn("not saved", rejected)
        invalid = by_name["search_memory"].forward(query="workspace", top_k=40)
        self.assertIn("top_k", invalid)

        corrected = by_name["update_memory"].forward(memory_id=1,
                                                      new_content="Prefers a silent workspace")
        self.assertIn("history preserved", corrected)
        self.assertEqual(self.repository.get_memory(1)["status"], "superseded")
        current = self.repository.connections.get().execute(
            "SELECT id FROM memories WHERE status='active' ORDER BY id DESC LIMIT 1").fetchone()[0]
        removed = by_name["delete_memory"].forward(memory_id=current)
        self.assertIn("was deleted", removed)

        sensitive = by_name["save_memory"].forward(
            content="I was diagnosed with diabetes", kind="profile_fact")
        self.assertIn("Want me to remember that?", sensitive)
        active_contents = [row["content"] for row in self.repository.list_memories()]
        self.assertNotIn("I was diagnosed with diabetes", active_contents)
        self.assertNotIn("Prefers a silent workspace", active_contents)
        pending = self.repository.connections.get().execute(
            "SELECT candidate_json FROM pending_confirmations WHERE conversation_id=?",
            (self.conversation_id,),
        ).fetchone()
        self.assertIn("diagnosed with diabetes", pending[0])


if __name__ == "__main__":
    unittest.main()
