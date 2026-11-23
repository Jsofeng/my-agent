import json
import tempfile
import threading
import time
import unittest
import signal
from unittest.mock import Mock, patch
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from edith.config import EdithConfig
from edith.db import MemoryRepository
from edith.embeddings.base import Embedder
from edith.extraction.confirmations import PendingConfirmationService
from edith.extraction.extractor import ExtractionService
from edith.extraction.extractor import SmolagentsLLMClient
from edith.extraction.prefilter import is_extractable_message
from edith.integration.background_worker import ExtractionWorker
from edith.integration.shutdown import register_shutdown_flush
from edith.integration.jarvis_adapter import JarvisMemoryAdapter
from edith.models import MemoryInput
from edith.vector import FaissVectorStore, VectorSynchronizer


class FakeEmbedder(Embedder):
    model_name = "extract-fake"
    dimension = 3

    def embed(self, texts):
        return np.tile(np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32), (len(texts), 1))


class FakeLLM:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    def complete(self, system_prompt, user_prompt, temperature=0.0):
        self.calls.append((system_prompt, user_prompt, temperature))
        if not self.outputs:
            raise AssertionError("unexpected LLM call")
        result = self.outputs.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


def candidate(content="Prefers quiet workspaces", importance=0.8, confidence=0.9,
              kind="preference", layer="semantic", occurred_at=None):
    return json.dumps([{
        "layer": layer,
        "kind": kind,
        "content": content,
        "subject": "user",
        "predicate": "workspace",
        "entities": [],
        "importance": importance,
        "confidence": confidence,
        "occurred_at": occurred_at,
        "provenance": "user_stated",
    }])


class ExtractionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.repo = MemoryRepository(self.root / "edith.db")
        self.addCleanup(self.repo.close)
        self.settings = EdithConfig(
            database_path=self.root / "edith.db", index_path=self.root / "edith.faiss",
            extraction_idle_minutes=60, extraction_batch_messages=8,
            pending_confirmation_ttl_turns=4,
        )
        self.conversation_id = self.repo.create_conversation()
        self.embedder = FakeEmbedder()
        self.sync = VectorSynchronizer(self.repo, self.embedder,
                                       FaissVectorStore(3), self.root / "edith.faiss")

    def user_turn(self, text, assistant="Response."):
        self.repo.add_message(self.conversation_id, "user", text)
        self.repo.add_message(self.conversation_id, "assistant", assistant)

    def extractor(self, *outputs):
        return ExtractionService(
            self.repo, FakeLLM(outputs), self.settings,
            clock=lambda: datetime(2026, 11, 22, 17, 0, tzinfo=timezone.utc),
            synchronizer=self.sync,
        )

    def test_prefilter_classifies_trivial_generic_and_personal_messages(self):
        self.assertFalse(is_extractable_message("Thanks!"))
        self.assertFalse(is_extractable_message("What is the weather in Toronto today?"))
        self.assertTrue(is_extractable_message("I prefer quiet workspaces."))
        self.assertTrue(is_extractable_message("Please remember that I started a new role."))

    def test_thresholds_user_only_batch_and_relative_date_prompt(self):
        self.user_turn("I prefer quiet workspaces.", "Assistant mentioned a different preference.")
        low = self.extractor(candidate(importance=0.2))
        stats = low.process_conversation(self.conversation_id)
        self.assertEqual(stats["saved"], 0)
        self.assertEqual(len(self.repo.list_memories()), 0)
        self.assertTrue(all(row["extracted"] == 1 for row in self.repo.get_messages(self.conversation_id)))

        self.user_turn("I have a project review next Tuesday.")
        output = candidate(content="Project review scheduled for 2026-12-01T09:00:00-05:00",
                           kind="event", layer="episodic", occurred_at="2026-12-01T09:00:00-05:00")
        llm = FakeLLM([output])
        extractor = ExtractionService(
            self.repo, llm, self.settings,
            clock=lambda: datetime(2026, 11, 22, 17, 0, tzinfo=timezone.utc),
            synchronizer=self.sync,
        )
        stats = extractor.process_conversation(self.conversation_id)
        self.assertEqual(stats["saved"], 1)
        self.assertIn("2026-11-22", llm.calls[0][1])
        self.assertIn("America/Toronto", llm.calls[0][1])
        self.assertNotIn("Assistant mentioned", llm.calls[0][1])
        self.assertEqual(llm.calls[0][2], 0.0)
        self.assertEqual(self.repo.list_memories(layer="episodic")[0]["occurred_at"],
                         "2026-12-01T14:00:00Z")

    def test_junk_batch_skips_llm_permanently(self):
        self.user_turn("Thanks!")
        llm = FakeLLM([])
        extractor = ExtractionService(self.repo, llm, self.settings)
        stats = extractor.process_conversation(self.conversation_id)
        self.assertEqual(stats["skipped"], 1)
        self.assertEqual(llm.calls, [])
        self.assertTrue(all(row["extracted"] == 1 for row in self.repo.get_messages(self.conversation_id)))

    def test_sensitive_candidate_waits_for_explicit_yes_and_is_purged(self):
        self.user_turn("I have a health condition I want to remember.")
        extraction = self.extractor(candidate(content="I was diagnosed with diabetes", kind="profile_fact"))
        self.assertEqual(extraction.process_conversation(self.conversation_id)["pending"], 1)
        self.assertEqual(len(self.repo.list_memories()), 0)
        db = self.repo.connections.get()
        pending_id, content_json = db.execute(
            "SELECT id,candidate_json FROM pending_confirmations WHERE conversation_id=?",
            (self.conversation_id,),
        ).fetchone()
        self.assertIn("diabetes", content_json)
        confirmations = PendingConfirmationService(self.repo, self.sync, ttl_turns=4)
        self.assertTrue(confirmations.claim_prompt(self.conversation_id))
        result = confirmations.resolve_reply(self.conversation_id, "yes, remember that")
        self.assertIn("remember that", result)
        memory = self.repo.list_memories()[0]
        self.assertEqual(memory["provenance"], "user_confirmed")
        self.assertEqual(db.execute("SELECT candidate_json FROM pending_confirmations WHERE id=?",
                                    (pending_id,)).fetchone()[0], "{}")

    def test_sensitive_candidate_no_and_expiry_purge_candidate(self):
        self.user_turn("I have a health issue.")
        extraction = self.extractor(candidate(content="User takes a prescribed medication", kind="profile_fact"))
        extraction.process_conversation(self.conversation_id)
        confirmations = PendingConfirmationService(self.repo, self.sync, ttl_turns=4)
        self.assertTrue(confirmations.claim_prompt(self.conversation_id))
        self.assertEqual(confirmations.resolve_reply(self.conversation_id, "no thanks"),
                         "Okay, I won’t save it.")
        self.assertEqual(len(self.repo.list_memories()), 0)

        pending_id = self.repo.add_pending_confirmation(
            self.conversation_id, {"content": "private candidate"}, "sensitive_category:health",
            expires_after_turn=1, asked=True,
        )
        self.user_turn("Can you help with something else?")
        self.user_turn("Another unrelated question for you.")
        confirmations.expire(self.conversation_id)
        row = self.repo.connections.get().execute(
            "SELECT candidate_json,status FROM pending_confirmations WHERE id=?", (pending_id,)
        ).fetchone()
        self.assertEqual(tuple(row), ("{}", "expired"))

    def test_invalid_json_gets_one_repair_retry_then_marks_batch_complete(self):
        self.user_turn("I prefer quiet workspaces.")
        llm = FakeLLM(["not json", candidate()])
        service = ExtractionService(self.repo, llm, self.settings)
        stats = service.process_conversation(self.conversation_id)
        self.assertEqual(stats["saved"], 1)
        self.assertEqual(len(llm.calls), 2)
        self.assertTrue(all(call[2] == 0.0 for call in llm.calls))

    def test_secret_candidate_dropped_and_messages_redacted_before_database(self):
        secret = "sk-1234567890abcdefghijklmn"
        message_id = self.repo.add_message(self.conversation_id, "user", "My key is " + secret)
        stored = self.repo.get_messages(self.conversation_id)[0]["content"]
        self.assertNotIn(secret, stored)
        self.assertEqual(stored, "My key is [REDACTED]")
        extractor = self.extractor(candidate(content="Candidate included " + secret))
        with self.assertLogs("edith.extraction.extractor", level="INFO") as logs:
            extractor.process_conversation(self.conversation_id)
        self.assertEqual(len(self.repo.list_memories()), 0)
        self.assertNotIn(secret.encode(), (self.root / "edith.db").read_bytes())
        self.assertNotIn(secret, "\n".join(logs.output))

    def test_secret_in_candidate_entity_field_is_dropped(self):
        self.user_turn("I am working with a fictional project.")
        secret = "sk-1234567890abcdefghijklmn"
        data = json.loads(candidate())
        data[0]["entities"] = [{"name": secret, "type": "technology"}]
        service = self.extractor(json.dumps(data))
        self.assertEqual(service.process_conversation(self.conversation_id)["saved"], 0)
        self.assertEqual(self.repo.list_memories(), [])
        self.assertNotIn(secret.encode(), (self.root / "edith.db").read_bytes())

    def test_transient_llm_failure_keeps_messages_pending(self):
        self.user_turn("I prefer quiet workspaces.")
        llm = FakeLLM([RuntimeError("temporary provider failure")])
        service = ExtractionService(self.repo, llm, self.settings)
        with self.assertRaisesRegex(RuntimeError, "temporary"):
            service.process_conversation(self.conversation_id)
        self.assertTrue(all(row["extracted"] == 0 for row in self.repo.get_messages(self.conversation_id)))

    def test_worker_retries_transient_failure_and_marks_after_success(self):
        self.user_turn("I prefer quiet workspaces.")
        class RetryExtractor:
            attempts = 0
            def process_conversation(inner, conversation_id):
                inner.attempts += 1
                if inner.attempts == 1:
                    raise RuntimeError("temporary failure")
                self.repo.mark_messages_extracted([row["id"] for row in
                                                   self.repo.unextracted_messages(conversation_id)])
                return {"processed": 2, "saved": 0, "pending": 0, "skipped": 0}
        extractor = RetryExtractor()
        worker = ExtractionWorker(self.repo, extractor, self.settings, max_retries=2,
                                  backoff_seconds=0.001)
        worker.start(catch_up=False)
        worker.enqueue(self.conversation_id)
        self.assertTrue(worker.flush(timeout=2.0))
        self.assertEqual(extractor.attempts, 2)
        self.assertEqual(self.repo.unextracted_conversation_ids(), [])
        worker.stop()

    def test_existing_smolagents_model_adapter_uses_temperature_zero(self):
        class Model:
            def generate(inner, messages, stop_sequences, temperature):
                from smolagents import ChatMessage
                from smolagents.models import MessageRole
                inner.messages = messages
                inner.temperature = temperature
                return ChatMessage(role=MessageRole.ASSISTANT, content="[]")
        model = Model()
        result = SmolagentsLLMClient(model).complete("system", "user", temperature=0.0)
        self.assertEqual(result, "[]")
        self.assertEqual(model.temperature, 0.0)
        self.assertEqual([message.role.value for message in model.messages], ["system", "user"])

    def test_end_conversation_triggers_immediate_extraction(self):
        self.user_turn("I prefer quiet workspaces.")
        done = threading.Event()
        class EventExtractor:
            def process_conversation(inner, conversation_id):
                done.set()
                self.repo.mark_messages_extracted([row["id"] for row in
                                                   self.repo.unextracted_messages(conversation_id)])
                return {"processed": 2, "saved": 0, "pending": 0, "skipped": 0}
        worker = ExtractionWorker(self.repo, EventExtractor(), self.settings, max_retries=0)
        worker.start(catch_up=False)
        worker.end_conversation(self.conversation_id)
        self.assertTrue(done.wait(2.0))
        self.assertTrue(worker.flush(timeout=2.0))
        worker.stop()

    def test_batch_trigger_enqueues_without_waiting_for_idle_timer(self):
        settings = EdithConfig(extraction_idle_minutes=60, extraction_batch_messages=2)
        self.user_turn("I prefer quiet workspaces.")
        self.user_turn("I prefer morning meetings.")
        done = threading.Event()
        class EventExtractor:
            def process_conversation(inner, conversation_id):
                done.set()
                self.repo.mark_messages_extracted([row["id"] for row in
                                                   self.repo.unextracted_messages(conversation_id)])
                return {"processed": 4, "saved": 0, "pending": 0, "skipped": 0}
        worker = ExtractionWorker(self.repo, EventExtractor(), settings, max_retries=0)
        start = time.monotonic()
        worker.after_response(self.conversation_id)
        self.assertLess(time.monotonic() - start, 0.2)
        self.assertTrue(done.wait(2.0))
        self.assertTrue(worker.flush(timeout=2.0))
        worker.stop()

    def test_shutdown_flush_and_startup_catchup_extract_pending_messages(self):
        self.user_turn("I prefer quiet workspaces.")
        llm = FakeLLM([candidate()])
        extractor = ExtractionService(self.repo, llm, self.settings, synchronizer=self.sync)
        worker = ExtractionWorker(self.repo, extractor, self.settings, max_retries=0)
        worker.start(catch_up=False)
        worker.enqueue(self.conversation_id)
        self.assertTrue(worker.flush(timeout=3.0))
        self.assertEqual(len(self.repo.list_memories()), 1)
        self.assertEqual(self.repo.unextracted_conversation_ids(), [])
        worker.stop()

        # Simulate a restart with an unprocessed turn left behind before idle extraction.
        second_conversation = self.repo.create_conversation()
        self.repo.add_message(second_conversation, "user", "I prefer morning meetings.")
        llm2 = FakeLLM([candidate(content="Prefers morning meetings")])
        worker2 = ExtractionWorker(
            self.repo, ExtractionService(self.repo, llm2, self.settings),
            self.settings, max_retries=0)
        worker2.start(catch_up=True)
        self.assertTrue(worker2.flush(timeout=3.0))
        self.assertEqual(len(self.repo.list_memories()), 2)
        worker2.stop()

    def test_idle_timer_runs_extraction_without_blocking_response_hook(self):
        settings = EdithConfig(extraction_idle_minutes=0, extraction_batch_messages=8)
        self.user_turn("I prefer quiet workspaces.")
        done = threading.Event()
        class EventExtractor:
            def process_conversation(inner, conversation_id):
                done.set()
                self.repo.mark_messages_extracted([row["id"] for row in
                                                   self.repo.unextracted_messages(conversation_id)])
                return {"processed": 2, "saved": 0, "pending": 0, "skipped": 0}
        worker = ExtractionWorker(self.repo, EventExtractor(), settings, max_retries=0)
        start = time.monotonic()
        worker.after_response(self.conversation_id)
        self.assertLess(time.monotonic() - start, 0.2)
        self.assertTrue(done.wait(2.0))
        self.assertTrue(worker.flush(timeout=2.0))
        worker.stop()

    def test_shutdown_registration_installs_atexit_and_signal_flush_once(self):
        fake_worker = Mock()
        fake_worker.flush.return_value = True
        with patch("edith.integration.shutdown.atexit.register") as at_exit, \
             patch("edith.integration.shutdown.signal.getsignal", return_value=signal.SIG_IGN), \
             patch("edith.integration.shutdown.signal.signal") as install_handler:
            register_shutdown_flush(fake_worker, timeout_seconds=1.0)
            self.assertEqual(at_exit.call_count, 1)
            self.assertEqual(install_handler.call_count, 2)
            handlers = [call.args[1] for call in install_handler.call_args_list]
            handlers[0](signal.SIGINT, None)
            handlers[1](signal.SIGTERM, None)
        fake_worker.flush.assert_called_once_with(1.0)


if __name__ == "__main__":
    unittest.main()
