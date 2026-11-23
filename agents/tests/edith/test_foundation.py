import concurrent.futures
import tempfile
import unittest
from pathlib import Path

from edith.db import MemoryRepository
from edith.models import MemoryInput
from edith.extraction.secrets import scan_secrets
from edith.layers.episodic import EpisodicMemoryService
from edith.layers.semantic import SemanticMemoryService


class FoundationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.db_path = Path(self.temp.name) / "edith.db"
        self.repo = MemoryRepository(self.db_path)
        self.addCleanup(self.repo.close)

    def test_conversation_messages_summary_and_restart(self):
        conversation_id = self.repo.create_conversation("test conversation")
        user_id = self.repo.add_message(conversation_id, "user", "I enjoy gardening.")
        self.repo.add_message(conversation_id, "assistant", "I will keep that in mind.")
        self.repo.save_summary(conversation_id, "User enjoys gardening.", user_id)
        self.repo.close()
        restarted = MemoryRepository(self.db_path)
        self.addCleanup(restarted.close)
        self.assertEqual([m["role"] for m in restarted.get_messages(conversation_id)],
                         ["user", "assistant"])
        self.assertEqual(restarted.latest_summary(conversation_id)["summary"],
                         "User enjoys gardening.")

    def test_layer_services_create_and_list(self):
        episode_id = EpisodicMemoryService(self.repo).add(
            "achievement", "Completed a personal milestone", occurred_at="2026-01-02T00:00:00Z")
        fact_id = SemanticMemoryService(self.repo).add(
            "preference", "Prefers quiet workspaces", subject="user", predicate="workspace")
        self.assertEqual(self.repo.get_memory(episode_id)["layer"], "episodic")
        self.assertEqual([x["id"] for x in SemanticMemoryService(self.repo).profile()], [fact_id])

    def test_soft_and_hard_delete_are_content_free(self):
        soft_id = self.repo.create_memory(MemoryInput(
            layer="semantic", kind="profile_fact", content="unique-soft-content-xyz"))
        self.repo.delete_memory(soft_id)
        self.assertEqual(self.repo.get_memory(soft_id)["status"], "deleted")
        hard_id = self.repo.create_memory(MemoryInput(
            layer="semantic", kind="profile_fact", content="unique-hard-content-abc"))
        entity_id = self.repo.add_entity("Temporary Entity", "other")
        self.repo.link_entity(hard_id, entity_id)
        self.repo.delete_memory(hard_id, hard=True)
        self.assertIsNone(self.repo.get_memory(hard_id))
        db = self.repo.connections.get()
        self.assertEqual(db.execute("SELECT count(*) FROM memory_entities WHERE memory_id=?",
                                    (hard_id,)).fetchone()[0], 0)
        self.assertEqual(db.execute("PRAGMA secure_delete").fetchone()[0], 1)
        event = db.execute("SELECT action,detail_json FROM memory_events WHERE memory_id=? ORDER BY id DESC",
                           (hard_id,)).fetchone()
        self.assertEqual(tuple(event), ("delete", "{}"))
        db.commit()
        self.assertEqual(db.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()[0], 0)
        self.repo.close()
        raw = self.db_path.read_bytes()
        self.assertNotIn(b"unique-hard-content-abc", raw)

    def test_migrations_are_idempotent(self):
        from edith.db.migrations import apply_migrations
        db = self.repo.connections.get()
        apply_migrations(db)
        apply_migrations(db)
        self.assertEqual(db.execute("SELECT version FROM schema_version").fetchone()[0], 1)
        self.assertEqual(db.execute("SELECT count(*) FROM sqlite_master WHERE type='table' "
                                    "AND name='memories'").fetchone()[0], 1)

    def test_concurrent_writes_share_repository(self):
        conversation_id = self.repo.create_conversation()
        def write(index):
            return self.repo.add_message(conversation_id, "user", "turn %d" % index)
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            ids = list(pool.map(write, range(40)))
        self.assertEqual(len(set(ids)), 40)
        self.assertEqual(len(self.repo.get_messages(conversation_id)), 40)

    def test_secret_scan_redacts_and_never_returns_secret_in_safe_text(self):
        secret = "sk-1234567890abcdefghijklmn"
        result = scan_secrets("My key is " + secret)
        self.assertTrue(result.has_secrets)
        self.assertNotIn(secret, result.safe_text)
        self.assertEqual(result.safe_text, "My key is [REDACTED]")


if __name__ == "__main__":
    unittest.main()
