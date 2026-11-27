import json
import tempfile
import unittest
from pathlib import Path

from edith.db import MemoryRepository
from edith.inspector import browse_memories, export_memories, memory_history
from edith.models import MemoryInput


class InspectorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.repo = MemoryRepository(self.root / "edith.db")
        self.addCleanup(self.repo.close)

    def test_filter_browse_history_and_fresh_json_export(self):
        original_id = self.repo.create_memory(MemoryInput(
            layer="semantic", kind="preference", content="Likes a quiet workspace",
            subject="user", predicate="workspace"))
        entity_id = self.repo.add_entity("FictionalProject", "project", ["Project Alias"])
        self.repo.link_entity(original_id, entity_id)
        updated_id = self.repo.supersede_memory(original_id, MemoryInput(
            layer="semantic", kind="preference", content="Prefers a quiet workspace",
            subject="user", predicate="workspace"))
        active = browse_memories(self.repo, layer="semantic", kind="preference",
                                 entity="Project Alias", status="active")
        self.assertEqual([row["id"] for row in active], [updated_id])
        self.assertNotIn("embedding", active[0])
        history = memory_history(self.repo, updated_id)
        self.assertEqual([row["id"] for row in history], [original_id, updated_id])
        export_path = Path(export_memories(self.repo))
        self.addCleanup(lambda: export_path.unlink(missing_ok=True))
        payload = json.loads(export_path.read_text(encoding="utf-8"))
        self.assertEqual([row["id"] for row in payload["memories"]], [original_id, updated_id])
        self.assertNotIn("embedding", payload["memories"][0])

    def test_invalid_filters_fail_clearly(self):
        with self.assertRaises(ValueError):
            browse_memories(self.repo, layer="other")


if __name__ == "__main__":
    unittest.main()
