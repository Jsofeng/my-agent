import tempfile
import unittest
from pathlib import Path

from edith.config import EdithConfig
from edith.db import MemoryRepository
from edith.layers.short_term import RollingSummaryService, build_short_term_context


class FakeSummaryClient:
    def __init__(self):
        self.prompts = []

    def complete(self, system_prompt, user_prompt, temperature=0.0):
        self.prompts.append((system_prompt, user_prompt, temperature))
        return "Synthetic rolling summary of the earlier turns."


class ShortTermTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.repo = MemoryRepository(Path(self.temp.name) / "edith.db")
        self.addCleanup(self.repo.close)
        self.conversation_id = self.repo.create_conversation()

    def test_maintains_summary_over_threshold_and_injects_recent_turns(self):
        for turn in range(10):
            self.repo.add_message(self.conversation_id, "user",
                                  "Synthetic turn %d discusses a fictional topic and has several useful words." % turn)
            self.repo.add_message(self.conversation_id, "assistant",
                                  "Synthetic reply explains the imaginary example in a few sentences.")
        settings = EdithConfig(rolling_summary_token_threshold=30,
                               rolling_summary_recent_messages=4,
                               short_term_context_token_budget=500)
        client = FakeSummaryClient()
        service = RollingSummaryService(self.repo, client, settings)
        self.assertTrue(service.maintain(self.conversation_id))
        summary = self.repo.latest_summary(self.conversation_id)
        self.assertIn("Synthetic rolling summary", summary["summary"])
        self.assertEqual(summary["covers_up_to_message_id"], 16)
        self.assertEqual(client.prompts[0][2], 0.0)
        context = build_short_term_context(
            self.repo, self.conversation_id,
            "Synthetic turn 9 discusses a fictional topic and has several useful words.", settings)
        self.assertIn("<conversation_context>", context)
        self.assertIn("rolling_summary", context)
        self.assertNotIn("Synthetic turn 9", context)
        self.assertIn("Synthetic reply", context)

    def test_summary_is_not_rewritten_without_new_covered_turns(self):
        for turn in range(6):
            self.repo.add_message(self.conversation_id, "user", "A" * 100)
            self.repo.add_message(self.conversation_id, "assistant", "B" * 100)
        settings = EdithConfig(rolling_summary_token_threshold=10,
                               rolling_summary_recent_messages=2)
        client = FakeSummaryClient()
        service = RollingSummaryService(self.repo, client, settings)
        self.assertTrue(service.maintain(self.conversation_id))
        first_id = self.repo.latest_summary(self.conversation_id)["id"]
        self.assertFalse(service.maintain(self.conversation_id))
        self.assertEqual(self.repo.latest_summary(self.conversation_id)["id"], first_id)
        self.assertEqual(len(client.prompts), 1)


if __name__ == "__main__":
    unittest.main()
