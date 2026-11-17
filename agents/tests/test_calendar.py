import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from tools import calendar as cal


class CalendarToolsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.db_path = Path(self.temp.name) / "pending.sqlite3"
        self.api = Mock()
        self.events = self.api.events.return_value
        self.service_patch = patch.object(cal, "_service", return_value=self.api)
        self.service_patch.start()
        self.addCleanup(self.service_patch.stop)
        self.db_patch = patch.object(cal, "_PENDING_DB", self.db_path)
        self.db_patch.start()
        self.addCleanup(self.db_patch.stop)

    @staticmethod
    def call(tool, **kwargs):
        return tool.forward(**kwargs)

    def test_missing_oauth_client_is_actionable_without_reading_file(self):
        self.service_patch.stop()
        with patch.object(cal, "_config_path", side_effect=[Path(self.temp.name) / "missing-client.json",
                                                             Path(self.temp.name) / "missing-token.json"]):
            with self.assertRaisesRegex(cal.CalendarSetupError, "not configured"):
                cal._service()
        self.service_patch.start()

    def test_oauth_token_refresh_and_service_creation_are_mocked(self):
        self.service_patch.stop()
        credentials = Mock(valid=False, expired=True, refresh_token="test-only")
        credentials.refresh.side_effect = lambda _request: setattr(credentials, "valid", True)
        mock_service = Mock()
        with patch.object(cal, "_config_path", side_effect=[Path(self.temp.name) / "mock-client.json",
                                                             Path(self.temp.name) / "mock-token.json"]), \
             patch.object(cal.Path, "is_file", return_value=True), \
             patch.object(cal.Path, "write_text") as write_token, \
             patch("google.oauth2.credentials.Credentials.from_authorized_user_file",
                   return_value=credentials) as load_token, \
             patch("google.auth.transport.requests.Request") as request_class, \
             patch("google_auth_oauthlib.flow.InstalledAppFlow.from_client_secrets_file",
                   side_effect=AssertionError("OAuth flow must not run in tests")), \
             patch("googleapiclient.discovery.build", return_value=mock_service) as build:
            self.assertIs(cal._service(), mock_service)
        credentials.refresh.assert_called_once_with(request_class.return_value)
        load_token.assert_called_once()
        build.assert_called_once_with("calendar", "v3", credentials=credentials, cache_discovery=False)
        write_token.assert_called_once()
        self.service_patch.start()

    def test_create_reminder_has_default_alert_and_recurrence(self):
        self.events.insert.return_value.execute.return_value = {"id": "evt-1", "summary": "Study"}
        result = self.call(cal.create_reminder, title="Study", description="Review", 
            start_time="2026-11-20T19:00", timezone="America/Toronto", duration_minutes=30,
            alert=None, recurrence="RRULE:FREQ=WEEKLY;BYDAY=FR")
        payload = self.events.insert.call_args.kwargs["body"]
        self.assertIn("Reminder created", result)
        self.assertEqual(payload["reminders"]["overrides"][0]["minutes"], 15)
        self.assertEqual(payload["recurrence"], ["RRULE:FREQ=WEEKLY;BYDAY=FR"])

    def test_create_event_and_invalid_range(self):
        self.events.insert.return_value.execute.return_value = {"id": "evt-2", "summary": "Meeting",
            "start": {"dateTime": "2026-11-20T10:00:00-05:00"}, "end": {"dateTime": "2026-11-20T11:00:00-05:00"}}
        self.call(cal.create_event, title="Meeting", description="", start_time="2026-11-20T10:00",
            end_time="2026-11-20T11:00", timezone="America/Toronto")
        with self.assertRaisesRegex(ValueError, "later"):
            self.call(cal.create_event, title="Bad", description="", start_time="2026-11-20T11:00",
                end_time="2026-11-20T10:00", timezone="America/Toronto")

    def test_list_availability_and_search(self):
        self.events.list.return_value.execute.return_value = {"items": [{"id": "evt-3", "summary": "Class",
            "start": {"dateTime": "2026-11-20T14:00:00-05:00"},
            "end": {"dateTime": "2026-11-20T15:00:00-05:00"}}]}
        self.assertIn("Class", self.call(cal.get_upcoming_events, days=7))
        self.assertIn("Busy", self.call(cal.check_availability,
            start_time="2026-11-20T14:00-05:00", end_time="2026-11-20T16:00-05:00"))
        self.assertIn("evt-3", self.call(cal.search_events, query="Class", days=30))

    def test_update_requires_confirmation_then_applies_once(self):
        original = {"id": "evt-4", "etag": "v1", "summary": "Old", "start": {"dateTime": "2026-11-20T10:00:00-05:00", "timeZone": "America/Toronto"},
                    "end": {"dateTime": "2026-11-20T11:00:00-05:00", "timeZone": "America/Toronto"}}
        self.events.get.return_value.execute.return_value = dict(original)
        self.events.update.return_value.execute.return_value = {**original, "summary": "New"}
        proposal = self.call(cal.update_event, event_id="evt-4", title="New", start_time="", end_time="", description="", alert=None)
        op_id = re.search(r"calendar confirm ([0-9a-f-]{36})", proposal).group(1)
        self.events.get.return_value.execute.return_value = dict(original)
        self.assertIn("New", cal.confirm_pending(op_id))
        self.events.update.assert_called_once()
        self.assertIn("already handled", cal.confirm_pending(op_id))

    def test_delete_is_only_applied_after_confirmation_and_rejection_is_safe(self):
        self.events.get.return_value.execute.return_value = {"id": "evt-5", "etag": "v1", "summary": "Delete me"}
        proposal = self.call(cal.delete_event, event_id="evt-5")
        op_id = re.search(r"calendar confirm ([0-9a-f-]{36})", proposal).group(1)
        self.assertIn("no event was changed", cal.reject_pending(op_id))
        self.events.delete.assert_not_called()
        proposal = self.call(cal.delete_event, event_id="evt-5")
        op_id = re.search(r"calendar confirm ([0-9a-f-]{36})", proposal).group(1)
        self.events.get.return_value.execute.return_value = {"id": "evt-5", "etag": "v1", "summary": "Delete me"}
        self.assertIn("deleted", cal.confirm_pending(op_id))
        self.events.delete.assert_called_once()

    def test_expired_confirmation_and_dst_validation(self):
        self.events.get.return_value.execute.return_value = {"id": "evt-6", "etag": "v1", "summary": "Old"}
        proposal = self.call(cal.delete_event, event_id="evt-6")
        op_id = re.search(r"calendar confirm ([0-9a-f-]{36})", proposal).group(1)
        with cal._pending_connect() as db:
            db.execute("UPDATE pending SET created=? WHERE id=?", ("2000-01-01T00:00:00+00:00", op_id))
        self.assertIn("expired", cal.confirm_pending(op_id))
        self.events.delete.assert_not_called()
        with self.assertRaisesRegex(ValueError, "does not exist"):
            cal._parse_time("2026-03-08T02:30", "America/Toronto")
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            cal._parse_time("2026-11-01T01:30", "America/Toronto")


if __name__ == "__main__":
    unittest.main()
