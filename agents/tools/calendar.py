"""Google Calendar tools with OAuth kept behind private helpers."""
import json
import os
import re
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from smolagents import tool

BASE_DIR = Path(__file__).resolve().parent.parent
_PENDING_DB = BASE_DIR / "calendar_pending.sqlite3"
_PENDING_TTL = timedelta(minutes=15)
_SCOPE = ["https://www.googleapis.com/auth/calendar.events"]
_ALERT_MINUTES = {"at_event": 0, "5_min_before": 5, "10_min_before": 10,
                  "15_min_before": 15, "30_min_before": 30, "1_hour_before": 60,
                  "1_day_before": 1440}


class CalendarSetupError(RuntimeError):
    """Safe, actionable calendar configuration error."""


def _config_path(env_name, default):
    value = os.getenv(env_name, default)
    path = Path(value).expanduser()
    return path if path.is_absolute() else BASE_DIR / path


def _service():
    """Load credentials only during a user-requested calendar operation."""
    client_secret = _config_path("GOOGLE_CLIENT_SECRET_FILE", "credentials.json")
    token_file = _config_path("GOOGLE_TOKEN_FILE", "token.json")
    if not client_secret.is_file():
        raise CalendarSetupError(
            "Google Calendar is not configured. Add your OAuth client file as "
            "credentials.json (or set GOOGLE_CLIENT_SECRET_FILE) and authenticate."
        )
    try:
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
        from googleapiclient.discovery import build
    except ImportError as exc:
        raise CalendarSetupError(
            "Google Calendar dependencies are missing. Install the Google API packages in requirements.txt."
        ) from exc

    try:
        credentials = None
        if token_file.is_file():
            credentials = Credentials.from_authorized_user_file(str(token_file), _SCOPE)
        if credentials and credentials.expired and credentials.refresh_token:
            credentials.refresh(Request())
        if not credentials or not credentials.valid:
            flow = InstalledAppFlow.from_client_secrets_file(str(client_secret), _SCOPE)
            credentials = flow.run_local_server(port=0)
        token_file.parent.mkdir(parents=True, exist_ok=True)
        token_file.write_text(credentials.to_json(), encoding="utf-8")
        return build("calendar", "v3", credentials=credentials, cache_discovery=False)
    except CalendarSetupError:
        raise
    except Exception as exc:
        # Do not include low-level Google auth exceptions; they may contain private paths/details.
        raise CalendarSetupError(
            "Google Calendar authentication failed. Check the local OAuth configuration and authenticate again."
        ) from exc


def _parse_time(value, tz_name=None):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Provide a date and time in ISO format, such as 2026-11-20T20:00.")
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("Invalid date/time. Use ISO format, such as 2026-11-20T20:00.") from exc
    if parsed.tzinfo is None:
        if not tz_name:
            raise ValueError("A timezone is required for local date/time values.")
        try:
            zone = ZoneInfo(tz_name)
        except ZoneInfoNotFoundError as exc:
            raise ValueError("Unknown timezone. Use an IANA timezone such as America/Toronto.") from exc
        candidate0 = parsed.replace(tzinfo=zone, fold=0)
        candidate1 = parsed.replace(tzinfo=zone, fold=1)
        valid0 = candidate0.astimezone(timezone.utc).astimezone(zone).replace(tzinfo=None) == parsed
        valid1 = candidate1.astimezone(timezone.utc).astimezone(zone).replace(tzinfo=None) == parsed
        if not valid0 and not valid1:
            raise ValueError("That local time does not exist because of a daylight-saving time change.")
        if valid0 and valid1 and candidate0.utcoffset() != candidate1.utcoffset():
            raise ValueError("That local time is ambiguous because of a daylight-saving time change; provide an ISO UTC offset.")
        parsed = candidate0 if valid0 else candidate1
    return parsed


def _validate_range(start, end, tz_name=None):
    start_dt = _parse_time(start, tz_name)
    end_dt = _parse_time(end, tz_name)
    if end_dt <= start_dt:
        raise ValueError("End time must be later than start time.")
    return start_dt, end_dt


def _event_body(title, description, start, end, tz_name, alert, location="", recurrence=None):
    if not isinstance(title, str) or not title.strip():
        raise ValueError("Event title cannot be empty.")
    if not isinstance(tz_name, str) or not tz_name:
        raise ValueError("Provide an IANA timezone, such as America/Toronto.")
    ZoneInfo(tz_name)  # Validate timezone before calling Google.
    start_dt, end_dt = _validate_range(start, end, tz_name)
    target_zone = ZoneInfo(tz_name)
    start_dt = start_dt.astimezone(target_zone)
    end_dt = end_dt.astimezone(target_zone)
    if start_dt <= datetime.now(target_zone):
        raise ValueError("Start time must be in the future.")
    body = {
        "summary": title.strip(), "description": description or "",
        "start": {"dateTime": start_dt.isoformat(), "timeZone": tz_name},
        "end": {"dateTime": end_dt.isoformat(), "timeZone": tz_name},
    }
    if location:
        body["location"] = location
    if recurrence:
        if not isinstance(recurrence, str) or not recurrence.startswith("RRULE:"):
            raise ValueError("Recurrence must be a Google Calendar rule beginning with RRULE:.")
        if not re.fullmatch(r"RRULE:FREQ=(DAILY|WEEKLY|MONTHLY|YEARLY)(;[A-Z]+=[A-Z0-9,-]+)*", recurrence):
            raise ValueError("Invalid recurrence rule.")
        body["recurrence"] = [recurrence]
    if alert is None:
        alert = "15_min_before"
    if alert == "None":
        body["reminders"] = {"useDefault": True}
    elif alert in _ALERT_MINUTES:
        body["reminders"] = {"useDefault": False, "overrides": [
            {"method": "popup", "minutes": _ALERT_MINUTES[alert]}]}
    else:
        raise ValueError("Unsupported alert. Use at_event, 5_min_before, 10_min_before, 15_min_before, 30_min_before, 1_hour_before, 1_day_before, or None.")
    return body, start_dt, end_dt


def _format_event(event):
    start = event.get("start", {}).get("dateTime", event.get("start", {}).get("date", "?"))
    end = event.get("end", {}).get("dateTime", event.get("end", {}).get("date", "?"))
    return (f"{event.get('summary', '(untitled)')} | {start} – {end}"
            f" | ID: {event.get('id', '?')}")


@contextmanager
def _pending_connect():
    db = sqlite3.connect(_PENDING_DB, timeout=10)
    try:
        db.execute("""CREATE TABLE IF NOT EXISTS pending (
            id TEXT PRIMARY KEY, action TEXT NOT NULL, event_id TEXT NOT NULL,
            changes TEXT NOT NULL, etag TEXT, created TEXT NOT NULL, status TEXT NOT NULL)""")
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def _stage(action, event_id, changes, etag):
    operation_id = str(uuid.uuid4())
    with _pending_connect() as db:
        db.execute("INSERT INTO pending VALUES (?, ?, ?, ?, ?, ?, 'pending')", (
            operation_id, action, event_id, json.dumps(changes), etag,
            datetime.now(timezone.utc).isoformat()))
    return (f"Proposed calendar {action} for event {event_id}: {json.dumps(changes, ensure_ascii=False)}\n"
            f"No change has been made. Reply `calendar confirm {operation_id}` to apply, "
            f"or `calendar reject {operation_id}` to cancel. This request expires in 15 minutes.")


def _apply_pending(operation_id, confirm):
    now = datetime.now(timezone.utc)
    with _pending_connect() as db:
        db.execute("BEGIN IMMEDIATE")
        row = db.execute("SELECT action,event_id,changes,etag,created,status FROM pending WHERE id=?",
                         (operation_id,)).fetchone()
        if not row or row[5] != "pending":
            return "Calendar confirmation is invalid, expired, or already handled."
        action, event_id, changes_json, etag, created, _ = row
        if now - datetime.fromisoformat(created) > _PENDING_TTL:
            db.execute("UPDATE pending SET status='expired' WHERE id=?", (operation_id,))
            return "Calendar confirmation expired; please prepare the change again."
        if not confirm:
            db.execute("UPDATE pending SET status='rejected' WHERE id=?", (operation_id,))
            return "Calendar change rejected; no event was changed."
        db.execute("UPDATE pending SET status='applying' WHERE id=?", (operation_id,))
        db.commit()
    try:
        service = _service()
        resource = service.events()
        current = resource.get(calendarId="primary", eventId=event_id).execute()
        if etag and current.get("etag") != etag:
            with _pending_connect() as db:
                db.execute("UPDATE pending SET status='stale' WHERE id=?", (operation_id,))
            return "The event changed since the proposal. Nothing was applied; search again and prepare a new change."
        if action == "delete":
            resource.delete(calendarId="primary", eventId=event_id).execute()
            result = f"Event {event_id} deleted."
        else:
            changes = json.loads(changes_json)
            # Preserve all untouched fields, including location and attendee data.
            current.update(changes)
            result = _format_event(resource.update(calendarId="primary", eventId=event_id,
                                                    body=current).execute())
        with _pending_connect() as db:
            db.execute("UPDATE pending SET status='applied' WHERE id=?", (operation_id,))
        return result
    except Exception as exc:
        with _pending_connect() as db:
            db.execute("UPDATE pending SET status='failed' WHERE id=?", (operation_id,))
        if isinstance(exc, CalendarSetupError):
            return str(exc)
        return "Google Calendar could not apply the change. Check the event and try again."


def confirm_pending(operation_id):
    return _apply_pending(operation_id, True)


def reject_pending(operation_id):
    return _apply_pending(operation_id, False)


@tool
def create_reminder(title: str, description: str, start_time: str, duration_minutes: int,
                    timezone: str = "America/Toronto", alert: str | None = None,
                    recurrence: str | None = None) -> str:
    """Create a calendar reminder, optionally recurring with a native RRULE.

    Args:
        title: Short reminder title.
        description: Reminder details.
        start_time: ISO local date/time, for example 2026-11-20T20:00.
        timezone: IANA timezone, for example America/Toronto.
        duration_minutes: Positive reminder duration.
        alert: at_event, 5_min_before, 10_min_before, 15_min_before, 30_min_before, 1_hour_before, 1_day_before, or None for Google defaults.
        recurrence: Optional Google RRULE, for example RRULE:FREQ=WEEKLY;BYDAY=SU.
    """
    if not isinstance(duration_minutes, int) or duration_minutes <= 0 or duration_minutes > 10080:
        raise ValueError("Duration must be a positive number of minutes no greater than one week.")
    start_dt = _parse_time(start_time, timezone)
    end_dt = start_dt + timedelta(minutes=duration_minutes)
    body, start_dt, _ = _event_body(title, description, start_dt.isoformat(), end_dt.isoformat(),
                                    timezone, alert, recurrence=recurrence)
    event = _service().events().insert(calendarId="primary", body=body).execute()
    return f"Reminder created: {event.get('summary')} on {start_dt.strftime('%Y-%m-%d %H:%M %Z')} (ID: {event.get('id')})."


@tool
def create_event(title: str, description: str, start_time: str, end_time: str,
                 timezone: str = "America/Toronto", location: str = "", alert: str | None = None) -> str:
    """Schedule an event with a local timezone and optional location.

    Args:
        title: Event title.
        description: Event description, or empty string.
        start_time: ISO date/time, for example 2026-11-20T14:00.
        end_time: ISO date/time later than the start.
        timezone: IANA timezone, for example America/Toronto.
        location: Optional location.
        alert: Supported reminder alert or None for a 15-minute alert.
    """
    body, start_dt, _ = _event_body(title, description, start_time, end_time, timezone, alert, location)
    event = _service().events().insert(calendarId="primary", body=body).execute()
    return f"Event created: {_format_event(event)}"


@tool
def get_upcoming_events(days: int = 7) -> str:
    """List calendar events chronologically over the next 1–90 days.

    Args:
        days: Number of days to look ahead (default 7).
    """
    if not isinstance(days, int) or not 1 <= days <= 90:
        raise ValueError("days must be between 1 and 90.")
    now = datetime.now(timezone.utc)
    items = _service().events().list(calendarId="primary", timeMin=now.isoformat(),
        timeMax=(now + timedelta(days=days)).isoformat(), singleEvents=True,
        orderBy="startTime", maxResults=250).execute().get("items", [])
    return "\n".join(_format_event(item) for item in items) if items else f"No upcoming events in the next {days} days."


@tool
def check_availability(start_time: str, end_time: str) -> str:
    """Check whether a period overlaps events. Times must be ISO date/times with offsets or timezone.

    Args:
        start_time: ISO date/time, e.g. 2026-11-20T14:00-05:00.
        end_time: ISO date/time after start.
    """
    start, end = _validate_range(start_time, end_time)
    items = _service().events().list(calendarId="primary", timeMin=start.astimezone(timezone.utc).isoformat(),
        timeMax=end.astimezone(timezone.utc).isoformat(), singleEvents=True, orderBy="startTime",
        maxResults=100).execute().get("items", [])
    return "Available: no overlapping events." if not items else "Busy during this period:\n" + "\n".join(_format_event(item) for item in items)


@tool
def delete_event(event_id: str) -> str:
    """Prepare deletion of an event; it is deleted only after the user confirms the shown operation ID.

    Args:
        event_id: Google Calendar event ID.
    """
    event = _service().events().get(calendarId="primary", eventId=event_id).execute()
    return _stage("delete", event_id, {"title": event.get("summary", "(untitled)")}, event.get("etag"))


@tool
def update_event(event_id: str, title: str = "", start_time: str = "", end_time: str = "",
                 description: str = "", alert: str | None = None) -> str:
    """Prepare an event update; omitted values remain unchanged, and user confirmation is required.

    Args:
        event_id: Google Calendar event ID.
        title: New title, or empty to keep the current title.
        start_time: New ISO start time, or empty to keep it.
        end_time: New ISO end time, or empty to keep it.
        description: New description, or empty to keep it.
        alert: New supported alert, or None to keep current reminders.
    """
    service = _service()
    event = service.events().get(calendarId="primary", eventId=event_id).execute()
    changes = {}
    if title:
        changes["summary"] = title
    if description:
        changes["description"] = description
    if bool(start_time) != bool(end_time):
        raise ValueError("To change the time, provide both start_time and end_time.")
    if start_time:
        tz_name = event.get("start", {}).get("timeZone")
        new_start, new_end = _validate_range(start_time, end_time, tz_name)
        if new_start <= datetime.now(new_start.tzinfo):
            raise ValueError("Start time must be in the future.")
        changes["start"] = {"dateTime": new_start.isoformat(), "timeZone": tz_name} if tz_name else {"dateTime": new_start.isoformat()}
        changes["end"] = {"dateTime": new_end.isoformat(), "timeZone": tz_name} if tz_name else {"dateTime": new_end.isoformat()}
    if alert is not None:
        if alert == "None":
            changes["reminders"] = {"useDefault": True}
        elif alert in _ALERT_MINUTES:
            changes["reminders"] = {"useDefault": False, "overrides": [{"method": "popup", "minutes": _ALERT_MINUTES[alert]}]}
        else:
            raise ValueError("Unsupported alert value.")
    if not changes:
        raise ValueError("Provide at least one value to update.")
    return _stage("update", event_id, changes, event.get("etag"))


@tool
def search_events(query: str, days: int = 90) -> str:
    """Search event titles and descriptions over the next 1–365 days.

    Args:
        query: Search text.
        days: Search horizon in days (default 90).
    """
    if not query or not query.strip():
        raise ValueError("Search query cannot be empty.")
    if not isinstance(days, int) or not 1 <= days <= 365:
        raise ValueError("days must be between 1 and 365.")
    now = datetime.now(timezone.utc)
    items = _service().events().list(calendarId="primary", timeMin=now.isoformat(),
        timeMax=(now + timedelta(days=days)).isoformat(), singleEvents=True,
        orderBy="startTime", q=query.strip(), maxResults=100).execute().get("items", [])
    return "\n".join(_format_event(item) for item in items) if items else f"No matching events found for {query!r}."
