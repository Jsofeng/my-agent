"""Email drafting and explicitly confirmed Gmail delivery."""
import hashlib
import os
import smtplib
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
from pathlib import Path

from dotenv import load_dotenv
from smolagents import tool

load_dotenv()

SENDER_EMAIL = os.getenv("SENDER_EMAIL")
SENDER_PASSWORD = os.getenv("SENDER_PASSWORD")
DRAFT_TTL = timedelta(hours=24)
_DB_PATH = Path(__file__).with_name("email_drafts.sqlite3")


def _connect():
    db = sqlite3.connect(_DB_PATH, timeout=10)
    db.execute("""CREATE TABLE IF NOT EXISTS drafts (
        id TEXT PRIMARY KEY, recipient TEXT NOT NULL, subject TEXT NOT NULL,
        body TEXT NOT NULL, digest TEXT NOT NULL, created_at TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'pending')""")
    return db


def _digest(recipient, subject, body):
    return hashlib.sha256("\0".join((recipient, subject, body)).encode()).hexdigest()


@tool
def generate_draft(recipient: str, subject: str, body: str) -> str:
    """Create and display an email preview. This operation never sends email.

    Args:
        recipient: Email address of the recipient.
        subject: Email subject.
        body: Complete plain text email body.
    """
    if not all(isinstance(v, str) and v.strip() for v in (recipient, subject, body)):
        raise ValueError("Recipient, subject, and body must all be non-empty.")
    draft_id = str(uuid.uuid4())
    digest = _digest(recipient, subject, body)
    created = datetime.now(timezone.utc)
    with _connect() as db:
        db.execute("INSERT INTO drafts VALUES (?, ?, ?, ?, ?, ?, 'pending')",
                   (draft_id, recipient, subject, body, digest, created.isoformat()))
    return (f"Email draft (ID: {draft_id})\nTo: {recipient}\nSubject: {subject}\n\n{body}\n\n"
            "Not sent. To send this exact draft, reply in chat: send " + draft_id +
            "\nTo discard it, reply: reject " + draft_id)


def reject_draft(draft_id: str) -> str:
    """Discard a pending draft without sending it."""
    with _connect() as db:
        cur = db.execute("UPDATE drafts SET status='rejected' WHERE id=? AND status='pending'", (draft_id,))
    return "Draft rejected; no email was sent." if cur.rowcount else "Draft is unavailable, expired, or already handled."


def send_confirmed_draft(draft_id: str) -> str:
    """Send only a pending, unexpired draft explicitly selected by the user in chat."""
    now = datetime.now(timezone.utc)
    with _connect() as db:
        db.execute("BEGIN IMMEDIATE")
        row = db.execute("SELECT recipient,subject,body,digest,created_at,status FROM drafts WHERE id=?", (draft_id,)).fetchone()
        if not row or row[5] != 'pending':
            raise ValueError("Draft is unavailable, expired, or already handled.")
        recipient, subject, body, digest, created_at, _ = row
        if now - datetime.fromisoformat(created_at) > DRAFT_TTL:
            db.execute("UPDATE drafts SET status='expired' WHERE id=?", (draft_id,))
            db.commit()
            raise ValueError("Draft expired; create a new draft.")
        if digest != _digest(recipient, subject, body):
            db.execute("UPDATE drafts SET status='invalid' WHERE id=?", (draft_id,))
            db.commit()
            raise ValueError("Draft changed; create and confirm a new draft.")
        if not SENDER_EMAIL or not SENDER_PASSWORD:
            raise RuntimeError("Gmail sender credentials are not configured.")
        # Reserve before delivery so the same confirmation cannot be replayed.
        db.execute("UPDATE drafts SET status='sending' WHERE id=?", (draft_id,))
        db.commit()
    msg = EmailMessage()
    msg["From"], msg["To"], msg["Subject"] = SENDER_EMAIL, recipient, subject
    msg.set_content(body)
    try:
        with smtplib.SMTP("smtp.gmail.com", 587, timeout=20) as server:
            server.starttls()
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.send_message(msg)
    except Exception:
        with _connect() as db:
            db.execute("UPDATE drafts SET status='failed' WHERE id=? AND status='sending'", (draft_id,))
        raise
    with _connect() as db:
        db.execute("UPDATE drafts SET status='sent' WHERE id=? AND status='sending'", (draft_id,))
    return f"Confirmed draft {draft_id} sent to {recipient}."
