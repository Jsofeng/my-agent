"""Regex-based removal of common secrets before any candidate reaches storage."""

from __future__ import annotations

import re

from ..models import SecretScanResult


PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("credential", re.compile(r"(?i)\b(?:sk-[A-Za-z0-9_-]{16,}|gh[pousr]_[A-Za-z0-9_]{20,}|xox[baprs]-[A-Za-z0-9-]{10,})\b")),
    ("token", re.compile(r"(?i)\b(?:bearer\s+)?[A-Za-z0-9_-]{32,}\b")),
    ("card_number", re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)")),
    ("government_id", re.compile(r"(?i)\b(?:\d{3}-\d{3}-\d{3}|\d{3}[- ]?\d{3}[- ]?\d{3}|\d{3}-\d{2}-\d{4})\b")),
)


def scan_secrets(text: str, redact: bool = True) -> SecretScanResult:
    """Find common credential, token, payment-card, and government-ID patterns."""
    matches: list[str] = []
    safe = text
    for category, pattern in PATTERNS:
        found = list(pattern.finditer(safe))
        if found:
            matches.extend([category] * len(found))
            if redact:
                safe = pattern.sub("[REDACTED]", safe)
    return SecretScanResult(safe_text=safe, matches=tuple(matches))
