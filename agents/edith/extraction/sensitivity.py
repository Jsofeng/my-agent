"""Conservative lexical screening for sensitive memory categories."""

from __future__ import annotations

import re


SENSITIVE_CATEGORIES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("health", re.compile(r"(?i)\b(?:health|medical|diagnos(?:is|ed)|medication|prescription|therapy|disability|illness|disease)\b")),
    ("financial_account_details", re.compile(r"(?i)\b(?:bank account|account number|routing number|credit card|debit card|financial account)\b")),
    ("precise_location", re.compile(r"(?i)\b(?:home address|street address|live at|gps coordinates|precise location)\b")),
    ("sexual_orientation", re.compile(r"(?i)\b(?:sexual orientation|gay|lesbian|bisexual|asexual|pansexual)\b")),
    ("religion", re.compile(r"(?i)\b(?:religion|religious|christian|muslim|jewish|hindu|sikh|buddhist|atheist)\b")),
    ("politics", re.compile(r"(?i)\b(?:political party|political affiliation|political ideology|voted for|campaign volunteer)\b")),
    ("immigration_status", re.compile(r"(?i)\b(?:immigration status|visa status|asylum status|undocumented|refugee status)\b")),
    ("minors_data", re.compile(r"(?i)\b(?:my minor child|my underage child|my son is \d{1,2}|my daughter is \d{1,2})\b")),
)


def sensitive_categories(text: str) -> tuple[str, ...]:
    """Return matching sensitive categories without retaining candidate content."""
    return tuple(category for category, pattern in SENSITIVE_CATEGORIES if pattern.search(text))
