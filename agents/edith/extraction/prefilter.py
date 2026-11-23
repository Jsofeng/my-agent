"""Cheap deterministic filter to avoid extraction calls for low-value turns."""

from __future__ import annotations

import re
from typing import Sequence


_TRIVIAL = re.compile(
    r"(?i)^(?:hi|hello|hey|thanks?|thank you|ok(?:ay)?|sure|great|got it|lol|haha|bye)[!. ]*$"
)
_COMMAND = re.compile(
    r"(?i)^(?:/|search\b|look up\b|calculate\b|what is\b|who is\b|when was\b|"
    r"tell me about\b|weather\b|translate\b|summarize\b)"
)
_PERSONAL = re.compile(
    r"(?i)\b(?:i|i'm|i am|i've|i have|my|me|we|our|remember that|please remember|"
    r"i prefer|i like|i dislike|i want|i plan|i decided|i chose|i started|i finished)\b"
)


def is_extractable_message(text: str) -> bool:
    """Return true only for a plausible user-specific statement or correction.

    The heuristic skips short acknowledgements, command-like requests, and generic
    questions without first-person or explicit remember/correction language. It is
    intentionally permissive once a user-specific signal appears; the extractor's
    schema and importance/confidence thresholds perform the deeper filtering.
    """
    stripped = text.strip()
    if len(stripped) < 12 or _TRIVIAL.fullmatch(stripped):
        return False
    if _COMMAND.match(stripped) and not _PERSONAL.search(stripped):
        return False
    return bool(_PERSONAL.search(stripped))


def filter_messages(messages: Sequence[str]) -> list[str]:
    return [message for message in messages if is_extractable_message(message)]
