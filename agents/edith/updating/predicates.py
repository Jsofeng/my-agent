"""Predicate cardinality registry for semantic facts."""

PREDICATE_CARDINALITY = {
    "current_employer": "single",
    "current_location": "single",
    "preferred_name": "single",
    "skill": "multi",
    "interest": "multi",
    "goal": "multi",
    "project": "multi",
}


def cardinality(predicate: str) -> str:
    """Return the configured cardinality, defaulting safely to multi-valued."""
    return PREDICATE_CARDINALITY.get(predicate, "multi")
