"""Lazy APScheduler setup; imports and starts only when explicitly requested."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from ..config import config
from .consolidator import MemoryConsolidator


def create_scheduler(consolidator: MemoryConsolidator, timezone_name: str | None = None):
    """Create a single-instance nightly scheduler without starting it."""
    from apscheduler.schedulers.background import BackgroundScheduler
    scheduler = BackgroundScheduler(timezone=timezone_name or config.user_timezone)
    scheduler.add_job(consolidator.run, "cron", hour=3, minute=0, id="edith_consolidation",
                      max_instances=1, coalesce=True, misfire_grace_time=3600)
    return scheduler


def run_if_overdue(repository, consolidator: MemoryConsolidator, now: datetime | None = None) -> bool:
    """Run a light catch-up when the last completed consolidation predates yesterday."""
    now = now or datetime.now(ZoneInfo(config.user_timezone))
    row = repository.connections.get().execute(
        "SELECT finished_at FROM consolidation_runs WHERE status='complete' "
        "ORDER BY id DESC LIMIT 1"
    ).fetchone()
    if row is None:
        consolidator.run()
        return True
    try:
        finished = datetime.fromisoformat(str(row[0]).replace("Z", "+00:00"))
        age_seconds = (now.astimezone(ZoneInfo("UTC")) - finished.astimezone(ZoneInfo("UTC"))).total_seconds()
    except (TypeError, ValueError):
        age_seconds = 86401
    if age_seconds > 86400:
        consolidator.run()
        return True
    return False
