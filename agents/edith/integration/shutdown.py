"""Bounded process-shutdown flush for pending extraction work."""

from __future__ import annotations

import atexit
import logging
import signal
import threading
from .background_worker import ExtractionWorker
from ..config import config


logger = logging.getLogger(__name__)
_registration_lock = threading.Lock()
_registered: set[int] = set()


def register_shutdown_flush(worker: ExtractionWorker,
                           timeout_seconds: float = config.shutdown_flush_timeout_seconds) -> None:
    worker_id = id(worker)
    with _registration_lock:
        if worker_id in _registered:
            return
        _registered.add(worker_id)

    def flush() -> None:
        with flush_lock:
            if flush_started.is_set():
                return
            flush_started.set()
        try:
            completed = worker.flush(timeout_seconds)
            if not completed:
                logger.error("Extraction shutdown flush timed out")
        except Exception as exc:
            logger.error("Extraction shutdown flush failed (error_type=%s)", type(exc).__name__)

    flush_started = threading.Event()
    flush_lock = threading.Lock()
    atexit.register(flush)
    for signum in (signal.SIGINT, signal.SIGTERM):
        previous = signal.getsignal(signum)

        def handle(signum, frame, previous_handler=previous):
            flush()
            if callable(previous_handler):
                previous_handler(signum, frame)
            elif previous_handler == signal.SIG_DFL:
                signal.signal(signum, signal.SIG_DFL)
                raise SystemExit(128 + signum)

        try:
            signal.signal(signum, handle)
        except ValueError:
            # Registration from a non-main thread cannot install signal handlers.
            continue
