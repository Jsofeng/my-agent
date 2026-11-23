"""Single-threaded extraction queue with idle triggers and bounded retries."""

from __future__ import annotations

import logging
import queue
import threading
import time
from time import monotonic
from typing import Optional

from ..config import EdithConfig, config
from ..db.repository import MemoryRepository
from ..extraction.extractor import ExtractionService


logger = logging.getLogger(__name__)


class _Barrier:
    def __init__(self):
        self.done = threading.Event()


class ExtractionWorker:
    def __init__(self, repository: MemoryRepository, extractor: ExtractionService,
                 settings: EdithConfig = config, max_retries: int = 3,
                 backoff_seconds: float = 0.25):
        self.repository = repository
        self.extractor = extractor
        self.settings = settings
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds
        self._queue: queue.Queue = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()
        self._queued: set[int] = set()
        self._idle_timers: dict[int, threading.Timer] = {}
        self._stopping = False

    def start(self, catch_up: bool = True) -> None:
        with self._lock:
            if self._thread and self._thread.is_alive():
                return
            self._stopping = False
            self._thread = threading.Thread(target=self._run, name="edith-extraction", daemon=True)
            self._thread.start()
        if catch_up:
            for conversation_id in self.repository.unextracted_conversation_ids():
                self.enqueue(conversation_id)

    def enqueue(self, conversation_id: int) -> None:
        self.start(catch_up=False)
        conversation_id = int(conversation_id)
        with self._lock:
            timer = self._idle_timers.pop(conversation_id, None)
            if timer:
                timer.cancel()
            if conversation_id not in self._queued:
                self._queued.add(conversation_id)
                self._queue.put(conversation_id)

    def schedule_idle(self, conversation_id: int) -> None:
        self.start(catch_up=False)
        with self._lock:
            existing = self._idle_timers.pop(int(conversation_id), None)
            if existing:
                existing.cancel()
            delay = max(0, self.settings.extraction_idle_minutes * 60)
            timer = threading.Timer(delay, self.enqueue, args=(int(conversation_id),))
            timer.daemon = True
            self._idle_timers[int(conversation_id)] = timer
            timer.start()

    def after_response(self, conversation_id: int) -> None:
        pending_users = self.repository.count_unextracted_user_messages(conversation_id)
        if pending_users >= self.settings.extraction_batch_messages:
            self.enqueue(conversation_id)
        else:
            self.schedule_idle(conversation_id)

    def end_conversation(self, conversation_id: int) -> None:
        self.enqueue(conversation_id)

    def flush(self, timeout: float = 10.0) -> bool:
        if self._thread is None or not self._thread.is_alive():
            self.start(catch_up=False)
        deadline = monotonic() + max(0.0, timeout)
        while True:
            for conversation_id in self.repository.unextracted_conversation_ids():
                self.enqueue(conversation_id)
            barrier = _Barrier()
            self._queue.put(barrier)
            if not barrier.done.wait(max(0.0, deadline - monotonic())):
                return False
            if not self.repository.unextracted_conversation_ids():
                return True
            if monotonic() >= deadline:
                return False

    def stop(self, timeout: float = 1.0) -> None:
        self._stopping = True
        self._queue.put(None)
        if self._thread is not None:
            self._thread.join(timeout=max(0.0, timeout))
        with self._lock:
            for timer in self._idle_timers.values():
                timer.cancel()
            self._idle_timers.clear()

    def _run(self) -> None:
        while True:
            task = self._queue.get()
            try:
                if task is None:
                    return
                if isinstance(task, _Barrier):
                    task.done.set()
                    continue
                conversation_id = int(task)
                try:
                    self._process_with_retries(conversation_id)
                finally:
                    with self._lock:
                        self._queued.discard(conversation_id)
                remaining_users = self.repository.count_unextracted_user_messages(conversation_id)
                remaining_messages = self.repository.count_unextracted_messages(conversation_id)
                if remaining_users >= self.settings.extraction_batch_messages:
                    self.enqueue(conversation_id)
                elif remaining_messages:
                    self.schedule_idle(conversation_id)
            finally:
                self._queue.task_done()

    def _process_with_retries(self, conversation_id: int) -> None:
        for attempt in range(self.max_retries + 1):
            try:
                stats = self.extractor.process_conversation(conversation_id)
                logger.info("Extraction completed (conversation_id=%d,processed=%d,saved=%d,pending=%d,skipped=%d)",
                            conversation_id, stats["processed"], stats["saved"],
                            stats["pending"], stats["skipped"])
                return
            except Exception as exc:
                logger.error("Extraction retry scheduled (conversation_id=%d,attempt=%d,error_type=%s)",
                             conversation_id, attempt + 1, type(exc).__name__)
                if attempt < self.max_retries:
                    time.sleep(self.backoff_seconds * (2 ** attempt))
        logger.error("Extraction retries exhausted (conversation_id=%d)", conversation_id)
