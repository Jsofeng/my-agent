"""Thread-local SQLite connections configured for Edith's workload."""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from typing import Union


class ConnectionManager:
    def __init__(self, path: Union[Path, str], busy_timeout_ms: int = 5000):
        self.path = Path(path)
        self.busy_timeout_ms = busy_timeout_ms
        self._local = threading.local()
        self._all_lock = threading.Lock()
        self._connections: list[sqlite3.Connection] = []
        self._wal_initialized = False

    def get(self) -> sqlite3.Connection:
        connection = getattr(self._local, "connection", None)
        if connection is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            connection = sqlite3.connect(
                str(self.path), timeout=self.busy_timeout_ms / 1000,
                check_same_thread=False, isolation_level="DEFERRED",
            )
            connection.row_factory = sqlite3.Row
            connection.execute(f"PRAGMA busy_timeout = {int(self.busy_timeout_ms)}")
            with self._all_lock:
                if not self._wal_initialized:
                    connection.execute("PRAGMA journal_mode = WAL")
                    self._wal_initialized = True
            connection.execute("PRAGMA secure_delete = ON")
            self._local.connection = connection
            with self._all_lock:
                self._connections.append(connection)
        return connection

    def close_thread(self) -> None:
        connection = getattr(self._local, "connection", None)
        if connection is not None:
            connection.close()
            self._local.connection = None

    def close_all(self) -> None:
        # Call only after application workers have stopped.
        with self._all_lock:
            for connection in self._connections:
                try:
                    connection.close()
                except sqlite3.Error:
                    pass
            self._connections.clear()
        self._local.connection = None
