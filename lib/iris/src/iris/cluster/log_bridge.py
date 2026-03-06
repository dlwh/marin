# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bridge between Python logging and the SQLite LogStore.

LogStoreHandler is a logging.Handler that buffers log records and periodically
flushes them to a LogStore under a fixed key. This lets process-level logs
(controller, worker) be queried through the same FetchLogs RPC as task logs.
"""

from __future__ import annotations

import logging
from threading import Lock

from iris.cluster.controller.logs import LogStore
from iris.logging import str_to_log_level
from iris.rpc import logging_pb2

_FLUSH_THRESHOLD = 100


class LogStoreHandler(logging.Handler):
    """Logging handler that writes formatted records into a LogStore.

    Buffers entries and flushes when the buffer reaches _FLUSH_THRESHOLD or
    when flush() is called explicitly. This avoids a SQLite commit per record.
    """

    def __init__(self, log_store: LogStore, key: str):
        super().__init__()
        self._log_store = log_store
        self._key = key
        self._buffer: list[logging_pb2.LogEntry] = []
        self._lock = Lock()

    def emit(self, record: logging.LogRecord) -> None:
        try:
            entry = logging_pb2.LogEntry(
                source="process",
                data=self.format(record),
                level=str_to_log_level(record.levelname),
            )
            entry.timestamp.epoch_ms = int(record.created * 1000)

            with self._lock:
                self._buffer.append(entry)
                if len(self._buffer) >= _FLUSH_THRESHOLD:
                    self._flush_locked()
        except Exception:
            self.handleError(record)

    def flush(self) -> None:
        with self._lock:
            self._flush_locked()

    def _flush_locked(self) -> None:
        """Flush buffered entries to the LogStore. Caller must hold self._lock."""
        if not self._buffer:
            return
        entries = self._buffer
        self._buffer = []
        self._log_store.append(self._key, entries)

    def close(self) -> None:
        self.flush()
        super().close()
