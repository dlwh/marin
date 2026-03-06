# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""SQLite-backed log store with string keys.

Stores log entries keyed by an arbitrary string, suitable for task attempt logs,
process logs, autoscaler logs, or any other log stream. Uses WAL mode for
concurrent reader/writer access.

Also provides LogStoreHandler, a logging.Handler that bridges Python's logging
framework into the LogStore so process-level logs (controller, worker) are
queryable through the same FetchLogs RPC as task logs.
"""

from __future__ import annotations

import logging
import re
import sqlite3
import tempfile
from dataclasses import dataclass
from pathlib import Path
from threading import Lock, RLock

from iris.cluster.types import JobName
from iris.logging import str_to_log_level
from iris.rpc import logging_pb2

_SCHEMA = """\
CREATE TABLE IF NOT EXISTS logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    key TEXT NOT NULL,
    source TEXT NOT NULL,
    data TEXT NOT NULL,
    epoch_ms INTEGER NOT NULL,
    level INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_key ON logs(key, id);
"""

_MAX_RECORDS = 100_000_000
_EVICT_CHECK_INTERVAL = 100_000  # check eviction every N appends

PROCESS_LOG_KEY = "/process"

_FLUSH_THRESHOLD = 100


def task_log_key(task_id: JobName, attempt_id: int) -> str:
    """Build a hierarchical key for task attempt logs."""
    return f"{task_id.to_wire()}:{attempt_id}"


@dataclass
class LogReadResult:
    entries: list[logging_pb2.LogEntry]
    lines_read: int  # total line count at end of read (cursor for next poll)


class LogStore:
    """SQLite-backed log store keyed by arbitrary strings.

    Thread-safe: writers and RPC readers may run concurrently. WAL mode
    allows readers to proceed without blocking the writer.
    """

    def __init__(self, log_dir: Path | None = None, *, max_records: int = _MAX_RECORDS):
        self._temp_dir: tempfile.TemporaryDirectory[str] | None = None
        if log_dir is not None:
            log_dir.mkdir(parents=True, exist_ok=True)
            db_path = str(log_dir / "logs.db")
        else:
            self._temp_dir = tempfile.TemporaryDirectory(prefix="iris_controller_logs_")
            db_path = str(Path(self._temp_dir.name) / "logs.db")

        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.execute("PRAGMA synchronous = NORMAL")
        self._conn.execute("PRAGMA busy_timeout = 5000")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

        self._lock = RLock()
        self._max_records = max_records
        self._append_count = 0

    def append(self, key: str, entries: list) -> None:
        if not entries:
            return
        rows = [(key, e.source, e.data, e.timestamp.epoch_ms, e.level) for e in entries]
        with self._lock:
            self._conn.executemany(
                "INSERT INTO logs (key, source, data, epoch_ms, level) VALUES (?, ?, ?, ?, ?)",
                rows,
            )
            self._conn.commit()
            self._append_count += 1
            if self._append_count % _EVICT_CHECK_INTERVAL == 0:
                self._evict_if_needed()

    def get_logs(
        self,
        key: str,
        *,
        since_ms: int = 0,
        skip_lines: int = 0,
        regex_filter: re.Pattern[str] | None = None,
        max_lines: int = 0,
        tail: bool = False,
        min_level: str = "",
    ) -> LogReadResult:
        min_level_enum = str_to_log_level(min_level) if min_level else 0

        with self._lock:
            total = self._count_lines(key)
            if total == 0:
                return LogReadResult(entries=[], lines_read=0)

            has_filter = regex_filter is not None or since_ms > 0 or min_level_enum > 0

            if tail and max_lines > 0 and not has_filter:
                effective_skip = max(skip_lines, total - max_lines)
                rows = self._conn.execute(
                    "SELECT source, data, epoch_ms, level FROM logs " "WHERE key = ? " "ORDER BY id LIMIT -1 OFFSET ?",
                    (key, effective_skip),
                ).fetchall()
            elif tail and max_lines > 0 and has_filter:
                params: list = [key]
                where_extra = ""
                if since_ms > 0:
                    where_extra += " AND epoch_ms > ?"
                    params.append(since_ms)
                rows = self._conn.execute(
                    "SELECT source, data, epoch_ms, level FROM logs " f"WHERE key = ?{where_extra} " "ORDER BY id",
                    params,
                ).fetchall()
                if regex_filter:
                    rows = [r for r in rows if regex_filter.search(r[1])]
                if min_level_enum > 0:
                    # level=0 (UNKNOWN) is always included so untagged output is never hidden.
                    rows = [r for r in rows if r[3] == 0 or r[3] >= min_level_enum]
                rows = rows[-max_lines:]
            else:
                # Forward mode
                params = [key]
                where_extra = ""
                if since_ms > 0:
                    where_extra += " AND epoch_ms > ?"
                    params.append(since_ms)

                query = (
                    "SELECT source, data, epoch_ms, level FROM logs "
                    f"WHERE key = ?{where_extra} "
                    "ORDER BY id LIMIT -1 OFFSET ?"
                )
                params.append(skip_lines)
                rows = self._conn.execute(query, params).fetchall()

                if regex_filter:
                    rows = [r for r in rows if regex_filter.search(r[1])]
                if min_level_enum > 0:
                    # level=0 (UNKNOWN) is always included so untagged output is never hidden.
                    rows = [r for r in rows if r[3] == 0 or r[3] >= min_level_enum]

                if max_lines > 0:
                    rows = rows[:max_lines]

        entries = [self._row_to_entry(r) for r in rows]
        return LogReadResult(entries=entries, lines_read=total)

    def has_logs(self, key: str) -> bool:
        with self._lock:
            row = self._conn.execute(
                "SELECT 1 FROM logs WHERE key = ? LIMIT 1",
                (key,),
            ).fetchone()
            return row is not None

    def clear(self, key: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM logs WHERE key = ?", (key,))
            self._conn.commit()

    def close(self) -> None:
        """Close the database connection and clean up temp dir if applicable."""
        self._conn.close()
        if self._temp_dir is not None:
            self._temp_dir.cleanup()

    @staticmethod
    def _row_to_entry(row: tuple) -> logging_pb2.LogEntry:
        # row: (source, data, epoch_ms, level)
        entry = logging_pb2.LogEntry(source=row[0], data=row[1], level=row[3])
        entry.timestamp.epoch_ms = row[2]
        return entry

    def _count_lines(self, key: str) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) FROM logs WHERE key = ?",
            (key,),
        ).fetchone()
        return row[0]

    def _evict_if_needed(self) -> None:
        """Delete oldest rows when the total count exceeds the cap. Must hold self._lock."""
        count = self._conn.execute("SELECT COUNT(*) FROM logs").fetchone()[0]
        if count > self._max_records:
            excess = count - self._max_records
            self._conn.execute(
                "DELETE FROM logs WHERE id IN (SELECT id FROM logs ORDER BY id LIMIT ?)",
                (excess,),
            )
            self._conn.commit()


class LogStoreHandler(logging.Handler):
    """Logging handler that writes formatted records into a LogStore.

    Buffers entries and flushes when the buffer reaches _FLUSH_THRESHOLD or
    when flush() is called explicitly. This avoids a SQLite commit per record.
    """

    def __init__(self, log_store: LogStore, key: str = PROCESS_LOG_KEY):
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
