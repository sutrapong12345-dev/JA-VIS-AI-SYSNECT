"""Tamper-evident append-only audit storage for J.A.R.V.I.S."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from typing import Any, Dict, List, Optional


class AuditStore:
    """Append-only SQLite audit log with a cryptographic hash chain.

    SQLite is the zero-setup local backend. The table shape is mirrored by
    ``backend/migrations/postgresql_audit.sql`` for organizational deployment.
    """

    def __init__(self, path: str):
        self.path = path
        self._lock = threading.Lock()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._initialize()

    @contextmanager
    def _connect(self):
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS audit_events (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT NOT NULL UNIQUE,
                    occurred_at REAL NOT NULL,
                    actor_id TEXT NOT NULL,
                    actor_role TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    action TEXT NOT NULL,
                    outcome TEXT NOT NULL,
                    target TEXT NOT NULL,
                    details_json TEXT NOT NULL,
                    previous_hash TEXT NOT NULL,
                    event_hash TEXT NOT NULL UNIQUE
                );
                CREATE TRIGGER IF NOT EXISTS audit_events_no_update
                BEFORE UPDATE ON audit_events
                BEGIN SELECT RAISE(ABORT, 'audit events are immutable'); END;
                CREATE TRIGGER IF NOT EXISTS audit_events_no_delete
                BEFORE DELETE ON audit_events
                BEGIN SELECT RAISE(ABORT, 'audit events are immutable'); END;
                """
            )

    @staticmethod
    def _canonical(payload: Dict[str, Any]) -> str:
        return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    def append(
        self,
        action: str,
        outcome: str,
        *,
        actor_id: str = "anonymous",
        actor_role: str = "staff",
        session_id: str = "",
        target: str = "",
        details: Optional[Dict[str, Any]] = None,
    ) -> str:
        event_id = str(uuid.uuid4())
        occurred_at = time.time()
        safe_details = details or {}
        with self._lock, self._connect() as connection:
            previous = connection.execute(
                "SELECT event_hash FROM audit_events ORDER BY sequence DESC LIMIT 1"
            ).fetchone()
            previous_hash = previous["event_hash"] if previous else "GENESIS"
            body = {
                "event_id": event_id,
                "occurred_at": occurred_at,
                "actor_id": actor_id,
                "actor_role": actor_role,
                "session_id": session_id,
                "action": action,
                "outcome": outcome,
                "target": target,
                "details": safe_details,
                "previous_hash": previous_hash,
            }
            event_hash = hashlib.sha256(self._canonical(body).encode("utf-8")).hexdigest()
            connection.execute(
                """
                INSERT INTO audit_events (
                    event_id, occurred_at, actor_id, actor_role, session_id,
                    action, outcome, target, details_json, previous_hash, event_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event_id, occurred_at, actor_id, actor_role, session_id,
                    action, outcome, target, self._canonical(safe_details),
                    previous_hash, event_hash,
                ),
            )
        return event_id

    def recent(self, limit: int = 100) -> List[Dict[str, Any]]:
        capped = max(1, min(limit, 500))
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM audit_events ORDER BY sequence DESC LIMIT ?", (capped,)
            ).fetchall()
        return [
            {
                **dict(row),
                "details": json.loads(row["details_json"]),
            }
            for row in rows
        ]

    def verify_chain(self) -> Dict[str, Any]:
        with self._connect() as connection:
            rows = connection.execute("SELECT * FROM audit_events ORDER BY sequence").fetchall()
        expected_previous = "GENESIS"
        for row in rows:
            details = json.loads(row["details_json"])
            body = {
                "event_id": row["event_id"],
                "occurred_at": row["occurred_at"],
                "actor_id": row["actor_id"],
                "actor_role": row["actor_role"],
                "session_id": row["session_id"],
                "action": row["action"],
                "outcome": row["outcome"],
                "target": row["target"],
                "details": details,
                "previous_hash": row["previous_hash"],
            }
            calculated = hashlib.sha256(self._canonical(body).encode("utf-8")).hexdigest()
            if row["previous_hash"] != expected_previous or row["event_hash"] != calculated:
                return {"valid": False, "events": len(rows), "broken_at": row["sequence"]}
            expected_previous = row["event_hash"]
        return {"valid": True, "events": len(rows), "head": expected_previous}
