"""Role-aware local RAG index with source citations."""

from __future__ import annotations

import hashlib
import os
import re
import sqlite3
import threading
import time
from contextlib import contextmanager
from typing import Any, Dict, Iterable, List, Set


ROLE_LEVEL = {"staff": 10, "manager": 20, "admin": 30}
ROLE_PREFIX = re.compile(r"^(staff|manager|admin)__", re.IGNORECASE)


def _terms(text: str) -> Set[str]:
    lowered = (text or "").lower()
    terms = set(re.findall(r"[a-z0-9_./-]{2,}|[\u0e00-\u0e7f]{2,}", lowered))
    thai_runs = re.findall(r"[\u0e00-\u0e7f]{3,}", lowered)
    for run in thai_runs:
        terms.update(run[index:index + 3] for index in range(max(0, len(run) - 2)))
    return terms


def _chunks(text: str, size: int = 1200, overlap: int = 200) -> Iterable[str]:
    normalized = text.replace("\r\n", "\n").strip()
    start = 0
    while start < len(normalized):
        end = min(len(normalized), start + size)
        if end < len(normalized):
            boundary = normalized.rfind("\n", start + size // 2, end)
            if boundary > start:
                end = boundary
        chunk = normalized[start:end].strip()
        if chunk:
            yield chunk
        if end >= len(normalized):
            break
        start = max(start + 1, end - overlap)


class RagStore:
    def __init__(self, path: str):
        self.path = path
        self._lock = threading.Lock()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS rag_documents (
                    source TEXT PRIMARY KEY,
                    minimum_role TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    indexed_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS rag_chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source TEXT NOT NULL REFERENCES rag_documents(source) ON DELETE CASCADE,
                    chunk_number INTEGER NOT NULL,
                    minimum_role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    terms TEXT NOT NULL,
                    UNIQUE(source, chunk_number)
                );
                """
            )

    @contextmanager
    def _connect(self):
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    @staticmethod
    def role_for_filename(filename: str) -> str:
        match = ROLE_PREFIX.match(filename)
        return match.group(1).lower() if match else "staff"

    def index_directory(self, directory: str) -> Dict[str, int]:
        indexed_sources: Set[str] = set()
        documents = 0
        chunks = 0
        with self._lock, self._connect() as connection:
            for filename in sorted(os.listdir(directory)) if os.path.isdir(directory) else []:
                if filename.startswith(("_", ".")) or not filename.lower().endswith((".md", ".txt")):
                    continue
                path = os.path.join(directory, filename)
                with open(path, "r", encoding="utf-8", errors="ignore") as handle:
                    content = handle.read().strip()
                if not content:
                    continue
                indexed_sources.add(filename)
                minimum_role = self.role_for_filename(filename)
                digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
                current = connection.execute(
                    "SELECT content_hash, minimum_role FROM rag_documents WHERE source=?", (filename,)
                ).fetchone()
                if not current or current["content_hash"] != digest or current["minimum_role"] != minimum_role:
                    connection.execute("DELETE FROM rag_documents WHERE source=?", (filename,))
                    connection.execute(
                        "INSERT INTO rag_documents VALUES (?, ?, ?, ?)",
                        (filename, minimum_role, digest, time.time()),
                    )
                    for number, chunk in enumerate(_chunks(content), start=1):
                        connection.execute(
                            """INSERT INTO rag_chunks
                            (source, chunk_number, minimum_role, content, terms)
                            VALUES (?, ?, ?, ?, ?)""",
                            (filename, number, minimum_role, chunk, " ".join(sorted(_terms(chunk)))),
                        )
                documents += 1
            existing = {
                row["source"] for row in connection.execute("SELECT source FROM rag_documents")
            }
            for removed in existing - indexed_sources:
                connection.execute("DELETE FROM rag_documents WHERE source=?", (removed,))
            chunks = connection.execute("SELECT COUNT(*) AS count FROM rag_chunks").fetchone()["count"]
        return {"documents": documents, "chunks": chunks}

    def search(self, query: str, role: str = "staff", top_k: int = 5) -> List[Dict[str, Any]]:
        query_terms = _terms(query)
        if not query_terms:
            return []
        level = ROLE_LEVEL.get(role, ROLE_LEVEL["staff"])
        allowed_roles = [name for name, value in ROLE_LEVEL.items() if value <= level]
        placeholders = ",".join("?" for _ in allowed_roles)
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT source, chunk_number, minimum_role, content, terms FROM rag_chunks "
                f"WHERE minimum_role IN ({placeholders})",
                allowed_roles,
            ).fetchall()
        scored = []
        for row in rows:
            content_terms = set(row["terms"].split())
            overlap = query_terms & content_terms
            if not overlap:
                continue
            score = len(overlap) / max(1, len(query_terms))
            scored.append((score, row))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            {
                "source": row["source"],
                "chunk": row["chunk_number"],
                "minimum_role": row["minimum_role"],
                "score": round(score, 4),
                "content": row["content"],
                "citation": f"[KB:{row['source']}#chunk-{row['chunk_number']}]",
            }
            for score, row in scored[:max(1, min(top_k, 10))]
        ]

    def context(self, query: str, role: str = "staff", top_k: int = 5) -> str:
        results = self.search(query, role, top_k)
        return "\n\n".join(f"{item['citation']}\n{item['content']}" for item in results)

    def info(self) -> Dict[str, int]:
        with self._connect() as connection:
            documents = connection.execute("SELECT COUNT(*) AS count FROM rag_documents").fetchone()["count"]
            chunks = connection.execute("SELECT COUNT(*) AS count FROM rag_chunks").fetchone()["count"]
        return {"documents": documents, "chunks": chunks}
