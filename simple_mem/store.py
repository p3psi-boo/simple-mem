"""A single-file memory store: raw evidence and distilled memories together.

Everything a machine has seen arrives as *evidence turns* -- verbatim user and
assistant messages -- and everything it has concluded lives beside them as
*memories*. Both are searchable through SQLite FTS5, and memories can also be
ranked by vector similarity, fused with Reciprocal Rank Fusion so that a hit
which is strong in both channels outranks one that is strong in only one.

Two design choices are load-bearing here:

* Evidence is keyed by a hash of its source identity, so shipping the same
  conversation twice is a no-op instead of a duplicate.
* FTS5's `unicode61` tokenizer treats a run of CJK text as a single token, so
  a Chinese substring query matches nothing on its own. Every lexical search
  therefore also runs a `LIKE` substring fallback.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import uuid
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np

from .project import resolve_project_id

__all__ = ["SimpleMemStore"]

DEFAULT_DB_PATH = "~/.simple-mem/simple-mem.db"

_RRF_K = 60

#: Below this cosine similarity a vector hit is noise, not a result.
_MIN_VECTOR_SIMILARITY = 0.25

_FTS_TOKEN = re.compile(r"[0-9A-Za-z_\u3400-\u4dbf\u4e00-\u9fff]+")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS evidence_turns (
    evidence_id     TEXT PRIMARY KEY,
    machine_id      TEXT NOT NULL DEFAULT '',
    source_id       TEXT NOT NULL,
    project_id      TEXT NOT NULL,
    conversation_id TEXT NOT NULL,
    message_id      TEXT NOT NULL,
    role            TEXT NOT NULL,
    content         TEXT NOT NULL,
    created_at      TEXT NOT NULL,
    processed       INTEGER NOT NULL DEFAULT 0
);

CREATE VIRTUAL TABLE IF NOT EXISTS evidence_fts USING fts5(
    evidence_id UNINDEXED,
    project_id UNINDEXED,
    source_id UNINDEXED,
    content,
    tokenize='unicode61'
);

CREATE TABLE IF NOT EXISTS memories (
    memory_id     TEXT PRIMARY KEY,
    project_id    TEXT NOT NULL,
    kind          TEXT NOT NULL DEFAULT 'fact',
    statement     TEXT NOT NULL,
    details       TEXT NOT NULL DEFAULT '',
    search_keys   TEXT NOT NULL DEFAULT '[]',
    boundary_envs TEXT NOT NULL DEFAULT '',
    source_id     TEXT NOT NULL DEFAULT 'manual',
    evidence_ids  TEXT NOT NULL DEFAULT '[]',
    confidence    REAL NOT NULL DEFAULT 0.9,
    status        TEXT NOT NULL DEFAULT 'active',
    superseded_by TEXT,
    embedding     BLOB,
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL
);

CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    memory_id UNINDEXED,
    project_id UNINDEXED,
    kind UNINDEXED,
    statement,
    details,
    search_keys,
    tokenize='unicode61'
);

CREATE TABLE IF NOT EXISTS sync_state (
    key        TEXT PRIMARY KEY,
    value      TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""

class SimpleMemStore:
    """One SQLite file holding both raw evidence and distilled memories."""

    def __init__(self, db_path: Union[str, Path] = DEFAULT_DB_PATH) -> None:
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    # -- plumbing ----------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.db_path), timeout=5.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        connection.execute("PRAGMA busy_timeout=5000")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def _initialize(self) -> None:
        connection = self._connect()
        try:
            connection.executescript(_SCHEMA)
            connection.commit()
        finally:
            connection.close()

    # -- ingestion ---------------------------------------------------------

    def ingest_turns(
        self,
        turns: Sequence[Mapping[str, Any]],
        *,
        machine_id: str = "",
        default_source_id: str = "unknown",
        default_project_id: Optional[str] = None,
    ) -> Dict[str, int]:
        """Store verbatim turns, ignoring ones already seen.

        Evidence identity is derived, not assigned, so re-shipping a
        conversation costs nothing and never duplicates a row.
        """
        accepted = 0
        inserted = 0
        with self._connection() as connection:
            for turn in turns:
                conversation_id = _clean(
                    turn.get("conversation_id") or turn.get("session_id")
                )
                message_id = _clean(
                    turn.get("message_id") or turn.get("turn_id") or turn.get("id")
                )
                if not conversation_id or not message_id:
                    continue

                source_id = _clean(turn.get("source_id")) or default_source_id
                role = _clean(turn.get("role")) or "user"
                content = _turn_content(turn)
                raw_cwd = _clean(turn.get("cwd") or turn.get("workspace"))
                project_source = (
                    turn.get("project_id")
                    or raw_cwd
                    or default_project_id
                )
                project_id = self._resolve_project_with_conn(connection, project_source)
                if raw_cwd and project_id and not project_id.startswith("dir://") and project_id != "global":
                    self._record_cwd_project(connection, raw_cwd, project_id)
                turn_machine = _clean(turn.get("machine_id")) or machine_id
                created_at = (
                    _clean(turn.get("created_at") or turn.get("timestamp"))
                    or _utcnow()
                )

                evidence_id = hashlib.sha256(
                    f"{source_id}\0{conversation_id}\0{message_id}".encode("utf-8")
                ).hexdigest()

                accepted += 1
                cursor = connection.execute(
                    """
                    INSERT OR IGNORE INTO evidence_turns (
                        evidence_id, machine_id, source_id, project_id,
                        conversation_id, message_id, role, content, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        evidence_id,
                        turn_machine,
                        source_id,
                        project_id,
                        conversation_id,
                        message_id,
                        role,
                        content,
                        created_at,
                    ),
                )
                if cursor.rowcount != 1:
                    continue
                inserted += 1
                connection.execute(
                    """
                    INSERT INTO evidence_fts (evidence_id, project_id, source_id, content)
                    VALUES (?, ?, ?, ?)
                    """,
                    (evidence_id, project_id, source_id, content),
                )

        return {
            "accepted": accepted,
            "inserted": inserted,
            "duplicates": accepted - inserted,
        }

    def fetch_unprocessed_turns(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT * FROM evidence_turns
                WHERE processed = 0
                ORDER BY created_at ASC, evidence_id ASC
                LIMIT ?
                """,
                (max(0, int(limit)),),
            ).fetchall()
        return [dict(row) for row in rows]

    def mark_turns_processed(self, evidence_ids: Sequence[str]) -> int:
        ids = [str(value) for value in evidence_ids if value]
        if not ids:
            return 0
        total = 0
        with self._connection() as connection:
            for start in range(0, len(ids), 500):
                chunk = ids[start : start + 500]
                placeholders = ",".join("?" for _ in chunk)
                cursor = connection.execute(
                    f"UPDATE evidence_turns SET processed = 1 "
                    f"WHERE evidence_id IN ({placeholders})",
                    chunk,
                )
                total += cursor.rowcount
        return total

    # -- memories ----------------------------------------------------------

    def remember(
        self,
        statement: str,
        *,
        project_id: str = "global",
        kind: str = "fact",
        details: str = "",
        search_keys: Sequence[str] = (),
        boundary_envs: str = "",
        source_id: str = "manual",
        evidence_ids: Sequence[str] = (),
        confidence: float = 0.9,
        embedding: Optional[Sequence[float]] = None,
        memory_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        with self._connection() as connection:
            return self._remember(
                connection,
                statement=statement,
                project_id=project_id,
                kind=kind,
                details=details,
                search_keys=search_keys,
                boundary_envs=boundary_envs,
                source_id=source_id,
                evidence_ids=evidence_ids,
                confidence=confidence,
                embedding=embedding,
                memory_id=memory_id,
            )

    def _remember(
        self,
        connection: sqlite3.Connection,
        *,
        statement: str,
        project_id: str,
        kind: str,
        details: str,
        search_keys: Sequence[str],
        boundary_envs: str,
        source_id: str,
        evidence_ids: Sequence[str],
        confidence: float,
        embedding: Optional[Sequence[float]],
        memory_id: Optional[str],
    ) -> Dict[str, Any]:
        if project_id and project_id != "global":
            project_id = self._resolve_project_with_conn(connection, project_id)
        identifier = memory_id or f"mem_{uuid.uuid4().hex}"
        now = _utcnow()
        keys_json = json.dumps([str(key) for key in search_keys], ensure_ascii=False)
        evidence_json = json.dumps(
            [str(value) for value in evidence_ids], ensure_ascii=False
        )
        blob = _embedding_bytes(embedding)

        connection.execute(
            """
            INSERT INTO memories (
                memory_id, project_id, kind, statement, details, search_keys,
                boundary_envs, source_id, evidence_ids, confidence, status,
                superseded_by, embedding, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'active', NULL, ?, ?, ?)
            ON CONFLICT(memory_id) DO UPDATE SET
                project_id = excluded.project_id,
                kind = excluded.kind,
                statement = excluded.statement,
                details = excluded.details,
                search_keys = excluded.search_keys,
                boundary_envs = excluded.boundary_envs,
                source_id = excluded.source_id,
                evidence_ids = excluded.evidence_ids,
                confidence = excluded.confidence,
                embedding = excluded.embedding,
                updated_at = excluded.updated_at
            """,
            (
                identifier,
                project_id,
                kind,
                statement,
                details,
                keys_json,
                boundary_envs,
                source_id,
                evidence_json,
                float(confidence),
                blob,
                now,
                now,
            ),
        )
        self._refresh_memory_fts(connection, identifier)
        return self._memory_row(connection, identifier)

    def update_memory(
        self,
        memory_id: str,
        *,
        statement: Optional[str] = None,
        details: Optional[str] = None,
        search_keys: Optional[Sequence[str]] = None,
        boundary_envs: Optional[str] = None,
        embedding: Optional[Sequence[float]] = None,
    ) -> Optional[Dict[str, Any]]:
        assignments: List[str] = []
        params: List[Any] = []
        if statement is not None:
            assignments.append("statement = ?")
            params.append(statement)
        if details is not None:
            assignments.append("details = ?")
            params.append(details)
        if search_keys is not None:
            assignments.append("search_keys = ?")
            params.append(json.dumps([str(key) for key in search_keys], ensure_ascii=False))
        if boundary_envs is not None:
            assignments.append("boundary_envs = ?")
            params.append(boundary_envs)
        if embedding is not None:
            assignments.append("embedding = ?")
            params.append(_embedding_bytes(embedding))

        with self._connection() as connection:
            existing = connection.execute(
                "SELECT memory_id FROM memories WHERE memory_id = ?", (memory_id,)
            ).fetchone()
            if existing is None:
                return None
            if assignments:
                assignments.append("updated_at = ?")
                params.append(_utcnow())
                params.append(memory_id)
                connection.execute(
                    f"UPDATE memories SET {', '.join(assignments)} WHERE memory_id = ?",
                    params,
                )
                self._refresh_memory_fts(connection, memory_id)
            return self._memory_row(connection, memory_id)

    def supersede_memory(
        self, old_memory_id: str, new_memory: Mapping[str, Any]
    ) -> Dict[str, Any]:
        """Replace a memory with a corrected one, in a single transaction.

        The old row is kept and marked `superseded` rather than deleted, so
        the correction remains auditable; only live search drops it.
        """
        with self._connection() as connection:
            stored = self._remember(
                connection,
                statement=new_memory.get("statement", ""),
                project_id=new_memory.get("project_id", "global"),
                kind=new_memory.get("kind", "fact"),
                details=new_memory.get("details", ""),
                search_keys=new_memory.get("search_keys", ()) or (),
                boundary_envs=new_memory.get("boundary_envs", ""),
                source_id=new_memory.get("source_id", "manual"),
                evidence_ids=new_memory.get("evidence_ids", ()) or (),
                confidence=float(new_memory.get("confidence", 0.9)),
                embedding=new_memory.get("embedding"),
                memory_id=new_memory.get("memory_id"),
            )
            connection.execute(
                """
                UPDATE memories
                SET status = 'superseded', superseded_by = ?, updated_at = ?
                WHERE memory_id = ?
                """,
                (stored["memory_id"], _utcnow(), old_memory_id),
            )
            connection.execute(
                "DELETE FROM memories_fts WHERE memory_id = ?", (old_memory_id,)
            )
        return stored

    def forget(self, memory_id: str) -> bool:
        with self._connection() as connection:
            cursor = connection.execute(
                "DELETE FROM memories WHERE memory_id = ?", (memory_id,)
            )
            connection.execute(
                "DELETE FROM memories_fts WHERE memory_id = ?", (memory_id,)
            )
            return cursor.rowcount > 0

    def get_memory(self, memory_id: str) -> Optional[Dict[str, Any]]:
        with self._connection() as connection:
            return self._memory_row(connection, memory_id)

    def list_memories(
        self,
        *,
        project_id: Optional[str] = None,
        status: str = "active",
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        clauses = ["status = ?"]
        params: List[Any] = [status]
        with self._connection() as connection:
            if project_id is not None:
                if project_id != "global":
                    project_id = self._resolve_project_with_conn(connection, project_id)
                clauses.append("project_id = ?")
                params.append(project_id)
            params.append(max(0, int(limit)))
            rows = connection.execute(
                f"SELECT * FROM memories WHERE {' AND '.join(clauses)} "
                f"ORDER BY updated_at DESC, memory_id ASC LIMIT ?",
                params,
            ).fetchall()
        return [_row_to_memory(row) for row in rows]

    # -- briefing and threads ----------------------------------------------

    def briefing(
        self, project_id: Optional[str] = None, limit: int = 20
    ) -> Dict[str, Any]:
        """A zero-query summary of what memory holds for a project.

        A workspace sees its own active memories plus the global ones, its
        own first; `global` sees only global memories. Kinds are bucketed so
        an agent can read preferences, decisions, experiences and facts
        without searching for any of them.
        """
        cap = max(1, int(limit))
        with self._connection() as connection:
            resolved_project_id = self._resolve_project_with_conn(connection, project_id)
            if resolved_project_id != "global":
                rows = connection.execute(
                    """
                    SELECT * FROM memories
                    WHERE status = 'active' AND project_id IN (?, 'global')
                    ORDER BY CASE WHEN project_id = ? THEN 0 ELSE 1 END,
                             updated_at DESC, memory_id ASC
                    LIMIT ?
                    """,
                    (resolved_project_id, resolved_project_id, cap),
                ).fetchall()
            else:
                rows = connection.execute(
                    """
                    SELECT * FROM memories
                    WHERE status = 'active' AND project_id = 'global'
                    ORDER BY updated_at DESC, memory_id ASC
                    LIMIT ?
                    """,
                    (cap,),
                ).fetchall()
            memories = [_row_to_memory(row) for row in rows]
            recent_threads = self._recent_threads(connection, resolved_project_id)

        buckets: Dict[str, List[Dict[str, Any]]] = {
            "preferences": [],
            "decisions": [],
            "experiences": [],
            "facts": [],
        }
        by_kind = {
            "preference": "preferences",
            "decision": "decisions",
            "experience": "experiences",
            "fact": "facts",
        }
        for memory in memories:
            buckets[by_kind.get(str(memory.get("kind") or ""), "facts")].append(memory)

        return {
            "project_id": resolved_project_id,
            "preferences": buckets["preferences"],
            "decisions": buckets["decisions"],
            "experiences": buckets["experiences"],
            "facts": buckets["facts"],
            "memories": memories,
            "recent_threads": recent_threads,
        }

    def _recent_threads(
        self, connection: sqlite3.Connection, project_id: str
    ) -> List[Dict[str, Any]]:
        """The five most recently active conversations, newest first."""
        scope_sql = ""
        params: List[Any] = []
        if project_id != "global":
            scope_sql = "WHERE project_id = ?"
            params.append(project_id)
        rows = connection.execute(
            f"""
            SELECT conversation_id, project_id, source_id, machine_id,
                   COUNT(*) AS turn_count, MAX(created_at) AS last_active_at
            FROM evidence_turns
            {scope_sql}
            GROUP BY conversation_id
            ORDER BY last_active_at DESC
            LIMIT 5
            """,
            params,
        ).fetchall()

        threads: List[Dict[str, Any]] = []
        for row in rows:
            last = connection.execute(
                """
                SELECT content FROM evidence_turns
                WHERE conversation_id = ?
                ORDER BY created_at DESC, rowid DESC
                LIMIT 1
                """,
                (row["conversation_id"],),
            ).fetchone()
            preview = last["content"] if last is not None else ""
            threads.append(
                {
                    "conversation_id": row["conversation_id"],
                    "project_id": row["project_id"],
                    "source_id": row["source_id"],
                    "machine_id": row["machine_id"],
                    "turn_count": row["turn_count"],
                    "last_active_at": row["last_active_at"],
                    "preview": preview[:160],
                }
            )
        return threads

    def fetch_thread(
        self,
        conversation_id: str,
        limit: int = 50,
        around_message_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Return a conversation's turns in order, optionally windowed.

        With `around_message_id` the window of `limit` turns is centered on
        that turn; without a match (or without a message id) the tail of the
        conversation is returned, which is where the current question lives.
        """
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT evidence_id, project_id, source_id, machine_id,
                       conversation_id, message_id, role, content, created_at
                FROM evidence_turns
                WHERE conversation_id = ?
                ORDER BY created_at ASC, rowid ASC
                """,
                (conversation_id,),
            ).fetchall()

        turns = [dict(row) for row in rows]
        total_turns = len(turns)
        selected = turns

        if limit and int(limit) > 0 and total_turns > int(limit):
            window = int(limit)
            index: Optional[int] = None
            if around_message_id:
                for position, turn in enumerate(turns):
                    if around_message_id in (turn["message_id"], turn["evidence_id"]):
                        index = position
                        break
            if index is None:
                selected = turns[-window:]
            else:
                start = max(0, index - window // 2)
                start = min(start, total_turns - window)
                selected = turns[start : start + window]

        return {
            "conversation_id": conversation_id,
            "project_id": turns[0]["project_id"] if turns else "global",
            "source_id": turns[0]["source_id"] if turns else "",
            "machine_id": turns[0]["machine_id"] if turns else "",
            "total_turns": total_turns,
            "turns": selected,
        }

    # -- search ------------------------------------------------------------

    def search(
        self,
        query: str,
        *,
        project_id: Optional[str] = None,
        scope: str = "workspace",
        limit: int = 10,
        query_embedding: Optional[Sequence[float]] = None,
        include_raw: bool = True,
    ) -> Dict[str, Any]:
        cap = max(1, int(limit) * 5)

        with self._connection() as connection:
            resolved_project_id = project_id
            if resolved_project_id is not None and resolved_project_id != "global":
                resolved_project_id = self._resolve_project_with_conn(
                    connection, resolved_project_id
                )

            workspace = bool(
                scope == "workspace"
                and resolved_project_id
                and resolved_project_id != "global"
            )
            lexical_ids = (
                _lexical_memory_ids(connection, query, workspace, resolved_project_id, cap)
                if query and query.strip()
                else []
            )
            vector_ids = (
                _vector_memory_ids(
                    connection, query_embedding, workspace, resolved_project_id, cap
                )
                if query_embedding is not None
                else []
            )
            fused = _fuse_rankings([lexical_ids, vector_ids], limit)

            memories = []
            for memory_id, score in fused:
                row = self._memory_row(connection, memory_id)
                if row is None:
                    continue
                row["score"] = round(score, 4)
                memories.append(row)

            raw_turns: List[Dict[str, Any]] = []
            if include_raw and query and query.strip():
                raw_turns = _search_raw_turns(
                    connection, query, workspace, resolved_project_id, limit
                )

        return {
            "query": query,
            "scope": scope,
            "project_id": resolved_project_id or "global",
            "memories": memories,
            "raw_turns": raw_turns,
        }

    # -- helpers -----------------------------------------------------------

    def _refresh_memory_fts(
        self, connection: sqlite3.Connection, memory_id: str
    ) -> None:
        row = connection.execute(
            "SELECT memory_id, project_id, kind, statement, details, search_keys "
            "FROM memories WHERE memory_id = ?",
            (memory_id,),
        ).fetchone()
        connection.execute("DELETE FROM memories_fts WHERE memory_id = ?", (memory_id,))
        if row is None:
            return
        connection.execute(
            """
            INSERT INTO memories_fts (
                memory_id, project_id, kind, statement, details, search_keys
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                row["memory_id"],
                row["project_id"],
                row["kind"],
                row["statement"],
                row["details"],
                row["search_keys"],
            ),
        )

    def _memory_row(
        self, connection: sqlite3.Connection, memory_id: str
    ) -> Optional[Dict[str, Any]]:
        row = connection.execute(
            "SELECT * FROM memories WHERE memory_id = ?", (memory_id,)
        ).fetchone()
        if row is None:
            return None
        return _row_to_memory(row)

    def _record_cwd_project(
        self, connection: sqlite3.Connection, cwd: str, project_id: str
    ) -> None:
        normalized = cwd.strip().rstrip("/")
        if not normalized:
            return
        connection.execute(
            """
            INSERT INTO sync_state (key, value, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET
                value = excluded.value,
                updated_at = excluded.updated_at
            """,
            (
                f"cwd:{normalized}",
                json.dumps({"project_id": project_id}, ensure_ascii=False),
                _utcnow(),
            ),
        )

    def _resolve_project_with_conn(
        self, connection: sqlite3.Connection, value: Optional[str | Path]
    ) -> str:
        if value is None:
            return "global"
        raw = str(value).strip()
        if not raw or raw == "global":
            return "global"
        if raw.startswith(("git://", "git-root://", "dir://", "id://")):
            return raw
        resolved = resolve_project_id(raw)
        if not resolved.startswith("dir://"):
            return resolved
        # When an Agent calls the central Hub's HTTP /mcp endpoint directly
        # from a remote machine, `cwd` may not exist on the Hub's disk. Walk up
        # the path prefixes against edge-reported (cwd -> project_id) mappings.
        candidate = raw.rstrip("/")
        while candidate and candidate != "/":
            row = connection.execute(
                "SELECT value FROM sync_state WHERE key = ?",
                (f"cwd:{candidate}",),
            ).fetchone()
            if row is not None:
                try:
                    mapped = json.loads(row["value"]).get("project_id")
                    if mapped:
                        return str(mapped)
                except Exception:
                    pass
            parent = candidate.rsplit("/", 1)[0]
            if parent == candidate:
                break
            candidate = parent
        return resolved

    def stats(self) -> Dict[str, Any]:
        with self._connection() as connection:
            evidence = connection.execute(
                "SELECT COUNT(*) AS n FROM evidence_turns"
            ).fetchone()["n"]
            unprocessed = connection.execute(
                "SELECT COUNT(*) AS n FROM evidence_turns WHERE processed = 0"
            ).fetchone()["n"]
            memories = connection.execute(
                "SELECT COUNT(*) AS n FROM memories"
            ).fetchone()["n"]
            active = connection.execute(
                "SELECT COUNT(*) AS n FROM memories WHERE status = 'active'"
            ).fetchone()["n"]
            projects = [
                row["project_id"]
                for row in connection.execute(
                    "SELECT DISTINCT project_id FROM memories ORDER BY project_id"
                ).fetchall()
            ]
        return {
            "db_path": str(self.db_path),
            "evidence_turns": evidence,
            "unprocessed_turns": unprocessed,
            "memories": memories,
            "active_memories": active,
            "projects": projects,
        }


# --------------------------------------------------------------------------
# ranking
# --------------------------------------------------------------------------


def _lexical_memory_ids(
    connection: sqlite3.Connection,
    query: str,
    workspace: bool,
    project_id: Optional[str],
    cap: int,
) -> List[str]:
    scope_sql = ""
    scope_params: List[Any] = []
    if workspace:
        scope_sql = " AND m.project_id IN (?, 'global')"
        scope_params.append(project_id)

    identifiers: List[str] = []
    seen = set()

    expression = _fts_expression(query)
    if expression is not None:
        try:
            rows = connection.execute(
                f"""
                SELECT m.memory_id AS mid
                FROM memories_fts f
                JOIN memories m ON m.memory_id = f.memory_id
                WHERE memories_fts MATCH ? AND m.status = 'active'{scope_sql}
                ORDER BY bm25(memories_fts)
                LIMIT ?
                """,
                [expression, *scope_params, cap],
            ).fetchall()
        except sqlite3.OperationalError:
            rows = []
        for row in rows:
            identifier = row["mid"]
            if identifier not in seen:
                seen.add(identifier)
                identifiers.append(identifier)

    pattern = _like_pattern(query)
    if pattern is not None:
        rows = connection.execute(
            f"""
            SELECT m.memory_id AS mid
            FROM memories m
            WHERE m.status = 'active'{scope_sql}
              AND LOWER(m.statement || ' ' || m.details || ' ' || m.search_keys)
                  LIKE ? ESCAPE '\\'
            ORDER BY m.updated_at DESC, m.memory_id ASC
            LIMIT ?
            """,
            [*scope_params, pattern, cap],
        ).fetchall()
        for row in rows:
            identifier = row["mid"]
            if identifier not in seen:
                seen.add(identifier)
                identifiers.append(identifier)

    return identifiers


def _vector_memory_ids(
    connection: sqlite3.Connection,
    query_embedding: Sequence[float],
    workspace: bool,
    project_id: Optional[str],
    cap: int,
) -> List[str]:
    scope_sql = ""
    scope_params: List[Any] = []
    if workspace:
        scope_sql = " AND project_id IN (?, 'global')"
        scope_params.append(project_id)

    query = np.asarray(query_embedding, dtype=np.float32).ravel()
    if query.size == 0:
        return []

    rows = connection.execute(
        f"""
        SELECT memory_id, embedding FROM memories
        WHERE status = 'active' AND embedding IS NOT NULL{scope_sql}
        """,
        scope_params,
    ).fetchall()

    scored: List[Tuple[str, float]] = []
    for row in rows:
        vector = np.frombuffer(row["embedding"], dtype=np.float32)
        if vector.shape != query.shape:
            continue
        similarity = _cosine_similarity(query, vector)
        if similarity >= _MIN_VECTOR_SIMILARITY:
            scored.append((row["memory_id"], similarity))
    scored.sort(key=lambda item: (-item[1], item[0]))
    return [identifier for identifier, _ in scored[:cap]]


def _search_raw_turns(
    connection: sqlite3.Connection,
    query: str,
    workspace: bool,
    project_id: Optional[str],
    limit: int,
) -> List[Dict[str, Any]]:
    scope_sql = ""
    scope_params: List[Any] = []
    if workspace:
        scope_sql = " AND e.project_id = ?"
        scope_params.append(project_id)

    cap = max(1, int(limit) * 5)
    identifiers: List[str] = []
    seen = set()

    expression = _fts_expression(query)
    if expression is not None:
        try:
            rows = connection.execute(
                f"""
                SELECT e.evidence_id AS eid
                FROM evidence_fts f
                JOIN evidence_turns e ON e.evidence_id = f.evidence_id
                WHERE evidence_fts MATCH ?{scope_sql}
                ORDER BY bm25(evidence_fts)
                LIMIT ?
                """,
                [expression, *scope_params, cap],
            ).fetchall()
        except sqlite3.OperationalError:
            rows = []
        for row in rows:
            identifier = row["eid"]
            if identifier not in seen:
                seen.add(identifier)
                identifiers.append(identifier)

    pattern = _like_pattern(query)
    if pattern is not None:
        rows = connection.execute(
            f"""
            SELECT e.evidence_id AS eid
            FROM evidence_turns e
            WHERE LOWER(e.content) LIKE ? ESCAPE '\\'{scope_sql}
            ORDER BY e.created_at DESC, e.evidence_id ASC
            LIMIT ?
            """,
            [pattern, *scope_params, cap],
        ).fetchall()
        for row in rows:
            identifier = row["eid"]
            if identifier not in seen:
                seen.add(identifier)
                identifiers.append(identifier)

    turns: List[Dict[str, Any]] = []
    for identifier in identifiers[:limit]:
        row = connection.execute(
            "SELECT * FROM evidence_turns WHERE evidence_id = ?", (identifier,)
        ).fetchone()
        if row is not None:
            turns.append(dict(row))
    return turns


def _fuse_rankings(
    rankings: Sequence[Sequence[str]], limit: int
) -> List[Tuple[str, float]]:
    scores: Dict[str, float] = {}
    for ranking in rankings:
        for rank, identifier in enumerate(ranking):
            scores[identifier] = scores.get(identifier, 0.0) + 1.0 / (_RRF_K + rank)
    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return ordered[: max(0, int(limit))]


def _fts_expression(query: str) -> Optional[str]:
    """A quoted OR-join of the query's tokens, or None when there are none.

    Quoting each token makes FTS5 read it as a literal string, so a query
    carrying `:`, `"`, `(`, `*` or a bare `AND` can never be mistaken for
    syntax and raise `sqlite3.OperationalError`.
    """
    tokens = _FTS_TOKEN.findall(query)
    if not tokens:
        return None
    return " OR ".join(f'"{token}"' for token in tokens[:24])


def _like_pattern(query: str) -> Optional[str]:
    if not query:
        return None
    escaped = (
        query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    )
    return f"%{escaped}%"


def _cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return float(np.dot(left, right) / (left_norm * right_norm))


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------


def _turn_content(turn: Mapping[str, Any]) -> str:
    for key in ("content", "verbatim", "text"):
        value = turn.get(key)
        if isinstance(value, str) and value:
            return value
    return ""


def _clean(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _embedding_bytes(embedding: Optional[Sequence[float]]) -> Optional[bytes]:
    if embedding is None:
        return None
    vector = np.asarray(embedding, dtype=np.float32)
    return vector.tobytes()


def _json_list(value: Any) -> List[Any]:
    if not isinstance(value, str) or not value:
        return []
    try:
        decoded = json.loads(value)
    except ValueError:
        return []
    return decoded if isinstance(decoded, list) else []


def _row_to_memory(row: sqlite3.Row) -> Dict[str, Any]:
    data = {key: row[key] for key in row.keys()}
    data["has_embedding"] = data.get("embedding") is not None
    data.pop("embedding", None)
    data["search_keys"] = _json_list(data.get("search_keys"))
    data["evidence_ids"] = _json_list(data.get("evidence_ids"))
    return data
