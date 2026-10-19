"""Tests for the Simple Mem Hub HTTP/MCP surface and the Hub CLI.

The store and extractor are faked: these tests are about the contract the Hub
and the CLI present to callers, not about SQLite. The real store is exercised
by its own workstream's tests. The edge client (shipping, hook capture, MCP
over stdio) lives in the standalone Go binary and is out of scope here.
"""

from __future__ import annotations

import io
import json

import httpx
import pytest
from fastapi.testclient import TestClient

from simple_mem import cli
from simple_mem import extractor as extractor_module
from simple_mem import hub
from simple_mem import project as project_module
from simple_mem import store as store_module
from simple_mem.hub import create_hub_app, handle_mcp_jsonrpc

AUTH = {"Authorization": "Bearer secret"}
WORKSPACE = "git://example.com/acme/repo"


class FakeStore:
    """An in-memory stand-in for SimpleMemStore."""

    def __init__(self) -> None:
        self.turns: list[dict] = []
        self.memories: dict[str, dict] = {}
        self.counter = 0
        self.ingest_calls: list[dict] = []
        self.search_calls: list[dict] = []
        self.remember_calls: list[dict] = []
        self.forget_calls: list[str] = []
        self.briefing_calls: list[dict] = []
        self.thread_calls: list[dict] = []

    def stats(self) -> dict:
        return {
            "db_path": ":memory:",
            "evidence_turns": len(self.turns),
            "memories": len(self.memories),
            "active_memories": len(self.memories),
            "projects": sorted({row["project_id"] for row in self.memories.values()}),
        }

    def ingest_turns(
        self,
        turns,
        *,
        machine_id="",
        default_source_id="unknown",
        default_project_id=None,
    ) -> dict:
        self.ingest_calls.append(
            {
                "turns": list(turns),
                "machine_id": machine_id,
                "default_source_id": default_source_id,
                "default_project_id": default_project_id,
            }
        )
        self.turns.extend(turns)
        return {"accepted": len(turns), "inserted": len(turns), "duplicates": 0}

    def search(
        self,
        query,
        *,
        project_id=None,
        scope="workspace",
        limit=10,
        query_embedding=None,
        include_raw=True,
    ) -> dict:
        self.search_calls.append(
            {
                "query": query,
                "project_id": project_id,
                "scope": scope,
                "limit": limit,
                "query_embedding": query_embedding,
            }
        )
        needle = (query or "").lower()
        hits = [
            memory
            for memory in self.memories.values()
            if needle and needle in memory["statement"].lower()
        ]
        return {
            "query": query,
            "scope": scope,
            "project_id": project_id or "global",
            "memories": hits[:limit],
            "raw_turns": [],
        }

    def remember(
        self,
        statement,
        *,
        project_id="global",
        kind="fact",
        details="",
        search_keys=(),
        boundary_envs="",
        source_id="manual",
        evidence_ids=(),
        confidence=0.9,
        embedding=None,
        memory_id=None,
    ) -> dict:
        self.counter += 1
        identifier = memory_id or f"mem_{self.counter}"
        record = {
            "memory_id": identifier,
            "statement": statement,
            "project_id": project_id,
            "kind": kind,
            "details": details,
            "search_keys": list(search_keys),
            "boundary_envs": boundary_envs,
            "source_id": source_id,
            "embedding": embedding,
        }
        self.memories[identifier] = record
        self.remember_calls.append(record)
        return record

    def forget(self, memory_id) -> bool:
        self.forget_calls.append(memory_id)
        return self.memories.pop(memory_id, None) is not None

    def briefing(self, project_id=None, limit=20) -> dict:
        self.briefing_calls.append({"project_id": project_id, "limit": limit})
        active = list(self.memories.values())
        return {
            "project_id": project_id or "global",
            "preferences": [m for m in active if m.get("kind") == "preference"],
            "decisions": [m for m in active if m.get("kind") == "decision"],
            "experiences": [m for m in active if m.get("kind") == "experience"],
            "facts": [m for m in active if m.get("kind") == "fact"],
            "memories": active[:limit],
            "recent_threads": [],
        }

    def fetch_thread(self, conversation_id, limit=50, around_message_id=None) -> dict:
        self.thread_calls.append(
            {
                "conversation_id": conversation_id,
                "limit": limit,
                "around_message_id": around_message_id,
            }
        )
        return {
            "conversation_id": conversation_id,
            "project_id": "global",
            "source_id": "",
            "machine_id": "",
            "total_turns": 0,
            "turns": [],
        }


class FakeExtractor:
    """A deterministic extractor that records what it was asked to do."""

    def __init__(self) -> None:
        self.embedded: list[str] = []
        self.reconciled: list[tuple[dict, list]] = []
        self.batches = 0

    def embed_text(self, text):
        self.embedded.append(text)
        return [0.1, 0.2, 0.3]

    def reconcile(self, candidate, existing):
        self.reconciled.append((candidate, existing))
        return {
            "memory_id": "reconciled_1",
            "statement": candidate["statement"],
            "project_id": candidate["project_id"],
            "kind": candidate["kind"],
        }

    def process_pending_batch(self, limit=30):
        self.batches += 1
        return {"turns_processed": limit, "processed": 3}


@pytest.fixture
def store() -> FakeStore:
    return FakeStore()


@pytest.fixture
def extractor() -> FakeExtractor:
    return FakeExtractor()


@pytest.fixture(autouse=True)
def _no_real_extractor(monkeypatch):
    """Keep the real extractor out of every test unless one is passed in."""
    monkeypatch.setattr(hub, "_load_extractor", lambda store: None)


@pytest.fixture
def client(store) -> TestClient:
    return TestClient(create_hub_app(store=store, api_key="secret"))


# -- REST -------------------------------------------------------------------


def test_health_is_open_and_reports_stats(store):
    client = TestClient(create_hub_app(store=store, api_key="secret"))
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["service"] == "simple-mem-hub"
    assert "stats" in body


def test_bearer_auth_is_enforced(client):
    assert client.post("/api/v1/ingest", json={"turns": []}).status_code == 401
    assert client.post("/api/v1/search", json={"query": "x"}).status_code == 401
    assert client.post("/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "ping"}).status_code == 401
    assert client.post("/api/v1/ingest", json={"turns": []}, headers=AUTH).status_code == 200


def test_x_api_key_is_accepted(client):
    response = client.post("/api/v1/search", json={"query": "x"}, headers={"X-API-Key": "secret"})
    assert response.status_code == 200


def test_ingest_endpoint_and_remote_alias(client, store):
    turns = [
        {
            "conversation_id": "c1",
            "message_id": "m1",
            "role": "user",
            "content": "hello",
        }
    ]
    response = client.post(
        "/api/v1/ingest",
        json={"machine_id": "m1", "source_id": "pi", "workspace": WORKSPACE, "turns": turns},
        headers=AUTH,
    )
    assert response.status_code == 200
    assert response.json() == {"accepted": 1, "inserted": 1, "duplicates": 0}
    call = store.ingest_calls[0]
    assert call["machine_id"] == "m1"
    assert call["default_source_id"] == "pi"
    assert call["default_project_id"] == WORKSPACE

    alias = client.post(
        "/api/v1/remote/ingest", json={"source_id": "codex", "turns": []}, headers=AUTH
    )
    assert alias.status_code == 200
    assert store.ingest_calls[1]["default_source_id"] == "codex"


def test_remember_search_and_forget(client, store):
    remembered = client.post(
        "/api/v1/remember",
        json={"statement": "The Hub uses SQLite FTS5", "kind": "decision", "cwd": WORKSPACE},
        headers=AUTH,
    )
    assert remembered.status_code == 200
    memory_id = remembered.json()["memory_id"]
    assert store.remember_calls[0]["project_id"] == WORKSPACE

    found = client.post(
        "/api/v1/search",
        json={"query": "SQLite", "project_id": WORKSPACE},
        headers=AUTH,
    )
    assert found.status_code == 200
    assert found.json()["query"] == "SQLite"
    assert store.search_calls[0]["project_id"] == WORKSPACE

    deleted = client.delete(f"/api/v1/memories/{memory_id}", headers=AUTH)
    assert deleted.status_code == 200
    assert deleted.json() == {"memory_id": memory_id, "deleted": True}
    again = client.delete(f"/api/v1/memories/{memory_id}", headers=AUTH)
    assert again.json()["deleted"] is False


def test_briefing_endpoint(client, store):
    store.remember("Prefer tabs", project_id=WORKSPACE, kind="preference")
    response = client.get(
        "/api/v1/briefing", params={"cwd": WORKSPACE, "limit": 5}, headers=AUTH
    )
    assert response.status_code == 200
    body = response.json()
    assert body["project_id"] == WORKSPACE
    assert [memory["statement"] for memory in body["preferences"]] == ["Prefer tabs"]
    assert store.briefing_calls == [{"project_id": WORKSPACE, "limit": 5}]


def test_briefing_endpoint_prefers_project_id(client, store):
    response = client.get(
        "/api/v1/briefing",
        params={"project_id": "proj://a", "cwd": "proj://b"},
        headers=AUTH,
    )
    assert response.status_code == 200
    assert store.briefing_calls == [{"project_id": "proj://a", "limit": 20}]


def test_thread_endpoint(client, store):
    response = client.get(
        "/api/v1/threads/conv-9",
        params={"limit": 3, "around_message_id": "m2"},
        headers=AUTH,
    )
    assert response.status_code == 200
    assert response.json()["conversation_id"] == "conv-9"
    assert store.thread_calls == [
        {"conversation_id": "conv-9", "limit": 3, "around_message_id": "m2"}
    ]


def test_extract_endpoint(store, extractor):
    client = TestClient(create_hub_app(store=store, extractor=extractor, api_key=""))
    response = client.post("/api/v1/extract")
    assert response.status_code == 200
    assert response.json()["processed"] == 3
    assert extractor.batches == 1


def test_extract_endpoint_accepts_limit(store, extractor):
    client = TestClient(create_hub_app(store=store, extractor=extractor, api_key=""))
    response = client.post("/api/v1/extract", json={"limit": 7})
    assert response.status_code == 200
    assert response.json()["turns_processed"] == 7
    assert extractor.batches == 1


def test_extract_endpoint_without_extractor(store):
    client = TestClient(create_hub_app(store=store, api_key=""))
    assert client.post("/api/v1/extract").status_code == 503


# -- MCP over HTTP ----------------------------------------------------------


def test_mcp_initialize(client):
    response = client.post(
        "/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "initialize"}, headers=AUTH
    )
    result = response.json()["result"]
    assert result["protocolVersion"] == "2024-11-05"
    assert result["capabilities"] == {"tools": {}}
    assert result["serverInfo"] == {"name": "simple-mem", "version": "0.2.0"}


def test_mcp_notification_returns_204(client):
    response = client.post(
        "/mcp",
        json={"jsonrpc": "2.0", "method": "notifications/initialized"},
        headers=AUTH,
    )
    assert response.status_code == 204


def test_mcp_tools_list(client):
    response = client.post(
        "/mcp", json={"jsonrpc": "2.0", "id": 2, "method": "tools/list"}, headers=AUTH
    )
    tools = response.json()["result"]["tools"]
    assert [tool["name"] for tool in tools] == [
        "simple_mem_briefing",
        "simple_mem_search",
        "simple_mem_thread",
        "simple_mem_remember",
        "simple_mem_forget",
    ]
    briefing = tools[0]
    assert set(briefing["inputSchema"]["properties"]) == {"cwd", "limit"}
    assert briefing["inputSchema"]["required"] == []
    assert briefing["inputSchema"]["properties"]["limit"]["default"] == 20
    search = tools[1]
    assert set(search["inputSchema"]["properties"]) == {"query", "scope", "cwd", "limit"}
    assert search["inputSchema"]["required"] == ["query"]
    thread = tools[2]
    assert set(thread["inputSchema"]["properties"]) == {
        "conversation_id",
        "around_message_id",
        "limit",
    }
    assert thread["inputSchema"]["required"] == ["conversation_id"]
    remember = tools[3]
    assert set(remember["inputSchema"]["properties"]) == {
        "statement",
        "kind",
        "details",
        "search_keys",
        "boundary_envs",
        "scope",
        "cwd",
    }
    assert tools[4]["inputSchema"]["required"] == ["memory_id"]


def test_mcp_ping_and_unknown_method(client):
    ping = client.post("/mcp", json={"jsonrpc": "2.0", "id": 3, "method": "ping"}, headers=AUTH)
    assert ping.json()["result"] == {}
    missing = client.post(
        "/mcp", json={"jsonrpc": "2.0", "id": 4, "method": "does/not/exist"}, headers=AUTH
    )
    assert missing.json()["error"]["code"] == -32601


def test_mcp_batch_drops_notifications(client):
    response = client.post(
        "/mcp",
        json=[
            {"jsonrpc": "2.0", "id": 1, "method": "ping"},
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
        ],
        headers=AUTH,
    )
    assert response.status_code == 200
    assert len(response.json()) == 1


def test_mcp_tools_call_search_uses_cwd_header(store, extractor):
    store.remember("The Hub uses SQLite FTS5", project_id=WORKSPACE)
    client = TestClient(create_hub_app(store=store, extractor=extractor, api_key=""))
    response = client.post(
        "/mcp",
        json={
            "jsonrpc": "2.0",
            "id": 5,
            "method": "tools/call",
            "params": {"name": "simple_mem_search", "arguments": {"query": "SQLite"}},
        },
        headers={"X-Simple-Mem-Cwd": WORKSPACE},
    )
    payload = json.loads(response.json()["result"]["content"][0]["text"])
    assert payload["memories"][0]["statement"] == "The Hub uses SQLite FTS5"
    assert store.search_calls[0]["project_id"] == WORKSPACE
    assert store.search_calls[0]["query_embedding"] == [0.1, 0.2, 0.3]
    assert extractor.embedded == ["SQLite"]


def test_mcp_tools_call_remember_via_extractor(store, extractor):
    response = client_call(
        store,
        extractor,
        "simple_mem_remember",
        {"statement": "Use WAL mode", "kind": "decision", "scope": "workspace"},
        default_cwd=WORKSPACE,
    )
    assert response["memory_id"] == "reconciled_1"
    candidate, existing = extractor.reconciled[0]
    assert candidate["statement"] == "Use WAL mode"
    assert candidate["project_id"] == WORKSPACE
    assert candidate["source_id"] == "mcp"
    assert existing == []
    assert store.remember_calls == []


def test_mcp_tools_call_remember_stores_without_extractor(store):
    response = client_call(
        store, None, "simple_mem_remember", {"statement": "Global fact", "scope": "global"}
    )
    assert store.remember_calls[0]["project_id"] == "global"
    assert response["statement"] == "Global fact"


def test_mcp_tools_call_forget(store):
    record = store.remember("Temporary", project_id="global")
    response = client_call(store, None, "simple_mem_forget", {"memory_id": record["memory_id"]})
    assert response["deleted"] is True
    assert store.forget_calls == [record["memory_id"]]


def test_mcp_tools_call_briefing_uses_cwd_header(store):
    store.remember("The Hub uses SQLite FTS5", project_id=WORKSPACE, kind="decision")
    client = TestClient(create_hub_app(store=store, api_key=""))
    response = client.post(
        "/mcp",
        json={
            "jsonrpc": "2.0",
            "id": 7,
            "method": "tools/call",
            "params": {"name": "simple_mem_briefing", "arguments": {}},
        },
        headers={"X-Simple-Mem-Cwd": WORKSPACE},
    )
    payload = json.loads(response.json()["result"]["content"][0]["text"])
    assert payload["project_id"] == WORKSPACE
    assert [memory["statement"] for memory in payload["decisions"]] == [
        "The Hub uses SQLite FTS5"
    ]
    assert store.briefing_calls == [{"project_id": WORKSPACE, "limit": 20}]


def test_mcp_tools_call_thread(store):
    payload = client_call(
        store,
        None,
        "simple_mem_thread",
        {"conversation_id": "conv-1", "limit": 5, "around_message_id": "m2"},
    )
    assert payload["conversation_id"] == "conv-1"
    assert store.thread_calls == [
        {"conversation_id": "conv-1", "limit": 5, "around_message_id": "m2"}
    ]


def test_mcp_tools_call_unknown_tool_is_error(store):
    response = handle_mcp_jsonrpc(
        {
            "jsonrpc": "2.0",
            "id": 6,
            "method": "tools/call",
            "params": {"name": "nope", "arguments": {}},
        },
        store=store,
    )
    assert response["result"]["isError"] is True


def test_mcp_notification_returns_none(store):
    assert (
        handle_mcp_jsonrpc(
            {"jsonrpc": "2.0", "method": "notifications/initialized"}, store=store
        )
        is None
    )


def client_call(store, extractor, tool, arguments, default_cwd=None):
    response = handle_mcp_jsonrpc(
        {
            "jsonrpc": "2.0",
            "id": 99,
            "method": "tools/call",
            "params": {"name": tool, "arguments": arguments},
        },
        store=store,
        extractor=extractor,
        default_cwd=default_cwd,
    )
    return json.loads(response["result"]["content"][0]["text"])


# -- CLI: help and dispatch -------------------------------------------------


def test_cli_usage_without_arguments():
    out = io.StringIO()
    assert cli.main([], stdout=out) == 2
    assert "Usage: simple-mem-hub" in out.getvalue()
    assert "serve" in out.getvalue()
    assert "extract" in out.getvalue()


def test_cli_unknown_command():
    out = io.StringIO()
    assert cli.main(["nonsense"], stdout=out) == 2
    assert "Usage:" in out.getvalue()


# -- CLI: serve -------------------------------------------------------------


class _ServeCapture:
    def __init__(self) -> None:
        self.app_kwargs: dict | None = None
        self.run_kwargs: dict = {}
        self.app = object()


@pytest.fixture
def serve_capture(monkeypatch) -> _ServeCapture:
    capture = _ServeCapture()

    def fake_create_hub_app(db_path=None, **kwargs):
        capture.app_kwargs = {"db_path": str(db_path), **kwargs}
        return capture.app

    import uvicorn

    monkeypatch.setattr(hub, "create_hub_app", fake_create_hub_app)
    monkeypatch.setattr(uvicorn, "run", lambda app, **kwargs: capture.run_kwargs.update(app=app, **kwargs))
    return capture


def test_cli_serve_starts_hub(serve_capture, monkeypatch):
    monkeypatch.delenv("SIMPLE_MEM_API_KEY", raising=False)
    out = io.StringIO()
    code = cli.main(
        [
            "serve",
            "--host",
            "127.0.0.1",
            "--port",
            "9123",
            "--db",
            "/tmp/hub.db",
            "--api-key",
            "secret",
            "--extract-interval",
            "15",
        ],
        stdout=out,
    )
    assert code == 0
    assert serve_capture.app_kwargs == {
        "db_path": "/tmp/hub.db",
        "api_key": "secret",
        "extract_interval": 15.0,
    }
    assert serve_capture.run_kwargs["app"] is serve_capture.app
    assert serve_capture.run_kwargs["host"] == "127.0.0.1"
    assert serve_capture.run_kwargs["port"] == 9123


def test_cli_serve_hub_alias(serve_capture):
    assert cli.main(["serve-hub", "--port", "9999"], stdout=io.StringIO()) == 0
    assert serve_capture.run_kwargs["port"] == 9999


def test_cli_bare_options_default_to_serve(serve_capture):
    assert cli.main(["--port", "9001"], stdout=io.StringIO()) == 0
    assert serve_capture.run_kwargs["port"] == 9001


def test_cli_serve_reads_api_key_from_environment(serve_capture, monkeypatch):
    monkeypatch.setenv("SIMPLE_MEM_API_KEY", "env-secret")
    assert cli.main(["serve"], stdout=io.StringIO()) == 0
    assert serve_capture.app_kwargs["api_key"] == "env-secret"
    assert serve_capture.app_kwargs["extract_interval"] == 0.0


def test_cli_serve_rejects_bad_option(serve_capture):
    out = io.StringIO()
    assert cli.main(["serve", "--nope"], stdout=out) == 2
    assert "unrecognized option" in out.getvalue()


# -- CLI: extract -----------------------------------------------------------


class _FakeResponse:
    def __init__(self, payload, status_code=200) -> None:
        self._payload = payload
        self.status_code = status_code
        self.raised = False

    def raise_for_status(self):
        self.raised = True

    def json(self):
        if self._payload is None:
            raise ValueError("no body")
        return self._payload


def test_cli_extract_local_uses_store_and_extractor(monkeypatch):
    seen: dict = {}

    class LocalStore:
        def __init__(self, db_path):
            seen["db_path"] = str(db_path)

    class LocalExtractor:
        def __init__(self, store):
            seen["store"] = store

        def process_pending_batch(self, limit=30):
            seen["limit"] = limit
            return {"turns_processed": limit, "added": 1}

        def close(self):
            seen["closed"] = True

    monkeypatch.setattr(store_module, "SimpleMemStore", LocalStore)
    monkeypatch.setattr(extractor_module, "MemoryExtractor", LocalExtractor)
    out = io.StringIO()

    code = cli.main(
        ["extract", "--db", "/tmp/hub.db", "--limit", "5", "--json"], stdout=out
    )

    assert code == 0
    assert seen["db_path"] == "/tmp/hub.db"
    assert isinstance(seen["store"], LocalStore)
    assert seen["limit"] == 5
    assert seen["closed"] is True
    assert json.loads(out.getvalue()) == {"turns_processed": 5, "added": 1}


def test_cli_extract_remote_posts_to_hub(monkeypatch):
    calls: list[dict] = []

    def fake_post(url, *, json=None, headers=None, timeout=None):
        calls.append({"url": url, "json": json, "headers": headers, "timeout": timeout})
        return _FakeResponse({"turns_processed": 4, "added": 2})

    monkeypatch.setattr(httpx, "post", fake_post)
    out = io.StringIO()

    code = cli.main(
        [
            "extract",
            "--remote-url",
            "http://hub.local:8765",
            "--api-key",
            "secret",
            "--limit",
            "4",
        ],
        stdout=out,
    )

    assert code == 0
    assert calls[0]["url"] == "http://hub.local:8765/api/v1/extract"
    assert calls[0]["json"] == {"limit": 4}
    assert calls[0]["headers"] == {"Authorization": "Bearer secret"}
    assert "turns_processed=4" in out.getvalue()


def test_cli_extract_remote_reports_failure(monkeypatch):
    def fake_post(*args, **kwargs):
        raise httpx.ConnectError("boom")

    monkeypatch.setattr(httpx, "post", fake_post)
    out = io.StringIO()

    assert cli.main(["extract", "--remote-url", "http://down"], stdout=out) == 1
    assert "extract failed" in out.getvalue()


# -- CLI: stats and search --------------------------------------------------


class _InspectStore:
    last_db: str | None = None
    search_result: dict = {"memories": []}

    def __init__(self, db_path):
        type(self).last_db = str(db_path)

    def stats(self):
        return {"evidence_turns": 3, "memories": 2}

    def search(self, query, *, project_id=None, scope="workspace", limit=10, **kwargs):
        type(self).last_call = {
            "query": query,
            "project_id": project_id,
            "scope": scope,
            "limit": limit,
        }
        return {"query": query, "memories": self.search_result["memories"]}


def test_cli_stats_reports_local_store(monkeypatch):
    monkeypatch.setattr(store_module, "SimpleMemStore", _InspectStore)
    out = io.StringIO()

    assert cli.main(["stats", "--db", "/tmp/hub.db", "--json"], stdout=out) == 0
    assert _InspectStore.last_db == "/tmp/hub.db"
    assert json.loads(out.getvalue()) == {"evidence_turns": 3, "memories": 2}


def test_cli_search_queries_local_store(monkeypatch):
    monkeypatch.setattr(store_module, "SimpleMemStore", _InspectStore)
    monkeypatch.setattr(
        project_module, "resolve_project_id", lambda path: f"proj://{path}"
    )
    out = io.StringIO()

    code = cli.main(
        ["search", "sqlite", "--cwd", "/work/repo", "--scope", "all", "--limit", "3", "--json"],
        stdout=out,
    )

    assert code == 0
    assert _InspectStore.last_call == {
        "query": "sqlite",
        "project_id": "proj:///work/repo",
        "scope": "all",
        "limit": 3,
    }
    assert json.loads(out.getvalue())["query"] == "sqlite"


def test_cli_search_reports_hit_count(monkeypatch):
    monkeypatch.setattr(store_module, "SimpleMemStore", _InspectStore)
    monkeypatch.setattr(project_module, "resolve_project_id", lambda path: path)
    _InspectStore.search_result = {"memories": [{"statement": "a"}, {"statement": "b"}]}
    out = io.StringIO()

    assert cli.main(["search", "sqlite", "--cwd", "/work/repo"], stdout=out) == 0
    assert out.getvalue().strip() == "2 memories"


def test_cli_search_requires_a_query():
    out = io.StringIO()
    assert cli.main(["search", "--json"], stdout=out) == 2
    assert "requires a query" in out.getvalue()
