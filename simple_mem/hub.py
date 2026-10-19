"""The Simple Mem Hub: one HTTP surface over a governed memory store.

It exposes verbatim evidence turns and distilled memories over HTTP, including
a native MCP JSON-RPC endpoint so an MCP host can call the Hub directly, plus a
small REST surface for shipping, searching, editing and extracting. The
JSON-RPC handler is transport-free, so the same three tools answer over
Streamable HTTP here and over stdio in the CLI.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response

__all__ = ["create_hub_app", "handle_mcp_jsonrpc"]

logger = logging.getLogger("simple_mem.hub")

PROTOCOL_VERSION = "2024-11-05"
SERVER_INFO: Dict[str, str] = {"name": "simple-mem", "version": "0.2.0"}
DEFAULT_DB_PATH = "~/.simple-mem/simple-mem.db"

#: The tools an agent is given. `scope` is workspace-local by default so an
#: agent sees its own project's memory without asking.
_TOOLS: List[Dict[str, Any]] = [
    {
        "name": "simple_mem_briefing",
        "description": (
            "Read the working-memory briefing for the current project without "
            "needing a search query: active preferences, architectural "
            "decisions, key facts, coding experiences, and recent "
            "cross-machine threads."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "cwd": {"type": "string"},
                "limit": {"type": "integer", "default": 20},
            },
            "required": [],
        },
    },
    {
        "name": "simple_mem_search",
        "description": (
            "Search across governed developer memories (facts, decisions, "
            "preferences, coding experiences) and cross-machine raw "
            "conversation turns."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "scope": {"type": "string", "enum": ["workspace", "all"], "default": "workspace"},
                "cwd": {"type": "string"},
                "limit": {"type": "integer", "default": 10},
            },
            "required": ["query"],
        },
    },
    {
        "name": "simple_mem_thread",
        "description": (
            "Fetch the ordered multi-turn conversation messages for a "
            "conversation_id so you can inspect the full question and "
            "assistant solution context."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "conversation_id": {"type": "string"},
                "around_message_id": {"type": "string"},
                "limit": {"type": "integer", "default": 20},
            },
            "required": ["conversation_id"],
        },
    },
    {
        "name": "simple_mem_remember",
        "description": (
            "Persist an important project fact, architectural decision, user "
            "preference, or coding experience."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "statement": {"type": "string"},
                "kind": {
                    "type": "string",
                    "enum": ["fact", "decision", "preference", "experience"],
                    "default": "fact",
                },
                "details": {"type": "string", "default": ""},
                "search_keys": {"type": "array", "items": {"type": "string"}, "default": []},
                "boundary_envs": {"type": "string", "default": ""},
                "scope": {"type": "string", "enum": ["workspace", "global"], "default": "workspace"},
                "cwd": {"type": "string"},
            },
            "required": ["statement"],
        },
    },
    {
        "name": "simple_mem_forget",
        "description": "Delete an outdated or incorrect memory by memory_id.",
        "inputSchema": {
            "type": "object",
            "properties": {"memory_id": {"type": "string"}},
            "required": ["memory_id"],
        },
    },
]


class _UnknownTool(LookupError):
    """A tools/call named a tool this server does not offer."""


# -- native MCP JSON-RPC ----------------------------------------------------


def handle_mcp_jsonrpc(
    payload: Mapping[str, Any],
    *,
    store: Any,
    extractor: Any | None = None,
    default_cwd: str | None = None,
) -> Optional[Dict[str, Any]]:
    """Answer one MCP JSON-RPC request, or None for a notification.

    Notifications carry no `id` and get no reply; the caller drops a `None`
    from a batch and answers 204 for a lone one.
    """
    if not isinstance(payload, Mapping):
        return {"jsonrpc": "2.0", "id": None, "error": {"code": -32600, "message": "Invalid Request"}}

    method = payload.get("method")
    identifier = payload.get("id")
    if method == "notifications/initialized" or (
        isinstance(method, str) and method.startswith("notifications/")
    ):
        return None
    params = payload.get("params")
    if not isinstance(params, Mapping):
        params = {}

    if method == "initialize":
        result: Any = {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {"tools": {}},
            "serverInfo": dict(SERVER_INFO),
        }
    elif method == "ping":
        result = {}
    elif method == "tools/list":
        result = {"tools": _TOOLS}
    elif method == "tools/call":
        result = _tool_call_result(
            params, store=store, extractor=extractor, default_cwd=default_cwd
        )
    else:
        if identifier is None:
            return None
        return {
            "jsonrpc": "2.0",
            "id": identifier,
            "error": {"code": -32601, "message": "Method not found"},
        }
    return {"jsonrpc": "2.0", "id": identifier, "result": result}


def _tool_call_result(params: Mapping[str, Any], *, store: Any, extractor: Any | None,
                      default_cwd: str | None) -> Dict[str, Any]:
    name = params.get("name")
    arguments = params.get("arguments")
    if not isinstance(arguments, Mapping):
        arguments = {}
    try:
        payload = _dispatch_tool(
            name, arguments, store=store, extractor=extractor, default_cwd=default_cwd
        )
    except _UnknownTool as error:
        return _content_result({"error": str(error)}, is_error=True)
    except Exception as error:  # a tool must not take the transport down with it
        return _content_result({"error": f"{type(error).__name__}: {error}"}, is_error=True)
    return _content_result(payload)


def _content_result(payload: Any, *, is_error: bool = False) -> Dict[str, Any]:
    text = json.dumps(payload, ensure_ascii=False, default=str)
    return {"content": [{"type": "text", "text": text}], "isError": is_error}


def _dispatch_tool(name: Any, arguments: Mapping[str, Any], *, store: Any,
                   extractor: Any | None, default_cwd: str | None) -> Dict[str, Any]:
    if name == "simple_mem_briefing":
        target_project = arguments.get("project_id") or arguments.get("cwd") or default_cwd
        return store.briefing(
            project_id=target_project,
            limit=_int(arguments.get("limit"), 20),
        )
    if name == "simple_mem_search":
        return _search_tool(arguments, store=store, extractor=extractor, default_cwd=default_cwd)
    if name == "simple_mem_thread":
        return store.fetch_thread(
            str(arguments.get("conversation_id") or ""),
            limit=_int(arguments.get("limit"), 20),
            around_message_id=arguments.get("around_message_id"),
        )
    if name == "simple_mem_remember":
        return _remember_tool(
            arguments, store=store, extractor=extractor, default_cwd=default_cwd
        )
    if name == "simple_mem_forget":
        memory_id = str(arguments.get("memory_id") or "")
        return {"memory_id": memory_id, "deleted": bool(store.forget(memory_id))}
    raise _UnknownTool(f"unknown tool: {name}")


def _search_tool(arguments: Mapping[str, Any], *, store: Any, extractor: Any | None,
                 default_cwd: str | None) -> Dict[str, Any]:
    query = str(arguments.get("query") or "")
    target_project = arguments.get("project_id") or arguments.get("cwd") or default_cwd
    return store.search(
        query,
        project_id=target_project,
        scope=str(arguments.get("scope") or "workspace"),
        limit=_int(arguments.get("limit"), 10),
        query_embedding=_embed(extractor, query),
    )


def _remember_tool(arguments: Mapping[str, Any], *, store: Any, extractor: Any | None,
                   default_cwd: str | None) -> Dict[str, Any]:
    scope = str(arguments.get("scope") or "workspace")
    if scope == "global":
        project_id = "global"
    else:
        project_id = (
            arguments.get("project_id")
            or arguments.get("cwd")
            or default_cwd
            or "global"
        )
    statement = str(arguments.get("statement") or "")
    candidate = {
        "statement": statement,
        "project_id": project_id,
        "kind": str(arguments.get("kind") or "fact"),
        "details": str(arguments.get("details") or ""),
        "search_keys": _string_list(arguments.get("search_keys")),
        "boundary_envs": str(arguments.get("boundary_envs") or ""),
        "source_id": "mcp",
        "embedding": _embed(extractor, statement),
    }
    return _remember_candidate(store, extractor, candidate, _existing_memories(store, candidate))


def _remember_candidate(store: Any, extractor: Any | None, candidate: Mapping[str, Any],
                        existing: Sequence[Any]) -> Dict[str, Any]:
    """Reconcile through the extractor when it can, store directly otherwise.

    Reconciliation is the extractor's to do -- it knows whether a candidate
    restates, refines or contradicts what is already held -- so a Hub without
    one simply records the candidate.
    """
    reconcile = getattr(extractor, "reconcile", None) if extractor is not None else None
    if callable(reconcile):
        try:
            outcome = reconcile(dict(candidate), list(existing))
        except TypeError:
            outcome = reconcile(dict(candidate))
        if isinstance(outcome, Mapping) and "action" in outcome and "memory_id" not in outcome:
            action = str(outcome.get("action") or "ADD").upper()
            target_id = str(outcome.get("target_memory_id") or "")
            merged = {
                "statement": str(outcome.get("statement") or candidate["statement"]),
                "project_id": candidate["project_id"],
                "kind": str(outcome.get("kind") or candidate["kind"]),
                "details": str(outcome.get("details") if outcome.get("details") is not None else candidate["details"]),
                "search_keys": _string_list(outcome.get("search_keys") or candidate["search_keys"]),
                "boundary_envs": str(
                    outcome.get("boundary_envs")
                    if outcome.get("boundary_envs") is not None
                    else candidate["boundary_envs"]
                ),
                "source_id": candidate.get("source_id", "mcp"),
                "embedding": candidate.get("embedding"),
            }
            if action == "UPDATE" and target_id and hasattr(store, "supersede_memory"):
                return store.supersede_memory(target_id, merged)
            if action == "DELETE" and target_id:
                return {"action": "DELETE", "memory_id": target_id, "deleted": bool(store.forget(target_id))}
            if action == "NOOP" and target_id:
                get_mem = getattr(store, "get_memory", None)
                existing_mem = get_mem(target_id) if callable(get_mem) else None
                return existing_mem or dict(outcome)
            return store.remember(**merged)
        if outcome is not None:
            return outcome
    return store.remember(
        candidate["statement"],
        project_id=candidate["project_id"],
        kind=candidate["kind"],
        details=candidate["details"],
        search_keys=candidate["search_keys"],
        boundary_envs=candidate["boundary_envs"],
        source_id=candidate.get("source_id", "mcp"),
        embedding=candidate.get("embedding"),
    )


def _existing_memories(store: Any, candidate: Mapping[str, Any]) -> List[Any]:
    search = getattr(store, "search", None)
    if not callable(search):
        return []
    try:
        found = search(
            candidate["statement"],
            project_id=candidate["project_id"],
            scope="workspace",
            limit=5,
        )
    except Exception:
        return []
    memories = found.get("memories") if isinstance(found, Mapping) else None
    return list(memories) if isinstance(memories, Sequence) else []


def _embed(extractor: Any | None, text: str) -> Optional[List[float]]:
    if extractor is None or not text:
        return None
    embed_text = getattr(extractor, "embed_text", None)
    if not callable(embed_text):
        return None
    try:
        return list(embed_text(text))
    except Exception:
        return None


def _int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _string_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, Sequence):
        return [str(item) for item in value if item is not None]
    return []


# -- FastAPI Hub ------------------------------------------------------------


def create_hub_app(
    db_path: str | Path | None = None,
    *,
    api_key: str | None = None,
    store: Any | None = None,
    extractor: Any | None = None,
    extract_interval: float = 0.0,
):
    """Build the Hub around a store and (optionally) an extractor.

    Both collaborators can be supplied directly -- how tests drive the routes
    without touching disk -- otherwise the store is opened from `db_path` and
    an extractor is imported when the package provides one.
    """
    resolved_api_key = (
        api_key if api_key is not None else os.environ.get("SIMPLE_MEM_API_KEY", "")
    )
    if store is None:
        from .store import SimpleMemStore

        store = SimpleMemStore(_resolve_db_path(db_path))
    if extractor is None:
        extractor = _load_extractor(store)

    app = FastAPI(title="simple-mem-hub", version=SERVER_INFO["version"])
    interval = _float(extract_interval, 0.0)
    if interval > 0:
        app.add_event_handler("startup", _start_extraction_loop(app, extractor, interval))
        app.add_event_handler("shutdown", _stop_extraction_loop(app))

    def require_api_key(request: Request) -> None:
        if resolved_api_key and _request_token(request) != resolved_api_key:
            raise HTTPException(status_code=401, detail="Unauthorized")

    protected = [Depends(require_api_key)]

    @app.get("/health")
    def health() -> Dict[str, Any]:
        return {"status": "ok", "service": "simple-mem-hub", "stats": _safe_stats(store)}

    @app.post("/api/v1/ingest", dependencies=protected)
    @app.post("/api/v1/remote/ingest", dependencies=protected)
    def ingest(payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        body = payload or {}
        return store.ingest_turns(
            body.get("turns") or [],
            machine_id=str(body.get("machine_id") or ""),
            default_source_id=str(body.get("source_id") or "unknown"),
            default_project_id=body.get("project_id") or body.get("workspace"),
        )

    @app.post("/api/v1/search", dependencies=protected)
    def search(payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        body = payload or {}
        query = str(body.get("query") or "")
        return store.search(
            query,
            project_id=body.get("project_id") or body.get("cwd"),
            scope=str(body.get("scope") or "workspace"),
            limit=_int(body.get("limit"), 10),
            query_embedding=_embed(extractor, query),
        )

    @app.get("/api/v1/briefing", dependencies=protected)
    def briefing(
        project_id: Optional[str] = None,
        cwd: Optional[str] = None,
        limit: int = 20,
    ) -> Dict[str, Any]:
        return store.briefing(project_id=project_id or cwd, limit=limit)

    @app.get("/api/v1/threads/{conversation_id}", dependencies=protected)
    def thread(
        conversation_id: str,
        limit: int = 50,
        around_message_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        return store.fetch_thread(
            conversation_id, limit=limit, around_message_id=around_message_id
        )

    @app.post("/api/v1/remember", dependencies=protected)
    def remember(payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        body = payload or {}
        return _remember_tool(
            body,
            store=store,
            extractor=extractor,
            default_cwd=body.get("cwd") or body.get("project_id"),
        )

    @app.delete("/api/v1/memories/{memory_id}", dependencies=protected)
    def forget(memory_id: str) -> Dict[str, Any]:
        return {"memory_id": memory_id, "deleted": bool(store.forget(memory_id))}

    @app.post("/api/v1/extract", dependencies=protected)
    def extract(payload: Optional[Dict[str, Any]] = None) -> Any:
        process_pending = (
            getattr(extractor, "process_pending_batch", None) if extractor else None
        )
        if not callable(process_pending):
            raise HTTPException(status_code=503, detail="extractor unavailable")
        limit = (payload or {}).get("limit")
        if limit is None:
            return process_pending()
        try:
            return process_pending(limit=_int(limit, 30))
        except TypeError:
            return process_pending()

    @app.post("/mcp", dependencies=protected)
    async def mcp(request: Request) -> Any:
        payload = await _read_json(request)
        default_cwd = (
            request.headers.get("x-simple-mem-project-id")
            or request.headers.get("x-simple-mem-cwd")
            or None
        )
        if isinstance(payload, list):
            responses = [
                handle_mcp_jsonrpc(
                    item, store=store, extractor=extractor, default_cwd=default_cwd
                )
                for item in payload
            ]
            responses = [response for response in responses if response is not None]
            if not responses:
                return Response(status_code=204)
            return JSONResponse(responses)
        response = handle_mcp_jsonrpc(
            payload, store=store, extractor=extractor, default_cwd=default_cwd
        )
        if response is None:
            return Response(status_code=204)
        return JSONResponse(response)

    return app


def _resolve_db_path(db_path: str | Path | None) -> str:
    if db_path is not None:
        return str(db_path)
    from_env = os.environ.get("SIMPLE_MEM_DB_PATH")
    if from_env:
        return from_env
    return str(Path(DEFAULT_DB_PATH).expanduser())


def _load_extractor(store: Any) -> Any | None:
    """Import the extractor when this checkout ships one, else None.

    The Hub still stores and searches without one, so a missing or unusable
    extractor degrades rather than breaks startup.
    """
    try:
        from .extractor import MemoryExtractor

        return MemoryExtractor(store)
    except Exception:
        return None


def _request_token(request: Any) -> str:
    authorization = request.headers.get("authorization", "")
    if authorization:
        scheme, _, value = authorization.partition(" ")
        if scheme.lower() == "bearer" and value.strip():
            return value.strip()
    return (request.headers.get("x-api-key") or "").strip()


async def _read_json(request: Any) -> Any:
    try:
        return await request.json()
    except Exception:
        return {}


def _safe_stats(store: Any) -> Any:
    stats = getattr(store, "stats", None)
    if not callable(stats):
        return {}
    try:
        return stats()
    except Exception:
        return {}


# -- periodic extraction ----------------------------------------------------


def _start_extraction_loop(app: FastAPI, extractor: Any | None, interval: float):
    """Return a startup hook that extracts a batch every `interval` seconds.

    The loop is a daemon thread: it never holds the server open on shutdown
    and a crashing batch is logged rather than allowed to kill the Hub.
    """

    def _start() -> None:
        process_pending = (
            getattr(extractor, "process_pending_batch", None) if extractor else None
        )
        if not callable(process_pending):
            logger.warning("[simple-mem] extract interval set but no extractor available")
            return
        stop = threading.Event()
        app.state.extraction_stop = stop

        def _run() -> None:
            while not stop.wait(interval):
                try:
                    process_pending()
                except Exception as error:  # noqa: BLE001 - keep the Hub alive
                    logger.warning("[simple-mem] scheduled extraction failed: %s", error)

        thread = threading.Thread(target=_run, name="simple-mem-extractor", daemon=True)
        app.state.extraction_thread = thread
        thread.start()

    return _start


def _stop_extraction_loop(app: FastAPI):
    def _stop() -> None:
        stop = getattr(app.state, "extraction_stop", None)
        if stop is not None:
            stop.set()

    return _stop
