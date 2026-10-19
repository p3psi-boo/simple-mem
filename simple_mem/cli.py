"""The ``simple-mem-hub`` command line.

This is the Hub server's own CLI: it starts the FastAPI Hub, drives one batch
of extraction (locally or against a running Hub), and offers the local
administrator two read-only inspection commands (``stats`` and ``search``).
The edge client that ships transcripts and speaks MCP lives in the standalone
Go ``simple-mem`` binary, so nothing here imports edge code.
"""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Sequence
from typing import Any, Dict, Optional, TextIO

__all__ = ["main"]

DEFAULT_DB_PATH = "~/.simple-mem/simple-mem.db"
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8765
DEFAULT_LIMIT = 30
DEFAULT_EXTRACT_INTERVAL = 0.0


def main(argv: Sequence[str] | None = None, *, stdout: TextIO | None = None) -> int:
    """Run one Hub command; the console script passes the process arguments."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    stream = stdout if stdout is not None else sys.stdout
    if not arguments:
        _usage(stream)
        return 2
    # ``simple-mem-hub --port 9000`` is the terse way to start the Hub.
    if arguments[0].startswith("-"):
        return _serve(arguments, stream)
    command, *options = arguments
    if command in ("serve", "serve-hub"):
        return _serve(options, stream)
    if command == "extract":
        return _extract(options, stream)
    if command == "stats":
        return _stats(options, stream)
    if command == "search":
        return _search(options, stream)
    _usage(stream)
    return 2


# -- option plumbing --------------------------------------------------------


def _parse(
    options: Sequence[str],
    value_options: Dict[str, str],
    flag_options: Dict[str, str],
    stream: TextIO,
) -> Optional[Dict[str, Any]]:
    """Turn a flat option list into a dict, or None with a message on error."""
    parsed: Dict[str, Any] = {}
    index = 0
    while index < len(options):
        option = options[index]
        if option in value_options:
            if index + 1 >= len(options):
                stream.write(f"simple-mem-hub: {option} requires a value\n")
                return None
            parsed[value_options[option]] = options[index + 1]
            index += 2
            continue
        if option in flag_options:
            parsed[flag_options[option]] = True
            index += 1
            continue
        stream.write(f"simple-mem-hub: unrecognized option: {option}\n")
        return None
    return parsed


def _env(*names: str) -> Optional[str]:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
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


def _db_path(explicit: Optional[str]) -> str:
    return explicit or _env("SIMPLE_MEM_DB_PATH") or DEFAULT_DB_PATH


def _project_id(explicit: Optional[str]) -> str:
    """Normalize a directory into the cross-machine project id the store uses."""
    candidate = explicit or os.environ.get("WORKSPACE_FOLDER_PATHS") or os.getcwd()
    try:
        from .project import resolve_project_id

        return resolve_project_id(candidate)
    except Exception:
        return candidate


def _write_json(stream: TextIO, payload: Any) -> None:
    stream.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
    stream.flush()


def _usage(stream: TextIO) -> None:
    stream.write(
        "Usage: simple-mem-hub <command> [options]\n"
        "  serve [--host HOST] [--port PORT] [--db PATH] [--api-key KEY]\n"
        "        [--extract-interval SECONDS]\n"
        "  extract [--db PATH] [--limit N] [--remote-url URL] [--api-key KEY] [--json]\n"
        "  stats [--db PATH] [--json]\n"
        "  search <query> [--scope workspace|all] [--cwd PATH] [--db PATH]\n"
        "         [--limit N] [--json]\n"
    )


# -- serve ------------------------------------------------------------------


def _serve(options: Sequence[str], stream: TextIO) -> int:
    parsed = _parse(
        options,
        {
            "--host": "host",
            "--port": "port",
            "--db": "db",
            "--db-path": "db",
            "--api-key": "api_key",
            "--api-token": "api_key",
            "--extract-interval": "extract_interval",
        },
        {},
        stream,
    )
    if parsed is None:
        return 2
    host = parsed.get("host") or _env("SIMPLE_MEM_HOST") or DEFAULT_HOST
    port = _int(parsed.get("port") or _env("SIMPLE_MEM_PORT"), DEFAULT_PORT)
    db_path = _db_path(parsed.get("db"))
    api_key = parsed.get("api_key") or _env("SIMPLE_MEM_API_KEY")
    interval = _float(
        parsed.get("extract_interval") or _env("SIMPLE_MEM_EXTRACT_INTERVAL"),
        DEFAULT_EXTRACT_INTERVAL,
    )

    from .hub import create_hub_app

    app = create_hub_app(db_path, api_key=api_key, extract_interval=interval)
    import uvicorn

    uvicorn.run(app, host=host, port=port)
    return 0


# -- extract ----------------------------------------------------------------


def _extract(options: Sequence[str], stream: TextIO) -> int:
    parsed = _parse(
        options,
        {
            "--db": "db",
            "--db-path": "db",
            "--limit": "limit",
            "--remote-url": "remote_url",
            "--api-key": "api_key",
            "--api-token": "api_key",
        },
        {"--json": "json"},
        stream,
    )
    if parsed is None:
        return 2
    limit = _int(parsed.get("limit"), DEFAULT_LIMIT)
    remote_url = parsed.get("remote_url") or _env("SIMPLE_MEM_REMOTE_URL")
    api_key = parsed.get("api_key") or _env("SIMPLE_MEM_API_KEY")
    if remote_url:
        result = _extract_remote(remote_url, api_key=api_key, limit=limit, stream=stream)
        if result is None:
            return 1
    else:
        result = _extract_local(_db_path(parsed.get("db")), limit=limit)
    if parsed.get("json"):
        _write_json(stream, result)
    else:
        stream.write("extracted: " + _summary_text(result) + "\n")
        stream.flush()
    return 0


def _extract_remote(
    remote_url: str, *, api_key: Optional[str], limit: int, stream: TextIO
) -> Optional[Any]:
    import httpx

    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    try:
        response = httpx.post(
            remote_url.rstrip("/") + "/api/v1/extract",
            json={"limit": limit},
            headers=headers,
            timeout=60.0,
        )
        response.raise_for_status()
        return response.json()
    except Exception as error:  # noqa: BLE001 - report, do not traceback
        stream.write(f"simple-mem-hub: extract failed: {error}\n")
        return None


def _extract_local(db_path: str, *, limit: int) -> Dict[str, Any]:
    from .extractor import MemoryExtractor
    from .store import SimpleMemStore

    store = SimpleMemStore(db_path)
    extractor = MemoryExtractor(store)
    try:
        return extractor.process_pending_batch(limit=limit)
    finally:
        extractor.close()


# -- stats ------------------------------------------------------------------


def _stats(options: Sequence[str], stream: TextIO) -> int:
    parsed = _parse(
        options,
        {"--db": "db", "--db-path": "db"},
        {"--json": "json"},
        stream,
    )
    if parsed is None:
        return 2
    from .store import SimpleMemStore

    store = SimpleMemStore(_db_path(parsed.get("db")))
    result = store.stats()
    if parsed.get("json"):
        _write_json(stream, result)
    else:
        stream.write(_summary_text(result) + "\n")
        stream.flush()
    return 0


# -- search -----------------------------------------------------------------


def _search(options: Sequence[str], stream: TextIO) -> int:
    if not options or options[0].startswith("-"):
        stream.write("simple-mem-hub: search requires a query\n")
        return 2
    query = options[0]
    parsed = _parse(
        options[1:],
        {
            "--scope": "scope",
            "--cwd": "cwd",
            "--db": "db",
            "--db-path": "db",
            "--limit": "limit",
        },
        {"--json": "json"},
        stream,
    )
    if parsed is None:
        return 2
    from .store import SimpleMemStore

    store = SimpleMemStore(_db_path(parsed.get("db")))
    result = store.search(
        query,
        project_id=_project_id(parsed.get("cwd")),
        scope=parsed.get("scope") or "workspace",
        limit=_int(parsed.get("limit"), 10),
    )
    if parsed.get("json"):
        _write_json(stream, result)
    else:
        stream.write(_search_text(result) + "\n")
        stream.flush()
    return 0


def _search_text(result: Any) -> str:
    if isinstance(result, dict):
        memories = result.get("memories")
        if isinstance(memories, list):
            return f"{len(memories)} memories"
    return _summary_text(result)


def _summary_text(result: Any) -> str:
    if isinstance(result, dict):
        return ", ".join(f"{key}={value}" for key, value in result.items())
    return json.dumps(result, ensure_ascii=False, default=str)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
