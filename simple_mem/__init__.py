"""simple-mem-hub: the central Hub over a single-SQLite developer memory store."""

from .extractor import LLMConfig, MemoryExtractor
from .hub import create_hub_app, handle_mcp_jsonrpc
from .project import canonicalize_git_url, resolve_project_id
from .store import SimpleMemStore

__version__ = "0.2.0"

__all__ = [
    "__version__",
    "LLMConfig",
    "MemoryExtractor",
    "SimpleMemStore",
    "canonicalize_git_url",
    "create_hub_app",
    "handle_mcp_jsonrpc",
    "resolve_project_id",
]
