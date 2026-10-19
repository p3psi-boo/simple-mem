# simple-mem

**`simple-mem`** is a minimal single-user, multi-machine, multi-Agent/Harness developer memory system split into two standalone programs:

1. **`simple-mem` (Go Edge Client — zero external Go dependencies, single static binary)**
   - Compiled with `CGO_ENABLED=0` (`go.mod` uses 100% Go standard library).
   - Distributed to any developer machine (macOS, Linux, Windows/WSL, dev containers) with `<5ms` startup latency for real-time agent hooks and `stdio` MCP.
   - Modules:
     - `internal/project`: 4-Tier Project Identity Waterfall (`.simple-mem.json` / `SIMPLE_MEM_PROJECT_ID` $\rightarrow$ canonicalized Git remote URL `git://<host>/<owner>/<repo>` including `git worktree` `commondir` $\rightarrow$ local Git root commit SHA `git-root://<sha>` $\rightarrow$ `dir://<slug>`).
     - `internal/transcript`: Incremental JSONL transcript parsers, auto-discovery, and hook payload parsers (`UserPromptSubmit` & `Stop` with `transcript_path` fallback) for `claude-code`, `factory-droid`, `codex`, and `pi`.
     - `internal/shipper`: Atomic cursor-based incremental shipper and Hub HTTP client (`Ingest`, `Briefing`, `Search`, `FetchThread`, `Remember`, `Forget`).
     - `internal/mcp`: Native JSON-RPC 2.0 `stdio` MCP server and edge config generator (`--write-droid`, `--write-claude`).
     - `cmd/simple-mem`: Unified Edge CLI (`project-id`, `ship`, `capture`, `mcp`, `briefing`, `search`, `thread`, `remember`, `forget`, `edge-config`).

2. **`simple-mem-hub` (Python Central Hub Server — single-file SQLite + FastAPI)**
   - Runs on a central host or inside Docker (`Dockerfile` included).
   - Modules:
     - `simple_mem/store.py`: Single-file SQLite (`WAL` + `FTS5` BM25 + CJK substring fallback + `float32` BLOB embeddings + NumPy cosine similarity + RRF hybrid fusion + working memory briefing + thread context expansion).
     - `simple_mem/extractor.py`: Dual-layer developer memory extractor (`fact` / `decision` / `preference` + `experience` with `search_keys` & `boundary_envs`) and `ADD / UPDATE / DELETE / NOOP` conflict reconciler.
     - `simple_mem/hub.py` & `simple_mem/cli.py`: FastAPI Hub (`/health`, `/api/v1/ingest`, `/api/v1/briefing`, `/api/v1/search`, `/api/v1/threads/{id}`, `/api/v1/remember`, `/api/v1/memories/{id}`, `/api/v1/extract`, and Streamable HTTP `POST /mcp`) plus `simple-mem-hub` CLI (`serve`, `extract`, `stats`, `search`).

## MCP Tools (Streamable HTTP `POST /mcp` & Edge `stdio`)

| Tool | Description |
| :--- | :--- |
| `simple_mem_briefing` | Read the working-memory briefing for the current project without needing a search query (`preferences`, `decisions`, `experiences`, `facts`, and `recent_threads`). |
| `simple_mem_search` | Hybrid RRF search (FTS5 BM25 + `float32` cosine similarity) across governed developer memories and cross-machine raw conversation turns. |
| `simple_mem_thread` | Fetch ordered multi-turn conversation context for a `conversation_id` (optionally centered around `around_message_id`). |
| `simple_mem_remember` | Persist or reconcile (`ADD` / `UPDATE` / `DELETE` / `NOOP`) a project fact, architectural decision, user preference, or coding experience. |
| `simple_mem_forget` | Delete an outdated or incorrect memory by `memory_id`. |

## Quick Start

### 1. Run Central Hub (`simple-mem-hub`) with Docker

```bash
docker build -t simple-mem-hub:latest .

docker run -d \
  --name simple-mem-hub \
  --restart unless-stopped \
  -p 8765:8765 \
  -v "$HOME/.simple-mem:/data" \
  -e SIMPLE_MEM_API_KEY="your-secret-api-key" \
  -e SIMPLE_MEM_EXTRACT_INTERVAL="60" \
  -e SIMPLE_MEM_LLM_BASE_URL="https://api.openai.com/v1" \
  -e SIMPLE_MEM_LLM_API_KEY="sk-your-llm-api-key" \
  -e SIMPLE_MEM_LLM_MODEL="gpt-4o-mini" \
  -e SIMPLE_MEM_EMBED_BASE_URL="https://api.openai.com/v1" \
  -e SIMPLE_MEM_EMBED_API_KEY="sk-your-embed-api-key" \
  -e SIMPLE_MEM_EMBED_MODEL="text-embedding-3-small" \
  simple-mem-hub:latest
```

### 2. Build & Configure Go Edge Client (`simple-mem`)

```bash
# Build static binary for current OS/Arch
CGO_ENABLED=0 go build -ldflags="-s -w" -o ./bin/simple-mem ./cmd/simple-mem

# Cross-compile for macOS Apple Silicon if needed
GOOS=darwin GOARCH=arm64 CGO_ENABLED=0 go build -ldflags="-s -w" -o ./bin/simple-mem-darwin-arm64 ./cmd/simple-mem

# Install MCP + Hook config for Factory Droid & Claude Code
./bin/simple-mem edge-config \
  --remote-url http://hub-host:8765 \
  --api-key your-secret-api-key \
  --machine-id workstation-1 \
  --write-droid \
  --write-claude

# Incrementally ship local transcripts from Claude Code, Factory Droid, Codex, and pi
./bin/simple-mem ship \
  --source all \
  --remote-url http://hub-host:8765 \
  --api-key your-secret-api-key \
  --watch
```
