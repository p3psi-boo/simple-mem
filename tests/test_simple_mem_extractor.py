"""Tests for the simple-mem developer memory extractor.

Everything here runs offline: model calls go through ``httpx.MockTransport``
and the store is a small in-memory fake that mirrors the real
``simple_mem.store.SimpleMemStore`` contract.
"""

from __future__ import annotations

import json
import sys
from itertools import count
from pathlib import Path
from typing import Any, Optional, Sequence

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from simple_mem.extractor import (  # noqa: E402
    EXTRACTION_SYSTEM_PROMPT,
    RECONCILE_SYSTEM_PROMPT,
    LLMConfig,
    MemoryExtractor,
)

LLM_CFG = LLMConfig(
    llm_base_url="https://llm.test/v1",
    llm_api_key="llm-key",
    llm_model="test-model",
    embed_base_url="https://embed.test/v1",
    embed_api_key="embed-key",
    embed_model="test-embed",
)


class FakeStore:
    """Minimal duck-typed store matching SimpleMemStore's public surface."""

    def __init__(self, turns=None, memories=None):
        self.turns = [dict(turn) for turn in (turns or [])]
        self.memories = {memory["memory_id"]: dict(memory) for memory in (memories or [])}
        self.remembered: list[dict[str, Any]] = []
        self.marked_batches: list[list[str]] = []
        self.forgotten: list[str] = []
        self._ids = count(1)

    # -- evidence ------------------------------------------------------

    def fetch_unprocessed_turns(self, limit: int = 50):
        pending = [turn for turn in self.turns if not turn.get("processed")]
        return pending[:limit]

    def mark_turns_processed(self, evidence_ids: Sequence[str]) -> int:
        wanted = {str(value) for value in evidence_ids}
        for turn in self.turns:
            if str(turn.get("evidence_id")) in wanted:
                turn["processed"] = 1
        self.marked_batches.append([str(value) for value in evidence_ids])
        return len(wanted)

    # -- memories ------------------------------------------------------

    def remember(self, statement, **fields):
        memory_id = fields.pop("memory_id", None) or f"mem_{next(self._ids)}"
        record = {
            "memory_id": memory_id,
            "statement": statement,
            "status": "active",
            **fields,
        }
        self.memories[memory_id] = record
        self.remembered.append(record)
        return record

    def supersede_memory(self, old_memory_id, new_memory):
        if old_memory_id in self.memories:
            self.memories[old_memory_id]["status"] = "superseded"
        return self.remember(
            new_memory.get("statement", ""),
            **{key: value for key, value in new_memory.items() if key != "statement"},
        )

    def search(self, query, *, project_id=None, scope="workspace", include_raw=True, **_):
        hits = []
        for record in self.memories.values():
            if record.get("status") != "active":
                continue
            if project_id and record.get("project_id") not in (None, project_id):
                continue
            hits.append(dict(record))
        return {"query": query, "memories": hits, "raw_turns": []}

    def forget(self, memory_id):
        removed = self.memories.pop(memory_id, None)
        if removed is not None:
            self.forgotten.append(memory_id)
        return removed is not None


class MockEndpoint:
    """A scripted OpenAI-compatible endpoint for extraction and reconcile."""

    def __init__(self, extract_payload, reconcile_payloads=()):
        self.extract_payload = extract_payload
        self.reconcile_payloads = list(reconcile_payloads)
        self.extract_calls = 0
        self.reconcile_calls = 0
        self.embedding_calls = 0

    def handler(self, request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/embeddings"):
            self.embedding_calls += 1
            return httpx.Response(200, json={"data": [{"embedding": [0.1, 0.2, 0.3]}]})
        body = json.loads(request.content)
        system = body["messages"][0]["content"]
        if system == EXTRACTION_SYSTEM_PROMPT:
            self.extract_calls += 1
            content = json.dumps(self.extract_payload)
        elif system == RECONCILE_SYSTEM_PROMPT:
            self.reconcile_calls += 1
            payload = self.reconcile_payloads.pop(0) if self.reconcile_payloads else {"action": "ADD"}
            content = json.dumps(payload)
        else:  # pragma: no cover - guards against prompt drift
            raise AssertionError(f"unexpected system prompt: {system[:60]!r}")
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    def client(self) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(self.handler))


def _turn(evidence_id: str, project_id: str, conversation_id: str, content: str, role="user"):
    return {
        "evidence_id": evidence_id,
        "project_id": project_id,
        "conversation_id": conversation_id,
        "role": role,
        "content": content,
    }


# ======================================================================
# LLMConfig
# ======================================================================

def test_from_env_prefers_simple_mem_and_falls_back_to_openai():
    config = LLMConfig.from_env(
        {
            "OPENAI_BASE_URL": "https://proxy/v1",
            "OPENAI_API_KEY": "openai-key",
            "SIMPLE_MEM_LLM_MODEL": "cheap-model",
            "SIMPLE_MEM_EMBED_MODEL": "small-embed",
        }
    )
    assert config.llm_base_url == "https://proxy/v1"
    assert config.llm_api_key == "openai-key"
    assert config.llm_model == "cheap-model"
    assert config.embed_base_url == "https://proxy/v1"  # falls back to llm base
    assert config.embed_api_key == "openai-key"
    assert config.embed_model == "small-embed"
    assert config.enable_heuristic_fallback is True


def test_from_env_explicit_values_win_and_heuristic_flag_parses():
    config = LLMConfig.from_env(
        {
            "SIMPLE_MEM_LLM_BASE_URL": "https://llm/v1",
            "SIMPLE_MEM_LLM_API_KEY": "k1",
            "SIMPLE_MEM_EMBED_BASE_URL": "https://embed/v1",
            "SIMPLE_MEM_EMBED_API_KEY": "k2",
            "SIMPLE_MEM_HEURISTIC_EXTRACT": "false",
        }
    )
    assert config.llm_base_url == "https://llm/v1"
    assert config.embed_base_url == "https://embed/v1"
    assert config.embed_api_key == "k2"
    assert config.enable_heuristic_fallback is False


# ======================================================================
# heuristics
# ======================================================================

def test_heuristic_extraction_finds_typed_markers():
    turns = [
        {"role": "user", "content": "记住：所有 API 返回的字段必须使用 snake_case。"},
        {"role": "assistant", "content": "架构决议：统一使用 PostgreSQL 存储事实层，放弃 MySQL。"},
        {"role": "user", "content": "踩坑：httpx 的 MockTransport 必须传 handler，否则请求无法路由。"},
        {"role": "assistant", "content": "Never use print() for logging; always use structlog."},
    ]
    extractor = MemoryExtractor(FakeStore(), LLMConfig())

    candidates = extractor.extract_candidates(turns)

    kinds = {candidate["kind"] for candidate in candidates}
    assert {"preference", "decision", "experience"} <= kinds
    assert all(2 <= len(candidate["search_keys"]) <= 6 for candidate in candidates)
    assert any("PostgreSQL" in candidate["statement"] for candidate in candidates)
    assert any("MockTransport" in candidate["details"] for candidate in candidates)


def test_heuristic_extraction_ignores_plain_chat():
    extractor = MemoryExtractor(FakeStore(), LLMConfig())
    turns = [{"role": "user", "content": "今天天气怎么样？顺便看看那个文件。"}]
    assert extractor.extract_candidates(turns) == []


def test_heuristic_fallback_can_be_disabled():
    extractor = MemoryExtractor(FakeStore(), LLMConfig(enable_heuristic_fallback=False))
    turns = [{"role": "user", "content": "记住：使用 PostgreSQL"}]
    assert extractor.extract_candidates(turns) == []


# ======================================================================
# LLM extraction & embeddings
# ======================================================================

def test_llm_extraction_parses_candidates_and_embeds():
    endpoint = MockEndpoint(
        {
            "memories": [
                {
                    "kind": "decision",
                    "statement": "统一使用 PostgreSQL 作为事实层存储",
                    "details": "放弃了 MySQL",
                    "search_keys": ["PostgreSQL", "事实层"],
                    "boundary_envs": "Python 3.12",
                }
            ]
        }
    )
    extractor = MemoryExtractor(FakeStore(), LLM_CFG, http_client=endpoint.client())

    candidates = extractor.extract_candidates([{"role": "user", "content": "换成 PostgreSQL 吧"}])

    assert endpoint.extract_calls == 1
    assert candidates == [
        {
            "kind": "decision",
            "statement": "统一使用 PostgreSQL 作为事实层存储",
            "details": "放弃了 MySQL",
            "search_keys": ["PostgreSQL", "事实层"],
            "boundary_envs": "Python 3.12",
        }
    ]
    assert extractor.embed_text("hello") == [0.1, 0.2, 0.3]
    assert endpoint.embedding_calls == 1


def test_llm_extraction_falls_back_to_heuristic_on_error():
    def failing_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "boom"})

    client = httpx.Client(transport=httpx.MockTransport(failing_handler))
    extractor = MemoryExtractor(FakeStore(), LLM_CFG, http_client=client)

    candidates = extractor.extract_candidates([{"role": "user", "content": "记住：使用 PostgreSQL"}])

    assert [candidate["kind"] for candidate in candidates] == ["preference"]


def test_embed_text_returns_none_without_config_or_on_error():
    assert MemoryExtractor(FakeStore(), LLMConfig()).embed_text("hello") is None

    def failing_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": "unavailable"})

    client = httpx.Client(transport=httpx.MockTransport(failing_handler))
    extractor = MemoryExtractor(FakeStore(), LLM_CFG, http_client=client)
    assert extractor.embed_text("hello") is None


# ======================================================================
# reconcile
# ======================================================================

def test_reconcile_adds_when_no_existing_memories():
    extractor = MemoryExtractor(FakeStore(), LLMConfig())
    result = extractor.reconcile({"kind": "fact", "statement": "A fresh fact"}, [])
    assert result == {"action": "ADD", "kind": "fact", "statement": "A fresh fact"}


def test_reconcile_noops_on_normalized_duplicate():
    extractor = MemoryExtractor(FakeStore(), LLMConfig())
    existing = [{"memory_id": "mem_1", "kind": "fact", "statement": "Use  PostgreSQL."}]
    result = extractor.reconcile({"kind": "fact", "statement": "use postgresql"}, existing)
    assert result == {"action": "NOOP", "target_memory_id": "mem_1"}


def test_reconcile_heuristic_updates_same_subject_with_new_conclusion():
    extractor = MemoryExtractor(FakeStore(), LLMConfig())
    existing = [
        {"memory_id": "mem_old", "kind": "decision", "statement": "统一使用 MySQL 存储事实层"}
    ]
    result = extractor.reconcile(
        {"kind": "decision", "statement": "统一使用 PostgreSQL 存储事实层"}, existing
    )
    assert result["action"] == "UPDATE"
    assert result["target_memory_id"] == "mem_old"
    assert result["statement"] == "统一使用 PostgreSQL 存储事实层"


def test_reconcile_llm_delete_and_invalid_target_fallback():
    delete_endpoint = MockEndpoint({}, [{"action": "DELETE", "target_memory_id": "mem_1"}])
    extractor = MemoryExtractor(FakeStore(), LLM_CFG, http_client=delete_endpoint.client())
    result = extractor.reconcile(
        {"kind": "fact", "statement": "X"}, [{"memory_id": "mem_1", "kind": "fact", "statement": "Y"}]
    )
    assert result == {"action": "DELETE", "target_memory_id": "mem_1"}

    bad_target = MockEndpoint({}, [{"action": "UPDATE", "target_memory_id": "missing"}])
    extractor = MemoryExtractor(FakeStore(), LLM_CFG, http_client=bad_target.client())
    result = extractor.reconcile(
        {"kind": "fact", "statement": "X"}, [{"memory_id": "mem_1", "kind": "fact", "statement": "Y"}]
    )
    assert result["action"] == "ADD"


# ======================================================================
# process_pending_batch
# ======================================================================

def test_process_pending_batch_empty_store():
    store = FakeStore()
    counts = MemoryExtractor(store, LLMConfig()).process_pending_batch()
    assert counts == {"turns_processed": 0, "added": 0, "updated": 0, "deleted": 0, "noop": 0}
    assert store.marked_batches == []


def test_process_pending_batch_llm_add_update_delete_noop():
    turns = [
        _turn("e1", "proj-a", "conv-a", "换成 PostgreSQL"),
        _turn("e2", "proj-b", "conv-b", "换成 PostgreSQL"),
        _turn("e3", "proj-c", "conv-c", "换成 PostgreSQL"),
        _turn("e4", "proj-d", "conv-d", "换成 PostgreSQL"),
    ]
    memories = [
        {"memory_id": "mem_a", "project_id": "proj-a", "kind": "decision",
         "statement": "统一使用 MySQL 存储事实层", "status": "active"},
        {"memory_id": "mem_b", "project_id": "proj-b", "kind": "fact",
         "statement": "an obsolete fact", "status": "active"},
        {"memory_id": "mem_c", "project_id": "proj-c", "kind": "fact",
         "statement": "already known", "status": "active"},
    ]
    store = FakeStore(turns=turns, memories=memories)
    endpoint = MockEndpoint(
        {
            "memories": [
                {
                    "kind": "decision",
                    "statement": "统一使用 PostgreSQL 存储事实层",
                    "details": "放弃了 MySQL",
                    "search_keys": ["PostgreSQL"],
                }
            ]
        },
        reconcile_payloads=[
            {"action": "UPDATE", "target_memory_id": "mem_a"},
            {"action": "DELETE", "target_memory_id": "mem_b"},
            {"action": "NOOP", "target_memory_id": "mem_c"},
            {"action": "ADD"},
        ],
    )
    extractor = MemoryExtractor(store, LLM_CFG, http_client=endpoint.client())

    counts = extractor.process_pending_batch(limit=10)

    assert counts == {"turns_processed": 4, "added": 1, "updated": 1, "deleted": 1, "noop": 1}
    assert store.memories["mem_a"]["status"] == "superseded"
    assert "mem_b" not in store.memories
    assert store.memories["mem_c"]["status"] == "active"
    assert any(
        record["statement"] == "统一使用 PostgreSQL 存储事实层"
        and record["project_id"] == "proj-a"
        and record["embedding"] == [0.1, 0.2, 0.3]
        for record in store.remembered
    )
    assert any(record["project_id"] == "proj-d" for record in store.remembered)
    assert endpoint.embedding_calls == 2  # one ADD and one UPDATE
    assert store.marked_batches == [["e1", "e2", "e3", "e4"]]
    assert all(turn["processed"] == 1 for turn in store.turns)


def test_process_pending_batch_heuristic_updates_and_marks_turns():
    store = FakeStore(
        turns=[_turn("e1", "proj-x", "conv-x", "架构决议：统一使用 PostgreSQL 存储事实层")],
        memories=[
            {"memory_id": "mem_old", "project_id": "proj-x", "kind": "decision",
             "statement": "统一使用 MySQL 存储事实层", "status": "active"}
        ],
    )
    extractor = MemoryExtractor(store, LLMConfig())

    counts = extractor.process_pending_batch()

    assert counts["updated"] == 1
    assert store.memories["mem_old"]["status"] == "superseded"
    assert any(
        record["statement"] == "统一使用 PostgreSQL 存储事实层" for record in store.remembered
    )
    assert store.turns[0]["processed"] == 1
    assert store.marked_batches == [["e1"]]


def test_process_pending_batch_groups_by_project_and_conversation():
    store = FakeStore(
        turns=[
            _turn("e1", "proj-x", "conv-1", "记住：使用 PostgreSQL 存储事实层"),
            _turn("e2", "proj-x", "conv-1", "记住：使用 Redis 做缓存层"),
            _turn("e3", "proj-y", "conv-2", "记住：使用 SQLite 做测试库"),
        ]
    )
    extractor = MemoryExtractor(store, LLMConfig())

    counts = extractor.process_pending_batch()

    assert counts["turns_processed"] == 3
    assert counts["added"] == 3  # two groups -> one extraction pass each
    projects = sorted(record["project_id"] for record in store.remembered)
    assert projects == ["proj-x", "proj-x", "proj-y"]
    assert store.marked_batches == [["e1", "e2", "e3"]]
