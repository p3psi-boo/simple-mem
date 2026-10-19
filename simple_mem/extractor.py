"""Distill durable developer memories from raw conversation turns.

The cognitive core of simple-mem, lifted from ``hy_memory/coding`` and
stripped of its database and pipeline dependencies:

* **Extraction** turns verbatim turns into typed candidates -- ``fact`` /
  ``decision`` / ``preference`` / ``experience``.
* **Reconciliation** compares each candidate with the memories already in
  the project and decides ``ADD`` / ``UPDATE`` / ``DELETE`` / ``NOOP``, so
  ``统一使用 MySQL`` is superseded by ``统一使用 PostgreSQL`` instead of
  both being kept.

With an LLM configured it calls an OpenAI-compatible ``/chat/completions``
endpoint; without one it scans for explicit developer markers, so the
pipeline still works offline. The store is duck-typed: ``fetch_unprocessed
_turns``, ``mark_turns_processed``, ``search``, ``remember``,
``supersede_memory`` and ``forget`` (see ``simple_mem.store``).
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Optional

import httpx

logger = logging.getLogger(__name__)

__all__ = ["LLMConfig", "MemoryExtractor", "EXTRACTION_SYSTEM_PROMPT", "RECONCILE_SYSTEM_PROMPT"]

_HTTP_TIMEOUT = httpx.Timeout(30.0)
_VALID_KINDS = ("fact", "decision", "preference", "experience")
_FALSEY = {"0", "false", "no", "off", "n", "disabled"}
_MAX_SEARCH_KEYS = 6

EXTRACTION_SYSTEM_PROMPT = """\
Distill durable developer memories from a slice of a coding session. Return ONLY
{"memories": [...]}, no markdown fence or commentary; 0-6 items, and an empty list
is valid when nothing is worth keeping. Each memory has:
  kind        "fact" (durable statement about the project/system) | "decision"
              (architecture choice plus the rejected options) | "preference" (a rule
              the team wants followed always) | "experience" (reusable how-to:
              symptom, solution, pitfalls).
  statement   one sentence whose subject is the code/decision/rule, never "the user"
              or "the agent"; keep exact identifiers and error strings inline, since
              that is how a future search finds it.
  details     for "experience": concrete steps, commands, pitfalls; otherwise
              supporting detail, paths, versions, rejected options.
  search_keys 2-6 triggers: error codes, module names, APIs, config keys or phrases.
  boundary_envs  OS/language/framework constraints if it only holds there, else "".
Never narrate the session; record the settled decision and name the losing option
once; if a choice was reversed keep the winner and mark the loser dropped.
"""

RECONCILE_SYSTEM_PROMPT = """\
Merge ONE new developer memory into the existing active memories of the same project.
Return ONLY a JSON object, no markdown fence. Choose one action:
  ADD     -- no existing memory answers the same question.
  UPDATE  -- an existing memory answers the same question but is now wrong, stale or
             less complete; the new statement supersedes it.
  DELETE  -- the new memory explicitly invalidates an existing memory that UPDATE
             cannot repair.
  NOOP    -- the existing memory already says the same thing.
Output:
  {"action": "ADD"|"UPDATE"|"DELETE"|"NOOP",
   "target_memory_id": "<required for UPDATE/DELETE/NOOP>",
   "statement": "...", "details": "...", "search_keys": ["..."], "boundary_envs": "..."}
"Same question" means a future search could return either one; if the solution
differs prefer UPDATE and never keep both. UPDATE/DELETE/NOOP must name a
target_memory_id from EXISTING. DELETE only on explicit evidence, never for
staleness. For ADD/UPDATE return the complete final text, not a diff.
"""


@dataclass
class LLMConfig:
    """Model endpoints; empty credentials disable that capability."""

    llm_base_url: str = ""
    llm_api_key: str = ""
    llm_model: str = "gpt-4o-mini"
    embed_base_url: str = ""
    embed_api_key: str = ""
    embed_model: str = "text-embedding-3-small"
    enable_heuristic_fallback: bool = True

    @classmethod
    def from_env(cls, env: Optional[Mapping[str, str]] = None) -> "LLMConfig":
        source = os.environ if env is None else env

        def read(*names: str, default: str = "") -> str:
            for name in names:
                value = source.get(name)
                if value is not None and str(value).strip():
                    return str(value).strip()
            return default

        llm_base_url = read("SIMPLE_MEM_LLM_BASE_URL", "OPENAI_BASE_URL")
        llm_api_key = read("SIMPLE_MEM_LLM_API_KEY", "OPENAI_API_KEY")
        raw = source.get("SIMPLE_MEM_HEURISTIC_EXTRACT")
        return cls(
            llm_base_url=llm_base_url,
            llm_api_key=llm_api_key,
            llm_model=read("SIMPLE_MEM_LLM_MODEL", default="gpt-4o-mini"),
            embed_base_url=read("SIMPLE_MEM_EMBED_BASE_URL", default=llm_base_url),
            embed_api_key=read("SIMPLE_MEM_EMBED_API_KEY", default=llm_api_key),
            embed_model=read("SIMPLE_MEM_EMBED_MODEL", default="text-embedding-3-small"),
            enable_heuristic_fallback=not (
                raw is not None and str(raw).strip().lower() in _FALSEY
            ),
        )


# Markers are checked most-specific first; ASCII entries match case-insensitively.
_HEURISTIC_MARKERS: tuple[tuple[str, str], ...] = (
    ("架构决议", "decision"), ("架构决策", "decision"), ("统一使用", "decision"),
    ("最终采用", "decision"), ("决定：", "decision"), ("决定:", "decision"),
    ("decision:", "decision"), ("we decided", "decision"), ("settled on", "decision"),
    ("禁止使用", "preference"), ("不允许使用", "preference"), ("不要使用", "preference"),
    ("偏好：", "preference"), ("偏好:", "preference"), ("记住：", "preference"),
    ("记住:", "preference"), ("remember:", "preference"), ("rule:", "preference"),
    ("always use", "preference"), ("never use", "preference"),
    ("踩坑：", "experience"), ("踩坑:", "experience"), ("坑点：", "experience"),
    ("坑点:", "experience"), ("pitfall:", "experience"), ("gotcha:", "experience"),
    ("workaround:", "experience"), ("fact:", "fact"), ("note:", "fact"), ("注意：", "fact"),
)
_SENTENCE_SPLIT = re.compile(r"(?<=[。！？!?；;])|\n+")
_CLAUSE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_LEADING_NOISE = " \t:：-—,，。.、"
_KEY_PATTERNS = (
    re.compile(r"`([^`\n]{2,60})`"),
    re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*[A-Z][A-Za-z0-9_]*)\b"),
    re.compile(r"\b([A-Z][A-Z0-9_]{2,})\b"),
    re.compile(r"\b([A-Za-z0-9_-]+\.(?:py|js|ts|tsx|jsx|go|rs|java|rb|json|ya?ml|toml|md|sh|cfg|ini))\b"),
    re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*_[A-Za-z0-9_]+)\b"),
)
_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with", "use",
    "using", "used", "is", "are", "be", "must", "should", "always", "never",
    "please", "we", "it", "this", "that", "not", "no", "使用", "统一", "禁止",
    "偏好", "记住", "决定", "我们", "项目",
}


def _split_sentences(text: str) -> list[str]:
    out: list[str] = []
    for block in _SENTENCE_SPLIT.split(text or ""):
        out.extend(piece.strip() for piece in _CLAUSE_SPLIT.split(block) if piece.strip())
    return out


def _content_tokens(text: str) -> list[str]:
    tokens: list[str] = []
    for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}|[\u4e00-\u9fff]{2,6}", text or ""):
        if token.lower() not in _STOPWORDS and token not in tokens:
            tokens.append(token)
        if len(tokens) >= _MAX_SEARCH_KEYS:
            break
    return tokens


def _derive_search_keys(statement: str) -> list[str]:
    """Best-effort 2-6 triggers so heuristic candidates stay searchable."""
    keys: list[str] = []
    for pattern in _KEY_PATTERNS:
        for match in pattern.finditer(statement):
            token = match.group(1).strip()
            if token and token not in keys:
                keys.append(token)
    if len(keys) < 2:
        keys.extend(token for token in _content_tokens(statement) if token not in keys)
    if len(keys) < 2 and statement.strip():  # one-word statements still need 2 keys
        phrase = statement.strip()[:60]
        if phrase not in keys:
            keys.append(phrase)
    return keys[:_MAX_SEARCH_KEYS]


def _normalize_candidate(raw: Any, fallback_kind: str = "fact") -> Optional[dict[str, Any]]:
    if not isinstance(raw, Mapping):
        return None
    statement = str(raw.get("statement") or raw.get("task") or "").strip()
    if not statement:
        return None
    kind = str(raw.get("kind") or fallback_kind).strip().lower()
    if kind not in _VALID_KINDS:
        kind = fallback_kind if fallback_kind in _VALID_KINDS else "fact"
    details = raw.get("details")
    details = raw.get("solution") or "" if details is None else details
    if not isinstance(details, str):
        details = json.dumps(details, ensure_ascii=False)
    keys = raw.get("search_keys") or []
    if isinstance(keys, str):
        keys = re.split(r"[,，;；]", keys)
    keys = [str(key).strip() for key in keys if str(key).strip()] or _derive_search_keys(statement)
    boundary = raw.get("boundary_envs") or ""
    if isinstance(boundary, (list, tuple)):
        boundary = ", ".join(str(item) for item in boundary)
    return {
        "kind": kind,
        "statement": statement[:600],
        "details": details.strip(),
        "search_keys": keys[:_MAX_SEARCH_KEYS],
        "boundary_envs": str(boundary).strip(),
    }


def _format_turns(turns: Sequence[Mapping[str, Any]]) -> str:
    lines = []
    for turn in turns:
        content = turn.get("content") or turn.get("verbatim") or turn.get("text") or ""
        lines.append(f"[{turn.get('role') or 'user'}] {str(content).strip()}")
    return "\n".join(line for line in lines if line.strip())


def _load_json_object(text: str) -> dict[str, Any]:
    """Parse a JSON object from model output, tolerating code fences."""
    if not text:
        return {}
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z0-9]*\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    try:
        parsed = json.loads(cleaned)
    except ValueError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start == -1 or end <= start:
            return {}
        try:
            parsed = json.loads(cleaned[start : end + 1])
        except ValueError:
            return {}
    return parsed if isinstance(parsed, dict) else {}


def _normalize_statement(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower()).strip("。.！!？?；; ")


def _token_set(text: str) -> set[str]:
    found = re.findall(r"[A-Za-z_][A-Za-z0-9_]{1,}|[\u4e00-\u9fff]", str(text or "").lower())
    return {token for token in found if token not in _STOPWORDS}


def _statements_related(left: str, right: str) -> bool:
    left_tokens, right_tokens = _token_set(left), _token_set(right)
    if left_tokens and right_tokens:
        overlap = len(left_tokens & right_tokens) / min(len(left_tokens), len(right_tokens))
        if overlap >= 0.6:
            return True
    parts = re.split(r"[：:]", str(left or ""), maxsplit=1)
    prefix = _normalize_statement(parts[0]) if len(parts) == 2 else ""
    return bool(prefix) and prefix == _normalize_statement(re.split(r"[：:]", str(right or ""), maxsplit=1)[0])


class MemoryExtractor:
    """Turn verbatim turns into reconciled memories in a duck-typed store."""

    def __init__(self, store: Any, config: Optional[LLMConfig] = None,
                 http_client: Optional[httpx.Client] = None) -> None:
        self.store = store
        self.config = config if config is not None else LLMConfig.from_env()
        self._client = http_client

    @property
    def has_llm(self) -> bool:
        return bool(self.config.llm_base_url and self.config.llm_api_key)

    @property
    def has_embedding(self) -> bool:
        return bool(self.config.embed_base_url and self.config.embed_api_key)

    def _http(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(timeout=_HTTP_TIMEOUT)
        return self._client

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    @staticmethod
    def _headers(api_key: str) -> dict[str, str]:
        return {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    def _chat(self, system_prompt: str, user_prompt: str) -> str:
        payload = {
            "model": self.config.llm_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0,
        }
        response = self._http().post(
            self.config.llm_base_url.rstrip("/") + "/chat/completions",
            headers=self._headers(self.config.llm_api_key), json=payload,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"] or ""

    def embed_text(self, text: str) -> Optional[list[float]]:
        """Return the embedding of ``text``, or ``None`` when unavailable."""
        if not text or not self.has_embedding:
            return None
        try:
            response = self._http().post(
                self.config.embed_base_url.rstrip("/") + "/embeddings",
                headers=self._headers(self.config.embed_api_key),
                json={"model": self.config.embed_model, "input": text},
            )
            response.raise_for_status()
            return [float(value) for value in response.json()["data"][0]["embedding"]]
        except Exception as error:  # noqa: BLE001 - embedding is best-effort
            logger.warning("[simple-mem] embedding failed: %s", error)
            return None

    # -- extraction ----------------------------------------------------

    def extract_candidates(self, turns: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        """Extract candidate memories from a batch of verbatim turns."""
        if not turns:
            return []
        if self.has_llm:
            try:
                payload = _load_json_object(self._chat(
                    EXTRACTION_SYSTEM_PROMPT,
                    "CONVERSATION TURNS (extract only from these):\n" + _format_turns(turns),
                )).get("memories")
                if isinstance(payload, list):
                    normalized = (_normalize_candidate(raw) for raw in payload)
                    return [item for item in normalized if item is not None]
            except Exception as error:  # noqa: BLE001
                logger.warning("[simple-mem] LLM extraction failed: %s", error)
            if not self.config.enable_heuristic_fallback:
                return []
        return self.extract_candidates_heuristic(turns) if self.config.enable_heuristic_fallback else []

    def extract_candidates_heuristic(
        self, turns: Sequence[Mapping[str, Any]]
    ) -> list[dict[str, Any]]:
        """Marker scan -- the offline path; casual chat yields nothing."""
        candidates: list[dict[str, Any]] = []
        seen: set[str] = set()
        for turn in turns:
            if str(turn.get("role") or "").strip().lower() not in ("", "user", "assistant", "human"):
                continue
            content = turn.get("content") or turn.get("verbatim") or turn.get("text") or ""
            for sentence in _split_sentences(str(content)):
                lowered = sentence.lower()
                for marker, kind in _HEURISTIC_MARKERS:
                    position = lowered.find(marker.lower())
                    if position == -1:
                        continue
                    if marker.endswith((":", "：")) or position == 0:
                        tail = sentence[position + len(marker):].strip(_LEADING_NOISE)
                    else:
                        head, sep, rest = re.split(r"([：:])", sentence, maxsplit=1) if re.search(r"[：:]", sentence[:position]) else ("", "", sentence)
                        tail = (rest if sep and len(head.strip()) <= 8 else sentence).strip(_LEADING_NOISE)
                    statement = tail if len(tail) >= 4 else sentence.strip()
                    keys = _derive_search_keys(statement)
                    if len(keys) < 2:  # widen the net when the tail is one word
                        for extra in _derive_search_keys(sentence) + _derive_search_keys(str(content)):
                            if extra not in keys:
                                keys.append(extra)
                    candidate = _normalize_candidate({
                        "kind": kind,
                        "statement": statement,
                        "details": sentence.strip(),
                        "search_keys": keys[:_MAX_SEARCH_KEYS],
                    })
                    if candidate is not None:
                        key = _normalize_statement(candidate["statement"])
                        if key and key not in seen:
                            seen.add(key)
                            candidates.append(candidate)
                    break
        return candidates

    # -- reconciliation ------------------------------------------------

    def reconcile(self, candidate: Mapping[str, Any],
                  existing_memories: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        """Decide ADD / UPDATE / DELETE / NOOP for one candidate."""
        candidate = dict(candidate)
        existing = [dict(memory) for memory in existing_memories
                    if isinstance(memory, Mapping) and memory.get("memory_id")]
        if not existing:
            return {"action": "ADD", **candidate}
        normalized = _normalize_statement(candidate.get("statement"))
        for memory in existing:
            if _normalize_statement(memory.get("statement")) == normalized:
                return {"action": "NOOP", "target_memory_id": str(memory["memory_id"])}
        if self.has_llm:
            try:
                return self._reconcile_with_llm(candidate, existing)
            except Exception as error:  # noqa: BLE001
                logger.warning("[simple-mem] LLM reconcile failed: %s", error)
        return self._reconcile_heuristic(candidate, existing)

    def _reconcile_with_llm(self, candidate: Mapping[str, Any],
                            existing: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        rows = [
            {"memory_id": memory.get("memory_id"), "kind": memory.get("kind"),
             "statement": memory.get("statement"),
             "details": str(memory.get("details") or "")[:800],
             "search_keys": list(memory.get("search_keys") or [])}
            for memory in existing
        ]
        payload = _load_json_object(self._chat(
            RECONCILE_SYSTEM_PROMPT,
            "CANDIDATE:\n" + json.dumps(candidate, ensure_ascii=False, indent=2)
            + "\n\nEXISTING MEMORIES:\n" + json.dumps(rows, ensure_ascii=False, indent=2),
        ))
        action = str(payload.get("action") or "ADD").strip().upper()
        if action not in ("ADD", "UPDATE", "DELETE", "NOOP"):
            action = "ADD"
        target = str(payload.get("target_memory_id") or payload.get("target_id") or "").strip()
        valid_ids = {str(memory["memory_id"]) for memory in existing}
        if action in ("UPDATE", "DELETE") and target not in valid_ids:
            return {"action": "ADD", **candidate}
        if action == "NOOP":
            if target in valid_ids:
                return {"action": "NOOP", "target_memory_id": target}
            return {"action": "ADD", **candidate}
        if action == "DELETE":
            return {"action": "DELETE", "target_memory_id": target}
        merged = dict(candidate)
        if payload.get("statement"):
            merged["statement"] = str(payload["statement"]).strip()
        if payload.get("details") is not None:
            merged["details"] = str(payload.get("details") or "").strip()
        if payload.get("boundary_envs") is not None:
            merged["boundary_envs"] = str(payload.get("boundary_envs") or "").strip()
        keys = payload.get("search_keys")
        if isinstance(keys, list) and keys:
            merged["search_keys"] = [str(key).strip() for key in keys if str(key).strip()][:_MAX_SEARCH_KEYS]
        kind = str(payload.get("kind") or "").strip().lower()
        if kind in _VALID_KINDS:
            merged["kind"] = kind
        result = {"action": action, **merged}
        if action == "UPDATE":
            result["target_memory_id"] = target
        return result

    def _reconcile_heuristic(self, candidate: Mapping[str, Any],
                             existing: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        """Lexical fallback: compatible kind + related text means the old one loses."""
        cand_kind = str(candidate.get("kind") or "fact")
        declarative = {"fact", "decision", "preference"}
        for memory in existing:
            mem_kind = str(memory.get("kind") or "fact")
            if mem_kind != cand_kind and not (mem_kind in declarative and cand_kind in declarative):
                continue
            if _statements_related(candidate.get("statement", ""), memory.get("statement", "")):
                return {"action": "UPDATE", "target_memory_id": str(memory["memory_id"]),
                        **dict(candidate)}
        return {"action": "ADD", **dict(candidate)}

    # -- batch driver --------------------------------------------------

    def process_pending_batch(self, limit: int = 30) -> dict[str, int]:
        """Extract, reconcile and persist one batch of unprocessed turns."""
        turns = list(self.store.fetch_unprocessed_turns(limit=limit))
        counts = {"turns_processed": len(turns), "added": 0, "updated": 0, "deleted": 0, "noop": 0}
        if not turns:
            return counts

        groups: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
        evidence_ids: list[str] = []
        for turn in turns:
            if turn.get("evidence_id"):
                evidence_ids.append(str(turn["evidence_id"]))
            key = (str(turn.get("project_id") or "global"), str(turn.get("conversation_id") or ""))
            groups.setdefault(key, []).append(turn)

        for (project_id, _conversation_id), group_turns in groups.items():
            for candidate in self.extract_candidates(group_turns):
                decision = self.reconcile(candidate, self._existing_memories(candidate, project_id))
                action = str(decision.get("action") or "ADD").upper()
                target = str(decision.get("target_memory_id") or "")
                statement = decision.get("statement", candidate.get("statement", ""))
                if action in ("ADD", "UPDATE"):
                    record = {
                        "statement": statement,
                        "project_id": project_id,
                        "kind": decision.get("kind", "fact"),
                        "details": decision.get("details", ""),
                        "search_keys": decision.get("search_keys", []),
                        "boundary_envs": decision.get("boundary_envs", ""),
                        "source_id": "extractor",
                        "evidence_ids": evidence_ids,
                        "embedding": self.embed_text(statement),
                    }
                    if action == "ADD":
                        self.store.remember(**record)
                        counts["added"] += 1
                    else:
                        self.store.supersede_memory(target, record)
                        counts["updated"] += 1
                elif action == "DELETE" and target:
                    self.store.forget(target)
                    counts["deleted"] += 1
                else:
                    counts["noop"] += 1

        self.store.mark_turns_processed(evidence_ids)
        return counts

    def _existing_memories(self, candidate: Mapping[str, Any], project_id: str) -> list[Mapping[str, Any]]:
        try:
            result = self.store.search(candidate.get("statement", ""), project_id=project_id,
                                       scope="workspace", include_raw=False)
        except Exception as error:  # noqa: BLE001 - search is best-effort
            logger.warning("[simple-mem] existing-memory search failed: %s", error)
            return []
        memories = result.get("memories") if isinstance(result, Mapping) else result
        return [memory for memory in (memories or []) if isinstance(memory, Mapping)]
