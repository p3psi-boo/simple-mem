"""Tests for the simple-mem project identity waterfall and SQLite store.

The project tests pin each tier of `resolve_project_id`: a declared identity,
the two spellings of one git remote, a worktree reached through a `commondir`
pointer, a local repository named by its root commit, and the directory-name
fallback. The store tests pin the behaviours the design depends on -- that
re-ingesting evidence is a no-op, that CJK substrings are still findable, that
vector and lexical hits fuse, and that a workspace sees its own project plus
the global one and nothing else.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from simple_mem import (
    SimpleMemStore,
    __version__,
    canonicalize_git_url,
    resolve_project_id,
)


# --------------------------------------------------------------------------
# fixtures and helpers
# --------------------------------------------------------------------------


def run_git(cwd: Path, *args: str) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env.update(
        {
            "GIT_AUTHOR_NAME": "Test",
            "GIT_AUTHOR_EMAIL": "test@example.com",
            "GIT_COMMITTER_NAME": "Test",
            "GIT_COMMITTER_EMAIL": "test@example.com",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_SYSTEM": os.devnull,
        }
    )
    return subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )


def make_repo(path: Path, *, remote: str | None = None) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    run_git(path, "init", "-q")
    (path / "tracked.txt").write_text("hello\n", encoding="utf-8")
    run_git(path, "add", "tracked.txt")
    run_git(path, "commit", "-q", "-m", "initial")
    if remote is not None:
        run_git(path, "remote", "add", "origin", remote)
    return path


@pytest.fixture
def store(tmp_path: Path) -> SimpleMemStore:
    return SimpleMemStore(tmp_path / "simple-mem.db")


# --------------------------------------------------------------------------
# canonicalize_git_url
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("git@GitHub.com:Acme/CoreEngine.git", "git://github.com/acme/coreengine"),
        ("ssh://git@github.com:2222/Org/Repo.git", "git://github.com/org/repo"),
        ("https://user:pass@GitHub.com/Org/Repo.git", "git://github.com/org/repo"),
        ("git://github.com/Org/Repo/", "git://github.com/org/repo"),
        ("http://example.com/team/project", "git://example.com/team/project"),
    ],
)
def test_canonicalize_git_url_variants(raw: str, expected: str) -> None:
    assert canonicalize_git_url(raw) == expected


@pytest.mark.parametrize(
    "raw",
    ["", "   ", "not a url", "https://github.com", "git@github.com:", "/a/b/c"],
)
def test_canonicalize_git_url_rejects_non_repositories(raw: str) -> None:
    assert canonicalize_git_url(raw) is None


# --------------------------------------------------------------------------
# Tier 0 -- the value is already an identity
# --------------------------------------------------------------------------


def test_tier0_passes_through_canonical_identities() -> None:
    assert resolve_project_id("global", env={}) == "global"
    assert resolve_project_id("proj://acme", env={}) == "proj://acme"
    assert resolve_project_id("dir://SomeDir/", env={}) == "dir://SomeDir"
    assert resolve_project_id("git-root://AbC123", env={}) == "git-root://AbC123"


def test_tier0_canonicalizes_git_identities() -> None:
    assert (
        resolve_project_id("git://GitHub.com/Acme/Core/", env={})
        == "git://github.com/acme/core"
    )
    assert (
        resolve_project_id("https://github.com/Acme/Core.git", env={})
        == "git://github.com/acme/core"
    )
    assert (
        resolve_project_id("git@github.com:Acme/Core.git", env={})
        == "git://github.com/acme/core"
    )


def test_global_without_any_cwd_is_ambient() -> None:
    assert resolve_project_id(None, env={}) == "global"
    assert resolve_project_id("", env={}) == "global"
    assert resolve_project_id(None, env={"SIMPLE_MEM_PROJECT_ID": " "}) == "global"
    assert (
        resolve_project_id(None, env={"SIMPLE_MEM_PROJECT_ID": "acme"})
        == "proj://acme"
    )
    assert (
        resolve_project_id(None, env={"SIMPLE_MEM_PROJECT_ID": "git://h/r"})
        == "git://h/r"
    )


# --------------------------------------------------------------------------
# Tier 1 -- an explicit declaration
# --------------------------------------------------------------------------


def test_tier1_reads_simple_mem_json_from_a_parent(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    nested = root / "packages" / "core"
    nested.mkdir(parents=True)
    (root / ".simple-mem.json").write_text(
        json.dumps({"project_id": "Acme-Core"}), encoding="utf-8"
    )
    assert resolve_project_id(nested, env={}) == "proj://Acme-Core"


def test_tier1_json_keeps_an_explicit_uri_and_global(tmp_path: Path) -> None:
    (tmp_path / ".simple-mem.json").write_text(
        json.dumps({"project_id": "git://host/team"}), encoding="utf-8"
    )
    assert resolve_project_id(tmp_path, env={}) == "git://host/team"

    (tmp_path / ".simple-mem.json").write_text(
        json.dumps({"project_id": "global"}), encoding="utf-8"
    )
    assert resolve_project_id(tmp_path, env={}) == "global"


def test_tier1_env_overrides_the_json_file(tmp_path: Path) -> None:
    (tmp_path / ".simple-mem.json").write_text(
        json.dumps({"project_id": "from-file"}), encoding="utf-8"
    )
    assert (
        resolve_project_id(tmp_path, env={"SIMPLE_MEM_PROJECT_ID": "from-env"})
        == "proj://from-env"
    )


def test_tier1_invalid_json_falls_through(tmp_path: Path) -> None:
    (tmp_path / ".simple-mem.json").write_text("{ not json", encoding="utf-8")
    assert resolve_project_id(tmp_path, env={}) == f"dir://{tmp_path.name}"


# --------------------------------------------------------------------------
# Tier 2a -- the git remote
# --------------------------------------------------------------------------


def test_tier2a_ssh_and_https_checkouts_agree(tmp_path: Path) -> None:
    ssh_repo = make_repo(tmp_path / "ssh-repo", remote="git@GitHub.com:Acme/CoreEngine.git")
    https_repo = make_repo(
        tmp_path / "https-repo",
        remote="https://user:pass@github.com/acme/coreengine.git",
    )
    assert resolve_project_id(ssh_repo, env={}) == "git://github.com/acme/coreengine"
    assert resolve_project_id(https_repo, env={}) == "git://github.com/acme/coreengine"


def test_tier2a_uses_first_remote_when_origin_is_absent(tmp_path: Path) -> None:
    repo = make_repo(tmp_path / "repo")
    run_git(repo, "remote", "add", "upstream", "https://gitlab.com/Team/Proj.git")
    assert resolve_project_id(repo, env={}) == "git://gitlab.com/team/proj"


def test_tier2a_worktree_follows_commondir(tmp_path: Path) -> None:
    main = make_repo(tmp_path / "main", remote="git@github.com:Acme/Core.git")
    worktree = tmp_path / "worktree"
    run_git(main, "worktree", "add", "-q", str(worktree), "-b", "feature")

    assert (worktree / ".git").is_file()
    assert (worktree / ".git").read_text(encoding="utf-8").startswith("gitdir:")
    assert resolve_project_id(worktree, env={}) == "git://github.com/acme/core"


# --------------------------------------------------------------------------
# Tier 2b -- a local repository named by its root commit
# --------------------------------------------------------------------------


def test_tier2b_local_repo_uses_root_commit_sha(tmp_path: Path) -> None:
    repo = make_repo(tmp_path / "local")
    expected = run_git(repo, "rev-parse", "HEAD").stdout.strip().lower()
    assert resolve_project_id(repo, env={}) == f"git-root://{expected}"


def test_tier2b_works_without_a_git_binary(tmp_path: Path, monkeypatch) -> None:
    repo = make_repo(tmp_path / "local")
    expected = run_git(repo, "rev-parse", "HEAD").stdout.strip().lower()

    import simple_mem.project as project

    def no_git(*args, **kwargs):
        raise FileNotFoundError("git is not installed")

    monkeypatch.setattr(project.subprocess, "run", no_git)
    assert resolve_project_id(repo, env={}) == f"git-root://{expected}"


def test_tier2b_worktree_without_remote_still_names_the_commit(tmp_path: Path) -> None:
    main = make_repo(tmp_path / "main")
    expected = run_git(main, "rev-parse", "HEAD").stdout.strip().lower()
    worktree = tmp_path / "worktree"
    run_git(main, "worktree", "add", "-q", str(worktree), "-b", "feature")
    assert resolve_project_id(worktree, env={}) == f"git-root://{expected}"


# --------------------------------------------------------------------------
# Tier 3 -- the directory names itself
# --------------------------------------------------------------------------


def test_tier3_directory_fallback(tmp_path: Path) -> None:
    plain = tmp_path / "some-project"
    plain.mkdir()
    assert resolve_project_id(plain, env={}) == "dir://some-project"
    assert resolve_project_id(tmp_path / "missing", env={}) == "dir://missing"


# --------------------------------------------------------------------------
# Store: schema and ingestion
# --------------------------------------------------------------------------


def test_schema_creates_the_expected_objects(tmp_path: Path) -> None:
    store = SimpleMemStore(tmp_path / "db" / "nested" / "mem.db")
    assert store.db_path.parent.is_dir()
    with store._connection() as connection:
        names = {row["name"] for row in connection.execute("SELECT name FROM sqlite_master")}
        journal = connection.execute("PRAGMA journal_mode").fetchone()[0]
    for expected in (
        "evidence_turns",
        "evidence_fts",
        "memories",
        "memories_fts",
        "sync_state",
    ):
        assert expected in names
    assert journal.lower() == "wal"


def test_ingest_turns_is_idempotent(store: SimpleMemStore) -> None:
    turns = [
        {
            "source_id": "claude-code",
            "conversation_id": "conv-1",
            "message_id": "msg-1",
            "role": "user",
            "content": "hello there",
            "project_id": "proj://alpha",
        }
    ]
    assert store.ingest_turns(turns) == {"accepted": 1, "inserted": 1, "duplicates": 0}
    assert store.ingest_turns(turns) == {"accepted": 1, "inserted": 0, "duplicates": 1}

    pending = store.fetch_unprocessed_turns()
    assert len(pending) == 1
    assert pending[0]["project_id"] == "proj://alpha"
    assert pending[0]["content"] == "hello there"

    assert store.mark_turns_processed([pending[0]["evidence_id"]]) == 1
    assert store.fetch_unprocessed_turns() == []


def test_ingest_skips_turns_without_identity(store: SimpleMemStore) -> None:
    result = store.ingest_turns(
        [
            {"conversation_id": "c", "content": "no message id"},
            {"message_id": "m", "content": "no conversation id"},
            {"conversation_id": "c", "message_id": "m", "verbatim": "kept"},
        ]
    )
    assert result == {"accepted": 1, "inserted": 1, "duplicates": 0}


def test_ingest_honours_defaults(store: SimpleMemStore) -> None:
    store.ingest_turns(
        [{"conversation_id": "c", "message_id": "m", "text": "body"}],
        machine_id="machine-7",
        default_source_id="opencode",
        default_project_id="proj://fallback",
    )
    turn = store.fetch_unprocessed_turns()[0]
    assert turn["source_id"] == "opencode"
    assert turn["machine_id"] == "machine-7"
    assert turn["role"] == "user"
    assert turn["content"] == "body"
    assert turn["project_id"] == "proj://fallback"


# --------------------------------------------------------------------------
# Store: lexical search
# --------------------------------------------------------------------------


def test_search_finds_a_cjk_substring(store: SimpleMemStore) -> None:
    memory = store.remember(
        "接口错误码统一用 problem+json", project_id="proj://alpha", search_keys=["错误码"]
    )
    result = store.search("错误码", project_id="proj://alpha")
    assert [hit["memory_id"] for hit in result["memories"]] == [memory["memory_id"]]
    assert result["project_id"] == "proj://alpha"


def test_search_survives_fts_punctuation(store: SimpleMemStore) -> None:
    memory = store.remember("错误码 统一约定", project_id="proj://alpha")
    for query in ['错误码: "quoted" (((', "AND OR NOT", "100% sure_thing"]:
        result = store.search(query, project_id="proj://alpha")
        assert isinstance(result["memories"], list)
    result = store.search("错误码", project_id="proj://alpha")
    assert memory["memory_id"] in [hit["memory_id"] for hit in result["memories"]]


def test_search_matches_ascii_terms(store: SimpleMemStore) -> None:
    memory = store.remember(
        "Use problem+json for error codes",
        project_id="proj://alpha",
        search_keys=["problem", "json"],
    )
    result = store.search("problem", project_id="proj://alpha")
    assert memory["memory_id"] in [hit["memory_id"] for hit in result["memories"]]


# --------------------------------------------------------------------------
# Store: hybrid ranking
# --------------------------------------------------------------------------


def test_hybrid_search_fuses_lexical_and_vector(store: SimpleMemStore) -> None:
    query_vector = [1.0, 0.0, 0.0, 0.0]
    lexical_only = store.remember(
        "alpha unique token", project_id="proj://alpha", embedding=[-1.0, 0.0, 0.0, 0.0]
    )
    vector_only = store.remember(
        "nothing relevant here",
        project_id="proj://alpha",
        embedding=[0.9, 0.1, 0.0, 0.0],
    )
    both = store.remember(
        "alpha again", project_id="proj://alpha", embedding=[1.0, 0.0, 0.0, 0.0]
    )

    result = store.search(
        "alpha", project_id="proj://alpha", query_embedding=query_vector
    )
    identifiers = [hit["memory_id"] for hit in result["memories"]]

    assert identifiers[0] == both["memory_id"]
    assert lexical_only["memory_id"] in identifiers
    assert vector_only["memory_id"] in identifiers
    assert all(0 < hit["score"] <= 1.0 for hit in result["memories"])


def test_vector_search_filters_below_threshold(store: SimpleMemStore) -> None:
    far = store.remember("unrelated", project_id="proj://alpha", embedding=[-1.0, 0.0])
    result = store.search("", project_id="proj://alpha", query_embedding=[1.0, 0.0])
    assert far["memory_id"] not in [hit["memory_id"] for hit in result["memories"]]


def test_remember_decodes_lists_and_hides_embedding(store: SimpleMemStore) -> None:
    memory = store.remember(
        "statement",
        project_id="proj://alpha",
        search_keys=["one", "two"],
        evidence_ids=["ev-1", "ev-2"],
        embedding=[0.1, 0.2, 0.3],
    )
    assert memory["search_keys"] == ["one", "two"]
    assert memory["evidence_ids"] == ["ev-1", "ev-2"]
    assert memory["has_embedding"] is True
    assert "embedding" not in memory

    plain = store.remember("plain", project_id="proj://alpha")
    assert plain["search_keys"] == []
    assert plain["evidence_ids"] == []
    assert plain["has_embedding"] is False


# --------------------------------------------------------------------------
# Store: scope isolation
# --------------------------------------------------------------------------


def test_workspace_scope_sees_itself_and_global_only(store: SimpleMemStore) -> None:
    mine = store.remember("错误码 用于本项目 alpha", project_id="proj://alpha")
    other = store.remember("错误码 用于另一个项目 beta", project_id="proj://beta")
    shared = store.remember("错误码 全局约定", project_id="global")

    store.ingest_turns(
        [
            {
                "source_id": "pi",
                "conversation_id": "c-alpha",
                "message_id": "m-alpha",
                "content": "错误码 raw alpha",
                "project_id": "proj://alpha",
            },
            {
                "source_id": "pi",
                "conversation_id": "c-beta",
                "message_id": "m-beta",
                "content": "错误码 raw beta",
                "project_id": "proj://beta",
            },
        ]
    )

    workspace = store.search("错误码", project_id="proj://alpha", scope="workspace")
    identifiers = {hit["memory_id"] for hit in workspace["memories"]}
    assert mine["memory_id"] in identifiers
    assert shared["memory_id"] in identifiers
    assert other["memory_id"] not in identifiers
    assert {turn["project_id"] for turn in workspace["raw_turns"]} == {"proj://alpha"}

    everything = store.search("错误码", project_id="proj://alpha", scope="all")
    all_identifiers = {hit["memory_id"] for hit in everything["memories"]}
    assert other["memory_id"] in all_identifiers
    assert {turn["project_id"] for turn in everything["raw_turns"]} == {
        "proj://alpha",
        "proj://beta",
    }

    without_raw = store.search("错误码", project_id="proj://alpha", include_raw=False)
    assert without_raw["raw_turns"] == []


def test_search_without_a_project_spans_everything(store: SimpleMemStore) -> None:
    alpha = store.remember("错误码 alpha", project_id="proj://alpha")
    beta = store.remember("错误码 beta", project_id="proj://beta")
    result = store.search("错误码")
    identifiers = {hit["memory_id"] for hit in result["memories"]}
    assert {alpha["memory_id"], beta["memory_id"]} <= identifiers
    assert result["project_id"] == "global"


# --------------------------------------------------------------------------
# Store: briefing and threads
# --------------------------------------------------------------------------


def test_briefing_groups_memories_by_kind(store: SimpleMemStore) -> None:
    preference = store.remember("prefer tabs", project_id="proj://alpha", kind="preference")
    decision = store.remember("use sqlite", project_id="proj://alpha", kind="decision")
    experience = store.remember("debugged the parser", project_id="global", kind="experience")
    fact = store.remember("the hub is fastapi", project_id="global", kind="fact")

    result = store.briefing("proj://alpha")

    assert result["project_id"] == "proj://alpha"
    assert [memory["memory_id"] for memory in result["preferences"]] == [
        preference["memory_id"]
    ]
    assert [memory["memory_id"] for memory in result["decisions"]] == [
        decision["memory_id"]
    ]
    assert [memory["memory_id"] for memory in result["experiences"]] == [
        experience["memory_id"]
    ]
    assert [memory["memory_id"] for memory in result["facts"]] == [fact["memory_id"]]
    assert {memory["memory_id"] for memory in result["memories"]} == {
        preference["memory_id"],
        decision["memory_id"],
        experience["memory_id"],
        fact["memory_id"],
    }


def test_briefing_scopes_to_workspace_and_global_only(store: SimpleMemStore) -> None:
    mine = store.remember("mine", project_id="proj://alpha", kind="fact")
    shared = store.remember("shared", project_id="global", kind="fact")
    other = store.remember("other", project_id="proj://beta", kind="fact")

    workspace = store.briefing("proj://alpha")
    identifiers = [memory["memory_id"] for memory in workspace["memories"]]
    assert identifiers == [mine["memory_id"], shared["memory_id"]]
    assert other["memory_id"] not in identifiers

    ambient = store.briefing(None)
    assert ambient["project_id"] == "global"
    assert [memory["memory_id"] for memory in ambient["memories"]] == [shared["memory_id"]]


def test_briefing_honours_the_limit(store: SimpleMemStore) -> None:
    for index in range(5):
        store.remember(f"fact {index}", project_id="proj://alpha", kind="fact")
    result = store.briefing("proj://alpha", limit=2)
    assert len(result["memories"]) == 2
    assert sum(
        len(result[key]) for key in ("preferences", "decisions", "experiences", "facts")
    ) == 2


def test_briefing_reports_recent_threads(store: SimpleMemStore) -> None:
    store.ingest_turns(
        [
            {
                "source_id": "pi",
                "conversation_id": "conv-a",
                "message_id": "a1",
                "role": "user",
                "content": "first question",
                "project_id": "proj://alpha",
                "created_at": "2026-01-01T00:00:00+00:00",
            },
            {
                "source_id": "pi",
                "conversation_id": "conv-a",
                "message_id": "a2",
                "role": "assistant",
                "content": "the assistant answer",
                "project_id": "proj://alpha",
                "created_at": "2026-01-02T00:00:00+00:00",
            },
            {
                "source_id": "claude-code",
                "conversation_id": "conv-b",
                "message_id": "b1",
                "role": "user",
                "content": "other question",
                "project_id": "proj://beta",
                "created_at": "2026-01-03T00:00:00+00:00",
            },
        ]
    )

    workspace = store.briefing("proj://alpha")
    assert [thread["conversation_id"] for thread in workspace["recent_threads"]] == [
        "conv-a"
    ]
    thread = workspace["recent_threads"][0]
    assert thread["project_id"] == "proj://alpha"
    assert thread["source_id"] == "pi"
    assert thread["turn_count"] == 2
    assert thread["last_active_at"] == "2026-01-02T00:00:00+00:00"
    assert thread["preview"] == "the assistant answer"

    everything = store.briefing(None)
    assert [thread["conversation_id"] for thread in everything["recent_threads"]] == [
        "conv-b",
        "conv-a",
    ]


def test_fetch_thread_orders_and_windows(store: SimpleMemStore) -> None:
    store.ingest_turns(
        [
            {
                "source_id": "pi",
                "conversation_id": "conv-1",
                "message_id": f"m{index}",
                "role": "user" if index % 2 else "assistant",
                "content": f"turn {index}",
                "project_id": "proj://alpha",
                "created_at": f"2026-01-0{index}T00:00:00+00:00",
            }
            for index in range(1, 5)
        ]
    )

    whole = store.fetch_thread("conv-1")
    assert whole["conversation_id"] == "conv-1"
    assert whole["project_id"] == "proj://alpha"
    assert whole["source_id"] == "pi"
    assert whole["total_turns"] == 4
    assert [turn["message_id"] for turn in whole["turns"]] == ["m1", "m2", "m3", "m4"]
    assert set(whole["turns"][0]) == {
        "evidence_id",
        "project_id",
        "source_id",
        "machine_id",
        "conversation_id",
        "message_id",
        "role",
        "content",
        "created_at",
    }

    tail = store.fetch_thread("conv-1", limit=2)
    assert [turn["message_id"] for turn in tail["turns"]] == ["m3", "m4"]

    centered = store.fetch_thread("conv-1", limit=3, around_message_id="m3")
    assert [turn["message_id"] for turn in centered["turns"]] == ["m2", "m3", "m4"]

    by_evidence = store.fetch_thread(
        "conv-1", limit=1, around_message_id=whole["turns"][0]["evidence_id"]
    )
    assert [turn["message_id"] for turn in by_evidence["turns"]] == ["m1"]


def test_fetch_thread_handles_an_unknown_conversation(store: SimpleMemStore) -> None:
    result = store.fetch_thread("missing")
    assert result == {
        "conversation_id": "missing",
        "project_id": "global",
        "source_id": "",
        "machine_id": "",
        "total_turns": 0,
        "turns": [],
    }


# --------------------------------------------------------------------------
# Store: lifecycle
# --------------------------------------------------------------------------


def test_supersede_replaces_without_deleting_history(store: SimpleMemStore) -> None:
    old = store.remember("错误码 用 code", project_id="proj://alpha")
    new = store.supersede_memory(
        old["memory_id"],
        {"statement": "错误码 改用 problem+json", "project_id": "proj://alpha"},
    )

    superseded = store.get_memory(old["memory_id"])
    assert superseded is not None
    assert superseded["status"] == "superseded"
    assert superseded["superseded_by"] == new["memory_id"]

    active = {memory["memory_id"] for memory in store.list_memories(project_id="proj://alpha")}
    assert new["memory_id"] in active
    assert old["memory_id"] not in active

    result = store.search("错误码", project_id="proj://alpha")
    identifiers = {hit["memory_id"] for hit in result["memories"]}
    assert new["memory_id"] in identifiers
    assert old["memory_id"] not in identifiers


def test_forget_removes_the_memory(store: SimpleMemStore) -> None:
    memory = store.remember("temporary", project_id="proj://alpha")
    assert store.forget(memory["memory_id"]) is True
    assert store.get_memory(memory["memory_id"]) is None
    assert store.forget(memory["memory_id"]) is False
    assert store.search("temporary", project_id="proj://alpha")["memories"] == []


def test_update_memory_refreshes_search(store: SimpleMemStore) -> None:
    memory = store.remember("old wording", project_id="proj://alpha")
    updated = store.update_memory(
        memory["memory_id"], statement="new wording", search_keys=["fresh"]
    )
    assert updated is not None
    assert updated["statement"] == "new wording"
    assert updated["search_keys"] == ["fresh"]
    assert store.update_memory("missing", statement="x") is None

    result = store.search("wording", project_id="proj://alpha")
    assert memory["memory_id"] in [hit["memory_id"] for hit in result["memories"]]


def test_stats_reports_a_usable_summary(store: SimpleMemStore) -> None:
    store.ingest_turns(
        [
            {
                "source_id": "pi",
                "conversation_id": "c",
                "message_id": "m",
                "content": "raw",
                "project_id": "proj://alpha",
            }
        ]
    )
    store.remember("kept", project_id="proj://alpha")
    removed = store.remember("removed", project_id="proj://beta")
    store.forget(removed["memory_id"])

    stats = store.stats()
    assert stats["evidence_turns"] == 1
    assert stats["unprocessed_turns"] == 1
    assert stats["memories"] == 1
    assert stats["active_memories"] == 1
    assert stats["projects"] == ["proj://alpha"]


# --------------------------------------------------------------------------
# package surface
# --------------------------------------------------------------------------


def test_package_exports() -> None:
    assert __version__ == "0.2.0"
    assert callable(resolve_project_id)
    assert callable(canonicalize_git_url)
    assert callable(SimpleMemStore)
