from __future__ import annotations

import json
import os
import socket
import subprocess
import time
from pathlib import Path

import httpx


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _ensure_go_client_binary(repo_root: Path) -> Path:
    bin_path = repo_root / "bin" / "simple-mem"
    if not bin_path.exists():
        subprocess.run(
            ["go", "build", "-ldflags=-s -w", "-o", str(bin_path), "./cmd/simple-mem"],
            cwd=str(repo_root),
            check=True,
        )
    return bin_path


def test_go_client_and_python_hub_cross_machine_e2e(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    hub_bin = str(repo_root / "bin" / "simple-mem-hub")
    client_bin = str(_ensure_go_client_binary(repo_root))

    port = _free_port()
    hub_url = f"http://127.0.0.1:{port}"
    api_key = "sm-go-py-e2e-secret"
    db_path = tmp_path / "hub" / "simple-mem.db"

    hub_proc = subprocess.Popen(
        [
            hub_bin,
            "serve",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--db",
            str(db_path),
            "--api-key",
            api_key,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        for _ in range(40):
            try:
                resp = httpx.get(f"{hub_url}/health", timeout=0.5)
                if resp.status_code == 200:
                    break
            except Exception:
                time.sleep(0.1)
        else:
            raise AssertionError("Python Hub failed to start")

        # 1. Machine A: Mac checkout with SSH remote in a deep subdirectory
        mac_repo = tmp_path / "mac" / "coreengine"
        (mac_repo / ".git").mkdir(parents=True)
        (mac_repo / ".git" / "config").write_text(
            '[remote "origin"]\n\turl = git@GitHub.com:AcmeCorp/CoreEngine.git\n',
            encoding="utf-8",
        )
        mac_subdir = mac_repo / "services" / "order-api"
        mac_subdir.mkdir(parents=True)

        # Verify Go client's `project-id` subcommand
        res_pid_mac = subprocess.run(
            [client_bin, "project-id", "--cwd", str(mac_subdir)],
            capture_output=True,
            text=True,
            check=True,
        )
        assert res_pid_mac.stdout.strip() == "git://github.com/acmecorp/coreengine"

        # Capture hook event from Machine A via Go `simple-mem capture`
        hook_env = {
            **os.environ,
            "SIMPLE_MEM_REMOTE_URL": hub_url,
            "SIMPLE_MEM_API_KEY": api_key,
            "SIMPLE_MEM_MACHINE_ID": "macbook-pro",
        }
        hook_payload = {
            "hook_event_name": "UserPromptSubmit",
            "session_id": "sess-mac-1",
            "cwd": str(mac_subdir),
            "prompt": "记住：订单服务统一使用 PostgreSQL 16，禁止使用 MySQL。",
        }
        res_capture = subprocess.run(
            [client_bin, "capture", "factory-droid"],
            input=json.dumps(hook_payload),
            env=hook_env,
            capture_output=True,
            text=True,
            check=True,
        )
        capture_out = json.loads(res_capture.stdout)
        assert capture_out["shipped"] == 1

        # 2. Machine B: Linux workstation with HTTPS remote + git worktree + pi transcript
        linux_main = tmp_path / "linux" / "coreengine-main"
        git_common = linux_main / ".git"
        (git_common / "worktrees" / "wt1").mkdir(parents=True)
        (git_common / "config").write_text(
            '[remote "origin"]\n\turl = https://github.com/acmecorp/coreengine.git\n',
            encoding="utf-8",
        )
        (git_common / "worktrees" / "wt1" / "commondir").write_text("../..\n", encoding="utf-8")
        linux_wt = tmp_path / "linux" / "coreengine-wt1"
        linux_wt.mkdir(parents=True)
        (linux_wt / ".git").write_text(
            f"gitdir: {git_common / 'worktrees' / 'wt1'}\n",
            encoding="utf-8",
        )

        res_pid_linux = subprocess.run(
            [client_bin, "project-id", "--cwd", str(linux_wt)],
            capture_output=True,
            text=True,
            check=True,
        )
        assert res_pid_linux.stdout.strip() == "git://github.com/acmecorp/coreengine"

        pi_jsonl = tmp_path / "sess-pi-1.jsonl"
        pi_jsonl.write_text(
            "\n".join(
                [
                    json.dumps({"type": "session", "id": "sess-pi-1", "cwd": str(linux_wt)}),
                    json.dumps(
                        {
                            "type": "message",
                            "id": "m-pi-1",
                            "timestamp": "2026-04-20T10:00:00Z",
                            "message": {
                                "role": "assistant",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": "踩坑：NixOS 下运行 numpy 需要在导入前通过 ctypes 预加载 libstdc++.so.6。",
                                    }
                                ],
                            },
                        }
                    ),
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        # Ship transcript from Machine B via Go `simple-mem ship`
        res_ship = subprocess.run(
            [
                client_bin,
                "ship",
                "--source",
                "pi",
                "--transcript",
                str(pi_jsonl),
                "--remote-url",
                hub_url,
                "--api-key",
                api_key,
                "--machine-id",
                "linux-workstation",
                "--cursor-file",
                str(tmp_path / "cursor.json"),
                "--json",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        assert json.loads(res_ship.stdout)["shipped"] == 1

        # 3. Trigger extraction on Python Hub via `simple-mem-hub extract`
        res_ext = subprocess.run(
            [hub_bin, "extract", "--remote-url", hub_url, "--api-key", api_key, "--json"],
            capture_output=True,
            text=True,
            check=True,
        )
        ext_stats = json.loads(res_ext.stdout)
        assert ext_stats["turns_processed"] == 2 and ext_stats["added"] >= 2

        # 4. Query & reconcile via Go `simple-mem mcp` stdio server from Machine B's git worktree
        mcp_requests = (
            "\n".join(
                [
                    json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}),
                    json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}),
                    json.dumps(
                        {
                            "jsonrpc": "2.0",
                            "id": 3,
                            "method": "tools/call",
                            "params": {
                                "name": "simple_mem_search",
                                "arguments": {"query": "PostgreSQL", "scope": "workspace"},
                            },
                        }
                    ),
                    json.dumps(
                        {
                            "jsonrpc": "2.0",
                            "id": 4,
                            "method": "tools/call",
                            "params": {
                                "name": "simple_mem_remember",
                                "arguments": {
                                    "statement": "订单服务统一使用 PostgreSQL 17，禁止使用 MySQL",
                                    "kind": "preference",
                                    "scope": "workspace",
                                },
                            },
                        }
                    ),
                    json.dumps(
                        {
                            "jsonrpc": "2.0",
                            "id": 5,
                            "method": "tools/call",
                            "params": {
                                "name": "simple_mem_search",
                                "arguments": {"query": "PostgreSQL", "scope": "workspace"},
                            },
                        }
                    ),
                    json.dumps(
                        {
                            "jsonrpc": "2.0",
                            "id": 6,
                            "method": "tools/call",
                            "params": {
                                "name": "simple_mem_briefing",
                                "arguments": {"limit": 10},
                            },
                        }
                    ),
                    json.dumps(
                        {
                            "jsonrpc": "2.0",
                            "id": 7,
                            "method": "tools/call",
                            "params": {
                                "name": "simple_mem_thread",
                                "arguments": {"conversation_id": "sess-pi-1", "limit": 10},
                            },
                        }
                    ),
                ]
            )
            + "\n"
        )
        res_mcp = subprocess.run(
            [client_bin, "mcp", "--remote-url", hub_url, "--api-key", api_key, "--cwd", str(linux_wt)],
            input=mcp_requests,
            capture_output=True,
            text=True,
            check=True,
        )
        responses = [json.loads(line) for line in res_mcp.stdout.splitlines() if line.strip()]
        assert len(responses) == 7

        tools = [t["name"] for t in responses[1]["result"]["tools"]]
        assert tools == [
            "simple_mem_briefing",
            "simple_mem_search",
            "simple_mem_thread",
            "simple_mem_remember",
            "simple_mem_forget",
        ]

        search_before = json.loads(responses[2]["result"]["content"][0]["text"])
        assert search_before["project_id"] == "git://github.com/acmecorp/coreengine"
        assert any("PostgreSQL 16" in m["statement"] for m in search_before["memories"])

        search_after = json.loads(responses[4]["result"]["content"][0]["text"])
        active_statements = [m["statement"] for m in search_after["memories"]]
        assert any("PostgreSQL 17" in s for s in active_statements)
        assert not any("PostgreSQL 16" in s for s in active_statements)

        briefing_out = json.loads(responses[5]["result"]["content"][0]["text"])
        assert briefing_out["project_id"] == "git://github.com/acmecorp/coreengine"
        assert any("PostgreSQL 17" in m["statement"] for m in briefing_out["memories"])
        assert any(t["conversation_id"] == "sess-pi-1" for t in briefing_out["recent_threads"])

        thread_out = json.loads(responses[6]["result"]["content"][0]["text"])
        assert thread_out["conversation_id"] == "sess-pi-1"
        assert thread_out["total_turns"] == 1
        assert "libstdc++.so.6" in thread_out["turns"][0]["content"]

        # 5. Also verify direct Streamable HTTP MCP endpoint (`POST /mcp`) on the central Hub
        http_mcp_resp = httpx.post(
            f"{hub_url}/mcp",
            json={
                "jsonrpc": "2.0",
                "id": 99,
                "method": "tools/call",
                "params": {
                    "name": "simple_mem_search",
                    "arguments": {
                        "query": "PostgreSQL",
                        "scope": "workspace",
                        "cwd": str(mac_subdir),
                    },
                },
            },
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=5.0,
        ).json()
        http_search = json.loads(http_mcp_resp["result"]["content"][0]["text"])
        assert http_search["project_id"] == "git://github.com/acmecorp/coreengine"
        assert any("PostgreSQL 17" in m["statement"] for m in http_search["memories"])
    finally:
        hub_proc.terminate()
        hub_proc.wait(timeout=5)
