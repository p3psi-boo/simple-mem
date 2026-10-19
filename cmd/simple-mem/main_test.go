package main

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
)

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

type hubCall struct {
	method  string
	path    string
	query   url.Values
	headers http.Header
	body    map[string]any
}

func mockHub(t *testing.T, status int, response string) (*httptest.Server, *[]hubCall) {
	t.Helper()
	var mu sync.Mutex
	calls := &[]hubCall{}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body := map[string]any{}
		if r.Body != nil {
			_ = json.NewDecoder(r.Body).Decode(&body)
		}
		mu.Lock()
		*calls = append(*calls, hubCall{method: r.Method, path: r.URL.Path, query: r.URL.Query(), headers: r.Header.Clone(), body: body})
		mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(status)
		_, _ = w.Write([]byte(response))
	}))
	t.Cleanup(server.Close)
	return server, calls
}

func projectDir(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, ".simple-mem.json"), []byte(`{"project_id":"demo"}`), 0o644); err != nil {
		t.Fatalf("marker: %v", err)
	}
	return dir
}

func run(t *testing.T, stdin string, args ...string) (int, string, string) {
	t.Helper()
	var stdout, stderr bytes.Buffer
	code := Run(context.Background(), args, strings.NewReader(stdin), &stdout, &stderr)
	return code, stdout.String(), stderr.String()
}

// ---------------------------------------------------------------------------
// project-id
// ---------------------------------------------------------------------------

func TestProjectIDSubcommand(t *testing.T) {
	dir := projectDir(t)
	code, stdout, stderr := run(t, "", "project-id", "--cwd", dir, "--json")
	if code != 0 {
		t.Fatalf("code = %d, stderr = %s", code, stderr)
	}
	var payload map[string]any
	if err := json.Unmarshal([]byte(stdout), &payload); err != nil {
		t.Fatalf("stdout = %q: %v", stdout, err)
	}
	if payload["project_id"] != "proj://demo" {
		t.Fatalf("project_id = %#v", payload["project_id"])
	}

	code, stdout, _ = run(t, "", "project-id", "--cwd", dir)
	if code != 0 || strings.TrimSpace(stdout) != "proj://demo" {
		t.Fatalf("plain output = %q (code %d)", stdout, code)
	}
}

func TestUnknownCommandPrintsUsage(t *testing.T) {
	code, _, stderr := run(t, "", "nonsense")
	if code != 2 {
		t.Fatalf("code = %d", code)
	}
	if !strings.Contains(stderr, "Usage:") {
		t.Fatalf("stderr = %q", stderr)
	}
}

// ---------------------------------------------------------------------------
// ship
// ---------------------------------------------------------------------------

func writeTranscript(t *testing.T, dir string) string {
	t.Helper()
	path := filepath.Join(dir, "pi.jsonl")
	content := strings.Join([]string{
		`{"type":"session","id":"s1","cwd":"` + dir + `"}`,
		`{"type":"message","id":"m1","timestamp":"t1","message":{"role":"user","content":"hello"}}`,
		`{"type":"message","id":"m2","message":{"role":"assistant","content":"hi there"}}`,
	}, "\n") + "\n"
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatalf("transcript: %v", err)
	}
	return path
}

func TestShipTranscriptCommand(t *testing.T) {
	server, calls := mockHub(t, http.StatusOK, `{"accepted":2}`)
	dir := projectDir(t)
	path := writeTranscript(t, dir)
	cursor := filepath.Join(dir, "cursor.json")

	code, stdout, stderr := run(t, "", "ship", "--source", "pi", "--transcript", path,
		"--remote-url", server.URL, "--api-key", "secret", "--machine-id", "m1", "--cursor-file", cursor, "--json")
	if code != 0 {
		t.Fatalf("code = %d, stderr = %s", code, stderr)
	}
	var result map[string]any
	if err := json.Unmarshal([]byte(stdout), &result); err != nil {
		t.Fatalf("stdout = %q: %v", stdout, err)
	}
	if result["shipped"].(float64) != 2 {
		t.Fatalf("result = %#v", result)
	}
	if len(*calls) != 1 || (*calls)[0].path != "/api/v1/ingest" {
		t.Fatalf("calls = %#v", *calls)
	}
	if (*calls)[0].headers.Get("Authorization") != "Bearer secret" {
		t.Fatalf("auth header missing: %#v", (*calls)[0].headers)
	}
	turns := (*calls)[0].body["turns"].([]any)
	if len(turns) != 2 {
		t.Fatalf("turns = %#v", turns)
	}
	first := turns[0].(map[string]any)
	if first["content"] != "hello" || first["project_id"] != "proj://demo" {
		t.Fatalf("first turn = %#v", first)
	}

	// A second pass ships nothing new.
	code, stdout, _ = run(t, "", "ship", "--source", "pi", "--transcript", path,
		"--remote-url", server.URL, "--cursor-file", cursor, "--json")
	if code != 0 {
		t.Fatalf("second code = %d", code)
	}
	if err := json.Unmarshal([]byte(stdout), &result); err != nil {
		t.Fatalf("second stdout = %q", stdout)
	}
	if result["shipped"].(float64) != 0 || result["skipped"].(float64) != 2 {
		t.Fatalf("second result = %#v", result)
	}
	if len(*calls) != 1 {
		t.Fatalf("idempotent ship must not post again, got %d calls", len(*calls))
	}
}

func TestShipRejectsUnknownSource(t *testing.T) {
	code, _, stderr := run(t, "", "ship", "--source", "bogus", "--remote-url", "http://x")
	if code != 2 || !strings.Contains(stderr, "unsupported source") {
		t.Fatalf("code = %d, stderr = %q", code, stderr)
	}
}

// ---------------------------------------------------------------------------
// capture
// ---------------------------------------------------------------------------

func TestCaptureCommandReadsStdin(t *testing.T) {
	server, calls := mockHub(t, http.StatusAccepted, `{"accepted":1}`)
	dir := projectDir(t)
	payload := `{"hook_event_name":"UserPromptSubmit","session_id":"s","cwd":"` + dir + `","prompt":"do the thing"}`
	code, stdout, stderr := run(t, payload, "capture", "claude-code", "--remote-url", server.URL, "--api-key", "k", "--machine-id", "m1")
	if code != 0 {
		t.Fatalf("code = %d, stderr = %s", code, stderr)
	}
	var result map[string]any
	if err := json.Unmarshal([]byte(stdout), &result); err != nil {
		t.Fatalf("stdout = %q: %v", stdout, err)
	}
	if result["mode"] != "remote" || result["shipped"].(float64) != 1 {
		t.Fatalf("result = %#v", result)
	}
	turn := (*calls)[0].body["turns"].([]any)[0].(map[string]any)
	if turn["content"] != "do the thing" || turn["role"] != "user" {
		t.Fatalf("turn = %#v", turn)
	}
	if turn["project_id"] != "proj://demo" {
		t.Fatalf("project_id = %#v", turn["project_id"])
	}
}

func TestCaptureIgnoresUnreachableHubUnlessStrict(t *testing.T) {
	server, _ := mockHub(t, http.StatusInternalServerError, `{"error":"boom"}`)
	dir := projectDir(t)
	payload := `{"hook_event_name":"Stop","session_id":"s","cwd":"` + dir + `","response":"done"}`

	code, _, _ := run(t, payload, "capture", "pi", "--remote-url", server.URL)
	if code != 0 {
		t.Fatalf("a hook must never block the agent, code = %d", code)
	}
	code, _, stderr := run(t, payload, "capture", "pi", "--remote-url", server.URL, "--strict")
	if code != 1 {
		t.Fatalf("strict capture should fail, code = %d", code)
	}
	if !strings.Contains(stderr, "boom") {
		t.Fatalf("stderr = %q", stderr)
	}
}

func TestCaptureRequiresKnownHarness(t *testing.T) {
	code, _, stderr := run(t, "{}", "capture", "bogus")
	if code != 2 || !strings.Contains(stderr, "unsupported harness") {
		t.Fatalf("code = %d, stderr = %q", code, stderr)
	}
}

// ---------------------------------------------------------------------------
// search / remember / forget
// ---------------------------------------------------------------------------

func TestSearchCommand(t *testing.T) {
	server, calls := mockHub(t, http.StatusOK, `{"query":"SQLite","memories":[{"statement":"uses FTS5"}]}`)
	dir := projectDir(t)
	code, stdout, stderr := run(t, "", "search", "SQLite", "--remote-url", server.URL, "--cwd", dir, "--top-k", "3", "--json")
	if code != 0 {
		t.Fatalf("code = %d, stderr = %s", code, stderr)
	}
	if !strings.Contains(stdout, "uses FTS5") {
		t.Fatalf("stdout = %q", stdout)
	}
	call := (*calls)[0]
	if call.path != "/api/v1/search" || call.body["project_id"] != "proj://demo" || call.body["limit"].(float64) != 3 {
		t.Fatalf("call = %#v", call)
	}
}

func TestRememberCommand(t *testing.T) {
	server, calls := mockHub(t, http.StatusOK, `{"memory_id":"mem_1"}`)
	dir := projectDir(t)
	code, stdout, stderr := run(t, "", "remember", "Use WAL mode", "--remote-url", server.URL, "--cwd", dir, "--kind", "decision", "--json")
	if code != 0 {
		t.Fatalf("code = %d, stderr = %s", code, stderr)
	}
	if !strings.Contains(stdout, "mem_1") {
		t.Fatalf("stdout = %q", stdout)
	}
	call := (*calls)[0]
	if call.body["statement"] != "Use WAL mode" || call.body["kind"] != "decision" || call.body["project_id"] != "proj://demo" {
		t.Fatalf("call = %#v", call)
	}
}

func TestForgetCommand(t *testing.T) {
	server, calls := mockHub(t, http.StatusOK, `{"memory_id":"mem_2","deleted":true}`)
	code, stdout, stderr := run(t, "", "forget", "mem_2", "--remote-url", server.URL, "--json")
	if code != 0 {
		t.Fatalf("code = %d, stderr = %s", code, stderr)
	}
	if !strings.Contains(stdout, "true") {
		t.Fatalf("stdout = %q", stdout)
	}
	call := (*calls)[0]
	if call.method != http.MethodDelete || call.path != "/api/v1/memories/mem_2" {
		t.Fatalf("call = %#v", call)
	}
}

// ---------------------------------------------------------------------------
// briefing / thread
// ---------------------------------------------------------------------------

func TestBriefingCommandResolvesProject(t *testing.T) {
	server, calls := mockHub(t, http.StatusOK, `{"project_id":"proj://demo","memories":[{"statement":"uses FTS5"}]}`)
	dir := projectDir(t)
	code, stdout, stderr := run(t, "", "briefing", "--remote-url", server.URL, "--cwd", dir, "--limit", "5", "--json")
	if code != 0 {
		t.Fatalf("code = %d, stderr = %s", code, stderr)
	}
	if !strings.Contains(stdout, "uses FTS5") {
		t.Fatalf("stdout = %q", stdout)
	}
	call := (*calls)[0]
	if call.method != http.MethodGet || call.path != "/api/v1/briefing" {
		t.Fatalf("call = %#v", call)
	}
	if call.query.Get("project_id") != "proj://demo" || call.query.Get("limit") != "5" {
		t.Fatalf("query = %#v", call.query)
	}
}

func TestBriefingCommandPrintsTextSummary(t *testing.T) {
	server, _ := mockHub(t, http.StatusOK, `{"project_id":"proj://demo","briefing":"Standing facts for demo"}`)
	dir := projectDir(t)
	code, stdout, stderr := run(t, "", "briefing", "--remote-url", server.URL, "--cwd", dir)
	if code != 0 {
		t.Fatalf("code = %d, stderr = %s", code, stderr)
	}
	if !strings.Contains(stdout, "Standing facts for demo") {
		t.Fatalf("stdout = %q", stdout)
	}
}

func TestThreadCommandPrintsMessages(t *testing.T) {
	server, calls := mockHub(t, http.StatusOK, `{"conversation_id":"conv_1","messages":[{"role":"user","content":"hello"},{"role":"assistant","content":"hi there"}]}`)
	code, stdout, stderr := run(t, "", "thread", "conv_1", "--remote-url", server.URL, "--around", "m2", "--limit", "4")
	if code != 0 {
		t.Fatalf("code = %d, stderr = %s", code, stderr)
	}
	if !strings.Contains(stdout, "[user] hello") || !strings.Contains(stdout, "[assistant] hi there") {
		t.Fatalf("stdout = %q", stdout)
	}
	call := (*calls)[0]
	if call.method != http.MethodGet || call.path != "/api/v1/threads/conv_1" {
		t.Fatalf("call = %#v", call)
	}
	if call.query.Get("around_message_id") != "m2" || call.query.Get("limit") != "4" {
		t.Fatalf("query = %#v", call.query)
	}
}

func TestThreadCommandJSON(t *testing.T) {
	server, _ := mockHub(t, http.StatusOK, `{"conversation_id":"conv_1","messages":[]}`)
	code, stdout, stderr := run(t, "", "thread", "conv_1", "--remote-url", server.URL, "--json")
	if code != 0 {
		t.Fatalf("code = %d, stderr = %s", code, stderr)
	}
	var result map[string]any
	if err := json.Unmarshal([]byte(stdout), &result); err != nil {
		t.Fatalf("stdout = %q: %v", stdout, err)
	}
	if result["conversation_id"] != "conv_1" {
		t.Fatalf("result = %#v", result)
	}
}

func TestThreadCommandRequiresConversationID(t *testing.T) {
	code, _, stderr := run(t, "", "thread")
	if code != 2 || !strings.Contains(stderr, "conversation id") {
		t.Fatalf("code = %d, stderr = %q", code, stderr)
	}
}

// ---------------------------------------------------------------------------
// env defaults, mcp, edge-config
// ---------------------------------------------------------------------------

func TestRemoteURLFallsBackToEnvironment(t *testing.T) {
	server, calls := mockHub(t, http.StatusOK, `{"memories":[]}`)
	t.Setenv("SIMPLE_MEM_REMOTE_URL", server.URL)
	t.Setenv("SIMPLE_MEM_API_KEY", "envkey")
	code, _, stderr := run(t, "", "search", "anything", "--project-id", "proj://demo", "--json")
	if code != 0 {
		t.Fatalf("code = %d, stderr = %s", code, stderr)
	}
	if (*calls)[0].headers.Get("X-API-Key") != "envkey" {
		t.Fatalf("api key from env not used: %#v", (*calls)[0].headers)
	}
}

func TestMCPCommandAnswersInitialize(t *testing.T) {
	code, stdout, stderr := run(t, `{"jsonrpc":"2.0","id":1,"method":"initialize"}`+"\n", "mcp", "--remote-url", "http://127.0.0.1:1")
	if code != 0 {
		t.Fatalf("code = %d, stderr = %s", code, stderr)
	}
	var response map[string]any
	if err := json.Unmarshal([]byte(strings.TrimSpace(stdout)), &response); err != nil {
		t.Fatalf("stdout = %q: %v", stdout, err)
	}
	info := response["result"].(map[string]any)["serverInfo"].(map[string]any)
	if info["name"] != "simple-mem" {
		t.Fatalf("response = %#v", response)
	}
}

func TestEdgeConfigCommand(t *testing.T) {
	code, stdout, stderr := run(t, "", "edge-config", "--remote-url", "http://hub:8765", "--api-key", "k", "--machine-id", "m1", "--bin-path", "/bin/sm")
	if code != 0 {
		t.Fatalf("code = %d, stderr = %s", code, stderr)
	}
	var cfg map[string]any
	if err := json.Unmarshal([]byte(stdout), &cfg); err != nil {
		t.Fatalf("stdout = %q: %v", stdout, err)
	}
	server := cfg["mcpServers"].(map[string]any)["simple-mem"].(map[string]any)
	if server["command"] != "/bin/sm" {
		t.Fatalf("command = %#v", server["command"])
	}
	if cfg["http_mcp"].(map[string]any)["url"] != "http://hub:8765/mcp" {
		t.Fatalf("cfg = %#v", cfg)
	}
}

func TestInitEdgeAlias(t *testing.T) {
	code, stdout, _ := run(t, "", "init-edge", "--remote-url", "http://hub", "--machine-id", "m1")
	if code != 0 || !strings.Contains(stdout, "simple-mem") {
		t.Fatalf("code = %d, stdout = %q", code, stdout)
	}
}
