package mcp

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

type captured struct {
	method  string
	path    string
	query   url.Values
	headers http.Header
	body    map[string]any
}

func hub(t *testing.T, status int, response string) (*httptest.Server, *[]captured) {
	t.Helper()
	var mu sync.Mutex
	calls := &[]captured{}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body := map[string]any{}
		if r.Body != nil {
			_ = json.NewDecoder(r.Body).Decode(&body)
		}
		mu.Lock()
		*calls = append(*calls, captured{method: r.Method, path: r.URL.Path, query: r.URL.Query(), headers: r.Header.Clone(), body: body})
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

func toolText(t *testing.T, response map[string]any) map[string]any {
	t.Helper()
	result, ok := response["result"].(map[string]any)
	if !ok {
		t.Fatalf("missing result: %#v", response)
	}
	content, ok := result["content"].([]map[string]any)
	if !ok || len(content) == 0 {
		t.Fatalf("missing content: %#v", result)
	}
	var decoded map[string]any
	if err := json.Unmarshal([]byte(content[0]["text"].(string)), &decoded); err != nil {
		t.Fatalf("content is not JSON: %v", err)
	}
	return decoded
}

// ---------------------------------------------------------------------------
// handshake and listing
// ---------------------------------------------------------------------------

func TestInitialize(t *testing.T) {
	server := NewServer("http://hub", "", "", "")
	response, ok := server.Handle(map[string]any{"jsonrpc": "2.0", "id": float64(1), "method": "initialize"})
	if !ok {
		t.Fatal("initialize must produce a response")
	}
	result := response["result"].(map[string]any)
	if result["protocolVersion"] != "2024-11-05" {
		t.Fatalf("protocolVersion = %#v", result["protocolVersion"])
	}
	if result["serverInfo"].(map[string]any)["name"] != "simple-mem" {
		t.Fatalf("serverInfo = %#v", result["serverInfo"])
	}
	if result["serverInfo"].(map[string]any)["version"] != "0.2.0" {
		t.Fatalf("version = %#v", result["serverInfo"])
	}
}

func TestNotificationHasNoResponse(t *testing.T) {
	server := NewServer("http://hub", "", "", "")
	if _, ok := server.Handle(map[string]any{"jsonrpc": "2.0", "method": "notifications/initialized"}); ok {
		t.Fatal("notifications must not produce a response")
	}
}

func TestPing(t *testing.T) {
	server := NewServer("http://hub", "", "", "")
	response, ok := server.Handle(map[string]any{"jsonrpc": "2.0", "id": float64(2), "method": "ping"})
	if !ok {
		t.Fatal("ping must produce a response")
	}
	if len(response["result"].(map[string]any)) != 0 {
		t.Fatalf("ping result = %#v", response["result"])
	}
}

func TestToolsList(t *testing.T) {
	server := NewServer("http://hub", "", "", "")
	response, _ := server.Handle(map[string]any{"jsonrpc": "2.0", "id": float64(3), "method": "tools/list"})
	tools := response["result"].(map[string]any)["tools"].([]map[string]any)
	names := make([]string, 0, len(tools))
	for _, tool := range tools {
		names = append(names, tool["name"].(string))
	}
	want := []string{"simple_mem_briefing", "simple_mem_search", "simple_mem_thread", "simple_mem_remember", "simple_mem_forget"}
	if strings.Join(names, ",") != strings.Join(want, ",") {
		t.Fatalf("tools = %v", names)
	}

	briefing := tools[0]["inputSchema"].(map[string]any)
	briefingProperties := briefing["properties"].(map[string]any)
	if len(briefingProperties) != 2 {
		t.Fatalf("briefing properties = %#v", briefingProperties)
	}
	for _, key := range []string{"cwd", "limit"} {
		if _, ok := briefingProperties[key]; !ok {
			t.Fatalf("briefing is missing %q", key)
		}
	}
	if len(briefing["required"].([]string)) != 0 {
		t.Fatalf("briefing required = %#v", briefing["required"])
	}

	search := tools[1]["inputSchema"].(map[string]any)
	properties := search["properties"].(map[string]any)
	if len(properties) != 4 {
		t.Fatalf("search properties = %#v", properties)
	}
	for _, key := range []string{"query", "scope", "cwd", "limit"} {
		if _, ok := properties[key]; !ok {
			t.Fatalf("search is missing %q", key)
		}
	}
	required := search["required"].([]string)
	if len(required) != 1 || required[0] != "query" {
		t.Fatalf("search required = %v", required)
	}

	thread := tools[2]["inputSchema"].(map[string]any)
	threadProperties := thread["properties"].(map[string]any)
	if len(threadProperties) != 3 {
		t.Fatalf("thread properties = %#v", threadProperties)
	}
	for _, key := range []string{"conversation_id", "around_message_id", "limit"} {
		if _, ok := threadProperties[key]; !ok {
			t.Fatalf("thread is missing %q", key)
		}
	}
	if thread["required"].([]string)[0] != "conversation_id" {
		t.Fatalf("thread required = %#v", thread["required"])
	}

	forget := tools[4]["inputSchema"].(map[string]any)
	if forget["required"].([]string)[0] != "memory_id" {
		t.Fatalf("forget required = %#v", forget["required"])
	}
}

func TestUnknownMethodAndParseError(t *testing.T) {
	server := NewServer("http://hub", "", "", "")
	response, _ := server.Handle(map[string]any{"jsonrpc": "2.0", "id": float64(4), "method": "does/not/exist"})
	code := response["error"].(map[string]any)["code"].(int)
	if code != -32601 {
		t.Fatalf("error code = %v", code)
	}
	raw, ok := server.HandleRaw([]byte("{not json"))
	if !ok {
		t.Fatal("a parse error must be answered")
	}
	var parsed map[string]any
	if err := json.Unmarshal(raw, &parsed); err != nil {
		t.Fatalf("bad parse response: %v", err)
	}
	if parsed["error"].(map[string]any)["code"].(float64) != -32700 {
		t.Fatalf("parse error = %#v", parsed)
	}
}

func TestBatchDropsNotifications(t *testing.T) {
	server := NewServer("http://hub", "", "", "")
	raw, ok := server.HandleRaw([]byte(`[{"jsonrpc":"2.0","id":1,"method":"ping"},{"jsonrpc":"2.0","method":"notifications/initialized"}]`))
	if !ok {
		t.Fatal("batch with a request must be answered")
	}
	var responses []map[string]any
	if err := json.Unmarshal(raw, &responses); err != nil {
		t.Fatalf("batch response: %v", err)
	}
	if len(responses) != 1 {
		t.Fatalf("expected 1 response, got %d", len(responses))
	}
}

// ---------------------------------------------------------------------------
// tool calls
// ---------------------------------------------------------------------------

func TestToolsCallSearchResolvesProjectAndForwards(t *testing.T) {
	server, calls := hub(t, http.StatusOK, `{"memories":[{"statement":"x"}]}`)
	cwd := projectDir(t)
	instance := NewServer(server.URL, "secret", cwd, "")
	response, _ := instance.Handle(map[string]any{
		"jsonrpc": "2.0", "id": float64(5), "method": "tools/call",
		"params": map[string]any{"name": "simple_mem_search", "arguments": map[string]any{"query": "SQLite", "limit": float64(3)}},
	})
	if result := toolText(t, response); result["memories"] == nil {
		t.Fatalf("result = %#v", result)
	}
	if len(*calls) != 1 {
		t.Fatalf("calls = %#v", *calls)
	}
	call := (*calls)[0]
	if call.method != http.MethodPost || call.path != "/api/v1/search" {
		t.Fatalf("call = %#v", call)
	}
	if call.body["project_id"] != "proj://demo" {
		t.Fatalf("project_id = %#v", call.body["project_id"])
	}
	if call.body["query"] != "SQLite" || call.body["limit"].(float64) != 3 {
		t.Fatalf("body = %#v", call.body)
	}
	if call.headers.Get("Authorization") != "Bearer secret" || call.headers.Get("X-API-Key") != "secret" {
		t.Fatalf("headers = %#v", call.headers)
	}
}

func TestToolsCallRememberUsesGlobalScope(t *testing.T) {
	server, calls := hub(t, http.StatusOK, `{"memory_id":"mem_1"}`)
	instance := NewServer(server.URL, "", projectDir(t), "")
	response, _ := instance.Handle(map[string]any{
		"jsonrpc": "2.0", "id": float64(6), "method": "tools/call",
		"params": map[string]any{"name": "simple_mem_remember", "arguments": map[string]any{"statement": "Global fact", "scope": "global"}},
	})
	if toolText(t, response)["memory_id"] != "mem_1" {
		t.Fatalf("result = %#v", response)
	}
	call := (*calls)[0]
	if call.body["project_id"] != "global" || call.body["source_id"] != "mcp" {
		t.Fatalf("body = %#v", call.body)
	}
}

func TestToolsCallForget(t *testing.T) {
	server, calls := hub(t, http.StatusOK, `{"memory_id":"mem_9","deleted":true}`)
	instance := NewServer(server.URL, "", "", "")
	response, _ := instance.Handle(map[string]any{
		"jsonrpc": "2.0", "id": float64(7), "method": "tools/call",
		"params": map[string]any{"name": "simple_mem_forget", "arguments": map[string]any{"memory_id": "mem_9"}},
	})
	if toolText(t, response)["deleted"] != true {
		t.Fatalf("result = %#v", response)
	}
	call := (*calls)[0]
	if call.method != http.MethodDelete || call.path != "/api/v1/memories/mem_9" {
		t.Fatalf("call = %#v", call)
	}
}

func TestToolsCallBriefingResolvesProjectAndForwards(t *testing.T) {
	server, calls := hub(t, http.StatusOK, `{"project_id":"proj://demo","memories":[]}`)
	cwd := projectDir(t)
	instance := NewServer(server.URL, "secret", cwd, "")
	response, _ := instance.Handle(map[string]any{
		"jsonrpc": "2.0", "id": float64(10), "method": "tools/call",
		"params": map[string]any{"name": "simple_mem_briefing", "arguments": map[string]any{"limit": float64(7)}},
	})
	if toolText(t, response)["project_id"] != "proj://demo" {
		t.Fatalf("result = %#v", response)
	}
	if len(*calls) != 1 {
		t.Fatalf("calls = %#v", *calls)
	}
	call := (*calls)[0]
	if call.method != http.MethodGet || call.path != "/api/v1/briefing" {
		t.Fatalf("call = %#v", call)
	}
	if call.query.Get("project_id") != "proj://demo" || call.query.Get("limit") != "7" {
		t.Fatalf("query = %#v", call.query)
	}
}

func TestToolsCallThreadForwardsWindow(t *testing.T) {
	server, calls := hub(t, http.StatusOK, `{"conversation_id":"conv_1","messages":[]}`)
	instance := NewServer(server.URL, "", "", "")
	response, _ := instance.Handle(map[string]any{
		"jsonrpc": "2.0", "id": float64(11), "method": "tools/call",
		"params": map[string]any{"name": "simple_mem_thread", "arguments": map[string]any{
			"conversation_id": "conv_1", "around_message_id": "m2", "limit": float64(4),
		}},
	})
	if toolText(t, response)["conversation_id"] != "conv_1" {
		t.Fatalf("result = %#v", response)
	}
	if len(*calls) != 1 {
		t.Fatalf("calls = %#v", *calls)
	}
	call := (*calls)[0]
	if call.method != http.MethodGet || call.path != "/api/v1/threads/conv_1" {
		t.Fatalf("call = %#v", call)
	}
	if call.query.Get("around_message_id") != "m2" || call.query.Get("limit") != "4" {
		t.Fatalf("query = %#v", call.query)
	}
}

func TestToolsCallThreadRequiresConversationID(t *testing.T) {
	server, calls := hub(t, http.StatusOK, `{}`)
	instance := NewServer(server.URL, "", "", "")
	response, _ := instance.Handle(map[string]any{
		"jsonrpc": "2.0", "id": float64(12), "method": "tools/call",
		"params": map[string]any{"name": "simple_mem_thread", "arguments": map[string]any{}},
	})
	result := response["result"].(map[string]any)
	if result["isError"] != true {
		t.Fatalf("expected isError, got %#v", result)
	}
	if len(*calls) != 0 {
		t.Fatalf("a missing conversation_id must not reach the hub, calls = %#v", *calls)
	}
}

func TestToolsCallUnknownToolIsError(t *testing.T) {
	instance := NewServer("http://hub", "", "", "")
	response, _ := instance.Handle(map[string]any{
		"jsonrpc": "2.0", "id": float64(8), "method": "tools/call",
		"params": map[string]any{"name": "nope", "arguments": map[string]any{}},
	})
	result := response["result"].(map[string]any)
	if result["isError"] != true {
		t.Fatalf("expected isError, got %#v", result)
	}
}

func TestToolsCallReportsHubFailureAsErrorContent(t *testing.T) {
	server, _ := hub(t, http.StatusInternalServerError, `{"error":"boom"}`)
	instance := NewServer(server.URL, "", "", "")
	response, _ := instance.Handle(map[string]any{
		"jsonrpc": "2.0", "id": float64(9), "method": "tools/call",
		"params": map[string]any{"name": "simple_mem_search", "arguments": map[string]any{"query": "x"}},
	})
	result := response["result"].(map[string]any)
	if result["isError"] != true {
		t.Fatalf("expected isError, got %#v", result)
	}
}

// ---------------------------------------------------------------------------
// stdio
// ---------------------------------------------------------------------------

func TestServeStdioRoundTrip(t *testing.T) {
	server := NewServer("http://hub", "", "", "")
	input := strings.NewReader("{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"ping\"}\n\n{\"jsonrpc\":\"2.0\",\"method\":\"notifications/initialized\"}\n")
	var output bytes.Buffer
	if err := server.ServeStdio(context.Background(), input, &output); err != nil {
		t.Fatalf("serve: %v", err)
	}
	lines := strings.Split(strings.TrimSpace(output.String()), "\n")
	if len(lines) != 1 {
		t.Fatalf("expected exactly one response line, got %q", output.String())
	}
	var parsed map[string]any
	if err := json.Unmarshal([]byte(lines[0]), &parsed); err != nil {
		t.Fatalf("response: %v", err)
	}
	if parsed["id"].(float64) != 1 {
		t.Fatalf("id = %#v", parsed["id"])
	}
}

func TestForwardStdioRelaysToHub(t *testing.T) {
	var mu sync.Mutex
	headers := &http.Header{}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		*headers = r.Header.Clone()
		mu.Unlock()
		if r.URL.Path != "/mcp" {
			t.Errorf("path = %q", r.URL.Path)
		}
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"jsonrpc":"2.0","id":1,"result":{"serverInfo":{}}}`))
	}))
	defer server.Close()

	cwd := projectDir(t)
	input := strings.NewReader("{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"initialize\"}\n")
	var output bytes.Buffer
	if err := ForwardStdio(context.Background(), input, &output, server.URL, "secret", cwd); err != nil {
		t.Fatalf("forward: %v", err)
	}
	if !strings.Contains(output.String(), "serverInfo") {
		t.Fatalf("output = %q", output.String())
	}
	if headers.Get("Authorization") != "Bearer secret" {
		t.Fatalf("Authorization = %q", headers.Get("Authorization"))
	}
	if headers.Get("X-Simple-Mem-Cwd") != "proj://demo" {
		t.Fatalf("cwd header = %q", headers.Get("X-Simple-Mem-Cwd"))
	}
}

func TestForwardStdioDropsNoContent(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusNoContent)
	}))
	defer server.Close()
	input := strings.NewReader("{\"jsonrpc\":\"2.0\",\"method\":\"notifications/initialized\"}\n")
	var output bytes.Buffer
	if err := ForwardStdio(context.Background(), input, &output, server.URL, "", ""); err != nil {
		t.Fatalf("forward: %v", err)
	}
	if output.Len() != 0 {
		t.Fatalf("expected no output, got %q", output.String())
	}
}

// ---------------------------------------------------------------------------
// edge config
// ---------------------------------------------------------------------------

func TestGenerateEdgeConfig(t *testing.T) {
	cfg := GenerateEdgeConfig("/usr/local/bin/simple-mem", "http://hub:8765/", "key", "m1")
	server := cfg["mcpServers"].(map[string]any)["simple-mem"].(map[string]any)
	if server["command"] != "/usr/local/bin/simple-mem" {
		t.Fatalf("command = %#v", server["command"])
	}
	if args := server["args"].([]string); len(args) != 1 || args[0] != "mcp" {
		t.Fatalf("args = %#v", server["args"])
	}
	env := server["env"].(map[string]any)
	if env["SIMPLE_MEM_REMOTE_URL"] != "http://hub:8765/" || env["SIMPLE_MEM_API_KEY"] != "key" || env["SIMPLE_MEM_MACHINE_ID"] != "m1" {
		t.Fatalf("env = %#v", env)
	}
	httpMCP := cfg["http_mcp"].(map[string]any)
	if httpMCP["url"] != "http://hub:8765/mcp" {
		t.Fatalf("http_mcp = %#v", httpMCP)
	}
	if httpMCP["headers"].(map[string]any)["Authorization"] != "Bearer key" {
		t.Fatalf("headers = %#v", httpMCP["headers"])
	}
	hooks := cfg["hooks"].(map[string]any)
	if _, ok := hooks["SessionStart"]; !ok {
		t.Fatalf("hooks = %#v", hooks)
	}
	if _, ok := hooks["UserPromptSubmit"]; !ok {
		t.Fatalf("hooks = %#v", hooks)
	}
	if _, ok := hooks["Stop"]; !ok {
		t.Fatalf("hooks = %#v", hooks)
	}
}

func TestWriteEdgeConfigsMergesExisting(t *testing.T) {
	home := t.TempDir()
	droidPath := filepath.Join(home, ".factory", "mcp.json")
	if err := os.MkdirAll(filepath.Dir(droidPath), 0o755); err != nil {
		t.Fatalf("mkdir: %v", err)
	}
	if err := os.WriteFile(droidPath, []byte(`{"theme":"dark","mcpServers":{"other":{"command":"x"}}}`), 0o644); err != nil {
		t.Fatalf("seed: %v", err)
	}
	cfg := GenerateEdgeConfig("sm", "http://hub", "key", "m1")
	if err := WriteEdgeConfigs(cfg, true, true, home); err != nil {
		t.Fatalf("write: %v", err)
	}

	droidData, err := os.ReadFile(droidPath)
	if err != nil {
		t.Fatalf("read droid: %v", err)
	}
	var droid map[string]any
	if err := json.Unmarshal(droidData, &droid); err != nil {
		t.Fatalf("droid json: %v", err)
	}
	if droid["theme"] != "dark" {
		t.Fatalf("existing keys must survive, got %#v", droid)
	}
	servers := droid["mcpServers"].(map[string]any)
	if _, ok := servers["other"]; !ok {
		t.Fatalf("existing servers must survive: %#v", servers)
	}
	if _, ok := servers["simple-mem"]; !ok {
		t.Fatalf("simple-mem server missing: %#v", servers)
	}
	if _, ok := droid["hooks"]; !ok {
		t.Fatalf("hooks missing from merged config: %#v", droid)
	}

	claudePath := filepath.Join(home, ".claude", "settings.json")
	claudeData, err := os.ReadFile(claudePath)
	if err != nil {
		t.Fatalf("read claude: %v", err)
	}
	var claude map[string]any
	if err := json.Unmarshal(claudeData, &claude); err != nil {
		t.Fatalf("claude json: %v", err)
	}
	if _, ok := claude["mcpServers"].(map[string]any)["simple-mem"]; !ok {
		t.Fatalf("claude config = %#v", claude)
	}
}
