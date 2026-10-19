// Package mcp is the edge's Model Context Protocol surface.
//
// An agent host speaks JSON-RPC 2.0 to us over stdio. We answer the handshake
// and tool listing natively, and translate each tool call into the Hub's REST
// surface after resolving the caller's directory into a project identity. The
// same three tools answer over Streamable HTTP at the Hub, so this package also
// carries a thin forwarder for hosts that prefer to talk HTTP directly.
package mcp

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"simple-mem/internal/project"
)

// ProtocolVersion and ServerInfo identify this server during the handshake.
const (
	ProtocolVersion = "2024-11-05"
	ServerName      = "simple-mem"
	ServerVersion   = "0.2.0"
)

// ServerInfo returns the server identity block.
func ServerInfo() map[string]any {
	return map[string]any{"name": ServerName, "version": ServerVersion}
}

// Tools returns the MCP tool definitions an agent is given.
func Tools() []map[string]any {
	return []map[string]any{
		{
			"name":        "simple_mem_briefing",
			"description": "Fetch the working memory briefing for the current project: standing facts, decisions and recent context to load before starting work.",
			"inputSchema": map[string]any{
				"type": "object",
				"properties": map[string]any{
					"cwd":   map[string]any{"type": "string"},
					"limit": map[string]any{"type": "integer", "default": 20},
				},
				"required": []string{},
			},
		},
		{
			"name":        "simple_mem_search",
			"description": "Search across governed developer memories (facts, decisions, preferences, coding experiences) and cross-machine raw conversation turns.",
			"inputSchema": map[string]any{
				"type": "object",
				"properties": map[string]any{
					"query": map[string]any{"type": "string"},
					"scope": map[string]any{"type": "string", "enum": []string{"workspace", "all"}, "default": "workspace"},
					"cwd":   map[string]any{"type": "string"},
					"limit": map[string]any{"type": "integer", "default": 10},
				},
				"required": []string{"query"},
			},
		},
		{
			"name":        "simple_mem_thread",
			"description": "Expand the full context of one conversation thread, optionally centred on a specific message.",
			"inputSchema": map[string]any{
				"type": "object",
				"properties": map[string]any{
					"conversation_id":   map[string]any{"type": "string"},
					"around_message_id": map[string]any{"type": "string"},
					"limit":             map[string]any{"type": "integer", "default": 20},
				},
				"required": []string{"conversation_id"},
			},
		},
		{
			"name":        "simple_mem_remember",
			"description": "Persist an important project fact, architectural decision, user preference, or coding experience.",
			"inputSchema": map[string]any{
				"type": "object",
				"properties": map[string]any{
					"statement":     map[string]any{"type": "string"},
					"kind":          map[string]any{"type": "string", "enum": []string{"fact", "decision", "preference", "experience"}, "default": "fact"},
					"details":       map[string]any{"type": "string", "default": ""},
					"search_keys":   map[string]any{"type": "array", "items": map[string]any{"type": "string"}, "default": []string{}},
					"boundary_envs": map[string]any{"type": "string", "default": ""},
					"scope":         map[string]any{"type": "string", "enum": []string{"workspace", "global"}, "default": "workspace"},
					"cwd":           map[string]any{"type": "string"},
				},
				"required": []string{"statement"},
			},
		},
		{
			"name":        "simple_mem_forget",
			"description": "Delete an outdated or incorrect memory by memory_id.",
			"inputSchema": map[string]any{
				"type": "object",
				"properties": map[string]any{
					"memory_id": map[string]any{"type": "string"},
				},
				"required": []string{"memory_id"},
			},
		},
	}
}

// Server answers MCP requests and forwards tool calls to the Hub.
type Server struct {
	RemoteURL string
	APIKey    string
	Cwd       string
	ProjectID string
	HTTP      *http.Client
}

// NewServer builds a server that resolves local directories to project ids and
// forwards tool calls to the Hub at remoteURL.
func NewServer(remoteURL, apiKey, cwd, projectID string) *Server {
	return &Server{
		RemoteURL: strings.TrimRight(remoteURL, "/"),
		APIKey:    apiKey,
		Cwd:       cwd,
		ProjectID: projectID,
		HTTP:      &http.Client{Timeout: 30 * time.Second},
	}
}

// Handle answers one JSON-RPC request. The bool is false for a notification,
// which takes no reply.
func (s *Server) Handle(payload map[string]any) (map[string]any, bool) {
	if payload == nil {
		return errorResponse(nil, -32600, "Invalid Request"), true
	}
	method, _ := payload["method"].(string)
	id, hasID := payload["id"]
	if strings.HasPrefix(method, "notifications/") {
		return nil, false
	}
	params, _ := payload["params"].(map[string]any)
	if params == nil {
		params = map[string]any{}
	}

	var result any
	switch method {
	case "initialize":
		result = map[string]any{
			"protocolVersion": ProtocolVersion,
			"capabilities":    map[string]any{"tools": map[string]any{}},
			"serverInfo":      ServerInfo(),
		}
	case "ping":
		result = map[string]any{}
	case "tools/list":
		result = map[string]any{"tools": Tools()}
	case "tools/call":
		result = s.callTool(params)
	default:
		if !hasID {
			return nil, false
		}
		return errorResponse(id, -32601, "Method not found"), true
	}
	return map[string]any{"jsonrpc": "2.0", "id": id, "result": result}, true
}

// HandleRaw answers one raw JSON-RPC frame, which may be a batch. The returned
// bytes are the frame to write back; the bool is false when nothing is owed.
func (s *Server) HandleRaw(raw []byte) ([]byte, bool) {
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 {
		return nil, false
	}
	if trimmed[0] == '[' {
		items, err := decodeBatch(trimmed)
		if err != nil {
			return mustMarshal(errorResponse(nil, -32700, "Parse error")), true
		}
		var responses []map[string]any
		for _, item := range items {
			if response, ok := s.Handle(item); ok {
				responses = append(responses, response)
			}
		}
		if len(responses) == 0 {
			return nil, false
		}
		return mustMarshal(responses), true
	}
	payload, err := decodeObject(trimmed)
	if err != nil {
		return mustMarshal(errorResponse(nil, -32700, "Parse error")), true
	}
	response, ok := s.Handle(payload)
	if !ok {
		return nil, false
	}
	return mustMarshal(response), true
}

// ServeStdio reads newline-delimited JSON-RPC requests and writes responses.
func (s *Server) ServeStdio(ctx context.Context, in io.Reader, out io.Writer) error {
	scanner := bufio.NewScanner(in)
	scanner.Buffer(make([]byte, 0, 64*1024), 16*1024*1024)
	writer := bufio.NewWriter(out)
	defer writer.Flush()
	for scanner.Scan() {
		if len(bytes.TrimSpace(scanner.Bytes())) == 0 {
			continue
		}
		response, ok := s.HandleRaw(scanner.Bytes())
		if !ok {
			continue
		}
		if _, err := writer.Write(response); err != nil {
			return err
		}
		if err := writer.WriteByte('\n'); err != nil {
			return err
		}
		if err := writer.Flush(); err != nil {
			return err
		}
	}
	return scanner.Err()
}

// --------------------------------------------------------------------------
// Tool dispatch
// --------------------------------------------------------------------------

func (s *Server) callTool(params map[string]any) map[string]any {
	name, _ := params["name"].(string)
	args, _ := params["arguments"].(map[string]any)
	if args == nil {
		args = map[string]any{}
	}
	switch name {
	case "simple_mem_briefing":
		return s.briefing(args)
	case "simple_mem_search":
		return s.search(args)
	case "simple_mem_thread":
		return s.thread(args)
	case "simple_mem_remember":
		return s.remember(args)
	case "simple_mem_forget":
		return s.forget(args)
	default:
		return contentResult(map[string]any{"error": "unknown tool: " + name}, true)
	}
}

func (s *Server) briefing(args map[string]any) map[string]any {
	query := url.Values{}
	query.Set("project_id", s.resolveProjectID(args))
	query.Set("limit", strconv.Itoa(intArg(args, "limit", 20)))
	response, err := s.do(context.Background(), http.MethodGet, "/api/v1/briefing?"+query.Encode(), nil)
	if err != nil {
		return contentResult(map[string]any{"error": err.Error()}, true)
	}
	return contentResult(response, false)
}

func (s *Server) thread(args map[string]any) map[string]any {
	conversationID := stringArg(args, "conversation_id", "")
	if conversationID == "" {
		return contentResult(map[string]any{"error": "conversation_id is required"}, true)
	}
	query := url.Values{}
	if around := stringArg(args, "around_message_id", ""); around != "" {
		query.Set("around_message_id", around)
	}
	query.Set("limit", strconv.Itoa(intArg(args, "limit", 20)))
	path := "/api/v1/threads/" + url.PathEscape(conversationID) + "?" + query.Encode()
	response, err := s.do(context.Background(), http.MethodGet, path, nil)
	if err != nil {
		return contentResult(map[string]any{"error": err.Error()}, true)
	}
	return contentResult(response, false)
}

func (s *Server) search(args map[string]any) map[string]any {
	body := map[string]any{
		"query":      stringArg(args, "query", ""),
		"scope":      stringArg(args, "scope", "workspace"),
		"limit":      intArg(args, "limit", 10),
		"project_id": s.resolveProjectID(args),
	}
	response, err := s.do(context.Background(), http.MethodPost, "/api/v1/search", body)
	if err != nil {
		return contentResult(map[string]any{"error": err.Error()}, true)
	}
	return contentResult(response, false)
}

func (s *Server) remember(args map[string]any) map[string]any {
	scope := stringArg(args, "scope", "workspace")
	projectID := "global"
	if scope != "global" {
		projectID = s.resolveProjectID(args)
	}
	body := map[string]any{
		"statement":     stringArg(args, "statement", ""),
		"kind":          stringArg(args, "kind", "fact"),
		"details":       stringArg(args, "details", ""),
		"search_keys":   stringSliceArg(args["search_keys"]),
		"boundary_envs": stringArg(args, "boundary_envs", ""),
		"scope":         scope,
		"project_id":    projectID,
		"source_id":     "mcp",
	}
	response, err := s.do(context.Background(), http.MethodPost, "/api/v1/remember", body)
	if err != nil {
		return contentResult(map[string]any{"error": err.Error()}, true)
	}
	return contentResult(response, false)
}

func (s *Server) forget(args map[string]any) map[string]any {
	memoryID := stringArg(args, "memory_id", "")
	response, err := s.do(context.Background(), http.MethodDelete, "/api/v1/memories/"+url.PathEscape(memoryID), nil)
	if err != nil {
		return contentResult(map[string]any{"error": err.Error()}, true)
	}
	return contentResult(response, false)
}

func (s *Server) resolveProjectID(args map[string]any) string {
	cwd := stringArg(args, "cwd", "")
	if cwd == "" {
		cwd = s.Cwd
	}
	if cwd == "" {
		if s.ProjectID != "" {
			return s.ProjectID
		}
		return project.ResolveProjectID("", "")
	}
	return project.ResolveProjectID(cwd, s.ProjectID)
}

func (s *Server) do(ctx context.Context, method, path string, body any) (map[string]any, error) {
	var reader io.Reader
	if body != nil {
		encoded, err := json.Marshal(body)
		if err != nil {
			return nil, err
		}
		reader = bytes.NewReader(encoded)
	}
	request, err := http.NewRequestWithContext(ctx, method, s.RemoteURL+path, reader)
	if err != nil {
		return nil, err
	}
	if body != nil {
		request.Header.Set("Content-Type", "application/json")
	}
	if s.APIKey != "" {
		request.Header.Set("Authorization", "Bearer "+s.APIKey)
		request.Header.Set("X-API-Key", s.APIKey)
	}
	response, err := s.httpClient().Do(request)
	if err != nil {
		return nil, err
	}
	defer response.Body.Close()
	payload, err := io.ReadAll(io.LimitReader(response.Body, 64<<20))
	if err != nil {
		return nil, err
	}
	if response.StatusCode < 200 || response.StatusCode >= 300 {
		return nil, fmt.Errorf("hub returned %s: %s", response.Status, strings.TrimSpace(string(payload)))
	}
	if len(bytes.TrimSpace(payload)) == 0 {
		return map[string]any{}, nil
	}
	var decoded map[string]any
	if err := json.Unmarshal(payload, &decoded); err != nil {
		return map[string]any{"result": string(payload)}, nil
	}
	return decoded, nil
}

func (s *Server) httpClient() *http.Client {
	if s.HTTP != nil {
		return s.HTTP
	}
	return http.DefaultClient
}

func contentResult(payload any, isError bool) map[string]any {
	text, err := json.Marshal(payload)
	if err != nil {
		text = []byte(fmt.Sprintf("%v", payload))
	}
	return map[string]any{
		"content": []map[string]any{{"type": "text", "text": string(text)}},
		"isError": isError,
	}
}

func errorResponse(id any, code int, message string) map[string]any {
	return map[string]any{
		"jsonrpc": "2.0",
		"id":      id,
		"error":   map[string]any{"code": code, "message": message},
	}
}

// --------------------------------------------------------------------------
// HTTP forwarder
// --------------------------------------------------------------------------

// ForwardStdio relays newline-delimited JSON-RPC frames to the Hub's /mcp
// endpoint, passing the caller's project identity along as a header.
func ForwardStdio(ctx context.Context, in io.Reader, out io.Writer, remoteURL, apiKey, cwd string) error {
	client := &http.Client{Timeout: 60 * time.Second}
	endpoint := strings.TrimRight(remoteURL, "/") + "/mcp"
	projectID := ""
	if cwd != "" {
		projectID = project.ResolveProjectID(cwd, "")
	}
	scanner := bufio.NewScanner(in)
	scanner.Buffer(make([]byte, 0, 64*1024), 16*1024*1024)
	writer := bufio.NewWriter(out)
	defer writer.Flush()
	for scanner.Scan() {
		line := bytes.TrimSpace(scanner.Bytes())
		if len(line) == 0 {
			continue
		}
		request, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(line))
		if err != nil {
			return err
		}
		request.Header.Set("Content-Type", "application/json")
		if apiKey != "" {
			request.Header.Set("Authorization", "Bearer "+apiKey)
			request.Header.Set("X-API-Key", apiKey)
		}
		if projectID != "" {
			request.Header.Set("X-Simple-Mem-Cwd", projectID)
		}
		response, err := client.Do(request)
		if err != nil {
			return err
		}
		body, readErr := io.ReadAll(io.LimitReader(response.Body, 64<<20))
		response.Body.Close()
		if readErr != nil {
			return readErr
		}
		if response.StatusCode == http.StatusNoContent {
			continue
		}
		if response.StatusCode < 200 || response.StatusCode >= 300 {
			return fmt.Errorf("hub returned %s: %s", response.Status, strings.TrimSpace(string(body)))
		}
		if _, err := writer.Write(bytes.TrimSpace(body)); err != nil {
			return err
		}
		if err := writer.WriteByte('\n'); err != nil {
			return err
		}
		if err := writer.Flush(); err != nil {
			return err
		}
	}
	return scanner.Err()
}

// --------------------------------------------------------------------------
// Edge config
// --------------------------------------------------------------------------

// GenerateEdgeConfig returns the MCP and hook configuration an edge machine
// needs, covering every supported harness.
func GenerateEdgeConfig(binPath, remoteURL, apiKey, machineID string) map[string]any {
	if binPath == "" {
		binPath = "simple-mem"
	}
	if machineID == "" {
		if host, err := os.Hostname(); err == nil {
			machineID = host
		}
	}
	base := strings.TrimRight(remoteURL, "/")
	env := map[string]any{
		"SIMPLE_MEM_REMOTE_URL": remoteURL,
		"SIMPLE_MEM_API_KEY":    apiKey,
		"SIMPLE_MEM_MACHINE_ID": machineID,
	}
	server := map[string]any{"command": binPath, "args": []string{"mcp"}, "env": env}
	return map[string]any{
		"mcpServers": map[string]any{"simple-mem": server},
		"http_mcp": map[string]any{
			"type":    "http",
			"url":     base + "/mcp",
			"headers": map[string]any{"Authorization": "Bearer " + apiKey},
		},
		"hooks": map[string]any{
			"SessionStart":     hookEntries(binPath),
			"UserPromptSubmit": hookEntries(binPath),
			"Stop":             hookEntries(binPath),
		},
	}
}

func hookEntries(binPath string) []map[string]any {
	var entries []map[string]any
	for _, harness := range []string{"claude-code", "factory-droid", "codex", "pi"} {
		entries = append(entries, map[string]any{
			"matcher": harness,
			"hooks": []map[string]any{{
				"type":    "command",
				"command": binPath + " capture " + harness,
			}},
		})
	}
	return entries
}

// WriteEdgeConfigs installs the generated MCP servers (and hooks, where the
// host reads them from the same file) into the requested harness configs.
func WriteEdgeConfigs(cfg map[string]any, writeClaude, writeDroid bool, homeDir string) error {
	if homeDir == "" {
		resolved, err := os.UserHomeDir()
		if err != nil {
			return err
		}
		homeDir = resolved
	}
	servers, _ := cfg["mcpServers"].(map[string]any)
	hooks, _ := cfg["hooks"].(map[string]any)
	merge := map[string]any{}
	if servers != nil {
		merge["mcpServers"] = servers
	}
	if hooks != nil {
		merge["hooks"] = hooks
	}
	if writeClaude {
		if err := mergeJSONFile(filepath.Join(homeDir, ".claude", "settings.json"), merge); err != nil {
			return err
		}
	}
	if writeDroid {
		if err := mergeJSONFile(filepath.Join(homeDir, ".factory", "mcp.json"), merge); err != nil {
			return err
		}
	}
	return nil
}

func mergeJSONFile(path string, additions map[string]any) error {
	existing := map[string]any{}
	if data, err := os.ReadFile(path); err == nil {
		_ = json.Unmarshal(data, &existing)
	}
	for key, value := range additions {
		if key == "mcpServers" {
			servers, _ := existing["mcpServers"].(map[string]any)
			if servers == nil {
				servers = map[string]any{}
			}
			added, _ := value.(map[string]any)
			for name, server := range added {
				servers[name] = server
			}
			existing["mcpServers"] = servers
			continue
		}
		existing[key] = value
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	data, err := json.MarshalIndent(existing, "", "  ")
	if err != nil {
		return err
	}
	temporary := filepath.Join(filepath.Dir(path), "."+filepath.Base(path)+".tmp")
	if err := os.WriteFile(temporary, data, 0o644); err != nil {
		return err
	}
	return os.Rename(temporary, path)
}

// --------------------------------------------------------------------------
// Helpers
// --------------------------------------------------------------------------

func decodeObject(raw []byte) (map[string]any, error) {
	decoder := json.NewDecoder(bytes.NewReader(raw))
	decoder.UseNumber()
	var payload map[string]any
	if err := decoder.Decode(&payload); err != nil {
		return nil, err
	}
	return payload, nil
}

func decodeBatch(raw []byte) ([]map[string]any, error) {
	decoder := json.NewDecoder(bytes.NewReader(raw))
	decoder.UseNumber()
	var items []map[string]any
	if err := decoder.Decode(&items); err != nil {
		return nil, err
	}
	return items, nil
}

func mustMarshal(value any) []byte {
	encoded, err := json.Marshal(value)
	if err != nil {
		return []byte(`{"jsonrpc":"2.0","id":null,"error":{"code":-32603,"message":"Internal error"}}`)
	}
	return encoded
}

func stringArg(args map[string]any, key, fallback string) string {
	if value, ok := args[key].(string); ok {
		return value
	}
	return fallback
}

func intArg(args map[string]any, key string, fallback int) int {
	switch value := args[key].(type) {
	case float64:
		return int(value)
	case json.Number:
		if parsed, err := value.Int64(); err == nil {
			return int(parsed)
		}
	case int:
		return value
	}
	return fallback
}

func stringSliceArg(value any) []string {
	switch typed := value.(type) {
	case string:
		if typed == "" {
			return []string{}
		}
		return []string{typed}
	case []any:
		var out []string
		for _, item := range typed {
			if text, ok := item.(string); ok {
				out = append(out, text)
			}
		}
		if out == nil {
			out = []string{}
		}
		return out
	case []string:
		return typed
	default:
		return []string{}
	}
}
