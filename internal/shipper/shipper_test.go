package shipper

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"sync"
	"testing"
	"time"

	"simple-mem/internal/transcript"
)

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

type recordedRequest struct {
	path    string
	rawPath string
	query   url.Values
	method  string
	headers http.Header
	body    map[string]any
}

func newHub(t *testing.T, status int) (*httptest.Server, *[]recordedRequest) {
	t.Helper()
	var mu sync.Mutex
	requests := &[]recordedRequest{}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body := map[string]any{}
		if r.Body != nil {
			_ = json.NewDecoder(r.Body).Decode(&body)
		}
		mu.Lock()
		*requests = append(*requests, recordedRequest{
			path: r.URL.Path, rawPath: r.URL.EscapedPath(), query: r.URL.Query(),
			method: r.Method, headers: r.Header.Clone(), body: body,
		})
		mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(status)
		_, _ = w.Write([]byte(`{"ok":true}`))
	}))
	t.Cleanup(server.Close)
	return server, requests
}

func write(t *testing.T, path, content string) {
	t.Helper()
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatalf("write %s: %v", path, err)
	}
}

// touch forces a distinct mtime, since a filesystem may otherwise reuse the
// previous timestamp and the cursor would (correctly) report "unchanged".
func touch(t *testing.T, path string) {
	t.Helper()
	when := time.Now().Add(2 * time.Second)
	if err := os.Chtimes(path, when, when); err != nil {
		t.Fatalf("chtimes %s: %v", path, err)
	}
}

func stubTranscript(t *testing.T, turns []transcript.Turn, offset int64) *[]int64 {
	t.Helper()
	seenOffsets := &[]int64{}
	original := parseTranscript
	t.Cleanup(func() { parseTranscript = original })
	parseTranscript = func(path, source string, startOffset int64, machineID, projectID string) ([]transcript.Turn, int64, error) {
		*seenOffsets = append(*seenOffsets, startOffset)
		if startOffset >= int64(len(turns)) {
			return nil, offset, nil
		}
		return turns[startOffset:], offset, nil
	}
	return seenOffsets
}

// ---------------------------------------------------------------------------
// client
// ---------------------------------------------------------------------------

func TestClientIngestSendsAuthHeadersAndBody(t *testing.T) {
	server, requests := newHub(t, http.StatusAccepted)
	client := NewClient(server.URL+"/", "secret")
	_, err := client.Ingest(context.Background(), "m1", "pi", "proj://demo", []transcript.Turn{{TurnID: "t1", Content: "hello"}})
	if err != nil {
		t.Fatalf("ingest: %v", err)
	}
	if len(*requests) != 1 {
		t.Fatalf("expected 1 request, got %d", len(*requests))
	}
	request := (*requests)[0]
	if request.path != "/api/v1/ingest" {
		t.Fatalf("unexpected path %q", request.path)
	}
	if got := request.headers.Get("Authorization"); got != "Bearer secret" {
		t.Fatalf("Authorization = %q", got)
	}
	if got := request.headers.Get("X-API-Key"); got != "secret" {
		t.Fatalf("X-API-Key = %q", got)
	}
	if request.body["machine_id"] != "m1" || request.body["source_id"] != "pi" || request.body["project_id"] != "proj://demo" {
		t.Fatalf("unexpected body %#v", request.body)
	}
	turns, ok := request.body["turns"].([]any)
	if !ok || len(turns) != 1 {
		t.Fatalf("turns = %#v", request.body["turns"])
	}
}

func TestClientConvenienceEndpoints(t *testing.T) {
	server, requests := newHub(t, http.StatusOK)
	client := NewClient(server.URL, "")
	if _, err := client.Search(context.Background(), map[string]any{"query": "x"}); err != nil {
		t.Fatalf("search: %v", err)
	}
	if _, err := client.Remember(context.Background(), map[string]any{"statement": "y"}); err != nil {
		t.Fatalf("remember: %v", err)
	}
	if _, err := client.Forget(context.Background(), "mem_1"); err != nil {
		t.Fatalf("forget: %v", err)
	}
	if len(*requests) != 3 {
		t.Fatalf("requests = %#v", *requests)
	}
	if (*requests)[0].path != "/api/v1/search" || (*requests)[0].method != http.MethodPost {
		t.Fatalf("search request = %#v", (*requests)[0])
	}
	if (*requests)[1].path != "/api/v1/remember" || (*requests)[1].method != http.MethodPost {
		t.Fatalf("remember request = %#v", (*requests)[1])
	}
	if (*requests)[2].path != "/api/v1/memories/mem_1" || (*requests)[2].method != http.MethodDelete {
		t.Fatalf("forget request = %#v", (*requests)[2])
	}
}

func TestClientReturnsHubError(t *testing.T) {
	server, _ := newHub(t, http.StatusInternalServerError)
	client := NewClient(server.URL, "")
	if _, err := client.Ingest(context.Background(), "m", "pi", "", nil); err == nil {
		t.Fatal("expected an error for a 500 response")
	}
}

func TestClientBriefingAndThreadEndpoints(t *testing.T) {
	server, requests := newHub(t, http.StatusOK)
	client := NewClient(server.URL, "")
	if _, err := client.Briefing(context.Background(), "proj://demo", 20); err != nil {
		t.Fatalf("briefing: %v", err)
	}
	if _, err := client.FetchThread(context.Background(), "conv/1", "msg_9", 5); err != nil {
		t.Fatalf("thread: %v", err)
	}
	if len(*requests) != 2 {
		t.Fatalf("requests = %#v", *requests)
	}

	briefing := (*requests)[0]
	if briefing.method != http.MethodGet || briefing.path != "/api/v1/briefing" {
		t.Fatalf("briefing request = %#v", briefing)
	}
	if briefing.query.Get("project_id") != "proj://demo" || briefing.query.Get("limit") != "20" {
		t.Fatalf("briefing query = %#v", briefing.query)
	}

	thread := (*requests)[1]
	if thread.method != http.MethodGet {
		t.Fatalf("thread method = %q", thread.method)
	}
	if thread.rawPath != "/api/v1/threads/conv%2F1" {
		t.Fatalf("thread path must be escaped, got %q", thread.rawPath)
	}
	if thread.query.Get("around_message_id") != "msg_9" || thread.query.Get("limit") != "5" {
		t.Fatalf("thread query = %#v", thread.query)
	}
}

func TestClientThreadOmitsEmptyAroundMessage(t *testing.T) {
	server, requests := newHub(t, http.StatusOK)
	client := NewClient(server.URL, "")
	if _, err := client.FetchThread(context.Background(), "conv_1", "", 0); err != nil {
		t.Fatalf("thread: %v", err)
	}
	request := (*requests)[0]
	if request.path != "/api/v1/threads/conv_1" {
		t.Fatalf("path = %q", request.path)
	}
	if _, ok := request.query["around_message_id"]; ok {
		t.Fatalf("around_message_id must be omitted, query = %#v", request.query)
	}
	if _, ok := request.query["limit"]; ok {
		t.Fatalf("limit must be omitted when zero, query = %#v", request.query)
	}
}

// ---------------------------------------------------------------------------
// ShipTranscript cursor
// ---------------------------------------------------------------------------

func TestShipTranscriptAdvancesAndStopsAtCursor(t *testing.T) {
	server, requests := newHub(t, http.StatusOK)
	dir := t.TempDir()
	path := filepath.Join(dir, "pi.jsonl")
	write(t, path, "line1\nline2\n")
	cursor := filepath.Join(dir, "cursor.json")

	turns := []transcript.Turn{{TurnID: "t1"}, {TurnID: "t2"}}
	offsets := stubTranscript(t, turns, 2)

	options := ShipOptions{Source: "pi", Transcript: path, RemoteURL: server.URL, CursorFile: cursor}

	first, err := ShipTranscript(context.Background(), options)
	if err != nil {
		t.Fatalf("first ship: %v", err)
	}
	if first.Shipped != 2 || first.Skipped != 0 || first.Unchanged {
		t.Fatalf("first = %#v", first)
	}
	if len(*requests) != 1 {
		t.Fatalf("expected 1 ingest, got %d", len(*requests))
	}

	second, err := ShipTranscript(context.Background(), options)
	if err != nil {
		t.Fatalf("second ship: %v", err)
	}
	if second.Shipped != 0 || !second.Unchanged || second.Skipped != 2 {
		t.Fatalf("second = %#v", second)
	}
	if len(*requests) != 1 {
		t.Fatalf("unchanged ship must not post again, got %d requests", len(*requests))
	}
	if len(*offsets) != 1 {
		t.Fatalf("unchanged ship must not reparse, offsets = %v", *offsets)
	}

	data, err := os.ReadFile(cursor)
	if err != nil {
		t.Fatalf("cursor not written: %v", err)
	}
	var state cursorState
	if err := json.Unmarshal(data, &state); err != nil {
		t.Fatalf("cursor invalid: %v", err)
	}
	entry, ok := state.Cursors[path]
	if !ok || entry.Offset != 2 || entry.Turns != 2 {
		t.Fatalf("cursor entry = %#v", entry)
	}
}

func TestShipTranscriptResumesFromStoredOffset(t *testing.T) {
	server, requests := newHub(t, http.StatusOK)
	dir := t.TempDir()
	path := filepath.Join(dir, "pi.jsonl")
	write(t, path, "a\n")
	cursor := filepath.Join(dir, "cursor.json")

	all := []transcript.Turn{{TurnID: "t1"}}
	offsets := &[]int64{}
	original := parseTranscript
	t.Cleanup(func() { parseTranscript = original })
	parseTranscript = func(path, source string, startOffset int64, machineID, projectID string) ([]transcript.Turn, int64, error) {
		*offsets = append(*offsets, startOffset)
		if startOffset >= int64(len(all)) {
			return nil, int64(len(all)), nil
		}
		return all[startOffset:], int64(len(all)), nil
	}
	options := ShipOptions{Source: "pi", Transcript: path, RemoteURL: server.URL, CursorFile: cursor}

	if _, err := ShipTranscript(context.Background(), options); err != nil {
		t.Fatalf("first ship: %v", err)
	}
	all = append(all, transcript.Turn{TurnID: "t2"}, transcript.Turn{TurnID: "t3"})
	write(t, path, "a\nb\nc\n") // longer content, so it is not mistaken for rotation
	touch(t, path)

	result, err := ShipTranscript(context.Background(), options)
	if err != nil {
		t.Fatalf("second ship: %v", err)
	}
	if result.Shipped != 2 || result.Skipped != 1 {
		t.Fatalf("second = %#v", result)
	}
	if len(*offsets) != 2 || (*offsets)[1] != 1 {
		t.Fatalf("expected resume offset 1, saw %v", *offsets)
	}
	bodyTurns, _ := (*requests)[1].body["turns"].([]any)
	if len(bodyTurns) != 2 {
		t.Fatalf("expected only the new turns, got %#v", bodyTurns)
	}
}

func TestShipTranscriptResetsCursorAfterRotation(t *testing.T) {
	server, _ := newHub(t, http.StatusOK)
	dir := t.TempDir()
	path := filepath.Join(dir, "pi.jsonl")
	write(t, path, "first transcript content that is fairly long\n")
	cursor := filepath.Join(dir, "cursor.json")

	turns := []transcript.Turn{{TurnID: "old"}}
	offsets := stubTranscript(t, turns, 1000) // pretend the first pass consumed 1000 units
	options := ShipOptions{Source: "pi", Transcript: path, RemoteURL: server.URL, CursorFile: cursor}
	if _, err := ShipTranscript(context.Background(), options); err != nil {
		t.Fatalf("first ship: %v", err)
	}

	write(t, path, "x\n") // rotated to something smaller than the stored offset
	touch(t, path)
	if _, err := ShipTranscript(context.Background(), options); err != nil {
		t.Fatalf("second ship: %v", err)
	}
	if len(*offsets) != 2 || (*offsets)[1] != 0 {
		t.Fatalf("rotation must reset the offset, saw %v", *offsets)
	}
}

func TestShipTranscriptErrorLeavesCursorUntouched(t *testing.T) {
	server, _ := newHub(t, http.StatusInternalServerError)
	dir := t.TempDir()
	path := filepath.Join(dir, "pi.jsonl")
	write(t, path, "one\n")
	cursor := filepath.Join(dir, "cursor.json")

	stubTranscript(t, []transcript.Turn{{TurnID: "t1"}}, 1)
	_, err := ShipTranscript(context.Background(), ShipOptions{
		Source: "pi", Transcript: path, RemoteURL: server.URL, CursorFile: cursor,
	})
	if err == nil {
		t.Fatal("expected a shipping error")
	}
	if _, statErr := os.Stat(cursor); !os.IsNotExist(statErr) {
		t.Fatalf("cursor should not exist after a failed ship, err = %v", statErr)
	}
}

func TestShipTranscriptRejectsBadOptions(t *testing.T) {
	if _, err := ShipTranscript(context.Background(), ShipOptions{Source: "nope", Transcript: "x"}); err == nil {
		t.Fatal("expected unsupported harness error")
	}
	if _, err := ShipTranscript(context.Background(), ShipOptions{Source: "pi", Transcript: "x"}); err == nil {
		t.Fatal("expected missing remote URL error")
	}
	if _, err := ShipTranscript(context.Background(), ShipOptions{Source: "pi", RemoteURL: "http://x"}); err == nil {
		t.Fatal("expected missing transcript error")
	}
}

func TestShipTranscriptBatchesTurns(t *testing.T) {
	server, requests := newHub(t, http.StatusOK)
	dir := t.TempDir()
	path := filepath.Join(dir, "pi.jsonl")
	write(t, path, "x\n")
	cursor := filepath.Join(dir, "cursor.json")

	turns := make([]transcript.Turn, 5)
	for i := range turns {
		turns[i] = transcript.Turn{TurnID: "t"}
	}
	stubTranscript(t, turns, 5)
	result, err := ShipTranscript(context.Background(), ShipOptions{
		Source: "pi", Transcript: path, RemoteURL: server.URL, CursorFile: cursor, BatchSize: 2,
	})
	if err != nil {
		t.Fatalf("ship: %v", err)
	}
	if result.Shipped != 5 {
		t.Fatalf("shipped = %d", result.Shipped)
	}
	if len(*requests) != 3 {
		t.Fatalf("expected 3 batched requests, got %d", len(*requests))
	}
}

// ---------------------------------------------------------------------------
// ShipAll
// ---------------------------------------------------------------------------

func TestShipAllAggregatesBySource(t *testing.T) {
	server, _ := newHub(t, http.StatusOK)
	dir := t.TempDir()
	first := filepath.Join(dir, "a.jsonl")
	second := filepath.Join(dir, "b.jsonl")
	write(t, first, "a\n")
	write(t, second, "b\n")

	originalDiscover := discoverTranscripts
	t.Cleanup(func() { discoverTranscripts = originalDiscover })
	discoverTranscripts = func(source, homeDir string) ([]transcript.TranscriptInfo, error) {
		switch source {
		case "claude-code":
			return []transcript.TranscriptInfo{{SourceID: "claude-code", Path: first}}, nil
		case "codex":
			return []transcript.TranscriptInfo{{SourceID: "codex", Path: second}}, nil
		default:
			return nil, nil
		}
	}

	originalParse := parseTranscript
	t.Cleanup(func() { parseTranscript = originalParse })
	parseTranscript = func(path, source string, startOffset int64, machineID, projectID string) ([]transcript.Turn, int64, error) {
		if source == "claude-code" {
			return []transcript.Turn{{TurnID: "c1"}, {TurnID: "c2"}}, 2, nil
		}
		return []transcript.Turn{{TurnID: "x1"}}, 1, nil
	}

	result, err := ShipAll(context.Background(), ShipOptions{
		Source: "all", RemoteURL: server.URL, CursorFile: filepath.Join(dir, "cursor.json"),
	})
	if err != nil {
		t.Fatalf("ship all: %v", err)
	}
	if result.FilesScanned != 2 || result.Shipped != 3 {
		t.Fatalf("result = %#v", result)
	}
	if result.BySource["claude-code"] != (SourceStats{Files: 1, Shipped: 2}) {
		t.Fatalf("claude-code stats = %#v", result.BySource["claude-code"])
	}
	if result.BySource["codex"] != (SourceStats{Files: 1, Shipped: 1}) {
		t.Fatalf("codex stats = %#v", result.BySource["codex"])
	}
	if result.BySource["pi"] != (SourceStats{}) {
		t.Fatalf("pi stats = %#v", result.BySource["pi"])
	}
}

func TestShipAllRejectsUnknownSource(t *testing.T) {
	if _, err := ShipAll(context.Background(), ShipOptions{Source: "bogus", RemoteURL: "http://x"}); err == nil {
		t.Fatal("expected unsupported source error")
	}
}

// ---------------------------------------------------------------------------
// CaptureHookEvent
// ---------------------------------------------------------------------------

func TestCaptureHookEventShipsTurns(t *testing.T) {
	server, requests := newHub(t, http.StatusAccepted)
	original := parseHookPayload
	t.Cleanup(func() { parseHookPayload = original })
	parseHookPayload = func(harness string, payload map[string]any, machineID, projectID string) ([]transcript.Turn, error) {
		return []transcript.Turn{{TurnID: "h1", Role: "user", Content: "hi"}}, nil
	}
	result, err := CaptureHookEvent(context.Background(), CaptureOptions{
		Harness: "pi", Payload: map[string]any{"event": "UserPromptSubmit"},
		RemoteURL: server.URL, APIKey: "k", MachineID: "m1", ProjectID: "proj://demo",
	})
	if err != nil {
		t.Fatalf("capture: %v", err)
	}
	if result.Mode != "remote" || result.Shipped != 1 {
		t.Fatalf("result = %#v", result)
	}
	if len(*requests) != 1 || (*requests)[0].path != "/api/v1/ingest" {
		t.Fatalf("requests = %#v", *requests)
	}
	if (*requests)[0].body["source_id"] != "pi" {
		t.Fatalf("source_id = %#v", (*requests)[0].body["source_id"])
	}
}

func TestCaptureHookEventDisabledWithoutRemote(t *testing.T) {
	original := parseHookPayload
	t.Cleanup(func() { parseHookPayload = original })
	parseHookPayload = func(harness string, payload map[string]any, machineID, projectID string) ([]transcript.Turn, error) {
		return []transcript.Turn{{TurnID: "h1"}}, nil
	}
	result, err := CaptureHookEvent(context.Background(), CaptureOptions{Harness: "pi", Payload: map[string]any{}})
	if err != nil {
		t.Fatalf("capture: %v", err)
	}
	if result.Mode != "disabled" || result.Shipped != 0 {
		t.Fatalf("result = %#v", result)
	}
}

func TestCaptureHookEventReportsUnreachableHub(t *testing.T) {
	server, _ := newHub(t, http.StatusInternalServerError)
	original := parseHookPayload
	t.Cleanup(func() { parseHookPayload = original })
	parseHookPayload = func(harness string, payload map[string]any, machineID, projectID string) ([]transcript.Turn, error) {
		return []transcript.Turn{{TurnID: "h1"}}, nil
	}
	result, err := CaptureHookEvent(context.Background(), CaptureOptions{
		Harness: "pi", Payload: map[string]any{}, RemoteURL: server.URL,
	})
	if err == nil {
		t.Fatal("expected transport error")
	}
	if result.Error == "" {
		t.Fatalf("expected error text in result, got %#v", result)
	}
}
