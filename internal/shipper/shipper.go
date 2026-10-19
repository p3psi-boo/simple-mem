// Package shipper is the edge half of Simple Mem: it reads local harness
// transcripts incrementally and moves them to the Hub.
//
// It has three jobs and nothing else. It speaks the Hub's HTTP surface with a
// reusable client; it remembers how far it has read each transcript in a
// cursor file written through a rename so a crash cannot half-write it; and it
// turns hook events into the same turns a transcript would have produced, so
// the Hub cannot tell live capture from a later backfill.
package shipper

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"

	"simple-mem/internal/transcript"
)

// DefaultCursorPath is where the per-file read cursors live.
const (
	defaultCursorFile = ".simple-mem/edge-cursor.json"
	defaultBatchSize  = 100
)

// Indirection for tests and for the transcript workstream's own semantics.
var (
	parseTranscript     = transcript.ParseTranscript
	discoverTranscripts = transcript.DiscoverTranscripts
	parseHookPayload    = transcript.ParseHookPayload
)

// DefaultCursorPath returns ~/.simple-mem/edge-cursor.json.
func DefaultCursorPath() string {
	home, err := os.UserHomeDir()
	if err != nil || home == "" {
		home = "."
	}
	return filepath.Join(home, defaultCursorFile)
}

// ShipOptions drives ShipTranscript and ShipAll.
type ShipOptions struct {
	Source     string // one harness, or "all"
	Transcript string // ship exactly this file when set
	RemoteURL  string
	APIKey     string
	MachineID  string
	ProjectID  string
	CursorFile string
	HomeDir    string
	BatchSize  int
}

// ShipResult is the outcome of shipping one transcript file.
type ShipResult struct {
	SourceID  string `json:"source_id"`
	Path      string `json:"path"`
	Shipped   int    `json:"shipped"`
	Skipped   int    `json:"skipped"`
	Unchanged bool   `json:"unchanged,omitempty"`
}

// SourceStats aggregates one harness inside a ShipAllResult.
type SourceStats struct {
	Files   int `json:"files"`
	Shipped int `json:"shipped"`
	Skipped int `json:"skipped"`
}

// ShipAllResult aggregates every transcript ShipAll visited.
type ShipAllResult struct {
	FilesScanned int                    `json:"files_scanned"`
	Shipped      int                    `json:"shipped"`
	Skipped      int                    `json:"skipped"`
	BySource     map[string]SourceStats `json:"by_source"`
	Errors       []string               `json:"errors,omitempty"`
}

// CaptureOptions is one harness hook event on its way to the Hub.
type CaptureOptions struct {
	Harness   string
	Payload   map[string]any
	RemoteURL string
	APIKey    string
	MachineID string
	ProjectID string
	Strict    bool
}

// CaptureResult reports where a hook event went.
type CaptureResult struct {
	Mode    string `json:"mode"`
	Shipped int    `json:"shipped"`
	Error   string `json:"error,omitempty"`
}

// --------------------------------------------------------------------------
// Hub client
// --------------------------------------------------------------------------

// Client is a minimal Hub client. It is safe for concurrent use.
type Client struct {
	BaseURL string
	APIKey  string
	HTTP    *http.Client
}

// NewClient builds a client for baseURL, defaulting to a 30s timeout.
func NewClient(baseURL, apiKey string) *Client {
	return &Client{
		BaseURL: strings.TrimRight(baseURL, "/"),
		APIKey:  apiKey,
		HTTP:    &http.Client{Timeout: 30 * time.Second},
	}
}

// Do issues one JSON request against the Hub and decodes the JSON response.
func (c *Client) Do(ctx context.Context, method, path string, body any) (map[string]any, error) {
	var reader io.Reader
	if body != nil {
		encoded, err := json.Marshal(body)
		if err != nil {
			return nil, err
		}
		reader = bytes.NewReader(encoded)
	}
	request, err := http.NewRequestWithContext(ctx, method, c.BaseURL+path, reader)
	if err != nil {
		return nil, err
	}
	if body != nil {
		request.Header.Set("Content-Type", "application/json")
	}
	if c.APIKey != "" {
		request.Header.Set("Authorization", "Bearer "+c.APIKey)
		request.Header.Set("X-API-Key", c.APIKey)
	}
	response, err := c.httpClient().Do(request)
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

func (c *Client) httpClient() *http.Client {
	if c.HTTP != nil {
		return c.HTTP
	}
	return http.DefaultClient
}

// Search posts a search request to /api/v1/search.
func (c *Client) Search(ctx context.Context, body map[string]any) (map[string]any, error) {
	return c.Do(ctx, http.MethodPost, "/api/v1/search", body)
}

// Remember posts a remember request to /api/v1/remember.
func (c *Client) Remember(ctx context.Context, body map[string]any) (map[string]any, error) {
	return c.Do(ctx, http.MethodPost, "/api/v1/remember", body)
}

// Forget deletes one memory by id.
func (c *Client) Forget(ctx context.Context, memoryID string) (map[string]any, error) {
	return c.Do(ctx, http.MethodDelete, "/api/v1/memories/"+url.PathEscape(memoryID), nil)
}

// Briefing fetches the working memory briefing for a project: the standing
// facts, decisions and recent turns an agent should have on hand before work
// starts.
func (c *Client) Briefing(ctx context.Context, projectID string, limit int) (map[string]any, error) {
	query := url.Values{}
	if projectID != "" {
		query.Set("project_id", projectID)
	}
	if limit > 0 {
		query.Set("limit", strconv.Itoa(limit))
	}
	return c.Do(ctx, http.MethodGet, withQuery("/api/v1/briefing", query), nil)
}

// FetchThread fetches one conversation's turns. When aroundMessageID is set the
// Hub returns a window centred on that message instead of the tail.
func (c *Client) FetchThread(ctx context.Context, conversationID string, aroundMessageID string, limit int) (map[string]any, error) {
	query := url.Values{}
	if aroundMessageID != "" {
		query.Set("around_message_id", aroundMessageID)
	}
	if limit > 0 {
		query.Set("limit", strconv.Itoa(limit))
	}
	path := "/api/v1/threads/" + url.PathEscape(conversationID)
	return c.Do(ctx, http.MethodGet, withQuery(path, query), nil)
}

func withQuery(path string, query url.Values) string {
	if encoded := query.Encode(); encoded != "" {
		return path + "?" + encoded
	}
	return path
}

// Ingest ships one batch of turns to /api/v1/ingest.
func (c *Client) Ingest(ctx context.Context, machineID, sourceID, projectID string, turns []transcript.Turn) (map[string]any, error) {
	body := map[string]any{
		"machine_id": machineID,
		"source_id":  sourceID,
		"turns":      turns,
	}
	if projectID != "" {
		body["project_id"] = projectID
	}
	return c.Do(ctx, http.MethodPost, "/api/v1/ingest", body)
}

// --------------------------------------------------------------------------
// Cursor
// --------------------------------------------------------------------------

type cursorEntry struct {
	MtimeNS   int64  `json:"mtime_ns"`
	Offset    int64  `json:"offset"`
	Turns     int64  `json:"turns,omitempty"`
	UpdatedAt string `json:"updated_at,omitempty"`
}

type cursorState struct {
	Version int                    `json:"version"`
	Cursors map[string]cursorEntry `json:"cursors"`
}

var cursorMu sync.Mutex

func loadCursor(path string) cursorState {
	state := cursorState{Version: 1, Cursors: map[string]cursorEntry{}}
	data, err := os.ReadFile(path)
	if err != nil {
		return state
	}
	var decoded cursorState
	if err := json.Unmarshal(data, &decoded); err != nil {
		return state
	}
	if decoded.Cursors == nil {
		decoded.Cursors = map[string]cursorEntry{}
	}
	if decoded.Version == 0 {
		decoded.Version = 1
	}
	return decoded
}

// saveCursor writes the cursor through a rename so a crash cannot leave a
// half-written file behind.
func saveCursor(path string, state cursorState) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	data, err := json.MarshalIndent(state, "", "  ")
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
// Shipping
// --------------------------------------------------------------------------

// ShipTranscript sends the not-yet-shipped turns of one transcript file.
func ShipTranscript(ctx context.Context, opts ShipOptions) (ShipResult, error) {
	result := ShipResult{SourceID: opts.Source, Path: opts.Transcript}
	if opts.Transcript == "" {
		return result, errors.New("shipper: a transcript path is required")
	}
	if !transcriptSupported(opts.Source) {
		return result, fmt.Errorf("shipper: unsupported harness: %q", opts.Source)
	}
	if opts.RemoteURL == "" {
		return result, errors.New("shipper: a remote URL is required")
	}
	client := NewClient(opts.RemoteURL, opts.APIKey)
	return shipOne(ctx, client, opts, opts.Source, opts.Transcript)
}

// ShipAll ships every discovered transcript for the requested sources.
func ShipAll(ctx context.Context, opts ShipOptions) (ShipAllResult, error) {
	sources, err := resolveSources(opts.Source)
	if err != nil {
		return ShipAllResult{}, err
	}
	if opts.RemoteURL == "" {
		return ShipAllResult{}, errors.New("shipper: a remote URL is required")
	}
	result := ShipAllResult{BySource: map[string]SourceStats{}}
	for _, source := range sources {
		result.BySource[source] = SourceStats{}
	}
	client := NewClient(opts.RemoteURL, opts.APIKey)

	homeDir := opts.HomeDir
	if homeDir == "" {
		homeDir, _ = os.UserHomeDir()
	}

	var pairs []transcript.TranscriptInfo
	if opts.Transcript != "" {
		source := opts.Source
		if source == "" || source == "all" {
			source = "claude-code"
		}
		pairs = append(pairs, transcript.TranscriptInfo{SourceID: source, Path: opts.Transcript})
	} else {
		for _, source := range sources {
			found, err := discoverTranscripts(source, homeDir)
			if err != nil {
				result.Errors = append(result.Errors, fmt.Sprintf("%s: %v", source, err))
				continue
			}
			pairs = append(pairs, found...)
		}
	}

	for _, info := range pairs {
		result.FilesScanned++
		stats := result.BySource[info.SourceID]
		stats.Files++
		shipped, err := shipOne(ctx, client, opts, info.SourceID, info.Path)
		stats.Shipped += shipped.Shipped
		stats.Skipped += shipped.Skipped
		result.BySource[info.SourceID] = stats
		result.Shipped += shipped.Shipped
		result.Skipped += shipped.Skipped
		if err != nil {
			result.Errors = append(result.Errors, fmt.Sprintf("%s: %v", info.Path, err))
		}
	}
	return result, nil
}

func shipOne(ctx context.Context, client *Client, opts ShipOptions, source, path string) (ShipResult, error) {
	result := ShipResult{SourceID: source, Path: path}
	info, err := os.Stat(path)
	if err != nil {
		return result, err
	}
	mtime := info.ModTime().UnixNano()
	cursorPath := opts.CursorFile
	if cursorPath == "" {
		cursorPath = DefaultCursorPath()
	}

	cursorMu.Lock()
	state := loadCursor(cursorPath)
	entry, exists := state.Cursors[path]
	cursorMu.Unlock()

	if exists && entry.MtimeNS == mtime && info.Size() >= entry.Offset {
		result.Unchanged = true
		result.Skipped = int(entry.Turns)
		return result, nil
	}
	if info.Size() < entry.Offset { // the file shrank or was rotated away
		entry = cursorEntry{}
	}

	turns, newOffset, err := parseTranscript(path, source, entry.Offset, opts.MachineID, opts.ProjectID)
	if err != nil {
		return result, err
	}
	result.Skipped = int(entry.Turns)

	if len(turns) > 0 {
		batchSize := opts.BatchSize
		if batchSize <= 0 {
			batchSize = defaultBatchSize
		}
		for start := 0; start < len(turns); start += batchSize {
			end := start + batchSize
			if end > len(turns) {
				end = len(turns)
			}
			if _, err := client.Ingest(ctx, opts.MachineID, source, opts.ProjectID, turns[start:end]); err != nil {
				return result, err // the cursor does not advance on failure
			}
		}
		result.Shipped = len(turns)
		entry.Turns += int64(len(turns))
	}

	entry.MtimeNS = mtime
	entry.Offset = newOffset
	entry.UpdatedAt = time.Now().UTC().Format(time.RFC3339Nano)
	cursorMu.Lock()
	state = loadCursor(cursorPath) // re-read under the lock so a concurrent writer's other paths survive
	state.Cursors[path] = entry
	if err := saveCursor(cursorPath, state); err != nil {
		cursorMu.Unlock()
		return result, err
	}
	cursorMu.Unlock()
	return result, nil
}

func resolveSources(source string) ([]string, error) {
	if source == "" || source == "all" {
		return append([]string(nil), transcript.SupportedHarnesses...), nil
	}
	if !transcriptSupported(source) {
		return nil, fmt.Errorf("shipper: unsupported harness: %q", source)
	}
	return []string{source}, nil
}

func transcriptSupported(harness string) bool {
	for _, candidate := range transcript.SupportedHarnesses {
		if candidate == harness {
			return true
		}
	}
	return false
}

// --------------------------------------------------------------------------
// Live hook capture
// --------------------------------------------------------------------------

// CaptureHookEvent turns one harness hook payload into turns and ships them.
func CaptureHookEvent(ctx context.Context, opts CaptureOptions) (CaptureResult, error) {
	result := CaptureResult{Mode: "disabled"}
	turns, err := parseHookPayload(opts.Harness, opts.Payload, opts.MachineID, opts.ProjectID)
	if err != nil {
		return result, err
	}
	if opts.RemoteURL == "" {
		return result, nil
	}
	result.Mode = "remote"
	if len(turns) == 0 {
		return result, nil
	}
	client := NewClient(opts.RemoteURL, opts.APIKey)
	if _, err := client.Ingest(ctx, opts.MachineID, opts.Harness, opts.ProjectID, turns); err != nil {
		result.Error = err.Error()
		return result, err
	}
	result.Shipped = len(turns)
	return result, nil
}
