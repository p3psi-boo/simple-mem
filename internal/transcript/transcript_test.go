package transcript

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func writeJSONL(t *testing.T, path string, records []string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	content := ""
	for _, record := range records {
		content += record + "\n"
	}
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
}

func mustParse(t *testing.T, path, sourceID string, startOffset int64) ([]Turn, int64) {
	t.Helper()
	turns, next, err := ParseTranscript(path, sourceID, startOffset, "machine-1", "demo")
	if err != nil {
		t.Fatalf("ParseTranscript(%s): %v", sourceID, err)
	}
	return turns, next
}

// --------------------------------------------------------------------------
// Per-harness parsers
// --------------------------------------------------------------------------

func TestParseClaudeCode(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "claude.jsonl")
	writeJSONL(t, path, []string{
		`{"type":"summary","sessionId":"sess-1","cwd":"` + dir + `","summary":"ignored"}`,
		`{"type":"user","sessionId":"sess-1","uuid":"u1","cwd":"` + dir + `","timestamp":"t1","message":{"role":"user","content":"hello"}}`,
		`{"type":"assistant","sessionId":"sess-1","uuid":"a1","cwd":"` + dir + `","timestamp":"t2","message":{"role":"assistant","content":[{"type":"thinking","thinking":"secret"},{"type":"text","text":"hi there"},{"type":"tool_use","name":"Bash","input":{}}]}}`,
	})
	turns, next := mustParse(t, path, "claude-code", 0)

	if next != 3 {
		t.Fatalf("nextOffset = %d, want 3", next)
	}
	if len(turns) != 2 {
		t.Fatalf("turns = %d, want 2", len(turns))
	}
	if got := []string{turns[0].Role, turns[1].Role}; got[0] != "user" || got[1] != "assistant" {
		t.Errorf("roles = %v", got)
	}
	if got := []string{turns[0].Content, turns[1].Content}; got[0] != "hello" || got[1] != "hi there" {
		t.Errorf("contents = %v", got)
	}
	for _, turn := range turns {
		if turn.SessionID != "sess-1" {
			t.Errorf("session = %q", turn.SessionID)
		}
		if turn.ProjectID != "proj://demo" {
			t.Errorf("project = %q", turn.ProjectID)
		}
		if turn.SourceID != "claude-code" || turn.MachineID != "machine-1" {
			t.Errorf("source/machine = %q/%q", turn.SourceID, turn.MachineID)
		}
		if len(turn.TurnID) != 24 {
			t.Errorf("turn id length = %d", len(turn.TurnID))
		}
	}
}

func TestParseFactoryDroid(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "droid.jsonl")
	writeJSONL(t, path, []string{
		`{"type":"session_start","id":"droid-1","cwd":"` + dir + `"}`,
		`{"type":"message","id":"m1","timestamp":"t1","message":{"role":"user","content":"build it"}}`,
		`{"type":"message","id":"m2","timestamp":"t2","message":{"role":"assistant","content":[{"type":"text","text":"done"}]}}`,
	})
	turns, _ := mustParse(t, path, "factory-droid", 0)
	if len(turns) != 2 || turns[0].Content != "build it" || turns[1].Content != "done" {
		t.Fatalf("turns = %+v", turns)
	}
	if turns[0].SessionID != "droid-1" || turns[0].Timestamp != "t1" {
		t.Errorf("session/timestamp = %q/%q", turns[0].SessionID, turns[0].Timestamp)
	}
}

func TestParseCodex(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "codex.jsonl")
	writeJSONL(t, path, []string{
		`{"type":"session_meta","payload":{"id":"codex-1","cwd":"` + dir + `"}}`,
		`{"type":"event_msg","payload":{"type":"user_message","message":"question"}}`,
		`{"type":"event_msg","payload":{"type":"agent_message","message":"answer"}}`,
		`{"type":"response_item","timestamp":"t2","payload":{"type":"message","role":"assistant","content":[{"type":"output_text","text":"second"}]}}`,
		`{"type":"response_item","payload":{"type":"function_call","name":"x"}}`,
	})
	turns, _ := mustParse(t, path, "codex", 0)
	if len(turns) != 3 {
		t.Fatalf("turns = %d, want 3: %+v", len(turns), turns)
	}
	if turns[0].Content != "question" || turns[1].Content != "answer" || turns[2].Content != "second" {
		t.Errorf("contents = %q %q %q", turns[0].Content, turns[1].Content, turns[2].Content)
	}
	if turns[2].Timestamp != "t2" {
		t.Errorf("timestamp = %q", turns[2].Timestamp)
	}
	if turns[0].SessionID != "codex-1" {
		t.Errorf("session = %q", turns[0].SessionID)
	}
}

func TestParsePi(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "pi.jsonl")
	writeJSONL(t, path, []string{
		`{"type":"session","id":"pi-1","cwd":"` + dir + `"}`,
		`{"type":"message","id":"p1","timestamp":"t1","message":{"role":"user","content":"hi pi"}}`,
		`{"type":"message","id":"p2","message":{"role":"assistant","content":"hi human"}}`,
		`{"type":"message","message":{"role":"tool","content":"ignored"}}`,
	})
	turns, _ := mustParse(t, path, "pi", 0)
	if len(turns) != 2 {
		t.Fatalf("turns = %d, want 2", len(turns))
	}
	if turns[0].Content != "hi pi" || turns[1].Content != "hi human" {
		t.Errorf("contents = %q %q", turns[0].Content, turns[1].Content)
	}
	if turns[0].SessionID != "pi-1" {
		t.Errorf("session = %q", turns[0].SessionID)
	}
}

func TestDeterministicTurnIDs(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "pi.jsonl")
	writeJSONL(t, path, []string{
		`{"type":"session","id":"pi-1","cwd":"` + dir + `"}`,
		`{"type":"message","message":{"role":"user","content":"same"}}`,
	})
	first, _ := mustParse(t, path, "pi", 0)
	second, _ := mustParse(t, path, "pi", 0)
	if len(first) != 1 || first[0].TurnID == "" {
		t.Fatalf("first = %+v", first)
	}
	if first[0].TurnID != second[0].TurnID {
		t.Errorf("ids differ: %s vs %s", first[0].TurnID, second[0].TurnID)
	}
}

func TestParseSkipsNoiseAndMalformedLines(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "pi.jsonl")
	writeJSONL(t, path, []string{
		`{"type":"session","id":"pi-1","cwd":"` + dir + `"}`,
		`not json at all`,
		`{"type":"message","message":{"role":"system","content":"ignored"}}`,
		`{"type":"message","message":{"role":"user","content":"kept"}}`,
	})
	turns, _ := mustParse(t, path, "pi", 0)
	if len(turns) != 1 || turns[0].Content != "kept" {
		t.Fatalf("turns = %+v", turns)
	}
}

func TestParseStripsIDENoise(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "pi.jsonl")
	writeJSONL(t, path, []string{
		`{"type":"message","message":{"role":"user","content":"<ide_opened_file>/tmp/x</ide_opened_file>please fix"}}`,
		`{"type":"message","message":{"role":"user","content":"<ide_selection>code</ide_selection>   "}}`,
	})
	turns, _ := mustParse(t, path, "pi", 0)
	if len(turns) != 1 || turns[0].Content != "please fix" {
		t.Fatalf("turns = %+v", turns)
	}
}

func TestParseRejectsUnknownHarness(t *testing.T) {
	if _, _, err := ParseTranscript("x.jsonl", "unknown", 0, "", ""); err == nil {
		t.Fatal("expected error for unsupported harness")
	}
}

func TestParseMissingFileIsEmpty(t *testing.T) {
	turns, next, err := ParseTranscript(filepath.Join(t.TempDir(), "missing.jsonl"), "pi", 5, "", "")
	if err != nil {
		t.Fatalf("err = %v", err)
	}
	if len(turns) != 0 || next != 5 {
		t.Errorf("turns=%d next=%d", len(turns), next)
	}
}

// --------------------------------------------------------------------------
// Incremental cursor
// --------------------------------------------------------------------------

func TestPartialTrailingLineNotConsumed(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "pi.jsonl")
	complete := `{"type":"session","id":"s","cwd":"` + dir + `"}` + "\n" +
		`{"type":"message","message":{"role":"user","content":"a"}}` + "\n" +
		`{"type":"message","message":{"role":"assistant","content":"b"}}` + "\n"
	if err := os.WriteFile(path, []byte(complete), 0o644); err != nil {
		t.Fatal(err)
	}

	turns, next := mustParse(t, path, "pi", 0)
	if len(turns) != 2 || next != 3 {
		t.Fatalf("turns=%d next=%d", len(turns), next)
	}

	// A partially written line appears; it must not advance the cursor.
	partial := complete + `{"type":"message","message":{"role":"user","content":"c`
	if err := os.WriteFile(path, []byte(partial), 0o644); err != nil {
		t.Fatal(err)
	}
	turns, next = mustParse(t, path, "pi", next)
	if len(turns) != 0 || next != 3 {
		t.Fatalf("after partial: turns=%d next=%d", len(turns), next)
	}

	// Once terminated, the line is complete and shipped.
	if err := os.WriteFile(path, []byte(partial+"\"}}\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	turns, next = mustParse(t, path, "pi", next)
	if len(turns) != 1 || turns[0].Content != "c" || next != 4 {
		t.Fatalf("after completion: %+v next=%d", turns, next)
	}
}

func TestStartOffsetStillReadsHeader(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "droid.jsonl")
	writeJSONL(t, path, []string{
		`{"type":"session_start","id":"sess-9","cwd":"` + dir + `"}`,
		`{"type":"message","message":{"role":"user","content":"hi"}}`,
		`{"type":"message","message":{"role":"assistant","content":"yo"}}`,
	})
	turns, next := mustParse(t, path, "factory-droid", 1)
	if len(turns) != 2 || next != 3 {
		t.Fatalf("turns=%d next=%d", len(turns), next)
	}
	if turns[0].SessionID != "sess-9" {
		t.Errorf("session = %q, want sess-9", turns[0].SessionID)
	}
}

// --------------------------------------------------------------------------
// Discovery
// --------------------------------------------------------------------------

func TestDiscoverTranscripts(t *testing.T) {
	home := t.TempDir()
	files := map[string]string{
		filepath.Join(home, ".claude", "projects", "p", "one.jsonl"):  `{"type":"user","sessionId":"cc","cwd":"/tmp","message":{"role":"user","content":"a"}}`,
		filepath.Join(home, ".factory", "sessions", "two.jsonl"):      `{"type":"session_start","id":"fd","cwd":"/tmp"}`,
		filepath.Join(home, ".codex", "sessions", "three.jsonl"):      `{"type":"session_meta","payload":{"id":"cx","cwd":"/tmp"}}`,
		filepath.Join(home, ".pi", "agent", "sessions", "four.jsonl"): `{"type":"session","id":"pi","cwd":"/tmp"}`,
	}
	for path, record := range files {
		writeJSONL(t, path, []string{record})
	}
	writeJSONL(t, filepath.Join(home, ".codex", "sessions", "decoy.txt"), []string{"{}"})

	infos, err := DiscoverTranscripts("", home)
	if err != nil {
		t.Fatal(err)
	}
	if len(infos) != 4 {
		t.Fatalf("infos = %d, want 4", len(infos))
	}
	order := []string{"claude-code", "codex", "factory-droid", "pi"}
	for i, want := range order {
		if infos[i].SourceID != want {
			t.Errorf("infos[%d].SourceID = %q, want %q", i, infos[i].SourceID, want)
		}
	}

	onlyCodex, err := DiscoverTranscripts("codex", home)
	if err != nil {
		t.Fatal(err)
	}
	if len(onlyCodex) != 1 || onlyCodex[0].SourceID != "codex" || onlyCodex[0].SessionID != "cx" {
		t.Errorf("codex only = %+v", onlyCodex)
	}

	empty, err := DiscoverTranscripts("pi", filepath.Join(home, "nope"))
	if err != nil || len(empty) != 0 {
		t.Errorf("empty = %+v err=%v", empty, err)
	}
}

// --------------------------------------------------------------------------
// Hook payloads
// --------------------------------------------------------------------------

func TestHookUserPromptSubmit(t *testing.T) {
	payload := map[string]any{
		"hook_event_name": "UserPromptSubmit",
		"session_id":      "sess",
		"cwd":             t.TempDir(),
		"prompt":          "do the thing",
		"timestamp":       "t",
	}
	turns, err := ParseHookPayload("factory-droid", payload, "m1", "demo")
	if err != nil {
		t.Fatal(err)
	}
	if len(turns) != 1 {
		t.Fatalf("turns = %+v", turns)
	}
	turn := turns[0]
	if turn.Role != "user" || turn.Content != "do the thing" || turn.SourceID != "factory-droid" {
		t.Errorf("turn = %+v", turn)
	}
	if turn.ProjectID != "proj://demo" || turn.SessionID != "sess" || turn.MachineID != "m1" {
		t.Errorf("turn = %+v", turn)
	}
}

func TestHookStopInlineResponse(t *testing.T) {
	payload := map[string]any{
		"event":             "Stop",
		"conversation_id":   "c1",
		"cwd":               t.TempDir(),
		"response":          "final answer",
		"assistant_message": "ignored because response wins",
	}
	turns, err := ParseHookPayload("pi", payload, "m1", "demo")
	if err != nil {
		t.Fatal(err)
	}
	if len(turns) != 1 || turns[0].Role != "assistant" || turns[0].Content != "final answer" {
		t.Fatalf("turns = %+v", turns)
	}
}

func TestHookStopFallsBackToTranscript(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "claude.jsonl")
	writeJSONL(t, path, []string{
		`{"type":"user","sessionId":"s","uuid":"u1","cwd":"` + dir + `","message":{"role":"user","content":"question"}}`,
		`{"type":"assistant","sessionId":"s","uuid":"a1","cwd":"` + dir + `","message":{"role":"assistant","content":"the answer"}}`,
		`{"type":"assistant","sessionId":"s","uuid":"a2","cwd":"` + dir + `","message":{"role":"assistant","content":"the last answer"}}`,
	})
	payload := map[string]any{
		"hook_event_name": "Stop",
		"session_id":      "s",
		"cwd":             dir,
		"transcript_path": path,
	}
	turns, err := ParseHookPayload("claude-code", payload, "m1", "demo")
	if err != nil {
		t.Fatal(err)
	}
	if len(turns) != 1 || turns[0].Role != "assistant" || turns[0].Content != "the last answer" {
		t.Fatalf("turns = %+v", turns)
	}
}

func TestHookWithoutDestinationHasNoTurns(t *testing.T) {
	turns, err := ParseHookPayload("pi", map[string]any{"event": "Notification"}, "m1", "")
	if err != nil {
		t.Fatal(err)
	}
	if len(turns) != 0 {
		t.Errorf("turns = %+v", turns)
	}
}

func TestSupportedHarnessesList(t *testing.T) {
	want := strings.Join([]string{"claude-code", "factory-droid", "codex", "pi"}, ",")
	if got := strings.Join(SupportedHarnesses, ","); got != want {
		t.Errorf("SupportedHarnesses = %q", got)
	}
	var turn Turn
	raw, err := json.Marshal(turn)
	if err != nil || !strings.Contains(string(raw), `"turn_id"`) {
		t.Errorf("Turn json = %s err=%v", raw, err)
	}
}
