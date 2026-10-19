// Package transcript reads the local transcript files written by coding
// harnesses and turns them into ship-ready Turn values.
//
// It is the pure-Go counterpart of simple_mem/edge.py's parsers: four formats
// (claude-code, factory-droid, codex and pi), incremental line-cursor shipping,
// transcript discovery and live hook capture.
package transcript

import (
	"bytes"
	"crypto/sha1"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"time"

	"simple-mem/internal/project"
)

// SupportedHarnesses lists every source_id ParseTranscript understands.
var SupportedHarnesses = []string{"claude-code", "factory-droid", "codex", "pi"}

// sessionDirs maps a harness to its transcript root under the home directory.
var sessionDirs = map[string][]string{
	"claude-code":   {".claude", "projects"},
	"factory-droid": {".factory", "sessions"},
	"codex":         {".codex", "sessions"},
	"pi":            {".pi", "agent", "sessions"},
}

// Turn is one user or assistant message extracted from a transcript.
type Turn struct {
	TurnID    string `json:"turn_id"`
	ProjectID string `json:"project_id"`
	SessionID string `json:"session_id"`
	SourceID  string `json:"source_id"`
	MachineID string `json:"machine_id"`
	Role      string `json:"role"`
	Content   string `json:"content"`
	Timestamp string `json:"timestamp"`
	Cwd       string `json:"cwd,omitempty"`
}

// TranscriptInfo describes a discovered transcript file.
type TranscriptInfo struct {
	SourceID  string `json:"source_id"`
	SessionID string `json:"session_id"`
	Path      string `json:"path"`
	Cwd       string `json:"cwd"`
	ProjectID string `json:"project_id"`
}

var (
	ideOpenedFile = regexp.MustCompile(`(?s)<ide_opened_file>.*?</ide_opened_file>`)
	ideSelection  = regexp.MustCompile(`(?s)<ide_selection>.*?</ide_selection>`)
)

// ParseTranscript extracts the not-yet-shipped turns from one transcript.
//
// startOffset is the number of lines already shipped; only turns after that
// are returned, while the header lines are still scanned so the session id and
// cwd they carry are recovered. The returned int64 is the new line cursor: the
// count of complete, newline-terminated lines, so a partially written trailing
// line is never consumed.
func ParseTranscript(path string, sourceID string, startOffset int64, machineID string, projectID string) ([]Turn, int64, error) {
	turns, next, _, _, err := parseFile(path, sourceID, startOffset, machineID, projectID)
	return turns, next, err
}

// DiscoverTranscripts finds every transcript under the harness session
// directories. An empty source (or "all") covers every supported harness.
func DiscoverTranscripts(source string, homeDir string) ([]TranscriptInfo, error) {
	home := homeDir
	if strings.TrimSpace(home) == "" {
		resolved, err := os.UserHomeDir()
		if err != nil {
			return nil, err
		}
		home = resolved
	}
	home = expandHome(home)

	var sources []string
	if source == "" || source == "all" {
		sources = append(sources, SupportedHarnesses...)
	} else if isSupported(source) {
		sources = []string{source}
	}

	var infos []TranscriptInfo
	for _, sourceID := range sources {
		parts := sessionDirs[sourceID]
		root := filepath.Join(append([]string{home}, parts...)...)
		if info, err := os.Stat(root); err != nil || !info.IsDir() {
			continue
		}
		_ = filepath.WalkDir(root, func(path string, entry fs.DirEntry, err error) error {
			if err != nil {
				return nil
			}
			if entry.IsDir() || !strings.HasSuffix(entry.Name(), ".jsonl") {
				return nil
			}
			turns, _, sessionID, cwd, _ := parseFile(path, sourceID, 0, "", "")
			info := TranscriptInfo{
				SourceID:  sourceID,
				SessionID: sessionID,
				Path:      path,
				Cwd:       cwd,
				ProjectID: project.ResolveProjectID(cwd, ""),
			}
			if len(turns) > 0 {
				if turns[0].SessionID != "" {
					info.SessionID = turns[0].SessionID
				}
				info.ProjectID = turns[0].ProjectID
			}
			infos = append(infos, info)
			return nil
		})
	}

	sort.Slice(infos, func(i, j int) bool {
		if infos[i].SourceID != infos[j].SourceID {
			return infos[i].SourceID < infos[j].SourceID
		}
		return infos[i].Path < infos[j].Path
	})
	return infos, nil
}

// ParseHookPayload turns one live harness hook payload into turns.
//
// UserPromptSubmit yields the submitted prompt. Stop (or any payload carrying
// a response) yields the assistant reply, falling back to the last assistant
// turn of transcript_path when no inline response is present.
func ParseHookPayload(harness string, payload map[string]any, machineID string, projectID string) ([]Turn, error) {
	if payload == nil {
		payload = map[string]any{}
	}
	eventName := firstString(payload, "hook_event_name", "event")
	sessionID := firstString(payload, "session_id", "conversation_id")
	if sessionID == "" {
		sessionID = "hook-session"
	}
	cwd := firstString(payload, "cwd", "workspace")
	if cwd == "" {
		if working, err := os.Getwd(); err == nil {
			cwd = working
		}
	}
	resolvedProject := project.ResolveProjectID(cwd, projectID)
	createdAt := firstString(payload, "timestamp")
	if createdAt == "" {
		createdAt = time.Now().UTC().Format(time.RFC3339)
	}

	var turns []Turn
	if eventName == "UserPromptSubmit" || keyPresent(payload, "prompt") {
		if prompt, ok := payload["prompt"].(string); ok && strings.TrimSpace(prompt) != "" {
			turn := hookTurn(harness, sessionID, "user", prompt, resolvedProject, createdAt, machineID)
			turn.Cwd = cwd
			turns = append(turns, turn)
		}
	}

	wantsAssistant := eventName == "Stop" ||
		keyPresent(payload, "response") ||
		keyPresent(payload, "assistant_message") ||
		keyPresent(payload, "transcript_path")
	if wantsAssistant {
		text := firstString(payload, "response", "assistant_message", "assistant")
		switch {
		case strings.TrimSpace(text) == "" && keyPresent(payload, "transcript_path"):
			transcriptPath := firstString(payload, "transcript_path")
			if transcriptPath != "" {
				path := expandHome(transcriptPath)
				if info, err := os.Stat(path); err == nil && info.Mode().IsRegular() {
					extracted, _, _, _, _ := parseFile(path, harness, 0, machineID, projectID)
					var assistants []Turn
					for _, turn := range extracted {
						if turn.Role == "assistant" {
							assistants = append(assistants, turn)
						}
					}
					if len(assistants) > 0 {
						latest := assistants[len(assistants)-1]
						turnSession := latest.SessionID
						if turnSession == "" {
							turnSession = sessionID
						}
						turnTimestamp := latest.Timestamp
						if turnTimestamp == "" {
							turnTimestamp = createdAt
						}
						turnProject := latest.ProjectID
						if turnProject == "" {
							turnProject = resolvedProject
						}
						turn := hookTurn(harness, turnSession, "assistant", latest.Content, turnProject, turnTimestamp, machineID)
						turn.Cwd = latest.Cwd
						if turn.Cwd == "" {
							turn.Cwd = cwd
						}
						turns = append(turns, turn)
					}
				}
			}
		case strings.TrimSpace(text) != "":
			turn := hookTurn(harness, sessionID, "assistant", text, resolvedProject, createdAt, machineID)
			turn.Cwd = cwd
			turns = append(turns, turn)
		}
	}

	if turns == nil {
		turns = []Turn{}
	}
	return turns, nil
}

// --------------------------------------------------------------------------
// File parsing
// --------------------------------------------------------------------------

type transcriptState struct {
	sessionID string
	cwd       string
}

func parseFile(path string, sourceID string, startOffset int64, machineID string, projectID string) ([]Turn, int64, string, string, error) {
	if !isSupported(sourceID) {
		return nil, startOffset, "", "", fmt.Errorf("transcript: unsupported harness %q", sourceID)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		// A missing or unreadable transcript simply has nothing to ship.
		return []Turn{}, startOffset, "", "", nil
	}

	lines := completeLines(data)
	nextOffset := int64(len(lines))
	fallbackSession := sourceID + ":" + strings.TrimSuffix(filepath.Base(path), filepath.Ext(path))

	state := transcriptState{}
	turns := []Turn{}
	for index, line := range lines {
		lineNumber := int64(index + 1)
		trimmed := strings.TrimSpace(line)
		if trimmed == "" {
			continue
		}
		var record map[string]any
		if err := json.Unmarshal([]byte(trimmed), &record); err != nil {
			continue
		}
		parsed := parseRecord(sourceID, record)
		if parsed.sessionID != "" {
			state.sessionID = parsed.sessionID
		}
		if parsed.cwd != "" {
			state.cwd = parsed.cwd
		}
		if !parsed.ok || parsed.content == "" {
			continue
		}
		if lineNumber <= startOffset {
			continue
		}
		sessionID := state.sessionID
		if sessionID == "" {
			sessionID = fallbackSession
		}
		turns = append(turns, Turn{
			TurnID:    deterministicTurnID(sourceID, sessionID, lineNumber, parsed.role, parsed.content),
			ProjectID: project.ResolveProjectID(state.cwd, projectID),
			SessionID: sessionID,
			SourceID:  sourceID,
			MachineID: machineID,
			Role:      parsed.role,
			Content:   parsed.content,
			Timestamp: parsed.timestamp,
			Cwd:       state.cwd,
		})
	}
	return turns, nextOffset, state.sessionID, state.cwd, nil
}

// completeLines returns every '\n'-terminated line without the terminator; a
// partially written trailing line is dropped.
func completeLines(data []byte) []string {
	var lines []string
	start := 0
	for i := 0; i < len(data); i++ {
		if data[i] != '\n' {
			continue
		}
		lines = append(lines, string(bytes.TrimSuffix(data[start:i], []byte("\r"))))
		start = i + 1
	}
	return lines
}

type parsedLine struct {
	role      string
	content   string
	timestamp string
	sessionID string
	cwd       string
	ok        bool
}

func parseRecord(sourceID string, record map[string]any) parsedLine {
	switch sourceID {
	case "claude-code":
		return parseClaudeRecord(record)
	case "factory-droid":
		return parseSimpleRecord(record, "session_start")
	case "codex":
		return parseCodexRecord(record)
	default:
		return parseSimpleRecord(record, "session")
	}
}

func parseClaudeRecord(record map[string]any) parsedLine {
	parsed := parsedLine{
		sessionID: stringValue(record, "sessionId"),
		cwd:       stringValue(record, "cwd"),
	}
	recordType := stringValue(record, "type")
	if recordType != "user" && recordType != "assistant" {
		return parsed
	}
	message, _ := record["message"].(map[string]any)
	var content any
	if message != nil {
		content = message["content"]
	}
	parsed.role = recordType
	parsed.content = textFromContent(content)
	parsed.timestamp = stringValue(record, "timestamp")
	parsed.ok = true
	return parsed
}

func parseSimpleRecord(record map[string]any, sessionType string) parsedLine {
	parsed := parsedLine{cwd: stringValue(record, "cwd")}
	recordType := stringValue(record, "type")
	if recordType == sessionType {
		parsed.sessionID = stringValue(record, "id")
		if parsed.sessionID == "" {
			parsed.sessionID = stringValue(record, "sessionId")
		}
		return parsed
	}
	if recordType != "message" {
		return parsed
	}
	message, _ := record["message"].(map[string]any)
	if message == nil {
		return parsed
	}
	role := stringValue(message, "role")
	if role != "user" && role != "assistant" {
		return parsed
	}
	parsed.role = role
	parsed.content = textFromContent(message["content"])
	parsed.timestamp = stringValue(record, "timestamp")
	parsed.ok = true
	return parsed
}

func parseCodexRecord(record map[string]any) parsedLine {
	parsed := parsedLine{}
	recordType := stringValue(record, "type")
	payload, _ := record["payload"].(map[string]any)

	switch recordType {
	case "session_meta":
		if payload != nil {
			parsed.sessionID = stringValue(payload, "id")
			parsed.cwd = stringValue(payload, "cwd")
		}
	case "event_msg":
		if payload == nil {
			return parsed
		}
		payloadType := stringValue(payload, "type")
		switch payloadType {
		case "user_message":
			parsed.role = "user"
		case "agent_message":
			parsed.role = "assistant"
		default:
			return parsed
		}
		parsed.content = textFromContent(firstValue(payload, "message", "text"))
		parsed.timestamp = firstString(payload, "timestamp")
		parsed.ok = true
	case "response_item":
		if payload == nil || stringValue(payload, "type") != "message" {
			return parsed
		}
		role := stringValue(payload, "role")
		if role != "user" && role != "assistant" {
			return parsed
		}
		parsed.role = role
		parsed.content = textFromContent(payload["content"])
		parsed.timestamp = firstString(record, "timestamp")
		if parsed.timestamp == "" {
			parsed.timestamp = firstString(payload, "timestamp")
		}
		parsed.ok = true
	}
	return parsed
}

// textFromContent extracts the human-readable text of a message body,
// ignoring thinking, tool_use and tool_result blocks.
func textFromContent(content any) string {
	switch value := content.(type) {
	case string:
		return cleanContent(value)
	case []any:
		var parts []string
		for _, item := range value {
			switch block := item.(type) {
			case string:
				if text := strings.TrimSpace(block); text != "" {
					parts = append(parts, text)
				}
			case map[string]any:
				blockType := stringValue(block, "type")
				text, _ := block["text"].(string)
				if blockType == "text" || blockType == "input_text" || blockType == "output_text" || (blockType == "" && text != "") {
					if trimmed := strings.TrimSpace(text); trimmed != "" {
						parts = append(parts, trimmed)
					}
				}
			}
		}
		return cleanContent(strings.Join(parts, "\n"))
	default:
		return ""
	}
}

// cleanContent trims a message and drops the IDE context tags harnesses splice
// into otherwise ordinary prompts.
func cleanContent(content string) string {
	content = ideOpenedFile.ReplaceAllString(content, "")
	content = ideSelection.ReplaceAllString(content, "")
	return strings.TrimSpace(content)
}

func deterministicTurnID(sourceID, sessionID string, offset int64, role string, content string) string {
	if len(content) > 128 {
		content = content[:128]
	}
	sum := sha1.Sum([]byte(sourceID + ":" + sessionID + ":" + strconv.FormatInt(offset, 10) + ":" + role + ":" + content))
	return hex.EncodeToString(sum[:])[:24]
}

func hookTurn(sourceID, sessionID, role, content, projectID, timestamp, machineID string) Turn {
	verbatim := strings.TrimSpace(content)
	return Turn{
		TurnID:    deterministicTurnID(sourceID, sessionID, 0, role, verbatim),
		ProjectID: projectID,
		SessionID: sessionID,
		SourceID:  sourceID,
		MachineID: machineID,
		Role:      role,
		Content:   verbatim,
		Timestamp: timestamp,
	}
}

// --------------------------------------------------------------------------
// Helpers
// --------------------------------------------------------------------------

func isSupported(sourceID string) bool {
	for _, harness := range SupportedHarnesses {
		if harness == sourceID {
			return true
		}
	}
	return false
}

func stringValue(record map[string]any, key string) string {
	if record == nil {
		return ""
	}
	value, ok := record[key].(string)
	if !ok {
		return ""
	}
	return value
}

func firstValue(record map[string]any, keys ...string) any {
	for _, key := range keys {
		if value, ok := record[key]; ok && value != nil {
			return value
		}
	}
	return nil
}

func firstString(record map[string]any, keys ...string) string {
	for _, key := range keys {
		if value := stringValue(record, key); value != "" {
			return value
		}
	}
	return ""
}

func keyPresent(record map[string]any, key string) bool {
	_, ok := record[key]
	return ok
}

func expandHome(path string) string {
	if path != "~" && !strings.HasPrefix(path, "~/") {
		return path
	}
	home, err := os.UserHomeDir()
	if err != nil || home == "" {
		return path
	}
	if path == "~" {
		return home
	}
	return filepath.Join(home, path[2:])
}
