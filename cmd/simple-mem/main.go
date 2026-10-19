// Command simple-mem is the Go edge client: the local end of the Simple Mem
// memory split. It resolves project identities, ships transcripts and hook
// events to the Hub, speaks MCP to an agent host, and answers a few direct
// memory queries.
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	"simple-mem/internal/mcp"
	"simple-mem/internal/project"
	"simple-mem/internal/shipper"
	"simple-mem/internal/transcript"
)

const (
	defaultRemoteURL = "http://127.0.0.1:8765"
	defaultInterval  = 30.0
)

func main() {
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	os.Exit(Run(ctx, os.Args[1:], os.Stdin, os.Stdout, os.Stderr))
}

// Run executes one command and returns the process exit code.
func Run(ctx context.Context, args []string, stdin io.Reader, stdout, stderr io.Writer) int {
	if len(args) == 0 {
		usage(stderr)
		return 2
	}
	command, rest := args[0], args[1:]
	switch command {
	case "project-id":
		return runProjectID(rest, stdout, stderr)
	case "ship":
		return runShip(ctx, rest, stdout, stderr)
	case "capture":
		return runCapture(ctx, rest, stdin, stdout, stderr)
	case "mcp":
		return runMCP(ctx, rest, stdin, stdout, stderr)
	case "search":
		return runSearch(ctx, rest, stdout, stderr)
	case "briefing":
		return runBriefing(ctx, rest, stdout, stderr)
	case "thread":
		return runThread(ctx, rest, stdout, stderr)
	case "remember":
		return runRemember(ctx, rest, stdout, stderr)
	case "forget":
		return runForget(ctx, rest, stdout, stderr)
	case "edge-config", "init-edge":
		return runEdgeConfig(rest, stdout, stderr)
	default:
		usage(stderr)
		return 2
	}
}

func usage(stream io.Writer) {
	fmt.Fprint(stream, `Usage: simple-mem <command> [options]
  project-id [--cwd DIR] [--project-id ID] [--json]
  ship [--source all|claude-code|factory-droid|codex|pi] [--transcript PATH]
       [--remote-url URL] [--api-key KEY] [--machine-id ID] [--project-id ID]
       [--cursor-file PATH] [--watch] [--interval SEC] [--json]
  capture <harness> [--remote-url URL] [--api-key KEY] [--machine-id ID]
          [--project-id ID] [--strict]
  mcp [--remote-url URL] [--api-key KEY] [--cwd DIR] [--project-id ID]
  search <query> [--remote-url URL] [--api-key KEY] [--cwd DIR]
         [--project-id ID] [--scope workspace|all] [--top-k N] [--json]
  briefing [--cwd DIR] [--limit N] [--remote-url URL] [--api-key KEY]
           [--project-id ID] [--json]
  thread <conversation_id> [--around MSG_ID] [--limit N] [--remote-url URL]
         [--api-key KEY] [--json]
  remember <statement> [--remote-url URL] [--api-key KEY] [--cwd DIR]
           [--project-id ID] [--kind fact|decision|preference|experience]
           [--details TEXT] [--scope workspace|global] [--json]
  forget <memory_id> [--remote-url URL] [--api-key KEY] [--json]
  edge-config [--remote-url URL] [--api-key KEY] [--machine-id ID]
              [--bin-path PATH] [--write-claude] [--write-droid]
`)
}

// --------------------------------------------------------------------------
// Shared flag plumbing
// --------------------------------------------------------------------------

type commonFlags struct {
	remoteURL *string
	apiKey    *string
	machineID *string
	projectID *string
}

func addCommon(fs *flag.FlagSet) commonFlags {
	return commonFlags{
		remoteURL: fs.String("remote-url", envOr("SIMPLE_MEM_REMOTE_URL", defaultRemoteURL), "Hub base URL"),
		apiKey:    fs.String("api-key", os.Getenv("SIMPLE_MEM_API_KEY"), "Hub API key"),
		machineID: fs.String("machine-id", envOr("SIMPLE_MEM_MACHINE_ID", hostname()), "stable machine id"),
		projectID: fs.String("project-id", os.Getenv("SIMPLE_MEM_PROJECT_ID"), "explicit project id"),
	}
}

func addJSON(fs *flag.FlagSet) *bool {
	return fs.Bool("json", false, "emit JSON")
}

func parse(fs *flag.FlagSet, args []string) bool {
	fs.SetOutput(io.Discard)
	fs.Usage = func() {}
	if err := fs.Parse(args); err != nil {
		return false
	}
	return fs.NArg() == 0
}

func splitPositional(args []string) (string, []string, bool) {
	if len(args) == 0 || strings.HasPrefix(args[0], "-") {
		return "", args, false
	}
	return args[0], args[1:], true
}

func envOr(key, fallback string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return fallback
}

func hostname() string {
	if host, err := os.Hostname(); err == nil && host != "" {
		return host
	}
	return "unknown"
}

func transcriptSupported(harness string) bool {
	for _, candidate := range transcript.SupportedHarnesses {
		if candidate == harness {
			return true
		}
	}
	return false
}

func currentCwd(explicit string) string {
	if explicit != "" {
		return explicit
	}
	if declared := os.Getenv("WORKSPACE_FOLDER_PATHS"); declared != "" {
		return declared
	}
	if wd, err := os.Getwd(); err == nil {
		return wd
	}
	return ""
}

func writeJSON(stream io.Writer, payload any) {
	encoded, err := json.Marshal(payload)
	if err != nil {
		fmt.Fprintf(stream, "simple-mem: %v\n", err)
		return
	}
	fmt.Fprintln(stream, string(encoded))
}

// --------------------------------------------------------------------------
// project-id
// --------------------------------------------------------------------------

func runProjectID(args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet("project-id", flag.ContinueOnError)
	cwd := fs.String("cwd", "", "directory to resolve")
	projectID := fs.String("project-id", os.Getenv("SIMPLE_MEM_PROJECT_ID"), "explicit project id")
	asJSON := addJSON(fs)
	if !parse(fs, args) {
		fmt.Fprintln(stderr, "simple-mem: invalid options for project-id")
		return 2
	}
	resolved := project.ResolveProjectID(currentCwd(*cwd), *projectID)
	if *asJSON {
		writeJSON(stdout, map[string]any{"project_id": resolved})
	} else {
		fmt.Fprintln(stdout, resolved)
	}
	return 0
}

// --------------------------------------------------------------------------
// ship
// --------------------------------------------------------------------------

func runShip(ctx context.Context, args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet("ship", flag.ContinueOnError)
	source := fs.String("source", "", "harness name or all")
	file := fs.String("transcript", "", "ship exactly this file")
	common := addCommon(fs)
	cursor := fs.String("cursor-file", "", "cursor file path")
	watch := fs.Bool("watch", false, "repeat forever")
	interval := fs.Float64("interval", defaultInterval, "seconds between watch passes")
	asJSON := addJSON(fs)
	if !parse(fs, args) {
		fmt.Fprintln(stderr, "simple-mem: invalid options for ship")
		return 2
	}

	selected := *source
	if selected == "" {
		if *file != "" {
			selected = "claude-code"
		} else {
			selected = "all"
		}
	}
	if selected != "all" && !transcriptSupported(selected) {
		fmt.Fprintf(stderr, "simple-mem: unsupported source: %s\n", selected)
		return 2
	}
	options := shipper.ShipOptions{
		Source:     selected,
		Transcript: *file,
		RemoteURL:  *common.remoteURL,
		APIKey:     *common.apiKey,
		MachineID:  *common.machineID,
		ProjectID:  *common.projectID,
		CursorFile: *cursor,
	}
	for {
		code := shipOnce(ctx, options, *asJSON, stdout, stderr)
		if code != 0 || !*watch {
			return code
		}
		select {
		case <-ctx.Done():
			return 0
		case <-time.After(time.Duration(*interval * float64(time.Second))):
		}
	}
}

func shipOnce(ctx context.Context, options shipper.ShipOptions, asJSON bool, stdout, stderr io.Writer) int {
	if options.Transcript != "" {
		result, err := shipper.ShipTranscript(ctx, options)
		if err != nil {
			fmt.Fprintf(stderr, "simple-mem: %v\n", err)
			return 1
		}
		reportShip(asJSON, result, result.Shipped, stdout)
		return 0
	}
	result, err := shipper.ShipAll(ctx, options)
	if err != nil {
		fmt.Fprintf(stderr, "simple-mem: %v\n", err)
		return 1
	}
	reportShip(asJSON, result, result.Shipped, stdout)
	for _, message := range result.Errors {
		fmt.Fprintf(stderr, "simple-mem: %s\n", message)
	}
	if len(result.Errors) > 0 {
		return 1
	}
	return 0
}

func reportShip(asJSON bool, payload any, shipped int, stdout io.Writer) {
	if asJSON {
		writeJSON(stdout, payload)
		return
	}
	fmt.Fprintf(stdout, "shipped %d messages\n", shipped)
}

// --------------------------------------------------------------------------
// capture
// --------------------------------------------------------------------------

func runCapture(ctx context.Context, args []string, stdin io.Reader, stdout, stderr io.Writer) int {
	harness, rest, ok := splitPositional(args)
	if !ok {
		fmt.Fprintln(stderr, "simple-mem: capture requires a harness")
		return 2
	}
	if !transcriptSupported(harness) {
		fmt.Fprintf(stderr, "simple-mem: unsupported harness: %s\n", harness)
		return 2
	}
	fs := flag.NewFlagSet("capture", flag.ContinueOnError)
	common := addCommon(fs)
	strict := fs.Bool("strict", false, "fail when the Hub is unreachable")
	if !parse(fs, rest) {
		fmt.Fprintln(stderr, "simple-mem: invalid options for capture")
		return 2
	}

	raw, err := io.ReadAll(stdin)
	if err != nil {
		fmt.Fprintf(stderr, "simple-mem: %v\n", err)
		return 2
	}
	payload := map[string]any{}
	if len(strings.TrimSpace(string(raw))) > 0 {
		decoder := json.NewDecoder(strings.NewReader(string(raw)))
		decoder.UseNumber()
		if err := decoder.Decode(&payload); err != nil {
			fmt.Fprintln(stderr, "simple-mem: capture expects JSON on stdin")
			return 2
		}
	}
	result, err := shipper.CaptureHookEvent(ctx, shipper.CaptureOptions{
		Harness:   harness,
		Payload:   payload,
		RemoteURL: *common.remoteURL,
		APIKey:    *common.apiKey,
		MachineID: *common.machineID,
		ProjectID: *common.projectID,
		Strict:    *strict,
	})
	if err != nil {
		fmt.Fprintf(stderr, "simple-mem: %v\n", err)
		if *strict {
			return 1
		}
		return 0
	}
	writeJSON(stdout, result)
	return 0
}

// --------------------------------------------------------------------------
// mcp
// --------------------------------------------------------------------------

func runMCP(ctx context.Context, args []string, stdin io.Reader, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet("mcp", flag.ContinueOnError)
	common := addCommon(fs)
	cwd := fs.String("cwd", "", "directory to resolve to a project id")
	if !parse(fs, args) {
		fmt.Fprintln(stderr, "simple-mem: invalid options for mcp")
		return 2
	}
	server := mcp.NewServer(*common.remoteURL, *common.apiKey, currentCwd(*cwd), *common.projectID)
	if err := server.ServeStdio(ctx, stdin, stdout); err != nil {
		fmt.Fprintf(stderr, "simple-mem: %v\n", err)
		return 1
	}
	return 0
}

// --------------------------------------------------------------------------
// search / remember / forget
// --------------------------------------------------------------------------

func runSearch(ctx context.Context, args []string, stdout, stderr io.Writer) int {
	query, rest, ok := splitPositional(args)
	if !ok {
		fmt.Fprintln(stderr, "simple-mem: search requires a query")
		return 2
	}
	fs := flag.NewFlagSet("search", flag.ContinueOnError)
	common := addCommon(fs)
	cwd := fs.String("cwd", "", "directory to resolve to a project id")
	scope := fs.String("scope", "workspace", "workspace or all")
	topK := fs.Int("top-k", 10, "maximum memories")
	asJSON := addJSON(fs)
	if !parse(fs, rest) {
		fmt.Fprintln(stderr, "simple-mem: invalid options for search")
		return 2
	}
	body := map[string]any{
		"query":      query,
		"scope":      *scope,
		"limit":      *topK,
		"project_id": project.ResolveProjectID(currentCwd(*cwd), *common.projectID),
	}
	client := shipper.NewClient(*common.remoteURL, *common.apiKey)
	result, err := client.Search(ctx, body)
	if err != nil {
		fmt.Fprintf(stderr, "simple-mem: %v\n", err)
		return 1
	}
	if *asJSON {
		writeJSON(stdout, result)
	} else {
		fmt.Fprintln(stdout, searchSummary(result))
	}
	return 0
}

func runRemember(ctx context.Context, args []string, stdout, stderr io.Writer) int {
	statement, rest, ok := splitPositional(args)
	if !ok {
		fmt.Fprintln(stderr, "simple-mem: remember requires a statement")
		return 2
	}
	fs := flag.NewFlagSet("remember", flag.ContinueOnError)
	common := addCommon(fs)
	cwd := fs.String("cwd", "", "directory to resolve to a project id")
	kind := fs.String("kind", "fact", "fact, decision, preference or experience")
	details := fs.String("details", "", "supporting detail")
	scope := fs.String("scope", "workspace", "workspace or global")
	asJSON := addJSON(fs)
	if !parse(fs, rest) {
		fmt.Fprintln(stderr, "simple-mem: invalid options for remember")
		return 2
	}
	projectID := project.ResolveProjectID(currentCwd(*cwd), *common.projectID)
	if *scope == "global" {
		projectID = "global"
	}
	body := map[string]any{
		"statement":  statement,
		"kind":       *kind,
		"details":    *details,
		"scope":      *scope,
		"project_id": projectID,
		"source_id":  "cli",
	}
	client := shipper.NewClient(*common.remoteURL, *common.apiKey)
	result, err := client.Remember(ctx, body)
	if err != nil {
		fmt.Fprintf(stderr, "simple-mem: %v\n", err)
		return 1
	}
	if *asJSON {
		writeJSON(stdout, result)
	} else if id, ok := result["memory_id"].(string); ok {
		fmt.Fprintf(stdout, "remembered %s\n", id)
	} else {
		writeJSON(stdout, result)
	}
	return 0
}

func runForget(ctx context.Context, args []string, stdout, stderr io.Writer) int {
	memoryID, rest, ok := splitPositional(args)
	if !ok {
		fmt.Fprintln(stderr, "simple-mem: forget requires a memory id")
		return 2
	}
	fs := flag.NewFlagSet("forget", flag.ContinueOnError)
	common := addCommon(fs)
	asJSON := addJSON(fs)
	if !parse(fs, rest) {
		fmt.Fprintln(stderr, "simple-mem: invalid options for forget")
		return 2
	}
	client := shipper.NewClient(*common.remoteURL, *common.apiKey)
	result, err := client.Forget(ctx, memoryID)
	if err != nil {
		fmt.Fprintf(stderr, "simple-mem: %v\n", err)
		return 1
	}
	if *asJSON {
		writeJSON(stdout, result)
	} else if deleted, ok := result["deleted"].(bool); ok {
		fmt.Fprintf(stdout, "deleted=%t\n", deleted)
	} else {
		writeJSON(stdout, result)
	}
	return 0
}

func searchSummary(result map[string]any) string {
	if memories, ok := result["memories"].([]any); ok {
		return fmt.Sprintf("%d memories", len(memories))
	}
	encoded, err := json.Marshal(result)
	if err != nil {
		return "0 memories"
	}
	return string(encoded)
}

// --------------------------------------------------------------------------
// briefing / thread
// --------------------------------------------------------------------------

func runBriefing(ctx context.Context, args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet("briefing", flag.ContinueOnError)
	common := addCommon(fs)
	cwd := fs.String("cwd", "", "directory to resolve to a project id")
	limit := fs.Int("limit", 20, "maximum briefing items")
	asJSON := addJSON(fs)
	if !parse(fs, args) {
		fmt.Fprintln(stderr, "simple-mem: invalid options for briefing")
		return 2
	}
	projectID := project.ResolveProjectID(currentCwd(*cwd), *common.projectID)
	client := shipper.NewClient(*common.remoteURL, *common.apiKey)
	result, err := client.Briefing(ctx, projectID, *limit)
	if err != nil {
		fmt.Fprintf(stderr, "simple-mem: %v\n", err)
		return 1
	}
	if *asJSON {
		writeJSON(stdout, result)
	} else {
		fmt.Fprintln(stdout, briefingSummary(result))
	}
	return 0
}

func runThread(ctx context.Context, args []string, stdout, stderr io.Writer) int {
	conversationID, rest, ok := splitPositional(args)
	if !ok {
		fmt.Fprintln(stderr, "simple-mem: thread requires a conversation id")
		return 2
	}
	fs := flag.NewFlagSet("thread", flag.ContinueOnError)
	common := addCommon(fs)
	around := fs.String("around", "", "centre the window on this message id")
	limit := fs.Int("limit", 20, "maximum messages")
	asJSON := addJSON(fs)
	if !parse(fs, rest) {
		fmt.Fprintln(stderr, "simple-mem: invalid options for thread")
		return 2
	}
	client := shipper.NewClient(*common.remoteURL, *common.apiKey)
	result, err := client.FetchThread(ctx, conversationID, *around, *limit)
	if err != nil {
		fmt.Fprintf(stderr, "simple-mem: %v\n", err)
		return 1
	}
	if *asJSON {
		writeJSON(stdout, result)
	} else {
		printThread(stdout, result)
	}
	return 0
}

func briefingSummary(result map[string]any) string {
	if text, ok := result["briefing"].(string); ok && strings.TrimSpace(text) != "" {
		return strings.TrimSpace(text)
	}
	var parts []string
	for _, key := range []string{"memories", "facts", "decisions", "preferences", "experiences", "recent_turns", "turns"} {
		if items, ok := result[key].([]any); ok {
			parts = append(parts, fmt.Sprintf("%d %s", len(items), key))
		}
	}
	if len(parts) > 0 {
		return "briefing: " + strings.Join(parts, ", ")
	}
	encoded, err := json.Marshal(result)
	if err != nil {
		return "briefing: empty"
	}
	return string(encoded)
}

func printThread(stream io.Writer, result map[string]any) {
	messages, ok := result["messages"].([]any)
	if !ok {
		messages, _ = result["turns"].([]any)
	}
	for _, item := range messages {
		message, ok := item.(map[string]any)
		if !ok {
			continue
		}
		role, _ := message["role"].(string)
		content, _ := message["content"].(string)
		fmt.Fprintf(stream, "[%s] %s\n", role, content)
	}
}

// --------------------------------------------------------------------------
// edge-config
// --------------------------------------------------------------------------

func runEdgeConfig(args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet("edge-config", flag.ContinueOnError)
	common := addCommon(fs)
	binPath := fs.String("bin-path", "simple-mem", "path to this binary")
	writeClaude := fs.Bool("write-claude", false, "write ~/.claude/settings.json")
	writeDroid := fs.Bool("write-droid", false, "write ~/.factory/mcp.json")
	if !parse(fs, args) {
		fmt.Fprintln(stderr, "simple-mem: invalid options for edge-config")
		return 2
	}
	cfg := mcp.GenerateEdgeConfig(*binPath, *common.remoteURL, *common.apiKey, *common.machineID)
	if *writeClaude || *writeDroid {
		if err := mcp.WriteEdgeConfigs(cfg, *writeClaude, *writeDroid, ""); err != nil {
			fmt.Fprintf(stderr, "simple-mem: %v\n", err)
			return 1
		}
	}
	writeJSON(stdout, cfg)
	return 0
}
