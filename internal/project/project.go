// Package project turns a working directory into the most stable cross-machine
// name it can justify.
//
// The four-tier waterfall mirrors simple_mem/project.py: a declared identity
// outranks the repository's own origin, which outranks its opaque root commit,
// which outranks the bare directory name.
//
//	Tier 0  a value already shaped like an identity (git://, proj://, ...)
//	Tier 1  an explicit declaration (SIMPLE_MEM_PROJECT_ID, .simple-mem.json)
//	Tier 2a the git remote URL, canonicalized (git://<host>/<path>)
//	Tier 2b the repository's root commit, when it has no remote
//	Tier 3 the directory's own name (dir://<name>)
package project

import (
	"bytes"
	"compress/zlib"
	"encoding/json"
	"io"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"
	"time"
)

const (
	globalID      = "global"
	homeWalkLimit = 4096
)

// canonicalPrefixes are values already shaped like an identity.
var canonicalPrefixes = []string{
	"git://", "git-root://", "proj://", "dir://", "global://",
}

// urlPrefixes name a repository but are handed to us in clone-URL form.
var urlPrefixes = []string{"https://", "http://", "git@", "ssh://"}

var (
	scpRemote = regexp.MustCompile(`^(?:[^@/]+@)?([^:/@]+):(.+)$`)
	hexSHA    = regexp.MustCompile(`^[0-9a-f]{7,64}$`)
)

// CanonicalizeGitURL renders any clone URL as git://<host>/<path>, folding the
// host and path to lower case. It returns "" when the value names no host.
func CanonicalizeGitURL(rawURL string) string {
	value := strings.TrimSpace(rawURL)
	if value == "" {
		return ""
	}

	var host, path string
	if !strings.Contains(value, "://") {
		// scp-like spellings need a user@ before the host.
		if !strings.Contains(value, "@") {
			return ""
		}
		match := scpRemote.FindStringSubmatch(value)
		if match == nil {
			return ""
		}
		host, path = match[1], match[2]
	} else {
		parsed, err := url.Parse(value)
		if err != nil || parsed.Scheme == "" {
			return ""
		}
		host, path = parsed.Hostname(), parsed.Path
	}

	host = strings.ToLower(strings.TrimSpace(host))
	path = strings.Trim(path, "/")
	path = strings.TrimSuffix(path, ".git")
	path = strings.Trim(path, "/")
	path = strings.ToLower(path)

	if host == "" || path == "" {
		return ""
	}
	return "git://" + host + "/" + path
}

// ResolveProjectID walks the identity waterfall for cwd.
//
// A non-empty explicitProjectID is an already-declared identity and short
// circuits every tier: a canonical identity is normalized, a plain string is
// promoted to proj://<name>. With no explicit value the directory itself is
// inspected, falling back to the ambient global identity when cwd is empty.
func ResolveProjectID(cwd string, explicitProjectID string) string {
	cwdStr := strings.TrimSpace(cwd)
	explicit := strings.TrimSpace(explicitProjectID)

	// Tier 0 -- a value that is already an identity, not a directory.
	if explicit != "" {
		return normalizeExplicit(explicit)
	}
	if id, ok := identityAsGiven(cwdStr); ok {
		return id
	}

	// Tier 1 -- an explicit declaration outranks anything discoverable.
	if declared := normalizeDeclared(os.Getenv("SIMPLE_MEM_PROJECT_ID")); declared != "" {
		return declared
	}
	if cwdStr == "" {
		return globalID
	}

	start := resolveStart(cwdStr)
	for _, directory := range walkDirectories(start) {
		marker := filepath.Join(directory, ".simple-mem.json")
		if info, err := os.Stat(marker); err == nil && info.Mode().IsRegular() {
			if declared := declaredFromJSON(marker); declared != "" {
				return declared
			}
		}
	}

	// Tier 2 -- the repository's own name, remote first, root commit second.
	for _, directory := range walkDirectories(start) {
		marker := filepath.Join(directory, ".git")
		info, err := os.Stat(marker)
		if err != nil {
			continue
		}
		if !info.IsDir() && !info.Mode().IsRegular() {
			continue
		}
		gitDir, commonDir, ok := gitLayout(marker)
		if !ok {
			continue
		}
		config := filepath.Join(commonDir, "config")
		if fi, err := os.Stat(config); err != nil || !fi.Mode().IsRegular() {
			config = filepath.Join(gitDir, "config")
		}
		if fi, err := os.Stat(config); err == nil && fi.Mode().IsRegular() {
			if remote := originURL(config); remote != "" {
				if canonical := CanonicalizeGitURL(remote); canonical != "" {
					return canonical
				}
			}
		}
		if root := rootCommitSHA(directory, gitDir, commonDir); root != "" {
			return "git-root://" + root
		}
	}

	// Tier 3 -- the directory names itself.
	return "dir://" + directorySlug(cwdStr)
}

// --------------------------------------------------------------------------
// Tier 0
// --------------------------------------------------------------------------

func identityAsGiven(value string) (string, bool) {
	if value == "" {
		return "", false
	}
	if value == globalID {
		return globalID, true
	}
	for _, prefix := range canonicalPrefixes {
		if !strings.HasPrefix(value, prefix) {
			continue
		}
		if prefix == "git://" {
			if canonical := CanonicalizeGitURL(value); canonical != "" {
				return canonical, true
			}
			return strings.ToLower(strings.TrimRight(value, "/")), true
		}
		return strings.TrimRight(value, "/"), true
	}
	for _, prefix := range urlPrefixes {
		if strings.HasPrefix(value, prefix) {
			if canonical := CanonicalizeGitURL(value); canonical != "" {
				return canonical, true
			}
		}
	}
	return "", false
}

func normalizeExplicit(value string) string {
	if value == globalID {
		return globalID
	}
	if id, ok := identityAsGiven(value); ok {
		return id
	}
	return "proj://" + value
}

// --------------------------------------------------------------------------
// Tier 1
// --------------------------------------------------------------------------

func normalizeDeclared(value string) string {
	value = strings.TrimSpace(value)
	if value == "" {
		return ""
	}
	if value == globalID || strings.Contains(value, "://") {
		return value
	}
	return "proj://" + value
}

func declaredFromJSON(marker string) string {
	data, err := os.ReadFile(marker)
	if err != nil {
		return ""
	}
	var payload map[string]any
	if err := json.Unmarshal(data, &payload); err != nil {
		return ""
	}
	for _, key := range []string{"project_id", "project", "id"} {
		if raw, ok := payload[key]; ok {
			if text, ok := raw.(string); ok {
				if declared := normalizeDeclared(text); declared != "" {
					return declared
				}
			}
		}
	}
	return ""
}

// --------------------------------------------------------------------------
// Directory walking
// --------------------------------------------------------------------------

// resolveStart expands ~ and resolves symlinks the way the Python original
// does before the parent walk.
func resolveStart(cwd string) string {
	path := expandHome(cwd)
	if info, err := os.Stat(path); err == nil && info.IsDir() {
		if resolved, err := filepath.EvalSymlinks(path); err == nil {
			path = resolved
		}
		if absolute, err := filepath.Abs(path); err == nil {
			path = absolute
		}
	}
	return path
}

// walkDirectories yields start and its parents, stopping before "/", "/tmp"
// and the user's home directory.
func walkDirectories(start string) []string {
	directory := start
	if info, err := os.Stat(directory); err != nil || !info.IsDir() {
		directory = filepath.Dir(directory)
	}

	excluded := map[string]bool{"/": true, "/tmp": true}
	if home, err := os.UserHomeDir(); err == nil && home != "" {
		excluded[filepath.Clean(home)] = true
	}

	var out []string
	seen := map[string]bool{}
	for i := 0; i < homeWalkLimit; i++ {
		current := filepath.Clean(directory)
		if current == "" || excluded[current] || seen[current] {
			break
		}
		seen[current] = true
		out = append(out, current)
		parent := filepath.Dir(current)
		if parent == current {
			break
		}
		directory = parent
	}
	return out
}

// expandHome replaces a leading "~" with the user's home directory.
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

// directorySlug lower-cases a base name and folds runs of unsupported
// characters into a single hyphen, preserving the characters already legal in
// an identity ([a-z0-9._-]).
func directorySlug(cwd string) string {
	trimmed := strings.TrimRight(expandHome(cwd), string(os.PathSeparator))
	name := filepath.Base(trimmed)
	if name == "." || name == string(os.PathSeparator) || name == "" {
		name = "root"
	}
	var builder strings.Builder
	previousDash := false
	for _, r := range strings.ToLower(name) {
		switch {
		case r >= 'a' && r <= 'z', r >= '0' && r <= '9', r == '_', r == '.', r == '-':
			builder.WriteRune(r)
			previousDash = false
		case !previousDash:
			builder.WriteByte('-')
			previousDash = true
		}
	}
	slug := strings.Trim(builder.String(), "-")
	if slug == "" {
		return "root"
	}
	return slug
}

// --------------------------------------------------------------------------
// Tier 2a -- the git remote
// --------------------------------------------------------------------------

// gitLayout returns (git dir, common dir) for a .git directory or a worktree
// pointer file, following a commondir hop when present.
func gitLayout(marker string) (string, string, bool) {
	info, err := os.Stat(marker)
	if err != nil {
		return "", "", false
	}

	gitDir := marker
	if !info.IsDir() {
		pointer, err := os.ReadFile(marker)
		if err != nil {
			return "", "", false
		}
		gitDir = ""
		for _, line := range strings.Split(string(pointer), "\n") {
			trimmed := strings.TrimSpace(line)
			if len(trimmed) < len("gitdir:") || !strings.EqualFold(trimmed[:len("gitdir:")], "gitdir:") {
				continue
			}
			candidate := expandHome(strings.TrimSpace(trimmed[len("gitdir:"):]))
			if !filepath.IsAbs(candidate) {
				candidate = filepath.Join(filepath.Dir(marker), candidate)
			}
			gitDir = candidate
			break
		}
		if gitDir == "" {
			return "", "", false
		}
	}

	commonDir := gitDir
	commondirFile := filepath.Join(gitDir, "commondir")
	if data, err := os.ReadFile(commondirFile); err == nil {
		for _, line := range strings.Split(string(data), "\n") {
			trimmed := strings.TrimSpace(line)
			if trimmed == "" {
				continue
			}
			candidate := expandHome(trimmed)
			if !filepath.IsAbs(candidate) {
				candidate = filepath.Join(gitDir, candidate)
			}
			commonDir = candidate
			break
		}
	}
	return gitDir, commonDir, true
}

// originURL reads the remote "origin" URL from a git config, falling back to
// the first remote when origin is absent.
func originURL(configFile string) string {
	data, err := os.ReadFile(configFile)
	if err != nil {
		return ""
	}
	section := ""
	origin := ""
	first := ""
	for _, raw := range strings.Split(string(data), "\n") {
		line := strings.TrimSpace(raw)
		if line == "" || line[0] == '#' || line[0] == ';' {
			continue
		}
		if strings.HasPrefix(line, "[") {
			if end := strings.Index(line, "]"); end >= 0 {
				section = strings.ToLower(strings.TrimSpace(line[1:end]))
			} else {
				section = strings.ToLower(strings.TrimSpace(line[1:]))
			}
			continue
		}
		if !strings.Contains(section, "remote") {
			continue
		}
		key, value, found := strings.Cut(line, "=")
		if !found || strings.ToLower(strings.TrimSpace(key)) != "url" {
			continue
		}
		value = strings.TrimSpace(value)
		if value == "" {
			continue
		}
		if strings.Contains(section, "origin") {
			origin = value
		} else if first == "" {
			first = value
		}
	}
	if origin != "" {
		return origin
	}
	return first
}

// --------------------------------------------------------------------------
// Tier 2b -- the repository's root commit
// --------------------------------------------------------------------------

func rootCommitSHA(repoDir, gitDir, commonDir string) string {
	if sha := rootCommitViaGit(repoDir); sha != "" {
		return sha
	}
	return rootCommitInGo(gitDir, commonDir)
}

func rootCommitViaGit(repoDir string) string {
	command := exec.Command("git", "-C", repoDir, "rev-list", "--max-parents=0", "HEAD")
	done := make(chan struct{})
	var output []byte
	var runErr error
	go func() {
		output, runErr = command.Output()
		close(done)
	}()
	select {
	case <-done:
	case <-time.After(2 * time.Second):
		if command.Process != nil {
			_ = command.Process.Kill()
		}
		<-done
		return ""
	}
	if runErr != nil {
		return ""
	}
	return firstSHA(string(output))
}

func rootCommitInGo(gitDir, commonDir string) string {
	head, err := os.ReadFile(filepath.Join(gitDir, "HEAD"))
	if err != nil {
		return ""
	}
	headText := strings.TrimSpace(string(head))

	var sha string
	if strings.HasPrefix(headText, "ref:") {
		resolved := readRef(commonDir, strings.TrimSpace(headText[len("ref:"):]))
		if resolved == "" {
			return ""
		}
		sha = resolved
	} else {
		sha = firstSHA(headText)
		if sha == "" {
			return ""
		}
	}

	for i := 0; i < homeWalkLimit; i++ {
		body := readLooseObject(commonDir, sha)
		if body == nil {
			return ""
		}
		parent := ""
		for _, line := range strings.Split(string(body), "\n") {
			if strings.HasPrefix(line, "parent ") {
				parent = strings.ToLower(strings.TrimSpace(line[len("parent "):]))
				break
			}
			if line == "" {
				break
			}
		}
		if parent == "" {
			return sha
		}
		sha = parent
	}
	return ""
}

func readRef(commonDir, ref string) string {
	refFile := filepath.Join(commonDir, filepath.FromSlash(ref))
	if data, err := os.ReadFile(refFile); err == nil {
		if sha := firstSHA(string(data)); sha != "" {
			return sha
		}
	}

	packed, err := os.ReadFile(filepath.Join(commonDir, "packed-refs"))
	if err != nil {
		return ""
	}
	for _, line := range strings.Split(string(packed), "\n") {
		trimmed := strings.TrimSpace(line)
		if trimmed == "" || strings.HasPrefix(trimmed, "#") || strings.HasPrefix(trimmed, "^") {
			continue
		}
		fields := strings.Fields(trimmed)
		if len(fields) >= 2 && fields[1] == ref {
			return firstSHA(fields[0])
		}
	}
	return ""
}

func readLooseObject(commonDir, sha string) []byte {
	if len(sha) < 3 {
		return nil
	}
	raw, err := os.ReadFile(filepath.Join(commonDir, "objects", sha[:2], sha[2:]))
	if err != nil {
		return nil
	}
	reader, err := zlib.NewReader(bytes.NewReader(raw))
	if err != nil {
		return nil
	}
	defer reader.Close()
	data, err := io.ReadAll(reader)
	if err != nil {
		return nil
	}
	if index := bytes.IndexByte(data, 0); index >= 0 {
		return data[index+1:]
	}
	return data
}

func firstSHA(text string) string {
	for _, line := range strings.Split(text, "\n") {
		candidate := strings.ToLower(strings.TrimSpace(line))
		if hexSHA.MatchString(candidate) {
			return candidate
		}
	}
	return ""
}
