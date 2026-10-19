package project

import (
	"bytes"
	"compress/zlib"
	"crypto/sha1"
	"encoding/hex"
	"os"
	"path/filepath"
	"strconv"
	"testing"
)

// clearProjectEnv removes an ambient SIMPLE_MEM_PROJECT_ID so a test observes
// only the tier it is exercising.
func clearProjectEnv(t *testing.T) {
	t.Helper()
	t.Setenv("SIMPLE_MEM_PROJECT_ID", "")
}

func writeFile(t *testing.T, path, content string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatalf("mkdir %s: %v", filepath.Dir(path), err)
	}
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatalf("write %s: %v", path, err)
	}
}

// --------------------------------------------------------------------------
// Canonicalization
// --------------------------------------------------------------------------

func TestCanonicalizeGitURLVariants(t *testing.T) {
	cases := map[string]string{
		"git@github.com:Owner/Repo.git":            "git://github.com/owner/repo",
		"ssh://git@github.com/Owner/Repo.git":      "git://github.com/owner/repo",
		"https://github.com/Owner/Repo":            "git://github.com/owner/repo",
		"https://user:pass@github.com/Group/P.git": "git://github.com/group/p",
		"https://github.com:443/Owner/Repo.git":    "git://github.com/owner/repo",
		"git://GitHub.com/Acme/Core/":              "git://github.com/acme/core",
		"  https://GitLab.com/Group/Proj.git  ":    "git://gitlab.com/group/proj",
	}
	for raw, want := range cases {
		if got := CanonicalizeGitURL(raw); got != want {
			t.Errorf("CanonicalizeGitURL(%q) = %q, want %q", raw, got, want)
		}
	}
}

func TestCanonicalizeGitURLRejectsNonRepositories(t *testing.T) {
	for _, raw := range []string{"", "   ", "not a url", "https://github.com", "git@github.com:", "/a/b/c", "relative/path"} {
		if got := CanonicalizeGitURL(raw); got != "" {
			t.Errorf("CanonicalizeGitURL(%q) = %q, want empty", raw, got)
		}
	}
}

// --------------------------------------------------------------------------
// Tier 0 -- the value is already an identity
// --------------------------------------------------------------------------

func TestTier0PassesThroughCanonicalIdentities(t *testing.T) {
	clearProjectEnv(t)
	cases := map[string]string{
		"global":            "global",
		"proj://acme":       "proj://acme",
		"dir://SomeDir/":    "dir://SomeDir",
		"git-root://AbC123": "git-root://AbC123",
		"global://team":     "global://team",
	}
	for raw, want := range cases {
		if got := ResolveProjectID(raw, ""); got != want {
			t.Errorf("ResolveProjectID(%q, \"\") = %q, want %q", raw, got, want)
		}
	}
}

func TestTier0CanonicalizesCloneURLs(t *testing.T) {
	clearProjectEnv(t)
	cases := map[string]string{
		"git://GitHub.com/Acme/Core/":      "git://github.com/acme/core",
		"https://github.com/Acme/Core.git": "git://github.com/acme/core",
		"git@github.com:Acme/Core.git":     "git://github.com/acme/core",
	}
	for raw, want := range cases {
		if got := ResolveProjectID(raw, ""); got != want {
			t.Errorf("ResolveProjectID(%q, \"\") = %q, want %q", raw, got, want)
		}
	}
}

func TestExplicitProjectIDWins(t *testing.T) {
	clearProjectEnv(t)
	if got := ResolveProjectID("/somewhere", "acme"); got != "proj://acme" {
		t.Errorf("plain explicit = %q, want proj://acme", got)
	}
	if got := ResolveProjectID("/somewhere", "git@github.com:Acme/Core.git"); got != "git://github.com/acme/core" {
		t.Errorf("url explicit = %q", got)
	}
	if got := ResolveProjectID("/somewhere", "global"); got != "global" {
		t.Errorf("global explicit = %q", got)
	}
}

// --------------------------------------------------------------------------
// Ambient / empty directory
// --------------------------------------------------------------------------

func TestEmptyDirectoryIsAmbient(t *testing.T) {
	clearProjectEnv(t)
	if got := ResolveProjectID("", ""); got != "global" {
		t.Errorf("empty cwd = %q, want global", got)
	}
	t.Setenv("SIMPLE_MEM_PROJECT_ID", "acme")
	if got := ResolveProjectID("", ""); got != "proj://acme" {
		t.Errorf("empty cwd with env = %q, want proj://acme", got)
	}
	if got := ResolveProjectID("", "explicit"); got != "proj://explicit" {
		t.Errorf("explicit over ambient = %q", got)
	}
}

// --------------------------------------------------------------------------
// Tier 1 -- an explicit declaration
// --------------------------------------------------------------------------

func TestTier1JSONFromParent(t *testing.T) {
	clearProjectEnv(t)
	root := filepath.Join(t.TempDir(), "workspace")
	nested := filepath.Join(root, "packages", "core")
	if err := os.MkdirAll(nested, 0o755); err != nil {
		t.Fatal(err)
	}
	writeFile(t, filepath.Join(root, ".simple-mem.json"), `{"project_id": "Acme-Core"}`)
	if got := ResolveProjectID(nested, ""); got != "proj://Acme-Core" {
		t.Errorf("got %q, want proj://Acme-Core", got)
	}
}

func TestTier1JSONKeysAndURI(t *testing.T) {
	clearProjectEnv(t)
	for _, key := range []string{"project_id", "project", "id"} {
		dir := filepath.Join(t.TempDir(), "k")
		if err := os.MkdirAll(dir, 0o755); err != nil {
			t.Fatal(err)
		}
		writeFile(t, filepath.Join(dir, ".simple-mem.json"), `{"`+key+`": "from-`+key+`"}`)
		if got := ResolveProjectID(dir, ""); got != "proj://from-"+key {
			t.Errorf("key %s: got %q", key, got)
		}
	}

	dir := filepath.Join(t.TempDir(), "uri")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	writeFile(t, filepath.Join(dir, ".simple-mem.json"), `{"project_id": "git://host/team"}`)
	if got := ResolveProjectID(dir, ""); got != "git://host/team" {
		t.Errorf("uri json = %q", got)
	}
}

func TestTier1EnvOverridesJSON(t *testing.T) {
	dir := filepath.Join(t.TempDir(), "env")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	writeFile(t, filepath.Join(dir, ".simple-mem.json"), `{"project_id": "from-file"}`)
	t.Setenv("SIMPLE_MEM_PROJECT_ID", "from-env")
	if got := ResolveProjectID(dir, ""); got != "proj://from-env" {
		t.Errorf("got %q, want proj://from-env", got)
	}
}

func TestTier1InvalidJSONFallsThrough(t *testing.T) {
	clearProjectEnv(t)
	dir := filepath.Join(t.TempDir(), "some-project")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	writeFile(t, filepath.Join(dir, ".simple-mem.json"), `{ not json`)
	if got := ResolveProjectID(dir, ""); got != "dir://some-project" {
		t.Errorf("got %q, want dir://some-project", got)
	}
}

// --------------------------------------------------------------------------
// Tier 2a -- the git remote
// --------------------------------------------------------------------------

func TestTier2aSSHAndHTTPSAgree(t *testing.T) {
	clearProjectEnv(t)
	sshRepo := filepath.Join(t.TempDir(), "ssh-repo")
	writeFile(t, filepath.Join(sshRepo, ".git", "config"), "[remote \"origin\"]\n\turl = git@GitHub.com:Acme/CoreEngine.git\n")
	httpsRepo := filepath.Join(t.TempDir(), "https-repo")
	writeFile(t, filepath.Join(httpsRepo, ".git", "config"), "[remote \"origin\"]\n\turl = https://user:pass@github.com/acme/coreengine.git\n")

	for _, repo := range []string{sshRepo, httpsRepo} {
		if got := ResolveProjectID(repo, ""); got != "git://github.com/acme/coreengine" {
			t.Errorf("%s: got %q", repo, got)
		}
	}
}

func TestTier2aFirstRemoteWhenOriginAbsent(t *testing.T) {
	clearProjectEnv(t)
	repo := filepath.Join(t.TempDir(), "repo")
	writeFile(t, filepath.Join(repo, ".git", "config"), "[remote \"upstream\"]\n\turl = https://gitlab.com/Team/Proj.git\n")
	if got := ResolveProjectID(repo, ""); got != "git://gitlab.com/team/proj" {
		t.Errorf("got %q", got)
	}
}

func TestTier2aWorktreeFollowsCommondir(t *testing.T) {
	clearProjectEnv(t)
	root := t.TempDir()
	mainGit := filepath.Join(root, "main", ".git")
	writeFile(t, filepath.Join(mainGit, "config"), "[remote \"origin\"]\n\turl = git@github.com:Acme/Core.git\n")
	gitDir := filepath.Join(mainGit, "worktrees", "wt")
	writeFile(t, filepath.Join(gitDir, "commondir"), "../..\n")
	worktree := filepath.Join(root, "worktree")
	if err := os.MkdirAll(worktree, 0o755); err != nil {
		t.Fatal(err)
	}
	writeFile(t, filepath.Join(worktree, ".git"), "gitdir: "+gitDir+"\n")

	if got := ResolveProjectID(worktree, ""); got != "git://github.com/acme/core" {
		t.Errorf("got %q", got)
	}
}

// --------------------------------------------------------------------------
// Tier 2b -- the repository's root commit
// --------------------------------------------------------------------------

// writeLooseObject stores a zlib-compressed git object and returns its SHA.
func writeLooseObject(t *testing.T, gitDir, kind, body string) string {
	t.Helper()
	payload := kind + " " + strconv.Itoa(len(body)) + "\x00" + body
	sum := sha1.Sum([]byte(payload))
	sha := hex.EncodeToString(sum[:])

	var compressed bytes.Buffer
	writer := zlib.NewWriter(&compressed)
	if _, err := writer.Write([]byte(payload)); err != nil {
		t.Fatal(err)
	}
	if err := writer.Close(); err != nil {
		t.Fatal(err)
	}
	objectPath := filepath.Join(gitDir, "objects", sha[:2], sha[2:])
	if err := os.MkdirAll(filepath.Dir(objectPath), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(objectPath, compressed.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}
	return sha
}

func TestTier2bRootCommitViaLooseObjects(t *testing.T) {
	clearProjectEnv(t)
	repo := filepath.Join(t.TempDir(), "local")
	gitDir := filepath.Join(repo, ".git")
	if err := os.MkdirAll(gitDir, 0o755); err != nil {
		t.Fatal(err)
	}
	rootSHA := writeLooseObject(t, gitDir, "commit", "tree 0000000000000000000000000000000000000000\nauthor a <a@b> 0 +0000\ncommitter a <a@b> 0 +0000\n\nroot\n")
	childSHA := writeLooseObject(t, gitDir, "commit", "tree 0000000000000000000000000000000000000000\nparent "+rootSHA+"\nauthor a <a@b> 0 +0000\ncommitter a <a@b> 0 +0000\n\nchild\n")
	writeFile(t, filepath.Join(gitDir, "HEAD"), "ref: refs/heads/main\n")
	writeFile(t, filepath.Join(gitDir, "refs", "heads", "main"), childSHA+"\n")

	if got := rootCommitInGo(gitDir, gitDir); got != rootSHA {
		t.Errorf("rootCommitInGo = %q, want %q", got, rootSHA)
	}
	if got := ResolveProjectID(repo, ""); got != "git-root://"+rootSHA {
		t.Errorf("ResolveProjectID = %q, want git-root://%s", got, rootSHA)
	}
}

func TestTier2aPackConfigParsesOrigin(t *testing.T) {
	config := filepath.Join(t.TempDir(), "config")
	writeFile(t, config, "# a comment\n[core]\n\trepositoryformatversion = 0\n[remote \"upstream\"]\n\turl = https://example.com/a/b\n[remote \"origin\"]\n\turl = git@github.com:Owner/Repo.git\n")
	if got := originURL(config); got != "git@github.com:Owner/Repo.git" {
		t.Errorf("originURL = %q", got)
	}
}

// --------------------------------------------------------------------------
// Tier 3 -- the directory names itself
// --------------------------------------------------------------------------

func TestTier3DirectoryFallback(t *testing.T) {
	clearProjectEnv(t)
	base := t.TempDir()
	plain := filepath.Join(base, "some-project")
	if err := os.MkdirAll(plain, 0o755); err != nil {
		t.Fatal(err)
	}
	if got := ResolveProjectID(plain, ""); got != "dir://some-project" {
		t.Errorf("got %q, want dir://some-project", got)
	}
	if got := ResolveProjectID(filepath.Join(base, "missing"), ""); got != "dir://missing" {
		t.Errorf("missing got %q, want dir://missing", got)
	}
}

func TestDirectorySlugLowercasesAndHyphenates(t *testing.T) {
	cases := map[string]string{
		"My Project": "my-project",
		"Some_Proj":  "some_proj",
		"a--b":       "a--b",
		"!!!":        "root",
		"":           "root",
	}
	for name, want := range cases {
		if got := directorySlug(name); got != want {
			t.Errorf("directorySlug(%q) = %q, want %q", name, got, want)
		}
	}
}

func TestDeclaredFromJSONIgnoresMalformed(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, ".simple-mem.json")
	if err := os.WriteFile(path, []byte("[1,2,3]"), 0o644); err != nil {
		t.Fatal(err)
	}
	if got := declaredFromJSON(path); got != "" {
		t.Errorf("declaredFromJSON(array) = %q, want empty", got)
	}
}
