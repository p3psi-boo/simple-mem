"""Project identity: one name every machine agrees a project has.

A caller sits in a directory. That directory is a local fact -- two checkouts
of one repository are two directories -- so an identity built from the path
cannot be compared across machines. The four-tier waterfall here turns the
directory into the most stable name it can justify, preferring a declared
identity over the repository's own origin over an opaque commit over the
bare directory name.

    Tier 0  a value already shaped like an identity (`git://`, `proj://`, ...)
    Tier 1  an explicit declaration (`SIMPLE_MEM_PROJECT_ID`, `.simple-mem.json`)
    Tier 2a the git remote URL, canonicalized (`git://<host>/<path>`)
    Tier 2b the repository's root commit, when it has no remote
    Tier 3  the directory's own name (`dir://<name>`)

Every tier returns a string a different machine could have produced for the
same project; only Tier 3 is knowingly machine-local.
"""

from __future__ import annotations

import configparser
import json
import os
import re
import subprocess
import zlib
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Optional, Tuple, Union
from urllib.parse import urlsplit

__all__ = ["canonicalize_git_url", "resolve_project_id"]

_CANONICAL_PREFIXES = ("git://", "git-root://", "proj://", "dir://")

#: A URL that already names a repository, handed back to us as a directory.
_URL_PREFIXES = ("https://", "http://", "git@", "ssh://")

_GITDIR_POINTER = re.compile(r"^\s*gitdir:\s*(?P<path>.+?)\s*$")

_REMOTE_SECTION = re.compile(r'^\[\s*remote\s+"(?P<name>[^"]+)"\s*\]$')

_REMOTE_URL = re.compile(r"^url\s*=\s*(?P<url>.+?)\s*$")

_HEX_SHA = re.compile(r"^[0-9a-f]{7,64}$")

_HOME_WALK_LIMIT = 4096


def canonicalize_git_url(url: str) -> Optional[str]:
    """`git://<host>/<path>` for any clone URL, or None if it names no host.

    SSH and HTTPS spellings of one repository become the same string, and the
    host and path are folded to lower case because GitHub, GitLab and friends
    treat them that way. A scp-like `user@host:path` is accepted as readily as
    a parsed URL, since the two are interchangeable on the command line.
    """
    if not url or not isinstance(url, str):
        return None
    value = url.strip()
    if not value:
        return None

    host = ""
    path = ""
    if "://" not in value and "@" in value:
        rest = value.split("@", 1)[1]
        host, separator, path = rest.partition(":")
        if not separator:
            return None
    else:
        if "://" not in value:
            return None
        try:
            parsed = urlsplit(value)
        except ValueError:
            return None
        if not parsed.scheme:
            return None
        host = parsed.hostname or ""
        path = parsed.path or ""

    host = host.strip().lower()
    path = path.strip().strip("/")
    if path.endswith(".git"):
        path = path[:-4]
    path = path.strip("/").lower()

    if not host or not path:
        return None
    return f"git://{host}/{path}"


def resolve_project_id(
    cwd: Union[str, Path, None] = None,
    *,
    env: Optional[Mapping[str, str]] = None,
) -> str:
    """The project identity for `cwd`, walking the waterfall in order.

    With no directory at all the caller is asking for the ambient project:
    the declared `SIMPLE_MEM_PROJECT_ID`, or `global` when nothing declares
    one. A directory always yields a project-local identity.
    """
    environment = env if env is not None else os.environ

    if cwd is None:
        return _declared_project(environment) or "global"
    cwd_str = str(cwd).strip()
    if not cwd_str:
        return _declared_project(environment) or "global"

    # Tier 0 -- the value is already an identity, not a directory.
    canonical = _identity_as_given(cwd_str)
    if canonical is not None:
        return canonical

    # Tier 1 -- an explicit declaration outranks anything discoverable.
    declared = _declared_project(environment)
    if declared is not None:
        return declared
    try:
        start = Path(cwd_str).expanduser()
        if start.exists():
            start = start.resolve()
    except (OSError, RuntimeError, ValueError):
        start = Path(cwd_str)
    for directory in _walk_directories(start):
        marker = directory / ".simple-mem.json"
        if marker.is_file():
            declared = _declared_from_json(marker)
            if declared is not None:
                return declared

    # Tier 2 -- the repository's own name, remote first, root commit second.
    for directory in _walk_directories(start):
        marker = directory / ".git"
        if not (marker.is_dir() or marker.is_file()):
            continue
        layout = _git_layout(marker)
        if layout is None:
            continue
        git_dir, common_dir = layout
        config = common_dir / "config"
        if not config.is_file():
            config = git_dir / "config"
        if config.is_file():
            remote = _remote_url_from_config(config)
            if remote:
                canonical = canonicalize_git_url(remote)
                if canonical is not None:
                    return canonical
        root = _root_commit_sha(directory, git_dir, common_dir)
        if root is not None:
            return f"git-root://{root}"

    # Tier 3 -- nothing above justifies a shared name, so the directory names
    # itself. This is the only tier that is knowingly machine-local.
    return f"dir://{Path(cwd_str).name or 'root'}"


# --------------------------------------------------------------------------
# Tier 0
# --------------------------------------------------------------------------


def _identity_as_given(cwd_str: str) -> Optional[str]:
    if cwd_str == "global":
        return "global"
    if cwd_str.startswith("git://"):
        canonical = canonicalize_git_url(cwd_str)
        return canonical if canonical is not None else cwd_str.rstrip("/").lower()
    if cwd_str.startswith(_CANONICAL_PREFIXES):
        return cwd_str.rstrip("/")
    if cwd_str.startswith(_URL_PREFIXES):
        canonical = canonicalize_git_url(cwd_str)
        if canonical is not None:
            return canonical
    return None


# --------------------------------------------------------------------------
# Tier 1
# --------------------------------------------------------------------------


def _declared_project(environment: Mapping[str, str]) -> Optional[str]:
    value = environment.get("SIMPLE_MEM_PROJECT_ID", "")
    return _normalize_declared(value)


def _normalize_declared(value: str) -> Optional[str]:
    if not isinstance(value, str):
        return None
    value = value.strip()
    if not value:
        return None
    if "://" in value or value == "global":
        return value
    return f"proj://{value}"


def _declared_from_json(marker: Path) -> Optional[str]:
    try:
        data = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    return _normalize_declared(data.get("project_id", ""))


def _walk_directories(start: Path) -> Iterator[Path]:
    """`start` upwards, stopping before `/`, `/tmp`, and the user's home."""
    excluded = {Path("/"), Path("/tmp")}
    try:
        excluded.add(Path.home())
    except (RuntimeError, ValueError):  # pragma: no cover - no home to name
        pass

    try:
        directory = start if start.is_dir() else start.parent
    except OSError:  # pragma: no cover - unreadable path
        directory = start
    seen = set()
    for _ in range(_HOME_WALK_LIMIT):
        if directory in excluded or directory in seen:
            return
        seen.add(directory)
        yield directory
        parent = directory.parent
        if parent == directory:
            return
        directory = parent


# --------------------------------------------------------------------------
# Tier 2a -- the git remote
# --------------------------------------------------------------------------


def _git_layout(marker: Path) -> Optional[Tuple[Path, Path]]:
    """(git dir, common dir) for a `.git` directory or worktree pointer file.

    A `git worktree` checkout keeps a `.git` *file* naming a git dir under the
    main repository, and a `commondir` file inside it naming where the shared
    config, refs and objects actually live. Both hops are followed.
    """
    if marker.is_dir():
        git_dir = marker
    elif marker.is_file():
        git_dir = _git_dir_from_pointer(marker)
        if git_dir is None:
            return None
    else:
        return None

    common_dir = git_dir
    commondir_file = git_dir / "commondir"
    if commondir_file.is_file():
        try:
            line = commondir_file.read_text(encoding="utf-8", errors="replace")
        except OSError:
            line = ""
        lines = [part for part in line.splitlines() if part.strip()]
        if lines:
            candidate = Path(lines[0].strip()).expanduser()
            if not candidate.is_absolute():
                candidate = git_dir / candidate
            common_dir = candidate
    return git_dir, common_dir


def _git_dir_from_pointer(marker: Path) -> Optional[Path]:
    try:
        text = marker.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    for line in text.splitlines():
        match = _GITDIR_POINTER.match(line)
        if match is None:
            continue
        path = Path(match.group("path")).expanduser()
        if not path.is_absolute():
            path = marker.parent / path
        return path
    return None


def _remote_url_from_config(config_file: Path) -> Optional[str]:
    try:
        text = config_file.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None

    parser = configparser.ConfigParser(strict=False, interpolation=None)
    try:
        parser.read_string(text)
    except configparser.Error:
        return _remote_url_from_lines(text)

    if parser.has_option('remote "origin"', "url"):
        value = parser.get('remote "origin"', "url").strip()
        if value:
            return value
    for section in parser.sections():
        if not section.startswith('remote "'):
            continue
        if parser.has_option(section, "url"):
            value = parser.get(section, "url").strip()
            if value:
                return value
    return _remote_url_from_lines(text)


def _remote_url_from_lines(text: str) -> Optional[str]:
    """Last-resort lookup for config syntax configparser rejects."""
    current = None
    origin_url = None
    first_url = None
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped[0] in "#;":
            continue
        if stripped.startswith("["):
            match = _REMOTE_SECTION.match(stripped)
            current = match.group("name") if match is not None else None
            continue
        if current is None:
            continue
        match = _REMOTE_URL.match(stripped)
        if match is None:
            continue
        value = match.group("url").strip()
        if not value:
            continue
        if current == "origin":
            origin_url = value
        elif first_url is None:
            first_url = value
    return origin_url or first_url


# --------------------------------------------------------------------------
# Tier 2b -- the repository's root commit
# --------------------------------------------------------------------------


def _root_commit_sha(repo_dir: Path, git_dir: Path, common_dir: Path) -> Optional[str]:
    from_subprocess = _root_commit_via_git(repo_dir)
    if from_subprocess is not None:
        return from_subprocess
    return _root_commit_in_python(git_dir, common_dir)


def _root_commit_via_git(repo_dir: Path) -> Optional[str]:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo_dir), "rev-list", "--max-parents=0", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return _first_sha(completed.stdout)


def _root_commit_in_python(git_dir: Path, common_dir: Path) -> Optional[str]:
    """Walk HEAD's first-parent chain to the commit that has no parent.

    Only loose objects are read; a packed repository simply returns None and
    lets the caller fall through to the next tier. That is enough to name a
    freshly created repository without a working `git` binary.
    """
    head_file = git_dir / "HEAD"
    try:
        head = head_file.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return None

    if head.startswith("ref:"):
        resolved = _read_ref(common_dir, head[4:].strip())
        if resolved is None:
            return None
        sha = resolved
    else:
        sha = _first_sha(head)
        if sha is None:
            return None

    for _ in range(_HOME_WALK_LIMIT):
        body = _read_loose_object(common_dir, sha)
        if body is None:
            return None
        parent = None
        for line in body.split(b"\n"):
            if line.startswith(b"parent "):
                parent = line[7:].strip().decode("ascii", "replace").lower()
                break
            if not line:
                break
        if parent is None:
            return sha
        sha = parent
    return None


def _read_ref(common_dir: Path, ref: str) -> Optional[str]:
    ref_file = common_dir / ref
    if ref_file.is_file():
        try:
            sha = _first_sha(ref_file.read_text(encoding="utf-8", errors="replace"))
        except OSError:
            sha = None
        if sha is not None:
            return sha

    packed = common_dir / "packed-refs"
    if packed.is_file():
        try:
            text = packed.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or stripped.startswith("^"):
                continue
            parts = stripped.split()
            if len(parts) >= 2 and parts[1] == ref:
                return _first_sha(parts[0])
    return None


def _read_loose_object(common_dir: Path, sha: str) -> Optional[bytes]:
    if len(sha) < 3:
        return None
    path = common_dir / "objects" / sha[:2] / sha[2:]
    try:
        raw = path.read_bytes()
    except OSError:
        return None
    try:
        data = zlib.decompress(raw)
    except zlib.error:
        return None
    _, _, body = data.partition(b"\x00")
    return body


def _first_sha(text: str) -> Optional[str]:
    if not isinstance(text, str):
        return None
    for line in text.splitlines():
        candidate = line.strip().lower()
        if _HEX_SHA.match(candidate):
            return candidate
    return None
