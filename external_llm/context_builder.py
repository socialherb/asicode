# external_llm/context_builder.py
"""
Enhanced Context Builder for External LLM Integration with asicode.

Provides rich project context by:
1. Using asicode's context_collector for related files (when available)
2. Target file with line numbers (full content — head+tail truncation removed per
   "discarding info first is actually token waste" principle, see commit 320365fa)
3. Git status and recent commits
4. Project structure hints

Compatibility:
- external_llm.service expects:
  - ContextBuilder (class alias)
  - enhance_user_request(text, ...) (function)
"""

from __future__ import annotations

import logging
import os
import subprocess
import time
from collections.abc import Callable
from pathlib import Path

from path_security import normalize_rel_path, resolve_inside_repo
from utils.string_helper import utf8_trailing_incomplete_len

from .agent.agent_context_manager import get_git_snapshot
from .languages.capabilities import AnalysisCapability, is_supported

logger = logging.getLogger(__name__)

# P21-3: file-context blocks are head-bounded. The builder used to read,
# line-number and embed the WHOLE file (and each related file) into the LLM
# context — a multi-hundred-MB target flooded the prompt. Same class as
# P19-1/P21-1 (webapp + service.py snippet paths); truncated reads carry an
# explicit marker so the model knows it only sees the head.
_FILE_CONTEXT_MAX_BYTES = 1024 * 1024  # 1 MiB

# P21-3: line-numbering expands the head (~1.6-2.5x with the "N | " prefix),
# so the OUTPUT is capped too — a 1 MiB head of short lines would otherwise
# still grow into a multi-MB prompt block. 5000 numbered lines ≈ 70-90 KB.
_FILE_CONTEXT_MAX_LINES = 5000


def _bounded_file_text(p: Path, max_bytes: int = _FILE_CONTEXT_MAX_BYTES) -> tuple[str, bool]:
    """Read up to ``max_bytes`` of ``p`` (UTF-8, latin-1 fallback).

    Returns ``(text, truncated)``; never loads more than ``max_bytes`` into
    memory and never splits a multi-byte UTF-8 char at the cut.
    """
    size = p.stat().st_size
    if size <= max_bytes:
        try:
            return p.read_text(encoding="utf-8"), False
        except UnicodeDecodeError:
            return p.read_text(encoding="latin-1"), False
    with p.open("rb") as f:
        raw = f.read(max_bytes)
    trim = utf8_trailing_incomplete_len(raw)
    if trim:
        raw = raw[:-trim]
    try:
        return raw.decode("utf-8"), True
    except UnicodeDecodeError:
        return raw.decode("latin-1"), True


# Process-wide TTL cache for project-structure hints.  Computing this scans
# every top-level directory recursively via rglob (~95ms on a ~900-file repo)
# and build_context() runs on every request, so the cost is paid per turn.
# The result is a coarse overview (top-level dirs + a few files) that changes
# rarely and is purely advisory — stale hints never affect correctness — so a
# generous TTL is safe and a fresh repo_root key naturally isolates projects.
_STRUCTURE_HINTS_TTL_S = 300.0
# Opportunistic-eviction trigger: once the cache holds more than this many
# entries, expired ones are dropped on the next miss.  Bounds memory in
# long-running processes that touch many distinct repo roots (e.g. test
# runners spinning up temp dirs); single-repo services never reach it.
_STRUCTURE_HINTS_GC_THRESHOLD = 16
_structure_hints_cache: dict[str, tuple[str, float]] = {}

# Process-wide TTL cache for the recent-commits fetch
# (`git log -N --oneline --decorate`, one subprocess spawn per miss).  The git
# STATUS half of build_context() delegates to agent_context_manager's
# get_git_snapshot SSOT (10s per-root TTL, invalidated after every successful
# mutating tool call) — but that snapshot only carries the single last_commit,
# so the decorated N-commit log keeps this small cache of its own.  Commits
# change rarely (and only via mutating tool calls); 2s of advisory staleness is
# far below any agent-visible threshold, and the (repo_root, count) key
# isolates projects exactly like _structure_hints_cache.
_GIT_LOG_TTL_S = 2.0
# Opportunistic-eviction trigger, same shape as _STRUCTURE_HINTS_GC_THRESHOLD.
_GIT_LOG_GC_THRESHOLD = 32
_git_log_cache: dict[tuple[str, int], tuple[str, float]] = {}


def _cached_git_log(repo_root: Path, count: int, fetch: Callable[[int], str]) -> str:
    """TTL-cached wrapper around a ``git log -N`` fetch.

    ``fetch`` runs only on a miss; its result (including the empty-string
    failure sentinel) is cached for ``_GIT_LOG_TTL_S`` seconds.  Expired
    entries are purged opportunistically once the cache exceeds the GC
    threshold, mirroring _structure_hints_cache to bound memory in processes
    that touch many distinct repo roots (e.g. test runners).
    """
    key = (str(repo_root), count)
    now = time.monotonic()
    cached = _git_log_cache.get(key)
    if cached is not None:
        value, expiry = cached
        if expiry > now:
            return value
        _git_log_cache.pop(key, None)
    if len(_git_log_cache) > _GIT_LOG_GC_THRESHOLD:
        for _k, (_v, _exp) in list(_git_log_cache.items()):
            if _exp <= now:
                _git_log_cache.pop(_k, None)
    value = fetch(count)
    _git_log_cache[key] = (value, now + _GIT_LOG_TTL_S)
    return value


class EnhancedContextBuilder:
    def __init__(self, repo_root: str):
        self.repo_root = Path(repo_root).resolve()

    def build_context(
        self,
        user_request: str,
        target_file: str | None = None,
        include_related_files: bool = True,
        include_git_context: bool = True,
        max_related_files: int = 3,
    ) -> str:
        sections: list[str] = []

        sections.append("# PROJECT CONTEXT FOR CODE EDITING")
        sections.append("")
        sections.append(f"**Repository**: `{self.repo_root.name}`")
        sections.append(f"**Path**: `{self.repo_root}`")
        sections.append("")

        if include_git_context:
            git_ctx = self._build_git_context()
            if git_ctx:
                sections.append("## Git Status")
                sections.append("")
                sections.append(git_ctx)
                sections.append("")

        if target_file:
            file_ctx = self._build_file_context(target_file)
            if file_ctx:
                sections.append(f"## Target File: `{target_file}`")
                sections.append("")
                sections.append(file_ctx)
                sections.append("")

        if include_related_files and target_file:
            related_ctx = self._build_related_files_context(
                target_file,
                max_files=max_related_files,
            )
            if related_ctx:
                sections.append("## Related Files")
                sections.append("")
                sections.append(related_ctx)
                sections.append("")

        structure_hints = self._get_project_structure_hints()
        if structure_hints:
            sections.append("## Project Structure")
            sections.append("")
            sections.append(structure_hints)
            sections.append("")

        sections.append("## User Request")
        sections.append("")
        sections.append(user_request)
        sections.append("")

        sections.append("## Instructions")
        sections.append("")
        sections.append(self._get_llm_instructions(target_file))

        return "\n".join(sections)

    def _build_git_context(self) -> str:
        parts: list[str] = []

        status = self._get_git_status()
        if status:
            parts.append("```")
            parts.append(status)
            parts.append("```")

        recent = self._get_recent_commits(count=3)
        if recent:
            parts.append("")
            parts.append("**Recent Changes**:")
            parts.append("```")
            parts.append(recent)
            parts.append("```")

        return "\n".join(parts) if parts else ""

    def _get_git_status(self) -> str:
        # Delegate to the shared get_git_snapshot SSOT (agent_context_manager):
        # it already runs `git -c core.quotePath=false status --short` with a
        # 10s per-root TTL that is invalidated after every successful mutating
        # tool call — fresher than any private cache, and one subprocess spawn
        # fewer per build_context() call.
        return get_git_snapshot(str(self.repo_root)).get("status", "")

    def _get_recent_commits(self, count: int = 3) -> str:
        # The SSOT snapshot carries only the single last_commit; the decorated
        # N-commit log keeps its own small TTL cache (see _cached_git_log).
        return _cached_git_log(self.repo_root, count, self._fetch_recent_commits)

    def _fetch_recent_commits(self, count: int) -> str:
        try:
            result = subprocess.run(
                ["git", "log", f"-{count}", "--oneline", "--decorate"],
                cwd=str(self.repo_root),
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
        except Exception as e:
            logger.debug("Git log failed: %s", e)
        return ""

    def _build_file_context(self, rel_path: str) -> str:
        try:
            file_path = self.repo_root / rel_path
            if not file_path.exists():
                logger.warning("File not found: %s", rel_path)
                return ""

            # P21-3: head-bounded read — see _FILE_CONTEXT_MAX_BYTES.
            try:
                content, truncated = _bounded_file_text(file_path)
            except OSError:
                logger.warning("File read failed: %s", rel_path)
                return ""

            lines = content.split("\n")
            total = len(lines)
            lang = self._detect_language(rel_path)

            out: list[str] = []
            out.append(f"```{lang}")
            for i, line in enumerate(lines[:_FILE_CONTEXT_MAX_LINES], 1):
                out.append(f"{i:4d} | {line}")
            if len(lines) > _FILE_CONTEXT_MAX_LINES:
                out.append(f"    ... (more lines omitted — showing first {_FILE_CONTEXT_MAX_LINES})")
            out.append("```")
            out.append("")
            if truncated or len(lines) > _FILE_CONTEXT_MAX_LINES:
                # State the TRUE cause: a byte-bound truncation and a line-cap
                # truncation are different facts, and telling the model "exceeds
                # 1 MiB" when the file is actually < 1 MiB (just long) is a
                # false statement about what it is not seeing.
                cause = "file exceeds 1 MiB" if truncated else f"showing first {_FILE_CONTEXT_MAX_LINES} lines"
                out.append(f"**Total lines**: >={total} (head only — {cause})")
            else:
                out.append(f"**Total lines**: {total}")

            return "\n".join(out)
        except Exception as e:
            logger.exception("Failed to build file context for %s: %s", rel_path, e)
            return ""

    def _build_related_files_context(
        self,
        target_file: str,
        max_files: int = 3,
    ) -> str:
        try:
            related_files = self._find_related_files(target_file, max_files)
            if not related_files:
                return ""

            parts: list[str] = []
            for idx, rel_file in enumerate(related_files, 1):
                file_path = self.repo_root / rel_file
                if not file_path.exists() or not file_path.is_file():
                    continue

                try:
                    # P21-3: head-bounded read per related file.
                    content, truncated = _bounded_file_text(file_path)
                except Exception:
                    logger.debug("related file read failed: %s", rel_file, exc_info=True)
                    continue

                # P21-3 output bound applies to related files too: the target
                # block is capped at 5000 numbered lines (~70-90 KB), but the
                # raw related head used to be embedded UNcapped — up to 1 MiB
                # of prompt per related file (x3), dwarfing the target's own
                # output bound.  Same line cap as the target block.
                snippet_lines = content.split("\n")
                if len(snippet_lines) > _FILE_CONTEXT_MAX_LINES:
                    snippet = "\n".join(snippet_lines[:_FILE_CONTEXT_MAX_LINES])
                    snippet += f"\n...[more lines omitted — showing first {_FILE_CONTEXT_MAX_LINES}]"
                else:
                    snippet = content
                if truncated:
                    snippet += "\n...[TRUNCATED — head only]..."

                lang = self._detect_language(rel_file)

                parts.append(f"### {idx}. `{rel_file}`")
                parts.append("")
                parts.append(f"```{lang}")
                parts.append(snippet)
                parts.append("```")
                parts.append("")

            return "\n".join(parts) if parts else ""
        except Exception as e:
            logger.debug("Failed to build related files context: %s", e)
            return ""

    def _find_related_files(self, target_file: str, max_files: int) -> list[str]:
        # Normalized ONCE and used by BOTH paths: the context_collector
        # comparison below and the fallback's candidate comparison.  The
        # fallback used to compare the raw ``target_file`` — a
        # "./pkg/__init__.py" form never matched "pkg/__init__.py", so a
        # target importing its own package leaked into its own Related
        # Files (defect A class, fallback half).
        rel = normalize_rel_path(target_file)
        # 1) Preferred: context_collector (shallow)
        try:
            from context_collector import collect_related_files_shallow  # type: ignore

            selected, _meta = collect_related_files_shallow(str(self.repo_root), target_file)
            # Must match context_collector.collect_related_files_shallow's
            # normalization exactly so the target can be excluded.
            # removeprefix("./") (NOT lstrip("./")) — lstrip takes a character
            # SET {'.','/'} and would strip a dotfile's leading dot, e.g.
            # ".config.py" -> "config.py", leaking the target into its own
            # Related Files list.  See go_provider.py for the same fix.
            related = [x for x in (selected or []) if x and x != rel]
            if related:
                return related[:max_files]
        except Exception as e:
            logger.debug("context_collector unavailable or failed, falling back: %s", e)

        # 2) Fallback: simple Python import parsing
        try:
            # P22-2: containment guard — target_file must not escape the repo.
            try:
                file_path = resolve_inside_repo(str(self.repo_root), str(target_file))
            except ValueError:
                return []
            if not file_path.exists() or not file_path.is_file():
                return []

            if not is_supported(target_file, AnalysisCapability.CONTEXT_BUILDING):
                return []

            try:
                # P21-3: import scanning only needs the head.
                content, _trunc = _bounded_file_text(file_path)
            except (OSError, UnicodeDecodeError, ValueError):  # unreadable target
                return []

            related: list[str] = []

            for line in content.splitlines():
                stripped = line.strip()
                if stripped.startswith("from ") and " import " in stripped:
                    module = stripped[5:].split(" import ", 1)[0].strip()
                elif stripped.startswith("import "):
                    module = stripped[7:].strip().split()[0].strip()
                else:
                    continue
                if not module or module.startswith("."):
                    continue

                top = module.split(".")[0]
                candidates = [
                    self.repo_root / f"{top}.py",
                    self.repo_root / top / "__init__.py",
                ]

                for cand in candidates:
                    if cand.exists() and cand.is_file():
                        relp = str(cand.relative_to(self.repo_root))
                        if relp != rel and relp not in related:
                            related.append(relp)
                            break

                if len(related) >= max_files:
                    break

            return related[:max_files]
        except Exception as e:
            logger.debug("Failed to find related files: %s", e)
            return []

    def _get_project_structure_hints(self) -> str:
        key = str(self.repo_root)
        now = time.monotonic()
        cached = _structure_hints_cache.get(key)
        if cached is not None:
            text, expires = cached
            if now < expires:
                return text
            # Stale entry: drop it so it can't linger.  Without this, keys
            # that expire but are never re-accessed would accumulate forever.
            _structure_hints_cache.pop(key, None)
        # Opportunistic GC: purge other expired entries to bound memory.
        # Only runs on the (rare) miss path; cache hits stay O(1).
        if len(_structure_hints_cache) > _STRUCTURE_HINTS_GC_THRESHOLD:
            for _k, (_t, _exp) in list(_structure_hints_cache.items()):
                if _exp <= now:
                    _structure_hints_cache.pop(_k, None)
        parts: list[str] = []
        try:
            dirs: list[str] = []
            files: list[str] = []
            for item in sorted(self.repo_root.iterdir(), key=lambda p: p.name.lower()):
                if item.name.startswith("."):
                    continue
                if item.is_dir():
                    # Count .py files with directory pruning instead of rglob,
                    # which would descend into node_modules/.git/etc.
                    _skip = {
                        ".git",
                        "__pycache__",
                        "node_modules",
                        ".venv",
                        "venv",
                        ".mypy_cache",
                        ".pytest_cache",
                        "build",
                        "dist",
                    }
                    py_count = 0
                    for _root, _dirs, _files in os.walk(item):
                        _dirs[:] = [d for d in _dirs if d not in _skip]
                        py_count += sum(1 for f in _files if f.endswith(".py"))
                    if py_count > 0:
                        dirs.append(f"  - `{item.name}/` ({py_count} .py files)")
                elif item.suffix in [".py", ".md", ".txt", ".yaml", ".yml", ".json"]:
                    files.append(f"  - `{item.name}`")

            if dirs or files:
                parts.append("```")
                parts.append(f"{self.repo_root.name}/")
                parts.extend(dirs[:10])
                parts.extend(files[:5])
                parts.append("```")
        except Exception as e:
            logger.debug("Failed to get project structure: %s", e)
            # Don't cache failures — let the next call retry.
            return ""
        result = "\n".join(parts) if parts else ""
        _structure_hints_cache[key] = (result, now + _STRUCTURE_HINTS_TTL_S)
        return result

    def _detect_language(self, filename: str) -> str:
        ext = Path(filename).suffix.lower()
        lang_map = {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".jsx": "jsx",
            ".tsx": "tsx",
            ".java": "java",
            ".cpp": "cpp",
            ".c": "c",
            ".go": "go",
            ".rs": "rust",
            ".rb": "ruby",
            ".sh": "bash",
            ".yaml": "yaml",
            ".yml": "yaml",
            ".json": "json",
            ".md": "markdown",
        }
        return lang_map.get(ext, "")

    def _get_llm_instructions(self, target_file: str | None = None) -> str:
        file_hint = f" for `{target_file}`" if target_file else ""
        return f"""**Your Task**: Generate a unified diff patch{file_hint}

**Critical Requirements**:
1. Output ONLY valid unified diff format
2. Start with `diff --git a/... b/...`
3. Include `---` and `+++` headers
4. Use `@@ -X,Y +A,B @@` hunk headers
5. Include 3 context lines before and after each change
6. Preserve exact indentation
7. Make minimal, focused changes
"""


# -----------------------------
# Module-level API (REQUIRED)
# -----------------------------
# external_llm.service imports these names.

ContextBuilder = EnhancedContextBuilder


def enhance_user_request(
    user_request: str,
    target_file: str | None = None,
    extra_hints: list[str] | None = None,
) -> str:
    """
    Small helper expected by some service implementations.

    Keeps behavior conservative: appends a short hint block to the user's request.
    """
    hints: list[str] = []
    if target_file:
        hints.append(f"- Target file: {target_file}")
    if extra_hints:
        hints.extend([f"- {h}" for h in extra_hints if h])

    if not hints:
        return user_request

    return user_request.rstrip() + "\n\n[HINTS]\n" + "\n".join(hints) + "\n"
