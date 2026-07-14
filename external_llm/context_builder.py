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
import subprocess
import time
import os
from pathlib import Path
from typing import Optional

from .languages.capabilities import AnalysisCapability, is_supported
from common import normalize_rel_path_fast
logger = logging.getLogger(__name__)

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


class EnhancedContextBuilder:
    def __init__(self, repo_root: str):
        self.repo_root = Path(repo_root).resolve()

    def build_context(
        self,
        user_request: str,
        target_file: Optional[str] = None,
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
        try:
            result = subprocess.run(
                ["git", "status", "--short"],
                cwd=str(self.repo_root),
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
        except Exception as e:
            logger.debug(f"Git status failed: {e}")
        return ""

    def _get_recent_commits(self, count: int = 3) -> str:
        try:
            result = subprocess.run(
                ["git", "log", f"-{count}", "--oneline", "--decorate"],
                cwd=str(self.repo_root),
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
        except Exception as e:
            logger.debug(f"Git log failed: {e}")
        return ""

    def _build_file_context(self, rel_path: str) -> str:
        try:
            file_path = self.repo_root / rel_path
            if not file_path.exists():
                logger.warning(f"File not found: {rel_path}")
                return ""

            try:
                content = file_path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                content = file_path.read_text(encoding="latin-1")

            lines = content.split("\n")
            total = len(lines)
            lang = self._detect_language(rel_path)

            out: list[str] = []
            out.append(f"```{lang}")
            for i, line in enumerate(lines, 1):
                out.append(f"{i:4d} | {line}")
            out.append("```")
            out.append("")
            out.append(f"**Total lines**: {total}")

            return "\n".join(out)
        except Exception as e:
            logger.error(f"Failed to build file context for {rel_path}: {e}")
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
                    try:
                        content = file_path.read_text(encoding="utf-8")
                    except UnicodeDecodeError:
                        content = file_path.read_text(encoding="latin-1")
                except Exception:
                    continue

                snippet = content

                lang = self._detect_language(rel_file)

                parts.append(f"### {idx}. `{rel_file}`")
                parts.append("")
                parts.append(f"```{lang}")
                parts.append(snippet)
                parts.append("```")
                parts.append("")

            return "\n".join(parts) if parts else ""
        except Exception as e:
            logger.debug(f"Failed to build related files context: {e}")
            return ""

    def _find_related_files(self, target_file: str, max_files: int) -> list[str]:
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
            rel = normalize_rel_path_fast(target_file)
            related = [x for x in (selected or []) if x and x != rel]
            if related:
                return related[:max_files]
        except Exception as e:
            logger.debug(f"context_collector unavailable or failed, falling back: {e}")

        # 2) Fallback: simple Python import parsing
        try:
            file_path = self.repo_root / target_file
            if not file_path.exists() or not file_path.is_file():
                return []

            if not is_supported(target_file, AnalysisCapability.CONTEXT_BUILDING):
                return []

            try:
                content = file_path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                content = file_path.read_text(encoding="latin-1")

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
                        if relp != target_file and relp not in related:
                            related.append(relp)
                            break

                if len(related) >= max_files:
                    break

            return related[:max_files]
        except Exception as e:
            logger.debug(f"Failed to find related files: {e}")
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
            for item in self.repo_root.iterdir():
                if item.name.startswith("."):
                    continue
                if item.is_dir():
                    # Count .py files with directory pruning instead of rglob,
                    # which would descend into node_modules/.git/etc.
                    _skip = {".git", "__pycache__", "node_modules", ".venv",
                             "venv", ".mypy_cache", ".pytest_cache", "build", "dist"}
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
            logger.debug(f"Failed to get project structure: {e}")
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

    def _get_llm_instructions(self, target_file: Optional[str] = None) -> str:
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
    target_file: Optional[str] = None,
    extra_hints: Optional[list[str]] = None,
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
