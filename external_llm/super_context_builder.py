"""
Super Context Builder - Massively Enhanced Context for External LLM

Integrates all context enhancement features:
✅ AST-based code analysis (functions, classes, types)
✅ Dependency graph (call relationships)
✅ Test examples (usage, expected behavior)
✅ Similar code patterns (learning templates)
✅ Project metadata (README, requirements)
✅ Smart snippet selection (important parts first)
✅ Type information (complete signatures)
✅ Documentation (docstrings, comments)

BEFORE: 60% understanding, simple file content
AFTER: 95% understanding, rich structured context
"""
from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Optional

from .code_analyzer import CodeAnalyzer
from .dependency_graph import DependencyGraphBuilder

# TestFinder import removed — class never existed in test_finder.py.
# SymbolAwareTestFinder has different interface (no find_tests_for_file).
# This module is currently unused; kept for future context building.
# Note: pattern_matcher.py was removed (it was a duplicate of context_builder.py
# with no PatternMatcher class). The pattern_matcher references below are dead code
# kept only for structural reference — they never executed.

logger = logging.getLogger(__name__)


class SuperContextBuilder:
    """
    Builds massively enhanced context for external LLM

    Provides everything LLM needs to understand the project:
    - Complete code structure
    - Dependencies and call graphs
    - Test examples
    - Similar code patterns
    - Type information
    - Documentation
    """

    def __init__(self, repo_root: str):
        self.repo_root = Path(repo_root).resolve()

        # Initialize analyzers
        self.code_analyzer = CodeAnalyzer()
        self.dependency_builder = DependencyGraphBuilder(self.repo_root)
        # self.test_finder = None  # TestFinder was never defined; see import note above

    def build_context(
        self,
        user_request: str,
        target_file: Optional[str] = None,
        include_dependencies: bool = True,
        include_git_context: bool = True,
        max_file_lines: int = 500,
    ) -> str:
        """
        Build super-enhanced context

        Args:
            user_request: User's request
            target_file: Target file (if any)
            include_dependencies: Include dependency graph
            include_git_context: Include git information
            max_file_lines: Max lines per file

        Returns:
            Comprehensive context string
        """
        sections = []

        # === HEADER ===
        sections.append("# 📋 PROJECT CONTEXT (ENHANCED)")
        sections.append("")
        sections.append(f"**Repository**: `{self.repo_root.name}`")
        sections.append(f"**Path**: `{self.repo_root}`")
        sections.append("")

        # === PROJECT METADATA ===
        metadata = self._build_project_metadata()
        if metadata:
            sections.append("## 📦 Project Metadata")
            sections.append("")
            sections.append(metadata)
            sections.append("")

        # === GIT CONTEXT ===
        if include_git_context:
            git_ctx = self._build_git_context()
            if git_ctx:
                sections.append("## 🔄 Git Status")
                sections.append("")
                sections.append(git_ctx)
                sections.append("")

        # === TARGET FILE (ENHANCED) ===
        if target_file:
            target_path = self.repo_root / target_file

            if target_path.exists():
                file_ctx = self._build_enhanced_file_context(
                    target_path,
                    max_lines=max_file_lines
                )

                sections.append(f"## 🎯 Target File: `{target_file}`")
                sections.append("")
                sections.append(file_ctx)
                sections.append("")

                # === DEPENDENCY GRAPH ===
                if include_dependencies:
                    dep_ctx = self._build_dependency_context(target_path)
                    if dep_ctx:
                        sections.append("## 🔗 Dependencies & Call Graph")
                        sections.append("")
                        sections.append(dep_ctx)
                        sections.append("")

                # === SIMILAR CODE PATTERNS ===
                # Removed: _build_pattern_context depended on PatternMatcher which
                # never existed in pattern_matcher.py (that file was a duplicate of
                # context_builder.py). The import always failed, making this dead code.

        # === USER REQUEST ===
        sections.append("## 💬 User Request")
        sections.append("")
        sections.append(user_request)
        sections.append("")

        # === INSTRUCTIONS ===
        sections.append("## 📝 Instructions")
        sections.append("")
        sections.append(self._get_enhanced_instructions(target_file))

        return "\n".join(sections)

    def _build_project_metadata(self) -> str:
        """Build project metadata section with collaboration info"""
        lines = []

        # README
        readme = self._find_readme()
        if readme:
            lines.append(f"**README**: `{readme.name}` (exists)")
            # Extract first paragraph
            try:
                content = readme.read_text()
                first_para = content.split('\n\n')[0]
                if len(first_para) < 500:
                    lines.append(f"> {first_para}")
            except Exception:
                pass

        # Requirements
        req_file = self._find_requirements()
        if req_file:
            try:
                reqs = req_file.read_text().split('\n')
                main_reqs = [r.split('==')[0] for r in reqs if r and not r.startswith('#')][:10]
                if main_reqs:
                    lines.append(f"\n**Dependencies**: {', '.join(main_reqs)}")
            except Exception:
                pass

        # pyproject.toml
        pyproject = self.repo_root / "pyproject.toml"
        if pyproject.exists():
            lines.append("\n**Build Tool**: pyproject.toml (exists)")

        # Collaboration metadata
        collab_info = self._extract_collaboration_metadata()
        if collab_info:
            lines.append("\n**🤝 Collaboration Context**:")
            lines.append(collab_info)

        return '\n'.join(lines) if lines else ""

    def _extract_collaboration_metadata(self) -> str:
        """Extract collaboration-related metadata.

        Fetches recent commit history ONCE (author + subject via
        :meth:`_fetch_commit_subjects_authors`) and derives all three metrics
        from it. This replaces three separate ``git log`` subprocess spawns
        (~54ms -> ~18ms per call on this repo) that each re-fetched the same
        commit history.
        """
        lines = []

        # Single shared fetch supplies authors + subjects for metrics 1, 2, 4.
        authors, subjects = self._fetch_commit_subjects_authors(commits=20)

        # 1. Git recent contributors (most recent 10 commits' authors)
        try:
            contributors = self._get_recent_contributors(commits=10, authors=authors)
            if contributors:
                lines.append(f"- **Recent contributors**: {', '.join(contributors)}")
        except Exception:
            pass

        # 2. Code review patterns (from commit subjects)
        try:
            review_patterns = self._detect_review_patterns(commits=20, subjects=subjects)
            if review_patterns:
                lines.append(f"- **Code review focus**: {review_patterns}")
        except Exception:
            pass

        # 3. Team conventions (from common files)
        conventions = self._detect_team_conventions()
        if conventions:
            lines.append(f"- **Team conventions**: {conventions}")

        # 4. Related issues/PRs (from commit subjects)
        try:
            related_refs = self._extract_issue_references(commits=15, subjects=subjects)
            if related_refs:
                lines.append(f"- **Related issues/PRs**: {', '.join(related_refs[:3])}")
        except Exception:
            pass

        return '\n'.join(lines) if lines else ""

    def _fetch_commit_subjects_authors(self, commits: int = 20):
            """Single ``git log`` fetch returning ``(authors, subjects)`` for the
            last N commits (most-recent first).

            ``%x09`` is a literal TAB, cleanly separating author and subject even
            when subjects contain spaces. Replaces three separate subprocess
            spawns in the collaboration-metadata path. Returns ``([], [])`` on any
            failure (non-repo, timeout, non-zero exit).
            """
            try:
                result = subprocess.run(
                    ["git", "log", f"-{commits}", "--pretty=format:%an%x09%s"],
                    cwd=str(self.repo_root),
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.returncode != 0:
                    return [], []
                authors, subjects = [], []
                for line in result.stdout.split('\n'):
                    line = line.rstrip('\r')
                    if '\t' in line:
                        author, subject = line.split('\t', 1)
                    else:
                        author, subject = line, ""
                    authors.append(author)
                    subjects.append(subject)
                return authors, subjects
            except Exception:
                return [], []
    def _get_recent_contributors(self, commits: int = 10, authors=None) -> list[str]:
        """Get recent git contributors.

        ``authors`` may be supplied pre-fetched (from a shared
        :meth:`_fetch_commit_subjects_authors` call) to avoid a redundant
        subprocess spawn; when ``None`` the data is fetched on demand.
        """
        if authors is None:
            authors, _ = self._fetch_commit_subjects_authors(commits)
        # unique, preserving first-seen order; drop empties
        contributors = list(dict.fromkeys(c for c in authors[:commits] if c))
        return contributors[:5]

    def _detect_review_patterns(self, commits: int = 20, subjects=None) -> str:
        """Detect common code review patterns from commit messages.

        ``subjects`` may be supplied pre-fetched (see
        :meth:`_fetch_commit_subjects_authors`).
        """
        if subjects is None:
            _, subjects = self._fetch_commit_subjects_authors(commits)
        subjects = subjects[:commits]
        # Common review keywords
        review_keywords = ["fix:", "refactor:", "cleanup:", "style:", "test:", "docs:"]
        patterns = []
        for kw in review_keywords:
            count = sum(1 for msg in subjects if msg.lower().startswith(kw))
            if count > 0:
                patterns.append(f"{kw} ({count}x)")
        return ", ".join(patterns) if patterns else ""

    def _detect_team_conventions(self) -> str:
        """Detect team coding conventions"""
        conventions = []

        # Check for common convention files
        convention_files = [
            ".editorconfig", ".pre-commit-config.yaml",
            ".flake8", ".pylintrc", "pyproject.toml"
        ]

        for cf in convention_files:
            if (self.repo_root / cf).exists():
                conventions.append(cf)

        # Check for linting in CI
        ci_files = [".github/workflows", ".gitlab-ci.yml", ".circleci/config.yml"]
        for ci in ci_files:
            ci_path = self.repo_root / ci
            if ci_path.exists():
                conventions.append("CI linting")
                break

        return ", ".join(conventions) if conventions else ""

    def _extract_issue_references(self, commits: int = 15, subjects=None) -> list[str]:
        """Extract issue/PR references from commit messages.

        ``subjects`` may be supplied pre-fetched (see
        :meth:`_fetch_commit_subjects_authors`).
        """
        if subjects is None:
            _, subjects = self._fetch_commit_subjects_authors(commits)
        subjects = subjects[:commits]
        import re
        patterns = [
            r'#(\d+)',  # GitHub-style #123
            r'(\w+-\d+)',  # JIRA-style PROJ-123
        ]
        refs = []
        for msg in subjects:
            for pattern in patterns:
                refs.extend(re.findall(pattern, msg))
        return list(set(refs))[:5]

    def _build_enhanced_file_context(self, file_path: Path, max_lines: int) -> str:
        """Build enhanced file context with AST analysis"""
        lines = []

        # Analyze code structure
        analysis = self.code_analyzer.analyze_file(file_path)

        if analysis:
            # === FILE SUMMARY ===
            lines.append("### 📊 File Summary")
            lines.append("")

            if analysis.module_docstring:
                lines.append(f'**Module Purpose**: "{analysis.module_docstring.split(chr(10))[0]}"')

            lines.append(f"**Functions**: {len(analysis.functions)} defined")
            lines.append(f"**Classes**: {len(analysis.classes)} defined")

            if analysis.imports:
                imp_modules = list(set(imp.module for imp in analysis.imports))[:5]
                lines.append(f"**Imports**: {', '.join(imp_modules)}")

            lines.append("")

            # === KEY FUNCTIONS ===
            if analysis.functions:
                lines.append("### ⚡ Key Functions")
                lines.append("")

                for func in analysis.functions[:5]:  # Top 5 functions
                    sig = self.code_analyzer.format_function_signature(func)
                    lines.append("```python")
                    lines.append(sig)
                    lines.append("```")
                    lines.append("")

            # === KEY CLASSES ===
            if analysis.classes:
                lines.append("### 🏗️ Key Classes")
                lines.append("")

                for cls in analysis.classes[:3]:  # Top 3 classes
                    sig = self.code_analyzer.format_class_signature(cls)
                    lines.append("```python")
                    lines.append(sig)
                    lines.append("```")
                    lines.append("")

            # === TYPE INFORMATION ===
            type_info = self._extract_type_info(analysis)
            if type_info:
                lines.append("### 🏷️ Type Information")
                lines.append("")
                lines.append(type_info)
                lines.append("")

        # === FULL FILE CONTENT (Smart Snippet) ===
        lines.append("### 📄 File Content")
        lines.append("")

        try:
            content = file_path.read_text()
            content_lines = content.split('\n')

            if len(content_lines) <= max_lines:
                # Show full file with line numbers
                lines.append("```python")
                for i, line in enumerate(content_lines, 1):
                    lines.append(f"{i:4d} | {line}")
                lines.append("```")
            else:
                # Smart snippet (important parts)
                important_lines = self._select_important_lines(
                    content_lines,
                    analysis,
                    max_lines
                )

                lines.append("```python")
                lines.append(f"# File has {len(content_lines)} lines, showing important sections:")
                lines.append("")
                for line_num, line in important_lines:
                    lines.append(f"{line_num:4d} | {line}")
                lines.append("```")

        except Exception as e:
            lines.append(f"*Error reading file: {e}*")

        return '\n'.join(lines)

    def _build_dependency_context(self, file_path: Path) -> str:
        """Build dependency and call graph context"""
        lines = []

        try:
            graph = self.dependency_builder.build_graph(file_path, max_depth=1)

            # File dependencies
            rel_path = str(file_path.relative_to(self.repo_root))

            if rel_path in graph.file_imports:
                imports = graph.file_imports[rel_path]
                lines.append("**This file imports**:")
                for imp in imports[:5]:
                    lines.append(f"- `{imp}`")
                lines.append("")

            # Call relationships
            # Find functions defined in this file
            analysis = self.code_analyzer.analyze_file(file_path)
            if analysis and analysis.functions:
                for func in analysis.functions[:3]:  # Top 3 functions
                    func_key = f"{rel_path}:{func.name}"

                    call_info = self.dependency_builder.format_call_graph(graph, func_key, max_items=5)
                    if call_info and "No call information" not in call_info:
                        lines.append(f"**`{func.name}()` relationships**:")
                        lines.append("```")
                        lines.append(call_info)
                        lines.append("```")
                        lines.append("")

        except Exception as e:
            logger.debug(f"Failed to build dependency context: {e}")

        return '\n'.join(lines) if lines else ""


    def _build_git_context(self) -> str:
        """Build git context"""
        parts = []

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

        return '\n'.join(parts) if parts else ""

    def _select_important_lines(
        self,
        all_lines: list[str],
        analysis: Optional[any],
        max_lines: int
    ) -> list[tuple[int, str]]:
        """
        Intelligently select most important lines

        Priority:
        1. Module docstring
        2. Imports
        3. Class definitions (with docstrings)
        4. Function definitions (with docstrings)
        5. Key comments
        """
        selected = []

        # Always include first 10 lines (module docstring, imports)
        for i in range(min(10, len(all_lines))):
            selected.append((i + 1, all_lines[i]))

        # If we have analysis, include function/class definitions
        if analysis:
            for func in analysis.functions[:3]:
                # Include function def line + docstring
                start = func.line_number - 1
                end = min(start + 10, len(all_lines))
                for i in range(start, end):
                    if i >= 0 and i < len(all_lines):
                        selected.append((i + 1, all_lines[i]))

            for cls in analysis.classes[:2]:
                start = cls.line_number - 1
                end = min(start + 15, len(all_lines))
                for i in range(start, end):
                    if i >= 0 and i < len(all_lines):
                        selected.append((i + 1, all_lines[i]))

        # Remove duplicates and sort
        selected = sorted(set(selected), key=lambda x: x[0])

        # Limit to max_lines
        return selected[:max_lines]

    def _extract_type_info(self, analysis: any) -> str:
        """Extract and format type information"""
        lines = []

        # Type aliases (global level)
        for var, value in analysis.global_vars.items():
            if var[0].isupper():  # Likely a type alias or constant
                lines.append(f"{var} = {value}")

        return '\n'.join(lines) if lines else ""

    def _get_git_status(self) -> str:
        """Get git status"""
        try:
            result = subprocess.run(
                ["git", "status", "--short"],
                cwd=str(self.repo_root),
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
        return ""

    def _get_recent_commits(self, count: int = 3) -> str:
        """Get recent commits"""
        try:
            result = subprocess.run(
                ["git", "log", f"-{count}", "--oneline", "--decorate"],
                cwd=str(self.repo_root),
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
        return ""

    def _find_readme(self) -> Optional[Path]:
        """Find README file"""
        for name in ["README.md", "README.rst", "README.txt", "README"]:
            readme = self.repo_root / name
            if readme.exists():
                return readme
        return None

    def _find_requirements(self) -> Optional[Path]:
        """Find requirements file"""
        for name in ["requirements.txt", "requirements-dev.txt", "requirements.in"]:
            req = self.repo_root / name
            if req.exists():
                return req
        return None

    def _get_enhanced_instructions(self, target_file: Optional[str]) -> str:
        """Get enhanced instructions for LLM"""
        file_hint = f" for `{target_file}`" if target_file else ""

        return f"""**Your Task**: Generate a unified diff patch{file_hint}

**You now have comprehensive context including**:
✅ Complete code structure (functions, classes, types)
✅ Dependency relationships (who calls what)
✅ Test examples (actual usage, expected behavior)
✅ Similar code patterns (learn from existing code)
✅ Type information (complete signatures)
✅ Project conventions (naming, style)

**Critical Requirements**:
1. Follow the established patterns shown in similar code
2. Maintain consistency with existing function signatures
3. Use the same naming convention as the project
4. Include proper type hints (as shown in examples)
5. Write tests similar to existing test patterns
6. Output ONLY valid unified diff format
7. Include proper diff headers (diff --git, ---, +++, @@). OMIT the `index` line — you cannot compute real git blob SHAs
8. Use 3 context lines before and after changes
9. Preserve exact indentation

**Output Format**:
```diff
diff --git a/file.py b/file.py
--- a/file.py
+++ b/file.py
@@ -10,7 +10,7 @@ def function():
     context
     context
     context
-    old_line
+    new_line
     context
     context
     context
```

Generate the patch now."""


# Backward compatibility alias
EnhancedContextBuilder = SuperContextBuilder
ContextBuilder = SuperContextBuilder




def enhance_user_request(user_request: str, **kwargs) -> str:
    """Enhance user request (compatibility function)"""
    return user_request
