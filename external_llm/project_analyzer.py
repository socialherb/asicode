"""
Project Structure Analyzer for asicode

Analyzes project structure to help LLM understand:
1. Directory organization
2. Naming conventions
3. Code patterns
4. File relationships
5. Common imports/dependencies

This provides rich context for general requests like "create login functionality".
"""
from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .languages import LanguageId

logger = logging.getLogger(__name__)


@dataclass
class ProjectStructure:
    """Analyzed project structure"""

    # Framework/tech detected
    framework: Optional[str] = None

    # All detected frameworks (multi-framework support)
    frameworks: list[str] = field(default_factory=list)

    # Detected source languages (ranked by file count, most common first)
    languages: list[str] = field(default_factory=list)

    # Primary (most common) source language, or None if undetermined
    primary_language: Optional[str] = None

    # Project types: 'web', 'cli', 'mobile', 'library', 'package'
    project_types: list[str] = field(default_factory=list)

    # Directory structure
    directories: dict[str, list[str]] = field(default_factory=dict)  # {purpose: [paths]}

    # File patterns
    file_patterns: dict[str, str] = field(default_factory=dict)  # {pattern_name: template}

    # Naming conventions
    naming_style: str = "snake_case"  # snake_case, camelCase, PascalCase

    # Common imports
    common_imports: list[str] = field(default_factory=list)

    # Entry points
    entry_points: list[str] = field(default_factory=list)

    # Test directory
    test_dir: Optional[str] = None

    # Example files for reference
    example_files: dict[str, str] = field(default_factory=dict)  # {type: path}


class ProjectAnalyzer:
    """
    Analyzes project structure to provide context for LLM

    Detects:
    - Framework (Django, Flask, FastAPI, React, etc.)
    - Directory organization (MVC, feature-based, etc.)
    - Coding patterns and conventions
    - File templates and structures
    """

    # Framework detection patterns.
    #
    # Each marker is ``(marker_type, value, weight)``.  Weights separate
    # *definitive* signals (an import of the framework, a declared dependency,
    # a framework-specific config file → weight 3) from *generic/shared* ones
    # (a bare ``migrations`` dir, ``wsgi.py``, ``app.py`` → weight 1) that
    # several frameworks—or none—could explain.  A framework must reach
    # ``MIN_FRAMEWORK_SCORE`` to be reported, so a lone generic marker can no
    # longer confirm a framework on its own (e.g. an Alembic ``migrations/``
    # dir must not imply Django).  See ``_detect_frameworks``.
    FRAMEWORK_MARKERS = {
        'django': [
            ('file', 'manage.py', 3),
            ('import', 'from django.', 3),
            ('py_dep', 'django', 4),
            ('file', 'wsgi.py', 1),
            ('dir', 'migrations', 1),
        ],
        'fastapi': [
            ('import', 'from fastapi', 3),
            ('import', 'import fastapi', 3),
            ('py_dep', 'fastapi', 4),
            ('file', 'main.py', 1),  # with FastAPI app
        ],
        'flask': [
            ('import', 'from flask', 3),
            ('import', 'import flask', 3),
            ('py_dep', 'flask', 4),
            ('file', 'app.py', 1),
        ],
        'react': [
            ('pkg_dep', 'react', 3),
            ('file_ext', '.jsx', 1),
            ('dir', 'node_modules', 1),
        ],
        'nextjs': [
            ('file', 'next.config.ts', 3),
            ('file', 'next.config.js', 3),
            ('dir', '.next', 2),
            ('dir', 'pages', 1),
        ],
        'vue': [
            ('pkg_dep', 'vue', 3),
            ('file_ext', '.vue', 2),
        ],
        'nuxt': [
            ('file', 'nuxt.config.ts', 3),
            ('file', 'nuxt.config.js', 3),
            ('dir', '.nuxt', 2),
        ],
        'astro': [
            ('file', 'astro.config.mjs', 3),
            ('file_ext', '.astro', 2),
        ],

        # Python CLI frameworks. ``argparse`` is Python stdlib, not a real
        # third-party framework — it is tracked in STDLIB_CLI (used only for
        # project-type classification) and deliberately omitted from here so a
        # bare ``import argparse`` can never inflate the framework list.
        'click': [
            ('import', 'import click', 2),
            ('import', 'from click', 2),
            ('py_dep', 'click', 4),
        ],
        'typer': [
            ('import', 'import typer', 2),
            ('import', 'from typer', 2),
            ('py_dep', 'typer', 4),
        ],

        # Python libraries commonly worth surfacing as a detected dependency.
        # Detected purely via manifest declaration — their import usage is too
        # ubiquitous / alias-prone to scan reliably.
        'pydantic': [
            ('py_dep', 'pydantic', 4),
        ],
        'rich': [
            ('py_dep', 'rich', 4),
        ],
        'libcst': [
            ('py_dep', 'libcst', 4),
        ],
        'httpx': [
            ('py_dep', 'httpx', 3),
        ],
        'requests': [
            ('py_dep', 'requests', 3),
        ],
        'sqlalchemy': [
            ('py_dep', 'sqlalchemy', 4),
            ('import', 'from sqlalchemy', 3),
        ],
        'pytest': [
            ('py_dep', 'pytest', 3),
            ('file', 'pytest.ini', 2),
            ('file', 'conftest.py', 2),
        ],
        # tree-sitter: import-based detection is unreliable (the library is
        # often wrapped, e.g. ``from external_llm.languages.tree_sitter_utils``),
        # so rely on the manifest + presence of grammar packages.
        'tree-sitter': [
            ('py_dep', 'tree-sitter', 4),
            ('py_dep', 'tree-sitter-python', 2),
            ('py_dep', 'tree-sitter-javascript', 2),
        ],

        # Go CLI frameworks
        'cobra': [
            ('go_dep', 'github.com/spf13/cobra', 3),
        ],
        'fang': [
            ('go_dep', 'charm.land/fang', 3),
        ],
        'urfave-cli': [
            ('go_dep', 'github.com/urfave/cli', 3),
        ],
        'kong': [
            ('go_dep', 'github.com/alecthomas/kong', 3),
        ],
        # Go web frameworks
        'gin': [
            ('go_dep', 'github.com/gin-gonic/gin', 3),
        ],
        'echo': [
            ('go_dep', 'github.com/labstack/echo', 3),
        ],
        'fiber': [
            ('go_dep', 'github.com/gofiber/fiber', 3),
        ],
        'chi': [
            ('go_dep', 'github.com/go-chi/chi', 3),
        ],
        'gorilla-mux': [
            ('go_dep', 'github.com/gorilla/mux', 3),
        ],

        # JVM / Android. Kotlin is reported as a *language* by
        # _detect_languages, so it is not listed as a framework here. Android
        # app/library modules and Jetpack Compose are detected via Gradle
        # build files + the version catalog (gradle_text) and Kotlin/Java
        # imports (jvm_import). The catalog path matters because projects
        # declare plugins as ``alias(libs.plugins.android.application)``
        # rather than inline ids — the real id lives in libs.versions.toml.
        'android': [
            ('gradle_text', 'com.android.application', 4),  # app plugin — definitive
            ('gradle_text', 'com.android.library', 3),      # library module
            ('gradle_text', 'applicationId', 2),            # app module config
        ],
        'jetpack-compose': [
            ('jvm_import', 'import androidx.compose', 3),
            ('gradle_text', 'androidx.compose', 3),
        ],
    }

    # Python standard-library modules sometimes used as a CLI entry point.
    # These are *not* third-party frameworks and must never appear in the
    # detected framework list, but they are still evidence of a CLI project
    # type. ``_detect_project_types`` consults this set directly.
    STDLIB_CLI = frozenset({'argparse'})


    # Maps each framework to the language(s) it can only appear in.  A
    # framework is scored only when one of its languages is actually present
    # in the repo, which prevents cross-language false positives (e.g. a Go
    # repo can never match a Python/JS framework). Frameworks omitted here are
    # not language-gated.
    FRAMEWORK_LANGUAGES = {
        'django': {'python'},
        'fastapi': {'python'},
        'flask': {'python'},
        'argparse': {'python'},
        'click': {'python'},
        'typer': {'python'},
        'react': {'javascript', 'typescript'},
        'nextjs': {'javascript', 'typescript'},
        'vue': {'javascript', 'typescript'},
        'nuxt': {'javascript', 'typescript'},
        'astro': {'javascript', 'typescript'},
        'cobra': {'go'},
        'fang': {'go'},
        'urfave-cli': {'go'},
        'kong': {'go'},
        'gin': {'go'},
        'echo': {'go'},
        'fiber': {'go'},
        'chi': {'go'},
        'gorilla-mux': {'go'},
        'android': {'kotlin', 'java'},
        'jetpack-compose': {'kotlin', 'java'},
    }

    # Minimum total score for a framework to be reported. With the weighting
    # above this means a single generic marker (weight 1) is never enough.
    MIN_FRAMEWORK_SCORE = 2

    # A language must account for at least this share of recognized source
    # files (and at least MIN_LANGUAGE_FILES files) to count as "present".
    # Guards against a couple of stray scripts (e.g. build .py files in a Go
    # repo) re-enabling another language's framework detection.
    MIN_LANGUAGE_SHARE = 0.05
    MIN_LANGUAGE_FILES = 2

    # Directories never worth scanning for language/framework signals.
    SKIP_DIRS = frozenset({
        '.git', 'node_modules', 'vendor', 'dist', 'build', '.next', '.nuxt',
        '__pycache__', '.venv', 'venv', 'env', '.mypy_cache', '.pytest_cache',
        'target', '.idea', '.vscode', 'site-packages',
    })

    TS_JS_EXTS = ['.ts', '.tsx', '.js', '.jsx', '.mjs', '.cjs']

    # Common directory purposes
    DIRECTORY_PURPOSES = {
        'models': frozenset(['model', 'models', 'db', 'database']),
        'views': frozenset(['view', 'views', 'controller', 'controllers', 'handler', 'handlers']),
        'routes': frozenset(['route', 'routes', 'router', 'routers', 'urls', 'api']),
        'services': frozenset(['service', 'services', 'business', 'logic']),
        'agents': frozenset(['agent', 'agents', 'llm', 'external_llm', 'ai', 'bots']),
        'utils': frozenset(['util', 'utils', 'helper', 'helpers', 'common']),
        'tests': frozenset(['test', 'tests', '__tests__']),
        'static': frozenset(['static', 'assets', 'public']),
        'templates': frozenset(['template', 'templates', 'views']),
        'config': frozenset(['config', 'settings', 'conf']),
    }

    def __init__(self, repo_root: str, max_depth: int = 5):
        self.repo_root = Path(repo_root).resolve()
        self.max_depth = max_depth
        # Lazily-materialized list of repo source files; walked once and reused by
        # _detect_languages + every _sample_files caller during one analyze() pass
        # (see _iter_source_files), so the repo tree is no longer re-walked ~7x.
        self._source_files_cache: Optional[list[Path]] = None
        # Lazily-populated cache for _read_pyproject_deps(); shared across all
        # py_dep marker lookups during one analyze() pass so the manifest is
        # parsed at most once.
        self._pyproject_deps_cache: Optional[set] = None
        # Lazily-populated cache for _read_gradle_text(); shared across all
        # gradle_text marker lookups during one analyze() pass so the Gradle
        # build files + version catalog are read at most once.
        self._gradle_text_cache: Optional[str] = None

    def analyze(self) -> ProjectStructure:
        """
        Analyze project structure

        Populates both `framework` (primary, backward compat) and
        `frameworks` (all detected) fields.  Any callers still reading
        `.framework` will continue to work unchanged.

        Returns:
            ProjectStructure with detected information
        """
        structure = ProjectStructure()

        # Detect source languages first — frameworks are gated on the
        # languages actually present in the repo.
        structure.languages = self._detect_languages()
        structure.primary_language = structure.languages[0] if structure.languages else None

        # Detect all frameworks (multi-framework support), gated by language
        structure.frameworks = self._detect_frameworks(structure.languages)
        structure.framework = structure.frameworks[0] if structure.frameworks else None

        # Analyze directories
        structure.directories = self._analyze_directories()

        # Find entry points (before project_types — entry_points needed as input)
        structure.entry_points = self._find_entry_points(structure.framework)

        # Detect project types (web, cli, library, package)
        structure.project_types = self._detect_project_types(
            structure.frameworks, structure.entry_points, structure.languages
        )

        # Detect file patterns
        structure.file_patterns = self._detect_file_patterns(structure.framework)

        # Detect naming style
        structure.naming_style = self._detect_naming_style()
        if structure.naming_style is None:
            structure.naming_style = "unknown"

        # Find common imports
        structure.common_imports = self._find_common_imports()

        # Find test directory
        structure.test_dir = self._find_test_dir()

        # Find example files
        structure.example_files = self._find_example_files(structure.framework)

        return structure

    def _walk_source_files(self):
        """Generator: yield repo files, skipping vendored/build/VCS directories."""
        try:
            stack = [self.repo_root]
            while stack:
                current = stack.pop()
                try:
                    entries = list(current.iterdir())
                except (OSError, PermissionError):
                    continue
                for entry in entries:
                    if entry.is_dir():
                        if entry.name in self.SKIP_DIRS or entry.name.startswith('.'):
                            continue
                        stack.append(entry)
                    elif entry.is_file():
                        yield entry
        except Exception as e:
            logger.debug(f"Error walking source files: {e}")

    def _iter_source_files(self):
        """Return the cached list of repo source files (walked once per instance).

        The tree walk is the dominant I/O cost of ``analyze()`` and is reached
        from ``_detect_languages`` (full walk) plus several ``_sample_files``
        calls — each formerly re-traversed the repo from the root. Materializing
        the walk once and returning the list lets every caller reuse it, so a
        single ``analyze()`` pass no longer re-walks the tree ~7x. Callers only
        iterate (and may ``break`` early), which is behavior-preserving over the
        old generator.
        """
        if self._source_files_cache is None:
            self._source_files_cache = list(self._walk_source_files())
        return self._source_files_cache

    def _dir_exists_pruned(self, name: str) -> bool:
            """Return True if a directory named *name* exists in the source tree.

            Pruned descent: each directory entry is TESTED against *name* (so a
            marker like ``('dir','node_modules')`` still matches), but os.walk never
            DESCENDS into vendored/build/VCS dirs (SKIP_DIRS / dotdirs) afterwards.
            The former ``rglob(name)`` walked the entire tree — including a huge
            node_modules — once per ``dir`` marker per ``_detect_frameworks`` pass,
            and could match a vendored nested copy (false positive). This caps the
            walk to the source tree and short-circuits on the first match.
            """
            try:
                stack = [self.repo_root]
                while stack:
                    current = stack.pop()
                    try:
                        entries = list(current.iterdir())
                    except (OSError, PermissionError):
                        continue
                    for entry in entries:
                        if not entry.is_dir():
                            continue
                        if entry.name == name:
                            return True
                        # Test the entry, but do not descend into vendor/build/VCS.
                        if entry.name in self.SKIP_DIRS or entry.name.startswith('.'):
                            continue
                        stack.append(entry)
            except Exception:
                pass
            return False
    def _sample_files(self, suffixes, limit: int) -> list[Path]:
        """Sample up to ``limit`` repo files with the given suffix(es),
        skipping vendored/build/VCS directories (.venv, node_modules, …).

        Scanning vendored trees produces both noise and false signals — e.g.
        a ``wsgi.py`` under ``.venv/.../starlette`` is not a project marker.
        """
        if isinstance(suffixes, str):
            suffixes = (suffixes,)
        out: list[Path] = []
        for path in self._iter_source_files():
            if path.suffix in suffixes:
                out.append(path)
                if len(out) >= limit:
                    break
        return out

    def _detect_languages(self) -> list[str]:
        """Detect source languages, ranked by file count (most common first).

        Uses the shared LanguageId extension map so language identification
        stays consistent with the rest of the codebase. A language must hold a
        minimum share of recognized files to count as present, so a couple of
        stray scripts can't masquerade as a real language.
        """
        counts: dict[str, int] = defaultdict(int)
        for path in self._iter_source_files():
            lang = LanguageId.from_path(str(path))
            if lang is LanguageId.UNKNOWN:
                continue
            # Config/markup formats aren't "the project's language".
            if lang in (LanguageId.JSON, LanguageId.CSS, LanguageId.HTML):
                continue
            # Gradle build scripts are Kotlin/Groovy *syntax* but build
            # configuration, not application source — a lone build.gradle.kts
            # in a Python repo must not make "kotlin" a present language (which
            # would unlock android/compose detection on a non-JVM project).
            if path.name.endswith('.gradle.kts') or path.name.endswith('.gradle'):
                continue
            counts[lang.value] += 1

        total = sum(counts.values())
        if not total:
            return []

        present = [
            lang for lang, n in counts.items()
            if n >= self.MIN_LANGUAGE_FILES and (n / total) >= self.MIN_LANGUAGE_SHARE
        ]
        # Fall back to the single dominant language if the share gate rejected
        # everything (e.g. a tiny repo with one source file).
        if not present and counts:
            present = [max(counts, key=lambda k: counts[k])]

        return sorted(present, key=lambda lang: counts[lang], reverse=True)

    def _detect_framework(self) -> Optional[str]:
        """Detect primary project framework (backward compat wrapper)"""
        frameworks = self._detect_frameworks(self._detect_languages())
        return frameworks[0] if frameworks else None

    def _detect_frameworks(self, languages: Optional[list[str]] = None) -> list[str]:
        """Detect all project frameworks (multi-framework support)

        Scoring is weighted (definitive markers count more than generic ones)
        and gated by language: a framework is only considered when one of its
        languages is present in the repo. A framework must reach
        MIN_FRAMEWORK_SCORE, then any framework scoring >= 50% of the top score
        is included (catches polyglot/web+CLI projects).
        """
        if languages is None:
            languages = self._detect_languages()
        lang_set = set(languages)

        scores = defaultdict(int)

        for framework, markers in self.FRAMEWORK_MARKERS.items():
            # Language gate: skip frameworks whose language isn't present.
            allowed = self.FRAMEWORK_LANGUAGES.get(framework)
            if allowed and not (allowed & lang_set):
                continue

            for marker_type, marker_value, weight in markers:
                if marker_type == 'file':
                    if (self.repo_root / marker_value).exists():
                        scores[framework] += weight

                elif marker_type == 'dir':
                    if self._dir_exists_pruned(marker_value):
                        scores[framework] += weight

                elif marker_type == 'import':
                    # Match real import *statements* (line starts with the
                    # marker), not the string appearing anywhere — otherwise a
                    # marker literal like 'from django.' in this very file, or
                    # a quoted mention in another, would self-trigger.
                    for py_file in self._sample_files('.py', 50):
                        try:
                            content = py_file.read_text()
                            if any(line.lstrip().startswith(marker_value)
                                   for line in content.splitlines()):
                                scores[framework] += weight
                                break
                        except Exception:
                            continue

                elif marker_type == 'pkg_dep':
                    pkg_path = self.repo_root / 'package.json'
                    if pkg_path.exists():
                        try:
                            import json
                            pkg = json.loads(pkg_path.read_text())
                            all_deps = {**pkg.get('dependencies', {}), **pkg.get('devDependencies', {})}
                            if marker_value in all_deps:
                                scores[framework] += weight
                        except Exception:
                            pass

                elif marker_type == 'go_dep':
                    if self._go_mod_requires(marker_value):
                        scores[framework] += weight

                elif marker_type == 'py_dep':
                    # Manifest-declared dependency — the strongest, alias- and
                    # sampling-immune signal (see _read_pyproject_deps).
                    deps = self._pyproject_deps_cache
                    if deps is None:
                        deps = self._read_pyproject_deps()
                        self._pyproject_deps_cache = deps
                    if marker_value in deps:
                        scores[framework] += weight

                elif marker_type == 'file_ext':
                    # Reuse the cached, pruned source-file list rather than a
                    # fresh rglob (which walks node_modules and can false-positive
                    # on a vendored .vue/.astro). Suffix membership is O(n) but
                    # cached, and only source files are considered.
                    if any(p.suffix == marker_value for p in self._iter_source_files()):
                        scores[framework] += weight

                elif marker_type == 'gradle_text':
                    # Substring across Gradle build scripts + version catalog.
                    # Catches projects that declare plugins/deps via catalog
                    # aliases (the id lives in libs.versions.toml, not the
                    # .gradle.kts). Cached per instance via _read_gradle_text.
                    if marker_value in self._read_gradle_text():
                        scores[framework] += weight

                elif marker_type == 'jvm_import':
                    # Import-statement prefix match in Kotlin/Java sources —
                    # parallel to the 'import' marker (Python) but on .kt/.java.
                    for src_file in self._sample_files(('.kt', '.java'), 50):
                        try:
                            content = src_file.read_text()
                            if any(line.lstrip().startswith(marker_value)
                                   for line in content.splitlines()):
                                scores[framework] += weight
                                break
                        except Exception:
                            continue

        # Drop frameworks that never cleared the minimum-evidence bar.
        scores = {fw: sc for fw, sc in scores.items() if sc >= self.MIN_FRAMEWORK_SCORE}
        if not scores:
            return []

        # Multi-framework: include any framework scoring >= 50% of top score
        max_score = max(scores.values())
        threshold = max(self.MIN_FRAMEWORK_SCORE, max_score * 0.5)
        sorted_frameworks = sorted(
            [fw for fw, sc in scores.items() if sc >= threshold],
            key=lambda fw: scores[fw],
            reverse=True,
        )

        return sorted_frameworks

    def _go_mod_requires(self, module_prefix: str) -> bool:
        """Return True if go.mod has a *direct* require matching module_prefix.

        Lines marked ``// indirect`` are transitive dependencies and are
        ignored, so a transitively pulled-in web framework can't be reported
        as one the project uses directly.
        """
        go_mod = self.repo_root / 'go.mod'
        if not go_mod.exists():
            return False
        try:
            for line in go_mod.read_text().splitlines():
                stripped = line.strip()
                if not stripped or stripped.startswith('//'):
                    continue
                if '// indirect' in line:
                    continue
                # Match the module path token (first field of a require line).
                token = stripped.split()[0] if stripped.split() else ''
                if token == module_prefix or token.startswith(module_prefix):
                    return True
        except Exception:
            return False
        return False

    def _read_gradle_text(self) -> str:
        """Concatenated text of Gradle build files + version catalog.

        Covers Kotlin DSL (``*.gradle.kts``), Groovy DSL (``*.gradle``), and
        the version catalog (``gradle/*.toml``) so framework detection works
        for projects that declare plugins/dependencies via catalog aliases
        (``alias(libs.plugins.android.application)``) rather than inline ids
        — the real plugin id then lives in ``libs.versions.toml``. Build
        output dirs (``build/``, ``.gradle/``) are skipped to avoid generated
        files. Cached per instance.
        """
        if self._gradle_text_cache is not None:
            return self._gradle_text_cache
        chunks: list[str] = []
        try:
            # Reuse the cached, pruned source-file walk (_walk_source_files never
            # descends into SKIP_DIRS or dotdirs, so build/ and .gradle/ output
            # are already excluded — the former per-path post-filter is now
            # redundant). The old rglob walked the whole tree twice (once per
            # extension), descending into node_modules/build each time.
            for path in self._iter_source_files():
                name = path.name
                if not (name.endswith(".gradle.kts") or name.endswith(".gradle")):
                    continue
                try:
                    chunks.append(path.read_text())
                except Exception:
                    continue
            for path in self.repo_root.glob("gradle/*.toml"):
                try:
                    chunks.append(path.read_text())
                except Exception:
                    continue
        except Exception as e:
            logger.debug(f"Error reading gradle files: {e}")
        self._gradle_text_cache = "\n".join(chunks)
        return self._gradle_text_cache

    def _read_pyproject_deps(self) -> set:
        """Return the set of *direct* Python dependency names declared in the
        project manifest (pyproject.toml or setup.py).

        This is the Python analogue of ``_go_mod_requires`` / ``pkg_dep``: a
        dependency declared in the manifest is the strongest, alias- and
        sampling-immune signal that a framework/library is actually used. Import
        scanning alone is defeated by aliases (``import tree_sitter as _ts``),
        self-wrapping modules (``from external_llm.languages.tree_sitter_utils``)
        and the fixed-size sample of source files.

        Handles both PEP 621 (``[project]``/``[project.optional-dependencies]``)
        and Poetry (``[tool.poetry.dependencies]``) layouts, plus a flat
        ``setup.py`` ``install_requires``. Returns lower-cased, version- and
        extras-stripped distribution names.
        """
        names: set = set()

        pyproject = self.repo_root / 'pyproject.toml'
        if pyproject.exists():
            try:
                data = self._parse_toml(pyproject.read_text())
            except Exception:
                data = None
            if isinstance(data, dict):
                # PEP 621 main deps
                project = data.get('project') or {}
                for dep in (project.get('dependencies') or []):
                    names.add(self._normalize_dep_name(dep))
                # PEP 621 optional-dependencies (extra groups)
                for group in (project.get('optional-dependencies') or {}).values():
                    for dep in group:
                        names.add(self._normalize_dep_name(dep))
                # Poetry deps
                poetry = (data.get('tool') or {}).get('poetry') or {}
                for dep in (poetry.get('dependencies') or {}).keys():
                    # Skip Python version constraint keys like 'python = "^3.9"'.
                    if dep.lower() == 'python':
                        continue
                    names.add(self._normalize_dep_name(dep))

        setup_py = self.repo_root / 'setup.py'
        if setup_py.exists():
            try:
                import ast as _ast
                content = setup_py.read_text()
                tree = _ast.parse(content)
                for node in _ast.walk(tree):
                    if isinstance(node, _ast.keyword) and node.arg == 'install_requires':
                        if isinstance(node.value, (_ast.List, _ast.Tuple)):
                            for elt in node.value.elts:
                                if isinstance(elt, _ast.Constant) and isinstance(elt.value, str):
                                    names.add(self._normalize_dep_name(elt.value))
            except Exception:
                pass

        return names

    @staticmethod
    def _normalize_dep_name(dep: str) -> str:
        """Normalize a PEP 508 dependency spec to a bare distribution name.

        ``'tree-sitter>=0.23'`` → ``'tree-sitter'``
        ``'fastapi[all]>=0.110,<1'`` → ``'fastapi'``
        ``'PyYAML'`` → ``'pyyaml'``  (PEP 503 normalization: case-fold + dash)
        """
        name = dep.strip()
        for sep in ('[', ' ', ';', '<', '>', '=', '~', '!'):
            name = name.split(sep, 1)[0]
        return name.strip().lower().replace('_', '-')

    def _parse_toml(self, text: str):
        """Parse TOML using the stdlib (3.11+) tomllib, else a minimal fallback.

        The fallback only understands the flat/table list shapes used for
        dependency extraction, so this module keeps no third-party dependency.
        """
        try:
            import tomllib
            return tomllib.loads(text)
        except ImportError:
            return self._parse_toml_fallback(text)

    @staticmethod
    def _parse_toml_fallback(text: str) -> dict:
        """Minimal TOML reader for [project]/[tool.poetry] dependency lists.

        Only supports the subset emitted by modern packaging tools for
        dependency arrays and poetry table values. Sufficient for framework
        detection when tomllib isn't available (Python < 3.11).
        """
        result: dict = {}

        def descend(path_parts):
            node = result
            for part in path_parts:
                existing = node.get(part)
                if not isinstance(existing, dict):
                    existing = {}
                    node[part] = existing
                node = existing
            return node

        current_table = result
        for raw in text.splitlines():
            line = raw.strip()
            if not line or line.startswith('#'):
                continue
            if line.startswith('[') and line.endswith(']'):
                header = line[1:-1].strip()
                current_table = descend([p.strip() for p in header.split('.')])
                continue
            if '=' not in line:
                continue
            key, _, value = line.partition('=')
            key = key.strip().strip('"').strip("'")
            value = value.strip()
            if value.startswith('['):
                inner = value.strip('[]').strip()
                items = []
                buf = ''
                in_str = False
                quote = ''
                for ch in inner:
                    if in_str:
                        if ch == quote:
                            in_str = False
                            items.append(buf)
                            buf = ''
                        else:
                            buf += ch
                    elif ch in ('"', "'"):
                        in_str = True
                        quote = ch
                    elif ch == ',':
                        if buf:
                            items.append(buf)
                            buf = ''
                    elif ch not in (' ', '\t'):
                        buf += ch
                if buf:
                    items.append(buf)
                current_table[key] = [s.strip() for s in items if s.strip()]
            elif value.startswith('"') or value.startswith("'"):
                current_table[key] = value.strip('"').strip("'")
            else:
                current_table[key] = value
        return result

    def _detect_project_types(
        self,
        frameworks: list[str],
        entry_points: list[str],
        languages: Optional[list[str]] = None,
    ) -> list[str]:
        """Detect project type(s): 'web', 'cli', 'library', 'package'"""
        types: list[str] = []
        languages = languages or []
        web_frameworks = {
            'fastapi', 'flask', 'django', 'nextjs', 'nuxt', 'astro',
            'gin', 'echo', 'fiber', 'chi', 'gorilla-mux',
        }
        cli_frameworks = {
            'click', 'typer',
            'cobra', 'fang', 'urfave-cli', 'kong',
        }
        # stdlib CLI modules (argparse) are not frameworks, but still evidence
        # of a CLI project type — scanned directly from source, not from the
        # framework list.
        stdlib_cli = self.STDLIB_CLI

        # Type 1: Web — any web framework detected
        if any(fw in web_frameworks for fw in frameworks):
            types.append('web')

        # Type 2: CLI — CLI framework detected, OR a language-specific entry
        # with CLI patterns.
        if any(fw in cli_frameworks for fw in frameworks):
            types.append('cli')
        elif 'cli' not in types:
            # Python fallback: a __main__ guard wired to argv/argparse, or a
            # direct stdlib CLI module import (argparse is no longer in the
            # framework list but still signals a CLI).
            for py_file in self._sample_files('.py', 30):
                try:
                    content = py_file.read_text()
                    stdlib_hit = any(
                        f'import {mod}' in content or f'from {mod}' in content
                        for mod in stdlib_cli
                    )
                    if ('if __name__ ==' in content and (
                        'sys.argv' in content or 'argparse' in content
                    )) or stdlib_hit:
                        types.append('cli')
                        break
                except Exception:
                    continue

        # Go fallback: a `package main` executable with no web framework is a
        # CLI/binary, not a web service or library.
        if 'go' in languages and 'web' not in types and 'cli' not in types:
            if self._go_has_main_package():
                types.append('cli')

        # Android / mobile: an Android app or library module. ('mobile' is the
        # platform-agnostic type — extensible to iOS/Flutter/RN later.)
        if 'android' in frameworks and 'mobile' not in types:
            types.append('mobile')

        # Type 3: Library — has pkg metadata but no web/CLI entry points
        if not types:
            has_pkg_meta = any(
                (self.repo_root / f).exists()
                for f in ['setup.py', 'pyproject.toml', 'setup.cfg']
            )
            # A Go module with no main package is a library too.
            if has_pkg_meta or ('go' in languages and not self._go_has_main_package()):
                types.append('library')

        # Default: at least 'package'
        if not types:
            types.append('package')

        return types

    def _go_has_main_package(self) -> bool:
        """Return True if the repo declares a Go `package main` (an executable)."""
        if (self.repo_root / 'main.go').exists():
            return True
        for go_file in self._sample_files('.go', 50):
            try:
                for line in go_file.read_text().splitlines():
                    stripped = line.strip()
                    if stripped.startswith('package '):
                        if stripped == 'package main':
                            return True
                        break  # only the package clause matters
            except Exception:
                continue
        return False

    def _analyze_directories(self) -> dict[str, list[str]]:
        """Analyze directory structure and categorize"""

        categorized = defaultdict(list)

        try:
            for item in self.repo_root.iterdir():
                if not item.is_dir():
                    continue

                if item.name.startswith('.'):
                    continue

                # Match against known patterns
                for purpose, variants in self.DIRECTORY_PURPOSES.items():
                    if item.name.lower() in variants:
                        categorized[purpose].append(item.name)
                        break
                else:
                    # Unknown purpose
                    categorized['other'].append(item.name)

        except Exception as e:
            logger.debug(f"Error analyzing directories: {e}")

        return dict(categorized)

    def _detect_file_patterns(
        self, framework: Optional[str], language: Optional[str] = None
    ) -> dict[str, str]:
        """Detect file naming patterns"""

        patterns = {}

        if framework == 'django':
            patterns['model'] = '{app}/models.py or {app}/models/{model}.py'
            patterns['view'] = '{app}/views.py or {app}/views/{view}.py'
            patterns['url'] = '{app}/urls.py'
            patterns['form'] = '{app}/forms.py'
            patterns['admin'] = '{app}/admin.py'

        elif framework == 'fastapi':
            patterns['router'] = 'routers/{feature}.py or api/{feature}.py'
            patterns['model'] = 'models/{model}.py'
            patterns['schema'] = 'schemas/{schema}.py'
            patterns['service'] = 'services/{service}.py'

        elif framework == 'flask':
            patterns['route'] = 'routes/{feature}.py or {feature}/routes.py'
            patterns['model'] = 'models/{model}.py or models.py'
            patterns['form'] = 'forms/{form}.py or forms.py'

        elif framework == 'react':
            patterns['component'] = 'components/{Component}.tsx or components/{Component}.jsx'
            patterns['page'] = 'pages/{route}.tsx or app/{route}/page.tsx'
            patterns['hook'] = 'hooks/{useHook}.ts or hooks/{useHook}.js'
            patterns['service'] = 'services/{service}.ts or lib/{service}.ts'

        elif framework == 'nextjs':
            patterns['page'] = 'app/{route}/page.tsx or pages/{route}.tsx'
            patterns['component'] = 'components/{Component}.tsx'
            patterns['api'] = 'app/api/{route}/route.ts or pages/api/{route}.ts'
            patterns['layout'] = 'app/layout.tsx'

        elif framework == 'vue':
            patterns['component'] = 'components/{Component}.vue'
            patterns['page'] = 'pages/{route}.vue or views/{View}.vue'

        elif framework == 'nuxt':
            patterns['page'] = 'pages/{route}.vue'
            patterns['component'] = 'components/{Component}.vue'
            patterns['layout'] = 'layouts/{layout}.vue'

        elif framework == 'astro':
            patterns['page'] = 'pages/{route}.astro'
            patterns['component'] = 'components/{Component}.astro or components/{Component}.tsx'

        elif language == 'go':
            patterns['package'] = 'internal/{pkg}/{file}.go or pkg/{pkg}/{file}.go'
            patterns['command'] = 'cmd/{command}/main.go or internal/cmd/{command}.go'
            patterns['test'] = '{file}_test.go'

        elif language in ('javascript', 'typescript'):
            patterns['module'] = '{feature}.ts or {feature}.js'

        elif language in (None, 'python'):
            # Generic Python (also the no-language fallback)
            patterns['module'] = '{feature}.py'
            patterns['class'] = '{feature}_class.py or {feature}.py'

        # Other languages: no reliable generic template — leave empty rather
        # than emit misleading guidance.

        return patterns

    def _detect_naming_style(self) -> Optional[str]:
        """Detect naming convention from existing files.

        Returns None when there is no recognized evidence, so callers can
        report "unknown" rather than a misleading default. Covers Python,
        JS/TS, Kotlin/Java, Go, Rust, Swift and C#. Previously this scanned
        only Python/JS and returned "snake_case" even for languages it never
        scanned — e.g. a Kotlin repo — claiming a convention it had zero data
        for. It also returns None for a repo whose files are all single-word
        lowercase (genuinely ambiguous) instead of defaulting to snake_case.
        """

        snake_count = 0
        camel_count = 0
        pascal_count = 0
        seen = 0

        try:
            for ext in ('.py', '.ts', '.tsx', '.js', '.jsx',
                        '.kt', '.java', '.go', '.rs', '.swift', '.cs'):
                for file in self._sample_files(ext, 30):
                    name = file.stem
                    if name in ('index', 'next', 'vite', 'nuxt', 'astro', 'tailwind',
                                'postcss', 'package', 'tsconfig', 'eslint', 'prettier'):
                        continue

                    seen += 1
                    if '_' in name:
                        snake_count += 1
                    elif name[0].isupper() and any(c.isupper() for c in name[1:]):
                        pascal_count += 1
                    elif name[0].islower() and any(c.isupper() for c in name[1:]):
                        camel_count += 1

        except Exception as e:
            logger.debug(f"Error detecting naming style: {e}")

        # No files of a recognized language → no basis to claim a convention.
        if seen == 0:
            return None

        # A convention requires at least one file exhibiting its distinctive
        # pattern. A repo whose files are all single-word lowercase (main.go,
        # app.py) is genuinely ambiguous → report unknown rather than silently
        # defaulting to snake_case.
        if snake_count == 0 and pascal_count == 0 and camel_count == 0:
            return None

        # Determine majority
        if snake_count > camel_count and snake_count > pascal_count:
            return "snake_case"
        elif pascal_count > snake_count:
            return "PascalCase"
        elif camel_count > 0:
            return "camelCase"

        return "snake_case"  # Default for Python (snake ties/leads)

    def _find_common_imports(self) -> list[str]:
        """Find most common imports in the project (Python + JS/TS)"""

        import_counts = defaultdict(int)

        try:
            # Python imports
            for py_file in self._sample_files('.py', 50):
                try:
                    content = py_file.read_text()
                    # import can appear anywhere in a file (lazy import, long module docstring/
                    # license header). Same full-line scan as _detect_frameworks.
                    for line in content.split('\n'):
                        line = line.strip()
                        if line.startswith('import '):
                            module = line.split()[1].split('.')[0]
                            import_counts[module] += 1
                        elif line.startswith('from '):
                            parts = line.split()
                            # Skip relative imports ('from . import', 'from .x import'):
                            # parts[1] like '.x' splits to '' and would pollute counts.
                            if len(parts) >= 2 and not parts[1].startswith('.'):
                                module = parts[1].split('.')[0]
                                if module:
                                    import_counts[module] += 1
                except Exception:
                    continue

            # JS/TS imports (ESM + CJS)
            for ts_ext in ['.ts', '.tsx', '.js', '.jsx']:
                for ts_file in self._sample_files(ts_ext, 50):
                    try:
                        content = ts_file.read_text()
                        # ESM/CJS imports can appear anywhere too — full line scan.
                        for line in content.split('\n'):
                            line = line.strip()
# ESM: `from 'module'` or `from "module"` — string ops instead of regex
                            for _q in ("'", '"'):
                                _marker = f'from {_q}'
                                _idx = line.find(_marker)
                                if _idx != -1:
                                    _start = _idx + len(_marker)
                                    _end = line.find(_q, _start)
                                    if _end != -1:
                                        _mod = line[_start:_end]
                                        if not _mod.startswith('.'):
                                            import_counts[_mod.split('/')[0]] += 1
                            # CJS: `require('module')` or `require("module")`
                            for _q in ("'", '"'):
                                _marker = f'require({_q}'
                                _idx = line.find(_marker)
                                if _idx != -1:
                                    _start = _idx + len(_marker)
                                    _end = line.find(_q, _start)
                                    if _end != -1:
                                        _mod = line[_start:_end]
                                        if not _mod.startswith('.'):
                                            import_counts[_mod.split('/')[0]] += 1
                    except Exception:
                        continue

        except Exception as e:
            logger.debug(f"Error finding imports: {e}")

        # Return top 10
        sorted_imports = sorted(import_counts.items(), key=lambda x: x[1], reverse=True)
        return [imp for imp, _ in sorted_imports[:10]]

    def _find_entry_points(self, framework: Optional[str]) -> list[str]:
        """Find main entry point files

        Scans in priority order:
        1. pyproject.toml → [project.scripts] or [tool.poetry.scripts]
        2. setup.py → entry_points / console_scripts
        3. setup.cfg → [options.entry_points]
        4. package.json → "bin" or "main"
        5. Fallback: well-known filenames (main.py, app.py, asi.py, index.ts, etc.)
        """

        entry_points = []

        # --- Priority 1: pyproject.toml ---
        pyproject_path = self.repo_root / 'pyproject.toml'
        if pyproject_path.exists():
            try:
                import tomllib  # Python 3.11+
                with open(pyproject_path, 'rb') as f:
                    data = tomllib.load(f)
                scripts = (data.get('project', {}).get('scripts', {})
                           or data.get('tool', {}).get('poetry', {}).get('scripts', {}))
                for _name, _target in scripts.items():
                    entry_points.append(f"{_name} ({_target})")
            except Exception:
                try:
                    import tomli
                    with open(pyproject_path, 'rb') as f:
                        data = tomli.load(f)
                    scripts = (data.get('project', {}).get('scripts', {})
                               or data.get('tool', {}).get('poetry', {}).get('scripts', {}))
                    for _name, _target in scripts.items():
                        entry_points.append(f"{_name} ({_target})")
                except Exception:
                    pass

        # --- Priority 2: setup.py (parse console_scripts via regex) ---
        setup_py_path = self.repo_root / 'setup.py'
        if setup_py_path.exists() and not entry_points:
            try:
                import ast as _ast
                with open(setup_py_path) as f:
                    tree = _ast.parse(f.read())
                for node in _ast.walk(tree):
                    if isinstance(node, _ast.Call) and hasattr(node.func, 'id') and node.func.id == 'setup':
                        for kw in node.keywords:
                            if kw.arg == 'entry_points':
                                try:
                                    ep_dict = _ast.literal_eval(kw.value)
                                    console_scripts = ep_dict.get('console_scripts', [])
                                    entry_points.extend(console_scripts)
                                except Exception:
                                    pass
            except Exception:
                pass

        # --- Priority 3: setup.cfg ---
        setup_cfg_path = self.repo_root / 'setup.cfg'
        if setup_cfg_path.exists() and not entry_points:
            try:
                import configparser
                cfg = configparser.ConfigParser()
                cfg.read(setup_cfg_path)
                if cfg.has_section('options.entry_points'):
                    for k, v in cfg['options.entry_points'].items():
                        if k == 'console_scripts':
                            entry_points.extend(line.strip() for line in v.split('\n') if line.strip())
            except Exception:
                pass

        # --- Priority 4: package.json ---
        pkg_json = self.repo_root / 'package.json'
        if pkg_json.exists():
            try:
                import json
                with open(pkg_json) as f:
                    pkg = json.load(f)
                bin_ = pkg.get('bin')
                if isinstance(bin_, str):
                    entry_points.append(bin_)
                elif isinstance(bin_, dict):
                    entry_points.extend(bin_.values())
                main_ = pkg.get('main')
                if main_ and main_ not in entry_points:
                    entry_points.append(main_)
            except Exception:
                pass

        # --- Priority 5: Well-known filename fallback (always checked) ---
        # Strong candidates: include immediately if file exists
        strong_candidates = [
            'main.py',
            'app.py',
            'manage.py',
            'wsgi.py',
            'asgi.py',
            'asi.py',
            '__main__.py',
            # JS/TS entry points
            'index.ts',
            'index.tsx',
            'index.js',
            'index.jsx',
            # Go entry point
            'main.go',
        ]

        for candidate in strong_candidates:
            if (self.repo_root / candidate).exists():
                if candidate not in entry_points:
                    entry_points.append(candidate)

        # Weak candidates: include only if file contains a real entry pattern
        weak_candidates = ['cli.py']
        for candidate in weak_candidates:
            path = self.repo_root / candidate
            if path.exists() and candidate not in entry_points:
                if self._has_entry_pattern(path):
                    entry_points.append(candidate)

        return entry_points

    @staticmethod
    def _has_entry_pattern(path: Path) -> bool:
        """Check if a Python file contains a real entry point pattern"""
        try:
            content = path.read_text()
            return ("if __name__ == '__main__'" in content
                    or 'if __name__ == "__main__"' in content
                    or 'def main():' in content)
        except Exception:
            return False

    def _find_test_dir(self) -> Optional[str]:
        """Find test directory"""

        test_dirs = ['tests', 'test', '__tests__', 'spec']

        for test_dir in test_dirs:
            if (self.repo_root / test_dir).is_dir():
                return test_dir

        return None

    def _first_glob(self, pattern: str) -> Optional[Path]:
            """Return the first glob match without materializing the full match list.

            Equivalent to ``list(self.repo_root.glob(pattern))[0]`` when a match
            exists, but short-circuits after the first hit (glob yields in the same
            order either way) and returns ``None`` when there is no match.
            """
            return next(self.repo_root.glob(pattern), None)
    def _find_example_files(self, framework: Optional[str]) -> dict[str, str]:
        """Find example files of each type for reference"""

        examples = {}

        try:
            if framework == 'django':
                for pattern in ['*/views.py', '*/models.py', '*/urls.py']:
                    match = self._first_glob(pattern)
                    if match is not None:
                        file_type = match.name.replace('.py', '')
                        examples[file_type] = str(match.relative_to(self.repo_root))

            elif framework == 'fastapi':
                for pattern in ['routers/*.py', 'models/*.py', 'schemas/*.py']:
                    match = self._first_glob(pattern)
                    if match is not None:
                        examples[match.parent.name] = str(match.relative_to(self.repo_root))

            elif framework == 'flask':
                for pattern in ['routes/*.py', 'models/*.py']:
                    match = self._first_glob(pattern)
                    if match is not None:
                        examples[match.parent.name] = str(match.relative_to(self.repo_root))

            elif framework in ('react', 'nextjs'):
                for pattern in ['components/*.tsx', 'components/*.jsx', 'pages/*.tsx', 'hooks/*.ts']:
                    match = self._first_glob(pattern)
                    if match is not None:
                        examples[match.parent.name] = str(match.relative_to(self.repo_root))

            elif framework == 'vue':
                for pattern in ['components/*.vue', 'pages/*.vue']:
                    match = self._first_glob(pattern)
                    if match is not None:
                        examples[match.parent.name] = str(match.relative_to(self.repo_root))

            elif framework == 'nuxt':
                for pattern in ['components/*.vue', 'pages/*.vue', 'layouts/*.vue']:
                    match = self._first_glob(pattern)
                    if match is not None:
                        examples[match.parent.name] = str(match.relative_to(self.repo_root))

        except Exception as e:
            logger.debug(f"Error finding example files: {e}")

        return examples

    def get_structure_summary(self, structure: ProjectStructure) -> str:
        """Get human-readable summary of project structure"""

        lines = ["# Project Overview"]
        lines.append("")

        if structure.languages:
            lines.append(f"**Languages**: {', '.join(structure.languages)}")

        if structure.frameworks:
            lines.append(f"**Frameworks**: {', '.join(structure.frameworks)}")
        elif structure.framework:
            lines.append(f"**Framework**: {structure.framework}")

        if structure.project_types:
            lines.append(f"**Project Type**: {', '.join(structure.project_types)}")

        if structure.entry_points:
            lines.append(f"**Entry Points**: {', '.join(structure.entry_points[:5])}")

        if structure.directories:
            lines.append("**Directory Organization**:")
            for purpose, dirs in structure.directories.items():
                if purpose != 'other':
                    lines.append(f"- {purpose.title()}: {', '.join(dirs)}")

        if structure.file_patterns:
            lines.append("")
            lines.append("**File Patterns**:")
            for pattern_type, template in structure.file_patterns.items():
                lines.append(f"- {pattern_type}: `{template}`")

        if structure.naming_style:
            lines.append("")
            lines.append(f"**Naming Convention**: {structure.naming_style}")

        if structure.common_imports:
            lines.append("")
            lines.append(f"**Common Imports**: {', '.join(structure.common_imports[:5])}")

        if structure.example_files:
            lines.append("")
            lines.append("**Example Files** (for reference):")
            for file_type, path in structure.example_files.items():
                lines.append(f"- {file_type}: `{path}`")

        return "\n".join(lines)

