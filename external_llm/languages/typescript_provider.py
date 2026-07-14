"""
TypeScript / TSX syntax provider.

Uses ``tsc --noEmit`` for validation and regex-based symbol detection.
Gracefully degrades when ``tsc`` is not installed.
"""
from __future__ import annotations

import json
import logging
import os
import re
import subprocess

from .base import SyntaxProvider, _replace_last_cmd_path, _tempfile_for_content, detect_project_root
from .models import (
    LanguageCapabilities,
    LanguageId,
    SymbolPattern,
    SyntaxError_,
    SyntaxValidationResult,
)

logger = logging.getLogger(__name__)

def _make_capabilities() -> LanguageCapabilities:
    from .tree_sitter_utils import is_available

    return LanguageCapabilities(
        has_ast_parser=False,  # no stdlib AST — uses regex
        has_syntax_validator=True,
        has_semantic_validator=True,
        has_linter=True,
        has_test_runner=True,
        has_symbol_search=True,
        has_tree_sitter=is_available(),
        supports_modify_symbol=True,
        supports_insert_after_symbol=True,
    )

# tsc error line: file.ts(10,5): error TS1005: ';' expected.
_TSC_ERROR_RE = re.compile(
    r"^(.+?)\((\d+),(\d+)\):\s+error\s+(TS\d+):\s+(.+)$"
)

# tsc diagnostic codes that live in the 1xxx (syntax) band but are actually
# module/interop CONFIG diagnostics — they fire on valid source whenever tsc
# runs without the project's compiler options (e.g. esModuleInterop). They must
# NOT be treated as genuine syntax errors by a config-blind check.
_TSC_CONFIG_DEPENDENT_1XXX = frozenset({
    "TS1192",  # module has no default export
    "TS1208",  # cannot be compiled under --isolatedModules (global script)
    "TS1259",  # can only be default-imported using esModuleInterop
    "TS1286",  # esModuleInterop required for '* as' default
    "TS1287",  # esModuleInterop / module setting
    "TS1288",  # esModuleInterop / module setting
    "TS1371",  # import never used as a value (verbatimModuleSyntax)
    "TS1479",  # CommonJS import needs esModuleInterop / dynamic import
})


def is_genuine_syntax_error(code: str) -> bool:
    """True only for config- and environment-independent PARSER syntax errors.

    tsc diagnostic bands: 1xxx = syntax, 2xxx = type/semantic, 5xxx = config,
    7xxx = implicit-any. Only true 1xxx *parser* errors are reproducible
    regardless of installed @types, lib config, module resolution or compiler
    flags — everything else depends on the environment and must not block an
    edit when tsc runs config-blind (single file / temp file / --ignoreConfig).
    The few 1xxx module/interop codes (see ``_TSC_CONFIG_DEPENDENT_1XXX``) are
    excluded too.
    """
    if not code or not code.startswith("TS"):
        return False
    try:
        num = int(code[2:])
    except ValueError:
        return False
    return 1000 <= num <= 1999 and code not in _TSC_CONFIG_DEPENDENT_1XXX


class TypeScriptSyntaxProvider(SyntaxProvider):
    """TypeScript language support (regex-based symbols, tsc validation)."""

    _caps: LanguageCapabilities | None = None

    def language_id(self) -> LanguageId:
        return LanguageId.TYPESCRIPT

    def capabilities(self) -> LanguageCapabilities:
        if self._caps is None:
            self._caps = _make_capabilities()
        return self._caps

    # ── Syntax validation ─────────────────────────────────────────────────

    def _validate_syntax_impl(self, file_path: str, content: str) -> SyntaxValidationResult:
        """Validate TypeScript source via ``tsc --noEmit`` on *content* (written to temp file).

        Falls back to ``ok=True`` when tsc is not available.
        """
        _suffix = os.path.splitext(file_path)[1] or ".ts"
        _tmp_path, _cleanup = _tempfile_for_content(content, _suffix)
        if not _tmp_path:
            return SyntaxValidationResult(ok=True, language=LanguageId.TYPESCRIPT)
        _cmd = _replace_last_cmd_path(
            ["npx", "tsc", "--noEmit", "--allowJs", "--pretty", "false",
             "--skipLibCheck", file_path],
            file_path, _tmp_path,
        )
        try:
            try:
                # NOTE: this is a SYNTAX validator. We deliberately do not pass
                # --isolatedModules: it emits module-constraint diagnostics
                # (e.g. TS1208 on a no-export script) that are not syntax errors.
                proc = subprocess.run(
                    _cmd,
                    capture_output=True, text=True, timeout=30,
                    cwd=os.path.dirname(_tmp_path) or ".",
                )
            except FileNotFoundError:
                logger.debug("tsc not installed; skipping validation")
                return SyntaxValidationResult(ok=True, language=LanguageId.TYPESCRIPT)
            except subprocess.TimeoutExpired:
                logger.debug("tsc timed out for %s", file_path)
                return SyntaxValidationResult(ok=True, language=LanguageId.TYPESCRIPT)
            except Exception as e:
                logger.debug("tsc error: %s", e)
                return SyntaxValidationResult(ok=True, language=LanguageId.TYPESCRIPT)

            if proc.returncode == 0:
                return SyntaxValidationResult(ok=True, language=LanguageId.TYPESCRIPT)

            errors: list[SyntaxError_] = []
            for line in (proc.stdout + proc.stderr).splitlines():
                m = _TSC_ERROR_RE.match(line)
                if not m:
                    continue
                # Only report genuine parser syntax errors. Type/semantic/config
                # diagnostics depend on installed @types, lib config and module
                # resolution — a single-file check cannot judge them, and reporting
                # them would wrongly fail valid code in a repo without node_modules
                # (e.g. `@types/node` missing → TS2591 "Cannot find name 'process'",
                # or esModuleInterop not applied → TS1259). See is_genuine_syntax_error.
                _code = m.group(4)  # e.g. "TS1005"
                if not is_genuine_syntax_error(_code):
                    continue
                errors.append(SyntaxError_(
                    # Report the real source path, not the internal temp file
                    # that tsc actually compiled (a long ../../var/folders/... path).
                    file=file_path,
                    line=int(m.group(2)),
                    col=int(m.group(3)),
                    message=f"{_code}: {m.group(5)}",
                ))
            if not errors:
                # No genuine syntax errors. tsc may have exited non-zero on
                # type/semantic/environmental diagnostics, which we intentionally
                # ignore here (this validates syntax, not types).
                return SyntaxValidationResult(ok=True, language=LanguageId.TYPESCRIPT)

            return SyntaxValidationResult(ok=False, errors=errors, language=LanguageId.TYPESCRIPT)
        finally:
            _cleanup()

    # ── Semantic validation ──────────────────────────────────────────────

    def validate_semantics(self, file_path: str) -> SyntaxValidationResult:
        """Run ``tsc --noEmit`` on the **on-disk** file with project config.

        Unlike :meth:`validate_syntax` (config-blind, single temp file), this
        runs tsc in **project mode** so it picks up ``tsconfig.json`` /
        ``jsconfig.json`` and ``node_modules``. This enables catching type
        errors (TS2xxx), missing imports (TS2307), and undefined names (TS2304)
        that the config-blind syntax check intentionally ignores.

        Design choices:
        - **Project mode (no file on cmdline)**: TS ≥6.0 errors with TS5112
          ("tsconfig.json will not be loaded if files are specified on
          commandline") when a file path is passed alongside a config. We
          instead write a temporary ``tsconfig.<pid>.json`` that ``extends``
          the real config and pins ``include`` to the target file. This
          preserves the project's compiler options (paths, baseUrl, module
          resolution) while checking exactly one file.
        - **Skips entirely if there is no tsconfig.json/jsconfig.json** —
          without config tsc floods output with environment diagnostics
          (missing @types, module resolution) that would wrongly fail valid
          code.
        - Only diagnostics in the 2xxx semantic band are reported; syntax
          (1xxx) is already covered by :meth:`validate_syntax`, and config
          (5xxx)/implicit-any (7xxx) bands are environment-dependent noise.
        - The target-file filter is kept as a defensive net: project mode
          should only compile the pinned file, but ``extends`` may pull in
          ambient declarations that surface sibling diagnostics.
        - Errors (``error TS2xxx``) make ``ok=False``; warnings surfaced.
        """
        return self._run_tsc_semantic(
            file_path,
            language=LanguageId.TYPESCRIPT,
            config_markers=("tsconfig.json",),
            config_filename="tsconfig.json",
            allow_js=False,
        )

    def _run_tsc_semantic(
        self,
        file_path: str,
        *,
        language: LanguageId,
        config_markers: tuple[str, ...],
        config_filename: str,
        allow_js: bool,
    ) -> SyntaxValidationResult:
        """Shared tsc project-mode semantic check for TS and JS providers.

        Writes a temporary ``tsconfig.<pid>.json`` that ``extends`` the real
        config (``tsconfig.json`` or ``jsconfig.json``) with ``include`` pinned
        to *file_path*, then runs ``tsc --noEmit --project <temp>``.

        Args:
            file_path: on-disk file to check (TS or JS).
            language: which ``LanguageId`` to tag the result with.
            config_markers: markers passed to :func:`detect_project_root`.
            config_filename: the real config file found at the project root
                (``tsconfig.json`` for TS, ``tsconfig.json`` or ``jsconfig.json``
                for JS).
            allow_js: whether to force ``--allowJs --checkJs`` for JS files
                whose config may not enable them.
        """
        if not file_path or not os.path.exists(file_path):
            return SyntaxValidationResult(ok=True, language=language)

        project_root = detect_project_root(file_path, markers=config_markers)
        real_config = os.path.join(project_root, config_filename)
        if not os.path.isfile(real_config):
            # No config → tsc would emit config/environment noise. Skip.
            return SyntaxValidationResult(ok=True, language=language)

        # Pin the check to exactly the target file via a temp tsconfig that
        # extends the real one. Relative path is required by tsc `include`.
        rel_target = os.path.relpath(file_path, project_root)
        # Name the temp config tsconfig.*.json so tsc treats it as a project
        # root config; random suffix avoids collisions across parallel checks.
        tmp_config = os.path.join(
            project_root, f".tsconfig.semcheck.{os.getpid()}.{id(file_path)}.json",
        )
        import json as _json
        temp_body: dict = {
            "extends": config_filename,
            "include": [rel_target],
        }
        if allow_js:
            # JS configs may omit allowJs/checkJs — force them so the JS file
            # is actually type-checked rather than just parsed.
            temp_body["compilerOptions"] = {"allowJs": True, "checkJs": True}
        try:
            with open(tmp_config, "w", encoding="utf-8") as fh:
                _json.dump(temp_body, fh)
        except OSError as e:
            logger.debug("could not write temp tsconfig: %s", e)
            return SyntaxValidationResult(ok=True, language=language)

        cmd = [
            "npx", "tsc", "--noEmit", "--pretty", "false",
            "--skipLibCheck", "--project", tmp_config,
        ]
        try:
            try:
                proc = subprocess.run(
                    cmd,
                    capture_output=True, text=True, timeout=120,
                    cwd=project_root,
                )
            except FileNotFoundError:
                logger.debug("tsc not installed; skipping semantic validation")
                return SyntaxValidationResult(ok=True, language=language)
            except subprocess.TimeoutExpired:
                logger.debug("tsc timed out for %s; skipping", file_path)
                return SyntaxValidationResult(ok=True, language=language)
            except Exception as e:
                logger.debug("tsc semantic check failed: %s", e)
                return SyntaxValidationResult(ok=True, language=language)

            if proc.returncode == 0:
                return SyntaxValidationResult(ok=True, language=language)

            # Parse: file.ts(10,5): error TS2304: Cannot find name 'foo'.
            target_norm = os.path.normpath(os.path.abspath(file_path))
            errors: list[SyntaxError_] = []
            has_error = False
            for line in (proc.stdout + proc.stderr).splitlines():
                m = _TSC_ERROR_RE.match(line)
                if not m:
                    continue
                _file, _line, _col, _code, _msg = m.groups()
                # Only report the file we asked about (defensive: project mode
                # pins include, but extends may pull ambient siblings).
                if _file and os.path.normpath(os.path.abspath(_file)) != target_norm:
                    continue
                # Only semantic (2xxx) band: syntax (1xxx) is handled by
                # validate_syntax, config (5xxx) and implicit-any (7xxx) are noise.
                try:
                    num = int(_code[2:])
                except ValueError:
                    continue
                if not (2000 <= num <= 2999):
                    continue
                errors.append(SyntaxError_(
                    file=file_path,
                    line=int(_line), col=int(_col),
                    message=f"{_code}: {_msg}",
                    severity="error",
                    code=_code,
                ))
                has_error = True
            return SyntaxValidationResult(
                ok=not has_error,
                errors=errors,
                language=language,
            )
        finally:
            try:
                os.unlink(tmp_config)
            except OSError:
                pass

    # ── Symbol patterns ───────────────────────────────────────────────────

    def get_symbol_patterns(self, kind: str = "any") -> list[SymbolPattern]:
        patterns: list[SymbolPattern] = []
        if kind in ("function", "any"):
            patterns.append(SymbolPattern(
                kind="function",
                regex=r"(?:export\s+)?(?:async\s+)?function\s+{name}\s*[\(<]",
                description="TS/JS function declaration",
            ))
            patterns.append(SymbolPattern(
                kind="function",
                regex=r"(?:export\s+)?(?:const|let)\s+{name}\s*=\s*(?:async\s*)?\(",
                description="TS/JS arrow / function expression",
            ))
        if kind in ("class", "any"):
            patterns.append(SymbolPattern(
                kind="class",
                regex=r"(?:export\s+)?(?:abstract\s+)?class\s+{name}\s*(?:extends|implements|<|\{)",
                description="TS/JS class declaration",
            ))
        if kind in ("interface", "any"):
            patterns.append(SymbolPattern(
                kind="interface",
                regex=r"(?:export\s+)?interface\s+{name}\s*(?:extends|<|\{)",
                description="TS interface",
            ))
        if kind in ("type", "any"):
            patterns.append(SymbolPattern(
                kind="type",
                regex=r"(?:export\s+)?type\s+{name}\s*(?:=|<)",
                description="TS type alias",
            ))
        return patterns

    # ── File globs ────────────────────────────────────────────────────────

    def get_file_globs(self) -> list[str]:
        return ["*.ts", "*.tsx"]

    # ── Lint / test commands ──────────────────────────────────────────────

    def get_lint_command(self, file_path: str) -> list[str] | None:
        return ["npx", "eslint", "--format=json", file_path]

    def get_test_directory(self, repo_root: str) -> str | None:
        """Detect test directory from jest/vitest config files.

        Checks these config files in order:
          1. jest.config.js / jest.config.ts  (roots field)
          2. vitest.config.ts / vitest.config.js  (test.dir or test.include)
          3. package.json (scripts.test or jest config inline)
          4. package.json devDependencies/dependencies (jest/vitest convention)

        Returns configured test root (e.g. '__tests__', 'tests', 'spec')
        or ``None`` to fall back to convention-based detection.
        """
        import re as _re

        # ── Helper: read and try to parse a config file ────────────────
        def _read_config(path: str) -> dict | None:
            full = os.path.join(repo_root, path)
            if not os.path.isfile(full):
                return None
            try:
                with open(full, encoding="utf-8", errors="replace") as f:
                    text = f.read()
            except OSError:
                return None
            # Try JSON (jest.config.json, package.json)
            if path.endswith(".json"):
                try:
                    return json.loads(text)
                except json.JSONDecodeError:
                    return None
            # For .js/.ts config files, try to extract the config object
            # by looking for common export patterns
            # jest.config.js: module.exports = { roots: ['<rootDir>/__tests__'] }
            _roots_m = _re.search(r"roots\s*[:=]\s*\[([^\]]+)\]", text)
            _dirs_m = _re.search(r"(?:test|dir)(?:s|Path|Directory)?\s*[:=]\s*[\"']([^\"']+)[\"']", text)
            _suites_m = _re.search(r"testMatch\s*[:=]\s*\[([^\]]+)\]", text)
            result = {}
            if _roots_m:
                items = _roots_m.group(1)
                dirs = _re.findall(r"[\"']([^\"']+)[\"']", items)
                # Replace <rootDir> with actual path relative to repo
                result["roots"] = [
                    d.replace("<rootDir>", ".") for d in dirs
                ]
            if _dirs_m:
                result["dir"] = _dirs_m.group(1).replace("<rootDir>", ".")
            if _suites_m:
                result["testMatch"] = _suites_m.group(0)[:200]
            return result if result else None

        # ── 1. jest.config.js / jest.config.ts ─────────────────────────
        for cfg_name in ("jest.config.js", "jest.config.ts", "jest.config.json", "jest.config.mjs"):
            cfg = _read_config(cfg_name)
            if cfg:
                roots = cfg.get("roots") or []
                if roots:
                    # Pick the first root that looks like a test directory
                    for r in roots:
                        _bare = r.replace("<rootDir>", "").strip("./")
                        # Prefer roots containing 'test' or 'spec'
                        if "test" in _bare.lower() or "spec" in _bare.lower():
                            return _bare
                    # Fallback: use first root
                    _first = roots[0].replace("<rootDir>", "").strip("./")
                    if _first:
                        return _first
                # Check inline dir
                _dir = cfg.get("dir", "")
                if _dir and _dir != ".":
                    return _dir.strip("./")

        # ── 2. vitest.config.ts / vitest.config.js ─────────────────────
        for cfg_name in ("vitest.config.ts", "vitest.config.js", "vitest.config.mjs"):
            cfg = _read_config(cfg_name)
            if cfg:
                _dir = cfg.get("dir", "") or cfg.get("testMatch", "")
                if "test" in _dir.lower() or "spec" in _dir.lower():
                    return _dir.strip("./").strip("*")

        # ── 3. package.json (scripts or jest config) ───────────────────
        pkg_path = os.path.join(repo_root, "package.json")
        if os.path.isfile(pkg_path):
            try:
                with open(pkg_path, encoding="utf-8") as f:
                    pkg = json.load(f)
                # Inline jest config: { "jest": { "roots": [...] } }
                jest_cfg = pkg.get("jest")
                if isinstance(jest_cfg, dict):
                    roots = jest_cfg.get("roots") or []
                    for r in roots:
                        _bare = r.replace("<rootDir>", "").strip("./")
                        if "test" in _bare.lower():
                            return _bare
                # Check test command for directory hints
                test_script = pkg.get("scripts", {}).get("test", "")
                _m = _re.search(r"--roots\s+(\S+)|(?:__tests__|tests/|spec/)", test_script)
                if _m:
                    _found = _m.group(1) or _m.group(0)
                    if "test" in _found.lower():
                        return _found.strip("./")
            except (json.JSONDecodeError, OSError):
                pass

        # ── 4. Convention: check if __tests__ or tests exists ──────────
        for _candidate in ("__tests__", "tests", "spec", "test"):
            _full = os.path.join(repo_root, _candidate)
            if os.path.isdir(_full):
                return _candidate

        return None

    def get_test_command(
        self, repo_root: str, test_args: list[str] | None = None
    ) -> list[str] | None:
        """Auto-detect test runner from package.json (jest/vitest)."""
        pkg_path = os.path.join(repo_root, "package.json")
        runner = "jest"  # default
        if os.path.isfile(pkg_path):
            try:
                with open(pkg_path) as f:
                    pkg = json.load(f)
                deps = {**pkg.get("dependencies", {}), **pkg.get("devDependencies", {})}
                if "vitest" in deps:
                    runner = "vitest"
            except Exception:
                pass
        return ["npx", runner, "--passWithNoTests"] + (test_args or [])

    # ── Symbol finder (regex + brace counting) ────────────────────────────

    def find_symbol_in_file(
        self, file_path: str, symbol_name: str, content: str
    ) -> tuple[int, int] | None:
        """Find *symbol_name* using tree-sitter (precise) or regex + brace counting (fallback)."""
        from .tree_sitter_utils import find_symbol_range, is_available

        if is_available():
            result = find_symbol_range(content, symbol_name, "typescript")
            if result:
                return result

        return self._find_symbol_regex(file_path, symbol_name, content)

    def _find_symbol_regex(
        self, file_path: str, symbol_name: str, content: str
    ) -> tuple[int, int] | None:
        """Fallback: regex match + brace counting for block end."""
        esc = re.escape(symbol_name)
        for sp in self.get_symbol_patterns("any"):
            pat = sp.regex.replace("{name}", esc)
            for m in re.finditer(pat, content, re.MULTILINE):
                start_offset = m.start()
                start_line = content[:start_offset].count("\n") + 1
                end_line = self._find_block_end(content, start_offset)
                return (start_line, end_line)
        return None

    @staticmethod
    def _find_block_end(content: str, offset: int) -> int:
        """Heuristic: find the matching closing brace from *offset*."""
        depth = 0
        started = False
        line = content[:offset].count("\n") + 1
        i = offset
        length = len(content)
        while i < length:
            ch = content[i]
            if ch == "\n":
                line += 1
            elif ch == "{":
                depth += 1
                started = True
            elif ch == "}":
                depth -= 1
                if started and depth == 0:
                    return line
            i += 1
        # Fallback: couldn't find matching brace, return start + 20
        return content[:offset].count("\n") + 21

    # ── Definition keywords ───────────────────────────────────────────────

    # ── Regex fallback for structural queries ─────────────────────────────

    def _find_top_level_definitions_regex(
        self, content: str,
    ) -> list[tuple[str, str, int, int]]:
        """Regex fallback: find all top-level TS/JS definitions via pattern + brace counting."""
        results: list[tuple[str, str, int, int]] = []
        # Functions: function Name(  or async function Name(
        for m in re.finditer(
            r'^(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*\(',
            content, re.MULTILINE,
        ):
            start_line = content[:m.start()].count("\n") + 1
            end_line = self._find_block_end(content, m.start())
            results.append((m.group(1), "function", start_line, end_line))
        # Classes
        for m in re.finditer(r'^(?:export\s+)?(?:abstract\s+)?class\s+(\w+)', content, re.MULTILINE):
            start_line = content[:m.start()].count("\n") + 1
            end_line = self._find_block_end(content, m.start())
            results.append((m.group(1), "class", start_line, end_line))
        # Interfaces
        for m in re.finditer(r'^(?:export\s+)?interface\s+(\w+)', content, re.MULTILINE):
            start_line = content[:m.start()].count("\n") + 1
            end_line = self._find_block_end(content, m.start())
            results.append((m.group(1), "interface", start_line, end_line))
        # Type aliases
        for m in re.finditer(r'^(?:export\s+)?type\s+(\w+)\s*=', content, re.MULTILINE):
            start_line = content[:m.start()].count("\n") + 1
            # Type aliases end at semicolon or newline, not brace
            semi = content.find(";", m.start())
            end_line = content[:len(content) if semi == -1 else semi + 1].count("\n")
            if end_line <= start_line:
                end_line = start_line + 1
            results.append((m.group(1), "type", start_line, end_line))
        return results

    @staticmethod
    def _find_block_end_offset(content: str, offset: int) -> int:
        """Find offset of matching closing brace."""
        depth = 0
        started = False
        for i in range(offset, len(content)):
            ch = content[i]
            if ch == "{":
                depth += 1
                started = True
            elif ch == "}":
                depth -= 1
                if started and depth == 0:
                    return i + 1
        return len(content)

    def _find_class_methods_regex(
        self, content: str, class_name: str,
    ) -> list[tuple[str, int, int]]:
        """Regex fallback: find methods inside a TS/JS class body."""
        results: list[tuple[str, int, int]] = []
        esc = re.escape(class_name)
        # Find class definition
        pat = r'(?:export\s+)?(?:abstract\s+)?class\s+' + esc + r'\s*(?:extends|implements|<|\{|[^{]+?\{)'
        for cm in re.finditer(pat, content):
            class_body_start = content.find("{", cm.start())
            if class_body_start == -1:
                continue
            class_end = self._find_block_end_offset(content, class_body_start)
            class_body = content[class_body_start:class_end]
            # Match methods: method_name(  or async method_name(  or get/set
            for mm in re.finditer(
                r'(?:(?:public|private|protected|static|async|get|set)\s+)*'
                r'(?:(\w+)\s*\(|get\s+(\w+)\s*\(|set\s+(\w+)\s*\()',
                class_body,
            ):
                _name = mm.group(1) or mm.group(2) or mm.group(3)
                if not _name or _name in ("if", "for", "while", "switch", "catch"):
                    continue
                method_start = class_body_start + mm.start()
                method_line = content[:method_start].count("\n") + 1
                method_end = self._find_block_end(content, method_start)
                results.append((_name, method_line, method_end))
        return results

    def _find_symbol_body_range_regex(
        self, content: str, symbol_name: str,
    ) -> tuple[int, int] | None:
        """Regex fallback: find function body via first { after definition."""
        esc = re.escape(symbol_name)
        for sp in self.get_symbol_patterns("any"):
            pat = sp.regex.replace("{name}", esc)
            for m in re.finditer(pat, content, re.MULTILINE):
                body_start = content.find("{", m.end())
                if body_start == -1:
                    continue
                body_start_line = content[:body_start].count("\n") + 1
                body_end_line = self._find_block_end(content, body_start)
                return (body_start_line, body_end_line)
        return None

    # ── Structural query methods (tree-sitter → regex fallback) ────────────

    def find_top_level_definitions(
        self, content: str,
    ) -> list[tuple[str, str, int, int]]:
        from .tree_sitter_utils import find_all_symbols, is_available
        result = find_all_symbols(content, "typescript") if is_available() else None
        if result:
            return result
        return self._find_top_level_definitions_regex(content)

    def find_class_methods(
        self, content: str, class_name: str,
    ) -> list[tuple[str, int, int]]:
        from .tree_sitter_utils import extract_class_methods, is_available
        result = extract_class_methods(content, class_name, "typescript") if is_available() else None
        if result:
            return result
        return self._find_class_methods_regex(content, class_name)

    def find_symbol_body_range(
        self, content: str, symbol_name: str,
    ) -> tuple[int, int] | None:
        from .tree_sitter_utils import extract_symbol_body, is_available
        result = extract_symbol_body(content, symbol_name, "typescript") if is_available() else None
        if result:
            return result
        return self._find_symbol_body_range_regex(content, symbol_name)

    def get_definition_keywords(self) -> list[str]:
        return [
            "function ",
            "async function ",
            "class ",
            "interface ",
            "type ",
            "const ",
            "export function ",
            "export async function ",
            "export class ",
            "export interface ",
            "export type ",
            "export const ",
        ]
