"""
TypeScript / TSX syntax provider.

Uses ``tsc --noEmit`` for validation and regex-based symbol detection.
Gracefully degrades when ``tsc`` is not installed.
"""
from __future__ import annotations

import contextlib
import json
import logging
import os
import re
import subprocess
import uuid

from .base import (
    SyntaxProvider,
    _replace_last_cmd_path,
    _tempfile_for_content,
    build_line_index,
    detect_project_root,
    find_brace_block_end,
    find_brace_block_end_offset,
    line_at_offset,
    resolve_tool_path,
    tree_sitter_syntax_fallback,
)
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
                    check=False,
                )
            except FileNotFoundError:
                logger.debug("tsc not installed; falling back to tree-sitter")
                return tree_sitter_syntax_fallback(content, LanguageId.TYPESCRIPT, file_path)
            except subprocess.TimeoutExpired:
                logger.debug("tsc timed out for %s", file_path)
                return tree_sitter_syntax_fallback(content, LanguageId.TYPESCRIPT, file_path)
            except Exception as e:
                logger.debug("tsc error: %s", e)
                return tree_sitter_syntax_fallback(content, LanguageId.TYPESCRIPT, file_path)

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
        return self.validate_semantics_batch([file_path])[file_path]

    def validate_semantics_batch(
        self, file_paths: list[str],
    ) -> dict[str, SyntaxValidationResult]:
        """Semantic-check *file_paths* with one tsc run per project root.

        Each file otherwise paid its own ``npx tsc`` startup, which is the bulk
        of a short check. tsc already type-checks everything in ``include``
        together and tags each diagnostic with its file, so widening ``include``
        to the whole group costs about one startup instead of N.

        Grouped by :func:`detect_project_root` because that is tsc's cwd and
        selects which ``tsconfig.json`` applies — a monorepo has one per
        package, and merging them would check files against the wrong config.
        """
        return self._batch_by_root(
            file_paths,
            language=LanguageId.TYPESCRIPT,
            config_markers=("tsconfig.json",),
            config_for=lambda _root: "tsconfig.json",
            allow_js=False,
        )

    def _batch_by_root(
        self,
        file_paths: list[str],
        *,
        language: LanguageId,
        config_markers: tuple[str, ...],
        config_for,
        allow_js: bool,
    ) -> dict[str, SyntaxValidationResult]:
        """Group *file_paths* by (project root, config) and run tsc once each.

        *config_for* maps a project root to the config filename to extend; JS
        needs it because a JS project may carry either ``jsconfig.json`` or
        ``tsconfig.json``, and the two cannot share one temp config.
        """
        # A fresh result per file, never one shared instance: the dataclass is
        # mutable and carries a list, so sharing would let one consumer's
        # append surface on every other skipped file.
        def _skip(reason: str) -> SyntaxValidationResult:
            return SyntaxValidationResult.unchecked(language, reason)

        out: dict[str, SyntaxValidationResult] = {}
        groups: dict[tuple[str, str], list[str]] = {}
        for p in file_paths:
            if not p or not os.path.exists(p):
                out[p] = _skip("the file is not on disk")
                continue
            root = detect_project_root(p, markers=config_markers)
            cfg = config_for(root)
            if cfg is None or not os.path.isfile(os.path.join(root, cfg)):
                # No config → tsc would emit config/environment noise. Skip.
                out[p] = _skip(
                    f"no {' or '.join(config_markers)} above this file, "
                    "so tsc has no project to check it against",
                )
                continue
            groups.setdefault((root, cfg), []).append(p)
        for (root, cfg), paths in groups.items():
            out.update(self._run_tsc_semantic(
                paths,
                language=language,
                project_root=root,
                config_filename=cfg,
                allow_js=allow_js,
            ))
        return out

    def _run_tsc_semantic(
        self,
        file_paths: list[str],
        *,
        language: LanguageId,
        project_root: str,
        config_filename: str,
        allow_js: bool,
    ) -> dict[str, SyntaxValidationResult]:
        """One tsc project-mode run over *file_paths*, split back out per file.

        Writes a temporary ``.tsconfig.semcheck.*.json`` that ``extends`` the
        real config (``tsconfig.json`` or ``jsconfig.json``) with ``include``
        pinned to *file_paths*, then runs ``tsc --noEmit --project <temp>``.

        Args:
            file_paths: on-disk files to check (TS or JS), all under
                *project_root* and sharing *config_filename*.
            language: which ``LanguageId`` to tag the results with.
            project_root: tsc's cwd; also where the temp config is written.
            config_filename: the real config file at *project_root*.
            allow_js: whether to force ``allowJs``/``checkJs`` for JS files
                whose config may not enable them.
        """
        def _skip(reason: str) -> dict[str, SyntaxValidationResult]:
            return {
                p: SyntaxValidationResult.unchecked(language, reason)
                for p in file_paths
            }

        # Pin the check to exactly the target files via a temp tsconfig that
        # extends the real one. Relative paths are required by tsc `include`.
        rel_targets = [os.path.relpath(p, project_root) for p in file_paths]
        # The temp config sits in project_root next to the real one; pid + a
        # random token keep concurrent checks from colliding. (``id()`` is NOT
        # usable here: CPython reuses addresses after GC, so it is unique only
        # among simultaneously-live objects.)
        tmp_config = os.path.join(
            project_root,
            f".tsconfig.semcheck.{os.getpid()}.{uuid.uuid4().hex}.json",
        )
        import json as _json
        temp_body: dict = {
            # MUST be "./name", not a bare "name": tsc resolves a bare extends
            # as a *node module* specifier, so the real config is never loaded
            # and the check silently runs on tsc defaults — losing the
            # project's strict/paths/jsx/lib settings. The resulting
            # "TS6053: File 'tsconfig.json' not found" lands in the 6xxx band
            # that the diagnostic filter below drops, so the failure is
            # invisible. Verified: bare -> TS6053, "./" -> clean.
            "extends": f"./{config_filename}",
            "include": rel_targets,
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
            return _skip("the temporary tsconfig could not be written")

        cmd = [
            "npx", "tsc", "--noEmit", "--pretty", "false",
            "--skipLibCheck", "--project", tmp_config,
        ]
        try:
            try:
                proc = subprocess.run(
                    cmd,
                    capture_output=True, text=True,
                    # Scaled by batch size; startup dominates, so the base
                    # budget still covers the common small batch.
                    timeout=30 + 5 * len(file_paths),
                    cwd=project_root,
                    check=False,
                )
            except FileNotFoundError:
                logger.debug("tsc not installed; skipping semantic validation")
                return _skip("tsc is not installed (npx could not resolve it)")
            except subprocess.TimeoutExpired:
                logger.debug("tsc timed out for %s; skipping", file_paths)
                return _skip("tsc timed out")
            except Exception as e:
                logger.debug("tsc semantic check failed: %s", e)
                return _skip("tsc could not be run")

            if proc.returncode == 0:
                # tsc exited clean — a real verdict, not a skip. Sharing the
                # skip constructor here would report a genuinely checked file as
                # unchecked, the same conflation in the other direction.
                return {
                    p: SyntaxValidationResult(ok=True, language=language)
                    for p in file_paths
                }

            # Parse: file.ts(10,5): error TS2304: Cannot find name 'foo'.
            by_norm = {os.path.normpath(os.path.abspath(p)): p for p in file_paths}
            collected: dict[str, list[SyntaxError_]] = {p: [] for p in file_paths}
            failed: set[str] = set()
            for line in (proc.stdout + proc.stderr).splitlines():
                m = _TSC_ERROR_RE.match(line)
                if not m:
                    continue
                _file, _line, _col, _code, _msg = m.groups()
                # Only report the files we asked about (project mode pins
                # include, but extends may pull ambient siblings, and a batched
                # run legitimately reports every file in the group).
                #
                # tsc prints paths relative to ITS cwd (= project_root), so a
                # bare abspath() would resolve them against the agent process's
                # cwd instead — and this process never chdirs, so that is
                # wherever the user launched from. Whenever it differs from
                # project_root (monorepo package, launch from a subdirectory)
                # EVERY diagnostic missed the target and was dropped, making a
                # broken file report ok=True/errors=[] — indistinguishable from
                # "checked, clean". Resolve against the tool's cwd, as
                # go_provider already does. normpath collapses the "./" prefix
                # tsc sometimes emits.
                owner = (
                    by_norm.get(resolve_tool_path(project_root, _file))
                    if _file else None
                )
                if owner is None:
                    continue
                # Only semantic (2xxx) band: syntax (1xxx) is handled by
                # validate_syntax, config (5xxx) and implicit-any (7xxx) are noise.
                # NOTE: no try/except around int(_code[2:]) — _TSC_ERROR_RE
                # captures `TS\d+` only, so the digits are guaranteed (a
                # ValueError guard here would be an unreachable branch).
                num = int(_code[2:])
                if not (2000 <= num <= 2999):
                    continue
                collected[owner].append(SyntaxError_(
                    file=owner,
                    line=int(_line), col=int(_col),
                    message=f"{_code}: {_msg}",
                    severity="error",
                    code=_code,
                ))
                failed.add(owner)
            return {
                p: SyntaxValidationResult(
                    ok=p not in failed,
                    errors=collected[p],
                    language=language,
                )
                for p in file_paths
            }
        finally:
            with contextlib.suppress(OSError):
                os.unlink(tmp_config)

    # ── Symbol patterns ───────────────────────────────────────────────────

    def get_symbol_patterns(self, kind: str = "any") -> list[SymbolPattern]:
        patterns: list[SymbolPattern] = []
        if kind in ("function", "any"):
            patterns.append(SymbolPattern(
                kind="function",
                regex=r"(?:export\s+)?(?:async\s+)?function\s*\*?\s*{name}\s*[\(<]",
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
        # .mts/.cts are TypeScript ESM/CJS module variants — first-class in
        # _EXT_MAP / the JS-TS family / _EXT_TO_GRAMMAR_KEY (all → typescript).
        return ["*.ts", "*.tsx", "*.mts", "*.cts"]

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
                logger.debug("could not read test config %s", full)
                return None
            # Try JSON (jest.config.json, package.json)
            if path.endswith(".json"):
                try:
                    return json.loads(text)
                except json.JSONDecodeError:
                    logger.debug("test config %s is not valid JSON", full)
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
            with contextlib.suppress(json.JSONDecodeError, OSError), open(pkg_path, encoding="utf-8") as f:
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
            with contextlib.suppress(OSError, ValueError), open(pkg_path, encoding="utf-8") as f:  # missing/unparseable package.json
                pkg = json.load(f)
                deps = {**pkg.get("dependencies", {}), **pkg.get("devDependencies", {})}
                if "vitest" in deps:
                    runner = "vitest"
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

        return self._find_symbol_regex(symbol_name, content)

    @staticmethod
    def _find_block_end(content: str, offset: int, nl: list[int] | None = None) -> int:
        """Heuristic: find the matching closing brace from *offset*.

        Delegates to the shared :func:`find_brace_block_end` (C-family SSOT)
        which skips string/char/template literals and ``//`` / ``/* */``
        comments so braces inside them do not corrupt the depth counter.
        *nl* is an optional precomputed line index (see ``base.build_line_index``)
        that keeps the internal line queries O(log n) in hot loops.
        """
        return find_brace_block_end(content, offset, nl)

    # ── Definition keywords ───────────────────────────────────────────────

    # ── Regex fallback for structural queries ─────────────────────────────

    def _find_top_level_definitions_regex(
        self, content: str,
    ) -> list[tuple[str, str, int, int]]:
        """Regex fallback: find all top-level TS/JS definitions via pattern + brace counting."""
        results: list[tuple[str, str, int, int]] = []
        nl = build_line_index(content)
        # Functions: function Name( / async function Name( / generator
        # function* Name( (the * is optional — a generator declaration was
        # previously invisible to the fallback).
        for m in re.finditer(
            r'^(?:export\s+)?(?:async\s+)?function\s*\*?\s+(\w+)\s*\(',
            content, re.MULTILINE,
        ):
            start_line = line_at_offset(nl, m.start())
            end_line = self._find_block_end(content, m.start(), nl)
            results.append((m.group(1), "function", start_line, end_line))
        # Classes
        for m in re.finditer(r'^(?:export\s+)?(?:abstract\s+)?class\s+(\w+)', content, re.MULTILINE):
            start_line = line_at_offset(nl, m.start())
            end_line = self._find_block_end(content, m.start(), nl)
            results.append((m.group(1), "class", start_line, end_line))
        # Interfaces
        for m in re.finditer(r'^(?:export\s+)?interface\s+(\w+)', content, re.MULTILINE):
            start_line = line_at_offset(nl, m.start())
            end_line = self._find_block_end(content, m.start(), nl)
            results.append((m.group(1), "interface", start_line, end_line))
        # Type aliases
        for m in re.finditer(r'^(?:export\s+)?type\s+(\w+)\s*=', content, re.MULTILINE):
            start_line = line_at_offset(nl, m.start())
            # Type aliases end at semicolon or newline, not brace. Use the
            # 1-based line_at_offset for the end too — the previous code
            # compared a 0-based line_index_at_offset against the 1-based
            # start_line, under-reporting end_line by one whenever the
            # semicolon sat two or more lines below the `type` keyword.
            semi = content.find(";", m.start())
            end_line = line_at_offset(nl, len(content) if semi == -1 else semi + 1)
            if end_line <= start_line:
                end_line = start_line + 1
            results.append((m.group(1), "type", start_line, end_line))
        return results

    @staticmethod
    def _find_block_end_offset(content: str, offset: int) -> int:
        """Offset (exclusive) of matching ``}`` for the class-body range.

        Delegates to :func:`base.find_brace_block_end_offset` (the shared SSOT) so
        braces inside string/char/template literals or comments cannot corrupt the
        depth counter. See base.py for the full contract.
        """
        return find_brace_block_end_offset(content, offset)

    def _find_class_methods_regex(
        self, content: str, class_name: str,
    ) -> list[tuple[str, int, int]]:
        """Regex fallback: find methods inside a TS/JS class body."""
        results: list[tuple[str, int, int]] = []
        nl = build_line_index(content)
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
                method_line = line_at_offset(nl, method_start)
                method_end = self._find_block_end(content, method_start, nl)
                results.append((_name, method_line, method_end))
        return results

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
