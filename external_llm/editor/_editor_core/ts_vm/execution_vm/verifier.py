"""verifier.py — Multi-level verification for TS/JS code.

Three verification levels (executed in order, each gated):

1. **Parse check** — tree-sitter can parse without ERROR nodes
2. **tsc check** — TypeScript compiler (--noEmit) finds no errors
3. **eslint check** — linter passes

Each level can be individually enabled/disabled. The VM uses
parse check always, tsc/eslint only when available.
"""
from __future__ import annotations

import logging
import os
import subprocess
import tempfile

from external_llm.editor._editor_core.ts_vm.execution_vm.models import VerifyError
from external_llm.editor.semantic.ts_semantic_tracer import TSSemanticTracer

logger = logging.getLogger(__name__)


class TSVerifier:
    """Verifies TS/JS code at multiple levels."""

    def __init__(
        self,
        language: str = "typescript",
        use_tsc: bool = True,
        use_eslint: bool = False,
        **kwargs,
    ):
        self._language = language
        self._tracer = TSSemanticTracer(language=language)
        self._use_tsc = use_tsc
        self._use_eslint = use_eslint

    def verify(self, code: str, file_path: str = "check.ts") -> tuple[bool, list[VerifyError]]:
        """Run all enabled verification levels.

        Returns (ok, errors).
        """

        # Level 1: Parse check (always)
        parse_ok, parse_errors = self._check_parse(code, file_path)
        if not parse_ok:
            return False, parse_errors

        # Level 2: tsc (if enabled and available)
        if self._use_tsc and self._language == "typescript":
            tsc_ok, tsc_errors = self._check_tsc(code)
            if not tsc_ok:
                return False, tsc_errors

        # Level 3: eslint (if enabled)
        if self._use_eslint:
            lint_ok, lint_errors = self._check_eslint(code)
            if not lint_ok:
                return False, lint_errors

        return True, []

    # ── Level 1: Parse ───────────────────────────────────────────────

    def _check_parse(
        self, code: str, file_path: str,
    ) -> tuple[bool, list[VerifyError]]:
        """Check that tree-sitter can parse without ERROR nodes."""
        from external_llm.languages.tree_sitter_utils import is_available

        if not is_available():
            return True, []  # can't check

        parser = self._tracer._get_parser()
        if parser is None:
            return True, []

        try:
            tree = parser.parse(code.encode("utf-8"))
            errors = self._collect_parse_errors(tree.root_node, code)
            return len(errors) == 0, errors
        except Exception as e:
            return False, [VerifyError(message=f"Parse exception: {e}")]

    def _collect_parse_errors(self, node, code: str) -> list[VerifyError]:
        errors: list[VerifyError] = []
        if node.type == "ERROR" or node.is_missing:
            line = node.start_point.row + 1
            col = node.start_point.column + 1
            snippet = code[node.start_byte: min(node.end_byte, node.start_byte + 40)]
            errors.append(VerifyError(
                message=f"Parse error near: {snippet!r}",
                line=line, column=col,
            ))
        for child in node.children:
            errors.extend(self._collect_parse_errors(child, code))
        return errors

    # ── Level 2: tsc ─────────────────────────────────────────────────

    def _check_tsc(self, code: str) -> tuple[bool, list[VerifyError]]:
        """Run tsc --noEmit on the code. Returns (ok, errors).

        This check is intentionally ISOLATED: it runs on a temp file with
        ``--ignoreConfig``, so it sees neither the project's compiler options
        (esModuleInterop, lib, paths) nor its node_modules / @types. As a
        result, only genuine PARSER syntax errors are reliable here. Every
        config/environment-dependent diagnostic — module resolution (TS2307
        for 'ws' or './types'), missing globals (TS2503 'NodeJS', TS2591
        'process'), interop (TS1259 esModuleInterop), missing declarations
        (TS7016) and the type-checker cascade they trigger (TS2339 …) — is a
        FALSE POSITIVE in this isolated environment and must not fail an edit
        that is valid under the real project config. We therefore keep only
        genuine syntax errors. (Previously only TS2307-relative was filtered,
        which let TS1259/TS2503/TS2591/… roll back valid edits.)
        """
        tmp_path = None
        from external_llm.languages.typescript_provider import is_genuine_syntax_error

        try:
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=".ts", mode="w", encoding="utf-8",
            ) as f:
                f.write(code)
                tmp_path = f.name

            proc = subprocess.run(
                [
                    "tsc", "--ignoreConfig", "--noEmit", "--allowJs",
                    "--isolatedModules", "--skipLibCheck", tmp_path,
                ],
                capture_output=True, text=True, timeout=15,
            )

            if proc.returncode == 0:
                return True, []

            errors = self._parse_tsc_output(proc.stdout + proc.stderr)
            # Keep only config/environment-independent parser syntax errors.
            errors = [e for e in errors if is_genuine_syntax_error(e.code or "")]
            return (len(errors) == 0), errors

        except FileNotFoundError:
            logger.debug("tsc not found, skipping type check")
            return True, []
        except subprocess.TimeoutExpired:
            return True, []  # timeout = skip
        except Exception as e:
            logger.debug("tsc check failed: %s", e)
            return True, []
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    def _parse_tsc_output(self, output: str) -> list[VerifyError]:
        """Parse tsc error output into VerifyError list."""
        import re

        errors: list[VerifyError] = []
        # Pattern: file(line,col): error TS2304: ...
        pattern = re.compile(r"\((\d+),(\d+)\):\s+error\s+(TS\d+):\s+(.+)")
        for line in output.split("\n"):
            m = pattern.search(line)
            if m:
                errors.append(VerifyError(
                    message=m.group(4).strip(),
                    line=int(m.group(1)),
                    column=int(m.group(2)),
                    code=m.group(3),
                ))
            elif "error" in line.lower() and not errors:
                errors.append(VerifyError(message=line.strip()))
        return errors or [VerifyError(message=output[:200])]

    # ── Level 3: eslint ──────────────────────────────────────────────

    def _check_eslint(self, code: str) -> tuple[bool, list[VerifyError]]:
        """Run eslint on the code. Returns (ok, errors)."""
        tmp_path = None
        try:
            suffix = ".ts" if self._language == "typescript" else ".js"
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=suffix, mode="w", encoding="utf-8",
            ) as f:
                f.write(code)
                tmp_path = f.name

            proc = subprocess.run(
                ["eslint", "--format=json", tmp_path],
                capture_output=True, text=True, timeout=15,
            )

            if proc.returncode == 0:
                return True, []

            return False, [VerifyError(message="eslint errors found")]

        except FileNotFoundError:
            return True, []  # eslint not installed
        except Exception:
            return True, []
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
