"""verifier.py — Multi-level verification for all supported languages.

Each language verifier supports:
1. Parse check (tree-sitter) — always, instant
2. Compiler check — compile() / javac / kotlinc / go build
3. Linter check — optional, if the tool is available

Verifiers fall back gracefully when tools are not installed.
"""
from __future__ import annotations

import ast
import logging
import os
import subprocess
import tempfile
from abc import ABC, abstractmethod

from external_llm.editor._editor_core.vm.models import VerifyError

logger = logging.getLogger(__name__)


class BaseVerifier(ABC):
    """Abstract base for language-specific verifiers."""

    @abstractmethod
    def verify(self, code: str, file_path: str = "") -> tuple[bool, list[VerifyError]]:
        """Run all enabled verification levels. Returns (ok, errors)."""
        ...


# ── Python ────────────────────────────────────────────────────────────

class PythonVerifier(BaseVerifier):
    """Verifies Python code via ast.parse + compile() + optional pyright."""

    def __init__(self, use_pyright: bool = False):
        self._use_pyright = use_pyright

    def verify(self, code: str, file_path: str = "check.py") -> tuple[bool, list[VerifyError]]:
        # Level 1: ast.parse
        try:
            ast.parse(code, filename=file_path)
        except SyntaxError as e:
            return False, [VerifyError(
                message=f"SyntaxError: {e.msg}",
                line=e.lineno or 0,
                column=e.offset or 0,
                code="E0001",
            )]

        # Level 2: compile() — stricter, catches了一些 ast.parse misses
        try:
            compile(code, file_path, "exec")
        except SyntaxError as e:
            return False, [VerifyError(
                message=f"CompileError: {e.msg}",
                line=e.lineno or 0,
                column=e.offset or 0,
                code="E0002",
            )]
        except ValueError as e:
            # e.g. "source code string cannot contain null bytes"
            return False, [VerifyError(message=str(e), code="E0003")]

        # Level 3: pyright (optional)
        if self._use_pyright:
            ok, errors = self._check_pyright(code)
            if not ok:
                return False, errors

        return True, []

    def _check_pyright(self, code: str) -> tuple[bool, list[VerifyError]]:
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=".py", mode="w", encoding="utf-8",
            ) as f:
                f.write(code)
                tmp_path = f.name

            # Use --outputjson for stable machine-readable output with rule codes
            proc = subprocess.run(
                ["pyright", "--outputjson", tmp_path],
                capture_output=True, text=True, timeout=30,
            )
            if proc.returncode == 0:
                return True, []

            errors = self._parse_pyright_output(proc.stdout)
            return False, errors

        except FileNotFoundError:
            logger.debug("pyright not found, skipping")
            return True, []
        except subprocess.TimeoutExpired:
            return True, []
        except Exception as e:
            logger.debug("pyright check failed: %s", e)
            return True, []
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    def _parse_pyright_output(self, output: str) -> list[VerifyError]:
        """Parse pyright --outputjson output.

        JSON format:
        {
            "version": "...",
            "summary": {...},
            "generalDiagnostics": [
                {
                    "file": "...",
                    "severity": "error",
                    "message": "...",
                    "rule": "reportUndefinedVariable",  # <-- stable code
                    "range": {"start": {"line": 4, "character": 0}, ...}
                }
            ]
        }
        """
        import json
        errors: list[VerifyError] = []
        try:
            data = json.loads(output)
            for diag in data.get("generalDiagnostics", []):
                if diag.get("severity") != "error":
                    continue
                rule = diag.get("rule", "PYRIGHT")  # e.g. "reportUndefinedVariable"
                start = diag.get("range", {}).get("start", {})
                errors.append(VerifyError(
                    message=diag.get("message", "").strip(),
                    line=start.get("line", 0) + 1,  # 0-based → 1-based
                    column=start.get("character", 0) + 1,
                    code=rule,
                ))
        except (json.JSONDecodeError, KeyError) as e:
            logger.debug("Failed to parse pyright JSON: %s", e)
        return errors


# ── Java ──────────────────────────────────────────────────────────────

class JavaVerifier(BaseVerifier):
    """Verifies Java code via javac compilation."""

    def verify(self, code: str, file_path: str = "Check.java") -> tuple[bool, list[VerifyError]]:
        tmp_dir = None
        try:
            tmp_dir = tempfile.mkdtemp()
            src_path = os.path.join(tmp_dir, os.path.basename(file_path))
            with open(src_path, "w", encoding="utf-8") as f:
                f.write(code)

            # Use -XDrawDiagnostics for stable machine-readable keys (locale-independent)
            # Use -J-Duser.language=en as fallback for older JDKs
            env = os.environ.copy()
            env["LC_ALL"] = "C"
            env["LANG"] = "C"

            proc = subprocess.run(
                ["javac", "-XDrawDiagnostics", "-J-Duser.language=en", src_path],
                capture_output=True, text=True, timeout=30,
                cwd=tmp_dir,
                env=env,
            )
            if proc.returncode == 0:
                return True, []

            errors = self._parse_javac_output(proc.stdout + proc.stderr)
            return False, errors

        except FileNotFoundError:
            logger.debug("javac not found, skipping verification")
            return True, []
        except subprocess.TimeoutExpired:
            return True, []
        except Exception as e:
            logger.debug("javac check failed: %s", e)
            return True, []
        finally:
            if tmp_dir:
                import shutil
                shutil.rmtree(tmp_dir, ignore_errors=True)

    def _parse_javac_output(self, output: str) -> list[VerifyError]:
        """Parse javac error output into VerifyError list.

        With -XDrawDiagnostics, javac outputs stable diagnostic keys like:
            Check.java:5:9: compiler.err.cant.resolve.location: ...
        Without -XDrawDiagnostics (fallback), format is:
            Check.java:5: error: cannot find symbol
        """
        import re
        errors: list[VerifyError] = []

        # Pattern 1: -XDrawDiagnostics format (stable, locale-independent)
        # e.g. "Check.java:5:9: compiler.err.cant.resolve.location: variable: x"
        # Note: trailing colon is optional when diagnostic has no arguments
        # e.g. "Check.java:3:5: compiler.err.missing.ret.stmt"
        diag_pattern = re.compile(r":(\d+):(\d+):\s+([\w.]+)(?::\s*(.*))?$")
        # Pattern 2: Legacy format (locale-dependent, fallback)
        # e.g. "Check.java:5: error: cannot find symbol"
        legacy_pattern = re.compile(r":(\d+)(?::(\d+))?:\s+(?:error|warning):\s+(.+)")

        for line in output.split("\n"):
            m = diag_pattern.search(line)
            if m:
                # If diagnostic has no arguments, group(4) is None — use the key itself as message
                message = m.group(4).strip() if m.group(4) else m.group(3)
                errors.append(VerifyError(
                    message=message,
                    line=int(m.group(1)),
                    column=int(m.group(2)),
                    code=m.group(3),  # e.g. "compiler.err.cant.resolve.location"
                ))
                continue

            m = legacy_pattern.search(line)
            if m:
                errors.append(VerifyError(
                    message=m.group(3).strip(),
                    line=int(m.group(1)),
                    column=int(m.group(2) or 0),
                    code="ERROR",  # Generic code for legacy format
                ))
            elif "error:" in line.lower() and not errors:
                errors.append(VerifyError(message=line.strip(), code="ERROR"))
        return errors


# ── Kotlin ────────────────────────────────────────────────────────────

class KotlinVerifier(BaseVerifier):
    """Verifies Kotlin code via kotlinc."""

    def verify(self, code: str, file_path: str = "check.kt") -> tuple[bool, list[VerifyError]]:
        tmp_dir = None
        try:
            tmp_dir = tempfile.mkdtemp()
            src_path = os.path.join(tmp_dir, os.path.basename(file_path))
            with open(src_path, "w", encoding="utf-8") as f:
                f.write(code)

            # kotlinc is locale-independent (always outputs English diagnostics),
            # so -J-Duser.language=en is a no-op but kept for defensive consistency.
            # LC_ALL=C/LANG=C are also no-ops for kotlinc but harmless.
            env = os.environ.copy()
            env["LC_ALL"] = "C"
            env["LANG"] = "C"

            proc = subprocess.run(
                ["kotlinc", "-J-Duser.language=en", src_path],
                capture_output=True, text=True, timeout=30,
                cwd=tmp_dir,
                env=env,
            )
            if proc.returncode == 0:
                return True, []

            errors = self._parse_kotlinc_output(proc.stdout + proc.stderr)
            return False, errors

        except FileNotFoundError:
            logger.debug("kotlinc not found, skipping verification")
            return True, []
        except subprocess.TimeoutExpired:
            return True, []
        except Exception as e:
            logger.debug("kotlinc check failed: %s", e)
            return True, []
        finally:
            if tmp_dir:
                import shutil
                shutil.rmtree(tmp_dir, ignore_errors=True)

    def _parse_kotlinc_output(self, output: str) -> list[VerifyError]:
        """Parse kotlinc error output into VerifyError list."""
        import re
        errors: list[VerifyError] = []
        # Pattern: file:line:col: error|warning: message
        pattern = re.compile(r":(\d+):(\d+):\s+(error|warning):\s+(.+)")
        for line in output.split("\n"):
            m = pattern.search(line)
            if m:
                errors.append(VerifyError(
                    message=m.group(4).strip(),
                    line=int(m.group(1)),
                    column=int(m.group(2)),
                    code=m.group(3).upper(),
                ))
            elif "error:" in line.lower() and not errors:
                errors.append(VerifyError(message=line.strip()))
        return errors


# ── Go ────────────────────────────────────────────────────────────────

class GoVerifier(BaseVerifier):
    """Verifies Go code via go build."""

    def verify(self, code: str, file_path: str = "check.go") -> tuple[bool, list[VerifyError]]:
        tmp_dir = None
        try:
            tmp_dir = tempfile.mkdtemp()
            src_path = os.path.join(tmp_dir, os.path.basename(file_path))
            with open(src_path, "w", encoding="utf-8") as f:
                f.write(code)

            # Force English locale for stable error messages
            env = os.environ.copy()
            env["LC_ALL"] = "C"
            env["LANG"] = "C"

            proc = subprocess.run(
                ["go", "build", "-o", os.devnull, src_path],
                capture_output=True, text=True, timeout=30,
                cwd=tmp_dir,
                env=env,
            )
            if proc.returncode == 0:
                return True, []

            errors = self._parse_go_output(proc.stdout + proc.stderr)
            return False, errors

        except FileNotFoundError:
            logger.debug("go not found, skipping verification")
            return True, []
        except subprocess.TimeoutExpired:
            return True, []
        except Exception as e:
            logger.debug("go build check failed: %s", e)
            return True, []
        finally:
            if tmp_dir:
                import shutil
                shutil.rmtree(tmp_dir, ignore_errors=True)

    def _parse_go_output(self, output: str) -> list[VerifyError]:
        """Parse go build error output into VerifyError list."""
        import re
        errors: list[VerifyError] = []
        # Pattern: file:line:col: message
        pattern = re.compile(r":(\d+):(\d+):\s+(.+)")
        for line in output.split("\n"):
            m = pattern.search(line)
            if m:
                errors.append(VerifyError(
                    message=m.group(3).strip(),
                    line=int(m.group(1)),
                    column=int(m.group(2)),
                ))
        return errors


# ── Factory ───────────────────────────────────────────────────────────

def create_verifier(language: str, **kwargs) -> BaseVerifier:
    """Factory: create the appropriate verifier for *language*."""
    # Lazy import to avoid circular dependency at module level
    _VERIFIERS = {
        "python": PythonVerifier,
        "java": JavaVerifier,
        "kotlin": KotlinVerifier,
        "go": GoVerifier,
    }
    cls = _VERIFIERS.get(language)
    if cls is not None:
        return cls(**kwargs)
    # TS/JS: use TSVerifier from ts_vm shim
    if language in ("typescript", "javascript"):
        from external_llm.editor._editor_core.ts_vm.execution_vm.verifier import TSVerifier
        return TSVerifier(language=language, **kwargs)
    raise ValueError(f"No verifier for language: {language}")
