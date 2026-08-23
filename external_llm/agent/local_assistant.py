"""
Helper Backend for asicode — LocalAssistant

This module provides the backend for the delegate_to_helper tool.
It is NOT an independent execution lane or mini-agent.
The Developer (AgentLoop) calls delegate_to_helper when it wants to offload
isolated code generation to a subordinate helper model.

Architecture:
  - The Developer (main LLM) remains in control at all times.
  - The Helper model (any model: API or Ollama) generates code for isolated subtasks.
  - The system validates helper output (syntax, patterns).
  - The Developer reviews and integrates helper output via write_plan or apply_patch.

Supported helper roles:
  - code_snippet : function body from signature + docstring
  - test_skeleton: pytest test stubs for given function
  - boilerplate  : imports, class scaffolds, config files
  - docstring    : triple-quoted docstrings
  - transform    : simple code transformations (rename, type hints)
  - fim          : Fill-in-Middle code completion

Helper is ON/OFF controlled by AgentConfig.helper_enabled.
All failures surface as ToolResult errors — the Developer decides how to recover.
"""

from __future__ import annotations

import ast
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .config.thresholds import config as _cfg

logger = logging.getLogger(__name__)


def _extract_fenced_blocks(text: str) -> list[str]:
    """Extract content from fenced code blocks (`` ```...``` ``) via string split."""
    parts = text.split("```")
    results: list[str] = []
    for i in range(1, len(parts), 2):
        block = parts[i]
        # Skip optional language tag (first line)
        nl = block.find("\n")
        if nl >= 0:
            results.append(block[nl + 1 :])
        else:
            results.append(block)
    return results


# ── Prompt Templates ──────────────────────────────────────────────────────────

_LOCAL_PROMPTS: dict[str, str] = {
    "code_snippet": (
        "Generate ONLY the function body for:\n\n"
        "{function_signature}\n\n"
        "Context:\n{context_code}\n\n"
        "{constraints}\n\n"
        "Output ONLY code. No markdown. No explanation."
    ),
    "test_skeleton": (
        "Generate a pytest unit test for this function:\n\n"
        "{function_signature}\n\n"
        "Context:\n{context_code}\n\n"
        "Use pytest style. Output ONLY the test code. No markdown. No explanation.\n"
        "{constraints}"
    ),
    "boilerplate": (
        "Generate boilerplate code:\n\n{instruction}\n\n{constraints}\n\nOutput ONLY code. No markdown. No explanation."
    ),
    "docstring": (
        "Generate a docstring for:\n\n"
        "{function_signature}\n\n"
        "Context:\n{context_code}\n\n"
        "Output ONLY the docstring (with triple quotes). No other text."
    ),
    "transform": (
        "Transform this code:\n\n"
        "{context_code}\n\n"
        "Transformation:\n{instruction}\n\n"
        "{constraints}\n\n"
        "Output ONLY the transformed code. No markdown. No explanation."
    ),
    "fim": (
        "Complete the code between PREFIX and SUFFIX:\n\n"
        "PREFIX:\n{context_code}\n\n"
        "SUFFIX:\n{constraints}\n\n"
        "Output ONLY the missing code. No markdown. No explanation."
    ),
}


# ── Data Classes ──────────────────────────────────────────────────────────────


@dataclass
class DelegationSpec:
    """A single subtask to be executed by the local model."""

    role: str  # code_snippet | test_skeleton | boilerplate | docstring | transform | fim
    instruction: str  # Natural language description of what to generate
    function_signature: str = ""  # For code_snippet / test_skeleton roles
    context_code: str = ""  # Surrounding code for context
    file_path: str = ""  # Target file (if known)
    language: str = "python"  # python | javascript | typescript
    constraints: str = ""  # Style / format constraints
    max_tokens: int = _cfg.tokens.LOCAL_ASSISTANT_SHORT


@dataclass
class DelegationResult:
    """Result of a single delegation to the local model."""

    spec: DelegationSpec
    raw_output: str = ""
    cleaned_output: str = ""
    validation: dict[str, Any] = field(default_factory=dict)
    # validation keys: syntax_ok, pattern_match, issues (list), overall_ok
    accepted: bool = False
    execution_time: float = 0.0


# ── Output Cleaner ────────────────────────────────────────────────────────────


class OutputCleaner:
    """Strip markdown fences, preamble text, and noise from model output."""

    _PREAMBLE_WORDS = ("here is", "here's", "below is", "the following", "this is")
    _POSTAMBLE_WORDS = ("note:", "this code", "the above", "remember", "hope this")

    def clean(self, raw: str) -> str:
        text = raw.strip()

        # 1. Extract from markdown fence if present
        fenced = _extract_fenced_blocks(text)
        if fenced:
            return fenced[0].strip()

        # 2. Strip leading explanation lines
        lines = text.split("\n")
        code_start = 0
        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped:
                continue
            lower = stripped.lower()
            if any(lower.startswith(w) for w in self._PREAMBLE_WORDS):
                code_start = i + 1
                continue
            if stripped.startswith("```"):
                code_start = i + 1
                continue
            # Looks like code — stop skipping
            code_start = i
            break

        lines = lines[code_start:]

        # 3. Strip trailing explanation lines
        code_end = len(lines)
        for i in range(len(lines) - 1, -1, -1):
            stripped = lines[i].strip()
            if not stripped:
                continue
            lower = stripped.lower()
            if any(lower.startswith(w) for w in self._POSTAMBLE_WORDS):
                code_end = i
            else:
                break

        return "\n".join(lines[:code_end]).strip()


# ── Output Validator ──────────────────────────────────────────────────────────


class OutputValidator:
    """
    Validates local model output before handing it to the main LLM.

    Checks:
    1. Syntax (ast.parse for Python, bracket balance for JS/TS)
    2. Role-specific patterns (def test_, triple-quote, etc.)
    3. Hallucination indicators (explanation text in output)
    """

    _HALLUCINATION_PREFIXES = ("here is", "here's", "the following", "below is", "note:", "hope this")

    def validate(self, output: str, spec: DelegationSpec) -> dict[str, Any]:
        result: dict[str, Any] = {
            "syntax_ok": True,
            "pattern_match": True,
            "issues": [],
            "overall_ok": True,
        }

        if not output.strip():
            result["overall_ok"] = False
            result["issues"].append("Empty output")
            return result

        lang = spec.language.lower()
        if lang in ("python", "py"):
            result = self._validate_python(output, spec, result)
        elif lang in ("javascript", "typescript", "js", "ts", "tsx", "jsx"):
            result = self._validate_js(output, result)

        result = self._validate_role(output, spec, result)
        result = self._check_hallucination(output, result)

        result["overall_ok"] = result["syntax_ok"] and result["pattern_match"] and len(result["issues"]) == 0
        return result

    # ── language validators ──────────────────────────────────────────────────

    def _validate_python(self, output: str, spec: DelegationSpec, result: dict[str, Any]) -> dict[str, Any]:
        try:
            ast.parse(output)
        except SyntaxError as e:
            # For function bodies, try wrapping in a dummy function
            if spec.role in ("code_snippet", "fim"):
                wrapped = "def _tmp():\n" + "\n".join(f"    {line}" for line in output.split("\n"))
                try:
                    ast.parse(wrapped)
                except SyntaxError as e:
                    result["syntax_ok"] = False
                    result["issues"].append(f"Python syntax error: {e}")
            else:
                # Reuse the first parse's exception — re-parsing the same
                # output deterministically raises the identical SyntaxError,
                # so the redundant second ast.parse is dropped.
                result["syntax_ok"] = False
                result["issues"].append(f"Python syntax error: {e}")
        return result

    def _validate_js(self, output: str, result: dict[str, Any]) -> dict[str, Any]:
        """Basic bracket balance check for JS/TS (string-aware)."""
        stack: list[str] = []
        pairs = {"(": ")", "[": "]", "{": "}"}
        in_str = False
        str_char = ""
        for i, ch in enumerate(output):
            if in_str:
                if ch == str_char and (i == 0 or output[i - 1] != "\\"):
                    in_str = False
                continue
            if ch in ('"', "'", "`"):
                in_str = True
                str_char = ch
            elif ch in pairs:
                stack.append(pairs[ch])
            elif ch in pairs.values():
                if not stack or stack[-1] != ch:
                    result["syntax_ok"] = False
                    result["issues"].append(f"Unmatched bracket '{ch}' at pos {i}")
                    return result
                stack.pop()
        if stack:
            result["syntax_ok"] = False
            result["issues"].append(f"Unclosed brackets: {''.join(reversed(stack))}")
        return result

    # ── role validators ──────────────────────────────────────────────────────

    def _validate_role(self, output: str, spec: DelegationSpec, result: dict[str, Any]) -> dict[str, Any]:
        if spec.role == "test_skeleton":
            if "def test_" not in output and "it(" not in output:
                result["pattern_match"] = False
                result["issues"].append("Test skeleton missing test function (def test_)")
        elif spec.role == "docstring":
            stripped = output.strip()
            if not (stripped.startswith(('"""', "'''"))):
                result["pattern_match"] = False
                result["issues"].append("Docstring must start with triple quotes")
        return result

    def _check_hallucination(self, output: str, result: dict[str, Any]) -> dict[str, Any]:
        first_line = output.strip().split("\n")[0]
        if first_line.lower().startswith(self._HALLUCINATION_PREFIXES):
            result["issues"].append("Likely explanation preamble detected in output")
        return result


# ── LocalAssistant ────────────────────────────────────────────────────────────


class LocalAssistant:
    """
    Helper backend: executes isolated code generation subtasks on behalf of the Developer.

    This class is the backend for the delegate_to_helper tool.
    It is NOT an independent execution lane or orchestrator.

    Primary entry point:
      delegate_single_task() — called by ToolRegistry._tool_delegate_to_helper()

    The Planner (main LLM) is ALWAYS in control of what gets delegated.
    """

    def __init__(
        self,
        local_model: str,
        repo_root: str,
        callback: Callable[[str, dict[str, Any]], None] | None = None,
        ollama_base_url: str = "http://127.0.0.1:11434",
        max_local_calls: int = 5,
    ):
        self._local_model = local_model
        self._repo_root = repo_root
        self._cb = callback or (lambda e, d: None)
        self._validator = OutputValidator()
        self._cleaner = OutputCleaner()
        self._max_local_calls = max(1, max_local_calls)
        self._delegation_count = 0

        # Create Ollama client for local model calls
        try:
            from external_llm.providers import OllamaClient

            self._local_client: Any = OllamaClient(
                api_key="",
                base_url=ollama_base_url,
                timeout=30,
            )
        except Exception as exc:
            logger.warning("OllamaClient init failed: %s — local calls will error", exc)
            self._local_client = None

    # ── public API ───────────────────────────────────────────────────────────

    def delegate_single_task(
        self,
        role: str,
        instruction: str,
        file_path: str = "",
        function_signature: str = "",
        context_code: str = "",
        constraints: str = "",
        language: str = "python",
        max_tokens: int = _cfg.tokens.LOCAL_ASSISTANT_SHORT,
    ) -> dict[str, Any]:
        """
        Delegate a single coding subtask to the local model.

        Args:
            role: One of "code_snippet", "test_skeleton", "boilerplate",
                  "docstring", "transform", "fim"
            instruction: Natural language description of what to generate
            file_path: Target file path (optional, for context)
            function_signature: Function signature (for code_snippet/test_skeleton)
            context_code: Surrounding code for context
            constraints: Style/format constraints
            language: Programming language (python, javascript, etc.)
            max_tokens: Maximum tokens for generation

        Returns:
            Dict with keys:
                success: bool
                code: str (cleaned generated code)
                raw_output: str (original model output)
                validation: Dict with syntax and pattern validation results
                issues: List of validation issues
                execution_time: float in seconds
        """

        # Enforce the per-session delegation budget (helper_max_calls). The
        # counter lives on the instance, so the limit spans the whole
        # AgentLoop session. The refused call is still counted, so once the
        # limit is hit every subsequent call is refused.
        self._delegation_count += 1
        if self._delegation_count > self._max_local_calls:
            logger.warning(
                "delegate_to_helper limit reached: %d/%d calls",
                self._delegation_count - 1,
                self._max_local_calls,
            )
            return {
                "success": False,
                "code": "",
                "raw_output": "",
                "validation": {"overall_ok": False, "issues": []},
                "issues": [
                    f"Delegation limit reached: max {self._max_local_calls} "
                    "local helper calls per session (helper_max_calls)"
                ],
                "error": (
                    f"Delegation limit reached: max {self._max_local_calls} "
                    "local helper calls per session (helper_max_calls)"
                ),
                "execution_time": 0.0,
                "role": role,
                "file_path": file_path,
            }

        # Create delegation spec — budget enforced at spec boundary for local model
        spec = DelegationSpec(
            role=role,
            instruction=instruction,
            function_signature=function_signature,
            context_code=context_code[: _cfg.tokens.LOCAL_MODEL_CONTEXT_CHARS],
            file_path=file_path,
            language=language,
            constraints=constraints,
            max_tokens=max_tokens,
        )

        # Execute delegation
        result = self._execute_delegation(spec)

        # Build response
        return {
            "success": result.accepted,
            "code": result.cleaned_output,
            "raw_output": result.raw_output,
            "validation": result.validation,
            "issues": result.validation.get("issues", []),
            "execution_time": result.execution_time,
            "role": role,
            "file_path": file_path,
        }

    # ── private helpers ───────────────────────────────────────────────────────

    def _execute_delegation(self, spec: DelegationSpec) -> DelegationResult:
        """Execute a single delegation on the local Ollama model."""
        from external_llm.client import LLMMessage, effective_content

        t0 = time.monotonic()

        if self._local_client is None:
            return DelegationResult(
                spec=spec,
                validation={"overall_ok": False, "issues": ["OllamaClient not available"]},
                accepted=False,
                execution_time=time.monotonic() - t0,
            )

        template = _LOCAL_PROMPTS.get(spec.role, _LOCAL_PROMPTS["code_snippet"])
        prompt = template.format(
            function_signature=spec.function_signature,
            context_code=spec.context_code,
            instruction=spec.instruction,
            constraints=spec.constraints,
        )

        try:
            response = self._local_client.chat(
                messages=[LLMMessage(role="user", content=prompt)],
                model=self._local_model,
                temperature=0.1,
                max_tokens=spec.max_tokens,
            )
            raw = effective_content(response).strip()
            cleaned = self._cleaner.clean(raw)
            validation = self._validator.validate(cleaned, spec)
            elapsed = time.monotonic() - t0

            return DelegationResult(
                spec=spec,
                raw_output=raw,
                cleaned_output=cleaned,
                validation=validation,
                accepted=validation.get("overall_ok", False),
                execution_time=elapsed,
            )
        except Exception as exc:
            elapsed = time.monotonic() - t0
            return DelegationResult(
                spec=spec,
                raw_output="",
                cleaned_output="",
                validation={"overall_ok": False, "issues": [str(exc)]},
                accepted=False,
                execution_time=elapsed,
            )
