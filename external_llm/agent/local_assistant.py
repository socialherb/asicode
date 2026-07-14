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
import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Optional

from .config.thresholds import config as _cfg

logger = logging.getLogger(__name__)


def _extract_fenced_blocks(text: str) -> list[str]:
    """Extract content from fenced code blocks (`` ```...``` ``) via string split."""
    parts = text.split('```')
    results: list[str] = []
    for i in range(1, len(parts), 2):
        block = parts[i]
        # Skip optional language tag (first line)
        nl = block.find('\n')
        if nl >= 0:
            results.append(block[nl + 1:])
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
        "Generate boilerplate code:\n\n"
        "{instruction}\n\n"
        "{constraints}\n\n"
        "Output ONLY code. No markdown. No explanation."
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

# Prompt sent to the main LLM to decide what to delegate
_DELEGATION_DECISION_PROMPT = (
    "You are analyzing a coding task to identify subtasks that can be delegated to a "
    "fast local code generation model (Qwen 7B).\n\n"
    "The local model is GOOD at:\n"
    "- Generating function bodies from signatures\n"
    "- Writing unit test skeletons\n"
    "- Creating boilerplate (imports, class scaffolds, configs)\n"
    "- Adding docstrings and comments\n"
    "- Simple code transformations (rename, type hints)\n"
    "- Fill-in-middle code completion\n\n"
    "The local model is BAD at:\n"
    "- Multi-file changes\n"
    "- Complex refactoring\n"
    "- Generating unified diffs directly\n"
    "- Cross-module understanding\n"
    "- Architectural decisions\n\n"
    "Task: {request}\n\n"
    "Current file context:\n{file_context}\n\n"
    "Respond with ONLY a JSON object (no markdown):\n"
    "{{\n"
    "  \"delegatable\": true/false,\n"
    "  \"subtasks\": [\n"
    "    {{\n"
    "      \"role\": \"code_snippet|test_skeleton|boilerplate|docstring|transform|fim\",\n"
    "      \"instruction\": \"what to generate\",\n"
    "      \"function_signature\": \"def foo(x: int) -> str:\",\n"
    "      \"file_path\": \"path/to/file.py\",\n"
    "      \"constraints\": \"style notes, max lines, etc.\"\n"
    "    }}\n"
    "  ],\n"
    "  \"main_llm_tasks\": [\"tasks that require the main LLM\"],\n"
    "  \"integration_plan\": \"how to combine local outputs into the final change\"\n"
    "}}\n\n"
    "If the task is NOT suitable for delegation, set \"delegatable\": false and empty subtasks."
)

# Prompt sent to main LLM to integrate local model outputs into a patch
_INTEGRATION_PROMPT = (
    "Integrate the following generated code snippets into a unified patch.\n\n"
    "Original request: {request}\n\n"
    "File context:\n{file_context}\n\n"
    "Generated code from local model:\n{delegation_summary}\n\n"
    "Create a unified diff patch that correctly integrates all the generated code.\n"
    "Include 3 lines of context before and after each change.\n"
    "Output ONLY the unified diff. No explanation."
)


# ── Data Classes ──────────────────────────────────────────────────────────────

@dataclass
class DelegationSpec:
    """A single subtask to be executed by the local model."""
    role: str                    # code_snippet | test_skeleton | boilerplate | docstring | transform | fim
    instruction: str             # Natural language description of what to generate
    function_signature: str = "" # For code_snippet / test_skeleton roles
    context_code: str = ""       # Surrounding code for context
    file_path: str = ""          # Target file (if known)
    language: str = "python"     # python | javascript | typescript
    constraints: str = ""        # Style / format constraints
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
    rejection_reason: str = ""
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
        lines = text.split('\n')
        code_start = 0
        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped:
                continue
            lower = stripped.lower()
            if any(lower.startswith(w) for w in self._PREAMBLE_WORDS):
                code_start = i + 1
                continue
            if stripped.startswith('```'):
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

        return '\n'.join(lines[:code_end]).strip()


# ── Output Validator ──────────────────────────────────────────────────────────

class OutputValidator:
    """
    Validates local model output before handing it to the main LLM.

    Checks:
    1. Syntax (ast.parse for Python, bracket balance for JS/TS)
    2. Role-specific patterns (def test_, triple-quote, etc.)
    3. Hallucination indicators (explanation text in output)
    """

    _HALLUCINATION_PREFIXES = (
        'here is', "here's", 'the following', 'below is', 'note:', 'hope this'
    )

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

        result["overall_ok"] = (
            result["syntax_ok"]
            and result["pattern_match"]
            and len(result["issues"]) == 0
        )
        return result

    # ── language validators ──────────────────────────────────────────────────

    def _validate_python(
        self, output: str, spec: DelegationSpec, result: dict[str, Any]
    ) -> dict[str, Any]:
        try:
            ast.parse(output)
        except SyntaxError:
            # For function bodies, try wrapping in a dummy function
            if spec.role in ("code_snippet", "fim"):
                wrapped = "def _tmp():\n" + "\n".join(
                    f"    {line}" for line in output.split("\n")
                )
                try:
                    ast.parse(wrapped)
                except SyntaxError as e:
                    result["syntax_ok"] = False
                    result["issues"].append(f"Python syntax error: {e}")
            else:
                try:
                    ast.parse(output)
                except SyntaxError as e:
                    result["syntax_ok"] = False
                    result["issues"].append(f"Python syntax error: {e}")
        return result

    def _validate_js(self, output: str, result: dict[str, Any]) -> dict[str, Any]:
        """Basic bracket balance check for JS/TS (string-aware)."""
        stack: list[str] = []
        pairs = {'(': ')', '[': ']', '{': '}'}
        in_str = False
        str_char = ''
        for i, ch in enumerate(output):
            if in_str:
                if ch == str_char and (i == 0 or output[i - 1] != '\\'):
                    in_str = False
                continue
            if ch in ('"', "'", '`'):
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

    def _validate_role(
        self, output: str, spec: DelegationSpec, result: dict[str, Any]
    ) -> dict[str, Any]:
        if spec.role == "test_skeleton":
            if 'def test_' not in output and "it(" not in output:
                result["pattern_match"] = False
                result["issues"].append("Test skeleton missing test function (def test_)")
        elif spec.role == "docstring":
            stripped = output.strip()
            if not (stripped.startswith('"""') or stripped.startswith("'''")):
                result["pattern_match"] = False
                result["issues"].append("Docstring must start with triple quotes")
        return result

    def _check_hallucination(
        self, output: str, result: dict[str, Any]
    ) -> dict[str, Any]:
        first_line = output.strip().split('\n')[0]
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

    Full-flow entry point (used when HELPER lane existed; kept for backward compat):
      execute() — plan delegation, run helper model, integrate, apply patch.
      Falls back to standard AgentLoop on any failure.

    The Planner (main LLM) is ALWAYS in control of what gets delegated.
    """

    def __init__(
        self,
        planner_llm_client: Any,
        planner_model: str,
        local_model: str,
        repo_root: str,
        callback: Optional[Callable[[str, dict[str, Any]], None]] = None,
        ollama_base_url: str = "http://127.0.0.1:11434",
        max_local_calls: int = 5,
    ):
        self._planner_client = planner_llm_client
        self._planner_model = planner_model
        self._local_model = local_model
        self._repo_root = repo_root
        self._cb = callback or (lambda e, d: None)
        self._ollama_base_url = ollama_base_url
        self._max_local_calls = max_local_calls
        self._validator = OutputValidator()
        self._cleaner = OutputCleaner()

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

    def execute(self, request: str, route_decision: Any, config: Any) -> Any:
        """
        Execute the local assistant flow.

        Returns AgentResult (compatible with standard AgentLoop output).
        On any failure, falls back transparently to AgentLoop.
        """
        from external_llm.agent.agent_loop import AgentResult

        start = time.monotonic()

        self._cb("local_assistant_start", {
            "request": request[:200],
            "local_model": self._local_model,
            "task_kind": str(getattr(route_decision, 'task_kind', '')),
        })

        try:
            # Step 1: gather file context
            file_context = self._gather_context(request, config)

            # Step 2: ask main LLM what to delegate
            delegation_specs = self._plan_delegation(request, file_context)

            if not delegation_specs:
                self._cb("local_assistant_fallback", {
                    "reason": "No delegatable subtasks identified by main LLM",
                })
                return self._fallback_to_main_agent(request, config)

            self._cb("local_assistant_plan", {
                "subtask_count": len(delegation_specs),
                "roles": [s.role for s in delegation_specs],
            })

            # Step 3: execute each delegation on local model
            delegations: list[DelegationResult] = []
            for i, spec in enumerate(delegation_specs[: self._max_local_calls]):
                self._cb("local_delegation_start", {
                    "index": i,
                    "role": spec.role,
                    "instruction": spec.instruction[:100],
                })
                dr = self._execute_delegation(spec)
                delegations.append(dr)
                self._cb("local_delegation_complete", {
                    "index": i,
                    "role": spec.role,
                    "accepted": dr.validation.get("overall_ok", False),
                    "issues": dr.validation.get("issues", []),
                    "execution_time": round(dr.execution_time, 2),
                })

            # Step 4: filter to valid outputs
            valid = [d for d in delegations if d.validation.get("overall_ok", False)]

            if not valid:
                self._cb("local_assistant_fallback", {
                    "reason": "All local model outputs failed validation",
                    "issues": [d.validation.get("issues", []) for d in delegations],
                })
                return self._fallback_to_main_agent(request, config)

            # Step 5: ask main LLM to integrate outputs into patch
            final_patch = self._integrate_outputs(request, file_context, valid)

            if not final_patch:
                self._cb("local_assistant_fallback", {
                    "reason": "Main LLM integration returned no valid patch",
                })
                return self._fallback_to_main_agent(request, config)

            # Step 6: apply patch
            from external_llm.agent.tool_registry import ToolRegistry
            registry = ToolRegistry(self._repo_root, config)
            patch_result = registry.dispatch("apply_patch", {"patch": final_patch})

            elapsed = time.monotonic() - start

            if patch_result.ok:
                self._cb("local_assistant_complete", {
                    "status": "success",
                    "delegations": len(delegations),
                    "valid": len(valid),
                    "execution_time": round(elapsed, 2),
                })
                return AgentResult(
                    status="success",
                    turns=[],
                    final_message=(
                        f"Local assistant completed. "
                        f"{len(valid)}/{len(delegations)} delegations applied."
                    ),
                    applied_patches=registry.applied_patches,
                    metadata={
                        "local_assistant": True,
                        "local_model": self._local_model,
                        "delegations": len(delegations),
                        "valid_delegations": len(valid),
                        "execution_time": elapsed,
                    },
                )
            else:
                self._cb("local_assistant_fallback", {
                    "reason": f"Patch application failed: {patch_result.error}",
                })
                return self._fallback_to_main_agent(request, config)

        except Exception as exc:
            logger.exception("LocalAssistant.execute() failed: %s", exc)
            self._cb("local_assistant_error", {"error": str(exc)})
            return self._fallback_to_main_agent(request, config)

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

        # Create delegation spec — budget enforced at spec boundary for local model
        spec = DelegationSpec(
            role=role,
            instruction=instruction,
            function_signature=function_signature,
            context_code=context_code[:_cfg.tokens.LOCAL_MODEL_CONTEXT_CHARS],
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

    def _gather_context(self, request: str, config: Any) -> str:
        """Read file context mentioned in the request via direct file I/O."""
        from pathlib import Path

        _known_exts = {'py', 'js', 'ts', 'tsx', 'jsx', 'html', 'css', 'json', 'yaml', 'yml'}
        _delims = ' :()[]{}<>"\'\t\n\r'
        file_patterns = []
        _buf = []
        for _ch in request:
            if _ch in _delims:
                if _buf:
                    _w = ''.join(_buf).strip('.,;!')
                    if '.' in _w:
                        _parts = _w.rsplit('.', 1)
                        if len(_parts) == 2 and _parts[1].lower() in _known_exts:
                            file_patterns.append(_w)
                    _buf = []
            else:
                _buf.append(_ch)
        if _buf:
            _w = ''.join(_buf).strip('.,;!')
            if '.' in _w:
                _parts = _w.rsplit('.', 1)
                if len(_parts) == 2 and _parts[1].lower() in _known_exts:
                    file_patterns.append(_w)
        parts: list[str] = []
        for path in list(dict.fromkeys(file_patterns))[:3]:  # dedup, max 3
            abs_path = Path(self._repo_root) / path if not Path(path).is_absolute() else Path(path)
            if abs_path.is_file():
                try:
                    content = abs_path.read_text(encoding="utf-8", errors="replace")
                    lines = content.splitlines()
                    part_content = "\n".join(f"{i+1}: {line}" for i, line in enumerate(lines[:200]))
                    parts.append(f"=== {path} ===\n[File: {path} | Lines: 1-{min(len(lines), 200)} of {len(lines)}]\n{part_content}")
                except Exception:
                    pass

        if not parts:
            # fallback: search for key terms
            import subprocess
            words = request.split()[:5]
            try:
                proc = subprocess.run(
                    ["rg", "--no-heading", "--line-number", "-C", "2", "--", " ".join(words), str(self._repo_root)],
                    capture_output=True, text=True, timeout=5,
                )
                if proc.returncode == 0 and proc.stdout:
                    parts.append(f"=== Search Results ===\n{proc.stdout}")
            except Exception:
                pass

        return "\n\n".join(parts) if parts else "(no file context available)"

    def _plan_delegation(self, request: str, file_context: str) -> list[DelegationSpec]:
        """Ask main LLM to identify which subtasks can be delegated to local model."""
        from external_llm.client import LLMMessage, effective_content

        prompt = _DELEGATION_DECISION_PROMPT.format(
            request=request,
            file_context=file_context,
        )
        try:
            response = self._planner_client.chat(
                messages=[
                    LLMMessage(role="system", content="You are a task decomposition expert."),
                    LLMMessage(role="user", content=prompt),
                ],
                model=self._planner_model,
                temperature=0.0,
                max_tokens=_cfg.tokens.LOCAL_ASSISTANT_DEFAULT,
            )
            text = effective_content(response).strip()
            start = text.find('{')
            end = text.rfind('}')
            if start == -1 or end == -1:
                return []

            data = json.loads(text[start:end + 1])
            if not data.get("delegatable", False):
                return []

            specs: list[DelegationSpec] = []
            for sub in data.get("subtasks", []):
                specs.append(DelegationSpec(
                    role=sub.get("role", "code_snippet"),
                    instruction=sub.get("instruction", ""),
                    function_signature=sub.get("function_signature", ""),
                    context_code=file_context[:_cfg.tokens.LOCAL_MODEL_CONTEXT_CHARS],
                    file_path=sub.get("file_path", ""),
                    constraints=sub.get("constraints", ""),
                ))
            return specs

        except Exception as exc:
            logger.warning("Delegation planning failed: %s", exc)
            return []

    def _execute_delegation(self, spec: DelegationSpec) -> DelegationResult:
        """Execute a single delegation on the local Ollama model."""
        from external_llm.client import LLMMessage, effective_content

        t0 = time.monotonic()

        if self._local_client is None:
            return DelegationResult(
                spec=spec,
                validation={"overall_ok": False, "issues": ["OllamaClient not available"]},
                accepted=False,
                rejection_reason="OllamaClient not available",
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
                rejection_reason=str(exc),
                execution_time=elapsed,
            )

    def _integrate_outputs(
        self,
        request: str,
        file_context: str,
        valid_delegations: list[DelegationResult],
    ) -> Optional[str]:
        """Ask main LLM to integrate local model outputs into a unified diff patch."""
        from external_llm.client import LLMMessage, effective_content

        summary_parts = []
        for i, dr in enumerate(valid_delegations):
            summary_parts.append(
                f"### Subtask {i + 1}: {dr.spec.role}\n"
                f"File: {dr.spec.file_path}\n"
                f"Generated code:\n```\n{dr.cleaned_output}\n```"
            )
        delegation_summary = "\n\n".join(summary_parts)

        prompt = _INTEGRATION_PROMPT.format(
            request=request,
            file_context=file_context,
            delegation_summary=delegation_summary,
        )

        try:
            response = self._planner_client.chat(
                messages=[
                    LLMMessage(
                        role="system",
                        content="You are an expert at creating unified diff patches.",
                    ),
                    LLMMessage(role="user", content=prompt),
                ],
                model=self._planner_model,
                temperature=0.0,
                max_tokens=_cfg.tokens.SUBAGENT_SHORT,
            )
            text = effective_content(response).strip()

            # Try extracting from markdown fence
            if "```" in text:
                fenced = _extract_fenced_blocks(text)
                if fenced:
                    return fenced[0].strip()

            # Accept raw diff
            if text.startswith("---") or text.startswith("diff "):
                return text

            return None

        except Exception as exc:
            logger.warning("Integration prompt failed: %s", exc)
            return None

    def _fallback_to_main_agent(self, request: str, config: Any) -> Any:
        """Transparently fall back to the standard AgentLoop."""
        from external_llm.agent.agent_loop import AgentLoop
        from external_llm.agent.tool_registry import ToolRegistry

        self._cb("local_assistant_fallback_agent_start", {
            "reason": "Falling back to standard AgentLoop",
        })

        registry = ToolRegistry(self._repo_root, config)
        loop = AgentLoop(
            llm_client=self._planner_client,
            registry=registry,
            config=config,
            model=self._planner_model,
        )
        return loop.run(request)
