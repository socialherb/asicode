"""
External LLM Service for asicode

Orchestrates the complete flow:
1) Build context from project (or accept prebuilt context)
2) Send to external LLM
3) Parse response (diff-first)
4) (auto mode) If not a diff: accept full-file rewrite blocks and synthesize a unified diff
5) Validate unified diff (git apply compatible)
6) Return final patch + metadata
"""
from __future__ import annotations

import contextlib
import io
import logging
import os
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any, Optional

from .agent.config.thresholds import config as _cfg
from .client import DEFAULT_LLM_TIMEOUT, OLLAMA_LLM_TIMEOUT, LLMClientError, LLMMessage, create_llm_client, effective_content
from .code_structure_utils import extract_symbol_name, is_function_def
from .output_parser import parse_file_blocks, parse_llm_output, validate_diff
from common import normalize_rel_path_fast
# context_builder.py in your repo has evolved over time.
# Keep imports resilient across revisions.
try:
    # newer
    from .context_builder import ContextBuilder, enhance_user_request  # type: ignore
except ImportError:
    # fallback (older revisions)
    from .context_builder import EnhancedContextBuilder as ContextBuilder  # type: ignore

    def enhance_user_request(user_request: str, *_, **__) -> str:  # type: ignore
        return str(user_request or "").strip()

# Super context builder (optional; only used when enabled by context_variant)
try:
    from .super_context_builder import SuperContextBuilder  # type: ignore
except ImportError:
    SuperContextBuilder = None  # type: ignore

logger = logging.getLogger(__name__)

# Patch engine for unified patch intelligence
try:
    from .patch_engine import PatchContext, PatchEngine
except ImportError as e:
    logger.warning("PatchEngine not available: %s", e)
    PatchEngine = None  # type: ignore
    PatchContext = None  # type: ignore

# Phase 1 and 2 completed - always use PatchEngine when available


def _asrp_text(s: str, max_chars: int) -> str:
    """
    Small, safe clip helper for UI-provided context snippets.

    - normalize newlines
    - trim surrounding whitespace
    - clip to max_chars (best-effort)
    """
    try:
        t = str(s or "").replace("\r\n", "\n").strip()
    except (TypeError, AttributeError):
        t = ""
    mc = int(max_chars or 0)
    if mc <= 0:
        return ""
    if len(t) <= mc:
        return t
    clipped = t[:mc].rstrip()
    return clipped + " …[CLIPPED]"




# --- Typed context pack parser helpers (replaces ad-hoc regex patterns) ---


def _is_failure_summary_header(line: str) -> bool:
    """Detect 'failure_summary:' section header in CLIP_SESSION_V2 context."""
    stripped = line.strip().lower()
    return stripped.startswith("failure_summary") and stripped.rstrip().endswith(":")


def _is_section_boundary(line: str) -> bool:
    """Detect CLIP_SESSION_V2 section boundaries without regex.
    Matches: ====, ----, ASICODE_*, TIP:, END_CTX_PACK.
    """
    stripped = line.strip()
    if not stripped:
        return False
    fc = stripped[0]
    # Repeating separator lines (====, ----)
    if fc in ("=", "-"):
        return len(stripped) >= 4 and all(c == fc for c in stripped[:4])
    if stripped.startswith("ASICODE_"):
        return True
    if stripped.startswith("TIP:"):
        return True
    if stripped.startswith("END_CTX_PACK"):
        return True
    return False


def _extract_failure_summary_block(txt: str) -> str:
    """Extract content after 'failure_summary:' header until section boundary.
    Returns clipped text (max 900 chars) or empty string.
    """
    try:
        lines = txt.splitlines()
        header_idx = None
        for i, line in enumerate(lines):
            if _is_failure_summary_header(line):
                header_idx = i
                break
        if header_idx is None:
            return ""

        picked: list[str] = []
        for line in lines[header_idx + 1:]:
            if _is_section_boundary(line):
                break
            if picked and not line.strip():
                break
            if line.strip():
                picked.append(line.rstrip())
            if len(picked) >= 12:
                break
        hint = "\n".join(picked).strip()
        return _asrp_text(hint, 900) if hint else ""
    except (TypeError, AttributeError):
        return ""


def _extract_failed_reason(txt: str) -> str:
    """Extract 'reason:' text from FAILED status block (fallback parser)."""
    try:
        lines = txt.splitlines()
        reason_lines: list[str] = []
        in_failed_block = False
        for line in lines:
            stripped = line.strip().lower()
            if "status" in stripped and "failed" in stripped:
                in_failed_block = True
                continue
            if in_failed_block:
                if stripped.startswith("reason:"):
                    reason = line.strip()[len("reason:"):].strip()
                    if reason:
                        reason_lines.append(reason)
                    if len(reason_lines) >= 3:
                        break
                elif _is_section_boundary(line):
                    break

        if reason_lines:
            return _asrp_text(f"reason: {reason_lines[0]}", 400)
        return ""
    except (TypeError, AttributeError):
        return ""


def _extract_identifiers(text: str) -> list[str]:
    """Extract identifiers (3+ chars) from text — string ops instead of regex."""
    result: list[str] = []
    buf: list[str] = []
    for ch in text:
        if ch.isalnum() or ch == '_':
            buf.append(ch)
        else:
            if buf:
                word = ''.join(buf)
                if len(word) >= 3 and (word[0].isalpha() or word[0] == '_'):
                    result.append(word)
                buf = []
    if buf:
        word = ''.join(buf)
        if len(word) >= 3 and (word[0].isalpha() or word[0] == '_'):
            result.append(word)
    return result


class ExternalLLMService:
    """
    Service to handle external LLM integration.

    IMPORTANT: This service returns a unified diff (git apply compatible).
    - output_mode="diff": require unified diff output from the model (legacy behavior)
    - output_mode="auto": accept diff OR full-file rewrite blocks (Cursor-like),
                          then synthesize unified diff server-side when possible.
    """

    # Safety caps for auto-mode FILE rewrites (MVP)
    _MAX_FILE_CHARS = 250_000
    _MAX_PATCH_CHARS = 350_000

    # Safety caps for FILE-block retry (git-apply-check failed -> FILE rewrite -> synth diff)
    # NOTE: FILE-block rewrite is powerful; keep it constrained to avoid large, risky rewrites.
    _MAX_FILE_RETRY_FILE_CHARS = 60_000            # only allow FILE retry when the target file is small
    _MAX_FILE_REWRITE_CHANGE_RATIO = 0.45          # reject if rewrite changes "too much" of the file
    _MAX_FILE_REWRITE_CHANGED_LINES = 400          # additional cap based on estimated changed lines


    @staticmethod
    @contextlib.contextmanager
    def _suppress_console_noise():
        buf_out = io.StringIO()
        buf_err = io.StringIO()
        with contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(buf_err):
            yield

    @staticmethod
    def _is_trivial_edit_request(user_request: str) -> bool:
        from .agent.agent_fast_path import _is_trivial_request
        return _is_trivial_request(user_request)

    @staticmethod
    def _extract_literal_needles_from_request(user_request: str) -> list[str]:
        """
        Extract literal snippets from the request that are likely intended to be inserted verbatim.
        Examples:
          - " ... '- Do NOT echo ...' add a line ..."
          - ' ... "SOME STRING" add ... '
        We take quoted strings and also " - <rule line> " style bullets.
        """
        req = str(user_request or "")
        if not req.strip():
            return []

        needles: list[str] = []

        # Quoted strings (single/double) — string ops instead of regex
        def _find_quoted(text: str, quote: str) -> list[str]:
            out: list[str] = []
            pos = 0
            while True:
                i = text.find(quote, pos)
                if i == -1:
                    break
                j = text.find(quote, i + 1)
                if j == -1:
                    break
                content = text[i + 1:j]
                if len(content) >= 6 and '\n' not in content and '\r' not in content:
                    out.append(content.strip())
                pos = j + 1
            return out
        needles.extend(_find_quoted(req, "'"))
        needles.extend(_find_quoted(req, '"'))

        # Bullet-like lines inside the request (common for prompt edits)
        # ex) - Do NOT echo...
        for ln in req.split('\n'):
            s = ln.strip()
            if s.startswith('- ') and len(s) >= 8:  # "- " + at least 6 chars
                needles.append(s)

        # Dedup while preserving order
        seen = set()
        out: list[str] = []
        for s in needles:
            if not s:
                continue
            if s in seen:
                continue
            seen.add(s)
            out.append(s)
        # Prefer longer/more specific needles first
        out.sort(key=len, reverse=True)
        return out

    @classmethod
    def _noop_precheck_for_literal_add(cls, repo_root: str, target_file: Optional[str], user_request: str) -> bool:
        """
        If the request looks like an "add this literal text" request AND the literal text
        already exists in the target file, return True (NOOP).
        """
        rr = str(repo_root or "").strip()
        tf = normalize_rel_path_fast(target_file)
        if not rr or not tf:
            return False

        try:
            p = (Path(rr).resolve() / tf).resolve()
            if not p.exists() or not p.is_file():
                return False
            file_text = p.read_text(encoding="utf-8", errors="replace")
        except (OSError, PermissionError):
            return False

        needles = cls._extract_literal_needles_from_request(user_request)
        if not needles:
            return False

        for needle in needles:
            if needle and (needle in file_text):
                return True
        return False


    def _read_target_file_snippet_best_effort(
        self,
        repo_root: str,
        target_file: str,
    ) -> str:
        """
        Best-effort read of target file content.
        """
        rr = str(repo_root or "").strip()
        tf = normalize_rel_path_fast(str(target_file))
        if not rr or not tf:
            return ""

        try:
            root = Path(rr).resolve()
            p = (root / tf).resolve()
            if not p.exists() or not p.is_file():
                return ""
            txt = p.read_text(encoding="utf-8", errors="replace")
        except (OSError, PermissionError):
            return ""

        if not txt:
            return ""

        return txt if txt.endswith("\n") else txt + "\n"

    @staticmethod
    def _extract_identifier_needles(user_request: str) -> list[str]:
        """
        Pull likely code identifiers from the request so we can send a focused snippet.
        Examples:
          - "In the _looks_like_unified_diff function ..."
          - "FooBar class ..."
        We prefer longer / underscore-heavy identifiers.
        """
        req = str(user_request or "")
        if not req.strip():
            return []

        # Extract identifiers (3+ chars) — string ops instead of regex
        raw = _extract_identifiers(req)
        if not raw:
            return []

        # De-dupe preserving order
        seen = set()
        ids: list[str] = []
        for s in raw:
            if s in seen:
                continue
            seen.add(s)
            ids.append(s)

        # Prefer "more specific" ones first (underscore + length)
        def _score(s: str) -> tuple[int, int]:
            return (s.count("_"), len(s))

        ids.sort(key=_score, reverse=True)
        return ids[:6]

    def _read_target_file_focused_snippet_best_effort(
        self,
        repo_root: str,
        target_file: str,
        *,
        needles: list[str],
        radius_lines: int = 120,
    ) -> str:
        """Focused snippet around the first occurrence of any needle."""
        rr = str(repo_root or "").strip()
        tf = normalize_rel_path_fast(str(target_file))
        if not rr or not tf or not needles:
            return ""

        try:
            root = Path(rr).resolve()
            p = (root / tf).resolve()
            if not p.exists() or not p.is_file():
                return ""
            txt = p.read_text(encoding="utf-8", errors="replace")
        except (OSError, PermissionError):
            return ""

        if not txt:
            return ""

        lines = txt.replace("\r\n", "\n").split("\n")
        hit_idx: Optional[int] = None
        for i, line in enumerate(lines):
            for needle in needles:
                if needle and (needle in line):
                    hit_idx = i
                    break
            if hit_idx is not None:
                break

        if hit_idx is None:
            return ""

        start = max(0, hit_idx - int(radius_lines))
        end = min(len(lines), hit_idx + int(radius_lines) + 1)
        snippet = "\n".join(lines[start:end]).strip()

        if snippet and not snippet.endswith("\n"):
            snippet += "\n"
        return snippet

    # ---------------------------------------------------------------------
    # LLM_CONTEXT (server-built, token-optimized)
    # ---------------------------------------------------------------------

    @staticmethod
    def _git_cmd_best_effort(repo_root: str, args: list[str]) -> str:
        rr = str(repo_root or "").strip()
        if not rr:
            return ""
        try:
            p = subprocess.run(
                ["git", *list(args)],
                cwd=rr,
                text=True,
                capture_output=True,
                timeout=5,
            )
            if p.returncode != 0:
                return ""
            return (p.stdout or "").strip()
        except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
            return ""

    @classmethod
    def _get_git_identity_best_effort(cls, repo_root: str) -> dict[str, str]:
        return {
            "branch": cls._git_cmd_best_effort(repo_root, ["rev-parse", "--abbrev-ref", "HEAD"]),
            "head_commit": cls._git_cmd_best_effort(repo_root, ["rev-parse", "HEAD"]),
        }

    @staticmethod
    def _read_file_text_best_effort(repo_root: str, target_file: str) -> str:
        rr = str(repo_root or "").strip()
        tf = normalize_rel_path_fast(str(target_file))
        if not rr or not tf:
            return ""
        try:
            root = Path(rr).resolve()
            p = (root / tf).resolve()
            if not p.exists() or not p.is_file():
                return ""
            return p.read_text(encoding="utf-8", errors="replace") or ""
        except (OSError, PermissionError):
            return ""

    @staticmethod
    def _classify_failure_hint_best_effort(previous_failure_hint: str) -> str:
        """
        Classify previous failure into a small set of buckets to adapt snippet budgets.
        Returns one of:
          - "block_not_found"
          - "ambiguous"
          - "apply_failed"
          - "path_or_header"
          - "" (unknown/none)
        """
        t = str(previous_failure_hint or "").lower()
        if not t:
            return ""
        if ("block_not_found" in t) or ("could not find block" in t) or ("not found" in t and "block" in t):
            return "block_not_found"
        if ("ambig" in t) or ("needs_disambiguation" in t) or ("match_count" in t) or ("selected_indices_missing" in t):
            return "ambiguous"
        if ("git apply" in t) or ("apply failed" in t) or ("patch failed" in t) or ("hunk" in t and ("failed" in t or "reject" in t)):
            return "apply_failed"
        if ("no such file or directory" in t) or ("path" in t and ("mismatch" in t or "wrong" in t)) or ("diff --git" in t) or ("--- a/" in t and "+++ b/" in t):
            return "path_or_header"
        return ""

    def _extract_previous_failure_hint_best_effort(self, extra_context: Optional[str]) -> str:
        """
        Best-effort extraction of a small "previous failure" hint from UI-provided context (CLIP_SESSION/CTX_PACK).
        This is intentionally tiny to avoid token bloat and to prevent stale context from dominating.

        Uses typed helper functions instead of ad-hoc regex patterns:
        - _is_failure_summary_header() for section header detection
        - _is_section_boundary() for section boundary detection
        - _extract_failure_summary_block() for structured block parsing
        - _extract_failed_reason() for fallback FAILED/reason extraction
        """
        txt = str(extra_context or "").strip()
        if not txt:
            return ""

        # Stage 1: Parse structured CLIP_SESSION_V2 failure_summary block
        hint = _extract_failure_summary_block(txt)
        if hint:
            return hint

        # Stage 2: Fallback — FAILED status + reason extraction
        hint = _extract_failed_reason(txt)
        if hint:
            return hint

        return ""

    def _build_llm_context_v7_best_effort(
        self,
        *,
        repo_root: str,
        target_file: Optional[str],
        user_request: str,
        is_trivial: bool,
        previous_failure_hint: str = "",
        output_mode: Optional[str] = None,
    ) -> tuple[str, dict[str, Any]]:
        """
        Server-built, token-optimized LLM context (NOT the CLIP_SESSION).
        - Always includes authoritative full-file snippet
        """
        rr = str(repo_root or "").strip()
        tgt = normalize_rel_path_fast(target_file) or ""
        meta: dict[str, Any] = {"kind": "LLM_CONTEXT_V7", "source": "server"}

        git_info = self._get_git_identity_best_effort(rr)
        if git_info.get("branch"):
            meta["branch"] = git_info["branch"]
        if git_info.get("head_commit"):
            meta["head_commit"] = git_info["head_commit"]

        snippet = ""

        if rr and tgt:
            meta["target_file_chars"] = len(self._read_file_text_best_effort(rr, tgt))

            fail_kind = self._classify_failure_hint_best_effort(previous_failure_hint)
            meta["snippet_failure_hint"] = fail_kind if fail_kind else None

            if is_trivial:
                needles = self._extract_identifier_needles(user_request)
                radius = 120
                if fail_kind in ("block_not_found", "apply_failed", "ambiguous"):
                    radius = 180

                focused = self._read_target_file_focused_snippet_best_effort(
                    rr,
                    tgt,
                    needles=needles,
                    radius_lines=radius,
                )
                if focused:
                    snippet = focused
                    meta["snippet_kind"] = "focused"
                else:
                    snippet = self._read_target_file_snippet_best_effort(rr, tgt)
                    meta["snippet_kind"] = "head_tail"
            else:
                snippet = self._read_target_file_snippet_best_effort(rr, tgt)
                meta["snippet_kind"] = "head_tail"

        meta["snippet_chars"] = len(snippet or "")

        # Keep as plain text (LLM-friendly), not JSON.
        parts: list[str] = []
        parts.append("ASICODE_LLM_CONTEXT v7")
        parts.append("--------------------------------------------------")
        parts.append(f"REPO_ROOT: {rr or '<EMPTY>'}")
        if git_info.get("branch"):
            parts.append(f"BRANCH: {git_info['branch']}")
        if git_info.get("head_commit"):
            parts.append(f"HEAD_COMMIT: {git_info['head_commit']}")
        parts.append(f"TARGET_FILE: {tgt or '<NONE>'}")
        parts.append("")
        parts.append("CONTRACT:")
        mode = (output_mode or "").strip().lower()
        if mode == "full_file":
            parts.append("- Output ONLY a single FILE block for the TARGET_FILE.")
            parts.append("- FILE block MUST start with: FILE: <file_path>")
            parts.append("- NO prose, NO markdown fences outside the FILE block, NO JSON.")
            parts.append("- MUST touch ONLY the target file path exactly as TARGET_FILE.")
            parts.append("- If already satisfied, output exactly: NOOP")
        elif mode == "diff":
            parts.append("- Output ONLY a valid unified diff (git apply compatible).")
            parts.append("- NO prose, NO markdown fences, NO JSON.")
            parts.append("- MUST touch ONLY the target file path exactly as TARGET_FILE.")
            parts.append("- Diff MUST include: diff --git a/<path> b/<path> and ---/+++ headers.")
            parts.append("- If already satisfied, output exactly: NOOP")
        else:  # auto or unknown/default
            parts.append("- Output MUST be EITHER:")
            parts.append("  (A) a valid unified diff (git apply compatible), OR")
            parts.append("  (B) exactly ONE full-file rewrite block for the target file.")
            parts.append("- Prefer (A). Use (B) ONLY if producing a correct diff is difficult.")
            parts.append("- NO prose, NO markdown fences outside the diff/FILE block, NO JSON.")
            parts.append("- MUST touch ONLY the target file path exactly as TARGET_FILE.")
            parts.append("- If already satisfied, output exactly: NOOP")
        parts.append("")

        pfh = str(previous_failure_hint or "").strip()
        if pfh:
            meta["previous_failure_hint_chars"] = len(pfh)
            parts.append("PREVIOUS_FAILURE_HINT (for avoiding repeated mistakes; may be stale):")
            parts.append(pfh)
            parts.append("")

        if snippet:
            parts.append("TARGET FILE SNIPPET (authoritative):")
            parts.append(f"```python\n{snippet.rstrip()}\n```")
        else:
            parts.append("TARGET FILE SNIPPET (authoritative): <EMPTY>")
        parts.append("")
        text = "\n".join(parts).strip()
        if text and not text.endswith("\n"):
            text += "\n"
        return (text, meta)

    def _build_llm_context_super_best_effort(
        self,
        *,
        repo_root: str,
        target_file: Optional[str],
        user_request: str,
        output_mode: Optional[str] = None,
    ) -> tuple[str, dict[str, Any]]:
        """
        Build a richer "super context" using SuperContextBuilder if available.

        This is optional and best-effort:
          - If SuperContextBuilder isn't present, we fallback to v7.
          - Keep the output as plain text (LLM-friendly).
        """
        rr = str(repo_root or "").strip()
        tgt = normalize_rel_path_fast(target_file) or None
        meta: dict[str, Any] = {"kind": "LLM_CONTEXT_SUPER", "source": "server", "target_file": tgt or ""}

        if not rr:
            return ("", {**meta, "reason": "repo_root_empty"})

        try:
            b = SuperContextBuilder(rr)  # type: ignore[call-arg]
            out = b.build_context(user_request=user_request, target_file=tgt)  # type: ignore[misc]
            text = str(out or "").strip()
            if text and not text.endswith("\n"):
                text += "\n"
            meta["length"] = len(text)
            return (text, meta)
        except Exception as e:
            # Fallback to v7 if super builder fails
            txt, m = self._build_llm_context_v7_best_effort(
                repo_root=rr,
                target_file=tgt,
                user_request=user_request,
                is_trivial=False,
                previous_failure_hint="",
                output_mode=output_mode,
            )
            m = dict(m or {})
            m["variant_fallback_from"] = "super"
            m["super_error"] = f"{type(e).__name__}: {e}"
            return (txt, m)


    def __init__(
        self,
        provider: str,
        api_key: str,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: int = DEFAULT_LLM_TIMEOUT,
    ):
        # Special handling for Ollama: use longer default timeout for model loading.
        # Only override when the caller used the cloud default (not an explicit value).
        provider_stripped = (provider or "").strip()
        if provider_stripped.lower() == "ollama" and timeout == DEFAULT_LLM_TIMEOUT:
            timeout = OLLAMA_LLM_TIMEOUT
            logger.info(f"Using extended timeout for Ollama: {timeout}s")

        self.provider = provider_stripped
        self.model = (model or "").strip() or self._get_default_model(self.provider)

        # LLM client
        self.client = create_llm_client(
            provider=self.provider,
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
        )

        logger.info(
            "Initialized ExternalLLMService: provider=%s, model=%s",
            self.provider,
            self.model,
        )

    # ---------------------------------------------------------------------
    # ── Default model per provider ─────────────────────────────────────────
    # Used as fallback when no model is explicitly provided.
    _PROVIDER_DEFAULT_MODELS: dict[str, str] = {
        "openai": "gpt-4-turbo-preview",
        "anthropic": "claude-sonnet-4-6",
        "google": "gemini-2.0-flash",
        "deepseek": "deepseek-chat",
        "zai": "glm-4-plus",
        "openrouter": "deepseek/deepseek-v4-flash",
        "opencode": "deepseek-v4-flash",
        "ollama": "",  # ollama picks its own default at runtime
    }

    @staticmethod
    def _get_default_model(provider: str) -> str:
        """Return the default model name for a provider, or '' if unknown."""
        return ExternalLLMService._PROVIDER_DEFAULT_MODELS.get(
            (provider or "").strip().lower(), ""
        )

    # Public API
    # ---------------------------------------------------------------------

    def generate_patch(
        self,
        repo_root: str,
        user_request: str,
        target_file: Optional[str] = None,
        extra_context: Optional[str] = None,
        temperature: float = 0.0,
        system_prompt: Optional[str] = None,
        output_mode: str = "diff",  # "diff" | "auto"
        max_tokens: int = _cfg.tokens.SERVICE_DEFAULT,
        context_variant: str = "v7",  # "v7" | "super" | "hybrid"
        progress_callback: Optional[Callable[[str, str, Optional[int], Optional[int]], None]] = None,
    ) -> dict[str, Any]:
        """
        Generate a patch using external LLM.

        Returns dict:
          {
            "success": bool,
            "patch": str,
            "error": str|None,
            "provider": str,
            "model": str,
            "tokens_used": int|None,
            "meta": {...}
          }
        """
        rr = str(repo_root or "").strip()
        if not rr:
            return {
                "success": False,
                "patch": "",
                "error": "repo_root is empty",
                "provider": self.provider,
                "model": self.model,
                "tokens_used": None,
                "meta": ({"reason": "bad_input"}),
            }

        mode = str(output_mode or "diff").strip().lower()
        if mode not in {"diff", "auto", "full_file"}:
            mode = "diff"

        tgt = normalize_rel_path_fast(target_file) or None
        is_trivial = self._is_trivial_edit_request(user_request)

        # Fast NOOP precheck (prevents "duplicate add" diffs like the one you saw):
        # If the user asks to "add X" but X already exists in the target file, return NOOP immediately.
        if tgt and self._noop_precheck_for_literal_add(rr, tgt, user_request):
            return {
                "success": True,
                "patch": "",
                "error": None,
                "provider": self.provider,
                "model": self.model,
                "tokens_used": None,
                "meta": ({
                    "reason": "noop",
                    "mode": mode,
                    "target_file": tgt,
                    "context_meta": {"source": "noop_precheck"},
                    "synth_reason": "noop_precheck",
                    "explanation": "",
                    "retry_used": False,
                    "first_fail_reason": "",
                    "second_fail_reason": "",
                    "noop_trust_level": "high",
                }),
            }

        try:
            # ---- Context: prefer UI-provided CTX pack (already curated), else build best-effort.
            # Token optimization (stage-1):
            # - For trivial edits, drop expensive repo context and use a smaller target snippet.
            if progress_callback:
                progress_callback("building_context", "Building context...", 1, 4)
            is_trivial = self._is_trivial_edit_request(user_request)
            if extra_context and str(extra_context).strip():
                context_text = str(extra_context).strip()
                context_meta = {"source": "provided", "length": len(context_text)}
            elif is_trivial:
                context_text = ""
                context_meta = {"source": "built", "context_dropped": "trivial"}
            else:
                # best-effort builder (kept tolerant)
                context_text, context_meta = self._build_context_best_effort(
                    repo_root=rr,
                    target_file=tgt,
                    user_request=user_request,
                )

            # Ground diff generation with authoritative target-file snippet (reduces "patch does not apply")
            target_snippet = ""
            if tgt:
                if is_trivial:
                    # For trivial requests, prefer a focused snippet around identifiers mentioned in the request.
                    needles = self._extract_identifier_needles(user_request)
                    target_snippet = self._read_target_file_focused_snippet_best_effort(
                        rr,
                        tgt,
                        needles=needles,
                        radius_lines=120,
                        max_chars=6_000,
                    )
                    if not target_snippet:
                        target_snippet = self._read_target_file_snippet_best_effort(
                            rr, tgt,
                        )
                else:
                    target_snippet = self._read_target_file_snippet_best_effort(rr, tgt)

                if target_snippet:
                    context_meta = dict(context_meta or {})
                    context_meta["target_snippet_chars"] = len(target_snippet)

            # System prompt
            if system_prompt is None:
                if mode == "diff":
                    system_prompt = self._build_patch_only_system_prompt()
                elif mode == "full_file" and tgt:
                    system_prompt = self._build_file_block_only_system_prompt(tgt)
                else:
                    system_prompt = self._build_auto_system_prompt()

            # User request enhancement (tolerant to older signatures)
            enhanced_request = enhance_user_request(
                user_request=user_request,
                target_file=tgt,
            )

            # Build a tiny "previous failure" hint from any UI-provided context (CLIP_SESSION/CTX_PACK).
            prev_fail_hint = self._extract_previous_failure_hint_best_effort(extra_context)

            # Build LLM_CONTEXT (server-built) for the external model.
            # - v7: target-file snippet optimized for "git apply" success
            # - super: richer repo context (deps/patterns/tests) for higher-level understanding
            # - hybrid: v7 primary + super as optional addendum
            cv = str(context_variant or "v7").strip().lower()
            if cv not in ("v7", "super", "hybrid"):
                cv = "v7"

            llm_context_text = ""
            llm_context_meta: dict[str, Any] = {"kind": "LLM_CONTEXT", "variant": cv}

            if cv == "super":
                llm_context_text, llm_context_meta = self._build_llm_context_super_best_effort(
                    repo_root=rr,
                    target_file=tgt,
                    user_request=user_request,
                    output_mode=mode,
                )
            else:
                llm_context_text, llm_context_meta = self._build_llm_context_v7_best_effort(
                    repo_root=rr,
                    target_file=tgt,
                    user_request=user_request,
                    is_trivial=is_trivial,
                    previous_failure_hint=prev_fail_hint,
                    output_mode=mode,
                )

            # Compose messages:
            # - Use LLM_CONTEXT as the primary context for the external model.
            # - Keep any UI-provided context as OPTIONAL extra (can be large), so don't prepend it by default.
            extra_user_ctx = str(context_text or "").strip()
            user_payload_parts = [
                llm_context_text.strip(),
                "",
                "---",
                "",
                "REQUEST:",
                str(enhanced_request or "").strip(),
            ]

            # Hybrid mode: add SuperContext as optional appendix (keeps v7 as the hard contract)
            if cv == "hybrid":
                sc_text, sc_meta = self._build_llm_context_super_best_effort(
                    repo_root=rr,
                    target_file=tgt,
                    user_request=user_request,
                )
                # Only append if super builder actually produced super content
                # (not the v7 fallback, which would duplicate v7 already included above)
                is_actual_super = sc_meta.get("kind") == "LLM_CONTEXT_SUPER"
                if sc_text.strip() and is_actual_super:
                    user_payload_parts += [
                        "",
                        "---",
                        "",
                        "SUPER_CONTEXT (optional; may be larger/noisier):",
                        sc_text.strip(),
                    ]
                    llm_context_meta = dict(llm_context_meta or {})
                    llm_context_meta["hybrid_super_meta"] = sc_meta
                    llm_context_meta["hybrid_super_chars"] = len(sc_text)
            if extra_user_ctx:
                user_payload_parts += [
                    "",
                    "---",
                    "",
                    "ADDITIONAL USER-PROVIDED CONTEXT (optional; may be stale/large):",
                    extra_user_ctx,
                ]
            user_payload = "\n".join(user_payload_parts).strip()

            # ---- E2E audit breadcrumbs (prove whether UI-provided context was injected) ----
            import hashlib
            _user_bytes = user_payload.encode("utf-8", errors="replace")
            _extra_bytes = extra_user_ctx.encode("utf-8", errors="replace") if extra_user_ctx else b""
            context_meta = dict(context_meta or {})
            context_meta.update(
                {
                    "prompt_user_payload_bytes": len(_user_bytes),
                    "prompt_user_payload_sha256": hashlib.sha256(_user_bytes).hexdigest(),
                    "extra_user_ctx_bytes": len(_extra_bytes),
                    "extra_user_ctx_sha256": hashlib.sha256(_extra_bytes).hexdigest() if _extra_bytes else None,
                }
            )

            messages = [
                LLMMessage(role="system", content=str(system_prompt or "").strip()),
                LLMMessage(role="user", content=user_payload),
            ]

            # Call LLM (1st attempt)
            if progress_callback:
                progress_callback("sending_to_llm", "Sending request to LLM...", 2, 4)
            resp = self.client.chat(
                messages=messages,
                model=self.model,
                temperature=float(temperature or 0.0),
                max_tokens=int(max_tokens or _cfg.tokens.SERVICE_DEFAULT),
                thinking_mode=bool(getattr(self, "thinking_mode", False)),
                reasoning_effort=getattr(self, "reasoning_effort", None),
                reasoning_callback=getattr(self, "reasoning_callback", None),
            )
            llm_text = effective_content(resp)

            # Token accounting (split): first vs retry, plus total.
            tokens_used_first = getattr(resp, "tokens_used", None)
            tokens_used_retry = None
            tokens_used_total = tokens_used_first

            def _evaluate_llm_text(
                llm_out: str,
                *,
                force_file_block: bool = False,
                allow_salvage: bool = True,
            ) -> tuple[bool, str, str, Optional[str], str]:
                """
                Returns:
                  (ok, patch, explanation, synth_reason, fail_reason)

                fail_reason is one of:
                  - "invalid_diff:<msg>"
                  - "git_apply_check_failed:<msg>"
                  - "empty_patch"
                """
                # Protocol: model may return EXACTLY "NOOP" to indicate "already satisfied".
                if str(llm_out or "").strip().upper() == "NOOP":
                    return (True, "", "", "noop", "")

                synth0: Optional[str] = None

                # Try using PatchEngine if available
                if PatchEngine is not None and tgt:
                    try:
                        engine = PatchEngine(rr)
                        output_mode = "full_file" if force_file_block else "auto"
                        PatchContext(
                            original_request=None,
                            file_content=None,
                            llm_output=llm_out,
                            output_mode=output_mode,
                            metadata={"force_file_block": force_file_block}
                        )
                        result = engine.synthesize_and_apply(llm_out, tgt, output_mode)

                        if result.success:
                            # Extract synth_reason from metadata
                            synth_reason_from_meta = result.metadata.get("synth_reason", "")
                            mode_from_meta = result.metadata.get("mode", "")
                            synth0 = f"patch_engine_{mode_from_meta}"
                            if synth_reason_from_meta:
                                synth0 = f"{synth0}:{synth_reason_from_meta}"

                            # Note: explanation is empty as we don't parse it from LLM output
                            return (True, result.patch_applied or "", "", synth0, "")
                        else:
                            # PatchEngine failed, fall back to legacy logic
                            logger.debug("PatchEngine failed: %s", result.error)
                    except Exception as e:
                        logger.debug("PatchEngine exception, falling back: %s", e)
                        # Continue with legacy logic

                # When forcing FILE-block only (retry), do NOT accept partial/garbled diffs or salvage.
                if force_file_block and tgt:
                    expl0 = ""
                    patch0 = ""
                    engine = PatchEngine(rr)
                    p2, r2 = engine._try_synthesize_diff_from_file_blocks(
                        repo_root=rr,
                        target_file=tgt,
                        llm_text=llm_out,
                    )
                    # Preserve synth reason even when patch is empty (e.g., file_rewrite_too_large)
                    synth0 = r2
                    if p2.strip():
                        patch0 = p2.strip()
                else:
                    with self._suppress_console_noise():
                        parsed0 = self._parse_llm_output_best_effort(llm_out or "")
                        # FORCE: ignore any diff generated by the LLM
                        patch0 = ""
                        expl0 = (parsed0.get("explanation") or "").strip()

                        if tgt:
                            # Import AST rewriter once for entire block
                            try:
                                from external_llm.ast_rewrite import ASTRewriter
                                rewriter = ASTRewriter(rr)
                            except Exception as e:
                                logger.debug("AST rewriter import failed: %s", e)
                                rewriter = None

                            # --- AST rewrite attempt (block-level patching) ---
                            try:
                                if rewriter:
                                    # Attempt symbol-based rewrite first
                                    parsed_blocks = parse_file_blocks(llm_out or "")
                                    if parsed_blocks:
                                        block = parsed_blocks[0]
                                        new_code = block.get("text") or block.get("content") or ""

                                        llm_header = (llm_out or "").strip().splitlines()[0].strip()

                                        if llm_header.startswith("FUNCTION:"):
                                            func_name = llm_header.split("FUNCTION:")[1].strip()

                                            result = rewriter.replace_function(
                                                tgt,
                                                func_name,
                                                new_code
                                            )

                                            patch0 = rewriter.generate_patch(tgt, result)
                                            synth0 = "ast_function"

                                        elif llm_header.startswith("CLASS:"):
                                            class_name = llm_header.split("CLASS:")[1].strip()

                                            result = rewriter.replace_class(
                                                tgt,
                                                class_name,
                                                new_code
                                            )

                                            patch0 = rewriter.generate_patch(tgt, result)
                                            synth0 = "ast_class"

                                        elif llm_header.startswith("METHOD:"):
                                            path = llm_header.split("METHOD:")[1].strip()
                                            class_name, method_name = path.split(".")

                                            result = rewriter.replace_method(
                                                tgt,
                                                class_name,
                                                method_name,
                                                new_code
                                            )

                                            patch0 = rewriter.generate_patch(tgt, result)
                                            synth0 = "ast_method"

                                        elif is_function_def(new_code):
                                            _fname, _ = extract_symbol_name(new_code.strip())
                                            if _fname:
                                                result = rewriter.replace_function(
                                                    tgt,
                                                    _fname,
                                                    new_code
                                                )

                                                patch0 = rewriter.generate_patch(tgt, result)
                                                synth0 = "ast_autodetect"

                            except Exception as e:
                                logger.debug("AST rewrite attempt failed: %s", e)

                            # 🔥 NEW: symbol search fallback
                            if not patch0:
                                try:
                                    from external_llm.agent.symbol_search import SymbolSearcher

                                    searcher = SymbolSearcher(rr)

                                    parsed_blocks = parse_file_blocks(llm_out or "")
                                    if parsed_blocks:
                                        block = parsed_blocks[0]
                                        new_code = block.get("text") or block.get("content") or ""

                                        header = new_code.strip().splitlines()[0].strip()

                                        # Extract symbol name from header
                                        symbol_name, symbol_kind = extract_symbol_name(header)

                                        if symbol_name:
                                            results = searcher.find_symbol(symbol_name, kind=symbol_kind if symbol_kind != "function" else "any")
                                        else:
                                            results = searcher.find_symbol(header)  # Fallback to original behavior

                                        if not results:
                                            sym = searcher.fuzzy_find_symbol(symbol_name or header)
                                            if sym:
                                                results = [sym]

                                        if results:
                                            sym = results[0]

                                            if sym.kind in ("function", "async_function", "method"):
                                                if rewriter:  # Safety check
                                                    result = rewriter.replace_function(
                                                        sym.file,  # was sym.file_path
                                                        sym.name,
                                                        new_code
                                                    )
                                                    patch0 = rewriter.generate_patch(sym.file, result)  # was sym.file_path
                                                    synth0 = "ast_symbol_function"
                                            elif sym.kind == "class":
                                                if rewriter:  # Safety check
                                                    result = rewriter.replace_class(
                                                        sym.file,  # was sym.file_path
                                                        sym.name,
                                                        new_code
                                                    )
                                                    patch0 = rewriter.generate_patch(sym.file, result)  # was sym.file_path
                                                    synth0 = "ast_symbol_class"

                                except Exception as e:
                                    logger.debug("Symbol search fallback failed: %s", e)

                            # semantic patch fallback
                            if not patch0:
                                try:
                                    from external_llm.semantic_patch import SemanticPatchEngine

                                    parsed_blocks = parse_file_blocks(llm_out or "")
                                    if parsed_blocks:
                                        block = parsed_blocks[0]
                                        new_code = block.get("text") or block.get("content") or ""

                                        semantic_engine = SemanticPatchEngine(rr)
                                        sem_result = semantic_engine.apply_semantic_patch(
                                            file_path=tgt,
                                            new_code=new_code,
                                        )

                                        if sem_result:
                                            patch0 = semantic_engine.generate_patch(tgt, sem_result)
                                            if sem_result.kind == "class":
                                                synth0 = "semantic_class"
                                            else:
                                                synth0 = "semantic_function"
                                except Exception as e:
                                    logger.debug("Semantic patch fallback failed: %s", e)

                            # fallback to file-block diff synthesis
                            if not patch0:
                                engine = PatchEngine(rr)
                                p2, r2 = engine._try_synthesize_diff_from_file_blocks(
                                    repo_root=rr,
                                    target_file=tgt,
                                    llm_text=llm_out,
                                )
                                synth0 = r2
                                if p2.strip():
                                    patch0 = p2.strip()


                # Centralized patch normalization pipeline (shared by diff/auto/fast paths)
                patch0 = self._normalize_candidate_patch(patch0, tgt if tgt else None)

                if not patch0:
                    fr = "empty_patch"
                    if force_file_block and synth0:
                        fr = f"file_block_failed:{synth0}"
                    return (False, "", expl0, synth0, fr)

                ok_v, err_v = self._validate_diff_best_effort(
                    patch0,
                    target_file=tgt if (mode == "auto" and tgt) else None,
                )
                if not ok_v:
                    return (False, "", expl0, synth0, f"invalid_diff:{err_v or 'validate_failed'}")

                engine = PatchEngine(rr)
                ok_g, err_g = engine._git_apply_check_best_effort(patch0)

                # 🔧 AST Auto-Repair attempt
                if not ok_g and tgt:
                    try:
                        repair_result = engine.repair_patch(patch0, tgt, "git_apply_failed", llm_out)
                        repaired_patch = repair_result.patch if repair_result.success else None

                        if repaired_patch:
                            ok2, _err2 = engine._git_apply_check_best_effort(repaired_patch)
                            if ok2:
                                patch0 = repaired_patch
                                ok_g = True
                    except Exception as e:
                        logger.debug("Auto-repair in pipeline failed: %s", e)

                if not ok_g:
                    return (False, "", expl0, synth0, f"git_apply_check_failed:{err_g}")

                return (True, patch0, expl0, synth0, "")

            if progress_callback:
                progress_callback("parsing_response", "Parsing LLM response...", 3, 4)
            # For full_file mode, force FILE block only parsing, disable salvage
            force_file = (mode == "full_file")
            allow_salvage = (mode != "full_file")
            ok1, patch, explanation, synth_reason, fail_reason = _evaluate_llm_text(
                llm_text,
                force_file_block=force_file,
                allow_salvage=allow_salvage
            )
            context_meta = dict(context_meta or {})
            # Expose LLM_CONTEXT meta for debugging (token policy / git identity / snippet sizes)
            context_meta["llm_context_meta"] = llm_context_meta
            context_meta["llm_context_chars"] = len(str(llm_context_text or ""))
            # Record output mode used
            context_meta["output_mode_used"] = mode

            # 2nd attempt (targeted): ONLY when the diff is syntactically valid but fails `git apply --check`.
            # In that case, force a single FILE block for the exact target file, then synthesize a diff server-side.
            retry_used = False
            retry_fail_reason = ""
            retry_triggers = ("git_apply_check_failed:", "invalid_diff:")
            if (not ok1) and tgt and any(str(fail_reason or "").startswith(trigger) for trigger in retry_triggers) and self._is_target_file_small_enough_for_file_retry(rr, tgt):
                retry_used = True
                retry_system = self._build_file_block_only_system_prompt(tgt)

                # Reuse the same LLM_CONTEXT (authoritative + token-optimized) for retry.
                # (includes the same previous_failure_hint we extracted above)
                # Keep context_text as optional extra only (can be stale/large).
                extra_user_ctx = str(context_text or "").strip()
                retry_user_parts = [
                    llm_context_text.strip(),
                    "",
                    "---",
                    "",
                    "REQUEST:",
                    str(enhanced_request or "").strip(),
                    "",
                    "---",
                    "",
                    "RETRY MODE: FILE-BLOCK ONLY. Output a single FILE block for TARGET_FILE, then stop.",
                ]
                if extra_user_ctx:
                    retry_user_parts += [
                        "",
                        "---",
                        "",
                        "ADDITIONAL USER-PROVIDED CONTEXT (optional; may be stale/large):",
                        extra_user_ctx,
                    ]
                retry_user_payload = "\n".join(retry_user_parts).strip()

                retry_messages = [
                    LLMMessage(role="system", content=str(retry_system or "").strip()),
                    LLMMessage(role="user", content=retry_user_payload),
                ]

                resp2 = self.client.chat(
                    messages=retry_messages,
                    model=self.model,
                    temperature=float(temperature or 0.0),
                    max_tokens=int(max_tokens or _cfg.tokens.SERVICE_REPAIR),
                    thinking_mode=bool(getattr(self, "thinking_mode", False)),
                    reasoning_effort=getattr(self, "reasoning_effort", None),
                    reasoning_callback=getattr(self, "reasoning_callback", None),
                )
                llm_text2 = effective_content(resp2)

                # Token accounting (retry): keep retry separately; also compute total if possible.
                tokens_used_retry = getattr(resp2, "tokens_used", None)
                if isinstance(tokens_used_first, int) and isinstance(tokens_used_retry, int):
                    tokens_used_total = tokens_used_first + tokens_used_retry
                else:
                    # best-effort fallback: prefer the most recent known value
                    tokens_used_total = tokens_used_retry if tokens_used_retry is not None else tokens_used_first

                # Retry is FILE-block-only: no salvage, no partial diffs. We synthesize + validate + git-apply-check.
                ok2, patch2, explanation2, synth2, fail2 = _evaluate_llm_text(
                    llm_text2,
                    force_file_block=True,
                    allow_salvage=False,
                )
                if ok2:
                    patch = patch2
                    explanation = explanation2
                    synth_reason = synth2 or synth_reason
                    fail_reason = ""
                else:
                    retry_fail_reason = fail2

            if (not patch) and (synth_reason != "noop"):
                # Normalize error shape
                if retry_used and retry_fail_reason:
                    final_fail = retry_fail_reason
                else:
                    final_fail = fail_reason or "empty_patch"

                if final_fail.startswith("invalid_diff:"):
                    err_msg = final_fail.split(":", 1)[1]
                    reason = "invalid_diff"
                    error = f"invalid_diff: {err_msg or 'validate_failed'}"
                elif final_fail.startswith("git_apply_check_failed:"):
                    err_msg = final_fail.split(":", 1)[1]
                    reason = "git_apply_check_failed"
                    error = f"git_apply_check_failed: {err_msg}"
                else:
                    reason = "invalid_diff"
                    error = "invalid_diff: empty_patch"

                return {
                    "success": False,
                    "patch": "",
                    "error": error,
                    "provider": self.provider,
                    "model": self.model,
                    "tokens_used": tokens_used_total,
                    "meta": ({
                        "reason": reason,
                        "mode": mode,
                        "target_file": tgt,
                        "context_meta": context_meta,
                        "synth_reason": synth_reason,
                        "explanation": explanation,
                        "retry_used": retry_used,
                        "first_fail_reason": fail_reason,
                        "second_fail_reason": retry_fail_reason,
                        # token split (debuggable, stable fields)
                        "tokens_used_first": tokens_used_first,
                        "tokens_used_retry": tokens_used_retry,
                        "tokens_used_total": tokens_used_total,
                    }),
                }

            reason = "ok"
            if synth_reason == "noop":
                reason = "noop"
            elif synth_reason == "file_block_synth":
                reason = "ok_synth"
            elif synth_reason == "salvaged":
                reason = "ok_salvaged"

            # -------------------------
            # NOOP trust classification
            # -------------------------
            noop_trust_level = None
            if reason == "noop" or (synth_reason == "noop"):
                # Default: LOW trust (LLM may have avoided change)
                noop_trust_level = "low"

                # If request is strict trivial and we did not need retry,
                # it's more likely a legitimate "already applied".
                if bool(is_trivial) and (not retry_used):
                    noop_trust_level = "medium"

            if progress_callback:
                progress_callback("finalizing", "Finalizing patch...", 4, 4)
            return {
                "success": True,
                "patch": patch,
                "error": None,
                "provider": self.provider,
                "model": self.model,
                "tokens_used": tokens_used_total,
                "meta": ({
                    "reason": reason,
                    "mode": mode,
                    "target_file": tgt,
                    "context_meta": context_meta,
                    "synth_reason": synth_reason,
                    "explanation": explanation,
                    "retry_used": retry_used,
                    "first_fail_reason": fail_reason,
                    "second_fail_reason": retry_fail_reason,
                    # token split (debuggable, stable fields)
                    "tokens_used_first": tokens_used_first,
                    "tokens_used_retry": tokens_used_retry,
                    "tokens_used_total": tokens_used_total,
                    "noop_trust_level": noop_trust_level,
                }),
            }

        except LLMClientError as e:
            logger.error("LLM client error: %s", e)
            return {
                "success": False,
                "patch": "",
                "error": str(e),
                "provider": self.provider,
                "model": self.model,
                "tokens_used": None,
                "meta": ({"reason": "llm_error", "mode": mode, "target_file": tgt}),
            }
        except Exception as e:
            logger.exception("Unexpected error in ExternalLLMService.generate_patch")
            return {
                "success": False,
                "patch": "",
                "error": f"{type(e).__name__}: {e}",
                "provider": self.provider,
                "model": self.model,
                "tokens_used": None,
                "meta": ({"reason": "internal_error", "mode": mode, "target_file": tgt}),
            }

    # ---------------------------------------------------------------------
    # Context building
    # ---------------------------------------------------------------------

    def _build_context_best_effort(
        self,
        repo_root: str,
        target_file: Optional[str],
        user_request: str,
    ) -> tuple[str, dict[str, Any]]:
        """
        Build context using ContextBuilder (with safe fallback).
        """
        rr = str(repo_root).strip()
        meta: dict[str, Any] = {"source": "built", "repo_root": rr, "target_file": target_file or ""}

        try:
            builder = ContextBuilder(rr)
            out = builder.build_context(user_request=user_request, target_file=target_file)
            if isinstance(out, tuple) and len(out) == 2:
                text, m = out
                text = str(text or "")
                if isinstance(m, dict):
                    meta.update(m)
                meta["length"] = len(text)
                return (text, meta)
            if isinstance(out, str):
                text = out
                meta["length"] = len(text)
                return (text, meta)
        except Exception as e:
            # Record builder error for debuggability (mirrors _build_llm_context_super_best_effort)
            meta["build_error"] = f"{type(e).__name__}: {e}"

        # Fallback: no context available
        meta["fallback"] = True
        return ("", meta)

    # ---------------------------------------------------------------------
    # Output parsing / validation
    # ---------------------------------------------------------------------

    @staticmethod
    def _parse_llm_output_best_effort(text: str) -> dict[str, str]:
        """
        Parse LLM output -> dict {"patch": ..., "explanation": ...}

        Wraps parse_llm_output() with safe fallback (empty on error).
        """
        try:
            out = parse_llm_output(text)
        except Exception:
            return {"explanation": "", "patch": ""}

        return {
            "explanation": str(out.get("explanation") or ""),
            "patch": str(out.get("patch") or ""),
        }

    @staticmethod
    def _validate_diff_best_effort(patch: str, target_file: Optional[str] = None) -> tuple[bool, str]:
        """
        validate_diff compatibility:
        - some revisions return (bool, msg)
        - some return bool only
        - newer validate_diff supports target_file filtering
        """
        try:
            out = validate_diff(patch or "", target_file=target_file)
        except TypeError:
            out = validate_diff(patch or "")

        if isinstance(out, tuple) and len(out) == 2:
            ok, msg = out
            return (bool(ok), str(msg or ""))
        return (bool(out), "" if out else "validate_diff_failed")

    @staticmethod
    def _normalize_candidate_patch(patch: str, target_file: Optional[str]) -> str:
        """
        Centralized patch normalization pipeline.
        Apply the same sanitation/repair steps across diff/auto/fast paths.
        """
        if not patch:
            return ""
        # Use PatchEngine normalization
        import os
        engine = PatchEngine(os.getcwd())
        normalized, _error = engine.normalize_and_validate(patch, target_file)
        # We ignore error for now, just return normalized patch
        return normalized




    @staticmethod
    def _build_patch_only_system_prompt() -> str:
        """
        System prompt that forces unified diff output.
        (Standalone restoration — was mistakenly removed as 'dead code' in f7f7312a.)
        """
        hard_rules = (
            "\n\n"
            "CRITICAL OUTPUT RULES:\n"
            "- Output ONLY a valid unified diff (git apply compatible).\n"
            "- Do NOT output markdown fences, JSON, ASICODE blocks, or explanations.\n"
            "- The diff MUST include file headers (--- / +++) and at least one hunk.\n"
            "- Prefer minimal, localized edits.\n"
            "- Do NOT echo or paste the user's request into the code.\n"
            "- If the requested change is ALREADY PRESENT in the target file, output EXACTLY: NOOP\n"
            "- If the request is ambiguous (e.g., 'add a comment'), interpret it as a real code comment\n"
            "  (e.g., a Python line starting with '#') and do NOT modify docstrings unless explicitly asked.\n"
        )
        return ("You are an expert code editor. You produce precise, minimal unified diffs.\n" + hard_rules).strip()

    @staticmethod
    def _build_file_block_only_system_prompt(target_file: str) -> str:
        """
        Strong fallback prompt: force Cursor-like single FILE block output for exactly the target file.
        The server will synthesize a unified diff from this.
        """
        tf = str(target_file or "").strip().lstrip("/").lstrip("./")
        rules = (
            "\n\n"
            "CRITICAL OUTPUT RULES (FILE BLOCK ONLY):\n"
            f"- Output MUST be exactly ONE file rewrite block for: {tf}\n"
            f"- First line MUST be: FILE: {tf}\n"
            "- Then output a fenced code block containing the COMPLETE updated file content.\n"
            "- Output NOTHING ELSE (no prose, no explanations, no extra markdown).\n"
            "- Do NOT mention or modify any other files.\n"
            "- Preserve all existing content except the requested change.\n"
        )
        return ("You are an expert code editor. Output a single file rewrite block.\n" + rules).strip()

    @staticmethod
    def _build_auto_system_prompt() -> str:
        """
        System prompt for Cursor-like behavior (auto mode):

        Goal:
          - Prefer unified diff (git apply compatible)
          - Fallback: full-file rewrite block for the TARGET FILE only

        The server will reject multi-file edits in this mode.
        """
        rules = (
            "\n\n"
            "CRITICAL OUTPUT RULES (AUTO MODE):\n"
            "- Output MUST be EITHER:\n"
            "  (A) a valid unified diff (git apply compatible), OR\n"
            "  (B) exactly ONE full-file rewrite block for the target file.\n"
            "- Prefer (A). Use (B) ONLY if producing a correct diff is difficult.\n"
            "- Do NOT include prose, markdown commentary, JSON, or explanations outside the diff/FILE block.\n"
            "- Do NOT modify or mention any files other than the target file.\n"
            "- Do NOT echo or paste the user's request into the code.\n"
            "- If the requested change is ALREADY PRESENT in the target file, output EXACTLY: NOOP\n"
            "- For 'add a comment' requests, add a real code comment (e.g., '# ...') and avoid docstring edits\n"
            "  unless explicitly requested.\n"
            "\n"
            "FULL FILE REWRITE BLOCK FORMAT (fallback B):\n"
            "FILE: <target_file_relative_path>\n"
            "```<any_language>\n"
            "... complete file content ...\n"
            "```\n"
        )
        return ("You are an expert code editor. Prefer unified diffs; fall back to file rewrite blocks.\n" + rules).strip()

    def _is_target_file_small_enough_for_file_retry(self, repo_root: str, target_file: str) -> bool:
        """
        Gate FILE-block retry to small files only.

        Why:
          FILE retry asks the model to output a complete file. For large files this is:
            - token-expensive
            - truncation-prone
            - higher risk of accidental large rewrite
        """
        try:
            rr = Path(str(repo_root or "")).resolve()
            tgt_rel = normalize_rel_path_fast(str(target_file))
            if not tgt_rel:
                return False
            p = (rr / tgt_rel).resolve()
            if not p.exists() or (not p.is_file()):
                return False
            txt = p.read_text(encoding="utf-8", errors="replace")
            return len(txt) <= int(self._MAX_FILE_RETRY_FILE_CHARS)
        except (OSError, PermissionError):
            return False

def create_service_from_env(
    provider: Optional[str] = None,
    model: Optional[str] = None,
) -> Optional[ExternalLLMService]:
    """
    Create ExternalLLMService from environment variables.

    Environment variables:
    - EXTERNAL_LLM_PROVIDER: Provider name (openai, anthropic, google, deepseek, ollama, zai, openrouter, opencode)
    - EXTERNAL_LLM_MODEL: Model to use (optional)
    - OPENAI_API_KEY / ANTHROPIC_API_KEY / GOOGLE_API_KEY / DEEPSEEK_API_KEY / OLLAMA_API_KEY / ZAI_API_KEY / OPENROUTER_API_KEY
    - EXTERNAL_LLM_BASE_URL: Optional base URL override
    """
    prov = (provider or os.getenv("EXTERNAL_LLM_PROVIDER", "") or "").strip().lower()
    if not prov:
        logger.debug("No external LLM provider configured")
        return None

    api_key_env_vars = {
        "openai": "OPENAI_API_KEY",
        "opencode": "OPENCODE_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "google": "GOOGLE_API_KEY",
        "deepseek": "DEEPSEEK_API_KEY",
        "ollama": "OLLAMA_API_KEY",
        "zai": "ZAI_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
    }
    api_key_var = api_key_env_vars.get(prov)
    if not api_key_var:
        logger.error("Unknown provider: %s", prov)
        return None

    api_key = (os.getenv(api_key_var, "") or "").strip()
    if not api_key:
        # Local providers (ollama) don't require an API key
        if prov != "ollama":
            logger.warning("No API key found for %s (set %s)", prov, api_key_var)
            return None

    m = (model or os.getenv("EXTERNAL_LLM_MODEL", "") or "").strip() or None
    # Provider-scoped base_url: a foreign provider's global base_url must not
    # leak in (see resolve_provider_base_url).
    from .client import resolve_provider_base_url
    base_url = resolve_provider_base_url(prov)

    try:
        svc = ExternalLLMService(
            provider=prov,
            api_key=api_key,
            model=m,
            base_url=base_url,
        )
        logger.info("External LLM service created: %s (%s)", prov, svc.model)
        return svc
    except Exception as e:
        logger.error("Failed to create external LLM service: %s", e)
        return None
