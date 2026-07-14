from __future__ import annotations

import hashlib
import re as _re
from dataclasses import dataclass, field
from typing import Any, Optional

# -----------------------------
# pytest entry-point plugins → pip package
# -----------------------------
# pytest dynamically loads options like --timeout/--cov/--asyncio-mode as entry-point
# plugins. If a package is not installed, pytest core emits an "unrecognized arguments:"
# error. This mapping is only used by the diagnostic layer (failure_context) to
# normalize "which package is missing" — actual install/re-run decisions are handled
# by the recovery layer (bash handler) via ask_user. New plugins just need to be
# added here.
#
# Keys are only the long-name of the option (--timeout, --cov). Even if =VALUE is
# attached (like "--timeout=60"), it is extracted from the "unrecognized arguments:"
# text via --[\w-]+, split on "=", and compared — no special handling needed.
_PYTEST_PLUGIN_OPTIONS: dict[str, str] = {
    "--timeout": "pytest-timeout",
    "--cov": "pytest-cov",
    "--asyncio-mode": "pytest-asyncio",
    "--reruns": "pytest-rerunfailures",
    "--html": "pytest-html",
    "--numprocesses": "pytest-xdist",  # -n auto / -n N also use the same package
    "--flakes": "pytest-flakes",
    "--pep8": "pytest-pep8",
    "--benchmark": "pytest-benchmark",
    "--randomly-seed": "pytest-randomly",
    "--mypy": "pytest-mypy",
    "--django-settings": "pytest-django",
}


def _extract_missing_pytest_plugins(raw: str) -> tuple[list[str], list[str]]:
    """Extract missing pytest plugins from "unrecognized arguments:" error.

    When pytest encounters an option from an uninstalled plugin, it immediately exits
    with:
        ERROR: usage: ... unrecognized arguments: --timeout=60 --cov=foo

    Returns:
        (offending_options, missing_packages)
        - offending_options: list of --xxx[=val] flags listed after "unrecognized arguments:"
        - missing_packages: pip package names mapped from offending_options via
          _PYTEST_PLUGIN_OPTIONS (unmapped options are marked unknown → empty package,
          prompting the caller to remove the option)
    """
    low = raw.lower()
    idx = low.find("unrecognized arguments")
    if idx < 0:
        return [], []
    tail = raw[idx:]
    offending = _re.findall(r"--[A-Za-z][\w-]*", tail)
    missing: list[str] = []
    seen = set()
    for opt in offending:
        pkg = _PYTEST_PLUGIN_OPTIONS.get(opt)
        if pkg and pkg not in seen:
            missing.append(pkg)
            seen.add(pkg)
    return offending, missing


# -----------------------------
# Data structures
# -----------------------------

@dataclass
class TraceFrame:
    file: str
    line: Optional[int] = None
    func: Optional[str] = None
    text: Optional[str] = None


@dataclass
class FailureContext:
    stage: str  # plan|llm_generate|diff_clean|git_apply_check|apply|tests|runtime
    type: str   # normalized failure type (AssertionError, ImportError, GitApplyError, ...)
    message: str

    primary_file: Optional[str] = None
    primary_line: Optional[int] = None
    primary_symbol: Optional[str] = None  # function/class/variable/module
    test_id: Optional[str] = None

    traceback: list[TraceFrame] = field(default_factory=list)
    raw_snippet: str = ""

    # meta signals for strategy engine
    fingerprint: str = ""
    tags: list[str] = field(default_factory=list)  # e.g., ["missing_hunks", "corrupt_patch", "collection_error"]
    details: dict[str, Any] = field(default_factory=dict)


# -----------------------------
# Public API
# -----------------------------

def analyze_failure(*, stage: str, raw_text: str, repo_root: Optional[str] = None) -> FailureContext:
    """
    Parse raw error/log text into FailureContext.

    repo_root:
      - if provided, can help choose "user code frame" by filtering paths.
      - keep optional to avoid tight coupling.
    """
    raw = (raw_text or "").strip()
    if not raw:
        ctx = FailureContext(stage=stage, type="UnknownError", message="empty error text", raw_snippet="")
        ctx.fingerprint = _fingerprint(ctx)
        return ctx

    # 1) Try specialized detectors first (git apply / diff / pytest)
    ctx = _try_parse_git_apply(stage, raw)
    if ctx:
        ctx.fingerprint = _fingerprint(ctx)
        return ctx

    ctx = _try_parse_diff_format(stage, raw)
    if ctx:
        ctx.fingerprint = _fingerprint(ctx)
        return ctx

    ctx = _try_parse_pytest(stage, raw, repo_root=repo_root)
    if ctx:
        ctx.fingerprint = _fingerprint(ctx)
        return ctx

    # 2) Fallback: python traceback parser
    ctx = _try_parse_python_traceback(stage, raw, repo_root=repo_root)
    if ctx:
        ctx.fingerprint = _fingerprint(ctx)
        return ctx

    # 3) Last resort
    ctx = FailureContext(stage=stage, type="UnknownError", message=_first_line(raw), raw_snippet=_snip(raw))
    ctx.fingerprint = _fingerprint(ctx)
    return ctx


# -----------------------------
# Parsers
# -----------------------------

def _try_parse_git_apply(stage: str, raw: str) -> Optional[FailureContext]:
    lower = raw.lower()

    # Common asicode pipeline errors you already use
    if "git_apply_check_failed" in lower or "git apply" in lower or "patch failed" in lower:
        ctx = FailureContext(
            stage=stage or "git_apply_check",
            type="GitApplyError",
            message=_first_line(raw),
            raw_snippet=_snip(raw),
        )
        # Try extract "path:line" (first matching line only)
        _pf_prefix = "patch failed:"
        _idx = lower.find(_pf_prefix)
        if _idx >= 0:
            _line_end = raw.find('\n', _idx)
            if _line_end == -1:
                _line_end = len(raw)
            _line = raw[_idx:_line_end]
            _suffix = _line[len(_pf_prefix):].strip()
            _parts = _suffix.split(":", 1)
            ctx.primary_file = _parts[0].strip()
            if len(_parts) > 1 and _parts[1].strip().isdigit():
                ctx.primary_line = int(_parts[1].strip())
            ctx.tags.append("patch_failed")

        if "corrupt patch" in lower or "corrupt" in lower:
            ctx.tags.append("corrupt_patch")

        return ctx

    return None


def _try_parse_diff_format(stage: str, raw: str) -> Optional[FailureContext]:
    lower = raw.lower()

    if "empty_patch" in lower:
        ctx = FailureContext(stage=stage, type="EmptyPatch", message=_first_line(raw), raw_snippet=_snip(raw))
        ctx.tags.append("empty_patch")
        return ctx

    if "missing_hunks" in lower or "no @@ hunk" in lower:
        ctx = FailureContext(stage=stage, type="InvalidUnifiedDiff", message=_first_line(raw), raw_snippet=_snip(raw))
        ctx.tags.append("missing_hunks")
        return ctx

    if "invalid_diff" in lower or "no diff found" in lower:
        ctx = FailureContext(stage=stage, type="InvalidUnifiedDiff", message=_first_line(raw), raw_snippet=_snip(raw))
        ctx.tags.append("invalid_diff")
        return ctx

    return None


def _try_parse_pytest(stage: str, raw: str, repo_root: Optional[str]) -> Optional[FailureContext]:
    """
    Parse pytest output (including collection/import errors).

    NOTE:
      - Your test expects "ImportError" for collection errors like:
        "ERROR collecting ..." + "ImportError: No module named ..."
      - So we normalize collection/import cases accordingly.
    """
    low = raw.lower()

    # ── Missing pytest plugin option detection (usage error) ────────────────
    # When pytest encounters an uninstalled entry-point plugin option (--timeout, --cov, ...),
    # it prints "ERROR: usage: ... unrecognized arguments: --timeout=60" at the core stage
    # and exits immediately. This error has no FAILED / E / "error collecting" markers,
    # so existing branches can't catch it — it gets buried as UnknownError. Detect it first
    # and normalize as MissingPytestPlugin type.
    if "unrecognized arguments" in low:
        offending, missing = _extract_missing_pytest_plugins(raw)
        ctx = FailureContext(
            stage=stage or "tests",
            type="MissingPytestPlugin" if missing else "PytestUsageError",
            message=_first_line(raw),
            raw_snippet=_snip(raw),
        )
        ctx.tags.append("unrecognized_argument")
        if missing:
            ctx.tags.append("missing_pytest_plugin")
            ctx.details["missing_packages"] = missing
        if offending:
            ctx.details["offending_options"] = offending
        return ctx

    has_core_failure_markers = ("E   " in raw) or ("FAILED" in raw) or ("short test summary info" in low)
    has_collection_markers = ("error collecting" in low) or ("importerror while importing" in low)
    if not (has_core_failure_markers or has_collection_markers):
        return None

    ctx = FailureContext(stage=stage or "tests", type="PytestFailure", message=_first_line(raw), raw_snippet=_snip(raw))

    # Collection error detection
    if ("error collecting" in low) or ("importerror while importing test module" in low) or ("collected 0 items" in low):
        ctx.tags.append("collection_error")
        ctx.type = "PytestCollectionError"

        if ("importerror" in low) or ("modulenotfounderror" in low):
            ctx.type = "ImportError"
            missing = _extract_missing_module(raw)
            if missing:
                ctx.details["missing_module"] = missing

    # Try to extract a FAILED test id if present
    for _line in raw.splitlines():
        if " FAILED" in _line:
            _parts = _line.rsplit(" FAILED", 1)[0].strip()
            if "::" in _parts:
                ctx.test_id = _parts
                break

    return ctx


def _try_parse_python_traceback(stage: str, raw: str, repo_root: Optional[str]) -> Optional[FailureContext]:
    if "Traceback (most recent call last):" not in raw:
        # Also catch SyntaxError blocks that might not have full traceback
        if "SyntaxError" not in raw and "IndentationError" not in raw:
            return None

    # Exception type is usually last line: "TypeError: ..."
    ex_type, ex_msg = _extract_exception_tail(raw)
    if not ex_type:
        ex_type = "RuntimeError"

    ctx = FailureContext(
        stage=stage or "runtime",
        type=_normalize_exception_type(ex_type),
        message=ex_msg or _first_line(raw),
        raw_snippet=_snip(raw),
    )

    if ex_type == "ModuleNotFoundError":
        ctx.type = "ImportError"
        ctx.details["missing_module"] = _extract_missing_module(raw)

    frames = _extract_traceback_frames(raw, repo_root=repo_root, max_frames=8)
    ctx.traceback = frames
    pf = _pick_primary_frame(frames)
    if pf:
        ctx.primary_file = pf.file
        ctx.primary_line = pf.line
        ctx.primary_symbol = pf.func

    return ctx


# -----------------------------
# Utilities
# -----------------------------

def _normalize_exception_type(t: str) -> str:
    t = (t or "").strip()
    if not t:
        return "UnknownError"
    if t == "ModuleNotFoundError":
        return "ImportError"
    return t


def _extract_exception_tail(raw: str) -> tuple[Optional[str], Optional[str]]:
    # typical tail: "TypeError: blah" (also allow pytest's "E   AssertionError: ...")

    lines = [ln.rstrip("\n") for ln in raw.splitlines() if ln.strip()]
    for ln in reversed(lines[-40:]):
        _stripped = ln.strip()
        # Remove pytest "E   " prefix
        if _stripped.startswith("E ") or _stripped.startswith("E\t"):
            _stripped = _stripped[1:].lstrip()
        _colon_idx = _stripped.find(":")
        if _colon_idx >= 0:
            _ex_type = _stripped[:_colon_idx].strip()
            if _ex_type.isidentifier() and (_ex_type.endswith("Error") or _ex_type.endswith("Exception") or _ex_type == "AssertionError"):
                return _ex_type, _stripped
        # Handle standalone AssertionError without colon
        if _stripped.startswith("AssertionError"):
            return "AssertionError", _stripped

    return None, None

def _extract_missing_module(raw: str) -> Optional[str]:
    # ModuleNotFoundError: No module named 'xyz'
    prefix = "No module named "
    for _q in ("'", '"'):
        _idx = raw.find(prefix + _q)
        if _idx >= 0:
            _start = _idx + len(prefix) + 1
            _end = raw.find(_q, _start)
            if _end >= 0:
                return raw[_start:_end]
    return None


def _extract_traceback_frames(raw: str, repo_root: Optional[str], max_frames: int) -> list[TraceFrame]:
    frames: list[TraceFrame] = []
    _lines = raw.splitlines()
    for i, _ln in enumerate(_lines):
        if _ln.startswith('  File "') and ', line ' in _ln and ', in ' in _ln:
            _file = _ln.split(', line ')[0].replace('  File "', '').rstrip('"')
            _rest = _ln.split(', line ', 1)[1]
            if ', in ' not in _rest:
                continue
            _line_str, _func = _rest.split(', in ', 1)
            _func = _func.strip()
            try:
                _line_no = int(_line_str.strip())
            except (ValueError, TypeError):
                continue
            _text = _lines[i + 1].strip() if i + 1 < len(_lines) else ""
            frames.append(TraceFrame(file=_file, line=_line_no, func=_func, text=_text))
            if len(frames) >= max_frames:
                break
    return frames


def _pick_primary_frame(frames: list[TraceFrame]) -> Optional[TraceFrame]:
    if not frames:
        return None
    return frames[0]


def _fingerprint(ctx: FailureContext) -> str:
    basis = "|".join([
        ctx.stage or "",
        ctx.type or "",
        ctx.primary_file or "",
        str(ctx.primary_line or ""),
        ctx.primary_symbol or "",
        ctx.test_id or "",
    ])
    return hashlib.sha256(basis.encode("utf-8")).hexdigest()[:16]


def _first_line(raw: str) -> str:
    for ln in raw.splitlines():
        if ln.strip():
            return ln.strip()
    return raw[:200].strip()


def _snip(raw: str, limit: int = 1200) -> str:
    """Truncate raw_snippet to limit length.

    pytest/python tracebacks have the exception type and message at the end (tail).
    Truncating only the head would cause the LLM to miss key diagnostics like
    `AssertionError: ...` when viewing the raw_snippet. Preserves head+tail evenly
    to ensure tail diagnostics are never lost.
    """
    s = raw.strip()
    if len(s) <= limit:
        return s
    _half = limit // 2
    return (
        s[:_half]
        + "\n...<snip — middle omitted; traceback tail preserved below>...\n"
        + s[-_half:]
    )

