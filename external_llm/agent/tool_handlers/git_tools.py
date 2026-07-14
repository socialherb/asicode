"""Shell tool handlers for ToolRegistry."""
from __future__ import annotations

import logging
import re as _re
import shutil as _shutil
import subprocess
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from ..background_job_manager import (
    BackgroundJobManager,
    get_global_background_job_manager,
    recover_communicate_partial,
)

if TYPE_CHECKING:
    from ..tool_registry import ToolResult

logger = logging.getLogger(__name__)

from .shell_policy import DANGEROUS_SHELL_COMMANDS as _DANGEROUS_SHELL_COMMANDS
from .shell_policy import FORBIDDEN_FLAGS as _FORBIDDEN_FLAGS

# LLM-generated shell commands are always executed under bash. LLMs are trained
# on bash, so bash is the dialect whose semantics match model expectations.
# Running under zsh (the user's $SHELL on macOS) causes subtle, hard-to-diagnose
# failures: zsh's default `nomatch` option rejects unquoted globs that bash
# silently passes through (e.g. `find . -name *.py` → zsh aborts the command
# with "no matches found"), and array indexing / pattern matching differ.
# Falling back to /bin/sh only on the rare systems without bash.
_BASH_EXECUTABLE = _shutil.which("bash") or "/bin/sh"

# ── Module-level compiled regexes for shell-command auto-correction ──────────
# Previously these were re-compiled on EVERY _tool_shell_exec invocation via
# __import__("re").compile(...). Compiling once at module load avoids the
# per-call dict lookup + cache-check overhead. See _tool_shell_exec usage below.
_PYTHON_CMD_RE = _re.compile(
    r"(?<![a-zA-Z0-9_.\-/])python(?![a-zA-Z0-9_.\-])(?=\s|[|&;`(\$]|$)"
)
_CAT_A_RE = _re.compile(r"\bcat\s+-A\b")
_FIND_RE = _re.compile(r"\bfind\s+")
_FIND_EXCLUDED_RE = _re.compile(r"-not\s+-path\s+['\"]?\./([^/'\")\s]+)")
_PIPE_SEP_RE = _re.compile(r"(\s*[|;]|\s+&&|\s+\|\|)")
_SORT_V_RE = _re.compile(r"\bsort\s+-V(\s+[^|&;<>]+)?")


# ── Capability shims for macOS (BSD userland, no GNU coreutils) ─────────────
# macOS lacks several GNU tools that LLMs emit frequently. A bare
# "command not found" is especially dangerous when the failing command heads a
# pipeline: the trailing `tail`/`head` runs against empty input and yields no
# output, so the agent can mis-read the silent failure as success.
#
# DESIGN PRINCIPLE — shim only when we can produce CORRECT output; for tools
# whose GNU-vs-BSD flag/regex semantics differ (sed -i, stat -c, ...), aliasing
# to the BSD tool would yield SILENT WRONG output, which is strictly worse than
# a loud error. Those get an explanatory stub that names the brew package.
#
# Each shim is guarded by `command -v <name>` so the whole prelude is a
# complete no-op on Linux / GNU-coreutils hosts: the function is defined only
# when the real binary is absent. The bash tool's own timeout→background
# transition at communicate() remains the outer safety net.
_SHELL_SHIM_PRELUDE = """# --- timeout: run a command with a wall-clock kill (GNU exit 124 on timeout)
if ! command -v timeout >/dev/null 2>&1; then
timeout() {
    local dur="$1"; shift
    [ $# -gt 0 ] || { echo "timeout: missing command" >&2; return 1; }
    "$@" &
    local pid=$!
    ( sleep "$dur" 2>/dev/null; kill -TERM "$pid" 2>/dev/null ) &
    local wpid=$!
    wait "$pid" 2>/dev/null
    local rc=$?
    if kill -0 "$wpid" 2>/dev/null; then
        kill "$wpid" 2>/dev/null; wait "$wpid" 2>/dev/null
    else
        rc=124
    fi
    return $rc
}
fi
# --- gtimeout: GNU coreutils alias of `timeout` — same semantics, delegate
if ! command -v gtimeout >/dev/null 2>&1; then
gtimeout() { timeout "$@"; }
fi
# --- tac: reverse line order — BSD `tail -r` is the native equivalent
if ! command -v tac >/dev/null 2>&1; then
tac() { tail -r "$@"; }
fi
# --- nproc: logical CPU count — sysctl (macOS has no online/offline split)
if ! command -v nproc >/dev/null 2>&1; then
nproc() {
    while [ $# -gt 0 ]; do case "$1" in -*) shift;; *) break;; esac; done
    sysctl -n hw.logicalcpu 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 1
}
fi
# --- shuf: random line permutation — BSD sort supports -R; handle -n N
if ! command -v shuf >/dev/null 2>&1; then
shuf() {
    local n=""
    while [ $# -gt 0 ]; do
        case "$1" in
            -n) n="$2"; shift 2;;
            -n*) n="${1#-n}"; shift;;
            -*) shift;;
            *) break;;
        esac
    done
    if [ -n "$n" ]; then sort -R "$@" | head -n "$n"; else sort -R "$@"; fi
}
fi
# --- realpath: canonical path (older macOS < ~10.15 lacks it). Pure bash via
#     cd+pwd -P — resolves symlinks for dirs; matches python3 os.path.realpath.
if ! command -v realpath >/dev/null 2>&1; then
realpath() {
    local p dir base
    for p in "$@"; do
        if [ -d "$p" ]; then
            ( cd "$p" && pwd -P )
        elif [ -f "$p" ]; then
            dir=$(cd "$(dirname "$p")" && pwd -P)
            base=$(basename "$p")
            printf '%s/%s\\n' "$dir" "$base"
        else
            printf '%s\\n' "$p"; return 1
        fi
    done
}
fi
# --- gsed / gstat: GNU sed/stat NOT aliasable to BSD (different -i / -c
#     semantics would silently corrupt output). Emit a clear install hint.
if ! command -v gsed >/dev/null 2>&1; then
gsed() {
    echo "asicode: 'gsed' (GNU sed) not installed; cannot alias to BSD sed (different -i/regex semantics)." >&2
    echo "          Install: brew install gnu-sed" >&2
    return 127
}
fi
if ! command -v gstat >/dev/null 2>&1; then
gstat() {
    echo "asicode: 'gstat' (GNU stat) not installed; cannot alias to BSD stat (no -c/--format support)." >&2
    echo "          Install: brew install coreutils" >&2
    return 127
}
fi
"""


def _apply_shell_shims(command: str) -> str:
    """Prepend macOS capability shims (timeout/tac/nproc/shuf/...) to a command.

    The prelude defines each shim only when the real binary is absent, so it is
    inert on Linux / GNU-coreutils hosts. GNU-vs-BSD-incompatible tools (gsed,
    gstat) get an explanatory error stub rather than a silently-wrong BSD alias.
    Applied at the execution boundary — after all command analysis and
    auto-correction, never to the stored command the parsers/auditors see.
    """
    return _SHELL_SHIM_PRELUDE + command

# Detect pytest invocations so pytest-specific recovery (missing entry-point
# plugin) only fires for pytest, not other argparse tools that also emit
# "unrecognized arguments:". Matches a pytest runner token at the start of a
# head segment (first command, or right after a pipe/&&/;/||). The token must
# be a standalone program name — 'py.test', 'pytest', or '<python> -m pytest'.
# Negatives: 'pip install pytest' (pytest is an arg, not the runner), quoted
# 'pytest' inside a grep pattern, 'python3 test_runner.py'.
_PYTEST_CMD_RE = _re.compile(
    r"(?:^|[|;]|&&|\|\|)\s*(?:\S*python\S*\s+-m\s+pytest|\bpytest\b|\bpy\.test\b)(?=\s|$)"
)

# Noise dirs auto-excluded when the LLM emits a bare `find *.py` without venv/
# node_modules exclusions. Module-level (not per-call) so the list is built once
# and stays consistent; previously it was a local inside _tool_shell_exec, rebuilt
# on every invocation.
_FIND_NOISE_DIRS = (".venv", "venv", "node_modules", "dist", "build", ".git")


def _quoted_intervals(command: str) -> list:
    """Return [start, end) intervals covering shell-quoted regions in *command*.

    Used by the shell-command auto-corrections below to AVOID rewriting tokens
    that appear inside a string literal. Bash quoting rules:
      * single-quote: every char is literal; no escapes; a single-quote cannot
        occur inside the region.
      * double-quote: only ``$ ` " \\`` are special; backslash escapes the
        next char.

    Example: ``grep -rln 'sort -V' tests/``  →  the ``sort -V`` substring lives
    inside a single-quoted region. Without this guard, the ``sort -V`` auto-
    correction would rewrite it to ``python3 -c "..."``, breaking the quoting
    and yielding ``syntax error near unexpected token '('``.

    An unterminated quote makes the remainder of the string a quoted region
    (conservative — protects against partial commands).
    """
    intervals = []
    i, n = 0, len(command)
    while i < n:
        c = command[i]
        if c == "'":
            j = command.find("'", i + 1)
            end = n if j == -1 else j + 1
            intervals.append((i, end))
            i = end
        elif c == '"':
            j = i + 1
            while j < n:
                if command[j] == "\\" and j + 1 < n:
                    j += 2
                    continue
                if command[j] == '"':
                    break
                j += 1
            end = n if j >= n else j + 1
            intervals.append((i, end))
            i = end
        else:
            i += 1
    return intervals


# ── Heredoc-body detection ──────────────────────────────────────────────
# A heredoc body (``<<DELIM ... DELIM``) is LITERAL program text fed to a child
# process (e.g. a python3 script). The shell-dialect auto-corrections below
# (python→python3, find-exclusion injection, sort -V, cat -A) must NEVER rewrite
# inside it — they are shell-syntax fixes with no business editing a script.
# Previously only shell quotes ('...' / "...") were protected via
# _quoted_intervals, so a comment like ``# find all *.py`` or a bare ``python``
# token inside a heredoc body was silently mangled → python SyntaxError on stdin
# or altered script semantics.
_HEREDOC_OPENER_RE = _re.compile(r"<<(-?)\s*([\"']?)([A-Za-z_][A-Za-z0-9_]*)\2")


def _heredoc_body_intervals(command: str) -> list:
    """Return ``[start, end)`` char-offset intervals covering heredoc BODIES.

    Recognises ``<<DELIM``, ``<<-DELIM``, ``<<'DELIM'``, ``<<"DELIM"`` (and an
    optional single space after ``<<``). The protected span runs from the first
    char after the opener line's newline through the end of the closing
    delimiter line (inclusive), so nothing in the body — delimiter line included
    — is touched by position-based rewrites. An unterminated heredoc protects to
    end-of-string (conservative). A nested ``<<`` that falls inside an
    already-claimed body is treated as literal text and skipped (``pos`` jumps
    past each body).

    Note: arithmetic bit-shift ``a << b`` can be misread as a heredoc opener
    (delim ``b``). That only ever OVER-protects (corrections get skipped), which
    is safe — it never corrupts. The common LLM pattern ``<< 'PYEOF'`` and bare
    ``<<EOF`` are handled correctly.
    """
    spans: list = []
    pos, n = 0, len(command)
    while pos < n:
        m = _HEREDOC_OPENER_RE.search(command, pos)
        if not m:
            break
        dash = m.group(1) == "-"
        delim = m.group(3)
        nl = command.find("\n", m.end())
        if nl == -1:
            break  # opener on the final line: no body to protect
        body_start = nl + 1
        if dash:
            close_re = _re.compile(r"(?:^|\n)[ \t]*" + _re.escape(delim) + r"[ \t]*(?=\n|$)")
        else:
            close_re = _re.compile(r"(?:^|\n)" + _re.escape(delim) + r"[ \t]*(?=\n|$)")
        cm = close_re.search(command, body_start)
        if cm:
            line_end = command.find("\n", cm.end())
            body_end = (line_end + 1) if line_end != -1 else n
        else:
            body_end = n  # unterminated → protect to end-of-string
        spans.append((body_start, body_end))
        pos = body_end  # resume after this body → a '<<' inside it is literal
    return spans


def _literal_intervals(command: str) -> list:
    """Combined "do-not-rewrite" regions: shell quotes AND heredoc bodies.

    The shell-command auto-corrections consult this via :func:`_match_in_quotes`
    so literal content — whether a quoted string or a heredoc script body — is
    never altered. See :func:`_quoted_intervals` / :func:`_heredoc_body_intervals`.
    """
    return _quoted_intervals(command) + _heredoc_body_intervals(command)


def _match_in_quotes(pos: int, intervals: list) -> bool:
    """True if *pos* lies within any protected (quoted / heredoc-body) interval."""
    return any(start <= pos < end for start, end in intervals)


def _truncate_bash_output(content: str, max_chars: int) -> str:
    """Truncate bash output to fit the token budget while preserving head+tail.

    pytest/traceback core diagnostics (``short test summary info``, ``N failed``,
    FAILED list) are placed at the **end (tail)** of the output. Truncating only the
    head would cause ``failure_context._try_parse_pytest`` to miss the core markers
    and fall through to UnknownError, so half is allocated to head and half to tail
    to preserve tail diagnostics.

    When non-ASCII (CJK/JSON) content is prevalent, the per-character token cost is
    higher, so the character cap is proportionally reduced.
    """
    if not content or len(content) <= max_chars:
        return content
    # ASCII ~3 chars/token, CJK ~1.5 chars/token. To maintain the same token budget,
    # non-ASCII output must be truncated at fewer characters.
    _sample = content[:4000]
    _ascii_ratio = (sum(ch.isascii() for ch in _sample) / len(_sample)) if _sample else 1.0
    _cap = max_chars if _ascii_ratio > 0.7 else int(max_chars * 0.5)
    if len(content) <= _cap:
        return content
    _truncated = len(content) - _cap
    _half = _cap // 2
    return (
        content[:_half]
        + f"\n... [truncated {_truncated} chars (middle) — bash output exceeded the "
        f"~{max_chars // 3000}K-token budget. Showing head+tail; pytest/traceback "
        f"diagnostics live at the tail. Re-run with a narrower filter "
        f"(grep, or `wc -c`/`wc -l` to size it first).]\n"
        + content[-_half:]
    )


def _run_bounded_subprocess(
    cmd, *, timeout: int = 120, shell: bool = False,
    executable: Optional[str] = None, cwd: Optional[str] = None,
    input: Optional[str] = None, env: Optional[dict[str, str]] = None,
) -> "subprocess.CompletedProcess":
    """``subprocess.run`` with a mandatory timeout and full process-group cleanup.

    Guarantees the agent never blocks indefinitely on a recovery-path subprocess.
    Recovery commands can stall waiting on input — ``pytest`` dropping into
    ``--pdb`` / ``input()``, a build prompt, or a network stall during ``pip
    install``. A bare ``subprocess.run`` (no timeout) hangs forever in that case;
    and since ``TimeoutExpired`` is a ``SubprocessError`` (not ``OSError``), it
    also escapes the surrounding ``except Exception`` only *after* the hang — by
    then the agent loop is wedged.

    Mirrors the safety discipline of the main bash path (``_tool_shell_exec``):
    ``start_new_session=True`` + ``killpg`` on timeout, so grandchildren
    (pytest-spawned server fixtures, child build servers) are torn down too,
    not orphaned. Returns a ``CompletedProcess`` (returncode=-9 + a trailing
    note on timeout) so callers keep their existing ``.returncode`` /
    ``.stdout`` / ``.stderr`` access and degrade gracefully.
    """
    import os as _os
    import signal as _signal
    proc = subprocess.Popen(
        cmd, shell=shell, executable=executable, cwd=cwd,
        stdin=subprocess.PIPE if input is not None else None,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, encoding="utf-8", errors="replace",
        start_new_session=True, env=env,
    )
    try:
        stdout, stderr = proc.communicate(input=input, timeout=timeout)
    except subprocess.TimeoutExpired:
        # Kill the whole process group (start_new_session created one) so
        # grandchildren are terminated too, not orphaned.
        try:
            _os.killpg(_os.getpgid(proc.pid), _signal.SIGKILL)
        except (ProcessLookupError, OSError):
            pass
        # Reap the killed process and drain partial output to avoid zombies.
        try:
            stdout, stderr = proc.communicate(timeout=5)
        except Exception:
            stdout, stderr = "", ""
        _note = f"\n[aborted: exceeded {timeout}s timeout]"
        return subprocess.CompletedProcess(
            args=cmd, returncode=-9,
            stdout=stdout or "", stderr=(stderr or "") + _note,
        )
    return subprocess.CompletedProcess(
        args=cmd, returncode=proc.returncode,
        stdout=stdout or "", stderr=stderr or "",
    )
class ShellToolsMixin:
    """Mixin providing shell tool implementations for ToolRegistry."""

    def _request_shell_danger_approval(self, dangerous_names: str, command: str) -> bool:
        """Request user approval for dangerous shell commands.

        Returns True if approved (or checkpoint unavailable), False if denied.
        """
        _config = getattr(self, "config", None)
        _enabled = getattr(_config, "user_checkpoint_enabled", False) if _config else False
        _callback = getattr(_config, "user_checkpoint_callback", None) if _config else None
        if not _enabled or not _callback:
            # Even in environments without checkpoint infra (Design Chat etc.),
            # ask directly if running in an interactive terminal.
            import sys as _sys
            if _sys.stdin.isatty():
                print()
                print(f"  ⚠️  Command execution requested: {dangerous_names}")
                print(f"      Command: {command[:200]}")
                try:
                    _answer = input("      Approve execution? (y/N): ").strip().lower()
                    return _answer in ("y", "yes")
                except (EOFError, KeyboardInterrupt):
                    return False
            return False  # non-interactive: deny by default

        _question_data = {
            "question": (
                f"The shell command contains dangerous operations ({dangerous_names}):\n"
                f"```\n{command[:500]}\n```\n"
                f"Allow execution?"
            ),
            "type": "yes_no",
            "options": ["yes", "no"],
            "reason": f"shell_exec requested dangerous command: {dangerous_names}",
            "default": "no",
            "source": "shell_danger_approval",
            "question_id": f"shell_danger_{uuid.uuid4().hex[:8]}",
        }
        try:
            _resp = _callback(_question_data)
            _answer = (_resp or {}).get("answer", "no") or "no"
            # Use LLM-based intent classifier to interpret the user's response.
            # This handles natural language variations and multi-language replies.
            from .._user_intent import UserApproval, classify_user_approval
            _verdict = classify_user_approval(_answer)
            return _verdict == UserApproval.APPROVED
        except Exception:
            return False  # deny on error

    def _maybe_recover_pytest_missing_plugin(
        self, command: str, stderr: str, original_command: str,
        timeout: int = 120,
    ) -> Optional[dict[str, Any]]:
        """Recover from pytest's "unrecognized arguments" for missing entry-point plugins.

        pytest aborts at core stage ("unrecognized arguments: --timeout=60") when an
        entry-point plugin option is passed but its package isn't installed. This layer
        diagnoses the missing plugin(s) via ``failure_context._extract_missing_pytest_plugins``,
        asks the user whether to install, and — on approval — pip-installs and re-runs.

        Returns a dict that the caller (``_tool_shell_exec``) interprets:
            - ``None`` → not a recovery target; caller proceeds normally.
            - ``{"_append_hint": "<text>"}`` → caller appends the hint to its result
              content (no override). Used for decline / unmapped options / install failure.
            - ``{"_override": ToolResult}`` → caller returns this result directly,
              replacing the original. Used after a successful install+rerun.

        The separation of intent (the keys) from the carrier (this dict) lets a single
        return contract cover three distinct caller behaviors without exception control-flow.
        """
        # ── Guard 1: only pytest commands ───────────────────────────────────
        # Other argparse tools (git, pip, ...) also emit "unrecognized arguments";
        # recovering them as missing-pytest-plugin would be wrong. _PYTEST_CMD_RE
        # matches the pytest runner at the head of a command segment.
        if not _PYTEST_CMD_RE.search(command):
            return None

        # ── Guard 2: must be a usage error, not a test failure ──────────────
        # A normal pytest failure (assertions, collection errors) is NOT a recovery
        # target — it has no "unrecognized arguments" line.
        from ..failure_context import _extract_missing_pytest_plugins
        offending_options, missing_packages = _extract_missing_pytest_plugins(stderr)
        if not offending_options:
            return None

        # ── Unmapped options → hint only (no install possible) ──────────────
        # e.g. --frobnicate isn't a known plugin option. We can't install an unknown
        # package, so surface a removal hint and let the LLM/model retry without it.
        if not missing_packages:
            return {
                "_append_hint": (
                    f"pytest rejected: {', '.join(offending_options)} are not recognized. "
                    f"These options are not mapped to any installable pytest plugin; "
                    f"remove them from the command and re-run."
                )
            }

        # ── Mapped options → ask user whether to install ────────────────────
        # Precedent: web_search_tools._ask_install_searxng uses _tool_ask_user with
        # metadata["answer"]. Recovery degrades gracefully on any exception (e.g.
        # checkpoint disabled / no callback) by treating it as a decline.
        try:
            _resp = self._tool_ask_user({
                "question": (
                    f"pytest failed because these plugins are not installed: "
                    f"{', '.join(missing_packages)}.\n"
                    f"Install them and re-run the command?"
                ),
                "type": "yes_no",
                "options": ["yes", "no"],
                "reason": f"missing pytest plugins: {', '.join(missing_packages)}",
                "default": "no",
            })
            _answer = (_resp.metadata or {}).get("answer", "no")
            from .._user_intent import UserApproval, classify_user_approval
            _approved = classify_user_approval(str(_answer)) == UserApproval.APPROVED
        except Exception:
            # ask_user unavailable (no checkpoint / callback error) → decline.
            _approved = False

        if not _approved:
            return {
                "_append_hint": (
                    f"pytest failed: missing plugin option(s) {', '.join(offending_options)} "
                    f"(would require: {', '.join(missing_packages)}). "
                    f"Install declined or unavailable; remove the option or install manually."
                )
            }

        # ── Approved → pip install then re-run ──────────────────────────────
        # Re-run the ORIGINAL command (the one the caller actually executed), not a
        # rewritten one — the plugins, once installed, make the original succeed.
        try:
            _install_cmd = f"pip install {' '.join(missing_packages)}"
            _inst = _run_bounded_subprocess(
                _install_cmd, shell=True,
                executable=_BASH_EXECUTABLE, cwd=self.repo_root,
                env={**__import__("os").environ.copy()},
            )
            if _inst.returncode != 0:
                _inst_err = (_inst.stderr or _inst.stdout or "unknown error").strip()
                return {
                    "_override": {
                        "ok": False,
                        "content": "",
                        "error": f"pip install failed for {', '.join(missing_packages)}: {_inst_err}",
                        "metadata": {
                            "recovered_pytest_plugin": False,
                            "installed_packages": [],
                        },
                        "retryable": False,
                    }
                }
            # Install succeeded → re-run the original command in the same repo_root.
            _rerun = _run_bounded_subprocess(
                _apply_shell_shims(original_command), shell=True,
                executable=_BASH_EXECUTABLE, cwd=self.repo_root,
                timeout=timeout,
                env={**__import__("os").environ.copy()},
            )
            _parts = []
            if _rerun.stdout:
                _parts.append(_rerun.stdout)
            if _rerun.stderr:
                _parts.append(f"[stderr]\n{_rerun.stderr}")
            _rerun_content = "\n".join(_parts) or "(no output)"
            from ..config.thresholds import config as _thresholds
            _rerun_content = _truncate_bash_output(_rerun_content, _thresholds.tokens.BASH_OUTPUT_MAX_CHARS)
            return {
                "_override": {
                    "ok": _rerun.returncode == 0,
                    "content": _rerun_content,
                    "metadata": {
                        "returncode": _rerun.returncode,
                        "background": False,
                        "recovered_pytest_plugin": True,
                        "installed_packages": list(missing_packages),
                    },
                }
            }
        except Exception as _e:
            return {
                "_override": {
                    "ok": False,
                    "content": "",
                    "error": f"Recovery execution failed: {_e}",
                    "metadata": {
                        "recovered_pytest_plugin": False,
                        "installed_packages": [],
                    },
                    "retryable": False,
                }
            }

    def _tool_shell_exec(self, args: dict[str, Any]) -> "ToolResult":
        if self.config.cancel_event and self.config.cancel_event.is_set():
            return self._make_result(
                ok=False,
                content="",
                error="Operation cancelled before shell execution",
                execution_time=0.0,
                retryable=False,
            )

        import shlex

        command = (args.get("command") or "").strip()
        timeout = int(args.get("timeout") or 120)

        if not command:
            return self._make_result(ok=False, content="", error="command is required")

        # ── LLM training-data path bias correction ────────────────────────
        # LLMs often generate hardcoded paths like /workspace from training data.
        # Since commands run with cwd=self.repo_root, replace bias paths with the actual repo_root.
        command = self._correct_bias_path(command)

        # ── python → python3 auto-fallback ──────────────────────────────
        # Prevent shell_exec failure on macOS where 'python' command may be absent.
        # Replace standalone 'python' (command, && python, | python etc.) with python3.
        # NOTE: matches inside shell-quoted regions (e.g. grep 'python|python3')
        # are deliberately skipped — rewriting them would corrupt the string
        # literal. See _quoted_intervals().
        _qiv = _literal_intervals(command)
        if _PYTHON_CMD_RE.search(command):
            def _repl_py(m):
                return "python3" if not _match_in_quotes(m.start(), _qiv) else m.group(0)
            fixed_cmd = _PYTHON_CMD_RE.sub(_repl_py, command)
            if fixed_cmd != command:
                logger.info("bash: auto-corrected python -> python3: %.200s", fixed_cmd)
                command = fixed_cmd

        # ── cat -A (GNU) → cat -vet (BSD) auto-fallback ──────────────
        # macOS BSD cat does not support the -A flag. -vet provides equivalent functionality.
        # (show non-printing + show $ at line ends + show tabs as ^I)
        # NOTE: `_qiv` is recomputed here because the python→python3 step above
        # may have lengthened the command and shifted every quoted interval; but
        # we only pay for the scan when cat -A is actually present (most commands
        # have neither, skipping the work entirely).
        if _CAT_A_RE.search(command):
            _qiv = _literal_intervals(command)
            def _repl_cat(m):
                return "cat -vet" if not _match_in_quotes(m.start(), _qiv) else m.group(0)
            fixed_cmd = _CAT_A_RE.sub(_repl_cat, command)
            if fixed_cmd != command:
                logger.info("bash: auto-corrected cat -A -> cat -vet: %.200s", fixed_cmd)
                command = fixed_cmd

        # ── find *.py/*.ts without venv/node_modules exclusions → auto-inject ─
        # When the LLM generates a find command without excluding .venv / node_modules etc.,
        # thousands of site-packages files flood the context, causing token explosion.
        # Auto-inject missing exclusion paths when a find command is detected.
        _find_match = _FIND_RE.search(command)
        # Skip the entire find auto-correction if the matched 'find' token lives
        # inside a shell-quoted region (e.g. grep 'find -name' ...). Rewriting
        # it would corrupt the literal string. _PIPE_SEP_RE / _FIND_EXCLUDED_RE
        # below also operate on the raw command, so they inherit this guard.
        _find_in_quotes = bool(_find_match and _match_in_quotes(_find_match.start(), _literal_intervals(command)))
        if _find_match and not _find_in_quotes:
            _already_excluded = set(
                m.group(1)
                for m in _FIND_EXCLUDED_RE.finditer(command)
            )
            _missing = [d for d in _FIND_NOISE_DIRS if d not in _already_excluded]
            if _missing:
                _exclude_flags = " ".join(
                    f'-not -path "./{d}/*"' for d in _missing
                )
                # find ... [existing flags] → find ... [existing flags] -not -path ...
                # Insert before pipe/redirect (first | ; && ahead)
                # \s* before [|;] to handle "2>/dev/null;echo" (no space before ;)
                # \s+ before &&/|| because those binary operators always have whitespace.
                # Separator search MUST start AFTER the find token.
                # If 'cd ... && find ...' has another segment before find,
                # the first separator (cd's &&) would catch the exclude flags and
                # attach them to the wrong cd command, causing 'zsh: too many arguments'.
                _pipe_match = _PIPE_SEP_RE.search(command, _find_match.end())
                _insert_pos = _pipe_match.start() if _pipe_match else len(command)

                # Split find command into [before][findcmd][after] segments.
                _before = command[:_find_match.start()]
                _findcmd = command[_find_match.start():_insert_pos]
                _after = command[_insert_pos:]

                # ── -o (OR) expression parentheses correction ─────────────────
                # find's -a (implicit AND) binds tighter than -o, so
                #   find p -name A -o -name B -not -path C
                # parses as 'A OR (B AND NOT C)', meaning files matching -name A
                # are NOT excluded. When -o is present, wrap the expression in
                # \( ... \) to produce:
                #   find p \( -name A -o -name B \) -not -path C
                # = '(A OR B) AND NOT C'. \(,\) is escaped to prevent the shell
                # from interpreting it as a subshell, with spaces on both sides
                # so find recognizes them as separate tokens.
                if _re.search(r"(^|\s)-o(\s|$)", _findcmd):
                    _kw_len = _find_match.end() - _find_match.start()
                    _head = _findcmd[:_kw_len]   # "find "
                    _rest = _findcmd[_kw_len:]   # "p -name A -o -name B"
                    # Separate leading path operands from expression (first predicate -X / ( / !).
                    _pred = _re.search(r"(^|\s)([-(!])", _rest)
                    if _pred:
                        _ps = _pred.start(2)
                        _paths, _expr = _rest[:_ps], _rest[_ps:]
                        _findcmd = _head + _paths + r"\( " + _expr + r" \)"

                fixed_cmd = _before + _findcmd + " " + _exclude_flags + _after
                if fixed_cmd != command:
                    logger.info(
                        "bash: auto-injected find exclusions (%s): %.300s",
                        ", ".join(_missing),
                        fixed_cmd,
                    )
                    command = fixed_cmd

        # ── sort -V (GNU) → python3 natural sort (BSD) auto-fallback ─────────
        # macOS BSD sort does not support -V (natural version sort).
        # Use python3 to split into numeric/text segments for natural sort.
        # Handles arbitrary formats (semver, mixed alpha-numeric, etc.).
        _SORT_V_NATURAL_SCRIPT = (
            "import sys,re;"
            "lines=sys.stdin.read().splitlines();"
            "lines.sort(key=lambda x:[int(s)if s.isdigit()"
            "else s.lower()for s in re.split(r'(\\d+)',x)]);"
            "print('\\n'.join(lines))"
        )
        if _SORT_V_RE.search(command):
            _qiv = _literal_intervals(command)

            def _repl_sort_v(m):
                # Skip matches inside shell-quoted regions — rewriting a
                # ``sort -V`` that lives inside e.g. grep's search pattern
                # injects a python3 -c "..." with literal parens and breaks
                # the quoting (bash: syntax error near unexpected token '(').
                if _match_in_quotes(m.start(), _qiv):
                    return m.group(0)
                _a = (m.group(1) or "").strip()
                _py = f'python3 -c "{_SORT_V_NATURAL_SCRIPT}"'
                return f"cat {_a} | {_py}" if _a else _py

            fixed_cmd = _SORT_V_RE.sub(_repl_sort_v, command)
            if fixed_cmd != command:
                logger.info("bash: auto-corrected sort -V -> python3: %.200s", fixed_cmd)
                command = fixed_cmd

        _SHELL_SYNTAX = {"for", "in", "do", "done", "if", "then", "else", "fi", "while", "until", "echo"}
        _SEGMENT_SEPARATORS = {"|", "&&", "||", ";"}

        # heredoc syntax (<<) cannot be parsed by shlex.split → extract only the header portion for permission check
        # NOTE: _re is the module-level `import re as _re` (see top of file).
        # Do NOT re-import here — a local `import re as _re` makes Python treat
        # _re as a function-local name across the WHOLE body, so the earlier
        # _re.search() calls (find -o grouping, ~L175) raise UnboundLocalError
        # before this line ever runs.
        _is_heredoc = bool(_re.search(r"<<\s*['\"]?\w", command))
        if _is_heredoc:
            # Parse only the heredoc head (first line or portion before <<) with shlex to extract the execution command
            _heredoc_header = _re.split(r"<<", command, maxsplit=1)[0].strip()
            try:
                parts = shlex.split(_heredoc_header) if _heredoc_header else []
            except Exception:
                parts = _heredoc_header.split()
        else:
            try:
                parts = shlex.split(command)
            except Exception as e:
                return self._make_result(ok=False, content="", error=f"Invalid command syntax: {e}")

        if not parts:
            return self._make_result(ok=False, content="", error="Empty command")

        executables = set()
        dangerous_executables = set()
        expect_executable = True
        for token in parts:
            if token in _SEGMENT_SEPARATORS:
                expect_executable = True
                continue
            if token.startswith(">") or token.startswith("<") or token.startswith("2>"):
                continue
            if token.startswith("-") or "/" in token or token.startswith("$") or "=" in token:
                if expect_executable:
                    expect_executable = False
                continue

            if expect_executable:
                expect_executable = False
                name = Path(token).name
                if name in _SHELL_SYNTAX:
                    continue
                if name in _DANGEROUS_SHELL_COMMANDS:
                    dangerous_executables.add(name)
                executables.add(name)
            else:
                # Check forbidden flags for already-registered executables
                for exe in executables:
                    if exe in _FORBIDDEN_FLAGS and token in _FORBIDDEN_FLAGS[exe]:
                        return self._make_result(
                            ok=False, content="",
                            error=f"Flag '{token}' is not allowed for '{exe}'. "
                                  f"Use apply_patch for file edits.",
                        )

        # User approval required for dangerous commands
        if dangerous_executables:
            _danger_str = ", ".join(sorted(dangerous_executables))
            _approval = self._request_shell_danger_approval(_danger_str, command)
            if not _approval:
                return self._make_result(
                    ok=False, content="",
                    error=(
                        f"User denied execution of dangerous command(s): {_danger_str}. "
                        f"Operation cancelled."
                    ),
                )

        # Background job manager for timeout→background transition
        _bg_mgr = self._get_bg_manager()

        try:
            import os as _os
            _env = _os.environ.copy()
            _env.pop("MallocStackLogging", None)
            _env.pop("MallocStackLoggingDirectory", None)

            # Use Popen for non-blocking start — allows timeout→background transition
            proc = subprocess.Popen(
                _apply_shell_shims(command), shell=True,
                executable=_BASH_EXECUTABLE,
                cwd=self.repo_root, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True,
                # Decode tolerantly: commands like `head -c N` (cuts a multibyte
                # char mid-sequence) or `cat -vet` (emits raw non-printing bytes)
                # routinely produce non-UTF-8 output. Strict decoding would raise
                # UnicodeDecodeError and surface as a spurious "Command execution
                # failed", blocking the agent on otherwise-successful commands.
                encoding="utf-8", errors="replace",
                # Create new process group so background kill can terminate children
                start_new_session=True,
                env=_env,
            )

            # Use communicate() for correct pipe I/O handling (prevents deadlock
            # when child process fills the pipe buffer while we wait).
            # On timeout: process remains running → background transition.
            # On success: stdout/stderr are fully captured strings.
            try:
                stdout_data, stderr_data = proc.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                # ── Timeout → background transition ──────────────────────────
                # Instead of returning a timeout error (which wastes the work done),
                # transition the process to background management.
                #
                # Salvage the output communicate() already consumed from the
                # pipes before timing out (lives in a CPython-private buffer,
                # unreadable via the raw fd) — see recover_communicate_partial.
                recover_communicate_partial(proc)

                job_id = _bg_mgr.start(command, proc)
                logger.info("bash: timed out, bg job=%s cmd=%.200s", job_id, command)
                return self._make_result(
                    ok=True,
                    content=f"⏳ Command moved to background (Job ID: {job_id}). The command exceeded {timeout}s timeout.",
                    metadata={"background_job_id": job_id},
                )

            stdout = stdout_data or ""
            raw_stderr = stderr_data or ""
            filtered_stderr_lines = [
                line for line in (raw_stderr.splitlines() if raw_stderr else [])
                if "MallocStackLogging" not in (line or "")
            ]
            stderr = "\n".join(filtered_stderr_lines).strip()

            parts_out = []
            if stdout:
                parts_out.append(stdout)
            if stderr:
                parts_out.append(f"[stderr]\n{stderr}")
            content = "\n".join(parts_out) or "(no output)"

            #── bash output size restriction ─────────────────────────────────────
            # Prevent context token explosion from large output (git diff, find, rg -r, etc.).
            # Limit managed as a single BASH_OUTPUT_MAX_CHARS threshold
            # (NO hardcoding — mismatch between threshold and actual cap would defeat tuning).
            # Head+tail preservation logic is encapsulated in _truncate_bash_output and tested.
            from ..config.thresholds import config as _thresholds
            content = _truncate_bash_output(content, _thresholds.tokens.BASH_OUTPUT_MAX_CHARS)

            # rg/grep etc. return exit code 1 for "no match" (exit code 2 is the real error).
            # exit code 1 + no stderr = normal execution but no result → not an error.
            # If stdout has meaningful output, treat exit code != 0 as ok.
            #   - find: partially fails due to permission denied after finding some files
            #   - rg/grep --with-filename: exits abnormally after match due to internal error (SIGPIPE etc.)
            # stderr and exit code are included in content for LLM visibility.
            # If stdout is empty and exit code != 0, treat as a real failure.
            ok = proc.returncode == 0 or (proc.returncode == 1 and not stderr) or bool(stdout.strip())

            # ── pytest missing-plugin recovery ────────────────────────────────
            # A non-zero exit with "unrecognized arguments" in stderr, for a pytest
            # command, signals an uninstalled entry-point plugin. Attempt recovery
            # (diagnose → ask_user → install → re-run) before returning the failure.
            # The recovery returns a plain dict contract: None = no recovery,
            # {"_override": {...}} = replace this result, {"_append_hint": str} =
            # annotate the failure so the model can self-correct. Only attempt on a
            # genuine failure — a successful command never needs recovery.
            if not ok and stderr and "unrecognized arguments" in stderr:
                _recovery = self._maybe_recover_pytest_missing_plugin(
                    command=command, stderr=stderr, original_command=command,
                    timeout=timeout,
                )
                if _recovery is not None:
                    if "_override" in _recovery:
                        # Recovery produced a replacement result (install+rerun, or a
                        # definitive install failure). Convert the plain-dict contract
                        # to a ToolResult so the caller sees a normal result.
                        _ov = _recovery["_override"]
                        return self._make_result(
                            ok=_ov.get("ok", False),
                            content=_ov.get("content", ""),
                            error=_ov.get("error"),
                            metadata=_ov.get("metadata", {}),
                            retryable=_ov.get("retryable", True),
                        )
                    if "_append_hint" in _recovery:
                        content = content + "\n\n" + _recovery["_append_hint"]

            return self._make_result(ok=ok, content=content, metadata={"returncode": proc.returncode, "background": False})
        except subprocess.TimeoutExpired:
            # Safety net: should not happen (Popen.wait timeout is handled above),
            # but keep as fallback for edge cases.
            return self._make_result(
                ok=False, content="",
                error=f"Command timed out after {timeout}s",
                metadata={"timeout": True},
            )
        except Exception as e:
            return self._make_result(ok=False, content="", error=f"Command execution failed: {e}")

    def _get_bg_manager(self) -> BackgroundJobManager:
        """Get or create the shared BackgroundJobManager instance."""
        _mgr = getattr(self, "_bg_manager", None)
        if _mgr is None:
            _mgr = get_global_background_job_manager()
            self._bg_manager = _mgr
        return _mgr

    def _tool_job(self, args: dict[str, Any]) -> "ToolResult":
        """Manage background shell jobs: list, output, kill."""
        action = str(args.get("action", "")).strip().lower()

        if not action:
            return self._make_result(
                ok=False, content="",
                error="'action' is required. Choose: list, output, kill",
            )

        _ACTIONS = {
            "list": self._job_list,
            "output": self._job_output,
            "kill": self._job_kill,
        }

        handler = _ACTIONS.get(action)
        if handler is None:
            return self._make_result(
                ok=False, content="",
                error=f"Unknown action: '{action}'. Available: list, output, kill",
            )

        return handler(args)

    def _job_list(self, args: dict[str, Any]) -> "ToolResult":
        """List all background jobs."""
        _bg_mgr = self._get_bg_manager()
        jobs = _bg_mgr.list_jobs(include_completed=True)

        if not jobs:
            return self._make_result(ok=True, content="No background jobs.")

        lines = [f"Background jobs ({len(jobs)} total):"]
        lines.append(f"{'ID':<14} {'STATUS':<12} {'ELAPSED':<10} {'CMD':<80}")
        lines.append("-" * 120)
        for j in jobs:
            # Truncate command to fit one line
            cmd = j.command.replace("\n", "\\n")[:77]
            elapsed = f"{j.elapsed:.1f}s"
            lines.append(f"{j.job_id:<14} {j.status:<12} {elapsed:<10} {cmd}")

        return self._make_result(ok=True, content="\n".join(lines))

    def _job_output(self, args: dict[str, Any]) -> "ToolResult":
        """Show current output of a background job.

        If *wait_timeout* > 0, blocks until the job finishes or the
        timeout expires (polling internally), then returns the final
        output.  This eliminates the need for the caller to poll
        repeatedly.
        """
        job_id = str(args.get("job_id", "")).strip()
        if not job_id:
            return self._make_result(
                ok=False, content="",
                error="'job_id' is required for output action.",
            )

        wait_timeout = args.get("wait_timeout", 0)
        try:
            wait_timeout = float(wait_timeout)
        except (TypeError, ValueError):
            wait_timeout = 0.0

        _bg_mgr = self._get_bg_manager()

        if wait_timeout > 0:
            info = _bg_mgr.wait_for_completion(job_id, timeout=wait_timeout)
        else:
            info = _bg_mgr.get_info(job_id)

        if info is None:
            return self._make_result(
                ok=False, content="",
                error=f"Job '{job_id}' not found. Use `job` with action='list' to see active jobs.",
            )

        parts = [f"Job ID: {info.job_id} | Status: {info.status} | Elapsed: {info.elapsed:.1f}s"]
        if info.stdout:
            parts.append(f"\n[stdout]\n{info.stdout}")
        if info.stderr:
            parts.append(f"\n[stderr]\n{info.stderr}")

        return self._make_result(ok=True, content="\n".join(parts))

    def _job_kill(self, args: dict[str, Any]) -> "ToolResult":
        """Kill a background job."""
        job_id = str(args.get("job_id", "")).strip()
        if not job_id:
            return self._make_result(
                ok=False, content="",
                error="'job_id' is required for kill action.",
            )

        _bg_mgr = self._get_bg_manager()
        status = _bg_mgr.kill(job_id)
        if status is None:
            return self._make_result(
                ok=False, content="",
                error=f"Job '{job_id}' not found. Use `job` with action='list' to see active jobs.",
            )
        return self._make_result(ok=True, content=f"Job '{job_id}' killed. Final status: {status}")

