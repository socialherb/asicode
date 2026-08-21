"""Shell tool handlers for ToolRegistry."""
from __future__ import annotations

import ast as _ast
import codecs as _codecs
import io as _io
import logging
import os
import re as _re
import selectors as _selectors
import shlex
import shutil as _shutil
import subprocess
import time as _time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from ...common.subprocess_utils import CANCEL_POLL_INTERVAL as _CANCEL_POLL_INTERVAL
from ...common.subprocess_utils import run_bounded_subprocess as _run_bounded_subprocess
from ..agent_loop_types import AgentCancelled
from ..background_job_manager import (
    BackgroundJobManager,
    get_global_background_job_manager,
    strip_malloc_noise,
)
from ..cancel_scope import effective_cancel

if TYPE_CHECKING:
    from ..tool_registry import ToolResult

from .shell_policy import ARG_COMMAND_INTRODUCERS as _ARG_COMMAND_INTRODUCERS
from .shell_policy import COMMAND_INTRODUCING_KEYWORDS as _COMMAND_INTRODUCING_KEYWORDS
from .shell_policy import COMMAND_STRING_BUILTINS as _COMMAND_STRING_BUILTINS
from .shell_policy import COMMAND_WRAPPERS as _COMMAND_WRAPPERS
from .shell_policy import DANGEROUS_EXECUTABLE_PREFIXES as _DANGEROUS_EXECUTABLE_PREFIXES
from .shell_policy import DANGEROUS_FLAG_COMBOS as _DANGEROUS_FLAG_COMBOS
from .shell_policy import DANGEROUS_SHELL_COMMANDS as _DANGEROUS_SHELL_COMMANDS
from .shell_policy import EVAL_BUILTINS as _EVAL_BUILTINS
from .shell_policy import FORBIDDEN_FLAGS as _FORBIDDEN_FLAGS
from .shell_policy import NON_OVERWRITING_FLAGS as _NON_OVERWRITING_FLAGS
from .shell_policy import OVERWRITING_EXECUTABLES as _OVERWRITING_EXECUTABLES
from .shell_policy import PYTHON_DESTRUCTIVE_CALLS as _PY_DESTRUCTIVE_CALLS
from .shell_policy import PYTHON_INTERPRETERS as _PYTHON_INTERPRETERS
from .shell_policy import PYTHON_SHELL_ESCAPES as _PY_SHELL_ESCAPES
from .shell_policy import SHELL_INTERPRETERS as _SHELL_INTERPRETERS
from .shell_policy import SHELL_TIMEOUT_DEFAULT as _SHELL_TIMEOUT_DEFAULT
from .shell_policy import SHELL_TIMEOUT_MAX as _SHELL_TIMEOUT_MAX
from .shell_policy import STDIN_INTERPRETERS as _STDIN_INTERPRETERS
from .shell_policy import WRAPPER_POSITIONAL_ARGS as _WRAPPER_POSITIONAL_ARGS
from .shell_policy import WRAPPER_SHELL_C_PAYLOAD as _WRAPPER_SHELL_C_PAYLOAD
from .shell_policy import WRAPPER_VALUE_FLAGS as _WRAPPER_VALUE_FLAGS

logger = logging.getLogger(__name__)

# _CANCEL_POLL_INTERVAL is imported above: the cadence is shared with the search
# path, since a cancel that feels instant in `bash` and laggy in `grep` is worse
# than either. See common.subprocess_utils.CANCEL_POLL_INTERVAL.
#
# Bytes taken from a pipe per ready-read. Large enough that a chatty command
# costs few syscalls, small enough that one read cannot outrun the poll.
_PIPE_READ_CHUNK = 64 * 1024
# Floor on how long a command may take to EXIT once both its pipes are at EOF.
# Without it a command that finishes exactly as its budget runs out is pushed to
# the background with its complete result already captured.
_EXIT_GRACE = 2.0

# `FOO=bar cmd` — an assignment prefix, not the command itself.
_ENV_ASSIGN_RE = _re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
# `$RM` / `${RM}` / `${VENV}/bin/python` in the EXECUTABLE slot. Anchored at the
# start because only a token that BEGINS with the expansion can have the
# expansion decide which program runs (`x$RM` is a word, not a command name).
_EXEC_SLOT_VAR_RE = _re.compile(
    r"^\$(?:\{(?P<brace>[A-Za-z_][A-Za-z0-9_]*)\}|(?P<bare>[A-Za-z_][A-Za-z0-9_]*))"
)
# `timeout 5 cmd` / `timeout 5s cmd` / `nice 10 cmd` — a wrapper's numeric arg.
_WRAPPER_NUM_ARG_RE = _re.compile(r"^[0-9]+(?:\.[0-9]+)?[smhd]?$")

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
#
# `timeout` decides "did we time out?" from a MARKER the watchdog writes, not
# from whether the watchdog is still alive. The liveness test was a race: the
# watchdog subshell has nothing to do after `kill -TERM`, so it normally exits
# before `wait` resumes the caller — but only normally. Resume the parent
# first and `kill -0 $wpid` still succeeds, so the timeout is read as "the
# command finished on its own" and the caller gets 143 (128+SIGTERM) instead of
# 124. Seen once in a 11,918-test suite run and reproduced deterministically by
# leaving the watchdog alive one extra millisecond past its kill (5/5 wrong).
# The marker is written BEFORE the signal, so a child that died from our TERM
# always finds it already on disk — no ordering left to lose. Hosts without
# mktemp keep the old heuristic rather than losing the feature.
#
# The leading GNU option block is not decoration. Taking "$1" as the duration
# outright made every option form silently destroy the command: `timeout -k 2 5
# cmd` bound dur="-k" and then ran `2 5 cmd`, so the caller got 127 and the
# command NEVER RAN — the exact silent-failure-in-a-pipeline hazard this
# prelude exists to prevent (`timeout -k 5 300 pytest ... | tail -20` yields an
# empty tail that reads as success). `-s`/`-k` are honoured rather than merely
# skipped, because a caller asking for KILL after a grace period is asking for
# behaviour, not syntax. Unknown `-*` options are dropped: GNU would reject
# them, but refusing to run is the failure mode we are removing.
_SHELL_SHIM_PRELUDE = """# --- timeout: run a command with a wall-clock kill (GNU exit 124 on timeout)
if ! command -v timeout >/dev/null 2>&1; then
timeout() {
    local sig="TERM" kill_after="" dur
    while [ $# -gt 0 ]; do
        case "$1" in
            -k) kill_after="$2"; shift 2;;
            -k*) kill_after="${1#-k}"; shift;;
            --kill-after=*) kill_after="${1#*=}"; shift;;
            --kill-after) kill_after="$2"; shift 2;;
            -s) sig="$2"; shift 2;;
            -s*) sig="${1#-s}"; shift;;
            --signal=*) sig="${1#*=}"; shift;;
            --signal) sig="$2"; shift 2;;
            --preserve-status|--foreground) shift;;
            --) shift; break;;
            -*) shift;;
            *) break;;
        esac
    done
    dur="$1"; shift
    [ $# -gt 0 ] || { echo "timeout: missing command" >&2; return 1; }
    local mark
    mark="$(mktemp -t asi_timeout.XXXXXX 2>/dev/null)" || mark=""
    # Own process group for the child (job control): the escalation below then
    # signals the GROUP (`kill -N -- -pid`), matching GNU timeout semantics —
    # a `timeout -k 1 1 bash -c '...; sleep 30'` must take the wrapper's
    # children with it. Without this the killed wrapper leaves its children
    # orphaned, and an orphan holding the caller's stdout/stderr pipes blocks
    # the caller until it finishes on its own (measured: a 2 s timeout turning
    # into a 30 s hang). Group kill needs the group to exist, which only job
    # control provides in a non-interactive shell. `kill -N -- -pid pid` keeps
    # the plain-pid fallback for shells where set -m could not apply.
    set -m
    "$@" &
    local pid=$!
    ( sleep "$dur" 2>/dev/null; [ -n "$mark" ] && printf t >"$mark"
      kill -"$sig" -- -"$pid" "$pid" 2>/dev/null
      if [ -n "$kill_after" ]; then
          sleep "$kill_after" 2>/dev/null; kill -KILL -- -"$pid" "$pid" 2>/dev/null
      fi ) &
    local wpid=$!
    wait "$pid" 2>/dev/null
    local rc=$?
    if [ -n "$mark" ]; then
        kill -- -"$wpid" "$wpid" 2>/dev/null; wait "$wpid" 2>/dev/null
        [ -s "$mark" ] && rc=124
        rm -f "$mark" 2>/dev/null
    elif kill -0 "$wpid" 2>/dev/null; then
        kill -- -"$wpid" "$wpid" 2>/dev/null; wait "$wpid" 2>/dev/null
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


def _blank_heredoc_bodies(command: str, *, interpreter_names: frozenset | None = None) -> str:
    """Return *command* with heredoc BODIES replaced by spaces — except those
    whose receiver is a shell interpreter.

    Length-preserving, because :func:`_split_shell_separators` locates
    separators by byte position in the string the tokens came from — a
    substitution that changed length would misalign every position after it.

    The danger scan needs "the command, minus data-only heredoc bodies": body
    text written to a file (``cat <<EOF > f.txt``) is data, not commands to
    run, so ``rm`` inside one must not prompt. But a shell interpreter reading
    a heredoc (``bash <<EOF``) IS executing that body as shell code, so
    blanking it hides the commands entirely: measured, ``bash <<EOF ; rm -rf x
    ; EOF`` ran unprompted because the body was blanked before the scan. The
    *interpreter_names* parameter selects which receivers keep their bodies
    visible, and only ``SHELL_INTERPRETERS`` is passed — ``python3`` etc. are
    deliberately excluded (their body is not shell code, same known-limit as
    ``python -c``).

    It previously got that by truncating at the first ``<<`` and
    scanning only what came before, which also discarded every command AFTER
    the body:

        cat <<EOF > f.txt
        hello
        EOF
        rm -rf victim          <- never scanned, ran with no prompt

    That is the defect fixed for a bare newline in 8955a266, reached by a
    different route, and it applied equally to `git reset --hard` and to a
    truncating `>` redirect on the opener line itself (`cat <<EOF > src/main.py`
    destroys the file whether or not the body is dangerous). Blanking instead of
    truncating keeps the body unscanned while leaving everything around it —
    opener line included — visible to the scan.

    Spans come from :func:`_heredoc_body_intervals`, so the closing delimiter
    line is blanked too (it is inside the span) and an unterminated heredoc
    blanks to end-of-string — conservative, and no worse than the truncation it
    replaces.
    """
    spans = _heredoc_body_intervals(command)
    if not spans:
        return command
    preserve: set = set()
    if interpreter_names is not None:
        # For each heredoc body, decide whether its RECEIVER is a shell
        # interpreter. If so the body IS shell code and must be scanned, not
        # blanked.
        #
        # The receiver is the executable of the command segment the `<<` sits
        # in, so it is resolved by :func:`_segment_executable` — the shared
        # implementation of the danger scan's own executable-slot rules —
        # applied to the opener line's prefix. Taking the prefix is exact rather
        # than approximate: a heredoc operator cannot be in a different segment
        # from the command it feeds, so the last segment of the text before `<<`
        # IS that command (`cat x; bash <<EOF` resolves to `bash`).
        #
        # Two string heuristics preceded this and both shipped bypasses. Reading
        # the last word before `<<` resolved `bash -s <<EOF` — the canonical
        # spelling for feeding bash a script — to `-s`, and equally missed
        # `bash -x`, `bash -euo pipefail`, `bash 2>/dev/null` and `/bin/bash -s`;
        # `sudo bash <<EOF` passed only because the wrapper happened to leave
        # `bash` last, never because wrappers were handled. Scanning every word
        # instead fixed those but charged `echo bash <<EOF` a false prompt and
        # still knew nothing of `sudo -u me bash` or `chroot /mnt bash`. Both
        # were re-derivations of slot rules that already existed twenty lines
        # away in the scan loop, which is exactly why they drifted.
        #
        # `-c` suppresses preservation — the interpreter then runs THAT string
        # and ignores stdin, so the body is dead code. Scoped to the receiver's
        # OWN segment tokens (not the whole line), so `cat -c f; bash <<EOF`
        # keeps its prompt: the `-c` there belongs to `cat`, not to `bash`.
        pos, idx = 0, 0
        while pos < len(command) and idx < len(spans):
            m = _HEREDOC_OPENER_RE.search(command, pos)
            # The span list was built by this same regex with this same skip
            # rule, so opener search and spans cannot disagree.
            if not m:  # pragma: no cover
                break
            line_start = command.rfind("\n", 0, m.start()) + 1
            prefix = command[line_start:m.start()]
            _recv, _seg = _segment_executable(prefix)
            if (
                _recv in interpreter_names
                and not any(_SHELL_C_FLAG_RE.match(_t) for _t in _seg)
            ):
                preserve.add(idx)
            idx += 1
            pos = spans[idx - 1][1]
    out = list(command)
    for idx, (start, end) in enumerate(spans):
        if idx in preserve:
            continue  # interpreter-targeted: body IS shell code, scan it
        for i in range(start, min(end, len(out))):
            # Newlines are separators to _normalize_for_scan; blank them too so
            # a body cannot re-arm the executable slot for its own text.
            out[i] = " "
    return "".join(out)


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


def _python_payload_effects(code: str) -> tuple:
    """``(destructive dotted names, shell command strings, opaque_escape)``.

    Reads a ``python -c`` payload with the parser for the language it is written
    in. The shell scan next door cannot: a Python one-liner is not shell code,
    and the gate treated "not shell" as "not checkable" — which is how
    ``python3 -c "import shutil; shutil.rmtree('/tmp/x')"`` ran unprompted while
    the ``rm -rf`` it is equivalent to did not.

    Names are RESOLVED, not matched textually, so aliasing does not evade it:
    ``import shutil as sh``, ``from shutil import rmtree as rt`` and
    ``__import__('shutil').rmtree`` all reduce to ``shutil.rmtree``, and a
    ``Path('x').unlink()`` reduces to ``pathlib.Path.unlink`` through the
    receiver call.

    Shell strings are returned rather than judged, so the caller can splice them
    into the token scan and let every existing rule apply — the same treatment
    ``bash -c`` gets. ``opaque_escape`` is True when a shell escape is handed
    something this cannot read (an f-string, a variable): the doctrine for an
    unreadable payload reaching a shell is already "ask", not "allow" — see the
    pipe rule.

    A payload that does not parse yields nothing. It also will not run.
    """
    try:
        _tree = _ast.parse(code)
    except (SyntaxError, ValueError):
        return set(), [], False

    _alias: dict = {}
    for _node in _ast.walk(_tree):
        if isinstance(_node, _ast.Import):
            for _a in _node.names:
                if _a.asname:
                    _alias[_a.asname] = _a.name
                else:
                    _alias[_a.name.split(".")[0]] = _a.name.split(".")[0]
        elif isinstance(_node, _ast.ImportFrom) and _node.module:
            for _a in _node.names:
                _alias[_a.asname or _a.name] = f"{_node.module}.{_a.name}"

    def _dotted(_n):
        if isinstance(_n, _ast.Name):
            return _alias.get(_n.id, _n.id)
        if isinstance(_n, _ast.Attribute):
            _base = _dotted(_n.value)
            return f"{_base}.{_n.attr}" if _base else None
        if isinstance(_n, _ast.Call):
            # `__import__('os').system` — the import is the expression, so the
            # module name is an argument rather than a binding.
            if (
                isinstance(_n.func, _ast.Name)
                and _n.func.id == "__import__"
                and _n.args
                and isinstance(_n.args[0], _ast.Constant)
                and isinstance(_n.args[0].value, str)
            ):
                return _n.args[0].value
            # `Path('x').unlink()` — the receiver's own call carries the type.
            return _dotted(_n.func)
        return None

    # `f = shutil.rmtree; f('/tmp/x')` — one more binding, resolved the same way
    # the shell scan resolves `RM=rm; $RM -f x`. Not a general dataflow
    # analysis: a name reached through a container, a getattr or a callback
    # stays unresolved, so this narrows the gap rather than closing it. That is
    # the right ambition for an advisory gate aimed at accidents.
    for _node in _ast.walk(_tree):
        if not isinstance(_node, _ast.Assign):
            continue
        _origin = _dotted(_node.value)
        if _origin is None:
            continue
        for _tgt in _node.targets:
            if isinstance(_tgt, _ast.Name):
                _alias.setdefault(_tgt.id, _origin)

    _destructive: set = set()
    _shell_strings: list = []
    _opaque = False
    for _node in _ast.walk(_tree):
        if not isinstance(_node, _ast.Call):
            continue
        _name = _dotted(_node.func)
        if _name is None:
            continue
        if _name in _PY_DESTRUCTIVE_CALLS:
            _destructive.add(_name)
            continue
        if _name not in _PY_SHELL_ESCAPES:
            continue
        _arg = _node.args[0] if _node.args else None
        if isinstance(_arg, _ast.Constant) and isinstance(_arg.value, str):
            _shell_strings.append(_arg.value)
        elif isinstance(_arg, (_ast.List, _ast.Tuple)) and _arg.elts and all(
            isinstance(_e, _ast.Constant) and isinstance(_e.value, str)
            for _e in _arg.elts
        ):
            # The argv form is not shell, but its executable slot is the thing
            # the scan reads, and joining is enough to expose it.
            _shell_strings.append(" ".join(_e.value for _e in _arg.elts))
        else:
            _opaque = True
    return _destructive, _shell_strings, _opaque


def _close_pipes(proc) -> None:
    """Release a finished command's pipe fds.

    ``communicate()`` used to do this, so nothing did after it was replaced —
    and a leaked read end per bash call is a file-descriptor leak in the
    longest-lived process there is. NOT called on the timeout path: there the
    BackgroundJobManager takes the pipes over and keeps draining them.
    """
    for _stream in (getattr(proc, "stdout", None), getattr(proc, "stderr", None)):
        if _stream is None:
            continue
        try:
            _stream.close()
        except (OSError, ValueError) as exc:
            logger.debug("bash: pipe close failed: %s", exc)


# Bounded stream accumulation is shared with the `run_tests` runner — see
# external_llm/common/bounded_capture for why it is not local to bash.
# Re-exported under this name: callers and tests reference
# ``git_tools._BoundedCapture``.
from external_llm.common.bounded_capture import _BoundedCapture  # noqa: E402


def _truncate_bash_output(content: str, max_chars: int, true_len: int = 0) -> str:
    """Truncate bash output to fit the token budget while preserving head+tail.

    pytest/traceback core diagnostics (``short test summary info``, ``N failed``,
    FAILED list) are placed at the **end (tail)** of the output. Truncating only the
    head would cause ``failure_context._try_parse_pytest`` to miss the core markers
    and fall through to UnknownError, so half is allocated to head and half to tail
    to preserve tail diagnostics.

    When non-ASCII (CJK/JSON) content is prevalent, the per-character token cost is
    higher, so the character cap is proportionally reduced.

    ``true_len`` is how many characters the command actually produced, which is no
    longer ``len(content)`` now that the capture is bounded while it reads. Only
    the reported figure uses it — the slicing still operates on what is here.
    Without it a command that printed 108 MB would be described as having lost the
    ~130 KB the capture happened to be holding.
    """
    if not content or len(content) <= max_chars:
        return content
    # ASCII ~3 chars/token, CJK ~1.5 chars/token. To maintain the same token budget,
    # non-ASCII output must be truncated at fewer characters.
    _sample = content[:4000]
    _ascii_ratio = (sum(ch.isascii() for ch in _sample) / len(_sample)) if _sample else 1.0
    _cap = max_chars if _ascii_ratio > 0.7 else int(max_chars * 0.5)
    # _cap <= max_chars, so len(content) > max_chars (the guard above) implies
    # len(content) > _cap; this branch is unreachable.
    if len(content) <= _cap:  # pragma: no cover
        return content
    _truncated = max(true_len, len(content)) - _cap
    _half = _cap // 2
    return (
        content[:_half]
        + f"\n... [truncated {_truncated} chars (middle) — bash output exceeded the "
        f"~{max_chars // 3000}K-token budget. Showing head+tail; pytest/traceback "
        f"diagnostics live at the tail. Re-run with a narrower filter "
        f"(grep, or `wc -c`/`wc -l` to size it first).]\n"
        + content[-_half:]
    )


_SHELL_SEPARATOR_SPLIT_RE = _re.compile(r"([;&|]+)")
# Any run of separator punctuation starts a new command segment. Matching the
# whole class, rather than the enumerated {"|", "&&", "||", ";"} this replaced,
# also covers a bare `&` (`sleep 30 & rm -rf x`) and `;;`, which the fixed set
# let slip past as if they were ordinary words.
# A bare `{` or `}` is a shell GROUPING token (`{ rm x; }`), which otherwise
# lands in the executable slot and leaves the real command behind it looking
# like an argument. Matched as a single character so `find -exec rm {} \;` —
# where the placeholder is the two-character token `{}` — is untouched.
_SEPARATOR_ONLY_RE = _re.compile(r"[;&|]+|[{}]")


_ANSI_C_SIMPLE_ESCAPES = {
    "a": "\a", "b": "\b", "e": "\x1b", "E": "\x1b", "f": "\f",
    "n": "\n", "r": "\r", "t": "\t", "v": "\v",
    "\\": "\\", "'": "'", '"': '"', "?": "?",
}


def _decode_ansi_c(body: str) -> str:
    """Decode the body of a bash ``$'...'`` ANSI-C quoted string.

    Covers the obfuscation vectors that hide a dangerous executable name:
    ``\\xNN`` hex (``$'\\x72\\x6d'`` → ``rm``), ``\\NNN`` octal
    (``$'\\162\\155'`` → ``rm``), and the simple named escapes. ``\\cX`` control
    chars are decoded too. Unknown escapes pass the following character through
    literally (bash's behaviour).
    """
    out: list[str] = []
    i = 0
    n = len(body)
    while i < n:
        c = body[i]
        if c != "\\" or i + 1 >= n:
            out.append(c)
            i += 1
            continue
        nxt = body[i + 1]
        if nxt == "x":
            j = i + 2
            hexs = ""
            while j < n and len(hexs) < 2 and body[j] in "0123456789abcdefABCDEF":
                hexs += body[j]
                j += 1
            if hexs:
                out.append(chr(int(hexs, 16)))
                i = j
            else:
                out.append("\\x")
                i += 2
        elif nxt in "01234567":
            j = i + 1
            octs = ""
            while j < n and len(octs) < 3 and body[j] in "01234567":
                octs += body[j]
                j += 1
            out.append(chr(int(octs, 8) & 0xFF))
            i = j
        elif nxt == "c" and i + 2 < n:
            ctrl = body[i + 2]
            out.append("\x7f" if ctrl == "?" else chr(ord(ctrl.upper()) & 0x1F))
            i += 3
        elif nxt in _ANSI_C_SIMPLE_ESCAPES:
            out.append(_ANSI_C_SIMPLE_ESCAPES[nxt])
            i += 2
        else:
            out.append(nxt)
            i += 2
    return "".join(out)


# Punctuation that the token scan reads as structure rather than as text:
# separators (_SEPARATOR_ONLY_RE), redirects, and substitution/quote openers.
# Neutralised inside a decoded ANSI-C literal — see _shlex_safe_literal.
_LITERAL_STRUCTURAL_CHARS = ";&|{}()<>`$\n"


def _shlex_safe_literal(s: str) -> str:
    """Render *s* as a single, structurally inert shlex word.

    A ``$'...'`` ANSI-C body is one literal word in bash, so it must survive
    the later ``shlex.split`` as one token (internal quotes/spaces must not
    re-split it or unbalance the whole scan). Single-quote wrapping with the
    standard ``'\\''`` escape does that much.

    Quoting alone is NOT enough, though, because ``shlex.split`` STRIPS the
    quotes: a body that decodes to ``;`` comes back as the bare token ``;``,
    which ``_SEPARATOR_ONLY_RE.fullmatch`` cannot tell from a real separator.
    Measured: ``echo $'\\x3b' rm -f /tmp/x`` raised an rm prompt, though bash
    passes that ``;`` to echo as data and never runs rm — data promoted to
    structure, breaking the "quoted prose never counts" contract this scan is
    built on.

    So structural punctuation is replaced with ``_`` here. The substitution is
    lossless for this scan's purposes: the only thing it ever asks of a decoded
    body is whether it NAMES a dangerous executable, and no executable basename
    contains any of these characters. A body that decodes to ``rm`` is
    untouched and still reaches the executable slot.
    """
    for ch in _LITERAL_STRUCTURAL_CHARS:
        s = s.replace(ch, "_")
    return "'" + s.replace("'", "'\\''") + "'"


def _normalize_for_scan(command: str) -> str:
    """Make implicit command boundaries explicit before the policy scan.

    ``shlex.split`` treats a NEWLINE as ordinary whitespace, so a multi-line
    script collapses into one segment and every executable after line 1 is
    classified as an argument of line 1's command. That is the same defect the
    ``;`` handling above exists to fix, for the separator a model reaches for
    most naturally — measured: ``cd build\\nrm -rf artifacts`` ran with no
    approval prompt, while the ``;`` and ``&&`` spellings of it both prompted.

    Subshell parentheses have the same effect by a different route: ``(rm x)``
    tokenises as ``['(rm', 'x)']`` and ``Path('(rm').name`` is not ``rm``.

    Both are rewritten to ``;`` — only in the copy the SCAN reads. The command
    that actually executes is never touched. Quoted regions are left alone, so
    a newline inside ``git commit -m "line1<newline>line2"`` stays part of the
    message, and ``grep "(foo)"`` keeps its literal parens.

    Command substitution is executed by the shell in BOTH quoting contexts, so
    both are surfaced, by different routes:

    * Unquoted — ``$(...)`` needs nothing beyond the ``(``/``)`` rule above, and
      backticks join it in the same rule (opener and closer both become ``;``).
    * Inside double quotes — the opener closes the surrounding double quote with
      ``"`` then inserts a ``;`` boundary, and the matching closer inserts
      ``; "`` to reopen it. Bare ``(`` without ``$`` is left alone here, because
      inside double quotes it is literal text: prose like ``"fix (rm) stuff"``
      must not invent a prompt.

    Single-quoted substitutions are deliberately untouched — ``'$(rm x)'`` and
    ``'`rm x`'`` are literal strings to the shell, so surfacing them would be a
    false prompt, not a catch.

    Substitution bodies (``$(...)`` and backtick) are themselves unquoted command
    lines, so a nested ``$(...)``, nested quoted ``"$(...)"``, or bare ``( ... )``
    inside one is surfaced too — depth-counted for ``$(...)`` and boundary-emitted
    for the backtick body, so the matching closer (not an inner ``)``) reopens the
    surrounding quote. Without that the inner command stayed literal: ``"`echo
    $(rm x)`"`` and ``"`echo "$(rm x)"`"`` both ran with no prompt.

    A genuinely nested command line — ``bash -c "rm x"`` — is handled separately
    in the token scan (``_shell_c_payload_index`` splices the payload back in and
    re-runs every rule on it), not by this normaliser.
    """
    out: list[str] = []
    in_single = in_double = False
    escaped = False
    # Command substitution state inside double quotes (0 = not inside one).
    # Positive = depth of unmatched ``(`` after ``$(``.
    cmdsub_depth = 0
    in_dq_backtick = False  # inside backtick substitution within double quotes
    i = 0
    while i < len(command):
        ch = command[i]

        if escaped:
            out.append(ch)
            escaped = False
            i += 1
            continue
        if ch == "\\" and not in_single:
            out.append(ch)
            escaped = True
            i += 1
            continue

        # ── bash $'...' ANSI-C quoting & ${IFS} word-split glue ─────────
        # Both execute real commands but shlex mis-tokenises them, so the
        # executable name never reaches the scan:
        #   `$'rm'`        → shlex token "$rm" (looks like a variable) → skipped
        #   `$'\x72\x6d'`  → "$rm" again (hex-obfuscated "rm")
        #   `rm${IFS}-f`   → one opaque token → basename is not "rm"
        # The ANSI-C body is decoded to one literal word (bash semantics) and
        # emitted shlex-safe; unquoted ${IFS}/$IFS becomes a split boundary.
        #
        # ANSI-C `$'...'` fires in EVERY unquoted context, which includes the
        # body of a double-quoted `$(...)` / backtick substitution: those bodies
        # are unquoted command lines even though we reached them from inside
        # double quotes, so `$'rm'` there is ANSI-C quoting, not literal text.
        # Without the body inclusion, `"$( $'rm' x )"` ran unprompted — the
        # `$'rm'` was emitted literally and shlex tokenised it as `$rm`. Normal
        # double-quoted text (cmdsub_depth == 0, no backtick) keeps `$'...'`
        # literal, because there it is genuinely not ANSI-C quoting. ${IFS}/$IFS
        # stays top-level-only: it never word-splits inside any double-quoted
        # region, substitution bodies included (the body is unquoted, but IFS
        # splitting happens at WORD-split time on the body's OWN words, which the
        # scan already separates — decoding it here would only mis-space them).
        _unquoted_ctx = (
            not in_single
            and (not in_double or cmdsub_depth > 0 or in_dq_backtick)
        )
        if (
            _unquoted_ctx
            and ch == "$"
            and i + 1 < len(command)
            and command[i + 1] == "'"
        ):
            j = i + 2
            body = []
            while j < len(command):
                c = command[j]
                if c == "\\" and j + 1 < len(command):
                    body.append(c)
                    body.append(command[j + 1])
                    j += 2
                    continue
                if c == "'":
                    break
                body.append(c)
                j += 1
            out.append(_shlex_safe_literal(_decode_ansi_c("".join(body))))
            i = j + 1  # consume the closing quote
            continue
        if (ch == "$" and not in_single and not in_double
                and cmdsub_depth == 0 and not in_dq_backtick):
            if command[i:i + 6] == "${IFS}":
                out.append(" ")
                i += 6
                continue
            if command[i:i + 4] == "$IFS" and (
                i + 4 >= len(command)
                or not (command[i + 4].isalnum() or command[i + 4] == "_")
            ):
                out.append(" ")
                i += 4
                continue

        # ── single-quote boundary ──────────────────────────────────
        if ch == "'" and not in_double:
            in_single = not in_single
            out.append(ch)
            i += 1
            continue

        # ── double-quote boundary ──────────────────────────────────
        if ch == '"' and not in_single:
            if in_dq_backtick or cmdsub_depth > 0:
                # Inside a substitution BODY (backtick or ``$(...)``) a ``"``
                # opens a NESTED quote, but the body still runs as an unquoted
                # command line. Emitting it verbatim re-wraps the body's
                # commands in a quoted token (hiding them from the scan), and
                # resetting state orphans the substitution — both left
                # ``"`echo "$(rm x)"`"`` running unprompted. A boundary
                # surfaces the body's commands while leaving quote balance
                # intact: only the substitution's own closer reopens the
                # surrounding double quote.
                #
                # The ``$(...)`` arm used to fall through to the toggle below,
                # whose ``cmdsub_depth = 0`` reset orphaned the still-open
                # substitution: its real ``)`` closer was then emitted
                # literally and the reopening ``"`` never came, so the scan
                # copy ended quote-unbalanced and shlex refused the WHOLE
                # command ("Invalid command syntax: No closing quotation") —
                # e.g. ``echo "tip: $(git rev-parse "$b")"``, live incident
                # 2026-08-02.
                out.append(" ; ")
                i += 1
                continue
            in_double = not in_double
            out.append(ch)
            i += 1
            continue

        # ── inside single quotes: literal ──────────────────────────
        if in_single:
            out.append(ch)
            i += 1
            continue

        # ── inside double-quoted backtick substitution ─────────────
        # The body is an unquoted command line, exactly like a ``$(...)`` body,
        # so a nested ``$(...)`` or bare ``( )`` inside it really executes and
        # must be surfaced. Emitting the body literally left
        # ``"`echo $(rm x)`"`` running with no prompt: the closing backtick was
        # the only char this branch acted on, so the inner ``$(rm x)`` survived
        # intact, shlex split it into the non-executable token ``$(rm``, and the
        # scan never saw ``rm``. ``$(`` here enters the cmdsub machinery below;
        # that handler reopens the *backtick* state (not the double quote) when
        # its depth returns to 0, which is why the guard excludes cmdsub_depth>0.
        if in_dq_backtick and cmdsub_depth == 0:
            if ch == "`":
                in_dq_backtick = False
                out.append(' ; "')
            elif ch == "$" and i + 1 < len(command) and command[i + 1] == "(":
                out.append(" ; ")
                cmdsub_depth = 1
                i += 2
                continue
            elif ch == "\n" or ch in "()":
                out.append(" ; ")
            else:
                out.append(ch)
            i += 1
            continue

        # ── inside double-quoted $(...) command substitution ───────
        # The BODY of a substitution is an unquoted command line, so the
        # unquoted rule applies inside it too: every ``(``, ``)`` and backtick
        # is a boundary, not a literal. Emitting them literally is what left
        # nested ``"$(echo $(rm x))"`` invisible — the inner ``(`` stayed glued
        # to its command, producing the token ``$(rm``, whose Path(...).name is
        # not ``rm``. Depth still counts the parens so the CLOSER of the outer
        # substitution is the one that reopens the double quote.
        if cmdsub_depth > 0:
            if ch == "(":
                cmdsub_depth += 1
                out.append(" ; ")
            elif ch == ")":
                cmdsub_depth -= 1
                if cmdsub_depth == 0:
                    # Nested inside a double-quoted backtick: the backtick's own
                    # closer reopens the double quote, so do not reopen it here.
                    out.append(" ; " if in_dq_backtick else ' ; "')
                else:
                    out.append(" ; ")
            elif ch in "\n`":
                out.append(" ; ")
            else:
                out.append(ch)
            i += 1
            continue

        # ── inside double quotes (normal, not in a cmdsub) ─────────
        if in_double:
            # $(
            if ch == "$" and i + 1 < len(command) and command[i + 1] == "(":
                out.append('" ; ')
                cmdsub_depth = 1
                i += 2
                continue
            # backtick
            if ch == "`":
                out.append('" ; ')
                in_dq_backtick = True
                i += 1
                continue
            out.append(ch)
            i += 1
            continue

        # ── unquoted ───────────────────────────────────────────────
        # The backtick is here for the same reason ``(`` is: it opens a command
        # substitution that the shell really executes, and without a boundary
        # ``echo `rm -rf x``` tokenises as ``['echo', '`rm', ...]`` whose
        # ``Path('`rm').name`` is not ``rm``. Both the opener and the closer map
        # to ``;``, exactly as ``(`` and ``)`` do — the substitution's body is a
        # command line, and what surrounds it is another one.
        if ch == "\n" or ch in "()`":
            out.append(" ; ")
        else:
            out.append(ch)
        i += 1

    return "".join(out)


def _matches_forbidden_flag(token: str, forbidden: set) -> bool:
    """True if *token* is one of *forbidden*, including its value-carrying forms.

    An exact comparison misses the spellings that actually appear: ``-i.bak``
    (BSD/GNU sed's in-place suffix form) and ``--in-place=.bak`` are the same
    restricted flag as ``-i``. Bundled short flags (``sed -ni``) are NOT matched —
    catching those needs per-flag arity knowledge, and this is an advisory guard
    steering the model toward apply_patch, not a sandbox.
    """
    if token in forbidden:
        return True
    head = token.split("=", 1)[0]
    if head in forbidden:
        return True
    # Short form with an attached value: `-i.bak` for `-i`.
    return any(
        len(flag) == 2 and not flag.startswith("--") and token.startswith(flag)
        for flag in forbidden
    )


def _split_shell_separators(tokens, original_command: Optional[str] = None) -> list:
    """Break unquoted ``;`` / ``&`` / ``|`` runs out of *tokens* into their own items.

    ``shlex.split`` does not treat those as separators, so a segment boundary can
    arrive glued to a word: ``ls;rm -rf x`` tokenises as ``['ls;rm', '-rf', 'x']``
    and ``for f in a b; do rm`` yields ``'b;'``. The scan then never sees the
    boundary, keeps treating the next word as an argument, and the command after
    the separator escapes the danger check.

    When *original_command* is provided (the raw shell command string), the
    function detects separators that appear inside single- or double-quoted
    regions and skips splitting them — preventing false positives on grep/rg
    patterns like ``grep -rn "foo|rm" .`` where ``|`` is literal, not a pipe.
    """
    # ── Build set of separator byte positions inside quoted regions ─────
    # A simple state machine: track whether each byte is inside a quote.
    # This lets us skip split-position bytes that are literal.
    _quoted_seps: set[int] = set()
    if original_command:
        in_single = False
        in_double = False
        for i, ch in enumerate(original_command):
            if ch == "'" and not in_double:
                in_single = not in_single
            elif ch == '"' and not in_single:
                in_double = not in_double
            elif ch in ";&|" and (in_single or in_double):
                _quoted_seps.add(i)

    out = []
    # Cursor into *original_command* so each token is located at or after the
    # previous one. Without it, a token that repeats (``rm a; rm b`` → two
    # ``rm``) always resolves to the FIRST occurrence and the quoted-region
    # lookup is answered about the wrong byte range.
    _cursor = 0
    for token in tokens:
        if any(ch in token for ch in ";&|"):
            # A `>|` is a noclobber redirect operator, not a pipe — splitting it
            # would leave `>` as a bare redirect with `|` as its target (and the
            # real filename as an ordinary argument).  `>>&` and `&>` also contain
            # a shell-separator character without being a segment boundary.
            #
            # The TARGET must still be checked, because _REDIRECT_RE's tail group
            # is `.*` and therefore swallows any separator glued to the filename.
            # shlex emits `echo hi >out.txt;rm -rf /` as the single token
            # `>out.txt;rm`, which matches the operator rule while hiding a whole
            # command: skipping the split there left `rm` out of the executable
            # slot entirely, so the danger gate never saw it (measured — the
            # `;`, `&&` and `|` spellings all ran unprompted). Only a target with
            # no separator of its own is exempt; anything else falls through to
            # the normal split, which at worst over-splits a `>|` (losing the
            # truncation target while still surfacing the command — fail-safe).
            _redir_tok = _REDIRECT_RE.match(token)
            if _redir_tok and not any(ch in _redir_tok.group(2) for ch in ";&|"):
                out.append(token)
                continue
            idx = -1
            if original_command:
                idx = original_command.find(token, _cursor)
                if idx < 0:  # shlex un-escaped it (``'a'"b"``) — not locatable
                    idx = original_command.find(token)
            if idx >= 0:
                _cursor = idx + len(token)
                # Split only separators that are NOT inside a quoted region.
                pieces: list[str] = []
                _buf: list[str] = []
                for j, ch in enumerate(token):
                    if ch in ";&|" and (idx + j) not in _quoted_seps:
                        if _buf:
                            pieces.append("".join(_buf))
                            _buf = []
                        pieces.append(ch)
                    else:
                        _buf.append(ch)
                if _buf:
                    pieces.append("".join(_buf))
                # Trust this result even when it did NOT split — "every
                # separator in this token was quoted" is the answer, not a
                # failure. The previous `if len(pieces) > 1` guard fell through
                # to the naive regex split in exactly that case, re-splitting
                # the token the quote scan had just cleared and making the whole
                # quote-awareness dead code for its own target
                # (``grep -rn "foo|rm" .`` kept raising an rm prompt).
                out.extend(p for p in pieces if p)
                continue
            # Not locatable in the raw command — fall back to the naive split,
            # which over-approximates toward asking (the safe direction).
            out.extend(piece for piece in _SHELL_SEPARATOR_SPLIT_RE.split(token) if piece)
        else:
            out.append(token)
    return out


# `-fdx` → {-f, -d, -x}: a bundle of single-letter flags, the spelling
# DANGEROUS_FLAG_COMBOS is written against. `--force` is excluded by the
# leading-single-dash anchor; so is a negative number (`-5`).
_SHORT_FLAG_BUNDLE_RE = _re.compile(r"^-[A-Za-z]{2,}$")

# `--flag=value` → two tokens so `--size=0` reaches the `{"--size","0"}` combo.
# Restricted to `--[A-Za-z]` so `git commit -m 'x=--hard'` (prose inside an
# assignment) is not mis-split.
_LONG_FLAG_EQ_RE = _re.compile(r"^(--[A-Za-z][A-Za-z0-9_-]*)=(.*)$")
# `-s0` / `-s0K` → `-s` + `0`/`0K` so glued short-flag values reach combos too.
_SHORT_FLAG_VALUE_RE = _re.compile(r"^(-[A-Za-z])(\d.*)$")
# Truncation to zero bytes with an explicit SI suffix (`-s 0K`, `-s 0M`, …).
# Canonicalised to "0" in the combo vocabulary so the single `{"-s","0"}` rule
# covers every zero-equivalent spelling.
_ZERO_SIZE_RE = _re.compile(r"^0[KkMmGgTtPp]?$")

# `>`, `>>`, `2>`, `&>`, `>|` (noclobber override), `<`, and their
# glued-target forms (`>out.txt`). `>|` is a truncating redirect just like
# bare `>` — bash's noclobber override is still a destructive write.
_REDIRECT_RE = _re.compile(r"^(?:\d+|&)?(>>|>\|?|<)(.*)$")


# A `-c` payload can itself be `bash -c "..."`. Bounded by DEPTH, not by a total
# count of expansions: siblings are self-limiting because each one costs input
# characters (`bash -c A; bash -c B; ...` needs the text to spell them all),
# whereas nesting multiplies. A total budget got that backwards — measured, a
# budget of 8 spent on nine harmless `bash -c "ls"` siblings left the `rm` in the
# tenth unscanned, while raising the same budget cost nothing on deep input
# (852 ms → 821 ms at depth 11, because the time is the outer string's length,
# not the expansion count).
_SHELL_C_MAX_DEPTH = 8

# `-c`, and the bundled spellings a model actually writes (`sh -lc`, `sh -ec`).
# The bundle is short flags only, so `--color` is excluded by the anchor.
_SHELL_C_FLAG_RE = _re.compile(r"^-[A-Za-z]*c$")

# Words that occupy the executable slot syntactically without being a command.
# Module-level so the main danger scan and _segment_executable share ONE
# definition — the heredoc receiver check used to reimplement the slot rules
# with a string heuristic, and the two drifted (see _segment_executable).
_SHELL_SYNTAX: frozenset = frozenset({
    "for", "in", "do", "done", "if", "then", "else", "fi", "while", "until",
    "echo",
})


def _segment_executable(text: str) -> tuple:
    """Resolve the executable of the LAST command segment in *text*.

    Returns ``(exe, segment_tokens)`` — ``exe`` is the basename-reduced
    executable name (``None`` if the segment has none), ``segment_tokens`` are
    that segment's own tokens (so callers can inspect its flags).

    This exists so a caller that needs "what command is this text running?"
    gets the SAME answer the danger scan's own executable-slot state machine
    would give, instead of re-deriving it. The heredoc receiver check is the
    first such caller: it previously took "the last whitespace-delimited word
    before ``<<``", which is not the executable whenever a flag, a redirect or
    a wrapper flag-value sits in between — ``bash -s <<EOF`` (the canonical
    spelling) resolved to ``-s`` and the body was blanked, shipping a bypass.

    The rules mirrored here are the slot-consuming ones: command wrappers and
    their positional operands (``chroot /mnt cmd``) and value flags
    (``sudo -u me cmd``), env assignments (``FOO=bar cmd``), numeric wrapper
    args (``timeout 5s cmd``), flags, substitution/assignment tokens, basename
    reduction, and _SHELL_SYNTAX words. Effects the scan performs while walking
    tokens — forbidden-flag rejection, ``-c`` / ``eval`` splicing, danger-name
    accumulation — are deliberately NOT reproduced: this is a pure query.
    """
    try:
        _norm = _normalize_for_scan(text)
    except Exception:
        return (None, [])
    try:
        _parts = shlex.split(_norm)
    except Exception:
        _parts = _norm.split()  # tolerant: unbalanced quotes in prose
    _toks = _split_shell_separators(_parts, _norm)

    # Track the most recently RESOLVED segment. Normalisation rewrites
    # `<(cmd)` into `< ; cmd ;`, so a trailing process substitution leaves the
    # final segment empty; falling back to the nearest resolved one keeps
    # `bash <(echo hi) <<EOF` pointing at `bash` rather than at nothing.
    _last: tuple = (None, [])
    exe: Optional[str] = None
    seg: list = []
    expect_executable = True
    wrapper_ctx: Optional[str] = None
    positional_skip = 0
    skip_flag_value = False

    for token in _toks:
        if _SEPARATOR_ONLY_RE.fullmatch(token):
            if exe is not None:
                _last = (exe, seg)
            exe, seg = None, []
            expect_executable = True
            wrapper_ctx, positional_skip, skip_flag_value = None, 0, False
            continue
        seg.append(token)
        if skip_flag_value:
            skip_flag_value = False
            continue
        if token.startswith("-"):
            if (expect_executable and wrapper_ctx
                    and token in _WRAPPER_VALUE_FLAGS.get(wrapper_ctx, frozenset())):
                skip_flag_value = True
            continue
        if expect_executable and token in _COMMAND_WRAPPERS:
            wrapper_ctx = Path(token).name
            positional_skip = _WRAPPER_POSITIONAL_ARGS.get(wrapper_ctx, 0)
            continue
        if expect_executable and positional_skip > 0:
            positional_skip -= 1
            continue
        if expect_executable and (
            _WRAPPER_NUM_ARG_RE.match(token) or _ENV_ASSIGN_RE.match(token)
        ):
            continue
        if token.startswith("$") or "=" in token:
            expect_executable = False
            continue
        if expect_executable:
            name = Path(token).name or token
            expect_executable = False
            if name in _SHELL_SYNTAX:
                continue
            exe = name
    if exe is not None:
        _last = (exe, seg)
    return _last


def _shell_c_payload_index(tokens: list, start: int) -> Optional[int]:
    """Index of the command string of a ``<shell> -c <payload>``, or None.

    *start* is the position just past the shell's own name. Scans that segment
    only — the segment bound comes from :func:`_segment_end_index`, the same
    primitive ``eval`` uses, so a ``-c`` after the next separator belongs to a
    different command, and ``bash script.sh; grep -c x`` must not hand grep's
    count flag to the shell.
    """
    end = _segment_end_index(tokens, start)
    for i in range(start, end):
        if _SHELL_C_FLAG_RE.match(tokens[i]):
            return i + 1 if i + 1 < len(tokens) else None
    return None


def _trap_payload_index(tokens: list, start: int) -> Optional[int]:
    """Index of ``trap``'s command-string operand, or None.

    *start* is the position just past ``trap`` itself. Scans that segment only,
    like :func:`_shell_c_payload_index` — ``trap 'x' EXIT; rm -rf y`` must not
    swallow the ``rm``.

    Only the FIRST non-flag operand is the action; everything after it is a
    signal name (``EXIT``, ``INT``, ``2``). Two operands are not commands and
    must not be spliced:

    * ``-`` resets the handler to its default,
    * ``''`` ignores the signal.

    Both would otherwise be handed to ``_expand_shell_c_payload``, which returns
    [] for them anyway — they are rejected here so the intent is stated rather
    than relied upon.
    """
    end = _segment_end_index(tokens, start)
    i = start
    while i < end:
        tok = tokens[i]
        # `trap -p`, `trap --` — options precede the action.
        if tok == "--" or (tok.startswith("-") and len(tok) > 1):
            i += 1
            continue
        if tok in ("-", ""):
            return None
        return i
    return None


def _expand_exec_slot_var(token: str, assigned: dict) -> Optional[str]:
    """Expand a leading ``$NAME`` / ``${NAME}`` in an executable-slot *token*.

    Returns the expanded token, or None when nothing is known about the name —
    in which case the caller leaves the token alone and the segment keeps its
    old "executable unknown" behaviour.

    Two sources, in order:

    1. *assigned* — ``NAME=value`` seen EARLIER IN THIS COMMAND. ``RM=rm; $RM -f
       x`` parks the command in a variable one statement before using it, which
       is the `eval` shape again: the command is in the string being scanned.
    2. ``os.environ`` — the command is about to run in this process's
       environment, so a name we can read is a name the shell will resolve the
       same way. This is what makes `$SHELL -c 'rm -rf x'` scannable, and with
       it the whole ``-c`` payload is re-entered rather than dropped.

    Only the executable slot is expanded. Arguments are deliberately left alone:
    a value that happens to read like a dangerous name (``MSG=rm; echo $MSG``)
    must not invent a prompt, and the flag-combo vocabulary is built from
    literal tokens on purpose (see DANGEROUS_FLAG_COMBOS).

    Suffixes survive the expansion — ``${VENV}/bin/python`` becomes a real path
    whose basename the caller reduces to ``python``. Previously the whole token
    was discarded, so the interpreter behind it was never identified.
    """
    _m = _EXEC_SLOT_VAR_RE.match(token)
    if _m is None:
        return None
    name = _m.group("brace") or _m.group("bare")
    value = assigned.get(name)
    if value is None:
        value = os.environ.get(name)
    if not value:
        return None
    expanded = value + token[_m.end():]
    # A value carrying shell metacharacters would change the token's structure
    # (a second command, a redirect), and re-tokenising mid-scan is not this
    # function's job. Refuse rather than feed the scan a token that no longer
    # means what its shape says.
    if any(c in expanded for c in ";|&<>()`$\n"):
        return None
    return expanded


def _segment_end_index(tokens: list, start: int) -> int:
    """Index one past the last token of the segment beginning at *start*.

    Shared segment-bound primitive for ``eval`` payloads and the ``-c`` flag
    scan of :func:`_shell_c_payload_index`: `eval` has no flag marking where
    its payload begins and ends, so the payload is "everything up to the next
    separator". ``eval rm -rf x; ls`` must not swallow the ``ls``.
    """
    for i in range(start, len(tokens)):
        if _SEPARATOR_ONLY_RE.fullmatch(tokens[i]):
            return i
    return len(tokens)


def _is_dangerous_executable(name: str) -> bool:
    """True if the reduced basename *name* is a DANGEROUS_SHELL_COMMANDS hit.

    Exact membership first, then the prefix families in
    ``DANGEROUS_EXECUTABLE_PREFIXES`` — see that constant for why `mkfs` needed
    one (its listed spelling is the only one nobody types).  A prefix match
    still reports the REAL name in the approval prompt, so the user is asked
    about `mkfs.ext4`, not about an abstraction of it.
    """
    if name in _DANGEROUS_SHELL_COMMANDS:
        return True
    return any(
        name.startswith(_pref) and len(name) > len(_pref)
        for _pref in _DANGEROUS_EXECUTABLE_PREFIXES
    )

def _expand_shell_c_payload(payload: str, *, interpreter_names: frozenset | None = None) -> list:
    """Tokenise a ``-c`` payload the same way the top-level command was.

    Returns [] when the payload cannot be read as a command line, which leaves
    the caller's token stream untouched — the payload then stays the opaque
    string it has always been. Failing open matches the rest of this scan: it is
    an advisory prompt, and a payload shlex cannot parse is one the shell will
    reject too.
    """
    if not payload or not payload.strip():
        return []
    _scan = _normalize_for_scan(_blank_heredoc_bodies(payload, interpreter_names=interpreter_names))
    try:
        _parts = shlex.split(_scan)
    except ValueError:
        return []
    return _split_shell_separators(_parts, _scan)


def _segment_flag_combo_hit(exe: Optional[str], tokens: list) -> bool:
    """True if *tokens* (one command segment) satisfy a combo for *exe*.

    Whole-token membership only — see the matching contract in
    ``shell_policy.DANGEROUS_FLAG_COMBOS``. Bundled short flags are expanded so
    ``-fdx``, ``-fd -x`` and ``-f -d -x`` are one rule rather than three.

    Flag=value glue is split before building the vocabulary:
    ``--size=0`` → ``--size`` + ``0``, ``-s0K`` → ``-s`` + ``0K``.
    Zero-size suffix tokens (``0K``, ``0M``, …) are canonicalised to ``"0"`` so
    the single ``{"-s","0"}`` rule covers every zero-equivalent spelling.
    """
    combos = _DANGEROUS_FLAG_COMBOS.get(exe or "")
    if not combos:
        return False
    # Start with every original token so bundle-targeting combos like
    # {"-delete"} still match (the original token is preserved alongside any
    # expansions from the normalisation loop below).
    vocab: set = set(tokens)
    for tok in tokens:
        if _SHORT_FLAG_BUNDLE_RE.fullmatch(tok):
            vocab.update("-" + ch for ch in tok[1:])
            continue  # bundle expanded; the original tok is already in vocab
        m = _LONG_FLAG_EQ_RE.match(tok)
        if m:
            vocab.add(m.group(1))
            val = m.group(2)
            vocab.add(val)
            if _ZERO_SIZE_RE.match(val):
                vocab.add("0")
            continue
        m = _SHORT_FLAG_VALUE_RE.match(tok)
        if m:
            vocab.add(m.group(1))
            val = m.group(2)
            vocab.add(val)
            if _ZERO_SIZE_RE.match(val):
                vocab.add("0")
            continue
        if _ZERO_SIZE_RE.match(tok):
            vocab.add("0")  # canonicalise 0K/0M/… so {"-s","0"} matches
    return any(combo <= vocab for combo in combos)


def _overwrite_targets(exe: Optional[str], tokens: list) -> list:
    """Files a segment's own ARGUMENTS name as overwrite destinations.

    The redirect gate sees ``echo '' > src/main.py`` because a ``>`` is present.
    ``echo '' | tee src/main.py``, ``cp /dev/null src/main.py`` and
    ``mv other src/main.py`` destroy the file just as completely with no
    redirect for it to see, and no dangerous executable name for the name gate
    to catch — measured, all three ran unprompted while the ``>`` spelling of
    the same act prompted.

    Returns raw argument strings; the caller passes them through
    :func:`_truncating_redirect_targets`, which applies the same narrow scoping
    it applies to redirects (in-repo, existing, non-empty). So this deliberately
    does no filesystem work of its own — a target outside the repo, or one that
    does not exist yet, is still never mentioned.

    ``cp``/``mv`` write only their LAST operand; the leading ones are sources
    and are read. Two operands are required for that reading to be true at all
    (``cp f`` is not a command), so a lone operand yields nothing rather than
    being mistaken for a destination.
    """
    mode = _OVERWRITING_EXECUTABLES.get(exe or "")
    if not mode:
        return []
    if any(t in _NON_OVERWRITING_FLAGS.get(exe or "", frozenset()) for t in tokens):
        return []  # -a / -n turns it into an append or a no-clobber
    args = []
    seen_exe = False
    for tok in tokens:
        if not seen_exe and (tok == exe or os.path.basename(tok) == exe):
            # The executable itself is in this list, possibly as a full path
            # (`/usr/bin/tee`) while *exe* is the reduced basename.
            seen_exe = True
            continue
        if tok.startswith("-"):
            continue
        args.append(tok)
    if mode == "last":
        return args[-1:] if len(args) >= 2 else []
    return args


def _truncating_redirect_targets(targets: list, repo_root: str) -> list:
    """Existing in-repo files that a ``>`` redirect would truncate to zero.

    ``echo '' > src/main.py`` destroys a source file as thoroughly as ``rm``
    does, but no *executable* in it is dangerous, so neither the name gate nor
    the flag gate can see it. Scoped deliberately narrowly to keep the prompt
    rare and meaningful:

    * ``>>`` (append) and ``<`` (read) are not truncation and never listed.
    * A target that does not exist yet, or is empty, loses nothing.
    * A target outside the repo is not listed — writing to ``/tmp/out.txt`` or
      ``/dev/null`` is ordinary agent behaviour, and prompting on it would
      train reflexive approval.
    * Unexpanded shell syntax (``$VAR``, globs, ``&1``) is skipped rather than
      guessed at.
    """
    root = Path(repo_root).resolve()
    hits = []
    for target in targets:
        if not target or target[0] in "&$" or any(c in target for c in "*?"):
            continue
        p = Path(target)
        if not p.is_absolute():
            p = root / target
        try:
            if not (p.is_file() and p.stat().st_size > 0):
                continue
            resolved = p.resolve()
            if resolved.is_relative_to(root):
                hits.append(str(resolved.relative_to(root)))
        # Path.is_file() already swallows OSError (returns False), so the
        # explicit stat can only fail by a TOCTOU race.
        except OSError:  # pragma: no cover
            # Unstattable target (broken symlink, permission, ELOOP). Treated
            # as "nothing to truncate" so the gate stays quiet, but logged:
            # every swallowed target here is one the user is NOT asked about.
            logger.debug("Redirect target not stattable: %r", target, exc_info=True)
            continue
    return hits


def _format_command_for_approval(command: str, limit: int = 1200) -> str:
    """Render *command* for a human approval prompt, never hiding text silently.

    The prompt used to cut the command at 200 chars with no marker, so a chained
    command's dangerous part could sit past the cutoff and the user would approve
    text they never saw. A cap is still needed (a here-doc payload can be tens of
    KB), but any elision has to announce itself.
    """
    if len(command) <= limit:
        return command
    return (
        command[:limit]
        + f"\n      … [{len(command) - limit} more chars hidden — approve only if "
        f"the visible part is what you intend]"
    )


class ShellToolsMixin:
    """Mixin providing shell tool implementations for ToolRegistry."""

    def _capture_bounded(self, proc, timeout: float, cap: int):
        """Drain a command's pipes to EOF, keeping only ``cap`` head + tail chars.

        The drain also ends as soon as the DIRECT CHILD has exited, even if a
        grandchild keeps the pipes open (``sleep 200 &`` — "start a dev server
        in the background, then talk to it" is a routine workflow): EOF would
        otherwise never arrive, and a command that already finished would be
        misreported as a timeout and "backgrounded" while dead. See the poll
        loop below.

        Returns ``(stdout_capture, stderr_capture, status)`` where status is
        ``"done"``, ``"cancelled"`` or ``"timeout"``.

        Replaces ``communicate()``, which had no way to bound what it holds: the
        whole output existed in memory before the ~130 KB budget could apply, so
        ``cat`` of a 108 MB log cost +360.6 MB of peak RSS (measured) — the same
        defect the foreground search path was just fixed for, on the tool that
        can run anything.

        Both pipes are polled in one loop for the reason ``communicate`` exists:
        draining stdout to exhaustion while the child fills the stderr pipe
        deadlocks. Waking every ``_CANCEL_POLL_INTERVAL`` keeps ESC responsive
        without a second mechanism — the poll IS the cancel check.

        Decoding is ours now, so it has to match what ``text=True`` did or every
        command emitting ``\\r`` (progress bars) changes shape:
        ``IncrementalNewlineDecoder`` gives the same universal-newline
        translation, statefully, so a ``\\r\\n`` split across two reads still
        collapses to one ``\\n``. Verified against ``communicate(text=True)`` at
        every split point of a mixed ``\\r``/``\\r\\n``/multibyte sample.

        Owning the buffers also retires ``recover_communicate_partial``: partial
        output no longer has to be dug out of CPython's private
        ``_fileobj2output`` after a timeout, because it is already here.
        """
        _out = _BoundedCapture(cap)
        _err = _BoundedCapture(cap)
        _pending: dict = {}
        _decoders: dict = {}
        _sel = _selectors.DefaultSelector()
        try:
            for _stream, _capture in ((proc.stdout, _out), (proc.stderr, _err)):
                if _stream is None:
                    continue
                _sel.register(_stream, _selectors.EVENT_READ)
                _pending[_stream] = _capture
                _decoders[_stream] = _io.IncrementalNewlineDecoder(
                    _codecs.getincrementaldecoder("utf-8")("replace"), True,
                )

            def _pump(_wait: float) -> None:
                """One select+read round: drain what is ready, handle EOF."""
                for _key, _ in _sel.select(timeout=_wait):
                    _stream = _key.fileobj
                    try:
                        _chunk = os.read(_stream.fileno(), _PIPE_READ_CHUNK)
                    except (OSError, ValueError) as exc:
                        logger.debug("bash: pipe read ended early: %s", exc)
                        _chunk = b""
                    _dec = _decoders[_stream]
                    if _chunk:
                        _pending[_stream].feed(_dec.decode(_chunk))
                        continue
                    # EOF. Flush the decoder so a truncated multibyte sequence
                    # becomes a replacement char rather than vanishing.
                    _pending[_stream].feed(_dec.decode(b"", True))
                    _sel.unregister(_stream)
                    del _pending[_stream]

            _deadline = _time.monotonic() + timeout
            while _pending:
                _remaining = _deadline - _time.monotonic()
                if _remaining <= 0:
                    return _out, _err, "timeout"
                _pump(min(_CANCEL_POLL_INTERVAL, _remaining))
                # Read off the LIVE config every poll — the design-chat REPL
                # swaps config.cancel_event per turn, so a value captured before
                # the loop goes stale. effective_cancel additionally merges any
                # per-call scope (MCP wait_for timeout / aborted parallel
                # batch): an abandoned bash aborts at the next poll tick
                # instead of running out the whole budget.
                _cancel = effective_cancel(getattr(self.config, "cancel_event", None))
                if _cancel is not None and _cancel.is_set():
                    _close_pipes(proc)
                    return _out, _err, "cancelled"
                # The direct child may have exited while a grandchild keeps the
                # pipes open (`sleep 200 &`): the EOF this loop waits for never
                # arrives, because the grandchild inherited the write end and
                # holds it for its whole (long) life. When the child is gone,
                # everything it will ever write has been written — drain the
                # group's buffered tail for one grace period and finish, instead
                # of burning the whole budget and misreporting a dead command
                # as "timeout → background job".
                if proc.poll() is not None:
                    _grace = _time.monotonic() + _EXIT_GRACE
                    while _pending and _time.monotonic() < _grace:
                        _pump(min(_CANCEL_POLL_INTERVAL, _grace - _time.monotonic()))
                    break
        finally:
            _sel.close()

        # Both pipes are at EOF, so the command has said everything it is going
        # to. Exiting still gets whatever is left of the budget — a command that
        # closes its streams early and keeps working must still reach the
        # background transition at the same moment it used to — plus a floor, so
        # a process that finishes as the budget expires is not backgrounded with
        # its result already in hand.
        try:
            proc.wait(timeout=max(_EXIT_GRACE, _deadline - _time.monotonic()))
        except subprocess.TimeoutExpired:
            return _out, _err, "timeout"
        _close_pipes(proc)
        return _out, _err, "done"

    def _cancel_running_command(
        self, proc, command: str, out_cap=None, err_cap=None,
    ) -> "ToolResult":
        """Tear down a command the user cancelled mid-run, and report it.

        The process is in its own session (``start_new_session=True``), so it is
        not reaped by anything on this side: without an explicit kill it keeps
        running to completion, keeps its side effects, and outlives the agent
        entirely if the CLI is killed. Signals the whole group — SIGTERM, then
        SIGKILL after a short grace — the same escalation ``BackgroundJob.kill``
        uses, so a shell's children go with it rather than reparenting to init.

        Whatever the command printed before the cancel is returned rather than
        dropped: the user stopped the command, which is not a reason to hide what
        it had already said. It arrives as the captures the pump was filling —
        no longer excavated from a CPython-private buffer after the fact.
        """
        import signal as _signal

        _partial = out_cap.text() if out_cap is not None else ""
        _partial_err = err_cap.text() if err_cap is not None else ""
        _true_len = (
            (out_cap.total if out_cap is not None else 0)
            + (err_cap.total if err_cap is not None else 0)
        )
        try:
            # The child is its own session leader (Popen start_new_session=True),
            # so pgid == proc.pid. Kill the stored pid, NOT a re-resolved
            # getpgid(): a shell that exited early (e.g. `cmd &` backgrounded)
            # is reaped by the time ESC lands, getpgid() then raises
            # ProcessLookupError, and the silently-skipped kill would orphan
            # the grandchildren — the exact leak this teardown exists to
            # prevent. killpg() targets the GROUP, which outlives its leader
            # while any member is alive.
            os.killpg(proc.pid, _signal.SIGTERM)
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, _signal.SIGKILL)
                try:
                    proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    logger.warning("cancel: process %s survived SIGKILL", proc.pid)
        except (ProcessLookupError, PermissionError, OSError) as exc:
            # Already gone, or not ours to signal — nothing left to tear down.
            logger.debug("cancel: could not signal process group: %s", exc)
        logger.info("bash: cancelled by user, cmd=%.200s", command)

        _parts = []
        if _partial.strip():
            _parts.append(_partial)
        if _partial_err.strip():
            _parts.append(f"[stderr]\n{_partial_err}")
        if _parts:
            from ..config.thresholds import config as _thresholds
            _joined = "\n".join(_parts)
            _content = _truncate_bash_output(
                _joined,
                _thresholds.tokens.BASH_OUTPUT_MAX_CHARS,
                true_len=max(_true_len, len(_joined)),
            )
        else:
            _content = ""
        return self._make_result(
            ok=False,
            content=_content,
            error="Operation cancelled",
            retryable=False,
            metadata={"cancelled": True, "partial_output": bool(_parts)},
        )

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
                print(f"      Command: {_format_command_for_approval(command)}")
                try:
                    _answer = input("      Approve execution? (y/N): ").strip().lower()
                except (EOFError, KeyboardInterrupt):
                    return False
                else:
                    return _answer in ("y", "yes")
            return False  # non-interactive: deny by default

        _question_data = {
            "question": (
                f"The shell command contains dangerous operations ({dangerous_names}):\n"
                f"```\n{_format_command_for_approval(command)}\n```\n"
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
        except Exception as _exc:
            logger.debug("shell danger approval callback failed — denying: %s", _exc)
            return False  # deny on error (fail-closed)
        else:
            return _verdict == UserApproval.APPROVED

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
            import sys as _sys
            # `pip` on PATH is not a given (this repo's runtime lacks it), and a
            # PATH `pip` may belong to a different interpreter than the one the
            # re-run will use. Install with the interpreter the ORIGINAL command
            # names when it names one (`python3 -m pytest` → `python3 -m pip`),
            # else with this process's own interpreter.
            _m = _re.search(r"(\S*python\S*)\s+-m\s+", original_command)
            if _m:
                _pip_cmd = f"{_m.group(1)} -m pip install " + shlex.join(missing_packages)
            else:
                _pip_cmd = f"{_sys.executable} -m pip install " + shlex.join(missing_packages)
            _inst = _run_bounded_subprocess(
                _pip_cmd, shell=True,
                executable=_BASH_EXECUTABLE, cwd=self.repo_root,
                # Explicit network-install budget: the bare 120s default would
                # kill a cold-cache / large-wheel download mid-flight and report
                # a bogus "pip install failed".
                timeout=300,
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
        # effective_cancel merges the agent-loop ESC event (whole-turn) with
        # any per-call scope this dispatch runs under (MCP wait_for timeout /
        # aborted parallel batch) — the dispatch-entry check must observe both.
        _ce = effective_cancel(getattr(self.config, "cancel_event", None))
        if _ce is not None and _ce.is_set():
            return self._make_result(
                ok=False,
                content="",
                error="Operation cancelled before shell execution",
                execution_time=0.0,
                retryable=False,
            )

        import shlex

        command = (args.get("command") or "").strip()
        # Clamp to the range the tool schema advertises ("default: 120, max: 300").
        # Unclamped, a model-supplied timeout=99999 pins a worker thread for
        # hours; the MCP layer derives its own ceiling from this same bound
        # (asi_mcp_adapter._resolve_mcp_timeout), so the two must agree.
        try:
            timeout = int(args.get("timeout") or _SHELL_TIMEOUT_DEFAULT)
        except (TypeError, ValueError):
            timeout = _SHELL_TIMEOUT_DEFAULT
        timeout = max(1, min(timeout, _SHELL_TIMEOUT_MAX))

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
            _already_excluded = {m.group(1) for m in _FIND_EXCLUDED_RE.finditer(command)}
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


        # Scan a copy with heredoc BODIES blanked and newlines / subshell parens
        # turned into explicit separators; the string that EXECUTES stays
        # untouched below.
        # NOTE: _re is the module-level `import re as _re` (see top of file).
        # Do NOT re-import here — a local `import re as _re` makes Python treat
        # _re as a function-local name across the WHOLE body, so the earlier
        # _re.search() calls (find -o grouping, above) raise UnboundLocalError
        # before this line ever runs.
        _scan_command = _normalize_for_scan(_blank_heredoc_bodies(command, interpreter_names=_SHELL_INTERPRETERS))
        # A heredoc body is script text, not shell syntax, so an apostrophe in
        # its prose ("don't") leaves shlex with an unbalanced quote. Blanking
        # the bodies removes that, but an opener whose body never starts (the
        # `<<EOF` is the final line) can still reach shlex unbalanced — so the
        # tolerant split is kept for commands that carry an opener at all.
        # Detected with the same _HEREDOC_OPENER_RE the blanking uses, so the
        # two cannot disagree about what a heredoc is (the previous inline
        # `<<\s*['\"]?\w` missed `<<-DELIM`, which the blanker handles).
        try:
            parts = shlex.split(_scan_command)
        except Exception as e:
            if not _HEREDOC_OPENER_RE.search(command):
                return self._make_result(ok=False, content="", error=f"Invalid command syntax: {e}")
            parts = _scan_command.split()

        # The empty-command guard above fires first (strip()), and shlex.split
        # never yields [] for a non-empty input.
        if not parts:  # pragma: no cover
            return self._make_result(ok=False, content="", error="Empty command")

        dangerous_executables = set()
        expect_executable = True
        segment_exe: Optional[str] = None  # executable of the segment being scanned
        # Per-segment token accumulator. The flag-combo check consults THIS,
        # not the raw command string: a combo is only meaningful once the
        # segment's own executable is known, and a raw-string regex cannot tell
        # `git reset --hard` from `echo "--hard"` or from a commit message that
        # merely says "--hard". See _segment_flag_combo_hit.
        segment_tokens: list = []
        # `NAME=value` bindings seen so far in THIS command, used to resolve a
        # later `$NAME` sitting in an executable slot. Command-wide rather than
        # per-segment: `RM=rm; $RM -f x` binds in one segment and uses it in the
        # next, which is the whole shape being closed. A prefix assignment
        # (`RM=rm cmd`) is scoped to one command in real shell semantics, so
        # carrying it forward can only make the scan look at MORE candidates,
        # never fewer — the safe direction for an advisory gate.
        _var_assignments: dict = {}
        flag_combo_exes: set = set()
        # `python -c` payload findings — see _python_payload_effects.
        python_destructive: set = set()
        python_opaque_escape = False
        redirect_targets: list = []
        expect_redirect_target = False
        # Destinations named as arguments rather than after a `>` — see
        # _overwrite_targets. Collected per segment, checked with the redirect
        # targets against the same in-repo/existing/non-empty scoping.
        overwrite_targets: list = []
        # Wrapper state, all scoped to the segment being scanned: which wrapper
        # opened it (its flag table), how many positional operands still precede
        # the real command, and whether the previous token was a flag whose value
        # is the next token.
        wrapper_ctx: Optional[str] = None
        positional_skip = 0
        skip_flag_value = False
        segment_is_piped_into = False
        piped_interpreters: set = set()
        # Process substitution (`bash <(echo rm)`) feeds the left-hand command
        # via a FIFO — the same "unreadable payload" shape as `curl | sh`.
        # Detected per-segment from the normalised token stream (NOT from the raw
        # command string): after normalisation `<(cmd)` becomes `< ; cmd ;`,
        # so a bare `<` whose next token is a separator `;` is a procsub.
        segment_has_procsub = False
        procsub_interpreters: set = set()

        def _close_segment() -> None:
            """Evaluate the finished segment's flag combos and procsub (call before reset)."""
            if _segment_flag_combo_hit(segment_exe, segment_tokens):
                flag_combo_exes.add(segment_exe)
            # Overwrite destinations named as ARGUMENTS (tee/cp/mv), which carry
            # no `>` for the redirect collector to have seen. Evaluated here for
            # the same reason the flag combos are: the segment's full token list
            # is only complete once it closes.
            overwrite_targets.extend(_overwrite_targets(segment_exe, segment_tokens))
            # Process substitution is detected mid-segment (the `<` redirect
            # appears AFTER the executable), so the check runs here at segment
            # close, not at the executable-assignment point.
            #
            # The `-c` suppression reads THIS segment's own accumulated tokens,
            # not a forward scan from _ti. By the time this runs _ti has already
            # advanced past the separator, so it points at the NEXT segment —
            # which, for `<shell> <(body)`, is the substitution BODY. A forward
            # scan therefore asked "does the body contain -c?", and any body
            # carrying an ordinary `-c` flag silenced the prompt: measured,
            # `bash <(curl -c jar http://x)`, `bash <(grep -c foo f)` and
            # `bash <(echo sh -c ls)` all ran unprompted. `curl -c`, `grep -c`,
            # `wc -c`, `sort -c` and `tar -c` are all common spellings, so this
            # was reachable without contriving anything.
            #
            # segment_tokens is the right source: it is the same per-segment
            # accumulator _segment_flag_combo_hit consults, and it holds the
            # interpreter's OWN flags — `bash -c 'echo hi' <(echo rm)` keeps its
            # suppression (bash runs the string, the FIFO is a mere argument)
            # while the body's flags no longer leak into the decision.
            if (
                segment_has_procsub
                and segment_exe is not None
                and (segment_exe in _SHELL_INTERPRETERS
                     or segment_exe in _STDIN_INTERPRETERS
                     or segment_exe in ("source", "."))
                and not any(_SHELL_C_FLAG_RE.match(t) for t in segment_tokens)
            ):
                procsub_interpreters.add(segment_exe)

        # _scan_command, not command: the splitter locates separators by byte
        # position in the string its tokens came from, so feeding it the raw
        # command after normalising would misalign every position.
        # Materialised, and walked by index, so a `<shell> -c "<payload>"` found
        # mid-scan can splice the payload's own tokens in right here. Re-entering
        # the SAME loop is deliberate: the nested command then gets every rule
        # this scan already implements — wrappers, env assignments, redirects,
        # flag combos, basename reduction — instead of a second, poorer copy of
        # them drifting alongside the original.
        _tokens = _split_shell_separators(parts, _scan_command)
        # Nesting depth of each token, spliced in lockstep with _tokens, so the
        # bound is per-payload rather than per-command — see _SHELL_C_MAX_DEPTH.
        _depths = [0] * len(_tokens)
        _ti = 0
        while _ti < len(_tokens):
            token = _tokens[_ti]
            _token_depth = _depths[_ti]
            _ti += 1
            if _SEPARATOR_ONLY_RE.fullmatch(token):
                _close_segment()
                expect_executable = True
                segment_exe = None
                segment_tokens = []
                expect_redirect_target = False
                wrapper_ctx = None
                positional_skip = 0
                skip_flag_value = False
                # A `|` makes the NEXT segment read this one's stdout. That
                # matters for interpreters: `curl url | sh` executes bytes this
                # scan will never see. Tracked here because the separator is the
                # only place the relationship is visible.
                segment_is_piped_into = "|" in token
                segment_has_procsub = False
                continue
            if expect_redirect_target:
                # Detached target of the `>` seen on the previous token.
                redirect_targets.append(token)
                expect_redirect_target = False
                continue
            _redir = _REDIRECT_RE.match(token)
            if _redir is not None:
                _op, _glued = _redir.group(1), _redir.group(2)
                # ── here-string `<<<` ─────────────────────────────────────────
                # Detected as a `<` redirect whose glued tail starts with `<<`
                # (`<<<` matches group(1)='<', group(2)='<<'; a bare `<` or heredoc
                # `<<` does not). The word a SHELL interpreter is fed on stdin via
                # `<<<` it reads as a command line, so the word is shell code —
                # exactly the `-c` shape — and is re-entered through the same
                # splice path. A non-shell (`cat <<< …`) just prints the string, so
                # only SHELL_INTERPRETERS qualify. Was a documented closable gap:
                # `bash <<< 'rm -rf x'` ran unprompted because `<<<` was read as a
                # plain `<` redirect and the quoted word landed in argument slot.
                if (
                    _op == "<"
                    and _glued.startswith("<<")
                    and _token_depth < _SHELL_C_MAX_DEPTH
                    and (
                        segment_exe in _SHELL_INTERPRETERS
                        or segment_exe in _STDIN_INTERPRETERS
                    )
                ):
                    if len(_glued) > 2:
                        # Glued form `<<<word`. The payload is glued onto the
                        # operator, so there is no separate word to consume:
                        # INSERT the re-entered tokens after the current one
                        # (the `<<<word` token is at _ti-1, already processed by
                        # this redirect block and never revisited). The leading
                        # `;` therefore lands at _ti, where the loop picks it up
                        # next iteration and resets expect_executable — without
                        # that, `rm` was read as an argument of the interpreter
                        # and the glued spelling `bash <<<'rm x'` ran unprompted.
                        _nested = _expand_shell_c_payload(_glued[2:], interpreter_names=_SHELL_INTERPRETERS)
                        if _nested:
                            _repl = [";", *_nested, ";"]
                            _tokens[_ti:_ti] = _repl
                            _depths[_ti:_ti] = [_token_depth + 1] * len(_repl)
                    elif _ti < len(_tokens):
                        # Separated form `<<< word` — replace the payload word in place.
                        _nested = _expand_shell_c_payload(_tokens[_ti], interpreter_names=_SHELL_INTERPRETERS)
                        if _nested:
                            _repl = [";", *_nested, ";"]
                            _tokens[_ti:_ti + 1] = _repl
                            _depths[_ti:_ti + 1] = [_token_depth + 1] * len(_repl)
                    continue
                # ── process substitution `<(` — per-segment ──────────────────
                # After normalisation `<(cmd)` becomes `< ; cmd ;` — the `(`
                # is rewritten to `;`, so a bare `<` whose next token is a
                # separator `;` is a procsub (as opposed to `< file`, where the
                # next token is the filename).  Only `<` with no glued target
                # and no here-string prefix is checked.
                if (
                    _op == "<"
                    and not _glued  # not here-string (`<<<`) and not glued file
                    and _ti < len(_tokens)
                    and _SEPARATOR_ONLY_RE.fullmatch(_tokens[_ti])
                ):
                    segment_has_procsub = True
                if _op in (">", ">|"):  # `>>` appends and `<` reads — neither truncates
                    if _glued:
                        redirect_targets.append(_glued)
                    else:
                        expect_redirect_target = True
                continue
            if skip_flag_value:
                # Value of a wrapper flag seen on the previous token (`-u` in
                # `sudo -u me rm`). Not appended to segment_tokens: it is the
                # flag's operand, not a flag of this segment, and letting a value
                # like `--hard` into the combo vocabulary would invent a prompt.
                skip_flag_value = False
                continue
            if expect_executable and token.startswith("$"):
                # `RM=rm; $RM -f x` / `$SHELL -c '…'` — expand BEFORE any other
                # rule sees the token, so the resolved name gets every one of
                # them (wrapper table, EVAL_BUILTINS, the interpreter `-c`
                # splice) instead of a second, poorer copy. Unresolvable names
                # fall through unchanged and keep the old behaviour.
                _expanded = _expand_exec_slot_var(token, _var_assignments)
                if _expanded is not None:
                    token = _expanded
            segment_tokens.append(token)
            # ── Tokens that PRECEDE a command without being one ──────────────
            # Each of these used to consume the executable slot, so the real
            # command behind them was classified as a mere argument and never
            # reached the danger check: `sudo rm -rf /`, `xargs rm -rf`,
            # `FOO=1 rm -rf x`, `find . -exec rm {} +`, `timeout 5 pkill -f x`
            # and `for f in *; do rm -rf $f; done` all ran unprompted.
            if token in _ARG_COMMAND_INTRODUCERS:
                # `find . -name x -exec rm {} +` — the command follows the flag,
                # from a position where nothing was expected.
                expect_executable = True
                continue
            if Path(token).name in _COMMAND_INTRODUCING_KEYWORDS:
                # `do` / `then` / `else` are followed by a command. Checked
                # regardless of expectation state: the `;` that precedes them is
                # glued to the previous token by shlex, so arriving here with
                # expect_executable already False is the normal case.
                # Exception: `!` is a shell keyword that negates exit codes
                # (`! rm file`), but also a `find` negation flag (`find . ! -name
                # rm`).  Only treat standalone `!` as a command introducer when
                # we're at a segment boundary — `find . ! -name` arrives with
                # expect_executable=False (find consumed it), so we skip it.
                if token == "!" and not expect_executable:
                    continue
                expect_executable = True
                continue
            if token.startswith("-"):
                # ── Forbidden-flag check ─────────────────────────────────────
                # This has to happen HERE. It used to live in the "not an
                # executable" branch below, unreachable because the generic skip
                # `continue`d on every token starting with "-": `sed -i` was
                # advertised as blocked (FORBIDDEN_FLAGS, and "sed (no -i)" in
                # the tool schema) while running unimpeded.
                # Scoped to the current segment's executable rather than every
                # executable seen so far, so `sed x; cat -i` no longer charges
                # cat with sed's restriction.
                _forbidden = _FORBIDDEN_FLAGS.get(segment_exe or "")
                if _forbidden and _matches_forbidden_flag(token, _forbidden):
                    return self._make_result(
                        ok=False, content="",
                        error=f"Flag '{token}' is not allowed for '{segment_exe}'. "
                              f"Use apply_patch for file edits.",
                    )
                # A flag is never the executable, so it must not consume the
                # expectation — `xargs -n1 rm` still has to reach `rm`.
                #
                # ...but its VALUE would, if the value is a separate token. Only
                # while still looking for the executable: once the segment has
                # one, a `-u` belongs to that command and its value is an
                # ordinary argument.
                if (
                    expect_executable
                    and wrapper_ctx
                    and token in _WRAPPER_VALUE_FLAGS.get(wrapper_ctx, frozenset())
                ):
                    skip_flag_value = True
                continue
            if expect_executable and token in _COMMAND_WRAPPERS:
                # Remember WHICH wrapper: its flag table decides whether a later
                # `-u me` swallows one more token. Basename-reduced so
                # `/usr/bin/sudo -u me rm` is the same rule as `sudo -u me rm`.
                wrapper_ctx = Path(token).name
                positional_skip = _WRAPPER_POSITIONAL_ARGS.get(wrapper_ctx, 0)
                if (wrapper_ctx in _WRAPPER_SHELL_C_PAYLOAD
                        and _token_depth < _SHELL_C_MAX_DEPTH):
                    # `runuser -c "rm -rf x"` / `script -c ...` / `flock f -c ...`
                    # hand their payload to a shell, exactly like `bash -c`. They
                    # cannot live in _SHELL_INTERPRETERS because a COMMAND_WRAPPERS
                    # hit is consumed right here and never reaches that check, so
                    # the same splice is applied from inside this branch.
                    _payload_at = _shell_c_payload_index(_tokens, _ti)
                    if _payload_at is not None:
                        _nested = _expand_shell_c_payload(_tokens[_payload_at], interpreter_names=_SHELL_INTERPRETERS)
                        if _nested:
                            _repl = [";", *_nested, ";"]
                            _tokens[_payload_at:_payload_at + 1] = _repl
                            _depths[_payload_at:_payload_at + 1] = (
                                [_token_depth + 1] * len(_repl)
                            )
                continue
            if expect_executable and positional_skip > 0:
                # `chroot /mnt rm -rf x` — the operand before the command is not
                # the command. Consumed here so `rm` still reaches the check.
                positional_skip -= 1
                continue
            if expect_executable and (
                _WRAPPER_NUM_ARG_RE.match(token)      # wrapper arg: `timeout 5s rm`
                or _ENV_ASSIGN_RE.match(token)        # `FOO=bar rm ...`
            ):
                if _ENV_ASSIGN_RE.match(token):
                    # Remember the value so a later `$NAME` in an executable
                    # slot resolves (see _expand_exec_slot_var). Recorded ONLY
                    # here, where the token really is a shell assignment: the
                    # `=` in `make CFLAGS=-O2` is an argument, and treating it
                    # as an assignment would seed the table with words the shell
                    # never binds.
                    _name, _, _value = token.partition("=")
                    _var_assignments[_name] = _value
                continue
            if token.startswith("$") or "=" in token:
                if expect_executable:
                    expect_executable = False
                continue

            if expect_executable:
                # Reduce a path form to its basename BEFORE the policy lookup:
                # `/bin/rm` and `./rm` are the same command as `rm`. The old
                # scan skipped every token containing "/" outright, which made
                # this Path(...).name reduction dead code for the one case it
                # exists for — an absolute path bypassed the gate entirely.
                name = Path(token).name or token
                expect_executable = False
                if name in _SHELL_SYNTAX:
                    continue
                segment_exe = name
                if _is_dangerous_executable(name):
                    dangerous_executables.add(name)
                if name in _EVAL_BUILTINS and _token_depth < _SHELL_C_MAX_DEPTH:
                    # `eval` re-parses its own operands as a command line, so
                    # the payload is every token up to the next separator,
                    # space-joined — which is not an approximation but eval's
                    # actual semantics (see shell_policy.EVAL_BUILTINS).
                    # Spliced through the same fence-and-re-enter path as
                    # `bash -c`, so the payload gets every rule this scan
                    # implements rather than a second, poorer copy of them.
                    _eval_end = _segment_end_index(_tokens, _ti)
                    _nested = _expand_shell_c_payload(" ".join(_tokens[_ti:_eval_end]), interpreter_names=_SHELL_INTERPRETERS)
                    if _nested:
                        _repl = [";", *_nested, ";"]
                        _tokens[_ti:_eval_end] = _repl
                        _depths[_ti:_eval_end] = [_token_depth + 1] * len(_repl)
                if name in _COMMAND_STRING_BUILTINS and _token_depth < _SHELL_C_MAX_DEPTH:
                    # `trap 'rm -rf x' EXIT` — one operand is a command string
                    # the shell runs later, the rest are signal names. Spliced
                    # through the same fence-and-re-enter path as `bash -c` and
                    # `eval`, so the payload gets every rule this scan
                    # implements. Only that ONE operand is replaced: joining the
                    # segment the way `eval` does would append `EXIT` to the
                    # command line and turn `trap - EXIT` into a command.
                    _trap_at = _trap_payload_index(_tokens, _ti)
                    if _trap_at is not None:
                        _nested = _expand_shell_c_payload(
                            _tokens[_trap_at], interpreter_names=_SHELL_INTERPRETERS,
                        )
                        if _nested:
                            _repl = [";", *_nested, ";"]
                            _tokens[_trap_at:_trap_at + 1] = _repl
                            _depths[_trap_at:_trap_at + 1] = (
                                [_token_depth + 1] * len(_repl)
                            )
                if (
                    (name in _SHELL_INTERPRETERS or name in _STDIN_INTERPRETERS)
                    and segment_is_piped_into
                    and _shell_c_payload_index(_tokens, _ti) is None
                ):
                    # `curl -s url | sh` — the interpreter runs whatever the
                    # upstream produced, which is not in this string and cannot
                    # be recovered by any amount of parsing. Unlike the other
                    # known-open shapes, though, the SHAPE alone is enough: an
                    # unreadable payload handed to a shell is exactly the case
                    # where "we cannot tell" should mean "ask", not "allow".
                    # Excluded when a `-c` payload is present — then the code IS
                    # visible, was spliced above, and stdin is ignored, so the
                    # prompt would be noise.
                    piped_interpreters.add(name)
                if name in _PYTHON_INTERPRETERS and _token_depth < _SHELL_C_MAX_DEPTH:
                    # A `python -c` payload is not shell, so it is not spliced —
                    # but it IS readable, with Python's own parser. What comes
                    # back is either an effect this gate already asks about
                    # under another spelling (`shutil.rmtree` is `rm -rf`) or a
                    # command handed back to the shell, and the latter is
                    # spliced so every rule in this scan applies to it.
                    _py_at = _shell_c_payload_index(_tokens, _ti)
                    if _py_at is not None:
                        _py_calls, _py_shell, _py_opaque = _python_payload_effects(
                            _tokens[_py_at],
                        )
                        python_destructive.update(_py_calls)
                        if _py_opaque:
                            python_opaque_escape = True
                        for _cmd in _py_shell:
                            _nested_py = _expand_shell_c_payload(
                                _cmd, interpreter_names=_SHELL_INTERPRETERS,
                            )
                            if _nested_py:
                                _repl = [";", *_nested_py, ";"]
                                _tokens[_py_at:_py_at] = _repl
                                _depths[_py_at:_py_at] = (
                                    [_token_depth + 1] * len(_repl)
                                )
                if name in _SHELL_INTERPRETERS and _token_depth < _SHELL_C_MAX_DEPTH:
                    _payload_at = _shell_c_payload_index(_tokens, _ti)
                    if _payload_at is not None:
                        _nested = _expand_shell_c_payload(_tokens[_payload_at], interpreter_names=_SHELL_INTERPRETERS)
                        if _nested:
                            # Fenced by separators so the payload cannot inherit
                            # this segment's executable slot, nor leak its own
                            # trailing state into what follows `bash -c "..."`.
                            _repl = [";", *_nested, ";"]
                            _tokens[_payload_at:_payload_at + 1] = _repl
                            _depths[_payload_at:_payload_at + 1] = (
                                [_token_depth + 1] * len(_repl)
                            )

        _close_segment()  # the last segment ends without a trailing separator

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
            logger.info("User approved dangerous command(s): %s", _danger_str)

        # ── Destructive-effect check (flag combos + truncating redirects) ──
        # Both are cases where no *executable name* is dangerous, so the gate
        # above cannot see them: `git reset --hard` and `echo '' > src/main.py`
        # destroy work as thoroughly as `rm` does. Collected during the token
        # scan, per segment, so the reason names the segment's real executable.
        _effect_reasons: list = []
        if flag_combo_exes:
            _effect_reasons.append(
                ", ".join(sorted(e for e in flag_combo_exes if e)) + " (dangerous flags)"
            )
        if piped_interpreters:
            _effect_reasons.append(
                "pipes into " + ", ".join(sorted(piped_interpreters))
                + " (the piped code is not visible to this check)"
            )
        if procsub_interpreters:
            _effect_reasons.append(
                "process substitution feeds " + ", ".join(sorted(procsub_interpreters))
                + " (the substitution output is not visible to this check)"
            )
        if python_destructive:
            _effect_reasons.append(
                "python -c calls " + ", ".join(sorted(python_destructive))
                + " (deletes files, same as rm)"
            )
        if python_opaque_escape:
            _effect_reasons.append(
                "python -c hands a command it builds at runtime to the shell "
                "(the command is not visible to this check)"
            )
        _truncated = _truncating_redirect_targets(redirect_targets, self.repo_root)
        if _truncated:
            _effect_reasons.append(
                "output redirection truncates " + ", ".join(sorted(set(_truncated)))
            )
        # Same check, same scoping, for destinations named as arguments. Kept a
        # separate reason so the prompt says which act it is — "cp overwrites
        # src/main.py" is a different sentence from "redirection truncates" it,
        # and the user is being asked to recognise the command they got.
        _overwritten = _truncating_redirect_targets(overwrite_targets, self.repo_root)
        if _overwritten:
            _effect_reasons.append(
                "overwrites " + ", ".join(sorted(set(_overwritten)))
            )
        if _effect_reasons:
            _danger_str = "; ".join(_effect_reasons)
            _approval = self._request_shell_danger_approval(_danger_str, command)
            if not _approval:
                return self._make_result(
                    ok=False, content="",
                    error=(
                        f"User denied execution of destructive operation: {_danger_str}. "
                        f"Operation cancelled."
                    ),
                )
            logger.info("User approved destructive operation(s): %s", _danger_str)

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

            # Both pipes are drained in one poll loop — the deadlock
            # `communicate()` exists to prevent — but into a BOUNDED capture, so
            # the output the tool is about to cut to ~130 KB never has to exist
            # in full first (`cat` of a 108 MB log: +360.6 MB peak RSS and 0.24 s
            # before, +0.4 MB and 0.03 s after, same answer).
            #
            # The poll is also the cancel check, so ESC reaches a command that is
            # ALREADY RUNNING. That check used to happen only at dispatch entry,
            # so a `cancel_event` set one second into a `bash sleep 12` was not
            # observed until second twelve (measured); with the CLI's default
            # 120 s budget and 300 s ceiling the command kept running — side
            # effects included — long after the user asked it to stop. The CLI's
            # own ESC path abandons the agent thread and returns the prompt,
            # which hid this: the user gets their prompt back while the
            # subprocess keeps going, unowned.
            from ..config.thresholds import config as _thresholds
            _out_cap, _err_cap, _status = self._capture_bounded(
                proc, timeout, _thresholds.tokens.BASH_OUTPUT_MAX_CHARS,
            )
            if _status == "cancelled":
                return self._cancel_running_command(proc, command, _out_cap, _err_cap)
            if _status == "timeout":
                # ── Timeout → background transition ──────────────────────────
                # Instead of returning a timeout error (which wastes the work done),
                # transition the process to background management.
                #
                # What the command printed so far comes straight off the captures
                # and rides the existing `_recovered_*` contract into the job's
                # first drain. No excavation of CPython's private
                # `_fileobj2output` any more — we own the buffer.
                proc._recovered_stdout = _out_cap.text()
                proc._recovered_stderr = _err_cap.text()

                job_id = _bg_mgr.start(command, proc)
                logger.info("bash: timed out, bg job=%s cmd=%.200s", job_id, command)
                return self._make_result(
                    ok=True,
                    content=f"⏳ Command moved to background (Job ID: {job_id}). The command exceeded {timeout}s timeout.",
                    metadata={"background_job_id": job_id},
                )

            stdout = _out_cap.text()
            stderr = strip_malloc_noise(_err_cap.text()).strip()

            parts_out = []
            if stdout:
                parts_out.append(stdout)
            if stderr:
                parts_out.append(f"[stderr]\n{stderr}")
            content = "\n".join(parts_out) or "(no output)"
            # What the command actually printed, which `len(content)` no longer
            # is. Only the truncation NOTICE uses it, so a 108 MB run is
            # described by its real size instead of by the slice we kept.
            _true_len = len(content) + _out_cap.dropped + _err_cap.dropped

            #── bash output size restriction ─────────────────────────────────────
            # Prevent context token explosion from large output (git diff, find, rg -r, etc.).
            # Limit managed as a single BASH_OUTPUT_MAX_CHARS threshold
            # (NO hardcoding — mismatch between threshold and actual cap would defeat tuning).
            # Head+tail preservation logic is encapsulated in _truncate_bash_output and tested.
            content = _truncate_bash_output(
                content, _thresholds.tokens.BASH_OUTPUT_MAX_CHARS, true_len=_true_len,
            )

            # rg/grep etc. return exit code 1 for "no match" (exit code 2 is the real error).
            # exit code 1 + no stderr = normal execution but no result → not an error.
            # If stdout has meaningful output, treat exit code != 0 as ok.
            #   - find: partially fails due to permission denied after finding some files
            #   - rg/grep --with-filename: exits abnormally after match due to internal error (SIGPIPE etc.)
            # stderr and exit code are included in content for LLM visibility.
            # If stdout is empty and exit code != 0, treat as a real failure.
            ok = proc.returncode == 0 or (proc.returncode == 1 and not stderr) or (proc.returncode >= 128 and bool(stdout.strip()))

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
            # Tail preview — merge BOTH streams: a verdict may land on stderr
            # only (test-failure summaries), so the old stdout-priority pick
            # (stdout or stderr) hid it whenever stdout held boilerplate.
            # Each stream keeps a tail slice so neither can swallow the other
            # within the one-line budget.  Single line: newlines collapsed.
            if j.stdout and j.stderr:
                _slice = 64
                preview = (
                    f"out: {j.stdout[-_slice:]} | err: {j.stderr[-_slice:]}"
                ).replace("\n", "\\n")
            else:
                preview = (j.stdout or j.stderr or "").replace("\n", "\\n")
            if preview:
                lines.append(f"{j.job_id:<14} {j.status:<12} {elapsed:<10} {cmd}")
                lines.append(f"{'':<38}│ {preview[-160:]}")
            else:
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
        # Clamp to the same bound bash's `timeout` uses (SHELL_TIMEOUT_MAX=300),
        # which the schema advertises and the MCP layer derives its ceiling
        # from (asi_mcp_adapter._INNER_TIMEOUT_TOOLS). Unclamped, a model-
        # supplied wait_timeout=99999 would pin this thread — and since `job`
        # is in _SERIAL_TOOLS, the whole turn — for hours.
        wait_timeout = max(0.0, min(wait_timeout, float(_SHELL_TIMEOUT_MAX)))

        _bg_mgr = self._get_bg_manager()

        if wait_timeout > 0:
            # Read the LIVE cancel event (the design-chat REPL swaps
            # config.cancel_event per turn, so a value captured earlier goes
            # stale — same pattern as _capture_bounded's poll loop) and merge
            # any per-call scope: an abandoned job wait returns at the next
            # poll tick instead of pinning the serial-turn thread.
            _cancel = effective_cancel(getattr(self.config, "cancel_event", None))
            info = _bg_mgr.wait_for_completion(
                job_id, timeout=wait_timeout, cancel_event=_cancel
            )
            if _cancel is not None and _cancel.is_set():
                raise AgentCancelled("job wait cancelled by user") from None
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
        content = "\n".join(parts)

        # F1: bound the render at the same budget as every other tool result.
        # The live buffer accumulates up to _OUTPUT_BUF_CAP (2 MiB) per stream
        # — 35x the bash cap — and even the reaped ring's 32 KiB tail is far
        # past it; an unbounded job payload would blow the turn's context
        # budget the way bash did before _truncate_bash_output (same
        # head+tail preserve, same notice — the cancel/rerun paths below
        # already do this).  `job` is in _SERIAL_TOOLS, so this protects the
        # whole turn, not just one message.
        # F5-followup: true_len uses the captures' `total` counters (what the
        # process actually printed) rather than len() of the capped text — a
        # job that wrote 100 MB must not be reported as having lost the ~130 KB
        # the bounded capture happened to retain.
        from ..config.thresholds import config as _thresholds
        content = _truncate_bash_output(
            content, _thresholds.tokens.BASH_OUTPUT_MAX_CHARS,
            true_len=info.stdout_total + info.stderr_total,
        )
        return self._make_result(ok=True, content=content)

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

