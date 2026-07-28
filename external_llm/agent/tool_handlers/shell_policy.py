"""Shell execution policy constants shared by tool_registry and git_tools.

Default-open model: ALL shell commands are allowed by default.
Only specific dangerous operations require user approval.
"""

# Commands that are allowed but have certain forbidden flags.
FORBIDDEN_FLAGS: dict = {
    "sed": {"-i", "--in-place"},  # in-place edit bypasses apply_patch pipeline
}

# Flag combinations that require user approval, even when the executable
# itself is not inherently dangerous.  Keys are executable basenames;
# values are lists of token sets — approval is requested when EVERY token of
# any one set appears among the arguments of a segment whose executable is
# that key.
#
# This is the "middle ground" between FORBIDDEN_FLAGS (always blocked) and
# DANGEROUS_SHELL_COMMANDS (prompt on the executable name): it asks the user
# when specific flags turn a normally-safe command destructive.
#
# ── Matching contract (git_tools._segment_flag_combo_hit) ────────────────
# Membership is tested against the *tokens of one command segment*, never
# against the raw command string, and only after that segment's executable is
# known.  Two consequences the earlier raw-string regex got wrong:
#   • The executable must actually be the one keyed here — `echo "--hard"` and
#     `sort -s -k 0 f.txt` no longer masquerade as `git`/`truncate`.
#   • Tokens match WHOLE, so quoted prose never counts: the `--hard` inside
#     `git commit -m 'force push --hard fix'` is one token of English, not a
#     flag, and `--force-with-lease` is not `--force` (prompting on the safe
#     alternative to `--force` would be exactly the reflexive-approval training
#     the `kill` note below warns about).
# Bundled short flags are expanded before the test (`-fdx` → `-f -d -x`), so a
# set of single-letter flags covers every spelling without enumerating them.
DANGEROUS_FLAG_COMBOS: dict[str, list[frozenset[str]]] = {
    "git": [
        frozenset({"--hard"}),          # git reset --hard — discards the worktree
        frozenset({"--force"}),         # git push --force — rewrites remote history
        frozenset({"-f", "-d"}),        # git clean -fd/-fdx — removes untracked files
        frozenset({"checkout", "--"}),  # git checkout -- . — discards uncommitted edits
        frozenset({"restore", "--"}),   # git restore -- . — the modern equivalent
    ],
    # ``find -delete`` removes every match, with no confirmation of its own.
    "find": [
        frozenset({"-delete"}),
    ],
    # ``truncate -s 0`` empties a file in place.
    "truncate": [
        frozenset({"-s", "0"}),
    ],
    # NOTE: `dd` is deliberately absent — it is already in
    # DANGEROUS_SHELL_COMMANDS, so it prompts on the executable name alone and
    # a combo entry could only ever be redundant.  (The old `frozenset({"of"})`
    # entry never matched anyway: the token is `of=/dev/sda`, not `of`.)
}

# Commands that require user approval before execution (destructive operations).
# When the LLM requests one of these, the system blocks it and asks the LLM
# to obtain explicit user consent before proceeding.
#
# Membership test is the *executable name only* (see the token scan in
# git_tools._tool_shell_exec), so a command earns a place here when it is
# dangerous by nature rather than dangerous under some flag — anything whose
# blast radius depends on a flag belongs in FORBIDDEN_FLAGS instead.
#
# Matching is on the reduced basename, EXACT — see DANGEROUS_EXECUTABLE_PREFIXES
# for the one family where exact matching was the wrong test.
DANGEROUS_SHELL_COMMANDS: frozenset = frozenset({
    "rm",
    # Process killers that select targets by NAME or PATTERN, not by PID: their
    # blast radius is the whole machine and cannot be narrowed by inspecting
    # arguments. A `pkill -f "sleep 30"` meant to reap one test child also kills
    # an unrelated build script, another agent session, or the parent shell whose
    # cmdline merely contains the pattern.
    # `kill` is deliberately absent: it takes explicit PIDs, which is the narrow,
    # legitimate form an agent needs for reaping jobs it started. Prompting on it
    # would train the user to approve reflexively, which is worse than the
    # residual `kill -9 -1` risk (that one is target-shaped, not name-shaped, so
    # this executable-name set cannot express it anyway).
    "pkill",
    "killall",
    # Whole-device / filesystem destroyers. A coding agent never legitimately
    # needs these, so the prompt costs nothing and the mistake is unrecoverable.
    "mkfs",
    "dd",
    # Machine-level state changes. These normally need sudo and would fail, but
    # passwordless sudo exists and the confirmation is free.
    "shutdown",
    "reboot",
    "halt",
})

# Executable-name PREFIXES that are dangerous for the same reason their base
# name is, spelled as a family rather than enumerated.
#
# `mkfs` earned its place in DANGEROUS_SHELL_COMMANDS above, but that decision
# was unreachable for every spelling anyone actually uses: the filesystem is
# selected by the program NAME (`mkfs.ext4`, `mkfs.xfs`, `mkfs.vfat`, …), and
# exact basename membership matches none of them. Measured before this entry
# existed: `mkfs /dev/sda1` prompted, `mkfs.ext4 /dev/sda1` ran unprompted —
# so the rule fired only for the one spelling nobody types.
#
# Kept deliberately small. A prefix is a blunt instrument (it matches every
# future `mkfs.*`, which is the point here but would be reckless for, say,
# `rm`), so a name belongs here only when the suffix is a VARIANT SELECTOR for
# the same operation rather than a different program.
DANGEROUS_EXECUTABLE_PREFIXES: frozenset = frozenset({
    "mkfs.",
})

# Builtins that re-parse their ARGUMENTS as a command line. Unlike the
# SHELL_INTERPRETERS `-c` form there is no flag marking the payload: `eval`
# concatenates every remaining operand with spaces and runs the result, so the
# scan reconstructs the payload the same way — every token up to the next
# separator (git_tools._segment_end_index), space-joined — and re-enters it
# through the same splice path `bash -c` uses.
#
# This is NOT the "genuine indirection" the known-limits note below describes.
# The command is right there in the string being scanned — it was simply parked
# in an argument slot, which is exactly the `bash -c "rm -rf x"` shape. All four
# spellings ran with no prompt before this existed:
#
#     eval "rm -rf ~/x"      eval 'rm -rf ~/x'
#     eval rm -rf ~/x        eval rm "-rf" ~/x
#
# Space-joining is not an approximation of eval — it IS eval's semantics
# (POSIX: "concatenate the arguments, separated by spaces, then parse and
# execute"), so quoting that the OUTER shell already consumed is genuinely gone
# by the time eval runs. `eval git commit -m "reset --hard"` really does hand
# git a bare `--hard` operand, and the flag-combo prompt it now raises is
# therefore correct, not a false positive. Quotes that survive into the payload
# (`eval "echo 'reset --hard'"`) are re-parsed by shlex and stay one token, so
# the "quoted prose never counts" contract above still holds.
EVAL_BUILTINS: frozenset = frozenset({
    "eval",
})

# ─── Executable-position tracking (danger-gate support) ───────────────────
# The gate identifies the *executable* of each command segment, so anything that
# occupies that slot without being the command hides the real one behind it.
# Every name below used to swallow the slot, leaving `sudo rm -rf /` and
# `xargs rm -rf` classified as harmless `sudo`/`xargs` invocations.

# Programs that execute the command that follows them.
COMMAND_WRAPPERS: frozenset = frozenset({
    "sudo", "doas", "env", "nohup", "setsid", "stdbuf", "nice", "ionice",
    "time", "timeout", "gtimeout", "xargs", "command", "exec", "builtin",
    "chroot",
})

# Shell keywords after which a command begins (handled separately from
# COMMAND_WRAPPERS: these are syntax, and the scan already reduces basenames
# before consulting them).
COMMAND_INTRODUCING_KEYWORDS: frozenset = frozenset({
    "if", "while", "until", "then", "do", "else", "elif", "!",
})

# Flags whose *value* is a command, reached from argument position.
ARG_COMMAND_INTRODUCERS: frozenset = frozenset({
    "-exec", "-execdir", "-ok", "-okdir",
})

# Shells whose `-c` argument is another command line. The scan re-enters that
# payload rather than treating it as an opaque string, so `bash -c "rm -rf x"`
# reaches the same checks the un-nested spelling does. Only shells belong here:
# `python -c` and `perl -e` also take code, but it is not SHELL code, and
# scanning it with shell rules would classify its contents wrongly.
SHELL_INTERPRETERS: frozenset = frozenset({
    "sh", "bash", "zsh", "dash", "ksh", "ash", "mksh", "busybox",
    # `su -c "rm -rf x"` hands its payload to the target user's shell, so the
    # payload is shell code by the same argument. `su someuser -c "..."` works
    # too — the scan looks for `-c` anywhere in the segment, not at a fixed
    # position — and a `su` with no `-c` simply has no payload to re-enter.
    "su",
})

# Wrapper flags that consume the NEXT token as their value. Without this the
# value lands in the executable slot and the real command behind it is read as a
# mere argument: `sudo -u me rm -rf x` scanned `me` as the executable and never
# looked at `rm` (measured — it ran unprompted, as did `env -u FOO rm`,
# `timeout -s KILL 5 rm` and `xargs -I {} rm`).
#
# Only the SEPARATED spelling needs listing. `sudo --user=me rm` and `sudo -uroot
# rm` already worked: the value is glued to the flag, so nothing extra occupies
# the executable slot. Exact-token matching therefore covers the gap without
# needing per-flag arity for the glued forms.
WRAPPER_VALUE_FLAGS: dict = {
    "sudo": frozenset({
        "-u", "-g", "-p", "-C", "-D", "-h", "-r", "-t", "-U",
        "--user", "--group", "--prompt", "--close-from", "--chdir",
        "--host", "--role", "--type", "--other-user",
    }),
    "doas": frozenset({"-u", "-C"}),
    "env": frozenset({"-u", "-C", "-S", "--unset", "--chdir", "--split-string"}),
    "timeout": frozenset({"-s", "-k", "--signal", "--kill-after"}),
    "gtimeout": frozenset({"-s", "-k", "--signal", "--kill-after"}),
    "nice": frozenset({"-n", "--adjustment"}),
    "ionice": frozenset({"-c", "-n", "-p", "-P", "--class", "--classdata", "--pid"}),
    "stdbuf": frozenset({"-i", "-o", "-e", "--input", "--output", "--error"}),
    "xargs": frozenset({
        "-I", "-i", "-n", "-P", "-d", "-E", "-L", "-s", "-a",
        "--replace", "--max-args", "--max-procs", "--delimiter", "--eof",
        "--max-lines", "--max-chars", "--arg-file",
    }),
    "time": frozenset({"-o", "-f", "--output", "--format"}),
}

# Wrappers that consume POSITIONAL operands before the command. `chroot /mnt rm
# -rf x` put `/mnt` in the executable slot, so `rm` was never checked. Keyed by
# how many operands precede the command.
WRAPPER_POSITIONAL_ARGS: dict = {
    "chroot": 1,
}

# Known limits of this scan — it is an advisory prompt, not a sandbox (bash is
# unrestricted by design). Indirection defeats any static reading of the command
# string, so these still reach the shell unprompted:
#   • variable/substitution indirection — `$CMD -rf x`, `$(echo rm) -rf x`
#   • a payload interpreted by a NON-shell — `python -c "os.system('rm -rf x')"`
#   • a payload interpreted on another HOST — `ssh host rm -rf x`
#   • a wrapper flag this file does not know takes a value — the table above is
#     enumerated per wrapper, so an unlisted one puts its value in the executable
#     slot. Bounded by WRAPPER_VALUE_FLAGS being the only thing to extend.
#
# Four entries have LEFT this list, each with the mechanism that replaced it:
#   • `bash -c "rm -rf x"` → SHELL_INTERPRETERS, re-entered as a command line
#   • `eval "rm -rf x"` → EVAL_BUILTINS, re-entered the same way
#   • `"$(rm x)"`, backticks, and their nested forms →
#     git_tools._normalize_for_scan
#   • `sudo -u me rm -rf x` → WRAPPER_VALUE_FLAGS / WRAPPER_POSITIONAL_ARGS
# What remains ABOVE is genuine indirection: the command is not IN the string
# being scanned, so no amount of parsing recovers it.
# Treat the gate as a guardrail against plausible mistakes, not as containment.
#
# Separately, these are shapes where the command IS recoverable (or the shape
# alone is enough) and the gate simply does not look yet. All measured running
# unprompted on 2026-07-29; listed so they are known gaps rather than decisions:
#   • a shell payload delivered other than by `-c` — `bash <<< 'rm -rf x'`,
#     `echo 'rm -rf x' | bash`, `curl -s url | sh`. The herestring form is
#     recoverable (the payload is in the string); the pipe forms are not, so
#     closing those means gating the SHAPE "pipes into a shell interpreter"
#     rather than reading the payload.
#   • command wrappers absent from COMMAND_WRAPPERS — `flock`, `parallel`,
#     `script`, `unbuffer`, `watch`, `runuser`, `proot` each still swallow the
#     executable slot the way `sudo`/`nohup` did before they were listed.
#   • executables that fit DANGEROUS_SHELL_COMMANDS' own criterion but are not
#     in it — `shred`, `srm` (unrecoverable erasure, the `dd`/`mkfs` category).
#     A policy call, not a parsing gap.
# `source evil.sh` / `. evil.sh` is NOT in that list: its payload lives in a
# FILE, which makes it the same genuine indirection as `$CMD` unless the gate
# starts reading files, which it deliberately does not.

# Bounds for the `bash` tool's own timeout argument. Advertised verbatim in the
# tool schema and enforced in git_tools._tool_shell_exec; the MCP ceiling is
# derived from the same numbers (asi_mcp_adapter._resolve_mcp_timeout), so all
# three must read them from here rather than restating the literals.
SHELL_TIMEOUT_DEFAULT: int = 120
SHELL_TIMEOUT_MAX: int = 300


# ── Verification-command recognition ─────────────────────────────────────────
# Used ONLY by the advisory phase machine (agent_phase_manager) to notice that
# the agent has actually verified its edit, so VERIFY can advance to FINISH.
#
# This is NOT a gate. Nothing is blocked or allowed on the strength of it, so a
# false negative costs a missed phase advance and a false positive costs one
# premature "you may answer now" nudge. That is the opposite risk profile to
# DANGEROUS_SHELL_COMMANDS above, where a miss means an unprompted `rm`, and it
# is why this can stay a name list instead of a parsed command policy.
#
# The transition previously keyed off the `run_lint` / `run_tests` tools. Those
# were removed from AGENT_TOOL_SCHEMAS ("bash equivalents; kept as internal
# dispatch only") but left as the sole route into FINISH, so FINISH became
# unreachable: the model has no schema for either, and get_tool_names()
# validation rejects them if it emits one anyway. Verification now happens
# through `bash`, so that is what this recognises.
#
# Anchored at a command-segment head, exactly like git_tools._PYTEST_CMD_RE
# (kept separate — that one drives pytest-plugin recovery and must stay
# pytest-only). A NEWLINE counts as a segment boundary alongside `;`/`|`/`&&`,
# the lesson of the shell-gate fix: `cd build\npytest` is two commands.
# Consequence of anchoring: `pip install pytest` and `grep pytest .` do not
# match, because there the runner name is an argument, not the executable.
import re as _re

_VERIFICATION_RUNNERS = (
    r"\S*python\S*\s+-m\s+(?:pytest|unittest|ruff|mypy|flake8|pylint)",
    r"pytest|py\.test|tox|nox",
    r"ruff|mypy|flake8|pylint|pyright",
    r"(?:npx\s+)?(?:eslint|tsc|vitest|jest)",
    r"go\s+(?:test|vet)",
    r"cargo\s+(?:test|clippy)",
    r"(?:npm|pnpm|yarn)\s+(?:run\s+)?(?:test|lint|typecheck)",
    r"make\s+(?:test|lint|check)",
    # uv/poetry/pdm/hatch/rye run — wraps a test/lint/typecheck runner.
    # ``python -m pytest`` is the common intermediary; the optional prefix
    # must not match bare ``python`` (e.g. ``uv run python script.py``) —
    # only ``-m`` followed by a known runner counts.
    r"(?:uv|poetry|pdm|hatch|rye)\s+run\s+(?:python\S*\s+-m\s+)?(?:pytest|ruff|mypy|flake8|pylint|eslint|tsc|vitest|jest|unittest|pyright|py\.test|tox|nox)",
)

VERIFICATION_CMD_RE = _re.compile(
    r"(?:^|[|;&\n]|&&|\|\|)\s*(?:" + "|".join(_VERIFICATION_RUNNERS) + r")(?=\s|$)"
)


def is_verification_command(command: str) -> bool:
    """True if *command* runs a test/lint/typecheck runner in any segment.

    See :data:`VERIFICATION_CMD_RE` for scope and for why a loose match is
    acceptable here but would not be in the danger gate.
    """
    return bool(command) and bool(VERIFICATION_CMD_RE.search(command))
