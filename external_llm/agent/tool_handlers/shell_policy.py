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

# ─── Executable-position tracking (danger-gate support) ───────────────────
# The gate identifies the *executable* of each command segment, so anything that
# occupies that slot without being the command hides the real one behind it.
# Every name below used to swallow the slot, leaving `sudo rm -rf /` and
# `xargs rm -rf` classified as harmless `sudo`/`xargs` invocations.

# Programs that execute the command that follows them.
COMMAND_WRAPPERS: frozenset = frozenset({
    "sudo", "doas", "env", "nohup", "setsid", "stdbuf", "nice", "ionice",
    "time", "timeout", "gtimeout", "xargs", "command", "exec", "builtin",
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

# Known limits of this scan — it is an advisory prompt, not a sandbox (bash is
# unrestricted by design). Indirection defeats any static reading of the command
# string, so these still reach the shell unprompted:
#   • variable/substitution indirection — `$CMD -rf x`, `$(echo rm) -rf x`
#   • a payload interpreted by another shell — `bash -c "rm -rf x"`, `ssh host rm`
#   • wrapper flags that take a value — `sudo -u me rm -rf x` (the value lands in
#     the executable slot before the real command)
# Treat the gate as a guardrail against plausible mistakes, not as containment.

# Bounds for the `bash` tool's own timeout argument. Advertised verbatim in the
# tool schema and enforced in git_tools._tool_shell_exec; the MCP ceiling is
# derived from the same numbers (asi_mcp_adapter._resolve_mcp_timeout), so all
# three must read them from here rather than restating the literals.
SHELL_TIMEOUT_DEFAULT: int = 120
SHELL_TIMEOUT_MAX: int = 300
