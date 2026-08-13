"""Shell execution policy constants shared by tool_registry and git_tools.

Default-open model: ALL shell commands are allowed by default.
Only specific dangerous operations require user approval.
"""

import re as _re

# Commands that are allowed but have certain forbidden flags.
#
# The entries are the stream editors' in-place flags. `sed -i` was here alone,
# which made the restriction a property of the SPELLING rather than of the act:
# `perl -i -pe 's/x/y/' f.py` edits f.py in place exactly as `sed -i` does, and
# ran unprompted while sed's identical form was rejected with "Use apply_patch"
# (measured by dispatching both). Same act, same answer.
FORBIDDEN_FLAGS: dict = {
    "sed": {"-i", "--in-place"},   # in-place edit bypasses apply_patch pipeline
    "perl": {"-i", "--in-place"},  # perl -i / -i.bak / -pi -e — same act as sed -i
    "ruby": {"-i", "--in-place"},  # ruby -i -pe — the third spelling of the same
}

# Executables whose *argument* names a file they overwrite, with no `>` for the
# redirect gate to see. `_truncating_redirect_targets` applies the same narrow
# scoping it applies to redirects (in-repo, existing, non-empty), so a target
# outside the repo or one that does not exist yet is still never mentioned.
#
# Value is the argument position: "all" = every non-flag argument (tee writes to
# each of them), "last" = only the final one (cp/mv's destination; the leading
# arguments are SOURCES and are read, not written).
#
# Deliberately excluded — `tee -a` and `cp -n`. Appending is not truncation, the
# same call the redirect gate makes for `>>`; the point of this table is to make
# the answer depend on the act, not the spelling, in both directions.
OVERWRITING_EXECUTABLES: dict[str, str] = {
    "tee": "all",
    "cp": "last",
    "mv": "last",
    "install": "last",
}

# Flags that turn an OVERWRITING_EXECUTABLES entry non-destructive.
NON_OVERWRITING_FLAGS: dict[str, frozenset[str]] = {
    "tee": frozenset({"-a", "--append"}),
    "cp": frozenset({"-n", "--no-clobber"}),
    "mv": frozenset({"-n", "--no-clobber"}),
    "install": frozenset(),
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
        frozenset({"push", "-f"}),      # git push -f — the short form of --force
        frozenset({"clean", "-f"}),     # git clean -f — removes untracked files (short form alone)
        frozenset({"-f", "-d"}),        # git clean -fd/-fdx — removes untracked files+dirs
        frozenset({"checkout", "--"}),  # git checkout -- . — discards uncommitted edits (long form)
        frozenset({"checkout", "."}),   # git checkout . — discards uncommitted edits
        frozenset({"checkout", "-f"}),  # git checkout -f — force-switch discarding changes
        frozenset({"restore", "--"}),   # git restore -- . — the modern equivalent (long form)
        frozenset({"restore", "."}),    # git restore . — discards uncommitted edits
        frozenset({"restore", "-W"}),   # git restore -W . — discards worktree changes (BSD compat)
        frozenset({"branch", "-f"}),    # git branch -f — force-reassign a branch ref
    ],
    # ``find -delete`` removes every match, with no confirmation of its own.
    "find": [
        frozenset({"-delete"}),
    ],
    # ``truncate -s 0`` empties a file in place.  The zero-size check in
    # ``_segment_flag_combo_hit`` also matches 0K/0M/0G/etc. and glued
    # forms ``-s0``/``--size=0`` after flag=value normalisation.
    "truncate": [
        frozenset({"-s", "0"}),
        frozenset({"--size", "0"}),
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
    # Secure / unrecoverable FILE erasure. `shred` overwrites in place (and
    # `-u` deletes afterwards) so the original bytes are gone; `srm` is the
    # secure-delete equivalent. The same "unrecoverable, agent never needs it"
    # argument as dd/mkfs applies — the survey at the bottom of this file
    # previously listed these as a deliberate omission; promoted because the
    # mistake is just as permanent and the false-positive cost is nil.
    "shred",
    "srm",
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

# Builtins whose *first operand* is a command string the shell runs later.
#
# `trap 'rm -rf "$tmp"' EXIT` is the `eval "rm -rf x"` shape wearing a different
# hat: the command is right there in the string being scanned, parked in an
# argument slot, and the scan read it as one opaque word. All three spellings
# ran with no prompt (measured 2026-07-29, dispatched through the real
# ToolRegistry):
#
#     trap 'rm -f X' EXIT      trap -- 'rm -f X' EXIT      trap 'rm -f X' INT EXIT
#
# It differs from EVAL_BUILTINS in WHICH operands are the payload. `eval` joins
# all of them; `trap` runs only the first, and the rest are signal names — so
# space-joining would hand the scan `rm -f X EXIT` and, worse, would make a
# reset (`trap - EXIT`) look like a command. git_tools._trap_payload_index
# therefore locates that single operand and splices it alone.
#
# A cleanup trap is the reason this matters in practice rather than in theory:
# `trap 'rm -rf "$workdir"' EXIT` is the idiomatic way to write one, so the
# model reaches for it without any intent to evade a gate.
COMMAND_STRING_BUILTINS: frozenset = frozenset({
    "trap",
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
    # Same slot-swallowing shape as the above, measured running unprompted
    # before they were listed (`flock /tmp/l rm -rf x`, `parallel rm -rf ::: x`,
    # `unbuffer rm -rf x`, `watch rm -rf x`, `runuser -u nobody rm -rf x`).
    # `flock` also takes a positional lock file first — see
    # WRAPPER_POSITIONAL_ARGS.
    "flock", "parallel", "unbuffer", "watch", "runuser",
    # `script` covers BOTH of its divergent spellings once it carries a
    # positional arity of 1 (see WRAPPER_POSITIONAL_ARGS): BSD's
    # `script <file> <command>` puts the command in the slot after the operand
    # is consumed, and GNU's `script -c <command> [file]` is caught by the
    # payload re-entry below. A GNU user writing `script out.log` simply has
    # the log name consumed as the operand, which is harmless.
    "script",
})

# Wrappers whose ``-c`` argument is a SHELL command line, exactly like
# SHELL_INTERPRETERS' — but which also occupy the executable slot, so they are
# listed above rather than there (a COMMAND_WRAPPERS hit is consumed before the
# interpreter check is ever reached). The token scan re-enters the payload for
# these too.
#
#   runuser -c "rm -rf x"     ("pass a single command to the shell with -c")
#   script  -c "rm -rf x"     ("run command rather than interactive shell")
#   flock /tmp/l -c "rm -rf x"
#
# `script` is here and NOT in COMMAND_WRAPPERS on purpose: its positional form
# is platform-divergent (GNU takes a typescript FILE, BSD takes file + command),
# so the operand count cannot be stated once. The `-c` form is unambiguous on
# both, and the positional form stays a known gap rather than a guess.
WRAPPER_SHELL_C_PAYLOAD: frozenset = frozenset({
    "runuser", "script", "flock",
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
# Interpreters that execute stdin when run without arguments (or with a script
# file argument). These are gated by the PIPE rule (`curl url | python3`) and the
# here-string rule (`python3 <<< 'code'`), but NOT by the `-c` splice — their
# `-c`/`-e` payload is not shell code, so re-entering it as a shell command line
# would classify its contents wrongly. Kept separate from SHELL_INTERPRETERS for
# that reason: the two sets differ in which rules apply.
STDIN_INTERPRETERS: frozenset = frozenset({
    "python3", "python", "python2",
    "perl", "ruby", "lua",
    "node",
})
# ── `python -c` — the one non-shell payload that CAN be read ────────────────
# The comment above is still true (a Python payload is not shell code and must
# not be scanned with shell rules), but "not shell" was being read as "not
# checkable", and it is not the same thing. Measured 2026-07-30 by dispatching
# 41 real command shapes at the gate: it prompted on 36 — backticks, $(), xargs,
# find -exec, eval, procsub, here-strings, trap, wrappers, loops, nested
# `bash -c` — and the survivors were exactly these two:
#
#     python3 -c "import shutil; shutil.rmtree('/tmp/x')"     ran unprompted
#     perl -e 'unlink "/tmp/x"'                               ran unprompted
#
# Python is checkable with the parser for the language it is actually written
# in, which is also this repo's stated preference over pattern matching. `perl
# -e` / `ruby -e` / `node -e` stay a documented limit: there is no parser here
# for them, and prompting on the FLAG alone would train exactly the reflexive
# approval the `kill` note above argues against.
PYTHON_INTERPRETERS: frozenset = frozenset({"python", "python3", "python2"})

# Calls whose blast radius is the same as an executable in
# DANGEROUS_SHELL_COMMANDS, so the same answer applies: ask. Matched on the
# RESOLVED dotted name, so every spelling is one entry — `import shutil as sh`,
# `from shutil import rmtree as rt` and `__import__('shutil').rmtree` all land
# on `shutil.rmtree`.
#
# Deliberately narrow. Writing, creating and reading are what the agent is for;
# only deletion and in-place truncation are here, mirroring `rm` / `truncate -s 0`.
PYTHON_DESTRUCTIVE_CALLS: frozenset = frozenset({
    "shutil.rmtree",
    "os.remove",
    "os.unlink",
    "os.rmdir",
    "os.removedirs",
    "os.truncate",
    "pathlib.Path.unlink",
    "pathlib.Path.rmdir",
})

# Calls that hand a command back to the shell. Their argument is re-entered
# through the SAME token scan as a `bash -c` payload, so `python3 -c
# "os.system('rm -rf x')"` reaches every rule the un-nested spelling does
# instead of getting a second, poorer copy of them.
PYTHON_SHELL_ESCAPES: frozenset = frozenset({
    "os.system",
    "os.popen",
    "os.execv", "os.execve", "os.execvp", "os.execvpe", "os.spawnv",
    "subprocess.run",
    "subprocess.call",
    "subprocess.check_call",
    "subprocess.check_output",
    "subprocess.Popen",
    "subprocess.getoutput",
    "subprocess.getstatusoutput",
})

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
    # Separated-spelling flag values for the wrappers added above; without
    # these the value lands in the executable slot and the real command behind
    # it reads as an argument (the `sudo -u me rm` shape).
    "flock": frozenset({"-w", "-E", "--wait", "--timeout", "--conflict-exit-code"}),
    "watch": frozenset({"-n", "--interval"}),
    "parallel": frozenset({
        "-j", "-N", "-n", "-L", "-S", "-d", "-a",
        "--jobs", "--delimiter", "--max-lines", "--arg-file",
    }),
    "runuser": frozenset({"-u", "-g", "-s", "--user", "--group", "--shell"}),
}

# Wrappers that consume POSITIONAL operands before the command. `chroot /mnt rm
# -rf x` put `/mnt` in the executable slot, so `rm` was never checked. Keyed by
# how many operands precede the command.
WRAPPER_POSITIONAL_ARGS: dict = {
    "chroot": 1,
    # `flock <file>|<dir> <command> ...` — the lock target precedes the command
    # (`flock [options] <file> <command>` per its own usage line).
    "flock": 1,
    # `script <file> <command>` (BSD) / `script [options] [file]` (GNU). One
    # operand either way; see the note in COMMAND_WRAPPERS.
    "script": 1,
}

# Known limits of this scan — it is an advisory prompt, not a sandbox (bash is
# unrestricted by design). Indirection defeats any static reading of the command
# string, so these still reach the shell unprompted:
#   • variable indirection whose value is NOT set in the same command and is not
#     in the environment — `$CMD -rf x` with CMD exported by a parent shell.
#     The resolvable half no longer belongs here: `RM=rm; $RM -f x` assigns in
#     the very string being scanned, and `$SHELL -c '…'` reads a variable this
#     process also has, so both are now expanded in the executable slot (see
#     git_tools._expand_exec_slot_var).
#   • substitution OUTPUT used as the executable — `$(echo rm) -rf x`. The body
#     is scanned as its own command line (`echo` — safe), but what the body
#     PRINTS becomes the command, and that is not in the string.
#   • a payload interpreted by a NON-shell — `python -c "os.system('rm -rf x')"`
#   • a payload interpreted on another HOST — `ssh host rm -rf x`
#   • a wrapper flag this file does not know takes a value — the table above is
#     enumerated per wrapper, so an unlisted one puts its value in the executable
#     slot. Bounded by WRAPPER_VALUE_FLAGS being the only thing to extend.
#
# Six entries have LEFT this list, each with the mechanism that replaced it:
#   • `bash -c "rm -rf x"` → SHELL_INTERPRETERS, re-entered as a command line
#   • `eval "rm -rf x"` → EVAL_BUILTINS, re-entered the same way
#   • `trap 'rm -rf x' EXIT` → COMMAND_STRING_BUILTINS, re-entered the same way
#   • `"$(rm x)"`, backticks, and their nested forms →
#     git_tools._normalize_for_scan
#   • `RM=rm; $RM -f x` → git_tools._expand_exec_slot_var
#   • `sudo -u me rm -rf x` → WRAPPER_VALUE_FLAGS / WRAPPER_POSITIONAL_ARGS
# What remains ABOVE is genuine indirection: the command is not IN the string
# being scanned, so no amount of parsing recovers it.
# Treat the gate as a guardrail against plausible mistakes, not as containment.
#
# Separately, these are shapes where the command IS recoverable (or the shape
# alone is enough) and the gate simply does not look yet. Measured running
# unprompted on 2026-07-29; listed so they are known gaps rather than decisions:
#   • `proot <cmd>` — a wrapper, but obscure enough that adding it buys little
#     next to the ones now listed. Left out deliberately rather than missed.
#   • `script <file> <command>` on BSD when the file operand is itself absent
#     or doubled: the arity is stated once (1) and cannot cover both platforms'
#     edge spellings.
#   • executables that fit DANGEROUS_SHELL_COMMANDS' own criterion but are not
#     in it — `shred` and `srm` WERE listed here; both are now in the set (see
#     the block above). Any remaining name is a deliberate policy call, not a
#     parsing gap.
#
# Closed since that survey, each verified against the pre-fix tree:
#   • the seven unlisted command wrappers → COMMAND_WRAPPERS (+
#     WRAPPER_POSITIONAL_ARGS for `flock`/`script`, + WRAPPER_VALUE_FLAGS)
#   • `runuser -c` / `script -c` / `flock … -c` → WRAPPER_SHELL_C_PAYLOAD,
#     re-entered like `bash -c`
#   • `curl url | sh` and friends → gated on the SHAPE. The piped bytes are
#     genuinely unreadable, which is precisely why an unreadable payload handed
#     to a shell should ask rather than allow. Suppressed when a `-c` payload
#     is present, since then the code IS visible and stdin is ignored.
#   • `bash <<< 'rm -rf x'` (here-string) → the `<<<` word a SHELL interpreter
#     is fed on stdin is re-entered as a command line, the same splice path as
#     `-c`. A non-shell (`cat <<< …`) just prints the word and is left alone.
#   • `shred` / `srm` → promoted into DANGEROUS_SHELL_COMMANDS (unrecoverable
#     erasure, the dd/mkfs category).
#   • `"$( $'rm' x )"` (ANSI-C `$'...'` inside a double-quoted command
#     substitution) → the substitution body is an unquoted command line, so the
#     ANSI-C decoder now fires there too (git_tools._normalize_for_scan). Normal
#     double-quoted text keeps `$'...'` literal.
#   • `bash <<EOF ; rm -rf x ; EOF` (heredoc body fed to a shell interpreter)
#     → the blanking is now receiver-aware: when the heredoc's receiver is in
#     SHELL_INTERPRETERS the body IS shell code and is NOT blanked, so the scan
#     sees it. Non-shell receivers (`cat`, `tee`) and non-shell interpreters
#     (`python3` — same known-limit as `python -c`) keep their bodies blanked.
#     The receiver is searched across every word of the opener-line prefix,
#     basename-reduced — reading only the word next to `<<` left `bash -s <<EOF`
#     (the canonical spelling), `bash -x`, `bash 2>/dev/null` and `/bin/bash -s`
#     all bypassing, and made `sudo bash <<EOF` pass by luck rather than by
#     wrapper handling. `-c` on the opener line suppresses it: stdin is ignored.
#   • `source <(curl url)` / `. <(curl url)` (procsub feeding source/dot) →
#     process substitution produces an anonymous FIFO, not a real file, so the
#     "payload lives in a FILE" exclusion does not apply. The procsub gate now
#     includes `source` and `.` alongside SHELL_INTERPRETERS. The dot-source
#     basename reduction (`Path(".").name == ""`) is also fixed so `.` survives
#     as an executable name (git_tools, token scan line ~1654).
#
# `source evil.sh` / `. evil.sh` with a REAL FILE is STILL genuine indirection
# (the payload lives on disk, not in this string) and is deliberately left
# unprompted.
#
# `eval "$(curl url)"` is left unprompted too, but for a WEAKER reason, and the
# distinction matters because it decides whether the entry is ever revisited.
# Its payload is unrecoverable — produced at runtime, never in the scanned
# string — which is the `$CMD` property. But unrecoverability alone stopped
# being this gate's test for silence: `curl url | sh` and `bash <(curl url)`
# have equally unrecoverable payloads and BOTH prompt, on the ground that the
# SHAPE — an opaque payload entering an interpreter through a visible channel —
# is itself statically readable. `eval "$(…)"` has that same readable shape.
# So this is a POLICY call about false positives (`eval "$(cat cfg)"` and
# `eval "$(ssh-agent -s)"` are ordinary and would start prompting), not a
# parsing limit. Revisit it as a policy question; do not file it under `$CMD`.

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
