"""Tests for the shell danger-approval gate and the bash timeout clamp.

The gate is a *default-open* policy: everything runs unless the executable name
appears in ``DANGEROUS_SHELL_COMMANDS``, in which case the user is asked first.
Its blind spot was scope: a machine-wide ``pkill -f <pattern>`` sailed through
unremarked while the ``rm`` in the same command line raised a prompt.

Every test here denies approval, so no dangerous command is ever executed — the
gate short-circuits before Popen. That is asserted explicitly, not assumed.
"""
from __future__ import annotations

import shlex
import time
from pathlib import Path

import pytest

import external_llm.agent.tool_handlers.git_tools as gt_mod
from external_llm.agent.tool_handlers.shell_policy import (
    DANGEROUS_SHELL_COMMANDS,
    SHELL_TIMEOUT_DEFAULT,
    SHELL_TIMEOUT_MAX,
)


class _Recorder:
    """Records approval requests and whether the command reached the shell."""

    def __init__(self):
        self.asked: list[tuple[str, str]] = []
        self.spawned: list[str] = []


@pytest.fixture
def gate(tool_registry, monkeypatch):
    """A registry whose approval prompt is recorded and denied, and whose
    subprocess spawn is recorded rather than performed."""
    rec = _Recorder()

    def _deny(dangerous_names, command):
        rec.asked.append((dangerous_names, command))
        return False

    monkeypatch.setattr(
        tool_registry, "_request_shell_danger_approval", _deny, raising=True
    )

    class _FakePopen:
        """Stands in for a successful, instant command."""

        def __init__(self, command, *a, **kw):
            # The command reaching Popen carries the macOS capability shim
            # prelude (_apply_shell_shims), so tests match on the tail.
            rec.spawned.append(command)
            self.returncode = 0

        def communicate(self, timeout=None):
            rec.timeout = timeout
            return ("", "")

    monkeypatch.setattr(gt_mod.subprocess, "Popen", _FakePopen)
    tool_registry._recorder = rec
    return tool_registry


def _run(reg, command, **args):
    return reg.dispatch("bash", {"command": command, **args})


# ── pattern-scoped process killers ────────────────────────────────────────


@pytest.mark.parametrize(
    "command,expected",
    [
        ('pkill -f "sleep 30"', "pkill"),
        ("pkill -9 python3", "pkill"),
        ("killall Terminal", "killall"),
        ("/usr/bin/pkill -f pytest", "pkill"),  # absolute path → basename match
    ],
)
def test_pattern_killers_require_approval(gate, command, expected):
    """`pkill`/`killall` select targets by name, so their blast radius is the
    whole machine and cannot be narrowed by inspecting arguments."""
    result = _run(gate, command)
    assert gate._recorder.asked, f"no approval requested for: {command}"
    assert expected in gate._recorder.asked[0][0]
    assert not result.ok
    assert expected in result.error
    assert gate._recorder.spawned == [], "denied command still reached the shell"


def test_kill_with_an_explicit_pid_is_not_gated(gate):
    """Deliberate exclusion: `kill <pid>` is the narrow form an agent needs to
    reap a job it started. Prompting on it would train reflexive approval."""
    result = _run(gate, "kill 12345")
    assert gate._recorder.asked == []
    assert result.ok
    assert gate._recorder.spawned[0].endswith("kill 12345")


# ── executable-position bypasses ──────────────────────────────────────────
# Each form below hid the real command from the gate by occupying the
# executable slot with something that merely precedes a command. Plain
# `sudo rm -rf /` ran with no prompt at all.


@pytest.mark.parametrize(
    "command,expected",
    [
        # path form — the basename reduction existed but was unreachable
        ("/bin/rm -rf /tmp/x", "rm"),
        ("./rm -rf /tmp/x", "rm"),
        ("sudo /bin/rm -rf /tmp/x", "rm"),
        # command wrappers
        ("sudo rm -rf /tmp/x", "rm"),
        ("env rm -rf /tmp/x", "rm"),
        ("nohup rm -rf /tmp/x", "rm"),
        ("ls | xargs rm -rf", "rm"),
        ("xargs -n1 rm -rf", "rm"),          # wrapper flag must not consume the slot
        ("timeout 5 pkill -f x", "pkill"),   # wrapper numeric arg
        ("timeout 5s killall foo", "killall"),
        ("nice -n 5 rm -rf /tmp/x", "rm"),
        # env-assignment prefix
        ("FOO=1 rm -rf /tmp/x", "rm"),
        # command-introducing keywords (the `;` is glued to the prior token)
        ("for f in a b; do rm -rf $f; done", "rm"),
        ("if true; then rm -rf /tmp/x; fi", "rm"),
        ("while true; do killall foo; done", "killall"),
        # separators shlex does not split on
        ("ls;rm -rf /tmp/x", "rm"),
        ("cd /tmp&&rm -rf x", "rm"),
        ("sleep 1 & rm -rf /tmp/x", "rm"),   # bare `&` was not a known separator
        ("true || rm -rf /tmp/x", "rm"),
        # a flag whose value is a command
        ("find . -name x -exec rm {} +", "rm"),
        # ANSI-C quoting hides the executable as a shlex "variable" token
        ("$'rm' -rf /tmp/x", "rm"),
        ("$'\\x72\\x6d' -rf /tmp/x", "rm"),  # hex-obfuscated "rm"
        ("$'\\162\\155' -rf /tmp/x", "rm"),  # octal-obfuscated "rm"
        # ${IFS} glues the command and its flags into one opaque token
        ("rm${IFS}-rf${IFS}/tmp/x", "rm"),
    ],
)
def test_executable_position_bypasses_are_closed(gate, command, expected):
    result = _run(gate, command)
    assert gate._recorder.asked, f"BYPASS — no approval requested for: {command}"
    assert expected in gate._recorder.asked[0][0]
    assert not result.ok
    assert gate._recorder.spawned == [], "denied command still reached the shell"


@pytest.mark.parametrize(
    "command",
    [
        "echo hello",
        "git status",
        "ls -la /tmp",
        "ls dd",                     # `dd` as a path argument, not the command
        "pytest -q",
        "timeout 5 pytest -q",
        "sudo -v",
        "python3 -m pytest tests/ -q",
        "git log --oneline | head -5",
        "cat a.txt && cat b.txt",
        "for f in *.py; do echo $f; done",
        "find . -name '*.py' -not -path './.venv/*'",
        "wc -l *.py",
    ],
)
def test_benign_commands_are_not_gated(gate, command):
    """Closing the bypasses must not turn ordinary work into a prompt storm."""
    result = _run(gate, command)
    assert gate._recorder.asked == [], f"false positive on: {command}"
    assert result.ok


@pytest.mark.parametrize(
    "command",
    [
        # `\x3b` is a semicolon; bash hands it to echo as DATA and never runs
        # rm. Decoding it and emitting it raw made shlex yield a bare `;`
        # token, which _SEPARATOR_ONLY_RE cannot distinguish from a real
        # separator — so `rm` landed in the executable slot and prompted.
        # Measured before _shlex_safe_literal neutralised structural chars.
        r"echo $'\x3b' rm -f /tmp/x",
        r"echo $'\073' rm -f /tmp/x",          # octal spelling of the same
        r"echo $'\x26\x26' rm -f /tmp/x",      # `&&`
        r"echo $'\x7c' rm -f /tmp/x",          # `|`
        r"echo $'\x0a' rm -f /tmp/x",          # newline — also a boundary here
        # A decoded quote must not break out of the wrapping and re-open the
        # scan's quote state.
        r"echo $'\x27; rm -f /tmp/x; \x27'",
        r"echo $'it\'s fine'",
    ],
)
def test_decoded_ansi_c_content_is_data_not_structure(gate, command):
    """A ``$'...'`` body is ONE bash word — decoding it must not manufacture
    separators, redirects or quotes out of its content.

    This is the "quoted prose never counts" contract (see
    ``shell_policy.DANGEROUS_FLAG_COMBOS``) applied to the ANSI-C decoder: the
    only question the scan may ask of a decoded body is whether it NAMES a
    dangerous executable. Everything else in it is text.
    """
    _run(gate, command)
    assert gate._recorder.asked == [], (
        f"decoded ANSI-C content was promoted to shell structure: {command!r}"
    )


def test_ansi_c_decoding_still_reaches_the_executable_slot(gate):
    """The neutralisation must not blunt the fix it protects: a body that
    decodes to a dangerous NAME still has to be caught."""
    _run(gate, r"$'\x72\x6d' -rf /tmp/x")
    assert gate._recorder.asked, "hex-obfuscated rm stopped being gated"
    assert "rm" in gate._recorder.asked[0][0]


# ── compound commands ─────────────────────────────────────────────────────


def test_compound_command_reports_every_dangerous_segment(gate):
    """The real case that motivated this: the `rm` was announced, the
    machine-wide `pkill` in the same line was not."""
    command = (
        'pkill -f "sleep 30" 2>/dev/null; cd /tmp && rm -rf _gitlock '
        "&& mkdir _gitlock && git init -q ."
    )
    result = _run(gate, command)
    assert gate._recorder.asked, "no approval requested"
    names = gate._recorder.asked[0][0]
    assert "pkill" in names
    assert "rm" in names
    assert not result.ok
    assert gate._recorder.spawned == []


def test_dangerous_command_after_a_pipe_is_still_seen(gate):
    result = _run(gate, "ls /tmp | grep lock && rm -rf /tmp/_gitlock")
    assert gate._recorder.asked
    assert "rm" in gate._recorder.asked[0][0]
    assert not result.ok


def test_harmless_command_is_never_gated(gate):
    result = _run(gate, "echo hello")
    assert gate._recorder.asked == []
    assert result.ok


def test_dangerous_word_inside_a_quoted_argument_is_not_an_executable(gate):
    """`grep rm ...` names `rm` as data, not as a command to run."""
    result = _run(gate, 'grep -r "rm -rf" .')
    assert gate._recorder.asked == []
    assert result.ok


# ── policy set shape ──────────────────────────────────────────────────────


def test_policy_covers_the_pattern_killers_and_excludes_bare_kill():
    assert {"pkill", "killall"} <= DANGEROUS_SHELL_COMMANDS
    assert "rm" in DANGEROUS_SHELL_COMMANDS
    # See shell_policy: `kill` targets explicit PIDs; its mass form is
    # target-shaped, which an executable-name set cannot express anyway.
    assert "kill" not in DANGEROUS_SHELL_COMMANDS


def test_policy_entries_are_bare_executable_names():
    """The membership test runs against Path(token).name, so an entry carrying a
    flag or a path could never match anything."""
    for name in DANGEROUS_SHELL_COMMANDS:
        assert "/" not in name and not name.startswith("-") and " " not in name


def test_schema_description_lists_the_policy_set():
    """The advertised contract is rendered from the policy, not restated — it
    previously named only `rm` after other commands became gated."""
    from external_llm.agent.tool_schemas import SCHEMA_BASH

    description = SCHEMA_BASH["description"]
    for name in DANGEROUS_SHELL_COMMANDS:
        assert name in description, f"{name} is gated but undocumented"


# ── forbidden flags ───────────────────────────────────────────────────────
# The FORBIDDEN_FLAGS check sat in the "token is not an executable" branch,
# which flags could never reach: the generic skip above it `continue`d on
# everything starting with "-". `sed -i` was advertised as blocked in two places
# and ran unimpeded.


@pytest.mark.parametrize(
    "command",
    [
        "sed -i 's/a/b/' f.py",
        "sed --in-place 's/a/b/' f.py",
        "sed -i.bak 's/a/b/' f.py",          # short form with attached value
        "sed --in-place=.bak 's/a/b/' f.py",  # long form with attached value
        "cat f | sed -i 's/a/b/' f.py",       # second segment
        "sudo sed -i 's/a/b/' f.py",          # behind a wrapper
    ],
)
def test_in_place_sed_is_blocked(gate, command):
    result = _run(gate, command)
    assert not result.ok, f"forbidden flag not enforced: {command}"
    assert "not allowed for 'sed'" in result.error
    assert "apply_patch" in result.error
    assert gate._recorder.spawned == [], "blocked command still reached the shell"


@pytest.mark.parametrize(
    "command",
    [
        "sed -e 's/a/b/' f.py",
        "sed -n '1,5p' f.py",
        "sed 's/a/b/' f.py > out.txt",
        "grep -i pattern f.py",   # -i is only forbidden for sed
        "git diff -i",
        "cat -i f.py",
    ],
)
def test_unrelated_flags_are_allowed(gate, command):
    assert _run(gate, command).ok, f"false block on: {command}"


def test_forbidden_flag_is_scoped_to_its_own_segment(gate):
    """`sed x; cat -i f` must not charge `cat` with sed's restriction — the old
    check tested every executable seen anywhere in the command line."""
    assert _run(gate, "sed x; cat -i f.py").ok


# ── timeout clamp ─────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "requested,expected",
    [
        (None, SHELL_TIMEOUT_DEFAULT),
        (0, SHELL_TIMEOUT_DEFAULT),      # falsy → default, not "no timeout"
        (60, 60),
        (SHELL_TIMEOUT_MAX, SHELL_TIMEOUT_MAX),
        (99999, SHELL_TIMEOUT_MAX),      # clamped: cannot pin a worker for hours
        (-5, 1),
        ("abc", SHELL_TIMEOUT_DEFAULT),  # model-supplied garbage must not raise
    ],
)
def test_timeout_is_clamped_to_the_advertised_range(gate, requested, expected):
    args = {} if requested is None else {"timeout": requested}
    result = _run(gate, "echo hi", **args)
    assert result.ok
    assert gate._recorder.timeout == expected


def test_schema_advertises_the_enforced_bounds():
    from external_llm.agent.tool_schemas import SCHEMA_BASH

    text = SCHEMA_BASH["parameters"]["properties"]["timeout"]["description"]
    assert str(SHELL_TIMEOUT_DEFAULT) in text
    assert str(SHELL_TIMEOUT_MAX) in text


# ── approval prompt legibility ────────────────────────────────────────────


def test_long_command_is_elided_with_an_explicit_marker():
    """A security prompt must not hide the text it is asking about: the old
    200-char cut could place the dangerous part past the cutoff silently."""
    command = "rm -rf /tmp/x && " + ("echo padding; " * 400)
    shown = gt_mod._format_command_for_approval(command)
    assert len(command) > len(shown)
    assert "more chars hidden" in shown
    assert shown.startswith("rm -rf /tmp/x")


def test_short_command_is_shown_verbatim():
    command = 'pkill -f "sleep 30"; rm -rf /tmp/_gitlock'
    assert gt_mod._format_command_for_approval(command) == command


def test_the_previously_truncated_real_command_is_now_fully_visible():
    """Regression for the observed prompt, which cut mid-token at 200 chars."""
    command = (
        'pkill -f "sleep 30" 2>/dev/null; cd /tmp && rm -rf _gitlock && '
        "mkdir _gitlock && cd _gitlock && git init -q . && "
        "git config user.email t@t && git config user.name t && "
        "printf '#!/bin/sh\\nsleep 12 &\\nsleep 1\\n' > .git/hooks/pre-commit && "
        "chmod +x .git/hooks/pre-commit && git commit -q --allow-empty -m x"
    )
    assert len(command) > 200
    shown = gt_mod._format_command_for_approval(command)
    assert shown == command
    assert "git commit" in shown


# ── flag combos: scoped to the segment's own executable ───────────────────
#
# The first implementation searched the RAW command string for each combo's
# flags and never checked that the keyed executable was present, so any command
# containing the words matched: `echo "--hard"` prompted as git, and the
# `truncate` combo {-s, 0} fired on `sort -s -k 0 f.txt`. A gate that prompts on
# ordinary commands trains reflexive approval — the failure mode shell_policy's
# own `kill` note calls worse than the risk it guards.


@pytest.mark.parametrize(
    "command,expected",
    [
        ("git reset --hard HEAD~5", "git"),
        ("git clean -fdx", "git"),            # bundled short flags
        ("git clean -fd", "git"),             # bundle without -x
        ("git clean -f -d -x", "git"),        # separate short flags
        ("git push --force origin main", "git"),
        ("git checkout -- .", "git"),
        ("git restore -- src/", "git"),
        ("find . -name '*.py' -delete", "find"),
        ("truncate -s 0 sample.py", "truncate"),
        ("ls && git reset --hard", "git"),    # combo in a later segment
    ],
)
def test_destructive_flag_combos_require_approval(gate, command, expected):
    result = _run(gate, command)
    assert gate._recorder.asked, f"no approval requested for: {command}"
    assert expected in gate._recorder.asked[0][0]
    assert not result.ok
    assert gate._recorder.spawned == [], "denied command still reached the shell"


@pytest.mark.parametrize(
    "command",
    [
        'echo "--hard"',                      # no git anywhere
        "echo hello -s 0",                    # not truncate
        "sort -s -k 0 file.txt",              # -s and 0 belong to sort
        'python3 -c "print(0)" -s',
        'grep -rn -- "--force" docs/',        # searching FOR the flag text
        "cat notes.md | grep -- -delete",     # -delete is grep's pattern
        "git commit -m 'force push --hard fix'",  # flag words inside a message
        "git push --force-with-lease origin x",   # the SAFE alternative
        "git diff -- src/main.py",            # `--` without checkout/restore
        "git checkout -b feature/x",          # branch creation destroys nothing
    ],
)
def test_flag_combos_do_not_fire_on_unrelated_commands(gate, command):
    result = _run(gate, command)
    assert gate._recorder.asked == [], f"spurious approval prompt for: {command}"
    assert result.ok or gate._recorder.spawned, f"command was blocked: {command}"


def test_bundled_short_flags_expand_to_their_letters():
    from external_llm.agent.tool_handlers.git_tools import _segment_flag_combo_hit

    assert _segment_flag_combo_hit("git", ["clean", "-fdx"])
    assert _segment_flag_combo_hit("git", ["clean", "-fd"])
    assert _segment_flag_combo_hit("git", ["clean", "-f", "-d"])
    # `--force` must NOT be reached by expanding a long flag letter-by-letter.
    assert not _segment_flag_combo_hit("git", ["push", "--force-with-lease"])
    assert not _segment_flag_combo_hit("git", ["status", "-s"])


# ── quoted separators are literal, not segment boundaries ─────────────────


@pytest.mark.parametrize(
    "command",
    [
        'grep -rn "foo|rm" .',      # alternation in a double-quoted pattern
        "grep -rn 'a;rm' .",        # single-quoted
        'rg "build|rm -rf" src/',
        'echo "a&rm"',
    ],
)
def test_quoted_separator_does_not_start_a_new_command(gate, command):
    """A `;`/`|`/`&` inside quotes is data. Splitting on it invented a segment
    whose first word looked like an executable, so `grep -rn "foo|rm" .` asked
    to approve an `rm` that does not exist."""
    _run(gate, command)
    assert gate._recorder.asked == [], f"spurious approval prompt for: {command}"


def test_unquoted_separator_still_starts_a_new_command(gate):
    """The quote-awareness must not weaken the real boundary case."""
    _run(gate, "ls;rm -rf /tmp/x")
    assert gate._recorder.asked, "glued `;rm` escaped the gate"
    assert "rm" in gate._recorder.asked[0][0]


def test_repeated_token_resolves_to_its_own_occurrence(gate):
    """The quoted-region lookup walks a cursor, so the second `rm|x` is judged
    on its own bytes rather than the first occurrence's."""
    _run(gate, "echo 'rm|x' && echo rm|x")
    assert gate._recorder.asked == []


# ── truncating output redirection ─────────────────────────────────────────


@pytest.mark.parametrize("template", ["> {f}", "echo '' > {f}", "cat /dev/null > {f}", "ls && echo x > {f}"])
def test_truncating_redirect_over_an_existing_repo_file_requires_approval(gate, template):
    """`echo '' > src/main.py` destroys a file as thoroughly as `rm` does, but
    no *executable* in it is dangerous, so the name gate cannot see it."""
    command = template.format(f="sample.py")   # created by the temp_repo_root fixture
    result = _run(gate, command)
    assert gate._recorder.asked, f"no approval requested for: {command}"
    assert "truncates" in gate._recorder.asked[0][0]
    assert "sample.py" in gate._recorder.asked[0][0]
    assert not result.ok
    assert gate._recorder.spawned == []


@pytest.mark.parametrize(
    "command",
    [
        "echo hi > brand_new_file.txt",       # target does not exist — creates it
        "echo hi >> sample.py",               # append, not truncate
        "python3 -m pytest -q 2>/dev/null",   # /dev/null is not a repo file
        "cat sample.py > /tmp/copy.py",       # outside the repo
        "cat < sample.py",                    # input redirection
        "grep x sample.py 2>&1",              # fd duplication, not a file
    ],
)
def test_non_truncating_redirects_are_not_gated(gate, command):
    _run(gate, command)
    assert gate._recorder.asked == [], f"spurious approval prompt for: {command}"


# ── implicit command boundaries: newline and subshell parens ──────────────
#
# `shlex.split` treats a newline as ordinary whitespace, so a multi-line script
# collapsed into ONE segment and every executable after line 1 was classified
# as an argument of line 1's command — the same defect the `;` handling was
# built to fix, for the separator a model reaches for most naturally. Measured
# before the fix: `cd build\nrm -rf artifacts` ran with no prompt, while the
# `;` and `&&` spellings of it both prompted.


@pytest.mark.parametrize(
    "command",
    [
        "true\nrm victim.txt",
        "cd build\nrm -rf artifacts",
        "echo start\ncd src\nrm -rf generated\necho done",   # separator on a later line
        "true\r\nrm victim.txt",                             # CRLF
        "true\n\n\nrm victim.txt",                           # blank lines between
        "  \n  rm victim.txt",                               # leading blank line
    ],
    ids=["bare", "build-cleanup", "third-line", "crlf", "blank-lines", "leading-blank"],
)
def test_newline_starts_a_new_command_segment(gate, command):
    _run(gate, command)
    assert [a[0] for a in gate._recorder.asked] == ["rm"], (
        f"newline did not end the previous segment: {command!r}"
    )
    assert gate._recorder.spawned == []


@pytest.mark.parametrize(
    "command",
    [
        "(rm victim.txt)",
        "echo x | (rm victim.txt)",
        "(cd build && rm -rf out)",
        "{ rm victim.txt; }",          # brace group, already word-separated
    ],
    ids=["subshell", "piped-subshell", "subshell-compound", "brace-group"],
)
def test_grouping_does_not_hide_the_executable(gate, command):
    """`(rm x)` tokenises as `['(rm', 'x)']`, and `Path('(rm').name` is not
    `rm` — the paren has to end the segment for the basename lookup to work."""
    _run(gate, command)
    assert "rm" in [a[0] for a in gate._recorder.asked], (
        f"grouping punctuation hid the executable: {command!r}"
    )
    assert gate._recorder.spawned == []


@pytest.mark.parametrize(
    "command",
    [
        "echo one\necho two",                      # multi-line, nothing dangerous
        "git commit -m 'line one\nline two' --allow-empty",   # newline INSIDE a quote
        'git commit -m "fix: rm handling\nsee #12" --allow-empty',
        "grep -rn '(foo)' .",                      # literal parens in a pattern
        'grep -rn "a(b|c)d" src',                  # regex parens AND a pipe
        "python3 -c 'print(1)'",                   # parens inside quoted code
    ],
    ids=["plain-multiline", "quoted-newline", "quoted-newline-dq",
         "literal-parens", "regex-parens", "quoted-code-parens"],
)
def test_normalisation_does_not_invent_prompts(gate, command):
    """Over-prompting is its own failure — it trains reflexive approval, which
    the `kill` note in shell_policy warns about explicitly."""
    _run(gate, command)
    assert gate._recorder.asked == [], f"spurious approval prompt for: {command!r}"


def test_heredoc_body_is_not_scanned_as_commands(gate):
    """The body is data. A `rm` inside it must not prompt — and the header
    still has to be scanned, which the next test covers."""
    _run(gate, "cat > out.txt << 'EOF'\nrm not-a-command\nEOF")
    assert gate._recorder.asked == []


def test_heredoc_header_is_still_scanned(gate):
    _run(gate, "rm victim.txt << 'EOF'\nignored\nEOF")
    assert [a[0] for a in gate._recorder.asked] == ["rm"]


# ── commands AFTER a heredoc body ─────────────────────────────────────────
# The scan used to truncate at the first `<<` and read only what came before,
# so a heredoc anywhere in a script hid every command that followed it. Writing
# a file and then running something is the plainest possible two-step script —
# nothing adversarial is involved, which is what made it worth fixing.


@pytest.mark.parametrize(
    "command,expected",
    [
        ("cat <<EOF > f.txt\nhello\nEOF\nrm -rf victim", "rm"),
        ("cat <<'EOF' > f.txt\nhello\nEOF\nrm -rf victim", "rm"),
        # <<-DELIM: the blanker's _HEREDOC_OPENER_RE has always handled the
        # dash form; the scan's own inline detector did not, so the two
        # disagreed about what a heredoc even was.
        ("cat <<-EOF > f.txt\nhello\n\tEOF\nrm -rf victim", "rm"),
        ("cat <<EOF > f.txt\nhi\nEOF\npkill -f pytest", "pkill"),
    ],
    ids=["bare", "quoted-delim", "dash-delim", "pattern-killer"],
)
def test_a_command_after_the_heredoc_body_is_scanned(gate, command, expected):
    _run(gate, command)
    assert expected in "".join(a[0] for a in gate._recorder.asked), (
        f"command after the heredoc body escaped the gate: {command!r}"
    )


def test_a_flag_combo_after_the_heredoc_body_is_scanned(gate):
    """The effect gate (no dangerous executable name) was hidden by the same
    truncation as the name gate."""
    _run(gate, "cat <<EOF > f.txt\nhi\nEOF\ngit reset --hard")
    assert gate._recorder.asked, "git reset --hard after a heredoc was not gated"


def test_a_truncating_redirect_on_the_opener_line_is_scanned(gate, tmp_path):
    """`cat <<EOF > src/main.py` destroys the file whether or not the body is
    dangerous — and the opener line is exactly what truncation discarded."""
    victim = Path(gate.repo_root) / "real_source.py"
    victim.write_text("print('important')\n")
    _run(gate, "cat <<EOF > real_source.py\nhi\nEOF")
    assert gate._recorder.asked, "heredoc truncating an existing source file was not gated"
    assert victim.read_text() == "print('important')\n", "denied command still ran"


def test_a_new_file_target_is_still_not_gated(gate):
    """The redirect gate stays narrow: a target that does not exist loses
    nothing, and prompting on it would train reflexive approval."""
    _run(gate, "cat <<EOF > brand_new.py\nhi\nEOF")
    assert gate._recorder.asked == []


@pytest.mark.parametrize(
    "command",
    [
        "cat <<EOF > f.txt\nrm -rf everything\nEOF",
        "cat <<EOF > f.txt\nrm -rf everything\nEOF\nls -la",
        "cat <<EOF > f.txt\ngit reset --hard\nEOF",
        # An apostrophe in body prose leaves shlex with an unbalanced quote;
        # blanking the body is what keeps this parseable at all.
        "cat <<EOF > f.txt\ndon't do this\nEOF\nls",
    ],
    ids=["body-rm", "body-rm-then-safe", "body-flag-combo", "body-apostrophe"],
)
def test_the_body_itself_is_still_never_scanned(gate, command):
    """Blanking the body must not turn into scanning it — the whole point of
    the original truncation is preserved."""
    _run(gate, command)
    assert gate._recorder.asked == [], f"body text prompted: {command!r}"


def test_the_executed_command_keeps_its_heredoc(gate):
    """Blanking applies to the SCAN copy only — a blanked body reaching the
    shell would write a file full of spaces."""
    command = "cat <<EOF > f.txt\nhello world\nEOF"
    _run(gate, command)
    assert gate._recorder.spawned, "command never reached the shell"
    assert command in gate._recorder.spawned[-1], "the executed heredoc was altered"


def test_the_executed_command_is_never_rewritten(gate):
    """Normalisation applies to the SCAN copy only — the multi-line script that
    reaches the shell must be byte-identical to what the model sent."""
    command = "echo one\necho two\necho three"
    _run(gate, command)
    assert gate._recorder.spawned, "command never reached the shell"
    assert command in gate._recorder.spawned[-1], (
        "the executed command was altered by scan normalisation"
    )


# ── Command substitution ──────────────────────────────────────────────────
# ``$(rm x)`` and ``` `rm x` ``` ARE executed by the shell in both the
# unquoted and the double-quoted context — the scan must surface all four
# spellings.  The two contexts reach the boundary by different routes
# (the unquoted ``(``/``)``/backtick rule vs the double-quote state machine),
# so they are asserted separately: a regression in one must not be masked by
# the other still working.
#
# Bare ``(`` without ``$`` is prose (e.g. commit messages) and must NOT
# prompt.  Single-quoted ``'$(rm x)'`` is literal and must also not prompt.


@pytest.mark.parametrize(
    "command,expected",
    [
        ('echo "$(rm -rf ~/x)"', "rm"),
        ('echo "$(git reset --hard)"', "git"),
        ('D="$(pkill -f node)"', "pkill"),
        # backtick form
        ('echo "`rm -rf ~/x`"', "rm"),
        # combined: cmdsub inside a larger double-quoted span
        ('echo "before $(rm x) after"', "rm"),
        # NESTED. The body of a substitution is itself an unquoted command line,
        # so the inner opener needs a boundary too. Without one the inner `(`
        # stayed glued to its command and produced the token `$(rm`, whose
        # Path(...).name is not `rm` — the outer fix alone did not reach this.
        ('echo "$(echo $(rm -rf ~/x))"', "rm"),
        ('echo "$(echo `rm -rf ~/x`)"', "rm"),
        ('echo "$(echo $(git reset --hard))"', "git"),
    ],
    ids=["dollar-paren-rm", "dollar-paren-git", "dollar-paren-pkill",
         "backtick-rm", "prefix-suffix", "nested-dollar-paren",
         "nested-backtick", "nested-flag-combo"],
)
def test_command_substitution_in_double_quotes_is_gated(gate, command, expected):
    _run(gate, command)
    assert gate._recorder.asked, (
        f"BYPASS — no approval requested for cmdsub in dq: {command!r}"
    )
    assert expected in gate._recorder.asked[0][0]
    assert gate._recorder.spawned == []


@pytest.mark.parametrize(
    "command,expected",
    [
        # Backtick substitution INSIDE double quotes, whose BODY itself contains a
        # command substitution. The backtick body is an unquoted command line, so
        # the inner $(...) / `...` really executes — but the body used to be emitted
        # verbatim, so the inner command survived intact, shlex split `$(rm` into a
        # non-executable token, and the gate never saw `rm`. Measured: ran=True,
        # asked=[]. Three nesting shapes, each a distinct state-machine path.
        ('echo "`echo $(rm -rf ~/x)`"', "rm"),
        # The deeper form: the inner substitution is itself double-quoted inside the
        # backtick body. The `"` there used to clobber in_dq_backtick mid-body.
        ('echo "`echo \\"$(rm -rf ~/x)\\"`"', "rm"),
        ('echo "`echo \\"$(echo $(rm -rf ~/x))\\"`"', "rm"),
        # flag combo carried inside the buried substitution
        ('echo "`echo \\"$(git reset --hard)\\"`"', "git"),
        # assignment spelling of the same shape
        ('x=`echo $(pkill -f node)`', "pkill"),
        # bare subshell parens inside the backtick body
        ('echo "`(rm -rf ~/x)`"', "rm"),
    ],
    ids=["backtick-body-dollar-paren", "backtick-body-quoted-dollar-paren",
         "backtick-body-nested-dollar-paren", "backtick-body-quoted-flag-combo",
         "backtick-body-assignment", "backtick-body-subshell"],
)
def test_substitution_inside_double_quoted_backtick_is_gated(gate, command, expected):
    """The body of a double-quoted backtick is an unquoted command line.

    Anything executable inside it — ``$(...)``, nested quoted ``"$(...)"``, or a
    bare ``( ... )`` subshell — runs for real, so it must reach the danger check.
    These were all silent executions (ran=True, no prompt) before the backtick-body
    handler surfaced its contents instead of emitting them verbatim.
    """
    _run(gate, command)
    assert gate._recorder.asked, (
        f"BYPASS — no approval requested for cmdsub inside dq backtick: {command!r}"
    )
    assert expected in gate._recorder.asked[0][0]
    assert gate._recorder.spawned == []


@pytest.mark.parametrize(
    "command,expected",
    [
        # Backticks unquoted: the sibling of the ``(``/``)`` rule, and the more
        # natural spelling of the two. Without a boundary the token is
        # ``` `rm ```, whose Path(...).name is not ``rm``, so the gate never
        # saw it — the double-quoted fix above did not reach this path.
        ("echo `rm -rf ~/x`", "rm"),
        ("echo `git reset --hard`", "git"),
        # Backtick body ending the command (no trailing text after the closer).
        ("D=`pkill -f node`", "pkill"),
        # $(...) unquoted — already covered by the paren rule; asserted here so
        # the four spellings live under one contract.
        ("echo $(rm -rf ~/x)", "rm"),
    ],
    ids=["backtick-rm", "backtick-git-hard", "backtick-assignment",
         "dollar-paren-rm"],
)
def test_unquoted_command_substitution_is_gated(gate, command, expected):
    _run(gate, command)
    assert gate._recorder.asked, (
        f"BYPASS — no approval requested for unquoted cmdsub: {command!r}"
    )
    assert expected in gate._recorder.asked[0][0]
    assert gate._recorder.spawned == []


@pytest.mark.parametrize(
    "command",
    [
        # bare paren without $ — prose, not a command
        'git commit -m "fix (rm) stuff"',
        # single-quoted — literal, never executes
        "echo '$(rm -rf ~/x)'",
        "echo '`rm x`'",
        # backtick inside SINGLE quotes is literal to the shell (verified with
        # `bash -n`), so the scan must not read it as a substitution opener.
        "grep -rn 'a`b' .",
        # ESCAPED backticks in double quotes are literal too — this is the one
        # dq spelling that does not execute, and it exercises the ``escaped``
        # flag rather than the backtick branch.
        'echo "a\\`rm -rf x\\`"',
        # rm inside plain dq text (no cmdsub syntax)
        'echo "rm -rf x"',
    ],
    ids=["bare-paren-prose", "single-quoted-dollar", "single-quoted-backtick",
         "single-quoted-backtick-pattern", "escaped-backtick-dq",
         "plain-dq-text"],
)
def test_command_substitution_not_invented_when_literal(gate, command):
    _run(gate, command)
    assert gate._recorder.asked == [], (
        f"spurious approval prompt for literal text: {command!r}"
    )


# ── Nested command lines: `<shell> -c "<payload>"` ────────────────────────
# The payload is another command line, so the scan re-enters it. `bash -c
# "cd sub && rm -rf node_modules"` is a shape a model reaches for naturally
# (one call for a compound job), and it used to reach the shell unprompted.
# The payload is spliced into the SAME token loop, so every existing rule —
# wrappers, flag combos, basename reduction — applies to it too.


@pytest.mark.parametrize(
    "command,expected",
    [
        ('bash -c "rm -rf ~/x"', "rm"),
        ("sh -c 'rm -rf ~/x'", "rm"),
        # the realistic shape: one call doing a compound job
        ('bash -c "cd sub && rm -rf node_modules"', "rm"),
        # bundled short flags — `-lc`/`-ec` still take the payload as their value
        ('bash -lc "rm -rf ~/x"', "rm"),
        # the flag-combo gate must reach into the payload as well
        ('bash -c "git reset --hard"', "git"),
        # payload that is itself a nested shell invocation
        ("""bash -c "bash -c 'rm -rf ~/x'" """.strip(), "rm"),
        # a wrapper in front must not hide the shell
        ('sudo bash -c "rm -rf ~/x"', "rm"),
        # path form reduces to a basename before the interpreter lookup
        ('/bin/sh -c "rm -rf ~/x"', "rm"),
    ],
    ids=["bash-c", "sh-c", "compound", "bundled-lc", "flag-combo",
         "nested-shell", "sudo-wrapped", "abs-path-shell"],
)
def test_shell_c_payload_is_scanned(gate, command, expected):
    _run(gate, command)
    assert gate._recorder.asked, (
        f"BYPASS — no approval requested for shell -c payload: {command!r}"
    )
    assert expected in gate._recorder.asked[0][0]
    assert gate._recorder.spawned == []


@pytest.mark.parametrize(
    "command",
    [
        # `-c` on a NON-shell is not a command line: python's is Python source,
        # and reading it with shell rules would classify its contents wrongly.
        'python3 -c "print(1)"',
        # grep's -c is --count; nothing follows it that is a command
        "grep -c pattern file.txt",
        # a `-c` past the segment boundary belongs to the next command, not to
        # the shell that opened the line
        "bash deploy.sh; grep -c rm file.txt",
        # a shell running a SCRIPT takes no payload to scan
        "bash deploy.sh",
        # harmless payload stays harmless
        'bash -c "ls -la"',
        # degenerate payloads must not raise or invent a prompt
        'bash -c ""',
        "bash -c",
    ],
    ids=["python-c", "grep-count", "grep-c-after-separator", "bash-script",
         "harmless-payload", "empty-payload", "dangling-c"],
)
def test_shell_c_scan_does_not_overreach(gate, command):
    _run(gate, command)
    assert gate._recorder.asked == [], (
        f"spurious approval prompt for: {command!r}"
    )


def test_shell_c_siblings_are_all_scanned(gate):
    """Every `bash -c` on one line is scanned, however many there are.

    The bound on re-entry is DEPTH, and this is why. An earlier spelling used a
    total budget of 8 expansions, which nine harmless `bash -c "ls"` siblings
    exhausted — the `rm` in the tenth was then never looked at. Siblings need no
    budget: each one costs input characters to spell, so the command string
    bounds them on its own.
    """
    command = "; ".join(['bash -c "ls"'] * 29 + ['bash -c "rm -rf ~/x"'])
    _run(gate, command)
    assert gate._recorder.asked, "the 30th sibling's rm was never scanned"
    assert "rm" in gate._recorder.asked[0][0]
    assert gate._recorder.spawned == []


def test_shell_c_nesting_is_scanned_and_terminates(gate):
    """Nested payloads are scanned to ``_SHELL_C_MAX_DEPTH``, and deeper input
    still returns promptly rather than recursing without end.

    Depth is where a bound is actually needed: each level of `bash -c '...'`
    re-quotes the one inside it, so the command string grows ~3x per level
    (measured: 20 chars at depth 1, 354 KB at depth 11). The time is the outer
    string's length, not the recursion — hence a generous depth limit costs
    nothing on the shapes a model actually writes (0.1 ms at depths 1-2).
    """
    payload = "rm -rf ~/x"
    for _ in range(gt_mod._SHELL_C_MAX_DEPTH):
        payload = f"bash -c {shlex.quote(payload)}"
    _run(gate, payload)
    assert gate._recorder.asked, "nesting at the depth limit lost the rm"
    assert gate._recorder.spawned == []

    # One level past the limit: the contract is termination, not a catch.
    deeper = f"bash -c {shlex.quote(payload)}"
    started = time.monotonic()
    _run(gate, deeper)
    assert time.monotonic() - started < 10.0, (
        "scan did not terminate promptly past the depth limit"
    )


# ── `eval` — a command line parked in an argument slot ────────────────────
# `eval` re-parses its operands as a command line, exactly like `bash -c` does,
# but with no flag marking the payload: it concatenates every operand with
# spaces and runs the result. The command is therefore right there in the
# string being scanned, merely sitting where an argument goes — which is what
# separates this from the genuine indirection (`$CMD`, `python -c`) the gate
# does not claim to catch. All four spellings below ran with no prompt at all.


@pytest.mark.parametrize(
    "command,expected",
    [
        # the four spellings, each a distinct token shape after shlex
        ('eval "rm -rf ~/x"', "rm"),
        ("eval 'rm -rf ~/x'", "rm"),
        ("eval rm -rf ~/x", "rm"),
        ('eval rm "-rf" ~/x', "rm"),
        # the realistic shape: one call doing a compound job
        ('eval "cd sub && rm -rf node_modules"', "rm"),
        # the flag-combo gate must reach into the payload too
        ('eval "git reset --hard"', "git"),
        # payload that is itself a nested re-parse, both spellings
        ("""eval "eval 'rm -rf ~/x'" """.strip(), "rm"),
        ("""eval "bash -c 'rm -rf ~/x'" """.strip(), "rm"),
        # a wrapper in front must not hide the eval
        ('sudo eval "rm -rf ~/x"', "rm"),
        # eval reached from a later segment, not from the head of the line
        ('ls; eval "rm -rf ~/x"', "rm"),
        # …and from a command-introducing keyword
        ('if true; then eval "rm -rf ~/x"; fi', "rm"),
    ],
    ids=["eval-dq", "eval-sq", "eval-bare", "eval-mixed-quotes", "compound",
         "flag-combo", "nested-eval", "nested-bash-c", "sudo-wrapped",
         "after-separator", "after-keyword"],
)
def test_eval_payload_is_scanned(gate, command, expected):
    _run(gate, command)
    assert gate._recorder.asked, (
        f"BYPASS — no approval requested for eval payload: {command!r}"
    )
    assert expected in gate._recorder.asked[0][0]
    assert gate._recorder.spawned == []


@pytest.mark.parametrize(
    "command",
    [
        # harmless payload stays harmless
        'eval "ls -la"',
        "eval ls",
        # degenerate payloads must not raise or invent a prompt
        'eval ""',
        "eval",
        # THE regression this needs most: quoting that SURVIVES into the payload
        # is re-parsed by shlex, so `--hard` stays one token of prose and the
        # git flag-combo must not fire. (Quoting the outer shell already
        # consumed is a different case — see test_eval_join_matches_eval_semantics.)
        """eval "echo 'reset --hard'" """.strip(),
        """eval "git commit -m 'force push --hard fix'" """.strip(),
        # the payload ends at the separator: `rm` here belongs to no eval
        "eval ls; echo done",
    ],
    ids=["harmless-payload", "bare-harmless", "empty-payload", "dangling-eval",
         "quoted-prose-flag", "quoted-commit-message", "segment-bound"],
)
def test_eval_scan_does_not_overreach(gate, command):
    _run(gate, command)
    assert gate._recorder.asked == [], f"spurious approval prompt for: {command!r}"


def test_eval_payload_stops_at_the_segment_boundary(gate):
    """`eval`'s payload is its own segment, not the rest of the line.

    Swallowing past the `;` would let a later harmless segment be re-parsed as
    part of the payload — and, worse, hide a dangerous one by re-splicing it at
    a depth that the recursion bound eventually refuses.
    """
    _run(gate, "eval ls; rm -rf ~/x")
    assert gate._recorder.asked, "the rm AFTER eval's segment was never scanned"
    assert "rm" in gate._recorder.asked[0][0]
    assert gate._recorder.spawned == []


def test_eval_join_matches_eval_semantics(gate):
    """Space-joining the operands is eval's own behaviour, not an approximation.

    POSIX: eval concatenates its arguments separated by spaces, then parses and
    executes the result. So quoting the OUTER shell already consumed is really
    gone by the time eval runs — `eval git commit -m "reset --hard"` genuinely
    hands git a bare `--hard` operand. Prompting here is faithful, and this test
    pins that reading so a future "fix" does not re-quote the payload and
    silently reopen the `eval rm -rf x` bypass along with it.
    """
    _run(gate, 'eval git commit -m "reset --hard"')
    assert gate._recorder.asked, "eval's space-join semantics were not applied"
    assert "git" in gate._recorder.asked[0][0]


def test_eval_payload_reaches_the_destructive_effect_checks_too(gate):
    """Re-entry hands the payload to EVERY rule, not just the name gate.

    The payload is spliced into the same token loop, so the truncating-redirect
    check sees `> sample.py` inside an eval exactly as it sees it at top level —
    the property that makes re-entry worth doing instead of bolting a second,
    poorer scan onto `eval`.
    """
    _run(gate, """eval "echo x > sample.py" """.strip())
    assert gate._recorder.asked, "the redirect inside the eval payload was not seen"
    assert "truncates" in gate._recorder.asked[0][0]
    assert "sample.py" in gate._recorder.asked[0][0]
    assert gate._recorder.spawned == []


def test_eval_nesting_terminates(gate):
    """Nested `eval` is depth-bounded like nested `bash -c`, and terminates."""
    payload = "rm -rf ~/x"
    for _ in range(gt_mod._SHELL_C_MAX_DEPTH + 2):
        payload = f"eval {shlex.quote(payload)}"
    started = time.monotonic()
    _run(gate, payload)
    assert time.monotonic() - started < 10.0, (
        "eval scan did not terminate promptly past the depth limit"
    )


# ── Dangerous-executable NAME FAMILIES ────────────────────────────────────
# `mkfs` is in DANGEROUS_SHELL_COMMANDS, but exact basename membership made
# that decision unreachable: the filesystem is selected by the program name, so
# every spelling anyone types is `mkfs.<fs>` and none of them matched. Measured
# before DANGEROUS_EXECUTABLE_PREFIXES existed: `mkfs /dev/sda1` prompted,
# `mkfs.ext4 /dev/sda1` ran unprompted — the rule fired only for the one
# spelling nobody uses.


@pytest.mark.parametrize(
    "command,expected",
    [
        ("mkfs /dev/sda1", "mkfs"),  # control: the listed spelling still works
        ("mkfs.ext4 /dev/sda1", "mkfs.ext4"),
        ("mkfs.xfs /dev/sda1", "mkfs.xfs"),
        ("mkfs.btrfs -f /dev/sda1", "mkfs.btrfs"),
        # path form reduces to a basename before the policy lookup
        ("/sbin/mkfs.vfat /dev/disk2", "mkfs.vfat"),
        # and the family is reachable through every existing indirection
        ("sudo mkfs.ext4 /dev/sda1", "mkfs.ext4"),
        ('bash -c "mkfs.ext4 /dev/sda1"', "mkfs.ext4"),
        ('eval "mkfs.ext4 /dev/sda1"', "mkfs.ext4"),
    ],
    ids=["mkfs-bare", "ext4", "xfs", "btrfs", "abs-path-vfat", "sudo-wrapped",
         "via-bash-c", "via-eval"],
)
def test_mkfs_family_requires_approval(gate, command, expected):
    _run(gate, command)
    assert gate._recorder.asked, f"BYPASS — no approval requested for: {command!r}"
    assert expected in gate._recorder.asked[0][0], (
        "the prompt should name the real executable, not an abstraction of it"
    )
    assert gate._recorder.spawned == []


@pytest.mark.parametrize(
    "command",
    [
        # A prefix match needs a non-empty suffix; the bare dot is not a family
        # member, and unrelated programs that merely START with the letters must
        # not be swept up.
        "mkfsck /dev/sda1",
        "mkfontdir fonts/",
        "grep mkfs.ext4 /etc/fstab",  # the name as an ARGUMENT, not the executable
    ],
    ids=["mkfsck", "mkfontdir", "name-as-argument"],
)
def test_prefix_match_does_not_overreach(gate, command):
    _run(gate, command)
    assert gate._recorder.asked == [], f"spurious approval prompt for: {command!r}"


# ── Wrapper flags and operands that precede the real command ──────────────
# A wrapper keeps the executable slot open (`sudo rm` already prompted), but a
# flag VALUE or a positional operand would fill that slot and leave the real
# command reading as a mere argument. Only the SEPARATED spelling was affected:
# `sudo --user=me rm` and `sudo -uroot rm` glue the value to the flag, so they
# already worked and are kept here as the controls.


@pytest.mark.parametrize(
    "command,expected",
    [
        ("sudo -u me rm -rf ~/x", "rm"),
        ("doas -u me rm -rf ~/x", "rm"),
        ("env -u FOO rm -rf ~/x", "rm"),
        ("env -C /tmp rm -rf ~/x", "rm"),
        ("timeout -s KILL 5 rm -rf ~/x", "rm"),
        ("xargs -I {} rm -rf {}", "rm"),
        ("ionice -c 3 pkill -f node", "pkill"),
        ("stdbuf -o 0 rm -rf ~/x", "rm"),
        # `su -c` hands its payload to the target user's shell
        ('su -c "rm -rf ~/x"', "rm"),
        ('su someuser -c "rm -rf ~/x"', "rm"),
        # positional operand before the command
        ("chroot /mnt rm -rf ~/x", "rm"),
        # the flag-combo gate has to survive the same skip
        ("sudo -u me git reset --hard", "git"),
        # controls: glued value, already working before this rule existed
        ("sudo --user=me rm -rf ~/x", "rm"),
        ("sudo -uroot rm -rf ~/x", "rm"),
        # control: no flag at all
        ("sudo rm -rf ~/x", "rm"),
    ],
    ids=["sudo-u", "doas-u", "env-u", "env-C", "timeout-s", "xargs-I",
         "ionice-c", "stdbuf-o", "su-c", "su-user-c", "chroot-positional",
         "flag-combo-after-skip", "glued-long", "glued-short", "no-flag"],
)
def test_wrapper_operand_does_not_hide_the_command(gate, command, expected):
    _run(gate, command)
    assert gate._recorder.asked, (
        f"BYPASS — wrapper operand swallowed the executable slot: {command!r}"
    )
    assert expected in gate._recorder.asked[0][0]
    assert gate._recorder.spawned == []


@pytest.mark.parametrize(
    "command",
    [
        # the same shapes with a harmless command must stay silent
        "sudo -u me ls -la",
        "env -u FOO ls",
        "xargs -I {} echo {}",
        "timeout -s KILL 5 pytest -q",
        "nice -n 10 make build",
        "ionice -c 3 cp a b",
        'stdbuf -o 0 grep foo .',
        '/usr/bin/time -f "%e" ls',
        # wrapper with nothing after it
        "chroot /mnt",
        "chroot --help",
        "su someuser",
        # A user NAMED rm is the flag's value, not the command. This one was a
        # false PROMPT before the skip existed — `rm` landed in the executable
        # slot — so the fix removes a spurious prompt as well as adding a real one.
        "sudo -u rm ls",
        # `-u me` inside a commit message is prose, not a flag
        'git commit -m "-u me"',
    ],
    ids=["sudo-ls", "env-ls", "xargs-echo", "timeout-pytest", "nice-make",
         "ionice-cp", "stdbuf-grep", "time-ls", "chroot-bare", "chroot-help",
         "su-bare", "user-named-rm", "prose-flag"],
)
def test_wrapper_skip_does_not_invent_a_prompt(gate, command):
    _run(gate, command)
    assert gate._recorder.asked == [], (
        f"spurious approval prompt: {command!r}"
    )


@pytest.mark.parametrize(
    "command",
    [
        # `-u` with its value never supplied: skip_flag_value is still pending
        # when the segment ends, and would eat `rm` in the next one.
        "sudo -u; rm -rf ~/x",
        # chroot with its operand never supplied: positional_skip is still 1.
        "chroot; rm -rf ~/x",
    ],
    ids=["dangling-value-flag", "dangling-positional"],
)
def test_wrapper_skip_state_does_not_leak_past_the_segment(gate, command):
    """Pending wrapper state must be cleared at a segment boundary.

    The shape that matters is a wrapper whose operand never ARRIVES. A consumed
    one proves nothing: in `sudo -u me ls; rm -rf x` the skip is already spent by
    the `;`, so that command still prompts with the reset deleted — verified by
    mutation, which is why it is not the case tested here.
    """
    _run(gate, command)
    assert gate._recorder.asked, (
        f"pending wrapper state leaked past `;` and swallowed rm: {command!r}"
    )
    assert "rm" in gate._recorder.asked[0][0]


def test_wrapper_value_flag_only_applies_before_the_executable(gate):
    """Once the segment HAS an executable, a matching flag is that command's own.

    `git -c user.name=x rm -r x` is git's `-c`, not a wrapper's: the `rm` here is
    `git rm`, a subcommand, and nothing should be skipped on its account.
    """
    _run(gate, "git -c user.name=x log -1")
    assert gate._recorder.asked == []

from external_llm.agent.tool_handlers.shell_policy import (
    is_verification_command,
)


class TestVerificationCommandDetection:
    """Pin the ``is_verification_command`` regex that lets VERIFY→FINISH advance.

    This is NOT a safety gate — a false negative just misses a phase transition.
    The risk of a false positive (premature nudge) is also low, so the policy
    stays loose. What matters: it must recognise all the real spellings the
    model uses and must NOT fire on obviously non-verification commands.
    """

    # ── Must match ────────────────────────────────────────────────────────

    @pytest.mark.parametrize("cmd", [
        "pytest",
        "pytest -x -v tests/test_a.py",
        "cd tests && pytest",
        "py.test",
        "tox",
        "nox",
        "python -m pytest",
        "python3 -m pytest",
        "python -m pytest tests/ -x",
        "python3.14 -m unittest",
        "ruff check .",
        "mypy src/",
        "flake8",
        "pylint --rcfile=.pylintrc mypkg",
        "pyright",
        "npx eslint src/",
        "eslint --fix src/",
        "tsc --noEmit",
        "npx vitest",
        "vitest run",
        "npx jest",
        "jest --coverage",
        "go test ./...",
        "go vet ./...",
        "cargo test",
        "cargo clippy",
        "npm test",
        "pnpm run test",
        "yarn lint",
        "make test",
        "make lint",
        "make check",
        "uv run pytest",
        "poetry run pytest",
        "pdm run pytest",
        "hatch run pytest",
        "rye run pytest",
        "uv run ruff check",
        "poetry run mypy src/",
        "uv run python -m pytest",
        "poetry run python -m pytest",
        "uv run python3 -m pytest tests/ -x -v",
    ])
    def test_matches(self, cmd):
        assert is_verification_command(cmd) is True, (
            f"'{cmd}' should be recognised as a verification command"
        )

    # ── Must NOT match ────────────────────────────────────────────────────

    @pytest.mark.parametrize("cmd", [
        "uv run python script.py",          # bare python, no -m
        "uv sync",
        "poetry install",
        "uv add pytest",
        "pip install pytest",
        "grep pytest .",
        "git commit -m 'add pytest'",
        "ls",
        "cat test_results.txt",
        "echo all tests pass",
        "",                                  # empty string
    ])
    def test_does_not_match(self, cmd):
        assert is_verification_command(cmd) is False, (
            f"'{cmd}' should NOT be recognised as a verification command"
        )
