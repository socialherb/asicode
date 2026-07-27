"""Tests for the shell danger-approval gate and the bash timeout clamp.

The gate is a *default-open* policy: everything runs unless the executable name
appears in ``DANGEROUS_SHELL_COMMANDS``, in which case the user is asked first.
Its blind spot was scope: a machine-wide ``pkill -f <pattern>`` sailed through
unremarked while the ``rm`` in the same command line raised a prompt.

Every test here denies approval, so no dangerous command is ever executed — the
gate short-circuits before Popen. That is asserted explicitly, not assumed.
"""
from __future__ import annotations

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
