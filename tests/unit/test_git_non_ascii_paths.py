"""Non-ASCII paths must survive every git call whose output we read.

``git`` C-quotes any path containing a non-ASCII byte unless told otherwise:

    한글파일.py  ->  "\\355\\225\\234\\352\\270\\200\\355\\214\\214\\354\\235\\274.py"

Two distinct damages follow, and they need different fixes:

* **Parsed** output — the quoted form's suffix is ``.py"`` (quote included), so
  ``LanguageId.from_path`` returns UNKNOWN. A repo whose sources are all
  Korean/CJK-named detected *no languages at all*. Fix: ``-z`` + NUL split, via
  the ``common.repo_files`` SSOT.
* **Displayed** output — the octal escapes go into the model's context ("Modified
  files (git status)"), where they are unusable as paths and produce
  file-not-found on a name nobody has. Fix: ``-c core.quotePath=false``.

``repo_files.git_list_repo_files`` already documented ``-z`` as REQUIRED for
exactly this reason; the language-detection and display layers were simply never
swept. The last test here is that sweep, as a gate.
"""
from __future__ import annotations

import ast
import pathlib
import subprocess

import pytest

from external_llm.agent.agent_context_manager import _run_git_raw
from external_llm.common.repo_files import git_list_repo_files
from external_llm.languages.dependency_checker import detect_repo_languages
from external_llm.languages.models import LanguageId

_KR_PY = "한글파일.py"
_KR_TS = "한글컴포넌트.ts"


@pytest.fixture
def kr_repo(tmp_path):
    """A real git repo whose source files have Korean names."""
    def _git(*args):
        subprocess.run(["git", *args], cwd=tmp_path, check=True,
                       capture_output=True, timeout=15)
    _git("init", "-q")
    _git("config", "user.email", "t@example.com")
    _git("config", "user.name", "t")
    (tmp_path / _KR_PY).write_text("def f():\n    return 1\n", encoding="utf-8")
    (tmp_path / _KR_TS).write_text("export const x = 1\n", encoding="utf-8")
    (tmp_path / "ascii.py").write_text("x = 1\n", encoding="utf-8")
    _git("add", "-A")
    _git("commit", "-qm", "init")
    return tmp_path


def test_git_really_does_c_quote_without_the_flag(kr_repo):
    """Pin the upstream behaviour the rest of this file defends against — if a
    future git stops quoting, these tests should say so rather than pass
    vacuously."""
    raw = subprocess.run(["git", "ls-files"], cwd=kr_repo, capture_output=True,
                         text=True, timeout=15).stdout
    assert '\\355' in raw, "git no longer C-quotes; this suite's premise changed"
    assert _KR_PY not in raw


# ── parsed output ─────────────────────────────────────────────────────────

def test_repo_files_ssot_returns_usable_paths(kr_repo):
    paths = git_list_repo_files(str(kr_repo))
    assert paths is not None
    assert set(paths) == {_KR_PY, _KR_TS, "ascii.py"}


def test_language_detection_sees_non_ascii_named_sources(kr_repo):
    """The regression: this returned an EMPTY set before the fix, because every
    quoted path's suffix was `.py"` / `.ts"` rather than `.py` / `.ts`."""
    langs = detect_repo_languages(str(kr_repo))
    assert LanguageId.PYTHON in langs
    assert LanguageId.TYPESCRIPT in langs


def test_a_quoted_path_is_genuinely_unclassifiable(kr_repo):
    """Why the above fails without -z, stated directly."""
    quoted = r'"\355\225\234\352\270\200\355\214\214\354\235\274.py"'
    assert pathlib.Path(quoted).suffix == '.py"'
    assert LanguageId.from_path(quoted) is LanguageId.UNKNOWN
    assert LanguageId.from_path(_KR_PY) is LanguageId.PYTHON


def test_detection_survives_a_non_git_directory(tmp_path):
    """git_list_repo_files returns None there; the callers must degrade to an
    empty set, not raise — matching the pre-fix contract."""
    assert git_list_repo_files(str(tmp_path)) is None
    assert detect_repo_languages(str(tmp_path)) == set()


# ── displayed output ──────────────────────────────────────────────────────

def test_status_shown_to_the_model_keeps_real_filenames(kr_repo):
    """`_run_git_raw` feeds `_build_session_context`'s "Modified files (git
    status)" block, i.e. the system prompt."""
    (kr_repo / _KR_PY).write_text("def f():\n    return 2\n", encoding="utf-8")
    out = _run_git_raw(str(kr_repo), "status", "--short")
    assert _KR_PY in out, f"model-facing git status was C-quoted: {out!r}"
    assert "\\355" not in out


def test_untracked_non_ascii_file_is_also_readable(kr_repo):
    (kr_repo / "새파일.py").write_text("y = 2\n", encoding="utf-8")
    out = _run_git_raw(str(kr_repo), "status", "--short")
    assert "새파일.py" in out
    assert "\\355" not in out


# ── the sweep, as a gate ──────────────────────────────────────────────────

# git subcommands whose stdout contains PATHS.
_PATH_PRINTING = {"status", "diff", "ls-files", "ls-tree", "diff-tree"}
# Flags that make a path-printing invocation actually print paths.
_PATH_FLAGS = ("--short", "--porcelain", "--name-only", "--name-status")

# Call sites that print paths but are provably unaffected. Each entry states
# WHY, because "it looked fine" is how the two bugs above survived a prior sweep.
_ALLOWED = {
    # rc-only probe: `--error-unmatch` is used for its exit status; stdout is
    # never read, so quoting cannot matter.
    ("external_llm/agent/orchestrator.py", "ls-files"),
    # emptiness-only: _is_worktree_clean() compares the whole blob to "", and
    # the debug dict keeps a truncated copy. No path is ever extracted.
    ("diff_apply.py", "status"),
}


def _shipping_files():
    roots = ["external_llm", "webapp", "utils", "services"]
    files: list[pathlib.Path] = []
    for r in roots:
        p = pathlib.Path(r)
        if p.exists():
            files += [f for f in p.rglob("*.py") if "/lane/" not in str(f)]
    files += list(pathlib.Path(".").glob("*.py"))
    return files


def test_no_shipping_git_call_prints_paths_unprotected():
    """Every path-printing git call must carry -z (parsed) or
    core.quotePath=false (displayed)."""
    offenders = []
    for f in _shipping_files():
        try:
            tree = ast.parse(f.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.List):
                continue
            elts = [e.value for e in node.elts
                    if isinstance(e, ast.Constant) and isinstance(e.value, str)]
            if not elts or elts[0] != "git":
                continue
            sub = next((e for e in elts[1:] if not e.startswith("-")), None)
            if sub not in _PATH_PRINTING:
                continue
            if "-z" in elts or "core.quotePath=false" in elts:
                continue
            if sub != "ls-files" and not any(x in elts for x in _PATH_FLAGS):
                continue  # e.g. `diff --stat`: a summary, not a path list
            if (str(f), sub) in _ALLOWED:
                continue
            offenders.append(f"{f}:{node.lineno}  {' '.join(elts)}")
    assert not offenders, (
        "git call prints paths without -z or core.quotePath=false — non-ASCII "
        "paths will be C-quoted:\n  " + "\n  ".join(offenders)
    )
