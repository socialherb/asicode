"""P20 round: context_collector path containment + bounded reads.

P20-1: read_file_snippet_context / collect_related_files_shallow resolved a
prompt-supplied rel path with normalize_rel_path_fast, which strips leading
./ and / but does NOT reject ".." — "Target file: ../SECRET.txt" read outside
the repo (instruction mode + legacy diff mode; the external-LLM / plan-json
paths already refused it). Fix: resolve_inside_repo (strict normalize +
resolve() + relative_to containment); failure → meta["reason"] =
"path_outside_repo".

P20-4: _read_text_best_effort read files FULLY even though callers only need a
bounded window (a 2-4 KB snippet / an import head). Fix: max_bytes param; the
snippet path reports meta["read_truncated"] and appends the P19-1 marker.

P20-5: a byte-cap cut that splits a multi-byte char used to fail all 4 strict
decodes and fall back to replace (4x slower + potential cp949/euc-kr mojibake).
Fix: trim the incomplete trailing UTF-8 sequence before the strict ladder.
"""

from __future__ import annotations

import os
from pathlib import Path

import context_collector as cc


def _repo_with_files(tmp_path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "app.py").write_text("import os\nprint('hi')\n", encoding="utf-8")
    return repo


# ---------- P20-1: path containment ----------


def test_snippet_rejects_dotdot_traversal(tmp_path):
    repo = _repo_with_files(tmp_path)
    outside = tmp_path / "SECRET.txt"
    outside.write_text("TOP-SECRET-CANARY-9931\n", encoding="utf-8")
    ctx, meta = cc.read_file_snippet_context(str(repo), "../SECRET.txt", around_regex="^")
    assert ctx == ""
    assert meta["included"] is False
    assert meta["reason"] == "missing_args"  # SSOT rejects .. at normalize → reln empty
    # (snippet path reports missing_args; shallow reports empty_target)


def test_snippet_rejects_symlink_escape(tmp_path):
    repo = _repo_with_files(tmp_path)
    outside = tmp_path / "SECRET.txt"
    outside.write_text("TOP-SECRET-CANARY-9931\n", encoding="utf-8")
    os.symlink(outside, repo / "link.txt")
    ctx, meta = cc.read_file_snippet_context(str(repo), "link.txt", around_regex="^")
    assert ctx == ""
    assert meta["reason"] == "path_outside_repo"


def test_snippet_normal_path_still_reads(tmp_path):
    repo = _repo_with_files(tmp_path)
    ctx, meta = cc.read_file_snippet_context(str(repo), "app.py", around_regex="import")
    assert meta["included"] is True
    assert "import os" in ctx
    assert meta["reason"] == "ok"


def test_snippet_absolute_path_missing_file_same_as_before(tmp_path):
    # "/etc/passwd" normalizes to the relative "etc/passwd", which does not
    # exist inside the repo → missing_file (same outcome as the old path).
    repo = _repo_with_files(tmp_path)
    ctx, meta = cc.read_file_snippet_context(str(repo), "/etc/passwd", around_regex="^")
    assert ctx == ""
    assert meta["reason"] == "missing_file"


def test_collect_shallow_rejects_dotdot_traversal(tmp_path):
    repo = _repo_with_files(tmp_path)
    outside = tmp_path / "SECRET.txt"
    outside.write_text("x = 1\n", encoding="utf-8")
    sel, meta = cc.collect_related_files_shallow(str(repo), "../SECRET.txt")
    assert sel == []
    assert meta["reason"] == "empty_target"  # SSOT rejects .. at normalize; resolve never reached


def test_collect_shallow_normal_path_ok(tmp_path):
    repo = _repo_with_files(tmp_path)
    sel, meta = cc.collect_related_files_shallow(str(repo), "app.py")
    assert "app.py" in sel
    assert meta["reason"] == "ok"


def test_collect_shallow_rejects_symlink_escape(tmp_path):
    repo = _repo_with_files(tmp_path)
    outside = tmp_path / "SECRET.txt"
    outside.write_text("x = 1\n", encoding="utf-8")
    os.symlink(outside, repo / "link.py")
    sel, meta = cc.collect_related_files_shallow(str(repo), "link.py")
    assert sel == []
    assert meta["reason"] == "path_outside_repo"


# ---------- P20-4: bounded reads ----------


def _write_big(p: Path):
    p.write_bytes(b"x = 1\n" * 200_000)  # ~1.2 MiB — beyond the 1 MiB window
    return p


def test_snippet_bounded_read_flags_read_truncated(tmp_path):
    repo = _repo_with_files(tmp_path)
    big = _write_big(repo / "big.py")
    ctx, meta = cc.read_file_snippet_context(str(repo), big.name, around_regex="^")
    assert meta["included"] is True
    assert meta.get("read_truncated") is True
    assert "...[TRUNCATED]..." in ctx, "model must know the head is not the whole file"


def test_snippet_small_file_no_read_truncated(tmp_path):
    repo = _repo_with_files(tmp_path)
    ctx, meta = cc.read_file_snippet_context(str(repo), "app.py", around_regex="^")
    assert meta["included"] is True
    assert meta.get("read_truncated") is None
    assert "...[TRUNCATED]..." not in ctx


def test_collect_bounded_read_still_parses_head_imports(tmp_path):
    repo = _repo_with_files(tmp_path)
    (repo / "helper.py").write_text("VALUE = 1\n", encoding="utf-8")
    mod = repo / "mod.py"
    mod.write_bytes(b"import helper\n" + b"y = 1\n" * 200_000)  # > 256 KiB
    sel, meta = cc.collect_related_files_shallow(str(repo), "mod.py")
    assert "mod.py" in sel
    assert meta["reason"] == "ok"
    assert "helper.py" in meta["candidates"], "head imports must still be parsed"
    assert meta.get("read_truncated") is True


def test_read_text_best_effort_max_bytes_bounds(tmp_path):
    p = _write_big(tmp_path / "big.py")
    text, enc, truncated = cc._read_text_best_effort(p, max_bytes=256 * 1024)
    assert truncated is True
    assert len(text) < 300_000
    assert enc == "utf-8"


def test_read_text_best_effort_no_max_bytes_unchanged(tmp_path):
    p = tmp_path / "s.py"
    p.write_text("x = 1\n", encoding="utf-8")
    text, enc, truncated = cc._read_text_best_effort(p)
    assert truncated is False
    assert text == "x = 1\n"
    assert enc == "utf-8"


# ---------- P20-5: UTF-8 safe cut ----------


def test_read_text_best_effort_trims_incomplete_trailing_char(tmp_path):
    # 1 MiB boundary splits a 3-byte Korean char (1 MiB % 3 == 1): the strict
    # ladder must still win on the first try — no 4x-fail + replace re-decode,
    # no U+FFFD, no accidental cp949/euc-kr mojibake.
    p = tmp_path / "hangul.py"
    p.write_bytes(b"a" * (1024 * 1024 - 1) + "한".encode())
    text, enc, truncated = cc._read_text_best_effort(p, max_bytes=1024 * 1024)
    assert truncated is True
    assert enc == "utf-8", f"strict decode should win after the trim, got {enc!r}"
    assert "\ufffd" not in text
    assert text.endswith("a")


def test_utf8_leading_incomplete_len_helper():
    from utils.string_helper import utf8_leading_incomplete_len

    # window starts mid-char: two orphan continuation bytes
    mid = "한".encode()[1:] + b"rest"
    assert utf8_leading_incomplete_len(mid) == 2
    assert utf8_leading_incomplete_len(b"clean") == 0
    assert utf8_leading_incomplete_len(b"") == 0
