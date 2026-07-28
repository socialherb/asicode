"""Edit-failure recovery must point at read_file, not `bash cat`.

`read_file` renders each line as ``   121 │ 4│     return x`` — the ``│N│``
gutter is the leading-whitespace count, the same metric the write tools compute
``min_indent``/``detect_indent_char`` from. Its whole reason to exist, per its
own docstring, is "eliminating the guesswork that causes indent mismatches in
edit_text/anchor_edit/modify_symbol".

`bash cat -n` numbers lines but shows no such column. So the advice given when
an edit fails *because the supplied text did not match* was routing the model to
the one reading method that hides what the mismatch is usually about — and, in
`_enrich_plan_error`, handing it a home-made ``"NNN: line"`` listing under the
instruction "copy the EXACT text from this block".

`edit_text::search_string_mismatch` is the most frequent real failure in this
repo's persistent failure store, which is what makes the routing worth pinning.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from external_llm.common.indent_utils import INDENT_GUTTER_BAR, format_numbered_line

_SRC = (
    "class Widget:\n"
    "    def render(self):\n"
    "        return self.value\n"
    "\n"
    "    def reset(self):\n"
    "        self.value = None\n"
)


@pytest.fixture
def repo(tool_registry, tmp_path):
    (Path(tool_registry.repo_root) / "widget.py").write_text(_SRC, encoding="utf-8")
    return tool_registry


def _plan(before: str):
    return {"ops": [{"op": "edit_blocks", "path": "widget.py",
                     "blocks": [{"before": before, "after": "pass"}]}]}


# ── the near-miss block the model is told to copy from ────────────────────

def test_closest_match_block_carries_the_indent_gutter(repo):
    """This block is followed by "Copy the EXACT text from this block", so it
    is the single most important place for indentation to be unambiguous."""
    hint = repo._enrich_plan_error(_plan("    def rendr(self):"), "before not found")
    assert "Closest match" in hint, hint
    assert format_numbered_line(2, "    def render(self):") in hint, hint


def test_closest_match_block_explains_the_gutter(repo):
    """A column the model has not seen before must be labelled, or it may copy
    the gutter into 'before' and fail a second time."""
    hint = repo._enrich_plan_error(_plan("    def rendr(self):"), "before not found")
    assert "leading-whitespace count" in hint
    assert "not the gutter" in hint


# ── the no-near-miss fallback preview ─────────────────────────────────────

def test_fallback_preview_carries_the_indent_gutter(repo):
    hint = repo._enrich_plan_error(_plan("zzz totally absent zzz"), "before not found")
    assert format_numbered_line(2, "    def render(self):") in hint, hint
    assert "   2:     def render" not in hint, "bare NNN: listing is back"


def test_fallback_preview_routes_to_read_file_not_bash(repo):
    hint = repo._enrich_plan_error(_plan("zzz totally absent zzz"), "before not found")
    assert "read_file" in hint
    assert "start_line" in hint
    assert "bash (cat)" not in hint


def test_the_reason_for_read_file_is_stated(repo):
    """Naming the tool without the reason invites the model to substitute an
    equivalent-looking one; the gutter is the whole point."""
    hint = repo._enrich_plan_error(_plan("zzz totally absent zzz"), "before not found")
    assert INDENT_GUTTER_BAR in hint


# ── the other two recovery paths, as source contracts ─────────────────────
# These build their advice deep inside a turn/tool loop that a unit test cannot
# reach without a full LLM round-trip, so the string itself is the contract.

@pytest.mark.parametrize(
    "path,anchor",
    [
        ("external_llm/agent/agent_loop.py", "BLOCK NOT FOUND:"),
        ("external_llm/agent/agent_turn_pipeline.py",
         "Do NOT call write_plan with the same arguments again."),
    ],
)
def test_recovery_advice_names_read_file(path, anchor):
    text = Path(path).read_text(encoding="utf-8")
    idx = text.find(anchor)
    assert idx != -1, f"anchor text moved in {path}"
    window = text[idx: idx + 600]
    assert "read_file" in window, f"{path}: recovery advice no longer names read_file"
    assert "bash (cat)" not in window, f"{path}: recovery advice steers to bash cat again"


def test_symbol_info_carries_no_bash_read_guidance(repo):
    """The dead `read_guidance` key told the model to use `bash (cat -n)`; it
    had no readers, so deleting it changed no behaviour — but it must not come
    back by way of a wholesale serialisation of that dict.

    Asserted on the returned VALUES rather than on the source text: the source
    still discusses `bash (cat -n)` in the comment explaining why the key is
    gone, and a grep-style test would match that prose forever.
    """
    from external_llm.agent.symbol_search import SymbolSearcher

    info = SymbolSearcher(str(repo.repo_root)).get_symbol_info("render", file_path="widget.py")
    if info is None:
        pytest.skip("symbol lookup unavailable in this environment")
    assert "read_guidance" not in info
    rendered = " ".join(str(v) for v in info.values())
    assert "bash" not in rendered.lower(), f"symbol info steers to bash: {info!r}"
