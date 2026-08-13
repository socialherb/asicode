"""One file must not be read and parsed whole just because it was named.

P26-4 gated the BATCH tree-sitter walker (``_index_via_treesitter``) at
``_NONPY_INPROC_MAX_BYTES`` because "a single minified dist/*.js (tens of MB is
common) was read + tree-sitter-parsed in full". Every PER-FILE entry point had
the identical hole, and each of them already held the answer: they ``stat()``
the file and then spent ``st_size`` on nothing but a cache signature.

Measured before the gate, on a generated 32 MB bundle:

    read_file (over-cap refusal)   13.31 s / 1,648 MB  -> 131 chars of output
    get_file_outline               48.62 s /   685 MB
    find_symbol                    21.87 s /   336 MB  -> "not found"

The refusal path is the perverse one: the whole cost buys a message that
returns no file content at all. 0.2.15 capped tree-sitter parse memos to hold
whole-process peak RSS at 76 MB; one of these calls was 20x that on its own.
"""
from __future__ import annotations

import time

import pytest

from external_llm.agent.symbol_search import (
    _NONPY_INPROC_MAX_BYTES,
    SymbolSearcher,
)

# Comfortably past the gate, small enough to stay a fast test.
_OVERSIZE = _NONPY_INPROC_MAX_BYTES + (1 << 20)


def _write_oversized(root: str, name: str, unit: str) -> str:
    with open(f"{root}/{name}", "w", encoding="utf-8") as fh:
        fh.write(unit * ((_OVERSIZE // len(unit)) + 1))
    return name


@pytest.mark.parametrize(
    ("name", "unit"),
    [
        ("huge.js", "function f(){return 1;}\n"),
        ("huge.ts", "export function g(): number { return 1; }\n"),
        ("huge.py", "def h():\n    return 1\n"),
        ("huge.go", "func Alpha() int { return 1 }\n"),
    ],
    ids=["js", "ts", "py", "go"],
)
def test_the_outline_skips_an_oversized_file(temp_repo_root, name, unit):
    """Every language branch of get_file_outline reads and parses the file."""
    _write_oversized(temp_repo_root, name, unit)
    searcher = SymbolSearcher(temp_repo_root)

    started = time.monotonic()
    out = searcher.get_file_outline(name)
    elapsed = time.monotonic() - started

    assert out == [], "an oversized file must not be parsed for an outline"
    # Generous by 2 orders of magnitude against the 13-48 s this used to take;
    # the point is that no parse happened, not the exact number.
    assert elapsed < 2.0, f"outline took {elapsed:.1f}s — the file was parsed"


def test_a_normal_file_is_still_outlined(temp_repo_root):
    """The gate must not cost the case it exists to protect."""
    with open(f"{temp_repo_root}/small.py", "w", encoding="utf-8") as fh:
        fh.write("def alpha():\n    return 1\n\n\nclass Beta:\n    pass\n")
    names = {s.name for s in SymbolSearcher(temp_repo_root).get_file_outline("small.py")}
    assert {"alpha", "Beta"} <= names


def test_the_refusal_message_survives_an_empty_outline(tool_registry, temp_repo_root):
    """read_file's over-cap guidance documents this fallback — exercise it.

    "Falls back to the plain count when the outline is empty (unsupported
    language, parse failure) so this path can never be worse than what it
    replaces." An oversized file is now a third way to get an empty outline.
    """
    _write_oversized(temp_repo_root, "huge.py", "x = 1\n")
    result = tool_registry.dispatch("read_file", {"path": "huge.py"})
    assert result.ok
    assert (result.metadata or {}).get("over_line_cap") is True
    assert "too long to return whole" in result.content
    assert "start_line" in result.content, "the caller must still be told what to do"
