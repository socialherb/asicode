"""Both Python tree-sitter grammars must produce the same analysis results.

Two grammars are reachable at runtime and they disagree on tree SHAPE:

    standalone tree-sitter-python :  module -> expression_statement -> assignment
    tree-sitter-language-pack     :  module -> assignment            (no wrapper)

Only ``tree-sitter-language-pack`` is declared in ``pyproject.toml``, while
``tree_sitter_utils`` documents the individual ``tree_sitter_<lang>`` modules as
the PRIMARY path and the pack as a fallback. Dev machines tend to have stray
individual grammar packages installed, so every consumer written against the
wrapper shape passed locally and silently returned less for real users:
``collect_defined_names`` dropped module-level assignments, and
``public_dead_code_scanner`` stopped honouring ``__all__`` and reported public
re-exports as dead code.

These tests pin the parity by running the SAME assertions against each grammar
that is importable, driving the real public entry points rather than poking at
nodes. Whichever grammar a machine happens to resolve, the other one is still
covered here. If only one is installed, the other parametrisation skips — so
this file never gates on an undeclared dependency.
"""

from __future__ import annotations

import pytest

from external_llm.languages import tree_sitter_utils as tsu


def _standalone_language():
    import tree_sitter as ts
    import tree_sitter_python as tsp

    return ts.Language(tsp.language())


def _pack_language():
    from tree_sitter_language_pack import get_language

    return get_language("python")


_SOURCES = {"standalone": _standalone_language, "pack": _pack_language}


@pytest.fixture(params=sorted(_SOURCES))
def forced_python_grammar(request):
    """Pin the ``python`` grammar to one specific source for the whole test.

    FOUR caches are grammar-blind, and missing any one makes this fixture
    silently vacuous — both parametrisations then run on whichever grammar was
    resolved first, and every assertion below passes while proving nothing (that
    is exactly what the first draft of this file did):

    * ``_LANG_CACHE`` — what ``_get_language`` consults.
    * ``_PARSER_TLS`` — ``get_parser`` memoises the *Parser* per thread keyed by
      LANGUAGE NAME ONLY, so it returns a parser still bound to the old grammar.
    * ``parse_to_tree`` (``@lru_cache``) — keyed by ``(content, language)``.
    * ``_QUERY_CACHE`` — keyed by ``(language, query_string)``; a compiled Query is
      bound to one Language and matches NOTHING against another's tree, which
      looks identical to a broken query pattern.

    ``invalidate_caches()`` handles all four, so the fixture calls it and then
    seeds ``_LANG_CACHE`` — in that order, since invalidation clears the seed.
    """
    try:
        lang = _SOURCES[request.param]()
    except Exception as exc:  # grammar source not installed here
        pytest.skip(f"{request.param} python grammar unavailable: {exc}")

    def _pin() -> None:
        tsu.invalidate_caches()
        tsu._LANG_CACHE["python"] = lang

    _pin()
    yield request.param
    # Leave no pinned grammar behind for unrelated tests in the same process.
    tsu.invalidate_caches()


def test_fixture_actually_switches_grammar(forced_python_grammar):
    """Guard the fixture itself: the pinned grammar must reach ``get_parser``.

    Without clearing ``_PARSER_TLS.cache`` this assertion fails for one of the two
    parametrisations, which is precisely how a vacuous parity suite looks.
    """
    expected = "expression_statement" if forced_python_grammar == "standalone" else "assignment"
    tree = tsu.parse_to_tree("X = 1\n", "python")
    assert tree is not None
    assert tree.root_node.children[0].type == expected


def test_grammars_really_do_differ_in_shape():
    """Guard the premise: if the shapes ever converge, these tests lose their point."""
    shapes = {}
    for name, factory in _SOURCES.items():
        try:
            lang = factory()
        except Exception:
            continue
        import tree_sitter as ts

        root = ts.Parser(lang).parse(b"X = 1\n").root_node
        shapes[name] = root.children[0].type
    if len(shapes) < 2:
        pytest.skip("need both grammar sources to compare shapes")
    assert shapes["standalone"] == "expression_statement"
    assert shapes["pack"] == "assignment"


def test_collect_defined_names_finds_module_level_assignment(forced_python_grammar):
    """``X = 1`` is a defined name under either shape."""
    from external_llm.code_structure_utils import collect_defined_names

    names = collect_defined_names("def f(): pass\nclass C: pass\nX = 1\n")
    assert {"f", "C", "X"}.issubset(names), f"{forced_python_grammar}: got {sorted(names)}"


def test_collect_defined_names_finds_annotated_and_chained(forced_python_grammar):
    from external_llm.code_structure_utils import collect_defined_names

    names = collect_defined_names("A: int = 1\nB = C = 2\nD += 3\n")
    assert {"A", "B", "C", "D"}.issubset(names), f"{forced_python_grammar}: got {sorted(names)}"


def test_all_literal_is_honoured(forced_python_grammar):
    """``__all__ = [...]`` must be parsed, or public names get reported dead."""
    from external_llm.analysis._dead_block_shared import _ts_extract_all_list

    assert _ts_extract_all_list('__all__ = ["a", "b"]\ndef f(): pass\n') == {"a", "b"}


def test_dynamic_all_is_detected(forced_python_grammar):
    """A non-literal ``__all__`` must yield the conservative sentinel, not silence."""
    from external_llm.analysis._dead_block_shared import _ts_extract_all_list

    assert _ts_extract_all_list("__all__ = [x for x in names]\n") == {"*__dynamic__*"}


def test_public_dead_code_scanner_respects_all(forced_python_grammar, tmp_path):
    """End-to-end: a name in ``__all__`` is never a dead-code candidate."""
    from external_llm.analysis.public_dead_code_scanner import scan_public_dead_blocks

    src = tmp_path / "m.py"
    src.write_text(
        'def public_api():\n    return 1\n\n\ndef _internal():\n    return 2\n\n__all__ = ["public_api"]\n',
        encoding="utf-8",
    )
    candidates = scan_public_dead_blocks(
        repo_root="",
        file_paths=[str(src)],
        cross_file_referenced_names=set(),
    )
    flagged = {m.name for c in candidates for m in c.members}
    assert "public_api" not in flagged, f"{forced_python_grammar}: __all__ ignored"


def test_module_level_assignment_is_a_symbol(forced_python_grammar):
    """The shared _SYMBOL_QUERIES python pattern must capture module-level names."""
    captures = tsu.query_captures(
        "X = 1\ndef f(): pass\n",
        "python",
        tsu._SYMBOL_QUERIES["python"],
    )
    names = {c.text for c in captures if c.capture_name == "name"}
    assert {"X", "f"}.issubset(names), f"{forced_python_grammar}: got {sorted(names)}"
