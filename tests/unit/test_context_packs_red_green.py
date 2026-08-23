"""HelperContextBuilder RED→GREEN: pure-helper context pack construction.

Previously 39% covered: the only consumer test
(``tests/unit/agent/test_agent_tools_contract.py``) substitutes a
``_FakeBuilder`` for the real class, so the actual runtime path used by
``agent_tools.py``'s lazy import was never exercised.

Contract under test (as consumed by ``_tool_delegate_to_helper``):
  * ``build(task, function_signature=None, local_snippet=None,
    constraints=None) -> ContextPack`` with ``.content`` (rendered markdown)
    and ``.metadata`` (per-section presence flags).
  * Optional sections are included iff truthy; ``task`` is always rendered
    (no guard), so the section order is stable: Task, Function Signature,
    Local Context, Constraints.
"""

from __future__ import annotations

import pytest

from external_llm.context.context_packs import ContextPack, HelperContextBuilder


def _builder(tmp_path) -> HelperContextBuilder:
    return HelperContextBuilder(tmp_path)


# ---------------------------------------------------------------------------
# Task-only build
# ---------------------------------------------------------------------------


def test_task_only_content_exact(tmp_path):
    pack = _builder(tmp_path).build(task="implement foo")
    assert pack.content == "## Task\nimplement foo\n"


def test_task_only_metadata_all_false(tmp_path):
    pack = _builder(tmp_path).build(task="implement foo")
    assert pack.metadata == {
        "has_signature": False,
        "has_snippet": False,
        "has_constraints": False,
    }


def test_empty_task_is_rendered_unconditionally(tmp_path):
    # task has no truthiness guard — even "" produces the Task section.
    pack = _builder(tmp_path).build(task="")
    assert pack.content == "## Task\n\n"


# ---------------------------------------------------------------------------
# Optional sections
# ---------------------------------------------------------------------------


def test_function_signature_section(tmp_path):
    pack = _builder(tmp_path).build(task="t", function_signature="def f() -> int:\n    return 1")
    assert "### Function Signature" in pack.content
    assert "```python\ndef f() -> int:\n    return 1\n```" in pack.content
    assert pack.metadata["has_signature"] is True
    assert pack.metadata["has_snippet"] is False
    assert pack.metadata["has_constraints"] is False


def test_local_snippet_section(tmp_path):
    pack = _builder(tmp_path).build(task="t", local_snippet="x = 1")
    assert "### Local Context" in pack.content
    assert "```python\nx = 1\n```" in pack.content
    assert pack.metadata["has_signature"] is False
    assert pack.metadata["has_snippet"] is True
    assert pack.metadata["has_constraints"] is False


def test_constraints_section_is_plain_text(tmp_path):
    # constraints are NOT wrapped in a python fence — only signatures/snippets are.
    pack = _builder(tmp_path).build(task="t", constraints="no network\nno threads")
    assert "### Constraints" in pack.content
    assert "no network\nno threads" in pack.content
    assert "```python" not in pack.content
    assert pack.metadata["has_constraints"] is True


def test_empty_optional_strings_are_excluded(tmp_path):
    # consumer passes `... or None`, but the direct contract is truthiness:
    # "" must not create a section nor set its metadata flag.
    pack = _builder(tmp_path).build(task="t", function_signature="", local_snippet="", constraints="")
    assert pack.content == "## Task\nt\n"
    assert pack.metadata["has_signature"] is False
    assert pack.metadata["has_snippet"] is False
    assert pack.metadata["has_constraints"] is False


def test_whitespace_only_sections_are_truthy(tmp_path):
    # documented truthiness semantics: " " is included (consumer strips args
    # before calling, so it cannot reach here in the real flow).
    pack = _builder(tmp_path).build(task="t", local_snippet=" ")
    assert "### Local Context" in pack.content
    assert pack.metadata["has_snippet"] is True


def test_all_sections_order_and_exact_content(tmp_path):
    pack = _builder(tmp_path).build(
        task="do it",
        function_signature="def f() -> int:\n    return 1",
        local_snippet="x = 1",
        constraints="no network",
    )
    assert pack.content == (
        "## Task\n"
        "do it\n"
        "\n"
        "### Function Signature\n"
        "```python\n"
        "def f() -> int:\n"
        "    return 1\n"
        "```\n"
        "\n"
        "### Local Context\n"
        "```python\n"
        "x = 1\n"
        "```\n"
        "\n"
        "### Constraints\n"
        "no network\n"
    )
    assert pack.metadata == {
        "has_signature": True,
        "has_snippet": True,
        "has_constraints": True,
    }


def test_section_order_is_task_signature_snippet_constraints(tmp_path):
    pack = _builder(tmp_path).build(
        task="t",
        function_signature="sig",
        local_snippet="snip",
        constraints="cons",
    )
    assert pack.content.index("## Task") < pack.content.index("### Function Signature")
    assert pack.content.index("### Function Signature") < pack.content.index("### Local Context")
    assert pack.content.index("### Local Context") < pack.content.index("### Constraints")


# ---------------------------------------------------------------------------
# Builder / result object contracts
# ---------------------------------------------------------------------------


def test_repo_root_is_stored(tmp_path):
    builder = _builder(tmp_path)
    assert builder.repo_root == tmp_path
    assert isinstance(builder.repo_root, type(tmp_path))


def test_repo_root_accepts_str_too(tmp_path):
    # consumer passes self.repo_root which may be str — no runtime validation.
    builder = HelperContextBuilder(str(tmp_path))
    assert builder.repo_root == str(tmp_path)


def test_build_requires_task(tmp_path):
    with pytest.raises(TypeError):
        _builder(tmp_path).build()  # type: ignore[call-arg]


def test_result_is_context_pack_with_content_and_metadata(tmp_path):
    # consumer reads via getattr(pack, "content", "") — attribute must exist.
    pack = _builder(tmp_path).build(task="t")
    assert isinstance(pack, ContextPack)
    assert getattr(pack, "content", "") == "## Task\nt\n"
    assert getattr(pack, "metadata", None) is not None


def test_context_pack_dataclass_direct(tmp_path):
    pack = ContextPack(content="c", metadata={"k": 1})
    assert pack.content == "c"
    assert pack.metadata == {"k": 1}
    assert "content=" in repr(pack)
