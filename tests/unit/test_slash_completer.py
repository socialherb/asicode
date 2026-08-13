"""
Unit tests for _SlashCommandCompleter — /model and /model dev_N auto-completion.

Covers the slot-token regex detection (dev / dev_ / dev_N) and the
dev_N <model> model-name completion path, including 'off'.
"""
from prompt_toolkit.formatted_text import FormattedText

import asi


def _run(prefix, dev_models=None):
    c = asi._SlashCommandCompleter(get_dev_models_fn=lambda: dev_models or {})
    return list(c._yield_model_completions(prefix))


def _txt(cs):
    return sorted(c.text for c in cs)


def _sp(cs):
    return [c.start_position for c in cs]


def _meta_str(c):
    m = c.display_meta
    return "".join(s for _, s in m) if isinstance(m, FormattedText) else str(m)


def _meta_of(cs, t):
    for c in cs:
        if c.text == t:
            return _meta_str(c)
    return None


def test_dev_slot_token_offers_dev1_to_dev8():
    """'dev' (no underscore) still completes to dev_1..dev_8."""
    assert _txt(_run("dev")) == [f"dev_{i}" for i in range(1, 9)]


def test_dev_underscore_offers_all_slots():
    assert len(_txt(_run("dev_"))) == 8


def test_dev_partial_matches_only_prefix():
    assert _txt(_run("dev_3")) == ["dev_3"]


def test_dev_out_of_range_yields_nothing():
    """dev_99 (99 > 8) → no slot suggestions."""
    assert _txt(_run("dev_99")) == []


def test_dev_slot_start_position_replaces_typed_prefix():
    assert all(s == -3 for s in _sp(_run("dev")))
    assert all(s == -5 for s in _sp(_run("dev_3")))


def test_configured_slot_shows_set_meta():
    dm = {"1": ("ollama", "qwen"), "3": ("anthropic", "claude")}
    r = _run("dev_1", dm)
    assert _meta_of(r, "dev_1") == "✓ set"


def test_unconfigured_slot_shows_generic_meta():
    dm = {"1": ("ollama", "qwen")}
    r = _run("dev_2", dm)
    assert _meta_of(r, "dev_2") == "sub-agent model slot"


def test_dev_model_part_completes_model_names():
    """'dev_1 q' → only model names (no dev_ tokens)."""
    r = _run("dev_1 q")
    assert len(_txt(r)) > 0
    assert all(not t.startswith("dev_") for t in _txt(r))


def test_dev_off_suggestion():
    assert "off" in _txt(_run("dev_1 of"))
    assert "off" in _txt(_run("dev_1 off"))


def test_dev_trailing_space_yields_models():
    """'dev_1 ' (trailing space) → model completion, not slot tokens."""
    r = _run("dev_1 ")
    assert len(_txt(r)) > 0
    assert all(not t.startswith("dev_") for t in _txt(r))


def test_dev_model_part_start_position():
    """'dev_1 qw' → completion replaces only 'qw' (2 chars)."""
    assert all(s == -2 for s in _sp(_run("dev_1 qw")))


def test_normal_model_path_not_triggered_for_dev():
    r = _run("qwen")
    assert len(_txt(r)) > 0
    assert all("dev_" not in t for t in _txt(r))


def test_invalid_slot_token_falls_through_to_normal_path():
    """'dev_1a' (non-digit suffix, no space) → normal model completion."""
    r = _run("dev_1a")
    assert all(not t.startswith("dev_") for t in _txt(r))


def _collect_async(completer, document):
    """Drive get_completions_async to completion and collect Completion texts.

    Uses a manual event loop so the test needs no pytest-asyncio dependency.
    """
    import asyncio

    async def _drive():
        return [c.text async for c in completer.get_completions_async(document, None)]

    return asyncio.run(_drive())


def test_get_completions_async_exists_and_matches_sync():
    """Regression: prompt_toolkit 3.x calls get_completions_async; a duck-typed
    completer that does not inherit from Completer must provide it explicitly,
    otherwise ``'_SlashCommandCompleter' object has no attribute
    'get_completions_async'`` is raised during Tab completion. Async results must
    match the synchronous path exactly.
    """
    from prompt_toolkit.document import Document

    c = asi._SlashCommandCompleter()
    doc = Document("/mod")
    sync_texts = sorted(x.text for x in c.get_completions(doc, None))
    async_texts = sorted(_collect_async(c, doc))
    assert sync_texts, "expected some /mod completions"
    assert async_texts == sync_texts


def test_get_completions_async_empty_for_non_slash():
    """Plain text (no leading '/') must yield nothing on both paths."""
    from prompt_toolkit.document import Document

    c = asi._SlashCommandCompleter()
    doc = Document("hello world")
    assert list(c.get_completions(doc, None)) == []
    assert _collect_async(c, doc) == []


# ── /think context sync (repl_impl._set_completer_context) ───────────────────
# The completer reads provider/model through the lambdas _collect_input wires
# to repl_impl's module globals, so a /model switch reaches /think ONLY if the
# switch actually rebinds those globals.  It did not: every assignment in
# _dispatch_command was a bare `x = ...` in a nested scope, which binds a local
# and evaporates, so /think kept offering the STARTUP model's reasoning values
# for the whole session.  Routing all writes through _set_completer_context()
# puts the `global` declaration in one place; scripts/check_missing_global.py
# gates the class so a bare assignment cannot come back.


import pytest

from external_llm.repl import repl_impl


@pytest.fixture
def _completer_ctx():
    """Restore the module globals — they outlive the test in a shared worker."""
    saved = (repl_impl._completer_provider, repl_impl._completer_model)
    yield repl_impl
    repl_impl._set_completer_context(*saved)


def _think_values(provider, model):
    """/think completions as the REPL wires them: lambdas over module globals."""
    repl_impl._set_completer_context(provider, model)
    c = asi._SlashCommandCompleter(
        get_provider_fn=lambda: repl_impl._completer_provider,
        get_model_fn=lambda: repl_impl._completer_model,
    )
    return sorted(x.text for x in c._yield_think_completions(""))


def test_set_completer_context_rebinds_the_module_globals(_completer_ctx):
    repl_impl._set_completer_context("deepseek", "deepseek-chat")
    assert repl_impl._completer_provider == "deepseek"
    assert repl_impl._completer_model == "deepseek-chat"


def test_set_completer_context_normalizes_none_to_empty(_completer_ctx):
    """svc.provider/model can be None; the completer contract is str."""
    repl_impl._set_completer_context(None, None)
    assert repl_impl._completer_provider == ""
    assert repl_impl._completer_model == ""


def test_think_completions_follow_a_model_switch(_completer_ctx):
    """The regression: switching provider must change /think's offered values."""
    before = _think_values("openai", "gpt-5.2")
    after = _think_values("deepseek", "deepseek-chat")
    assert before != after, "/think ignored the model switch"
    # deepseek's list is on/off/high/max; openai's is on/off/none/low/medium/high
    assert "max" in after and "max" not in before
    assert "none" in before and "none" not in after


def test_think_completions_follow_a_switch_back(_completer_ctx):
    """Not one-way: switching back restores the original list."""
    first = _think_values("openai", "gpt-5.2")
    _think_values("deepseek", "deepseek-chat")
    assert _think_values("openai", "gpt-5.2") == first


def test_dispatch_command_has_no_bare_write_to_completer_globals():
    """The original bug, pinned by name.

    _dispatch_command lives inside _run_repl_impl, so a bare assignment there
    binds a local and is discarded — with no ruff diagnostic, because the names
    are never read in that scope (F823 needs a read).  Python's own symbol
    table is the oracle: outside the single writer, these names must never
    resolve as a local.
    """
    import symtable
    from pathlib import Path

    src = Path(repl_impl.__file__).read_text(encoding="utf-8")

    def scopes(table, path=""):
        yield table, path
        for child in table.get_children():
            yield from scopes(child, f"{path}/{child.get_name()}")

    offenders = []
    for table, path in scopes(symtable.symtable(src, "repl_impl.py", "exec")):
        if table.get_type() != "function" or table.get_name() == "_set_completer_context":
            continue
        for name in ("_completer_provider", "_completer_model"):
            try:
                sym = table.lookup(name)
            except KeyError:
                continue
            if sym.is_local() and not sym.is_declared_global():
                offenders.append(f"{path}: {name}")
    assert not offenders, (
        "bare assignment to a completer global (binds a local, silently "
        f"discarded) in: {offenders}; use _set_completer_context()"
    )
