"""Test configuration for agent unit tests — resets module-level global state.

``context_budget._context_window_overrides`` and ``_override_meta`` are
module-level dicts that persist across test files.  Without explicit cleanup,
override tests from one file pollute basic-resolve tests in another.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _reset_context_overrides():
    """Clear runtime context-override state before every test.

    This is an autouse fixture — it runs automatically for every test in the
    tests/unit/agent/ directory, regardless of the test class. ``_override_dirty``
    is reset too so the dirty-guard / merge-on-write behaviour starts from a
    known clean baseline each test (a leftover dirty flag from one test would
    make a clean-flush test in another unexpectedly write).
    """
    import external_llm.agent.context_budget as _cb

    _cb._context_window_overrides.clear()
    _cb._override_meta.clear()
    _cb._override_dirty = False
    yield
    # Teardown: noop — the clear is done at setup time.
