"""Contract tests for the credential-env leak guard in ``tests/conftest.py``.

The guard is a pytest HOOK WRAPPER, not a fixture: fixture finalization runs
INSIDE ``pytest_runtest_teardown``, so only a post-yield wrapper sees the state a
test really leaves behind — a fixture's own teardown would still have
monkeypatch's undo pending and would report every legitimate
``monkeypatch.setenv`` as a leak. A hook that stops being registered, or that
silently loses its comparison, looks exactly like a suite with no leaks, so the
wiring and the verdict are asserted here rather than assumed: dropping the
``wrapper=True`` registration or the check inside the wrapper fails this file.

The hooks are driven directly through the generator protocol — same source, no
nested pytest run, nothing written into the repo.
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Callable

import pytest

import asi
import tests.conftest as guard

_LEAKED = "sk-leaked-value"


class _Item:
    """Stand-in for a pytest item — the guard's hooks touch only ``.stash``."""

    def __init__(self) -> None:
        self.stash = pytest.Stash()


def _run_guard(item: _Item, body: Callable[[], None]) -> None:
    """Run setup hook -> *body* -> teardown hook in the order pytest uses.

    The baseline is therefore captured before *body* runs, exactly as the real
    wrapper does it (before any fixture for the item). Re-raises whatever the
    teardown wrapper raises — ``Failed`` when it reports a leak.
    """
    setup_gen = guard.pytest_runtest_setup(item)
    next(setup_gen)
    with contextlib.suppress(StopIteration):
        next(setup_gen)

    body()

    teardown_gen = guard.pytest_runtest_teardown(item, None)
    next(teardown_gen)
    with contextlib.suppress(StopIteration):
        next(teardown_gen)


def _write(key: str, value: str = _LEAKED) -> None:
    """The shape of the real bug: production assigns ``os.environ[key]`` itself."""
    os.environ[key] = value


def test_direct_write_after_registration_is_reported(monkeypatch):
    """Registered key, then overwritten by the code under test — the CI shape."""
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    item = _Item()

    def body() -> None:
        monkeypatch.setenv("DEEPSEEK_API_KEY", "placeholder")
        _write("DEEPSEEK_API_KEY")

    with pytest.raises(pytest.fail.Exception) as exc:
        _run_guard(item, body)

    message = str(exc.value)
    assert "DEEPSEEK_API_KEY" in message
    # Redaction contract: report the key and the shape, never the value — the
    # leaked value can be a real credential and this message lands in CI logs.
    assert _LEAKED not in message
    assert f"set ({len(_LEAKED)} chars)" in message


def test_unregistered_deletion_is_reported(monkeypatch):
    """Removing a credential the code under test did not register is a leak too."""
    monkeypatch.setenv("OPENAI_API_KEY", "placeholder")
    item = _Item()

    def body() -> None:
        os.environ.pop("OPENAI_API_KEY", None)

    with pytest.raises(pytest.fail.Exception) as exc:
        _run_guard(item, body)

    assert "OPENAI_API_KEY" in str(exc.value)
    assert "unset" in str(exc.value)


def test_monkeypatch_managed_change_is_not_a_leak(monkeypatch):
    """No false positive: the undo lands before the check, so nothing changed.

    The fixture's calls belong INSIDE ``body``: pytest captures the guard's
    baseline before any fixture runs, so a ``monkeypatch`` action taken outside
    sits outside the watched window — and its undo then restores the value that
    was current when it was registered, which the guard reads as a change. With
    an ambient credential exported (a developer shell, not CI) that produced a
    real-looking report: ``DEEPSEEK_API_KEY (unset → set (35 chars))``.
    """
    item = _Item()

    def body() -> None:
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        monkeypatch.setenv("DEEPSEEK_API_KEY", "placeholder")
        monkeypatch.setenv("OPENAI_API_KEY", "placeholder")
        monkeypatch.delenv("ZAI_API_KEY", raising=False)
        monkeypatch.undo()  # what fixture finalization does before the check runs

    _run_guard(item, body)


def test_non_credential_env_var_is_out_of_scope(monkeypatch):
    """Scope contract: the watched set is the credential SSOT, not every var."""
    monkeypatch.setenv("ASICODE_ENV_GUARD_PROBE", "placeholder")
    item = _Item()

    def body() -> None:
        _write("ASICODE_ENV_GUARD_PROBE", "leaked")

    _run_guard(item, body)


def test_missing_baseline_is_a_noop():
    """A teardown with no recorded baseline has nothing to judge — must not raise."""
    item = _Item()
    teardown_gen = guard.pytest_runtest_teardown(item, None)
    next(teardown_gen)
    with contextlib.suppress(StopIteration):
        next(teardown_gen)


def test_watched_set_is_derived_from_the_production_ssot():
    """A provider added to ``_API_KEY_ENV_MAP`` is watched without test edits."""
    assert set(guard._provider_key_env_names()) == {v for v in asi._API_KEY_ENV_MAP.values() if v}


def test_hooks_stay_registered_as_wrappers():
    """Registration contract — a non-wrapper hook would judge before the undo."""
    for hook in (guard.pytest_runtest_setup, guard.pytest_runtest_teardown):
        opts = getattr(hook, "pytest_impl", None)
        assert isinstance(opts, dict), hook.__name__
        assert opts.get("wrapper") is True, hook.__name__
