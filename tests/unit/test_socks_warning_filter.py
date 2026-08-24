"""Regression: urllib3's SOCKS DependencyWarning must never reach the banner.

``requests`` silences that warning itself, but only for the ~10ms between
installing its filter and importing ``requests.adapters``. ``catch_warnings()``
is not thread-safe (it restores a *snapshot* of the global filter list), so a
background thread leaving such a block inside that window wipes requests' filter
and the warning leaks to stderr — which is how it landed in the middle of the
asi startup banner (emb-warmup thread vs. the main thread's requests import).

``asi._silence_socks_dependency_warning()`` installs the filter at import time,
before any thread exists, so every later snapshot already contains it.

Both tests drive the leak with a churn thread in a subprocess; the *control*
test asserts the harness still reproduces the leak without asi's filter, so the
positive test cannot silently pass for the wrong reason.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

# PySocks present → urllib3 imports it cleanly and never warns: nothing to test.
pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("socks") is not None,
    reason="PySocks installed — urllib3 emits no SOCKS DependencyWarning",
)

_CHURN = """
    import threading, time, warnings, builtins

    # Hook requests' import chain at the exact spot where the SOCKS warning
    # would be emitted (urllib3.contrib.socks, imported from
    # requests.adapters). Firing the event there, then sleeping, makes the
    # churn thread's catch_warnings EXIT land deterministically BETWEEN
    # requests installing its DependencyWarning filter (requests/__init__.py)
    # and the warning being emitted — the race window the module docstring
    # documents. The original design slept 0.05s and hoped for a collision;
    # this is race-free by construction.
    real_import = builtins.__import__
    fired = threading.Event()

    def hooked(name, *a, **kw):
        if name == "urllib3.contrib.socks":
            fired.set()
            time.sleep(0.005)  # widen the window: hold the import open
        return real_import(name, *a, **kw)

    builtins.__import__ = hooked

    def churn():
        # Mimics sentence_transformers/torch: sit inside a catch_warnings
        # block until the SOCKS import begins, then exit — restoring the
        # pre-requests filter snapshot at the worst possible moment.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            fired.wait(10)

    threading.Thread(target=churn, daemon=True).start()
    import requests  # noqa: F401  — the import that triggers the SOCKS warning
    time.sleep(0.01)
"""


def _run(script: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=120,
        env=os.environ.copy(),
        check=False,
    )


def test_churn_thread_reproduces_the_leak_without_the_filter():
    """Control: the race is real — without asi's filter the warning does leak."""
    proc = _run(_CHURN)
    assert proc.returncode == 0, proc.stderr
    assert "DependencyWarning" in proc.stderr, (
        "control arm no longer reproduces the leak — the positive test below "
        f"would pass vacuously. stderr:\n{proc.stderr}"
    )


def test_importing_asi_survives_a_concurrent_catch_warnings_restore():
    """Importing asi first keeps the warning suppressed through the same race."""
    proc = _run("    import asi  # noqa: F401  — installs the filter at import time\n" + _CHURN)
    assert proc.returncode == 0, proc.stderr
    assert "DependencyWarning" not in proc.stderr, (
        f"SOCKS DependencyWarning leaked despite asi's filter:\n{proc.stderr}"
    )
