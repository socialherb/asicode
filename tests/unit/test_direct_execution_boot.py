"""Regression guard for the ``python asi.py`` direct-execution boot path.

Two latent bugs, both previously masked because ``--version``/``--help`` exit
inside argparse *before* reaching the crash site:

1. **main()-before-barrel ordering** — ``if __name__ == "__main__": main()``
   sat at ~line 3427, ABOVE the P6-2 barrel re-export
   ``from external_llm.repl.repl_impl import (...)`` at ~line 3435. Running
   ``python asi.py`` therefore called ``main()`` before the barrel bound
   ``_resolve_repo_root`` / ``run_repl`` / ``run_once`` into the module
   namespace → ``NameError`` at the very first statement of ``main()``
   (``_repo_root = _resolve_repo_root(args.repo)``).

2. **__main__ circular re-execution** — even after moving ``main()`` past the
   barrel, ``import asi`` inside ``repl_impl`` re-executes asi.py as a *separate*
   module object ``asi`` (distinct from the running ``__main__``), which then
   hits the barrel re-export again and fails with a circular-import
   ``ImportError``. Fixed by aliasing the running module under ``asi`` in
   ``sys.modules`` near the top of asi.py so the one-way cycle resolves to the
   already-running object (no re-execution).

The installed entry point ``from asi import main`` (asi.py imported as module
``asi`` with ``__name__ == "asi"``) was never affected — both bugs are specific
to ``python asi.py`` where ``__name__ == "__main__"``.

The guard invocation ``--prompt-stdin`` with empty stdin reaches
``_resolve_repo_root`` (~line 3317, the original crash site) and the barrel
import, then exits 1 at the "empty input from stdin" check — *before*
``run_repl`` / ``run_once`` boot vector/tree-sitter, so it completes in ~150ms.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ASI_PY = REPO / "asi.py"


def _run(args: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(ASI_PY), *args],
        cwd=str(REPO),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=30,
        **kw,
        check=False,
    )


class TestDirectExecutionBoot:
    """``python asi.py`` must boot past the barrel import without crashing."""

    def test_version_exits_clean(self):
        """Smoke: --version resolves inside argparse (exits before line 3317)."""
        proc = _run(["--version"])
        assert proc.returncode == 0
        assert "asicode" in (proc.stdout + proc.stderr).lower()

    def test_help_exits_clean(self):
        """Smoke: --help resolves inside argparse."""
        proc = _run(["--help"])
        assert proc.returncode == 0

    def test_direct_execution_reaches_resolve_repo_root(self):
        """THE regression: ``python asi.py`` must get past ``_resolve_repo_root``
        (~line 3317) AND the barrel re-export without ``NameError`` (fix #1) or a
        circular-import ``ImportError`` (fix #2).

        ``--prompt-stdin`` + empty stdin reaches that code path then exits 1 at
        the empty-stdin guard — a fast path that never boots the REPL.
        """
        proc = _run(["--prompt-stdin"])
        combined = proc.stdout + proc.stderr
        # A clean exit 1 from the empty-stdin guard, NOT a Python crash:
        assert "Traceback" not in combined
        assert "empty input from stdin" in combined
        assert proc.returncode == 1

    def test_normal_import_entrypoint_unaffected(self):
        """``from asi import main`` (the installed ``asi`` command) still works:
        the sys.modules guard is a no-op when ``__name__ == "asi"`` (not
        ``"__main__"``), and the barrel-bound public names are callable."""
        proc = subprocess.run(
            [
                sys.executable,
                "-c",
                "import asi; "
                "assert asi.__name__ == 'asi', asi.__name__; "
                "assert callable(asi.main); "
                "assert callable(asi.run_repl); "
                "assert callable(asi.run_once); "
                "assert callable(asi._resolve_repo_root); "
                "print('OK')",
            ],
            cwd=str(REPO),
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        assert proc.returncode == 0, proc.stderr
        assert proc.stdout.strip() == "OK"
