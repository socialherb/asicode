"""Environment-aware pip flags for in-process auto-installers (PEP 668).

Several code paths install optional dependencies on demand into the running
interpreter's environment (``asi._pip_install`` for tree-sitter / vector / the
Claude SDK, ``browser_tools`` for Playwright). On PEP 668 *externally-managed*
environments (Homebrew Python, Debian/Ubuntu system Python) a plain
``pip install`` is refused, so these installers need extra flags.

``pip_install_flags`` centralizes that decision so every **import-package**
installer makes the SAME choice: outside a virtualenv, install into the user
site with ``--user --break-system-packages`` — the managed *system* tree is
never touched and the user site is on ``sys.path``, so an in-process import
still succeeds. Inside a virtualenv no flags are added (venvs are never
externally-managed, and pip forbids ``--user`` there).

IMPORTANT — this is for packages that get **imported**. It is deliberately NOT
used by ``external_llm.languages.dependency_checker``, which installs **CLI
tools** (ruff, pyright) resolved via ``shutil.which``: ``pip install --user``
drops the console script into a user ``bin`` dir that is usually not on
``$PATH``, so a ``--user`` install would read back as "still missing". That
path intentionally keeps a plain ``--break-system-packages`` (system tree, so
the script lands on ``$PATH``).
"""
from __future__ import annotations

import os
import sys
import sysconfig


def pip_install_flags() -> list[str]:
    """Extra ``pip install`` flags required for the current environment.

    Returns ``["--user", "--break-system-packages"]`` on a PEP 668
    externally-managed environment outside a virtualenv, else ``[]``.

    The externally-managed state is detected structurally (the PEP 668
    ``EXTERNALLY-MANAGED`` marker in the stdlib dir), not by parsing pip's
    localized stderr — the same marker pip itself keys off, so the decision is
    consistent with pip's own refusal.
    """
    in_venv = sys.prefix != sys.base_prefix
    if in_venv:
        return []
    stdlib = sysconfig.get_path("stdlib") or ""
    if os.path.exists(os.path.join(stdlib, "EXTERNALLY-MANAGED")):
        return ["--user", "--break-system-packages"]
    return []


def ensure_user_site_importable() -> None:
    """Make a (possibly just-created) user-site dir importable in-process.

    A ``--user`` install may land in a user-site directory that did not exist
    at interpreter startup and so was never added to ``sys.path``. After such
    an install, appending it lets the freshly written package import without a
    process restart. No-op when user site is disabled, absent, or already on
    the path.
    """
    try:
        import site
        user_site = site.getusersitepackages()
    except Exception:
        return
    if user_site and os.path.isdir(user_site) and user_site not in sys.path:
        sys.path.append(user_site)
