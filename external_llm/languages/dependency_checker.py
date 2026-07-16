#!/usr/bin/env python3
"""Semantic validation tool dependency check + interactive install.

Decorates :mod:`asi` startup with a one-time check for
``pyright``, ``tsc``, and ``go``—the tools needed for
:meth:`SyntaxProvider.validate_semantics`.  Missing tools are
discovered before the agent loop starts, and the user is prompted
to install them interactively.

Graceful degradation
--------------------
- If ``npm`` is absent, pyright can fall back to ``pip install pyright``
  (tsc has no pip fallback).
- If neither installation method works, the tool is skipped
  (semantic validation degrades gracefully at runtime in each provider).
- Non-interactive sessions (pipe/redirect) skip prompts automatically.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
from dataclasses import dataclass
from typing import Optional

from .models import LanguageId

_logger = logging.getLogger(__name__)

# ── Persisted dismissal state ────────────────────────────────────────────────
# Per-user (machine-global): a tool's availability is a property of the host,
# not of any single repo.  When the user chooses "pretend installed" or "skip"
# we record it here so the prompt does not recur on every launch.  Deleting
# this file (or installing the tool) restores the default behaviour.
_STATE_PATH = os.path.join(os.path.expanduser("~"), ".asicode", "tool_state.json")
_STATE_VERSION = 1


# ── Tool descriptors ────────────────────────────────────────────────────────


@dataclass
class _Tool:
    """A single optional semantic-validation tool.

    Instances in :data:`_LANGUAGE_TOOL_MAP` / :data:`_TOOLS` are **templates**:
    they carry only the static spec (``use_npx`` flags install strategy) and
    default runtime state (``found=False``).  Callers must build fresh copies
    via :func:`_clone_tools` so that mutable state (``found`` / ``skipped`` /
    ``version``) never leaks between invocations or between the install prompt
    loop and the status-line renderer.
    """

    #: CLI command name.  When ``use_npx`` is True the provider actually runs
    #: ``npx <cmd>`` and this name is *not* required on ``$PATH`` globally.
    cmd: str
    #: Human-readable label.
    label: str
    #: ``None`` means "cannot auto-install, instructions only".
    npm_package: Optional[str] = None
    #: Fallback pip package (pyright only).
    pip_package: Optional[str] = None
    #: Install command shown to the user when auto-install is impossible.
    manual_hint: str = ""
    #: If True, the provider invokes this tool via ``npx <cmd>`` (e.g. tsc,
    #: eslint).  Such tools are considered available when ``npx`` is on
    #: ``$PATH`` even if the bare command is not globally installed, because
    #: npx transparently resolves a local ``node_modules/.bin`` entry or
    #: fetches on demand.
    use_npx: bool = False
    #: Whether the tool was found usable (runtime state — reset per check).
    found: bool = False
    #: Whether the user chose to skip (runtime state — reset per check).
    skipped: bool = False
    #: Version string (runtime state — reset per check).
    version: str = ""
    #: True when the user chose "Mark as done (pretend installed)" — distinguishes
    #: a pretend-found tool from one genuinely resolved on ``$PATH`` (runtime state).
    pretend_installed: bool = False


_TOOLS: list[_Tool] = [
    _Tool(
        cmd="pyright",
        label="Pyright (Python type checker)",
        npm_package="pyright",
        pip_package="pyright",
    ),
    _Tool(
        cmd="tsc",
        label="TypeScript compiler (TS type checker)",
        npm_package="typescript",
        use_npx=True,
    ),
    _Tool(
        cmd="go",
        label="Go toolchain (go vet/gofmt)",
        manual_hint=(
            "Install Go from https://go.dev/dl/ or via your package manager\n"
            "  macOS : brew install go\n"
            "  Ubuntu: sudo apt install golang-go\n"
            "  Fedora: sudo dnf install golang"
        ),
    ),
    _Tool(
        cmd="javac",
        label="JDK (javac — Java semantic checker)",
        manual_hint=(
            "Install a JDK from https://adoptium.net/ or via your package manager\n"
            "  macOS : brew install openjdk@17\n"
            "  Ubuntu: sudo apt install default-jdk\n"
            "  Fedora: sudo dnf install java-*-openjdk-devel"
        ),
    ),
    _Tool(
        cmd="kotlinc",
        label="Kotlin compiler (kotlinc — Kotlin semantic checker)",
        manual_hint=(
            "Install the Kotlin compiler from https://kotlinlang.org/docs/command-line.html\n"
            "  macOS : brew install kotlin\n"
            "  SDKMAN: sdk install kotlin"
        ),
    ),
]

# ── Language → tool mapping ────────────────────────────────────────────────

_LANGUAGE_TOOL_MAP: dict[LanguageId, list[_Tool]] = {
    LanguageId.PYTHON: [
        _Tool(
            cmd="pyright",
            label="Pyright (Python type checker)",
            npm_package="pyright",
            pip_package="pyright",
        ),
        _Tool(
            cmd="ruff",
            label="Ruff (Python linter & F821 fixer)",
            pip_package="ruff",
        ),
    ],
    LanguageId.TYPESCRIPT: [
        _Tool(
            cmd="tsc",
            label="TypeScript compiler (TS type checker)",
            npm_package="typescript",
            use_npx=True,
        ),
        _Tool(
            cmd="eslint",
            label="ESLint (JS/TS linter)",
            npm_package="eslint",
            use_npx=True,
        ),
    ],
    LanguageId.JAVASCRIPT: [
        _Tool(
            cmd="tsc",
            label="TypeScript compiler (JS type checker via --checkJs)",
            npm_package="typescript",
            use_npx=True,
        ),
        _Tool(
            cmd="eslint",
            label="ESLint (JS/TS linter)",
            npm_package="eslint",
            use_npx=True,
        ),
    ],
    LanguageId.GO: [
        _Tool(
            cmd="go",
            label="Go toolchain (go vet/gofmt)",
            manual_hint=(
                "Install Go from https://go.dev/dl/ or via your package manager\n"
                "  macOS : brew install go\n"
                "  Ubuntu: sudo apt install golang-go\n"
                "  Fedora: sudo dnf install golang"
            ),
        ),
    ],
    LanguageId.JAVA: [
        _Tool(
            cmd="javac",
            label="JDK (javac — Java semantic checker)",
            manual_hint=(
                "Install a JDK from https://adoptium.net/ or via your package manager\n"
                "  macOS : brew install openjdk@17\n"
                "  Ubuntu: sudo apt install default-jdk\n"
                "  Fedora: sudo dnf install java-*-openjdk-devel"
            ),
        ),
    ],
    LanguageId.KOTLIN: [
        _Tool(
            cmd="kotlinc",
            label="Kotlin compiler (kotlinc — Kotlin semantic checker)",
            manual_hint=(
                "Install the Kotlin compiler from https://kotlinlang.org/docs/command-line.html\n"
                "  macOS : brew install kotlin\n"
                "  SDKMAN: sdk install kotlin"
            ),
        ),
    ],
    LanguageId.C: [
        _Tool(
            cmd="gcc",
            label="GCC / Clang (gcc — C syntax checker)",
            manual_hint=(
                "Install a C compiler via your package manager\n"
                "  macOS : xcode-select --install   (provides clang)\n"
                "         brew install gcc\n"
                "  Ubuntu: sudo apt install gcc\n"
                "  Fedora: sudo dnf install gcc\n"
                "clang is used automatically if gcc is absent."
            ),
        ),
    ],
    LanguageId.CPP: [
        _Tool(
            cmd="g++",
            label="GCC / Clang (g++ — C++ syntax checker)",
            manual_hint=(
                "Install a C++ compiler via your package manager\n"
                "  macOS : xcode-select --install   (provides clang++)\n"
                "         brew install gcc\n"
                "  Ubuntu: sudo apt install g++\n"
                "  Fedora: sudo dnf install gcc-c++\n"
                "clang++ is used automatically if g++ is absent."
            ),
        ),
    ],
}


# ── helpers ──────────────────────────────────────────────────────────────────


def _detect_version(cmd: str) -> str:
    """Try to get a one-line version string for *cmd*.

    Most tools support ``<cmd> --version``, but Go requires ``go version``.
    """
    try:
        # Go uses "go version", not "go --version"
        version_cmd = [cmd, "version"] if cmd == "go" else [cmd, "--version"]
        proc = subprocess.run(
            version_cmd,
            capture_output=True, text=True, timeout=5,
        )
        if proc.returncode == 0:
            return proc.stdout.strip() or proc.stderr.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return ""


def _is_interactive() -> bool:
    """True if stdin is a real TTY (safe to prompt)."""
    try:
        return sys.stdin.isatty() and sys.stdout.isatty()
    except (OSError, AttributeError):
        return False


def _ask_yes_no(prompt: str, default: bool = True) -> bool:
    """Ask a Y/n or y/N question via stdin.

    Returns *default* on empty input or EOF.
    """
    suffix = " [Y/n] " if default else " [y/N] "
    try:
        raw = input(prompt + suffix).strip().lower()
    except (EOFError, KeyboardInterrupt):
        sys.stdout.write("\n")
        return default
    if raw in ("y", "yes"):
        return True
    if raw in ("n", "no"):
        return False
    return default


def _npm_install(package: str) -> bool:
    """Install an npm global package.

    Returns True on success.
    """
    _logger.debug("Installing npm package %r globally\u2026", package)
    print(f"    $ npm install -g {package}")
    try:
        proc = subprocess.run(
            ["npm", "install", "-g", package],
            capture_output=True, text=True, timeout=120,
        )
    except FileNotFoundError:
        print("    \u2717 npm not found \u2014 cannot install")
        return False
    except subprocess.TimeoutExpired:
        print("    \u2717 npm install timed out (120s)")
        return False
    except OSError as e:
        print(f"    \u2717 install failed: {e}")
        return False

    if proc.returncode != 0:
        lines = (proc.stderr or "").strip().splitlines()[-3:]
        for ln in lines:
            print(f"      {ln}")
        print(f"    \u2717 npm install failed (exit {proc.returncode})")
        return False

    print("    \u2713 Installed")
    return True


def _pip_install(package: str) -> bool:
    """Install a Python package via pip."""
    _logger.debug("Installing pip package %r\u2026", package)
    print(f"    $ pip install {package}")
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pip", "install", package],
            capture_output=True, text=True, timeout=120,
        )
    except subprocess.TimeoutExpired:
        print("    \u2717 pip install timed out (120s)")
        return False
    except OSError as e:
        print(f"    \u2717 install failed: {e}")
        return False

    if proc.returncode != 0:
        combined = (proc.stderr or "") + "\n" + (proc.stdout or "")
        for line in combined.strip().splitlines()[-3:]:
            print(f"      {line}")
        # PEP 668: externally-managed. NOTE: unlike the import-package
        # installers (asi / browser_tools, which use
        # external_llm.pip_env.pip_install_flags \u2192 --user --break-system-packages),
        # this path installs CLI tools (ruff, pyright) resolved via
        # shutil.which. --user drops the console script into a user bin dir
        # usually not on $PATH, so we deliberately install into the system tree
        # with --break-system-packages (no --user) to keep the script findable.
        if "externally-managed-environment" in combined:
            print(
                "    \u21b3 Python externally managed (PEP 668) "
                "\u2014 retrying with --break-system-packages"
            )
            try:
                proc = subprocess.run(
                    [sys.executable, "-m", "pip", "install",
                     "--break-system-packages", package],
                    capture_output=True, text=True, timeout=120,
                )
            except (OSError, subprocess.SubprocessError) as e2:
                print(f"    \u2717 retry failed: {e2}")
                return False
            if proc.returncode != 0:
                print(f"    \u2717 install failed (exit {proc.returncode})")
                return False
        else:
            print(f"    \u2717 install failed (exit {proc.returncode})")
            return False
    print("    \u2713 Installed")
    return True


# ── repo language detection ───────────────────────────────────────────────────


def detect_repo_languages(repo_root: str) -> set[LanguageId]:
    """Scan *repo_root* tracked files and return the set of LanguageId present.

    Uses ``git ls-files`` for speed and to skip ignored/generated files.
    Returns an empty set on any failure (non-git repo, timeout, etc.).
    """
    try:
        out = subprocess.run(
            ["git", "ls-files"],
            cwd=repo_root, capture_output=True, text=True, timeout=5,
        )
        files = out.stdout.splitlines() if out.returncode == 0 else []
    except (OSError, subprocess.SubprocessError):
        files = []

    langs: set[LanguageId] = set()
    for path in files:
        lang = LanguageId.from_path(path)
        if lang != LanguageId.UNKNOWN:
            langs.add(lang)
    return langs


def _clone_tools(langs) -> list[_Tool]:
    """Build fresh ``_Tool`` instances for *langs*, deduplicated by ``cmd``.

    The entries in :data:`_LANGUAGE_TOOL_MAP` are shared **templates**; this
    factory returns private copies so that the install-prompt loop and the
    status-line renderer never observe each other's mutable state (``found``
    / ``skipped`` / ``version``) across calls.
    """
    seen: set[str] = set()
    fresh: list[_Tool] = []
    for lang in langs:
        for tmpl in _LANGUAGE_TOOL_MAP.get(lang, []):
            if tmpl.cmd in seen:
                continue
            seen.add(tmpl.cmd)
            # dataclass(replace=False) — explicit copy keeps it cheap and clear.
            fresh.append(_Tool(
                cmd=tmpl.cmd, label=tmpl.label,
                npm_package=tmpl.npm_package, pip_package=tmpl.pip_package,
                manual_hint=tmpl.manual_hint, use_npx=tmpl.use_npx,
            ))
    return fresh


def _resolve_tool(t: _Tool) -> bool:
    """Decide whether *t* is usable, honoring ``use_npx``.

    Resolution order mirrors what the providers actually run:

    1. Bare command on ``$PATH`` (``pyright``, ``ruff``, ``go``, …) — works for
       tools the providers invoke directly.
    2. For ``use_npx`` tools (``tsc``, ``eslint``), the provider runs
       ``npx <cmd>``.  npx resolves a local ``node_modules/.bin`` entry or
       fetches on demand, so the bare command need *not* be globally
       installed — the presence of ``npx`` itself is sufficient.
    """
    if shutil.which(t.cmd):
        return True
    if t.use_npx and shutil.which("npx"):
        return True
    return False


def _check_tools(tools: list[_Tool]) -> None:
    """Populate ``found`` / ``version`` on each tool via :func:`_resolve_tool`."""
    for t in tools:
        usable = _resolve_tool(t)
        t.found = usable
        if usable and not t.use_npx:
            # npx-fetched tools have no stable local version to probe.
            t.version = _detect_version(t.cmd)
        else:
            t.version = ""


def _load_tool_state() -> dict[str, str]:
    """Return persisted dismissal decisions as ``{cmd: 'pretend' | 'skip'}``.

    Returns ``{}`` when the state file is absent or unreadable so a missing or
    corrupt file degrades to the legacy re-prompt-every-launch behaviour.
    """
    try:
        with open(_STATE_PATH, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    out: dict[str, str] = {}
    pretend = data.get("pretend", {})
    skipped = data.get("skipped", {})
    if isinstance(pretend, dict):
        out.update({str(k): "pretend" for k, v in pretend.items() if v})
    if isinstance(skipped, dict):
        out.update({str(k): "skip" for k, v in skipped.items() if v})
    return out


def _save_tool_state(decisions: dict[str, str]) -> None:
    """Atomically persist dismissal *decisions* (``{cmd: 'pretend'|'skip'}``).

    Failures are logged at debug level and swallowed — persistence is a
    best-effort convenience, never a hard requirement.
    """
    payload = {
        "version": _STATE_VERSION,
        "pretend": {k: True for k, v in decisions.items() if v == "pretend"},
        "skipped": {k: True for k, v in decisions.items() if v == "skip"},
    }
    base = os.path.dirname(_STATE_PATH) or "."
    try:
        os.makedirs(base, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=base, prefix=".tool_state_", suffix=".json")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2)
            os.replace(tmp, _STATE_PATH)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
    except OSError:
        _logger.debug("failed to persist tool state to %s", _STATE_PATH, exc_info=True)


def _sync_tool_state(tools: list[_Tool], baseline: dict[str, str]) -> None:
    """Reconcile persisted dismissal state with *tools*' current flags.

    The state file is **machine-global** (shared across every repo on the
    host).  *tools* holds only the tools relevant to the *current* repo's
    detected language set, so we must rewrite dismissal state for exactly
    those tools while **preserving** entries for tools that belong to other
    repos' languages — otherwise launching asi from a repo that does not
    contain (say) Kotlin would silently erase the user's ``kotlinc`` dismissal,
    making the prompt recur next time they open a Kotlin project.

    Reconciliation rules
    --------------------
    * A tool in *tools* that carries ``pretend_installed`` / ``skipped`` is
      "dismissed" and recorded as such.
    * A tool in *tools* that is genuinely ``found`` (no flag) is no longer
      dismissed → dropped.  This is what clears a stale entry once a tool is
      actually installed.
    * A tool **not** in *tools* (absent from this repo's languages) is left
      exactly as it was in *baseline* — we did not examine it this run, so we
      must not infer anything about it.

    No write occurs when the derived set matches *baseline*.
    """
    decisions: dict[str, str] = {}
    examined: set[str] = set()
    for t in tools:
        examined.add(t.cmd)
        if t.pretend_installed:
            decisions[t.cmd] = "pretend"
        elif t.skipped:
            decisions[t.cmd] = "skip"
    # Preserve dismissals for tools whose language is absent from this repo.
    # The global state spans all repos; only the tools we actually examined
    # this run may be re-derived.
    for cmd, dec in baseline.items():
        if cmd not in examined:
            decisions[cmd] = dec
    if decisions != baseline:
        _save_tool_state(decisions)
def _check_tools_with_state(
    langs,
    *,
    no_prompt: bool = False,
) -> list[_Tool]:
    """Core check + install routine returning fresh ``_Tool`` instances.

    This is the state-bearing core used by :func:`check_and_install_all`
    and by callers that need the richer ``skipped``/``version`` fields
    (e.g. the REPL status line).

    Persistence
    -----------
    Dismissal decisions ("pretend installed" / "skip") are loaded from the
    per-user state file (:data:`_STATE_PATH`) *before* prompting, so a tool
    the user already dismissed is not re-prompted on every launch.  Genuine
    ``$PATH`` availability always wins: a tool that becomes truly installed
    is reported as ``found`` and cleared from the dismissal set.  State is
    reconciled (and stale entries dropped) on every call, interactive or
    not; fresh decisions are written after the interactive prompt loop.

    Parameters
    ----------
    langs : set[LanguageId]
        Languages present in the repository.
    no_prompt : bool
        If True, skip prompts and just report status.

    Returns
    -------
    list[_Tool]
        Fresh tool instances with ``found`` / ``skipped`` / ``version``
        populated by this run.  Never aliases the module-level templates.
    """
    # 1. Build fresh, private tool instances (no shared mutable state).
    wanted = _clone_tools(langs)
    if not wanted:
        return []

    # 2. Scan all tools (npx-aware)
    _check_tools(wanted)

    # 2b. Apply persisted dismissal decisions so the prompt does not recur on
    #     every launch. Only act on tools NOT genuinely available — a tool
    #     later actually installed is reported truthfully as 'found'.
    persisted = _load_tool_state()
    for t in wanted:
        if t.found:
            continue
        decision = persisted.get(t.cmd)
        if decision == "pretend":
            t.found = True
            t.pretend_installed = True
        elif decision == "skip":
            t.skipped = True

    # 3. Report status — but only print the block when there is something
    #    missing *and* we're in an interactive session (avoid CI noise).
    missing = [t for t in wanted if not t.found and not t.skipped]
    if missing and _is_interactive():
        _print_status_block(wanted)

    if not _is_interactive() or no_prompt or not missing:
        # Still reconcile state (drops stale dismissals) even when not prompting.
        _sync_tool_state(wanted, persisted)
        return wanted

    # 4. Interactive install loop — prompt only for tools that are still
    #    missing AND not previously dismissed.
    print()
    for t in missing:
        if t.skipped:
            continue
        _prompt_and_install(t)

    # 5. Re-check after installs (npx-aware)
    for t in wanted:
        if not t.found and not t.skipped:
            if _resolve_tool(t):
                t.found = True
                if not t.use_npx:
                    t.version = _detect_version(t.cmd)

    # 6. Persist dismissal decisions (incl. new ones from the prompt loop).
    _sync_tool_state(wanted, persisted)

    return wanted


# ── public API ───────────────────────────────────────────────────────────────


def check_and_install_all(
    *,
    no_prompt: bool = False,
    include_go: bool = True,
) -> dict[str, bool]:
    """Check semantic-validation tools and interactively install missing ones.

    Legacy entry point operating on the flat :data:`_TOOLS` list.
    Callers needing per-language scoping should call
    :func:`_check_tools_with_state` directly with a filtered language set.

    Parameters
    ----------
    no_prompt : bool
        If True, skip prompts and just report status.
    include_go : bool
        If True, also check Go.  (Go is harder to auto-install so some
        callers may choose to skip in non-interactive mode.)

    Returns
    -------
    dict[str, bool]
        Mapping of tool command name → whether it is available *after*
        the check (and any install).
    """
    # Build fresh private instances from the shared templates so the
    # module-level _TOOLS never carries stale found/skipped state.
    templates = _TOOLS if include_go else [t for t in _TOOLS if t.cmd != "go"]
    tools = [_Tool(
        cmd=t.cmd, label=t.label, npm_package=t.npm_package,
        pip_package=t.pip_package, manual_hint=t.manual_hint, use_npx=t.use_npx,
    ) for t in templates]

    # 1. Scan all tools (npx-aware)
    _check_tools(tools)

    # 2. Report status for missing tools — only when interactive (avoid CI noise)
    missing = [t for t in tools if not t.found]
    if not missing:
        return {t.cmd: True for t in tools}

    if _is_interactive():
        _print_status_block(tools)

    if not _is_interactive() or no_prompt:
        return {t.cmd: t.found for t in tools}

    # 3. Interactive install loop
    print()
    for t in missing:
        if t.skipped:
            continue
        _prompt_and_install(t)

    # 4. Re-check after installs (npx-aware)
    for t in tools:
        if not t.found and not t.skipped:
            if _resolve_tool(t):
                t.found = True
                if not t.use_npx:
                    t.version = _detect_version(t.cmd)

    return {t.cmd: t.found for t in tools}


def _print_status_block(tools: list[_Tool]) -> None:
    """Print a compact status block."""
    print()
    print("\u2550" * 50)
    print("  \U0001f527  Semantic validation tools")
    print()
    for t in tools:
        if t.found:
            ver = f" ({t.version})" if t.version else ""
            print(f"    \u2713 {t.cmd:<12} found{ver}")
        elif t.skipped:
            print(f"    \u2015 {t.cmd:<12} skipped by user")
        else:
            print(f"    \u2717 {t.cmd:<12} not found")
    print()


def _prompt_and_install(t: _Tool) -> None:
    """Ask user whether to install *t* and do it."""
    print(f"  {t.label}")

    # Determine install strategy
    methods: list[tuple[str, str, str]] = []  # (method_name, pkg, kind)

    if t.npm_package:
        if shutil.which("npm"):
            methods.append(("npm", t.npm_package, "npm"))
        else:
            print("    (npm not available on $PATH)")

    if t.pip_package:
        methods.append(("pip", t.pip_package, "pip"))

    if t.manual_hint:
        if not methods:
            print("    Auto-install not available.")
            print(textwrap.indent(t.manual_hint, "    "))
            if _ask_yes_no("    Mark as done (pretend installed)", default=False):
                t.found = True
                t.pretend_installed = True
            else:
                t.skipped = True
            return

    if not methods:
        t.skipped = True
        return

    # Ask
    if not _ask_yes_no("    Install now", default=True):
        t.skipped = True
        print("    \u2015 Skipped (semantic validation disabled for this language)")
        return

    # Try each method in order
    for _method_name, pkg, kind in methods:
        if kind == "npm":
            ok = _npm_install(pkg)
        elif kind == "pip":
            ok = _pip_install(pkg)
        else:
            continue
        if ok:
            t.found = True
            return

    print("    \u2717 All install methods failed \u2014 skipping")
    t.skipped = True


# ── standalone dry-run ---


def main() -> None:
    """CLI entry point \u2014 dry-run without prompting."""
    result = check_and_install_all(no_prompt=True)
    print()
    print("  Summary:")
    for cmd, ok in result.items():
        _chk = "\u2713" if ok else "\u2717"
        print(f"    {_chk} {cmd}")
    print()
    sys.exit(0 if all(result.values()) else 1)


if __name__ == "__main__":
    main()
