"""Browser automation tool handler for ToolRegistry.

Provides Playwright-based browser automation actions:
  - navigate: Open a URL and extract rendered page text
  - click:    Click on an element by CSS selector
  - type:     Type text into an input field
  - extract:  Extract text content from the current page
  - screenshot: Take a full-page screenshot (returns file path)
  - evaluate: Execute JavaScript in the page context
  - wait:     Wait for a CSS selector to appear or a timeout
  - close:    Close the browser and free resources

Usage:
    browser_action(action="navigate", url="https://example.com", timeout=30000)
    browser_action(action="click", selector="#submit-btn")
    browser_action(action="type", selector="#email", text="user@example.com")
    browser_action(action="extract", max_chars=15000)
    browser_action(action="screenshot")
    browser_action(action="evaluate", js="document.title")
    browser_action(action="wait", selector=".result-loaded")
    browser_action(action="close")
"""

from __future__ import annotations

import atexit
import contextlib
import importlib
import importlib.util
import logging
import os
import pathlib
import subprocess
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as _FutureTimeout
from typing import TYPE_CHECKING, Any

from external_llm.pip_env import ensure_user_site_importable, pip_install_flags

from ...client import interruptible_sleep
from ..agent_loop_types import AgentCancelled
from ..cancel_scope import (
    _CompositeCancel,
    call_cancel_scope,
    current_cancel_event,
    effective_cancel,
)

if TYPE_CHECKING:
    from ..tool_registry import ToolResult

logger = logging.getLogger(__name__)

# ── Optional Playwright dependency ───────────────────────────────────── #
# Availability is probed WITHOUT executing the package: importing
# playwright.sync_api costs ~22ms and this module is loaded on every
# ToolRegistry construction (BrowserActionToolsMixin is a base class, so it
# cannot be deferred), while the browser tools are used by almost no run.
# Same pattern as vector_cache's numpy/faiss/sentence_transformers probes.
HAS_PLAYWRIGHT = importlib.util.find_spec("playwright") is not None


# True only when the Chromium BROWSER BINARY is present, not just the Python
# package. ``HAS_PLAYWRIGHT`` reflects the package; ``playwright install`` (a
# separate ~150MB download) is required before any browser can launch.
# Distinguishing them lets tests SKIP on package-present/binary-absent
# machines instead of FAILING with ``Executable doesn't exist``. Worst case
# (Playwright cache layout change) this returns False and over-skips, never
# makes the suite red. Filesystem-only — no driver subprocess, no import cost.
def _playwright_browser_installed() -> bool:
    if not HAS_PLAYWRIGHT:
        return False
    base = os.environ.get("PLAYWRIGHT_BROWSERS_PATH")
    if base:
        cache = pathlib.Path(base)
    elif sys.platform == "darwin":
        cache = pathlib.Path.home() / "Library" / "Caches" / "ms-playwright"
    elif sys.platform == "win32":
        cache = pathlib.Path.home() / "AppData" / "Local" / "ms-playwright"
    else:
        cache = pathlib.Path.home() / ".cache" / "ms-playwright"
    return bool(cache.is_dir() and any(cache.glob("chromium-*/chrome-*")))


PLAYWRIGHT_BROWSER_AVAILABLE = _playwright_browser_installed()

# Bound on first use by _ensure_playwright_imported(); _reload_playwright_module()
# rebinds the same three names after a late install.
sync_playwright: Any = None  # callable once imported: sync_playwright().start()
_PlaywrightTimeout: Any = Exception


def _ensure_playwright_imported() -> bool:
    """Import playwright.sync_api on first use; return True when usable."""
    global sync_playwright, _PlaywrightTimeout
    if sync_playwright is not None:
        return True
    if not HAS_PLAYWRIGHT:
        return False
    try:
        from playwright.sync_api import TimeoutError as _PwTimeout
        from playwright.sync_api import sync_playwright as _sync_pw
    except ImportError as e:
        logger.debug("playwright import failed (HAS_PLAYWRIGHT was stale): %s", e)
        return False
    sync_playwright = _sync_pw
    _PlaywrightTimeout = _PwTimeout
    return True


# ── Dedicated single-thread executor for all Playwright work ─────────── #
# Playwright's sync API binds its greenlet driver to the thread that called
# sync_playwright().start(); the browser/page objects can only be used from
# that same thread. Tool calls, however, are dispatched on a shared
# ThreadPoolExecutor (design_chat_loop) where browser_action — a read tool —
# runs unserialized on any of N workers. Reusing the shared browser singleton
# from a different (or already-exited) worker raises Playwright's
# "cannot switch to a different thread (which happens to have exited)".
#
# Pinning every Playwright call to one persistent worker thread guarantees
# affinity (and incidentally serializes access to the single shared page).
# The worker is created lazily on first submit and lives for the process.
_browser_executor_lock = threading.Lock()


def _new_browser_executor() -> ThreadPoolExecutor:
    """Create a fresh single-thread executor pinned for Playwright work.

    Centralised so both the initial module-level executor and the wedge-recovery
    path (_reset_browser_on_wedge) build an identical executor (same affinity
    contract). Teardown is owned by a SINGLE module-level atexit handler
    (_shutdown_browser_executor_at_exit) that always references the current
    global — so repeated wedge recoveries do not pile up dead handlers, each of
    which would retain a reference to its (stuck-worker) executor until exit.
    """
    return ThreadPoolExecutor(max_workers=1, thread_name_prefix="browser-action")


_BROWSER_EXECUTOR = _new_browser_executor()


def _shutdown_browser_executor_at_exit() -> None:
    """Shut down the current global browser executor at process exit.

    Registered ONCE at import; always operates on whichever executor is current
    at exit time (the module global is reassigned on wedge recovery). Replaces
    the per-executor atexit.register that grew the handler list by one on every
    wedge and held dead executors (with their orphaned worker threads).
    """
    with contextlib.suppress(RuntimeError):  # shutdown() from within its own worker thread
        # Best-effort: cancel_futures (3.9+) drops queued submits; requires-python
        # is >=3.10, so no legacy fallback is needed. The *running* future cannot
        # be interrupted — the worker becomes an orphan by design.
        _BROWSER_EXECUTOR.shutdown(wait=False, cancel_futures=True)


atexit.register(_shutdown_browser_executor_at_exit)


def _reset_browser_on_wedge() -> None:
    """Recover from a hard-timeout wedge by abandoning the stuck worker.

    Called when a browser action exceeds ``_BROWSER_HARD_TIMEOUT_SEC``: the
    single dedicated worker is still blocked inside an uninterruptible
    Playwright call, so every subsequent submit would queue behind it and time
    out too — wedging the whole session until process restart.

    We abandon the stuck worker (its thread + Playwright driver become orphans;
    best-effort, reaped on process exit) and spin up a fresh executor. The
    class-level browser refs are cleared WITHOUT calling ``.close()``:
    Playwright's sync objects are thread-affine, so closing from this (caller)
    thread would itself hang. The next ``_get_browser()`` lazily re-initialises
    a brand-new browser on the new worker thread.
    """
    global _BROWSER_EXECUTOR
    with _browser_executor_lock:
        old = _BROWSER_EXECUTOR
        with contextlib.suppress(RuntimeError):  # shutdown() from within its own worker thread
            # cancel_futures (3.9+) drops queued submits; the *running* future
            # cannot be interrupted, so its worker becomes an orphan by design.
            old.shutdown(wait=False, cancel_futures=True)
        _BROWSER_EXECUTOR = _new_browser_executor()
        BrowserActionToolsMixin._page = None
        BrowserActionToolsMixin._browser = None
        BrowserActionToolsMixin._playwright = None
    logger.warning(
        "browser_action: hard timeout exceeded — abandoned the wedged worker "
        "thread and recreated the browser executor; a new browser will start on "
        "the next call."
    )


# Hard upper bound (seconds) for any single browser action on the dedicated
# executor. Playwright per-call timeouts above only apply to the specific page
# operation (goto, click, fill, wait_for_selector); extract/screenshot/evaluate
# carry NO per-call timeout and an ill-behaved page (infinite-loop JS, hung
# renderer) would block the executor worker forever, and in turn the calling
# shared_pool worker that blocks on .result(). This is a safety net so a stuck
# browser cannot wedge an entire agent session.
_BROWSER_HARD_TIMEOUT_SEC = 120

# Per-call Playwright timeout ceiling (ms). The LLM-supplied ``timeout`` arg is
# unbounded, so a generous value (e.g. 180000 for a slow page) would exceed the
# dedicated-executor hard timeout above: the hard timeout would fire FIRST,
# abandoning the worker via ``_reset_browser_on_wedge`` and destroying the whole
# browser session (login state, current page) — even though Playwright was still
# happily waiting within the requested budget. Clamp every per-call timeout
# below the hard ceiling (minus a margin) so a clean Playwright per-call timeout
# always resolves before the wedge path, leaving the session intact.
_PER_CALL_TIMEOUT_MARGIN_SEC = 5
_PER_CALL_TIMEOUT_CEIL_MS = max((_BROWSER_HARD_TIMEOUT_SEC - _PER_CALL_TIMEOUT_MARGIN_SEC) * 1000, 1000)


def _clamp_per_call_timeout_ms(requested: Any) -> int:
    """Clamp a caller-requested Playwright timeout (ms) below the hard ceiling.

    Returns at least 1000ms (Playwright rejects <= 0). Applied by every browser
    action that forwards a ``timeout`` to Playwright (navigate/click/type/wait)
    so the per-call timeout always resolves before ``_BROWSER_HARD_TIMEOUT_SEC``
    would trigger a session-resetting wedge. Bad/non-int input falls back to the
    standard 30000ms default.
    """
    try:
        requested = int(requested)
    except (TypeError, ValueError):
        requested = 30000
    return max(1000, min(requested, _PER_CALL_TIMEOUT_CEIL_MS))


class BrowserActionToolsMixin:
    """Mixin providing browser_action tool implementation for ToolRegistry.

    Maintains a singleton browser instance (lazy-initialized) across calls
    within the same session. Call ``browser_action(action="close")`` to
    release resources.
    """

    # ── Shared browser state (class-level, lazy) ──────────────────────── #
    _browser = None
    _playwright = None
    _page = None
    _user_agent = None  # de-headlessed UA, derived once per browser (see _browser_user_agent)
    _pw_install_lock = threading.Lock()  # serialise Playwright install across threads

    # ── Host contract (provided by ToolRegistry / AgentToolsMixin) ───── #
    # These names are supplied by the host classes this mixin is mounted
    # onto; class-level annotations keep pyright from flagging the calls
    # below (P29 pattern — same as web_search_tools / read_tools).
    _make_result: Any
    _tool_ask_user: Any
    repo_root: str

    # ── Public dispatch entry point ───────────────────────────────────── #

    def _live_cancel_event(self) -> threading.Event | _CompositeCancel | None:
        """The registry's current cancel events as a cooperative-channel source.

        Merges the agent-loop ``config.cancel_event`` (whole-turn ESC) with any
        per-call scope this dispatch runs under (MCP wait_for timeout / aborted
        parallel batch) via :func:`effective_cancel` — a mid-wait abandon must
        release the browser worker's pool slot too, not just a whole-turn ESC.
        Returns ``None`` when no source exists (no behavior change on serial
        dispatch); a single source is returned as-is (its plain Event).

        Defensive ``getattr``: the mixin is also mounted on duck-typed test
        hosts that carry no ``config`` attribute.
        """
        return effective_cancel(getattr(getattr(self, "config", None), "cancel_event", None))

    def _tool_browser_action(self, args: dict[str, Any]) -> ToolResult:
        """Browser automation: navigate, click, type, extract, screenshot, evaluate, wait, close."""
        action = str(args.get("action", "")).strip().lower()

        if not action:
            return self._make_result(
                ok=False,
                content="",
                error="'action' is required. Choose: navigate, click, type, extract, screenshot, evaluate, wait, close",
            )

        if not HAS_PLAYWRIGHT or not _ensure_playwright_imported():
            with BrowserActionToolsMixin._pw_install_lock:
                if not self._ensure_playwright_installed():
                    return self._make_result(
                        ok=False,
                        content="",
                        error=(
                            "Playwright is not available — automatic installation was "
                            "declined or failed.\n"
                            "Install manually:\n"
                            "  pip install playwright && playwright install chromium"
                        ),
                    )
            # Module-level names updated by _reload_playwright_module; proceed.

        _actions = {
            "navigate": self._browser_navigate,
            "click": self._browser_click,
            "type": self._browser_type,
            "extract": self._browser_extract,
            "screenshot": self._browser_screenshot,
            "evaluate": self._browser_evaluate,
            "wait": self._browser_wait,
            "close": self._browser_close,
        }

        handler = _actions.get(action)
        if handler is None:
            return self._make_result(
                ok=False,
                content="",
                error=f"Unknown action: '{action}'. Available: {', '.join(sorted(_actions))}",
            )

        # Per-call cancel scope capture: this method runs on the calling
        # (dispatch/executor) thread, but the handler itself runs on the
        # dedicated browser thread. The cooperative scope is thread-local, so
        # capture the caller's innermost scope here and re-install it on the
        # browser thread — an abandoned call (MCP timeout / aborted parallel
        # batch) must reach a mid-wait interrupt inside _browser_wait, not just
        # a whole-turn ESC. None when no scope (serial dispatch) → no-op.
        _caller_scope = current_cancel_event()

        def _run() -> ToolResult:
            try:
                if _caller_scope is not None:
                    with call_cancel_scope(_caller_scope):
                        return handler(args)
                return handler(args)
            except AgentCancelled:
                # ESC mid-action must abort the turn, not become a generic
                # "Browser action failed" ToolResult (B904).
                raise
            except _PlaywrightTimeout:
                return self._make_result(
                    ok=False,
                    content="",
                    error="Playwright timeout: page or element did not load within the specified timeout.",
                )
            except Exception as e:
                return self._make_result(
                    ok=False,
                    content="",
                    error=f"Browser action '{action}' failed: {type(e).__name__}: {e}",
                )

        # Run on the dedicated browser thread so the sync Playwright objects are
        # always created and used from the same thread (see _BROWSER_EXECUTOR).
        # The calling (shared_pool) worker blocks on the result, which is fine:
        # browser_action is inherently serial against its single shared page.
        #
        # A hard timeout caps the wait: extract/screenshot/evaluate carry no
        # per-call Playwright timeout and a hung page would otherwise block the
        # executor (and this caller) forever.
        try:
            return _BROWSER_EXECUTOR.submit(_run).result(timeout=_BROWSER_HARD_TIMEOUT_SEC)
        except _FutureTimeout:
            # The worker is still blocked inside the (uninterruptible) Playwright
            # call, so this submit would wedge every later browser_action. Recover
            # by abandoning the stuck worker and recreating the executor; a fresh
            # browser starts on the next call. Without this, the session stays
            # wedged until process restart.
            _reset_browser_on_wedge()
            return self._make_result(
                ok=False,
                content="",
                error=(
                    f"Browser action '{action}' did not complete within "
                    f"{_BROWSER_HARD_TIMEOUT_SEC}s. The page may be unresponsive; "
                    f"the browser session has been reset — retry the action."
                ),
            )

    # ── On-the-fly Playwright install with user consent ────────────────── #

    def _ensure_playwright_installed(self) -> bool:
        """Ensure Playwright is available — prompt, install, and reload if needed.

        Returns True if Playwright is now ready (either was already installed
        after a concurrent call, or was just installed successfully).
        """
        if HAS_PLAYWRIGHT:
            return True
        # Frozen (PyInstaller / py2exe / etc.) environments cannot run
        # sys.executable -m pip; skip auto-install and fall through to the
        # manual-instructions error path.
        if getattr(sys, "frozen", False):
            logger.info("browser_action: frozen environment detected, skipping automatic Playwright install")
            return False
        if not self._ask_install_playwright():
            return False
        if not self._install_playwright():
            return False
        return self._reload_playwright_module()

    def _ask_install_playwright(self) -> bool:
        """Ask the user if they want to install Playwright.

        Uses the agent's ask_user mechanism. Falls back to 'no' if
        checkpoint/prompting is unavailable.
        """
        try:
            result = self._tool_ask_user(
                {
                    "question": (
                        "Playwright (browser automation) is needed for the "
                        "browser_action tool but is not installed.\n\n"
                        "Install it now?\n"
                        "  pip install playwright && playwright install chromium"
                    ),
                    "type": "confirm",
                    "options": ["yes", "no"],
                    "default": "no",
                    "reason": "Playwright required for browser_action tool",
                }
            )
            answer = result.metadata.get("answer", "no").lower().strip()
        except Exception as e:
            logger.warning("browser_action: ask_user failed (%s), skipping Playwright install", e)
            return False
        else:
            return answer == "yes"

    @staticmethod
    def _pip_install_flags() -> list[str]:
        """Extra ``pip install`` flags required for the current environment.

        Thin delegate to the shared :func:`external_llm.pip_env.pip_install_flags`
        so browser / asi / (import-package) installers make the same PEP 668
        decision. Kept as a method so tests can patch it per-instance.
        """
        return pip_install_flags()

    def _install_playwright(self) -> bool:
        """Install Playwright Python package + Chromium browser via pip.

        Uses ``_pip_install_flags`` so the pip step works on PEP 668
        externally-managed environments too. The ``playwright install
        chromium`` step downloads browser binaries into a cache dir (not a
        Python package), so it is unaffected by PEP 668 and needs no flags.
        """
        flags = self._pip_install_flags()
        try:
            logger.info(
                "Installing playwright package%s...",
                " into user site (externally-managed env)" if flags else "",
            )
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "playwright", *flags],
                check=True,
                capture_output=True,
                timeout=120,
            )
            logger.info("Installing Chromium for Playwright...")
            subprocess.run(
                [sys.executable, "-m", "playwright", "install", "chromium"],
                check=True,
                capture_output=True,
                timeout=300,
            )
        except subprocess.CalledProcessError as e:
            stderr = e.stderr.decode("utf-8", errors="replace") if e.stderr else ""
            logger.exception("Playwright installation failed (rc=%d): %s", e.returncode, stderr)
            return False
        except Exception as e:
            logger.exception("Playwright installation failed: %s", e)
            return False
        else:
            return True

    def _reload_playwright_module(self) -> bool:
        """Dynamically import Playwright after install and update module-level refs.

        After ``pip install playwright`` the package becomes importable.
        Updates ``HAS_PLAYWRIGHT``, ``sync_playwright``, and
        ``_PlaywrightTimeout`` in the module's global namespace so existing
        code paths (guard, ``_get_browser``, ``_run`` exception handler) pick
        up the new values without requiring a process restart.
        """
        global sync_playwright, _PlaywrightTimeout, HAS_PLAYWRIGHT
        try:
            # A just-completed ``--user`` install may land in a user-site dir
            # that was absent (hence not on sys.path) at interpreter startup;
            # ensure it is importable now, and drop stale import caches so the
            # freshly written package files are discovered.
            ensure_user_site_importable()
            importlib.invalidate_caches()
            sync_mod = importlib.import_module("playwright.sync_api")
            sync_playwright = sync_mod.sync_playwright
            _PlaywrightTimeout = sync_mod.TimeoutError
            HAS_PLAYWRIGHT = True
        except ImportError as e:
            logger.exception("Failed to import Playwright after installation: %s", e)
            return False
        else:
            return True

    # ── Browser lifecycle helpers ─────────────────────────────────────── #

    def _get_browser(self) -> Any:
        """Lazy-init and return the shared Playwright browser instance."""
        if BrowserActionToolsMixin._browser is None:
            # Callers reach here only past a HAS_PLAYWRIGHT guard, but bind
            # explicitly so a direct _get_browser() cannot hit `None()`.
            _ensure_playwright_imported()
            p = sync_playwright().start()
            try:
                BrowserActionToolsMixin._browser = p.chromium.launch(headless=True)
            except Exception:
                # launch() failed (missing browser binary, sandbox error, …).
                # Stop the just-started Playwright driver so its node process
                # does not leak — otherwise _browser stays None and the next
                # call starts ANOTHER driver, accumulating orphans. Re-raise so
                # the caller surfaces the real launch error.
                try:
                    p.stop()
                except Exception as _exc:  # driver teardown must not mask the launch error
                    logger.debug("playwright driver stop failed: %s", _exc)
                raise
            BrowserActionToolsMixin._playwright = p
        return BrowserActionToolsMixin._browser

    def _browser_user_agent(self):
        """The shared browser's UA with the ``Headless`` marker removed.

        Chromium's headless build advertises ``HeadlessChrome/<ver>`` in its
        User-Agent, and that literal substring is by itself enough for some
        anti-bot systems to refuse the request. Measured 2026-08-05 against
        Startpage from one IP within a few minutes: headless with the default UA
        was blocked 6/6, a HEADED browser passed 3/3, and the same headless
        browser with only this substring rewritten passed 6/6. So the block was
        never the IP or "being automated" — it was the UA string.

        Derived from the running browser rather than hardcoded so it tracks
        whatever Chromium ``playwright install`` put on the machine (149.x here)
        instead of drifting the way a literal would. Returns None if the probe
        fails, which simply leaves the page on Playwright's default UA.
        """
        if BrowserActionToolsMixin._user_agent is None:
            browser = self._get_browser()
            try:
                # Raw new_page(): this probe must not route back through here.
                probe = browser.new_page()
                try:
                    default_ua = probe.evaluate("navigator.userAgent")
                finally:
                    probe.close()
                BrowserActionToolsMixin._user_agent = default_ua.replace("HeadlessChrome/", "Chrome/")
            except Exception as e:  # probe is best-effort; never block a render on it
                logger.debug("browser: user-agent probe failed (%s); using the default", e)
                return None
        return BrowserActionToolsMixin._user_agent

    def _get_page(self) -> Any:
        """Get or create a page in the shared browser.

        Recreates the page if the existing one was closed (crash / user close).
        """
        browser = self._get_browser()
        page = BrowserActionToolsMixin._page
        if page is None or page.is_closed():
            BrowserActionToolsMixin._page = browser.new_page(user_agent=self._browser_user_agent())
        return BrowserActionToolsMixin._page

    def _close_shared_browser(self):
        """Release all browser resources."""
        try:
            if BrowserActionToolsMixin._page:
                BrowserActionToolsMixin._page.close()
        except Exception as _exc:  # teardown must never crash; log for diagnosability
            logger.debug("browser page close failed: %s", _exc)
        try:
            if BrowserActionToolsMixin._browser:
                BrowserActionToolsMixin._browser.close()
        except Exception as _exc:  # teardown must never crash; log for diagnosability
            logger.debug("browser close failed: %s", _exc)
        try:
            if BrowserActionToolsMixin._playwright:
                BrowserActionToolsMixin._playwright.stop()
        except Exception as _exc:  # teardown must never crash; log for diagnosability
            logger.debug("playwright stop failed: %s", _exc)
        BrowserActionToolsMixin._page = None
        BrowserActionToolsMixin._browser = None
        BrowserActionToolsMixin._playwright = None
        # The next browser may be a different Chromium build, so its UA must be
        # re-derived rather than inherited from the one just torn down.
        BrowserActionToolsMixin._user_agent = None

    def _render_and_eval(
        self,
        url: str,
        js: str,
        *,
        timeout_ms: int = 20000,
        wait_until: str = "networkidle",
        wait_for_selector: str | None = None,
    ) -> Any:
        """Navigate an ISOLATED throwaway page to ``url``, run ``js``, return its value.

        Reusable browser primitive for backends that need a real (JS-rendering)
        browser rather than an httpx scrape — currently ``search_web``'s Naver
        engine. Two deliberate properties:

        * **Isolated page.** Uses a fresh ``browser.new_page()`` that is closed
          afterwards, NOT the shared ``_page``, so an automated background render
          never clobbers the user's interactive ``browser_action`` session (open
          tabs, login/cookie state).

        ``wait_for_selector`` waits for a specific element after navigation. For a
        client-hydrated results page this is both the correct and the FAST option:
        measured on Startpage 2026-08-05, ``domcontentloaded``/``load`` alone
        returned a page that parsed to 0 results (5/5 queries — the markup had not
        hydrated yet), ``networkidle`` parsed 10/10 but cost ~6.5s, and
        ``domcontentloaded`` + this selector wait parsed 10/10 in ~3.1s.
        * **Same executor contract as ``_tool_browser_action``.** Runs on the
          dedicated single-thread ``_BROWSER_EXECUTOR`` (Playwright sync objects
          are thread-affine) under the same ``_BROWSER_HARD_TIMEOUT_SEC`` +
          wedge-recovery net, so a hung render cannot wedge the session.

        Returns the JSON-serialisable value produced by ``js`` (whatever
        ``page.evaluate`` returns). Raises ``RuntimeError`` when Playwright is
        unavailable or the render wedges; Playwright per-call timeouts / eval
        errors propagate as their own exception types for the caller to handle.
        """
        if not HAS_PLAYWRIGHT or not _ensure_playwright_imported():
            with BrowserActionToolsMixin._pw_install_lock:
                if not self._ensure_playwright_installed():
                    raise RuntimeError("Playwright is not available (automatic install declined or failed)")

        per_call = _clamp_per_call_timeout_ms(timeout_ms)
        if wait_until not in ("load", "domcontentloaded", "networkidle", "commit"):
            wait_until = "networkidle"

        def _run() -> Any:
            browser = self._get_browser()
            # isolated — never the shared _page
            page = browser.new_page(user_agent=self._browser_user_agent())
            try:
                page.goto(url, timeout=per_call, wait_until=wait_until)
                if wait_for_selector:
                    page.wait_for_selector(wait_for_selector, timeout=per_call)
                return page.evaluate(js)
            finally:
                try:
                    page.close()
                except Exception as _exc:  # teardown must never crash; log for diagnosability
                    logger.debug("browser page close failed: %s", _exc)

        # Mirrors _tool_browser_action's submit contract: pin to the browser
        # thread, cap with the hard timeout, and on a wedge abandon the stuck
        # worker + recreate the executor (see _reset_browser_on_wedge) so later
        # browser work is not blocked behind the hung render.
        try:
            return _BROWSER_EXECUTOR.submit(_run).result(timeout=_BROWSER_HARD_TIMEOUT_SEC)
        except _FutureTimeout:
            _reset_browser_on_wedge()
            raise RuntimeError(
                f"browser render did not complete within {_BROWSER_HARD_TIMEOUT_SEC}s; the browser session was reset"
            ) from None

    def _screenshot_dir(self) -> str:
        """Return the screenshots directory under ``.asicode`` (not the repo root).

        Writing to ``<repo_root>/screenshots/`` polluted the user's working tree
        and showed up in ``git status``; ``.asicode`` is the established tooling
        scratch dir (memory.md, design_sessions, …), so screenshots live there.
        """
        d = os.path.join(self.repo_root, ".asicode", "screenshots")
        os.makedirs(d, exist_ok=True)
        return d

    # ── Action handlers ───────────────────────────────────────────────── #

    def _browser_navigate(self, args: dict[str, Any]) -> ToolResult:
        url = str(args.get("url", "")).strip()
        timeout = _clamp_per_call_timeout_ms(args.get("timeout", 30000))
        max_chars = int(args.get("max_chars", 15000))
        max_chars = max(1000, min(max_chars, 50000))

        # Default to "load" rather than "networkidle": real pages with ads,
        # analytics, or long-polling rarely reach network-idle and instead burn
        # the full timeout. Callers can still opt into a stricter wait.
        wait_until = str(args.get("wait_until", "load")).strip().lower()
        if wait_until not in ("load", "domcontentloaded", "networkidle", "commit"):
            wait_until = "load"

        if not url:
            return self._make_result(ok=False, content="", error="'url' is required for navigate action")

        page = self._get_page()
        page.goto(url, timeout=timeout, wait_until=wait_until)

        text = page.inner_text("body")
        title = page.title()

        # Capture the real content length BEFORE appending the truncation marker
        # so metadata["length"] reflects actual content (not the ~90-char
        # informational suffix) and "total_length" tells the caller how much was
        # clipped — mirroring web_fetch's reported_len/total_length contract.
        total_len = len(text)
        reported_len = min(total_len, max_chars)
        if total_len > max_chars:
            text = text[:max_chars] + f"\n\n...[TRUNCATED at {max_chars} chars]..."

        final_url = page.url
        result = f"Title: {title}\nURL: {final_url}\n\n{text}"
        return self._make_result(
            ok=True,
            content=result,
            metadata={"title": title, "url": final_url, "length": reported_len, "total_length": total_len},
        )

    def _browser_click(self, args: dict[str, Any]) -> ToolResult:
        selector = str(args.get("selector", "")).strip()
        timeout = _clamp_per_call_timeout_ms(args.get("timeout", 30000))

        if not selector:
            return self._make_result(ok=False, content="", error="'selector' is required for click action")

        page = self._get_page()
        page.click(selector, timeout=timeout)
        page.wait_for_load_state("domcontentloaded")

        return self._make_result(
            ok=True,
            content=f"Clicked '{selector}'",
            metadata={"selector": selector},
        )

    def _browser_type(self, args: dict[str, Any]) -> ToolResult:
        selector = str(args.get("selector", "")).strip()
        text = args.get("text", "")
        timeout = _clamp_per_call_timeout_ms(args.get("timeout", 30000))

        if not selector:
            return self._make_result(ok=False, content="", error="'selector' and 'text' are required for type action")

        page = self._get_page()
        page.fill(selector, str(text), timeout=timeout)

        snippet = text[:50] + "..." if len(text) > 50 else text
        return self._make_result(
            ok=True,
            content=f"Typed '{snippet}' into '{selector}'",
            metadata={"selector": selector, "text_length": len(text)},
        )

    def _browser_extract(self, args: dict[str, Any]) -> ToolResult:
        page = self._get_page()
        text = page.inner_text("body")
        title = page.title()
        url = page.url

        max_chars = int(args.get("max_chars", 15000))
        max_chars = max(1000, min(max_chars, 50000))
        # See _browser_navigate: report real content length + total_length, not
        # the marker-inflated len(text).
        total_len = len(text)
        reported_len = min(total_len, max_chars)
        if total_len > max_chars:
            text = text[:max_chars] + f"\n\n...[TRUNCATED at {max_chars} chars]..."

        result = f"Title: {title}\nURL: {url}\n\n{text}"
        return self._make_result(
            ok=True,
            content=result,
            metadata={"title": title, "url": url, "length": reported_len, "total_length": total_len},
        )

    def _browser_screenshot(self, args: dict[str, Any]) -> ToolResult:
        page = self._get_page()
        filename = f"browser_{int(time.time())}_{uuid.uuid4().hex[:6]}.png"
        filepath = os.path.join(self._screenshot_dir(), filename)

        page.screenshot(path=filepath, full_page=True)

        return self._make_result(
            ok=True,
            content=f"Screenshot saved to {filepath}",
            metadata={"filepath": filepath, "url": page.url},
        )

    def _browser_evaluate(self, args: dict[str, Any]) -> ToolResult:
        js = str(args.get("js", "")).strip()

        if not js:
            return self._make_result(ok=False, content="", error="'js' is required for evaluate action")

        page = self._get_page()
        result = page.evaluate(js)

        return self._make_result(
            ok=True,
            content=str(result),
            metadata={"result_type": type(result).__name__},
        )

    def _browser_wait(self, args: dict[str, Any]) -> ToolResult:
        selector = args.get("selector")
        timeout = _clamp_per_call_timeout_ms(args.get("timeout", 30000))

        page = self._get_page()

        if selector:
            page.wait_for_selector(str(selector), timeout=timeout)
            return self._make_result(ok=True, content=f"Selector '{selector}' appeared.")
        # Clamped above, so a no-selector wait can never sleep past the hard
        # executor ceiling (which would trip the session-resetting wedge).
        wait_ms = max(timeout, 1)
        # interruptible_sleep (client.py SSOT): a mid-wait ESC aborts the turn
        # instead of freezing it for up to the full clamped timeout on the
        # single browser worker (which would also block every later action).
        if interruptible_sleep(wait_ms / 1000, self._live_cancel_event()):
            raise AgentCancelled("browser wait cancelled by user")
        return self._make_result(
            ok=True,
            content=f"Waited {wait_ms / 1000:.1f}s (no selector given).",
        )

    def _browser_close(self, args: dict[str, Any]) -> ToolResult:
        self._close_shared_browser()
        return self._make_result(ok=True, content="Browser closed and resources released.")
