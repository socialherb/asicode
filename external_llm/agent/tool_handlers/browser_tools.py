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
import importlib
import logging
import os
import subprocess
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as _FutureTimeout
from typing import TYPE_CHECKING, Any

from external_llm.pip_env import ensure_user_site_importable, pip_install_flags

if TYPE_CHECKING:
    from ..tool_registry import ToolResult

logger = logging.getLogger(__name__)

# ── Optional Playwright dependency ───────────────────────────────────── #
try:
    from playwright.sync_api import TimeoutError as _PlaywrightTimeout
    from playwright.sync_api import sync_playwright
    HAS_PLAYWRIGHT = True
except ImportError:
    HAS_PLAYWRIGHT = False
    _PlaywrightTimeout = Exception

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
_BROWSER_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="browser-action")
atexit.register(_BROWSER_EXECUTOR.shutdown, wait=False)

# Hard upper bound (seconds) for any single browser action on the dedicated
# executor. Playwright per-call timeouts above only apply to the specific page
# operation (goto, click, fill, wait_for_selector); extract/screenshot/evaluate
# carry NO per-call timeout and an ill-behaved page (infinite-loop JS, hung
# renderer) would block the executor worker forever, and in turn the calling
# shared_pool worker that blocks on .result(). This is a safety net so a stuck
# browser cannot wedge an entire agent session.
_BROWSER_HARD_TIMEOUT_SEC = 120


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
    _pw_install_lock = threading.Lock()  # serialise Playwright install across threads

    # ── Public dispatch entry point ───────────────────────────────────── #

    def _tool_browser_action(self, args: dict[str, Any]) -> "ToolResult":
        """Browser automation: navigate, click, type, extract, screenshot, evaluate, wait, close."""
        action = str(args.get("action", "")).strip().lower()

        if not action:
            return self._make_result(
                ok=False, content="",
                error="'action' is required. Choose: navigate, click, type, extract, screenshot, evaluate, wait, close",
            )

        if not HAS_PLAYWRIGHT:
            with BrowserActionToolsMixin._pw_install_lock:
                if not self._ensure_playwright_installed():
                    return self._make_result(
                        ok=False, content="",
                        error=(
                            "Playwright is not available — automatic installation was "
                            "declined or failed.\n"
                            "Install manually:\n"
                            "  pip install playwright && playwright install chromium"
                        ),
                    )
            # Module-level names updated by _reload_playwright_module; proceed.

        _ACTIONS = {
            "navigate": self._browser_navigate,
            "click": self._browser_click,
            "type": self._browser_type,
            "extract": self._browser_extract,
            "screenshot": self._browser_screenshot,
            "evaluate": self._browser_evaluate,
            "wait": self._browser_wait,
            "close": self._browser_close,
        }

        handler = _ACTIONS.get(action)
        if handler is None:
            return self._make_result(
                ok=False, content="",
                error=f"Unknown action: '{action}'. Available: {', '.join(sorted(_ACTIONS))}",
            )

        def _run() -> "ToolResult":
            try:
                return handler(args)
            except _PlaywrightTimeout:
                return self._make_result(
                    ok=False, content="",
                    error="Playwright timeout: page or element did not load within the specified timeout.",
                )
            except Exception as e:
                return self._make_result(
                    ok=False, content="",
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
            return _BROWSER_EXECUTOR.submit(_run).result(
                timeout=_BROWSER_HARD_TIMEOUT_SEC
            )
        except _FutureTimeout:
            return self._make_result(
                ok=False, content="",
                error=(
                    f"Browser action '{action}' did not complete within "
                    f"{_BROWSER_HARD_TIMEOUT_SEC}s. The page may be unresponsive."
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
            result = self._tool_ask_user({
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
            })
            answer = result.metadata.get("answer", "no").lower().strip()
            return answer == "yes"
        except Exception as e:
            logger.warning("browser_action: ask_user failed (%s), skipping Playwright install", e)
            return False

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
                check=True, capture_output=True, timeout=120,
            )
            logger.info("Installing Chromium for Playwright...")
            subprocess.run(
                [sys.executable, "-m", "playwright", "install", "chromium"],
                check=True, capture_output=True, timeout=300,
            )
            return True
        except subprocess.CalledProcessError as e:
            stderr = e.stderr.decode("utf-8", errors="replace") if e.stderr else ""
            logger.error("Playwright installation failed (rc=%d): %s", e.returncode, stderr)
            return False
        except Exception as e:
            logger.error("Playwright installation failed: %s", e)
            return False

    def _reload_playwright_module(self) -> bool:
        """Dynamically import Playwright after install and update module-level refs.

        After ``pip install playwright`` the package becomes importable.
        Updates ``HAS_PLAYWRIGHT``, ``sync_playwright``, and
        ``_PlaywrightTimeout`` in the module's global namespace so existing
        code paths (guard, ``_get_browser``, ``_run`` exception handler) pick
        up the new values without requiring a process restart.
        """
        try:
            # A just-completed ``--user`` install may land in a user-site dir
            # that was absent (hence not on sys.path) at interpreter startup;
            # ensure it is importable now, and drop stale import caches so the
            # freshly written package files are discovered.
            ensure_user_site_importable()
            importlib.invalidate_caches()
            sync_mod = importlib.import_module("playwright.sync_api")
            mod = sys.modules[__name__]
            mod.sync_playwright = sync_mod.sync_playwright
            mod._PlaywrightTimeout = sync_mod.TimeoutError
            mod.HAS_PLAYWRIGHT = True
            return True
        except ImportError as e:
            logger.error("Failed to import Playwright after installation: %s", e)
            return False
    # ── Browser lifecycle helpers ─────────────────────────────────────── #

    def _get_browser(self):
        """Lazy-init and return the shared Playwright browser instance."""
        if BrowserActionToolsMixin._browser is None:
            p = sync_playwright().start()
            BrowserActionToolsMixin._playwright = p
            BrowserActionToolsMixin._browser = p.chromium.launch(headless=True)
        return BrowserActionToolsMixin._browser

    def _get_page(self):
        """Get or create a page in the shared browser.

        Recreates the page if the existing one was closed (crash / user close).
        """
        browser = self._get_browser()
        page = BrowserActionToolsMixin._page
        if page is None or page.is_closed():
            BrowserActionToolsMixin._page = browser.new_page()
        return BrowserActionToolsMixin._page

    def _close_shared_browser(self):
        """Release all browser resources."""
        try:
            if BrowserActionToolsMixin._page:
                BrowserActionToolsMixin._page.close()
        except Exception:
            pass
        try:
            if BrowserActionToolsMixin._browser:
                BrowserActionToolsMixin._browser.close()
        except Exception:
            pass
        try:
            if BrowserActionToolsMixin._playwright:
                BrowserActionToolsMixin._playwright.stop()
        except Exception:
            pass
        BrowserActionToolsMixin._page = None
        BrowserActionToolsMixin._browser = None
        BrowserActionToolsMixin._playwright = None

    def _screenshot_dir(self) -> str:
        """Return the screenshots directory (relative to repo root)."""
        d = os.path.join(self.repo_root, "screenshots")
        os.makedirs(d, exist_ok=True)
        return d

    # ── Action handlers ───────────────────────────────────────────────── #

    def _browser_navigate(self, args: dict[str, Any]) -> "ToolResult":
        url = str(args.get("url", "")).strip()
        timeout = int(args.get("timeout", 30000))
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

        if len(text) > max_chars:
            text = text[:max_chars] + f"\n\n...[TRUNCATED at {max_chars} chars]..."

        final_url = page.url
        result = f"Title: {title}\nURL: {final_url}\n\n{text}"
        return self._make_result(
            ok=True, content=result,
            metadata={"title": title, "url": final_url, "length": len(text)},
        )

    def _browser_click(self, args: dict[str, Any]) -> "ToolResult":
        selector = str(args.get("selector", "")).strip()
        timeout = int(args.get("timeout", 30000))

        if not selector:
            return self._make_result(ok=False, content="", error="'selector' is required for click action")

        page = self._get_page()
        page.click(selector, timeout=timeout)
        page.wait_for_load_state("domcontentloaded")

        return self._make_result(
            ok=True, content=f"Clicked '{selector}'",
            metadata={"selector": selector},
        )

    def _browser_type(self, args: dict[str, Any]) -> "ToolResult":
        selector = str(args.get("selector", "")).strip()
        text = args.get("text", "")
        timeout = int(args.get("timeout", 30000))

        if not selector:
            return self._make_result(ok=False, content="", error="'selector' and 'text' are required for type action")

        page = self._get_page()
        page.fill(selector, str(text), timeout=timeout)

        snippet = text[:50] + "..." if len(text) > 50 else text
        return self._make_result(
            ok=True, content=f"Typed '{snippet}' into '{selector}'",
            metadata={"selector": selector, "text_length": len(text)},
        )

    def _browser_extract(self, args: dict[str, Any]) -> "ToolResult":
        page = self._get_page()
        text = page.inner_text("body")
        title = page.title()
        url = page.url

        max_chars = int(args.get("max_chars", 15000))
        max_chars = max(1000, min(max_chars, 50000))
        if len(text) > max_chars:
            text = text[:max_chars] + f"\n\n...[TRUNCATED at {max_chars} chars]..."

        result = f"Title: {title}\nURL: {url}\n\n{text}"
        return self._make_result(
            ok=True, content=result,
            metadata={"title": title, "url": url, "length": len(text)},
        )

    def _browser_screenshot(self, args: dict[str, Any]) -> "ToolResult":
        page = self._get_page()
        filename = f"browser_{int(time.time())}_{uuid.uuid4().hex[:6]}.png"
        filepath = os.path.join(self._screenshot_dir(), filename)

        page.screenshot(path=filepath, full_page=True)

        return self._make_result(
            ok=True, content=f"Screenshot saved to {filepath}",
            metadata={"filepath": filepath, "url": page.url},
        )

    def _browser_evaluate(self, args: dict[str, Any]) -> "ToolResult":
        js = str(args.get("js", "")).strip()

        if not js:
            return self._make_result(ok=False, content="", error="'js' is required for evaluate action")

        page = self._get_page()
        result = page.evaluate(js)

        return self._make_result(
            ok=True, content=str(result),
            metadata={"result_type": type(result).__name__},
        )

    def _browser_wait(self, args: dict[str, Any]) -> "ToolResult":
        selector = args.get("selector")
        timeout = int(args.get("timeout", 30000))

        page = self._get_page()

        if selector:
            page.wait_for_selector(str(selector), timeout=timeout)
            return self._make_result(ok=True, content=f"Selector '{selector}' appeared.")
        else:
            wait_ms = max(timeout, 1)
            time.sleep(wait_ms / 1000)
            return self._make_result(
                ok=True,
                content=f"Waited {wait_ms / 1000:.1f}s (no selector given).",
            )

    def _browser_close(self, args: dict[str, Any]) -> "ToolResult":
        self._close_shared_browser()
        return self._make_result(ok=True, content="Browser closed and resources released.")
