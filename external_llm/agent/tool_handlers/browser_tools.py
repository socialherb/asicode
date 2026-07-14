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
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as _FutureTimeout
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..tool_registry import ToolResult
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
            return self._make_result(
                ok=False, content="",
                error=(
                    "Playwright is not installed. Install it with:\n"
                    "  pip install playwright && playwright install chromium"
                ),
            )

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

    # ── Browser lifecycle helpers ─────────────────────────────────────── #

    def _get_browser(self):
        """Lazy-init and return the shared Playwright browser instance."""
        if BrowserActionToolsMixin._browser is None:
            p = sync_playwright().start()
            BrowserActionToolsMixin._playwright = p
            BrowserActionToolsMixin._browser = p.chromium.launch(headless=True)
        return BrowserActionToolsMixin._browser

    def _get_page(self):
        """Get or create a page in the shared browser."""
        browser = self._get_browser()
        if BrowserActionToolsMixin._page is None:
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

        result = f"Title: {title}\nURL: {url}\n\n{text}"
        return self._make_result(
            ok=True, content=result,
            metadata={"title": title, "url": url, "length": len(text)},
        )

    def _browser_click(self, args: dict[str, Any]) -> "ToolResult":
        selector = str(args.get("selector", "")).strip()
        timeout = int(args.get("timeout", 30000))

        if not selector:
            return self._make_result(ok=False, content="", error="'selector' is required for click action")

        page = self._get_page()
        page.click(selector, timeout=timeout)
        time.sleep(0.3)

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
            time.sleep(1)
            return self._make_result(ok=True, content="Waited 1 second (no selector given).")

    def _browser_close(self, args: dict[str, Any]) -> "ToolResult":
        self._close_shared_browser()
        return self._make_result(ok=True, content="Browser closed and resources released.")
