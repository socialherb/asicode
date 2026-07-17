"""Web search tool handler for ToolRegistry.

Provides web search via multiple backends:
    1. SearXNG self-hosted (set SEARXNG_BASE_URL env var, or auto-install with Docker)
       — private, self-hosted, preferred. If SEARXNG_BASE_URL is unset but Docker/Colima
       is available, the user is prompted to install SearXNG on first search.
    2. DuckDuckGo via html.duckduckgo.com/html/ (no setup needed) — fallback
    3. Brave Search API (set BRAVE_API_KEY env var) — clean JSON API

Usage: search_web(query="...", max_results=5, site_filter="...")
"""

from __future__ import annotations

import html as html_mod
import logging
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.parse
from html.parser import HTMLParser
from typing import TYPE_CHECKING, Any, Optional

import httpx

if TYPE_CHECKING:
    from ..tool_registry import ToolResult

logger = logging.getLogger(__name__)

# Browser User-Agent shared by all HTTP-based backends (DDG html, web_fetch) so a
# UA change is a one-line edit instead of a duplicated literal drifting apart.
_BROWSER_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)

# Transient (retryable) HTTP errors shared by all search backends. A single tuple
# keeps SearXNG / DuckDuckGo / Brave on one retry policy instead of each backend
# ad-hoc-listing a subset (SearXNG used to be the only one with any retry).
_TRANSIENT_HTTP_ERRORS = (
    httpx.ConnectError,
    httpx.RemoteProtocolError,
    httpx.ReadTimeout,
    httpx.ConnectTimeout,
    httpx.PoolTimeout,
)

class _DDGResultParser(HTMLParser):
    """Structured HTML parser that extracts search results from DuckDuckGo search page.

    Uses html.parser.HTMLParser (stdlib) instead of regex, making it resilient
    to HTML structure changes inside result blocks, nested tags, and attribute
    ordering differences. Approach mirrors Crush's tokenizer-based parsing.
    """

    def __init__(self, max_results: int = 10):
        super().__init__()
        self.max_results = max_results
        self.results: list[dict[str, str]] = []

        # Parser state
        self._current: dict[str, str] | None = None
        self._text_parts: list[str] = []
        self._capturing = False
        self._in_result_a = False
        self._in_snippet = False

    # ── helpers ──

    @staticmethod
    def _get_attr(attrs: list[tuple[str, str | None]], name: str, default: str = "") -> str:
        for n, v in attrs:
            if n == name:
                return v or default
        return default

    @staticmethod
    def _has_class(attrs: list[tuple[str, str | None]], cls: str) -> bool:
        class_val = _DDGResultParser._get_attr(attrs, "class")
        return cls in class_val.split() if class_val else False

    @staticmethod
    def _decode_ddg_url(href: str) -> str:
        """Decode DuckDuckGo's /l/?uddg= redirect URLs."""
        if "uddg=" in href:
            qs = urllib.parse.urlparse(href).query
            decoded = urllib.parse.parse_qs(qs).get("uddg", [None])[0]
            return urllib.parse.unquote(decoded) if decoded else href
        return href

    # ── parser callbacks ──

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if len(self.results) >= self.max_results:
            return

        if tag == "a" and self._has_class(attrs, "result__a"):
            self._in_result_a = True
            self._capturing = True
            self._text_parts = []
            href = self._decode_ddg_url(self._get_attr(attrs, "href"))
            self._current = {"url": href, "title": "", "snippet": ""}

        elif tag == "a" and self._has_class(attrs, "result__snippet"):
            self._in_snippet = True
            self._capturing = True
            self._text_parts = []

    def handle_endtag(self, tag: str) -> None:
        if len(self.results) >= self.max_results:
            return

        if tag == "a" and self._in_result_a and self._current is not None:
            self._current["title"] = html_mod.unescape("".join(self._text_parts)).strip()
            self._in_result_a = False
            self._capturing = False
            self._current["_got_title"] = True  # marker for finalization check

        elif tag == "a" and self._in_snippet:
            if self._current is not None and self._current.get("_got_title"):
                self._current["snippet"] = html_mod.unescape("".join(self._text_parts)).strip()
                # Drop the internal finalization marker before exposing the result dict
                # to callers (it leaked into every returned result previously).
                self._current.pop("_got_title", None)
                # Only emit if we collected a title (legitimate result)
                if self._current.get("title"):
                    self.results.append(self._current)
            # Reset for next result block
            self._in_snippet = False
            self._capturing = False
            self._current = None

    def handle_data(self, data: str) -> None:
        if self._capturing:
            self._text_parts.append(data)


class WebSearchToolsMixin:
    """Mixin providing web search tool implementations for ToolRegistry."""

    # ── Session-level decision cache: avoid duplicate Checkpoints ──
    # _ask_start_searxng / _ask_install_searxng cache the user's answer
    # for the lifetime of the ToolRegistry. This prevents the same user
    # Checkpoint from being issued twice in one turn when the LLM retries
    # search_web after all backends fail.
    # None = unasked; True = yes (start/install); False = no (skip)
    _searxng_start_decision: Optional[bool] = None
    _searxng_install_decision: Optional[bool] = None

    def _tool_search_web(self, args: dict[str, Any]) -> "ToolResult":
        """Search the web for information relevant to the current task.

        Supports three backends in priority order:
            1. SearXNG            (env SEARXNG_BASE_URL) — preferred
            2. DuckDuckGo         (no setup, always available)
            3. Brave Search API   (env BRAVE_API_KEY)

        Automatic fallback chain: if the primary backend fails, the next is
        tried transparently so the caller always gets results if any backend
        is reachable.
        """
        query = str(args.get("query", "")).strip()
        max_results = int(args.get("max_results", 5))
        max_results = max(1, min(max_results, 15))  # safety clamp

        site_filter = str(args.get("site_filter", "")).strip()
        if site_filter:
            query = f"{query} site:{site_filter}"

        if not query:
            return self._make_result(ok=False, content="", error="'query' is required")

        brave_key = os.environ.get("BRAVE_API_KEY", "")
        searxng_url = os.environ.get("SEARXNG_BASE_URL", "")

        # ── Build ordered backend list with fallback (SearXNG → DuckDuckGo → Brave) ──
        backends: list[tuple] = []
        if searxng_url:
            backends.append(("SearXNG", lambda: self._search_searxng(query, max_results, searxng_url)))
        elif self._has_docker_or_colima():
            # SEARXNG_BASE_URL not set, but Docker/Colima available → offer to install
            backends.append(("SearXNG", lambda: self._setup_and_search_searxng(query, max_results)))
        backends.append(("DuckDuckGo", lambda: self._search_duckduckgo(query, max_results)))
        if brave_key:
            backends.append(("Brave", lambda: self._search_brave(query, max_results, brave_key)))

        # ── Try each backend with fallback ──
        last_error = ""
        results: list[dict[str, str]] = []
        for name, search_fn in backends:
            try:
                results = search_fn()
                if results:
                    logger.info("web_search: %s returned %d results for '%s'", name, len(results), query)
                    break
            except (httpx.ConnectError, httpx.RemoteProtocolError) as e:
                if name == "SearXNG":
                    last_error = self._handle_searxng_connect_error(e, query, max_results, searxng_url)
                    # _handle_searxng_connect_error may have retried after installing;
                    # if results is now populated, we have a successful install+search
                    if last_error is None:
                        # Retry the search OUTSIDE the ConnectError handler above. A
                        # fresh connect failure here must fall through to the next
                        # backend (DuckDuckGo) instead of crashing search_web — wrap it.
                        try:
                            results = self._search_searxng(query, max_results, os.environ.get("SEARXNG_BASE_URL", "http://localhost:8080"))
                            if results:
                                logger.info("web_search: SearXNG succeeded after install")
                                break
                        except Exception as retry_err:
                            logger.warning("web_search: SearXNG retry after install failed (%s), falling back", retry_err)
                            last_error = f"SearXNG: {retry_err}"
                    continue
                last_error = f"{name}: {e}"
                logger.warning("web_search: %s failed (%s), trying next backend", name, e)
                continue
            except Exception as e:
                last_error = f"{name}: {e}"
                logger.warning("web_search: %s failed (%s), trying next backend", name, e)
                continue

        if not results:
            error_msg = last_error or "No results found from any backend."
            return self._make_result(ok=True, content=f"No results found. ({error_msg})", metadata={"result_count": 0})

        # ── Format response ──
        lines = [f"Web search results for: {query}", ""]
        for i, r in enumerate(results, 1):
            lines.append(f"{i}. {r['title']}")
            lines.append(f"   URL: {r['url']}")
            if r.get("snippet"):
                snippet = r["snippet"][:400]
                lines.append(f"   {snippet}")
            lines.append("")

        content = "\n".join(lines).strip()
        return self._make_result(
            ok=True,
            content=content if content else "(empty results)",
            metadata={"result_count": len(results), "query": query},
        )

    # ── Shared HTTP retry helper ─────────────────────────────────────

    @staticmethod
    def _http_request_with_retry(
        client: httpx.Client,
        method: str,
        url: str,
        *,
        params: Optional[dict] = None,
        data: Optional[dict] = None,
        headers: Optional[dict] = None,
        retries: int = 2,
        backoff: float = 1.5,
    ) -> httpx.Response:
        """Execute an HTTP GET/POST with transient-error retry.

        Shared by all three search backends so SearXNG / DuckDuckGo / Brave use one
        retry policy. Non-transient errors (e.g. HTTPStatusError / 4xx) are re-raised
        immediately; transient errors are retried ``retries`` times with ``backoff``
        seconds between attempts, then the last error is propagated to the caller's
        fallback chain.
        """
        last_err: Optional[Exception] = None
        for attempt in range(retries):
            try:
                if method.upper() == "GET":
                    return client.get(url, params=params, headers=headers)
                return client.post(url, data=data, headers=headers)
            except _TRANSIENT_HTTP_ERRORS as e:
                last_err = e
                if attempt < retries - 1:
                    logger.warning(
                        "web_search: transient HTTP error (attempt %d/%d: %s), retrying in %.1fs…",
                        attempt + 1, retries, e, backoff,
                    )
                    time.sleep(backoff)
                    continue
                raise  # final attempt failed — propagate
        if last_err is None:  # unreachable: retries >= 1 guarantees a try
            raise AssertionError("invariant: retries>=1 guarantees last_err is set")
        raise last_err

    # ── DuckDuckGo (no API key, via duckduckgo_search) ──────────────

    def _search_duckduckgo(self, query: str, max_results: int) -> list[dict[str, str]]:
        """Search using DuckDuckGo via html.duckduckgo.com/html/ (no API key needed).

        Uses _DDGResultParser (html.parser.HTMLParser) for structured HTML parsing
        instead of regex, making it resilient to HTML structure changes.
        """
        url = "https://html.duckduckgo.com/html/"
        headers = {
            "User-Agent": _BROWSER_UA,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Content-Type": "application/x-www-form-urlencoded",
        }
        data = {"q": query}

        with httpx.Client(timeout=15.0, follow_redirects=True, headers=headers) as client:
            resp = self._http_request_with_retry(client, "POST", url, data=data)
            resp.raise_for_status()

        parser = _DDGResultParser(max_results=max_results)
        parser.feed(resp.text)
        return parser.results

    def _search_brave(self, query: str, max_results: int, api_key: str) -> list[dict[str, str]]:
        """Search using Brave Search API (requires BRAVE_API_KEY)."""
        url = "https://api.search.brave.com/res/v1/web/search"
        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": api_key,
        }
        params = {"q": query, "count": max_results}

        with httpx.Client(timeout=15.0) as client:
            resp = self._http_request_with_retry(client, "GET", url, params=params, headers=headers)
            resp.raise_for_status()
            data = resp.json()

        results: list[dict[str, str]] = []
        for item in data.get("web", {}).get("results", []):
            results.append(
                {
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "snippet": item.get("description", ""),
                }
            )
        return results

    # ── SearXNG (self-hosted) ────────────────────────────────────────

    def _search_searxng(self, query: str, max_results: int, base_url: str) -> list[dict[str, str]]:
        """Search using self-hosted SearXNG instance, with retry on transient errors."""
        base_url = base_url.rstrip("/")
        url = f"{base_url}/search"
        params = {
            "q": query,
            "format": "json",
            "language": "en-US",
            "categories": "general",
            "pageno": 1,
        }

        # Retry policy is shared with the other backends via _http_request_with_retry
        # (previously SearXNG was the only backend with any retry; DDG/Brave had none).
        with httpx.Client(timeout=15.0, follow_redirects=True) as client:
            resp = self._http_request_with_retry(client, "GET", url, params=params)
            resp.raise_for_status()
            data = resp.json()

        results: list[dict[str, str]] = []
        for item in data.get("results", []):
            if len(results) >= max_results:
                break
            results.append(
                {
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "snippet": item.get("content", ""),
                }
            )
        return results

    # ── SearXNG installation helpers ────────────────────────────

    def _is_searxng_installed(self) -> bool:
        """Check if SearXNG can run on this system.

        Returns True if any of:
          - Docker image ``searxng/searxng`` exists (or a container with ``searxng`` in name)
          - ``pip show searxng`` / ``searx`` succeeds
          - Docker + Colima binaries exist (daemon may be off — start attempt will verify)
        """
        # ── pip package check (fast, no Docker needed) ──
        for pkg in ("searxng", "searx"):
            try:
                r = subprocess.run(
                    [sys.executable, "-m", "pip", "show", pkg],
                    capture_output=True, text=True, timeout=10,
                )
                if r.returncode == 0:
                    return True
            except (subprocess.TimeoutExpired, OSError):
                continue

        # ── Docker / Colima binary check ──
        docker_path = shutil.which("docker")
        if not docker_path:
            return False

        # Try to reach the Docker daemon
        try:
            info = subprocess.run(
                [docker_path, "info"],
                capture_output=True, text=True, timeout=10,
            )
            daemon_alive = info.returncode == 0
        except (subprocess.TimeoutExpired, OSError):
            daemon_alive = False

        if daemon_alive:
            # Check for image or stopped container
            try:
                r = subprocess.run(
                    [docker_path, "image", "inspect", "searxng/searxng"],
                    capture_output=True, text=True, timeout=10,
                )
                if r.returncode == 0:
                    return True
            except (subprocess.TimeoutExpired, OSError):
                pass
            try:
                r = subprocess.run(
                    [docker_path, "ps", "-a", "--filter", "name=searxng", "--format", "{{.Names}}"],
                    capture_output=True, text=True, timeout=10,
                )
                if r.stdout.strip():
                    return True
            except (subprocess.TimeoutExpired, OSError):
                pass
            return False  # daemon alive but no searxng found
        else:
            # Daemon is down — optimistic: docker binary exists, so installed
            return True

    def _ask_start_searxng(self) -> bool:
        """Ask the user if they want to start SearXNG (installed but not running).

        Uses the agent's ask_user mechanism. Falls back to 'no' if
        checkpoint/prompting is unavailable.

        Caches the decision in self._searxng_start_decision so the prompt
        is issued at most once per session (prevents duplicate Checkpoints
        when the LLM retries search_web after all backends fail).
        """
        if self._searxng_start_decision is not None:
            return self._searxng_start_decision
        try:
            result = self._tool_ask_user({
                "question": (
                    "SearXNG is installed but not currently running.\n"
                    "Would you like to start it now?"
                ),
                "type": "confirm",
                "options": ["yes", "no"],
                "default": "no",
                "reason": "SearXNG is installed but not running",
            })
            answer = result.metadata.get("answer", "no").lower().strip()
            self._searxng_start_decision = (answer == "yes")
            return self._searxng_start_decision
        except Exception as e:
            logger.warning("web_search: ask_user failed (%s), skipping SearXNG start", e)
            self._searxng_start_decision = False
            return False

    def _wait_for_searxng(self, base_url: str, timeout: float = 15.0,
                         interval: float = 0.5) -> bool:
        """Poll SearXNG /healthz until it responds or ``timeout`` elapses.

        Replaces a fixed ``time.sleep(5)``: returns as soon as the service is
        ready (often <1s for a warm container) instead of always blocking 5s,
        and waits longer (up to ``timeout``) when the image is still warming
        up. Best-effort — a final return value of False is logged but does not
        block the caller, since the first search may still succeed shortly
        after. Healthz is SearXNG's built-in readiness endpoint.
        """
        health_url = f"{base_url.rstrip('/')}/healthz"
        deadline = time.monotonic() + timeout
        last_err = None
        while time.monotonic() < deadline:
            try:
                resp = httpx.get(health_url, timeout=1.0)
                if resp.status_code < 500:
                    logger.info("web_search: SearXNG ready (healthz=%s)",
                                resp.status_code)
                    return True
            except Exception as e:  # ConnectionError, Timeout, etc.
                last_err = e
            time.sleep(interval)
        logger.warning("web_search: SearXNG not ready after %.1fs (%s)",
                       timeout, last_err)
        return False

    def _start_searxng(self) -> bool:
        """Start SearXNG: start Colima (if needed) then run the SearXNG container.

        Returns True if SearXNG is successfully running after this call.
        """
        docker_path = shutil.which("docker")
        if not docker_path:
            logger.warning("web_search: Docker not found, cannot start SearXNG")
            return False

        # ── Start Colima (if installed and not running) ──
        colima_path = shutil.which("colima")
        if colima_path:
            try:
                status = subprocess.run(
                    ["colima", "status"],
                    capture_output=True, text=True, timeout=10,
                )
                if status.returncode != 0:
                    logger.info("web_search: Starting Colima...")
                    subprocess.run(
                        ["colima", "start"],
                        capture_output=True, text=True, timeout=120,
                        check=True,
                    )
                    logger.info("web_search: Colima started")
            except (subprocess.TimeoutExpired, subprocess.CalledProcessError, OSError) as e:
                logger.warning("web_search: Failed to start Colima: %s", e)
                return False

        # ── Start SearXNG container ──
        try:
            # Check for existing container (stopped or running)
            ps_result = subprocess.run(
                [docker_path, "ps", "-a", "--filter", "name=searxng", "--format", "{{.Names}}"],
                capture_output=True, text=True, timeout=10,
            )
            container_names = ps_result.stdout.strip().split()
            if container_names:
                # Start all existing searxng containers
                all_ok = True
                for name in container_names:
                    try:
                        subprocess.run(
                            [docker_path, "start", name],
                            capture_output=True, text=True, timeout=30,
                            check=True,
                        )
                        logger.info("web_search: Started SearXNG container '%s'", name)
                    except subprocess.CalledProcessError:
                        all_ok = False
                # Healthz poll ONCE after all containers started — not inside the loop,
                # where it would block up to N × timeout serially on a multi-container
                # setup (each warm container still passes the same readiness check).
                if all_ok:
                    base_url = os.environ.get("SEARXNG_BASE_URL", "http://localhost:8080")
                    self._wait_for_searxng(base_url)
                return all_ok

            # No existing container — pull and run
            logger.info("web_search: Pulling SearXNG Docker image...")
            subprocess.run(
                [docker_path, "pull", "searxng/searxng"],
                capture_output=True, text=True, timeout=120,
                check=True,
            )
            base_url = os.environ.get("SEARXNG_BASE_URL", "http://localhost:8080")
            port = base_url.split(":")[-1].rstrip("/")
            subprocess.run(
                [
                    docker_path, "run", "-d",
                    "--name", "asicode-searxng",
                    "-p", f"{port}:8080",
                    "searxng/searxng",
                ],
                capture_output=True, text=True, timeout=30,
                check=True,
            )
            self._wait_for_searxng(base_url)
            logger.info("web_search: SearXNG container started (pulled+run)")
            return True

        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, OSError) as e:
            logger.warning("web_search: Failed to start SearXNG: %s", e)
            return False

    def _ask_install_searxng(self) -> bool:
        """Ask the user if they want to install SearXNG.

        Uses the agent's ask_user mechanism. Falls back to 'no' if
        checkpoint/prompting is unavailable.

        Caches the decision in self._searxng_install_decision so the
        prompt is issued at most once per session (prevents duplicate
        Checkpoints when the LLM retries search_web after all backends
        fail).
        """
        if self._searxng_install_decision is not None:
            return self._searxng_install_decision
        try:
            result = self._tool_ask_user({
                "question": (
                    "SearXNG is needed for web search but is not installed.\n"
                    "Would you like to install and start a local SearXNG instance?"
                ),
                "type": "confirm",
                "options": ["yes", "no"],
                "default": "no",
                "reason": "SearXNG required for web search, but not installed",
            })
            answer = result.metadata.get("answer", "no").lower().strip()
            self._searxng_install_decision = (answer == "yes")
            return self._searxng_install_decision
        except Exception as e:
            logger.warning("web_search: ask_user failed (%s), skipping SearXNG install", e)
            self._searxng_install_decision = False
            return False

    def _install_searxng(self) -> bool:
        """Install and start a local SearXNG instance.

        Uses Docker/Colima under the hood. Returns True if SearXNG is running
        after installation.
        """
        # Delegate to _start_searxng which handles pull+run and Colima start
        return self._start_searxng()

    def _has_docker_or_colima(self) -> bool:
        """Check if Docker or Colima binary is available on this system."""
        return shutil.which("docker") is not None or shutil.which("colima") is not None

    def _setup_and_search_searxng(self, query: str, max_results: int) -> list[dict[str, str]]:
        """Install SearXNG on-the-fly (with user consent) and search.

        Used when SEARXNG_BASE_URL is not set but Docker/Colima is available.
        Flows: ask user → install → set env var → search.
        If user declines or install fails, raises RuntimeError to fall through
        to the next backend (DuckDuckGo).
        """
        if not self._ask_install_searxng():
            raise RuntimeError("User declined SearXNG installation")

        if not self._install_searxng():
            raise RuntimeError("SearXNG installation failed")

        base_url = "http://localhost:8080"
        os.environ["SEARXNG_BASE_URL"] = base_url
        return self._search_searxng(query, max_results, base_url)

    def _handle_searxng_connect_error(self, error: Exception, query: str, max_results: int, searxng_url: str) -> Optional[str]:
        """Handle SearXNG ConnectError.

        Flow:
          1. Not installed + no Docker/Colima → skip question, fallback (too heavy)
          2. Not installed + has Docker/Colima → ask user → install → retry
          3. Installed but down → ask user → start → retry

        Returns None (retry) or error string (fall through to next backend).
        """
        # Normalize URL for logging — searxng_url may be empty in the
        # "auto-setup" flow (set env var not set but Docker available)
        display_url = searxng_url or os.environ.get("SEARXNG_BASE_URL", "http://localhost:8080")

        installed = self._is_searxng_installed()
        if not installed:
            # Docker/Colima not available → skip install question, too heavy
            if not self._has_docker_or_colima():
                logger.warning(
                    "web_search: SearXNG URL configured (%s) but Docker/Colima not installed. "
                    "Skipping install prompt, falling back to next backend.",
                    display_url,
                )
                return f"SearXNG: {error}"

            logger.warning(
                "web_search: SearXNG URL configured (%s) but not installed",
                display_url,
            )
            if self._ask_install_searxng():
                if self._install_searxng():
                    os.environ["SEARXNG_BASE_URL"] = display_url
                    return None  # Signal caller to retry
                return "SearXNG: install attempted but failed"
            return f"SearXNG: {error}"

        # Installed but not running → ask to start
        logger.warning(
            "web_search: SearXNG installed but not running at %s",
            searxng_url,
        )
        if self._ask_start_searxng():
            if self._start_searxng():
                return None  # Signal caller to retry
            return "SearXNG: start attempted but failed"
        return f"SearXNG (not running): {error}"

    def _tool_web_fetch(self, args: dict[str, Any]) -> "ToolResult":
        """Fetch and read content from a URL."""
        url = str(args.get("url", "")).strip()
        max_chars = int(args.get("max_chars", 15000))
        max_chars = max(1000, min(max_chars, 50000))

        if not url:
            return self._make_result(ok=False, content="", error="'url' is required")

        if not url.startswith(("http://", "https://")):
            url = "https://" + url

        #── Reddit: www.reddit.com → old.reddit.com (www blocking bypass) ──
        if "www.reddit.com" in url:
            old_url = url
            url = url.replace("www.reddit.com", "old.reddit.com")
            logger.info("URL rewrite: %s → %s (bypassing Reddit www block)", old_url, url)

        headers = {
            "User-Agent": _BROWSER_UA,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        }

        try:
            import httpx
            with httpx.Client(timeout=30.0, follow_redirects=True, headers=headers) as client:
                resp = client.get(url)
                resp.raise_for_status()
                content_type = resp.headers.get("content-type", "").lower()
                raw = resp.content

            # Determine if it's HTML or plain text
            if "application/json" in content_type:
                import json as _json
                try:
                    formatted = _json.dumps(resp.json(), indent=2, ensure_ascii=False)
                except Exception:
                    formatted = raw.decode("utf-8", errors="replace")
            elif not any(mime in content_type for mime in ["text/html", "application/xhtml", "text/plain"]):
                formatted = raw.decode("utf-8", errors="replace")
            else:
                # Strip HTML tags, extract text
                text = raw.decode("utf-8", errors="replace")

                # Remove scripts and styles
                text = re.sub(r'(?is)<script[^>]*>.*?</script>', '', text)
                text = re.sub(r'(?is)<style[^>]*>.*?</style>', '', text)

                # Remove HTML tags
                text = re.sub(r'<[^>]+>', ' ', text)

                # Decode HTML entities
                text = html_mod.unescape(text)

                # Collapse whitespace
                text = re.sub(r'\s+', ' ', text).strip()

                # Split into lines, rejoin by paragraph
                lines = [_item_.strip() for _item_ in text.split('\n') if _item_.strip()]
                formatted = '\n'.join(lines)

            # Truncate to max_chars
            if len(formatted) > max_chars:
                formatted = formatted[:max_chars] + f"\n\n...[TRUNCATED at {max_chars} chars]..."

            result = f"URL: {url}\nContent-Type: {content_type}\n\n{formatted}"
            return self._make_result(
                ok=True,
                content=result,
                metadata={"url": url, "content_type": content_type, "length": len(formatted)},
            )

        except httpx.TimeoutException:
            return self._make_result(ok=False, content="", error=f"Timeout fetching {url} (30s)")
        except httpx.HTTPStatusError as e:
            return self._make_result(ok=False, content="", error=f"HTTP {e.response.status_code} fetching {url}")
        except Exception as e:
            return self._make_result(ok=False, content="", error=f"Failed to fetch {url}: {type(e).__name__}: {e}")
