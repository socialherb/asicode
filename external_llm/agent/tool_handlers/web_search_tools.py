"""Web search tool handler for ToolRegistry.

Provides web search via multiple backends, tried in fallback order:
    1. SearXNG self-hosted (set SEARXNG_BASE_URL) — private; first ONLY when the
       user explicitly configured it.
    2. Startpage — keyless, unmetered, proxies Google's index. Default primary.
    3. Brave Search API (set BRAVE_API_KEY) — clean JSON API, stable but metered.
    4. Naver (headless browser) — Korean-query last resort.
    5. SearXNG auto-install offer (Docker/Colima present, SEARXNG_BASE_URL unset).
       Last because it raises a user Checkpoint.
    (-) DuckDuckGo — OPT-IN only via ASICODE_DDG_FALLBACK=on; see _should_try_ddg
       for why it left the default chain.

Every HTML-scraping backend routes its empty result set through
``_guard_block_wall``: an engine that answers HTTP 200 with a CAPTCHA/consent page
parses to zero results and would otherwise be reported as a genuine "nothing
matched", hiding an infrastructure failure behind a plausible answer. A wall
raises ``_BlockWallError``, which also trips the session circuit breaker so a
refusing engine is not asked again for the cooldown.

Usage: search_web(query="...", max_results=5, site_filter="...")
"""

from __future__ import annotations

import html as html_mod
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import TimeoutError as FuturesTimeoutError
from html.parser import HTMLParser
from typing import TYPE_CHECKING, Any, ClassVar, Optional

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

# Upper bound (bytes) on a web_fetch response body. web_fetch materialises the
# full body in memory (resp.text / resp.json) before truncating to max_chars, so
# a huge binary URL could balloon RSS / OOM the process. Content-Length is only
# advisory (absent on chunked responses), so this guards only the clear-cut
# oversized case — it never blocks a normal text page (max_chars caps at 50k).
_WEB_FETCH_MAX_BYTES = 20 * 1024 * 1024

# Content-Type prefixes that mark a response as non-textual. Without this guard
# such bodies fell through to the ``else`` branch and were UTF-8 replace-decoded
# into thousands of garbage characters that polluted the LLM context. Rejecting
# them up-front yields a clean error pointing at browser_action (which can at
# least screenshot) instead. Matched as prefixes so vendor subtypes
# (application/vnd.openxmlformats-...) are covered by their root.
_FETCH_BINARY_CONTENT_PREFIXES = (
    "application/pdf",
    "application/zip",
    "application/x-7z-compressed",
    "application/gzip",
    "application/x-bzip2",
    "application/x-tar",
    "application/x-rar-compressed",
    "application/msword",
    "application/vnd.",          # office formats: docx/xlsx/pptx/…
    "application/octet-stream",
    "image/",
    "audio/",
    "video/",
)

# Transient (retryable) HTTP errors shared by all search backends. A single tuple
# keeps SearXNG / DuckDuckGo / Brave on one retry policy instead of each backend
# ad-hoc-listing a subset (SearXNG used to be the only one with any retry).
_TRANSIENT_HTTP_ERRORS = (
    httpx.ConnectError,
    httpx.RemoteProtocolError,
    httpx.ReadTimeout,
    httpx.ReadError,          # parent of ReadTimeout; also covers abrupt stream drops
    httpx.ConnectTimeout,
    httpx.PoolTimeout,
)

# Connect-level failures ("host unreachable / TCP handshake blocked"). These are
# a SUBSET of _TRANSIENT_HTTP_ERRORS that must NOT be retried: unlike a slow read
# or a 429, an immediate re-connect to an unreachable host just re-pays the whole
# connect timeout. They fail fast so the fallback chain (and the session circuit
# breaker) move on — the fix for an IP-blocked DuckDuckGo burning ~15s x2 per search.
_CONNECT_ERRORS = (httpx.ConnectError, httpx.ConnectTimeout)

# Transient HTTP status codes worth retrying: rate-limiting (429) and gateway /
# overload responses (502/503/504). Shared by every backend routed through
# _http_request_with_retry, so a rate-limited DuckDuckGo scrape or an overloaded
# SearXNG gateway backs off instead of failing the whole search on the first hit.
_RETRYABLE_HTTP_STATUSES = frozenset({429, 502, 503, 504})

# httpx timeout for search backends: a SHORT connect budget (a TCP handshake that
# can't complete in a few seconds won't — a blocked/unreachable host should fail
# fast) paired with a patient read budget (some engines are genuinely slow). The
# old flat ``timeout=15.0`` spent the full 15s on every connect attempt, so an
# IP-blocked DuckDuckGo cost ~15s x2 retries ~= 31s per search; connect=4 caps that.
_SEARCH_HTTP_TIMEOUT = httpx.Timeout(connect=4.0, read=15.0, write=15.0, pool=15.0)

# Seconds a backend is skipped (session circuit breaker) after a connect-level
# failure. Long enough that a run of searches does not each re-pay a hard IP
# block's connect timeout; short enough that a transient blip only briefly
# sidelines the backend before it is retried.
_BACKEND_COOLDOWN_SEC = 90.0

# Wall-clock budget for the tier-1 parallel phase. A merge is only as fast as its
# slowest participant, so without a deadline one slow engine sets the latency of
# every search. Measured 2026-07-19: Startpage 1.6-1.7s (stable), SearXNG across
# 13 engines 3.0s-20.1s (it waits on its own slowest engine). 8s keeps SearXNG's
# common case while capping the outlier — whatever has arrived by then is merged
# and the rest is abandoned, so a 20s engine costs 8s, not 20.
_TIER1_DEADLINE_SEC = 8.0

# SearXNG's value is its per-engine request recipes, and those rot: 7 general
# engines took 57 commits in 12 months, almost all of them CAPTCHA-detection,
# changed-request-param and site-redesign fixes. But `docker pull` runs only when
# _start_searxng has to CREATE a container, so an existing install keeps starting
# the image it was first pulled with and never receives any of that work.
#
# Not hypothetical — measured on this machine 2026-07-20. A 2026-06-08 image had
# SearXNG's naver engine returning 0 results; the upstream fix ("[fix] naver:
# update HTML parsing for redesigned Naver search") landed 2026-06-28 and pulling
# 2026-07-18 took the same query to 15 results. Six weeks stale cost an entire
# engine, silently.
#
# 21 days is chosen against that churn rate: long enough that a healthy install
# is not nagged, short enough that a broken engine is not carried for six weeks.
_SEARXNG_IMAGE_STALE_DAYS = 21.0

# Minimum gap between notices. The staleness is real but not urgent, and a notice
# repeated on every session is a notice that gets ignored.
_SEARXNG_STALE_NOTICE_INTERVAL_DAYS = 7.0

# Engines SearXNG is asked for by name, instead of letting `categories=general`
# pick. Measured 2026-07-20 on a bot-flagged IP, per engine, two queries each.
#
# The default general category behaves badly here: brave (suspended) and
# duckduckgo (CAPTCHA) drop out, leaving **naver as the only engine answering
# every query** — including English technical ones. Measured against this list:
#
#   query                    categories=general      curated
#   climate policy 2026      0.31s  15 hits/14 dom   1.96s  130 hits/89 dom
#   전세 사기 대처법           0.37s  15 hits/10 dom   1.98s  102 hits/76 dom
#   python asyncio timeout   0.26s  15 hits/ 9 dom   2.01s  157 hits/59 dom
#
# So this trades ~1.7s for 6-9x the distinct domains, and for not answering an
# English query out of a single Korean index. 2s sits well inside the 8s tier-1
# deadline, and tier 1 runs SearXNG in PARALLEL with Startpage anyway, so the
# added latency is absorbed rather than added to the search.
#
# Selection is still LATENCY-AWARE — SearXNG waits for its slowest engine, and
# the measurements make the choice easy because **the slow engines are the
# failing ones**: yacy took 5.02s to time out, gabanza 4.02s, 360search 3.90s
# for 7 results, while the working ones answer fast (bing 0.23s/10, naver
# 0.52s/15, baidu 0.96s/10, yandex 1.38s/14, yep 1.86s/20). Excluding the slow
# failures costs no coverage.
#
# Engines that are dead HERE but healthy on a clean IP are deliberately KEPT
# (google 0.27s, duckduckgo 0.31s, brave 0.01s, bing 0.23s): they fail fast, so
# they cost ~nothing where they are blocked and carry the search where they are
# not. This list must work for users who are NOT behind a flagged IP, and these
# measurements are from one IP at one moment — engine health is volatile (mojeek
# went 10/10 → 0/0 within an hour of testing), which is also why the list is
# overridable rather than baked in.
_SEARXNG_DEFAULT_ENGINES = (
    "google,duckduckgo,brave,bing,naver,yandex,baidu,qwant,yep,zapmeta,mwmbl"
)


def _retry_after_seconds(resp: httpx.Response, default: float) -> float:
    """Parse a ``Retry-After`` header (delta-seconds or HTTP-date) → wait seconds.

    Returns ``default`` when the header is absent or unparseable. Capped at 30s
    so a hostile or misconfigured server cannot stall search for minutes; floored
    at 0 so ``Retry-After: 0`` never sleeps a negative amount.
    """
    raw = resp.headers.get("retry-after")
    if not raw:
        return default
    raw = raw.strip()
    try:  # delta-seconds form (RFC 7231 §7.1.3)
        return min(max(float(raw), 0.0), 30.0)
    except ValueError:
        pass
    try:  # HTTP-date form
        import email.utils as _eut
        from datetime import datetime, timezone

        dt = _eut.parsedate_to_datetime(raw)
        if dt is not None:
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return min(max((dt - datetime.now(timezone.utc)).total_seconds(), 0.0), 30.0)
    except Exception:
        pass
    return default


# Charset detection for web_fetch: HTML5 prescan of the first 1KB. httpx's
# ``Response.encoding`` returns ONLY the HTTP ``Content-Type`` header charset
# (falling back to UTF-8) — it never inspects the body — so pages that declare
# their encoding solely via ``<meta charset="euc-kr">`` (common for Korean /
# East-Asian legacy sites) were UTF-8 replace-decoded into mojibake. This sniffs
# the BOM and ``<meta>`` charset declaration BEFORE decoding, which is the only
# point at which it is possible (a full HTML parser cannot run on bytes whose
# encoding is unknown). A bounded scan of the document head is exactly the
# HTML5 "prescan" algorithm; a structured tokeniser is not applicable here, so
# the regex is the correct tool (this is text-format sniffing, not code parsing).
_META_CHARSET_RE = re.compile(
    r"""<meta\b[^>]*?charset\s*=\s*['"]?\s*([A-Za-z0-9_\-:.]+)""",
    re.IGNORECASE,
)


def _sniff_html_encoding(body_bytes: bytes) -> Optional[str]:
    """Best-effort HTML encoding sniff from BOM / ``<meta charset>`` (HTML5 prescan).

    Returns ``None`` when no declaration is found so the caller falls back to its
    own default. Operates on the first 1024 bytes (the HTML5 prescan window)
    decoded as ASCII-with-ignore: we are looking for ASCII charset tokens inside
    ASCII-structured ``<meta>`` tags, so non-ASCII body content cannot fool it.
    """
    if not body_bytes:
        return None
    head = body_bytes[:1024]
    # BOM checks (UTF-8 / UTF-16) take precedence per the encoding sniffing spec.
    if head.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig"
    if head.startswith((b"\xff\xfe", b"\xfe\xff")):
        return "utf-16"
    head_ascii = head.decode("ascii", errors="ignore")
    m = _META_CHARSET_RE.search(head_ascii)
    return m.group(1) if m else None


class _ResultParserBase(HTMLParser):
    """Attribute helpers shared by every engine's result parser.

    SSOT for the two operations each parser needs from a tag's attribute list.
    Kept in one place so a fix (e.g. class matching that must not treat
    ``result-title-extra`` as ``result-title``) lands for every engine at once,
    rather than being fixed in one twin and left broken in the other.
    """

    @staticmethod
    def _get_attr(attrs: list[tuple[str, str | None]], name: str, default: str = "") -> str:
        for n, v in attrs:
            if n == name:
                return v or default
        return default

    @staticmethod
    def _has_class(attrs: list[tuple[str, str | None]], cls: str) -> bool:
        """True when ``cls`` is one of the tag's whitespace-separated classes.

        Token-wise on purpose: a substring test would match the rotating
        ``css-<hash>`` companions and neighbouring names that merely share a
        prefix.
        """
        class_val = _ResultParserBase._get_attr(attrs, "class")
        return cls in class_val.split() if class_val else False


class _DDGResultParser(_ResultParserBase):
    """Structured HTML parser that extracts search results from DuckDuckGo search page.

    Uses html.parser.HTMLParser (stdlib) instead of regex, making it resilient
    to HTML structure changes inside result blocks, nested tags, and attribute
    ordering differences. Approach mirrors Crush's tokenizer-based parsing.

    A DDG result block is a title link (``result__a``) optionally followed by a
    snippet link (``result__snippet``). Results are emitted via ``_flush()``
    which is triggered on the snippet's closing tag *and* — crucially — when the
    next result starts or at EOF. The latter is the fix for the old bug where a
    result with a title but no snippet (DDG sometimes omits it) was silently
    dropped because emission happened only inside the snippet-endtag branch.
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
        # True once _current has been appended to self.results, so flushing it
        # again (next result start / EOF) is a no-op rather than a duplicate.
        self._emitted = False

    # ── helpers ──

    @staticmethod
    def _decode_ddg_url(href: str) -> str:
        """Decode DuckDuckGo's /l/?uddg= redirect URLs."""
        if "uddg=" in href:
            qs = urllib.parse.urlparse(href).query
            decoded = urllib.parse.parse_qs(qs).get("uddg", [None])[0]
            return urllib.parse.unquote(decoded) if decoded else href
        return href

    def _flush(self) -> None:
        """Emit the pending result if it has a title and room remains.

        Called from three sites: the snippet closing tag, the start of the next
        result (catches results whose snippet was missing), and ``close()``
        (catches the trailing result). The ``_emitted`` flag plus the
        ``max_results`` guard make repeated flushes safe and bounded.
        """
        if self._current is None or self._emitted:
            return
        if len(self.results) >= self.max_results:
            return
        if self._current.get("title"):
            self.results.append(self._current)
        self._emitted = True

    # ── parser callbacks ──

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if len(self.results) >= self.max_results:
            return

        if tag == "a" and self._has_class(attrs, "result__a"):
            # A new result block begins. Flush any pending result that collected
            # a title but never reached a snippet closing tag (snippet omitted).
            self._flush()
            self._in_result_a = True
            self._capturing = True
            self._text_parts = []
            href = self._decode_ddg_url(self._get_attr(attrs, "href"))
            self._current = {"url": href, "title": "", "snippet": ""}
            self._emitted = False

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

        elif tag == "a" and self._in_snippet:
            if self._current is not None:
                self._current["snippet"] = html_mod.unescape("".join(self._text_parts)).strip()
            self._in_snippet = False
            self._capturing = False
            self._flush()

    def handle_data(self, data: str) -> None:
        if self._capturing:
            self._text_parts.append(data)

    def close(self) -> None:
        super().close()
        # Flush a trailing result that had a title but no snippet closing tag.
        if len(self.results) < self.max_results:
            self._flush()


# Substrings marking a bot-detection / consent / rate-limit interstitial served
# with HTTP 200 and NO result links. Chain-common on purpose: every HTML-scraping
# engine has this failure mode and each one used to need its own detector.
#
# Empirically observed (2026-07-19, one IP, all engines probed directly):
#   DuckDuckGo  → HTTP 202 + "an anomaly in" / redirect to /anomaly/...
#   Mojeek      → HTTP 200 + "Verification required. Please complete the challenge"
#   Marginalia  → HTTP 200 + "Wait A Moment ... seeing a lot of fairly aggressive
#                 bot activity"
#   Google      → HTTP 200 + "detected unusual traffic" consent/interstitial wall
#   Cloudflare  → HTTP 200 + "Just a moment" / "Checking your browser"
#
# The danger this guards is NOT the wall itself but its shape: HTTP 200 + zero
# results is indistinguishable from a genuine miss, so without this the fallback
# chain reports "No results found" and the caller reads infrastructure failure as
# an honest empty answer. Same trap as the rg_fallback counter in CLAUDE.md —
# absence of evidence silently read as evidence of health.
_BLOCK_WALL_MARKERS = (
    # rate-limit / anomaly
    "an anomaly in",
    "unusual traffic",
    "rate limit",
    "too many requests",
    "duckduckgo.com/anomaly",
    # explicit bot challenge. Phrases are taken VERBATIM from the live pages —
    # a paraphrase does not match. DDG's current CAPTCHA, for instance, contains
    # neither the word "captcha" nor "complete the challenge"; it says "complete
    # the FOLLOWING challenge" and "confirm this search was made by a human".
    "verification required",
    "complete the challenge",
    "complete the following challenge",
    "made by a human",
    "bots use duckduckgo",
    "verify you are human",
    "are you a robot",
    "captcha",
    # throttle interstitials
    "wait a moment",
    "bot activity",
    "just a moment",
    "checking your browser",
    # consent / JS walls
    "before you continue",
    "enable javascript and cookies",
)

# Upper bound on the body prefix scanned for wall markers. Wall pages are small
# (Mojeek 5KB, Marginalia 37KB) and put the message near the top, so a bounded
# scan finds every observed wall while keeping the check O(1) in page size.
_BLOCK_WALL_SCAN_CHARS = 20_000


def _normalize_result_url(url: str) -> str:
    """Canonical key for deciding that two backends returned the SAME page.

    Deliberately conservative — over-normalising merges genuinely different
    pages. The query string is KEPT (``?id=123``, ``?v=...`` and ``?q=`` select
    real content on many sites); only the parts that never change what is served
    are dropped: the fragment, a default port, a ``www.`` prefix, host case, and
    a trailing slash. Falls back to the raw string if the URL will not parse, so
    an odd input degrades to "no dedup" rather than collapsing unrelated results.
    """
    if not url:
        return ""
    try:
        p = urllib.parse.urlsplit(url.strip())
    except ValueError:
        return url.strip()
    if not p.netloc:
        return url.strip()
    host = (p.hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    if p.port and not ((p.scheme == "http" and p.port == 80) or (p.scheme == "https" and p.port == 443)):
        host = f"{host}:{p.port}"
    path = p.path.rstrip("/")
    return f"{host}{path}?{p.query}" if p.query else f"{host}{path}"


def _merge_search_results(
    per_backend: list[tuple[str, list[dict[str, str]]]],
    max_results: int,
) -> list[dict[str, str]]:
    """Merge several backends' result lists into one deduplicated ranking.

    Ranking uses cross-backend agreement as the primary signal: a URL returned
    by three independent indexes is a stronger answer than one seen only by the
    engine that happened to run first. This is the ranking signal a single
    backend structurally cannot provide, and it is the whole point of querying
    more than one. Ties fall back to the best (lowest) position the URL achieved
    in any single backend, so within one agreement level each engine's own
    ordering is preserved.

    Field selection prefers the most informative variant rather than the first
    seen: backends differ in how much of a title they truncate and whether they
    return a snippet at all.

    ``per_backend`` order is used only as the final tiebreaker, so callers may
    pass backends in priority order without that dominating agreement.
    """
    merged: dict[str, dict[str, Any]] = {}
    for backend_index, (name, results) in enumerate(per_backend):
        for position, r in enumerate(results):
            url = (r.get("url") or "").strip()
            if not url:
                continue
            key = _normalize_result_url(url)
            entry = merged.get(key)
            if entry is None:
                merged[key] = {
                    "url": url,
                    "title": (r.get("title") or "").strip(),
                    "snippet": (r.get("snippet") or "").strip(),
                    "sources": [name],
                    "best_position": position,
                    "first_backend": backend_index,
                }
                continue
            if name not in entry["sources"]:
                entry["sources"].append(name)
            entry["best_position"] = min(entry["best_position"], position)
            entry["first_backend"] = min(entry["first_backend"], backend_index)
            # Keep the fuller title/snippet — engines truncate differently.
            title = (r.get("title") or "").strip()
            if len(title) > len(entry["title"]):
                entry["title"] = title
            snippet = (r.get("snippet") or "").strip()
            if len(snippet) > len(entry["snippet"]):
                entry["snippet"] = snippet

    # Drop unusable entries BEFORE truncating — slicing first would let a
    # titleless result consume one of the caller's max_results slots and return
    # fewer usable results than were available.
    ranked = sorted(
        (e for e in merged.values() if e["title"]),
        key=lambda e: (-len(e["sources"]), e["best_position"], e["first_backend"]),
    )
    return [
        {
            "url": e["url"],
            "title": e["title"],
            "snippet": e["snippet"],
            "sources": ",".join(e["sources"]),
        }
        for e in ranked[:max_results]
    ]


class _BlockWallError(RuntimeError):
    """An engine answered with a bot-detection wall instead of results.

    A distinct type (still a ``RuntimeError``, so existing handlers and callers
    are unaffected) so the fallback chain can tell "this engine is refusing to
    serve us" apart from an ordinary backend error and sideline it. Retrying a
    walled engine does not just waste a request — each attempt feeds the same
    bot-detection that escalates to a hard IP block.
    """


def _body_is_block_wall(body: str) -> bool:
    """Heuristic: does this HTML body look like a bot-detection / consent wall?

    Callers MUST only consult this for an EMPTY result set — see
    ``_guard_block_wall``. Several markers ("rate limit", "captcha") are ordinary
    words that legitimately appear in real search results ABOUT those topics, so
    matching them on a populated page would be a false positive. Gating on
    "zero results parsed" is what makes the heuristic safe.
    """
    if not body:
        return False
    low = body[:_BLOCK_WALL_SCAN_CHARS].lower()
    return any(m in low for m in _BLOCK_WALL_MARKERS)


# Control characters that must never reach a result field. Startpage interleaves
# server-rendered markup with hydration padding that occasionally leaves stray NUL
# / C0 bytes inside title text (observed live: '전세사기 유\x00형별 사례').
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


class _StartpageResultParser(_ResultParserBase):
    """Extracts search results from a Startpage (``/sp/search``) results page.

    Startpage proxies Google's index, so this yields general-web quality results
    without Google's own scraper detection. Structure of one result:

        <a class="result-title result-link css-<hash>" href="https://real.url/">
            <style data-emotion="...">.css-xxx{...}</style>
            <h2 class="wgl-title css-<hash>">The Title</h2>
        </a>
        <p class="description css-<hash>">The snippet…</p>

    Two structural facts drive this parser:

    * **Hook on the semantic class, not the hash.** Startpage ships CSS-in-JS
      (emotion) classes like ``css-1bggj8v`` that rotate on every frontend
      deploy; ``result-title`` / ``result-link`` / ``description`` are stable
      semantic names. Same lesson as the Naver backend's class-hash-free
      extraction — key off what the markup MEANS, not what the build emitted.
    * **``<style>`` blocks live INSIDE the result anchor.** ``HTMLParser`` treats
      style/script content as CDATA and still fires ``handle_data`` for it, so a
      naive "capture everything between <a> and </a>" pulls raw CSS into the
      title. ``_in_raw_text`` suppresses capture for exactly that reason.

    ``href`` is the real destination URL (no redirect wrapper to decode, unlike
    DuckDuckGo's ``/l/?uddg=``). Emission uses the same three-site flush as
    ``_DDGResultParser`` so a result whose snippet is missing is not dropped.
    """

    def __init__(self, max_results: int = 10):
        super().__init__()
        self.max_results = max_results
        self.results: list[dict[str, str]] = []

        self._current: dict[str, str] | None = None
        self._text_parts: list[str] = []
        self._capturing = False
        self._in_title = False
        self._in_snippet = False
        self._in_raw_text = False   # inside <style>/<script>: never capture
        self._emitted = False

    # ── helpers ──

    @staticmethod
    def _clean(text: str) -> str:
        """Unescape entities, drop control chars, collapse whitespace."""
        out = html_mod.unescape(text)
        out = _CONTROL_CHARS_RE.sub("", out)
        return " ".join(out.split())

    def _flush(self) -> None:
        if self._current is None or self._emitted:
            return
        if len(self.results) >= self.max_results:
            return
        # A result needs both a destination and a label to be usable.
        if self._current.get("title") and self._current.get("url"):
            self.results.append(self._current)
        self._emitted = True

    # ── parser callbacks ──

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in ("style", "script"):
            self._in_raw_text = True
            return
        if len(self.results) >= self.max_results:
            return

        if tag == "a" and self._has_class(attrs, "result-title"):
            # New result block: flush any pending one that never got a snippet.
            self._flush()
            href = self._get_attr(attrs, "href")
            self._current = {"url": href, "title": "", "snippet": ""}
            self._emitted = False
            self._in_title = True
            self._capturing = True
            self._text_parts = []

        elif tag == "p" and self._has_class(attrs, "description") and self._current is not None:
            self._in_snippet = True
            self._capturing = True
            self._text_parts = []

    def handle_endtag(self, tag: str) -> None:
        if tag in ("style", "script"):
            self._in_raw_text = False
            return
        if len(self.results) >= self.max_results:
            return

        if tag == "a" and self._in_title and self._current is not None:
            self._current["title"] = self._clean("".join(self._text_parts))
            self._in_title = False
            self._capturing = False

        elif tag == "p" and self._in_snippet:
            if self._current is not None:
                self._current["snippet"] = self._clean("".join(self._text_parts))
            self._in_snippet = False
            self._capturing = False
            self._flush()

    def handle_data(self, data: str) -> None:
        # The max_results check also stops a capture that was in flight when the
        # limit was reached from accumulating the rest of the page (Startpage
        # result pages run ~250KB) — its end tag returns early, so nothing else
        # would clear _capturing.
        if self._capturing and not self._in_raw_text and len(self.results) < self.max_results:
            self._text_parts.append(data)

    def close(self) -> None:
        super().close()
        if len(self.results) < self.max_results:
            self._flush()


class _HTMLTextExtractor(HTMLParser):
    """Block-aware HTML→text converter (stdlib ``html.parser``).

    Structural replacement for the old regex strip in ``_tool_web_fetch``. The
    previous approach ran ``re.sub(r'\\s+', ' ', text)`` — which collapses every
    newline to a space — and *then* ``text.split('\\n')``, so the split was dead
    code and the output was always a single run-on line.

    This parser inserts a newline at block-level boundaries (``<p>``, ``<div>``,
    ``<li>``, headings, ``<br>``, table rows, …) and skips ``<script>/<style>``
    entirely, preserving paragraph structure for LLM readability. ``convert_charrefs``
    is left at its default so character references in text are already decoded by
    the time ``handle_data`` sees them.
    """

    # Tags that introduce a visual block boundary → emit a newline so paragraphs
    # and list items survive the text extraction instead of being glued together.
    _BLOCK_TAGS = frozenset(
        {
            "address",
            "article",
            "aside",
            "blockquote",
            "br",
            "caption",
            "dd",
            "div",
            "dl",
            "dt",
            "fieldset",
            "figcaption",
            "figure",
            "footer",
            "form",
            "h1",
            "h2",
            "h3",
            "h4",
            "h5",
            "h6",
            "header",
            "hr",
            "li",
            "main",
            "nav",
            "ol",
            "p",
            "pre",
            "section",
            "table",
            "tbody",
            "td",
            "tfoot",
            "th",
            "thead",
            "tr",
            "ul",
            "option",
            "optgroup",
        }
    )
    # Tags whose textual content is never useful for reading (dropped wholesale).
    _SKIP_TAGS = frozenset({"script", "style", "noscript", "template"})

    def __init__(self, base_url: str = "") -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._skip_depth = 0
        # Hyperlink preservation. When ``base_url`` is given, each ``<a href>`` is
        # resolved to an absolute URL and emitted inline as ``text (url)`` so a
        # research agent can follow links it discovers in a fetched page. Without a
        # ``base_url`` relative links cannot be resolved, so the historical
        # text-only behaviour is preserved for callers that pass nothing.
        self._base = base_url or ""
        self._pending_href: str | None = None
        self._pending_anchor_text: list[str] = []
        self._seen_links: set[tuple[str, str]] = set()

    def _resolve_href(self, href: str | None) -> str | None:
        """Resolve an ``<a href>`` to a usable absolute URL, or ``None``.

        Returns ``None`` for empty/non-http(s) schemes (``mailto:``, ``tel:``,
        ``javascript:``), in-page ``#fragment`` anchors, and relative links when no
        ``base_url`` was supplied — none of which a research agent can follow.
        """
        if not href:
            return None
        href = href.strip()
        if not href or href.startswith("#"):
            return None
        resolved = urllib.parse.urljoin(self._base, href)
        resolved = urllib.parse.urldefrag(resolved).url
        if not (resolved.startswith("http://") or resolved.startswith("https://")):
            return None
        return resolved

    def _flush_anchor(self) -> None:
        """Emit the pending hyperlink as a trailing `` (url)`` and clear the slot.

        The anchor's visible text was already streamed into ``_parts`` by
        ``handle_data``; this only appends the resolved URL once ``</a>`` is seen,
        so the output reads ``visible text (https://target)``.
        """
        href = self._pending_href
        text = "".join(self._pending_anchor_text)
        self._pending_href = None
        self._pending_anchor_text = []
        if href is None:
            return
        link_text = re.sub(r"\s+", " ", text).strip()
        if not link_text or link_text == href:
            # Empty anchor, or the text is already the URL — appending it again
            # would only add noise.
            return
        key = (link_text, href)
        if key in self._seen_links:
            return
        self._seen_links.add(key)
        self._parts.append(f" ({href})")

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self._SKIP_TAGS:
            self._skip_depth += 1
            return
        if self._skip_depth:
            return
        if tag == "a":
            # Nested <a> is invalid HTML; flush the open link before starting fresh.
            if self._pending_href is not None:
                self._flush_anchor()
            self._pending_href = self._resolve_href(dict(attrs).get("href"))
            if self._pending_href is not None:
                self._pending_anchor_text = []
            return
        if tag in self._BLOCK_TAGS:
            self._parts.append("\n")

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        # XHTML self-closing forms (<br/>, <hr/>) — emit a single boundary newline.
        if tag in self._SKIP_TAGS:
            return
        if self._skip_depth:
            return
        if tag == "a":
            # <a href="..."/> carries no text node — nothing useful to emit.
            return
        if tag in self._BLOCK_TAGS:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP_TAGS:
            if self._skip_depth:
                self._skip_depth -= 1
            return
        if self._skip_depth:
            return
        if tag == "a":
            self._flush_anchor()
            return
        if tag in self._BLOCK_TAGS:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        self._parts.append(data)
        if self._pending_href is not None:
            self._pending_anchor_text.append(data)

    def get_text(self) -> str:
        """Return extracted text with block boundaries as newlines.

        Collapses intra-line whitespace (spaces/tabs) but preserves the newlines
        injected at block boundaries, then drops empty lines.
        """
        lines: list[str] = []
        for line in "".join(self._parts).split("\n"):
            line = re.sub(r"[ \t\r\f\v]+", " ", line).strip()
            if line:
                lines.append(line)
        return "\n".join(lines)


def _format_unresponsive_engine(entry: Any) -> str:
    """Render one SearXNG ``unresponsive_engines`` entry as ``name: reason``.

    Entries are ``[engine_name, reason, ...]`` lists (JSON has no tuples); newer
    SearXNG appends extra fields (e.g. a suspended flag), so only the first two
    are used. Any unexpected shape falls back to ``str(entry)`` so a schema
    change can never mask the underlying "all engines down" signal.
    """
    if isinstance(entry, (list, tuple)):
        name = str(entry[0]) if len(entry) >= 1 else "?"
        reason = str(entry[1]) if len(entry) >= 2 else "unresponsive"
        return f"{name}: {reason}"
    return str(entry)


def _searx_host_port(base_url: str) -> str:
    """Derive the host-side port for the SearXNG ``docker run -p`` mapping.

    Centralised so the port-derivation rule is testable in isolation. The former
    inline ``base_url.split(":")[-1]`` produced ``//searx.example.com`` for a
    portless URL, yielding an invalid ``-p`` value that silently broke
    ``docker run``. ``urlparse`` falls back to 8080 when no port is present.
    """
    try:
        return str(urllib.parse.urlparse(base_url).port or 8080)
    except ValueError:
        return "8080"


# Any Hangul syllable — gates the (heavier, browser-backed) Naver fallback so it
# fires only where its Korean-local coverage pays for the browser cost.
_HANGUL_RE = re.compile(r"[가-힣]")

# Class-hash-FREE extractor for Naver's web-vertical SERP, run in-page via a
# headless browser. Naver renders results client-side and hooks them with
# CSS-in-JS class names (``ukOeokM2JecBbNO3``…) that rotate on EVERY frontend
# deploy — so a class-based parser would break constantly. Instead we exploit the
# stable STRUCTURAL invariant: every result card renders several <a> to the SAME
# target href — a breadcrumb display-URL line, a title, and a snippet. We
# group anchors by href, drop Naver-internal links and accessibility noise
# ("새 창 열림", rating widgets), then split each group into {title, snippet}. Only
# a Naver DOM *structural* redesign (rare) breaks this, not a routine restyle.
_NAVER_EXTRACT_JS = r"""
() => {
  const isNaver = h => /naver\.com|nid\.naver|help\.naver|ads\.naver/.test(h);
  const NOISE = t =>
    !t
    || /^새 창 열림$/.test(t)
    || /^평점(\s|$)/.test(t)
    || /^\d+(\.\d+)?\/5(\s+\d+\s*참여)?$/.test(t)
    || /^\d+\s*참여$/.test(t)
    || /^\d{1,2}:\d{2}(:\d{2})?$/.test(t);
  const norm = s => (s || '').replace(/새 창 열림/g, '').replace(/\s+/g, ' ').trim();
  const SEP = String.fromCharCode(8250);    // breadcrumb separator U+203A (ASCII-safe source)
  const groups = new Map();                 // href(no #frag) -> ordered unique texts
  for (const a of document.querySelectorAll('a[href^="http"]')) {
    if (isNaver(a.href)) continue;
    const key = a.href.split('#')[0];       // merge one card's in-page fragment anchors
    const t = norm(a.innerText);
    if (NOISE(t)) continue;
    if (!groups.has(key)) groups.set(key, []);
    const arr = groups.get(key);
    if (!arr.includes(t)) arr.push(t);      // dedupe repeated anchor texts
  }
  const results = [];
  for (const [href, texts] of groups) {
    const host = new URL(href).host.replace(/^www\./, '');
    const crumb = texts.find(x => x.includes(SEP) || x.replace(/^www\./, '').startsWith(host));
    const rest = texts.filter(x => x !== crumb);
    if (!rest.length) continue;
    const snippet = rest.reduce((a, b) => (b.length > a.length ? b : a), '');
    const title = rest.filter(x => x !== snippet).sort((a, b) => a.length - b.length)[0] || rest[0];
    if (title && title.length >= 6)
      results.push({ title: title, url: href, snippet: snippet === title ? '' : snippet.slice(0, 300) });
  }
  return results.slice(0, 30);
}
"""


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

    # ── SearXNG setup serialization ──
    # Only ONE thread may drive the ask→install/start sequence at a time.
    #
    # The decision caches above are written only AFTER _tool_ask_user returns,
    # and that call blocks for as long as the Checkpoint is on screen (up to the
    # 60s auto-default). search_web runs on the shared tool-executor thread pool,
    # so when the LLM dispatches two searches in one batch both threads read the
    # cache as None inside that window and BOTH raise a Checkpoint — the user
    # sees the same question twice and one of them times out to the default.
    # Observed live 2026-07-19 (two "start SearXNG?" prompts; one answered "yes",
    # the other auto-applied "no").
    #
    # Guarding the ask alone is not enough: the loser of the race would then read
    # the cached "yes" and run _start_searxng() a second time, racing a concurrent
    # `docker run --name asicode-searxng` against itself. The lock therefore also
    # covers _start_searxng, which makes the second caller take the idempotent
    # "container already exists → docker start" path instead.
    #
    # RLock, not Lock: ask→start are sequential today, but a future refactor that
    # nests them must not self-deadlock.
    _searxng_setup_lock: ClassVar = threading.RLock()

    # ── Session circuit breaker ──
    # backend-name → monotonic deadline until which the backend is skipped after a
    # connect-level failure OR a bot-detection wall (_BlockWallError) — in both
    # cases re-asking is pure cost, and for a wall it actively deepens the block.
    # CLASS-level on purpose: it is shared process-wide state
    # (the whole point — avoid re-paying a dead backend's connect timeout on every
    # search), not per-request context. Guarded by a lock because searches run
    # concurrently on the shared tool-executor pool.
    _backend_cooldown: ClassVar[dict[str, float]] = {}
    _backend_cooldown_lock: ClassVar = threading.Lock()

    def _backend_in_cooldown(self, name: str) -> bool:
        """True if ``name`` is currently sidelined by the circuit breaker.

        Lazily evicts an expired entry so a recovered backend is retried once the
        cooldown lapses.
        """
        with WebSearchToolsMixin._backend_cooldown_lock:
            deadline = WebSearchToolsMixin._backend_cooldown.get(name)
            if deadline is None:
                return False
            if time.monotonic() >= deadline:
                del WebSearchToolsMixin._backend_cooldown[name]
                return False
            return True

    def _trip_backend_cooldown(self, name: str) -> None:
        """Sideline ``name`` for ``_BACKEND_COOLDOWN_SEC`` (connect failure or wall)."""
        with WebSearchToolsMixin._backend_cooldown_lock:
            WebSearchToolsMixin._backend_cooldown[name] = time.monotonic() + _BACKEND_COOLDOWN_SEC

    @staticmethod
    def _guard_block_wall(
        engine: str,
        body: str,
        results: list[dict[str, str]],
        status: Optional[int] = None,
    ) -> None:
        """Raise when an EMPTY result set is really a bot-detection wall.

        Every HTML-scraping backend shares one failure mode: the engine answers
        HTTP 200 with a challenge/consent page instead of results, which parses to
        ``[]`` and is then indistinguishable from an honest "nothing matched".
        Silently returning that empty list makes an infrastructure failure look
        like a genuine miss — so it is converted into an explicit error that both
        drives the fallback chain and names the real cause in ``last_error``.

        Deliberately a no-op when ``results`` is non-empty: several markers are
        ordinary words ("rate limit", "captcha") that legitimately occur in real
        results about those very topics. Gating on "zero results parsed" is what
        keeps the heuristic from firing on a healthy page.

        Two independent signals, because text matching alone proved too brittle to
        rely on. DDG's live CAPTCHA was caught by a single incidental marker (a
        ``duckduckgo.com/anomaly`` URL reference) — it contains neither "captcha"
        nor any phrase the original list matched — so one wording change would
        have silently reopened the trap:

        * ``status``: a 2xx that is NOT 200 means the engine acknowledged the
          request without running the search (DDG answers its challenge page with
          **HTTP 202**, and 200 with real results). This is structural, so it
          survives any rewording of the page.
        * ``body``: the marker list, for engines that wall with a plain 200.
        """
        if results:
            return
        if status is not None and 200 < status < 300:
            raise _BlockWallError(
                f"{engine} returned HTTP {status} with no results "
                f"(request acknowledged but not served — bot challenge)"
            )
        if _body_is_block_wall(body):
            raise _BlockWallError(f"{engine} served a bot-detection/block wall (no results parsed)")

    # ── SearXNG image freshness ──────────────────────────────────────

    # Checked at most once per process: the notice is informational, and the
    # `docker image inspect` behind it costs ~100ms that no search should re-pay.
    _searxng_staleness_checked: ClassVar[bool] = False

    def _searxng_image_age_days(self) -> Optional[float]:
        """Age of the LOCAL ``searxng/searxng`` image in days, or None.

        Deliberately local-only — no registry round trip. "Is a newer image
        published?" needs the network and belongs nowhere near a search; "has
        this install been carrying the same image for weeks?" answers the
        question that actually matters and is a fast local lookup.
        """
        docker_path = shutil.which("docker")
        if not docker_path:
            return None
        try:
            proc = subprocess.run(
                [docker_path, "image", "inspect", "searxng/searxng", "--format", "{{.Created}}"],
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (subprocess.TimeoutExpired, OSError):
            return None
        if proc.returncode != 0:
            return None  # image not present locally
        raw = (proc.stdout or "").strip().splitlines()
        if not raw:
            return None
        try:
            from datetime import datetime, timezone

            # Docker emits RFC3339 with nanoseconds; datetime handles at most
            # microseconds, so trim the fractional part to 6 digits.
            stamp = re.sub(r"(\.\d{6})\d+", r"\1", raw[0].replace("Z", "+00:00"))
            created = datetime.fromisoformat(stamp)
            if created.tzinfo is None:
                created = created.replace(tzinfo=timezone.utc)
            return max((datetime.now(timezone.utc) - created).total_seconds() / 86400.0, 0.0)
        except (ValueError, TypeError) as e:
            logger.debug("web_search: could not parse image timestamp %r (%s)", raw[0], e)
            return None

    def _searxng_stale_state_path(self) -> str:
        return os.path.join(str(getattr(self, "repo_root", ".")), ".asicode", "searxng_image_check.json")

    def _stale_searxng_image_notice(self) -> Optional[str]:
        """Actionable one-liner when the local SearXNG image is too old, else None.

        Rate-limited on disk so it does not reappear every session. Any failure
        along the way returns None: a freshness *hint* must never be able to
        break a search that is otherwise working, so the whole body is wrapped —
        guarding only the individual subprocess/IO calls would still let an
        unforeseen error propagate out of a purely advisory code path.
        """
        try:
            return self._stale_searxng_image_notice_inner()
        except Exception as e:
            logger.debug("web_search: staleness check failed (%s); ignoring", e)
            return None

    def _stale_searxng_image_notice_inner(self) -> Optional[str]:
        """Body of :meth:`_stale_searxng_image_notice` (errors handled there)."""
        if WebSearchToolsMixin._searxng_staleness_checked:
            return None
        WebSearchToolsMixin._searxng_staleness_checked = True

        path = self._searxng_stale_state_path()
        now = time.time()
        try:
            with open(path, encoding="utf-8") as fh:
                last = float(json.load(fh).get("last_notified", 0.0))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            last = 0.0
        if now - last < _SEARXNG_STALE_NOTICE_INTERVAL_DAYS * 86400.0:
            return None

        age = self._searxng_image_age_days()
        if age is None or age < _SEARXNG_IMAGE_STALE_DAYS:
            return None

        try:
            from ...common.atomic_io import atomic_write_json

            atomic_write_json(path, {"last_notified": now, "age_days_at_notice": round(age, 1)})
        except Exception as e:  # persistence is best-effort; never fail the search
            logger.debug("web_search: could not persist staleness notice state (%s)", e)

        return (
            f"[SearXNG] The local searxng/searxng image is {age:.0f} days old. Its per-engine "
            f"scraping recipes go stale — a six-week-old image was measured serving 0 results "
            f"from an engine that a newer one restored. Refresh with `docker pull "
            f"searxng/searxng`, then recreate the container REUSING its existing volumes and "
            f"port mapping (`docker inspect searxng` first — a plain `docker rm` without "
            f"reattaching /etc/searxng loses your settings.yml)."
        )

    def _run_tier_parallel(
        self,
        backends: list[tuple[str, Any]],
        deadline: float,
    ) -> tuple[list[tuple[str, list[dict[str, str]]]], list[str], set[str]]:
        """Query ``backends`` concurrently; return what finished within ``deadline``.

        Returns ``(per_backend_results, errors, connect_failed)``.

        Deliberately NON-INTERACTIVE: a worker never raises a user Checkpoint.
        The SearXNG install/start prompt stays on the sequential path, because
        prompting from inside a parallel phase means the user is asked a question
        while other backends are still racing, and the answer can no longer
        influence the phase it belongs to.

        A backend that misses the deadline is abandoned, not cancelled — an
        in-flight HTTP request cannot be interrupted, so its thread finishes on
        its own and its result is simply dropped. That is the price of bounding
        the merge's latency to the deadline instead of the slowest engine.
        """
        collected: list[tuple[str, list[dict[str, str]]]] = []
        errors: list[str] = []
        connect_failed: set[str] = set()
        runnable = [(n, fn) for n, fn in backends if not self._backend_in_cooldown(n)]
        for name, _ in backends:
            if self._backend_in_cooldown(name):
                logger.debug("web_search: skipping %s (in cooldown)", name)
        if not runnable:
            return collected, errors, connect_failed

        # NOT a `with` block: ThreadPoolExecutor.__exit__ calls shutdown(wait=True),
        # which blocks until every worker finishes — so a 20s engine would still
        # cost 20s despite the deadline firing on time, making the deadline
        # decorative. Shut down without waiting and let the straggler's thread
        # finish on its own; its result is simply dropped. (A live HTTP request
        # cannot be cancelled, but it is bounded by _SEARCH_HTTP_TIMEOUT.)
        pool = ThreadPoolExecutor(max_workers=len(runnable), thread_name_prefix="websearch")
        try:
            futures = {pool.submit(fn): name for name, fn in runnable}
            try:
                for fut in as_completed(futures, timeout=deadline):
                    name = futures[fut]
                    try:
                        results = fut.result()
                    except _BlockWallError as e:
                        # Refusing to serve us; retrying feeds the detection that
                        # escalates to a hard IP block. Sideline for the cooldown.
                        self._trip_backend_cooldown(name)
                        errors.append(f"{name}: {e}")
                        logger.warning("web_search: %s walled (%s); sidelining it", name, e)
                    except _CONNECT_ERRORS as e:
                        connect_failed.add(name)
                        errors.append(f"{name}: {e}")
                        logger.warning("web_search: %s connect-failed (%s)", name, e)
                    except Exception as e:
                        errors.append(f"{name}: {e}")
                        logger.warning("web_search: %s failed (%s)", name, e)
                    else:
                        if results:
                            logger.info("web_search: %s returned %d results", name, len(results))
                            collected.append((name, results))
            except FuturesTimeoutError:
                pending = [futures[f] for f in futures if not f.done()]
                logger.warning(
                    "web_search: tier-1 deadline %.1fs reached; proceeding without %s",
                    deadline,
                    ", ".join(pending) or "(none)",
                )
                errors.append(f"deadline {deadline:.0f}s: {', '.join(pending)}")
        finally:
            pool.shutdown(wait=False, cancel_futures=True)
        # Restore the caller's ordering so _merge_search_results' final tiebreaker
        # reflects backend priority rather than which thread happened to finish first.
        order = {n: i for i, (n, _) in enumerate(runnable)}
        collected.sort(key=lambda kv: order.get(kv[0], 0))
        return collected, errors, connect_failed

    def _tool_search_web(self, args: dict[str, Any]) -> "ToolResult":
        """Search the web for information relevant to the current task.

        Backends in priority order:
            1. SearXNG            (env SEARXNG_BASE_URL) — only when explicitly
                                   configured; an opt-in private instance outranks
                                   any third-party engine.
            2. Startpage          (no key) — proxies Google's index, unmetered.
                                   Default primary: the only general-web backend
                                   that answered 10/10 probe queries (EN+KO) from
                                   this IP without a key, prompt or CAPTCHA.
            3. Brave Search API   (env BRAVE_API_KEY) — stable keyed API, but a
                                   metered free tier, so it sits behind Startpage.
            4. Naver (browser)    — headless-browser render of Naver's web vertical,
                                   gated to Korean queries by default
                                   (_should_try_naver); needed because Naver's
                                   results are JS-hydrated and invisible to httpx.
            6. SearXNG auto-setup — Docker present but no SEARXNG_BASE_URL. Last,
                                   because installing raises a user Checkpoint and
                                   must not preempt a backend that just works.

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

        # ── Tier 1: unmetered backends, queried in PARALLEL and merged ──
        # These two cost nothing per use — no quota, no bot-detection escalation,
        # no browser — so there is no reason to stop at whichever answers first.
        # Querying both and merging buys the one ranking signal a single backend
        # cannot produce: cross-index agreement (see _merge_search_results).
        #
        # It also fixes a real gap. SearXNG's own Google engine is dead from a
        # flagged IP (HTTP 200 consent wall → 0 results, and SILENTLY: it never
        # appears in unresponsive_engines), and its startpage engine currently
        # fails with "parsing error". Measured 2026-07-19: SearXNG's 24 working
        # engines overlap Startpage's Google-derived results by only 0-2/10 on
        # most queries. So a first-wins chain with SearXNG in front would drop
        # Google's index entirely — the two are complements, not substitutes.
        tier1: list[tuple] = []
        searxng_autosetup: Optional[tuple] = None
        if searxng_url:
            # Explicitly configured self-hosted instance: the user opted into a
            # private backend, so it participates in the merge.
            tier1.append(("SearXNG", lambda: self._search_searxng(query, max_results, searxng_url)))
        elif self._has_docker_or_colima():
            # SEARXNG_BASE_URL unset but Docker/Colima present → we CAN offer to
            # install SearXNG, but that offer raises a user Checkpoint, so it must
            # not preempt a backend that already works without any prompt, and it
            # must not run inside the parallel phase. Deferred to tier 2.
            searxng_autosetup = ("SearXNG", lambda: self._setup_and_search_searxng(query, max_results))
        tier1.append(("Startpage", lambda: self._search_startpage(query, max_results)))

        per_backend, tier1_errors, connect_failed = self._run_tier_parallel(tier1, _TIER1_DEADLINE_SEC)
        merged = _merge_search_results(per_backend, max_results)
        if merged:
            return self._format_search_results(
                query,
                merged,
                [n for n, _ in per_backend],
                notice=self._stale_searxng_image_notice() if searxng_url else None,
            )

        # ── Tier 2: sequential fallback — each of these has a per-use cost ──
        # Reached only when tier 1 produced NOTHING. They are deliberately not in
        # the merge: Brave is a metered free tier (2000/month — merging would
        # spend one on every search), DuckDuckGo feeds the bot-detection that
        # escalates to a hard IP block on every attempt, and Naver spins a
        # headless browser. First-wins is the right policy when each try costs.
        backends: list[tuple] = []
        last_error = "; ".join(tier1_errors)
        if searxng_url and "SearXNG" in connect_failed:
            # Connect failure against an explicitly configured instance means it
            # is not running — that is the interactive install/start path, which
            # must run here rather than in a parallel worker.
            backends.append(("SearXNG", lambda: self._search_searxng(query, max_results, searxng_url)))
        if brave_key:
            backends.append(("Brave", lambda: self._search_brave(query, max_results, brave_key)))
        # DuckDuckGo is OPT-IN (ASICODE_DDG_FALLBACK=on). See _should_try_ddg for
        # the measurements: on a TLS-fingerprint-flagged IP it runs ~2/6 and every
        # failure feeds the escalation to a full TCP-level block.
        if self._should_try_ddg():
            backends.append(("DuckDuckGo", lambda: self._search_duckduckgo(query, max_results)))
        # Naver (browser-rendered) is the LAST resort: it spins a headless browser
        # (heavy) and shines mainly on Korean-local queries, so it is gated by
        # _should_try_naver and only reached when every lighter backend above
        # returned nothing/failed.
        if self._should_try_naver(query):
            backends.append(("Naver", lambda: self._search_naver_browser(query, max_results)))
        # Last: the SearXNG install offer (deferred above). It prompts the user, so
        # it runs only when every no-prompt backend has already come up empty.
        if searxng_autosetup is not None:
            backends.append(searxng_autosetup)

        # ── Try each tier-2 backend with fallback ──
        results: list[dict[str, str]] = []
        for name, search_fn in backends:
            # Circuit breaker: skip a backend that recently connect-failed instead
            # of re-paying its connect timeout on every search this session.
            if self._backend_in_cooldown(name):
                logger.debug("web_search: skipping %s (in cooldown after a recent connect failure)", name)
                continue
            try:
                results = search_fn()
                if results:
                    logger.info("web_search: %s returned %d results for '%s'", name, len(results), query)
                    break
            except (httpx.ConnectError, httpx.ConnectTimeout, httpx.RemoteProtocolError) as e:
                if name == "SearXNG":
                    # ConnectTimeout (host not responding / packet drop) means the
                    # same thing as ConnectError for a self-hosted instance — it is
                    # not running — so it must trigger the install/start prompt too.
                    # ConnectTimeout is NOT a subclass of ConnectError in httpx, so
                    # it has to be listed explicitly.
                    last_error = self._handle_searxng_connect_error(e, query, max_results, searxng_url)
                    # _handle_searxng_connect_error may have retried after installing;
                    # if results is now populated, we have a successful install+search
                    if last_error is None:
                        # Retry the search OUTSIDE the ConnectError handler above. A
                        # fresh connect failure here must fall through to the next
                        # backend (DuckDuckGo) instead of crashing search_web — wrap it.
                        try:
                            results = self._search_searxng(
                                query, max_results, os.environ.get("SEARXNG_BASE_URL", "http://localhost:8080")
                            )
                            if results:
                                logger.info("web_search: SearXNG succeeded after install")
                                break
                        except Exception as retry_err:
                            logger.warning(
                                "web_search: SearXNG retry after install failed (%s), falling back", retry_err
                            )
                            last_error = f"SearXNG: {retry_err}"
                    continue
                # Non-SearXNG connect-level failure (unreachable / IP-blocked host):
                # trip the session breaker so later searches skip it during cooldown.
                if isinstance(e, _CONNECT_ERRORS):
                    self._trip_backend_cooldown(name)
                last_error = f"{name}: {e}"
                logger.warning("web_search: %s failed (%s), trying next backend", name, e)
                continue
            except _BlockWallError as e:
                # The engine is refusing to serve this client, and every further
                # attempt reinforces its bot-detection (which escalates to a hard
                # IP block). Sideline it for the cooldown rather than re-asking on
                # each search — the same reasoning as the connect-failure breaker,
                # a different trigger.
                self._trip_backend_cooldown(name)
                last_error = f"{name}: {e}"
                logger.warning("web_search: %s walled (%s); sidelining it, trying next backend", name, e)
                continue
            except Exception as e:
                last_error = f"{name}: {e}"
                logger.warning("web_search: %s failed (%s), trying next backend", name, e)
                continue

        if not results:
            error_msg = last_error or "No results found from any backend."
            return self._make_result(ok=True, content=f"No results found. ({error_msg})", metadata={"result_count": 0})

        return self._format_search_results(query, results, [name])

    def _format_search_results(
        self,
        query: str,
        results: list[dict[str, str]],
        backends: list[str],
        notice: Optional[str] = None,
    ) -> "ToolResult":
        """Render results for the model. Shared by the merged and fallback paths.

        A merged result carries a ``sources`` field; it is surfaced only when
        MORE than one backend agreed, because that is the case where it tells the
        model something (independent indexes converged on this page) rather than
        just naming which engine happened to answer.

        ``notice`` (operational hints such as a stale SearXNG image) is placed in
        the CONTENT, not only the log. A warning that only reaches a log file is
        a warning nobody acts on — the model reads tool output, so that is the
        channel that can actually surface it to the user.
        """
        lines = [f"Web search results for: {query}", ""]
        if notice:
            lines += [notice, ""]
        for i, r in enumerate(results, 1):
            lines.append(f"{i}. {r['title']}")
            lines.append(f"   URL: {r['url']}")
            sources = (r.get("sources") or "").split(",")
            if len(sources) > 1:
                lines.append(f"   [confirmed by {len(sources)} sources: {', '.join(sources)}]")
            if r.get("snippet"):
                lines.append(f"   {r['snippet'][:400]}")
            lines.append("")

        content = "\n".join(lines).strip()
        return self._make_result(
            ok=True,
            content=content if content else "(empty results)",
            metadata={
                "result_count": len(results),
                "query": query,
                "backends": ",".join(backends),
            },
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
        retry_statuses: frozenset[int] = _RETRYABLE_HTTP_STATUSES,
    ) -> httpx.Response:
        """Execute an HTTP GET/POST with transient-error AND transient-status retry.

        Shared by all three search backends so SearXNG / DuckDuckGo / Brave use one
        retry policy. Two retry triggers, each backed off ``retries`` times:

        * transient NETWORK errors (``_TRANSIENT_HTTP_ERRORS``: connect/read
          timeouts, protocol errors) — retry after ``backoff`` seconds.
        * transient HTTP STATUS codes (``retry_statuses``: 429 rate-limit,
          502/503/504 gateway overload) — retry after the server's ``Retry-After``
          hint (capped at 30s via ``_retry_after_seconds``), or ``backoff``.

        A non-retryable status (e.g. 4xx other than 429) is returned as-is so the
        caller's ``raise_for_status()`` raises the precise HTTPStatusError, and the
        fallback chain keeps its existing contract. The final attempt's response is
        likewise returned for the caller to raise.
        """
        last_err: Optional[Exception] = None
        resp: Optional[httpx.Response] = None
        for attempt in range(retries):
            try:
                if method.upper() == "GET":
                    resp = client.get(url, params=params, headers=headers)
                else:
                    resp = client.post(url, data=data, headers=headers)
            except _TRANSIENT_HTTP_ERRORS as e:
                last_err = e
                # Connect-level failures are not retried (see _CONNECT_ERRORS): the
                # host is unreachable, so an immediate re-connect just re-pays the
                # connect timeout. Fail fast to the caller / fallback chain.
                if isinstance(e, _CONNECT_ERRORS):
                    raise
                if attempt < retries - 1:
                    logger.warning(
                        "web_search: transient HTTP error (attempt %d/%d: %s), retrying in %.1fs…",
                        attempt + 1,
                        retries,
                        e,
                        backoff,
                    )
                    time.sleep(backoff)
                    continue
                raise  # final attempt failed — propagate

            # Transient status code (rate-limit / gateway): back off and retry,
            # honouring Retry-After when the server supplies it.
            if resp.status_code in retry_statuses and attempt < retries - 1:
                wait = _retry_after_seconds(resp, backoff)
                logger.warning(
                    "web_search: HTTP %d (attempt %d/%d); retrying in %.1fs…",
                    resp.status_code,
                    attempt + 1,
                    retries,
                    wait,
                )
                time.sleep(wait)
                continue
            return resp

        # Exhausted status retries on the final attempt: return the last response
        # so the caller's raise_for_status() surfaces the precise HTTP status.
        # (With retries>=1 the loop always returns or raises before exiting, so
        # this is defensive — keeps the type checker and the old transient-error
        # contract intact.)
        if resp is not None:
            return resp
        if last_err is not None:
            raise last_err
        raise AssertionError("invariant: _http_request_with_retry exited without a response")

    # ── DuckDuckGo (no API key; OPT-IN, see _should_try_ddg) ─────────

    def _should_try_ddg(self) -> bool:
        """Whether DuckDuckGo participates in the fallback chain. Default: NO.

        Removed from the default chain 2026-07-19 after measuring what it
        actually costs. DDG discriminates on **TLS fingerprint**, not on IP and
        not on a usage quota — verified three ways from one IP:

        * same IP, same second: httpx got HTTP 202 + 0 results while a
          browser-TLS client got 200 + 10 results (so: not an IP block);
        * 12 consecutive browser-TLS queries succeeded 12/12 with no decay
          (so: not a usage limit) while httpx failed after 2-3;
        * the 202 body is DDG's OWN duck-CAPTCHA — "bots use DuckDuckGo too"
          (so: DDG itself, not an upstream index provider).

        httpx cannot pass that check (enabling HTTP/2 made it *worse*, 1/6 vs
        2/6 — the tell is the TLS ClientHello, not the protocol), so on a
        flagged IP DDG runs at ~2/6 while every failure feeds the bot-detection
        that escalates to the TCP-level IP block which previously took out
        *every* engine at once. A backend that is mostly-broken AND raises the
        odds of a total outage does not belong in the default path.

        Kept behind ``ASICODE_DDG_FALLBACK=on`` rather than deleted: the
        evidence is from ONE IP, and DDG may well be healthy from a clean one.
        The circuit breaker also trips on a wall now, so even when enabled it
        sidelines itself instead of hammering.
        """
        return os.environ.get("ASICODE_DDG_FALLBACK", "off").strip().lower() in ("on", "always", "1", "true")

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

        with httpx.Client(timeout=_SEARCH_HTTP_TIMEOUT, follow_redirects=True, headers=headers) as client:
            resp = self._http_request_with_retry(client, "POST", url, data=data)
            resp.raise_for_status()

        parser = _DDGResultParser(max_results=max_results)
        parser.feed(resp.text)
        parser.close()  # flushes a trailing title-only result (no snippet endtag)
        results = parser.results
        self._guard_block_wall("DuckDuckGo", resp.text, results, status=resp.status_code)
        return results

    # ── Startpage (no API key; proxies Google's index) ───────────────

    def _search_startpage(self, query: str, max_results: int) -> list[dict[str, str]]:
        """Search using Startpage (``/sp/search``) — keyless, unmetered, Google-quality.

        Startpage is a privacy proxy in FRONT of Google's index, so it returns
        general-web results of Google quality without Google's scraper detection
        (Google itself serves this IP a consent/"unusual traffic" wall, and even a
        headless browser does not get past it — see the Naver backend's notes).

        Measured 2026-07-19 from a single IP: 10/10 varied queries succeeded back
        to back — English and Korean alike — with no CAPTCHA, no rate-limit and no
        API key, using this same plain-httpx client. That is why it leads the
        chain; it is also why the wall guard below matters, since the day
        Startpage does start challenging us the failure would otherwise look
        exactly like a genuine empty result set.
        """
        url = "https://www.startpage.com/sp/search"
        headers = {
            "User-Agent": _BROWSER_UA,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        }
        params = {"query": query}

        with httpx.Client(timeout=_SEARCH_HTTP_TIMEOUT, follow_redirects=True, headers=headers) as client:
            resp = self._http_request_with_retry(client, "GET", url, params=params)
            resp.raise_for_status()

        parser = _StartpageResultParser(max_results=max_results)
        parser.feed(resp.text)
        parser.close()
        results = parser.results
        self._guard_block_wall("Startpage", resp.text, results, status=resp.status_code)
        return results

    def _search_brave(self, query: str, max_results: int, api_key: str) -> list[dict[str, str]]:
        """Search using Brave Search API (requires BRAVE_API_KEY)."""
        url = "https://api.search.brave.com/res/v1/web/search"
        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": api_key,
        }
        params = {"q": query, "count": max_results}

        with httpx.Client(timeout=_SEARCH_HTTP_TIMEOUT) as client:
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

    # ── Naver (browser-rendered, JS-hydrated results) ────────────────

    def _should_try_naver(self, query: str) -> bool:
        """Whether to attempt the browser-backed Naver fallback for this query.

        Env ``ASICODE_NAVER_FALLBACK`` selects the policy:
          * ``off``    — never (opt out of the browser fallback entirely)
          * ``always`` — for any query that reaches this last-resort slot
          * default / ``korean`` — only when the query contains Hangul, where
            Naver's Korean-local coverage justifies the browser cost (the lighter
            backends already handle Latin-script queries well).
        """
        mode = os.environ.get("ASICODE_NAVER_FALLBACK", "korean").strip().lower()
        if mode == "off":
            return False
        if mode == "always":
            return True
        return bool(_HANGUL_RE.search(query))

    def _search_naver_browser(self, query: str, max_results: int) -> list[dict[str, str]]:
        """Search Naver's web vertical via a headless browser (JS-rendered results).

        Naver serves its result list as client-hydrated JSON, so — unlike the
        httpx-scraping backends — the results are absent from the raw HTML and
        only a real browser sees them. ``_render_and_eval`` (on the browser mixin)
        navigates an ISOLATED page and runs ``_NAVER_EXTRACT_JS``, whose
        class-hash-free structural extraction survives Naver's per-deploy restyles.
        Raises ``RuntimeError`` (via ``_render_and_eval``) when the browser is
        unavailable, so the fallback chain records a real reason.
        """
        url = "https://search.naver.com/search.naver?where=web&query=" + urllib.parse.quote(query)
        raw = self._render_and_eval(url, _NAVER_EXTRACT_JS, timeout_ms=20000)

        results: list[dict[str, str]] = []
        for item in raw or []:
            if len(results) >= max_results:
                break
            if not isinstance(item, dict):
                continue
            title = str(item.get("title", "")).strip()
            href = str(item.get("url", "")).strip()
            if not title or not href.startswith("http"):
                continue
            results.append(
                {"title": title, "url": href, "snippet": str(item.get("snippet", "")).strip()}
            )
        return results

    # ── SearXNG (self-hosted) ────────────────────────────────────────

    @staticmethod
    def _searxng_engines() -> str:
        """Comma-separated engine list for SearXNG, or "" to use the category.

        ``ASICODE_SEARXNG_ENGINES`` overrides the measured default
        (:data:`_SEARXNG_DEFAULT_ENGINES`); the literal value ``category``
        restores SearXNG's own ``categories=general`` selection. Engine health is
        volatile and instance-specific, so this has to be tunable without a
        release — a hardcoded list is guaranteed to drift.
        """
        raw = os.environ.get("ASICODE_SEARXNG_ENGINES", "").strip()
        if not raw:
            return _SEARXNG_DEFAULT_ENGINES
        if raw.lower() == "category":
            return ""
        # Normalise: tolerate spaces after commas and stray empties.
        return ",".join(part.strip() for part in raw.split(",") if part.strip())

    def _search_searxng(self, query: str, max_results: int, base_url: str) -> list[dict[str, str]]:
        """Search using self-hosted SearXNG instance, with retry on transient errors."""
        base_url = base_url.rstrip("/")
        url = f"{base_url}/search"
        # Result language is configurable via ASICODE_SEARCH_LANG (SearXNG value,
        # e.g. "ko-KR", "en-US", "all"). Default "all" is neutral: it imposes no
        # language filter so the query's own language dominates — the old hardcoded
        # "en-US" put Korean queries at a disadvantage.
        params = {
            "q": query,
            "format": "json",
            "language": os.environ.get("ASICODE_SEARCH_LANG", "all"),
            "pageno": 1,
        }
        # Ask for engines BY NAME rather than letting categories=general choose —
        # see _SEARXNG_DEFAULT_ENGINES for why the category default is a bad set.
        engines = self._searxng_engines()
        if engines:
            params["engines"] = engines
        else:
            params["categories"] = "general"

        # Retry policy is shared with the other backends via _http_request_with_retry
        # (previously SearXNG was the only backend with any retry; DDG/Brave had none).
        with httpx.Client(timeout=_SEARCH_HTTP_TIMEOUT, follow_redirects=True) as client:
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

        # SearXNG is a metasearch aggregator: an EMPTY result set while one or
        # more upstream engines are flagged "unresponsive" (rate-limited /
        # access-denied / timed-out) is an infrastructure failure, not a genuine
        # "nothing matched". Silently returning [] here masks that behind an
        # ordinary miss and falls through to the next backend with no diagnostic
        # — the exact trap the block-wall check (_guard_block_wall) exists to
        # prevent for scraping backends. Raise so the chain records the real reason
        # (which engines died and why). A real empty set — every engine answered
        # with nothing — has an EMPTY unresponsive_engines and still returns [].
        if not results:
            unresponsive = data.get("unresponsive_engines") or []
            if unresponsive:
                detail = "; ".join(_format_unresponsive_engine(e) for e in unresponsive)
                # No "SearXNG:" prefix — the fallback chain's ``f"{name}: {e}"``
                # already supplies the backend label (mirrors the DDG anomaly raise).
                raise RuntimeError(f"all upstream engines unresponsive ({detail})")

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
                    capture_output=True,
                    text=True,
                    timeout=10,
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
                capture_output=True,
                text=True,
                timeout=10,
            )
            daemon_alive = info.returncode == 0
        except (subprocess.TimeoutExpired, OSError):
            daemon_alive = False

        if daemon_alive:
            # Check for image or stopped container
            try:
                r = subprocess.run(
                    [docker_path, "image", "inspect", "searxng/searxng"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if r.returncode == 0:
                    return True
            except (subprocess.TimeoutExpired, OSError):
                pass
            try:
                r = subprocess.run(
                    [docker_path, "ps", "-a", "--filter", "name=searxng", "--format", "{{.Names}}"],
                    capture_output=True,
                    text=True,
                    timeout=10,
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
        is issued at most once per session — covering both the LLM retrying
        search_web sequentially AND two searches running concurrently on the
        tool-executor pool (see _searxng_setup_lock).
        """
        if self._searxng_start_decision is not None:
            return self._searxng_start_decision
        with WebSearchToolsMixin._searxng_setup_lock:
            # Re-check under the lock: the thread we just queued behind may have
            # asked this very question and cached the answer while we waited.
            # Without this second read we would prompt again with the answer
            # already in hand — the whole bug this lock exists to prevent.
            if self._searxng_start_decision is not None:
                return self._searxng_start_decision
            try:
                result = self._tool_ask_user(
                    {
                        "question": (
                            "SearXNG is installed but not currently running.\nWould you like to start it now?"
                        ),
                        "type": "confirm",
                        "options": ["yes", "no"],
                        "default": "no",
                        "reason": "SearXNG is installed but not running",
                    }
                )
                answer = result.metadata.get("answer", "no").lower().strip()
                self._searxng_start_decision = answer == "yes"
                return self._searxng_start_decision
            except Exception as e:
                logger.warning("web_search: ask_user failed (%s), skipping SearXNG start", e)
                self._searxng_start_decision = False
                return False

    def _wait_for_searxng(self, base_url: str, timeout: float = 15.0, interval: float = 0.5) -> bool:
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
                    logger.info("web_search: SearXNG ready (healthz=%s)", resp.status_code)
                    return True
            except Exception as e:  # ConnectionError, Timeout, etc.
                last_err = e
            time.sleep(interval)
        logger.warning("web_search: SearXNG not ready after %.1fs (%s)", timeout, last_err)
        return False

    def _start_searxng(self) -> bool:
        """Start SearXNG: start Colima (if needed) then run the SearXNG container.

        Returns True if SearXNG is successfully running after this call.

        Serialized on ``_searxng_setup_lock``: two concurrent searches that both
        got a "yes" would otherwise race their own ``docker run --name
        asicode-searxng`` (the second failing on a duplicate container name) and
        double-pull the image. Holding the lock makes the second caller run the
        ``docker ps -a`` branch below, find the container the first one created,
        and take the idempotent ``docker start`` path.
        """
        with WebSearchToolsMixin._searxng_setup_lock:
            return self._start_searxng_locked()

    def _start_searxng_locked(self) -> bool:
        """Body of :meth:`_start_searxng`; call only with the setup lock held."""
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
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if status.returncode != 0:
                    logger.info("web_search: Starting Colima...")
                    subprocess.run(
                        ["colima", "start"],
                        capture_output=True,
                        text=True,
                        timeout=120,
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
                capture_output=True,
                text=True,
                timeout=10,
            )
            container_names = ps_result.stdout.strip().split()
            if container_names:
                # Start all existing searxng containers
                all_ok = True
                for name in container_names:
                    try:
                        subprocess.run(
                            [docker_path, "start", name],
                            capture_output=True,
                            text=True,
                            timeout=30,
                            check=True,
                        )
                        logger.info("web_search: Started SearXNG container '%s'", name)
                    except subprocess.CalledProcessError:
                        all_ok = False
                # Healthz poll ONCE after all containers started — not inside the loop,
                # where it would block up to N x timeout serially on a multi-container
                # setup (each warm container still passes the same readiness check).
                if all_ok:
                    base_url = os.environ.get("SEARXNG_BASE_URL", "http://localhost:8080")
                    self._wait_for_searxng(base_url)
                return all_ok

            # No existing container — pull and run
            logger.info("web_search: Pulling SearXNG Docker image...")
            subprocess.run(
                [docker_path, "pull", "searxng/searxng"],
                capture_output=True,
                text=True,
                timeout=120,
                check=True,
            )
            base_url = os.environ.get("SEARXNG_BASE_URL", "http://localhost:8080")
            port = _searx_host_port(base_url)
            subprocess.run(
                [
                    docker_path,
                    "run",
                    "-d",
                    "--name",
                    "asicode-searxng",
                    "-p",
                    f"{port}:8080",
                    "searxng/searxng",
                ],
                capture_output=True,
                text=True,
                timeout=30,
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

        Caches the decision in self._searxng_install_decision so the prompt is
        issued at most once per session — covering both sequential retries and
        concurrent searches (see _searxng_setup_lock).
        """
        if self._searxng_install_decision is not None:
            return self._searxng_install_decision
        with WebSearchToolsMixin._searxng_setup_lock:
            # Re-check under the lock — see _ask_start_searxng.
            if self._searxng_install_decision is not None:
                return self._searxng_install_decision
            try:
                result = self._tool_ask_user(
                    {
                        "question": (
                            "SearXNG is needed for web search but is not installed.\n"
                            "Would you like to install and start a local SearXNG instance?"
                        ),
                        "type": "confirm",
                        "options": ["yes", "no"],
                        "default": "no",
                        "reason": "SearXNG required for web search, but not installed",
                    }
                )
                answer = result.metadata.get("answer", "no").lower().strip()
                self._searxng_install_decision = answer == "yes"
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

    def _handle_searxng_connect_error(
        self, error: Exception, query: str, max_results: int, searxng_url: str
    ) -> Optional[str]:
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
        """Fetch and read content from a URL.

        HTML pages are converted to text via ``_HTMLTextExtractor`` (block-aware,
        paragraph-preserving) instead of the former regex strip that collapsed
        every page to a single run-on line. Charset resolution follows the HTML5
        sniffing order: HTTP ``Content-Type`` header charset first, then a BOM /
        ``<meta charset>`` prescan of the body (``_sniff_html_encoding``), so pages
        that declare encoding only in markup (EUC-KR/CP949 legacy Korean sites)
        are no longer UTF-8 mangled. ``start_index`` lets the caller resume reading
        past a previous TRUNCATION point. Clearly-binary Content-Types (PDF,
        image, archive, …) are rejected rather than decode-mangled into context.

        Uses ``client.stream("GET", …)`` so the Content-Length OOM guard actually
        prevents materialising a huge body (the pre-streaming version used a
        non-streaming ``client.get()`` that downloaded the full body *before*
        checking Content-Length — the guard was dead code for the download, only
        saving the decode copy). Transient network errors and HTTP 429/5xx status
        codes are retried with backoff, matching the search backends' policy.
        """
        url = str(args.get("url", "")).strip()
        max_chars = int(args.get("max_chars", 15000))
        max_chars = max(1000, min(max_chars, 50000))
        start_index = int(args.get("start_index", 0))
        start_index = max(0, start_index)

        if not url:
            return self._make_result(ok=False, content="", error="'url' is required")

        if not url.startswith(("http://", "https://")):
            url = "https://" + url

        # ── Reddit: www.reddit.com → old.reddit.com (www blocking bypass) ──
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
            with httpx.Client(timeout=30.0, follow_redirects=True, headers=headers) as client:
                # ── Streaming GET with retry + OOM guard ──────────────────────
                # Retry loop mirrors _http_request_with_retry: transient network
                # errors AND transient HTTP status codes (429/5xx) are retried
                # with backoff (via _retry_after_seconds for status codes).
                last_err: Optional[Exception] = None
                body_bytes: Optional[bytes] = None
                content_type = ""
                header_charset: Optional[str] = None

                for attempt in range(3):  # retries=2 → 3 total attempts
                    try:
                        with client.stream("GET", url) as stream_resp:
                            stream_resp.raise_for_status()
                            content_type = stream_resp.headers.get("content-type", "").lower()

                            # ── Reject clearly-binary content BEFORE reading body ──
                            # Catch PDF/image/archive at the header level so we never
                            # download megabytes of binary just to throw it away.
                            if content_type and any(
                                content_type.startswith(p) for p in _FETCH_BINARY_CONTENT_PREFIXES
                            ):
                                return self._make_result(
                                    ok=False,
                                    content="",
                                    error=(
                                        f"Refusing to fetch {url}: Content-Type "
                                        f"'{content_type}' is binary/non-text and cannot "
                                        f"be read as text. Use browser_action "
                                        f"(action='navigate' or 'screenshot') to view "
                                        f"it, or a dedicated download tool."
                                    ),
                                )

                            # OOM guard: refuse oversized responses BEFORE reading body.
                            # Content-Length check catches the clear-cut case; the
                            # streaming byte-cap catches chunked / no-CL responses.
                            _cl = stream_resp.headers.get("content-length")
                            if _cl:
                                try:
                                    if int(_cl) > _WEB_FETCH_MAX_BYTES:
                                        _mb = int(_cl) // (1024 * 1024)
                                        _lim = _WEB_FETCH_MAX_BYTES // (1024 * 1024)
                                        return self._make_result(
                                            ok=False,
                                            content="",
                                            error=(
                                                f"Refusing to fetch {url}: response is ~{_mb}MB "
                                                f"(limit {_lim}MB). Use browser_action or a "
                                                f"dedicated download tool for large files."
                                            ),
                                        )
                                except ValueError:
                                    pass  # malformed Content-Length — ignore

                            # Stream body with byte cap (catches chunked responses
                            # that have no Content-Length at all).
                            buf = bytearray()
                            for chunk in stream_resp.iter_bytes():
                                buf.extend(chunk)
                                if len(buf) > _WEB_FETCH_MAX_BYTES:
                                    _lim = _WEB_FETCH_MAX_BYTES // (1024 * 1024)
                                    return self._make_result(
                                        ok=False,
                                        content="",
                                        error=(
                                            f"Refusing to fetch {url}: streaming body "
                                            f"exceeded {_lim}MB limit. Use browser_action "
                                            f"or a dedicated download tool."
                                        ),
                                    )

                            body_bytes = bytes(buf)
                            # Capture the HTTP header charset (None when the
                            # header carries no charset); a body-level <meta
                            # charset> is resolved post-loop via prescan, since
                            # httpx never inspects the body for encoding.
                            header_charset = stream_resp.charset_encoding
                            break  # success — exit retry loop

                    except _TRANSIENT_HTTP_ERRORS as e:
                        last_err = e
                        if attempt < 2:
                            logger.warning(
                                "web_fetch: transient HTTP error (attempt %d/3: %s), "
                                "retrying in %.1fs…",
                                attempt + 1, e, 1.5,
                            )
                            time.sleep(1.5)
                            continue
                        raise  # final attempt — propagate

                    except httpx.HTTPStatusError as e:
                        code = e.response.status_code
                        if code in _RETRYABLE_HTTP_STATUSES and attempt < 2:
                            wait = _retry_after_seconds(e.response, 1.5)
                            logger.warning(
                                "web_fetch: HTTP %d (attempt %d/3); retrying in %.1fs…",
                                code, attempt + 1, wait,
                            )
                            time.sleep(wait)
                            continue
                        raise  # non-retryable status or final attempt — propagate

                if body_bytes is None:
                    if last_err is not None:
                        raise last_err
                    raise AssertionError("invariant: _tool_web_fetch broke out of retry loop without body")

                # ── Decode with charset detection ────────────────────────
                # Header charset wins; otherwise sniff <meta charset>/BOM from
                # the body (HTML5 prescan) so EUC-KR/CP949 pages that declare
                # encoding only in markup are not UTF-8 mangled. A bogus charset
                # name (rare) falls back to UTF-8 rather than raising LookupError.
                resp_encoding = header_charset or _sniff_html_encoding(body_bytes) or "utf-8"
                try:
                    text = body_bytes.decode(resp_encoding, errors="replace")
                except LookupError:
                    text = body_bytes.decode("utf-8", errors="replace")

                if "application/json" in content_type:
                    import json as _json

                    try:
                        formatted = _json.dumps(_json.loads(text), indent=2, ensure_ascii=False)
                    except Exception:
                        formatted = text
                elif "text/html" in content_type or "application/xhtml" in content_type:
                    extractor = _HTMLTextExtractor(base_url=url)
                    extractor.feed(text)
                    extractor.close()
                    formatted = extractor.get_text()
                else:
                    # text/plain, binary-as-text, or unknown — best-effort text view.
                    formatted = text

                # start_index: resume reading past a previous truncation.
                total_len = len(formatted)
                if start_index > 0:
                    if start_index >= total_len:
                        return self._make_result(
                            ok=True,
                            content=(
                                f"URL: {url}\nContent-Type: {content_type}\n\n"
                                f"(start_index={start_index} is past the end of the "
                                f"{total_len}-char content; there is nothing more to read.)"
                            ),
                            metadata={
                                "url": url,
                                "content_type": content_type,
                                "length": 0,
                                "start_index": start_index,
                                "total_length": total_len,
                            },
                        )
                    formatted = formatted[start_index:]

                # Capture the real content length BEFORE appending the truncation
                # marker so metadata["length"] reflects actual content, not the
                # ~90-char informational suffix. resume_at uses the un-truncated
                # offset and is unaffected.
                reported_len = len(formatted)
                if len(formatted) > max_chars:
                    resume_at = start_index + max_chars
                    reported_len = max_chars
                    formatted = (
                        formatted[:max_chars] + f"\n\n...[TRUNCATED at {max_chars} chars — call web_fetch again "
                        f"with start_index={resume_at} to continue reading]..."
                    )

                result = f"URL: {url}\nContent-Type: {content_type}\n\n{formatted}"
                return self._make_result(
                    ok=True,
                    content=result,
                    metadata={
                        "url": url,
                        "content_type": content_type,
                        "length": reported_len,
                        "start_index": start_index,
                    },
                )

        except httpx.TimeoutException:
            return self._make_result(ok=False, content="", error=f"Timeout fetching {url} (30s)")
        except httpx.HTTPStatusError as e:
            code = e.response.status_code
            hint = ""
            if code in (401, 403):
                hint = (
                    " — the site likely blocks automated requests or needs a "
                    "session; try browser_action navigate (renders via a real browser)"
                )
            return self._make_result(ok=False, content="", error=f"HTTP {code} fetching {url}{hint}")
        except Exception as e:
            return self._make_result(ok=False, content="", error=f"Failed to fetch {url}: {type(e).__name__}: {e}")
