"""Unit tests for web search/fetch tool handlers.

Covers regressions fixed in this change:

  - ``_DDGResultParser`` no longer drops results that have a title but no
    snippet (emit-on-snippet-endtag-only bug).
  - ``_HTMLTextExtractor`` preserves paragraph structure instead of collapsing
    every page to a single run-on line (the old ``re.sub(r'\\s+',' ')`` then
    ``split('\\n')`` was dead code).
  - ``_tool_web_fetch`` honors the page charset via ``resp.text`` (was a
    hardcoded UTF-8 decode), supports ``start_index`` to resume past a
    truncation, and hints at ``browser_action`` on 401/403.
  - ``_searx_host_port`` derives the docker ``-p`` port structurally instead of
    ``base_url.split(':')[-1]`` which broke on portless URLs.

No network: every HTTP call is stubbed via ``monkeypatch`` on ``httpx.Client``.
"""
from __future__ import annotations

import json
import re
import threading
import time

import httpx
import pytest

import external_llm.agent.tool_handlers.web_search_tools as wst
from external_llm.agent.tool_handlers.web_search_tools import (
    WebSearchToolsMixin,
    _DDGResultParser,
    _HTMLTextExtractor,
    _searx_host_port,
)


class _Host(WebSearchToolsMixin):
    """Minimal concrete host: stubs ``_make_result`` for offline testing."""

    repo_root = "."

    def _make_result(self, ok=False, content="", error=None, metadata=None, **kw):
        return {"ok": ok, "content": content, "error": error, "metadata": metadata or {}}


@pytest.fixture(autouse=True)
def _reset_backend_cooldown(monkeypatch, tmp_path):
    """Isolate the process-wide circuit-breaker state between tests.

    Covers BOTH breakers. The wall backoff additionally persists to
    ``<repo_root>/.asicode/search_backend_walls.json``, and ``_Host.repo_root``
    is ``"."`` — so without the redirect below, any test that trips a wall writes
    into the developer's real repo AND the next run loads it back and sidelines a
    backend that the test never blocked. Observed exactly that: one run persisted
    a Startpage wall, and the following run's chain-order test failed because
    Startpage was skipped before it could be called.

    Tests that assert on the state file set their own ``repo_root`` (a later
    monkeypatch wins); this is the default that keeps everyone else off the real
    tree.
    """
    mixin = wst.WebSearchToolsMixin
    monkeypatch.setattr(_Host, "repo_root", str(tmp_path), raising=False)

    def _clear():
        mixin._backend_cooldown.clear()
        mixin._wall_state.clear()
        mixin._wall_pending_strikes.clear()
        mixin._wall_since.clear()
        mixin._wall_state_loaded = False

    _clear()
    yield
    _clear()


@pytest.fixture(autouse=True)
def _neutralise_exa(monkeypatch):
    """Default Exa to an empty result set — same no-network contract as Startpage.

    Exa joined tier 1, so without this every ``_tool_search_web`` test issues a
    real POST to mcp.exa.ai and then waits out its connect timeout before the
    chain continues. Tests that are ABOUT Exa override this or call the real
    method unbound via ``_real_search_exa``.
    """
    monkeypatch.setattr(_Host, "_search_exa", lambda self, q, m: [], raising=False)


def _real_search_exa(host, query: str, max_results: int):
    """Invoke the genuine ``_search_exa``, bypassing the autouse stub."""
    return wst.WebSearchToolsMixin._search_exa(host, query, max_results)


@pytest.fixture(autouse=True)
def _neutralise_startpage(monkeypatch):
    """Default Startpage to a genuine empty result set for chain tests.

    Startpage leads the backend chain, so without this every test that exercises
    ``_tool_search_web`` to reach a LATER backend would issue a real HTTP request
    — breaking this module's "no network" contract and making the suite depend on
    a live third party. Returning ``[]`` (an honest miss, not a wall) makes the
    chain fall through exactly as those tests intend.

    Tests that are ABOUT Startpage either override this with their own stub (a
    later monkeypatch wins) or, to exercise the REAL implementation, call it
    unbound via ``_real_search_startpage`` below — going through ``_Host`` would
    hit this stub instead.
    """
    monkeypatch.setattr(_Host, "_search_startpage", lambda self, q, m: [], raising=False)


def _real_search_startpage(host, query: str, max_results: int):
    """Invoke the genuine ``_search_startpage``, bypassing the autouse stub."""
    return wst.WebSearchToolsMixin._search_startpage(host, query, max_results)


@pytest.fixture(autouse=True)
def _stub_dns(monkeypatch):
    """Resolve hostnames to a public IP so the SSRF guard never touches the network.

    web_fetch's SSRF guard resolves hostnames via ``socket.getaddrinfo``; without
    this, every web_fetch test would issue a real DNS query, violating the
    module's "no network" contract. Tests that exercise DNS resolution
    monkeypatch ``wst.socket.getaddrinfo`` themselves (later patches win).
    """
    monkeypatch.setattr(
        wst.socket, "getaddrinfo",
        lambda host, port, *a, **k: [
            (wst.socket.AF_INET, wst.socket.SOCK_STREAM, 6, "", ("93.184.216.34", port or 0))
        ],
    )


# ── _DDGResultParser ────────────────────────────────────────────────────

def test_ddg_parser_keeps_title_only_result():
    """A result with a title but no snippet must survive (was silently dropped)."""
    html = '<a class="result__a" href="https://only-title.example.com">Only Title</a>'
    p = _DDGResultParser(max_results=5)
    p.feed(html)
    p.close()
    assert len(p.results) == 1
    assert p.results[0]["title"] == "Only Title"
    assert p.results[0]["url"] == "https://only-title.example.com"
    assert p.results[0]["snippet"] == ""


def test_ddg_parser_keeps_leading_result_without_snippet_when_next_starts():
    """The first result is flushed when the next result__a begins, even mid-block."""
    html = (
        '<a class="result__a" href="https://a.example.com">First</a>'
        '<a class="result__a" href="https://b.example.com">Second</a>'
        '<a class="result__snippet" href="https://b.example.com">second snip</a>'
    )
    p = _DDGResultParser(max_results=5)
    p.feed(html)
    p.close()
    titles = [r["title"] for r in p.results]
    assert titles == ["First", "Second"]
    assert p.results[0]["snippet"] == ""   # no snippet ever appeared for "First"
    assert p.results[1]["snippet"] == "second snip"


def test_ddg_parser_normal_results_with_snippets():
    html = (
        '<a class="result__a" href="https://a.example.com">A Title</a>'
        '<a class="result__snippet" href="https://a.example.com">A snippet</a>'
        '<a class="result__a" href="https://b.example.com">B Title</a>'
        '<a class="result__snippet" href="https://b.example.com">B snippet</a>'
    )
    p = _DDGResultParser(max_results=5)
    p.feed(html)
    p.close()
    assert [(r["title"], r["snippet"]) for r in p.results] == [
        ("A Title", "A snippet"),
        ("B Title", "B snippet"),
    ]


def test_ddg_parser_respects_max_results():
    links = "".join(
        f'<a class="result__a" href="https://x{i}.example.com">T{i}</a>'
        f'<a class="result__snippet" href="https://x{i}.example.com">s{i}</a>'
        for i in range(10)
    )
    p = _DDGResultParser(max_results=3)
    p.feed(links)
    p.close()
    assert len(p.results) == 3


def test_ddg_parser_decodes_uddg_redirect():
    html = (
        '<a class="result__a" '
        'href="//duckduckgo.com/l/?uddg=https%3A%2F%2Freal.example.com%2Fpath">T</a>'
    )
    p = _DDGResultParser(max_results=5)
    p.feed(html)
    p.close()
    assert p.results[0]["url"] == "https://real.example.com/path"


# ── _HTMLTextExtractor ──────────────────────────────────────────────────

def test_html_extractor_preserves_paragraph_structure():
    """Block boundaries become newlines — NOT a single run-on line (regression)."""
    page = (
        "<html><head><title>T</title></head><body>"
        "<h1>Main Heading</h1>"
        "<p>First paragraph here.</p>"
        "<p>Second paragraph here.</p>"
        "<ul><li>Item one</li><li>Item two</li></ul>"
        "</body></html>"
    )
    out = _HTMLTextExtractor.extract(page)
    lines = out.split("\n")
    # Multiple distinct lines prove paragraph structure survived.
    assert len(lines) > 1, f"paragraph structure lost:\n{out}"
    assert "Main Heading" in out
    assert "First paragraph here." in out
    assert "Second paragraph here." in out
    assert "Item one" in out and "Item two" in out
    # No two paragraphs glued onto the same line.
    assert "First paragraph here.Second" not in out


def test_html_extractor_drops_script_and_style():
    page = (
        "<p>visible text</p>"
        "<script>var secret = 'leak';</script>"
        "<style>.x { color: red; }</style>"
        "<p>more visible text</p>"
    )
    out = _HTMLTextExtractor.extract(page)
    assert "visible text" in out
    assert "more visible text" in out
    assert "secret" not in out
    assert "leak" not in out
    assert "color" not in out


def _extract(page: str) -> str:
    e = _HTMLTextExtractor()
    e.feed(page)
    e.close()
    return e.get_text()


# Attach a tiny convenience classmethod-style helper for the tests above.
_HTMLTextExtractor.extract = staticmethod(_extract)  # type: ignore[attr-defined]


# ── _HTMLTextExtractor link preservation ────────────────────────────────
# The extractor emits ``visible text (absolute url)`` for followable <a href>
# when a base_url is supplied, so a research agent can chase links in a fetched
# page. These guard every branch of that path (scheme filter, base resolution,
# fragment strip, dedup, backward compat); the plain `.extract()` helper above
# passes no base_url, so it exercises only the text-only path.

_DOC_BASE = "https://example.com/docs/guide.html"


def _extract_linked(page: str, base: str = _DOC_BASE) -> str:
    e = _HTMLTextExtractor(base_url=base)
    e.feed(page)
    e.close()
    return e.get_text()


def test_html_extractor_preserves_absolute_link():
    out = _extract_linked('<p>See <a href="https://docs.python.org/3/">the docs</a> now</p>')
    assert "the docs (https://docs.python.org/3/)" in out


def test_html_extractor_resolves_relative_link_via_base():
    # ../ and root-relative hrefs both resolve against the fetched page's URL.
    assert "API (https://example.com/api/ref.html)" in _extract_linked('<a href="../api/ref.html">API</a>')
    assert "Other (https://example.com/other/page)" in _extract_linked('<a href="/other/page">Other</a>')


def test_html_extractor_rejects_non_followable_schemes():
    # mailto:/tel:/javascript: cannot be followed — keep the text, drop the URL.
    for href in ("mailto:x@y.com", "tel:+123", "javascript:void(0)"):
        out = _extract_linked(f'<a href="{href}">label</a>')
        assert "label" in out
        assert "(" not in out, f"{href} leaked a URL: {out!r}"


def test_html_extractor_rejects_fragment_only_anchor():
    out = _extract_linked('<a href="#section2">jump</a>')
    assert "jump" in out
    assert "(http" not in out and "section2" not in out


def test_html_extractor_strips_fragment_from_url():
    out = _extract_linked('<a href="https://x.com/p#frag">L</a>')
    assert "L (https://x.com/p)" in out
    assert "#frag" not in out


def test_html_extractor_skips_empty_anchor_text():
    # Image-only / whitespace-only anchor has no visible text to attach a URL to.
    out = _extract_linked('<a href="https://x.com/img"><img src="a.png"></a>text')
    assert "(https://x.com/img)" not in out


def test_html_extractor_skips_when_text_equals_url():
    # The visible text is already the URL — appending it again is pure noise.
    out = _extract_linked('<a href="https://x.com/p">https://x.com/p</a>')
    assert "https://x.com/p" in out
    assert "(https://x.com/p)" not in out


def test_html_extractor_dedups_repeated_link():
    # Header/footer nav repeats the same (text, url); emit the URL only once.
    out = _extract_linked(
        '<a href="https://x.com/a">Home</a> mid <a href="https://x.com/a">Home</a>'
    )
    assert out.count("(https://x.com/a)") == 1


def test_html_extractor_without_base_keeps_absolute_drops_relative():
    # Backward compat: no base_url → relative links can't be resolved and are
    # dropped (historical text-only behaviour), absolute links still preserved.
    out = _extract_linked('<a href="rel/path">rel</a> and <a href="https://abs.com/">abs</a>', base="")
    assert "abs (https://abs.com/)" in out
    assert "rel/path" not in out and "(rel" not in out


def test_html_extractor_nested_anchor_does_not_crash():
    # Nested <a> is invalid HTML; the open link is flushed before the inner one.
    out = _extract_linked('<a href="https://x.com/1">out<a href="https://x.com/2">in</a></a>')
    assert "(https://x.com/1)" in out and "(https://x.com/2)" in out


def test_html_extractor_ignores_link_inside_script():
    out = _extract_linked('<script><a href="https://evil.com">x</a></script>real')
    assert "evil.com" not in out
    assert "real" in out


# ── _searx_host_port ────────────────────────────────────────────────────

def test_searx_host_port_explicit():
    assert _searx_host_port("http://localhost:8080") == "8080"
    assert _searx_host_port("https://searx.example.com:9999") == "9999"


def test_searx_host_port_defaults_when_missing():
    # Portless URL used to yield "//searx.example.com" via split(":")[-1].
    assert _searx_host_port("https://searx.example.com") == "8080"
    assert _searx_host_port("http://localhost:8080/") == "8080"


# ── _tool_web_fetch (HTTP stubbed) ──────────────────────────────────────

class _FakeStreamResponse:
    """Wraps httpx.Response so it works inside a ``client.stream()`` context.

    ``stream_resp.encoding``, ``raise_for_status()``, ``headers`` and
    ``iter_bytes()`` come directly from the wrapped ``httpx.Response``.
    """

    def __init__(self, response: httpx.Response):
        self._response = response

    def __enter__(self):
        return self._response  # delegates .headers, .encoding, .raise_for_status, .iter_bytes

    def __exit__(self, *exc):
        return False


class _FakeClient:
    """Stub httpx.Client for offline web_fetch tests.

    Supports both ``get(url)`` (for search backends) and
    ``stream(method, url)`` (for web_fetch's streaming OOM-guard path).
    """

    def __init__(self, response: httpx.Response):
        self._response = response

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get(self, url, **kw):
        return self._response

    def stream(self, method, url, **kw):
        return _FakeStreamResponse(self._response)


def _stub_fetch(monkeypatch, response: httpx.Response):
    """Route web_fetch's httpx.Client() to return ``response`` for any method."""
    monkeypatch.setattr(wst.httpx, "Client", lambda *a, **k: _FakeClient(response))


def _html_response(text: str, ctype: str = "text/html; charset=utf-8") -> httpx.Response:
    return httpx.Response(200, request=httpx.Request("GET", "https://x/"), headers={"content-type": ctype}, text=text)


# ── web_fetch SSRF guard ─────────────────────────────────────────────────

def test_ssrf_guard_blocks_loopback_ips():
    for host in ("127.0.0.1", "127.0.0.2", "127.8.8.8", "::1"):
        with pytest.raises(wst._SSRFBlockedError):
            wst._assert_public_fetch_host(host)


def test_ssrf_guard_blocks_private_link_local_and_special_ips():
    for host in (
        "10.1.2.3", "172.16.0.1", "172.31.255.255", "192.168.0.1",
        "169.254.169.254", "fe80::1", "fc00::1", "fd12:3456::1",
        "0.0.0.0", "::", "224.0.0.1", "ff02::1", "240.0.0.1",
    ):
        with pytest.raises(wst._SSRFBlockedError):
            wst._assert_public_fetch_host(host)


def test_ssrf_guard_allows_public_ips():
    for host in ("8.8.8.8", "1.1.1.1", "93.184.216.34", "2606:2800:220:1:248:1893:25c8:1946"):
        wst._assert_public_fetch_host(host)  # must not raise


def test_ssrf_guard_unwraps_ipv4_mapped_ipv6():
    with pytest.raises(wst._SSRFBlockedError):
        wst._assert_public_fetch_host("::ffff:127.0.0.1")


def test_ssrf_guard_blocks_hostname_resolving_to_private(monkeypatch):
    monkeypatch.setattr(
        wst.socket, "getaddrinfo",
        lambda host, port, *a, **k: [
            (wst.socket.AF_INET, wst.socket.SOCK_STREAM, 6, "", ("127.0.0.1", port or 0))
        ],
    )
    with pytest.raises(wst._SSRFBlockedError):
        wst._assert_public_fetch_host("internal.example.com")


def test_ssrf_guard_blocks_mixed_resolution_any_private_hit(monkeypatch):
    monkeypatch.setattr(
        wst.socket, "getaddrinfo",
        lambda host, port, *a, **k: [
            (wst.socket.AF_INET, wst.socket.SOCK_STREAM, 6, "", ("93.184.216.34", port or 0)),
            (wst.socket.AF_INET, wst.socket.SOCK_STREAM, 6, "", ("10.0.0.9", port or 0)),
        ],
    )
    with pytest.raises(wst._SSRFBlockedError):
        wst._assert_public_fetch_host("dual.example.com")


def test_ssrf_guard_allows_public_hostname(monkeypatch):
    monkeypatch.setattr(
        wst.socket, "getaddrinfo",
        lambda host, port, *a, **k: [
            (wst.socket.AF_INET, wst.socket.SOCK_STREAM, 6, "", ("93.184.216.34", port or 0)),
            (wst.socket.AF_INET, wst.socket.SOCK_STREAM, 6, "", ("93.184.216.35", port or 0)),
        ],
    )
    wst._assert_public_fetch_host("example.com")  # must not raise


def test_ssrf_guard_passes_resolution_failure(monkeypatch):
    def _fail(host, port, *a, **k):
        raise wst.socket.gaierror("Name or service not known")
    monkeypatch.setattr(wst.socket, "getaddrinfo", _fail)
    wst._assert_public_fetch_host("no-such-host.invalid")  # must not raise


def test_ssrf_guard_env_hatch_disables(monkeypatch):
    for value in ("1", "on", "true"):
        monkeypatch.setenv(wst._SSRF_ALLOW_ENV, value)
        wst._assert_public_fetch_host("127.0.0.1")  # must not raise
        wst._assert_public_fetch_host("10.0.0.1")  # must not raise


def test_ssrf_request_hook_rejects_private_hop():
    class _FakeReq:
        def __init__(self, host):
            self.url = type("_FakeURL", (), {"host": host})()
    with pytest.raises(wst._SSRFBlockedError):
        wst._ssrf_request_hook(_FakeReq("::1"))
    wst._ssrf_request_hook(_FakeReq("example.com"))  # must not raise (autouse DNS stub)


def test_web_fetch_blocks_private_url(monkeypatch):
    host = _Host()
    _stub_fetch(monkeypatch, _html_response("<p>hi</p>"))
    res = host._tool_web_fetch({"url": "http://127.0.0.1:11434/api/tags"})
    assert not res["ok"]
    assert "SSRF" in res["error"]
    assert "127.0.0.1" in res["error"]


def test_web_fetch_private_url_allowed_with_env_hatch(monkeypatch):
    monkeypatch.setenv(wst._SSRF_ALLOW_ENV, "1")
    host = _Host()
    _stub_fetch(monkeypatch, _html_response("<p>hi</p>"))
    res = host._tool_web_fetch({"url": "http://127.0.0.1:8080/health"})
    assert res["ok"]


def test_web_fetch_preserves_paragraph_structure(monkeypatch):
    host = _Host()
    page = "<h1>Heading</h1><p>Para one.</p><p>Para two.</p>"
    _stub_fetch(monkeypatch, _html_response(page))
    res = host._tool_web_fetch({"url": "https://example.com"})
    assert res["ok"], res.get("error")
    # Both paragraphs present on separate lines (regression for single-line bug).
    assert "Para one." in res["content"]
    assert "Para two." in res["content"]
    assert "Para one.Para two" not in res["content"]


def test_web_fetch_truncation_reports_resume_index(monkeypatch):
    host = _Host()
    body = "<p>" + ("x" * 2500) + "</p>"
    _stub_fetch(monkeypatch, _html_response(body))
    res = host._tool_web_fetch({"url": "https://example.com", "max_chars": 1000})
    assert res["ok"]
    assert "TRUNCATED" in res["content"]
    assert "start_index=1000" in res["content"]


def test_web_fetch_start_index_resumes(monkeypatch):
    host = _Host()
    body = "<p>" + ("abcdefghij" * 250) + "</p>"  # 2500 chars
    _stub_fetch(monkeypatch, _html_response(body))
    # First read 1000 chars (the enforced minimum).
    first = host._tool_web_fetch({"url": "https://example.com", "max_chars": 1000})
    assert first["ok"] and "start_index=1000" in first["content"]
    # Resume at 1000 — the continuation must be reachable.
    second = host._tool_web_fetch({"url": "https://example.com", "max_chars": 1000, "start_index": 1000})
    assert second["ok"]
    assert second["metadata"]["start_index"] == 1000


def test_web_fetch_start_index_past_end(monkeypatch):
    host = _Host()
    _stub_fetch(monkeypatch, _html_response("<p>short</p>"))
    res = host._tool_web_fetch({"url": "https://example.com", "start_index": 99999})
    assert res["ok"]
    assert "nothing more to read" in res["content"]


def test_web_fetch_403_hints_browser_action(monkeypatch):
    host = _Host()
    resp = httpx.Response(403, request=httpx.Request("GET", "https://example.com"), text="forbidden")
    _stub_fetch(monkeypatch, resp)
    res = host._tool_web_fetch({"url": "https://example.com"})
    assert not res["ok"]
    assert "HTTP 403" in res["error"]
    assert "browser_action" in res["error"]


def test_web_fetch_json_pretty_printed(monkeypatch):
    host = _Host()
    resp = httpx.Response(
        200, request=httpx.Request("GET", "https://example.com"),
        headers={"content-type": "application/json"}, text='{"b": 2, "a": 1}',
    )
    _stub_fetch(monkeypatch, resp)
    res = host._tool_web_fetch({"url": "https://example.com"})
    assert res["ok"]
    assert '"a": 1' in res["content"]  # indented JSON, not raw minified


def test_web_fetch_language_env_passed_to_searxng(monkeypatch):
    """ASICODE_SEARCH_LANG overrides the SearXNG language param (was hardcoded en-US)."""
    monkeypatch.setenv("ASICODE_SEARCH_LANG", "ko-KR")
    captured = {}

    class _CapClient:
        def __enter__(self): return self
        def __exit__(self, *e): return False
        def get(self, url, params=None, headers=None):
            captured["params"] = params
            return httpx.Response(200, request=httpx.Request("GET", url), headers={"content-type": "application/json"}, text='{"results": []}')

    monkeypatch.setattr(wst.httpx, "Client", lambda *a, **k: _CapClient())
    host = _Host()
    host._search_searxng("한글 질의", 5, "http://localhost:8080")
    assert captured["params"]["language"] == "ko-KR"


# ── SearXNG: all-engines-unresponsive surfaced, not silently swallowed ───

class _SearxStubClient:
    """Stub httpx.Client for _search_searxng (GET → canned JSON body)."""

    def __init__(self, payload: dict):
        import json as _json

        self._text = _json.dumps(payload)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get(self, url, params=None, headers=None):
        return httpx.Response(
            200,
            request=httpx.Request("GET", url),
            headers={"content-type": "application/json"},
            text=self._text,
        )


def test_format_unresponsive_engine_shapes():
    assert wst._format_unresponsive_engine(["duckduckgo", "timeout"]) == "duckduckgo: timeout"
    # newer SearXNG appends extra fields → only the first two are used
    assert wst._format_unresponsive_engine(["brave", "Suspended", True]) == "brave: Suspended"
    # single-element / unexpected shapes degrade gracefully, never raise
    assert wst._format_unresponsive_engine(["lonely"]) == "lonely: unresponsive"
    assert wst._format_unresponsive_engine("weird") == "weird"


def test_searxng_all_engines_unresponsive_raises(monkeypatch):
    """0 results + a non-empty unresponsive_engines set is an infrastructure
    failure (every engine rate-limited/blocked), NOT a genuine miss: raise so
    the fallback chain records the real reason instead of a silent fall-through."""
    payload = {
        "results": [],
        "unresponsive_engines": [
            ["brave", "Suspended: too many requests"],
            ["duckduckgo", "timeout"],
        ],
    }
    monkeypatch.setattr(wst.httpx, "Client", lambda *a, **k: _SearxStubClient(payload))
    host = _Host()
    import pytest

    with pytest.raises(RuntimeError, match="unresponsive") as ei:
        host._search_searxng("anything", 5, "http://localhost:8080")
    # the offending engines/reasons are surfaced in the message for diagnosis
    assert "duckduckgo: timeout" in str(ei.value)


def test_searxng_empty_with_no_unresponsive_returns_empty(monkeypatch):
    """A genuine empty set — every engine answered with nothing — must NOT raise."""
    payload = {"results": [], "unresponsive_engines": []}
    monkeypatch.setattr(wst.httpx, "Client", lambda *a, **k: _SearxStubClient(payload))
    host = _Host()
    assert host._search_searxng("obscure query", 5, "http://localhost:8080") == []


def test_searxng_results_present_ignore_unresponsive(monkeypatch):
    """When some engines returned results, a partial unresponsive set is not
    fatal: return the results without raising."""
    payload = {
        "results": [{"title": "T", "url": "https://x/", "content": "snip"}],
        "unresponsive_engines": [["wikidata", "Suspended: access denied"]],
    }
    monkeypatch.setattr(wst.httpx, "Client", lambda *a, **k: _SearxStubClient(payload))
    host = _Host()
    results = host._search_searxng("q", 5, "http://localhost:8080")
    assert len(results) == 1
    assert results[0]["title"] == "T"
    assert results[0]["snippet"] == "snip"


# ── Naver browser fallback: gating + structured mapping + wiring ─────────

def test_should_try_naver_korean_by_default(monkeypatch):
    """Default policy: the browser Naver fallback fires on Hangul queries only."""
    monkeypatch.delenv("ASICODE_NAVER_FALLBACK", raising=False)
    host = _Host()
    # A Hangul-containing string is required here — this asserts Hangul detection,
    # so it cannot be an English string. Uses a neutral generic phrase.
    assert host._should_try_naver("서울의 관광지")          # Hangul → yes
    assert not host._should_try_naver("python asyncio")   # Latin-only → no


def test_should_try_naver_env_modes(monkeypatch):
    host = _Host()
    monkeypatch.setenv("ASICODE_NAVER_FALLBACK", "off")
    assert not host._should_try_naver("서울의 관광지")       # opt-out wins over Hangul
    monkeypatch.setenv("ASICODE_NAVER_FALLBACK", "always")
    assert host._should_try_naver("python asyncio")       # always, even Latin
    monkeypatch.setenv("ASICODE_NAVER_FALLBACK", "korean")
    assert host._should_try_naver("서울") and not host._should_try_naver("tokyo")


def test_search_naver_browser_maps_and_filters(monkeypatch):
    """Maps JS output → {title,url,snippet}, drops junk (empty title / non-http
    url / non-dict), and honours max_results."""
    raw = [
        {"title": "Seoul Tourist Attractions Guide", "url": "https://example.com/seoul", "snippet": "Top places to visit in Seoul"},
        {"title": "", "url": "https://example.com/no-title", "snippet": "drop: empty title"},
        {"title": "bad scheme", "url": "ftp://bad", "snippet": "drop: non-http url"},
        "not-a-dict",
        {"title": "second good", "url": "https://example.com/2", "snippet": "keep"},
        {"title": "third good", "url": "https://example.com/3", "snippet": "beyond cap"},
    ]
    host = _Host()
    host._render_and_eval = lambda url, js, **k: raw  # stub the browser primitive
    out = host._search_naver_browser("Seoul tourist attractions", max_results=2)
    assert [r["title"] for r in out] == ["Seoul Tourist Attractions Guide", "second good"]  # cap=2, junk dropped
    assert out[0]["url"] == "https://example.com/seoul"
    assert out[0]["snippet"] == "Top places to visit in Seoul"


def test_search_naver_browser_targets_web_vertical(monkeypatch):
    captured = {}
    host = _Host()

    def _stub(url, js, **k):
        captured["url"] = url
        return []

    host._render_and_eval = _stub
    host._search_naver_browser("Seoul weather", 5)
    assert "where=web" in captured["url"]
    assert "search.naver.com" in captured["url"]


def test_search_web_falls_back_to_naver_when_others_empty(monkeypatch):
    """SearXNG/DDG yield nothing → the browser Naver backend runs last and its
    results are returned and formatted. (ASICODE_NAVER_FALLBACK=always makes the
    gate language-independent, so the query need not be Hangul here.)"""
    monkeypatch.setenv("ASICODE_NAVER_FALLBACK", "always")
    monkeypatch.delenv("SEARXNG_BASE_URL", raising=False)
    monkeypatch.delenv("BRAVE_API_KEY", raising=False)
    host = _Host()
    host._has_docker_or_colima = lambda: False          # no SearXNG auto-setup backend
    host._search_duckduckgo = lambda q, n: []           # DDG: genuine empty
    naver_calls = {"n": 0}

    def _naver(q, n):
        naver_calls["n"] += 1
        return [{"title": "Seoul attractions result", "url": "https://example.com/n", "snippet": "a snippet"}]

    host._search_naver_browser = _naver
    res = host._tool_search_web({"query": "Seoul tourist attractions"})
    assert res["ok"]
    assert naver_calls["n"] == 1                          # Naver was reached
    assert "Seoul attractions result" in res["content"]
    assert res["metadata"]["result_count"] == 1


def test_search_web_skips_naver_for_latin_query(monkeypatch):
    """Default gating: a Latin-only query must NOT spin up the browser backend
    even when every other backend comes back empty."""
    monkeypatch.delenv("ASICODE_NAVER_FALLBACK", raising=False)
    monkeypatch.delenv("SEARXNG_BASE_URL", raising=False)
    monkeypatch.delenv("BRAVE_API_KEY", raising=False)
    host = _Host()
    host._has_docker_or_colima = lambda: False
    host._search_duckduckgo = lambda q, n: []

    def _boom(q, n):  # must never be called for a Latin query
        raise AssertionError("Naver browser backend should not run for Latin query")

    host._search_naver_browser = _boom
    res = host._tool_search_web({"query": "python asyncio"})
    assert res["metadata"]["result_count"] == 0          # no results, no browser spin


# ── _http_request_with_retry: status-code retry ─────────────────────────

class _SequenceClient:
    """Returns canned responses in order; records how many requests were made."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0

    def get(self, url, params=None, headers=None):
        self.calls += 1
        return self._responses.pop(0)

    def post(self, url, data=None, headers=None):
        self.calls += 1
        return self._responses.pop(0)


def _resp(status, **kw):
    return httpx.Response(status, request=httpx.Request("GET", "https://x/"), **kw)


def test_retry_after_seconds_parses_delta_capped_and_absent():
    assert wst._retry_after_seconds(_resp(429, headers={"retry-after": "5"}), 1.5) == 5.0
    # capped at 30s even if the server asks for more
    assert wst._retry_after_seconds(_resp(429, headers={"retry-after": "999"}), 1.5) == 30.0
    # absent header falls back to the caller default
    assert wst._retry_after_seconds(_resp(429), 1.5) == 1.5
    # an HTTP-date in the past floors to 0 (no negative sleep)
    past = wst._retry_after_seconds(
        _resp(429, headers={"retry-after": "Wed, 21 Oct 2015 07:28:00 GMT"}), 1.5
    )
    assert past == 0.0


def test_http_retry_retries_on_429_then_succeeds(monkeypatch):
    """A transient 429 must be retried and the later success returned."""
    sleeps = []
    monkeypatch.setattr(wst.time, "sleep", lambda s: sleeps.append(s))
    client = _SequenceClient([
        _resp(429, headers={"retry-after": "0"}),  # rate-limited, retry immediately
        _resp(200, text="ok"),                      # success on retry
    ])
    resp = wst.WebSearchToolsMixin._http_request_with_retry(client, "GET", "https://x/")
    assert resp.status_code == 200
    assert client.calls == 2
    assert sleeps  # did back off once


def test_http_retry_returns_bad_status_after_exhausting_retries(monkeypatch):
    """A persistent 503 is retried, then the final bad status is returned so the
    caller's raise_for_status() surfaces the precise error."""
    monkeypatch.setattr(wst.time, "sleep", lambda s: None)
    client = _SequenceClient([_resp(503), _resp(503)])
    resp = wst.WebSearchToolsMixin._http_request_with_retry(client, "GET", "https://x/", retries=2)
    assert resp.status_code == 503
    assert client.calls == 2


def test_http_retry_non_retryable_status_returned_immediately(monkeypatch):
    """A 404 is not retryable: returned on the first attempt with no back off."""
    monkeypatch.setattr(wst.time, "sleep", lambda s: None)
    client = _SequenceClient([_resp(404)])
    resp = wst.WebSearchToolsMixin._http_request_with_retry(client, "GET", "https://x/", retries=3)
    assert resp.status_code == 404
    assert client.calls == 1


def test_http_retry_honours_retry_after_for_wait(monkeypatch):
    """The Retry-After hint (not the default backoff) is used as the wait."""
    sleeps = []
    monkeypatch.setattr(wst.time, "sleep", lambda s: sleeps.append(s))
    client = _SequenceClient([
        _resp(429, headers={"retry-after": "7"}),
        _resp(200, text="ok"),
    ])
    wst.WebSearchToolsMixin._http_request_with_retry(client, "GET", "https://x/", backoff=1.5)
    assert sleeps == [7.0]


# ── connect-error fail-fast + session circuit breaker ────────────────────

class _RaisingClient:
    """httpx.Client stub whose get/post raise a fixed exception; counts calls."""

    def __init__(self, exc):
        self._exc = exc
        self.calls = 0

    def get(self, url, params=None, headers=None):
        self.calls += 1
        raise self._exc

    def post(self, url, data=None, headers=None):
        self.calls += 1
        raise self._exc


def test_http_retry_does_not_retry_connect_errors(monkeypatch):
    """A ConnectTimeout fails fast — no retry, no backoff sleep (an unreachable
    host would just re-pay the connect timeout)."""
    sleeps = []
    monkeypatch.setattr(wst.time, "sleep", lambda s: sleeps.append(s))
    client = _RaisingClient(httpx.ConnectTimeout("timed out"))
    with pytest.raises(httpx.ConnectTimeout):
        wst.WebSearchToolsMixin._http_request_with_retry(client, "GET", "https://x/", retries=3)
    assert client.calls == 1   # single attempt, no retry
    assert sleeps == []        # no backoff


def test_http_retry_still_retries_read_timeout(monkeypatch):
    """A non-connect transient error (ReadTimeout = slow server) is still retried."""
    monkeypatch.setattr(wst.time, "sleep", lambda s: None)
    calls = {"n": 0}

    class _C:
        def get(self, url, params=None, headers=None):
            calls["n"] += 1
            if calls["n"] == 1:
                raise httpx.ReadTimeout("slow")
            return _resp(200, text="ok")

    resp = wst.WebSearchToolsMixin._http_request_with_retry(_C(), "GET", "https://x/", retries=2)
    assert resp.status_code == 200
    assert calls["n"] == 2     # retried once


def test_connect_failure_trips_breaker_and_skips_backend(monkeypatch):
    """A backend that connect-fails is tripped into cooldown and skipped on the
    NEXT search — it does not re-pay its connect timeout every time.

    (DDG is used as the vehicle for breaker mechanics, so it must be opted in —
    it is no longer part of the default chain.)"""
    monkeypatch.setenv("ASICODE_DDG_FALLBACK", "on")
    monkeypatch.delenv("ASICODE_NAVER_FALLBACK", raising=False)
    monkeypatch.delenv("SEARXNG_BASE_URL", raising=False)
    monkeypatch.delenv("BRAVE_API_KEY", raising=False)
    host = _Host()
    host._has_docker_or_colima = lambda: False   # backends = [DuckDuckGo] (Latin → no Naver)
    ddg_calls = {"n": 0}

    def _ddg(q, n):
        ddg_calls["n"] += 1
        raise httpx.ConnectTimeout("timed out")

    host._search_duckduckgo = _ddg
    host._tool_search_web({"query": "python asyncio"})       # 1st: connect-fails → trips
    assert ddg_calls["n"] == 1
    assert host._backend_in_cooldown("DuckDuckGo")
    host._tool_search_web({"query": "python asyncio"})       # 2nd: skipped
    assert ddg_calls["n"] == 1                                # NOT called again


def test_read_error_does_not_trip_breaker(monkeypatch):
    """An ordinary RuntimeError must NOT sideline the backend — only unreachable
    (connect) failures and bot-detection walls (``_BlockWallError``) should."""
    monkeypatch.setenv("ASICODE_DDG_FALLBACK", "on")
    monkeypatch.delenv("ASICODE_NAVER_FALLBACK", raising=False)
    monkeypatch.delenv("SEARXNG_BASE_URL", raising=False)
    monkeypatch.delenv("BRAVE_API_KEY", raising=False)
    host = _Host()
    host._has_docker_or_colima = lambda: False
    host._search_duckduckgo = lambda q, n: (_ for _ in ()).throw(RuntimeError("anomaly"))
    host._tool_search_web({"query": "python asyncio"})
    assert not host._backend_in_cooldown("DuckDuckGo")        # not tripped


def test_backend_cooldown_expires(monkeypatch):
    """An expired cooldown entry is evicted so the backend is retried."""
    host = _Host()
    wst.WebSearchToolsMixin._backend_cooldown["DuckDuckGo"] = 0.0  # deadline in the past
    assert not host._backend_in_cooldown("DuckDuckGo")            # expired → False
    assert "DuckDuckGo" not in wst.WebSearchToolsMixin._backend_cooldown  # and evicted
def test_parallel_merge_probes_breaker_once_per_backend(monkeypatch):
    """Regression: the parallel-merge path (_run_tier_parallel) used to call
    _backend_in_cooldown TWICE per backend — once in the runnable listcomp and once
    in the logging loop. Each call acquires the cooldown lock AND re-runs its lazy
    eviction side-effect, so a just-expired entry was deleted then looked up again.
    The single-pass rewrite probes each backend exactly once."""
    host = _Host()
    # Sideline one of three backends so both the skip-log branch and the runnable
    # partition are exercised in the same call.
    wst.WebSearchToolsMixin._backend_cooldown["B"] = time.monotonic() + 3600

    probes = {"n": 0}
    orig = host._backend_in_cooldown

    def _counting(name):
        probes["n"] += 1
        return orig(name)

    monkeypatch.setattr(host, "_backend_in_cooldown", _counting)

    backends = [
        ("A", lambda: [{"t": "1"}]),
        ("B", lambda: [{"t": "2"}]),   # in cooldown → skipped, never submitted
        ("C", lambda: [{"t": "3"}]),
    ]
    collected, _errors, _connect_failed = host._run_tier_parallel(backends, deadline=2.0)

    assert probes["n"] == 3                       # exactly once per backend (was 6)
    assert {name for name, _ in collected} == {"A", "C"}   # B was skipped


# ── DuckDuckGo anomaly detection + close() flush ────────────────────────

class _DDGStubClient:
    """Stub httpx.Client for _search_duckduckgo (handles POST only)."""

    def __init__(self, body: str, status: int = 200):
        self._body = body
        self._status = status

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def post(self, url, data=None, headers=None):
        return httpx.Response(
            self._status,
            request=httpx.Request("POST", url),
            headers={"content-type": "text/html; charset=utf-8"},
            text=self._body,
        )


def test_body_is_block_wall_heuristic():
    """The detector is chain-common: it must recognise the wall each engine
    actually serves, all observed live on 2026-07-19 from one IP."""
    # DuckDuckGo anomaly / rate-limit
    assert wst._body_is_block_wall("<h1>we detected an anomaly in your requests</h1>")
    assert wst._body_is_block_wall("HTTP 429 rate limit exceeded")
    assert wst._body_is_block_wall("redirect to duckduckgo.com/anomaly/abc")
    # Mojeek: HTTP 200 + challenge
    assert wst._body_is_block_wall("Verification required. Please complete the challenge to continue.")
    # Marginalia: HTTP 200 + throttle interstitial
    assert wst._body_is_block_wall("Wait A Moment — seeing a lot of fairly aggressive bot activity")
    # Google / Cloudflare style
    assert wst._body_is_block_wall("Our systems have detected unusual traffic")
    assert wst._body_is_block_wall("<title>Just a moment...</title>")
    # genuine result page / empty → False
    assert not wst._body_is_block_wall('<a class="result__a">Title</a>')
    assert not wst._body_is_block_wall("")
    assert not wst._body_is_block_wall("just some ordinary text about ducks")


def test_block_wall_scan_is_bounded():
    """Only a bounded prefix is scanned, so the check stays O(1) in page size."""
    body = ("x" * (wst._BLOCK_WALL_SCAN_CHARS + 100)) + "captcha"
    assert not wst._body_is_block_wall(body)          # marker past the window
    assert wst._body_is_block_wall("captcha" + body)  # marker inside it


def test_guard_block_wall_ignores_populated_results():
    """THE false-positive guard: 'captcha' / 'rate limit' are ordinary words that
    legitimately appear in real results ABOUT those topics. A populated result set
    must never be reclassified as a wall, whatever the body says."""
    real = [{"title": "How CAPTCHA works", "url": "https://ex.com/a", "snippet": "rate limit basics"}]
    # No raise: results are present, so the markers are just page content.
    wst.WebSearchToolsMixin._guard_block_wall("Startpage", "captcha rate limit too many requests", real)


def test_guard_block_wall_empty_and_clean_is_a_genuine_miss():
    """Zero results with no wall markers is an honest 'nothing matched' — it must
    stay an empty list, not become an error."""
    wst.WebSearchToolsMixin._guard_block_wall("Startpage", "<html>No results for xyzzy</html>", [])


def test_guard_block_wall_empty_and_walled_raises():
    """Zero results + wall markers = infrastructure failure, surfaced by name."""
    with pytest.raises(RuntimeError, match="Startpage"):
        wst.WebSearchToolsMixin._guard_block_wall("Startpage", "Verification required", [])


# Visible text of DuckDuckGo's LIVE challenge page, captured verbatim 2026-07-19.
# Verbatim on purpose: the original marker list was written from paraphrase and
# matched this page only by an incidental ``duckduckgo.com/anomaly`` URL — the
# body contains neither "captcha" nor "complete the challenge".
_DDG_LIVE_CAPTCHA = (
    "DuckDuckGo Unfortunately, bots use DuckDuckGo too. Please complete the "
    "following challenge to confirm this search was made by a human. Select all "
    "squares containing a duck: Submit Images not loading? Please email the "
    "following code to: error-lite@duckduckgo.com Code: d4cd0dabcf4caa22a"
)


def test_live_ddg_captcha_matches_on_wording_not_just_url():
    """The live challenge page must be caught by its WORDING, not only by the
    incidental anomaly-URL reference — otherwise one copy edit at DDG silently
    reopens the 'HTTP 200 + zero results looks like a genuine miss' trap."""
    assert wst._body_is_block_wall(_DDG_LIVE_CAPTCHA)
    # And specifically: not merely because a URL happened to appear in the markup.
    assert "duckduckgo.com/anomaly" not in _DDG_LIVE_CAPTCHA.lower()
    matched = [m for m in wst._BLOCK_WALL_MARKERS if m in _DDG_LIVE_CAPTCHA.lower()]
    assert len(matched) >= 2, f"only {matched} matched — too few threads holding"


def test_guard_block_wall_flags_non_200_success_status():
    """Structural signal: a 2xx that is not 200 means the engine acknowledged the
    request without running the search (DDG answers its challenge with HTTP 202).
    This must fire even when the body carries no known marker at all."""
    with pytest.raises(RuntimeError, match="202"):
        wst.WebSearchToolsMixin._guard_block_wall("DuckDuckGo", "<html>totally novel wall</html>", [], status=202)


def test_guard_block_wall_status_ignored_when_results_present():
    """A populated result set is never reclassified, whatever the status."""
    hits = [{"title": "t", "url": "u", "snippet": "s"}]
    wst.WebSearchToolsMixin._guard_block_wall("DuckDuckGo", "body", hits, status=202)


# ── web_fetch: HTTP 200 + bot challenge ──────────────────────────────────────
#
# Extracted text (i.e. what _tool_web_fetch actually inspects, after
# _HTMLTextExtractor) of three pages captured live on 2026-08-05. Each answered
# HTTP 200. Kept verbatim for the same reason as _DDG_LIVE_CAPTCHA: every one of
# these was MISSED by the marker list as it stood that morning — "checking your
# browser" does not match "verifying your browser".
_LIVE_CHALLENGE_PAGES = {
    # www.reddit.com — the whole page extracts to just its title.
    "reddit-www": "Reddit - Please wait for verification",
    # redlib.privacyredirect.com — Anubis proof-of-work interstitial.
    "redlib-anubis": (
        "Making sure you're not a bot!\nMaking sure you're not a bot!\nLoading...\n"
        "Please wait a moment while we ensure the security of your connection.\n"
        "Protected by Anubis from Techaro. Made with ❤️ by Xe Iaso."
    ),
    # safereddit.com — same Anubis stack, different wording.
    "safereddit": (
        "Verifying your browser…\nVerifying your browser…\n"
        "You are seeing this because the administrator of this website has set up "
        "Anubis to protect the server against the scourge of AI companies "
        "aggressively scraping websites."
    ),
}


@pytest.mark.parametrize("name", sorted(_LIVE_CHALLENGE_PAGES))
def test_fetch_challenge_page_catches_live_200_walls(name):
    """Each live page must be caught by its own wording.

    These are the pages that make web_fetch dangerous rather than merely broken:
    HTTP 200, so an unguarded fetch formats the interstitial exactly like a
    successful read and the caller mistakes a block for the site's content.
    """
    assert wst._fetch_is_challenge_page(_LIVE_CHALLENGE_PAGES[name])


def test_fetch_challenge_page_ignores_real_content():
    """The real old.reddit.com thread reached through the same fetch must pass."""
    real = "Introducing Claude Opus 5 : ClaudeAI\njump to content\nmy subreddits\n" + ("comment body. " * 500)
    assert not wst._fetch_is_challenge_page(real)
    assert not wst._fetch_is_challenge_page("")
    assert not wst._fetch_is_challenge_page("   \n  ")


def test_fetch_challenge_page_needs_both_signals():
    """Length alone and wording alone must each be insufficient.

    This is the property that lets web_fetch consult the markers without the
    "zero results parsed" gate that _body_is_block_wall's contract requires: a
    long page keeps its content even if it discusses bot checks, and a short page
    is not condemned merely for being short.
    """
    # Wording present, but on a page long enough to be a real article about it.
    article = "Why sites are checking your browser: a deep dive. " * 200
    assert len(article) > wst._FETCH_CHALLENGE_MAX_TEXT
    assert wst._fetch_is_challenge_page(article) is False
    # Short page, no challenge wording — a legitimately terse page.
    assert not wst._fetch_is_challenge_page("404 Not Found\nThe page you requested does not exist.")


def test_loose_markers_stay_out_of_the_fetch_path():
    """"captcha" / "rate limit" must never reach the ungated web_fetch check.

    They are ordinary words in pages ABOUT those topics; _body_is_block_wall is
    only safe with them because the search path gates on zero parsed results.
    """
    for loose in ("captcha", "rate limit", "too many requests", "unusual traffic"):
        assert loose in wst._BLOCK_WALL_MARKERS
        assert loose not in wst._CHALLENGE_PAGE_MARKERS
        assert not wst._fetch_is_challenge_page(f"Our API returns 429 when you hit the {loose}.")


def test_challenge_markers_remain_a_subset_of_the_wall_markers():
    """Splitting the list must not drop a phrase from the search-path detector."""
    assert set(wst._CHALLENGE_PAGE_MARKERS) <= set(wst._BLOCK_WALL_MARKERS)


# ── web_fetch: Reddit host rewrite ───────────────────────────────────────────


@pytest.mark.parametrize(
    "host",
    ["reddit.com", "www.reddit.com", "np.reddit.com", "new.reddit.com", "m.reddit.com", "amp.reddit.com"],
)
def test_rewrite_reddit_url_covers_every_challenge_host(host):
    """The bare apex 302s to www and lands on the same interstitial, so matching
    only "www.reddit.com" left the commonest hand-typed form broken."""
    out = wst._rewrite_reddit_url(f"https://{host}/r/ClaudeAI/comments/abc/title/")
    assert out == "https://old.reddit.com/r/ClaudeAI/comments/abc/title/"


def test_rewrite_reddit_url_preserves_query_and_fragment():
    out = wst._rewrite_reddit_url("https://www.reddit.com/r/x/search?q=opus&restrict_sr=1#top")
    assert out == "https://old.reddit.com/r/x/search?q=opus&restrict_sr=1#top"


def test_fetch_headers_carry_the_full_browser_fingerprint():
    """A partial browser header set is itself a bot signature.

    Measured 2026-08-05 against old.reddit.com, 4 requests per variant: UA alone,
    UA+Accept, and UA+Accept-Language each returned 200 (4/4), while
    UA+Accept+Accept-Language — exactly what web_fetch used to send — returned 403
    (4/4). Adding the Sec-Fetch block restored 200 (4/4). Dropping any member here
    re-creates that fingerprint, so the set is asserted whole.
    """
    required = {
        "Accept", "Accept-Language", "sec-ch-ua", "sec-ch-ua-mobile", "sec-ch-ua-platform",
        "Sec-Fetch-Dest", "Sec-Fetch-Mode", "Sec-Fetch-Site", "Sec-Fetch-User",
        "Upgrade-Insecure-Requests",
    }
    assert required <= set(wst._BROWSER_FETCH_HEADERS)
    # User-Agent stays out of the constant — it is shared with the search backends.
    assert "User-Agent" not in wst._BROWSER_FETCH_HEADERS


def test_client_hint_version_matches_the_user_agent():
    """sec-ch-ua must not drift from _BROWSER_UA: a Chrome UA advertising one
    major while the client hint advertises another is a contradiction no real
    browser produces, which is the very signature these headers exist to avoid."""
    ua_major = re.search(r"Chrome/(\d+)", wst._BROWSER_UA).group(1)
    hint_majors = set(re.findall(r'"Chromium";v="(\d+)"|"Google Chrome";v="(\d+)"',
                                 wst._BROWSER_FETCH_HEADERS["sec-ch-ua"]))
    hint_majors = {v for pair in hint_majors for v in pair if v}
    assert hint_majors == {ua_major}, f"UA says Chrome/{ua_major}, sec-ch-ua says {hint_majors}"


def test_rewrite_reddit_url_leaves_other_urls_alone():
    """Host-based, not substring: a URL that merely mentions the string is not a
    Reddit fetch, and old.reddit.com must not be rewritten onto itself."""
    for untouched in (
        "https://example.com/?ref=www.reddit.com",
        "https://notreddit.com/r/x",
        "https://old.reddit.com/r/x",
        "https://i.redd.it/abc.png",
        "not a url at all",
    ):
        assert wst._rewrite_reddit_url(untouched) == untouched


def test_guard_block_wall_plain_200_miss_still_passes():
    """HTTP 200 + zero results + no markers stays a genuine miss."""
    wst.WebSearchToolsMixin._guard_block_wall("Startpage", "<html>no matches</html>", [], status=200)


def test_ddg_search_raises_on_anomaly_page(monkeypatch):
    """A 200 anomaly interstitial (0 results + markers) must raise so the
    fallback chain records a meaningful error and tries the next backend.

    The detector is now the chain-common ``_guard_block_wall``, so the message is
    generic — it must still name the engine that produced the wall."""
    anomaly = (
        "<html><body><h2>If this error persists, we have detected "
        "an anomaly in your requests.</h2></body></html>"
    )
    monkeypatch.setattr(wst.httpx, "Client", lambda *a, **k: _DDGStubClient(anomaly))
    host = _Host()

    with pytest.raises(RuntimeError, match="DuckDuckGo"):
        host._search_duckduckgo("anything", 5)


def test_ddg_search_returns_results_on_normal_page(monkeypatch):
    page = (
        '<a class="result__a" href="https://a.example.com">A Title</a>'
        '<a class="result__snippet" href="https://a.example.com">A snippet</a>'
    )
    monkeypatch.setattr(wst.httpx, "Client", lambda *a, **k: _DDGStubClient(page))
    host = _Host()
    results = host._search_duckduckgo("query", 5)
    assert len(results) == 1
    assert results[0]["title"] == "A Title"
    assert results[0]["snippet"] == "A snippet"


def test_ddg_search_flushes_trailing_title_only_result(monkeypatch):
    """_search_duckduckgo must call parser.close() so a trailing title-only
    result (no snippet endtag) is flushed instead of dropped."""
    page = '<a class="result__a" href="https://only.example.com">Only Title</a>'
    monkeypatch.setattr(wst.httpx, "Client", lambda *a, **k: _DDGStubClient(page))
    host = _Host()
    results = host._search_duckduckgo("query", 5)
    assert len(results) == 1
    assert results[0]["title"] == "Only Title"


def test_web_fetch_rejects_oversized_content_length(monkeypatch):
    """A Content-Length beyond _WEB_FETCH_MAX_BYTES is refused before the body is
    fully decoded, preventing an OOM on a huge binary URL."""
    huge = httpx.Response(
        200,
        request=httpx.Request("GET", "https://x/big.bin"),
        headers={
            "content-type": "text/plain",
            "content-length": str(500 * 1024 * 1024),
        },
        content=b"x" * 10,
    )
    _stub_fetch(monkeypatch, huge)
    host = _Host()
    res = host._tool_web_fetch({"url": "https://x/big.bin"})
    assert not res["ok"]
    err = res["error"].lower()
    assert "refusing" in err or "limit" in err


def test_web_fetch_allows_normal_content_length(monkeypatch):
    """A normal-sized Content-Length is fetched normally (guard never trips)."""
    ok = httpx.Response(
        200,
        request=httpx.Request("GET", "https://x/page"),
        headers={"content-type": "text/html; charset=utf-8", "content-length": "42"},
        text="<p>hello world</p>",
    )
    _stub_fetch(monkeypatch, ok)
    host = _Host()
    res = host._tool_web_fetch({"url": "https://x/page"})
    assert res["ok"], res
    assert "hello world" in res["content"]
def test_web_fetch_rejects_streaming_exceeding_byte_cap(monkeypatch):
    """A chunked response (no Content-Length) that exceeds the byte cap during
    streaming must be refused, preventing OOM on unbounded chunked responses."""
    # httpx.Response with no content-length → streaming byte cap is the only guard
    huge_body = b"x" * (wst._WEB_FETCH_MAX_BYTES + 1)
    resp = httpx.Response(
        200,
        request=httpx.Request("GET", "https://x/streaming.bin"),
        headers={"content-type": "text/plain"},
        content=huge_body,
    )
    _stub_fetch(monkeypatch, resp)
    host = _Host()
    res = host._tool_web_fetch({"url": "https://x/streaming.bin"})
    assert not res["ok"]
    err = res["error"].lower()
    assert "exceeded" in err or "limit" in err


# ── charset sniffing (_sniff_html_encoding) ─────────────────────────────


def test_sniff_html_encoding_bom_utf8():
    assert wst._sniff_html_encoding(b"\xef\xbb\xbf<html>") == "utf-8-sig"


def test_sniff_html_encoding_bom_utf16():
    assert wst._sniff_html_encoding(b"\xff\xfe<html>") == "utf-16"
    assert wst._sniff_html_encoding(b"\xfe\xff<html>") == "utf-16"


def test_sniff_html_encoding_meta_charset_attribute():
    body = b'<html><head><meta charset="euc-kr"></head><body></body></html>'
    assert wst._sniff_html_encoding(body).lower() == "euc-kr"


def test_sniff_html_encoding_meta_http_equiv():
    body = (
        b'<html><head><meta http-equiv="Content-Type" '
        b'content="text/html; charset=cp949"></head></html>'
    )
    assert wst._sniff_html_encoding(body).lower() == "cp949"


def test_sniff_html_encoding_none_when_absent():
    body = b'<html><head><title>no charset</title></head></html>'
    assert wst._sniff_html_encoding(body) is None


def test_sniff_html_encoding_reads_only_first_1kb():
    # A <meta charset> placed AFTER the first 1024 bytes must be ignored
    # (HTML5 prescan window), returning None.
    body = b"x" * 1100 + b'<meta charset="euc-kr">'
    assert wst._sniff_html_encoding(body) is None


def test_sniff_html_encoding_scans_non_ascii_body_safely():
    # The prescan decodes the head as ASCII-with-ignore, so a multi-byte body
    # whose <meta> tag is still ASCII-structured is found without choking.
    korean = "안녕".encode("euc-kr")
    body = b'<html><head><meta charset="euc-kr"></head><body>' + korean + b"</body></html>"
    assert wst._sniff_html_encoding(body).lower() == "euc-kr"


# ── web_fetch charset / binary regressions (end-to-end, HTTP stubbed) ───


def test_web_fetch_meta_charset_euckr_not_mangled(monkeypatch):
    """A page declaring charset only via <meta charset="euc-kr"> (no HTTP header
    charset) must be decoded via the body prescan, not UTF-8 mangled.

    Regression: httpx's Response.encoding returns only the header charset, so
    Korean legacy pages were replace-decoded into mojibake."""
    host = _Host()
    korean = "안녕하세요 세계"
    body = (
        f'<html><head><meta charset="euc-kr"></head>'
        f'<body><p>{korean}</p></body></html>'
    ).encode("euc-kr")
    # Content-Type deliberately carries NO charset → header path yields None.
    resp = httpx.Response(
        200, request=httpx.Request("GET", "https://x/"),
        headers={"content-type": "text/html"}, content=body,
    )
    _stub_fetch(monkeypatch, resp)
    res = host._tool_web_fetch({"url": "https://example.com"})
    assert res["ok"], res.get("error")
    assert korean in res["content"]


def test_web_fetch_header_charset_overrides_meta(monkeypatch):
    """When both header and <meta> declare a charset, the HTTP header wins."""
    host = _Host()
    body = b'<html><head><meta charset="euc-kr"></head><body>plain ascii</body></html>'
    resp = httpx.Response(
        200, request=httpx.Request("GET", "https://x/"),
        headers={"content-type": "text/html; charset=utf-8"}, content=body,
    )
    _stub_fetch(monkeypatch, resp)
    res = host._tool_web_fetch({"url": "https://example.com"})
    assert res["ok"], res.get("error")
    assert "plain ascii" in res["content"]


def test_web_fetch_rejects_binary_pdf(monkeypatch):
    """A PDF must be rejected with a clean error, not decode-mangled into context."""
    host = _Host()
    resp = httpx.Response(
        200, request=httpx.Request("GET", "https://x/doc.pdf"),
        headers={"content-type": "application/pdf"}, content=b"%PDF-1.4\n%\xe2\xe3\xcf\xd3",
    )
    _stub_fetch(monkeypatch, resp)
    res = host._tool_web_fetch({"url": "https://example.com/doc.pdf"})
    assert not res["ok"]
    assert "binary" in res["error"].lower()
    assert "browser_action" in res["error"]


def test_web_fetch_rejects_binary_image(monkeypatch):
    """An image Content-Type is rejected as binary too."""
    host = _Host()
    resp = httpx.Response(
        200, request=httpx.Request("GET", "https://x/img.png"),
        headers={"content-type": "image/png"}, content=b"\x89PNG\r\n\x1a\n",
    )
    _stub_fetch(monkeypatch, resp)
    res = host._tool_web_fetch({"url": "https://example.com/img.png"})
    assert not res["ok"]
    assert "binary" in res["error"].lower()


# ── SearXNG ConnectTimeout routing ──────────────────────────────────────


def test_searxng_connect_timeout_triggers_install_branch(monkeypatch):
    """A ConnectTimeout (packet-drop / host not responding) must enter the
    SearXNG install/start branch — not the generic fallback — because
    ConnectTimeout is NOT a subclass of ConnectError in httpx, so it has to be
    listed explicitly in the except tuple."""
    monkeypatch.setenv("SEARXNG_BASE_URL", "http://localhost:8080")
    host = _Host()

    handle_calls = {"n": 0}

    def _raise_connect_timeout(*a, **k):
        raise httpx.ConnectTimeout("timed out connecting")

    def _handle(*a, **k):
        handle_calls["n"] += 1
        return "install failed"  # non-None → skip the post-install retry

    monkeypatch.setattr(host, "_search_searxng", _raise_connect_timeout)
    monkeypatch.setattr(host, "_handle_searxng_connect_error", _handle)
    monkeypatch.setattr(host, "_search_duckduckgo", lambda *a, **k: [])
    res = host._tool_search_web({"query": "test"})
    assert handle_calls["n"] == 1  # ConnectTimeout reached the SearXNG branch
    assert res["ok"]  # search still returns (no crash)


class _FetchRetryClient:
    """Returns canned responses in order for web_fetch retry testing.

    ``stream(method, url)`` returns a ``_FakeStreamResponse`` around each
    response in sequence. Records how many ``stream()`` calls were made.
    """

    def __init__(self, responses: list[httpx.Response]):
        self._responses = list(responses)
        self.calls = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def stream(self, method, url, **kw):
        self.calls += 1
        return _FakeStreamResponse(self._responses.pop(0))


def test_web_fetch_retries_on_429_then_succeeds(monkeypatch):
    """A 429 response from web_fetch must be retried (matching the search
    backends' retry policy) and succeed on the second attempt."""
    sleeps = []
    monkeypatch.setattr(wst.time, "sleep", lambda s: sleeps.append(s))

    client = _FetchRetryClient([
        httpx.Response(
            429,
            request=httpx.Request("GET", "https://x/rate"),
            headers={"retry-after": "0", "content-type": "text/plain"},
        ),
        httpx.Response(
            200,
            request=httpx.Request("GET", "https://x/rate"),
            headers={"content-type": "text/plain; charset=utf-8"},
            text="finally ok",
        ),
    ])
    monkeypatch.setattr(wst.httpx, "Client", lambda *a, **k: client)
    host = _Host()
    res = host._tool_web_fetch({"url": "https://x/rate"})
    assert res["ok"], res.get("error")
    assert "finally ok" in res["content"]
    assert client.calls == 2
    assert sleeps  # did back off


def test_web_fetch_retries_on_transient_error_then_succeeds(monkeypatch):
    """A transient network error (e.g. ConnectError) in web_fetch must be
    retried and succeed on the second attempt."""
    sleeps = []
    monkeypatch.setattr(wst.time, "sleep", lambda s: sleeps.append(s))

    import httpx as _real_httpx

    ok_resp = httpx.Response(
        200,
        request=httpx.Request("GET", "https://x/unstable"),
        headers={"content-type": "text/plain; charset=utf-8"},
        text="recovered",
    )

    attempts = [0]

    class _RetryClient:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def stream(self, method, url, **kw):
            attempts[0] += 1
            if attempts[0] == 1:
                raise _real_httpx.ConnectError("first attempt failed")
            # second attempt succeeds
            return _FakeStreamResponse(ok_resp)

    monkeypatch.setattr(wst.httpx, "Client", lambda *a, **k: _RetryClient())
    host = _Host()
    res = host._tool_web_fetch({"url": "https://x/unstable"})
    assert res["ok"], res.get("error")
    assert "recovered" in res["content"]
    assert attempts[0] == 2
    assert sleeps


def test_search_web_prefers_brave_over_ddg_when_key_set(monkeypatch):
    """When BRAVE_API_KEY is set and SearXNG is unavailable, the stable keyed
    Brave API is tried BEFORE the rate-limit/anomaly-prone DuckDuckGo scraper
    (see _guard_block_wall). Previously DDG was always first, so a flaky
    scraper burned a request before the reliable backend was ever consulted.

    (Startpage now precedes both; the autouse fixture makes it return empty so
    this asserts the Brave-vs-DDG relative order it was written to protect.)"""
    monkeypatch.setenv("BRAVE_API_KEY", "fake-key")
    monkeypatch.delenv("SEARXNG_BASE_URL", raising=False)
    monkeypatch.setattr(_Host, "_has_docker_or_colima", lambda self: False)

    order: list[str] = []
    monkeypatch.setattr(
        _Host, "_search_brave",
        lambda self, q, m, k: order.append("brave") or [{"title": "t", "url": "u", "snippet": "s"}],
    )
    monkeypatch.setattr(
        _Host, "_search_duckduckgo", lambda self, q, m: order.append("ddg") or []
    )

    host = _Host()
    res = host._tool_search_web({"query": "test"})
    assert res["ok"], res.get("error")
    assert order == ["brave"], (
        f"Brave must be tried first when BRAVE_API_KEY is set; got order={order}"
    )


def test_search_web_excludes_ddg_from_default_chain(monkeypatch):
    """DuckDuckGo is OPT-IN as of 2026-07-19 and must NOT run by default.

    It discriminates on TLS fingerprint (measured: httpx 2/6 while a browser-TLS
    client scored 12/12 from the same IP in the same window), and every failed
    attempt feeds the bot-detection that escalates to a full IP block. Running it
    by default is both mostly-useless and actively harmful."""
    monkeypatch.delenv("ASICODE_DDG_FALLBACK", raising=False)
    monkeypatch.delenv("BRAVE_API_KEY", raising=False)
    monkeypatch.delenv("SEARXNG_BASE_URL", raising=False)
    monkeypatch.delenv("ASICODE_NAVER_FALLBACK", raising=False)
    monkeypatch.setattr(_Host, "_has_docker_or_colima", lambda self: False)

    def _boom(self, q, m):
        raise AssertionError("DuckDuckGo must not run unless ASICODE_DDG_FALLBACK is on")

    monkeypatch.setattr(_Host, "_search_duckduckgo", _boom)
    res = _Host()._tool_search_web({"query": "test"})
    assert res["metadata"]["result_count"] == 0   # Startpage stubbed empty; nothing else runs


def test_search_web_includes_ddg_when_opted_in(monkeypatch):
    """The opt-in escape hatch works: users on a clean IP can still enable it."""
    monkeypatch.setenv("ASICODE_DDG_FALLBACK", "on")
    monkeypatch.delenv("BRAVE_API_KEY", raising=False)
    monkeypatch.delenv("SEARXNG_BASE_URL", raising=False)
    monkeypatch.setattr(_Host, "_has_docker_or_colima", lambda self: False)

    order: list[str] = []
    monkeypatch.setattr(
        _Host, "_search_duckduckgo",
        lambda self, q, m: order.append("ddg") or [{"title": "t", "url": "u", "snippet": "s"}],
    )
    res = _Host()._tool_search_web({"query": "test"})
    assert order == ["ddg"]
    assert res["metadata"]["result_count"] == 1


def test_should_try_ddg_env_modes(monkeypatch):
    host = _Host()
    monkeypatch.delenv("ASICODE_DDG_FALLBACK", raising=False)
    assert host._should_try_ddg() is False              # default: off
    for on in ("on", "always", "1", "true", "ON", " on "):
        monkeypatch.setenv("ASICODE_DDG_FALLBACK", on)
        assert host._should_try_ddg() is True, on
    for off in ("off", "no", "0", ""):
        monkeypatch.setenv("ASICODE_DDG_FALLBACK", off)
        assert host._should_try_ddg() is False, off


def test_block_wall_trips_breaker(monkeypatch):
    """A walled engine must be sidelined, not re-asked every search: each retry
    feeds the same bot-detection that escalates to a hard IP block."""
    monkeypatch.delenv("SEARXNG_BASE_URL", raising=False)
    monkeypatch.delenv("BRAVE_API_KEY", raising=False)
    monkeypatch.delenv("ASICODE_NAVER_FALLBACK", raising=False)
    monkeypatch.setattr(_Host, "_has_docker_or_colima", lambda self: False)

    calls = {"n": 0}

    def _walled(self, q, m):
        calls["n"] += 1
        raise wst._BlockWallError("Startpage served a bot-detection/block wall")

    monkeypatch.setattr(_Host, "_search_startpage", _walled)
    host = _Host()
    host._tool_search_web({"query": "test"})          # 1st: walls → trips breaker
    assert calls["n"] == 1
    assert host._backend_in_cooldown("Startpage")
    host._tool_search_web({"query": "test"})          # 2nd: skipped entirely
    assert calls["n"] == 1, "walled backend was asked again despite the breaker"


def test_block_wall_error_is_a_runtime_error():
    """Subclassing RuntimeError keeps every existing handler and caller working."""
    assert issubclass(wst._BlockWallError, RuntimeError)


# ── Startpage backend ───────────────────────────────────────────────────

# Verbatim shape of one Startpage result, captured live 2026-07-19. Kept faithful
# on purpose: the inline <style> INSIDE the result anchor and the rotating
# ``css-<hash>`` emotion classes are the two things the parser must survive.
_SP_RESULT = (
    '<a class="result-title result-link css-1bggj8v"'
    ' href="https://docs.python.org/3/library/asyncio-task.html"'
    ' target="_blank" rel="noopener nofollow noreferrer" data-testid="gl-title-link">'
    '<style data-emotion="css i3irj7">.css-i3irj7{line-height:18px;color:#2E39B3;}</style>'
    '<h2 class="wgl-title css-i3irj7">Coroutines and tasks — Python 3.14.6 documentation</h2>'
    "</a>"
    '<style data-emotion="css 1507v2l">.css-1507v2l{color:#1e222d;}</style>'
    '<p class="description css-1507v2l"><b>timeout</b>() transforms the '
    "<b>asyncio</b>.CancelledError into a TimeoutError</p>"
)


def test_startpage_parser_extracts_title_url_snippet():
    p = wst._StartpageResultParser(max_results=5)
    p.feed(_SP_RESULT)
    p.close()
    assert len(p.results) == 1
    r = p.results[0]
    assert r["url"] == "https://docs.python.org/3/library/asyncio-task.html"
    assert r["title"] == "Coroutines and tasks — Python 3.14.6 documentation"
    assert "transforms the" in r["snippet"]
    assert "asyncio.CancelledError" in r["snippet"].replace(" ", "")


def test_startpage_parser_excludes_css_from_title():
    """REGRESSION: <style> blocks sit INSIDE the result anchor, and HTMLParser
    fires handle_data for CDATA content, so a naive capture pulls raw CSS into
    the title. Nothing that looks like a stylesheet may survive into a field."""
    p = wst._StartpageResultParser(max_results=5)
    p.feed(_SP_RESULT)
    p.close()
    r = p.results[0]
    for field in ("title", "snippet"):
        assert "line-height" not in r[field], f"CSS leaked into {field}: {r[field]!r}"
        assert "css-" not in r[field], f"CSS leaked into {field}: {r[field]!r}"
        assert "{" not in r[field], f"CSS leaked into {field}: {r[field]!r}"


def test_startpage_parser_survives_rotated_emotion_hashes():
    """Startpage ships CSS-in-JS classes that change on every frontend deploy.
    Extraction must key off the stable semantic names only, so swapping every
    hash must not change the outcome (same lesson as the Naver backend)."""
    rotated = _SP_RESULT.replace("css-1bggj8v", "css-ZZZZZZ").replace("css-1507v2l", "css-QQQQQQ")
    p = wst._StartpageResultParser(max_results=5)
    p.feed(rotated)
    p.close()
    assert len(p.results) == 1
    assert p.results[0]["title"].startswith("Coroutines and tasks")


def test_startpage_parser_strips_control_chars():
    """Observed live: '전세사기 유\\x00형별 사례'. Stray C0 bytes must not reach
    the LLM context."""
    dirty = _SP_RESULT.replace("Coroutines", "Coro\x00uti\x01nes")
    p = wst._StartpageResultParser(max_results=5)
    p.feed(dirty)
    p.close()
    title = p.results[0]["title"]
    assert "\x00" not in title and "\x01" not in title
    assert title.startswith("Coroutines and tasks")


def test_startpage_parser_keeps_result_without_snippet():
    """A title with no following <p class="description"> must still be emitted
    (same three-site flush contract as the DDG parser)."""
    only_title = _SP_RESULT.split("<style data-emotion=\"css 1507v2l\"", maxsplit=1)[0]
    p = wst._StartpageResultParser(max_results=5)
    p.feed(only_title)
    p.close()
    assert len(p.results) == 1
    assert p.results[0]["snippet"] == ""


def test_startpage_parser_respects_max_results():
    p = wst._StartpageResultParser(max_results=2)
    p.feed(_SP_RESULT * 5)
    p.close()
    assert len(p.results) == 2


def test_attr_helpers_live_in_base_ssot():
    """``_get_attr``/``_has_class`` must exist ONCE, on the shared base. Both
    parsers needed them and the pair was briefly duplicated; this pins the SSOT so
    a future engine cannot reintroduce a twin that drifts."""
    base = wst._ResultParserBase
    for parser in (wst._DDGResultParser, wst._StartpageResultParser):
        assert issubclass(parser, base)
        for helper in ("_get_attr", "_has_class"):
            assert helper not in vars(parser), (
                f"{parser.__name__}.{helper} shadows the base SSOT — delete it and inherit"
            )
            assert getattr(parser, helper) is getattr(base, helper)


def test_flush_and_state_init_live_in_base_ssot():
    """``_flush`` and the common parser-state block must exist ONCE, on the
    shared base. Both parsers used to carry identical flush skeletons and
    identical ``__init__`` state blocks; this pins the SSOT the same way the
    attr helpers are pinned above."""
    base = wst._ResultParserBase
    for parser in (wst._DDGResultParser, wst._StartpageResultParser):
        assert "_flush" not in vars(parser), (
            f"{parser.__name__}._flush shadows the base SSOT — delete it and inherit"
        )
        assert parser._flush is base._flush
    # The common state must come from the base constructor, not per-parser copies.
    for parser in (wst._DDGResultParser, wst._StartpageResultParser):
        p = parser(max_results=7)
        assert p.max_results == 7 and p.results == []
        assert p._current is None and p._text_parts == []
        assert p._capturing is False and p._in_snippet is False
        assert p._emitted is False


def test_flush_required_fields_per_engine():
    """DDG emits on a title alone (the URL comes from the anchor itself);
    Startpage additionally demands an explicit URL — the per-engine
    ``_REQUIRED_FIELDS`` difference, pinned behaviorally."""
    ddg = wst._DDGResultParser(max_results=5)
    ddg.feed('<a class="result__a">Title without href</a>')
    ddg.close()
    assert len(ddg.results) == 1
    assert ddg.results[0]["url"] == ""

    sp = wst._StartpageResultParser(max_results=5)
    sp.feed('<a class="result-title">Title without href</a>')
    sp.close()
    assert len(sp.results) == 0     # no destination -> not usable


def test_has_class_matches_whole_tokens_only():
    """Class matching is token-wise: a substring test would match the rotating
    ``css-<hash>`` companions and prefix-sharing neighbours."""
    attrs = [("class", "result-title result-link css-1bggj8v")]
    assert wst._ResultParserBase._has_class(attrs, "result-title")
    assert wst._ResultParserBase._has_class(attrs, "result-link")
    assert not wst._ResultParserBase._has_class(attrs, "result")       # prefix, not a token
    assert not wst._ResultParserBase._has_class(attrs, "result-tit")   # partial token
    assert not wst._ResultParserBase._has_class([], "result-title")


class _StartpageStubClient:
    """Stub httpx.Client for _search_startpage (GET only)."""

    def __init__(self, body: str, status: int = 200):
        self._body, self._status = body, status

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get(self, url, params=None, headers=None):
        return httpx.Response(
            self._status,
            request=httpx.Request("GET", url),
            headers={"content-type": "text/html; charset=utf-8"},
            text=self._body,
        )


def test_search_startpage_parses_live_shaped_page(monkeypatch):
    monkeypatch.setattr(wst.httpx, "Client", lambda *a, **k: _StartpageStubClient(_SP_RESULT))
    results = _real_search_startpage(_Host(), "python asyncio timeout", 5)
    assert len(results) == 1
    assert results[0]["url"].startswith("https://docs.python.org/")


def test_search_startpage_raises_on_block_wall(monkeypatch):
    """The day Startpage starts challenging us, it must surface as an error —
    not as a plausible-looking 'no results found'."""
    wall = "<html><body><h1>Verification required</h1><p>complete the challenge</p></body></html>"
    monkeypatch.setattr(wst.httpx, "Client", lambda *a, **k: _StartpageStubClient(wall))
    with pytest.raises(RuntimeError, match="Startpage"):
        _real_search_startpage(_Host(), "anything", 5)


def test_search_startpage_empty_page_is_a_genuine_miss(monkeypatch):
    """No results and no wall markers → honest empty list, no exception."""
    monkeypatch.setattr(
        wst.httpx, "Client",
        lambda *a, **k: _StartpageStubClient("<html><body><p>No results.</p></body></html>"),
    )
    assert _real_search_startpage(_Host(), "zzzq unlikely", 5) == []


def test_search_web_tries_startpage_first(monkeypatch):
    """Startpage leads the chain: when it answers, no other backend is consulted."""
    monkeypatch.delenv("SEARXNG_BASE_URL", raising=False)
    monkeypatch.setenv("BRAVE_API_KEY", "fake-key")
    monkeypatch.setattr(_Host, "_has_docker_or_colima", lambda self: False)

    order: list[str] = []
    monkeypatch.setattr(
        _Host, "_search_startpage",
        lambda self, q, m: order.append("startpage") or [{"title": "t", "url": "u", "snippet": "s"}],
    )
    monkeypatch.setattr(_Host, "_search_brave", lambda self, q, m, k: order.append("brave") or [])
    monkeypatch.setattr(_Host, "_search_duckduckgo", lambda self, q, m: order.append("ddg") or [])

    res = _Host()._tool_search_web({"query": "test"})
    assert res["ok"], res.get("error")
    assert order == ["startpage"], f"Startpage must lead the chain; got {order}"


def test_tier1_queries_searxng_and_startpage_together(monkeypatch):
    """Tier 1 MERGES rather than stopping at the first success.

    SearXNG and Startpage are complements, not substitutes: SearXNG's own Google
    engine is dead from a flagged IP and its startpage engine fails to parse, so
    a first-wins chain with SearXNG in front would silently drop Google's index.
    Both must be queried even when the first one answers."""
    monkeypatch.setenv("SEARXNG_BASE_URL", "http://localhost:8080")
    monkeypatch.delenv("BRAVE_API_KEY", raising=False)

    ran: list[str] = []
    monkeypatch.setattr(
        _Host, "_search_searxng",
        lambda self, q, m, u: ran.append("searxng") or [
            {"title": "from searxng", "url": "https://a.example/x", "snippet": "s"}
        ],
    )
    monkeypatch.setattr(
        _Host, "_search_startpage",
        lambda self, q, m: ran.append("startpage") or [
            {"title": "from startpage", "url": "https://b.example/y", "snippet": "s"}
        ],
    )

    res = _Host()._tool_search_web({"query": "test"})
    assert sorted(ran) == ["searxng", "startpage"], f"both must run; got {ran}"
    assert res["metadata"]["result_count"] == 2, "both backends' results must survive the merge"
    assert "from searxng" in res["content"] and "from startpage" in res["content"]


def test_tier2_not_reached_when_tier1_returns_results(monkeypatch):
    """Brave is a metered free tier (2000/month) and DDG feeds bot-detection, so
    tier 2 must stay untouched whenever tier 1 produced anything at all."""
    monkeypatch.delenv("SEARXNG_BASE_URL", raising=False)
    monkeypatch.setenv("BRAVE_API_KEY", "fake-key")
    monkeypatch.setenv("ASICODE_DDG_FALLBACK", "on")
    monkeypatch.setattr(_Host, "_has_docker_or_colima", lambda self: False)
    monkeypatch.setattr(
        _Host, "_search_startpage",
        lambda self, q, m: [{"title": "t", "url": "https://x.example/1", "snippet": "s"}],
    )

    def _must_not_run(*a, **k):
        raise AssertionError("tier 2 ran despite tier 1 returning results")

    monkeypatch.setattr(_Host, "_search_brave", _must_not_run)
    monkeypatch.setattr(_Host, "_search_duckduckgo", _must_not_run)
    res = _Host()._tool_search_web({"query": "test"})
    assert res["metadata"]["result_count"] == 1


def test_tier2_reached_when_tier1_empty(monkeypatch):
    """Conversely, an empty tier 1 must still fall through to the paid/costly
    backends — merging must not become a way to return nothing."""
    monkeypatch.delenv("SEARXNG_BASE_URL", raising=False)
    monkeypatch.setenv("BRAVE_API_KEY", "fake-key")
    monkeypatch.setattr(_Host, "_has_docker_or_colima", lambda self: False)
    monkeypatch.setattr(_Host, "_search_startpage", lambda self, q, m: [])
    monkeypatch.setattr(
        _Host, "_search_brave",
        lambda self, q, m, k: [{"title": "brave hit", "url": "https://b.example/1", "snippet": "s"}],
    )
    res = _Host()._tool_search_web({"query": "test"})
    assert res["metadata"]["result_count"] == 1
    assert "brave hit" in res["content"]


def test_tier1_deadline_returns_partial_instead_of_waiting(monkeypatch):
    """A merge is only as fast as its slowest participant, so one slow engine
    must not set the latency of every search. Measured: Startpage 1.7s, SearXNG
    up to 20.1s. Whatever arrived by the deadline is returned."""
    import time

    monkeypatch.setenv("SEARXNG_BASE_URL", "http://localhost:8080")
    monkeypatch.delenv("BRAVE_API_KEY", raising=False)
    monkeypatch.setattr(wst, "_TIER1_DEADLINE_SEC", 0.4)

    def _slow(self, q, m, u):
        time.sleep(5)  # far past the deadline
        return [{"title": "too late", "url": "https://slow.example/1", "snippet": ""}]

    monkeypatch.setattr(_Host, "_search_searxng", _slow)
    monkeypatch.setattr(
        _Host, "_search_startpage",
        lambda self, q, m: [{"title": "fast", "url": "https://fast.example/1", "snippet": ""}],
    )

    t0 = time.perf_counter()
    res = _Host()._tool_search_web({"query": "test"})
    elapsed = time.perf_counter() - t0

    assert elapsed < 3.0, f"deadline not enforced — took {elapsed:.1f}s"
    assert "fast" in res["content"]
    assert "too late" not in res["content"]


def test_search_web_defers_searxng_autosetup_behind_startpage(monkeypatch):
    """The Docker auto-install offer raises a user Checkpoint, so it must not
    preempt a backend that works with no prompt at all. Startpage answering means
    the user is never asked to install anything."""
    monkeypatch.delenv("SEARXNG_BASE_URL", raising=False)
    monkeypatch.delenv("BRAVE_API_KEY", raising=False)
    monkeypatch.setattr(_Host, "_has_docker_or_colima", lambda self: True)  # offer is available

    def _must_not_prompt(self, q, m):
        raise AssertionError("SearXNG install prompt ran before/instead of Startpage")

    monkeypatch.setattr(_Host, "_setup_and_search_searxng", _must_not_prompt)
    monkeypatch.setattr(
        _Host, "_search_startpage",
        lambda self, q, m: [{"title": "t", "url": "u", "snippet": "s"}],
    )

    res = _Host()._tool_search_web({"query": "test"})
    assert res["ok"]
    assert res["metadata"]["result_count"] == 1


# ── Result merging ──────────────────────────────────────────────────────

def test_normalize_url_ignores_cosmetic_differences():
    n = wst._normalize_result_url
    base = n("https://example.com/path")
    assert n("http://example.com/path") == base          # scheme
    assert n("https://www.example.com/path") == base     # www.
    assert n("https://EXAMPLE.com/path") == base         # host case
    assert n("https://example.com/path/") == base        # trailing slash
    assert n("https://example.com/path#frag") == base    # fragment
    assert n("https://example.com:443/path") == base     # default port


def test_normalize_url_keeps_meaningful_query():
    """Over-normalising merges genuinely different pages: ?id=/?v= select real
    content. The query string must survive."""
    n = wst._normalize_result_url
    assert n("https://ex.com/watch?v=aaa") != n("https://ex.com/watch?v=bbb")
    assert n("https://ex.com/p?id=1") != n("https://ex.com/p")
    # Non-default port is content-addressing too.
    assert n("https://ex.com:8443/x") != n("https://ex.com/x")
    # Unparseable input degrades to "no dedup", never to a collapsed key.
    assert n("not a url") == "not a url"
    assert n("") == ""


def test_merge_ranks_cross_backend_agreement_first():
    """The point of querying more than one index: a URL two backends agree on
    outranks one that only the higher-priority backend returned."""
    shared = "https://agreed.example/doc"
    merged = wst._merge_search_results(
        [
            ("SearXNG", [
                {"title": "solo top", "url": "https://solo.example/a", "snippet": ""},
                {"title": "agreed", "url": shared, "snippet": "short"},
            ]),
            ("Startpage", [
                {"title": "agreed (fuller title)", "url": shared + "/", "snippet": "a longer snippet"},
            ]),
        ],
        max_results=5,
    )
    assert len(merged) == 2, "the two spellings of the shared URL must dedupe to one"
    assert merged[0]["url"] == shared, "the agreed-on result must rank first"
    assert merged[0]["sources"] == "SearXNG,Startpage"
    # Field selection keeps the most informative variant, not the first seen.
    assert merged[0]["title"] == "agreed (fuller title)"
    assert merged[0]["snippet"] == "a longer snippet"


def test_merge_preserves_backend_order_within_same_agreement():
    """With no agreement to separate them, each backend's own ordering stands and
    the caller's backend priority breaks the final tie."""
    merged = wst._merge_search_results(
        [
            ("First", [{"title": "f1", "url": "https://f.example/1", "snippet": ""}]),
            ("Second", [{"title": "s1", "url": "https://s.example/1", "snippet": ""}]),
        ],
        max_results=5,
    )
    assert [r["title"] for r in merged] == ["f1", "s1"]


def test_merge_drops_untitled_and_respects_max_results():
    merged = wst._merge_search_results(
        [("X", [
            {"title": "", "url": "https://no-title.example/1", "snippet": "s"},
            {"title": "ok1", "url": "https://a.example/1", "snippet": ""},
            {"title": "ok2", "url": "https://b.example/1", "snippet": ""},
            {"title": "ok3", "url": "https://c.example/1", "snippet": ""},
        ])],
        max_results=2,
    )
    assert [r["title"] for r in merged] == ["ok1", "ok2"]


def test_merge_handles_empty_and_urlless_input():
    assert wst._merge_search_results([], max_results=5) == []
    assert wst._merge_search_results([("X", [])], max_results=5) == []
    assert wst._merge_search_results(
        [("X", [{"title": "t", "url": "", "snippet": "s"}])], max_results=5
    ) == []


def test_consensus_is_surfaced_only_when_more_than_one_source(monkeypatch):
    """Naming the single engine that answered tells the model nothing; naming
    several that converged does."""
    host = _Host()
    one = host._format_search_results(
        "q", [{"title": "t", "url": "u", "snippet": "s", "sources": "Startpage"}], ["Startpage"]
    )
    assert "confirmed by" not in one["content"]
    two = host._format_search_results(
        "q", [{"title": "t", "url": "u", "snippet": "s", "sources": "SearXNG,Startpage"}], ["SearXNG"]
    )
    assert "confirmed by 2 sources" in two["content"]


# ── SearXNG engine curation ─────────────────────────────────────────────

def test_searxng_engines_default_is_the_curated_list(monkeypatch):
    monkeypatch.delenv("ASICODE_SEARXNG_ENGINES", raising=False)
    assert _Host()._searxng_engines() == wst._SEARXNG_DEFAULT_ENGINES


def test_searxng_engines_env_override_and_category_escape(monkeypatch):
    """Engine health is volatile and instance-specific, so the list must be
    tunable without a release — including all the way back to SearXNG's own
    category selection."""
    monkeypatch.setenv("ASICODE_SEARXNG_ENGINES", "bing, mojeek ,, yandex ")
    assert _Host()._searxng_engines() == "bing,mojeek,yandex"   # spaces/empties normalised
    monkeypatch.setenv("ASICODE_SEARXNG_ENGINES", "category")
    assert _Host()._searxng_engines() == ""                      # "" == use the category
    monkeypatch.setenv("ASICODE_SEARXNG_ENGINES", "CATEGORY")
    assert _Host()._searxng_engines() == ""                      # case-insensitive
    monkeypatch.setenv("ASICODE_SEARXNG_ENGINES", "   ")
    assert _Host()._searxng_engines() == wst._SEARXNG_DEFAULT_ENGINES  # blank → default


def test_default_engine_list_excludes_slow_failing_engines():
    """The list is latency-aware: SearXNG waits for its slowest engine, and the
    measured slow ones were all FAILING (yacy 5.02s timeout, gabanza 4.02s
    timeout, 360search 3.90s). Including them would spend the tier-1 deadline on
    engines that return nothing."""
    engines = set(wst._SEARXNG_DEFAULT_ENGINES.split(","))
    for slow_failure in ("yacy", "gabanza", "360search"):
        assert slow_failure not in engines, f"{slow_failure} was measured slow AND failing"


def test_default_engine_list_keeps_fast_failing_western_engines():
    """google/duckduckgo/brave are dead from a bot-flagged IP but healthy from a
    clean one, and they fail in ~0.3s. Dropping them would optimise this one IP
    at the expense of every user who is not blocked."""
    engines = set(wst._SEARXNG_DEFAULT_ENGINES.split(","))
    for fast_elsewhere in ("google", "duckduckgo", "brave"):
        assert fast_elsewhere in engines


def test_search_searxng_sends_engines_not_category(monkeypatch):
    """The whole point: ask for engines BY NAME. With categories=general the
    measured result was naver answering every query alone, English included."""
    seen: dict = {}

    class _C:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def get(self, url, params=None, headers=None):
            seen.update(params or {})
            return httpx.Response(200, request=httpx.Request("GET", url), json={"results": []})

    monkeypatch.delenv("ASICODE_SEARXNG_ENGINES", raising=False)
    monkeypatch.setattr(wst.httpx, "Client", lambda *a, **k: _C())
    _Host()._search_searxng("q", 5, "http://localhost:8080")
    assert seen.get("engines") == wst._SEARXNG_DEFAULT_ENGINES
    assert "categories" not in seen, "engines= and categories= must not both be sent"


def test_search_searxng_category_escape_restores_categories(monkeypatch):
    seen: dict = {}

    class _C:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def get(self, url, params=None, headers=None):
            seen.update(params or {})
            return httpx.Response(200, request=httpx.Request("GET", url), json={"results": []})

    monkeypatch.setenv("ASICODE_SEARXNG_ENGINES", "category")
    monkeypatch.setattr(wst.httpx, "Client", lambda *a, **k: _C())
    _Host()._search_searxng("q", 5, "http://localhost:8080")
    assert seen.get("categories") == "general"
    assert "engines" not in seen


# ── SearXNG image freshness ─────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _reset_staleness_check():
    """The once-per-process guard is class state — isolate it between tests."""
    wst.WebSearchToolsMixin._searxng_staleness_checked = False
    yield
    wst.WebSearchToolsMixin._searxng_staleness_checked = False


def _proc(stdout="", rc=0):
    class _P:
        returncode = rc
    p = _P()
    p.stdout = stdout
    return p


def test_image_age_parses_docker_nanosecond_timestamp(monkeypatch):
    """Docker emits RFC3339 with NANOseconds; datetime accepts at most micro, so
    an unmodified string raises ValueError and the age silently becomes None."""
    from datetime import datetime, timedelta, timezone

    created = datetime.now(timezone.utc) - timedelta(days=42)
    stamp = created.strftime("%Y-%m-%dT%H:%M:%S") + ".762521632Z"   # 9 fractional digits
    monkeypatch.setattr(wst.shutil, "which", lambda _: "/usr/bin/docker")
    monkeypatch.setattr(wst.subprocess, "run", lambda *a, **k: _proc(stamp))
    age = _Host()._searxng_image_age_days()
    assert age is not None, "nanosecond precision must not defeat parsing"
    assert 41.5 < age < 42.5


def test_image_age_none_when_docker_or_image_absent(monkeypatch):
    monkeypatch.setattr(wst.shutil, "which", lambda _: None)
    assert _Host()._searxng_image_age_days() is None          # no docker
    monkeypatch.setattr(wst.shutil, "which", lambda _: "/usr/bin/docker")
    monkeypatch.setattr(wst.subprocess, "run", lambda *a, **k: _proc("", rc=1))
    assert _Host()._searxng_image_age_days() is None          # image not pulled
    monkeypatch.setattr(wst.subprocess, "run", lambda *a, **k: _proc("not-a-timestamp"))
    assert _Host()._searxng_image_age_days() is None          # unparseable


def test_stale_notice_fires_only_past_threshold(tmp_path, monkeypatch):
    host = _Host()
    monkeypatch.setattr(_Host, "repo_root", str(tmp_path), raising=False)

    monkeypatch.setattr(_Host, "_searxng_image_age_days", lambda self: 5.0)
    assert host._stale_searxng_image_notice() is None, "a fresh image must not nag"

    wst.WebSearchToolsMixin._searxng_staleness_checked = False
    monkeypatch.setattr(_Host, "_searxng_image_age_days", lambda self: 45.0)
    notice = host._stale_searxng_image_notice()
    assert notice and "45 days old" in notice
    # Must warn against the destructive shortcut that loses settings.yml.
    assert "volumes" in notice and "docker pull" in notice


def test_stale_notice_is_rate_limited_across_processes(tmp_path, monkeypatch):
    """A notice repeated every session is a notice that gets ignored, so the
    suppression window has to survive process restart — i.e. live on disk."""
    monkeypatch.setattr(_Host, "repo_root", str(tmp_path), raising=False)
    monkeypatch.setattr(_Host, "_searxng_image_age_days", lambda self: 45.0)

    assert _Host()._stale_searxng_image_notice() is not None      # first: notified
    wst.WebSearchToolsMixin._searxng_staleness_checked = False    # simulate a new process
    assert _Host()._stale_searxng_image_notice() is None, "state file did not suppress the repeat"

    state = tmp_path / ".asicode" / "searxng_image_check.json"
    assert state.exists(), "suppression must be persisted, not in-memory"


def test_stale_notice_checked_once_per_process(tmp_path, monkeypatch):
    """The docker call behind the notice costs ~100ms; no search should re-pay it."""
    monkeypatch.setattr(_Host, "repo_root", str(tmp_path), raising=False)
    calls = {"n": 0}

    def _age(self):
        calls["n"] += 1
        return 45.0

    monkeypatch.setattr(_Host, "_searxng_image_age_days", _age)
    host = _Host()
    host._stale_searxng_image_notice()
    host._stale_searxng_image_notice()
    host._stale_searxng_image_notice()
    assert calls["n"] == 1


def test_stale_notice_never_breaks_a_working_search(tmp_path, monkeypatch):
    """A freshness HINT must never be able to fail a search that otherwise works.

    Not merely "the subprocess call is guarded" — ANY unforeseen error inside a
    purely advisory path must be swallowed, so this raises from the very first
    thing the notice does."""
    monkeypatch.setattr(_Host, "repo_root", str(tmp_path), raising=False)

    def _boom(self):
        raise OSError("docker socket exploded")

    monkeypatch.setattr(_Host, "_searxng_image_age_days", _boom)
    assert _Host()._stale_searxng_image_notice() is None

    # And a state file that is unreadable/corrupt is likewise not fatal.
    wst.WebSearchToolsMixin._searxng_staleness_checked = False
    state = tmp_path / ".asicode" / "searxng_image_check.json"
    state.parent.mkdir(parents=True, exist_ok=True)
    state.write_text("{ this is not json", encoding="utf-8")
    monkeypatch.setattr(_Host, "_searxng_image_age_days", lambda self: 45.0)
    assert _Host()._stale_searxng_image_notice() is not None  # corrupt state → treat as never notified


def test_stale_notice_survives_a_walled_search_path(tmp_path, monkeypatch):
    """The notice is computed on the success path, so a search that returns
    results must still carry it — and one that fails must not crash on it.

    Must isolate repo_root like the rest of this block: this test drives the
    full _tool_search_web path, and without isolation _stale_searxng_image_notice
    reads/writes the REAL .asicode/searxng_image_check.json. A prior passing run
    then seeds a 7-day suppression window that fails every later run AND leaks
    bogus age data (45.0) into the live state file."""
    monkeypatch.setattr(_Host, "repo_root", str(tmp_path), raising=False)
    monkeypatch.setenv("SEARXNG_BASE_URL", "http://localhost:8080")
    monkeypatch.delenv("BRAVE_API_KEY", raising=False)
    monkeypatch.setattr(_Host, "_searxng_image_age_days", lambda self: 45.0)
    monkeypatch.setattr(
        _Host, "_search_searxng",
        lambda self, q, m, u: [{"title": "t", "url": "https://x.example/1", "snippet": "s"}],
    )
    monkeypatch.setattr(_Host, "_search_startpage", lambda self, q, m: [])
    res = _Host()._tool_search_web({"query": "test"})
    assert res["metadata"]["result_count"] == 1
    assert "days old" in res["content"]


def test_stale_notice_surfaced_in_search_content(monkeypatch):
    """The notice belongs in the tool OUTPUT — the model reads that; nobody reads
    the log file."""
    host = _Host()
    res = host._format_search_results(
        "q", [{"title": "t", "url": "u", "snippet": "s"}], ["SearXNG"], notice="[SearXNG] stale image"
    )
    assert "[SearXNG] stale image" in res["content"]
    assert "1. t" in res["content"], "the notice must not displace the results"


def test_no_notice_when_searxng_not_configured(tmp_path, monkeypatch):
    """A user with no SearXNG must never see SearXNG maintenance advice.

    Isolated even though the notice path should not be reached at all here —
    that is the property under test, and if it ever regresses this test would
    otherwise start writing state into the real repo instead of just failing."""
    monkeypatch.setattr(_Host, "repo_root", str(tmp_path), raising=False)
    monkeypatch.delenv("SEARXNG_BASE_URL", raising=False)
    monkeypatch.delenv("BRAVE_API_KEY", raising=False)
    monkeypatch.setattr(_Host, "_has_docker_or_colima", lambda self: False)
    monkeypatch.setattr(
        _Host, "_search_startpage",
        lambda self, q, m: [{"title": "t", "url": "https://x.example/1", "snippet": "s"}],
    )
    monkeypatch.setattr(_Host, "_searxng_image_age_days", lambda self: 999.0)
    res = _Host()._tool_search_web({"query": "test"})
    assert "SearXNG" not in res["content"]


# ── SearXNG Checkpoint: concurrency ─────────────────────────────────────

class _AnswerResult:
    """Minimal stand-in for the ToolResult returned by _tool_ask_user."""

    def __init__(self, answer: str):
        self.metadata = {"answer": answer}


def _run_concurrently(fn, n=2, timeout=10):
    """Call ``fn`` from ``n`` threads released together; return their results."""
    import threading

    ready = threading.Barrier(n, timeout=timeout)
    out: list = [None] * n

    def worker(i):
        ready.wait()          # release all threads at the same instant
        out[i] = fn()

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=timeout)
        assert not t.is_alive(), "worker thread hung — possible deadlock"
    return out


def test_concurrent_start_prompt_is_issued_once(monkeypatch):
    """REGRESSION: two searches dispatched in one batch both raised the same
    'start SearXNG?' Checkpoint, because the decision cache is written only AFTER
    _tool_ask_user returns — and that call blocks while the prompt is on screen.
    Observed live: one prompt answered "yes", the other auto-applied "no"."""
    import threading
    import time

    host = _Host()
    prompts: list[str] = []
    guard = threading.Lock()

    def _ask(args):
        with guard:
            prompts.append(args["question"])
        time.sleep(0.3)       # hold the check→write window open, as a real prompt does
        return _AnswerResult("yes")

    monkeypatch.setattr(host, "_tool_ask_user", _ask, raising=False)
    results = _run_concurrently(host._ask_start_searxng)

    assert len(prompts) == 1, f"user was prompted {len(prompts)}x for one decision"
    assert results == [True, True], "both callers must receive the cached decision"


def test_concurrent_install_prompt_is_issued_once(monkeypatch):
    """Same race, same fix, for the install Checkpoint."""
    import threading
    import time

    host = _Host()
    prompts: list[str] = []
    guard = threading.Lock()

    def _ask(args):
        with guard:
            prompts.append(args["question"])
        time.sleep(0.3)
        return _AnswerResult("no")

    monkeypatch.setattr(host, "_tool_ask_user", _ask, raising=False)
    results = _run_concurrently(host._ask_install_searxng)

    assert len(prompts) == 1, f"user was prompted {len(prompts)}x for one decision"
    assert results == [False, False]


def test_ask_searxng_decision_exception_falls_back_to_no(monkeypatch):
    """ask_user raising must not propagate: cache False + return False.

    Pins the shared fallback contract of ``_ask_searxng_decision`` (single
    source for the start/install prompts) — checkpoint/prompting unavailable
    degrades to 'no' and the decision is cached so we never re-prompt.
    """
    host = _Host()

    def _boom(args):
        raise RuntimeError("checkpoint unavailable")

    monkeypatch.setattr(host, "_tool_ask_user", _boom, raising=False)
    assert host._ask_start_searxng() is False
    assert host._searxng_start_decision is False
    # Second call: served from cache — no second prompt attempt.
    assert host._ask_start_searxng() is False


def test_ask_searxng_decision_cached_fast_path_skips_prompt(monkeypatch):
    """Once decided, the answer is served from the cache without re-prompting."""
    host = _Host()
    calls = []

    def _ask(args):
        calls.append(args)
        return _AnswerResult("yes")

    monkeypatch.setattr(host, "_tool_ask_user", _ask, raising=False)
    assert host._ask_install_searxng() is True
    assert host._ask_install_searxng() is True  # fast path — no second prompt
    assert len(calls) == 1
    assert host._searxng_install_decision is True


def test_concurrent_start_searxng_never_overlaps(monkeypatch):
    """Two callers that both got "yes" must not run `docker run` concurrently —
    the second would fail on a duplicate container name. _start_searxng is
    serialized so the loser takes the idempotent 'container exists' path."""
    import threading
    import time

    host = _Host()
    concurrent = {"now": 0, "max": 0}
    guard = threading.Lock()

    def _body(self):
        with guard:
            concurrent["now"] += 1
            concurrent["max"] = max(concurrent["max"], concurrent["now"])
        time.sleep(0.2)
        with guard:
            concurrent["now"] -= 1
        return True

    monkeypatch.setattr(_Host, "_start_searxng_locked", _body, raising=False)
    _run_concurrently(host._start_searxng)

    assert concurrent["max"] == 1, (
        f"{concurrent['max']} concurrent _start_searxng bodies — docker run can race itself"
    )


def test_searxng_setup_lock_is_shared_and_reentrant():
    """One lock covers ask+start (so a 'yes' decision and the container start it
    triggers cannot interleave), and it is reentrant so a future refactor that
    nests them cannot self-deadlock."""
    import threading

    lock = wst.WebSearchToolsMixin._searxng_setup_lock
    assert isinstance(lock, type(threading.RLock())), "must be an RLock"
    with lock:
        assert lock.acquire(blocking=False), "RLock must be re-acquirable by its owner"
        lock.release()


def test_web_fetch_truncation_length_excludes_marker(monkeypatch):
    """metadata['length'] must report the real content size (max_chars), NOT
    max_chars + the ~90-char TRUNCATED marker appended to the body. Pagination
    (start_index) is computed before the marker and is unaffected; only the
    reported length was inflated."""
    host = _Host()
    body = "<p>" + ("x" * 2500) + "</p>"
    _stub_fetch(monkeypatch, _html_response(body))
    res = host._tool_web_fetch({"url": "https://example.com", "max_chars": 1000})
    assert res["ok"]
    assert "TRUNCATED" in res["content"]
    assert res["metadata"]["length"] == 1000, (
        "length must reflect real content (max_chars), not include the marker — "
        f"got {res['metadata']['length']}"
    )


# ══════════════════════════════════════════════════════════════════════════
# Persistent escalating wall backoff
#
# A bot-detection wall is an IP-reputation decision measured in DAYS (Startpage
# was measured suspended for 14 days straight), so it needs a different breaker
# than a connect failure: one that survives process exit and escalates, because
# every re-probe is another flagged request that deepens the block.
# ══════════════════════════════════════════════════════════════════════════


def test_wall_backoff_escalates_per_strike(monkeypatch, tmp_path):
    """Consecutive walls double the backoff, capped at _WALL_BACKOFF_MAX_SEC."""
    monkeypatch.setattr(_Host, "repo_root", str(tmp_path), raising=False)
    host = _Host()
    seen = []
    for _ in range(10):
        host._trip_backend_cooldown("Startpage", wall=True)
        st = wst.WebSearchToolsMixin._wall_state["Startpage"]
        seen.append(round(st["until"] - time.time()))
        # Re-tripping while still walled must keep escalating, so clear only the
        # in-memory connect cooldown between strikes.
        wst.WebSearchToolsMixin._backend_cooldown.clear()

    assert seen[0] == pytest.approx(wst._WALL_BACKOFF_BASE_SEC, abs=2)
    assert seen[1] == pytest.approx(wst._WALL_BACKOFF_BASE_SEC * 2, abs=2)
    assert seen[2] == pytest.approx(wst._WALL_BACKOFF_BASE_SEC * 4, abs=2)
    assert seen[-1] == pytest.approx(wst._WALL_BACKOFF_MAX_SEC, abs=2), (
        f"backoff must cap at {wst._WALL_BACKOFF_MAX_SEC}s so a recovered backend "
        f"is still re-probed ~daily; got {seen[-1]}"
    )


def test_wall_backoff_survives_process_restart(monkeypatch, tmp_path):
    """The whole point: a new process must not re-probe a backend it knows is walled.

    Startpage stayed blocked for two weeks while every `asi` start re-probed it —
    ~1.7s of latency each time AND another bot-flagged request. Simulated here by
    dropping the in-memory state and forcing a reload, which is exactly what a
    fresh interpreter does.
    """
    monkeypatch.setattr(_Host, "repo_root", str(tmp_path), raising=False)
    _Host()._trip_backend_cooldown("Startpage", wall=True)
    assert (tmp_path / ".asicode" / "search_backend_walls.json").exists()

    # ── simulate a fresh process ──
    wst.WebSearchToolsMixin._wall_state.clear()
    wst.WebSearchToolsMixin._backend_cooldown.clear()
    wst.WebSearchToolsMixin._wall_state_loaded = False

    assert _Host()._backend_in_cooldown("Startpage") is True, (
        "a persisted wall must sideline the backend in a new process"
    )


def test_lapsed_wall_keeps_its_strike_ladder(monkeypatch, tmp_path):
    """A probe after the deadline must not reset the ladder to strike 1.

    Otherwise a permanently blocked backend oscillates forever at the base
    interval: probe → wall → 15min → probe → wall → 15min. The ladder may only be
    reset by a real success (_clear_backend_wall).
    """
    monkeypatch.setattr(_Host, "repo_root", str(tmp_path), raising=False)
    host = _Host()
    host._trip_backend_cooldown("Startpage", wall=True)
    wst.WebSearchToolsMixin._backend_cooldown.clear()

    # Expire the backoff, then let the breaker observe the lapse (the probe).
    wst.WebSearchToolsMixin._wall_state["Startpage"]["until"] = time.time() - 1
    assert host._backend_in_cooldown("Startpage") is False, "lapsed backoff must allow a probe"

    # The probe walls again → this is strike 2, not strike 1.
    host._trip_backend_cooldown("Startpage", wall=True)
    st = wst.WebSearchToolsMixin._wall_state["Startpage"]
    assert st["strikes"] == 2
    assert st["until"] - time.time() == pytest.approx(wst._WALL_BACKOFF_BASE_SEC * 2, abs=2)


def test_success_clears_wall_ladder(monkeypatch, tmp_path):
    monkeypatch.setattr(_Host, "repo_root", str(tmp_path), raising=False)
    host = _Host()
    host._trip_backend_cooldown("Startpage", wall=True)
    wst.WebSearchToolsMixin._backend_cooldown.clear()

    host._clear_backend_wall("Startpage")
    assert "Startpage" not in wst.WebSearchToolsMixin._wall_state
    assert host._backend_in_cooldown("Startpage") is False
    with open(tmp_path / ".asicode" / "search_backend_walls.json", encoding="utf-8") as fh:
        assert json.load(fh) == {}, "recovery must be persisted, not only held in memory"


def test_connect_failure_does_not_persist_a_wall(monkeypatch, tmp_path):
    """Only walls escalate. A connect failure keeps the transient 90s cooldown."""
    monkeypatch.setattr(_Host, "repo_root", str(tmp_path), raising=False)
    host = _Host()
    host._trip_backend_cooldown("SearXNG")  # wall=False (default)

    assert "SearXNG" not in wst.WebSearchToolsMixin._wall_state
    assert not (tmp_path / ".asicode" / "search_backend_walls.json").exists(), (
        "a host that is merely down must not be written off for hours"
    )
    assert host._backend_in_cooldown("SearXNG") is True


def test_walled_backend_notice_only_after_threshold(monkeypatch, tmp_path):
    """Degraded coverage is invisible in the results, so it is stated in the output.

    But not for a blip — the notice fires only once a backend has been blocked
    long enough that the user's search quality is genuinely reduced.
    """
    monkeypatch.setattr(_Host, "repo_root", str(tmp_path), raising=False)
    # Pinned: with a browser route available the notice reports substitution
    # instead (covered below), and whether Playwright is installed is a property
    # of the machine running the suite, not of the behaviour under test.
    monkeypatch.setattr(_Host, "_startpage_browser_available", lambda self: False, raising=False)
    host = _Host()
    host._trip_backend_cooldown("Startpage", wall=True)
    assert host._walled_backend_notice() is None, "a fresh wall is not yet newsworthy"

    wst.WebSearchToolsMixin._wall_state["Startpage"]["since"] = time.time() - 3 * 86400
    notice = host._walled_backend_notice()
    assert notice is not None and "Startpage" in notice and "3.0d" in notice
    assert "coverage is reduced" in notice


def test_walled_notice_reports_substitution_not_degradation(monkeypatch, tmp_path):
    """A backend carried by its other transport is NOT degrading coverage.

    Reporting it as blocked would be the same class of misreport this notice was
    written to prevent, only pointed the other way — the user would be told their
    general-web coverage dropped while it was in fact intact.
    """
    monkeypatch.setattr(_Host, "repo_root", str(tmp_path), raising=False)
    monkeypatch.setattr(_Host, "_startpage_browser_available", lambda self: True, raising=False)
    host = _Host()
    host._trip_backend_cooldown("Startpage", wall=True)
    wst.WebSearchToolsMixin._wall_state["Startpage"]["since"] = time.time() - 3 * 86400

    notice = host._walled_backend_notice()
    assert notice is not None and "Startpage" in notice
    assert "browser route" in notice
    assert "NOT reduced" in notice
    assert "skipped" not in notice


def test_startpage_browser_route_registers_only_when_the_http_route_is_walled(monkeypatch):
    """The browser costs ~3s and a Chromium process, so it must stay out of the
    tier while the ~1s httpx route still works."""
    monkeypatch.setattr(wst, "HAS_PLAYWRIGHT", True, raising=False)
    host = _Host()
    monkeypatch.setattr(_Host, "_backend_in_cooldown", lambda self, name: False, raising=False)
    assert host._startpage_browser_available() is False

    monkeypatch.setattr(_Host, "_backend_in_cooldown", lambda self, name: name == "Startpage", raising=False)
    import external_llm.agent.tool_handlers.browser_tools as bt

    monkeypatch.setattr(bt, "HAS_PLAYWRIGHT", True)
    monkeypatch.setattr(bt, "PLAYWRIGHT_BROWSER_AVAILABLE", True)
    assert host._startpage_browser_available() is True
    # No Playwright → no route, and above all no install prompt inside a search.
    monkeypatch.setattr(bt, "PLAYWRIGHT_BROWSER_AVAILABLE", False)
    assert host._startpage_browser_available() is False


def test_startpage_browser_parses_with_the_shared_parser(monkeypatch):
    """The browser route must reuse _StartpageResultParser, not grow a second
    parser that can drift from the httpx one."""
    html = (
        '<a class="result-title result-link css-xx" href="https://example.com/a">'
        '<h2 class="wgl-title">Alpha</h2></a><p class="description">snip a</p>'
        '<a class="result-title result-link css-yy" href="https://example.com/b">'
        '<h2 class="wgl-title">Beta</h2></a><p class="description">snip b</p>'
    )
    seen = {}

    def _fake_render(self, url, js, **kw):
        seen.update({"url": url, "js": js, **kw})
        return html

    monkeypatch.setattr(_Host, "_render_and_eval", _fake_render, raising=False)
    out = _Host()._search_startpage_browser("claude opus 5", 10)

    assert [r["title"] for r in out] == ["Alpha", "Beta"]
    assert out[0]["url"] == "https://example.com/a"
    assert "claude+opus+5" in seen["url"]
    # Client-hydrated page: without the selector wait the render parses to ZERO
    # results (measured 5/5 queries), so the wait is a correctness requirement.
    assert seen["wait_for_selector"] == "a.result-link"


def test_startpage_browser_route_reports_its_own_wall(monkeypatch):
    """If the browser route is ALSO walled it must raise, not return an empty list
    that the chain would read as an honest 'nothing matched'."""
    monkeypatch.setattr(
        _Host, "_render_and_eval",
        lambda self, url, js, **kw: "<html><body>Verification required</body></html>",
        raising=False,
    )
    with pytest.raises(RuntimeError, match="Startpage \\(browser\\)"):
        _Host()._search_startpage_browser("q", 5)


def test_wall_state_read_failure_is_survivable(monkeypatch, tmp_path):
    """Corrupt state must degrade to "no walls known", never break a search."""
    monkeypatch.setattr(_Host, "repo_root", str(tmp_path), raising=False)
    path = tmp_path / ".asicode"
    path.mkdir(parents=True, exist_ok=True)
    (path / "search_backend_walls.json").write_text("{not json", encoding="utf-8")

    assert _Host()._backend_in_cooldown("Startpage") is False


# ══════════════════════════════════════════════════════════════════════════
# Exa backend (keyless MCP)
# ══════════════════════════════════════════════════════════════════════════

_EXA_PAYLOAD = (
    "Title: Exceptions\n"
    "URL: https://www.python-httpx.org/exceptions/\n"
    "Published: 2026-07-28T22:50:35.095Z\n"
    "Author: N/A\n"
    "Highlights:\n"
    "## The exception hierarchy\n"
    "...\n"
    "ConnectTimeout is a TimeoutException, not a ConnectError.\n"
    "\n---\n\n"
    "Title: docs/exceptions.md at master\n"
    "URL: https://github.com/encode/httpx/blob/master/docs/exceptions.md\n"
    "Published: N/A\n"
    "Author: N/A\n"
    "Highlights:\n"
    "Timed out while connecting to the host.\n"
)


def test_parse_exa_results_reads_the_wire_format():
    out = wst._parse_exa_results(_EXA_PAYLOAD, 5)
    assert [r["url"] for r in out] == [
        "https://www.python-httpx.org/exceptions/",
        "https://github.com/encode/httpx/blob/master/docs/exceptions.md",
    ]
    assert out[0]["title"] == "Exceptions"
    assert "ConnectTimeout is a TimeoutException" in out[0]["excerpt"]
    assert out[0]["published"] == "2026-07-28T22:50:35.095Z"
    assert out[1]["published"] == "", "Exa's literal 'N/A' is absence, not a date"
    assert out[0]["snippet"] == "", "an excerpt must not masquerade as a SERP snippet"
    assert "..." not in out[0]["excerpt"].splitlines(), "Exa's elision marker carries no information"


def test_parse_exa_markdown_rule_does_not_split_a_result():
    """Highlights are page text and can contain a horizontal rule.

    Splitting on the separator alone would turn one result into two malformed
    halves — the second with no URL, silently dropping a real hit.
    """
    payload = (
        "Title: Design doc\n"
        "URL: https://example.com/doc\n"
        "Published: N/A\n"
        "Author: N/A\n"
        "Highlights:\n"
        "Section one.\n"
        "\n---\n\n"
        "Section two, after a markdown rule.\n"
    )
    out = wst._parse_exa_results(payload, 5)
    assert len(out) == 1, f"a markdown rule must not fabricate a second result; got {out}"
    assert "Section two" in out[0]["excerpt"]


def test_parse_exa_skips_untitled_results():
    """`Title: N/A` entries are dropped downstream anyway — do not spend a slot."""
    payload = (
        "Title: N/A\n"
        "URL: https://example.com/a\n"
        "Highlights:\nnothing useful\n"
        "\n---\n\n"
        "Title: Real\n"
        "URL: https://example.com/b\n"
        "Highlights:\nreal content\n"
    )
    out = wst._parse_exa_results(payload, 5)
    assert [r["title"] for r in out] == ["Real"]


def test_parse_exa_respects_max_results():
    payload = "\n---\n\n".join(
        f"Title: T{i}\nURL: https://example.com/{i}\nHighlights:\nbody {i}\n" for i in range(10)
    )
    assert len(wst._parse_exa_results(payload, 3)) == 3


def test_parse_exa_tolerates_garbage():
    assert wst._parse_exa_results("", 5) == []
    assert wst._parse_exa_results("no headers here at all", 5) == []


def test_mcp_result_text_flattens_and_raises():
    ok = {"result": {"content": [{"type": "text", "text": "hello"}, {"type": "image"}]}}
    assert wst._mcp_result_text(ok) == "hello"

    with pytest.raises(RuntimeError, match="MCP error"):
        wst._mcp_result_text({"error": {"message": "quota exceeded"}})
    with pytest.raises(TypeError, match="no result"):
        wst._mcp_result_text({"jsonrpc": "2.0"})
    with pytest.raises(TypeError, match="malformed MCP response"):
        wst._mcp_result_text(["not", "a", "dict"])
    with pytest.raises(RuntimeError, match="MCP tool error"):
        wst._mcp_result_text({"result": {"isError": True, "content": [{"type": "text", "text": "upstream down"}]}})


def _mcp_response(body: str, *, sse: bool, status: int = 200) -> httpx.Response:
    return httpx.Response(
        status,
        request=httpx.Request("POST", wst._EXA_MCP_URL),
        headers={"content-type": "text/event-stream" if sse else "application/json"},
        text=(f"event: message\ndata: {body}\n\n" if sse else body),
    )


def test_parse_mcp_body_handles_sse_and_plain_json():
    """The endpoint picks the encoding per response; both must decode identically."""
    body = json.dumps({"result": {"content": [{"type": "text", "text": "x"}]}})
    for sse in (True, False):
        assert wst.WebSearchToolsMixin._parse_mcp_body(_mcp_response(body, sse=sse))["result"]["content"]

    with pytest.raises(RuntimeError, match="no decodable data frame"):
        wst.WebSearchToolsMixin._parse_mcp_body(
            httpx.Response(
                200,
                request=httpx.Request("POST", wst._EXA_MCP_URL),
                headers={"content-type": "text/event-stream"},
                text="event: ping\n\n",
            )
        )


class _ExaStubClient:
    """Stub httpx.Client for _search_exa (POST only)."""

    def __init__(self, response: httpx.Response):
        self._response = response
        self.calls: list[dict] = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def post(self, url, json=None, data=None, headers=None):
        self.calls.append({"url": url, "json": json})
        return self._response


def test_search_exa_parses_a_live_shaped_response(monkeypatch):
    body = json.dumps({"result": {"content": [{"type": "text", "text": _EXA_PAYLOAD}]}})
    stub = _ExaStubClient(_mcp_response(body, sse=True))
    monkeypatch.setattr(wst.httpx, "Client", lambda *a, **k: stub)

    out = _real_search_exa(_Host(), "httpx exceptions", 5)
    assert len(out) == 2
    assert stub.calls[0]["json"]["method"] == "tools/call", (
        "the endpoint serves tools/call without an initialize handshake — "
        "sending one would cost an extra round trip on every search"
    )
    assert stub.calls[0]["json"]["params"]["arguments"]["numResults"] == 5


def test_search_exa_quota_refusal_is_a_wall(monkeypatch):
    """HTTP 429 from a keyless endpoint is a quota block, not a transient blip.

    It must reach the escalating ladder rather than being retried on the shared
    ~1.5s policy, which cannot outlast a quota that resets on a timer.
    """
    stub = _ExaStubClient(_mcp_response("{}", sse=False, status=429))
    monkeypatch.setattr(wst.httpx, "Client", lambda *a, **k: stub)

    with pytest.raises(wst._BlockWallError, match="429"):
        _real_search_exa(_Host(), "anything", 5)
    assert len(stub.calls) == 1, "a daily quota must not be re-asked 1.5s later"


@pytest.mark.parametrize("value,expected", [("off", False), ("0", False), ("no", False),
                                            ("", True), ("on", True)])
def test_exa_opt_out_switch(monkeypatch, value, expected):
    monkeypatch.setenv(wst._EXA_ENV, value)
    assert wst.WebSearchToolsMixin._should_try_exa() is expected


def test_tier1_includes_exa(monkeypatch):
    """Exa participates in the merge, not as a fallback behind a dead backend."""
    monkeypatch.delenv("SEARXNG_BASE_URL", raising=False)
    monkeypatch.setattr(_Host, "_has_docker_or_colima", lambda self: False)
    called: list[str] = []
    monkeypatch.setattr(
        _Host, "_search_startpage",
        lambda self, q, m: called.append("startpage") or [{"title": "sp", "url": "https://a/", "snippet": "s"}],
    )
    monkeypatch.setattr(
        _Host, "_search_exa",
        lambda self, q, m: called.append("exa") or [{"title": "exa", "url": "https://b/", "excerpt": "e"}],
    )

    res = _Host()._tool_search_web({"query": "test"})
    assert sorted(called) == ["exa", "startpage"], (
        "Startpage answering must not short-circuit Exa — tier 1 merges"
    )
    assert "https://b/" in res["content"]


# ══════════════════════════════════════════════════════════════════════════
# Excerpt rendering
# ══════════════════════════════════════════════════════════════════════════


def test_excerpt_supersedes_snippet_and_carries_date():
    res = _Host()._format_search_results(
        "q",
        [{"title": "T", "url": "https://x/", "snippet": "short serp line",
          "excerpt": "the real page text", "published": "2026-07-28T22:50:35.095Z", "sources": "Exa"}],
        ["Exa"],
    )
    assert "the real page text" in res["content"]
    assert "short serp line" not in res["content"], (
        "printing both spends tokens twice on the same result"
    )
    assert "Published: 2026-07-28" in res["content"]


def test_excerpt_falls_back_to_snippet_when_absent():
    res = _Host()._format_search_results(
        "q", [{"title": "T", "url": "https://x/", "snippet": "serp line", "sources": "SearXNG"}], ["SearXNG"]
    )
    assert "serp line" in res["content"]


def test_excerpt_total_budget_is_enforced():
    """Excerpts are ~8x a snippet; unbounded they dominate every search's cost."""
    results = [
        {"title": f"T{i}", "url": f"https://x/{i}", "snippet": "", "excerpt": "y" * 4000, "sources": "Exa"}
        for i in range(6)
    ]
    content = _Host()._format_search_results("q", results, ["Exa"])["content"]
    spent = content.count("y")
    assert spent <= wst._EXCERPT_TOTAL_BUDGET, f"budget blown: {spent} chars of excerpt"
    assert "…" in content or "[…]" in content, "truncation must be visible to the model"
    for i in range(6):
        assert f"https://x/{i}" in content, "budget may drop excerpts, never whole results"


def test_merge_preserves_excerpt_and_prefers_the_fuller_one():
    merged = wst._merge_search_results(
        [
            ("SearXNG", [{"title": "T", "url": "https://x/", "snippet": "serp"}]),
            ("Exa", [{"title": "T", "url": "https://x/", "excerpt": "long page text", "published": "2026-01-01"}]),
        ],
        5,
    )
    assert len(merged) == 1
    assert merged[0]["excerpt"] == "long page text"
    assert merged[0]["snippet"] == "serp", (
        "snippet and excerpt are different fields — merging them lets a snippet "
        "win on length and discard the better content"
    )
    assert merged[0]["published"] == "2026-01-01"
    assert merged[0]["sources"] == "SearXNG,Exa"


def test_wall_state_is_concurrency_safe(monkeypatch, tmp_path):
    """Searches run concurrently on the shared tool-executor pool.

    The wall bookkeeping is process-wide class state mutated from those threads,
    and the load path REPLACES the whole dict — so both the load and every
    read-modify-write must happen under the one lock. Exercised here rather than
    reasoned about, because the failure mode (a lost strike, a backend briefly
    seen as un-walled) is silent.
    """
    import sys
    import threading

    monkeypatch.setattr(_Host, "repo_root", str(tmp_path), raising=False)
    host = _Host()
    errors: list[BaseException] = []
    barrier = threading.Barrier(8)

    def worker(i: int) -> None:
        try:
            barrier.wait()
            for _ in range(25):
                host._backend_in_cooldown("Startpage")
                host._trip_backend_cooldown("Startpage", wall=True)
                host._walled_backend_notice()
                if i == 0:
                    host._clear_backend_wall("Exa")
        except BaseException as e:
            # Collected, not raised: an exception in a worker thread is invisible
            # to pytest, so without this the test would pass through a crash.
            errors.append(e)

    # Force the interpreter to preempt INSIDE the read-modify-write. The strike
    # update is "read prev → +1 → store", which at the default 5ms switch interval
    # completes between thread switches nearly every time — so an UNLOCKED
    # implementation passes this test. Confirmed by mutation: removing the lock
    # survived 5/5 runs at the default interval and is caught at this one. Without
    # this line the test asserts nothing about locking.
    prev_interval = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    try:
        threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)
    finally:
        sys.setswitchinterval(prev_interval)

    assert not errors, f"concurrent wall bookkeeping raised: {errors[:3]}"
    assert not any(t.is_alive() for t in threads), "a thread deadlocked on the cooldown lock"
    st = wst.WebSearchToolsMixin._wall_state["Startpage"]
    assert st["strikes"] == 200, f"lost strikes under concurrency: {st['strikes']}"
    assert st["until"] - time.time() == pytest.approx(wst._WALL_BACKOFF_MAX_SEC, abs=2)


# ══════════════════════════════════════════════════════════════════════════
# Merge relevance tiebreaker
#
# Every backend's own #0 arrives at (agreement=1, best_position=0), so before
# this the winner was decided by which backend was appended to tier1 first.
# SearXNG leads that list and keyword-matches badly on its top hit, so its junk
# systematically outranked Exa's correct #0 (measured 2026-08-03, 8 queries:
# "ruff F821 undefined name" -> Maine Coon on Wikipedia; "postgres index only
# scan" -> postgresql.org's front page; r/LocalLLaMA... -> r-project.org).
# ══════════════════════════════════════════════════════════════════════════


def test_relevance_breaks_the_tie_that_backend_order_used_to_decide():
    """The reported defect, minimised: two #0 results, one of them junk."""
    per_backend = [
        # Listed FIRST — under the old key this won on first_backend alone.
        ("SearXNG", [{"title": "The R Project for Statistical Computing",
                      "url": "https://www.r-project.org/",
                      "snippet": "R is a free software environment for statistical computing"}]),
        ("Exa", [{"title": "r/LocalLLaMA",
                  "url": "https://www.reddit.com/r/LocalLLaMA/",
                  "excerpt": "LocalLLaMA subreddit discussing DeepSeek and local models"}]),
    ]
    query = "r/LocalLLaMA DeepSeek reddit thread"

    assert _merge_top_url(per_backend, "") == "https://www.r-project.org/", (
        "precondition: without relevance, declaration order decides"
    )
    assert _merge_top_url(per_backend, query) == "https://www.reddit.com/r/LocalLLaMA/"


def _merge_top_url(per_backend, query: str) -> str:
    return wst._merge_search_results(per_backend, 5, query)[0]["url"]


def test_relevance_does_not_override_cross_backend_agreement():
    """Agreement stays the primary signal — relevance only breaks its ties.

    Two independent indexes converging on a page is evidence relevance scoring
    over ten short documents cannot produce, so a 2-source result must keep
    outranking a 1-source result that merely shares more query words.
    """
    agreed = {"title": "Result", "url": "https://agreed.example/", "snippet": "generic"}
    per_backend = [
        ("SearXNG", [agreed]),
        ("Exa", [agreed, {"title": "postgres index only scan visibility map",
                          "url": "https://solo.example/postgres-index-only-scan-visibility-map",
                          "excerpt": "postgres index only scan visibility map"}]),
    ]
    top = wst._merge_search_results(per_backend, 5, "postgres index only scan visibility map")[0]
    assert top["url"] == "https://agreed.example/"
    assert top["sources"] == "SearXNG,Exa"


def test_relevance_does_not_override_engine_position():
    """Containment: the engines' own ordering still wins above the tie.

    A ten-document BM25 has far less signal than an engine's link graph, click
    data and freshness, so relevance sits BELOW best_position — a result the
    engine ranked #0 keeps beating one it ranked #1.
    """
    per_backend = [
        ("Exa", [
            {"title": "Generic overview page", "url": "https://a.example/", "excerpt": "overview"},
            {"title": "postgres index only scan visibility map",
             "url": "https://b.example/", "excerpt": "postgres index only scan visibility map"},
        ]),
    ]
    assert _merge_top_url(per_backend, "postgres index only scan visibility map") == "https://a.example/"


def test_relevance_tie_falls_back_to_backend_order():
    """Determinism when nothing matches: a query sharing no token scores 0.0 for
    every candidate, and the ordering must stay stable rather than arbitrary."""
    per_backend = [
        ("SearXNG", [{"title": "Alpha", "url": "https://alpha.example/", "snippet": "aaa"}]),
        ("Exa", [{"title": "Beta", "url": "https://beta.example/", "excerpt": "bbb"}]),
    ]
    assert _merge_top_url(per_backend, "zzzz qqqq") == "https://alpha.example/"


def test_merge_without_a_query_keeps_the_old_ordering():
    """query='' disables relevance, so existing callers/tests are unaffected."""
    per_backend = [
        ("SearXNG", [{"title": "Junk", "url": "https://junk.example/", "snippet": "unrelated"}]),
        ("Exa", [{"title": "postgres index only scan", "url": "https://good.example/", "excerpt": "postgres"}]),
    ]
    assert _merge_top_url(per_backend, "") == "https://junk.example/"


def test_relevance_scores_korean_queries():
    """Hangul must score, or every Korean query silently falls back to backend order.

    This is why the tokenizer is reused from rag_configs rather than written as a
    ``\\w+``/ASCII split: its regex has a dedicated Hangul-run alternative.

    Deliberately Hangul-ONLY on both sides. An earlier version of this test used
    "파이썬 asyncio CancelledError 처리 방법" against a doc containing those same
    Latin tokens — so an ASCII-only tokenizer still ranked it first and the test
    passed while asserting nothing about Hangul (confirmed by mutation). With no
    Latin token to fall back on, an ASCII tokenizer yields an empty query, scores
    every candidate 0.0, and the junk result wins on backend order.
    """
    per_backend = [
        ("SearXNG", [{"title": "Unrelated page", "url": "https://junk.example/",
                      "snippet": "nothing to do with the question"}]),
        ("Exa", [{"title": "전세 사기 대처 방법 정리",
                  "url": "https://ko.example/guide", "excerpt": "전세 사기 피해 대처 절차"}]),
    ]
    assert _merge_top_url(per_backend, "전세 사기 대처 방법") == "https://ko.example/guide"


def test_relevance_tokens_keep_hangul_and_do_not_split_camel_case():
    toks = wst._relevance_tokens("파이썬 asyncio CancelledError 처리")
    assert "파이썬" in toks and "처리" in toks, f"Hangul runs dropped: {toks}"
    assert "cancellederror" in toks, (
        "CamelCase must stay whole — splitting it into cancelled/error makes "
        f"generic pages match a specific API name: {toks}"
    )


def test_url_contributes_relevance():
    """The URL often names what the title omits (reddit.com/r/LocalLLaMA)."""
    assert "localllama" in wst._url_text("https://www.reddit.com/r/LocalLLaMA/rising/").lower()
    assert "index only scans" in wst._url_text(
        "https://www.postgresql.org/docs/current/indexes-index-only-scans.html"
    ).replace("-", " ").lower()

    per_backend = [
        ("SearXNG", [{"title": "Untitled", "url": "https://a.example/", "snippet": ""}]),
        # Title says nothing; the path carries the whole signal.
        ("Exa", [{"title": "Untitled", "url": "https://www.reddit.com/r/LocalLLaMA/", "excerpt": ""}]),
    ]
    assert _merge_top_url(per_backend, "LocalLLaMA reddit") == "https://www.reddit.com/r/LocalLLaMA/"


def test_relevance_is_not_leaked_into_results():
    """`relevance` is internal bookkeeping, not part of the tool's output shape."""
    merged = wst._merge_search_results(
        [("Exa", [{"title": "T", "url": "https://x/", "excerpt": "body"}])], 5, "body"
    )
    assert "relevance" not in merged[0]


def test_relevance_scores_empty_inputs_safely():
    assert wst._relevance_scores("", ["a"]) == [0.0]
    assert wst._relevance_scores("q", []) == []
    assert wst._relevance_scores("!!! ???", ["a"]) == [0.0], "punctuation-only query has no tokens"


# ── ESC-cancelable retry backoff ────────────────────────────────────────
# Regression: the tool-layer retry sleeps were raw ``time.sleep``, so ESC was
# unresponsive for up to the 30s Retry-After cap (the client-layer backoff was
# made cancelable in the same change family; the tool layer lagged behind).

def test_live_cancel_event_absent_and_present():
    """``_live_cancel_event`` must be None on duck-typed hosts without a config
    (plain-sleep legacy behavior) and live on hosts that carry one."""
    host = _Host()
    assert host._live_cancel_event() is None
    import types

    ce = threading.Event()
    host.config = types.SimpleNamespace(cancel_event=ce)
    assert host._live_cancel_event() is ce


def test_http_retry_cancel_already_set_aborts_before_any_request():
    """An already-set cancel_event aborts before the first request — no point
    issuing a request the user no longer wants."""
    ce = threading.Event()
    ce.set()
    client = _SequenceClient([_resp(200, text="ok")])
    with pytest.raises(wst.AgentCancelled):
        wst.WebSearchToolsMixin._http_request_with_retry(
            client, "GET", "https://x/", cancel_event=ce
        )
    assert client.calls == 0


def test_http_retry_cancel_mid_backoff_aborts_retry():
    """ESC during a Retry-After wait must abort with AgentCancelled instead of
    sleeping out the (capped 30s) wait."""
    ce = threading.Event()

    def _late_set():
        time.sleep(0.05)
        ce.set()

    t = threading.Thread(target=_late_set, daemon=True)
    t.start()
    client = _SequenceClient([
        _resp(429, headers={"retry-after": "30"}),
        _resp(200, text="ok"),
    ])
    with pytest.raises(wst.AgentCancelled):
        wst.WebSearchToolsMixin._http_request_with_retry(
            client, "GET", "https://x/", cancel_event=ce
        )
    t.join(timeout=5)
    assert client.calls == 1  # no retry after the interrupt


def test_http_retry_cancel_mid_transient_backoff_aborts_retry():
    """Same interruptibility for the transient-error (e.g. slow server) retry
    sleep."""
    ce = threading.Event()

    def _late_set():
        time.sleep(0.05)
        ce.set()

    t = threading.Thread(target=_late_set, daemon=True)
    t.start()

    class _C:
        def __init__(self):
            self.calls = 0

        def get(self, url, params=None, headers=None):
            self.calls += 1
            raise httpx.ReadTimeout("slow")

    c = _C()
    with pytest.raises(wst.AgentCancelled):
        wst.WebSearchToolsMixin._http_request_with_retry(
            c, "GET", "https://x/", retries=3, cancel_event=ce
        )
    t.join(timeout=5)
    assert c.calls == 1


def test_http_retry_no_cancel_event_keeps_legacy_plain_sleep(monkeypatch):
    """Without a cancel_event the helper must keep its exact legacy behavior —
    plain ``time.sleep`` with the same backoff — so non-cancel callers are
    unaffected."""
    sleeps = []
    monkeypatch.setattr(wst.time, "sleep", lambda s: sleeps.append(s))
    client = _SequenceClient([
        _resp(429, headers={"retry-after": "0"}),
        _resp(200, text="ok"),
    ])
    resp = wst.WebSearchToolsMixin._http_request_with_retry(client, "GET", "https://x/")
    assert resp.status_code == 200
    assert client.calls == 2
    assert sleeps  # still backed off


def test_wait_for_searxng_cancel_raises(monkeypatch):
    """ESC during the SearXNG readiness poll must abort with AgentCancelled
    instead of polling out the full 15s timeout."""

    def _no_server(*a, **k):
        raise httpx.ConnectError("no server")

    monkeypatch.setattr(wst.httpx, "get", _no_server)
    ce = threading.Event()
    ce.set()
    host = _Host()
    with pytest.raises(wst.AgentCancelled):
        host._wait_for_searxng("http://localhost:1", cancel_event=ce)


def test_web_fetch_cancel_mid_backoff_aborts(monkeypatch):
    """ESC during web_fetch's inline retry sleep must abort with AgentCancelled
    instead of sleeping out the Retry-After cap."""
    ce = threading.Event()

    def _late_set():
        time.sleep(0.05)
        ce.set()

    t = threading.Thread(target=_late_set, daemon=True)
    t.start()
    client = _FetchRetryClient([
        httpx.Response(
            429,
            request=httpx.Request("GET", "https://x/rate"),
            headers={"retry-after": "30", "content-type": "text/plain"},
        ),
        httpx.Response(
            200,
            request=httpx.Request("GET", "https://x/rate"),
            headers={"content-type": "text/plain; charset=utf-8"},
            text="finally ok",
        ),
    ])
    monkeypatch.setattr(wst.httpx, "Client", lambda *a, **k: client)
    import types

    host = _Host()
    host.config = types.SimpleNamespace(cancel_event=ce)
    with pytest.raises(wst.AgentCancelled):
        host._tool_web_fetch({"url": "https://x/rate"})
    t.join(timeout=5)
    assert client.calls == 1  # no retry after the interrupt


def test_search_web_propagates_agent_cancelled(monkeypatch):
    """A cancellation raised by a backend (ESC during SearXNG setup / retry)
    must propagate out of ``_tool_search_web`` — neither the parallel tier-1
    collector nor the sequential backend loop may swallow it into a
    'trying next backend' fallback."""
    host = _Host()

    def _boom(*a, **k):
        raise wst.AgentCancelled("cancelled by user")

    # Startpage is the unconditional first tier-1 backend; make it abort and
    # keep every other backend harmless so no straggler future holds a raised
    # AgentCancelled.
    monkeypatch.setattr(host, "_search_startpage", _boom)
    monkeypatch.setattr(host, "_search_exa", lambda *a, **k: [])
    monkeypatch.setattr(host, "_search_startpage_browser", lambda *a, **k: [])
    with pytest.raises(wst.AgentCancelled):
        host._tool_search_web({"query": "python asyncio"})
