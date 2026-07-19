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
def _reset_backend_cooldown():
    """Isolate the process-wide circuit-breaker state between tests."""
    wst.WebSearchToolsMixin._backend_cooldown.clear()
    yield
    wst.WebSearchToolsMixin._backend_cooldown.clear()


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
    only_title = _SP_RESULT.split("<style data-emotion=\"css 1507v2l\"")[0]
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
        pass
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
