"""Live proof of the browser computer-use loop: the pixel seen is the pixel clicked.

Everything else about this feature is unit-tested against fakes, and a fake can
agree with a wrong assumption. The one thing that has to be true in the world is
the coordinate contract, so this drives a REAL Chromium through the REAL
dispatch path:

    navigate -> screenshot -> the model reads the target's pixel off the image
             -> click_at(that pixel) -> the PAGE records where the click landed

The page records ``clientX/clientY`` of its own click event, so the assertion is
made by the browser, not by our code. Skipped where no browser is installed
(``PLAYWRIGHT_BROWSER_AVAILABLE``) — the loop is a demonstration, not a gate, and
a skip here must never be mistaken for a passing gate (the unit suite covers the
contract regardless).
"""

from __future__ import annotations

import base64
import json
import urllib.parse

import pytest

from external_llm.agent.tool_handlers import browser_tools
from external_llm.agent.tool_handlers.browser_tools import BrowserActionToolsMixin, _png_size

pytestmark = pytest.mark.integration

_live_browser = pytest.mark.skipif(
    not (browser_tools.HAS_PLAYWRIGHT and browser_tools.PLAYWRIGHT_BROWSER_AVAILABLE),
    reason="no Playwright browser on this machine (pip install playwright && playwright install chromium)",
)

# A green 160x60 target at CSS (400, 500) that reports where it was clicked, and
# a tall pad above it so a mis-scaled click cannot land there by accident.
_PAGE = """<!doctype html><html><body style="margin:0">
<div style="height:300px;background:#eee">padding</div>
<button id="t" style="position:absolute;left:400px;top:500px;width:160px;height:60px;
  background:#22c55e;border:0;font:20px sans-serif">TARGET</button>
<div id="log" style="position:absolute;left:10px;top:700px;font:16px monospace"></div>
<!-- In-flow height, so the document is actually scrollable: every element above
     is absolutely positioned, which alone leaves the page exactly viewport-tall
     and makes a wheel event unobservable (measured: window.scrollY stayed 0). -->
<div style="height:2000px"></div>
<script>
  document.getElementById('t').addEventListener('click', (e) => {
    document.getElementById('log').textContent = JSON.stringify({x: e.clientX, y: e.clientY});
  });
</script></body></html>"""


class _Host(BrowserActionToolsMixin):
    """The real mixin with only the result factory stubbed out."""

    def __init__(self, repo_root):
        self.repo_root = repo_root

    def _make_result(self, ok=False, content="", error=None, metadata=None, **kw):
        return {"ok": ok, "content": content, "error": error, "metadata": metadata or {}}


def _ok(host, args) -> dict:
    """Dispatch an action through the real path and require success."""
    res = host._tool_browser_action(args)
    assert res["ok"], res["error"]
    return res


def _rect(host) -> dict:
    return json.loads(
        _ok(
            host,
            {"action": "evaluate", "js": "JSON.stringify(document.getElementById('t').getBoundingClientRect())"},
        )["content"]
    )


@_live_browser
def test_click_at_lands_on_the_pixel_the_screenshot_showed(tmp_path):
    host = _Host(str(tmp_path))
    try:
        _ok(host, {"action": "navigate", "url": "data:text/html," + urllib.parse.quote(_PAGE), "wait_until": "load"})

        shot = _ok(host, {"action": "screenshot"})
        meta = shot["metadata"]
        payload = base64.b64decode(meta["attach_images"][0]["data"])

        # The attached bytes ARE the capture, and the geometry we report comes
        # from those bytes — not from a value we assumed alongside them.
        assert payload[:8] == b"\x89PNG\r\n\x1a\n"
        assert _png_size(payload) == (meta["image"]["width"], meta["image"]["height"])
        assert meta["scale"] == meta["image"]["width"] / meta["viewport"]["width"]

        rect = _rect(host)
        centre_css = (rect["left"] + rect["width"] / 2, rect["top"] + rect["height"] / 2)
        # What the MODEL does: measure the target on the screenshot it was shown.
        image_x = round(centre_css[0] * meta["scale"])
        image_y = round(centre_css[1] * meta["scale"])

        _ok(host, {"action": "click_at", "x": image_x, "y": image_y})

        recorded = _ok(host, {"action": "evaluate", "js": "document.getElementById('log').textContent"})["content"]
        assert recorded and recorded != "None", "the page saw no click at all"
        got = json.loads(recorded)

        # The browser's own account of the click, checked against the target it
        # drew — the assertion the feature exists to satisfy.
        assert rect["left"] <= got["x"] <= rect["left"] + rect["width"]
        assert rect["top"] <= got["y"] <= rect["top"] + rect["height"]
        assert (got["x"], got["y"]) == (image_x, image_y)
    finally:
        BrowserActionToolsMixin()._close_shared_browser()


@_live_browser
def test_keyboard_and_scroll_reach_the_page(tmp_path):
    host = _Host(str(tmp_path))
    try:
        _ok(host, {"action": "navigate", "url": "data:text/html," + urllib.parse.quote(_PAGE), "wait_until": "load"})
        _ok(host, {"action": "screenshot"})

        _ok(host, {"action": "click_at", "x": 480, "y": 530})
        _ok(host, {"action": "key", "keys": "Tab"})
        _ok(host, {"action": "type_text", "text": "abc"})
        _ok(host, {"action": "scroll_at", "x": 480, "y": 530, "dy": 250})

        scrolled = _ok(host, {"action": "evaluate", "js": "window.scrollY"})["content"]
        assert int(scrolled) > 0, "the wheel event never reached the document"
    finally:
        BrowserActionToolsMixin()._close_shared_browser()
