"""Computer use in the browser: see the page, then act where you saw it.

The loop this covers is ``screenshot -> click_at -> screenshot``. Two things
have to hold for it to be more than a demo:

* **The model must actually receive the pixels.** ``screenshot`` declares the
  PNG through ``image_transport``, so the image is attached to the conversation
  instead of leaving the model to reason from a file path.
* **A coordinate must land where the model pointed.** The model reports image
  pixels because that is what it was shown; the pointer moves in CSS pixels.
  The two differ whenever the browser context has a device scale factor, and
  the difference is invisible in a result that only echoes what was asked — so
  the scale is derived from the captured BYTES, applied on the way in, and both
  spaces are reported on the way out.

Also sealed here: the actions that ACT on a page are classified as mutating and
browser calls stay strictly ordered (one page is a sequence, not a set).
"""

from __future__ import annotations

import ast
import base64
import inspect
import struct
from unittest import mock

import pytest

from external_llm.agent.tool_handlers import browser_tools
from external_llm.agent.tool_handlers.browser_tools import (
    INTERACTION_ACTIONS,
    BrowserActionToolsMixin,
    _as_coordinate,
    _png_size,
)
from external_llm.agent.tool_registry import ToolRegistry
from external_llm.agent.tool_schemas import SCHEMA_BROWSER_ACTION


def _png(width: int, height: int) -> bytes:
    """A byte string whose IHDR says width x height — all _png_size reads."""
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 13) + b"IHDR" + ihdr + b"\x00\x00\x00\x00"


class _Mouse:
    def __init__(self):
        self.calls: list = []

    def move(self, x, y, steps=None):
        self.calls.append(("move", x, y, steps))

    def click(self, x, y, button="left"):
        self.calls.append(("click", x, y, button))

    def dblclick(self, x, y, button="left"):
        self.calls.append(("dblclick", x, y, button))

    def down(self):
        self.calls.append(("down",))

    def up(self):
        self.calls.append(("up",))

    def wheel(self, dx, dy):
        self.calls.append(("wheel", dx, dy))


class _Keyboard:
    def __init__(self):
        self.calls: list = []

    def press(self, keys):
        self.calls.append(("press", keys))

    def type(self, text, delay=None):
        self.calls.append(("type", text, delay))


class _Page:
    """Fake Playwright page: records pointer/keyboard events, writes real bytes."""

    def __init__(self, *, image_size=(1280, 800), viewport=(1280, 800), url="https://x/"):
        self._image_size = image_size
        self.viewport_size = {"width": viewport[0], "height": viewport[1]}
        self._url = url
        self.mouse = _Mouse()
        self.keyboard = _Keyboard()
        self.shot = None
        self.load_state_waits = 0
        # Scripted answers for the scroll watch: what the counter reads next.
        self.scroll_reads: list[int] = []

    @property
    def url(self):
        return self._url

    def is_closed(self):
        return False

    def screenshot(self, path=None, full_page=None):
        self.shot = {"path": path, "full_page": full_page}
        with open(path, "wb") as fh:
            fh.write(_png(*self._image_size))

    def wait_for_load_state(self, state):
        self.load_state_waits += 1

    def evaluate(self, js):
        if js == browser_tools._SCROLL_WATCH_ARM_JS:
            return None
        if js == browser_tools._SCROLL_WATCH_READ_JS:
            return self.scroll_reads.pop(0) if self.scroll_reads else 0
        raise AssertionError(f"unexpected evaluate: {js!r}")

    def title(self):
        return "T"

    def inner_text(self, selector):
        return "body"


class _Host(BrowserActionToolsMixin):
    """Concrete host keeping the REAL handlers; the page is injected."""

    def __init__(self, page, repo_root):
        self._page_obj = page
        self.repo_root = repo_root

    def _make_result(self, ok=False, content="", error=None, metadata=None, **kw):
        return {"ok": ok, "content": content, "error": error, "metadata": metadata or {}}

    def _get_page(self):
        return self._page_obj


@pytest.fixture(autouse=True)
def _reset_class_state(monkeypatch):
    monkeypatch.setattr(BrowserActionToolsMixin, "_view_scale", None, raising=False)
    monkeypatch.setattr(BrowserActionToolsMixin, "_headless", True, raising=False)
    yield


@pytest.fixture
def page():
    return _Page()


@pytest.fixture
def host(page, tmp_path):
    return _Host(page, str(tmp_path))


def _attachment(result) -> dict:
    images = result["metadata"]["attach_images"]
    assert len(images) == 1
    return images[0]


# ══════════════════════════════════════════════════════════════════════════════
# 1. Geometry helpers
# ══════════════════════════════════════════════════════════════════════════════


def test_png_size_reads_the_ihdr():
    assert _png_size(_png(320, 200)) == (320, 200)


@pytest.mark.parametrize("payload", [b"", b"not a png at all", b"\x89PNG\r\n\x1a\n" + b"\x00" * 8])
def test_png_size_degrades_to_unknown(payload):
    assert _png_size(payload) == (0, 0)


@pytest.mark.parametrize("value", [None, "10", True, False, float("nan"), float("inf"), [1]])
def test_as_coordinate_rejects_non_coordinates(value):
    assert _as_coordinate(value) is None


def test_as_coordinate_accepts_numbers():
    assert _as_coordinate(10) == 10.0
    assert _as_coordinate(10.5) == 10.5
    assert _as_coordinate(0) == 0.0


# ══════════════════════════════════════════════════════════════════════════════
# 2. The screenshot hands over the pixels
# ══════════════════════════════════════════════════════════════════════════════


def test_screenshot_attaches_the_image_it_captured(host, page):
    res = host._browser_screenshot({})

    assert res["ok"]
    image = _attachment(res)
    assert image["media_type"] == "image/png"
    assert base64.b64decode(image["data"]) == _png(1280, 800)
    assert res["metadata"]["filepath"] in res["content"]


def test_screenshot_caption_states_the_coordinate_contract(host):
    image = _attachment(host._browser_screenshot({}))

    assert "IMAGE pixels" in image["caption"]
    assert "click_at" in image["caption"]


def test_screenshot_geometry_is_reported(host):
    meta = host._browser_screenshot({})["metadata"]

    assert meta["image"] == {"width": 1280, "height": 800}
    assert meta["viewport"] == {"width": 1280, "height": 800}
    assert meta["scale"] == 1.0
    assert meta["full_page"] is False


def test_screenshot_defaults_to_the_viewport(host, page):
    """A full-page image cannot be clicked on: its y is page-absolute."""
    host._browser_screenshot({})

    assert page.shot["full_page"] is False


def test_full_page_screenshot_warns_that_its_coordinates_are_not_clickable(host):
    res = host._browser_screenshot({"full_page": True})

    assert _attachment(res)["caption"].count("viewport screenshot (full_page=false)") == 1
    assert "NOT where the pointer is" in _attachment(res)["caption"]


def test_screenshot_that_cannot_be_read_back_says_so_instead_of_failing(host, monkeypatch):
    """The capture succeeded — only the hand-over failed. A failure here would
    send the agent to retake a screenshot it already has, so the result stays
    ok=True and states that the pixels were NOT attached."""
    monkeypatch.setattr(browser_tools.pathlib.Path, "read_bytes", mock.Mock(side_effect=OSError("gone")))

    res = host._browser_screenshot({})

    assert res["ok"] is True
    assert "NOT attached" in res["content"]
    assert "read_image" in res["content"]
    assert res["metadata"] == {
        "filepath": res["metadata"]["filepath"],
        "url": "https://x/",
        "full_page": False,
        "attached": False,
    }
    assert "attach_images" not in res["metadata"]


# ══════════════════════════════════════════════════════════════════════════════
# 3. Coordinates land where the model pointed
# ══════════════════════════════════════════════════════════════════════════════


def test_coordinates_are_image_pixels_at_scale_1(host, page):
    host._browser_screenshot({})

    host._browser_click_at({"x": 640, "y": 400})

    assert ("click", 640.0, 400.0, "left") in page.mouse.calls


def test_coordinates_are_scaled_down_from_a_retina_capture(tmp_path):
    """A 2x image means image pixel 200 is CSS pixel 100. The model clicked what
    it saw, so the pointer must follow the image, not the raw number."""
    page = _Page(image_size=(2560, 1600), viewport=(1280, 800))
    host = _Host(page, str(tmp_path))

    assert host._browser_screenshot({})["metadata"]["scale"] == 2.0

    host._browser_click_at({"x": 200, "y": 100})

    assert ("click", 100.0, 50.0, "left") in page.mouse.calls


def test_scale_survives_until_the_next_screenshot(host, page):
    host._browser_screenshot({})
    host._browser_mouse_move({"x": 10, "y": 20})
    host._browser_mouse_move({"x": 30, "y": 40})

    assert ("move", 10.0, 20.0, None) in page.mouse.calls
    assert ("move", 30.0, 40.0, None) in page.mouse.calls


def test_result_reports_both_coordinate_spaces(host):
    """A mis-scaled pointer is invisible in a result that only echoes the ask."""
    host._browser_screenshot({})

    res = host._browser_click_at({"x": 640, "y": 400})

    assert "image (640, 400)" in res["content"]
    assert "CSS (640, 400)" in res["content"]


def test_click_without_coordinates_is_an_actionable_error(host):
    res = host._browser_click_at({})

    assert res["ok"] is False
    assert "'x' and 'y' are required" in res["error"]


def test_click_rejects_a_string_coordinate(host):
    res = host._browser_click_at({"x": "640", "y": 400})

    assert res["ok"] is False
    assert "'x' and 'y' are required" in res["error"]


def test_browser_close_resets_the_scale(monkeypatch):
    """A new browser is a new page; an inherited scale would misplace clicks."""
    monkeypatch.setattr(BrowserActionToolsMixin, "_view_scale", 2.0, raising=False)

    BrowserActionToolsMixin()._close_shared_browser()

    assert BrowserActionToolsMixin._view_scale is None


# ══════════════════════════════════════════════════════════════════════════════
# 4. Pointer and keyboard actions
# ══════════════════════════════════════════════════════════════════════════════


def test_double_click_uses_dblclick(host, page):
    host._browser_double_click_at({"x": 5, "y": 6})

    assert ("dblclick", 5.0, 6.0, "left") in page.mouse.calls


def test_right_click_passes_the_button(host, page):
    host._browser_right_click_at({"x": 5, "y": 6})

    assert ("click", 5.0, 6.0, "right") in page.mouse.calls


def test_click_rejects_an_unknown_button(host, page):
    res = host._browser_click_at({"x": 1, "y": 2, "button": "thumb"})

    assert res["ok"] is False
    assert "left, middle, right" in res["error"]
    assert page.mouse.calls == []


def test_click_settles_the_load_state(host, page):
    host._browser_click_at({"x": 1, "y": 2})

    assert page.load_state_waits == 1


def test_drag_presses_moves_then_releases(host, page):
    host._browser_drag({"x": 10, "y": 20, "to_x": 100, "to_y": 200, "steps": 7})

    assert page.mouse.calls == [
        ("move", 10.0, 20.0, None),
        ("down",),
        ("move", 100.0, 200.0, 7),
        ("up",),
    ]


def test_drag_requires_both_ends(host, page):
    res = host._browser_drag({"x": 1, "y": 2})

    assert res["ok"] is False
    assert "'to_x'/'to_y'" in res["error"]
    assert page.mouse.calls == []


def test_drag_clamps_the_step_count(host, page):
    host._browser_drag({"x": 0, "y": 0, "to_x": 9, "to_y": 9, "steps": 9999})

    assert ("move", 9.0, 9.0, 100) in page.mouse.calls


def test_scroll_moves_the_pointer_first(host, page):
    """A wheel event scrolls whatever pane the pointer is over."""
    host._browser_scroll_at({"x": 100, "y": 200, "dy": 500})

    assert page.mouse.calls == [("move", 100.0, 200.0, None), ("wheel", 0.0, 500.0)]


def test_scroll_requires_dy(host):
    res = host._browser_scroll_at({"x": 1, "y": 2})

    assert res["ok"] is False
    assert "'dy' is required" in res["error"]


def test_key_presses_a_chord(host, page):
    res = host._browser_key({"keys": "Control+a"})

    assert res["ok"]
    assert page.keyboard.calls == [("press", "Control+a")]


def test_key_accepts_the_singular_alias(host, page):
    host._browser_key({"key": "Enter"})

    assert page.keyboard.calls == [("press", "Enter")]


def test_key_requires_a_key(host):
    res = host._browser_key({})

    assert res["ok"] is False
    assert "'keys' is required" in res["error"]


def test_type_text_types_at_the_focus(host, page):
    host._browser_type_text({"text": "hello"})

    assert page.keyboard.calls == [("type", "hello", 0)]


def test_type_text_rejects_empty_text(host, page):
    res = host._browser_type_text({"text": ""})

    assert res["ok"] is False
    assert page.keyboard.calls == []


# ── the wheel is applied asynchronously: the loop MUST wait for it ──────────


def test_scroll_at_arms_the_watch_before_the_wheel_and_settles_after(host, page):
    """Ordering is the whole point: arming after the wheel would miss the event,
    and settling before it would measure nothing."""
    order: list[str] = []
    monkeypatch_target = page.mouse
    monkeypatch_target.wheel = lambda dx, dy: order.append("wheel")
    host._arm_scroll_watch = lambda _page: (order.append("arm"), 0)[1]
    host._settle_scroll = lambda _page, _armed: order.append("settle")

    host._browser_scroll_at({"x": 1, "y": 2, "dy": 3})

    assert order == ["arm", "wheel", "settle"]


def test_settle_waits_until_the_counter_stops_moving(host, page):
    # armed_at=0, then the counter climbs and stops at 2.
    page.scroll_reads = [1, 2, 2]

    host._settle_scroll(page, 0)

    assert page.scroll_reads == []  # it polled through the climb and one stable read


def test_settle_returns_after_one_poll_when_nothing_moved(host, page):
    page.scroll_reads = [0, 0, 0]

    host._settle_scroll(page, 0)

    # Two reads only (the poll, then the equality check) — a pane that does not
    # scroll must not cost the full timeout.
    assert len(page.scroll_reads) == 2


def test_settle_is_a_noop_when_the_watch_could_not_be_armed(host, page):
    host._settle_scroll(page, -1)

    assert page.scroll_reads == []  # never even polled


def test_arm_returns_minus_one_when_evaluate_fails(host):
    class _Refusing(_Page):
        def evaluate(self, js):
            raise RuntimeError("frame detached")

    assert host._arm_scroll_watch(_Refusing()) == -1


def test_settle_gives_up_when_a_later_read_fails(host, page):
    class _Flaky(_Page):
        def evaluate(self, js):
            if js == browser_tools._SCROLL_WATCH_READ_JS:
                raise RuntimeError("gone")

    host._settle_scroll(_Flaky(), 0)  # must not raise


def test_scroll_at_still_reports_success_when_the_watch_is_unavailable(host, page, monkeypatch):
    monkeypatch.setattr(host, "_arm_scroll_watch", lambda _page: -1)

    res = host._browser_scroll_at({"x": 1, "y": 2, "dy": 3})

    assert res["ok"] is True


# ══════════════════════════════════════════════════════════════════════════════
# 5. Headless / headed mode
# ══════════════════════════════════════════════════════════════════════════════


def test_mode_switch_restarts_the_browser(host):
    host._ensure_browser_mode(False)

    assert BrowserActionToolsMixin._headless is False


def test_mode_that_matches_is_a_noop(monkeypatch):
    closed = []
    monkeypatch.setattr(BrowserActionToolsMixin, "_close_shared_browser", lambda self: closed.append(1))

    BrowserActionToolsMixin()._ensure_browser_mode(True)

    assert closed == []


def test_omitting_the_mode_keeps_the_running_browser(monkeypatch):
    closed = []
    monkeypatch.setattr(BrowserActionToolsMixin, "_close_shared_browser", lambda self: closed.append(1))

    BrowserActionToolsMixin()._ensure_browser_mode(None)

    assert closed == []


def test_mode_switch_closes_before_relaunch(monkeypatch):
    """The teardown must happen BEFORE the flag flips: ``_get_browser`` builds
    from the flag, so a flip without a close would leave the old browser
    running while later calls believe they are in the new mode."""
    order = []
    monkeypatch.setattr(BrowserActionToolsMixin, "_close_shared_browser", lambda self: order.append("close"))

    host = BrowserActionToolsMixin()
    host._ensure_browser_mode(False)
    order.append(f"headless={BrowserActionToolsMixin._headless}")

    assert order == ["close", "headless=False"]


# ══════════════════════════════════════════════════════════════════════════════
# 6. An act on a page is not an observation
# ══════════════════════════════════════════════════════════════════════════════


def _registry() -> ToolRegistry:
    reg = ToolRegistry.__new__(ToolRegistry)
    reg._WRITE_TOOLS = set()
    return reg


@pytest.mark.parametrize("action", sorted(INTERACTION_ACTIONS))
def test_interaction_actions_mutate(action):
    assert _registry()._tool_call_mutates("browser_action", {"action": action}) is True


@pytest.mark.parametrize(
    "action",
    ["navigate", "extract", "screenshot", "evaluate", "wait", "mouse_move", "close"],
)
def test_observing_actions_do_not_mutate(action):
    assert _registry()._tool_call_mutates("browser_action", {"action": action}) is False


def test_an_unknown_browser_action_is_rejected_before_any_handler():
    """Not mutating, and that is safe only because dispatch refuses the call:
    ``_tool_browser_action`` looks the verb up in its closed table and returns
    "Unknown action" without reaching a handler. The classifier's job is to
    describe the acts that CAN run — the same argument ``is_read_only_call``
    makes for malformed args, which ``dispatch`` also rejects first.
    """
    host = _Host(_Page(), ".")

    assert _registry()._tool_call_mutates("browser_action", {"action": "never-heard-of-it"}) is False
    res = host._tool_browser_action({"action": "never-heard-of-it"})
    assert res["ok"] is False
    assert "Unknown action" in res["error"]


def test_an_unknown_action_is_rejected_before_playwright_is_consulted(monkeypatch):
    """A typo must be a typo in EVERY environment — including one without Playwright.

    The action table is pure argument validation, so it is consulted first.
    That ordering is not cosmetic: the Playwright branch below it asks the user
    for permission to ``pip install playwright``. A malformed call must not be
    able to raise that prompt or start that install for a verb that was never
    an action — and it must not answer a typo with "Playwright is not
    available", which is the wrong thing to send anyone off to fix.
    """
    monkeypatch.setattr(browser_tools, "HAS_PLAYWRIGHT", False)
    monkeypatch.setattr(browser_tools, "_ensure_playwright_imported", lambda: False)

    consulted: list = []

    def _resolve_dependency(self):
        consulted.append(True)
        return False

    monkeypatch.setattr(BrowserActionToolsMixin, "_ensure_playwright_installed", _resolve_dependency)
    host = _Host(_Page(), ".")

    res = host._tool_browser_action({"action": "never-heard-of-it"})

    assert res["ok"] is False
    assert "Unknown action" in res["error"]
    assert consulted == [], "an unknown action must not reach dependency resolution"

    # Non-vacuity: the same armed guard DOES fire for a real action, so the
    # assertion above is about ordering, not about a branch that never runs.
    res = host._tool_browser_action({"action": "navigate", "url": "https://example.com"})

    assert res["ok"] is False
    assert "Playwright is not available" in res["error"]
    assert consulted == [True]


def _dispatched_browser_actions() -> set[str]:
    """The dispatch table as the SOURCE declares it.

    Read from the AST rather than by calling the handler: the table holds bound
    methods and is built per call, so running it would need a live host — and a
    gate that runs the code it is checking cannot notice a verb the code lost.
    """
    tree = ast.parse(inspect.getsource(browser_tools))
    tables = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Dict)
        and any(isinstance(target, ast.Name) and target.id == "_actions" for target in node.targets)
    ]
    assert len(tables) == 1, f"expected exactly one _actions table, found {len(tables)}"
    return {key.value for key in tables[0].value.keys}


def test_the_dispatched_actions_are_exactly_what_the_schema_offers():
    """Three lists describe one closed set — the dispatcher, the enum the model
    is shown, and the mutating subset — so they are compared, not trusted.

    Drift is silent in both directions: a verb in the enum but not the table is
    a call the model can make and that can only ever error, while a verb in the
    table but not the enum is an action the model cannot ask for. The mutating
    set is the subset that must BE offerable, since only an offerable verb can
    be classified before it runs.
    """
    enum = set(SCHEMA_BROWSER_ACTION["parameters"]["properties"]["action"]["enum"])

    assert enum == _dispatched_browser_actions()
    assert enum >= INTERACTION_ACTIONS, "every act must be one the model can ask for"


def test_browser_calls_are_always_serial():
    """One page is a sequence, not a set: [screenshot, click_at] must not be
    able to run the click before the capture it was aimed at."""
    reg = _registry()

    assert reg._tool_call_is_serial("browser_action", {"action": "click_at"}) is True
    assert reg._tool_call_is_serial("browser_action", {"action": "extract"}) is True
