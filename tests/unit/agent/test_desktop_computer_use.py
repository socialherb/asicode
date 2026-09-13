"""``computer`` — the host desktop: three gates, none of which may fail open.

The machine this drives is the user's own. Every refusal here is deliberate:

* **Off by default.** Upgrading the tool must not be the same act as handing a
  model the mouse.
* **Permission preflight.** Without Screen Recording ``screencapture`` succeeds
  and writes a PNG of the wallpaper alone; without Accessibility ``CGEventPost``
  succeeds and the events are dropped. Both failures are invisible from inside
  the conversation, so both gates are checked BEFORE the act, with the calls
  that ask rather than prompt.
* **A closed action table.** An action nobody classified is refused, not
  guessed at — and the interaction actions are named so a click on the user's
  real screen can never be classified as an observation.

The permission state of the machine running these tests is irrelevant: the
grant checks are monkeypatched, and the capture/synthesis edges are injectable
(``runner``, ``set_event_poster``), which is what makes the whole path testable
on a CI runner where no grant will ever exist.
"""

from __future__ import annotations

import struct
from types import SimpleNamespace
from unittest import mock

import pytest

from external_llm.agent import desktop_backend as backend
from external_llm.agent.desktop_backend import KEYCODES, MODIFIER_FLAGS, DisplayGeometry
from external_llm.agent.tool_handlers import desktop_tools
from external_llm.agent.tool_handlers.desktop_tools import (
    DESKTOP_INTERACTION_ACTIONS,
    DESKTOP_OBSERVATION_ACTIONS,
    DesktopToolsMixin,
)

_RETINA = DisplayGeometry(display_id=1, pixel_width=2560, pixel_height=1440, point_width=1280, point_height=720)
_OFFSET = DisplayGeometry(
    display_id=1,
    pixel_width=1920,
    pixel_height=1080,
    point_width=1920,
    point_height=1080,
    point_origin_x=1920,
    point_origin_y=0,
)


def _png(width: int, height: int) -> bytes:
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 13) + b"IHDR" + ihdr + b"\x00\x00\x00\x00"


class _Host(DesktopToolsMixin):
    repo_root = "."

    def _make_result(self, ok=False, content="", error=None, metadata=None, **kw):
        return {"ok": ok, "content": content, "error": error, "metadata": metadata or {}}


@pytest.fixture
def granted(monkeypatch, tmp_path):
    """Opted in, both permissions granted, captures write into tmp_path."""
    monkeypatch.setenv(desktop_tools.OPT_IN_ENV, "1")
    monkeypatch.setattr(desktop_tools, "screen_recording_granted", lambda: True)
    monkeypatch.setattr(desktop_tools, "accessibility_granted", lambda: True)
    monkeypatch.setattr(desktop_tools, "main_display_geometry", lambda: _RETINA)
    monkeypatch.setattr(desktop_tools, "capture_main_display", lambda: (_png(2560, 1440), ""))
    return _Host()


@pytest.fixture(autouse=True)
def _reset_poster():
    yield
    backend.set_event_poster(None)


# ══════════════════════════════════════════════════════════════════════════════
# 1. Permission preflight fails closed
# ══════════════════════════════════════════════════════════════════════════════


def test_screen_recording_is_false_when_the_framework_is_missing(monkeypatch):
    monkeypatch.setattr(backend, "_core_graphics", lambda: None)

    assert backend.screen_recording_granted() is False


def test_accessibility_is_false_when_the_framework_is_missing(monkeypatch):
    monkeypatch.setattr(backend, "_application_services", lambda: None)

    assert backend.accessibility_granted() is False


def test_capture_refuses_without_screen_recording_and_never_runs_screencapture(monkeypatch):
    monkeypatch.setattr(backend, "screen_recording_granted", lambda: False)
    runner = mock.Mock()

    png, error = backend.capture_main_display(runner=runner)

    assert png is None
    assert "Screen Recording" in error
    assert "Settings" in error  # the operator gets the path, not just "denied"
    runner.assert_not_called()


def test_capture_reports_an_unreadable_display_geometry(monkeypatch):
    monkeypatch.setattr(backend, "screen_recording_granted", lambda: True)
    monkeypatch.setattr(backend, "main_display_geometry", lambda: None)

    png, error = backend.capture_main_display(runner=mock.Mock())

    assert png is None
    assert "display geometry" in error


# ══════════════════════════════════════════════════════════════════════════════
# 2. Capture argv: -R is in POINTS, which is what makes the PNG 2x on Retina
# ══════════════════════════════════════════════════════════════════════════════


def test_capture_asks_for_the_display_in_points(monkeypatch, tmp_path):
    monkeypatch.setattr(backend, "screen_recording_granted", lambda: True)
    monkeypatch.setattr(backend, "main_display_geometry", lambda: _RETINA)
    monkeypatch.setattr(backend, "_capture_dir", lambda: str(tmp_path))
    captured: dict = {}

    def runner(argv, **kw):
        captured["argv"] = argv
        captured["kwargs"] = kw
        target = argv[-1]
        with open(target, "wb") as fh:
            fh.write(_png(2560, 1440))
        return SimpleNamespace(returncode=0, stderr="")

    png, error = backend.capture_main_display(runner=runner)

    assert error == ""
    assert png == _png(2560, 1440)
    argv = captured["argv"]
    assert argv[0] == backend._SCREENCAPTURE
    assert "-x" in argv  # no shutter sound
    assert argv[argv.index("-t") + 1] == "png"
    # 1280x720 POINTS — a Retina capture therefore writes a 2560x1440 PNG, which
    # is the image-pixel space the model measures on.
    assert argv[argv.index("-R") + 1] == "0,0,1280,720"
    assert captured["kwargs"]["timeout"] == backend._RUN_TIMEOUT_SEC
    assert captured["kwargs"]["check"] is False


def test_capture_reports_a_nonzero_exit_with_its_stderr(monkeypatch, tmp_path):
    monkeypatch.setattr(backend, "screen_recording_granted", lambda: True)
    monkeypatch.setattr(backend, "main_display_geometry", lambda: _RETINA)
    monkeypatch.setattr(backend, "_capture_dir", lambda: str(tmp_path))

    png, error = backend.capture_main_display(
        runner=lambda argv, **kw: SimpleNamespace(returncode=1, stderr="no display")
    )

    assert png is None
    assert "rc=1" in error
    assert "no display" in error


def test_capture_reports_a_runner_that_cannot_start(monkeypatch):
    monkeypatch.setattr(backend, "screen_recording_granted", lambda: True)
    monkeypatch.setattr(backend, "main_display_geometry", lambda: _RETINA)

    def runner(argv, **kw):
        raise FileNotFoundError("screencapture")

    png, error = backend.capture_main_display(runner=runner)

    assert png is None
    assert "FileNotFoundError" in error


def test_capture_reports_success_with_no_readable_file(monkeypatch, tmp_path):
    monkeypatch.setattr(backend, "screen_recording_granted", lambda: True)
    monkeypatch.setattr(backend, "main_display_geometry", lambda: _RETINA)
    monkeypatch.setattr(backend, "_capture_dir", lambda: str(tmp_path))

    png, error = backend.capture_main_display(runner=lambda argv, **kw: SimpleNamespace(returncode=0, stderr=""))

    assert png is None
    assert "could not be read" in error


# ══════════════════════════════════════════════════════════════════════════════
# 3. The coordinate bridge
# ══════════════════════════════════════════════════════════════════════════════


def test_to_points_divides_by_the_backing_scale():
    assert backend.to_points(_RETINA, 200, 100) == (100.0, 50.0)


def test_to_points_is_identity_at_scale_one():
    assert backend.to_points(_OFFSET, 200, 100) == (2120.0, 100.0)


def test_to_points_without_geometry_still_returns_a_point():
    assert backend.to_points(None, 200, 100) == (200.0, 100.0)


def test_display_scale_is_derived_from_the_two_spaces():
    assert _RETINA.scale == 2.0
    assert _OFFSET.scale == 1.0


def test_display_scale_degrades_to_one_when_a_dimension_is_unknown():
    assert DisplayGeometry(1, 0, 0, 0, 0).scale == 1.0


# ══════════════════════════════════════════════════════════════════════════════
# 4. Event synthesis (poster injected — no input reaches the real machine)
# ══════════════════════════════════════════════════════════════════════════════


def test_unknown_mouse_event_is_refused():
    with pytest.raises(ValueError, match="unknown mouse event"):
        backend.post_mouse("teleport", (1, 2))


def test_post_key_attaches_the_modifier_flags(monkeypatch):
    """A chord must ride on the event: posting a modifier key separately would
    leave Shift held down for whatever runs next."""
    seen: list = []
    monkeypatch.setattr(backend, "_new_mouse_event", mock.Mock())
    fake_cg = SimpleNamespace(
        CGEventCreateKeyboardEvent=lambda _n, code, down: {"code": code, "down": down},
        CGEventSetFlags=lambda event, flags: event.update(flags=flags),
    )
    monkeypatch.setattr(backend, "_core_graphics", lambda: fake_cg)
    backend.set_event_poster(lambda event: seen.append(event))

    backend.post_key(48, flags=backend.MODIFIER_FLAGS["cmd"], down=True)

    assert seen == [{"code": 48, "down": True, "flags": backend.MODIFIER_FLAGS["cmd"]}]


def test_post_text_chunks_a_long_string(monkeypatch):
    """Chunked so the window server does not drop a large single event."""
    posted: list = []

    class _FakeCG:
        class CGEventKeyboardSetUnicodeString:
            pass

    fake = SimpleNamespace(
        CGEventCreateKeyboardEvent=lambda _n, _code, down: {"down": down},
        CGEventKeyboardSetUnicodeString=lambda event, length, buf: event.update(text=length),
    )
    monkeypatch.setattr(backend, "_core_graphics", lambda: fake)
    backend.set_event_poster(lambda event: posted.append(event))

    count = backend.post_text("x" * 100)

    assert count == 100
    # One down + one up per chunk, 100 chars over 64-char chunks = 2 chunks.
    assert len(posted) == 4
    assert sum(e.get("text", 0) for e in posted) == 100


def test_post_text_returns_none_without_coregraphics(monkeypatch):
    monkeypatch.setattr(backend, "_core_graphics", lambda: None)

    assert backend.post_text("hi") is None


def test_cursor_position_is_reported_in_image_pixels(monkeypatch):
    fake = SimpleNamespace(
        CGEventCreate=lambda _n: "event",
        CGEventGetLocation=lambda _e: SimpleNamespace(x=100.0, y=50.0),
    )
    monkeypatch.setattr(backend, "_core_graphics", lambda: fake)
    monkeypatch.setattr(backend, "main_display_geometry", lambda: _RETINA)

    # 100 points on a 2x display is image pixel 200 — the space the model sees.
    assert backend.cursor_position_image() == (200.0, 100.0)


# ══════════════════════════════════════════════════════════════════════════════
# 5. The tool: opt-in first, then permissions
# ══════════════════════════════════════════════════════════════════════════════


def test_the_tool_is_off_unless_opted_in(monkeypatch):
    monkeypatch.delenv(desktop_tools.OPT_IN_ENV, raising=False)
    host = _Host()

    res = host._tool_computer({"action": "screenshot"})

    assert res["ok"] is False
    assert desktop_tools.OPT_IN_ENV in res["error"]
    assert "browser_action" in res["error"]  # the safer alternative is named


def test_opt_in_alone_does_not_grant_screen_recording(monkeypatch):
    monkeypatch.setenv(desktop_tools.OPT_IN_ENV, "1")
    monkeypatch.setattr(desktop_tools, "screen_recording_granted", lambda: False)

    res = _Host()._tool_computer({"action": "screenshot"})

    assert res["ok"] is False
    assert "Screen Recording" in res["error"]


def test_observation_needs_no_accessibility(monkeypatch, granted):
    monkeypatch.setattr(desktop_tools, "accessibility_granted", lambda: False)

    res = granted._tool_computer({"action": "screenshot"})

    assert res["ok"] is True


def test_interaction_needs_accessibility(monkeypatch):
    monkeypatch.setenv(desktop_tools.OPT_IN_ENV, "1")
    monkeypatch.setattr(desktop_tools, "accessibility_granted", lambda: False)

    res = _Host()._tool_computer({"action": "click", "x": 1, "y": 2})

    assert res["ok"] is False
    assert "Accessibility" in res["error"]


def test_unknown_action_is_refused():
    res = _Host()._tool_computer({"action": "levitate"})

    assert res["ok"] is False
    assert "Unknown action" in res["error"]


def test_empty_action_lists_the_table():
    res = _Host()._tool_computer({})

    assert res["ok"] is False
    assert "screenshot" in res["error"]


# ══════════════════════════════════════════════════════════════════════════════
# 6. Actions through the tool
# ══════════════════════════════════════════════════════════════════════════════


def test_screenshot_attaches_the_capture(granted):
    res = granted._tool_computer({"action": "screenshot"})

    assert res["ok"] is True
    image = res["metadata"]["attach_images"][0]
    assert image["media_type"] == "image/png"
    assert res["metadata"]["scale"] == 2.0
    assert "IMAGE pixels" in image["caption"]


def test_click_posts_a_press_and_release_at_the_scaled_point(granted, monkeypatch):
    events: list = []
    monkeypatch.setattr(desktop_tools, "post_mouse", lambda name, point: events.append((name, point)))

    res = granted._tool_computer({"action": "click", "x": 200, "y": 100})

    assert res["ok"] is True
    assert events == [("mouse_moved", (100.0, 50.0)), ("left_down", (100.0, 50.0)), ("left_up", (100.0, 50.0))]
    assert "image (200, 100)" in res["content"]
    assert "screen (100, 50)" in res["content"]


def test_double_click_posts_two_pairs(granted, monkeypatch):
    events: list = []
    monkeypatch.setattr(desktop_tools, "post_mouse", lambda name, point: events.append(name))

    granted._tool_computer({"action": "double_click", "x": 10, "y": 10})

    assert events == ["mouse_moved", "left_down", "left_up", "left_down", "left_up"]


def test_right_click_uses_the_right_button(granted, monkeypatch):
    events: list = []
    monkeypatch.setattr(desktop_tools, "post_mouse", lambda name, point: events.append(name))

    granted._tool_computer({"action": "right_click", "x": 10, "y": 10})

    assert events == ["mouse_moved", "right_down", "right_up"]


def test_drag_interpolates_between_the_ends(granted, monkeypatch):
    events: list = []
    monkeypatch.setattr(desktop_tools, "post_mouse", lambda name, point: events.append((name, point)))

    granted._tool_computer({"action": "drag", "x": 0, "y": 0, "to_x": 120, "to_y": 60})

    names = [n for n, _ in events]
    assert names[0] == "mouse_moved" and names[1] == "left_down" and names[-1] == "left_up"
    drags = [p for n, p in events if n == "left_drag"]
    assert len(drags) == desktop_tools._DRAG_STEPS
    assert drags[-1] == (60.0, 30.0)  # 120 image px is 60 points
    # Monotonic, so a target watching pointermove sees a real gesture.
    assert drags == sorted(drags)


def test_scroll_requires_dy_but_not_dx(granted, monkeypatch):
    calls: list = []
    monkeypatch.setattr(desktop_tools, "post_mouse", lambda *_a: None)
    monkeypatch.setattr(desktop_tools, "post_scroll", lambda point, dx, dy: calls.append((point, dx, dy)))

    missing = granted._tool_computer({"action": "scroll", "x": 5, "y": 5})
    ok = granted._tool_computer({"action": "scroll", "x": 5, "y": 5, "dy": -300})

    assert missing["ok"] is False
    assert ok["ok"] is True
    assert calls == [((2.5, 2.5), 0.0, -300.0)]


def test_type_uses_unicode_synthesis(granted, monkeypatch):
    typed: list = []
    monkeypatch.setattr(desktop_tools, "post_text", lambda text: typed.append(text) or len(text))

    res = granted._tool_computer({"action": "type", "text": "한글 café 🌱"})

    assert res["ok"] is True
    assert typed == ["한글 café 🌱"]


def test_type_refuses_empty_text(granted):
    res = granted._tool_computer({"action": "type", "text": ""})

    assert res["ok"] is False


def test_key_posts_a_chord_with_flags(granted, monkeypatch):
    seen: list = []
    monkeypatch.setattr(desktop_tools, "post_key", lambda code, flags=0, down=True: seen.append((code, flags, down)))

    res = granted._tool_computer({"action": "key", "keys": "cmd+escape"})

    assert res["ok"] is True
    assert seen == [
        (KEYCODES["escape"], MODIFIER_FLAGS["cmd"], True),
        (KEYCODES["escape"], MODIFIER_FLAGS["cmd"], False),
    ]


def test_a_chord_may_stack_modifiers(granted, monkeypatch):
    seen: list = []
    monkeypatch.setattr(desktop_tools, "post_key", lambda code, flags=0, down=True: seen.append(flags))

    granted._tool_computer({"action": "key", "keys": "cmd+shift+escape"})

    assert seen[0] == MODIFIER_FLAGS["cmd"] | MODIFIER_FLAGS["shift"]


def test_a_non_modifier_before_the_key_is_named_in_the_error(granted):
    """'cmd+nonsense+escape' must not be silently joined into a name no table
    has — the refusal says which element is wrong."""
    res = granted._tool_computer({"action": "key", "keys": "cmd+nonsense+escape"})

    assert res["ok"] is False
    assert "'nonsense'" in res["error"]


def test_unknown_key_names_the_table_and_points_at_type(granted):
    res = granted._tool_computer({"action": "key", "keys": "hyper"})

    assert res["ok"] is False
    assert "action='type'" in res["error"]
    assert "escape" in res["error"]


def test_a_bare_modifier_is_not_a_key(granted):
    res = granted._tool_computer({"action": "key", "keys": "cmd"})

    assert res["ok"] is False
    assert "No keycode" in res["error"]


def test_coordinates_are_required_and_validated(granted):
    for args in ({}, {"x": 1}, {"x": "5", "y": 5}, {"x": True, "y": 5}):
        res = granted._tool_computer({"action": "move", **args})
        assert res["ok"] is False, args
        assert "'x' and 'y' are required" in res["error"]


# ══════════════════════════════════════════════════════════════════════════════
# 7. Audit + classification
# ══════════════════════════════════════════════════════════════════════════════


def test_every_action_is_audited_outside_the_repo(granted, monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(desktop_tools, "post_mouse", lambda *_a: None)

    granted._tool_computer({"action": "click", "x": 1, "y": 1})

    log = tmp_path / ".asicode" / "computer_use.log"
    assert log.exists()
    line = log.read_text(encoding="utf-8")
    assert "click" in line
    assert str(tmp_path) not in line  # relative to the audit, not the repo


def test_typed_text_is_not_written_to_the_audit_log(granted, monkeypatch, tmp_path):
    """A log of what was typed is a log of passwords."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(desktop_tools, "post_text", lambda text: len(text))

    granted._tool_computer({"action": "type", "text": "hunter2"})

    assert "hunter2" not in (tmp_path / ".asicode" / "computer_use.log").read_text(encoding="utf-8")


def test_a_failing_handler_is_reported_not_swallowed(granted, monkeypatch):
    def boom(*a, **kw):
        raise RuntimeError("window server said no")

    monkeypatch.setattr(desktop_tools, "post_mouse", boom)

    res = granted._tool_computer({"action": "click", "x": 1, "y": 1})

    assert res["ok"] is False
    assert "window server said no" in res["error"]


def test_the_action_tables_do_not_overlap_and_cover_the_dispatcher():
    assert DESKTOP_INTERACTION_ACTIONS.isdisjoint(DESKTOP_OBSERVATION_ACTIONS)
    dispatched = {
        "screenshot",
        "cursor_position",
        "move",
        "click",
        "double_click",
        "right_click",
        "drag",
        "scroll",
        "type",
        "key",
    }
    assert dispatched == DESKTOP_INTERACTION_ACTIONS | DESKTOP_OBSERVATION_ACTIONS
