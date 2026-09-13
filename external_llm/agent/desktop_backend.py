"""Host-desktop computer use: capture the screen, synthesise input at a point.

Phase 2. Where Phase 1 drives a browser Playwright owns, this drives the machine
the user is actually sitting at — which is why almost everything here is a
permission question rather than a code question.

The two permission gates, and why they are PREFLIGHT
----------------------------------------------------
macOS gates screen capture (TCC ``Screen Recording``) and input synthesis
(``Accessibility``) behind per-application consent. Both failures are SILENT:

* Without Screen Recording, ``screencapture`` succeeds and writes a PNG. That
  PNG contains the desktop wallpaper and nothing else — no windows, no menu bar.
  An agent that treats it as a real capture will confidently describe a screen
  it never saw, and the only external symptom is an image.
* Without Accessibility, ``CGEventPost`` also succeeds, and the events are
  dropped. The agent clicks "at" a button and nothing happens, forever.

So both gates are checked with ``CGPreflightScreenCaptureAccess`` /
``AXIsProcessTrusted`` — the two calls that ANSWER the question instead of
PROMPTING for it (``CGRequestScreenCaptureAccess`` and
``AXIsProcessTrustedWithOptions`` are the prompting ones, and a CLI agent cannot
answer a system dialog on the user's behalf). A refusal here is a first-class
result with the exact System Settings path, never a degraded capture.

Coordinate spaces
-----------------
Three of them, and they are not the same number:

    display pixels    what the PNG contains   (2560x1440 on the 2x display here)
    display points    what CGEvent moves to   (1280x720 — the same display)
    image pixels      what the model measured on the screenshot it was shown

The model gets image pixels, which equal display pixels for a full-display
capture. Everything posted divides by the backing scale (``to_points``) so the
pointer lands where the model pointed — the same discipline Phase 1 applies to
the browser, and for the same reason: a mis-scaled pointer is invisible in a
result that only echoes the request.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import logging
import pathlib
import subprocess
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


SCREEN_CAPTURE_SETTINGS_HINT = (
    "System Settings → Privacy & Security → Screen Recording → enable the app that runs "
    "asicode (Terminal/iTerm/the IDE), then restart it"
)
ACCESSIBILITY_SETTINGS_HINT = (
    "System Settings → Privacy & Security → Accessibility → enable the app that runs "
    "asicode (Terminal/iTerm/the IDE), then restart it"
)

# ``screencapture`` is the only capture path used here: it is part of macOS, it
# writes a real PNG (so the image transport and the PNG-header geometry parser
# both work unchanged), and it is what the Screen Recording grant applies to.
_SCREENCAPTURE = "/usr/sbin/screencapture"

# A capture or a posted event that takes longer than this is not slow, it is
# broken (a stuck window server); the caller gets a result instead of a hang.
_RUN_TIMEOUT_SEC = 20


@dataclass(frozen=True)
class DisplayGeometry:
    """One display, in both spaces it has to be known in."""

    display_id: int
    pixel_width: int
    pixel_height: int
    point_width: int
    point_height: int
    point_origin_x: int = 0
    point_origin_y: int = 0

    @property
    def scale(self) -> float:
        """Display pixels per point — 2.0 on a Retina panel, 1.0 otherwise."""
        if self.point_width and self.pixel_width:
            return self.pixel_width / self.point_width
        return 1.0


# ── ctypes loads (lazy + cached: --help paths must not touch the frameworks) ──
_frameworks: dict[str, Any] = {}


def _framework(name: str) -> Any:
    """Load a macOS framework by library name, or ``None`` off-macOS."""
    if name in _frameworks:
        return _frameworks[name]
    lib = None
    try:
        path = ctypes.util.find_library(name)
        if path:
            lib = ctypes.CDLL(path)
    except Exception:  # pragma: no cover - a broken framework path is not actionable here
        logger.debug("could not load %s", name, exc_info=True)
        lib = None
    _frameworks[name] = lib
    return lib


def _core_graphics() -> Any:
    return _framework("CoreGraphics")


def _application_services() -> Any:
    return _framework("ApplicationServices")


def screen_recording_granted() -> bool:
    """Whether this process may capture the screen. False when unknowable.

    Fail-CLOSED: an unreadable answer is treated as "no", because the failure
    mode of guessing "yes" is a wallpaper-only image the agent believes is the
    user's screen.
    """
    cg = _core_graphics()
    if cg is None:
        return False
    try:
        cg.CGPreflightScreenCaptureAccess.restype = ctypes.c_bool
        return bool(cg.CGPreflightScreenCaptureAccess())
    except Exception:
        logger.debug("CGPreflightScreenCaptureAccess unavailable", exc_info=True)
        return False


def accessibility_granted() -> bool:
    """Whether this process may synthesise input. False when unknowable."""
    ax = _application_services()
    if ax is None:
        return False
    try:
        ax.AXIsProcessTrusted.restype = ctypes.c_bool
        return bool(ax.AXIsProcessTrusted())
    except Exception:
        logger.debug("AXIsProcessTrusted unavailable", exc_info=True)
        return False


def main_display_geometry() -> DisplayGeometry | None:
    """Geometry of the main display, or ``None`` when it cannot be read.

    ``CGDisplayBounds`` is in POINTS (it is a CGRect in the global display
    space) while ``CGDisplayPixels*`` is in pixels, so one call pair yields both
    spaces AND the scale between them — the number every input event divides by.
    """
    cg = _core_graphics()
    if cg is None:
        return None

    class _CGPoint(ctypes.Structure):
        _fields_ = [("x", ctypes.c_double), ("y", ctypes.c_double)]

    class _CGSize(ctypes.Structure):
        _fields_ = [("width", ctypes.c_double), ("height", ctypes.c_double)]

    class _CGRect(ctypes.Structure):
        _fields_ = [("origin", _CGPoint), ("size", _CGSize)]

    try:
        cg.CGMainDisplayID.restype = ctypes.c_uint32
        cg.CGDisplayPixelsWide.restype = ctypes.c_size_t
        cg.CGDisplayPixelsHigh.restype = ctypes.c_size_t
        cg.CGDisplayBounds.restype = _CGRect
        cg.CGDisplayBounds.argtypes = [ctypes.c_uint32]

        display_id = int(cg.CGMainDisplayID())
        bounds = cg.CGDisplayBounds(display_id)
        return DisplayGeometry(
            display_id=display_id,
            pixel_width=int(cg.CGDisplayPixelsWide(display_id)),
            pixel_height=int(cg.CGDisplayPixelsHigh(display_id)),
            # CGDisplayBounds is a CGRect in points; round() already yields an
            # int for an integral width, and a fractional one is a display
            # configuration we must not silently truncate into a wrong origin.
            point_width=round(bounds.size.width),
            point_height=round(bounds.size.height),
            point_origin_x=round(bounds.origin.x),
            point_origin_y=round(bounds.origin.y),
        )
    except Exception:
        logger.debug("main display geometry unavailable", exc_info=True)
        return None


def to_points(geometry: DisplayGeometry | None, image_x: float, image_y: float) -> tuple[float, float]:
    """Convert image pixels (what the model measured) to screen points.

    A full-display capture's image pixels ARE display pixels, so the only
    conversion needed is the backing scale — and its inverse is what makes a
    Retina click land on the pixel the model pointed at.
    """
    scale = geometry.scale if geometry else 1.0
    if scale <= 0:
        scale = 1.0
    origin_x = geometry.point_origin_x if geometry else 0
    origin_y = geometry.point_origin_y if geometry else 0
    return (image_x / scale + origin_x, image_y / scale + origin_y)


# ── Capture ──────────────────────────────────────────────────────────────────
def capture_main_display(*, runner: Callable[..., Any] | None = None) -> tuple[bytes | None, str]:
    """Capture the main display as PNG **pixels**. Returns ``(png_bytes, error)``.

    Refuses without the Screen Recording grant: ``screencapture`` would return
    success and a wallpaper-only image, which is worse than an error because the
    agent cannot tell the difference from inside the conversation.

    ``runner`` is injectable so the whole path is testable on a machine (or CI
    runner) where the grant will never exist; it receives the argv list and must
    return an object with ``returncode`` and ``stdout`` like ``subprocess.run``.
    """
    if not screen_recording_granted():
        return None, (
            "Screen Recording permission is not granted, so a capture would return only the "
            f"desktop wallpaper. Grant it: {SCREEN_CAPTURE_SETTINGS_HINT}"
        )

    geometry = main_display_geometry()
    if geometry is None:
        return None, "Could not read the main display geometry (CoreGraphics unavailable)."

    target = pathlib.Path(_capture_dir()) / f"screen_{int(time.time())}_{uuid.uuid4().hex[:6]}.png"
    argv = [
        _SCREENCAPTURE,
        "-x",  # no shutter sound
        "-t",
        "png",
        # -R takes POINTS; a Retina display therefore writes a 2x-pixel PNG,
        # which is exactly the image-pixel space the model will measure on.
        "-R",
        f"{geometry.point_origin_x},{geometry.point_origin_y},{geometry.point_width},{geometry.point_height}",
        str(target),
    ]
    run = runner or subprocess.run
    try:
        proc = run(argv, capture_output=True, text=True, timeout=_RUN_TIMEOUT_SEC, check=False)
    except Exception as exc:
        return None, f"screencapture could not be run: {type(exc).__name__}: {exc}"

    if getattr(proc, "returncode", 1) != 0:
        detail = (getattr(proc, "stderr", "") or "").strip()
        return None, f"screencapture failed (rc={proc.returncode}){': ' + detail if detail else ''}"

    try:
        return target.read_bytes(), ""
    except OSError as exc:
        return None, f"screencapture reported success but {target} could not be read: {exc}"


def _capture_dir() -> str:
    """Where captures live: ``~/.asicode/captures``.

    NOT inside the repo: a desktop capture has nothing to do with the working
    tree, and writing 2 MB PNGs into it would show up in ``git status`` for
    every screenshot the agent takes.
    """
    directory = pathlib.Path.home() / ".asicode" / "captures"
    directory.mkdir(parents=True, exist_ok=True)
    return str(directory)


# ── Input synthesis ──────────────────────────────────────────────────────────
# ``kCGEventLeftMouseDown`` etc. Names, not numbers: the constants are a stable
# part of the public API and a wrong integer here is a wrong click somewhere.
_EVENT_TYPES = {
    "left_down": 1,  # kCGEventLeftMouseDown
    "left_up": 2,  # kCGEventLeftMouseUp
    "right_down": 3,  # kCGEventRightMouseDown
    "right_up": 4,  # kCGEventRightMouseUp
    "mouse_moved": 5,  # kCGEventMouseMoved
    "left_drag": 6,  # kCGEventLeftMouseDragged
    "key_down": 10,  # kCGEventKeyDown
    "key_up": 11,  # kCGEventKeyUp
    "other_down": 25,  # kCGEventOtherMouseDown (middle button)
    "other_up": 26,  # kCGEventOtherMouseUp
    "scroll": 22,  # kCGEventScrollWheel
}

# macOS virtual keycodes for the keys that are not text. Deliberately tiny — the
# full table is a keymap, and anything absent is better refused than guessed at.
# Text never goes through here: ``post_text`` types Unicode directly.
KEYCODES: dict[str, int] = {
    "return": 36,
    "enter": 36,
    "tab": 48,
    "space": 49,
    "delete": 51,  # backspace
    "escape": 53,
    "esc": 53,
    "forwarddelete": 117,
    "home": 115,
    "end": 119,
    "pageup": 116,
    "pagedown": 121,
    "up": 126,
    "down": 125,
    "left": 123,
    "right": 124,
    "f1": 122,
    "f2": 120,
    "f3": 99,
    "f4": 118,
    "f5": 96,
    "f6": 97,
}

# Modifier flag masks (CGEventFlags). Named so a chord reads as the chord, and
# so a wrong edit to one of these numbers is a visibly wrong number.
MODIFIER_FLAGS: dict[str, int] = {
    "cmd": 1 << 20,  # kCGEventFlagMaskCommand
    "command": 1 << 20,
    "shift": 1 << 17,  # kCGEventFlagMaskShift
    "alt": 1 << 19,  # kCGEventFlagMaskAlternate
    "option": 1 << 19,
    "ctrl": 1 << 18,  # kCGEventFlagMaskControl
    "control": 1 << 18,
}

_poster: Callable[[Any], None] | None = None


def set_event_poster(poster: Callable[[Any], None] | None) -> None:
    """Override how a built event is delivered (tests, and future backends).

    Kept as a module seam rather than a constructor arg because the poster is a
    process-wide capability: exactly one implementation (``CGEventPost``) exists
    for a given process, and swapping it per call would suggest otherwise.
    """
    global _poster
    _poster = poster


def _default_poster(event: Any) -> None:
    cg = _core_graphics()
    if cg is None:
        raise RuntimeError("CoreGraphics unavailable")
    # kCGHIDEventTap: the same tap the hardware posts to, i.e. what an
    # application sees as real input.
    cg.CGEventPost(0, event)


def _new_mouse_event(event_type: int, point: tuple[float, float]) -> Any:
    cg = _core_graphics()
    if cg is None:
        raise RuntimeError("CoreGraphics unavailable")

    class _CGPoint(ctypes.Structure):
        _fields_ = [("x", ctypes.c_double), ("y", ctypes.c_double)]

    cg.CGEventCreateMouseEvent.restype = ctypes.c_void_p
    cg.CGEventCreateMouseEvent.argtypes = [ctypes.c_void_p, ctypes.c_uint32, _CGPoint, ctypes.c_uint32]
    return cg.CGEventCreateMouseEvent(None, event_type, _CGPoint(point[0], point[1]), 0)


def post_mouse(event_name: str, point: tuple[float, float]) -> None:
    """Post a mouse event at *point* (screen points). Raises on an unknown name.

    The event NAME selects the button (``left_down`` / ``right_up`` / …), so
    there is deliberately no second way to say it.
    """
    if event_name not in _EVENT_TYPES:
        raise ValueError(f"unknown mouse event {event_name!r}")
    event = _new_mouse_event(_EVENT_TYPES[event_name], point)
    (_poster or _default_poster)(event)


def post_scroll(point: tuple[float, float], dx: float, dy: float) -> None:
    """Post a wheel event at *point*. Positive *dy* scrolls content up (see the tool)."""
    cg = _core_graphics()
    if cg is None:
        raise RuntimeError("CoreGraphics unavailable")

    class _CGPoint(ctypes.Structure):
        _fields_ = [("x", ctypes.c_double), ("y", ctypes.c_double)]

    cg.CGEventCreateScrollWheelEvent.restype = ctypes.c_void_p
    cg.CGEventCreateScrollWheelEvent.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32]
    event = cg.CGEventCreateScrollWheelEvent(None, 0, 2, int(dy), int(dx))
    cg.CGEventSetLocation.restype = None
    cg.CGEventSetLocation.argtypes = [ctypes.c_void_p, _CGPoint]
    cg.CGEventSetLocation(event, _CGPoint(point[0], point[1]))
    (_poster or _default_poster)(event)


def post_key(keycode: int, *, flags: int = 0, down: bool = True) -> None:
    """Post a keyboard event for a macOS virtual keycode, with modifier *flags*."""
    cg = _core_graphics()
    if cg is None:
        raise RuntimeError("CoreGraphics unavailable")
    cg.CGEventCreateKeyboardEvent.restype = ctypes.c_void_p
    cg.CGEventCreateKeyboardEvent.argtypes = [ctypes.c_void_p, ctypes.c_uint16, ctypes.c_bool]
    event = cg.CGEventCreateKeyboardEvent(None, keycode, down)
    if flags:
        # The modifier state rides on the event itself; posting the modifier key
        # separately would leak a held-down Shift into whatever runs next.
        cg.CGEventSetFlags.restype = None
        cg.CGEventSetFlags.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
        cg.CGEventSetFlags(event, flags)
    (_poster or _default_poster)(event)


_TEXT_CHUNK = 64


def post_text(text: str) -> int | None:
    """Type *text* as Unicode. Returns the number of characters posted, or None.

    ``CGEventKeyboardSetUnicodeString`` attaches the string to a keyboard event,
    so characters with NO virtual keycode (CJK, emoji, accents, the symbols in a
    password) type exactly as given. A keycode table cannot do that — it would
    silently drop them, which in a password field is a lockout rather than a
    typo, and there is no later signal that a character went missing.

    Chunked: one event per ``_TEXT_CHUNK`` characters keeps a long paste from
    being dropped by the window server, which is measured behaviour for very
    large single events.
    """
    cg = _core_graphics()
    if cg is None:
        return None
    cg.CGEventCreateKeyboardEvent.restype = ctypes.c_void_p
    cg.CGEventCreateKeyboardEvent.argtypes = [ctypes.c_void_p, ctypes.c_uint16, ctypes.c_bool]
    cg.CGEventKeyboardSetUnicodeString.restype = None
    cg.CGEventKeyboardSetUnicodeString.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.POINTER(ctypes.c_uint16)]

    posted = 0
    for start in range(0, len(text), _TEXT_CHUNK):
        chunk = text[start : start + _TEXT_CHUNK]
        buffer = (ctypes.c_uint16 * len(chunk))(*[ord(ch) for ch in chunk])
        event = cg.CGEventCreateKeyboardEvent(None, 0, True)
        cg.CGEventKeyboardSetUnicodeString(event, len(chunk), buffer)
        (_poster or _default_poster)(event)
        (_poster or _default_poster)(cg.CGEventCreateKeyboardEvent(None, 0, False))
        posted += len(chunk)
    return posted


def cursor_position_image() -> tuple[float, float] | None:
    """Pointer position in IMAGE pixels — the space the model measures in.

    ``CGEventGetLocation`` answers in points; the model's coordinates are image
    pixels. Converting here (rather than reporting points and letting the model
    compare them with what it saw) is what makes ``cursor_position`` usable as an
    anchor: on a 2x display the two differ by exactly the factor that would
    otherwise look like the pointer being somewhere it is not.
    """
    cg = _core_graphics()
    if cg is None:
        return None

    class _CGPoint(ctypes.Structure):
        _fields_ = [("x", ctypes.c_double), ("y", ctypes.c_double)]

    try:
        cg.CGEventCreate.restype = ctypes.c_void_p
        cg.CGEventCreate.argtypes = [ctypes.c_void_p]
        cg.CGEventGetLocation.restype = _CGPoint
        cg.CGEventGetLocation.argtypes = [ctypes.c_void_p]
        point = cg.CGEventGetLocation(cg.CGEventCreate(None))
    except Exception:
        logger.debug("pointer position unavailable", exc_info=True)
        return None

    geometry = main_display_geometry()
    scale = geometry.scale if geometry else 1.0
    origin_x = geometry.point_origin_x if geometry else 0
    origin_y = geometry.point_origin_y if geometry else 0
    return ((point.x - origin_x) * scale, (point.y - origin_y) * scale)
