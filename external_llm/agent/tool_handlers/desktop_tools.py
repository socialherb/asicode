"""``computer`` — the host desktop, as a tool the model may drive.

Opt-in, permission-gated and audited, because the machine being driven is the
user's own: their mail, their banking tab, their password manager. The browser
tool of Phase 1 is sandboxed by Playwright; this is not sandboxed by anything,
so the refusals below are the feature, not the friction.

Three independent gates, all fail-closed:

1. **Opt-in** (``ASICODE_COMPUTER_USE=1``). Off by default, so upgrading the
   tool is not the same act as handing a model the mouse.
2. **Screen Recording** for anything that captures, **Accessibility** for
   anything that acts. Both are checked with their PREFLIGHT calls (see
   ``desktop_backend``): without them the OS returns success and delivers a
   wallpaper-only image or a dropped click, which is indistinguishable from a
   working screen inside the conversation.
3. **The declared action table.** An action that is not listed is refused —
   never guessed at — and the read-only ones are named explicitly so the
   mutation classifier cannot treat a click as an observation.

Coordinates are IMAGE pixels from the model's most recent ``screenshot``; the
backend divides by the display's backing scale, and the result reports both
spaces so a mis-scaled pointer cannot hide.
"""

from __future__ import annotations

import logging
import os
import time
from typing import TYPE_CHECKING, Any

from ...image_utils import png_size as _png_size
from ..desktop_backend import (
    ACCESSIBILITY_SETTINGS_HINT,
    KEYCODES,
    MODIFIER_FLAGS,
    SCREEN_CAPTURE_SETTINGS_HINT,
    accessibility_granted,
    capture_main_display,
    cursor_position_image,
    main_display_geometry,
    post_key,
    post_mouse,
    post_scroll,
    post_text,
    screen_recording_granted,
    to_points,
)

if TYPE_CHECKING:
    from ..tool_registry import ToolResult

logger = logging.getLogger(__name__)


OPT_IN_ENV = "ASICODE_COMPUTER_USE"

# Actions that ACT on the desktop. Single source of truth, imported by
# ``tool_registry._tool_call_mutates`` so the four consumers of "is this
# mutating?" agree: a click on the user's real screen is an act with
# consequences no cache layer may treat as a read.
DESKTOP_INTERACTION_ACTIONS: frozenset[str] = frozenset(
    {"move", "click", "double_click", "right_click", "drag", "scroll", "type", "key"}
)

# Actions that only observe. Named, not inferred: an action added later lands in
# neither set and is refused by the dispatcher until someone classifies it.
DESKTOP_OBSERVATION_ACTIONS: frozenset[str] = frozenset({"screenshot", "cursor_position"})

# How many intermediate moves a drag posts. One jump reads as a teleport to
# drag-and-drop targets that watch pointermove between down and up.
_DRAG_STEPS = 12


class DesktopToolsMixin:
    """Mixin providing the ``computer`` tool for ToolRegistry."""

    # ── Host contract (provided by ToolRegistry) ───────────────────────── #
    _make_result: Any
    repo_root: str

    # ── Gates ──────────────────────────────────────────────────────────── #

    @staticmethod
    def _computer_use_enabled() -> bool:
        return (os.environ.get(OPT_IN_ENV, "") or "").strip().lower() in ("1", "true", "yes", "on")

    def _computer_refusal(self, action: str) -> str:
        """The reason this action cannot run right now, or ``""``."""
        if not self._computer_use_enabled():
            return (
                "Desktop control is OFF. It drives the machine you are sitting at, so it is opt-in: "
                f"set {OPT_IN_ENV}=1 in the environment that runs asicode and start a new session. "
                "Prefer browser_action for anything a browser can do — it is sandboxed by Playwright."
            )
        if action == "screenshot" and not screen_recording_granted():
            return (
                "Screen Recording permission is not granted, so a capture would return only the "
                f"desktop wallpaper. Grant it: {SCREEN_CAPTURE_SETTINGS_HINT}"
            )
        if action in DESKTOP_INTERACTION_ACTIONS and not accessibility_granted():
            return (
                "Accessibility permission is not granted, so synthesised input is silently dropped "
                f"by the system. Grant it: {ACCESSIBILITY_SETTINGS_HINT}"
            )
        return ""

    # ── Dispatch ───────────────────────────────────────────────────────── #

    def _tool_computer(self, args: dict[str, Any]) -> ToolResult:
        action = str(args.get("action", "")).strip().lower()
        if not action:
            return self._make_result(
                ok=False,
                content="",
                error=(
                    "'action' is required. Choose: screenshot, cursor_position, move, click, "
                    "double_click, right_click, drag, scroll, type, key"
                ),
            )

        handlers = {
            "screenshot": self._computer_screenshot,
            "cursor_position": self._computer_cursor_position,
            "move": self._computer_move,
            "click": self._computer_click,
            "double_click": self._computer_click,
            "right_click": self._computer_click,
            "drag": self._computer_drag,
            "scroll": self._computer_scroll,
            "type": self._computer_type,
            "key": self._computer_key,
        }
        # Two shapes, both honest: the observation actions read no arguments at
        # all, and the rest take the call's args. A single uniform signature
        # would leave half the table carrying parameters it never reads, which
        # is how a handler that should have read one goes unnoticed.
        takes_args = action in DESKTOP_INTERACTION_ACTIONS
        handler = handlers.get(action)
        if handler is None:
            return self._make_result(
                ok=False,
                content="",
                error=f"Unknown action: '{action}'. Available: {', '.join(sorted(handlers))}",
            )

        refusal = self._computer_refusal(action)
        if refusal:
            logger.info("computer tool refused action %r: %s", action, refusal.split(".")[0])
            return self._make_result(ok=False, content="", error=refusal)

        try:
            self._audit(action, args)
            return handler(args) if takes_args else handler()
        except Exception as exc:
            # A failure here is a real failure of a real input path; it is
            # reported, never swallowed into a "done".
            return self._make_result(
                ok=False,
                content="",
                error=f"Desktop action '{action}' failed: {type(exc).__name__}: {exc}",
            )

    def _audit(self, action: str, args: dict[str, Any]) -> None:
        """Append one line per action to ``~/.asicode/computer_use.log``.

        The desktop has no undo and no sandbox, so the record of what was
        driven, when, and with what, lives outside the conversation where a
        user can review it after the fact. Best-effort: a failure to write an
        audit line must not become the reason an action did not happen.
        """
        try:
            import pathlib

            log = pathlib.Path.home() / ".asicode" / "computer_use.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            safe = {k: (f"<{len(v)} chars>" if k == "text" and isinstance(v, str) else v) for k, v in args.items()}
            with log.open("a", encoding="utf-8") as fh:
                fh.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')}\t{action}\t{safe}\n")
        except Exception as exc:
            logger.debug("computer_use audit write failed: %s", exc)

    # ── Shared coordinate handling ─────────────────────────────────────── #

    def _point_from_image(self, args: dict[str, Any], *, prefix: str = "") -> tuple[float, float] | None:
        """``(point_x, point_y)`` from the IMAGE coordinates the model reported."""
        x = _as_number(args.get(f"{prefix}x"))
        y = _as_number(args.get(f"{prefix}y"))
        if x is None or y is None:
            return None
        return to_points(main_display_geometry(), x, y)

    def _label(self, args: dict[str, Any], prefix: str, point: tuple[float, float]) -> str:
        return f"image ({args.get(f'{prefix}x')}, {args.get(f'{prefix}y')}) -> screen ({point[0]:.0f}, {point[1]:.0f})"

    # ── Observation ────────────────────────────────────────────────────── #

    def _computer_screenshot(self) -> ToolResult:
        png, error = capture_main_display()
        if png is None:
            return self._make_result(ok=False, content="", error=error)

        geometry = main_display_geometry()
        width, height = _png_size(png)
        scale = geometry.scale if geometry else 1.0
        caption = (
            f"screen capture, {width}x{height} px, display scale {scale:g} "
            f"({geometry.point_width}x{geometry.point_height} pt). "
            "Coordinates for move/click/double_click/right_click/drag/scroll are IMAGE pixels on THIS image."
        )
        import base64 as _b64

        return self._make_result(
            ok=True,
            content=(
                f"Captured the main display: {width}x{height} px, scale {scale:g}. "
                "The image is attached above; measure coordinates on it."
            ),
            metadata={
                "image": {"width": width, "height": height},
                "display_points": {"width": geometry.point_width, "height": geometry.point_height} if geometry else {},
                "scale": scale,
                "attach_images": [
                    {"media_type": "image/png", "data": _b64.b64encode(png).decode("utf-8"), "caption": caption}
                ],
            },
        )

    def _computer_cursor_position(self) -> ToolResult:
        position = cursor_position_image()
        if position is None:
            return self._make_result(ok=False, content="", error="Could not read the pointer position.")
        return self._make_result(
            ok=True,
            content=f"Pointer is at image ({position[0]:.0f}, {position[1]:.0f})",
            metadata={"x": position[0], "y": position[1]},
        )

    # ── Interaction ────────────────────────────────────────────────────── #

    def _computer_move(self, args: dict[str, Any]) -> ToolResult:
        point = self._point_from_image(args)
        if point is None:
            return self._make_result(ok=False, content="", error="'x' and 'y' are required for move")
        post_mouse("mouse_moved", point)
        return self._make_result(ok=True, content=f"Moved pointer to {self._label(args, '', point)}")

    def _computer_click(self, args: dict[str, Any]) -> ToolResult:
        # One handler, three actions: the verb the caller chose is IN the args,
        # so reading it here beats three near-identical methods.
        action = str(args.get("action", "")).strip().lower()
        point = self._point_from_image(args)
        if point is None:
            return self._make_result(ok=False, content="", error="'x' and 'y' are required for click actions")

        if action == "right_click":
            down, up, button = "right_down", "right_up", "right"
        elif action == "middle_click":
            down, up, button = "other_down", "other_up", "middle"
        else:
            down, up, button = "left_down", "left_up", "left"
        clicks = 2 if action == "double_click" else 1

        post_mouse("mouse_moved", point)
        for _ in range(clicks):
            post_mouse(down, point)
            post_mouse(up, point)
        label = {1: f"{button}-clicked", 2: "double-clicked"}[clicks]
        return self._make_result(
            ok=True,
            content=f"{label} {self._label(args, '', point)}",
            metadata={"x": point[0], "y": point[1], "button": button, "clicks": clicks},
        )

    def _computer_drag(self, args: dict[str, Any]) -> ToolResult:
        start = self._point_from_image(args)
        end = self._point_from_image(args, prefix="to_")
        if start is None or end is None:
            return self._make_result(
                ok=False, content="", error="'x'/'y' (start) and 'to_x'/'to_y' (end) are required for drag"
            )

        post_mouse("mouse_moved", start)
        post_mouse("left_down", start)
        # Intermediate moves: drag targets watch pointermove between down and up
        # and ignore a single jump to the destination.
        for step in range(1, _DRAG_STEPS + 1):
            fraction = step / _DRAG_STEPS
            mid = (
                start[0] + (end[0] - start[0]) * fraction,
                start[1] + (end[1] - start[1]) * fraction,
            )
            post_mouse("left_drag", mid)
        post_mouse("left_up", end)

        return self._make_result(
            ok=True,
            content=f"Dragged from image ({args.get('x')}, {args.get('y')}) to ({args.get('to_x')}, {args.get('to_y')})",
            metadata={"from": list(start), "to": list(end)},
        )

    def _computer_scroll(self, args: dict[str, Any]) -> ToolResult:
        point = self._point_from_image(args)
        if point is None:
            return self._make_result(ok=False, content="", error="'x' and 'y' are required for scroll")
        dy = _as_number(args.get("dy"))
        dx = _as_number(args.get("dx")) or 0.0
        if dy is None:
            return self._make_result(
                ok=False, content="", error="'dy' is required for scroll (positive scrolls content up)"
            )

        post_mouse("mouse_moved", point)
        post_scroll(point, dx, dy)
        return self._make_result(ok=True, content=f"Scrolled by ({dx:g}, {dy:g}) at {self._label(args, '', point)}")

    def _computer_type(self, args: dict[str, Any]) -> ToolResult:
        text = args.get("text", "")
        if not isinstance(text, str) or not text:
            return self._make_result(ok=False, content="", error="'text' is required for type")
        typed = post_text(text)
        if typed is None:
            return self._make_result(ok=False, content="", error="Could not synthesise text input.")
        snippet = text[:50] + "..." if len(text) > 50 else text
        return self._make_result(
            ok=True, content=f"Typed '{snippet}' at the focused element", metadata={"text_length": len(text)}
        )

    def _computer_key(self, args: dict[str, Any]) -> ToolResult:
        chord = str(args.get("keys", args.get("key", ""))).strip()
        if not chord:
            return self._make_result(
                ok=False, content="", error="'keys' is required for key (e.g. 'return', 'escape', 'cmd+tab')"
            )

        parts = [p.strip().lower() for p in chord.split("+") if p.strip()]
        if not parts:
            return self._make_result(ok=False, content="", error="'keys' is required for key")
        # Everything before the last element must BE a modifier, and the last
        # element is the key. Refusing anything else (rather than joining the
        # leftovers into a name no table has) is what turns "cmd+nonsense" into
        # an answer the model can act on.
        *modifier_names, name = parts
        flags = 0
        for modifier in modifier_names:
            mask = MODIFIER_FLAGS.get(modifier)
            if mask is None:
                return self._make_result(
                    ok=False,
                    content="",
                    error=(
                        f"'{modifier}' in '{chord}' is neither a modifier "
                        f"({', '.join(sorted(set(MODIFIER_FLAGS)))}) nor the final key."
                    ),
                )
            flags |= mask
        keycode = KEYCODES.get(name)
        if keycode is None:
            return self._make_result(
                ok=False,
                content="",
                error=(
                    f"No keycode for '{name}'. Named keys: {', '.join(sorted(KEYCODES))}. "
                    "For anything that produces text, use action='type' instead — it types Unicode directly."
                ),
            )

        post_key(keycode, flags=flags, down=True)
        post_key(keycode, flags=flags, down=False)
        return self._make_result(ok=True, content=f"Pressed '{chord}'", metadata={"keycode": keycode, "flags": flags})


def _as_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return number
