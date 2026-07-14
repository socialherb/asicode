"""
Image utility function module.

Functions extracted from asi.py:
- _check_clipboard_image: Detect image data on macOS clipboard and return as base64
- _extract_images_from_input: Detect image file paths in user input, read as base64, and return text with paths removed
"""

from __future__ import annotations


def _check_clipboard_image() -> list[dict[str, str]]:
    """Detect image data on macOS clipboard and return as base64.

    Reads the clipboard directly when using Cmd+Ctrl+Shift+4 (clipboard screenshot)
    and converts to base64 data when entering a prompt.
    """
    import base64 as _b64
    import platform as _platform
    import subprocess as _sp

    if _platform.system() != "Darwin":
        return []

    try:
        # Read clipboard TIFF/PNG data as base64 via osascript
        _script = (
            'use framework "AppKit"\n'
            'set pb to current application\'s NSPasteboard\'s generalPasteboard()\n'
            'set cls to pb\'s pasteboardItems()\n'
            'if cls is missing value or cls\'s count() = 0 then return ""\n'
            'set item to cls\'s objectAtIndex:0\n'
            'set types to item\'s types()\n'
            'if not (types\'s containsObject:"public.png" as string) then return ""\n'
            'set data to item\'s dataForType:"public.png"\n'
            'if data is missing value then return ""\n'
            'set b64 to data\'s base64EncodedStringWithOptions:0\n'
            'return b64 as string\n'
        )
        _result = _sp.run(
            ['osascript', '-l', 'JavaScript', '-e', _script],
            capture_output=True, text=True, timeout=3.0,
        )
        if _result.returncode == 0 and _result.stdout.strip():
            _b64_data = _result.stdout.strip()
            # Validate it's valid base64
            if len(_b64_data) > 100:  # minimum plausible image size
                try:
                    _b64.b64decode(_b64_data, validate=True)
                    return [{"media_type": "image/png", "data": _b64_data}]
                except Exception:
                    pass
    except (FileNotFoundError, _sp.TimeoutExpired, OSError):
        pass

    return []


def _extract_images_from_input(text: str) -> tuple[str, list[dict[str, str]]]:
    """Detect image file paths in user input, read as base64, and return text with paths removed.

    Detects image files dragged into the terminal (e.g. "path/to/img.png"),
    base64-encodes them, and removes the paths from text.
    Also handles shell-escaped spaces (`\\ `) pasted from macOS/iTerm2,
    and quotation-wrapped paths ("/path/with spaces/file.png").
    Also detects `data:image/...;base64,...` data URLs.
    """
    import base64 as _b64
    import mimetypes as _mimetypes
    from pathlib import Path as _Path

    _IMAGE_EXTENSIONS = frozenset((
        ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff", ".tif", ".svg",
    ))

    #--- 1st pass: shell-escaped blank/space (\ ) processing ---
    _MARKER = "\x00_IMG_ESC_SP_\x00"
    _escaped = text.replace("\\ ", _MARKER)
    words = _escaped.split()
    cleaned_words = []
    images = []

    for w in words:
        w_restored = w.replace(_MARKER, " ")
        p = _Path(w_restored.strip("'\""))
        if p.suffix.lower() in _IMAGE_EXTENSIONS and p.exists() and p.is_file():
            try:
                raw = p.read_bytes()
                b64 = _b64.b64encode(raw).decode("ascii")
                mt = _mimetypes.guess_type(str(p))[0] or "image/png"
                images.append({"media_type": mt, "data": b64})
            except (OSError, PermissionError, MemoryError):
                cleaned_words.append(w.replace(_MARKER, " "))
                continue
        else:
            cleaned_words.append(w.replace(_MARKER, " "))

    result = " ".join(cleaned_words).strip()

    # --- 2nd pass: quotation-wrapped paths (missed by 1st pass) ---
    if not images:
        # If starts with a quote, extract up to the closing quote and verify file
        _q = None
        if result.startswith("\""):
            _q = "\""
        elif result.startswith("'"):
            _q = "'"
        if _q:
            _end = result.find(_q, 1)
            if _end > 1:
                _candidate = result[1:_end]
                p = _Path(_candidate)
                if p.suffix.lower() in _IMAGE_EXTENSIONS and p.exists() and p.is_file():
                    try:
                        raw = p.read_bytes()
                        b64 = _b64.b64encode(raw).decode("ascii")
                        mt = _mimetypes.guess_type(str(p))[0] or "image/png"
                        images.append({"media_type": mt, "data": b64})
                        result = result[_end + 1:].strip()
                    except (OSError, PermissionError, MemoryError):
                        pass

    # --- 3rd pass: data URL detection (base64-encoded images) ---
    #"data:image/png;base64,iVBOR..." form processing
    import re as _re
    _DATA_URL_RE = _re.compile(
        r'data:image/(?P<fmt>\w+);base64,(?P<b64>[A-Za-z0-9+/=]+)',
    )
    _new_result_parts = []
    _last_end = 0
    for _m in _DATA_URL_RE.finditer(result):
        _fmt = _m.group("fmt")
        _b64_data = _m.group("b64")
        _mt = f"image/{_fmt}"
        if _fmt == "png":
            _mt = "image/png"
        elif _fmt == "jpg":
            _mt = "image/jpeg"
        elif _fmt == "jpeg":
            _mt = "image/jpeg"
        images.append({"media_type": _mt, "data": _b64_data})
        _new_result_parts.append(result[_last_end:_m.start()])
        _last_end = _m.end()
    _new_result_parts.append(result[_last_end:])
    result = "".join(_new_result_parts).strip()

    return result.strip(), images
