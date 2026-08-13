"""Tests for macOS clipboard image detection in image_utils.

Locks the P1-2 fix plus a latent P0 it exposed:

1. P0 — the osascript probe used ``-l JavaScript`` (JXA) but an *AppleScript*
   script body (``use framework``, ``set ... to``). JXA cannot parse that, so
   the call always errored and ``_check_clipboard_image`` silently returned
   ``[]``: clipboard images were attached on *no* machine. The probe now uses
   valid JXA (``ObjC.import``, ``$.``).

2. P1-2 — even when the call worked it only checked ``public.png``, dropping
   the TIFF-only flavour that browsers/apps place on the pasteboard. The probe
   now iterates PNG then TIFF (PNG preferred) and carries the media_type through.

The script-building and result-parsing helpers are pure functions and are
tested directly; the ``_check_clipboard_image`` orchestrator is exercised with
``platform.system`` / ``subprocess.run`` stubbed so no real osascript runs.
"""
from __future__ import annotations

import base64
import platform
import subprocess
from types import SimpleNamespace

from external_llm.image_utils import (
    _CLIPBOARD_IMAGE_UTIS,
    _build_clipboard_probe_script,
    _check_clipboard_image,
    _parse_clipboard_probe_result,
)

_VALID_B64 = base64.b64encode(b"x" * 200).decode()  # >100 chars, decodes cleanly


# --------------------------------------------------------------------------- #
# _parse_clipboard_probe_result — pure, no osascript
# --------------------------------------------------------------------------- #
class TestParseProbeResult:
    def test_empty_returns_none(self):
        assert _parse_clipboard_probe_result("") is None

    def test_whitespace_returns_none(self):
        assert _parse_clipboard_probe_result("   \n\t ") is None

    def test_no_tab_returns_none(self):
        # A bare UTI without the <tab><b64> separator is malformed.
        assert _parse_clipboard_probe_result("public.png") is None

    def test_unknown_uti_returns_none(self):
        assert _parse_clipboard_probe_result("public.jpeg\tdata") is None

    def test_empty_b64_returns_none(self):
        assert _parse_clipboard_probe_result("public.png\t") is None

    def test_png_parsed(self):
        assert _parse_clipboard_probe_result("public.png\tabc") == ("image/png", "abc")

    def test_tiff_parsed(self):
        assert _parse_clipboard_probe_result("public.tiff\txyz") == ("image/tiff", "xyz")

    def test_surrounding_whitespace_stripped(self):
        assert _parse_clipboard_probe_result("  public.tiff\txyz\n") == ("image/tiff", "xyz")


# --------------------------------------------------------------------------- #
# _build_clipboard_probe_script — pure, no osascript
# --------------------------------------------------------------------------- #
class TestBuildProbeScript:
    def test_probes_both_png_and_tiff(self):
        s = _build_clipboard_probe_script(_CLIPBOARD_IMAGE_UTIS)
        assert "'public.png'" in s
        assert "'public.tiff'" in s

    def test_png_listed_before_tiff(self):
        # PNG is the preferred flavour — it must appear first in the probe list.
        s = _build_clipboard_probe_script(_CLIPBOARD_IMAGE_UTIS)
        assert s.index("'public.png'") < s.index("'public.tiff'")

    def test_uses_valid_jxa_not_applescript(self):
        # Regression for the latent P0: the old body used AppleScript syntax
        # (`use framework`, `set ... to`) under `-l JavaScript`, which JXA
        # cannot parse. The probe must use the JS ObjC-bridge form instead.
        s = _build_clipboard_probe_script(_CLIPBOARD_IMAGE_UTIS)
        assert "use framework" not in s          # AppleScript ObjC import
        assert "ObjC.import('AppKit')" in s      # JXA ObjC import
        assert "$.NSPasteboard" in s             # JXA ObjC access
        assert "base64EncodedStringWithOptions" in s

    def test_emits_uti_tab_b64_format(self):
        # The script must separate the matched UTI from the base64 payload with
        # a tab so the parser can recover the media_type.
        s = _build_clipboard_probe_script(_CLIPBOARD_IMAGE_UTIS)
        assert "\\t" in s  # the JS source contains a '\t' escape

    def test_deterministic_for_same_utis(self):
        a = _build_clipboard_probe_script(_CLIPBOARD_IMAGE_UTIS)
        b = _build_clipboard_probe_script(_CLIPBOARD_IMAGE_UTIS)
        assert a == b

    def test_custom_utis_propagated(self):
        s = _build_clipboard_probe_script((("public.foo", "image/foo"),))
        assert "'public.foo'" in s
        assert "'public.png'" not in s


# --------------------------------------------------------------------------- #
# _check_clipboard_image — orchestrator, osascript stubbed
# --------------------------------------------------------------------------- #
def _run(returncode=0, stdout="", stderr=""):
    return SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)


class TestCheckClipboardImage:
    def test_non_darwin_returns_empty(self, monkeypatch):
        monkeypatch.setattr(platform, "system", lambda: "Linux")
        assert _check_clipboard_image() == []

    def test_nonzero_returncode_returns_empty(self, monkeypatch):
        monkeypatch.setattr(platform, "system", lambda: "Darwin")
        monkeypatch.setattr(subprocess, "run", lambda *a, **k: _run(returncode=1, stderr="boom"))
        assert _check_clipboard_image() == []

    def test_empty_stdout_returns_empty(self, monkeypatch):
        monkeypatch.setattr(platform, "system", lambda: "Darwin")
        monkeypatch.setattr(subprocess, "run", lambda *a, **k: _run(returncode=0, stdout=""))
        assert _check_clipboard_image() == []

    def test_png_clipboard_attached(self, monkeypatch):
        monkeypatch.setattr(platform, "system", lambda: "Darwin")
        monkeypatch.setattr(
            subprocess, "run",
            lambda *a, **k: _run(returncode=0, stdout=f"public.png\t{_VALID_B64}"),
        )
        assert _check_clipboard_image() == [{"media_type": "image/png", "data": _VALID_B64}]

    def test_tiff_clipboard_attached(self, monkeypatch):
        # The P1-2 fix: a TIFF-only pasteboard flavour is now attached instead
        # of silently dropped, and carries media_type image/tiff.
        monkeypatch.setattr(platform, "system", lambda: "Darwin")
        monkeypatch.setattr(
            subprocess, "run",
            lambda *a, **k: _run(returncode=0, stdout=f"public.tiff\t{_VALID_B64}"),
        )
        assert _check_clipboard_image() == [{"media_type": "image/tiff", "data": _VALID_B64}]

    def test_png_preferred_over_tiff(self, monkeypatch):
        # When both flavours are present the probe returns PNG first; the
        # orchestrator must surface that PNG, not a TIFF.
        monkeypatch.setattr(platform, "system", lambda: "Darwin")
        monkeypatch.setattr(
            subprocess, "run",
            lambda *a, **k: _run(returncode=0, stdout=f"public.png\t{_VALID_B64}"),
        )
        result = _check_clipboard_image()
        assert result and result[0]["media_type"] == "image/png"

    def test_invalid_base64_returns_empty(self, monkeypatch):
        monkeypatch.setattr(platform, "system", lambda: "Darwin")
        monkeypatch.setattr(
            subprocess, "run",
            lambda *a, **k: _run(returncode=0, stdout="public.png\t" + "!" * 120),
        )
        assert _check_clipboard_image() == []

    def test_too_small_payload_returns_empty(self, monkeypatch):
        # <100-char payloads are rejected as implausible image fragments.
        monkeypatch.setattr(platform, "system", lambda: "Darwin")
        monkeypatch.setattr(
            subprocess, "run",
            lambda *a, **k: _run(returncode=0, stdout="public.png\t" + "A" * 50),
        )
        assert _check_clipboard_image() == []

    def test_subprocess_missing_returns_empty(self, monkeypatch):
        # osascript absent (non-mac or stripped image) — suppress(FileNotFoundError).
        monkeypatch.setattr(platform, "system", lambda: "Darwin")

        def _raise(*a, **k):
            raise FileNotFoundError("osascript")

        monkeypatch.setattr(subprocess, "run", _raise)
        assert _check_clipboard_image() == []

    def test_passes_jxa_language_flag(self, monkeypatch):
        # The call must invoke `osascript -l JavaScript` (JXA), matching the
        # JS body the builder now produces.
        captured = {}

        def _fake_run(argv, *a, **k):
            captured["argv"] = argv
            return _run(returncode=0, stdout="")

        monkeypatch.setattr(platform, "system", lambda: "Darwin")
        monkeypatch.setattr(subprocess, "run", _fake_run)
        _check_clipboard_image()
        assert captured["argv"][0] == "osascript"
        assert "-l" in captured["argv"]
        assert "JavaScript" in captured["argv"]
        # the script body must be the JXA the builder produces
        assert "ObjC.import('AppKit')" in captured["argv"][-1]
