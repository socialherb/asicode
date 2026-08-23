"""OCR language-pack fallback for ``_detect_image_ocr_lang``.

Regression for the silent OCR-neutralization bug: when the highest-priority
language pack (``kor+eng``) was unavailable — the common case on an
English-only ``tesseract`` install — ``pytesseract.image_to_data`` raised
``TesseractError`` and the *outer* ``except Exception`` aborted the whole
function, so ``eng`` alone was *never* tried and every image came back with
``data=None`` (OCR completely dead). The fix wraps each pack in its own
try/except and continues to the next candidate. A module-level cache also
remembers the winning pack so a multi-image session probes it first.

These tests inject fake ``pytesseract`` / ``PIL`` modules via ``sys.modules``
so they run even when the real libraries (and the ``tesseract`` binary) are
absent.
"""

from __future__ import annotations

import base64
import sys

import pytest

import external_llm.providers as providers_module
from external_llm.providers import _detect_image_ocr_lang

# ── fakes ──────────────────────────────────────────────────────────────────


class _TesseractError(Exception):
    """Stand-in for ``pytesseract.TesseractError`` (which may be unavailable)."""


class _Out:
    DICT = "DICT"


class _FakeImg:
    size = (100, 50)


class _FakeImage:
    @staticmethod
    def open(_buf):
        return _FakeImg()


def _ocr_data(words: list[tuple[str, int]]) -> dict:
    """Build an ``image_to_data`` DICT payload. ``words``: list of (text, conf)."""
    return {
        "text": [w[0] for w in words],
        "conf": [w[1] for w in words],
        "left": [0] * len(words),
        "top": [0] * len(words),
        "width": [0] * len(words),
        "height": [0] * len(words),
    }


def _install_tess(
    monkeypatch, lang_to_outcome: dict, langs: list | None = None, probe_calls: list | None = None
) -> list[str]:
    """Inject a fake ``pytesseract`` module.

    ``lang_to_outcome`` maps each lang to either a result dict (available,
    produces that text) or an ``Exception`` (unavailable pack → raises).
    ``langs`` (optional) is what ``get_languages`` (``tesseract --list-langs``)
    reports; ``None`` makes the probe fail (legacy per-pack try/skip path).
    ``probe_calls`` (optional) records each ``get_languages`` invocation.
    Records the probe order on the returned list.
    """
    order: list[str] = []

    def _image_to_data(_img, lang=None, output_type=None):
        order.append(lang)
        outcome = lang_to_outcome.get(lang)
        if isinstance(outcome, Exception):
            raise outcome
        if outcome is None:
            # Available pack but not specified in the map → no text.
            return _ocr_data([])
        return outcome

    def _get_languages(config=""):
        if probe_calls is not None:
            probe_calls.append(config)
        if langs is None:
            raise AttributeError("no tesseract binary for --list-langs")
        return list(langs)

    class _TessModule:
        image_to_data = staticmethod(_image_to_data)
        get_languages = staticmethod(_get_languages)
        Output = _Out
        TesseractError = _TesseractError

    class _PILModule:
        Image = _FakeImage

    monkeypatch.setitem(sys.modules, "pytesseract", _TessModule)
    monkeypatch.setitem(sys.modules, "PIL", _PILModule)
    monkeypatch.setitem(sys.modules, "PIL.Image", _FakeImage)
    return order


@pytest.fixture(autouse=True)
def _reset_ocr_cache():
    """Each test starts (and ends) with an empty OCR-lang cache."""
    providers_module._OCR_RESOLVED_LANG = None
    providers_module._OCR_AVAILABLE_LANGS = None
    providers_module._OCR_LANG_PROBE_FAILED = False
    providers_module._OCRTextCacheAccess.clear()
    yield
    providers_module._OCR_RESOLVED_LANG = None
    providers_module._OCR_AVAILABLE_LANGS = None
    providers_module._OCR_LANG_PROBE_FAILED = False
    providers_module._OCRTextCacheAccess.clear()


_B64 = base64.b64encode(b"\x89PNG\r\n\x1a\n").decode()


# ── the bug regression ─────────────────────────────────────────────────────


def test_falls_back_when_first_lang_pack_missing(monkeypatch):
    """kor+eng unavailable → must try eng and return its data (not None).

    Pre-fix the TesseractError escaped the for-loop and hit the outer
    ``except Exception``; ``eng`` was never probed and OCR returned
    ``data=None`` for every image.
    """
    order = _install_tess(
        monkeypatch,
        {
            "kor+eng": _TesseractError("kor.traineddata not found"),
            "eng": _ocr_data([("hello", 80)]),
        },
    )

    lang, w, h, data = _detect_image_ocr_lang(_B64)

    assert order == ["kor+eng", "eng"]
    assert lang == "eng"
    assert (w, h) == (100, 50)
    assert data is not None
    assert data["text"] == ["hello"]


def test_falls_back_across_multiple_missing_packs(monkeypatch):
    """kor+eng AND eng both missing → chi_sim+eng still gets a chance."""
    order = _install_tess(
        monkeypatch,
        {
            "kor+eng": _TesseractError("no kor"),
            "eng": _TesseractError("no eng"),
            "chi_sim+eng": _ocr_data([("你好", 85)]),
            "jpn+eng": _ocr_data([("hi", 70)]),
        },
    )
    lang, _w, _h, data = _detect_image_ocr_lang(_B64)
    assert order == ["kor+eng", "eng", "chi_sim+eng"]
    assert lang == "chi_sim+eng"
    assert data is not None


# ── caching (P1-1) ─────────────────────────────────────────────────────────


def test_caches_winning_lang_and_probes_it_first(monkeypatch):
    """First image's winning pack is cached and probed first on the next image."""
    # Image 1: kor+eng produces text → cached.
    order1 = _install_tess(
        monkeypatch,
        {
            "kor+eng": _ocr_data([("안녕", 90)]),
            "eng": _ocr_data([("hi", 70)]),
        },
    )
    lang1, *_ = _detect_image_ocr_lang(_B64)
    assert lang1 == "kor+eng"
    assert providers_module._OCR_RESOLVED_LANG == "kor+eng"
    assert order1 == ["kor+eng"]  # first call: priority order, short-circuit

    # Image 2: kor+eng probed first (cached) and short-circuits — eng untouched.
    order2 = _install_tess(
        monkeypatch,
        {
            "kor+eng": _ocr_data([("안녕", 90)]),
            "eng": _ocr_data([("hi", 70)]),
        },
    )
    lang2, *_ = _detect_image_ocr_lang(_B64)
    assert lang2 == "kor+eng"
    assert order2 == ["kor+eng"]


def test_cached_unavailable_lang_does_not_shadow_later_winners(monkeypatch):
    """A cached winner that later finds no text still falls through to others."""
    # Image 1: eng wins and is cached.
    order1 = _install_tess(
        monkeypatch,
        {
            "kor+eng": _TesseractError("no kor"),
            "eng": _ocr_data([("hello", 80)]),
        },
    )
    _detect_image_ocr_lang(_B64)
    assert providers_module._OCR_RESOLVED_LANG == "eng"
    assert order1 == ["kor+eng", "eng"]

    # Image 2: eng available but produces no text; jpn+eng has the text.
    order2 = _install_tess(
        monkeypatch,
        {
            "kor+eng": _TesseractError("no kor"),
            "eng": _ocr_data([]),
            "chi_sim+eng": _ocr_data([]),
            "jpn+eng": _ocr_data([("こんにちは", 88)]),
        },
    )
    lang2, _w, _h, data = _detect_image_ocr_lang(_B64)
    # eng probed first (cached) → no text → falls through; the rest stay in
    # priority order and the unavailable kor+eng is skipped via try/except.
    assert order2 == ["eng", "kor+eng", "chi_sim+eng", "jpn+eng"]
    assert lang2 == "jpn+eng"
    assert data is not None
    assert providers_module._OCR_RESOLVED_LANG == "jpn+eng"


# ── edge cases ─────────────────────────────────────────────────────────────


def test_no_text_in_any_pack_returns_default(monkeypatch):
    """All packs available but no text → default lang, data None, no cache."""
    order = _install_tess(
        monkeypatch,
        {
            "kor+eng": _ocr_data([]),
            "eng": _ocr_data([]),
            "chi_sim+eng": _ocr_data([]),
            "jpn+eng": _ocr_data([]),
        },
    )
    lang, _w, _h, data = _detect_image_ocr_lang(_B64)
    assert lang == "eng"
    assert data is None
    assert order == ["kor+eng", "eng", "chi_sim+eng", "jpn+eng"]
    assert providers_module._OCR_RESOLVED_LANG is None


def test_importerror_returns_default(monkeypatch):
    """Missing pytesseract → ImportError path → default lang, data None."""
    # None in sys.modules makes ``import pytesseract`` raise ImportError.
    monkeypatch.setitem(sys.modules, "pytesseract", None)
    lang, w, h, data = _detect_image_ocr_lang(_B64)
    assert lang == "eng"
    assert data is None
    assert (w, h) == (0, 0)


# ── language-pack probe (P1-1) ─────────────────────────────────────────────


def test_probe_filters_unavailable_packs_before_first_pass(monkeypatch):
    """--list-langs says only ``eng`` exists → ``kor+eng`` is filtered BEFORE
    any image_to_data pass.

    Pre-fix, an English-only install burned a whole ``kor+eng`` OCR pass
    (~1-5s) merely discovering ``kor`` was missing; the probe order was
    ``["kor+eng", "eng"]``.  Post-fix it is a single ``["eng"]`` pass.
    """
    order = _install_tess(
        monkeypatch,
        {
            "kor+eng": _TesseractError("kor.traineddata not found"),
            "eng": _ocr_data([("hello", 80)]),
        },
        langs=["eng"],
    )
    lang, _w, _h, data = _detect_image_ocr_lang(_B64)
    assert order == ["eng"]  # single pass — the missing pack never ran OCR
    assert lang == "eng"
    assert data is not None


def test_probe_combined_pack_needs_all_components(monkeypatch):
    """``kor+eng`` requires BOTH ``kor`` and ``eng`` in --list-langs to stay
    in the probe order; ``chi_sim+eng``/``jpn+eng`` are filtered out."""
    order = _install_tess(
        monkeypatch,
        {
            "kor+eng": _ocr_data([("안녕", 90)]),
            "eng": _ocr_data([("hi", 70)]),
        },
        langs=["kor", "eng"],  # no chi_sim / jpn packs installed
    )
    lang, _w, _h, data = _detect_image_ocr_lang(_B64)
    assert order == ["kor+eng"]  # combined pack probed, short-circuits
    assert lang == "kor+eng"
    assert data is not None


def test_probe_failure_falls_back_to_legacy_probing(monkeypatch):
    """--list-langs unavailable (no binary / build without the flag) → the
    legacy per-pack try/skip behavior is preserved unchanged."""
    order = _install_tess(
        monkeypatch,
        {
            "kor+eng": _TesseractError("kor.traineddata not found"),
            "eng": _ocr_data([("hello", 80)]),
        },
        langs=None,  # get_languages raises → probe fails
    )
    lang, _w, _h, data = _detect_image_ocr_lang(_B64)
    assert order == ["kor+eng", "eng"]  # absence discovered by failing passes
    assert lang == "eng"
    assert data is not None
    assert providers_module._OCR_LANG_PROBE_FAILED is True


def test_probe_empty_install_skips_all_passes(monkeypatch):
    """--list-langs reports no packs at all → no OCR pass is attempted and the
    default ``("eng", w, h, None)`` is returned."""
    order = _install_tess(monkeypatch, {}, langs=[])
    lang, _w, _h, data = _detect_image_ocr_lang(_B64)
    assert order == []
    assert lang == "eng"
    assert data is None


def test_probe_runs_once_per_process(monkeypatch):
    """The --list-langs subprocess runs once; later images reuse the cached set
    (and keep filtering against it)."""
    probe_calls: list[str] = []
    order = _install_tess(
        monkeypatch,
        {
            "kor+eng": _TesseractError("no kor"),
            "eng": _ocr_data([("hi", 70)]),
        },
        langs=["eng"],
        probe_calls=probe_calls,
    )
    lang1, *_ = _detect_image_ocr_lang(_B64)
    assert lang1 == "eng"
    lang2, *_ = _detect_image_ocr_lang(_B64)
    assert lang2 == "eng"
    assert len(probe_calls) == 1  # second image reuses the cached probe
    assert order == ["eng", "eng"]
