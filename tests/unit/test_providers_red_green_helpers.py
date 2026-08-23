"""RED→GREEN: helpers in external_llm/providers.py not covered by the legacy
test_providers_* suite — _safe_callback failure isolation, multi-system message
merging, _try_ocr_base64 body, and the _images_to_text OCR path."""

from __future__ import annotations

import base64
import sys

import pytest

import external_llm.providers as providers_module
from external_llm.client import LLMMessage
from external_llm.providers import (
    _images_to_text,
    _normalize_ollama_system_messages,
    _OCRTextCacheAccess,
    _ollama_apply_images,
    _safe_callback,
    _try_ocr_base64,
)

# ── _safe_callback ────────────────────────────────────────────────────────────


def test_safe_callback_invokes_callback() -> None:
    got: list[str] = []
    _safe_callback(lambda c: got.append(c), "tok")
    assert got == ["tok"]


def test_safe_callback_isolates_callback_failure(caplog) -> None:
    """A buggy streaming callback must not propagate — it is logged at debug."""
    with caplog.at_level("DEBUG", logger="external_llm.providers"):
        _safe_callback(lambda _c: (_ for _ in ()).throw(RuntimeError("cb broke")), "x")
    assert any("streaming callback" in r.getMessage() for r in caplog.records)


def test_safe_callback_reports_name_when_available(caplog) -> None:
    def named_cb(_c: str) -> None:  # pragma: no cover - never called
        raise ValueError("boom")

    with caplog.at_level("DEBUG", logger="external_llm.providers"):
        _safe_callback(named_cb, "x")
    assert any("named_cb" in r.getMessage() for r in caplog.records)


# ── _normalize_ollama_system_messages ────────────────────────────────────────


def test_normalize_system_messages_single_unchanged() -> None:
    msgs = [{"role": "system", "content": "A"}, {"role": "user", "content": "U"}]
    assert _normalize_ollama_system_messages(msgs) is msgs


def test_normalize_system_messages_multiple_merged() -> None:
    msgs = [
        {"role": "system", "content": "A"},
        {"role": "user", "content": "U1"},
        {"role": "system", "content": "B"},
        {"role": "assistant", "content": "R"},
    ]
    out = _normalize_ollama_system_messages(msgs)
    assert out[0] == {"role": "system", "content": "A\nB"}
    assert out[1:] == [{"role": "user", "content": "U1"}, {"role": "assistant", "content": "R"}]


def test_normalize_system_messages_preserves_first_sys_extra_keys() -> None:
    """Non-payload keys (e.g. 'images') from the first system message survive."""
    msgs = [
        {"role": "system", "content": "A", "images": ["x"]},
        {"role": "system", "content": ""},  # empty content contributes nothing
        {"role": "user", "content": "U"},
    ]
    out = _normalize_ollama_system_messages(msgs)
    assert out[0]["content"] == "A"
    assert out[0]["images"] == ["x"]
    assert out[1] == {"role": "user", "content": "U"}


def test_normalize_system_messages_empty_contents_fall_back() -> None:
    """Multiple system messages with NO content → merge impossible → untouched."""
    msgs = [
        {"role": "system", "content": ""},
        {"role": "system", "content": ""},
        {"role": "user", "content": "U"},
    ]
    assert _normalize_ollama_system_messages(msgs) is msgs


# ── _try_ocr_base64 ──────────────────────────────────────────────────────────


class _TesseractError(Exception):
    pass


class _Out:
    DICT = "DICT"


class _FakeImg:
    size = (100, 50)


class _FakeImage:
    @staticmethod
    def open(_buf):
        return _FakeImg()


def _ocr_data(words: list[tuple]) -> dict:
    """words: (text, conf, left, top, width, height)."""
    return {
        "text": [w[0] for w in words],
        "conf": [w[1] for w in words],
        "left": [w[2] for w in words],
        "top": [w[3] for w in words],
        "width": [w[4] for w in words],
        "height": [w[5] for w in words],
    }


class _PILModule:
    Image = _FakeImage


def _install_fake_ocr(monkeypatch, lang_to_outcome: dict) -> list[str]:
    order: list[str] = []

    def _image_to_data(_img, lang=None, output_type=None):
        order.append(lang)
        outcome = lang_to_outcome.get(lang)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome if outcome is not None else _ocr_data([])

    def _get_languages(config=""):
        return ["eng"]

    class _TessModule:
        image_to_data = staticmethod(_image_to_data)
        get_languages = staticmethod(_get_languages)
        Output = _Out
        TesseractError = _TesseractError

    monkeypatch.setitem(sys.modules, "pytesseract", _TessModule)
    monkeypatch.setitem(sys.modules, "PIL", _PILModule)
    monkeypatch.setitem(sys.modules, "PIL.Image", _FakeImage)
    return order


_B64 = base64.b64encode(b"\x89PNG\r\n\x1a\n").decode()


@pytest.fixture(autouse=True)
def _reset_ocr_cache():
    providers_module._OCR_RESOLVED_LANG = None
    providers_module._OCR_AVAILABLE_LANGS = None
    providers_module._OCR_LANG_PROBE_FAILED = False
    providers_module._OCRTextCacheAccess.clear()
    yield
    providers_module._OCR_RESOLVED_LANG = None
    providers_module._OCR_AVAILABLE_LANGS = None
    providers_module._OCR_LANG_PROBE_FAILED = False
    providers_module._OCRTextCacheAccess.clear()


def test_try_ocr_base64_empty_input() -> None:
    assert _try_ocr_base64("") == ""


def test_try_ocr_base64_no_deps_returns_empty() -> None:
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setitem(sys.modules, "pytesseract", None)
    try:
        assert _try_ocr_base64(_B64) == ""
    finally:
        monkeypatch.undo()


def test_try_ocr_base64_builds_position_labels(monkeypatch) -> None:
    """Two lines at different y → separate [v-h] labelled lines; conf filter."""
    _install_fake_ocr(
        monkeypatch,
        {
            # top-left line: y=5 (top), x=10 (left); plus a low-conf word dropped
            "eng": _ocr_data(
                [
                    ("hello", 80, 10, 5, 20, 10),
                    ("noise", 5, 40, 5, 20, 10),  # conf < 20 → filtered
                    ("world", 70, 80, 40, 20, 10),  # y=40 (bottom), x=80 (right)
                ]
            ),
        },
    )
    out = _try_ocr_base64(_B64)
    assert "[Image OCR — Position Includes (100\u00d750px):" in out
    assert "  [top-left] hello" in out
    assert "  [bottom-right] world" in out
    assert "noise" not in out


def test_try_ocr_base64_center_labels(monkeypatch) -> None:
    """middle-* and *-center labels for mid-range coordinates."""
    _install_fake_ocr(
        monkeypatch,
        {"eng": _ocr_data([("mid", 60, 40, 20, 10, 10)])},  # x=40 center, y=20 middle
    )
    out = _try_ocr_base64(_B64)
    assert "  [middle-center] mid" in out


def test_try_ocr_base64_no_words_returns_empty(monkeypatch) -> None:
    """All words below confidence threshold → empty string."""
    _install_fake_ocr(
        monkeypatch,
        {"eng": _ocr_data([("low", 3, 0, 0, 0, 0)])},
    )
    assert _try_ocr_base64(_B64) == ""


def test_try_ocr_base64_no_text_data_returns_empty(monkeypatch) -> None:
    """OCR succeeded but text list empty → empty string."""
    _install_fake_ocr(monkeypatch, {"eng": _ocr_data([])})
    assert _try_ocr_base64(_B64) == ""


# ── _images_to_text OCR path ─────────────────────────────────────────────────


def test_images_to_text_runs_ocr_and_caches(monkeypatch) -> None:
    monkeypatch.setattr(providers_module, "_try_ocr_base64", lambda _b: "FAKE OCR TEXT")
    imgs: list[dict] = [{"data": "abc"}]
    out = _images_to_text(imgs)
    assert "[Image 1 — OCR Extracted Text:\nFAKE OCR TEXT\n]" in out
    assert imgs[0]["ocr_text"] == "FAKE OCR TEXT"  # cached for reuse


def test_images_to_text_multiple_images_numbered(monkeypatch) -> None:
    monkeypatch.setattr(providers_module, "_try_ocr_base64", lambda _b: "T1")
    out = _images_to_text([{"data": "a"}, {"data": "b"}])
    assert "[Image 1 —" in out and "[Image 2 —" in out


# ── _ollama_apply_images non-vision fold ─────────────────────────────────────


def test_ollama_apply_images_non_vision_folds_ocr_text(monkeypatch) -> None:
    monkeypatch.setattr(providers_module, "_images_to_text", lambda imgs: "OCR!")
    m = {"role": "user", "content": "describe"}
    _ollama_apply_images(m, LLMMessage(role="user", content="describe", images=[{"data": "x"}]), False)
    assert m["content"] == "OCR!\ndescribe"


def test_ollama_apply_images_non_vision_empty_content(monkeypatch) -> None:
    monkeypatch.setattr(providers_module, "_images_to_text", lambda imgs: "OCR!")
    m = {"role": "user", "content": ""}
    _ollama_apply_images(m, LLMMessage(role="user", content="", images=[{"data": "x"}]), False)
    assert m["content"] == "OCR!"


def test_ollama_apply_images_no_images_noop() -> None:
    m = {"role": "user", "content": "c"}
    _ollama_apply_images(m, LLMMessage(role="user", content="c"), True)
    assert m == {"role": "user", "content": "c"}


def test_ollama_apply_images_vision_passes_base64(monkeypatch) -> None:
    monkeypatch.setattr(providers_module, "_images_to_text", lambda imgs: "NEVER")
    m = {"role": "user", "content": "c"}
    _ollama_apply_images(m, LLMMessage(role="user", content="c", images=[{"data": "QUJD"}]), True)
    assert m["images"] == ["QUJD"]


def test_images_to_text_placeholder_when_ocr_empty(monkeypatch) -> None:
    monkeypatch.setattr(providers_module, "_try_ocr_base64", lambda _b: "")
    imgs: list[dict] = [{"data": "x", "media_type": "image/png"}]
    out = _images_to_text(imgs)
    assert "Image 1 Attached (image/png)" in out
    assert imgs[0]["ocr_text"] == ""


def test_images_to_text_reuses_preexisting_ocr_text() -> None:
    imgs = [{"data": "x", "ocr_text": "pre"}]
    assert "[Image 1 — OCR Extracted Text:\npre\n]" in _images_to_text(imgs)


def test_images_to_text_preexisting_empty_ocr_text_placeholder() -> None:
    """A pre-computed empty ``ocr_text`` reuses the placeholder branch (no OCR)."""
    imgs = [{"data": "x", "ocr_text": "", "media_type": "image/jpeg"}]
    out = _images_to_text(imgs)
    assert "Image 1 Attached (image/jpeg)" in out


# ── module-level OCR text LRU (P18-7) ──────────────────────────────────────


def test_ocr_text_lru_memoizes_same_base64(monkeypatch) -> None:
    """Identical base64 in a FRESH dict (no pre-computed ``ocr_text``) must not
    re-run tesseract — the module-level LRU short-circuits it."""
    _OCRTextCacheAccess.clear()
    calls: list[str] = []

    def _fake_try(b64: str) -> str:
        calls.append(b64)
        return "FAKE OCR TEXT"

    monkeypatch.setattr(providers_module, "_try_ocr_base64", _fake_try)
    imgs1 = [{"data": "abc"}]
    imgs2 = [{"data": "abc"}]  # a NEW dict, no ocr_text key
    out1 = _images_to_text(imgs1)
    out2 = _images_to_text(imgs2)
    assert out1 == out2
    assert calls == ["abc"]  # single OCR pass for two separate dicts
    assert _OCRTextCacheAccess.keys() == ["abc"]
    assert _OCRTextCacheAccess.values() == ["FAKE OCR TEXT"]


def test_ocr_text_lru_distinct_data_separate_keys(monkeypatch) -> None:
    """Different base64 payloads are cached under separate keys (no cross-talk)."""
    _OCRTextCacheAccess.clear()
    seen: list[str] = []

    def _fake(b64: str) -> str:
        seen.append(b64)
        return f"OCR[{b64}]"

    monkeypatch.setattr(providers_module, "_try_ocr_base64", _fake)
    out_a = _images_to_text([{"data": "a"}])
    out_b = _images_to_text([{"data": "b"}])
    assert "[Image 1 — OCR Extracted Text:\nOCR[a]\n]" in out_a
    assert "[Image 1 — OCR Extracted Text:\nOCR[b]\n]" in out_b
    assert seen == ["a", "b"]
    assert set(_OCRTextCacheAccess.keys()) == {"a", "b"}


def test_ocr_text_lru_caches_empty_result(monkeypatch) -> None:
    """A textless image ('' result) is cached too — a repeat must not re-run."""
    _OCRTextCacheAccess.clear()
    calls: list[str] = []

    def _fake(b64: str) -> str:
        calls.append(b64)
        return ""

    monkeypatch.setattr(providers_module, "_try_ocr_base64", _fake)
    out1 = _images_to_text([{"data": "empty"}])
    out2 = _images_to_text([{"data": "empty"}])
    assert "Image 1 Attached (image)" in out1
    assert "Image 1 Attached (image)" in out2
    assert calls == ["empty"]


def test_ocr_text_lru_is_bounded(monkeypatch) -> None:
    """LRU evicts oldest entries past ``_OCR_TEXT_CACHE_MAX`` — memory bounded."""
    _OCRTextCacheAccess.clear()
    try:
        providers_module._OCR_TEXT_CACHE_MAX = 2
        monkeypatch.setattr(providers_module, "_try_ocr_base64", lambda b: f"OCR[{b}]")
        _images_to_text([{"data": "k1"}])
        _images_to_text([{"data": "k2"}])
        _images_to_text([{"data": "k3"}])  # evicts k1
        assert _OCRTextCacheAccess.keys() == ["k2", "k3"]
    finally:
        _OCRTextCacheAccess.clear()
        providers_module._OCR_TEXT_CACHE_MAX = 128
