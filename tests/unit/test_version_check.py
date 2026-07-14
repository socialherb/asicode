"""Unit tests for utils.version_check — background PyPI update notice.

Design properties under test (see module docstring):
* non-blocking: interval-gated, daemon-thread fetch;
* crash-safe / resilient-load cache: corrupt JSON ignored & repaired;
* env override (ASICODE_VERSION_CHECK_CACHE) for hermetic tests;
* pluggable fetcher so no real network is hit.

Per repo convention, imports are deferred to test bodies where convenient and
``monkeypatch.setenv`` is used to redirect the cache to a temp file.
"""
from __future__ import annotations

import json
import time


def _set_cache(monkeypatch, tmp_path, data):
    """Point the version-check cache at a temp path seeded with ``data``."""
    path = tmp_path / "version_check.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    monkeypatch.setenv("ASICODE_VERSION_CHECK_CACHE", str(path))
    return path


# ─── version comparison ──────────────────────────────────────────────────────
class TestIsNewer:
    def test_semver_precedence(self):
        from utils.version_check import is_newer

        # String comparison would wrongly say "0.9.0" > "0.10.0".
        assert is_newer("0.10.0", "0.9.0") is True
        assert is_newer("0.9.0", "0.10.0") is False

    def test_equal_is_not_newer(self):
        from utils.version_check import is_newer

        assert is_newer("1.2.3", "1.2.3") is False

    def test_release_after_pre_release(self):
        from utils.version_check import is_newer

        assert is_newer("1.0.0", "1.0.0rc1") is True

    def test_invalid_versions_are_not_newer(self):
        from utils.version_check import is_newer

        assert is_newer("not-a-version", "1.0.0") is False
        assert is_newer("", "1.0.0") is False


# ─── current version ─────────────────────────────────────────────────────────
def test_get_current_version_is_string(monkeypatch):
    from utils.version_check import get_current_version

    v = get_current_version()
    assert isinstance(v, str)
    assert v != ""


# ─── cache path override & resilient load ────────────────────────────────────
def test_env_override_cache_path(monkeypatch, tmp_path):
    target = tmp_path / "vc.json"
    monkeypatch.setenv("ASICODE_VERSION_CHECK_CACHE", str(target))

    from utils.version_check import _cache_path

    assert _cache_path() == target


def test_read_cache_missing_returns_empty(monkeypatch, tmp_path):
    monkeypatch.setenv("ASICODE_VERSION_CHECK_CACHE", str(tmp_path / "nope.json"))

    from utils.version_check import _read_cache

    assert _read_cache() == {}


def test_read_cache_corrupt_returns_empty(monkeypatch, tmp_path):
    # A torn/partial write of a previous version must not raise — resilient load.
    path = tmp_path / "vc.json"
    path.write_text("{not valid json", encoding="utf-8")
    monkeypatch.setenv("ASICODE_VERSION_CHECK_CACHE", str(path))

    from utils.version_check import _read_cache

    assert _read_cache() == {}


def test_write_then_read_roundtrip(monkeypatch, tmp_path):
    monkeypatch.setenv("ASICODE_VERSION_CHECK_CACHE", str(tmp_path / "vc.json"))

    from utils.version_check import _read_cache, _write_cache

    _write_cache({"last_check_ts": 123.0, "latest": "9.9.9"})
    got = _read_cache()
    assert got["latest"] == "9.9.9"
    assert got["last_check_ts"] == 123.0


def test_write_replaces_corrupt_file(monkeypatch, tmp_path):
    path = tmp_path / "vc.json"
    path.write_text("garbage", encoding="utf-8")
    monkeypatch.setenv("ASICODE_VERSION_CHECK_CACHE", str(path))

    from utils.version_check import _read_cache, _write_cache

    _write_cache({"latest": "2.0.0"})
    assert _read_cache().get("latest") == "2.0.0"


# ─── start_update_check: interval gating & immediate notice ──────────────────
def test_cached_newer_version_gives_immediate_notice(monkeypatch, tmp_path):
    _set_cache(monkeypatch, tmp_path, {"latest": "0.9.0", "last_check_ts": time.time()})

    from utils.version_check import start_update_check

    handle = start_update_check(current_version="0.2.6", fetcher=lambda *_: None)
    # Within interval → no thread started; notice comes straight from cache.
    assert handle._thread is None
    notice = handle.collect(wait_s=0.0)
    assert notice is not None
    assert "0.9.0" in notice
    assert "0.2.6" in notice


def test_no_notice_when_up_to_date(monkeypatch, tmp_path):
    _set_cache(monkeypatch, tmp_path, {"latest": "0.2.6", "last_check_ts": time.time()})

    from utils.version_check import start_update_check

    handle = start_update_check(current_version="0.2.6", fetcher=lambda *_: None)
    assert handle.notice is None
    assert handle.collect(wait_s=0.0) is None


def test_overdue_starts_background_thread(monkeypatch, tmp_path):
    # last_check far in the past → interval exceeded → a fetch thread must start.
    _set_cache(monkeypatch, tmp_path, {"latest": "0.2.6", "last_check_ts": 0})

    calls = {"n": 0}

    def fake_fetcher(url, timeout):
        calls["n"] += 1
        return "0.3.0"

    from utils.version_check import start_update_check

    handle = start_update_check(
        current_version="0.2.6", fetcher=fake_fetcher, now_ts=time.time()
    )
    assert handle._thread is not None
    # Give the daemon thread a moment to finish its (fake, instant) fetch.
    notice = handle.collect(wait_s=2.0)
    handle._thread.join(timeout=2.0)

    assert calls["n"] == 1
    assert notice is not None
    assert "0.3.0" in notice


def test_background_fetch_writes_cache(monkeypatch, tmp_path):
    path = tmp_path / "vc.json"
    monkeypatch.setenv("ASICODE_VERSION_CHECK_CACHE", str(path))

    from utils.version_check import _read_cache, start_update_check

    handle = start_update_check(
        current_version="0.2.6",
        fetcher=lambda *_: "1.2.3",
        now_ts=time.time(),
    )
    handle._thread.join(timeout=2.0)

    cache = _read_cache()
    assert cache.get("latest") == "1.2.3"
    assert float(cache["last_check_ts"]) > 0


def test_fetcher_exception_is_swallowed(monkeypatch, tmp_path):
    # A raising fetcher must not propagate; cache stays unset, no notice.
    monkeypatch.setenv("ASICODE_VERSION_CHECK_CACHE", str(tmp_path / "vc.json"))

    from utils.version_check import _read_cache, start_update_check

    def boom(*_a, **_k):
        raise RuntimeError("network down")

    handle = start_update_check(current_version="0.2.6", fetcher=boom)
    handle._thread.join(timeout=2.0)
    assert handle.collect(wait_s=0.0) is None
    assert _read_cache() == {}


def test_start_never_raises_on_bad_env(monkeypatch, tmp_path):
    # Point the cache at an unwritable path inside a (to-be-removed) parent.
    bad = tmp_path / "subdir" / "vc.json"
    monkeypatch.setenv("ASICODE_VERSION_CHECK_CACHE", str(bad))
    # Make the parent disappear so writes fail; reads of missing file are fine.
    # (subdir doesn't exist → _atomic_write_text will mkdir it, so instead deny
    # by pointing into a path whose parent is a file.)
    file_parent = tmp_path / "blocker"
    file_parent.write_text("x", encoding="utf-8")
    monkeypatch.setenv("ASICODE_VERSION_CHECK_CACHE", str(file_parent / "vc.json"))

    from utils.version_check import start_update_check

    # Should not raise regardless of the IO situation (fail-open contract).
    handle = start_update_check(
        current_version="0.2.6", fetcher=lambda *_: "9.9.9", now_ts=time.time()
    )
    handle._thread.join(timeout=2.0)
    # collect must also never raise.
    assert handle.collect(wait_s=0.0) is None or isinstance(handle.collect(wait_s=0.0), str)


def test_packaging_parse_not_required(monkeypatch, tmp_path):
    # is_newer must still return a boolean even if packaging import is sabotaged.
    import builtins

    real_import = builtins.__import__

    def blocking_import(name, *args, **kwargs):
        if name == "packaging.version":
            raise ImportError("blocked for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocking_import)

    from utils.version_check import is_newer

    # Falls back to inequality comparison → different versions are "newer".
    assert is_newer("2.0.0", "1.0.0") is True
    assert is_newer("1.0.0", "1.0.0") is False


def test_uninstalled_sentinel_skips_check(tmp_path, monkeypatch):
    # current == "0.0.0" (source checkout, not pip-installed): no notice, no
    # fetch — a pip-upgrade hint would be wrong when there is nothing pip
    # manages, and the cached-latest path must not fire either.
    cache = tmp_path / "vc.json"
    monkeypatch.setenv("ASICODE_VERSION_CHECK_CACHE", str(cache))
    import json as _json

    cache.write_text(_json.dumps({"latest": "9.9.9", "last_check_ts": 0}))

    from utils.version_check import start_update_check

    fetched = []
    handle = start_update_check(
        current_version="0.0.0", fetcher=lambda *_: fetched.append(1) or "9.9.9"
    )
    assert handle.notice is None
    assert handle._thread is None  # fetch not even started
    assert handle.collect(wait_s=0.5) is None
    assert fetched == []
