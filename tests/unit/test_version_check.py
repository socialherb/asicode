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

import pytest


@pytest.fixture(autouse=True)
def _force_non_editable_host(monkeypatch):
    """Hermeticity: decouple every test from the host's editable-install state.

    On a developer checkout ``_is_editable_install()`` legitimately returns
    True, which makes ``start_update_check`` short-circuit (skip + no background
    thread). Tests here exercise the fetch / comparison / detection logic and
    must not depend on how the package happens to be installed on the host, so
    the default is a non-editable install. Tests asserting the editable-SKIP
    path override this explicitly (``monkeypatch.setattr(vc, "_is_editable_install",
    lambda: True)``); the detection unit tests call ``_has_editable_distribution``
    directly and are thus unaffected.
    """
    monkeypatch.setattr("utils.version_check._is_editable_install", lambda: False)


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
    # A raising fetcher must not propagate and must show no notice. The cache
    # now records last_check_ts (the *attempt* time) so the interval gate
    # suppresses retries — an offline host won't spawn a fetch thread on every
    # launch — but it never fabricates a `latest` value.
    monkeypatch.setenv("ASICODE_VERSION_CHECK_CACHE", str(tmp_path / "vc.json"))

    from utils.version_check import _read_cache, start_update_check

    def boom(*_a, **_k):
        raise RuntimeError("network down")

    handle = start_update_check(current_version="0.2.6", fetcher=boom)
    handle._thread.join(timeout=2.0)
    assert handle.collect(wait_s=0.0) is None
    cache = _read_cache()
    assert not cache.get("latest")  # nothing fabricated → no false notice
    assert float(cache.get("last_check_ts") or 0) > 0  # attempt recorded → gates retry


def test_failed_fetch_gates_retry_on_next_launch(monkeypatch, tmp_path):
    """Regression: a failed fetch must bump ``last_check_ts`` so the next launch
    (within the interval) does NOT start another fetch thread.

    Previously ``_bg_fetch`` wrote the cache only on success; on failure the
    cache stayed empty, so ``now - last_check`` was forever overdue and every
    CLI launch re-spawned a daemon thread + network call — wasteful on a
    persistently-offline host (airplane, air-gapped network, DNS outage).
    """
    cache = tmp_path / "vc.json"
    monkeypatch.setenv("ASICODE_VERSION_CHECK_CACHE", str(cache))

    from utils.version_check import _read_cache, start_update_check

    def offline(*_a, **_k):
        raise OSError("offline")

    # Launch 1: empty cache → overdue → fetch thread starts and fails offline.
    h1 = start_update_check(current_version="0.2.6", fetcher=offline)
    assert h1._thread is not None
    h1._thread.join(timeout=2.0)
    after = _read_cache()
    assert float(after.get("last_check_ts") or 0) > 0  # attempt recorded
    assert not after.get("latest")  # nothing fabricated

    # Launch 2 a moment later: the recorded attempt timestamp must gate the
    # retry → NO new fetch thread within the interval.
    h2 = start_update_check(current_version="0.2.6", fetcher=offline)
    assert h2._thread is None


def test_failed_fetch_preserves_known_latest(monkeypatch, tmp_path):
    """A failed fetch must not clobber a previously-known ``latest``.

    If PyPI was reachable yesterday (latest=0.9.0 cached) but is unreachable
    today, the failed fetch must preserve 0.9.0 so the user still sees the
    notice — while still bumping ``last_check_ts`` to rate-limit retries.
    """
    _set_cache(
        monkeypatch,
        tmp_path,
        {"latest": "0.9.0", "last_check_ts": 0, "current_at_check": "0.2.6"},
    )

    from utils.version_check import _read_cache, start_update_check

    def offline(*_a, **_k):
        raise OSError("offline")

    handle = start_update_check(current_version="0.2.6", fetcher=offline)
    handle._thread.join(timeout=2.0)

    cache = _read_cache()
    assert cache.get("latest") == "0.9.0"  # preserved, not clobbered
    assert float(cache.get("last_check_ts") or 0) > 0  # retry gated


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


# ─── editable install (PEP 610) skip ────────────────────────────────────────
# An editable install (`pip install -e .`) is developer-managed: an "update
# available" notice is harmful (following the hint replaces the checkout with a
# wheel). _is_editable_install must detect it and start_update_check must skip.


def test_editable_install_skips_check(tmp_path, monkeypatch):
    # Real version on disk + editable install → must skip (no notice, no fetch),
    # even though a cached newer version exists. This is the regression: an
    # editable checkout used to get "update available: X (current: Y)" and
    # suggest pip-upgrade, which would clobber the checkout.
    cache = tmp_path / "vc.json"
    monkeypatch.setenv("ASICODE_VERSION_CHECK_CACHE", str(cache))
    import json as _json

    cache.write_text(_json.dumps({"latest": "9.9.9", "last_check_ts": 0}))

    import utils.version_check as vc

    monkeypatch.setattr(vc, "_is_editable_install", lambda: True)

    fetched = []
    handle = vc.start_update_check(
        current_version="0.2.6", fetcher=lambda *_: fetched.append(1) or "9.9.9"
    )
    assert handle.notice is None
    assert handle._thread is None  # fetch not started
    assert handle.collect(wait_s=0.5) is None
    assert fetched == []


def test_non_editable_install_runs_check(tmp_path, monkeypatch):
    # Mirror of the above: editable=False must NOT short-circuit — the normal
    # cached-newer-version immediate-notice path still fires.
    cache = tmp_path / "vc.json"
    monkeypatch.setenv("ASICODE_VERSION_CHECK_CACHE", str(cache))
    import json as _json

    cache.write_text(_json.dumps({"latest": "9.9.9", "last_check_ts": time.time()}))

    import utils.version_check as vc

    monkeypatch.setattr(vc, "_is_editable_install", lambda: False)

    handle = vc.start_update_check(current_version="0.2.6", fetcher=lambda *_: None)
    assert handle.notice is not None  # immediate notice from cache


def test_is_editable_install_parses_direct_url(monkeypatch):
    # _is_editable_install enumerates distributions() and returns True if ANY
    # distribution carries a direct_url.json with dir_info.editable set.
    import importlib.metadata as _meta
    import utils.version_check as vc

    class _FakeDist:
        def __init__(self, name, payload):
            self.metadata = {"Name": name}
            self._payload = payload

        def read_text(self, name):
            if name == "direct_url.json":
                return self._payload
            return None

    def _with(payload):
        monkeypatch.setattr(
            _meta, "distributions", lambda: [_FakeDist("asicode", payload)]
        )
        # Call the real detection logic directly (the autouse fixture replaces
        # the _is_editable_install wrapper with a constant).
        return vc._has_editable_distribution(vc.PACKAGE_NAME)

    assert _with('{"dir_info": {"editable": true}}') is True
    assert _with('{"dir_info": {"editable": false}}') is False
    assert _with('{"url": "file:///x"}') is False  # no dir_info key
    assert _with(None) is False  # no direct_url.json (non-editable wheel install)


def test_editable_ignores_unrelated_distributions(monkeypatch):
    # A non-asicode editable install must NOT trip the asicode guard; the name
    # filter inside _has_editable_distribution scopes the enumeration.
    import importlib.metadata as _meta
    import utils.version_check as vc

    class _FakeDist:
        def __init__(self, name, payload):
            self.metadata = {"Name": name}
            self._payload = payload

        def read_text(self, name):
            return self._payload if name == "direct_url.json" else None

    monkeypatch.setattr(
        _meta,
        "distributions",
        lambda: [
            _FakeDist("other-pkg", '{"dir_info": {"editable": true}}'),
            _FakeDist("asicode", None),  # asicode present but NOT editable
        ],
    )
    assert vc._has_editable_distribution(vc.PACKAGE_NAME) is False


def test_is_editable_install_resilient(monkeypatch):
    # Any metadata error must fail-open to False (benign: worst case a redundant
    # notice), never raise.
    import importlib.metadata as _meta
    import utils.version_check as vc

    def _boom():
        raise RuntimeError("metadata explosion")

    monkeypatch.setattr(_meta, "distributions", _boom)
    assert vc._has_editable_distribution(vc.PACKAGE_NAME) is False


def test_editable_not_fooled_by_egginfo_shadowing(tmp_path, monkeypatch):
    # Regression (real-world in-situ): asi.py prepends the repo root to sys.path,
    # so the checkout's *.egg-info (NO direct_url.json) shadows the real editable
    # *.dist-info in site-packages for distribution(name) first-match resolution.
    # The guard must still detect editable mode by enumerating ALL distributions.
    # Uses a throwaway package name that cannot collide with the host install.
    import importlib
    import importlib.metadata as _meta
    import json as _json
    import utils.version_check as vc

    pkg = "asicodeshadowfixturerepro"

    # checkout *.egg-info — on sys.path ahead of site-packages, no direct_url.json
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    egg_info = repo_root / f"{pkg}.egg-info"
    egg_info.mkdir()
    (egg_info / "PKG-INFO").write_text(
        "Metadata-Version: 2.1\nName: %s\nVersion: 0.0.0\n" % pkg
    )

    # real editable *.dist-info in site-packages — carries direct_url.json
    site_dir = tmp_path / "site"
    site_dir.mkdir()
    dist_info = site_dir / f"{pkg}-0.2.10.dist-info"
    dist_info.mkdir()
    (dist_info / "PKG-INFO").write_text(
        "Metadata-Version: 2.1\nName: %s\nVersion: 0.2.10\n" % pkg
    )
    (dist_info / "direct_url.json").write_text(
        _json.dumps({"url": repo_root.as_uri(), "dir_info": {"editable": True}})
    )

    # Mirror asi.py sys.path ordering: repo root first (egg-info wins first-match),
    # site-packages second (holds the editable dist-info).
    monkeypatch.syspath_prepend(str(site_dir))
    monkeypatch.syspath_prepend(str(repo_root))
    importlib.invalidate_caches()

    # Sanity: the shadowing is real — distribution() resolves to the egg-info,
    # which has no direct_url.json (the exact false-negative path the fix targets).
    try:
        shadowed = _meta.distribution(pkg)
        assert shadowed.read_text("direct_url.json") is None, (
            "test invariant: egg-info must shadow the editable dist-info for "
            "distribution() first-match; otherwise this regression is moot"
        )
    except _meta.PackageNotFoundError:
        pass  # some importlib versions; the enumeration assertion below is decisive

    # The fix: enumerating ALL distributions finds the editable dist-info anyway.
    assert vc._has_editable_distribution(pkg) is True


# ─── editable memo: version-keyed scan cache ─────────────────────────────────
# _resolve_editable_install memoises the (~95ms) site-packages scan by the
# installed version so steady-state launches skip it. A cache MISS delegates to
# _is_editable_install (the function the autouse fixture / tests monkeypatch),
# so the resolved value honours an override rather than a stale cache entry;
# a HIT returns the cached value without any scan.
class TestEditableResolutionCaching:
    def test_cache_hit_skips_live_scan(self, monkeypatch, tmp_path):
        # Seed a hit for the current version → _is_editable_install must NOT be
        # called. We monkeypatch it to raise to PROVE the cached value is used.
        _set_cache(
            monkeypatch, tmp_path,
            {"editable": True, "editable_version": "0.2.6",
             "latest": "9.9.9", "last_check_ts": time.time()},
        )
        import utils.version_check as vc

        def _must_not_scan():
            raise AssertionError("cache hit must not trigger a live scan")

        monkeypatch.setattr(vc, "_is_editable_install", _must_not_scan)
        handle = vc.start_update_check(
            current_version="0.2.6", fetcher=lambda *_: None,
        )
        # editable=True (cached) → skip: no notice, no fetch thread.
        assert handle.notice is None
        assert handle._thread is None

    def test_version_change_recomputes_and_persists(self, monkeypatch, tmp_path):
        # Cached for an OLD version → the new version must miss, recompute via
        # the (monkeypatched) scan, and persist the fresh value.
        path = _set_cache(
            monkeypatch, tmp_path,
            {"editable": True, "editable_version": "0.2.5"},
        )
        import utils.version_check as vc

        calls = {"n": 0}

        def counting_scan():
            calls["n"] += 1
            return False

        monkeypatch.setattr(vc, "_is_editable_install", counting_scan)
        vc.start_update_check(current_version="0.2.6", fetcher=lambda *_: None)

        assert calls["n"] == 1  # miss → scan invoked exactly once
        cache = json.loads(path.read_text())
        assert cache["editable"] is False
        assert cache["editable_version"] == "0.2.6"

    def test_second_launch_same_version_skips_scan(self, monkeypatch, tmp_path):
        # The core perf win: after the first launch persists the memo, a second
        # launch of the SAME version must NOT re-scan. last_check_ts seeded to
        # now so no background thread is started (avoids an async cache write
        # racing the second launch's read).
        _set_cache(monkeypatch, tmp_path, {"last_check_ts": time.time()})
        import utils.version_check as vc

        calls = {"n": 0}

        def counting_scan():
            calls["n"] += 1
            return False

        monkeypatch.setattr(vc, "_is_editable_install", counting_scan)

        vc.start_update_check(current_version="0.2.6", fetcher=lambda *_: None)
        assert calls["n"] == 1
        vc.start_update_check(current_version="0.2.6", fetcher=lambda *_: None)
        assert calls["n"] == 1  # unchanged → memo hit, scan NOT re-invoked

    def test_bg_fetch_preserves_editable_memo(self, monkeypatch, tmp_path):
        # Regression: _bg_fetch used to write a FRESH dict, dropping the
        # editable memo and forcing a re-scan next launch. It now
        # read-merge-writes so the memo survives a fetch.
        path = _set_cache(
            monkeypatch, tmp_path,
            {"editable": False, "editable_version": "0.2.6", "last_check_ts": 0},
        )
        import utils.version_check as vc

        handle = vc.start_update_check(
            current_version="0.2.6", fetcher=lambda *_: "1.0.0",
        )
        assert handle._thread is not None  # last_check_ts 0 → overdue → fetch
        handle._thread.join(timeout=2.0)

        cache = json.loads(path.read_text())
        # Fetch refreshed latest / last_check_ts …
        assert cache.get("latest") == "1.0.0"
        assert float(cache.get("last_check_ts") or 0) > 0
        # … WITHOUT dropping the editable memo (the merge fix).
        assert cache.get("editable") is False
        assert cache.get("editable_version") == "0.2.6"
