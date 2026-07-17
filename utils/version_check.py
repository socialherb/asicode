"""Background PyPI update check for the asicode CLI.

Once per ``interval`` (default 24h) the PyPI JSON API is queried for the latest
published version of the package; if it is newer than the running version a
one-line upgrade hint is produced. The network call runs on a daemon thread so
it never blocks CLI startup: a previously cached *latest* version is returned
immediately, and a fresh fetch merely refreshes the cache for the next run.

Design invariants
-----------------
* **Non-blocking.** The cache is read synchronously (cheap file read); the
  network call is delegated to a daemon thread. ``collect(wait_s=0)`` never
  blocks.
* **Crash-safe cache.** Writes use ``mkstemp`` + ``fsync`` + ``os.replace``
  (POSIX-atomic). A corrupt / partially-written / unreadable cache is ignored
  on read and silently overwritten on the next write (resilient load).
* **Testable.** The cache path is overridable via the
  ``ASICODE_VERSION_CHECK_CACHE`` env var (mirrors the
  ``ASICODE_WRITE_TOOL_FAILURE_LOG`` pattern), and the network call is pluggable
  via the ``fetcher`` argument. No real network is touched by the unit tests.
* **Fail-open.** Any unexpected error is swallowed — the update check must
  *never* break the CLI itself.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Optional

__all__ = [
    "PACKAGE_NAME",
    "UpdateCheckHandle",
    "get_current_version",
    "is_newer",
    "start_update_check",
]

# ─── configuration ───────────────────────────────────────────────────────────
PACKAGE_NAME = "asicode"
_PYPI_URL = f"https://pypi.org/pypi/{PACKAGE_NAME}/json"
# Network budget: PyPI JSON is tiny and fast; this is only a safety net against
# a hung connection (mirrors the subprocess-hang discipline of bounding reads).
_FETCH_TIMEOUT_S: float = 4.0
_DEFAULT_INTERVAL_HOURS: float = 24.0


# ─── current version ─────────────────────────────────────────────────────────
def get_current_version() -> str:
    """Return the installed package version, or ``"0.0.0"`` if unresolvable.

    Uses ``importlib.metadata`` (stdlib). On a source checkout that is not
    installed, the metadata lookup fails — we return the ``"0.0.0"`` sentinel,
    which :func:`start_update_check` treats as "not pip-installed" and skips
    the check entirely (a pip-upgrade hint would be wrong there).
    """
    try:
        from importlib.metadata import PackageNotFoundError, version

        try:
            return str(version(PACKAGE_NAME))
        except PackageNotFoundError:
            return "0.0.0"
    except Exception:
        return "0.0.0"


def _has_editable_distribution(package_name: str = PACKAGE_NAME) -> bool:
    """True if ANY discovered distribution of *package_name* is editable (PEP 610).

    Enumerates **all** distributions rather than taking the first match from
    ``distribution(name)``. Rationale: when the CLI runs from a source checkout,
    the repo root is prepended to ``sys.path`` (see ``asi.py``), so the
    checkout's ``*.egg-info`` shadows the real ``*.dist-info`` in site-packages.
    ``distribution(name)`` then resolves to the egg-info, which carries no
    ``direct_url.json`` and yields a false negative — the guard silently fails
    in the very "developer checkout" scenario it exists to protect. Iterating
    every distribution finds the editable dist-info regardless of which path
    entry wins first-match resolution.

    The *package_name* argument (defaulting to :data:`PACKAGE_NAME`) lets a
    regression test reproduce the egg-info shadowing with a throwaway package
    name that cannot collide with the real install on the host.

    Fail-open: any metadata lookup error returns ``False`` (treat as a normal
    install; worst case a redundant notice appears, which is benign).
    """
    try:
        from importlib.metadata import distributions

        target = (package_name or "").lower()
        for dist in distributions():
            try:
                name = dist.metadata["Name"]
            except Exception:
                continue
            if not name or name.lower() != target:
                continue
            raw = dist.read_text("direct_url.json")
            if not raw:
                continue
            try:
                if json.loads(raw).get("dir_info", {}).get("editable"):
                    return True
            except Exception:
                continue
        return False
    except Exception:
        return False


def _is_editable_install() -> bool:
    """True if the package is installed in editable/development mode (PEP 610).

    Thin wrapper over :func:`_has_editable_distribution` bound to
    :data:`PACKAGE_NAME`. See that function for the egg-info shadowing
    rationale and the editable-skip policy in :func:`start_update_check`.
    """
    return _has_editable_distribution(PACKAGE_NAME)


def _resolve_editable_install(current: str) -> bool:
    """Resolve editable-install status, memoised by the installed version.

    The site-packages scan in :func:`_is_editable_install` (≈95 ms: it
    enumerates *every* distribution to defeat egg-info shadowing) is a pure
    function of the installed version — between two ``pip install`` events the
    answer never changes. Persisting it next to the version it was computed for
    lets every subsequent launch of the SAME version skip the scan entirely (a
    cheap cache read instead); after an upgrade/reinstall the version differs
    and it is recomputed exactly once.

    Keyed on a dedicated ``editable_version`` stamp (NOT ``current_at_check``,
    which :func:`_bg_fetch` rewrites on every fetch) so the memo is decoupled
    from the fetch cadence. A cache MISS delegates to
    :func:`_is_editable_install` (the function tests monkeypatch), so the
    resolved value honours any test override rather than a stale cached one.
    Fully fail-open: on any error falls back to a live scan, never blocks.
    """
    try:
        cache = _read_cache()
        if cache.get("editable_version") == current and "editable" in cache:
            return bool(cache["editable"])
        editable = _is_editable_install()
        # Persist so subsequent launches of the same version skip the scan.
        # Merge (not replace) preserves fetch keys; _write_cache swallows errors.
        _write_cache({**cache, "editable": bool(editable), "editable_version": current})
        return bool(editable)
    except Exception:
        # Never let a cache hiccup block the check — scan live as a fallback.
        return _is_editable_install()


# ─── cache path ──────────────────────────────────────────────────────────────
def _cache_path() -> Path:
    """Resolve the cache file path.

    Honors ``ASICODE_VERSION_CHECK_CACHE`` (absolute path override, used by
    tests to redirect to a temp file). Defaults to ``~/.asicode/version_check.json``.
    """
    override = os.environ.get("ASICODE_VERSION_CHECK_CACHE")
    if override:
        return Path(override)
    return Path(os.path.expanduser("~")) / ".asicode" / "version_check.json"


def _read_cache() -> dict[str, Any]:
    """Read & parse the cache. Returns ``{}`` if missing or corrupt.

    Resilient load: a truncated/partial JSON file (e.g. crash mid-write of a
    previous version that predated atomic writes, or external tampering) must
    not raise — we treat it as an empty cache and let the next write repair it.
    """
    path = _cache_path()
    try:
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw)
        if isinstance(data, dict):
            return data
        return {}
    except (OSError, ValueError):
        return {}


def _atomic_write_text(path: Path, content: str) -> None:
    """Crash-safe text write: ``mkstemp`` + ``fsync`` + ``os.replace``.

    A truncating ``open(path, "w")`` can leave a partial/empty file if the
    process is interrupted mid-write; the atomic temp-file + rename sequence
    guarantees the cache is either the previous contents or the new contents,
    never a torn write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    base_dir = str(path.parent)
    fd, tmp_path = tempfile.mkstemp(dir=base_dir, prefix=".vc-atomic-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        # Best-effort cleanup of the temp file on any failure path.
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _write_cache(data: dict[str, Any]) -> None:
    """Persist the cache atomally (errors swallowed — fail-open)."""
    try:
        _atomic_write_text(_cache_path(), json.dumps(data))
    except OSError:
        pass


# ─── version comparison ──────────────────────────────────────────────────────
def is_newer(latest: str, current: str) -> bool:
    """Return True iff ``latest`` is strictly newer than ``current``.

    Uses ``packaging.version`` (bundled with pip/setuptools, so virtually always
    present). Falls back to a conservative inequality check so a missing
    ``packaging`` never breaks the feature.
    """
    try:
        from packaging.version import InvalidVersion
        from packaging.version import parse as _parse

        try:
            return _parse(latest) > _parse(current)
        except InvalidVersion:
            return False
    except Exception:
        return bool(latest) and bool(current) and latest != current


# ─── network fetch (pluggable for tests) ─────────────────────────────────────
def _default_fetch(url: str, timeout: float) -> Optional[str]:
    """Query the PyPI JSON API and return the latest version string, or None.

    Uses the project's existing ``httpx`` dependency (sync client). Any error —
    DNS, timeout, non-200, malformed body — yields None so callers can treat
    "no answer" uniformly.
    """
    import httpx

    try:
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            resp = client.get(url)
            resp.raise_for_status()
            data = resp.json()
    except Exception:
        return None
    version = str((data.get("info") or {}).get("version", "")).strip()
    return version or None


# ─── notice formatting ───────────────────────────────────────────────────────
def _format_notice(latest: str, current: str) -> str:
    return (
        f"[{PACKAGE_NAME}] update available: {latest} (current: {current})"
        f" · pip install --upgrade {PACKAGE_NAME}"
    )


# ─── handle & entry point ────────────────────────────────────────────────────
class UpdateCheckHandle:
    """Opaque handle returned by :func:`start_update_check`.

    Carries the immediately-known (cached) notice and (optionally) the daemon
    thread performing a fresh fetch. ``collect`` lets a caller briefly wait for
    the fetch to finish so a first-ever run (empty cache) can still surface a
    notice this session.
    """

    __slots__ = ("_current", "_notice", "_thread")

    def __init__(self, current: str) -> None:
        self._thread: Optional[threading.Thread] = None
        self._notice: Optional[str] = None
        self._current: str = current

    @property
    def notice(self) -> Optional[str]:
        """Notice string if a newer version is already known (cached); else None."""
        return self._notice

    def collect(self, wait_s: float = 0.0) -> Optional[str]:
        """Optionally wait for the background fetch, then return the notice.

        ``wait_s=0`` is fully non-blocking: it returns the cached notice
        immediately. If a fetch thread is alive and ``wait_s>0`` is given, this
        waits up to ``wait_s`` seconds; once the thread has finished it re-reads
        the cache so a freshly-discovered version is reflected. Never raises.
        """
        thread = self._thread
        if thread is not None and thread.is_alive() and wait_s > 0:
            thread.join(wait_s)
        if thread is not None and not thread.is_alive():
            # Thread finished (now or just above): pick up any newer cached value.
            try:
                latest = _read_cache().get("latest")
            except Exception:
                latest = None
            if latest and is_newer(str(latest), self._current):
                self._notice = _format_notice(str(latest), self._current)
        return self._notice


def _bg_fetch(current: str, fetcher: Callable[[str, float], Optional[str]]) -> None:
    """Daemon-thread body: attempt a fetch, then refresh the cache.

    Always records ``last_check_ts`` (the *attempt* time) so the interval gate
    in :func:`start_update_check` suppresses retries for ``interval`` even when
    the fetch fails. Without this an offline host would spawn a fresh daemon
    thread + network call on **every** CLI launch: the failed fetch left the
    cache untouched, so ``now - last_check`` stayed forever overdue and the gate
    re-fired each run. On failure the last-known ``latest`` is preserved
    (re-read from the cache) so a previously-discovered update is neither lost
    nor fabricated (never write a made-up value). All errors swallowed.
    """
    try:
        latest = fetcher(_PYPI_URL, _FETCH_TIMEOUT_S)
    except Exception:
        latest = None
    now = time.time()
    # Read once: reused for both latest-preservation (failure path) and the
    # merge below. Merging — instead of replacing the whole dict — preserves
    # the editable memo written by ``_resolve_editable_install`` so a fetch
    # never forces a fresh site-packages scan on the next launch.
    cache = _read_cache()
    if not latest:
        # Preserve the last-known good value; never fabricate one on failure.
        latest = cache.get("latest")
    cache.update(
        {
            "last_check_ts": now,
            "latest": latest,
            "current_at_check": current,
        }
    )
    _write_cache(cache)


def start_update_check(
    *,
    interval_hours: float = _DEFAULT_INTERVAL_HOURS,
    current_version: Optional[str] = None,
    fetcher: Optional[Callable[[str, float], Optional[str]]] = None,
    now_ts: Optional[float] = None,
) -> UpdateCheckHandle:
    """Start a (possibly skipped) background update check; return a handle.

    Behaviour:
    * The cache is read first; if it records a *latest* newer than the running
      version, the handle's ``notice`` is populated **immediately** (no network).
    * If the last successful check is older than ``interval_hours`` (or there is
      no cache yet), a daemon thread performs a fresh fetch and refreshes the
      cache for subsequent runs. The fetcher defaults to :func:`_default_fetch`
      but is injectable for tests.

    This function never raises — a failure here must not impair the CLI.
    """
    current = current_version if current_version is not None else get_current_version()
    handle = UpdateCheckHandle(current)

    # Uninstalled source checkout OR editable install: there is no pip-managed
    # install to upgrade, so a notice would be actionable-looking but wrong
    # ("current: 0.0.0", or — for an editable checkout — actively harmful since
    # following the upgrade hint would replace the checkout with a wheel).
    # Skip both the notice and the fetch entirely.
    if current == "0.0.0" or _resolve_editable_install(current):
        return handle

    try:
        cache = _read_cache()
        cached_latest = cache.get("latest")
        if cached_latest and is_newer(str(cached_latest), current):
            handle._notice = _format_notice(str(cached_latest), current)

        last_check = float(cache.get("last_check_ts") or 0)
        now = now_ts if now_ts is not None else time.time()
        interval_s = max(0.0, interval_hours) * 3600.0
        if now - last_check >= interval_s:
            thread = threading.Thread(
                target=_bg_fetch,
                args=(current, fetcher or _default_fetch),
                name=f"{PACKAGE_NAME}-version-check",
                daemon=True,
            )
            handle._thread = thread
            thread.start()
    except Exception:
        # Fail-open: any hiccup just yields a handle with no notice.
        pass

    return handle
