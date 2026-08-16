"""Cross-process edit leases (F1) — advisory WIP-file ownership.

Two asicode sessions in parallel terminals have no working-tree ownership
signal, so they silently clobber each other's uncommitted WIP (documented
incidents: apply_patch AUTO-REPAIR resurrecting deleted symbols, git checkout
losing a session's edits, ruff reformatting mid-edit files). The lease module
records a per-file JSON lease under <repo_root>/.asicode/edit_leases/ on every
successful write-tool edit, and every write tool refuses a file carrying a
LIVE FOREIGN lease (mirroring the intra-session _refuse_session_edited UX).

Liveness matrix (the core contract under test):

  - own identity (host+pid+token)          -> never a conflict
  - same host, pid dead                    -> stale, no conflict
  - same host, pid == ours, token differs  -> recycled pid, no conflict
  - same host, pid alive, age <= 12h       -> CONFLICT
  - other host, age <= 30min               -> CONFLICT (pid not probeable)
  - older / corrupt / absent / empty root  -> no conflict (fail-open)

Plus the integration gates: apply_patch and edit_text refuse on a live
foreign lease WITHOUT mutating the working tree, and proceed once the lease
is gone.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from external_llm.common import edit_lease as el


def _write_lease(repo_root: Path, file_path: str, *, pid: int, host: str, token: str, ts: float) -> Path:
    """Plant a lease record directly (simulating a foreign session's acquire)."""
    key = el.normalize_lease_key(str(repo_root), file_path)
    lease_file = el._lease_file(str(repo_root), key)
    lease_file.parent.mkdir(parents=True, exist_ok=True)
    lease_file.write_text(
        json.dumps({"v": 1, "path": key, "pid": pid, "host": host, "token": token, "ts": ts}),
        encoding="utf-8",
    )
    return lease_file


def _conflict(repo_root: Path, file_path: str, **kw) -> list:
    return el.find_live_foreign_leases(str(repo_root), [file_path], **kw)


class TestLivenessMatrix:
    def test_own_lease_is_never_a_conflict(self, tmp_path):
        el.acquire_edit_lease(str(tmp_path), "src/a.py", tool="edit_text")
        assert el.read_edit_lease(str(tmp_path), "src/a.py") is not None
        assert _conflict(tmp_path, "src/a.py") == []

    def test_foreign_live_pid_conflicts_then_clears_on_death(self, tmp_path):
        proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
        try:
            _write_lease(
                tmp_path,
                "src/a.py",
                pid=proc.pid,
                host=el.session_identity()["host"],
                token="foreign-token",
                ts=time.time(),
            )
            conflicts = _conflict(tmp_path, "src/a.py")
            assert len(conflicts) == 1
            assert conflicts[0]["pid"] == proc.pid
            assert "lease_file" in conflicts[0]
        finally:
            proc.kill()
            proc.wait()
        # Owner process is gone -> lease goes stale immediately.
        assert _conflict(tmp_path, "src/a.py") == []

    def test_recycled_own_pid_with_foreign_token_is_not_a_conflict(self, tmp_path):
        """Our pid + foreign token = a dead predecessor whose pid was recycled
        into THIS process — not a live foreign owner."""
        ident = el.session_identity()
        _write_lease(
            tmp_path, "src/a.py", pid=ident["pid"], host=ident["host"], token="predecessor-token", ts=time.time()
        )
        assert _conflict(tmp_path, "src/a.py") == []

    def test_same_host_alive_pid_but_ancient_ts_stale(self, tmp_path):
        proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
        try:
            _write_lease(
                tmp_path,
                "src/a.py",
                pid=proc.pid,
                host=el.session_identity()["host"],
                token="foreign-token",
                ts=time.time() - (el.LEASE_TTL_SAME_HOST_S + 60),
            )
            assert _conflict(tmp_path, "src/a.py") == []
        finally:
            proc.kill()
            proc.wait()

    def test_cross_host_fresh_conflicts_and_old_does_not(self, tmp_path):
        _write_lease(tmp_path, "src/a.py", pid=1, host="other-host", token="t", ts=time.time() - 60)
        assert len(_conflict(tmp_path, "src/a.py")) == 1
        _write_lease(
            tmp_path, "src/a.py", pid=1, host="other-host", token="t", ts=time.time() - (el.LEASE_TTL_CROSS_HOST_S + 60)
        )
        assert _conflict(tmp_path, "src/a.py") == []


class TestFailOpen:
    def test_corrupt_lease_is_ignored(self, tmp_path):
        key = el.normalize_lease_key(str(tmp_path), "src/a.py")
        lease_file = el._lease_file(str(tmp_path), key)
        lease_file.parent.mkdir(parents=True, exist_ok=True)
        lease_file.write_bytes(b"{not json")
        assert _conflict(tmp_path, "src/a.py") == []

    def test_empty_repo_root_is_a_noop(self):
        el.acquire_edit_lease("", "src/a.py")
        assert el.find_live_foreign_leases("", ["src/a.py"]) == []
        assert el.read_edit_lease("", "src/a.py") is None

    def test_env_kill_switch_disables_both_directions(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ASICODE_EDIT_LEASES", "0")
        el.acquire_edit_lease(str(tmp_path), "src/a.py")
        assert el.read_edit_lease(str(tmp_path), "src/a.py") is None
        _write_lease(tmp_path, "src/a.py", pid=1, host="other-host", token="t", ts=time.time())
        assert _conflict(tmp_path, "src/a.py") == []


class TestAcquireAndSweep:
    def test_acquire_reclaims_and_rewrites_stale_lease(self, tmp_path):
        ident = el.session_identity()
        lease_file = _write_lease(
            tmp_path, "src/a.py", pid=999999999, host=ident["host"], token="dead-owner", ts=time.time()
        )
        el.acquire_edit_lease(str(tmp_path), "src/a.py", tool="edit_text")
        data = json.loads(lease_file.read_text(encoding="utf-8"))
        assert data["token"] == ident["token"]
        assert data["pid"] == ident["pid"]
        assert data["tool"] == "edit_text"

    def test_acquire_sweeps_week_old_lease_files(self, tmp_path):
        ancient = el._lease_dir(str(tmp_path)) / ("a" * 20 + ".json")
        ancient.parent.mkdir(parents=True, exist_ok=True)
        ancient.write_text("{}", encoding="utf-8")
        old = ancient.stat().st_mtime
        os.utime(ancient, (old - (el.SWEEP_MAX_AGE_S + 3600),) * 2)
        el.acquire_edit_lease(str(tmp_path), "src/other.py")
        assert not ancient.exists()
        # The freshly acquired lease itself survives the sweep.
        assert el.read_edit_lease(str(tmp_path), "src/other.py") is not None

    def test_absolute_and_relative_paths_share_one_key(self, tmp_path):
        el.acquire_edit_lease(str(tmp_path), "src/a.py")
        abs_form = str(tmp_path / "src" / "a.py")
        lease = el.read_edit_lease(str(tmp_path), abs_form)
        assert lease is not None and lease["path"] == "src/a.py"


# ── Integration: write tools refuse on a live foreign lease ──────────────────

_PATCH = """diff --git a/sample.py b/sample.py
--- a/sample.py
+++ b/sample.py
@@ -1,4 +1,4 @@
 def foo():
-    return 1
+    return 100

 def bar():
"""

_ORIG = "def foo():\n    return 1\n\ndef bar():\n    return 2\n"


class _Stub:
    """Minimal WriteToolsMixin host — mirrors test_apply_patch_dirty_warning."""

    def __init__(self, repo_root):
        from external_llm.agent.tool_handlers.write_tools import WriteToolsMixin

        # Dynamic mixin so this class keeps only the harness attributes here.
        class _Host(WriteToolsMixin):
            def _secure_path(self, p, confine=False, **_kw):
                """Minimal confinement-resolving stand-in for the registry's."""
                cand = Path(p) if Path(p).is_absolute() else Path(self.repo_root) / p
                resolved = cand.resolve()
                root = Path(self.repo_root).resolve()
                if confine and resolved != root and root not in resolved.parents:
                    return None
                return resolved

            def _make_result(self, **kw):
                from external_llm.agent.tool_registry import ToolResult

                kw.setdefault("content", "")
                return ToolResult(**kw)

        self._impl = _Host()
        self._impl.repo_root = str(repo_root)
        self._impl._effective_repo_root = str(repo_root)
        self._impl._applied_patches = []
        self._impl._text_edited_files = set()

    def __getattr__(self, name):
        return getattr(self._impl, name)


def _init_repo(tmp_path):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "t@t.com"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "test"], cwd=tmp_path, check=True)
    (tmp_path / "sample.py").write_text(_ORIG, encoding="utf-8")
    subprocess.run(["git", "add", "sample.py"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=tmp_path, check=True)
    return _Stub(tmp_path)


class TestWriteToolRefusal:
    def test_apply_patch_refuses_live_foreign_lease_without_mutating(self, tmp_path):
        stub = _init_repo(tmp_path)
        proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
        try:
            _write_lease(
                tmp_path, "sample.py", pid=proc.pid, host=el.session_identity()["host"], token="foreign", ts=time.time()
            )
            res = stub._apply_patch_text(_PATCH, path_hint="sample.py")
            assert not res.ok
            assert res.metadata["reason"] == "foreign_edit_lease"
            conflicts = res.metadata["foreign_lease_conflicts"]
            assert conflicts[0]["path"] == "sample.py"
            # The working tree was NOT mutated by the refused apply.
            assert (tmp_path / "sample.py").read_text(encoding="utf-8") == _ORIG

            # Escape hatch stated in the error: removing the lease unblocks.
            Path(conflicts[0]["lease_file"]).unlink()
            res2 = stub._apply_patch_text(_PATCH, path_hint="sample.py")
            assert res2.ok
        finally:
            proc.kill()
            proc.wait()

    def test_edit_text_refuses_live_foreign_lease(self, tmp_path):
        stub = _init_repo(tmp_path)
        proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
        try:
            _write_lease(
                tmp_path, "sample.py", pid=proc.pid, host=el.session_identity()["host"], token="foreign", ts=time.time()
            )
            res = stub._tool_edit_text(
                {
                    "file_path": "sample.py",
                    "old_string": "return 1",
                    "new_string": "return 42",
                }
            )
            assert not res.ok
            assert res.metadata["reason"] == "foreign_edit_lease"
            assert (tmp_path / "sample.py").read_text(encoding="utf-8") == _ORIG
        finally:
            proc.kill()
            proc.wait()

    def test_apply_patch_success_acquires_lease(self, tmp_path):
        """A successful apply must stake our lease so the OTHER session's
        write tools now see the file as our WIP."""
        stub = _init_repo(tmp_path)
        res = stub._apply_patch_text(_PATCH, path_hint="sample.py")
        assert res.ok
        lease = el.read_edit_lease(str(tmp_path), "sample.py")
        assert lease is not None
        assert lease["pid"] == os.getpid()
        # And our own lease never blocks our own follow-up edit.
        assert _conflict(tmp_path, "sample.py") == []
