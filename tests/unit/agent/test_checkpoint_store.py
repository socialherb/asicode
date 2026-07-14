"""Unit tests for CheckpointStore — CRUD + scan + error paths."""
import json
import shutil
from pathlib import Path
from unittest.mock import patch

import pytest

from external_llm.agent.checkpoint_store import CheckpointStore

# NOTE: CheckpointStore._scan_files() returns dict with PosixPath keys.
# All dict-key assertions use Path('...') or str(k) conversions.


# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture
def repo_root(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "file_a.py").write_text("x = 1")
    (root / "file_b.txt").write_text("hello")
    (root / ".git").mkdir()
    (root / ".git" / "HEAD").write_text("ref: main")
    (root / "__pycache__").mkdir()
    (root / "__pycache__" / "cache.pyc").write_text("cached")
    (root / ".asicode").mkdir()
    (root / ".asicode" / "config.json").write_text('{"key": "val"}')
    # Invalid UTF-8 bytes → UnicodeDecodeError on read
    (root / "binary.bin").write_bytes(b"\xff\xfe\x00\x01")
    return root


@pytest.fixture
def store_dir(tmp_path: Path) -> Path:
    d = tmp_path / "stores"
    d.mkdir()
    return d


@pytest.fixture
def store(repo_root: Path, store_dir: Path) -> CheckpointStore:
    return CheckpointStore(str(repo_root), str(store_dir))


# ── _scan_files ───────────────────────────────────────────────────────────

class TestScanFiles:
    def test_returns_path_keys(self, store):
        """_scan_files() returns PosixPath keys."""
        hashes = store._scan_files()
        keys = list(hashes.keys())
        assert all(isinstance(k, Path) for k in keys)

    def test_returns_sha256_hashes(self, store):
        hashes = store._scan_files()
        assert Path("file_a.py") in hashes
        assert Path("file_b.txt") in hashes
        h = hashes[Path("file_a.py")]
        assert isinstance(h, str)
        assert len(h) == 64  # SHA256 hex

    def test_excludes_excluded_dirs(self, store):
        hashes = store._scan_files()
        keys_s = [str(k) for k in hashes]
        assert not any(k.startswith(".git") for k in keys_s)
        assert not any(k.startswith("__pycache__") for k in keys_s)
        assert not any(k.startswith(".asicode") for k in keys_s)
    def test_excludes_dot_dirs_like_ruff_cache(self, store, repo_root):
            """Dot-dirs are pruned wholesale — a single .ruff_cache held 26k files
            (95% of scan time). Regression guard against re-introducing the brittle
            allowlist that missed cache/tool dot-dirs."""
            cache_dir = repo_root / ".ruff_cache" / "0"
            cache_dir.mkdir(parents=True)
            (cache_dir / "junk.py").write_text("x")
            (repo_root / ".sometool").mkdir()
            (repo_root / ".sometool" / "data.json").write_text("{}")
            hashes = store._scan_files()
            keys_s = [str(k) for k in hashes]
            assert not any(".ruff_cache" in k for k in keys_s)
            assert not any(".sometool" in k for k in keys_s)

    def test_excludes_pyc_extensions(self, store):
        hashes = store._scan_files()
        keys_s = [str(k) for k in hashes]
        assert not any(k.endswith(".pyc") for k in keys_s)

    def test_excludes_own_store_dir(self, store):
        hashes = store._scan_files()
        keys_s = [str(k) for k in hashes]
        assert not any("stores" in k for k in keys_s)

    def test_skips_unreadable_files(self, store, repo_root):
        f = repo_root / "unreadable.txt"
        f.write_text("data")
        f.chmod(0o000)
        try:
            hashes = store._scan_files()
            assert Path("unreadable.txt") not in hashes
        finally:
            f.chmod(0o644)

    def test_new_file_appears(self, store, repo_root):
        h1 = store._scan_files()
        (repo_root / "extra.py").write_text("new")
        h2 = store._scan_files()
        assert Path("extra.py") in h2
        assert len(h2) == len(h1) + 1


# ── Create ────────────────────────────────────────────────────────────────

class TestCreate:
    def test_returns_id(self, store):
        cid = store.create("first")
        assert cid.startswith("checkpoint_")
        assert len(store.checkpoints) == 1
        assert store.checkpoints[0]["id"] == cid

    def test_empty_description(self, store):
        store.create("")
        assert store.checkpoints[0]["description"] == ""

    def test_multiple_sorted_newest_first(self, store):
        cid1 = store.create("a")
        cid2 = store.create("b")
        cid3 = store.create("c")
        assert [cp["id"] for cp in store.checkpoints] == [cid3, cid2, cid1]

    def test_writes_json_file(self, store):
        cid = store.create("persist")
        cp_file = store.checkpoint_dir / f"{cid}.json"
        assert cp_file.exists()
        data = json.loads(cp_file.read_text())
        assert data["id"] == cid
        assert data["description"] == "persist"
        assert "file_a.py" in data["files"]
        assert data["files"]["file_a.py"] == "x = 1"

    def test_tracks_new_file_content(self, store, repo_root):
        (repo_root / "new_file.py").write_text("y = 2")
        cid = store.create("track")
        data = json.loads((store.checkpoint_dir / f"{cid}.json").read_text())
        assert "new_file.py" in data["files"]
        assert data["files"]["new_file.py"] == "y = 2"

    def test_binary_file_stored_as_base64_sentinel(self, store):
        """Binary files are stored as sentinel + base64 (not empty string).

        Storing '' would silently produce a 0-byte file on restore, corrupting
        binary assets. The sentinel lets restore() decode exact bytes back.
        """
        cid = store.create("binary")
        data = json.loads((store.checkpoint_dir / f"{cid}.json").read_text())
        stored = data["files"]["binary.bin"]
        assert stored.startswith("__asr_binary_b64__:")
        import base64 as _b64
        raw = _b64.b64decode(stored[len("__asr_binary_b64__:"):])
        assert raw == b"\xff\xfe\x00\x01"  # exact bytes from fixture

    def test_excludes_dirs_in_json(self, store):
        cid = store.create("exclude")
        data = json.loads((store.checkpoint_dir / f"{cid}.json").read_text())
        keys = list(data["file_hashes"].keys())
        assert not any(".git" in k for k in keys)
        assert not any("__pycache__" in k for k in keys)
        assert not any(".asicode" in k for k in keys)

    def test_writes_checkpoints_metadata(self, store):
        cid = store.create("meta")
        meta = json.loads(store.checkpoint_file.read_text())
        assert len(meta) == 1
        assert meta[0]["id"] == cid


# ── List ──────────────────────────────────────────────────────────────────

class TestList:
    def test_empty(self, store):
        assert store.list() == []

    def test_after_create(self, store):
        cid = store.create("list test")
        lst = store.list()
        assert len(lst) == 1
        assert lst[0]["id"] == cid
        assert lst[0]["description"] == "list test"
        assert lst[0]["file_count"] >= 2
        assert "timestamp" in lst[0]
        assert "path" not in lst[0]

    def test_fields(self, store):
        store.create("fields")
        assert set(store.list()[0].keys()) == {"id", "timestamp", "description", "scope", "file_count"}

    def test_order_newest_first(self, store):
        ids = [store.create(f"cp{i}") for i in range(3)]
        assert [cp["id"] for cp in store.list()] == ids[::-1]


# ── Restore ───────────────────────────────────────────────────────────────

class TestRestore:
    def test_restores_file_content(self, store, repo_root):
        (repo_root / "file_a.py").write_text("modified")
        cid = store.create("before")
        (repo_root / "file_a.py").write_text("corrupted")
        assert store.restore(cid) is True
        assert repo_root.joinpath("file_a.py").read_text() == "modified"

    def test_nonexistent_id(self, store):
        assert store.restore("nonexistent") is False

    def test_missing_checkpoint_file(self, store):
        cid = store.create("missing")
        (store.checkpoint_dir / f"{cid}.json").unlink()
        assert store.restore(cid) is False

    def test_corrupted_json(self, store):
        cid = store.create("corrupt")
        (store.checkpoint_dir / f"{cid}.json").write_text("not json")
        assert store.restore(cid) is False

    def test_creates_missing_directories(self, store, repo_root):
        (repo_root / "subdir" / "nested.py").parent.mkdir()
        (repo_root / "subdir" / "nested.py").write_text("nested")
        cid = store.create("nested")
        shutil.rmtree(repo_root / "subdir")
        assert not (repo_root / "subdir").exists()
        assert store.restore(cid) is True
        assert (repo_root / "subdir" / "nested.py").read_text() == "nested"

    def test_restores_multiple_files(self, store, repo_root):
        (repo_root / "extra.py").write_text("extra")
        cid = store.create("multi")
        for f in ["file_a.py", "file_b.txt", "extra.py"]:
            (repo_root / f).write_text("modified")
        assert store.restore(cid) is True
        assert repo_root.joinpath("file_a.py").read_text() == "x = 1"
        assert repo_root.joinpath("file_b.txt").read_text() == "hello"
        assert repo_root.joinpath("extra.py").read_text() == "extra"


# ── Delete ────────────────────────────────────────────────────────────────

class TestDelete:
    def test_removes_entry(self, store):
        cid = store.create("del")
        assert store.delete(cid) is True
        assert len(store.checkpoints) == 0

    def test_removes_file(self, store):
        cid = store.create("del file")
        f = store.checkpoint_dir / f"{cid}.json"
        assert f.exists()
        store.delete(cid)
        assert not f.exists()

    def test_updates_metadata(self, store):
        cid = store.create("del meta")
        assert len(json.loads(store.checkpoint_file.read_text())) == 1
        store.delete(cid)
        assert json.loads(store.checkpoint_file.read_text()) == []

    def test_nonexistent(self, store):
        assert store.delete("nonexistent") is False

    def test_preserves_others(self, store):
        ids = [store.create(f"cp{i}") for i in range(3)]
        store.delete(ids[1])
        remaining = [cp["id"] for cp in store.checkpoints]
        assert ids[0] in remaining
        assert ids[1] not in remaining
        assert ids[2] in remaining


# ── Init ──────────────────────────────────────────────────────────────────

class TestInit:
    def test_reloads_persisted(self, repo_root, store_dir):
        cid = CheckpointStore(str(repo_root), str(store_dir)).create("persist")
        s2 = CheckpointStore(str(repo_root), str(store_dir))
        assert len(s2.checkpoints) == 1
        assert s2.checkpoints[0]["id"] == cid

    def test_empty_store(self, repo_root, tmp_path):
        s = CheckpointStore(str(repo_root), str(tmp_path / "new"))
        assert s.checkpoints == []
        assert s.checkpoint_dir.exists()

    def test_corrupted_metadata(self, repo_root, store_dir):
        meta_file = store_dir / repo_root.name / "checkpoints.json"
        meta_file.parent.mkdir(parents=True, exist_ok=True)
        meta_file.write_text("bad json")
        s = CheckpointStore(str(repo_root), str(store_dir))
        assert s.checkpoints == []

    def test_creates_nested_dirs(self, repo_root, tmp_path):
        nested = tmp_path / "a" / "b" / "c"
        s = CheckpointStore(str(repo_root), str(nested))
        assert s.checkpoint_dir.exists()


# ── Error paths for inline coverage ───────────────────────────────────────

class TestScanFilesErrorPaths:
    """Coverage for _scan_files error paths: excluded extension (80), checkpoint dir (84)."""

    def test_excludes_pyc_file_in_repo(self, store, repo_root):
        """Line 80: a .pyc file in the repo is skipped."""
        (repo_root / "compiled.pyc").write_text("fake bytecode")
        hashes = store._scan_files()
        assert Path("compiled.pyc") not in hashes

    def test_checkpoint_dir_inside_repo(self, repo_root, tmp_path):
        """Line 84: files under checkpoint_dir are skipped when store_dir is inside repo."""
        # Place store_dir INSIDE repo_root so checkpoint_dir in _scan_files overlaps
        store_dir = repo_root / ".ckpts"
        store_dir.mkdir()
        s = CheckpointStore(str(repo_root), str(store_dir))
        # checkpoint_dir = store_dir / repo_root.name = .ckpts/repo
        (s.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        (s.checkpoint_dir / "inside_ckpt.py").write_text("x = 1")
        hashes = s._scan_files()
        assert not any("inside_ckpt.py" in str(k) for k in hashes)


class TestCreateErrorPaths:
    """Coverage for create() error paths: IOError writing checkpoint file (140-142)."""

    def test_ioerror_writing_checkpoint_file(self, store):
        """Line 140-142: IOError when writing individual checkpoint file propagates."""
        # Make checkpoint_dir read-only so creating a new file fails
        original_mode = store.checkpoint_dir.stat().st_mode
        store.checkpoint_dir.chmod(0o555)
        try:
            with pytest.raises(IOError):
                store.create("fail_on_write")
        finally:
            store.checkpoint_dir.chmod(original_mode)


class TestRestoreErrorPaths:
    """Coverage for restore() error paths: IOError writing file (225-227), partial restore (232)."""

    def test_ioerror_writing_restored_file(self, store, repo_root):
        """Lines 225-227: IOError when writing a restored file marks failure."""
        cid = store.create("restore_io")
        # Make file_a.py read-only so restore fails on it
        f = repo_root / "file_a.py"
        f.write_text("modified")
        f.chmod(0o444)
        try:
            result = store.restore(cid)
            assert result is False
        finally:
            f.chmod(0o644)

    def test_partial_restore_logs_warning(self, store, repo_root):
        """Line 232: one file fails but others succeed → partial restore."""
        (repo_root / "extra.py").write_text("extra content")
        cid = store.create("partial")
        # Make extra.py read-only
        extra = repo_root / "extra.py"
        extra.write_text("modified extra")
        extra.chmod(0o444)
        try:
            result = store.restore(cid)
            assert result is False
            # file_a.py should still be restored
            assert repo_root.joinpath("file_a.py").read_text() == "x = 1"
        finally:
            extra.chmod(0o644)


class TestDeleteErrorPaths:
    """Coverage for delete() error paths: OSError on unlink (263-265)."""

    def test_oserror_deleting_checkpoint_file(self, store):
        """Lines 263-265: OSError when unlinking checkpoint file returns False."""
        cid = store.create("del_oserror")
        with patch.object(Path, 'unlink', side_effect=OSError("Permission denied")):
            result = store.delete(cid)
            assert result is False
        # Verify checkpoint still exists (not removed from list)
        assert any(cp['id'] == cid for cp in store.checkpoints)


class TestSaveCheckpointsErrorPaths:
    """Coverage for _save_checkpoints: IOError (55-57)."""

    def test_ioerror_saving_checkpoints(self, store):
        """_save_checkpoints IOError/OSError propagates.

        With the atomic write (tmp + os.replace), forcing a failure by
        chmod-ing the existing checkpoints.json no longer works (os.replace
        succeeds regardless of the target file's mode, as long as the
        directory is writable). Instead, patch os.replace to raise OSError so
        the exception-propagation contract of _save_checkpoints is verified.
        """
        cid = store.create("save_io")
        # Patch os.replace (used inside checkpoint_store) to raise OSError.
        import external_llm.agent.checkpoint_store as csm
        with patch.object(csm.os, "replace", side_effect=OSError("replace blocked")):
            with pytest.raises(OSError):
                store.delete(cid)


# ── Edge cases ────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_empty_repo(self, tmp_path):
        root = tmp_path / "empty"
        root.mkdir()
        s = CheckpointStore(str(root), str(tmp_path / "cp"))
        cid = s.create("empty")
        assert cid is not None
        assert s.checkpoints[0]["file_count"] == 0

    def test_large_file(self, store, repo_root):
        (repo_root / "large.py").write_text("x\n" * 10000)
        cid = store.create("large")
        data = json.loads((store.checkpoint_dir / f"{cid}.json").read_text())
        assert data["files"]["large.py"] == "x\n" * 10000

    def test_unicode_content(self, store, repo_root):
        (repo_root / "uni.py").write_text("print('한글')")
        cid = store.create("uni")
        data = json.loads((store.checkpoint_dir / f"{cid}.json").read_text())
        assert data["files"]["uni.py"] == "print('한글')"

    def test_create_then_restore_idempotent(self, store, repo_root):
        original = repo_root.joinpath("file_a.py").read_text()
        cid = store.create("idem")
        assert store.restore(cid) is True
        assert repo_root.joinpath("file_a.py").read_text() == original

    def test_scan_after_empty_repo(self, tmp_path):
        root = tmp_path / "empty"
        root.mkdir()
        s = CheckpointStore(str(root), str(tmp_path / "cp"))
        assert s._scan_files() == {}


# ── Binary round-trip + eviction + concurrency ────────────────────────────

class TestBinaryRoundTrip:
    """F4 fix: binary files survive a create→restore cycle byte-for-byte."""

    def test_binary_restore_exact_bytes(self, store, repo_root):
        original = b"\x89PNG\r\n\x1a\n\x00\x00\x00\x01\xff\xfe"
        (repo_root / "img.png").write_bytes(original)
        cid = store.create("img")
        # Corrupt then restore.
        (repo_root / "img.png").write_bytes(b"corrupted")
        assert store.restore(cid) is True
        assert repo_root.joinpath("img.png").read_bytes() == original

    def test_text_file_unchanged_by_sentinel_logic(self, store, repo_root):
        """Plain UTF-8 text is stored verbatim (no sentinel prefix)."""
        cid = store.create("text")
        data = json.loads((store.checkpoint_dir / f"{cid}.json").read_text())
        stored = data["files"]["file_a.py"]
        assert not stored.startswith("__asr_binary_b64__:")
        assert stored == "x = 1"


class TestMaxCheckpointsEviction:
    """P3 fix: oldest checkpoints are evicted when max_checkpoints is exceeded."""

    def test_evicts_oldest_when_limit_exceeded(self, repo_root, store_dir):
        s = CheckpointStore(str(repo_root), str(store_dir), max_checkpoints=3)
        ids = []
        for i in range(5):
            ids.append(s.create(f"cp{i}"))
            # Ensure strictly increasing timestamps so eviction order is
            # deterministic (otherwise same-second ties make sort unstable).
            import time as _t
            _t.sleep(0.005)
        # Only the 3 newest should remain.
        remaining = {cp["id"] for cp in s.checkpoints}
        assert remaining == set(ids[2:5])  # cp2, cp3, cp4
        # Evicted checkpoint files are gone from disk.
        for evicted_id in ids[:2]:
            assert not (s.checkpoint_dir / f"{evicted_id}.json").exists()

    def test_zero_max_disables_eviction(self, repo_root, store_dir):
        s = CheckpointStore(str(repo_root), str(store_dir), max_checkpoints=0)
        for i in range(6):
            s.create(f"cp{i}")
        assert len(s.checkpoints) == 6

    def test_default_max_is_50(self, repo_root, store_dir):
        s = CheckpointStore(str(repo_root), str(store_dir))
        assert s.max_checkpoints == 50


class TestConcurrentSaveMerge:
    """B4 fix: two processes checkpointing concurrently don't lose entries.

    Simulates a concurrent writer by directly manipulating the on-disk index
    to represent another process's commit, then verifies _save_checkpoints
    merges it rather than clobbering.
    """

    def test_merges_concurrent_addition(self, store, store_dir, repo_root):
        # Process A (our `store`) creates one checkpoint.
        cid_a = store.create("A")
        # Simulate process B committing directly to disk under the lock.
        # B must also leave its checkpoint .json file on disk, because the
        # merge only resurrects entries whose data file still exists.
        cid_b = "checkpoint_concurrent_b"
        (store.checkpoint_dir / f"{cid_b}.json").write_text(f'{{"id": "{cid_b}"}}')
        import json as _json

        from external_llm.common.file_lock import cross_process_flock
        lock_path = store.checkpoint_file.with_suffix('.json.lock')
        with cross_process_flock(lock_path):
            disk = _json.loads(store.checkpoint_file.read_text())
            disk.append({
                'id': cid_b, 'timestamp': 9999999999.0, 'description': 'B',
                'file_count': 1, 'path': f'{cid_b}.json',
            })
            store.checkpoint_file.write_text(_json.dumps(disk))
        # Now process A creates another — must preserve cid_b.
        cid_a2 = store.create("A2")
        ids = {cp["id"] for cp in store.checkpoints}
        assert cid_a in ids
        assert cid_b in ids
        assert cid_a2 in ids


# ── os.walk pruning ──────────────────────────────────────────────────────────

def test_scan_files_prunes_vendor_dirs(tmp_path: Path):
    """_scan_files() prunes node_modules/.venv/etc via os.walk dirs[:] instead
    of rglob('*') which descends into every directory before filtering."""
    root = tmp_path / "repo"
    root.mkdir()
    (root / "src").mkdir()
    (root / "src" / "main.py").write_text("print('hello')")
    (root / "node_modules" / "pkg").mkdir(parents=True)
    (root / "node_modules" / "pkg" / "index.js").write_text("module.exports = 1;")
    (root / ".venv" / "lib").mkdir(parents=True)
    (root / ".venv" / "lib" / "site.py").write_text("# venv")

    s = CheckpointStore(str(root), str(tmp_path / "store"))
    hashes = s._scan_files()

    assert Path("src/main.py") in hashes
    assert not any("node_modules" in str(k) for k in hashes), \
        "node_modules should be pruned"
    assert not any(".venv" in str(k) for k in hashes), \
        ".venv should be pruned"


# ── Scoped (file-list) checkpoints ───────────────────────────────────────────

class TestScopedCheckpoint:
    def test_scan_listed_files_only_hashes_given_paths(self, store, repo_root):
        hashes = store._scan_files(files=["file_a.py"])
        assert set(hashes) == {Path("file_a.py")}

    def test_accepts_absolute_paths_under_repo_root(self, store, repo_root):
        hashes = store._scan_files(files=[str(repo_root / "file_b.txt")])
        assert set(hashes) == {Path("file_b.txt")}

    def test_skips_paths_outside_repo_root(self, store, tmp_path):
        outside = tmp_path / "outside.py"
        outside.write_text("evil = 1")
        hashes = store._scan_files(files=[str(outside), "../outside.py"])
        assert hashes == {}

    def test_skips_missing_files(self, store):
        """A plan may target a file it is about to create — not an error."""
        hashes = store._scan_files(files=["to_be_created.py", "file_a.py"])
        assert set(hashes) == {Path("file_a.py")}

    def test_skips_empty_and_directory_entries(self, store, repo_root):
        (repo_root / "adir").mkdir()
        hashes = store._scan_files(files=["", None, "adir", "file_a.py"])
        assert set(hashes) == {Path("file_a.py")}

    def test_create_scoped_snapshot_and_restore(self, store, repo_root):
        cid = store.create("scoped", files=["file_a.py"])
        # Only the listed file is stored.
        cp_path = store.checkpoint_dir / f"{cid}.json"
        data = json.loads(cp_path.read_text())
        assert data["scope"] == "files"
        assert set(data["files"]) == {"file_a.py"}
        # Mutate both files; restore must revert only the scoped one.
        (repo_root / "file_a.py").write_text("x = 999")
        (repo_root / "file_b.txt").write_text("changed")
        assert store.restore(cid) is True
        assert (repo_root / "file_a.py").read_text() == "x = 1"
        assert (repo_root / "file_b.txt").read_text() == "changed"

    def test_full_snapshot_scope_marker(self, store):
        cid = store.create("full")
        cp_path = store.checkpoint_dir / f"{cid}.json"
        data = json.loads(cp_path.read_text())
        assert data["scope"] == "full"
        entry = next(cp for cp in store.list() if cp["id"] == cid)
        assert entry["scope"] == "full"

    def test_list_defaults_scope_full_for_legacy_entries(self, store):
        store.create("legacy")
        for cp in store.checkpoints:
            cp.pop("scope", None)
        assert all(e["scope"] == "full" for e in store.list())


# ── Pre-write checkpoint gate (_checkpoint_plan_files) ───────────────────────

class TestCheckpointPlanFiles:
    """Mode/target resolution for the PLANNER lane's pre-write checkpoint."""

    @staticmethod
    def _ops(*paths):
        class _Op:
            def __init__(self, p):
                self.path = p
        return [_Op(p) for p in paths]

    def _gate(self):
        from external_llm.agent.agent_planner_pipeline import _checkpoint_plan_files
        return _checkpoint_plan_files

    @pytest.mark.parametrize("mode", ["0", "off", "false", "no", "OFF", " No "])
    def test_disabled_modes(self, mode):
        assert self._gate()(mode, self._ops("a.py")) == (False, None)

    @pytest.mark.parametrize("mode", [None, "", "1", "scoped", "true", "SCOPED"])
    def test_scoped_default_collects_sorted_unique_existing_paths(self, mode, tmp_path):
        """Scoped mode snapshots only target paths that resolve to an existing
        regular file under repo_root (mirrors CheckpointStore._scan_listed_files)."""
        (tmp_path / "a.py").write_text("a")
        (tmp_path / "b.py").write_text("b")
        ops = self._ops("b.py", "a.py", "b.py", None)
        assert self._gate()(mode, ops, repo_root=str(tmp_path)) == (True, ["a.py", "b.py"])

    def test_full_mode_returns_none_files(self):
        assert self._gate()("full", self._ops("a.py")) == (True, None)

    def test_scoped_with_no_paths_is_disabled(self):
        """Never pay the full-repo walk implicitly."""
        assert self._gate()("scoped", self._ops(None, None)) == (False, None)

    def test_no_operations_is_disabled(self):
        assert self._gate()("full", []) == (False, None)

    def test_scoped_create_only_plan_is_disabled(self, tmp_path):
        """A plan whose targets don't exist yet (create-only) has nothing to
        roll back to — skip the checkpoint so Undo isn't offered as a no-op
        that reports success while restoring nothing."""
        ops = self._ops("new_file.py", "brand_new.py")
        assert self._gate()("scoped", ops, repo_root=str(tmp_path)) == (False, None)

    def test_scoped_mixed_plan_snapshots_only_existing(self, tmp_path):
        """Existing file is snapshotted; to-be-created file is filtered out
        (create-only sibling does not block the snapshot nor get snapshotted)."""
        (tmp_path / "exists.py").write_text("x")
        ops = self._ops("exists.py", "to_create.py")
        assert self._gate()("scoped", ops, repo_root=str(tmp_path)) == (True, ["exists.py"])

    def test_scoped_path_outside_repo_is_filtered(self, tmp_path):
        """Paths escaping repo_root via "../" are not snapshotted (defense)."""
        ops = self._ops("../escape.py")
        assert self._gate()("scoped", ops, repo_root=str(tmp_path)) == (False, None)
