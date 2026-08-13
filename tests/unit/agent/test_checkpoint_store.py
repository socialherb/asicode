"""Unit tests for CheckpointStore — CRUD + scan + error paths."""
import json
import os
import shutil
from pathlib import Path
from unittest.mock import patch

import pytest

from external_llm.agent.checkpoint_store import CheckpointStore

# NOTE: CheckpointStore._scan_files() returns dict with PosixPath keys.
# All dict-key assertions use Path('...') or str(k) conversions.


def stored_bytes(store: CheckpointStore, cid: str, rel: str) -> bytes:
    """Exact bytes a checkpoint holds for *rel*, whichever storage it used.

    Content moved out of the index JSON into per-file blobs, because extend()
    rewrites the index on every write of a run and the index used to carry
    every captured file's full text — making the per-write cost grow with
    everything captured so far. Tests that reached into ``data["files"]`` were
    asserting that layout rather than the guarantee (that the exact pre-write
    bytes come back), so they go through here instead.
    """
    data = json.loads((store.checkpoint_dir / f"{cid}.json").read_text())
    inline = data.get("files", {})
    if rel in inline:
        content = inline[rel]
        if content.startswith("__asr_binary_b64__:"):
            import base64 as _b64
            return _b64.b64decode(content[len("__asr_binary_b64__:"):])
        return content.encode("utf-8")
    return (store.checkpoint_dir / f"{cid}.files" / data["file_hashes"][rel]).read_bytes()


def stored_paths(store: CheckpointStore, cid: str) -> set:
    """Repo-relative paths a checkpoint holds content for."""
    data = json.loads((store.checkpoint_dir / f"{cid}.json").read_text())
    return set(data["file_hashes"])


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

class TestResolveRepoRelative:
    """Direct contract of the shared _resolve_repo_relative helper (R1)."""

    def test_absolute_in_root_returns_resolved_and_relative(self, store, repo_root):
        hit = store._resolve_repo_relative(str(repo_root / "file_a.py"))
        assert hit is not None
        resolved, relative = hit
        assert resolved == (repo_root / "file_a.py").resolve()
        assert relative == Path("file_a.py")

    def test_repo_relative_input_resolves_under_root(self, store, repo_root):
        hit = store._resolve_repo_relative("file_b.txt")
        assert hit is not None
        assert hit[1] == Path("file_b.txt")

    def test_empty_entry_returns_none(self, store):
        assert store._resolve_repo_relative("") is None
        assert store._resolve_repo_relative(None) is None

    def test_path_escaping_root_returns_none(self, store, repo_root):
        outside = repo_root.parent / "outside.py"
        outside.write_text("x = 1")
        assert store._resolve_repo_relative(str(outside)) is None
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
        assert "file_a.py" in data["file_hashes"]
        assert stored_bytes(store, cid, "file_a.py") == b"x = 1"

    def test_tracks_new_file_content(self, store, repo_root):
        (repo_root / "new_file.py").write_text("y = 2")
        cid = store.create("track")
        assert "new_file.py" in stored_paths(store, cid)
        assert stored_bytes(store, cid, "new_file.py") == b"y = 2"

    def test_binary_file_stored_as_exact_bytes(self, store):
        """Binary files keep their exact bytes (never an empty string).

        Storing '' would silently produce a 0-byte file on restore, corrupting
        binary assets. Blobs hold raw bytes, so binary and text are the same
        case and the sentinel + base64 encoding the inline format needed to
        survive JSON — which also inflated every binary by a third — is gone.
        """
        cid = store.create("binary")
        assert stored_bytes(store, cid, "binary.bin") == b"\xff\xfe\x00\x01"

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
            with pytest.raises(PermissionError):
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
        with (
            patch.object(csm.os, "replace", side_effect=OSError("replace blocked")),
            pytest.raises(OSError, match="replace blocked"),
        ):
            store.delete(cid)


class TestSaveCheckpointsDurability:
    """The index write must be durable (fsync), not just atomic-rename."""

    def test_index_save_uses_atomic_write_json(self, store):
        """_save_checkpoints routes the index through atomic_write_json
        (sibling tmp → fsync → os.replace) rather than a hand-rolled tmp +
        os.replace that skipped fsync. Without fsync a crash/power-loss could
        leave a 0-byte index and _load_checkpoints() would silently reset to
        [] — losing the whole index even though each payload was fsync'd."""
        from external_llm.agent import checkpoint_store as csm

        cid = store.create("durability")
        with patch.object(
            csm, "atomic_write_json", wraps=csm.atomic_write_json
        ) as spy:
            store.delete(cid)  # triggers _save_checkpoints on the index
        index_calls = [
            ca for ca in spy.call_args_list
            if str(ca.args[0]) == str(store.checkpoint_file)
        ]
        assert index_calls, (
            "index write did not go through atomic_write_json (no fsync)"
        )


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
        assert stored_bytes(store, cid, "large.py") == b"x\n" * 10000

    def test_unicode_content(self, store, repo_root):
        (repo_root / "uni.py").write_text("print('한글')")
        cid = store.create("uni")
        assert stored_bytes(store, cid, "uni.py") == "print('한글')".encode()

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

    def test_text_file_stored_verbatim(self, store, repo_root):
        """Plain UTF-8 text is stored verbatim, with no encoding wrapper."""
        cid = store.create("text")
        raw = stored_bytes(store, cid, "file_a.py")
        assert not raw.startswith(b"__asr_binary_b64__:")
        assert raw == b"x = 1"


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
        assert stored_paths(store, cid) == {"file_a.py"}
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


# ── Blob storage: per-write cost must not grow with what is already captured ──

class TestBlobStorage:
    """extend() must not rewrite captured CONTENT on every write.

    The index used to carry every captured file's full text, and extend()
    rewrites the index on every write of a run — so the per-write cost grew
    with everything captured so far. Measured on ~24 KB source files before the
    split: 10 writes 134 ms, 20 writes 113 ms, 40 writes 341 ms, 80 writes
    1406 ms, i.e. 5.7 -> 17.6 ms per write and a 2.2 MB JSON. After: a flat
    ~1.1 ms per write and an 8 KB index.
    """

    def test_index_size_is_independent_of_file_size(self, tmp_path):
        """The structural property behind the speedup, asserted without a clock.

        Two runs capturing the SAME number of files, whose contents differ by
        100x in size, must produce near-identical indexes — that can only hold
        if the index stores paths and hashes rather than content.
        """
        sizes = {}
        for label, line_count in (("small", 10), ("large", 1000)):
            root = tmp_path / label
            root.mkdir()
            paths = []
            for i in range(12):
                p = root / f"m{i}.py"
                p.write_text("".join(f"v{i}_{j} = {j}\n" for j in range(line_count)))
                paths.append(str(p))
            s = CheckpointStore(str(root), str(tmp_path / f"store_{label}"))
            cid = s.create("scoped", files=[paths[0]])
            for p in paths[1:]:
                s.extend(cid, [p])
            sizes[label] = (s.checkpoint_dir / f"{cid}.json").stat().st_size
            assert len(stored_paths(s, cid)) == 12

        assert sizes["large"] < sizes["small"] * 1.2, (
            f"index grew with content: {sizes} — extend() is storing file text "
            f"in the index again, which makes each write O(everything captured)"
        )

    def test_identical_content_is_stored_once(self, tmp_path):
        """Blobs are content-addressed, so duplicate content costs one copy."""
        root = tmp_path / "repo"
        root.mkdir()
        paths = []
        for i in range(5):
            p = root / f"dup{i}.py"
            p.write_text("same content everywhere\n")
            paths.append(str(p))
        s = CheckpointStore(str(root), str(tmp_path / "store"))
        cid = s.create("scoped", files=paths)

        blobs = list((s.checkpoint_dir / f"{cid}.files").iterdir())
        assert len(blobs) == 1, f"expected one shared blob, got {len(blobs)}"
        assert len(stored_paths(s, cid)) == 5
        for p in paths:
            Path(p).write_text("clobbered\n")
        assert s.restore(cid) is True
        for p in paths:
            assert Path(p).read_text() == "same content everywhere\n"

    def test_eviction_removes_blobs(self, repo_root, store_dir):
        """Retention must bound BYTES, not just index entries.

        Blobs are the bulk of a checkpoint; leaving them behind would cap the
        number of checkpoints while the store grew without limit.
        """
        import time as _t

        s = CheckpointStore(str(repo_root), str(store_dir), max_checkpoints=2)
        ids = []
        for i in range(4):
            ids.append(s.create(f"cp{i}"))
            _t.sleep(0.005)
        for evicted in ids[:2]:
            assert not (s.checkpoint_dir / f"{evicted}.json").exists()
            assert not (s.checkpoint_dir / f"{evicted}.files").exists(), (
                "evicted checkpoint left its blobs behind"
            )
        for kept in ids[2:]:
            assert (s.checkpoint_dir / f"{kept}.files").is_dir()

    def test_delete_removes_blobs(self, store):
        cid = store.create("doomed")
        assert (store.checkpoint_dir / f"{cid}.files").is_dir()
        assert store.delete(cid) is True
        assert not (store.checkpoint_dir / f"{cid}.files").exists()

    def test_legacy_inline_checkpoint_still_restores(self, store, repo_root):
        """Checkpoints written before blob storage must keep working.

        Users upgrade with a populated .asicode/checkpoints/; a format change
        that silently stopped restoring them would take away the Undo they
        already had.
        """
        cid = store.create("legacy", files=[str(repo_root / "file_a.py")])
        cp_path = store.checkpoint_dir / f"{cid}.json"
        data = json.loads(cp_path.read_text())
        # Rewrite it in the OLD shape: content inline, no blobs at all.
        data["files"] = {"file_a.py": "x = 1"}
        data.pop("storage", None)
        cp_path.write_text(json.dumps(data))
        shutil.rmtree(store.checkpoint_dir / f"{cid}.files")

        (repo_root / "file_a.py").write_text("clobbered")
        assert store.restore(cid) is True
        assert (repo_root / "file_a.py").read_text() == "x = 1"

    def test_legacy_checkpoint_extended_after_upgrade_restores_both(self, store, repo_root):
        """A checkpoint can hold inline AND blob entries at once.

        A run in flight across an upgrade produces exactly this, so the storage
        choice is made per path rather than per checkpoint.
        """
        cid = store.create("legacy", files=[str(repo_root / "file_a.py")])
        cp_path = store.checkpoint_dir / f"{cid}.json"
        data = json.loads(cp_path.read_text())
        data["files"] = {"file_a.py": "x = 1"}
        data.pop("storage", None)
        cp_path.write_text(json.dumps(data))
        shutil.rmtree(store.checkpoint_dir / f"{cid}.files")

        assert store.extend(cid, [str(repo_root / "file_b.txt")]) == 1
        (repo_root / "file_a.py").write_text("clobbered a")
        (repo_root / "file_b.txt").write_text("clobbered b")
        assert store.restore(cid) is True
        assert (repo_root / "file_a.py").read_text() == "x = 1"
        assert (repo_root / "file_b.txt").read_text() == "hello"

    def test_unreadable_file_is_dropped_not_stored_empty(self, store, repo_root):
        """A file that cannot be read must not be recorded as empty content.

        The inline format stored '' for it, and restore() wrote that back —
        truncating the very file the checkpoint existed to protect.
        """
        f = repo_root / "locked.py"
        f.write_text("precious\n")
        cid = store.create("scoped", files=[str(f)])
        # Simulate the unreadable case by removing the blob after capture.
        with patch.object(CheckpointStore, "_copy_file_to_blob", side_effect=OSError("nope")):
            cid2 = store.create("scoped2", files=[str(f)])
        assert "locked.py" not in stored_paths(store, cid2)

        f.write_text("changed\n")
        assert store.restore(cid2) is True
        assert f.read_text() == "changed\n", "restore must not truncate what it never captured"
        assert store.restore(cid) is True
        assert f.read_text() == "precious\n"


# ── Index recovery: a corrupt index must not silently lose the timeline ──────

class TestIndexRecovery:
    """_load_checkpoints treats an unreadable index as a REBUILD trigger, not
    as an authoritative empty store. Every checkpoint's payload file survives
    independently, so the timeline is re-derived from those files instead of
    being silently reset to [] (which made a corrupt index look identical to a
    fresh store and let the next save confirm the loss)."""

    def test_corrupt_index_rebuilt_from_payload_files(self, repo_root, store_dir):
        s1 = CheckpointStore(str(repo_root), str(store_dir))
        cid1 = s1.create("alpha")
        cid2 = s1.create("beta")
        # The historical signature: a truncated/0-byte index (pre-fsync era).
        s1.checkpoint_file.write_text("not json")
        s2 = CheckpointStore(str(repo_root), str(store_dir))
        assert [cp["id"] for cp in s2.checkpoints] == [cid2, cid1]
        by_id = {cp["id"]: cp for cp in s2.checkpoints}
        assert by_id[cid2]["description"] == "beta"
        assert by_id[cid2]["file_count"] >= 2

    def test_zero_byte_index_rebuilt(self, repo_root, store_dir):
        s1 = CheckpointStore(str(repo_root), str(store_dir))
        cid = s1.create("zero")
        s1.checkpoint_file.write_text("")  # 0-byte index: the old crash signature
        s2 = CheckpointStore(str(repo_root), str(store_dir))
        assert [cp["id"] for cp in s2.checkpoints] == [cid]
        # The recovered entry is fully functional, not just listed.
        (repo_root / "file_a.py").write_text("clobbered")
        assert s2.restore(cid) is True
        assert (repo_root / "file_a.py").read_text() == "x = 1"

    def test_wrong_shape_index_rebuilt(self, repo_root, store_dir):
        """Valid JSON that is not a list is corruption too — a dict would
        otherwise blow up on the first merge/eviction."""
        s1 = CheckpointStore(str(repo_root), str(store_dir))
        cid = s1.create("shape")
        s1.checkpoint_file.write_text('{"not": "a list"}')
        s2 = CheckpointStore(str(repo_root), str(store_dir))
        assert [cp["id"] for cp in s2.checkpoints] == [cid]

    def test_rebuilt_index_healed_on_next_save(self, repo_root, store_dir):
        s1 = CheckpointStore(str(repo_root), str(store_dir))
        cid = s1.create("heal")
        s1.checkpoint_file.write_text("garbage")
        s2 = CheckpointStore(str(repo_root), str(store_dir))
        assert [cp["id"] for cp in s2.checkpoints] == [cid]
        cid3 = s2.create("fresh")  # any save rewrites the index
        disk = json.loads(s2.checkpoint_file.read_text())
        assert {cp["id"] for cp in disk} == {cid, cid3}

    def test_missing_index_with_payloads_recovered(self, repo_root, store_dir):
        s1 = CheckpointStore(str(repo_root), str(store_dir))
        cid = s1.create("orphan")
        s1.checkpoint_file.unlink()  # index gone, payload survives
        s2 = CheckpointStore(str(repo_root), str(store_dir))
        assert [cp["id"] for cp in s2.checkpoints] == [cid]


# ── Retention cap holds across the save-merge ────────────────────────────────

class TestMergeRetentionCap:
    """The save-merge resurrects concurrent additions; the max_checkpoints
    cap must be re-applied AFTER the merge, not only before it (a concurrent
    writer's entries could otherwise grow the index past the cap forever)."""

    def test_merge_reapplies_retention_cap(self, repo_root, store_dir):
        s = CheckpointStore(str(repo_root), str(store_dir), max_checkpoints=2)
        s.create("A1")
        s.create("A2")
        # Process B commits two NEWER entries directly to disk, with payload
        # files present so the merge resurrects them.
        from external_llm.common.file_lock import cross_process_flock
        lock_path = s.checkpoint_file.with_suffix('.json.lock')
        with cross_process_flock(lock_path):
            disk = json.loads(s.checkpoint_file.read_text())
            for i in range(2):
                cid_b = f"checkpoint_concurrent_b{i}"
                (s.checkpoint_dir / f"{cid_b}.json").write_text(f'{{"id": "{cid_b}"}}')
                disk.append({
                    'id': cid_b, 'timestamp': 9999999999.0 + i, 'description': f'B{i}',
                    'file_count': 1, 'path': f'{cid_b}.json',
                })
            s.checkpoint_file.write_text(json.dumps(disk))
        s.create("A3")  # triggers merge + save
        ids = {cp["id"] for cp in s.checkpoints}
        assert len(ids) == 2, f"retention cap violated after merge: {ids}"
        # The two newest (B's, timestamp 9.99e9) survive; A's were evicted.
        assert ids == {"checkpoint_concurrent_b0", "checkpoint_concurrent_b1"}


# ── Index path/id hardening (tampered index must not escape the store) ───────

class TestIndexPathGuard:
    """Index entries are read from disk; a tampered 'path' must not turn into
    a file operation outside checkpoint_dir — restore() reads it and
    delete()/eviction unlink it."""

    def test_delete_refuses_tampered_path(self, repo_root, store_dir, tmp_path):
        s = CheckpointStore(str(repo_root), str(store_dir))
        cid = s.create("guard")
        victim = tmp_path / "victim.json"
        victim.write_text("precious")
        s.checkpoints[0]["path"] = "../../victim.json"
        assert s.delete(cid) is False
        assert victim.read_text() == "precious"
        assert any(cp["id"] == cid for cp in s.checkpoints), \
            "refused delete must not drop the entry"

    def test_restore_refuses_tampered_path(self, repo_root, store_dir, tmp_path):
        s = CheckpointStore(str(repo_root), str(store_dir))
        cid = s.create("guard")
        s.checkpoints[0]["path"] = "../../victim.json"
        assert s.restore(cid) is False

    def test_extend_refuses_tampered_path(self, repo_root, store_dir, tmp_path):
        s = CheckpointStore(str(repo_root), str(store_dir))
        cid = s.create("guard", files=["file_a.py"])
        s.checkpoints[0]["path"] = "../../victim.json"
        assert s.extend(cid, ["file_b.txt"]) == 0

    def test_evict_skips_tampered_path_unlink(self, repo_root, store_dir, tmp_path):
        victim = tmp_path / "victim.json"
        victim.write_text("precious")
        s = CheckpointStore(str(repo_root), str(store_dir), max_checkpoints=1)
        cid = s.create("first")
        s.checkpoints[0]["path"] = "../../victim.json"
        s.create("second")  # evicts the tampered entry
        assert victim.read_text() == "precious"
        assert all(cp["id"] != cid for cp in s.checkpoints), \
            "entry still evicted from the index"

    def test_delete_refuses_suspicious_id_blob_removal(self, repo_root, store_dir, tmp_path):
        """A tampered 'id' must not turn _remove_blobs into a path outside the
        store (checkpoint_dir / '<id>.files')."""
        s = CheckpointStore(str(repo_root), str(store_dir))
        s.create("guard")
        s.checkpoints[0]["id"] = "../../evil"
        victim_dir = tmp_path / "evil.files"
        victim_dir.mkdir()
        (victim_dir / "x.bin").write_bytes(b"x")
        assert s.delete("../../evil") is True  # own payload removed, index updated
        assert (victim_dir / "x.bin").exists(), "blob removal must not escape the store"


# ── Blob copy streams (bounded memory for huge artifacts) ────────────────────

class TestStreamingBlobCopy:
    """_copy_file_to_blob streams in 64 KiB chunks so a captured file costs
    bounded memory no matter how large it is — same rationale as the chunked
    _sha256_file."""

    def test_large_file_streamed_to_blob_and_restored(self, store, repo_root):
        payload = os.urandom(65536 * 4 + 123)  # >4 chunks, non-aligned tail
        (repo_root / "big.bin").write_bytes(payload)
        cid = store.create("big", files=["big.bin"])
        assert stored_bytes(store, cid, "big.bin") == payload
        (repo_root / "big.bin").write_bytes(b"clobbered")
        assert store.restore(cid) is True
        assert (repo_root / "big.bin").read_bytes() == payload


# ── P27-4: byte-cap retention + per-file blob gate ─────────────────────────

class TestRetentionBytes:
    def test_byte_cap_evicts_oldest(self, repo_root, store_dir):
        """A 5 KB file per checkpoint blows a 10 KB store cap at 2 checkpoints —
        the OLDEST are evicted until the store fits (count cap disabled)."""
        (repo_root / "data.txt").write_text("y" * 5000)
        store = CheckpointStore(str(repo_root), str(store_dir),
                                max_checkpoints=0, max_bytes=10_000)
        store.create("one")
        store.create("two")
        store.create("three")
        assert [c["description"] for c in store.list()] == ["three"]
        assert store._store_bytes() <= 10_000

    def test_byte_cap_zero_disables(self, repo_root, store_dir):
        """max_bytes=0 → only the count cap applies (backward compatible)."""
        store = CheckpointStore(str(repo_root), str(store_dir),
                                max_checkpoints=2, max_bytes=0)
        store.create("one")
        store.create("two")
        store.create("three")
        assert [c["description"] for c in store.list()] == ["three", "two"]

    def test_blob_gate_skips_oversized_file(self, repo_root, store_dir):
        """max_blob_bytes: the giant file is hashed but NOT stored — restore()
        leaves it untouched instead of writing wrong bytes."""
        big = repo_root / "big.bin"
        big.write_bytes(b"x" * 5000)
        store = CheckpointStore(str(repo_root), str(store_dir), max_blob_bytes=1000)
        cid = store.create("with-big")
        data = json.loads((store.checkpoint_dir / f"{cid}.json").read_text())
        assert "big.bin" in data["skipped_files"]
        assert "big.bin" not in data["file_hashes"]
        assert data["file_count"] == len(data["file_hashes"])

        big.write_text("changed\n")
        assert store.restore(cid) is True
        assert big.read_text() == "changed\n"  # untouched — not restorable

    def test_blob_gate_zero_disables(self, repo_root, store_dir):
        """max_blob_bytes=0 (default) captures everything as before."""
        (repo_root / "big.bin").write_bytes(b"x" * 5000)
        store = CheckpointStore(str(repo_root), str(store_dir))
        cid = store.create("with-big")
        data = json.loads((store.checkpoint_dir / f"{cid}.json").read_text())
        assert "big.bin" not in data.get("skipped_files", [])
        assert "big.bin" in data["file_hashes"]
