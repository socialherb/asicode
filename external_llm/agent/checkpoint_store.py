import base64
import hashlib
import json
import logging
import os
import time
import uuid
from pathlib import Path

from external_llm.common.atomic_io import atomic_write_json
from external_llm.common.file_lock import cross_process_flock

logger = logging.getLogger(__name__)

# Sentinel prefix marking a base64-encoded binary file in checkpoint['files'].
# restore() checks for this to decide whether to decode bytes (binary) or
# write the stored string as UTF-8 text. Chosen to be extremely unlikely to
# appear at the start of a legitimate source file.
_BINARY_SENTINEL = "__asr_binary_b64__:"

# Marker in a checkpoint index saying its file CONTENTS live in a sibling blob
# directory rather than inline under 'files'. Checkpoints written before blob
# storage carry no 'storage' key and are still readable — see restore().
_STORAGE_BLOBS = "blobs"

# P27-4: default cap on the store's TOTAL on-disk bytes (index + payloads +
# blobs). The count cap (max_checkpoints) says nothing about how big a
# checkpoint is: 50 checkpoints of a repo carrying a multi-hundred-MB
# artifact grow without bound. 0 disables the byte dimension (then the store
# is bounded only by the count cap).
_DEFAULT_MAX_STORE_BYTES = 256 * 1024 * 1024


class CheckpointStore:
    """
    Manages file checkpoints with timeline UI support.
    """

    def __init__(self, repo_root: str, store_dir: str = '.asicode/checkpoints',
                 max_checkpoints: int = 50,
                 max_bytes: int = _DEFAULT_MAX_STORE_BYTES,
                 max_blob_bytes: int = 0):
        """
        Initialize checkpoint store.

        Args:
            repo_root: Root directory of the repository to track.
            store_dir: Directory where checkpoints will be stored.
                Relative paths are resolved against *repo_root*, not CWD
                (Bug #4 fix: CWD-relative resolution silently lost checkpoints
                when the server started from a different directory and caused
                cross-repo key collision via basename-only repo identification).
            max_checkpoints: Maximum number of checkpoints to retain. When
                exceeded, the oldest are evicted automatically on create().
                Set to 0 to disable eviction (unbounded retention).
            max_bytes: Maximum TOTAL on-disk bytes for the store (index +
                payload files + blobs). When exceeded, the oldest checkpoints
                are evicted until the store fits, on every save. Set to 0 to
                disable the byte dimension.
            max_blob_bytes: Per-file cap for captured CONTENT. A file larger
                than this is hashed (streamed) but its blob is NOT written —
                the checkpoint records it in ``skipped_files`` and restore()
                leaves it untouched, so a single giant tracked artifact cannot
                force the store over *max_bytes* in every checkpoint at once.
                Set to 0 to disable (capture everything).
        """
        self.repo_root = Path(repo_root).resolve()
        store_path = Path(store_dir)
        if not store_path.is_absolute():
            store_path = self.repo_root / store_dir
        self.store_dir = store_path.resolve()
        self.max_checkpoints = max_checkpoints
        self.max_bytes = max_bytes
        self.max_blob_bytes = max_blob_bytes

        # Create store directory if it doesn't exist
        self.store_dir.mkdir(parents=True, exist_ok=True)

        # Determine repository name for subdirectory
        repo_name = self.repo_root.name
        self.checkpoint_dir = self.store_dir / repo_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.checkpoint_file = self.checkpoint_dir / 'checkpoints.json'
        self._load_checkpoints()

    def _load_checkpoints(self) -> None:
        """Load checkpoints from JSON file.

        An unreadable index (0-byte file, truncated JSON, wrong shape) is NOT
        treated as an authoritative empty store: each checkpoint's payload
        file survives independently in checkpoint_dir, so the timeline is
        rebuilt from those files instead of being silently reset to [] (which
        made a corrupt index look identical to a fresh store and let the next
        save confirm the loss). The rebuilt index is persisted on the next
        save, healing the file.
        """
        if not self.checkpoint_file.exists():
            self.checkpoints = self._rebuild_index_from_disk()
            if self.checkpoints:
                logger.warning(
                    "checkpoint index %s missing; recovered %d checkpoint(s) "
                    "from payload files",
                    self.checkpoint_file, len(self.checkpoints),
                )
            return
        try:
            with open(self.checkpoint_file, encoding='utf-8') as f:
                self.checkpoints = json.load(f)
            if not isinstance(self.checkpoints, list):
                raise TypeError("index is not a JSON list")
        except (OSError, json.JSONDecodeError, ValueError, TypeError) as e:
            logger.exception("checkpoint index %s unreadable (%s)", self.checkpoint_file, e)
            self.checkpoints = self._rebuild_index_from_disk()
            if self.checkpoints:
                logger.warning(
                    "recovered %d checkpoint(s) from payload files; "
                    "next save will heal the index",
                    len(self.checkpoints),
                )

    def _rebuild_index_from_disk(self) -> list[dict]:
        """Reconstruct the index from the per-checkpoint payload files.

        The index is a derived listing: each checkpoint's authoritative data
        (id, timestamp, description, scope, file_count, file_hashes, absent)
        lives in its own ``checkpoint_<id>.json`` file, so a lost or corrupt
        index can always be re-derived. Payloads that cannot be read are
        skipped (they are gone either way). Sorted newest-first like a freshly
        saved index.
        """
        rebuilt: list[dict] = []
        for cp_file in sorted(self.checkpoint_dir.glob('checkpoint_*.json')):
            try:
                with open(cp_file, encoding='utf-8') as f:
                    data = json.load(f)
            except (OSError, json.JSONDecodeError) as e:
                logger.debug("rebuild: cannot read payload %s: %s", cp_file, e)
                continue
            cid = data.get('id')
            if not cid:
                continue
            rebuilt.append({
                'id': cid,
                'timestamp': data.get('timestamp', 0),
                'description': data.get('description', ''),
                'scope': data.get('scope', 'full'),
                'file_count': data.get('file_count', 0),
                'path': cp_file.name,
            })
        rebuilt.sort(key=lambda x: x.get('timestamp', 0), reverse=True)
        return rebuilt

    def _index_payload_path(self, entry: dict) -> Path | None:
        """Payload path for an index *entry*, verified inside checkpoint_dir.

        Index entries are read from disk like checkpoint payloads, so a
        tampered 'path' (``../``-style) must not turn into a file operation
        anywhere — restore() reads it, delete()/eviction unlink it. Returns
        None when the entry has no path or the path would resolve outside the
        checkpoint dir.
        """
        rel = entry.get('path')
        if not rel:
            return None
        p = Path(rel)
        if p.is_absolute():
            return None
        resolved = (self.checkpoint_dir / p).resolve()
        try:
            resolved.relative_to(self.checkpoint_dir)
        except ValueError:
            logger.debug("index entry path escapes checkpoint dir: %r", rel)
            return None
        return resolved

    def _save_checkpoints(self) -> None:
        """Save checkpoints to JSON file (atomic + durable write).

        Uses ``atomic_write_json`` (sibling tmp file → ``fsync`` →
        ``os.replace``), so the index is durable on disk before the rename.
        This prevents a truncated/partial checkpoints.json from being left
        behind if the process is interrupted mid-write (e.g. disk full,
        SIGKILL, power loss), which would otherwise cause
        _load_checkpoints() to silently reset self.checkpoints to [] and lose
        the whole checkpoint index — even though each payload was fsync'd.
        Mirrors the atomic-write pattern in external_llm/common/atomic_io.py.

        Concurrency: acquires an exclusive ``fcntl`` flock on the metadata
        file for the entire read-modify-write window so that two processes
        checkpointing concurrently cannot lose each other's entries (the
        last-writer would otherwise clobber the other's append). Before
        writing, we re-load the on-disk index and merge any entries added by
        a concurrent process since our last load. On non-POSIX platforms
        without ``fcntl`` (e.g. Windows), the lock is a no-op but the
        atomic-rename + merge still mitigates most races.
        """
        lock_path = self.checkpoint_file.with_suffix('.json.lock')
        with cross_process_flock(lock_path):
            # Re-load under the lock and merge concurrent additions.
            try:
                disk = []
                if self.checkpoint_file.exists():
                    with open(self.checkpoint_file, encoding='utf-8') as f:
                        disk = json.load(f)
            except (OSError, json.JSONDecodeError) as e:
                # Corrupt disk index: the in-memory state is authoritative and
                # the write below heals the file — but this is NOT an empty
                # disk, so say so (an unreadable index must not look like a
                # legitimately empty one).
                logger.warning("on-disk index unreadable (%s); keeping in-memory entries", e)
                disk = []
            if not isinstance(disk, list):
                logger.warning(
                    "on-disk index has unexpected shape %s; keeping in-memory entries",
                    type(disk).__name__,
                )
                disk = []
            known = {cp['id'] for cp in self.checkpoints}
            merged = list(self.checkpoints)
            for cp in disk:
                cid = cp.get('id')
                if cid in known:
                    continue
                # Only resurrect a disk entry if its checkpoint file still
                # exists. delete()/eviction remove the .json file before the
                # index is rewritten, so this prevents a deleted/evicted id
                # from being misclassified as a concurrent addition and
                # resurrected by the merge. The path is verified to stay
                # inside checkpoint_dir like every other index-path use.
                cp_file = self._index_payload_path(cp)
                if cp_file is not None and cp_file.exists():
                    merged.append(cp)
            merged.sort(key=lambda x: x.get('timestamp', 0), reverse=True)
            self.checkpoints = merged

            # The merge can grow the list past the retention caps again (a
            # concurrent process's entries resurrected), so re-apply them now
            # that the merged set is known — under the same lock, so eviction
            # order is consistent with the write below.
            self._enforce_retention()

            # atomic_write_json = tmp + fsync + os.replace, so the index is
            # durable on disk before the rename. The previous hand-rolled form
            # (tmp + os.replace) forgot the fsync: a crash/power-loss could
            # leave a 0-byte index, and _load_checkpoints() would then silently
            # reset self.checkpoints to [] — losing the whole index even though
            # each payload was fsync'd (line 288). atomic_write_json is already
            # imported above and used for the payloads.
            try:
                atomic_write_json(
                    self.checkpoint_file, self.checkpoints,
                    indent=2, ensure_ascii=False,
                )
            except OSError as e:
                logger.exception("Failed to save checkpoints: %s", e)
                raise

    def _scan_files(self, files=None) -> dict[Path, str]:
        """
        Compute SHA256 of tracked files.

        Args:
            files: Optional iterable of paths (absolute, or relative to
                *repo_root*) to snapshot instead of walking the whole repo.
                Paths outside repo_root, non-existent paths, and non-regular
                files are skipped (a plan may target a file it is about to
                create — nothing to snapshot yet).

        Returns:
            Dictionary mapping repo-relative file paths to their SHA256 hashes.
        """
        if files is not None:
            return self._scan_listed_files(files)
        file_hashes = {}
        exclude_dirs = {'.git', '.asicode', '__pycache__', '.pytest_cache', '.mypy_cache',
                        'node_modules', '.venv', 'venv', 'env', 'dist', 'build', '.eggs', '.tox'}
        exclude_extensions = {'.pyc', '.pyo', '.pyd', '.so', '.dll', '.exe'}

        # Use os.walk with directory pruning instead of rglob('*') to avoid
        # descending into excluded subtrees (node_modules, .git, etc.).
        # rglob visits every entry then filters, which is 70x+ slower on
        # repos with large vendor trees.
        for root, dirs, fnames in os.walk(self.repo_root):
            # Prune excluded directories in-place so os.walk skips their subtrees.
            # Dot-dirs are pruned wholesale: caches (.ruff_cache, .mypy_cache, …)
            # and tool dirs (.git, .asicode, .tenet, .claude) are never edit
            # targets and hashing them dominated scan cost (a single .ruff_cache
            # held 26k files = 95% of the SHA256 time). Aligns with
            # common.walk_policy._walk_should_skip_dir's dot-dir heuristic.
            dirs[:] = sorted(
                d for d in dirs if d not in exclude_dirs and not d.startswith('.')
            )
            for fname in sorted(fnames):
                file_path = Path(root) / fname

                # Skip excluded file extensions
                if file_path.suffix.lower() in exclude_extensions:
                    continue

                # Skip the checkpoint store directory itself
                if file_path.is_relative_to(self.checkpoint_dir):
                    continue

                try:
                    file_hash = self._sha256_file(file_path)
                    relative_path = file_path.relative_to(self.repo_root)
                    file_hashes[relative_path] = file_hash
                except OSError as e:
                    logger.warning("Could not read file %s: %s", file_path, e)
                    continue

        return file_hashes

    def _resolve_repo_relative(self, entry) -> tuple[Path, Path] | None:
        """Resolve *entry* to an absolute path under repo_root.

        Returns ``(resolved_abs_path, repo_relative_path)``, or ``None`` when
        *entry* is empty or escapes repo_root (warned and skipped). This is the
        shared resolution block for :meth:`_scan_listed_files`,
        :meth:`_scan_absent_files` and :meth:`_relativize`, which each used to
        duplicate it inline.
        """
        if not entry:
            return None
        p = Path(entry)
        if not p.is_absolute():
            p = self.repo_root / p
        try:
            resolved = p.resolve()
            return resolved, resolved.relative_to(self.repo_root)
        except (ValueError, OSError):
            logger.warning("Skipping checkpoint path outside repo root: %r", entry)
            return None

    def _scan_listed_files(self, files) -> dict[Path, str]:
        """Hash only the given paths (scoped snapshot; see :meth:`_scan_files`)."""
        file_hashes: dict[Path, str] = {}
        for entry in files:
            hit = self._resolve_repo_relative(entry)
            if hit is None:
                continue
            p, relative = hit
            if not p.is_file():
                continue  # to-be-created target or directory — nothing to snapshot
            try:
                file_hashes[relative] = self._sha256_file(p)
            except OSError as e:
                logger.warning("Could not read file %s: %s", p, e)
        return file_hashes

    def _relativize(self, entry) -> str:
        """Repo-relative form of *entry*, or '' if it escapes repo_root.

        Tombstones are keyed repo-relative like every other record, and
        restore() unlinks them — so a path that cannot be expressed relative to
        the root must be dropped here rather than stored and rejected later.
        """
        hit = self._resolve_repo_relative(entry)
        if hit is None:
            return ''
        return str(hit[1])

    def _scan_absent_files(self, files) -> list[str]:
        """Repo-relative paths from *files* that do not currently exist.

        The complement of :meth:`_scan_listed_files`, and the reason a scoped
        checkpoint can undo a file the run CREATED. Content alone cannot express
        "this file did not exist": restore() only writes what it stored, so a
        created file survived every undo. Worse under the accumulating gate — a
        file created by one write and edited by the next exists by the time the
        second write is gated, so it was captured at its half-written content
        and restore() left the tree in a state the run never actually passed
        through, reporting success.

        Directories are excluded, not just existing files: a path that names a
        directory is not a file this run created, and unlinking it on restore
        would be destructive far beyond an undo.
        """
        absent: list[str] = []
        for entry in files:
            hit = self._resolve_repo_relative(entry)
            if hit is None:
                continue
            p, relative = hit
            if p.exists():
                continue
            absent.append(str(relative))
        return absent

    def _blob_dir(self, checkpoint_id: str) -> Path:
        """Directory holding one checkpoint's file contents, one blob per file.

        Contents live here rather than inline in the index because ``extend()``
        rewrites the index on every write of the run, and the index used to
        carry every captured file's full text. That made the per-write cost
        proportional to everything captured SO FAR — quadratic over a run.
        Measured before the split, on ~24 KB source files:

            10 writes ->   134 ms (13.4 ms/write),  275 KB
            20 writes ->   113 ms ( 5.7 ms/write),  549 KB
            40 writes ->   341 ms ( 8.5 ms/write), 1098 KB
            80 writes ->  1406 ms (17.6 ms/write), 2196 KB

        Blobs are named by the SHA256 the index already stores, so identical
        content is written once no matter how many paths hold it, and a rewrite
        of the index moves only paths and hashes.
        """
        if (
            "/" in checkpoint_id or "\\" in checkpoint_id
            or checkpoint_id in (".", "..")
        ):
            raise ValueError(f"suspicious checkpoint id {checkpoint_id!r}")
        return self.checkpoint_dir / f"{checkpoint_id}.files"

    def _copy_file_to_blob(self, checkpoint_id: str, file_hash: str, file_path: Path) -> bool:
        """Stream *file_path* into the blob named by its content hash (idempotent).

        Chunked copy so a captured file costs bounded memory no matter how
        large it is — the same rationale as _sha256_file: a multi-hundred-MB
        artifact (weights, datasets) must not be slurped into RAM just to be
        checkpointed. Raw bytes: a blob is opaque, so text and binary are the
        same case and restore() writes back the exact bytes.

        Write-then-rename so a crash mid-write cannot leave a truncated blob
        sitting at the name of a hash the index already trusts.

        Returns False (blob skipped; the caller must drop the hash from the
        checkpoint and record the path in ``skipped_files``) when the file
        exceeds ``max_blob_bytes``. The store byte cap alone cannot protect a
        store whose EVERY checkpoint holds the same giant file — eviction
        would have to delete every checkpoint to get back under the cap. A
        skipped file's pre-write state is not restorable: restore() leaves it
        untouched rather than write wrong bytes.
        """
        if self.max_blob_bytes > 0:
            size = file_path.stat().st_size
            if size > self.max_blob_bytes:
                logger.warning(
                    "Skipping blob for %s (%d bytes > %d cap) — restore() will "
                    "leave this file untouched",
                    file_path, size, self.max_blob_bytes,
                )
                return False
        blob_dir = self._blob_dir(checkpoint_id)
        blob_dir.mkdir(parents=True, exist_ok=True)
        blob_path = blob_dir / file_hash
        if blob_path.exists():
            return True  # content-addressed: same hash means same bytes
        tmp_path = blob_path.with_suffix('.tmp')
        with open(tmp_path, "wb") as out, open(file_path, "rb") as src:
            for chunk in iter(lambda: src.read(65536), b""):
                out.write(chunk)
        os.replace(tmp_path, blob_path)
        return True

    def _read_blob(self, checkpoint_id: str, file_hash: str) -> bytes:
        """Return the bytes stored under *file_hash* (raises OSError if absent)."""
        with open(self._blob_dir(checkpoint_id) / file_hash, 'rb') as f:
            return f.read()

    def _remove_blobs(self, checkpoint_id: str) -> None:
        """Delete a checkpoint's blob directory, if any."""
        if (
            "/" in checkpoint_id or "\\" in checkpoint_id
            or checkpoint_id in (".", "..")
        ):
            logger.error("Refusing to remove blob dir for suspicious id %r", checkpoint_id)
            return
        blob_dir = self._blob_dir(checkpoint_id)
        if not blob_dir.is_dir():
            return
        for blob in blob_dir.iterdir():
            try:
                blob.unlink()
            except OSError as e:
                logger.warning("Could not remove blob %s: %s", blob, e)
        try:
            blob_dir.rmdir()
        except OSError as e:
            logger.warning("Could not remove blob dir %s: %s", blob_dir, e)

    @staticmethod
    def _sha256_file(file_path: Path, chunk_size: int = 65536) -> str:
        """Stream a file through SHA256 in fixed-size chunks.

        Reading the whole file into memory (``f.read()``) risks OOM on large
        repos with multi-hundred-MB artifacts (model weights, datasets, build
        outputs). Chunked hashing produces an identical digest at bounded
        memory cost. Python 3.11+ has ``hashlib.file_digest`` but chunking is
        portable across all supported versions.
        """
        h = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(chunk_size), b""):
                h.update(chunk)
        return h.hexdigest()

    def create(self, description: str = '', files=None, absent=None) -> str:
        """
        Create a timestamped checkpoint with SHA256 + contents of tracked files.

        Args:
            description: Optional description for the checkpoint.
            files: Optional iterable of paths to snapshot instead of the whole
                repo (scoped checkpoint). Cheap enough to run on every write
                turn — the full-repo walk reads/stores every source file.
                restore() only writes files present in the checkpoint, so a
                scoped restore never touches unrelated files. Listed paths that
                do not exist yet are recorded as ABSENT tombstones, so a file
                the run goes on to create is deleted by restore() rather than
                surviving it.

        Returns:
            Checkpoint ID.
        """
        checkpoint_id = f"checkpoint_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        timestamp = time.time()

        # Scan files and compute hashes
        file_hashes = self._scan_files(files)
        # Tombstones are SCOPED-only. A full-repo snapshot has no listed
        # targets, and every path the run later creates is by definition absent
        # from it — "delete everything not in the snapshot" would reach far
        # outside what the run touched.
        # An explicit list wins: confirm_writes() supplies paths it has already
        # established the run CREATED, and by then they exist — so re-deriving
        # absence from the filesystem here would find none of them.
        absent = (self._scan_absent_files(files) if files is not None else []) if absent is None else list(absent)

        # Create checkpoint data
        checkpoint_data = {
            'id': checkpoint_id,
            'timestamp': timestamp,
            'description': description,
            'scope': 'files' if files is not None else 'full',
            'file_count': len(file_hashes),
            'file_hashes': {str(k): v for k, v in file_hashes.items()},
            # Contents live in _blob_dir(checkpoint_id), keyed by the hashes
            # above. 'files' stays as an empty dict so a reader that predates
            # blob storage sees a well-formed (if empty) index rather than a
            # KeyError.
            'storage': _STORAGE_BLOBS,
            'files': {},
            # Repo-relative paths that did not exist when captured; restore()
            # unlinks them. Absent from checkpoints written before tombstones
            # existed, so every reader must default it.
            'absent': absent,
            # Paths whose content was NOT captured because they exceed
            # max_blob_bytes — restore() leaves them untouched (never writes
            # wrong bytes), and file_count excludes them.
            'skipped_files': [],
        }

        # Store file contents as blobs. A file that cannot be read is dropped
        # from file_hashes entirely rather than recorded with empty content:
        # the inline format stored '' for it, which restore() then wrote back —
        # truncating a file it was supposed to be protecting.
        unreadable: list[str] = []
        skipped: list[str] = []
        for relative_path_str, _file_hash in checkpoint_data['file_hashes'].items():
            relative_path = Path(relative_path_str)
            file_path = self.repo_root / relative_path
            try:
                if not self._copy_file_to_blob(checkpoint_id, _file_hash, file_path):
                    skipped.append(relative_path_str)
            except OSError as e:
                logger.warning("Could not read file content for %s: %s", file_path, e)
                unreadable.append(relative_path_str)
        for relative_path_str in unreadable + skipped:
            del checkpoint_data['file_hashes'][relative_path_str]
        checkpoint_data['skipped_files'] = skipped
        checkpoint_data['file_count'] = len(checkpoint_data['file_hashes'])
        file_hashes = {
            k: v for k, v in file_hashes.items()
            if str(k) not in unreadable and str(k) not in skipped
        }

        # Save checkpoint to individual file
        checkpoint_path = self.checkpoint_dir / f"{checkpoint_id}.json"
        try:
            atomic_write_json(checkpoint_path, checkpoint_data, indent=2, ensure_ascii=True)
        except OSError as e:
            logger.exception("Failed to save checkpoint %s: %s", checkpoint_id, e)
            raise

        # Update checkpoints list
        self.checkpoints.append({
            'id': checkpoint_id,
            'timestamp': timestamp,
            'description': description,
            'scope': 'files' if files is not None else 'full',
            'file_count': len(file_hashes),
            'path': str(checkpoint_path.relative_to(self.checkpoint_dir))
        })

        # Sort checkpoints by timestamp (newest first)
        self.checkpoints.sort(key=lambda x: x['timestamp'], reverse=True)

        # Evict oldest checkpoints when retention (count or bytes) is exceeded.
        self._enforce_retention()

        # Save updated checkpoints list
        self._save_checkpoints()

        logger.info("Created checkpoint %s with %s files", checkpoint_id, len(file_hashes))
        return checkpoint_id

    def extend(self, checkpoint_id: str, files, absent=None) -> int:
        """Add not-yet-captured files to an existing scoped checkpoint.

        Files already present keep their ORIGINAL stored content — that is the
        whole point: a run-level checkpoint accumulates each file's *pre-write*
        state the first time the run touches it, so one ``restore()`` rolls the
        entire run back to where it started. Re-capturing on every write would
        instead pin the most recent content and make Undo a no-op.

        Paths that do not exist yet are added as ABSENT tombstones under the
        same rule. First-seen-wins governs the two records jointly, not each in
        isolation: a path already tombstoned must NOT later be captured as
        content (it would resurrect a file the run created), and a path already
        captured as content must NOT later be tombstoned (deleting it would
        overshoot the undo). Both directions are checked against the union.

        A full-scope checkpoint already covers every file, so extending one is a
        no-op. Returns the number of records newly captured — files plus
        tombstones (0 when nothing was added, including when ``checkpoint_id``
        is unknown).
        """
        entry = next((c for c in self.checkpoints if c['id'] == checkpoint_id), None)
        if entry is None:
            logger.warning("extend: unknown checkpoint %s", checkpoint_id)
            return 0
        if entry.get('scope') != 'files':
            return 0  # full-repo snapshot already contains everything

        checkpoint_path = self._index_payload_path(entry)
        if checkpoint_path is None:
            logger.warning("extend: refusing suspicious stored path for %s", checkpoint_id)
            return 0
        try:
            with open(checkpoint_path, encoding='utf-8') as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("extend: cannot read checkpoint %s: %s", checkpoint_id, e)
            return 0

        # `files` may be a one-shot iterable and is consumed three times below.
        files = list(files)
        stored_absent = list(data.get('absent', []))
        # The union, so first-seen-wins holds ACROSS the two record kinds and
        # not merely within each. A path seen absent then written must stay a
        # tombstone; a path captured with content must never become one.
        known = set(data.get('file_hashes', {})) | set(stored_absent)
        fresh = {
            rel: h for rel, h in self._scan_listed_files(files).items()
            if str(rel) not in known
        }
        # As in create(): an explicit list is authoritative, because the caller
        # confirming a creation looks at paths that now exist.
        candidate_absent = (
            self._scan_absent_files(files) if absent is None
            else [self._relativize(p) for p in absent]
        )
        fresh_absent = [
            rel for rel in candidate_absent if rel and rel not in known
        ]
        if not fresh and not fresh_absent:
            return 0

        # A checkpoint created before blob storage keeps its inline contents;
        # only its NEW entries go to blobs. Both are then readable together,
        # which restore() handles per path rather than per checkpoint.
        for relative_path, file_hash in fresh.items():
            rel_str = str(relative_path)
            try:
                if not self._copy_file_to_blob(
                    checkpoint_id, file_hash, self.repo_root / relative_path
                ):
                    # Over max_blob_bytes: record so the skip is diagnosable,
                    # but keep the path OUT of file_hashes — restore() then
                    # never touches it (a hash without a blob would fall to
                    # the legacy inline branch and write empty content).
                    data.setdefault('skipped_files', []).append(rel_str)
                    continue
            except OSError as e:
                logger.warning("extend: could not read %s: %s", relative_path, e)
                continue
            data['file_hashes'][rel_str] = file_hash
            data.setdefault('blob_paths', []).append(rel_str)

        data['absent'] = stored_absent + fresh_absent
        data['file_count'] = len(data['file_hashes'])
        try:
            atomic_write_json(checkpoint_path, data, indent=2, ensure_ascii=True)
        except OSError as e:
            logger.exception("extend: failed to save checkpoint %s: %s", checkpoint_id, e)
            return 0

        # Against the union, so a checkpoint extended only with tombstones
        # still reports the work it did. Counting file_count against `known`
        # (which now includes tombstones) would report a negative number.
        added = (data['file_count'] + len(data['absent'])) - len(known)
        entry['file_count'] = data['file_count']
        self._save_checkpoints()
        logger.debug("Extended checkpoint %s with %d file(s)", checkpoint_id, added)
        return added

    def _store_bytes(self) -> int:
        """Total on-disk bytes of this store (index + payloads + blobs + lock).

        Walks the whole checkpoint dir, so callers should only use it on the
        over-cap path (see ``_enforce_retention``), not per checkpoint write.
        """
        total = 0
        for p in self.checkpoint_dir.rglob("*"):
            try:
                if p.is_file():
                    total += p.stat().st_size
            except OSError as e:
                logger.debug("store byte scan skipped %s: %s", p, e)
                continue
        return total

    def _enforce_retention(self) -> None:
        """Evict oldest checkpoints until the count AND byte caps are satisfied.

        Called under the save lock (from create() and _save_checkpoints()) so
        eviction order is consistent with the index write. Each dimension is
        independently disabled by 0 (``max_checkpoints`` / ``max_bytes``). The
        byte loop re-walks the store after each eviction — at most a few dozen
        walks on the rare over-cap path, none on the common under-cap path.
        """
        if self.max_checkpoints > 0:
            while len(self.checkpoints) > self.max_checkpoints:
                self._evict_oldest()
        if self.max_bytes > 0:
            while self.checkpoints and self._store_bytes() > self.max_bytes:
                self._evict_oldest()

    def _evict_oldest(self) -> None:
        """Remove the single oldest checkpoint (file + index entry).

        List is kept sorted newest-first, so the oldest is the last element.
        """
        if not self.checkpoints:
            return
        oldest = self.checkpoints.pop()
        try:
            cp_path = self._index_payload_path(oldest)
            if cp_path is not None and cp_path.exists():
                cp_path.unlink()
            # Blobs are the bulk of a checkpoint's bytes; leaving them behind
            # would make retention bound the index count while the store grew
            # without limit.
            self._remove_blobs(oldest['id'])
            logger.info("Evicted old checkpoint %s", oldest['id'])
        except (OSError, KeyError) as e:
            logger.warning("Failed to evict checkpoint %s: %s", oldest.get('id', '?'), e)

    def list(self) -> list[dict]:
        """
        Get sorted list of checkpoints.

        Returns:
            List of checkpoints with id, timestamp, description, file_count.
        """
        return [
            {
                'id': cp['id'],
                'timestamp': cp['timestamp'],
                'description': cp['description'],
                'scope': cp.get('scope', 'full'),
                'file_count': cp['file_count']
            }
            for cp in self.checkpoints
        ]

    def restore(self, checkpoint_id: str) -> bool:
        """
        Restore files from checkpoint.
        Args:
            checkpoint_id: ID of checkpoint to restore.
        Returns:
            True if successful, False otherwise.
        """
        # Find checkpoint
        checkpoint_info = None
        for cp in self.checkpoints:
            if cp['id'] == checkpoint_id:
                checkpoint_info = cp
                break
        if not checkpoint_info:
            logger.error("Checkpoint %s not found", checkpoint_id)
            return False
        # Load checkpoint data
        checkpoint_path = self._index_payload_path(checkpoint_info)
        if checkpoint_path is None:
            logger.error("Refusing to load checkpoint %s: invalid stored path", checkpoint_id)
            return False
        if not checkpoint_path.exists():
            logger.error("Checkpoint file %s not found", checkpoint_path)
            return False
        try:
            with open(checkpoint_path, encoding='utf-8') as f:
                checkpoint_data = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            logger.exception("Failed to load checkpoint %s: %s", checkpoint_id, e)
            return False
        # Every path this checkpoint holds content for, from either storage.
        # Blob and inline entries coexist inside one checkpoint: a checkpoint
        # created before blob storage and extended after it has both, so the
        # choice is made per path (see _content_for) rather than per file.
        inline = checkpoint_data.get('files', {})
        blob_paths = set(checkpoint_data.get('blob_paths', []))
        if checkpoint_data.get('storage') == _STORAGE_BLOBS:
            blob_paths |= set(checkpoint_data.get('file_hashes', {}))

        # Restore files
        success = True
        for relative_path_str in list(inline) + sorted(blob_paths - set(inline)):
            # Defense-in-depth: resolve and verify each path stays within
            # repo_root before writing. relative_path_str is read directly from
            # the checkpoint JSON (untrusted) — a tampered checkpoint could
            # otherwise write outside the repo via "../" traversal.
            target_path = (self.repo_root / relative_path_str).resolve()
            try:
                target_path.relative_to(self.repo_root)
            except ValueError:
                logger.exception("Refusing to restore path outside repo root: %r", relative_path_str)
                success = False
                continue
            file_path = target_path

            # Skip if file content hasn't changed (preserves mtime, avoids
            # invalidating build/test caches unnecessarily — Bug #5 fix).
            stored_hash = checkpoint_data.get('file_hashes', {}).get(relative_path_str)
            if stored_hash and file_path.exists():
                try:
                    current_hash = self._sha256_file(file_path)
                    if current_hash == stored_hash:
                        logger.debug("Skipping unchanged file %s", relative_path_str)
                        continue
                except OSError:
                    logger.debug("sha256 read failed, re-write %s", relative_path_str, exc_info=True)  # Fall through to re-write on read error.

            # Resolve the stored bytes BEFORE touching the target. A blob that
            # cannot be read must abort this path, not truncate the file it was
            # supposed to protect.
            if relative_path_str in blob_paths and relative_path_str not in inline:
                try:
                    raw = self._read_blob(checkpoint_id, stored_hash)
                except (OSError, TypeError, ValueError) as e:
                    logger.exception("Failed to read stored content for %s: %s", relative_path_str, e)
                    success = False
                    continue
            else:
                # Legacy inline entry. Binary files were stored as a sentinel +
                # base64 payload so the exact bytes come back (a plain UTF-8
                # write would corrupt them / silently truncate to 0 bytes).
                content = inline.get(relative_path_str, '')
                if isinstance(content, str) and content.startswith(_BINARY_SENTINEL):
                    raw = base64.b64decode(content[len(_BINARY_SENTINEL):])
                else:
                    raw = content.encode('utf-8')

            # Ensure parent directory exists
            file_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                with open(file_path, 'wb') as f:
                    f.write(raw)
                logger.debug("Restored file %s", relative_path_str)
            except OSError as e:
                logger.exception("Failed to restore file %s: %s", relative_path_str, e)
                success = False

        # Undo the run's file CREATIONS. Without this a scoped restore could
        # only ever rewind content, so a file the run created outlived the undo
        # — and under the accumulating gate a created-then-edited file was
        # captured at its half-written state, leaving a tree the run never
        # passed through while reporting success.
        for relative_path_str in checkpoint_data.get('absent', []):
            # Same defence-in-depth as the write loop above, and it matters
            # strictly more here: this branch UNLINKS, so a tampered checkpoint
            # containing "../" would delete outside the repo.
            target_path = (self.repo_root / relative_path_str).resolve()
            try:
                target_path.relative_to(self.repo_root)
            except ValueError:
                logger.exception("Refusing to delete path outside repo root: %r", relative_path_str)
                success = False
                continue
            try:
                target_path.unlink()
                logger.debug("Deleted run-created file %s", relative_path_str)
            except FileNotFoundError:
                # Already gone — the desired end state either way, so not an
                # error. Logged because a tombstone that never finds its file
                # is also what a wrongly-recorded creation looks like.
                logger.debug(
                    "Run-created file %s was already absent", relative_path_str
                )
            except IsADirectoryError:
                # _scan_absent_files excludes directories, so the path became a
                # directory after capture. Removing a tree is far beyond an
                # undo of one file.
                logger.warning(
                    "Refusing to delete directory %r recorded as absent", relative_path_str
                )
                success = False
            except OSError as e:
                logger.exception("Failed to delete %s: %s", relative_path_str, e)
                success = False
        if success:
            logger.info("Successfully restored checkpoint %s", checkpoint_id)
        else:
            logger.warning("Partially restored checkpoint %s with errors", checkpoint_id)
        return success

    def delete(self, checkpoint_id: str) -> bool:
        """
        Delete a checkpoint.

        Args:
            checkpoint_id: ID of checkpoint to delete.

        Returns:
            True if successful, False otherwise.
        """
        # Find checkpoint
        checkpoint_info = None
        for i, cp in enumerate(self.checkpoints):
            if cp['id'] == checkpoint_id:
                checkpoint_info = cp
                checkpoint_index = i
                break

        if not checkpoint_info:
            logger.error("Checkpoint %s not found", checkpoint_id)
            return False

        # Delete checkpoint file FIRST. _save_checkpoints()'s concurrent-merge
        # only resurrects disk entries whose .json file still exists, so a
        # concurrent writer cannot accidentally re-add this id after we remove
        # it from the index.
        checkpoint_path = self._index_payload_path(checkpoint_info)
        if checkpoint_path is None:
            logger.error("Refusing to delete checkpoint %s: invalid stored path", checkpoint_id)
            return False
        try:
            if checkpoint_path.exists():
                checkpoint_path.unlink()
        except OSError as e:
            logger.exception("Failed to delete checkpoint file %s: %s", checkpoint_path, e)
            return False
        # After the index file, so a failure here leaves orphan blobs rather
        # than an index pointing at content that is already gone.
        self._remove_blobs(checkpoint_id)

        # Remove from checkpoints list
        self.checkpoints.pop(checkpoint_index)
        self._save_checkpoints()

        logger.info("Deleted checkpoint %s", checkpoint_id)
        return True

