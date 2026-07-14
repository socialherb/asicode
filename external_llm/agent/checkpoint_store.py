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


class CheckpointStore:
    """
    Manages file checkpoints with timeline UI support.
    """

    def __init__(self, repo_root: str, store_dir: str = '.asicode/checkpoints',
                 max_checkpoints: int = 50):
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
        """
        self.repo_root = Path(repo_root).resolve()
        store_path = Path(store_dir)
        if not store_path.is_absolute():
            store_path = self.repo_root / store_dir
        self.store_dir = store_path.resolve()
        self.max_checkpoints = max_checkpoints

        # Create store directory if it doesn't exist
        self.store_dir.mkdir(parents=True, exist_ok=True)

        # Determine repository name for subdirectory
        repo_name = self.repo_root.name
        self.checkpoint_dir = self.store_dir / repo_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.checkpoint_file = self.checkpoint_dir / 'checkpoints.json'
        self._load_checkpoints()

    def _load_checkpoints(self) -> None:
        """Load checkpoints from JSON file."""
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file, encoding='utf-8') as f:
                    self.checkpoints = json.load(f)
            except (OSError, json.JSONDecodeError) as e:
                logger.warning(f"Failed to load checkpoints: {e}")
                self.checkpoints = []
        else:
            self.checkpoints = []

    def _save_checkpoints(self) -> None:
        """Save checkpoints to JSON file (atomic write via tmp + os.replace).

        Writes to a sibling .tmp file first, then atomically renames it into
        place. This prevents a truncated/partial checkpoints.json from being
        left behind if the process is interrupted mid-write (e.g. disk full,
        SIGKILL), which would otherwise cause _load_checkpoints() to silently
        reset self.checkpoints to [] and lose the whole checkpoint index.
        Mirrors the atomic-write pattern in session_state.py:save_state().

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
            except (OSError, json.JSONDecodeError):
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
                # resurrected by the merge.
                cp_file = self.checkpoint_dir / cp.get('path', f"{cid}.json")
                if cp_file.exists():
                    merged.append(cp)
            merged.sort(key=lambda x: x.get('timestamp', 0), reverse=True)
            self.checkpoints = merged

            tmp_path = self.checkpoint_file.with_suffix('.json.tmp')
            try:
                with open(tmp_path, 'w', encoding='utf-8') as f:
                    json.dump(self.checkpoints, f, indent=2, ensure_ascii=False)
                os.replace(tmp_path, self.checkpoint_file)  # POSIX atomic rename
            except OSError as e:
                logger.error(f"Failed to save checkpoints: {e}")
                try:
                    tmp_path.unlink(missing_ok=True)
                except Exception:
                    pass
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
        for root, dirs, files in os.walk(self.repo_root):
            # Prune excluded directories in-place so os.walk skips their subtrees.
            # Dot-dirs are pruned wholesale: caches (.ruff_cache, .mypy_cache, …)
            # and tool dirs (.git, .asicode, .tenet, .claude) are never edit
            # targets and hashing them dominated scan cost (a single .ruff_cache
            # held 26k files = 95% of the SHA256 time). Aligns with
            # _shared_utils._walk_should_skip_dir's dot-dir heuristic.
            dirs[:] = [d for d in dirs if d not in exclude_dirs and not d.startswith('.')]
            for fname in files:
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
                    logger.warning(f"Could not read file {file_path}: {e}")
                    continue

        return file_hashes

    def _scan_listed_files(self, files) -> dict[Path, str]:
        """Hash only the given paths (scoped snapshot; see :meth:`_scan_files`)."""
        file_hashes: dict[Path, str] = {}
        for entry in files:
            if not entry:
                continue
            p = Path(entry)
            if not p.is_absolute():
                p = self.repo_root / p
            try:
                p = p.resolve()
                relative = p.relative_to(self.repo_root)
            except (ValueError, OSError):
                logger.warning(f"Skipping checkpoint path outside repo root: {entry!r}")
                continue
            if not p.is_file():
                continue  # to-be-created target or directory — nothing to snapshot
            try:
                file_hashes[relative] = self._sha256_file(p)
            except OSError as e:
                logger.warning(f"Could not read file {p}: {e}")
        return file_hashes

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

    def create(self, description: str = '', files=None) -> str:
        """
        Create a timestamped checkpoint with SHA256 + contents of tracked files.

        Args:
            description: Optional description for the checkpoint.
            files: Optional iterable of paths to snapshot instead of the whole
                repo (scoped checkpoint). Cheap enough to run on every write
                turn — the full-repo walk reads/stores every source file.
                restore() only writes files present in the checkpoint, so a
                scoped restore never touches unrelated files. Limitation: a
                file *created* after a scoped checkpoint is not deleted by
                restore (it wasn't in the snapshot).

        Returns:
            Checkpoint ID.
        """
        checkpoint_id = f"checkpoint_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        timestamp = time.time()

        # Scan files and compute hashes
        file_hashes = self._scan_files(files)

        # Create checkpoint data
        checkpoint_data = {
            'id': checkpoint_id,
            'timestamp': timestamp,
            'description': description,
            'scope': 'files' if files is not None else 'full',
            'file_count': len(file_hashes),
            'file_hashes': {str(k): v for k, v in file_hashes.items()},
            'files': {}
        }

        # Store actual file contents. Binary files that fail UTF-8 decoding
        # are base64-encoded (with a sentinel marker) so restore() can write
        # back the exact bytes — storing '' would silently produce a 0-byte
        # file and corrupt binary assets (images, compiled objects, etc.).
        for relative_path_str, _file_hash in checkpoint_data['file_hashes'].items():
            relative_path = Path(relative_path_str)
            file_path = self.repo_root / relative_path
            try:
                with open(file_path, 'rb') as fb:
                    raw = fb.read()
                # Try to store as plain UTF-8 text (the common case).
                try:
                    raw.decode('utf-8')
                    checkpoint_data['files'][relative_path_str] = raw.decode('utf-8')
                    continue
                except UnicodeDecodeError:
                    pass
                # Binary: base64-encode with a sentinel so restore() knows to
                # decode bytes rather than write the text verbatim.
                checkpoint_data['files'][relative_path_str] = (
                    _BINARY_SENTINEL + base64.b64encode(raw).decode('ascii')
                )
            except OSError as e:
                logger.warning(f"Could not read file content for {file_path}: {e}")
                checkpoint_data['files'][relative_path_str] = ''

        # Save checkpoint to individual file
        checkpoint_path = self.checkpoint_dir / f"{checkpoint_id}.json"
        try:
            atomic_write_json(checkpoint_path, checkpoint_data, indent=2, ensure_ascii=True)
        except OSError as e:
            logger.error(f"Failed to save checkpoint {checkpoint_id}: {e}")
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

        # Evict oldest checkpoints when retention limit is exceeded.
        if self.max_checkpoints > 0:
            while len(self.checkpoints) > self.max_checkpoints:
                self._evict_oldest()

        # Save updated checkpoints list
        self._save_checkpoints()

        logger.info(f"Created checkpoint {checkpoint_id} with {len(file_hashes)} files")
        return checkpoint_id

    def _evict_oldest(self) -> None:
        """Remove the single oldest checkpoint (file + index entry).

        List is kept sorted newest-first, so the oldest is the last element.
        """
        if not self.checkpoints:
            return
        oldest = self.checkpoints.pop()
        try:
            cp_path = self.checkpoint_dir / oldest['path']
            if cp_path.exists():
                cp_path.unlink()
            logger.info(f"Evicted old checkpoint {oldest['id']}")
        except (OSError, KeyError) as e:
            logger.warning(f"Failed to evict checkpoint {oldest.get('id', '?')}: {e}")

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
            logger.error(f"Checkpoint {checkpoint_id} not found")
            return False
        # Load checkpoint data
        checkpoint_path = self.checkpoint_dir / checkpoint_info['path']
        if not checkpoint_path.exists():
            logger.error(f"Checkpoint file {checkpoint_path} not found")
            return False
        try:
            with open(checkpoint_path, encoding='utf-8') as f:
                checkpoint_data = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            logger.error(f"Failed to load checkpoint {checkpoint_id}: {e}")
            return False
        # Restore files
        success = True
        for relative_path_str, content in checkpoint_data.get('files', {}).items():
            # Defense-in-depth: resolve and verify each path stays within
            # repo_root before writing. relative_path_str is read directly from
            # the checkpoint JSON (untrusted) — a tampered checkpoint could
            # otherwise write outside the repo via "../" traversal.
            target_path = (self.repo_root / relative_path_str).resolve()
            try:
                target_path.relative_to(self.repo_root)
            except ValueError:
                logger.error(
                    f"Refusing to restore path outside repo root: {relative_path_str!r}"
                )
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
                        logger.debug(f"Skipping unchanged file {relative_path_str}")
                        continue
                except OSError:
                    pass  # Fall through to re-write on read error.

            # Ensure parent directory exists
            file_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                # Binary files are stored as a sentinel + base64 payload so
                # that the exact bytes are restored (a plain UTF-8 write would
                # corrupt them / silently truncate to 0 bytes).
                if isinstance(content, str) and content.startswith(_BINARY_SENTINEL):
                    raw = base64.b64decode(content[len(_BINARY_SENTINEL):])
                    with open(file_path, 'wb') as f:
                        f.write(raw)
                else:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                logger.debug(f"Restored file {relative_path_str}")
            except OSError as e:
                logger.error(f"Failed to restore file {relative_path_str}: {e}")
                success = False
        if success:
            logger.info(f"Successfully restored checkpoint {checkpoint_id}")
        else:
            logger.warning(f"Partially restored checkpoint {checkpoint_id} with errors")
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
            logger.error(f"Checkpoint {checkpoint_id} not found")
            return False

        # Delete checkpoint file FIRST. _save_checkpoints()'s concurrent-merge
        # only resurrects disk entries whose .json file still exists, so a
        # concurrent writer cannot accidentally re-add this id after we remove
        # it from the index.
        checkpoint_path = self.checkpoint_dir / checkpoint_info['path']
        try:
            if checkpoint_path.exists():
                checkpoint_path.unlink()
        except OSError as e:
            logger.error(f"Failed to delete checkpoint file {checkpoint_path}: {e}")
            return False

        # Remove from checkpoints list
        self.checkpoints.pop(checkpoint_index)
        self._save_checkpoints()

        logger.info(f"Deleted checkpoint {checkpoint_id}")
        return True

