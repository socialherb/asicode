import json
import os
from pathlib import Path
from typing import Any, Optional


class SessionState:
    """Represents the state of an editing session."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.edit_history: list[dict[str, Any]] = []
        self.plan: Optional[dict[str, Any]] = None

    def save(self) -> None:
        """Save the current state to disk via SessionStateManager."""
        SessionStateManager().save_state(self)

    def load_state(self) -> None:
        """Load state from disk via SessionStateManager."""
        loaded = SessionStateManager().load_state(self.session_id)
        if loaded:
            self.edit_history = loaded.edit_history
            self.plan = loaded.plan


class SessionStateManager:
    """Manages persistence of SessionState objects to JSON files."""

    def __init__(self, base_dir=None):
        if base_dir is None:
            self.base_dir = Path.cwd() / ".asicode" / "sessions"
        else:
            self.base_dir = Path(base_dir) / ".asicode" / "sessions"
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _session_path(self, session_id: str) -> Path:
        """Return the full path for a session JSON file (session_id sanitized)."""
        return self.base_dir / f"{self._safe_id(session_id)}.json"

    @staticmethod
    def _safe_id(session_id: str) -> str:
        """Sanitize session_id: strip path-traversal chars, keep alphanumeric + -_."""
        return "".join(c for c in session_id if c.isalnum() or c in "-_")

    def save_state(self, state: SessionState) -> None:
        """Serialize a SessionState to JSON and write to disk."""
        data = {
            "session_id": state.session_id,
            "edit_history": state.edit_history,
            "plan": state.plan,
        }
        path = self._session_path(state.session_id)
        tmp_path = path.with_suffix(".tmp")
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            os.replace(tmp_path, path)  # POSIX atomic rename
        except OSError as e:
            # Clean up temp file if atomic write failed
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass
            raise RuntimeError(f"Failed to save session state to {path}: {e}")

    def load_state(self, session_id: str) -> Optional[SessionState]:
        """Load a SessionState from disk, returning None if file does not exist."""
        path = self._session_path(session_id)
        if not path.exists():
            return None
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            raise RuntimeError(f"Failed to load session state from {path}: {e}")
        state = SessionState(session_id)
        state.edit_history = data.get("edit_history", [])
        state.plan = data.get("plan")
        return state
