"""
Integration test helpers for asicode.
"""
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any


def create_test_file(repo_root: str, filepath: str, content: str) -> Path:
    """Create a test file in the repository root."""
    full_path = Path(repo_root) / filepath
    full_path.parent.mkdir(parents=True, exist_ok=True)
    full_path.write_text(content)
    return full_path


def git_add_and_commit(repo_root: str, message: str = "test commit") -> None:
    """Add all changes and commit in the git repository."""
    subprocess.run(["git", "add", "."], cwd=repo_root, capture_output=True, check=True)
    subprocess.run(["git", "commit", "-m", message], cwd=repo_root, capture_output=True, check=True)


def apply_patch_and_verify(repo_root: str, patch_content: str, expected_changes: list[str]) -> bool:
    """
    Apply a patch and verify the expected changes were made.

    Args:
        repo_root: Repository root directory
        patch_content: Unified diff patch
        expected_changes: List of regex patterns to match in changed files

    Returns:
        True if patch applied successfully and all expected changes are present
    """
    # Write patch to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.patch', delete=False) as f:
        f.write(patch_content)
        patch_file = f.name

    try:
        # Apply patch
        result = subprocess.run(
            ["git", "apply", "--verbose", patch_file],
            cwd=repo_root,
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            # Try with 3-way fallback
            result = subprocess.run(
                ["git", "apply", "--3way", "--verbose", patch_file],
                cwd=repo_root,
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                return False

        # Verify expected changes
        for pattern in expected_changes:
            # Get diff of working directory
            diff_result = subprocess.run(
                ["git", "diff", "HEAD"],
                cwd=repo_root,
                capture_output=True,
                text=True
            )
            if not re.search(pattern, diff_result.stdout, re.MULTILINE | re.DOTALL):
                return False

        return True
    finally:
        os.unlink(patch_file)


def create_sample_repo_structure(repo_root: str) -> None:
    """Create a sample repository structure for testing."""
    # Create some Python files
    files = {
        "main.py": """def main():
    print("Hello, world!")

if __name__ == "__main__":
    main()
""",
        "utils/helpers.py": """def helper_function():
    return "help"

def another_helper():
    return 42
""",
        "tests/test_basic.py": """def test_helper():
    assert True
""",
    }

    for filepath, content in files.items():
        create_test_file(repo_root, filepath, content)

    git_add_and_commit(repo_root, "Initial sample structure")


def capture_sse_events(response_lines: list[str]) -> list[dict[str, Any]]:
    """
    Parse SSE event lines from a response.

    Args:
        response_lines: Lines from a Server-Sent Events response

    Returns:
        List of parsed event objects
    """
    events = []
    current_event = {}

    for line in response_lines:
        line = line.strip()
        if not line:
            if current_event:
                events.append(current_event)
                current_event = {}
            continue

        if line.startswith("event:"):
            current_event["event"] = line[6:].strip()
        elif line.startswith("data:"):
            data = line[5:].strip()
            if data:
                try:
                    current_event["data"] = json.loads(data)
                except json.JSONDecodeError:
                    current_event["data"] = data
        elif line.startswith("id:"):
            current_event["id"] = line[3:].strip()
        elif line.startswith("retry:"):
            current_event["retry"] = int(line[6:].strip())

    if current_event:
        events.append(current_event)

    return events


def verify_event_sequence(events: list[dict[str, Any]], expected_pattern: list[str]) -> bool:
    """
    Verify that events occur in the expected sequence.

    Args:
        events: List of event objects with 'event' field
        expected_pattern: List of event types in expected order

    Returns:
        True if events match expected pattern
    """
    event_types = [e.get("event") for e in events if "event" in e]

    if len(event_types) != len(expected_pattern):
        return False

    return all(actual == expected for actual, expected in zip(event_types, expected_pattern, strict=False))


def create_memory_file(repo_root: str, content: str = "# Test Memory\n\nTest content.") -> Path:
    """Create .asicode/memory.md file."""
    memory_dir = Path(repo_root) / ".asicode"
    memory_dir.mkdir(exist_ok=True)
    memory_file = memory_dir / "memory.md"
    memory_file.write_text(content)
    return memory_file


def create_session_history_file(repo_root: str, session_data: dict[str, Any]) -> Path:
    """Create .asicode/sessions.jsonl entry."""
    session_dir = Path(repo_root) / ".asicode"
    session_dir.mkdir(exist_ok=True)
    session_file = session_dir / "sessions.jsonl"

    # Append to existing file or create new
    with open(session_file, "a") as f:
        f.write(json.dumps(session_data) + "\n")

    return session_file


def get_file_content(repo_root: str, filepath: str) -> str:
    """Get content of a file in the repository."""
    return (Path(repo_root) / filepath).read_text()


def assert_file_contains(repo_root: str, filepath: str, expected_content: str) -> None:
    """Assert that a file contains the expected content."""
    content = get_file_content(repo_root, filepath)
    assert expected_content in content, f"Expected content not found in {filepath}"


def assert_file_matches(repo_root: str, filepath: str, pattern: str) -> None:
    """Assert that a file matches a regex pattern."""
    content = get_file_content(repo_root, filepath)
    assert re.search(pattern, content, re.MULTILINE | re.DOTALL) is not None, \
        f"Pattern not found in {filepath}"


def cleanup_test_files(*paths: str) -> None:
    """Clean up test files and directories."""
    for path in paths:
        if os.path.exists(path):
            if os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)
            else:
                os.unlink(path)
