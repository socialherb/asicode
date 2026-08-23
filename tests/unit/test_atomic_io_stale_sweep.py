"""Crash-leftover sweep for atomic_io temp files.

Each writer wraps its temp file in ``except BaseException: os.unlink(tmp_path)``,
which covers the EXCEPTION path only. A SIGKILL, an OOM kill or a power loss
runs no Python handler at all, so a half-written ``.atomic_*.tmp`` survives
indefinitely — observed in this repo as a 96 MB orphan in
``.asicode/vector_cache/``, left by an interrupted 124 MB metadata dump and
still present a day later, because nothing ever looked for one.

The sweep is age-gated rather than unconditional: a *concurrent* writer in
another process holds its own live temp file in the same directory, and
deleting that would corrupt its ``os.replace``. These tests pin both halves —
stale leftovers go, live ones stay.
"""

from __future__ import annotations

import json
import os
import time

import pytest

from external_llm.common import atomic_io


@pytest.fixture(autouse=True)
def _reset_sweep_memo():
    """The sweep runs once per directory per process; clear the memo between
    tests so each gets a fresh scan of its own tmp_path."""
    atomic_io._swept_dirs.clear()
    yield
    atomic_io._swept_dirs.clear()


def _leftover(d, name: str, age_s: float, size: int = 32) -> str:
    p = os.path.join(str(d), name)
    with open(p, "w", encoding="utf-8") as fh:
        fh.write("x" * size)
    stamp = time.time() - age_s
    os.utime(p, (stamp, stamp))
    return p


def test_stale_leftover_is_reclaimed_on_the_next_write(tmp_path):
    stale = _leftover(tmp_path, ".atomic_crashed.tmp", age_s=7200)
    atomic_io.atomic_write_json(tmp_path / "data.json", {"a": 1})
    assert not os.path.exists(stale), "crash leftover survived the sweep"
    assert json.loads((tmp_path / "data.json").read_text()) == {"a": 1}


def test_live_temp_file_of_a_concurrent_writer_is_preserved(tmp_path):
    """The dangerous direction: deleting another process's in-flight temp file
    would corrupt its rename. Anything younger than the age gate is off-limits."""
    live = _leftover(tmp_path, ".atomic_inflight.tmp", age_s=0)
    atomic_io.atomic_write_json(tmp_path / "data.json", {"a": 1})
    assert os.path.exists(live), "a live concurrent temp file was deleted"


def test_non_temp_files_are_never_touched(tmp_path):
    real = tmp_path / "metadata.json"
    real.write_text('{"keep": true}', encoding="utf-8")
    old_unrelated = _leftover(tmp_path, "notes.txt", age_s=7200)
    atomic_io.atomic_write_json(tmp_path / "data.json", {"a": 1})
    assert real.exists() and os.path.exists(old_unrelated)


def test_directories_matching_the_prefix_are_not_removed(tmp_path):
    d = tmp_path / ".atomic_subdir"
    d.mkdir()
    stamp = time.time() - 7200
    os.utime(d, (stamp, stamp))
    atomic_io.atomic_write_json(tmp_path / "data.json", {"a": 1})
    assert d.is_dir(), "sweep removed a directory"


def test_sweep_runs_once_per_directory_per_process(tmp_path):
    atomic_io.atomic_write_json(tmp_path / "data.json", {"a": 1})
    stale = _leftover(tmp_path, ".atomic_later.tmp", age_s=7200)
    atomic_io.atomic_write_json(tmp_path / "data.json", {"b": 2})
    assert os.path.exists(stale), (
        "second write re-scanned; the memo exists because orphans come from a "
        "process that already died, so one scan per directory is enough"
    )


def test_sweep_reports_how_much_it_reclaimed(tmp_path):
    _leftover(tmp_path, ".atomic_a.tmp", age_s=7200)
    _leftover(tmp_path, ".atomic_b.tmp", age_s=7200)
    _leftover(tmp_path, ".atomic_fresh.tmp", age_s=0)
    assert atomic_io.sweep_stale_temp_files(str(tmp_path)) == 2


def test_sweep_never_raises_on_a_missing_directory(tmp_path):
    assert atomic_io.sweep_stale_temp_files(str(tmp_path / "nope")) == 0


@pytest.mark.parametrize(
    "writer,payload",
    [
        ("atomic_write_json", {"a": 1}),
        ("atomic_write_jsonl", [{"a": 1}, {"b": 2}]),
    ],
)
def test_every_writer_sweeps(tmp_path, writer, payload):
    """All three temp-file creators must sweep, not just the JSON one."""
    stale = _leftover(tmp_path, ".atomic_crashed.tmp", age_s=7200)
    getattr(atomic_io, writer)(tmp_path / "out.dat", payload)
    assert not os.path.exists(stale), f"{writer} did not sweep"


def test_write_namespace_json_sweeps(tmp_path):
    stale = _leftover(tmp_path, ".atomic_crashed.tmp", age_s=7200)
    atomic_io.write_namespace_json(tmp_path / "ns.json", "sect", {"k": "v"})
    assert not os.path.exists(stale)
    assert json.loads((tmp_path / "ns.json").read_text())["sect"] == {"k": "v"}


# ── Shared pipeline (_atomic_replace) ─────────────────────────────────────────


def test_write_body_failure_removes_temp_and_keeps_target(tmp_path):
    """The shared pipeline must clean up its temp file when the payload
    serializer raises, and must never touch the existing target."""
    target = tmp_path / "data.json"
    target.write_text('{"old": true}', encoding="utf-8")

    def _boom(fh):
        raise RuntimeError("serializer exploded")

    with pytest.raises(RuntimeError):
        atomic_io._atomic_replace(target, ".tmp", _boom)
    assert json.loads(target.read_text()) == {"old": True}
    leftovers = [p for p in os.listdir(tmp_path) if p.startswith(".atomic_")]
    assert leftovers == [], f"temp file leaked on write failure: {leftovers}"


def test_finalize_failure_removes_temp_and_keeps_target(tmp_path):
    """The same cleanup contract holds when the post-write hook raises."""
    target = tmp_path / "data.txt"
    target.write_text("old", encoding="utf-8")

    def _finalize(tmp_path, target_path):
        raise OSError("chmod exploded")

    with pytest.raises(OSError, match="chmod exploded"):
        atomic_io._atomic_replace(
            target,
            ".tmp",
            lambda fh: fh.write("new"),
            finalize=_finalize,
        )
    assert target.read_text() == "old"
    leftovers = [p for p in os.listdir(tmp_path) if p.startswith(".atomic_")]
    assert leftovers == [], f"temp file leaked on finalize failure: {leftovers}"


def test_every_writer_routes_through_the_shared_pipeline(tmp_path, monkeypatch):
    """All three public writers must go through _atomic_replace — the dedup is
    structural, not incidental: json uses .tmp, jsonl .jsonl, and only text
    passes the mode-preserving finalize hook."""
    calls: list = []
    real = atomic_io._atomic_replace

    def spy(*args, **kwargs):
        calls.append((args[1], kwargs.get("finalize") is not None))
        return real(*args, **kwargs)

    monkeypatch.setattr(atomic_io, "_atomic_replace", spy)
    atomic_io.atomic_write_json(tmp_path / "a.json", {"a": 1})
    atomic_io.atomic_write_jsonl(tmp_path / "b.jsonl", [{"b": 2}])
    atomic_io.atomic_write_text(tmp_path / "c.txt", "x", mode=0o600)
    assert calls == [(".tmp", False), (".jsonl", False), (".tmp", True)]
