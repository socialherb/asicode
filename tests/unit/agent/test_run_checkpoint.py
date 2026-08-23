"""Pre-write Undo checkpoint gate (MAIN_AGENT).

The invariant worth guarding hardest is *first-seen wins*: the run checkpoint
must hold each file's content from before the run touched it. If a later write
to the same file re-captured it, ``restore()`` would put back the edited
content and Undo would silently become a no-op — which is exactly the failure a
green "a checkpoint exists" assertion would miss.
"""

from __future__ import annotations

import sys
import threading

import pytest

from external_llm.agent.checkpoint_store import CheckpointStore
from external_llm.agent.run_checkpoint import (
    MODE_FULL,
    MODE_OFF,
    MODE_SCOPED,
    RunCheckpointGate,
    resolve_checkpoint_mode,
    resolve_in_repo_paths,
)


@pytest.fixture
def repo(tmp_path):
    (tmp_path / "app.py").write_text("original app\n", encoding="utf-8")
    (tmp_path / "lib.py").write_text("original lib\n", encoding="utf-8")
    return tmp_path


# ── mode resolution ──────────────────────────────────────────────────────────


@pytest.mark.parametrize("raw", ["0", "off", "false", "no", "OFF", " No ", "FALSE"])
def test_disabled_modes(raw):
    assert resolve_checkpoint_mode(raw) == MODE_OFF


@pytest.mark.parametrize("raw", ["full", "FULL", " Full "])
def test_full_mode(raw):
    assert resolve_checkpoint_mode(raw) == MODE_FULL


@pytest.mark.parametrize("raw", [None, "", "  ", "scoped", "SCOPED", "anything-else", "1"])
def test_scoped_is_the_default(raw):
    assert resolve_checkpoint_mode(raw) == MODE_SCOPED


def test_gate_reads_the_env_var(repo, monkeypatch):
    monkeypatch.setenv("ASICODE_CHECKPOINT_ON_WRITE", "off")
    assert RunCheckpointGate(str(repo)).enabled is False
    monkeypatch.setenv("ASICODE_CHECKPOINT_ON_WRITE", "full")
    assert RunCheckpointGate(str(repo)).mode == MODE_FULL


# ── path filtering ───────────────────────────────────────────────────────────


def test_resolve_in_repo_paths_keeps_missing_targets(repo):
    """Non-existent paths survive the filter — they become tombstones.

    They used to be dropped here, which is what let a file the run CREATED
    outlive the Undo. Existence is decided downstream (content snapshot vs
    absent tombstone), so this filter is about repo containment only.
    """
    (repo / "sub").mkdir()
    got = resolve_in_repo_paths([str(repo / "app.py"), str(repo / "missing.py"), str(repo / "sub")], str(repo))
    assert got == [
        str((repo / "app.py").resolve()),
        str((repo / "missing.py").resolve()),
        str((repo / "sub").resolve()),
    ]


def test_resolve_in_repo_paths_accepts_relative_paths(repo):
    assert resolve_in_repo_paths(["app.py"], str(repo)) == [str((repo / "app.py").resolve())]


def test_resolve_in_repo_paths_rejects_escapes(repo, tmp_path):
    """Containment is still enforced — restore() both writes AND unlinks these."""
    outside = tmp_path.parent / "outside.py"
    outside.write_text("x\n", encoding="utf-8")
    try:
        assert resolve_in_repo_paths([str(outside), "../outside.py"], str(repo)) == []
    finally:
        outside.unlink()


# ── the first-seen invariant ─────────────────────────────────────────────────


def test_repeated_writes_keep_the_pre_run_content(repo):
    """The whole point of Undo: restore must rewind to before the FIRST write."""
    gate = RunCheckpointGate(str(repo), mode="scoped")

    gate.before_write([str(repo / "app.py")])
    (repo / "app.py").write_text("edit one\n", encoding="utf-8")
    gate.before_write([str(repo / "app.py")])
    (repo / "app.py").write_text("edit two\n", encoding="utf-8")

    assert CheckpointStore(str(repo)).restore(gate.checkpoint_id) is True
    assert (repo / "app.py").read_text(encoding="utf-8") == "original app\n"


def test_second_file_is_added_to_the_same_checkpoint(repo):
    gate = RunCheckpointGate(str(repo), mode="scoped")
    gate.before_write([str(repo / "app.py")])
    first_id = gate.checkpoint_id
    (repo / "app.py").write_text("edited\n", encoding="utf-8")

    gate.before_write([str(repo / "lib.py")])
    (repo / "lib.py").write_text("edited\n", encoding="utf-8")

    assert gate.checkpoint_id == first_id, "a run must produce ONE undo point"
    assert CheckpointStore(str(repo)).restore(first_id) is True
    assert (repo / "app.py").read_text(encoding="utf-8") == "original app\n"
    assert (repo / "lib.py").read_text(encoding="utf-8") == "original lib\n"


def test_new_file_only_write_is_undone_by_deleting_the_file(repo):
    """A run that only CREATES files is still undoable — via tombstones.

    This used to assert the opposite (no checkpoint at all), on the reasoning
    that a to-be-created file has no content to snapshot and an empty checkpoint
    restores nothing while reporting success. The premise was right and the
    conclusion wrong: the pre-run state of a file that does not exist is
    *absence*, which restore() can reproduce by unlinking it.
    """
    gate = RunCheckpointGate(str(repo), mode="scoped")
    gate.before_write([str(repo / "brand_new.py")])
    (repo / "brand_new.py").write_text("created by the run\n", encoding="utf-8")
    # What ToolRegistry.dispatch does after a SUCCESSFUL write: absence seen
    # before the handler only becomes a tombstone once the file really exists.
    gate.confirm_writes([str(repo / "brand_new.py")])
    assert gate.checkpoint_id is not None

    assert CheckpointStore(str(repo)).restore(gate.checkpoint_id) is True
    assert not (repo / "brand_new.py").exists()


def test_created_then_edited_file_is_deleted_not_left_mid_run(repo):
    """The accumulating gate's worst case: created by one write, edited by the next.

    By the second write the file exists, so it was captured as CONTENT — its
    half-written first version. restore() then left the tree at a state the run
    never passed through (neither pre-run absence nor the final content) and
    returned True. First-seen-wins has to span both record kinds for this to
    come out right: the tombstone from write #1 must survive write #2.
    """
    gate = RunCheckpointGate(str(repo), mode="scoped")
    new = repo / "brand_new.py"

    gate.before_write([str(new)])
    new.write_text("v1\n", encoding="utf-8")
    gate.confirm_writes([str(new)])
    gate.before_write([str(new), str(repo / "app.py")])
    new.write_text("v2\n", encoding="utf-8")
    (repo / "app.py").write_text("edited\n", encoding="utf-8")
    gate.confirm_writes([str(new), str(repo / "app.py")])

    assert CheckpointStore(str(repo)).restore(gate.checkpoint_id) is True
    assert not new.exists(), "created file must be gone, not left at v1"
    assert (repo / "app.py").read_text(encoding="utf-8") == "original app\n"


def test_a_captured_file_is_never_later_tombstoned(repo):
    """The other direction of first-seen-wins: content wins over a later absence.

    A file the run edited and then DELETED must come back with its pre-run
    content. Tombstoning it on the second sighting would make restore() unlink
    a file that existed before the run — an undo that destroys the user's work.
    """
    gate = RunCheckpointGate(str(repo), mode="scoped")
    gate.before_write([str(repo / "app.py")])
    (repo / "app.py").unlink()
    gate.before_write([str(repo / "app.py")])

    assert CheckpointStore(str(repo)).restore(gate.checkpoint_id) is True
    assert (repo / "app.py").read_text(encoding="utf-8") == "original app\n"


def test_restore_refuses_to_delete_a_directory_recorded_as_absent(repo):
    """A tombstoned path that became a DIRECTORY must not be removed as a tree."""
    gate = RunCheckpointGate(str(repo), mode="scoped")
    gate.before_write([str(repo / "later_a_dir")])
    (repo / "later_a_dir").mkdir()
    (repo / "later_a_dir" / "keep.py").write_text("keep me\n", encoding="utf-8")

    assert CheckpointStore(str(repo)).restore(gate.checkpoint_id) is False
    assert (repo / "later_a_dir" / "keep.py").read_text(encoding="utf-8") == "keep me\n"


def test_new_file_then_existing_file_still_gets_a_checkpoint(repo):
    gate = RunCheckpointGate(str(repo), mode="scoped")
    gate.before_write([str(repo / "brand_new.py")])
    gate.before_write([str(repo / "app.py")])
    assert gate.checkpoint_id is not None


def test_unknown_target_paths_do_not_crash(repo):
    gate = RunCheckpointGate(str(repo), mode="scoped")
    gate.before_write(None)  # _extract_write_target_paths returned None
    gate.before_write([])
    assert gate.checkpoint_id is None


# ── modes ────────────────────────────────────────────────────────────────────


def test_disabled_gate_never_touches_the_store(repo):
    gate = RunCheckpointGate(str(repo), mode="off")
    gate.before_write([str(repo / "app.py")])
    assert gate.checkpoint_id is None
    assert not (repo / ".asicode" / "checkpoints").exists()


def test_full_mode_snapshots_once(repo):
    gate = RunCheckpointGate(str(repo), mode="full")
    gate.before_write([str(repo / "app.py")])
    first = gate.checkpoint_id
    gate.before_write([str(repo / "lib.py")])
    assert gate.checkpoint_id == first
    entries = CheckpointStore(str(repo)).list()
    assert len(entries) == 1 and entries[0]["scope"] == "full"


def test_full_mode_fires_even_for_a_brand_new_file(repo):
    """Scoped needs an existing target; full snapshots the repo regardless."""
    gate = RunCheckpointGate(str(repo), mode="full")
    gate.before_write([str(repo / "brand_new.py")])
    assert gate.checkpoint_id is not None


def test_gate_never_raises_on_store_failure(repo, monkeypatch):
    gate = RunCheckpointGate(str(repo), mode="scoped")
    monkeypatch.setattr(gate, "_get_store", lambda: (_ for _ in ()).throw(OSError("disk full")))
    gate.before_write([str(repo / "app.py")])  # must not propagate
    assert gate.checkpoint_id is None


# ── CheckpointStore.extend ───────────────────────────────────────────────────


def test_extend_is_a_noop_for_unknown_id(repo):
    assert CheckpointStore(str(repo)).extend("nope", [str(repo / "app.py")]) == 0


def test_extend_is_a_noop_for_a_full_checkpoint(repo):
    store = CheckpointStore(str(repo))
    cid = store.create("full")
    assert store.extend(cid, [str(repo / "app.py")]) == 0


def test_extend_skips_already_captured_files(repo):
    store = CheckpointStore(str(repo))
    cid = store.create("scoped", files=[str(repo / "app.py")])
    assert store.extend(cid, [str(repo / "app.py")]) == 0
    assert store.extend(cid, [str(repo / "lib.py")]) == 1


def test_extend_updates_the_index_file_count(repo):
    store = CheckpointStore(str(repo))
    cid = store.create("scoped", files=[str(repo / "app.py")])
    store.extend(cid, [str(repo / "lib.py")])
    assert [c["file_count"] for c in store.list() if c["id"] == cid] == [2]
    # and a freshly loaded store sees the persisted count
    assert [c["file_count"] for c in CheckpointStore(str(repo)).list()] == [2]


def test_extend_round_trips_binary_files(repo):
    blob = repo / "logo.bin"
    blob.write_bytes(b"\x89PNG\r\n\x1a\n\xff\xfe")
    store = CheckpointStore(str(repo))
    cid = store.create("scoped", files=[str(repo / "app.py")])
    assert store.extend(cid, [str(blob)]) == 1
    blob.write_bytes(b"corrupted")
    assert store.restore(cid) is True
    assert blob.read_bytes() == b"\x89PNG\r\n\x1a\n\xff\xfe"


# ── concurrency ──────────────────────────────────────────────────────────────


def test_concurrent_first_writes_produce_one_checkpoint(repo):
    """Subagents write in parallel; they must share a single Undo point.

    sys.setswitchinterval is lowered so the GIL actually preempts inside the
    unlocked window — at the default 5ms this test passes even with the lock
    removed, asserting nothing.
    """
    for i in range(8):
        (repo / f"f{i}.py").write_text(f"orig {i}\n", encoding="utf-8")
    gate = RunCheckpointGate(str(repo), mode="scoped")
    old = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    try:
        barrier = threading.Barrier(8)

        def worker(i):
            barrier.wait()
            gate.before_write([str(repo / f"f{i}.py")])

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    finally:
        sys.setswitchinterval(old)

    assert len(CheckpointStore(str(repo)).list()) == 1
    for i in range(8):
        (repo / f"f{i}.py").write_text("clobbered\n", encoding="utf-8")
    assert CheckpointStore(str(repo)).restore(gate.checkpoint_id) is True
    for i in range(8):
        assert (repo / f"f{i}.py").read_text(encoding="utf-8") == f"orig {i}\n"


# ── wiring into the write path ───────────────────────────────────────────────


def test_write_tool_dispatch_creates_and_restores_a_checkpoint(repo):
    from external_llm.agent.tool_registry import AgentConfig, ToolRegistry

    reg = ToolRegistry(repo_root=str(repo), config=AgentConfig())
    assert reg.run_checkpoint_id is None, "read-only run must not create a checkpoint"

    result = reg.dispatch(
        "edit_text",
        {"file_path": "app.py", "old_text": "original app", "new_text": "rewritten"},
    )
    assert result.ok, result.error
    cid = reg.run_checkpoint_id
    assert cid is not None
    assert (repo / "app.py").read_text(encoding="utf-8") == "rewritten\n"

    assert CheckpointStore(str(repo)).restore(cid) is True
    assert (repo / "app.py").read_text(encoding="utf-8") == "original app\n"


def test_subagent_clone_shares_the_run_checkpoint(repo):
    from external_llm.agent.tool_registry import AgentConfig, ToolRegistry

    cfg = AgentConfig()
    reg = ToolRegistry(repo_root=str(repo), config=cfg)
    reg.dispatch(
        "edit_text",
        {"file_path": "app.py", "old_text": "original app", "new_text": "parent edit"},
    )
    clone = reg.clone_for_subagent(cfg)
    assert clone._run_checkpoint_gate is reg._run_checkpoint_gate
    assert clone.run_checkpoint_id == reg.run_checkpoint_id


def test_clone_taken_before_any_write_still_shares_the_gate(repo):
    """The clone must share the gate even when the parent has never written.

    This is the ONLY shape that occurs in production and the one the test above
    cannot see: ``OrchestratorAgent`` clones ``_registry_proto``, a registry it
    documents as "used only for repo_root / config access" and never dispatches
    through. With the gate built lazily on first write, that clone copied None
    and built its own — but ``is`` comparison of two Nones passes, so the
    sharing assertion above stayed green while every multi-agent run split its
    Undo point across N checkpoints.
    """
    from external_llm.agent.tool_registry import AgentConfig, ToolRegistry

    cfg = AgentConfig()
    reg = ToolRegistry(repo_root=str(repo), config=cfg)
    assert reg._run_checkpoint_gate is not None, "gate must exist before the first write, else clones copy None"
    assert reg.run_checkpoint_id is None, "still nothing captured"

    clone = reg.clone_for_subagent(cfg)
    assert clone._run_checkpoint_gate is reg._run_checkpoint_gate

    # A write by the SUBAGENT must land in the parent's checkpoint, and be the
    # only checkpoint the run produces.
    clone.dispatch(
        "edit_text",
        {"file_path": "app.py", "old_text": "original app", "new_text": "subagent edit"},
    )
    assert reg.run_checkpoint_id is not None, (
        "parent must see the subagent's checkpoint — agent_loop stamps the id from the parent registry"
    )
    assert clone.run_checkpoint_id == reg.run_checkpoint_id
    assert len(CheckpointStore(str(repo)).list()) == 1

    assert CheckpointStore(str(repo)).restore(reg.run_checkpoint_id) is True
    assert (repo / "app.py").read_text(encoding="utf-8") == "original app\n"


def test_constructing_a_registry_does_not_create_the_store(repo, tmp_path):
    """Eager gate, still-lazy store: a read-only run touches no checkpoint dir.

    The gate was lazy so that constructing a registry could not create
    .asicode/checkpoints/. Making it eager keeps that guarantee only because
    CheckpointStore construction (which mkdirs) stays behind
    ``RunCheckpointGate._get_store``.
    """
    from external_llm.agent.tool_registry import AgentConfig, ToolRegistry

    reg = ToolRegistry(repo_root=str(repo), config=AgentConfig())
    reg.dispatch("read_file", {"file_path": "app.py"})
    assert reg._run_checkpoint_gate is not None
    assert reg._run_checkpoint_gate._store is None
    assert not (repo / ".asicode" / "checkpoints").exists()


# ── A refused write must leave no tombstone ─────────────────────────────────


def test_a_refused_write_leaves_no_tombstone(repo):
    """before_write fires BEFORE the handler, so absence is not yet creation.

    A write can be refused after the gate has already seen its target missing —
    the post-edit syntax gate, a scoped write filter, a malformed argument. If
    that speculative absence became a tombstone, Undo would DELETE the path
    later, and the file it deletes could well be one the USER created by hand
    after the agent failed to. Data loss caused by a write that never happened.
    """
    gate = RunCheckpointGate(str(repo), mode="scoped")
    gate.before_write([str(repo / "never_written.py")])
    # The write failed, so dispatch never reaches confirm_writes for it at all.

    assert gate.checkpoint_id is None, "a run that changed nothing has no undo point"
    assert CheckpointStore(str(repo)).list() == []

    # The user creates that exact path by hand afterwards.
    (repo / "never_written.py").write_text("mine\n", encoding="utf-8")

    # A LATER successful write elsewhere gives the run a real checkpoint.
    gate.before_write([str(repo / "app.py")])
    (repo / "app.py").write_text("edited\n", encoding="utf-8")
    gate.confirm_writes([str(repo / "app.py")])
    assert gate.checkpoint_id is not None

    assert CheckpointStore(str(repo)).restore(gate.checkpoint_id) is True
    assert (repo / "app.py").read_text(encoding="utf-8") == "original app\n"
    assert (repo / "never_written.py").read_text(encoding="utf-8") == "mine\n", (
        "undo deleted a file the run never created"
    )


def test_a_pending_absence_survives_until_a_later_write_creates_it(repo):
    """Still-missing paths stay pending — a later write in the run may create them."""
    gate = RunCheckpointGate(str(repo), mode="scoped")
    target = repo / "eventually.py"

    gate.before_write([str(target)])
    gate.confirm_writes([str(target)])  # first attempt failed: still missing
    assert gate.checkpoint_id is None

    gate.before_write([str(target)])
    target.write_text("second attempt worked\n", encoding="utf-8")
    gate.confirm_writes([str(target)])  # now it exists
    assert gate.checkpoint_id is not None

    assert CheckpointStore(str(repo)).restore(gate.checkpoint_id) is True
    assert not target.exists()


def test_dispatch_of_a_refused_write_records_no_tombstone(repo):
    """The same guarantee through the real write path, not the gate alone."""
    from external_llm.agent.tool_registry import AgentConfig, ToolRegistry

    reg = ToolRegistry(repo_root=str(repo), config=AgentConfig())
    result = reg.dispatch(
        "edit_text",
        {"file_path": "broken.py", "old_text": "", "new_text": "def ("},
    )
    assert not result.ok, "precondition: this write must be refused"
    assert reg.run_checkpoint_id is None
    assert not (repo / "broken.py").exists()
