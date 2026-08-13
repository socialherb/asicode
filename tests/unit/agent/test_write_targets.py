"""Parity contract for write-tool target resolution.

Four consumers must agree on "which files does this call touch?" — the Undo
checkpoint, the rollback snapshot, the approval gate and the file-lock manager.
They used to carry four private copies of the extraction, and all four resolved
targets from the RAW arguments while every write handler normalises its own
(``__raw_arguments`` recovery; for ``write_plan`` a JSON-string / fenced / bare
list / top-level-``ops`` plan). Every shape a handler repaired was therefore
invisible to the gates: the write landed normally and the run silently got no
Undo point, no rollback snapshot and no file lock.

The matrix below is the guard. For each accepted argument shape it dispatches
the tool for real and asserts that the file the write actually changed is the
file all four consumers saw. A future copy of the extraction that misses a shape
fails here rather than in a user's un-undoable run.
"""
from __future__ import annotations

import json

import pytest

from external_llm.agent.checkpoint_store import CheckpointStore
from external_llm.agent.orchestrator import FileLockManager
from external_llm.agent.tool_registry import AgentConfig, ToolRegistry
from external_llm.agent.tool_safety import WriteSafetyManager
from external_llm.agent.write_targets import (
    normalize_plan,
    parse_patch_targets,
    plan_target_paths,
    write_target_paths,
)

ORIGINAL = "original = 1\n"
REWRITTEN = "rewritten = 2\n"


@pytest.fixture
def repo(tmp_path):
    (tmp_path / "app.py").write_text(ORIGINAL, encoding="utf-8")
    (tmp_path / ".git").mkdir()
    return tmp_path


# ── the shapes each tool accepts ────────────────────────────────────────────

_OPS = [{"op": "replace_file", "path": "app.py", "content": REWRITTEN}]
_PLAN = {"kind": "ASICODE_PLAN_V1", "ops": _OPS}

_PATCH_BODY = "@@ -1 +1 @@\n-original = 1\n+rewritten = 2\n"

WRITE_PLAN_SHAPES = {
    "plan-dict": {"plan": _PLAN},
    "plan-json-string": {"plan": json.dumps(_PLAN)},
    "plan-fenced": {"plan": "```json\n" + json.dumps(_PLAN) + "\n```"},
    "plan-bare-list": {"plan": _OPS},
    "top-level-ops": {"ops": _OPS},
    "top-level-operations": {"operations": _OPS},
    "raw-arguments": {"__raw_arguments": json.dumps({"plan": _PLAN})},
}

APPLY_PATCH_SHAPES = {
    "git-prefix": {
        "patch": "diff --git a/app.py b/app.py\n--- a/app.py\n+++ b/app.py\n" + _PATCH_BODY
    },
    "bare-ab-prefix": {"patch": "--- a/app.py\n+++ b/app.py\n" + _PATCH_BODY},
    # What plain `diff -u old new` and `git diff --no-prefix` emit. Every former
    # copy required an a/ b/ prefix, so these resolved to nothing.
    "no-prefix": {"patch": "--- app.py\n+++ app.py\n" + _PATCH_BODY},
    "no-prefix-timestamps": {
        "patch": "--- app.py\t2020-01-01 00:00:00.000000000 +0900\n"
                 "+++ app.py\t2020-01-02 00:00:00.000000000 +0900\n" + _PATCH_BODY
    },
}

EDIT_FILE_SHAPES = {
    "normal": {
        "path": "app.py",
        "operations": [{"op": "replace", "anchor": "original = 1", "content": "rewritten = 2"}],
    },
    "raw-arguments": {
        "__raw_arguments": json.dumps({
            "path": "app.py",
            "operations": [
                {"op": "replace", "anchor": "original = 1", "content": "rewritten = 2"}
            ],
        })
    },
}

ALL_SHAPES = [
    ("write_plan", name, args) for name, args in WRITE_PLAN_SHAPES.items()
] + [
    ("apply_patch", name, args) for name, args in APPLY_PATCH_SHAPES.items()
] + [
    ("edit_file", name, args) for name, args in EDIT_FILE_SHAPES.items()
]


@pytest.mark.parametrize(
    "tool_name,shape,args", ALL_SHAPES, ids=[f"{t}-{s}" for t, s, _ in ALL_SHAPES]
)
def test_every_consumer_sees_the_file_the_write_touched(repo, tool_name, shape, args):
    """All four target consumers resolve app.py, and the write really changes it."""
    reg = ToolRegistry(repo_root=str(repo), config=AgentConfig())
    target = str(repo / "app.py")

    extracted = reg._extract_write_target_paths(tool_name, args)
    snapshots = WriteSafetyManager(str(repo)).snapshot_target_files(tool_name, args)
    locks = FileLockManager(str(repo)).acquire_relevant(dict(args), tool_name)

    # A deep copy so a handler that mutates its arguments cannot invalidate the
    # comparison above (write_plan normalises ops in place).
    result = reg.dispatch(tool_name, json.loads(json.dumps(args)))
    assert result.ok, result.error
    assert (repo / "app.py").read_text(encoding="utf-8").strip() == "rewritten = 2"

    assert extracted == frozenset({target}), f"{shape}: checkpoint/cache scope"
    assert set(snapshots) == {target}, f"{shape}: rollback snapshot"
    assert len(locks) == 1, f"{shape}: file lock ({locks})"


@pytest.mark.parametrize(
    "tool_name,shape,args", ALL_SHAPES, ids=[f"{t}-{s}" for t, s, _ in ALL_SHAPES]
)
def test_every_shape_leaves_an_undoable_checkpoint(repo, tool_name, shape, args):
    """Undo restores the pre-write content whichever shape the call arrived in."""
    reg = ToolRegistry(repo_root=str(repo), config=AgentConfig())
    assert reg.dispatch(tool_name, json.loads(json.dumps(args))).ok

    cid = reg.run_checkpoint_id
    assert cid is not None, f"{shape}: write produced no Undo point"
    assert CheckpointStore(str(repo)).restore(cid) is True
    assert (repo / "app.py").read_text(encoding="utf-8") == ORIGINAL


# ── the diff parser ─────────────────────────────────────────────────────────

def test_patch_parser_strips_the_diff_u_timestamp():
    targets = parse_patch_targets(
        "--- app.py\t2020-01-01 00:00:00\n+++ app.py\t2020-01-02 00:00:00\n"
        "@@ -1 +1 @@\n-a\n+b\n"
    )
    assert set(targets) == {"app.py"}


def test_patch_parser_ignores_dev_null():
    targets = parse_patch_targets(
        "--- /dev/null\n+++ b/new.py\n@@ -0,0 +1 @@\n+x = 1\n"
    )
    assert targets == ["new.py"]


def test_patch_parser_does_not_read_hunk_body_as_a_header():
    """A deleted ``-- x`` line renders as ``--- x``; a header it is not.

    This is the ambiguity the a/-prefix requirement used to dodge by refusing
    the no-prefix form entirely. The hunk line counts resolve it instead, so
    both shapes are handled without false positives.
    """
    patch = (
        "--- a/real.py\n"
        "+++ b/real.py\n"
        "@@ -1,3 +1,3 @@\n"
        " context\n"
        "--- not_a_header.py\n"
        "+++ also_not_a_header.py\n"
    )
    assert set(parse_patch_targets(patch)) == {"real.py"}


def test_patch_parser_reads_a_header_after_a_completed_hunk():
    """Multi-file diffs: the second file's header follows the first file's hunk."""
    patch = (
        "--- a/one.py\n+++ b/one.py\n@@ -1 +1 @@\n-a\n+b\n"
        "--- a/two.py\n+++ b/two.py\n@@ -1 +1 @@\n-c\n+d\n"
    )
    assert set(parse_patch_targets(patch)) == {"one.py", "two.py"}


def test_count_patch_files_sees_no_prefix_patches():
    """The approval gate counts a no-prefix multi-file patch it used to miss."""
    patch = "".join(
        f"--- {n}.py\n+++ {n}.py\n@@ -1 +1 @@\n-a\n+b\n" for n in ("a", "b", "c")
    )
    assert WriteSafetyManager.count_patch_files(patch) == 3


def test_no_approval_prompt_for_the_nonexistent_delete_file_tool(repo):
    """A hallucinated tool name must not raise a deletion prompt.

    ``_gate_check`` runs before dispatch resolves the handler, so the removed
    ``delete_file`` branch made the user approve "DELETE FILE: app.py" for a
    tool that does not exist — the call then failed with "Unknown tool" and the
    file was never touched. Overstating a no-op's danger trains users to click
    through the prompts that are real.
    """
    prompts: list = []
    config = AgentConfig()
    config.approval_callback = lambda tool, args, preview: (
        prompts.append((tool, preview)) or True
    )
    reg = ToolRegistry(repo_root=str(repo), config=config)

    result = reg.dispatch("delete_file", {"path": "app.py"})

    assert prompts == [], f"phantom approval prompt: {prompts}"
    assert result.ok is False
    assert "Unknown tool" in (result.error or "")
    assert (repo / "app.py").read_text(encoding="utf-8") == ORIGINAL


# ── robustness: the gates run on the write path and must never raise ────────

@pytest.mark.parametrize("plan", [
    {"kind": "ASICODE_PLAN_V1", "ops": ["create_file app.py"]},   # ops of strings
    {"kind": "ASICODE_PLAN_V1", "ops": [None]},
    {"kind": "ASICODE_PLAN_V1", "ops": "not-a-list"},
    42,
    None,
])
def test_malformed_plans_resolve_to_no_targets_instead_of_raising(plan):
    assert write_target_paths("write_plan", {"plan": plan}) == []


def test_malformed_ops_reach_the_handler_error_not_a_traceback(repo):
    """A bad op must produce the handler's guidance, not an AttributeError.

    Two of the former copies did ``op.get("path")`` unguarded, so the pre-write
    checkpoint gate raised out of dispatch and the model received a raw Python
    traceback in place of a message telling it how to fix the plan.
    """
    reg = ToolRegistry(repo_root=str(repo), config=AgentConfig())
    result = reg.dispatch(
        "write_plan", {"plan": {"kind": "ASICODE_PLAN_V1", "ops": ["oops"]}}
    )
    assert result.ok is False
    assert "AttributeError" not in (result.error or "")
    assert "not a JSON object" in (result.error or "")


def test_write_target_paths_never_raises_on_junk():
    for junk in (None, {}, {"plan": object()}, {"patch": 3}, {"file_path": 7}):
        assert write_target_paths("write_plan", junk) == []


# ── normalisation is shared with the handler ────────────────────────────────

def test_normalize_plan_lifts_every_accepted_shape_to_one_dict():
    expected_paths = ["app.py"]
    for args in WRITE_PLAN_SHAPES.values():
        if "__raw_arguments" in args:
            continue  # recovered a level up, by write_target_paths
        assert plan_target_paths(normalize_plan(args)) == expected_paths


def test_normalize_plan_returns_unparseable_text_fence_stripped():
    """The handler quotes this value back in its rejection, so it must be the
    fence-stripped text — and it is also what the diff parser should see when a
    model wraps a patch in a code fence."""
    assert normalize_plan({"plan": "```\nnot json\n```"}) == "not json"
