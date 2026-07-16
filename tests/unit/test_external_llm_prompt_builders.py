"""
Regression tests for ExternalLLMService prompt builders.

These builders were mistakenly removed as "dead code" in f7f7312a despite
having live callers (see generate_patch lines 840-844).  These tests exist
to ensure the same regression cannot happen again — if a builder is ever
removed again, the corresponding test will fail first.
"""
from __future__ import annotations

from external_llm.service import ExternalLLMService


# ============================================================
# Builder content tests — verify each builder returns the
# expected key phrases that callers rely on.
# ============================================================


class TestBuildPatchOnlySystemPrompt:
    """_build_patch_only_system_prompt returns a diff-focused prompt."""

    def test_contains_noop_rule(self) -> None:
        prompt = ExternalLLMService._build_patch_only_system_prompt()
        assert "NOOP" in prompt, (
            "Callers (generate_patch mode='diff') rely on the NOOP output rule"
        )

    def test_contains_core_diff_instruction(self) -> None:
        prompt = ExternalLLMService._build_patch_only_system_prompt()
        assert "unified diff" in prompt.lower(), (
            "Mode 'diff' must instruct the model to produce a unified diff"
        )

    def test_contains_diff_format_requirement(self) -> None:
        prompt = ExternalLLMService._build_patch_only_system_prompt()
        assert "---" in prompt and "+++" in prompt, (
            "Diff format must mention file headers (+/- lines)"
        )

    def test_output_type(self) -> None:
        prompt = ExternalLLMService._build_patch_only_system_prompt()
        assert isinstance(prompt, str) and len(prompt) > 100


class TestBuildFileBlockOnlySystemPrompt:
    """_build_file_block_only_system_prompt returns a FILE-block prompt."""

    def test_contains_target_file(self) -> None:
        target = "some/deep/path.py"
        prompt = ExternalLLMService._build_file_block_only_system_prompt(target)
        assert target in prompt, (
            "Builder must embed the target file path so the model knows which file to output"
        )

    def test_contains_file_header(self) -> None:
        prompt = ExternalLLMService._build_file_block_only_system_prompt("foo.py")
        assert "FILE:" in prompt, (
            "Callers expect the FILE: header instruction in the prompt"
        )

    def test_contains_target_in_file_header(self) -> None:
        target = "bar/baz.py"
        prompt = ExternalLLMService._build_file_block_only_system_prompt(target)
        assert f"FILE: {target}" in prompt, (
            "FILE: header must reference the exact target path"
        )

    def test_dotfile_preserves_leading_dot(self) -> None:
        """Regression: lstrip("./") would mangle .gitignore → gitignore.
        normalize_rel_path_fast uses a while-loop, not character-set lstrip.
        """
        target = ".gitignore"
        prompt = ExternalLLMService._build_file_block_only_system_prompt(target)
        assert "FILE: .gitignore" in prompt, (
            "Dotfile's leading dot must survive normalization — "
            "the model would otherwise try to rewrite 'gitignore' instead of '.gitignore'"
        )

    def test_rejects_empty_target(self) -> None:
        # Should not crash with empty string
        prompt = ExternalLLMService._build_file_block_only_system_prompt("")
        assert isinstance(prompt, str) and len(prompt) > 50

    def test_output_type(self) -> None:
        prompt = ExternalLLMService._build_file_block_only_system_prompt("test.py")
        assert isinstance(prompt, str) and len(prompt) > 100


class TestBuildAutoSystemPrompt:
    """_build_auto_system_prompt returns the auto-mode prompt."""

    def test_contains_noop_rule(self) -> None:
        prompt = ExternalLLMService._build_auto_system_prompt()
        assert "NOOP" in prompt, (
            "Callers (generate_patch mode='auto') rely on the NOOP output rule"
        )

    def test_contains_diff_format_option(self) -> None:
        prompt = ExternalLLMService._build_auto_system_prompt()
        assert "unified diff" in prompt.lower(), (
            "Auto mode must offer unified diff as the preferred format"
        )

    def test_contains_file_block_option(self) -> None:
        prompt = ExternalLLMService._build_auto_system_prompt()
        assert "FILE:" in prompt, (
            "Auto mode must mention the FILE: block fallback format"
        )

    def test_output_type(self) -> None:
        prompt = ExternalLLMService._build_auto_system_prompt()
        assert isinstance(prompt, str) and len(prompt) > 100


# ============================================================
# Call-site tests — verify generate_patch selects the correct
# builder for each mode.  This prevents the "dead code" mistake:
# if a builder is no longer reachable from generate_patch, the
# builder itself may still exist but the integration is broken.
# ============================================================


class TestGeneratePatchRoutesToCorrectBuilder:
    """generate_patch must call the appropriate builder per mode."""

    def test_diff_mode_routes_to_patch_only(self, monkeypatch) -> None:
        """mode='diff' when system_prompt is None → _build_patch_only_system_prompt is called."""
        tracking = _install_builder_tracker(
            monkeypatch, "_build_patch_only_system_prompt",
        )
        _run_minimal_generate_patch(monkeypatch, output_mode="diff")
        assert tracking["called"], (
            "mode='diff' must call _build_patch_only_system_prompt; "
            "if this fails the builder is no longer reachable from generate_patch"
        )

    def test_full_file_mode_routes_to_file_block(self, monkeypatch) -> None:
        """mode='full_file' and target_file set → _build_file_block_only_system_prompt is called."""
        tracking = _install_builder_tracker(
            monkeypatch, "_build_file_block_only_system_prompt",
        )
        _run_minimal_generate_patch(monkeypatch, output_mode="full_file", target="any.py")
        assert tracking["called"], (
            "mode='full_file' must call _build_file_block_only_system_prompt; "
            "if this fails the builder is no longer reachable from generate_patch"
        )

    def test_auto_mode_routes_to_auto(self, monkeypatch) -> None:
        """mode='auto' when system_prompt is None → _build_auto_system_prompt is called."""
        tracking = _install_builder_tracker(
            monkeypatch, "_build_auto_system_prompt",
        )
        _run_minimal_generate_patch(monkeypatch, output_mode="auto")
        assert tracking["called"], (
            "mode='auto' must call _build_auto_system_prompt; "
            "if this fails the builder is no longer reachable from generate_patch"
        )


# ============================================================
# Helpers
# ============================================================


def _install_builder_tracker(monkeypatch, method_name: str) -> dict:
    """Monkey-patch a static builder method with a tracking wrapper.

    Returns a dict with a 'called' flag.
    """
    original = getattr(ExternalLLMService, method_name)
    tracker = {"called": False}

    @staticmethod
    def tracking(*args, **kwargs):
        tracker["called"] = True
        return original(*args, **kwargs)

    monkeypatch.setattr(ExternalLLMService, method_name, tracking)
    return tracker


def _run_minimal_generate_patch(
    monkeypatch,
    output_mode: str = "auto",
    target: str | None = None,
) -> None:
    """Run generate_patch with minimal mocked I/O so the routing path is exercised.

    We patch out the expensive/async operations (LLM call, file reads) and
    only verify that the correct system-prompt builder was selected.
    """
    import os
    import tempfile

    from external_llm.service import ExternalLLMService

    # Create temp repo-like structure.
    tmp = tempfile.mkdtemp()
    if target:
        tf_path = os.path.join(tmp, target)
        os.makedirs(os.path.dirname(tf_path), exist_ok=True)
        with open(tf_path, "w") as f:
            f.write("existing content")

    # Patch client so no real network call happens.
    class _FakeResponse:
        content = "NOOP"
        finish_reason = "stop"
        tokens_used = 10
        usage = {"total_tokens": 10}

    class _FakeClient:
        def chat(self, *args, **kwargs):
            return _FakeResponse()

    # Stub out all the expensive methods that generate_patch calls before the
    # LLM invocation, so the test stays fast and hermetic.
    monkeypatch.setattr(ExternalLLMService, "_is_trivial_edit_request", lambda *a: True)
    monkeypatch.setattr(ExternalLLMService, "_noop_precheck_for_literal_add", lambda *a: False)
    monkeypatch.setattr(ExternalLLMService, "_build_context_best_effort", lambda *a: ("", {}))
    monkeypatch.setattr(ExternalLLMService, "_read_target_file_focused_snippet_best_effort", lambda *a, **kw: "")
    monkeypatch.setattr(ExternalLLMService, "_read_target_file_snippet_best_effort", lambda *a, **kw: "")
    monkeypatch.setattr(ExternalLLMService, "_extract_identifier_needles", lambda *a: [])
    monkeypatch.setattr(ExternalLLMService, "_build_llm_context_v7_best_effort", lambda *a, **kw: ("", {}))
    monkeypatch.setattr(ExternalLLMService, "_build_llm_context_super_best_effort", lambda *a, **kw: ("", {}))
    monkeypatch.setattr(ExternalLLMService, "_extract_previous_failure_hint_best_effort", lambda *a, **kw: "")
    monkeypatch.setattr("external_llm.service.enhance_user_request", lambda *a, **kw: a[0] if a else "")

    # Instantiating ExternalLLMService normally would try to set up a provider.
    # We short-circuit by creating an instance and overriding client.
    svc = ExternalLLMService.__new__(ExternalLLMService)
    svc.provider = "test"
    svc.model = "test-model"
    svc.client = _FakeClient()
    svc.thinking_mode = False
    svc.reasoning_effort = None
    svc.reasoning_callback = None

    svc.generate_patch(
        repo_root=tmp,
        user_request="add a comment",
        target_file=target,
        extra_context=None,
        temperature=0.0,
        system_prompt=None,  # ← triggers builder selection
        output_mode=output_mode,
        max_tokens=100,
        context_variant="v7",
        progress_callback=None,
    )
