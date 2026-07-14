"""Regression tests for the safe single-line replace path contract parity.

``_try_deterministic_replace_block``'s "SAFE LINE-REPLACE MODE" (detected when
BEFORE is a single unique line) previously returned raw synthesizer output
directly to ``_finalize``, bypassing ``_clean_diff`` — the ONLY deterministic
synth path to do so. An external analysis flagged this as a "latent bug"
claiming it missed EOF-no-newline normalization. That claim is FALSE: EOF
marker normalization lives in the SYNTHESIZER (patch_synth.py, commit
d0dc24ef), which both paths call, so markers are identical either way. Fuzz
proves ``_clean_diff`` is IDEMPOTENT on deterministic synth output (268/268
byte-identical, 0 marker mismatch, 0 apply failures). The hardening is
therefore behavior-preserving: it restores the uniform contract "all synth
output is cleaned before finalize" purely to prevent silent sibling drift.

These tests lock (a) the idempotency property that makes the change safe, and
(b) the structural invariant that the safe path routes through ``_clean_diff``.
"""
from __future__ import annotations

import os
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from diff_apply import _clean_diff  # noqa: E402
from patch_synth import synthesize_replace_line_unified_diff  # noqa: E402


def _git_apply_ok(patch_text: str, cwd: str) -> bool:
    p = subprocess.run(
        ["git", "apply", "--check"], input=patch_text, cwd=cwd, text=True, capture_output=True
    )
    return p.returncode == 0


# ===========================================================================
# idempotency lock — the property that makes the hardening safe
# ===========================================================================
SAFE_MODE_CASES = [
    # (name, file_content, old_line, new_line)
    ("mid_with_nl", "line1\nline2\nline3\nline4\n", "line2", "LINE_TWO"),
    ("last_with_nl", "a\nb\nc\n", "c", "C"),
    ("last_no_nl_eof_marker", "a\nb\nc", "c", "C"),          # forces \ No newline
    ("single_line_no_nl", "only", "only", "ONLY"),            # forces \ No newline
    ("cjk_boundary", "x\n한글\nz\n", "한글", "KOREAN"),
    ("emoji", "a\n🎉\nb\n", "🎉", "PARTY"),
]


@pytest.mark.parametrize("name,content,old,new", SAFE_MODE_CASES)
def test_safe_mode_synth_is_clean_diff_idempotent(name, content, old, new, tmp_path):
    """The synth call used by the safe-single-line path must satisfy
    raw == _clean_diff(raw). If this ever breaks, the consistency hardening
    would become a real behavior change and must be re-evaluated."""
    d = str(tmp_path)
    rel = "sample.txt"
    with open(os.path.join(d, rel), "w") as f:
        f.write(content)
    # safe-single-line path uses context_lines=12, require_unique=True
    file_lines = content.splitlines(keepends=False)
    raw = synthesize_replace_line_unified_diff(
        d, rel, old, new, require_unique=True, context_lines=12, lines=file_lines
    )
    assert raw, f"[{name}] synth produced empty output"
    cleaned = _clean_diff(raw, d, file_path_hint=rel)
    assert raw == cleaned, (
        f"[{name}] _clean_diff is NOT idempotent on this synth output — "
        f"the consistency hardening would change behavior here.\nRAW:\n{raw}\nCLEANED:\n{cleaned}"
    )
    # both must apply cleanly and carry identical EOF markers
    assert _git_apply_ok(raw, d), f"[{name}] raw patch does not apply"
    assert ("\\ No newline" in raw) == ("\\ No newline" in cleaned)


def test_safe_mode_eof_marker_present_when_file_lacks_trailing_newline(tmp_path):
    """Directly disproves the 'missed EOF normalization' claim: the safe-mode
    synthesizer EMITS the \\ No newline marker itself (it is not _clean_diff's
    job), so the bypass path never actually missed it."""
    d = str(tmp_path)
    rel = "sample.txt"
    with open(os.path.join(d, rel), "w") as f:
        f.write("alpha\nbeta\ngamma")  # NO trailing newline
    raw = synthesize_replace_line_unified_diff(
        d, rel, "gamma", "GAMMA", require_unique=True, context_lines=12,
        lines=["alpha", "beta", "gamma"],
    )
    assert "\\ No newline at end of file" in raw, "synthesizer must emit EOF marker directly"


# ===========================================================================
# structural drift guard — safe path must route through _clean_diff
# ===========================================================================
def test_safe_single_line_path_routes_through_clean_diff():
    """Source-level guard scoped to the SAFE LINE-REPLACE MODE region:
    the safe-single-line branch must clean synth output before finalizing
    (contract parity with _try_deterministic_replace_line/range). Catches a
    future reversion that re-bypasses _clean_diff. Scoped to the safe-mode
    try-block so it does not pass via the downstream block-level finalize."""
    src_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "webapp", "llm_execution.py",
    )
    if not os.path.exists(src_path):
        pytest.skip("webapp/ not present (public CLI-only snapshot)")
    src = open(src_path).read()
    assert "def _try_deterministic_replace_block(" in src
    start = src.index("def _try_deterministic_replace_block(")
    nxt = src.find("\ndef ", start + 1)
    fn_body = src[start:nxt]
    # isolate the SAFE LINE-REPLACE MODE region: from its marker comment to the
    # `except` that logs "safe single-line replace mode failed"
    i = fn_body.index("# SAFE LINE-REPLACE MODE")
    j = fn_body.index("safe single-line replace mode failed")
    region = fn_body[i:j]
    assert "_clean_diff(det" in region, (
        "safe-single-line path must call _clean_diff on synth output before finalize"
    )
    assert "diff_patch=cleaned" in region, (
        "safe-single-line path must finalize with _clean_diff output, not raw synth"
    )
    assert "diff_patch_raw=det" in region, (
        "safe-single-line path should expose the raw synth via diff_patch_raw for parity"
    )
