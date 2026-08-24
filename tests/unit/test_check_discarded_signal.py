"""Tests for the net-new discarded-signal baseline-diff gate.

The live invariant (no new computed-and-dropped patterns beyond baseline) is
inherently vacuously-passable if the scanner is weakened, so the precision
parametrize tests (dangerous vs safe shapes) are what actually guard
detection capability — same pattern as test_check_no_new_silent_except.py.
"""

import importlib.util
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "check_discarded_signal.py"
_spec = importlib.util.spec_from_file_location("check_discarded_signal", _SCRIPT)
g = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
_spec.loader.exec_module(g)  # type: ignore[union-attr]


# ── precision: computed-then-dropped shapes ARE flagged (>=1 key) ──────────
@pytest.mark.parametrize(
    "body",
    [
        # direct pair at function top level
        '    x = parse_retry_after(headers)\n    raise LLMRateLimitError(f"retry after {x}s")',
        # pair nested inside an if guard
        '    if status == 429:\n        x = parse_retry_after(headers)\n        raise LLMRateLimitError(f"retry after {x}s")',
        # name interpolated twice in the message
        '    x = compute()\n    raise Err(f"a={x} b={x}")',
        # inside a try block
        '    try:\n        x = parse(h)\n        raise Err(f"bad {x}")\n    except ValueError:\n        pass',
    ],
    ids=["direct", "nested-if", "twice", "try-block"],
)
def test_scan_flags_computed_and_dropped(body):
    src = "def f():\n" + body + "\n"
    keys = g._scan_source(src, "m.py")
    assert len(keys) == 1, keys


# ── precision: safe shapes are NOT flagged (0 keys) ─────────────────────────
@pytest.mark.parametrize(
    "body",
    [
        # structured kwarg use → not dropped
        '    x = parse_retry_after(headers)\n    raise LLMRateLimitError(f"retry after {x}s", retry_after=x)',
        # positional use
        "    x = parse_retry_after(headers)\n    raise LLMRateLimitError(x)",
        # raise without a Call exc (re-raise)
        "    x = parse(h)\n    raise",
        # name not used in the raise at all
        '    x = parse(h)\n    raise Err("static message")',
        # non-Call assignment (constant)
        '    x = 5\n    raise Err(f"n={x}")',
        # tuple target
        '    x, y = parse(h)\n    raise Err(f"a={x}")',
        # statement between assignment and raise (not adjacent)
        '    x = parse(h)\n    log(x)\n    raise Err(f"a={x}")',
    ],
    ids=["kwarg", "positional", "bare-reraise", "unused-in-raise", "non-call", "tuple-target", "not-adjacent"],
)
def test_scan_ignores_safe_shapes(body):
    src = "def f():\n" + body + "\n"
    keys = g._scan_source(src, "m.py")
    assert keys == [], keys


def test_key_is_drift_stable_scope_qualname():
    src = 'class C:\n    def m(self):\n        x = parse(h)\n        raise Err(f"{x}")\n'
    keys = g._scan_source(src, "m.py")
    assert keys == ["m.py::<module>::C::m::0"], keys


def test_async_function_pair_flagged():
    src = 'async def f():\n    x = fetch()\n    raise Err(f"got {x}")' + "\n"
    keys = g._scan_source(src, "m.py")
    assert len(keys) == 1, keys


def test_module_level_pair_also_flagged():
    src = 'x = parse(h)\nraise Err(f"{x}")\n'
    keys = g._scan_source(src, "m.py")
    assert keys == ["m.py::<module>::0"], keys


# ── scan pruning: os.walk must not descend into skipped dirs (F-4) ──────────
def test_iter_repo_py_prunes_skipped_dirs_entirely(tmp_path, monkeypatch):
    """Full scan must skip SKIP_DIRS subtrees before traversal (rglob era
    walked everything and discarded afterwards; os.walk prunes in place).

    A planted skip dir with *.py files must never surface in the scan.
    """
    monkeypatch.setattr(g, "REPO", tmp_path)
    monkeypatch.setattr(g, "_SCAN_ROOTS", ("ext",))
    prod = tmp_path / "ext"
    prod.mkdir()
    (prod / "real.py").write_text("_d = 0\ndef f():\n    _d = 1\n", encoding="utf-8")
    for skipped in (".venv", "__pycache__", "build", "node_modules", ".git"):
        d = tmp_path / "ext" / skipped
        d.mkdir(parents=True, exist_ok=True)
        (d / "mod.py").write_text("_d = 0\ndef f():\n    _d = 1\n", encoding="utf-8")
    scanned = g._iter_repo_py()
    assert scanned == [tmp_path / "ext" / "real.py"], scanned
