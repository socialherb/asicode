"""Tests for the net-new silent-except baseline-diff gate.

Mirrors the IO/analysis split + precision-parametrize + vacuous-trap pattern of
``test_ast_cache_mutation_guard.py``: the live invariant (no new handlers beyond
baseline) is inherently vacuously-passable if the scanner is weakened, so the
*precision* parametrize tests (dangerous vs safe handler shapes) are what
actually guard detection capability.
"""

import importlib.util
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "check_no_new_silent_except.py"
_spec = importlib.util.spec_from_file_location("check_no_new_silent_except", _SCRIPT)
g = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
_spec.loader.exec_module(g)  # type: ignore[union-attr]


# --- precision: pure-suppression bodies ARE flagged (≥1 key) -----------------
@pytest.mark.parametrize(
    "body",
    [
        "        pass",
        "        return None",
        "        return",
        "        continue",
        "        break",
    ],
    ids=["pass", "return-None", "bare-return", "continue", "break"],
)
def test_scan_flags_pure_suppression(body):
    src = "def f():\n    try:\n        x = 1\n    except Exception:\n" + body + "\n"
    keys = g._scan_source(src, "m.py")
    assert keys == ["m.py::<module>::f::0"], keys


# --- precision: observability-bearing bodies are NOT flagged (0 keys) ---------
@pytest.mark.parametrize(
    "body",
    [
        "        raise",
        "        logger.error('boom')",
        "        self.log.warning('boom')",
        "        logging.exception('boom')",
        "        print('boom')",
        "        warnings.warn('boom')",  # attr 'warn' ∈ _LOGGING_ATTRS
    ],
    ids=["raise", "logger.error", "self.log.warning", "logging.exception", "print", "warnings.warn"],
)
def test_scan_passes_observability(body):
    src = "def f():\n    try:\n        x = 1\n    except Exception:\n" + body + "\n"
    assert g._scan_source(src, "m.py") == []


# --- precision boundary: a *value* fallback is NOT pure suppression ----------
def test_scan_passes_value_fallback():
    # return <value> does something (a fallback), so it is not the black-hole
    # pattern; documented conservative boundary of the gate.
    src = "def f():\n    try:\n        x = 1\n    except Exception:\n        return self._default\n"
    assert g._scan_source(src, "m.py") == []


def test_scan_passes_side_effect_body():
    # assignment + return None: body has a non-noop stmt → not pure suppression.
    src = "def f():\n    try:\n        x = 1\n    except Exception:\n        self._bad = True\n        return None\n"
    assert g._scan_source(src, "m.py") == []


# --- key stability under line drift ------------------------------------------
def test_key_stable_under_line_drift():
    base = "def f():\n    try:\n        x = 1\n    except Exception:\n        pass\n"
    drifted = "def f():\n\n\n\n\n    try:\n        x = 1\n    except Exception:\n        pass\n"
    assert g._scan_source(base, "m.py") == g._scan_source(drifted, "m.py")


def test_ordinal_distinguishes_handlers_in_same_scope():
    # Two silent handlers in one function → ordinals 0 and 1 (stable, distinct).
    src = (
        "def f():\n"
        "    try:\n        x = 1\n    except Exception:\n        pass\n"
        "    try:\n        y = 2\n    except Exception:\n        pass\n"
    )
    assert g._scan_source(src, "m.py") == [
        "m.py::<module>::f::0",
        "m.py::<module>::f::1",
    ]


# --- net-new detection via main() exit code -----------------------------------
def test_main_returns_nonzero_on_net_new(tmp_path, monkeypatch):
    prod = tmp_path / "ext"
    prod.mkdir()
    (prod / "mod.py").write_text(
        "def a():\n    try:\n        pass\n    except Exception:\n        pass\n\n"
        "def b():\n    try:\n        pass\n    except Exception:\n        pass\n"
    )
    monkeypatch.setattr(g, "REPO", tmp_path)
    monkeypatch.setattr(g, "_SCAN_ROOTS", ("ext",))
    monkeypatch.setattr(g, "BASELINE", tmp_path / "b.txt")
    # Only handler 'a' is baselined; 'b' is NET-NEW.
    (tmp_path / "b.txt").write_text("ext/mod.py::<module>::a::0\n")
    assert g.main() == 1


def test_main_returns_zero_when_in_sync(tmp_path, monkeypatch):
    prod = tmp_path / "ext"
    prod.mkdir()
    (prod / "mod.py").write_text(
        "def a():\n    try:\n        pass\n    except Exception:\n        pass\n"
    )
    monkeypatch.setattr(g, "REPO", tmp_path)
    monkeypatch.setattr(g, "_SCAN_ROOTS", ("ext",))
    monkeypatch.setattr(g, "BASELINE", tmp_path / "b.txt")
    (tmp_path / "b.txt").write_text("ext/mod.py::<module>::a::0\n")
    assert g.main() == 0


# --- live repo invariant: current ⊆ baseline (drift guard) -------------------
# If a future commit adds a silent except without re-baselining, this fails —
# the intended regression signal.  NOTE: vacuously-passable if the scanner is
# weakened; the precision tests above guard detection capability.
def test_no_new_silent_except_beyond_baseline():
    current = g._get_current_keys()
    baseline = g._load_baseline()
    new = current - baseline
    assert not new, f"{len(new)} new silent-except handler(s) beyond baseline:\n" + "\n".join(sorted(new))
