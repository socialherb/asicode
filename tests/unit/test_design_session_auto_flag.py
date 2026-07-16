"""Auto-continue turns must be tagged in the session so the chain of
self-driven steps can be reconstructed post-hoc (and later feed loop/repetition
detection). The `auto` flag is persisted only when True to keep manual turns lean.
"""
from external_llm.design_session import DesignSessionManager


def test_auto_true_is_persisted(tmp_path):
    mgr = DesignSessionManager(str(tmp_path))
    mgr.add_turn("s1", "user", "auto input", auto=True)
    mgr.add_turn("s1", "assistant", "auto resp", auto=True)
    turns = mgr.get_or_create("s1").turns
    assert turns[0].get("auto") is True
    assert turns[1].get("auto") is True


def test_auto_false_default_is_lean(tmp_path):
    """Manual turns must NOT carry an `auto` key (storage stays lean)."""
    mgr = DesignSessionManager(str(tmp_path))
    mgr.add_turn("s1", "user", "manual input")
    mgr.add_turn("s1", "assistant", "manual resp")
    turns = mgr.get_or_create("s1").turns
    assert "auto" not in turns[0]
    assert "auto" not in turns[1]


def test_auto_flag_round_trips_through_disk(tmp_path):
    mgr = DesignSessionManager(str(tmp_path))
    mgr.add_turn("s1", "user", "auto input", auto=True)
    mgr.add_turn("s1", "user", "manual input")
    # A fresh manager reading the same session dir must recover the flag.
    mgr2 = DesignSessionManager(str(tmp_path))
    turns = mgr2.get_or_create("s1").turns
    assert turns[0].get("auto") is True
    assert "auto" not in turns[1]
