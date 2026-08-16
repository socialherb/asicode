"""Bulky cold namespaces must not tax the hot ones' persistence path.

Every ``write_namespace`` is a read-merge-atomic-write of the WHOLE file under
a machine-global flock. ``experience_store`` is 89 KB of a real 123 KB
strategy_state.json and is written only from the disabled PLANNER lane, while
``adaptive_hub`` (3 KB) is written by the live agent loop every few seconds —
so the live loop paid for the dead one on every flush, and concurrent sessions
serialised on a single lock while doing it.

Routing the bulky namespace to its own file is only safe if reads keep working
across the transition, which is what most of this file pins.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import pytest

from external_llm.editor.learning import strategy_state as ss


@pytest.fixture()
def state_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    main = tmp_path / "strategy_state.json"
    monkeypatch.setattr(ss, "_STRATEGY_STATE_PATH", str(main))
    ss._migrated.discard(str(main))
    ss._warned_corrupt.clear()
    yield tmp_path
    ss._migrated.discard(str(main))
    ss._warned_corrupt.clear()


def _main(state_dir: Path) -> dict:
    p = state_dir / "strategy_state.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def _sidecar(state_dir: Path) -> dict:
    p = state_dir / "experience_store.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


class TestRouting:
    def test_hot_namespace_stays_in_the_shared_file(self, state_dir: Path):
        ss.write_namespace("adaptive_hub", {"a": 1})
        assert _main(state_dir)["adaptive_hub"] == {"a": 1}
        assert not (state_dir / "experience_store.json").exists()

    def test_routed_namespace_goes_to_its_own_file(self, state_dir: Path):
        ss.write_namespace("experience_store", [{"r": 1}])
        assert _sidecar(state_dir)["experience_store"] == [{"r": 1}]
        assert "experience_store" not in _main(state_dir)

    def test_round_trip_through_the_public_api(self, state_dir: Path):
        ss.write_namespace("experience_store", [{"r": 1}])
        ss.write_namespace("adaptive_hub", {"a": 1})
        assert ss.read_namespace("experience_store") == [{"r": 1}]
        assert ss.read_namespace("adaptive_hub") == {"a": 1}

    def test_model_keyed_namespaces_route_by_family(self, state_dir: Path):
        """'weights/{model}' must not be mistaken for an unrouted key."""
        ss.write_namespace("weights/glm-5.2", {"w": 1})
        assert "weights/glm-5.2" in _main(state_dir)

    def test_explicit_path_disables_routing(self, tmp_path: Path):
        """Callers owning a private file (exploration run-store, fallback-score
        store, tests) expect every namespace to land in the file they named."""
        private = tmp_path / "private.json"
        ss.write_namespace("experience_store", [{"r": 1}], path=str(private))
        assert json.loads(private.read_text(encoding="utf-8"))["experience_store"] == [{"r": 1}]
        assert ss.read_namespace("experience_store", path=str(private)) == [{"r": 1}]


class TestMigration:
    """Existing installs have the bulky namespace inside the shared file."""

    @staticmethod
    def _seed_pre_split(state_dir: Path) -> None:
        (state_dir / "strategy_state.json").write_text(json.dumps({
            "experience_store": [{"r": i} for i in range(200)],
            "adaptive_hub": {"a": 1},
            "weights": {"w": 1},
        }), encoding="utf-8")

    def test_read_works_before_any_migration(self, state_dir: Path):
        """Read-through keeps the move invisible regardless of when it runs."""
        self._seed_pre_split(state_dir)
        assert len(ss.read_namespace("experience_store")) == 200

    def test_first_hot_write_moves_it_out(self, state_dir: Path):
        self._seed_pre_split(state_dir)
        before = (state_dir / "strategy_state.json").stat().st_size
        ss.write_namespace("adaptive_hub", {"a": 2})
        after = (state_dir / "strategy_state.json").stat().st_size
        assert "experience_store" not in _main(state_dir)
        assert len(_sidecar(state_dir)["experience_store"]) == 200
        assert after < before, "the shared file must actually shrink"

    def test_value_survives_the_move(self, state_dir: Path):
        self._seed_pre_split(state_dir)
        original = ss.read_namespace("experience_store")
        ss.write_namespace("adaptive_hub", {"a": 2})
        assert ss.read_namespace("experience_store") == original

    def test_unrouted_namespaces_are_untouched(self, state_dir: Path):
        self._seed_pre_split(state_dir)
        ss.write_namespace("adaptive_hub", {"a": 2})
        main = _main(state_dir)
        assert main["weights"] == {"w": 1}
        assert main["adaptive_hub"] == {"a": 2}

    def test_migration_runs_once_per_process(self, state_dir: Path, monkeypatch):
        self._seed_pre_split(state_dir)
        ss.write_namespace("adaptive_hub", {"a": 2})
        calls = []
        monkeypatch.setattr(ss, "atomic_write_json",
                            lambda *a, **k: calls.append(a[0]))
        ss._migrate_split_namespaces()
        assert calls == []

    def test_nothing_to_migrate_is_a_noop(self, state_dir: Path):
        (state_dir / "strategy_state.json").write_text(
            json.dumps({"adaptive_hub": {"a": 1}}), encoding="utf-8")
        ss.write_namespace("adaptive_hub", {"a": 2})
        assert not (state_dir / "experience_store.json").exists()

    def test_missing_file_is_a_noop(self, state_dir: Path):
        ss._migrate_split_namespaces()  # must not raise


class TestBatchWrite:
    def test_batch_splits_across_target_files(self, state_dir: Path):
        ss.batch_write_namespaces({
            "adaptive_hub": {"a": 1},
            "weights": {"w": 1},
            "experience_store": [{"r": 1}],
        })
        main = _main(state_dir)
        assert set(main) == {"adaptive_hub", "weights"}
        assert _sidecar(state_dir)["experience_store"] == [{"r": 1}]

    def test_batch_preserves_unlisted_keys(self, state_dir: Path):
        ss.write_namespace("policy", {"p": 1})
        ss.batch_write_namespaces({"adaptive_hub": {"a": 1}})
        assert _main(state_dir)["policy"] == {"p": 1}

    def test_batch_with_explicit_path_stays_single_file(self, tmp_path: Path):
        private = tmp_path / "private.json"
        ss.batch_write_namespaces(
            {"adaptive_hub": {"a": 1}, "experience_store": [{"r": 1}]},
            path=str(private),
        )
        data = json.loads(private.read_text(encoding="utf-8"))
        assert set(data) == {"adaptive_hub", "experience_store"}
        assert not (tmp_path / "experience_store.json").exists()


def test_each_file_gets_its_own_lock(state_dir: Path):
    """A shared lock would keep the contention the split exists to remove."""
    ss.write_namespace("adaptive_hub", {"a": 1})
    ss.write_namespace("experience_store", [{"r": 1}])
    assert (state_dir / "strategy_state.json.lock").exists()
    assert (state_dir / "experience_store.json.lock").exists()


def test_hot_file_stays_small_after_split(state_dir: Path):
    """The measurable outcome: the file a hub flush rewrites is small."""
    TestMigration._seed_pre_split(state_dir)
    ss.write_namespace("adaptive_hub", {"a": 2})
    hot = (state_dir / "strategy_state.json").stat().st_size
    cold = (state_dir / "experience_store.json").stat().st_size
    assert hot < cold / 10, f"hot file {hot} B is not decisively smaller than cold {cold} B"
    assert os.path.getsize(state_dir / "strategy_state.json") < 4096


class TestMigrationPreservation:
    """A re-run migration must never clobber a newer sidecar write.

    The crash window documented on ``_migrate_split_namespaces`` leaves the
    value in BOTH files; the sidecar is then the fresher one, so the move must
    keep it (presence wins) instead of overwriting it with the stale main copy.
    """

    def test_migration_keeps_newer_sidecar_value(self, state_dir: Path):
        (state_dir / "strategy_state.json").write_text(json.dumps({
            "experience_store": [{"r": i} for i in range(200)],
            "adaptive_hub": {"a": 1},
        }), encoding="utf-8")
        # A crashed migration already wrote the sidecar, and a later write
        # updated it with NEWER records than main still holds.
        (state_dir / "experience_store.json").write_text(json.dumps({
            "experience_store": [{"r": "newer"}],
        }), encoding="utf-8")
        ss.write_namespace("adaptive_hub", {"a": 2})  # triggers the migration
        assert _sidecar(state_dir)["experience_store"] == [{"r": "newer"}]
        assert "experience_store" not in _main(state_dir)

    def test_migration_heals_corrupt_sidecar_from_main_copy(self, state_dir: Path):
        """A corrupt sidecar cannot be merged; main's copy is the only readable
        one, so the move writes it and the atomic write heals the file."""
        (state_dir / "strategy_state.json").write_text(json.dumps({
            "experience_store": [{"r": i} for i in range(5)],
            "adaptive_hub": {"a": 1},
        }), encoding="utf-8")
        (state_dir / "experience_store.json").write_text("{not json", encoding="utf-8")
        ss.write_namespace("adaptive_hub", {"a": 2})
        assert _sidecar(state_dir)["experience_store"] == [{"r": i} for i in range(5)]
        assert "experience_store" not in _main(state_dir)


class TestCorruption:
    """A broken state file must never masquerade as empty state (A113)."""

    LOGGER = "external_llm.editor.learning.strategy_state"

    def test_corrupt_read_returns_none_with_warning(self, state_dir: Path, caplog):
        (state_dir / "strategy_state.json").write_text("{not json", encoding="utf-8")
        with caplog.at_level(logging.WARNING, logger=self.LOGGER):
            assert ss.read_namespace("adaptive_hub") is None
        assert any("corrupt JSON" in r.message and "strategy_state.json" in r.message
                   for r in caplog.records)

    def test_corrupt_write_is_refused_and_file_untouched(self, state_dir: Path, caplog):
        """A corrupt file is never overwritten: its other namespaces would be
        silently dropped by the merge, so writes refuse and surface it."""
        p = state_dir / "strategy_state.json"
        p.write_text("{not json", encoding="utf-8")
        with caplog.at_level(logging.WARNING, logger=self.LOGGER):
            assert ss.write_namespace("adaptive_hub", {"a": 1}) is False
        assert p.read_text(encoding="utf-8") == "{not json"
        assert any("corrupt JSON" in r.message for r in caplog.records)

    def test_corrupt_warning_fires_once_per_path(self, state_dir: Path, caplog):
        p = state_dir / "strategy_state.json"
        p.write_text("{not json", encoding="utf-8")
        with caplog.at_level(logging.WARNING, logger=self.LOGGER):
            ss.read_namespace("adaptive_hub")
            ss.read_namespace("adaptive_hub")
            ss.write_namespace("adaptive_hub", {"a": 1})
        assert sum("corrupt JSON" in r.message for r in caplog.records) == 1

    def test_batch_aborts_on_corrupt_target(self, state_dir: Path, caplog):
        p = state_dir / "strategy_state.json"
        p.write_text("{not json", encoding="utf-8")
        with caplog.at_level(logging.WARNING, logger=self.LOGGER):
            assert ss.batch_write_namespaces({"adaptive_hub": {"a": 1}}) is False
        assert p.read_text(encoding="utf-8") == "{not json"
        assert any("corrupt JSON" in r.message for r in caplog.records)


# ── RED→GREEN: uncovered branches ────────────────────────────────────────────


def test_get_path_returns_configured_path(state_dir: Path):
    """get_path() reflects the redirected state file (L70)."""
    assert ss.get_path() == str(state_dir / "strategy_state.json")


def test_read_top_level_non_object_warns_corrupt(state_dir: Path, caplog):
    """A JSON array at the top level is corruption, not data (L124)."""
    (state_dir / "strategy_state.json").write_text("[1, 2, 3]", encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger="external_llm.editor.learning.strategy_state"):
        assert ss.read_namespace("adaptive_hub") is None
    assert any("corrupt" in r.message for r in caplog.records)


def test_read_routed_namespace_falls_back_to_consolidated(state_dir: Path):
    """Pre-split install: a routed namespace still readable from the shared
    file when its sidecar does not exist yet (L186-187)."""
    (state_dir / "strategy_state.json").write_text(
        json.dumps({"experience_store": [{"r": 1}]}), encoding="utf-8"
    )
    assert ss.read_namespace("experience_store") == [{"r": 1}]


def test_read_other_exception_returns_none(state_dir: Path, monkeypatch):
    """Non-JSON read failures are swallowed into None (L220-222)."""
    def _boom(_path):
        raise RuntimeError("io exploded")

    monkeypatch.setattr(ss, "_read_json_dict", _boom)
    assert ss.read_namespace("adaptive_hub") is None


def test_write_other_exception_returns_false(state_dir: Path, monkeypatch):
    """Non-JSON write failures return False (L252-254)."""
    def _boom(*_a, **_k):
        raise RuntimeError("lock exploded")

    monkeypatch.setattr(ss, "write_namespace_json", _boom)
    assert ss.write_namespace("adaptive_hub", {"a": 1}) is False


def test_batch_write_other_exception_returns_false(state_dir: Path, monkeypatch):
    """Non-JSON batch write failures return False (L303-305)."""
    def _boom(*_a, **_k):
        raise RuntimeError("write exploded")

    monkeypatch.setattr(ss, "atomic_write_json", _boom)
    assert ss.batch_write_namespaces({"adaptive_hub": {"a": 1}}) is False


def test_migration_failure_is_logged_and_swallowed(state_dir: Path, monkeypatch, caplog):
    """A failed split migration degrades gracefully: the write continues and
    the failure is logged, never propagated (L186-187)."""
    (state_dir / "strategy_state.json").write_text(
        json.dumps({"experience_store": [{"r": 1}], "adaptive_hub": {"a": 1}}),
        encoding="utf-8",
    )

    def _boom(*_a, **_k):
        raise RuntimeError("migration exploded")

    monkeypatch.setattr(ss, "atomic_write_json", _boom)
    with caplog.at_level(logging.DEBUG, logger="external_llm.editor.learning.strategy_state"):
        assert ss.write_namespace("adaptive_hub", {"a": 2}) is True
    assert any("split migration failed" in r.message for r in caplog.records)
