"""Gate: pytest xdist distribution mode must stay ``loadgroup``.

P5 (2026-08-24): ``worksteal`` (P10's pick, 1.24x faster than ``load`` on
2026-08-20) STEALS ``xdist_group`` tests onto other workers — repo_scan's
5 whole-REPO scan files ran CONCURRENTLY across 8 workers (observed gw1..gw7
in one run), re-creating the .cache/structural_graph_v1.json read+rewrite
storm 64a0917e6 fixed: those tests measured 17-18s apiece vs 4.5-6.4s under
``loadgroup``. The P10 A/B predated repo_scan's xdist_group; a fair re-A/B
(2026-08-24, identical tree/command, Chrome closed, x2)::

    worksteal  293.8s / 302.4s   (median 298.1)
    loadgroup  276.7s / 313.7s   (median 295.2)   -> <1% total-time diff

``loadgroup`` keeps the same wall time while restoring group serialization,
so it strictly dominates.  ``load`` (round-robin) is NOT restored because
P10's top-heavy straggler measurement still stands for non-group tests.

The mode lives in ``[tool.pytest.ini_options].addopts``, which nothing
consumes structurally: reverting to ``worksteal``/``load`` (or silently
dropping the flag) would resurface only as slower CI / group-storm jitter —
easy to miss, hard to attribute.

Contract (definition <-> behaviour, same shape as test_version_drift_gate):

* R1  ``addopts`` parsed with ``tomllib`` (no regex on the toml side)
       contains ``--dist`` immediately followed by ``loadgroup`` as SEPARATE
       elements — pytest 9 passes each element to argv verbatim, and this
       file's convention keeps flag/value pairs split (see the "-n"/"auto"
       NOTE in pyproject.toml);
* R2  xdist is engaged: ``-n`` followed by a value — ``--dist`` is inert
       under ``-n 0`` serial runs, so the parallelism flag must co-exist;
* R3  the ``dev`` extra pins ``pytest-xdist >= 3.1`` (xdist_group landed in
       3.1; older xdist rejects the value at startup with a hard error).
"""

from __future__ import annotations

from pathlib import Path

import tomllib

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PYPROJECT = _REPO_ROOT / "pyproject.toml"


def _parse(pyproject: Path = _PYPROJECT) -> dict:
    with pyproject.open("rb") as fh:
        return tomllib.load(fh)


def _addopts(doc: dict) -> list[str]:
    return doc["tool"]["pytest"]["ini_options"]["addopts"]


def _flag_value(opts: list[str], flag: str) -> str | None:
    """Return the element right after ``flag`` (repo convention: split pairs)."""
    for i, tok in enumerate(opts):
        if tok == flag and i + 1 < len(opts):
            return opts[i + 1]
    return None


# --- R1: the real tree pins loadgroup ---------------------------------------


def test_addopts_pins_loadgroup_as_split_pair() -> None:
    opts = _addopts(_parse())
    assert "--dist" in opts, "addopts lost the --dist flag entirely"
    assert _flag_value(opts, "--dist") == "loadgroup", (
        f"--dist value = {_flag_value(opts, '--dist')!r}, expected 'loadgroup' "
        "(worksteal breaks xdist_group serialization; see module docstring)"
    )


# --- R2: dist mode only matters under xdist parallelism --------------------


def test_xdist_parallelism_flag_present() -> None:
    opts = _addopts(_parse())
    assert _flag_value(opts, "-n") is not None, "--dist=loadgroup is inert without -n; keep the parallelism flag"


# --- R3: the installed xdist must understand loadgroup ----------------------


def test_dev_extra_pins_loadgroup_capable_xdist() -> None:
    deps = _parse()["project"]["optional-dependencies"]["dev"]
    pin = next((d for d in deps if d.startswith("pytest-xdist")), "")
    major_minor = tuple(int(x) for x in pin.split(">=")[1].split(".")[:2])
    assert major_minor >= (3, 1), f"{pin!r} predates --dist=loadgroup/xdist_group (xdist 3.1.0)"


# --- vacuity guards: the extractor must catch real mutations ---------------


def test_extractor_rejects_missing_flag() -> None:
    assert _flag_value(["-n", "auto"], "--dist") is None


def test_extractor_rejects_wrong_value() -> None:
    assert _flag_value(["--dist", "load"], "--dist") == "load"


def test_extractor_rejects_fused_single_token() -> None:
    # "--dist=loadgroup" as ONE element is legal pytest but breaks this
    # repo's split-pair convention — the guard must not silently accept it.
    assert _flag_value(["--dist=worksteal"], "--dist") is None
