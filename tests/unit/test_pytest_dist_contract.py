"""Gate: pytest xdist distribution mode must stay ``worksteal``.

P10 (2026-08-20): the unit suite is top-heavy — the stage3 spawned-REPL
cluster (~19 tests x 2-5s) plus 13.8s/12.6s/8.5s heavyweights
(.verify_artifacts/verify-durations-full-20260820-204549.txt) — so the
default round-robin ``load`` scheduler left worker stragglers. Interleaved
A/B x2 on 8 workers (identical tree/command) measured::

    load      199.1s / 202.1s
    worksteal 156.1s / 167.1s      -> 1.24x median

The mode lives in ``[tool.pytest.ini_options].addopts``, which nothing
consumes structurally: reverting to ``load`` (or silently dropping the flag)
would resurface only as slower CI — easy to miss, hard to attribute.

Contract (definition <-> behaviour, same shape as test_version_drift_gate):

* R1  ``addopts`` parsed with ``tomllib`` (no regex on the toml side)
       contains ``--dist`` immediately followed by ``worksteal`` as SEPARATE
       elements — pytest 9 passes each element to argv verbatim, and this
       file's convention keeps flag/value pairs split (see the "-n"/"auto"
       NOTE in pyproject.toml);
* R2  xdist is engaged: ``-n`` followed by a value — ``--dist`` is inert
       under ``-n 0`` serial runs, so the parallelism flag must co-exist;
* R3  the ``dev`` extra pins ``pytest-xdist >= 3.6`` (worksteal landed in
       3.5; older xdist rejects the value at startup with a hard error).
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


# --- R1: the real tree pins worksteal --------------------------------------


def test_addopts_pins_worksteal_as_split_pair() -> None:
    opts = _addopts(_parse())
    assert "--dist" in opts, "addopts lost the --dist flag entirely"
    assert _flag_value(opts, "--dist") == "worksteal", (
        f"--dist value = {_flag_value(opts, '--dist')!r}, expected 'worksteal' "
        "(measured 1.24x slower under round-robin load; see module docstring)"
    )


# --- R2: dist mode only matters under xdist parallelism --------------------


def test_xdist_parallelism_flag_present() -> None:
    opts = _addopts(_parse())
    assert _flag_value(opts, "-n") is not None, "--dist=worksteal is inert without -n; keep the parallelism flag"


# --- R3: the installed xdist must understand worksteal ---------------------


def test_dev_extra_pins_worksteal_capable_xdist() -> None:
    deps = _parse()["project"]["optional-dependencies"]["dev"]
    pin = next((d for d in deps if d.startswith("pytest-xdist")), "")
    major_minor = tuple(int(x) for x in pin.split(">=")[1].split(".")[:2])
    assert major_minor >= (3, 5), f"{pin!r} predates --dist=worksteal (xdist 3.5.0)"


# --- vacuity guards: the extractor must catch real mutations ---------------


def test_extractor_rejects_missing_flag() -> None:
    assert _flag_value(["-n", "auto"], "--dist") is None


def test_extractor_rejects_wrong_value() -> None:
    assert _flag_value(["--dist", "load"], "--dist") == "load"


def test_extractor_rejects_fused_single_token() -> None:
    # "--dist=worksteal" as ONE element is legal pytest but breaks this
    # repo's split-pair convention — the guard must not silently accept it.
    assert _flag_value(["--dist=worksteal"], "--dist") is None
