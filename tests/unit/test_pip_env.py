"""Tests for external_llm.pip_env — env-aware PEP 668 pip flag selection.

Shared by the import-package auto-installers (asi._pip_install,
browser_tools). The CLI-tool installer (dependency_checker) intentionally does
NOT use this (see pip_env module docstring).
"""
import sys

from external_llm import pip_env


def test_flags_in_venv(monkeypatch):
    """Inside a virtualenv → no extra flags (plain install works, --user forbidden)."""
    monkeypatch.setattr(sys, "prefix", "/venv")
    monkeypatch.setattr(sys, "base_prefix", "/usr")  # prefix != base_prefix → venv
    assert pip_env.pip_install_flags() == []


def test_flags_externally_managed(monkeypatch, tmp_path):
    """Not a venv + PEP 668 marker present → --user --break-system-packages."""
    monkeypatch.setattr(sys, "prefix", "/usr")
    monkeypatch.setattr(sys, "base_prefix", "/usr")
    (tmp_path / "EXTERNALLY-MANAGED").write_text("[externally-managed]\n")
    monkeypatch.setattr(
        pip_env.sysconfig, "get_path",
        lambda name: str(tmp_path) if name == "stdlib" else None,
    )
    assert pip_env.pip_install_flags() == ["--user", "--break-system-packages"]


def test_flags_normal_env(monkeypatch, tmp_path):
    """Not a venv + no PEP 668 marker → no extra flags."""
    monkeypatch.setattr(sys, "prefix", "/usr")
    monkeypatch.setattr(sys, "base_prefix", "/usr")
    monkeypatch.setattr(
        pip_env.sysconfig, "get_path",
        lambda name: str(tmp_path) if name == "stdlib" else None,
    )  # tmp_path has no EXTERNALLY-MANAGED marker
    assert pip_env.pip_install_flags() == []


def test_flags_stdlib_path_none(monkeypatch):
    """sysconfig.get_path returning None must not crash → no flags."""
    monkeypatch.setattr(sys, "prefix", "/usr")
    monkeypatch.setattr(sys, "base_prefix", "/usr")
    monkeypatch.setattr(pip_env.sysconfig, "get_path", lambda name: None)
    assert pip_env.pip_install_flags() == []


def test_ensure_user_site_appends_when_missing(monkeypatch, tmp_path):
    """A real, on-disk user-site dir absent from sys.path gets appended."""
    us = tmp_path / "usersite"
    us.mkdir()
    monkeypatch.setattr(pip_env, "sys", sys)  # keep real sys
    import site
    monkeypatch.setattr(site, "getusersitepackages", lambda: str(us))
    # Ensure it's not already present.
    if str(us) in sys.path:
        sys.path.remove(str(us))
    pip_env.ensure_user_site_importable()
    assert str(us) in sys.path
    sys.path.remove(str(us))  # cleanup


def test_ensure_user_site_noop_when_absent_dir(monkeypatch):
    """A user-site path that does not exist on disk is not added."""
    import site
    ghost = "/nonexistent/usersite/xyz"
    monkeypatch.setattr(site, "getusersitepackages", lambda: ghost)
    before = list(sys.path)
    pip_env.ensure_user_site_importable()
    assert ghost not in sys.path
    assert sys.path == before


def test_ensure_user_site_swallows_errors(monkeypatch):
    """getusersitepackages raising must not propagate."""
    import site
    monkeypatch.setattr(site, "getusersitepackages", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    pip_env.ensure_user_site_importable()  # must not raise
