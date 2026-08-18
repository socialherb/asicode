"""Self-check for check_standalone_imports.py's environment-skip policy.

lint.yml has been red since v0.2.21 for two reasons this file pins down:

1. Lossy index materialization -- the tmp copy contained ONLY the scan
   roots (external_llm/, services/, webapp/, asi.py), but in-scope modules
   import root-level siblings (common.py, utils/, context_collector.py,
   path_security.py) that were never copied, so ``--index-only`` phantom-
   failed with ``No module named 'common'`` in CI while the working tree
   (where the siblings exist by construction) always passed.
2. Environment-dependent third-party absence -- the baseline-diff job
   installs only ruff, so ~50 modules failed on requests/rich even though
   their import ORDER is fine. Those are classified as env-skips now, not
   failures; dependency presence stays the unit-test job's contract.

Three-rules compliance (a gate test must be non-vacuous): the classifier
must skip PURE third-party absence, must NOT skip anything touching a
first-party module, and must NOT skip any other import error.
"""

import importlib.util
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "check_standalone_imports.py"
_spec = importlib.util.spec_from_file_location("check_standalone_imports", _SCRIPT)
g = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
_spec.loader.exec_module(g)  # type: ignore[union-attr]

_FP = {"common", "utils", "external_llm", "asi"}


class TestThirdPartySkipClassifier:
    def test_pure_third_party_missing_is_skipped(self):
        err = "\n".join([
            "Traceback (most recent call last):",
            '  File "<string>", line 1, in <module>',
            "ModuleNotFoundError: No module named 'requests'",
        ])
        assert g._third_party_only_missing(err, _FP) == {"requests"}

    def test_dotted_third_party_reduces_to_top_segment(self):
        err = "ModuleNotFoundError: No module named 'rich.tooltips'"
        assert g._third_party_only_missing(err, _FP) == {"rich"}

    def test_missing_first_party_top_level_is_real_failure(self):
        # 'common' IS shipped by the repo -> its absence is a real
        # standalone/import-order failure, never an environment artifact.
        err = "ModuleNotFoundError: No module named 'common'"
        assert g._third_party_only_missing(err, _FP) is None

    def test_missing_first_party_submodule_is_real_failure(self):
        err = "ModuleNotFoundError: No module named 'external_llm.agent.x'"
        assert g._third_party_only_missing(err, _FP) is None

    def test_mixed_first_and_third_party_is_real_failure(self):
        # Imports run top-down and the FIRST failure wins, so a first-party
        # miss anywhere in the chain must surface, never be masked by an
        # earlier/better-known third-party one.
        err = "\n".join([
            "ModuleNotFoundError: No module named 'requests'",
            "ModuleNotFoundError: No module named 'common'",
        ])
        assert g._third_party_only_missing(err, _FP) is None

    def test_cannot_import_name_is_real_failure(self):
        err = ("ImportError: cannot import name 'resolve_inside_repo' "
               "from 'path_security'")
        assert g._third_party_only_missing(err, _FP) is None

    def test_non_module_errors_are_real_failures(self):
        err = "AttributeError: module 'asi' has no attribute 'run_repl'"
        assert g._third_party_only_missing(err, _FP) is None


@pytest.mark.skipif(shutil.which("git") is None, reason="git required")
class TestIndexMaterializationIncludesSiblingModules:
    """End-to-end guard for the v0.2.21~24 phantom failures (Fix A)."""

    def test_tmp_copy_carries_root_siblings_and_resolves_imports(
            self, monkeypatch, tmp_path):
        repo = tmp_path / "repo"
        (repo / "external_llm").mkdir(parents=True)
        (repo / "common.py").write_text("NAME = 'common'\n", encoding="utf-8")
        (repo / "external_llm" / "__init__.py").write_text("", encoding="utf-8")
        (repo / "external_llm" / "svc.py").write_text(
            "from common import NAME\nVALUE = NAME\n", encoding="utf-8")
        subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
        subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
        monkeypatch.setattr(g, "REPO", repo)

        with tempfile.TemporaryDirectory(prefix="asi-policy-") as td:
            tmp = Path(td)
            mods = g._materialize_index(tmp)
            # scan scope unchanged: only external_llm is a scan root here
            assert mods == ["external_llm", "external_llm.svc"]
            # the root-level sibling WAS materialized next to it
            assert (tmp / "common.py").exists()
            proc = subprocess.run(
                [sys.executable, "-c",
                 "import external_llm.svc as m; assert m.VALUE == 'common'"],
                cwd=tmp, capture_output=True, check=False)
            assert proc.returncode == 0, proc.stderr.decode("utf-8", "replace")
