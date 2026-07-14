"""Regression: lstrip("./") dotfile-mangle defect across path-normalization sites.

str.lstrip("./") strips a character SET {'.','/'}, not a prefix — so a dotfile's
leading dot was stripped (.gitignore -> gitignore).  In the patch-synthesis
WRITE path this reported a present dotfile as ``target_missing``; in dedup /
NOOP / related-files paths it broke matching.  Fixed at 17 sites by replacing
``.lstrip("./")`` with ``.lstrip("/").removeprefix("./").lstrip("/")`` (matches
the context_builder.py / go_provider.py precedent from c5d6c4de).

These tests pin the two highest-value entry points:
  * output_parser._norm_rel  — the only extracted normalization helper.
  * patch_engine WRITE path  — synthesizing a diff for a dotfile target.
The remaining 15 inline sites use the identical normalization expression,
so if _norm_rel is correct they are correct.
"""
import pytest

from external_llm.output_parser import EnhancedOutputParser
from external_llm.patch_engine import PatchEngine

# ── output_parser._norm_rel: the canonical normalization ────────────────────

class TestNormRelPreservesDotfiles:
    """_norm_rel must NOT strip a dotfile's leading dot.

    OLD lstrip("./") turned ".gitignore" into "gitignore" (char-set {'.','/'}
    consumed the leading dot).  removeprefix("./") matches the exact "./"
    prefix only.
    """

    @pytest.mark.parametrize("raw,expected", [
        # dotfiles preserved — the bug mangled every one of these
        (".gitignore", ".gitignore"),
        (".env.example", ".env.example"),
        (".config.py", ".config.py"),
        (".github/workflows/ci.yml", ".github/workflows/ci.yml"),
        # "./" prefix stripped, dotfile body preserved
        ("./.gitignore", ".gitignore"),
        ("./.vscode/settings.json", ".vscode/settings.json"),
        # leading "/" stripped, dotfile preserved
        ("/.gitignore", ".gitignore"),
        # git a/ b/ prefixes stripped, dotfile preserved
        ("a/.gitignore", ".gitignore"),
        ("b/.config.py", ".config.py"),
        # ordinary "./" / leading-slash normalization still works
        ("./src/foo.py", "src/foo.py"),
        ("/src/foo.py", "src/foo.py"),
        ("src/foo.py", "src/foo.py"),
        # whitespace trimmed too
        ("  ./.env  ", ".env"),
        # empty / None safe
        ("", ""),
    ])
    def test_norm_rel(self, raw, expected):
        assert EnhancedOutputParser._norm_rel(raw) == expected

    def test_leading_dot_survives(self):
        """The exact regression: the leading dot must NOT be consumed."""
        result = EnhancedOutputParser._norm_rel(".gitignore")
        assert result.startswith("."), f"dotfile mangled: {result!r}"


# ── patch_engine WRITE path: dotfile target not reported missing ─────────────

class TestPatchEngineDotfileTarget:
    """_try_synthesize_diff_from_file_blocks must locate a dotfile target.

    The existence check (patch_engine.py:599) runs BEFORE block parsing, so an
    empty ``llm_text`` isolates the normalization defect cleanly:
      OLD:  tgt_rel = '.gitignore'.lstrip('./') == 'gitignore'  -> target_missing
      NEW:  tgt_rel = '.gitignore' (preserved)                  -> no_file_block
    """

    @pytest.fixture
    def repo(self, tmp_path):
        (tmp_path / ".git").mkdir()
        return str(tmp_path)

    def test_dotfile_not_reported_missing(self, repo, tmp_path):
        (tmp_path / ".gitignore").write_text("old\n")
        engine = PatchEngine(repo)
        _diff, status = engine._try_synthesize_diff_from_file_blocks(
            repo, ".gitignore", ""
        )
        assert status != "target_missing", (
            f"dotfile target '.gitignore' was mangled to 'gitignore' "
            f"(status={status!r})"
        )

    def test_prefixed_dotfile_not_reported_missing(self, repo, tmp_path):
        """ './.env' must resolve to the same on-disk '.env' file."""
        (tmp_path / ".env").write_text("KEY=val\n")
        engine = PatchEngine(repo)
        _diff, status = engine._try_synthesize_diff_from_file_blocks(
            repo, "./.env", ""
        )
        assert status != "target_missing", (
            f"'./.env' did not resolve to '.env' (status={status!r})"
        )

    def test_nondotfile_still_works(self, repo, tmp_path):
        """Sanity: ordinary file normalization is unaffected by the fix."""
        (tmp_path / "foo.py").write_text("x = 1\n")
        engine = PatchEngine(repo)
        _diff, status = engine._try_synthesize_diff_from_file_blocks(
            repo, "./foo.py", ""
        )
        assert status != "target_missing"
