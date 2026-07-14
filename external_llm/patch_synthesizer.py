"""
Patch Synthesizer

Converts all output modes to Unified Diff
"""
from __future__ import annotations

import difflib
import logging
from pathlib import Path

from .hybrid_parser import ParseResult
from .output_modes import OutputMode

logger = logging.getLogger(__name__)


class PatchSynthesizer:
    """Convert parse result to Unified Diff"""

    def __init__(self, repo_root: str):
        self.repo_root = Path(repo_root)

    def synthesize(
        self,
        parse_result: ParseResult,
        target_file: str,
    ) -> str:
        """
        Convert all modes to unified diff

        Args:
            parse_result: Parsing result
            target_file: Target file

        Returns:
            Unified diff string
        """

        if not parse_result.success:
            raise ValueError("Cannot synthesize from failed parse result")

        # NEEDS_DISAMBIGUATION etc.: success=True but mode may be None
        if parse_result.mode is None:
            raise ValueError("cannot_synthesize: mode is None (needs_disambiguation)")

        if parse_result.mode == OutputMode.UNIFIED_DIFF:
            #already diff form
            return str(parse_result.diff or "")

        elif parse_result.mode == OutputMode.ASICODE_BLOCK:
            return self._from_asicode_block(parse_result, target_file)

        elif parse_result.mode == OutputMode.TARGETED_BLOCK:
            return self._from_targeted_block(parse_result, target_file)

        elif parse_result.mode == OutputMode.FULL_FILE:
            return self._from_full_file(parse_result, target_file)

        else:
            raise ValueError(f"Unknown mode: {parse_result.mode}")

    def _from_asicode_block(self, result: ParseResult, target_file: str) -> str:
        """Convert ASICODE_BLOCK to unified diff"""

        file_path = self.repo_root / target_file
        old_content = file_path.read_text(encoding="utf-8", errors="ignore") if file_path.exists() else ""
        new_content = old_content

        #each block apply
        for block in (result.blocks or []):
            before = block["before"]
            after = block["after"]

            # Replace (sequential fallback)
            if before in new_content:
                new_content = new_content.replace(before, after, 1)
            else:
                # Fallback 1: match ignoring trailing whitespace, replacing by line window.
                # CAUTION: replacing a normalized needle in unnormalized new_content
                # silently becomes a no-op, losing the change. Must replace the actual
                # content range at the line-window level.
                _content_lines = new_content.split("\n")
                _before_norm_lines = [ln.rstrip() for ln in before.split("\n")]
                _n_bl = len(_before_norm_lines)
                _matched = False
                for _i in range(len(_content_lines) - _n_bl + 1):
                    _window = _content_lines[_i:_i + _n_bl]
                    if [ln.rstrip() for ln in _window] == _before_norm_lines:
                        new_content = "\n".join(
                            _content_lines[:_i]
                            + after.split("\n")
                            + _content_lines[_i + _n_bl:]
                        )
                        _matched = True
                        break

                # Fallback 2: indent-normalized matching — normalize each of before and
                # the content window to its own minimum indent before comparison. The
                # previous implementation clipped both to before's min_indent, so
                # different indentation would never match (effectively dead code).
                if not _matched:
                    _before_lines = before.split("\n")
                    _before_min = min(
                        (len(line) - len(line.lstrip()) for line in _before_lines if line.strip()),
                        default=0,
                    )
                    _before_canon = [
                        (line[_before_min:] if line.strip() else line)
                        for line in _before_lines
                    ]
                    for _i in range(len(_content_lines) - len(_before_lines) + 1):
                        _window = _content_lines[_i:_i + len(_before_lines)]
                        _win_min = min(
                            (len(line) - len(line.lstrip()) for line in _window if line.strip()),
                            default=0,
                        )
                        _window_canon = [
                            (line[_win_min:] if line.strip() else line)
                            for line in _window
                        ]
                        if _window_canon == _before_canon:
                            # Shift 'after' uniformly to the matched window's base indent
                            _shift = _win_min - _before_min
                            _after_lines = after.split("\n")
                            if _shift != 0:
                                _after_lines = []
                                for _l in after.split("\n"):
                                    if not _l.strip():
                                        _after_lines.append(_l)
                                    elif _shift > 0:
                                        _after_lines.append((" " * _shift) + _l)
                                    else:
                                        _cut = -_shift
                                        _after_lines.append(
                                            _l[_cut:] if len(_l) >= _cut else _l.lstrip()
                                        )
                            new_content = "\n".join(
                                _content_lines[:_i]
                                + _after_lines
                                + _content_lines[_i + len(_before_lines):]
                            )
                            _matched = True
                            break

                if not _matched:
                    logger.warning("BEFORE block not found in file")

        # Generate diff
        return self._generate_diff(
            old_content,
            new_content,
            target_file
        )

    def _from_targeted_block(self, result: ParseResult, target_file: str) -> str:
        """Convert TARGETED_BLOCK to unified diff"""

        file_path = self.repo_root / target_file
        if not file_path.exists():
            raise ValueError("target_file_missing_for_targeted_block")
        old_content = file_path.read_text(encoding="utf-8", errors="ignore")
        old_lines = old_content.split('\n')

        # Find insertion position
        insert_point = result.insert_point
        insert_index = -1

        if insert_point.startswith("line "):
            # "line 45" (1-based). Insert AFTER that line.
            line_num = int(insert_point.split()[1])
            insert_index = max(0, min(line_num, len(old_lines)))
        else:
            # Find pattern match
            for i, line in enumerate(old_lines):
                if insert_point in line:
                    insert_index = i + 1
                    break

        if insert_index == -1:
            raise ValueError(f"Insert point not found: {insert_point}")

        # Insert code
        new_lines = (
            old_lines[:insert_index] +
            result.code.split('\n') +
            old_lines[insert_index:]
        )

        new_content = '\n'.join(new_lines)

        return self._generate_diff(
            old_content,
            new_content,
            target_file
        )

    def _from_full_file(self, result: ParseResult, target_file: str) -> str:
        """Convert FULL_FILE to unified diff"""

        file_path = self.repo_root / target_file

        if file_path.exists():
            old_content = file_path.read_text(encoding="utf-8", errors="ignore")
        else:
            old_content = ""

        new_content = result.content

        return self._generate_diff(
            old_content,
            new_content,
            target_file
        )

    def _generate_diff(
        self,
        old_content: str,
        new_content: str,
        file_path: str,
    ) -> str:
        """Generate unified diff"""

        old_lines = old_content.split('\n')
        new_lines = new_content.split('\n')

        diff = difflib.unified_diff(
            old_lines,
            new_lines,
            fromfile=f"a/{file_path}",
            tofile=f"b/{file_path}",
            lineterm='',
        )

        diff_text = '\n'.join(diff)

        # Empty check
        if not diff_text or diff_text.strip() == "":
            logger.info("No changes detected (empty diff)")
            return ""

        # Add git-style header (optional for validity, but aids downstream cleaner/logging)
        header = f"diff --git a/{file_path} b/{file_path}"
        if not diff_text.startswith("diff --git "):
            diff_text = header + "\n" + diff_text

        #trailing newline ensure
        if not diff_text.endswith("\n"):
            diff_text += "\n"

        return diff_text
