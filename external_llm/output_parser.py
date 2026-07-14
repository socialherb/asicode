# external_llm/output_parser.py
"""
Enhanced LLM Output Parser for asicode.

- Extract fenced/unfenced unified diffs
- Clean markdown/HTML artifacts
- Auto-correct common mistakes (---/+++ prefixes, missing diff --git headers)
- Validate structure and (optionally) ensure it targets a specific file
- Extract full-file rewrite blocks (Cursor-like) for auto-mode synthesis in service layer

IMPORTANT:
- external_llm.service expects module-level functions:
  - parse_llm_output(llm_output) -> dict OR (explanation, diff) depending on revision
  - validate_diff(diff, target_file=None) -> (ok, error_message_str)
- external_llm.__init__ expects module-level function:
  - extract_diff(llm_output) -> diff
"""
from __future__ import annotations

import logging
import re
from typing import Any, Optional

from common import normalize_rel_path_fast

logger = logging.getLogger(__name__)


class EnhancedOutputParser:
    # fenced blocks: ```diff ... ```
    DIFF_FENCE_RE = re.compile(
        r"```(?:diff|patch|unified|text)?\s*\n(.*?)\n```",
        re.DOTALL | re.IGNORECASE,
    )

    # unified hunk header
    HUNK_HEADER_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
    # file headers (used to detect orphan hunks)
    _FILE_HEADER_PREFIXES = ("diff --git", "--- ", "+++ ")

    # Cursor-like full rewrite blocks (support a few common labels)
    # Supports fenced OR unfenced bodies.
    # Example:
    #   FILE: path/to/file.py
    #   ```python
    #   <full content>
    #   ```
    # or
    #   FILE: path/to/file.py
    #   <full content ...>
    #   (until next FILE:/Path:/Target file: or EOF)
    #
    # NOTE: must NOT use DOTALL ('s'). The path group uses [^\n\r]+? and code1
    # uses [\s\S]*?, both of which are newline-safe regardless of flags. But the
    # unfenced code2 branch uses '.*' — under DOTALL '.' crosses newlines, so a
    # single repetition of (?:(?!^...FILE...).*\r?\n)* gobbles many physical
    # lines and the per-line (?!^...FILE...) negative lookahead is evaluated
    # only once per gobble. That let code2 swallow all subsequent FILE blocks
    # into the first one (silent data loss: later files "disappeared"). Keeping
    # '.*' line-bounded via 'im' (no 's') restores the per-line stop guard.
    FILE_BLOCK_RE = re.compile(
        r"(?im)"
        r"(?:^|\n)\s*(?:FILE|Path|Target file)\s*:\s*(?P<path>[^\n\r]+?)\s*\r?\n"
        r"(?:```[^\n\r]*\r?\n(?P<code1>[\s\S]*?)\r?\n```|"
        r"(?P<code2>(?:(?!^\s*(?:FILE|Path|Target file)\s*:).*\r?\n)*))?"
    )

    def extract_diff(self, llm_output: str) -> str:
        if not llm_output:
            logger.debug("LLM output is empty")
            return ""

        # Log first 500 chars of LLM output for debugging
        logger.debug(f"LLM output (first 500 chars): {llm_output[:500]}")

        diff = self._extract_from_fences(llm_output)
        if not diff:
            diff = self._extract_unfenced_diff(llm_output)
        if not diff:
            logger.warning("No diff found in LLM output")
            return ""

        diff = self._clean_diff(diff)
        diff = self._auto_correct_common_mistakes(diff)
        # Drop orphan hunks that appear without file headers (common truncation artifact).
        diff = self._drop_orphan_hunks(diff)
        # Fix common LLM mistake: hunk body lines missing leading " ", "+", "-" prefixes.
        diff = self._fix_hunk_body_prefixes(diff)

        if not self._has_valid_structure(diff):
            logger.warning("Extracted diff has invalid structure")
            return ""

        # Hard sanity: many LLM diffs are "structurally" valid but hunks are truncated/incomplete.
        # If hunk counts don't match the @@ header, git apply will fail (corrupt patch).
        if not self._hunks_have_consistent_line_counts(diff):
            logger.warning("Extracted diff has inconsistent hunk line counts (likely truncated)")
            return ""

        return diff

    # -----------------------------
    # Diff extraction
    # -----------------------------

    def _extract_from_fences(self, text: str) -> str:
        matches = self.DIFF_FENCE_RE.findall(text or "")
        if not matches:
            return ""
        if len(matches) > 1:
            logger.info("Found %d fenced blocks, combining", len(matches))
            return "\n\n".join(matches)
        return matches[0]

    def _extract_unfenced_diff(self, text: str) -> str:
        """
        Best-effort extraction when the model output is plain text diff without fences.

        Heuristic:
        - Start when we see typical diff starters
        - Keep lines that look diff-like
        - Stop after a few consecutive non-diff lines once in diff
        """
        lines = (text or "").split("\n")
        diff_lines: list[str] = []
        in_diff = False
        # We used to stop after a few non-diff lines, but many LLMs insert harmless
        # blank/commentary and then continue the diff, which caused truncation.
        # Now we keep collecting until the end, but only retain diff-like lines.
        saw_any_hunk = False

        for line in lines:
            if self._is_diff_starter(line):
                in_diff = True
                diff_lines.append(line)
                if line.startswith("@@ "):
                    saw_any_hunk = True
                continue

            if not in_diff:
                continue

            if self._is_diff_body_line(line):
                diff_lines.append(line)
                if line.startswith("@@ "):
                    saw_any_hunk = True
                continue

            # tolerate some blank lines inside diff
            if not line.strip():
                diff_lines.append(line)
                continue

            # Non-diff line while in diff:
            # - If we already saw a hunk, treat it as "likely explanation" and stop.
            # - If we haven't even seen a hunk yet, keep scanning (LLM sometimes prefaces headers).
            if saw_any_hunk:
                break

        return "\n".join(diff_lines) if diff_lines else ""

    @staticmethod
    def _is_diff_starter(line: str) -> bool:
        if not line:
            return False
        return line.startswith(("diff --git", "--- ", "+++ ", "@@ ", "index "))

    @staticmethod
    def _is_diff_body_line(line: str) -> bool:
        if line is None:
            return False
        if line.startswith(("diff --git", "--- ", "+++ ", "@@ ", "index ")):
            return True
        # diff body lines may start with +, -, space, or "\"
        if line[:1] in {"+", "-", " ", "\\"}:
            return True
        return False

    # -----------------------------
    # Cleaning & autocorrect
    # -----------------------------

    @staticmethod
    def _clean_diff(diff: str) -> str:
        if not diff:
            return ""
        d = str(diff)
        # strip accidental fence tokens that survived
        d = d.replace("```diff", "").replace("```patch", "").replace("```", "")
        # HTML entities
        d = d.replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&")
        d = d.strip()
        if d and not d.endswith("\n"):
            d += "\n"
        return d

    def _auto_correct_common_mistakes(self, diff: str) -> str:
        """
        Fix common issues:
        - missing a/ b/ prefixes for ---/+++
        - +++ accidentally uses a/ prefix
        - missing diff --git headers (inject based on ---/+++)
        """
        if not diff:
            return ""

        raw_lines = diff.split("\n")
        corrected: list[str] = []

        # pass1: normalize ---/+++ a/ b/
        for line in raw_lines:
            if line.startswith("---"):
                parts = line.split(maxsplit=1)
                path = parts[1].strip() if len(parts) > 1 else ""
                if path and path != "/dev/null" and not (path.startswith("a/") or path.startswith("b/")):
                    corrected.append(f"--- a/{path}")
                else:
                    corrected.append(line)
                continue

            if line.startswith("+++"):
                parts = line.split(maxsplit=1)
                path = parts[1].strip() if len(parts) > 1 else ""
                if path and path != "/dev/null" and not (path.startswith("a/") or path.startswith("b/")):
                    corrected.append(f"+++ b/{path}")
                else:
                    if path.startswith("a/") and path != "/dev/null":
                        corrected.append("+++ b/" + path[2:])
                    else:
                        corrected.append(line)
                continue

            if line.startswith("@@") and not self.HUNK_HEADER_RE.match(line):
                # Keep line, but log
                logger.warning("Invalid hunk header: %s", line)

            corrected.append(line)

        # pass2: inject diff --git headers if missing
        if any(_item_.startswith("diff --git") for _item_ in corrected):
            return "\n".join(corrected)

        def _norm(p: str) -> str:
            p = (p or "").strip()
            if p.startswith("a/") or p.startswith("b/"):
                p = p[2:]
            return p

        injected: list[str] = []
        i = 0
        n = len(corrected)

        while i < n:
            line = corrected[i]
            if line.startswith("---"):
                old_path = line.split(maxsplit=1)[1].strip() if len(line.split(maxsplit=1)) > 1 else ""
                new_path = ""
                if i + 1 < n and corrected[i + 1].startswith("+++"):
                    new_path = (
                        corrected[i + 1].split(maxsplit=1)[1].strip()
                        if len(corrected[i + 1].split(maxsplit=1)) > 1
                        else ""
                    )

                if old_path == "/dev/null":
                    p = _norm(new_path) if new_path and new_path != "/dev/null" else "unknown"
                    a_path = p
                    b_path = p
                elif new_path == "/dev/null":
                    p = _norm(old_path) if old_path and old_path != "/dev/null" else "unknown"
                    a_path = p
                    b_path = p
                else:
                    a_path = _norm(old_path) if old_path else "unknown"
                    b_path = _norm(new_path) if new_path else a_path

                injected.append(f"diff --git a/{a_path} b/{b_path}")
                injected.append(line)
                i += 1
                continue

            injected.append(line)
            i += 1

        return "\n".join(injected)

    def _drop_orphan_hunks(self, diff: str) -> str:
        """
        Relaxed orphan-hunk handling.

        Rationale:
          - This parser runs before the service-layer normalization/repair.
          - The service has stronger, target-aware repairs (e.g., injecting ---/+++ before @@),
            and a hard guardrail (`git apply --check`).
          - Dropping hunks here can remove real changes when the model emits hunks before headers
            or interleaves small non-diff artifacts.

        So we keep hunks and rely on downstream repair + git-apply validation.
        """
        if not diff:
            return ""

        txt = str(diff).replace("\r\n", "\n")
        lines = txt.split("\n")

        # If there are no recognizable file headers at all, keep as-is and let downstream decide.
        has_any_header = any(_item_.startswith(("diff --git", "--- ", "+++ ")) for _item_ in lines)
        if not has_any_header:
            kept = txt.strip()
            if kept and not kept.endswith("\n"):
                kept += "\n"
            return kept

        # Even if hunks appear before headers, keep them; service-layer normalization can repair.
        kept = txt.strip()
        if kept and not kept.endswith("\n"):
            kept += "\n"
        return kept


    def _fix_hunk_body_prefixes(self, diff: str) -> str:
        """
        LLMs sometimes emit unified diff hunks without the required per-line prefix.
        Inside a hunk, each line MUST start with: ' ', '+', '-', or '\\'.
        If a line doesn't, we treat it as a context line and prefix a single space.

        This repair is intentionally conservative:
        - Only applies after we've entered a hunk (after '@@ ... @@')
        - Stops when we hit a new file header ('diff --git', '--- ', '+++ ') or a new hunk header.
        """
        if not diff:
            return ""

        lines = diff.replace("\r\n", "\n").split("\n")
        # Drop the single trailing empty token produced by the final newline.
        # It is the terminator of the last body line, NOT an empty context line,
        # so converting it to a " " context line below would inject a phantom
        # line that overflows the @@ header counts and gets the whole diff
        # rejected by _hunks_have_consistent_line_counts.  (A genuine empty
        # context line mid-hunk keeps its own token and is preserved.)
        if lines and lines[-1] == "":
            lines.pop()
        out: list[str] = []

        in_hunk = False
        for line in lines:
            if line.startswith(("diff --git", "--- ", "+++ ")):
                in_hunk = False
                out.append(line)
                continue

            if line.startswith("@@"):
                in_hunk = True
                out.append(line)
                continue

            if not in_hunk:
                out.append(line)
                continue

            # in hunk: every line must start with ' ', '+', '-', or '\\'
            if line == "":
                # represent an empty context line as a single-space-prefixed line
                out.append(" ")
                continue

            if line[:1] in {" ", "+", "-", "\\"}:
                out.append(line)
                continue

            # Missing prefix -> assume context line
            out.append(" " + line)

        fixed = "\n".join(out)
        if fixed and not fixed.endswith("\n"):
            fixed += "\n"
        return fixed

    def _hunks_have_consistent_line_counts(self, diff: str) -> bool:
        """
        Ensure each hunk body matches the @@ header counts.

        Many LLM outputs include a correct-looking @@ header but omit/truncate body lines,
        which causes `git apply` to fail with "corrupt patch".

        Returns True if all hunks match their declared old/new line counts.
        """
        if not diff:
            return False

        lines = diff.replace("\r\n", "\n").split("\n")
        # The final newline yields a trailing empty token that is the last body
        # line's terminator, not a body line — counting it as a context line
        # inflates cur_old/cur_new past the @@ header and falsely rejects an
        # otherwise valid diff.  Drop exactly one (a real mid-hunk empty context
        # line keeps its token).
        if lines and lines[-1] == "":
            lines.pop()

        in_hunk = False
        exp_old = 0
        exp_new = 0
        saw_any_hunk = False

        cur_old = 0
        cur_new = 0

        def _finish_hunk() -> bool:
            nonlocal in_hunk, exp_old, exp_new, cur_old, cur_new, saw_any_hunk
            if not in_hunk:
                return True
            saw_any_hunk = True
            # Allow truncation (cur <= exp) but not excess (cur > exp)
            # Also require at least one line unless exp is 0 (unlikely)
            ok = (cur_old <= exp_old) and (cur_new <= exp_new) and (cur_old > 0 or cur_new > 0 or exp_old == 0 or exp_new == 0)
            in_hunk = False
            exp_old = exp_new = 0
            cur_old = cur_new = 0
            return ok

        for line in lines:
            if line.startswith(("diff --git", "--- ", "+++ ")):
                if not _finish_hunk():
                    return False
                continue

            if line.startswith("@@"):
                if not _finish_hunk():
                    return False
                m = self.HUNK_HEADER_RE.match(line)
                if not m:
                    return False
                exp_old = int(m.group(2) or 1)
                exp_new = int(m.group(4) or 1)
                cur_old = 0
                cur_new = 0
                in_hunk = True
                continue

            if not in_hunk:
                continue

            # inside hunk: count lines by prefix
            if not line:
                # empty should have been normalized to " " by _fix_hunk_body_prefixes
                # treat as context line anyway
                cur_old += 1
                cur_new += 1
                continue

            pfx = line[:1]
            if pfx == " ":
                cur_old += 1
                cur_new += 1
            elif pfx == "-":
                cur_old += 1
            elif pfx == "+":
                cur_new += 1
            elif pfx == "\\":
                # "\ No newline at end of file" does not count toward lines
                pass
            else:
                # invalid; should have been fixed already
                return False

        if not _finish_hunk():
            return False

        return saw_any_hunk

    @staticmethod
    def _has_valid_structure(diff: str) -> bool:
        if not diff:
            return False
        lines = diff.split("\n")
        has_minus = any(_item_.startswith("---") for _item_ in lines)
        has_plus = any(_item_.startswith("+++") for _item_ in lines)
        has_hunk = any(_item_.startswith("@@") for _item_ in lines)
        return has_minus and has_plus and has_hunk

    # -----------------------------
    # Explanation + patch parse
    # -----------------------------

    def parse_llm_response(self, llm_output: str) -> tuple[str, str]:
        """
        Returns: (explanation, diff)

        Explanation is "everything except diff blocks" (best-effort).
        """
        diff = self.extract_diff(llm_output)

        explanation = str(llm_output or "")
        explanation = self.DIFF_FENCE_RE.sub("", explanation)
        if diff and diff in explanation:
            explanation = explanation.replace(diff, "")

        return explanation.strip(), diff

    def parse_llm_output_dict(self, llm_output: str) -> dict[str, str]:
        """
        Newer service code often expects dict: {"patch": ..., "explanation": ...}
        """
        expl, diff = self.parse_llm_response(llm_output)
        return {"explanation": expl, "patch": diff}

    # -----------------------------
    # Validation
    # -----------------------------

    @staticmethod
    def _norm_rel(p: str) -> str:
        p = (p or "").strip()
        if p.startswith("a/") or p.startswith("b/"):
            p = p[2:]
        p = normalize_rel_path_fast(p)
        return p

    def validate_diff(
        self,
        diff: str,
        target_file: Optional[str] = None,
        require_additions: bool = False,
    ) -> tuple[bool, str]:
        """
        Validate diff structure.

        Args:
            diff: unified diff text
            target_file: restrict to single file
            require_additions: if True, ensure at least one real '+' line exists
                               (used for CREATE enforcement)

        Returns:
            (is_valid, error_message)
        """
        if not diff or not str(diff).strip():
            return False, "empty_patch"

        # 🔥 CREATE enforcement: must contain at least one real added line
        if require_additions:
            has_added_line = any(
                line.startswith("+") and not line.startswith("+++")
                for line in str(diff).splitlines()
            )
            if not has_added_line:
                return False, "no_added_lines"

        lines = str(diff).replace("\r\n", "\n").split("\n")
        has_file_headers = False
        has_hunks = False
        found_files: list[str] = []

        in_hunk = False

        for line in lines:
            if line.startswith("diff --git"):
                in_hunk = False
                # record b/<path>
                m = re.search(r"\sb/(.+?)(?:\s|$)", line)
                if m:
                    found_files.append(self._norm_rel(m.group(1)))
                continue

            if line.startswith("---") or line.startswith("+++"):
                in_hunk = False
                has_file_headers = True
                parts = line.split(maxsplit=1)
                if len(parts) > 1 and parts[1].strip() != "/dev/null":
                    found_files.append(self._norm_rel(parts[1]))
                continue

            if line.startswith("@@"):
                in_hunk = True
                has_hunks = True
                if not self.HUNK_HEADER_RE.match(line):
                    return False, f"invalid_hunk_header: {line}"
                continue

            # inside hunk: enforce per-line prefixes
            if in_hunk:
                if line == "":
                    # empty line should still have a prefix in real diff;
                    # tolerate here (extract_diff() normalizes), but still accept.
                    continue
                if line[:1] not in {" ", "+", "-", "\\"}:
                    return False, f"invalid_hunk_line_prefix: {line}"

        if not has_hunks:
            return False, "missing_hunks"
        if not has_file_headers:
            return False, "missing_file_headers"

        # dedup preserve order
        dedup: list[str] = []
        seen = set()
        for f in found_files:
            if f and f not in seen:
                seen.add(f)
                dedup.append(f)
        found_files = dedup

        if target_file:
            tf = self._norm_rel(target_file)
            if not found_files:
                # If headers missing file names (rare), we can't be sure; treat as failure.
                return False, f"target_scope_unknown (expected={tf})"
            uniq = found_files
            if len(uniq) != 1 or uniq[0] != tf:
                return False, f"touched_files_not_target: {uniq} (target={tf})"

        return True, ""

    # -----------------------------
    # Full-file blocks (auto-mode helper)
    # -----------------------------

    def parse_file_blocks(self, llm_output: str) -> list[dict[str, str]]:
        """
        Parse Cursor-like full rewrite blocks.

        REQUIRED SPEC (server/service consumption):
          - Recognize: FILE: path / Path: path / Target file: path
          - Body: fenced code OR unfenced until next FILE header or EOF
          - Return list:
              [{"path": "...", "text": "..."}, ...]
        Backward-compat:
          - also include "content" == "text" for older callers
        """
        out: list[dict[str, str]] = []
        text = str(llm_output or "").replace("\r\n", "\n")

        logger.debug(f"parse_file_blocks input length: {len(text)}")
        logger.debug(f"parse_file_blocks first 1000 chars: {text[:1000]}")

        for m in self.FILE_BLOCK_RE.finditer(text):
            path = (m.group("path") or "").strip().strip('"').strip("'")
            code = m.group("code1")
            if code is None:
                code = m.group("code2")
            if code is None:
                code = ""
            code = str(code).replace("\r\n", "\n").rstrip("\n") + "\n"

            logger.debug(f"Found FILE block: path='{path}', code length={len(code)}")
            if path:
                out.append({"path": path, "text": code, "content": code})

        logger.debug(f"parse_file_blocks returning {len(out)} blocks")
        return out

    def extract_file_hunks(self, diff: str) -> dict[str, list[dict[str, Any]]]:
        """
        Utility: parse hunks grouped by file (used by some UI inspectors).
        """
        result: dict[str, list[dict[str, Any]]] = {}
        current_file: Optional[str] = None
        current_hunk: Optional[dict[str, Any]] = None

        for line in (diff or "").replace("\r\n", "\n").split("\n"):
            if line.startswith("diff --git"):
                m = re.search(r"\sb/(.+?)(?:\s|$)", line)
                if m:
                    current_file = m.group(1)
                    result.setdefault(current_file, [])
                    current_hunk = None
                continue

            if line.startswith("@@") and current_file:
                m = self.HUNK_HEADER_RE.match(line)
                if m:
                    old_start = int(m.group(1))
                    old_lines = int(m.group(2) or 1)
                    new_start = int(m.group(3))
                    new_lines = int(m.group(4) or 1)
                    current_hunk = {
                        "old_start": old_start,
                        "old_lines": old_lines,
                        "new_start": new_start,
                        "new_lines": new_lines,
                        "header": line,
                        "content": [],
                    }
                    result[current_file].append(current_hunk)
                continue

            if current_hunk is not None:
                current_hunk["content"].append(line)

        return result


# -----------------------------
# Module-level API (REQUIRED)
# -----------------------------
_parser = EnhancedOutputParser()


def extract_diff(llm_output: str) -> str:
    """Used by external_llm.__init__ and possibly other callers."""
    return _parser.extract_diff(llm_output)


def parse_llm_output(llm_output: str) -> dict[str, str]:
    """
    Backward-compatible entrypoint.

    Some callers expect a tuple (explanation, diff).
    Other callers expect a dict {"explanation":..., "patch":...}.

    We return dict by default.
    """
    return _parser.parse_llm_output_dict(llm_output)


def validate_diff(diff: str, target_file: Optional[str] = None) -> tuple[bool, str]:
    """Backward-compatible helper used by external_llm.service."""
    return _parser.validate_diff(diff, target_file=target_file)


def parse_file_blocks(llm_output: str) -> list[dict[str, str]]:
    """Module-level helper for services that want full-file rewrite blocks."""
    return _parser.parse_file_blocks(llm_output)

def parse_tool_args(raw: Any) -> dict[str, Any]:
    """
    Parse LLM tool call arguments from raw JSON string/dict.

    Handles:
      - Already a dict -> pass through
      - None -> empty dict
      - JSON string -> parse
      - Malformed JSON -> salvage by extracting first {...} region
    """
    import json as _json

    if isinstance(raw, dict):
        return raw
    if raw is None:
        return {}
    if not isinstance(raw, str):
        return {"__raw_arguments": str(raw)}
    s = raw.strip()
    if not s:
        return {}
    try:
        obj = _json.loads(s)
        return obj if isinstance(obj, dict) else {"__raw_arguments": s}
    except _json.JSONDecodeError:
        # salvage: extract first {...} region
        left = s.find("{")
        r = s.rfind("}")
        if left != -1 and r != -1 and r > left:
            mid = s[left : r + 1]
            try:
                obj2 = _json.loads(mid)
                return obj2 if isinstance(obj2, dict) else {"__raw_arguments": s}
            except _json.JSONDecodeError:
                pass
        return {"__raw_arguments": s}
