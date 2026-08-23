# 쓰기 툴 통합 설계안 — `edit` 단일 툴 + mode enum vs 스키마 슬리밍

**작성일**: 2026-08-07
**상태**: Phase 1 = B (슬리밍) **구현 완료** — 실측 5종 7,002→4,248 (−2,754, −39.3%), 전체 21,737→18,982 (−12.7%). 옵션 A 재평가는 Phase 1 후 실측 기준으로 보류.
**측정**: repo 표준 추정기 `_cjk_aware_tokens`(utf8_bytes//2, context_budget과 동일 기준)

---

## 1. 목표와 현황 측정

**목표**: 매 LLM 호출마다 전송되는 툴 스키마 토큰 절감. 쓰기 툴 계열이 전체 스키마의 3분의 1을 차지.

| 툴 | 토큰 | description | 가장 비싼 파라미터 |
|---|---|---|---|
| anchor_edit | 2,237 | 640 | anchor_pattern 559 · anchor_ast_lineno 258 · occurrence 250 |
| edit_text | 2,119 | 747 | edits 578 · scope_start_line 291 · old_string 194 |
| edit_ast | 917 | 156 | ops 484 (미니 매뉴얼) |
| modify_symbol | 903 | 579 | code 86 · symbol 63 |
| apply_patch | 826 | 639 | path 90 · patch 31 |
| **5종 소계** | **7,002 (32.2%)** | **2,761 (39.4%)** | |
| write_plan | 572 | 150 | plan (멀티파일 원자 플랜 — 통합 대상 아님) |
| **6종 합계** | **7,574 (34.8%)** | | |
| 전체 AGENT_TOOL_SCHEMAS | 21,737 | | |

**핵심 발견**: 5종의 39%가 툴 description이고, 파라미터 설명 중 상당수가 "미니 튜토리얼" (anchor_pattern 559, edits 578, ops 484, scope_start_line 291). **절감의 대부분은 통합이 아니라 교육 텍스트 제거에서 나온다.**

**공유 인프라 (이미 단일화 완료 — 통합 시 재사용)**: 3-layer syntax gate(문법 검사 → origin-skip → rollback), session-edit 추적, `_WRITE_TOOLS` 집합, write_targets, `_parse_unified_diff_files`, 파일 인덱스 캐시, 블록 인트로듀서 가드.

---

## 2. 옵션 A — 단일 `edit` 툴 + mode enum (통합)

### 2.1 스키마 형태

5개 툴 → 1개. mode 값은 기존 툴 이름을 그대로 사용 (모델 지식·마이그레이션 1:1):

```json
{
  "name": "edit",
  "description": "Edit a file. Pick the mode matching the change shape: "
                 "apply_patch (unified diff, line ranges), edit_text (exact "
                 "unique string substitution), modify_symbol (whole symbol), "
                 "anchor_edit (pattern/anchor insertion), edit_ast (Python AST ops).",
  "parameters": {
    "properties": {
      "file_path": { "type": "string", "description": "Relative path to the file to edit" },
      "mode": { "type": "string",
                "enum": ["apply_patch", "edit_text", "modify_symbol", "anchor_edit", "edit_ast"],
                "description": "Edit strategy — see description for when to use each." },
      "dry_run": { "type": "boolean", "description": "Preview diff without writing" },
      "patch": { "type": "string", ... }, "path": { "type": "string", ... },
      "old_string": { ... }, "new_string": { ... }, "replace_all": { ... },
      "scope_start_line": { ... }, "scope_end_line": { ... }, "edits": { ... },
      "symbol": { ... }, "code": { ... },
      "anchor_pattern": { ... }, "edit_mode": { ... }, "code_snippet": { ... },
      "occurrence": { ... }, "context_before": { ... }, "context_after": { ... },
      "anchor_ast_lineno": { ... },
      "ops": { ... }
    },
    "required": ["file_path", "mode"]
  }
}
```

### 2.2 절감 계산 (실측 — 2.1 스키마를 실제 구성해 측정)

| 항목 | 토큰 |
|---|---|
| description 5종 → 1종 | −2,331 |
| file_path 4회 → 1회 | −129 |
| symbol 2회 → 1회 | −63 |
| dry_run 2회 → 1회 | −43 |
| mode enum 추가 | +90 |
| **순 절감 (실측)** | **3,027 (전체의 13.9%, 5종의 43%)** — 7,002 → 3,975 |

주의: 실측은 **파라미터 설명을 원문 그대로** 유지한 보수적 수치. 절감의 77%가 description 단일화에서 나오며, 파라미터 설명(3,975 중 ~3,370)이 병합 스키마의 대부분을 차지.

### 2.3 디스패치 설계

```python
# tool_registry.py
_TOOL_HANDLER_MAP["edit"] = "_tool_edit"


def _tool_edit(self, args: dict) -> ToolResult:
    mode = (args or {}).get("mode")
    handler = {
        "apply_patch": self._tool_apply_patch,
        "edit_text": self._tool_edit_text,
        "modify_symbol": self._tool_modify_symbol,
        "anchor_edit": self._tool_anchor_edit,
        "edit_ast": self._tool_edit_ast,
    }.get(mode)
    if handler is None:
        return self._tool_error("edit", f"unknown mode {mode!r} — is required: one of ...")
    return handler(args)  # 각 핸들러가 자체 required 검증을 이미 수행
```

- 기존 핸들러 메서드는 **그대로 유지** — 단위 테스트 대부분 무변경.
- **조건부 required 상실 흡수**: JSON Schema는 mode별 required를 표현 못 함 (oneOf/if-then은 모델 성능·provider 호환 리스크로 기각). 대신 각 핸들러가 이미 "is required" 오류를 emit하고, failure_classifier의 `_TEXT_MISSING_ARGS`가 이를 분류 — 기존 경로가 그대로 커버.
- **환각 호환 shim**: `_TOOL_HANDLER_MAP`에 "apply_patch" → `_tool_edit_legacy` (args에 mode 주입 후 위임) 유지. 스키마 비용 0, 전이 기간 동안 옛 이름 호출도 정상 동작. 1 릴리스 후 제거.

### 2.4 마이그레이션 지도 (이름-키 참조 12곳)

| # | 위치 | 변경 |
|---|---|---|
| 1 | `tool_schemas.py` | 5 스키마 → `SCHEMA_EDIT` 1개, `AGENT_TOOL_SCHEMAS` 순서 |
| 2 | `tool_registry.py` | `_TOOL_HANDLER_MAP`, `_WRITE_TOOLS`(5→1), **2190-2481 self-validating/rollback skip 이름 목록** ("edit_text"/"edit_ast"/"anchor_edit" 세 이름이 mode 기준으로 변경 필요), 2702 병렬화 금지(자동), 2969 |
| 3 | `tool_failure_log.py` | `WRITE_TOOLS`, `_ERROR_PATTERNS`의 `only_tool` 키 ("anchor_edit" 등) |
| 4 | `failure_classifier.py` | `_TEXT_MATCHING_EDIT_TOOLS` + 툴별 복구 힌트 |
| 5 | `work_state_digest.py` | `_WRITE_TOOLS` |
| 6 | `agent_loop_types.py` | `write_tools` 기본값 |
| 7 | `argument_repairer.py` | `_ARG_ALIASES` 5개 툴 → mode-aware 적용 |
| 8 | `agent_turn_pipeline.py` | 87-100 path/file_path 추출 + apply_patch 패치 파싱 특수 처리 |
| 9 | `agent_loop.py` | 1546-1600 syntax gate 경로 ("apply_patch/edit_text/etc."), edit_ast 팁 |
| 10 | `asi_mcp_adapter.py` | `_DESTRUCTIVE_TOOLS`, 195-196 툴 목록, 262 |
| 11 | 프롬프트 3곳 | `orchestrator.py:3759`, `agent_context_manager.py:198`, `insights_manager.py:1135-1166` |
| 12 | `repl_impl.py:6245` | `edit_file:`/`edit_text:`/`modify_symbol:` 결과 접두사 (cosmetic) |

테스트 영향: 213개 파일, ~1,487 참조 라인. 핸들러 단위 테스트(메서드 직접 호출)는 대부분 유지, 스키마/분류기/수리기/MCP 테스트는 마이그레이션.

### 2.5 옵션 A 리스크

| 리스크 | 심각도 | 완화 |
|---|---|---|
| 라우팅 신호 약화: 툴 이름이 전략 신호 → mode 값으로 | 중 | mode 값을 기존 이름과 동일하게; description에 "when to use each" 명시; 파일럿 실측 |
| 조건부 required 상실 (mode별 필수 인자 미강제) | 저-중 | 핸들러 검증 + "is required" 분류기 기존 경로 |
| 옛 이름 환각 호출 | 저 | shim (스키마 비용 0) |
| mode에 안 맞는 파라미터 혼합 전송 | 저 | 핸들러가 무시 (기존 대비 변화 없음 — 현재도 여분 인자 무시) |
| 마이그레이션 비용 (12곳 + 테스트) | 중 | 핸들러 무변경 + shim으로 완충 |

---

## 3. 옵션 B — 스키마 슬리밍 (툴 수 유지, 교육 텍스트 이동)

절감 대상: **교육 텍스트를 스키마에서 제거하고 핸들러 실패 메시지로 이동** (실패 메시지는 이미 같은 문체로 존재 — apply_patch dirty-target 거절, edit_text 유일성 요구 등. "did you mean" 힌트 인프라 `_ast_fail_hint`/`_near_match_hint` 확장).

| 대상 | 현재 | 목표 | 절감 |
|---|---|---|---|
| description 5종 | 2,761 | 5×~140 (라우팅 문장만 유지) | ~2,060 |
| anchor_pattern | 559 | 230 | ~330 |
| edits | 578 | 300 | ~280 |
| scope_start_line | 291 | 140 | ~150 |
| ops | 484 | 270 | ~215 |
| occurrence | 250 | 140 | ~110 |
| anchor_ast_lineno | 258 | 170 | ~90 |
| old_string | 194 | 150 | ~45 |
| **총 절감** | | | **≈ 3,280 (전체의 15%)** |

실측 검증: description만 라우팅 문장(2문장 내외)으로 교체 시 7,002 → 4,598 (**−2,404, 11.1%**). 파라미터 슬리밍까지 하면 ~3,280.

**유지 원칙**: 각 description의 첫 1-2문장 (라우팅: "★ PREFERRED over apply_patch for symbol-level changes", "Use when ...")은 반드시 보존. 제거 대상은 "작동 원리/실패 모드/주의사항" 교육.

**리스크**: 모델이 축약 설명으로 혼동 (검증 필요) — 실패 시 힌트가 백업. **API/이름/인프라 불변** → 테스트·마이그레이션 비용 0.

---

## 4. 비교와 권고

| | 옵션 A (통합) | 옵션 B (슬리밍) |
|---|---|---|
| 절감 (실측) | 3,027 (13.9%) | 2,404~3,280 (11.1~15%) |
| API 변경 | 파괴적 (LLM-facing) | 없음 |
| 인프라 마이그레이션 | 12곳 + 테스트 1,487라인 | 없음 |
| 라우팅 리스크 | 중 (mode 선택) | 저 (설명 축약) |
| 검증 가능성 | 파일럿 실측 필요 | 기존 스위트 + 역검증 |

**권고: B 선행 → A 재평가 (하이브리드 C)**.

1. **Phase 1 = B**: 저위험 대절감. 검증: 풀 스위트 + 역검증(stash 후 실패 확인) + "수정 전 설명으로 모델 호출 실패율" 회귀 지표.
2. **Phase 2 = A 재평가**: B 적용 후 실측 — **B를 완료하면 A의 추가 절감은 ~400-600으로 붕괴** (이미 description이 줄었고, 파라미터 설명은 통합으로도 제거 안 됨). 즉 A의 가치는 B를 먼저 하면 대부분 소멸. A가 정말 필요하면 **mode 값 = 기존 이름** + shim 전이로 리스크 완충.
3. **P1-2 (병렬 세션 `ContextWindowCollapseError`, minimal 스키마 변종)와 상호보완**: B/A는 모든 윈도우의 상수 절감, minimal 변종은 소형 윈도우 한정 툴 제거 — 충돌 없음.

**참고**: write_plan(572토큰)은 멀티파일 원자 플랜 시맨틱이라 통합 대상에서 제외. edit_file은 LLM 미노출 내부 별칭 — 변경 불필요.

---

## 5. 검증 계획 (Phase 1 = B 기준)

1. **측정 재현**: `estimate_tokens_from_tool_schemas`로 적용 전후 비교 — **완료: 21,737 → 18,982 (−2,755)**.
2. **역검증**: 수정 전 스키마 stash → 신규 테스트 RED — **완료** (토큰 하드코딩 단언은 의도적으로 미추가: 재슬리밍 시 fragile. 변경 증거는 실측 7,002→4,248 + test_context_budget floor 의존 단언의 RED→견고화).
3. **파일럿 실측 (A 진행 시)**: 로그의 tools 배열 실전송 토큰 + 잘못된 mode 선택률 (이전 툴 선택 오류율과 비교).
4. **풀 스위트** + ruff + pre-commit — **완료: 9,022 passed**.

### Phase 1 적용 내역 (2026-08-07)

- **description 5종** 2,761 → 1,083: 라우팅 문장만 유지 (★ PREFERRED/Use for/대안 툴 안내). 작동 원리·실패 모드·언어 목록·모드 상세 제거.
- **파라미터 슬리밍** (실측): anchor_pattern 559→200, edits 578→401, scope_start_line 291→144, ops 484→387, occurrence 250→150, anchor_ast_lineno 258→182, old_string 194→105.
- **교육 텍스트의 실패 경로 존재 확인** (제거분 전부 이미 커버): edit_text 유일성/스코프 안내 (write_tools_edit_mixin 1217/1224/1366), anchor_miss/not_unique (anchor_shared), edit_ast unknown op (ast_op_executor:209), apply_patch dirty/untracked (patch_mixin).
- **스키마 내용 단언 테스트 없음** — parity 테스트는 이름/핸들러만 검사. 유일 파급: test_context_budget의 `_structural_window_floor()` 의존 단언 (스키마 토큰 축소로 floor 이동 → floor 기준 견고화).
