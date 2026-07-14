# Typed Failure Classifier — tree-sitter 에러 노드 기반 전환 설계

> 장기 과제. CLAUDE.md 설계 인사이트("keyword/regex → AST/graph/typed policy 전환")의
> 1순위 적용 대상인 `vm/failure_classifier.py`(+ `ts_vm/repair/failure_classifier.py`)의
> 전환 설계.

## 1. 현황과 문제

현재 파이프라인 (`vm/vm.py`):

```
verifier (compile/javac/kotlinc/go build)
    → VerifyError(message, line, column, code)
    → BaseFailureClassifier._classify_single(errors[0])
        1) error_code_map   (Python E0602/F821 등 일부만)
        2) keyword_map      (소문자 substring 매칭)
        3) regex_patterns   (메시지 regex)
    → FailureType enum
    → RepairPlanner → RepairRegistry[FailureType] → 전략 실행
```

문제점:

| # | 문제 | 근거 |
|---|------|------|
| P1 | **로케일 취약**: javac는 JDK 로케일을 따라 한국어("오류:")로 출력. `_parse_javac_output`의 `(error\|warning)` regex와 keyword_map 전체가 무력화 → 에러가 0건으로 파싱되어 `returncode!=0`인데 errors=[] | `verifier.py:169`, `failure_classifier.py:125-137` |
| P2 | **컴파일러 버전/문구 취약**: 메시지 문구가 바뀌면 조용히 UNKNOWN으로 강등. 실패해도 아무 신호 없음 | keyword/regex 3층 전부 |
| P3 | **심볼 추출 부정확**: `\w+` 기반이라 한정자(`pkg.Foo`), 제네릭, Kotlin 백틱 식별자, 유니코드 식별자에서 잘리거나 실패 | `_PY_EXTRACT_SYMBOL` 등 |
| P4 | **첫 에러만 분류**: `classify()`가 `errors[0]`만 봄. 근본 원인이 2번째 에러인 경우(연쇄 에러) 오분류 | `failure_classifier.py:39-42` |
| P5 | **FailureType enum 중복**: vm/과 ts_vm/에 거의 동일한 enum 2벌 (PROPERTY_NOT_EXIST, MISSING_VARIABLE, UNUSED_IMPORT만 차이) | 두 파일 비교 |
| P6 | **가장 신뢰할 신호를 안 씀**: 실패한 *코드 자체*가 손에 있는데 컴파일러 *메시지 문자열*만 파싱 | 설계 전반 |

## 2. 핵심 원칙

**"메시지 파싱은 최후 수단."** 신뢰도 내림차순 3계층:

1. **Layer A — 코드 구조 (tree-sitter)**: 실패한 코드를 직접 파싱해 ERROR/MISSING
   노드에서 구문 오류를 *구조적으로* 판정. 로케일/컴파일러 무관, 8개 grammar 이미
   `pyproject.toml` 의존성에 존재.
2. **Layer B — 기계용 진단 코드**: 의미 오류(타입/임포트/미정의 심볼)는 tree-sitter가
   못 본다. 대신 컴파일러의 **안정적 기계용 코드**를 확보해 typed 테이블로 매핑.
3. **Layer C — 메시지 폴백**: 기존 keyword/regex를 축소 유지하되, 폴백 사용을
   계측해서 수렴을 측정.

정직한 한계: tree-sitter 에러 노드는 **구문 오류 축**만 담당한다. 전환의 실익은
(a) SYNTAX_ERROR 판정의 구조화, (b) 심볼 추출의 구조화, (c) repair에 넘길
typed FixHint 생성이며, 의미 오류 분류의 탈-regex는 Layer B(진단 코드)가 담당한다.

## 3. 데이터 모델 (typed 출력)

`FailureType` 단일 enum 반환 → `Classification` dataclass로 확장:

```python
# vm/classification.py (신규, vm/ts_vm 공유)

class EvidenceSource(str, Enum):
    TREE_SITTER = "tree_sitter"      # Layer A
    ERROR_CODE = "error_code"        # Layer B
    MESSAGE_FALLBACK = "message"     # Layer C
    NONE = "none"                    # UNKNOWN

@dataclass(frozen=True)
class FixHint:
    """repair 전략에 넘기는 구조화 힌트 (선택적)."""
    kind: str                  # "insert_token" | "remove_import" | "rename" ...
    token: Optional[str]       # 예: ";"  (MISSING 노드의 기대 토큰)
    line: Optional[int]
    column: Optional[int]

@dataclass(frozen=True)
class Classification:
    type: FailureType
    source: EvidenceSource
    symbol: Optional[str] = None      # extract_symbol 통합 (단일 패스)
    fix_hint: Optional[FixHint] = None
    error_index: int = 0              # 어느 VerifyError에서 판정했는지
```

- `classify()` 시그니처는 유지하되 `classify_typed()` 신설 → 기존 소비자
  (`repair_planner.py:61`)는 `classify_typed().type`으로 점진 전환.
- `extract_symbol()`은 `Classification.symbol`로 흡수 (regex 재실행 제거).
- vm/ts_vm의 FailureType을 공유 모듈로 승격하고 합집합으로 통일
  (PROPERTY_NOT_EXIST + MISSING_VARIABLE + UNUSED_IMPORT 모두 포함).

## 4. Layer A — tree-sitter 구문 분류기

`languages/tree_sitter_utils.py`에 기존 `has_error()`(bool)를 확장한 유틸 추가:

```python
@dataclass(frozen=True)
class SyntaxErrorNode:
    kind: str            # "ERROR" | "MISSING"
    missing_token: str   # MISSING 노드의 기대 토큰 (예: ";", ")")
    line: int            # 0-based → 소비자에서 1-based 변환
    column: int
    context_snippet: str # 주변 소스 (repair 프롬프트/전략용)

def find_error_nodes(content: str, language: str) -> Optional[list[SyntaxErrorNode]]:
    """ERROR/MISSING 노드 전수 수집. tree-sitter 미설치 시 None (폴백 신호)."""
```

- 구현은 `has_error()`와 동일한 반복 DFS(재귀 한도 안전) — 조기 return 대신 수집.
- MISSING 노드는 `node.type`이 곧 기대 토큰이므로 `FixHint(kind="insert_token",
  token=node.type, line=..., column=...)`을 공짜로 얻는다. 현재
  `py_repair_syntax_error`류가 메시지에서 재유도하던 정보의 상위 호환.
- 분류 우선순위 규칙:
  - **tree-sitter ERROR/MISSING 존재 ⇒ SYNTAX_ERROR (고신뢰)**. 메시지는 안 봄.
  - **트리 클린 + 컴파일러 에러 존재 ⇒ 구문 오류 아님이 보장** → Layer B/C로.
    이 규칙만으로 현재 keyword_map의 구문 관련 절반(“expected ';'”, “unclosed”,
    “expecting”, “invalid syntax”, “unexpected indent”…)이 삭제 가능.
- 폴백: `is_available()==False` 또는 `find_error_nodes()==None`이면 Layer A 생략
  (레포 전반에서 이미 쓰는 guard 패턴과 동일).
- 주의(리스크 R1): tree-sitter grammar는 컴파일러보다 관대/엄격이 어긋날 수 있다
  (Python soft keyword, 최신 문법). 따라서 "트리 클린 ⇒ 구문 오류 아님"은
  **컴파일러가 구문 코드를 명시한 경우(Layer B) 뒤집을 수 있는 약한 보장**으로 둔다.

## 5. Layer B — 진단 코드 정규화 (verifier 강화 선행)

분류기가 typed가 되려면 verifier가 기계용 코드를 실어줘야 한다:

| 언어 | 현재 | 전환 | 얻는 것 |
|------|------|------|---------|
| Python | `pyright <file>` 텍스트 파싱, code="PYRIGHT" 고정 | `pyright --outputjson` | `rule` 필드 (reportUndefinedVariable, reportMissingImports…) → 코드맵 직행, regex 파싱 삭제 |
| Java | `javac` 로케일 의존 텍스트 | `javac -XDrawDiagnostics` | `compiler.err.cant.resolve.location`, `compiler.err.expected` 등 **로케일 무관 안정 키** → P1 근본 해결 |
| Kotlin | kotlinc 텍스트 | kotlinc는 **로케일 무관**(항상 영어 출력, 실측 확인) + 안정코드 미제공 → **메시지 폴백 유지**. `-J-Duser.language=en`은 no-op이지만 방어적 일관성 유지 | P1 해당 없음 (로케일 취약성 없음) |
| Go | `go build` 텍스트 | 안정 코드 없음(메시지는 사실상 안정) → 폴백 유지, `LANG=C` 고정 | 현상 유지 (Go 메시지는 영어 고정이라 위험 낮음) |
| TS | tsc TS-코드 (ts_vm) | 이미 모범 사례 (`_TSC_CODE_MAP`) | Layer B의 참조 구현 |

- `error_code_map`은 언어별 **데이터 테이블**(dict)로 남긴다 — 이것은 regex가 아니라
  typed 매핑이므로 인사이트 위반이 아님. ts_vm의 `_TSC_CODE_MAP`이 표본.
- 즉시 수정 가치(설계와 별도 선행 가능): javac 로케일 고정만 먼저 넣어도 P1의
  "에러 0건 파싱" 실버그가 사라진다.
- **kotlinc 실측 결과** (2026-07-05): `LANG=C`, `LANG=ko_KR.UTF-8`, `-J-Duser.language=en`
  세 조합 모두 동일한 영어 출력 (`Bad.kt:1:22: error: unresolved reference 'undefinedSymbol'.`).
  kotlinc는 로케일라이즈되지 않으므로 P1의 로케일 취약성이 Kotlin에는 해당하지 않음.
  `-J` 플래그는 무해하지만 no-op — 방어적 일관성을 위해 유지.

## 6. Layer C — 메시지 폴백 + 계측

- 기존 keyword/regex 유지하되 구문 관련 항목 제거(Layer A가 흡수)로 표면적 축소.
- `Classification.source`로 폴백 사용을 노출하고, 분류 시 카운터 로깅:
  `logger.info("classify: %s via %s", type, source)`. 기존 adaptive/weight_learning
  인프라에 연결하면 "폴백률" 텔레메트리로 전환 수렴을 정량 측정 가능.
- 폴백률이 특정 언어에서 0에 수렴하면 해당 언어의 Layer C 항목 삭제.

## 7. 심볼 추출 전환 (P3)

메시지 regex(`\w+`) → **위치 기반 트리 조회**:

```
컴파일러가 line/column을 준다 (VerifyError.line/column 이미 존재)
    → tree.root_node.descendant_for_point_range((line-1, col-1), ...)
    → identifier/type_identifier 노드 (또는 최근접 식별자 조상/형제)
    → node.text = 정확한 심볼
```

- 한정자·제네릭·백틱·유니코드 식별자 모두 정확. 한국어 식별자 심볼도 안전 (다국어 요건).
- line/column이 없는 에러만 기존 메시지 regex로 폴백 (source에 기록).

## 8. 마이그레이션 페이즈

| Phase | 내용 | 검증 |
|-------|------|------|
| 0 | FailureType 공유 모듈 통일 + `Classification`/`classify_typed()` 도입 (동작 불변, 기존 regex 그대로 위임) | **골든 코퍼스 스냅샷**: 언어×실패유형 매트릭스의 실제 컴파일러 출력을 fixture로 채집, 현재 분류 결과를 golden으로 고정 |
| 1 | verifier 로케일 고정 + `pyright --outputjson` + `javac -XDrawDiagnostics` + 코드맵 확충 (Layer B) | golden diff — UNKNOWN 감소만 허용 |
| 2 | `find_error_nodes()` 추가 + Layer A 우선순위 적용, 구문 keyword/regex 삭제 | golden diff + FixHint 스냅샷 |
| 3 | 위치 기반 심볼 추출 + extract_symbol 흡수 | 심볼 추출 정확도 fixture (한정자/백틱/유니코드 케이스 포함) |
| 4 | ts_vm 분류기를 공유 구현으로 교체, 폴백률 계측 연결 | ts_vm 기존 테스트 + 폴백률 로그 |

각 Phase는 독립 커밋/독립 테스트. 골든 코퍼스가 회귀 안전망
(레포 원칙: 서브시스템 테스트 선행 — memory/run-scope-tests-before-commit).

## 9. 리스크

- **R1 grammar-컴파일러 불일치**: §4 우선순위 규칙으로 완화 — Layer B의 명시적
  구문 코드가 tree-sitter 판정을 뒤집을 수 있게 설계.
- **R2 tree-sitter 미설치 환경**: optional 의존 그룹이므로 `is_available()` 폴백 필수.
  Layer A 없이도 Layer B/C만으로 현재 수준 이상 동작해야 함 (Phase 1이 2보다 선행하는 이유).
- **R3 ERROR 노드 위치 부정확**: tree-sitter의 에러 복구는 휴리스틱이라 ERROR 노드
  위치가 실제 원인과 어긋날 수 있음 → FixHint는 "힌트"로만 쓰고 repair 전략이
  검증(재파싱) 후 적용. VM 루프가 이미 re-verify를 하므로 안전망 존재.
- **R4 classify(errors[0]) 유지 여부 (P4)**: classify_all 결과에서
  `SYNTAX_ERROR > MISSING_IMPORT > 기타` 순의 근본원인 우선 규칙을 Phase 2에서 도입
  (구문 오류가 있으면 연쇄 의미 에러는 노이즈).
