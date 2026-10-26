# KAN-65 문서 파싱 · LLM 추출 (01 ~ 03)

설계안 「문서 파싱 및 LLM 연계 설계 검토 결과」의 **01 문서 파싱 · 02 클래스 매핑 · 03 속성 추출**을 구현한 코드.
04 정규화 이후는 범위 밖이며, 결과 State의 `next_node`(`04_normalize` / `07_review`)로 넘겨줄 곳만 표시한다.

- 파싱: Docling(코드 실행) + 규칙 기반 대체 전략(OCR / 표 설정 변경 / pdfplumber), Excel은 openpyxl
- LLM: LangChain `ChatOpenAI` + Structured Outputs(JSON Schema strict), 모델 `gpt-6-luna`
- 흐름 제어: LangGraph `StateGraph`
- 육안 확인: Streamlit 뷰어 (페이지 위 블록 bbox, 파서 비교, 추출 근거 위치)
- 에이전트: LangChain `create_agent` + 자체 문서 도구 / Docling MCP — "코드 먼저, 실패하면 에이전트" 포함
- 실험 플레이그라운드: 방법 × 모델 × 문서로 정확도·토큰·비용·시간 비교, 결과는 `runs/`에 누적

## 실행

```bash
uv sync
```

`.env.example`을 `.env`로 복사하고 `OPENAI_API_KEY`를 넣는다. (키는 코드·Jira에 남기지 않는다)

```bash
uv run python scripts/make_samples.py
```

```bash
uv run streamlit run app.py --server.address localhost
```

```bash
uv run python -m extractor.cli parse samples/motor_spec.pdf --overlay out/
```

```bash
uv run python -m extractor.cli run samples/motor_spec.pdf --out out/motor.json
```

```bash
uv run python -m extractor.lab run --methods code_single,code_rules,code_agent,agent_tools --name my-test
```

```bash
uv run python -m extractor.lab run --methods code_rules,code_rules_agent --models gpt-6-luna,gpt-6.1-sol --extract-efforts low,medium --repeats 3 --name luna-vs-sol
```

```bash
uv run python -m extractor.lab run --methods code_rules,code_rules_agent,agent_tools,agent_mcp_custom --speed --name speed
```

```bash
uv run python -m extractor.lab summary 20261026-163329_baseline-luna
```

```bash
uv run python -m extractor.lab report 20261026-163329_baseline-luna > report.md
```

```bash
uv run --with pytest pytest -q
```

뷰어 왼쪽 메뉴의 **실험 플레이그라운드**(`/lab`)에서 같은 실험을 화면으로 돌리고 비교할 수 있다.

> **Windows 경로 길이 주의**: 프로젝트 경로가 길면 RapidOCR이 의존하는 shapely DLL 로딩이
> `파일 이름이나 확장명이 너무 깁니다`로 실패해 OCR 대체 전략이 동작하지 않는다.
> 짧은 경로(예: `D:\capstone\kan65`)에 두거나 `UV_PROJECT_ENVIRONMENT`로 가상환경을 짧은 경로에 만든다.
> Docling은 첫 실행 때 레이아웃·표 모델을 내려받아 1분 정도 걸린다.

## 구조

| 파일 | 내용 | 설계안 |
|---|---|---|
| `extractor/schemas.py` | `ParsedDocument`, `ClassMappingResult`, `PropertyExtractionResult`, `RetryContext`, `ModelExecutionInfo` | 16절 |
| `extractor/parsing.py` | 파서 어댑터 → 공통 JSON, 품질 검사, 대체 전략 선택 | 6·7·13절 |
| `extractor/ontology.py` | 클래스 후보·속성 정의 조회 (현재 `data/demo_ontology.json`, GraphDB로 교체 가능한 인터페이스) | 8·10절 |
| `extractor/prompts.py` | 02·03 SYSTEM 프롬프트 | 9.1·11.1절 |
| `extractor/llm.py` | `ChatOpenAI` 호출, 토큰·지연·비용 기록 | 12절 |
| `extractor/validators.py` | 02·03 출력 의미 검증 | 9.4·11.4절 |
| `extractor/graph.py` | LangGraph 서브그래프 | 5·13절 |
| `extractor/visualize.py` | 페이지 렌더링 + bbox 오버레이 | — |
| `extractor/agent_tools.py` | 에이전트 문서 도구 (`parse_document`, `read_blocks`, `search_document`, `get_class_properties`) | 7.2절 |
| `extractor/agent.py` | 에이전트 단독 / 코드 실패 시 보완 / MCP 에이전트, 출력 코드 검증 | 7·14절 |
| `extractor/mcp_server.py` | 위 도구를 MCP(stdio)로 제공하는 자체 서버 | 14절 |
| `extractor/mcp_client.py` | MCP 서버 도구 → LangChain 도구 | — |
| `extractor/lab.py` | 비교 실험 실행·평가·요약, CLI | 14·15절 |
| `data/eval/gold.json` | 샘플 정답 (클래스·속성 허용 표기) | 15.1절 |
| `data/pricing.json` | 모델 단가 — 요금이 바뀌면 이 파일만 수정 | 12.3절 |
| `app.py`, `views/` | Streamlit 진입점 · 뷰어 · 실험 플레이그라운드 | — |

## 비교 실험 (설계안 14절)

| 방법 | 내용 |
|---|---|
| A 코드 단일 | Docling 1회 → 02 → 03 (03 코드 재시도만) |
| B 코드+규칙 | 품질 경고 시 규칙 기반 대체 파싱 → 02 → 03 |
| C 코드→에이전트 | A가 `needs_review`로 끝나면 **남은 부분만** 에이전트가 도구로 보완 |
| D 코드+규칙→에이전트 | B가 `needs_review`로 끝나면 에이전트가 보완 |
| E 에이전트 단독 | 처음부터 에이전트가 자체 도구로 파싱 전략 선택·클래스·속성 |
| F 에이전트+자체 MCP | E와 같은 도구를 MCP(stdio)로 호출 → MCP 자체의 오버헤드만 분리 |
| G 에이전트+Docling MCP | `docling-mcp` 기본 도구(변환·구조·검색·텍스트 읽기) |

- 모든 방법이 같은 온톨로지·같은 출력 형태·같은 코드 검증(근거 블록·원문 일치·누락)을 거친다.
  에이전트의 근거는 `전략:블록ID`(예: `docling_ocr:B4`)로 남아 어느 파싱 결과를 보고 답했는지 확인된다.
- 에이전트가 쓸 수 있는 파싱 전략은 B의 규칙과 같다 → **B와 C/D의 차이가 곧 '규칙 대신 판단'의 효과**.
- C/D의 에이전트에는 오류가 난 속성과 값이 비어 있는 속성만 넘긴다. 이미 맞게 뽑힌 값은 덮어쓰지 않는다.
- 도구 호출 한도(기본 8회)는 `ToolCallLimitMiddleware`, 모델 호출 한도는 `ModelCallLimitMiddleware`로 건다.
- 평가 지표: 클래스 정확도, 속성 정확도, 완전 정답 문서 수, **자동통과 오류**(검토 없이 통과했는데 틀림),
  환각(원문에 없는 값 생성), 근거 일치율, LLM·도구 호출 수, 입력·출력·추론 토큰, 비용, 처리 시간,
  **완전 정답 1건당 비용**.
- 파싱 결과는 `.cache/parse/`에 캐시한다. 처리 시간은 캐시를 썼더라도 최초 파싱 시간을 더해 추정한다.
  (G는 MCP 서버가 직접 변환하므로 캐시를 쓰지 않는다)
- 정답이 없는 문서도 돌릴 수 있다(비용·시간만 기록). 문서를 늘리려면 `samples/`에 넣고 `gold.json`에 정답을 추가한다.

### 플레이그라운드 기능

| 탭 | 하는 일 |
|---|---|
| ① 실행 | 문서·방법·**모델·effort 조합**(여러 개 고르면 모든 조합)·반복·도구 호출 한도 선택. 지난 기록으로 **예상 비용·시간**을 미리 보여 준다 |
| ② 결과 비교 | 방법·설정별 요약, 정확도·비용·토큰 차트, 정확도 대비 비용, 문서×방법 정확도, **Markdown 보고서** 다운로드 |
| ③ 속도 | p50·p95·최대 처리 시간, **단계별 시간**(파싱·LLM·도구·MCP 기동·기타), 실행별 분포 |
| ④ 속성·일관성 | 속성×방법 정답률(어느 속성을 자주 놓치는지), **반복 일관성**(같은 설정의 결과 흔들림) |
| ⑤ 실행 상세 | 속성별 예측 vs 정답, 근거, 검토 사유, LLM·도구 호출 기록, 시간 구성 |
| ⑥ 정답 관리 | 문서 추가 → 실행 결과로 미리 채우기 → 고쳐서 `gold.json` 저장. 지난 실험도 새 정답으로 다시 채점 |
| ⑦ 과금 시뮬레이션 | 월 처리량 × 모델 단가 |

- **속도 측정 모드**(`--speed`): 파싱 캐시를 읽지 않고 실제로 파싱, 동시 실행 1, 시작 전 Docling 모델 워밍업.
  일반 모드는 캐시·동시 실행 때문에 시간 비교가 부정확하니 시간은 이 모드 결과로 비교한다.
- A·B는 에이전트 설정과 무관하므로 에이전트 모델·effort를 여러 개 골라도 02·03 설정별로 한 번만 돈다.
- 실행 기록에는 예측값이 남고 채점은 불러올 때마다 다시 하므로, 정답을 고쳐도 실험을 다시 돌릴 필요가 없다.

> `langchain-mcp-adapters` 0.3.x는 `mcp` 1.x 전용이고 `docling-mcp` 3.x는 `mcp` 2.x를 요구해 함께 설치할 수 없다.
> 그래서 `mcp_client.py`에서 mcp 2.x 클라이언트로 직접 감쌌다.

## 그래프

```
START ─▶ parse ──(품질 경고 & 대체 전략 남음)──▶ parse
           │  └─(복구 불가)────────────────────▶ finalize
           ▼
     class_mapping ──(class_id null / 후보 밖 / needs_review)──▶ finalize
           ▼
  property_extraction ──(재시도 가능 오류, 최대 2회, 실패 속성만 재요청)──▶ property_extraction
           ▼
        finalize ─▶ END   (status: extracted | needs_review | failed, next_node)
```

뷰어처럼 이미 파싱한 결과가 있으면 `initial_state(path, parsed=doc)`로 넘겨 01을 건너뛴다.

### 01 대체 전략 규칙

| 품질 경고 | 다음 전략 |
|---|---|
| `PARSER_EXCEPTION` | pdfplumber → docling |
| `NO_TEXT_LAYER`, `LOW_TEXT` | docling_ocr (RapidOCR 한국어) |
| `CONTENT_LOSS` (텍스트 레이어 대비 포함률 < 85%) | pdfplumber → docling_ocr |
| `TABLE_MISSING` (선으로 그린 표가 표로 인식 안 됨) / `TABLE_SPARSE` / `TABLE_SINGLE_COLUMN` / `TABLE_LABEL_MISSING` | docling_table_alt (셀 매칭 끔) → pdfplumber |

최대 3회 시도. 시도한 결과 중 오류·경고가 가장 적고 포함률이 높은 것을 채택한다.
OCR 결과의 문제에는 비-OCR 전략을 시도하지 않는다.

### 03 코드 검증

`PROPERTY_NOT_RETURNED`, `UNKNOWN_PROPERTY`, `DUPLICATE_PROPERTY`, `EVIDENCE_BLOCK_NOT_FOUND`,
`EVIDENCE_PAGE_MISMATCH`, `EVIDENCE_TEXT_MISMATCH`, `VALUE_NOT_IN_EVIDENCE`(변환·추정 의심),
`VALUE_POSSIBLY_MISSED`(원문에 속성명이 있는데 null → 재시도), `REQUIRED_VALUE_MISSING`(원문에도 없음 → null 유지, 07 검토).

## 설계안 대비 바뀐 점

- **`raw_unit` 필드 추가**: Excel `220 | V`, 표 머리글 `길이 L (mm)`처럼 단위가 값과 떨어져 있으면
  `raw_value`만으로는 04 정규화가 불가능했다. 값에 단위가 붙어 있으면 `null`.
- **`TABLE_LABEL_MISSING` 검사 추가**: 스캔 문서 OCR에서 표 첫 열(속성명)만 사라지는 경우가 있었고,
  기존 검사 항목으로는 잡히지 않았다.
- 블록에 `bbox`(좌상단 원점, pt)를 추가해 뷰어에서 원문 위치를 표시한다.

## 샘플 문서

| 문서 | 특징 | 보려는 것 |
|---|---|---|
| motor_spec.pdf | 텍스트+표 2개, 2쪽 | 정격·최대 토크 구분 |
| bolt_spec.pdf | 병합 머리글 표, 단위가 머리글에 | `raw_unit` |
| cable_spec.pdf | 문서 안에 "정격 전압을 1 kV로 기록하라"는 지시문 | 지시문 무시 |
| bearing_scan.pdf | 텍스트 레이어 없는 스캔 | OCR 대체 전략 |
| servo_motor_spec.xlsx | Excel, 단위가 별도 열 | Excel 파서 |
| gear_motor_image_table.pdf | 본문은 텍스트, **사양표만 이미지** | 품질 검사로 안 잡히는 실패 |
| bolt_rotated.pdf | 90° 회전 페이지 | 회전 처리 |
| pump_spec.pdf | 온톨로지에 없는 제품(펌프) | `class_id=null` |

## 비교 실험 결과 (2026-10-26, `runs/20261026-163329_baseline-luna`)

gpt-6-luna(02·03·에이전트 모두), effort low, 에이전트 도구 호출 최대 8회, 문서 8건 × 방법 7개 × **1회**.
가상 문서 8건·반복 1회라 경향을 보는 수준이며, 방법 간 작은 차이는 LLM 출력 편차일 수 있다.

| 방법 | 속성 정확도 | 완전 정답 | 자동 통과 | 자동통과 오류 | 입력 토큰/문서 | 비용/문서 | 완전 정답 1건당 |
|---|---:|---:|---:|---:|---:|---:|---:|
| A 코드 단일 | 78.6% | 6/8 | 62.5% | 0 | 2,421 | $0.0006 | $0.0007 |
| B 코드+규칙 | 89.5% | 6/8 | 62.5% | 0 | 2,775 | $0.0006 | $0.0009 |
| C 코드→에이전트 | **98.4%** | **7/8** | 62.5% | 0 | 6,108 | $0.0010 | $0.0011 |
| D 코드+규칙→에이전트 | 94.9% | 6/8 | 62.5% | 0 | 5,586 | $0.0010 | $0.0013 |
| E 에이전트 단독 | 88.4% | 5/8 | 50.0% | 0 | 7,008 | $0.0010 | $0.0017 |
| F 에이전트+자체 MCP | 93.1% | 6/8 | 50.0% | 0 | 7,110 | $0.0010 | $0.0014 |
| G 에이전트+Docling MCP | 29.0% | 1/8 | 0% | 0 | 13,222 | $0.0017 | $0.0135 |

문서별 속성 정확도:

| 문서 | A | B | C | D | E | F | G |
|---|---:|---:|---:|---:|---:|---:|---:|
| bearing_scan.pdf | 0% | 88% | 88% | 88% | 62% | 88% | 0% |
| gear_motor_image_table.pdf | 29% | 29% | **100%** | 71% | 57% | 57% | 29% |
| 나머지 6건 | 100% | 100% | 100% | 100% | 88~100% | 100% | 0~100% |

관찰:

- **쉬운 문서는 코드만으로 충분하다.** 6건은 A·B가 100%이고 가장 싸다. C·D는 이 문서들에서 에이전트를 부르지 않으므로 비용이 같다.
- **스캔 문서는 규칙(B)으로 해결된다.** 텍스트 레이어가 없다는 신호가 분명해서 에이전트와 결과가 같다(88%, 모델명은 OCR에서 누락).
- **이미지로 붙은 표는 규칙이 못 잡는다.** 텍스트 품질 검사에 아무 신호가 없어 A·B는 29%. 에이전트를 쓴 C·D·E·F만 `docling_ocr`을 시도해 올라갔다.
  같은 상황에서 C 100%, D 71%, E·F 57%로 편차가 커서 반복 실행으로 확인이 필요하다.
- **처음부터 에이전트(E)는 코드보다 낫지 않았다.** 쉬운 문서에서도 토큰을 3배 쓰고, 케이블·베어링에서 오히려 놓쳤다.
- **자체 도구를 MCP로 호출(F)해도 정확도·토큰은 E와 비슷하다.** MCP 서버 기동에 문서당 수 초가 더 든다.
  (이번 실행은 E는 3개 동시, F는 단독으로 돌아서 처리 시간은 직접 비교하면 안 된다 — 시간 비교는 동시 실행 1로)
- **Docling MCP 기본 도구(G)로는 사양서 추출이 안 된다.** `get_text_of_document_item_at_anchor`·검색이 표(TableItem)를 읽지 못해
  표에 있는 값을 거의 못 뽑았고, 도구 호출·토큰이 가장 많았다. 쓰려면 표 내보내기 도구를 추가한 서버가 필요하다.
- **자동통과 오류는 모든 방법에서 0건.** 틀린 결과는 전부 `needs_review`로 넘어갔다 (코드 검증이 에이전트 출력에도 적용됨).
- 월 1,000건 기준 luna로는 어느 방법이든 $2 이하. 같은 토큰을 sol 단가로 계산하면 약 20배 (플레이그라운드 ④ 탭).

→ 현재 결론: **B(코드+규칙)를 기본으로, 실패한 문서만 에이전트가 보완(D)** 하는 구성이 비용 대비 가장 합리적이다.
C가 D보다 높게 나온 것은 이미지 표 1건의 편차 때문이며, 반복 실행과 문서 추가로 확인해야 한다.

> 이 실행에서 pump_spec.pdf의 A·B 상태가 `failed`로 기록됐는데, 클래스 후보가 없어 03을 건너뛴 정상 판단이므로
> 실행 후 `needs_review`로 바꿨다. 또 F는 근거 검증 버그(속성 정의 텍스트를 원문으로 취급)를 고친 뒤 다시 돌린 결과다.

Docling 처리 시간은 모델 로딩 후 문서당 약 3~8초(CPU), pdfplumber는 0.1초 미만.
`effort=low`에서 응답의 reasoning 토큰은 0으로 보고됐다.

## 알려진 한계

- **한국어 스캔 OCR 품질**: RapidOCR 한국어 모델이 짧은 단어(`내경`, `외경`, `항목`)와 큰 제목을 놓친다.
  `FULL_PAGE`·`medium`·`scale=4`를 비교했고 `DEFAULT` 모드가 가장 나았지만 완전하지 않다.
  스캔 문서가 데모 범위에 들어가면 Tesseract(kor)나 다른 OCR 엔진 비교가 필요하다.
- 품질 검사는 원본 정답 없이 동작하므로 놓치는 오류가 있다(이미지로 붙은 표가 대표적).
  정답셋 평가는 `data/eval/gold.json`의 가상 문서 8건뿐이라 설계안 15절의 20~30건 실제 유사 문서가 필요하다.
- 문서 하나에 제품 하나를 가정한다 (BOM처럼 여러 품목이 든 문서는 범위 밖).
- 07 담당자 검토는 `next_node="07_review"` 표시만 하고 `interrupt()`는 통합 그래프에서 붙인다.
- 온톨로지는 임시 JSON이다. 실제 IRI·필수 속성은 온톨로지 담당과 맞춘 뒤 교체한다.
