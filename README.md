# KAN-65 문서 파싱 · LLM 추출 (01 ~ 03)

설계안 「문서 파싱 및 LLM 연계 설계 검토 결과」의 **01 문서 파싱 · 02 클래스 매핑 · 03 속성 추출**을 구현한 코드.
04 정규화 이후는 범위 밖이며, 결과 State의 `next_node`(`04_normalize` / `07_review`)로 넘겨줄 곳만 표시한다.

- 파싱: Docling(코드 실행) + 규칙 기반 대체 전략(OCR / 표 설정 변경 / pdfplumber), Excel은 openpyxl
- LLM: LangChain `ChatOpenAI` + Structured Outputs(JSON Schema strict), 모델 `gpt-6-luna`
- 흐름 제어: LangGraph `StateGraph`
- 육안 확인: Streamlit 뷰어 (페이지 위 블록 bbox, 파서 비교, 추출 근거 위치)

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
uv run --with pytest pytest -q
```

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
| `app.py` | Streamlit 뷰어 | — |

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

## 샘플 실행 결과 (2026-10-26, gpt-6-luna, effort low)

가상 사양서 5건, 실측값이며 평가셋 규모의 결과는 아니다.

| 문서 | 파싱 | 02 | 03 | 상태 | 비용 |
|---|---|---|---|---|---|
| motor_spec.pdf (텍스트+표 2개, 2쪽) | docling, 포함률 100% | Motor | 7/7 정답, 정격·최대 토크 구분 | extracted | $0.0007 |
| bolt_spec.pdf (병합 머리글 표) | docling | Bolt | 8/8, 길이 `30` + `mm` | extracted | $0.0010 |
| cable_spec.pdf (문서 내 지시문 포함) | docling | Cable | 지시문 무시하고 `450/750 V`, 심선 수는 `단심`(04에서 1로 정규화 필요) | extracted | $0.0007 |
| bearing_scan.pdf (텍스트 레이어 없음) | docling 실패 → docling_ocr | Bearing | 모델명 OCR 누락 → null | needs_review | $0.0006 |
| servo_motor_spec.xlsx | excel | Motor | 7/7, 단위는 `raw_unit` | extracted | $0.0006 |

Docling 처리 시간은 모델 로딩 후 문서당 약 3~8초(CPU), pdfplumber는 0.1초 미만.
`effort=low`에서 응답의 reasoning 토큰은 0으로 보고됐다.

## 알려진 한계

- **한국어 스캔 OCR 품질**: RapidOCR 한국어 모델이 짧은 단어(`내경`, `외경`, `항목`)와 큰 제목을 놓친다.
  `FULL_PAGE`·`medium`·`scale=4`를 비교했고 `DEFAULT` 모드가 가장 나았지만 완전하지 않다.
  스캔 문서가 데모 범위에 들어가면 Tesseract(kor)나 다른 OCR 엔진 비교가 필요하다.
- 품질 검사는 원본 정답 없이 동작하므로 놓치는 오류가 있다 → 정답셋 기반 평가(설계안 15절)는 아직 없음.
- 문서 하나에 제품 하나를 가정한다 (BOM처럼 여러 품목이 든 문서는 범위 밖).
- 07 담당자 검토는 `next_node="07_review"` 표시만 하고 `interrupt()`는 통합 그래프에서 붙인다.
- 온톨로지는 임시 JSON이다. 실제 IRI·필수 속성은 온톨로지 담당과 맞춘 뒤 교체한다.
