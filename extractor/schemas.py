"""노드 간 데이터 계약 (설계안 16절).

- ParsedDocument: 01 파싱 출력 (파서 고유 객체 대신 공통 JSON)
- ClassMappingResult: 02 LLM 출력
- PropertyExtractionResult: 03 LLM 출력
- RetryContext: 03 재시도 입력
- ModelExecutionInfo: 모델 호출 기록
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

BlockType = Literal["text", "heading", "list_item", "table", "caption", "other"]
Severity = Literal["info", "warning", "error"]


# ---------------------------------------------------------------------------
# 01 문서 파싱
# ---------------------------------------------------------------------------


class Block(BaseModel):
    block_id: str
    page: int = Field(description="1부터 시작. Excel은 시트 순번")
    type: BlockType
    text: str
    rows: list[list[str]] | None = Field(default=None, description="표일 때만 존재")
    bbox: list[float] | None = Field(
        default=None, description="[left, top, right, bottom], 좌상단 원점, PDF pt 단위"
    )
    label: str | None = Field(default=None, description="파서가 붙인 원래 라벨")
    sheet: str | None = None


class ParseWarning(BaseModel):
    code: str
    message: str
    severity: Severity = "warning"
    page: int | None = None
    block_id: str | None = None


class PageInfo(BaseModel):
    page: int
    width: float | None = None
    height: float | None = None
    text_layer_chars: int | None = Field(default=None, description="PDF 텍스트 레이어 글자 수")
    coverage: float | None = Field(
        default=None, description="텍스트 레이어 토큰 중 파싱 결과에 포함된 비율"
    )


class ParsedDocument(BaseModel):
    doc_id: str
    source_file: str
    file_type: Literal["pdf", "xlsx"]
    parser: str = Field(description="사용한 파싱 전략 이름")
    status: Literal["success", "degraded", "failed"]
    pages: list[PageInfo] = []
    blocks: list[Block] = []
    warnings: list[ParseWarning] = []
    elapsed_ms: int | None = None
    from_cache: bool = False

    def block_map(self) -> dict[str, Block]:
        return {b.block_id: b for b in self.blocks}

    def llm_view(self) -> dict:
        """LLM 입력용으로 bbox·라벨 등 불필요한 정보를 뺀 형태."""
        return {
            "doc_id": self.doc_id,
            "source_file": self.source_file,
            "blocks": [
                {"block_id": b.block_id, "page": b.page, "type": b.type, "text": b.text}
                for b in self.blocks
            ],
        }


# ---------------------------------------------------------------------------
# 02 · 03 LLM 출력 (Structured Outputs 스키마로도 사용)
# ---------------------------------------------------------------------------


class Evidence(BaseModel):
    block_id: str
    page: int
    text: str = Field(description="원문 그대로의 근거 문구")


class ClassMappingResult(BaseModel):
    class_id: str | None
    reason: str
    evidence: Evidence | None
    needs_review: bool


class ExtractedAttribute(BaseModel):
    property_id: str
    raw_value: str | None
    raw_unit: str | None = Field(
        description="값과 떨어진 열·표 머리글에 단위가 있을 때만 원문 그대로. 값에 단위가 붙어 있으면 null"
    )
    evidence: Evidence | None


class PropertyExtractionResult(BaseModel):
    class_id: str
    attributes: list[ExtractedAttribute]
    needs_review: bool
    warnings: list[str]


# ---------------------------------------------------------------------------
# 코드 검증 · 재시도 · 실행 기록
# ---------------------------------------------------------------------------


class ValidationIssue(BaseModel):
    code: str
    message: str
    property_id: str | None = None
    retryable: bool = False


class RetryContext(BaseModel):
    failed_property_ids: list[str]
    errors: list[dict]
    instruction: str = (
        "해당 속성값이 원문에 실제로 있는지 다시 확인한다. 값이 없으면 null을 유지한다."
    )


class ModelExecutionInfo(BaseModel):
    node: str
    model: str
    reasoning_effort: str | None
    input_tokens: int | None = None
    output_tokens: int | None = None
    reasoning_tokens: int | None = None
    latency_ms: int
    cost_usd: float | None = None
    error: str | None = None
