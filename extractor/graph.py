"""01~03 LangGraph 서브그래프.

parse ──(대체 전략 있음)──▶ parse
  │  └─(복구 불가)──────────▶ finalize (07 검토 대상)
  ▼
class_mapping ──(판단 불가/불확실)──▶ finalize (07 검토 대상)
  ▼
property_extraction ──(재시도 가능 오류, 횟수 이내)──▶ property_extraction
  ▼
finalize ──▶ END   (다음 단계: 04 정규화 또는 07 담당자 검토)

04~08은 범위 밖이므로 finalize는 다음 노드 이름만 next_node로 남긴다.
"""

from __future__ import annotations

import operator
from dataclasses import asdict
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph

from .llm import LLMConfig, call_structured
from .ontology import JsonOntology, OntologySource
from .parsing import initial_strategy, next_strategy, parse_with_strategy, score
from .prompts import CLASS_MAPPING_SYSTEM, PROPERTY_EXTRACTION_SYSTEM
from .schemas import (
    ClassMappingResult,
    ParsedDocument,
    PropertyExtractionResult,
    RetryContext,
)
from .validators import validate_class_mapping, validate_extraction

REVIEW_CLASS_CODES = {"CLASS_UNDECIDED", "CLASS_NOT_IN_CANDIDATES", "LLM_ERROR"}


class PipelineState(TypedDict, total=False):
    source_path: str
    config: dict
    # 01
    parse_strategy: str | None
    parse_attempts: Annotated[list[dict], operator.add]
    parsed: dict
    # 02
    class_candidates: list[dict]
    class_result: dict | None
    class_issues: list[dict]
    # 03
    properties: list[dict]
    extraction: dict | None
    extraction_issues: list[dict]
    extraction_attempts: int
    retry_context: dict | None
    # 공통
    model_runs: Annotated[list[dict], operator.add]
    review_reasons: Annotated[list[str], operator.add]
    status: str
    next_node: str


def build_graph(ontology: OntologySource | None = None):
    ontology = ontology or JsonOntology()

    # ---------------------------------------------------------------- 01
    def parse(state: PipelineState) -> dict:
        path = state["source_path"]
        strategy = state.get("parse_strategy") or initial_strategy(path)
        doc = parse_with_strategy(path, strategy)
        attempts = [*state.get("parse_attempts", []), doc.model_dump()]
        docs = [ParsedDocument.model_validate(a) for a in attempts]
        best = max(docs, key=score)
        return {
            "parse_attempts": [doc.model_dump()],
            "parsed": best.model_dump(),
            "parse_strategy": next_strategy(doc, [d.parser for d in docs]),
        }

    def after_parse(state: PipelineState) -> str:
        if state.get("parse_strategy"):
            return "parse"
        return "finalize" if state["parsed"]["status"] == "failed" else "class_mapping"

    # ---------------------------------------------------------------- 02
    def class_mapping(state: PipelineState) -> dict:
        cfg = LLMConfig(**state.get("config", {}))
        doc = ParsedDocument.model_validate(state["parsed"])
        candidates = ontology.class_candidates()
        result, info = call_structured(
            node="02_class_mapping",
            schema=ClassMappingResult,
            system=CLASS_MAPPING_SYSTEM,
            payload={"task": "class_mapping", "document": doc.llm_view(), "class_candidates": candidates},
            model=cfg.model, effort=cfg.class_effort, timeout=cfg.timeout,
        )
        update: dict = {"class_candidates": candidates, "model_runs": [info.model_dump()]}
        if result is None:
            update |= {"class_result": None,
                       "class_issues": [{"code": "LLM_ERROR", "message": info.error or "응답 없음"}]}
            return update
        issues = validate_class_mapping(result, candidates, doc)
        update |= {"class_result": result.model_dump(), "class_issues": [i.model_dump() for i in issues]}
        return update

    def after_class(state: PipelineState) -> str:
        res = state.get("class_result")
        codes = {i["code"] for i in state.get("class_issues", [])}
        if res is None or res["needs_review"] or codes & REVIEW_CLASS_CODES:
            return "finalize"
        return "property_extraction"

    # ---------------------------------------------------------------- 03
    def property_extraction(state: PipelineState) -> dict:
        cfg = LLMConfig(**state.get("config", {}))
        doc = ParsedDocument.model_validate(state["parsed"])
        class_id = state["class_result"]["class_id"]
        all_props = ontology.properties_of(class_id)
        retry = state.get("retry_context")
        prev = state.get("extraction")

        props = all_props
        payload: dict = {"task": "property_extraction", "class_id": class_id}
        if retry and prev:
            props = [p for p in all_props if p["id"] in retry["failed_property_ids"]]
            payload["retry_context"] = retry
        payload |= {"properties": props, "document": doc.llm_view()}

        result, info = call_structured(
            node="03_property_extraction" + ("_retry" if retry else ""),
            schema=PropertyExtractionResult,
            system=PROPERTY_EXTRACTION_SYSTEM,
            payload=payload,
            model=cfg.model, effort=cfg.extract_effort, timeout=cfg.timeout,
        )
        attempts = state.get("extraction_attempts", 0) + 1
        update: dict = {"properties": all_props, "extraction_attempts": attempts,
                        "model_runs": [info.model_dump()]}
        if result is None:
            if prev is None:
                update |= {"extraction": None, "retry_context": None,
                           "extraction_issues": [{"code": "LLM_ERROR", "message": info.error or "응답 없음"}]}
            return update

        if retry and prev:  # 실패 속성만 새 결과로 교체 (중복 반환분도 하나로 정리)
            merged = PropertyExtractionResult.model_validate(prev)
            failed = set(retry["failed_property_ids"])
            by_id = {a.property_id: a for a in reversed(merged.attributes)}
            by_id |= {a.property_id: a for a in result.attributes if a.property_id in failed}
            order = {p["id"]: i for i, p in enumerate(all_props)}
            merged.attributes = sorted(by_id.values(), key=lambda a: order.get(a.property_id, len(order)))
            merged.needs_review = merged.needs_review or result.needs_review
            merged.warnings = [*merged.warnings, *result.warnings]
            result = merged

        issues = validate_extraction(result, all_props, doc, class_id)
        retry_ids = sorted({i.property_id for i in issues if i.retryable and i.property_id})
        update |= {
            "extraction": result.model_dump(),
            "extraction_issues": [i.model_dump() for i in issues],
            "retry_context": RetryContext(
                failed_property_ids=retry_ids,
                errors=[{"code": i.code, "property_id": i.property_id, "message": i.message}
                        for i in issues if i.retryable],
            ).model_dump() if retry_ids and attempts <= cfg.max_extraction_retries else None,
        }
        return update

    def after_extraction(state: PipelineState) -> str:
        return "property_extraction" if state.get("retry_context") else "finalize"

    # ---------------------------------------------------------------- 종료
    def finalize(state: PipelineState) -> dict:
        reasons: list[str] = []
        parsed = state.get("parsed") or {}
        if parsed.get("status") == "failed":
            reasons.append("01 파싱 복구 실패")
        elif parsed.get("status") == "degraded":
            reasons.append("01 파싱 품질 경고 있음")
        cr = state.get("class_result")
        if "class_result" in state and (cr is None or cr.get("class_id") is None):
            reasons.append("02 클래스 판단 불가")
        elif cr and cr.get("needs_review"):
            reasons.append("02 LLM이 검토 필요로 표시")
        for i in state.get("class_issues", []):
            reasons.append(f"02 {i['code']}: {i['message']}")
        ex = state.get("extraction")
        if ex and ex.get("needs_review"):
            reasons.append("03 LLM이 검토 필요로 표시 (값 충돌 등)")
        for i in state.get("extraction_issues", []):
            reasons.append(f"03 {i['code']} [{i.get('property_id')}]: {i['message']}")

        blocking = parsed.get("status") == "failed" or not ex
        return {
            "review_reasons": reasons,
            "status": "failed" if blocking else ("needs_review" if reasons else "extracted"),
            "next_node": "07_review" if reasons else "04_normalize",
        }

    g = StateGraph(PipelineState)
    g.add_node("parse", parse)
    g.add_node("class_mapping", class_mapping)
    g.add_node("property_extraction", property_extraction)
    g.add_node("finalize", finalize)
    # parsed를 넘기면(뷰어에서 이미 파싱한 경우) 01을 건너뛴다.
    g.add_conditional_edges(START, lambda s: after_parse(s) if s.get("parsed") else "parse",
                            ["parse", "class_mapping", "finalize"])
    g.add_conditional_edges("parse", after_parse, ["parse", "class_mapping", "finalize"])
    g.add_conditional_edges("class_mapping", after_class, ["property_extraction", "finalize"])
    g.add_conditional_edges("property_extraction", after_extraction, ["property_extraction", "finalize"])
    g.add_edge("finalize", END)
    return g.compile()


def initial_state(
    path: str, config: LLMConfig | None = None, parsed: ParsedDocument | None = None
) -> PipelineState:
    state: PipelineState = {"source_path": str(path), "config": asdict(config or LLMConfig())}
    if parsed is not None:
        state["parsed"] = parsed.model_dump()
    return state
