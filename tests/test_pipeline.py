"""LLM 없이 검증·재시도 흐름을 확인하는 테스트.  uv run --with pytest pytest -q"""

from __future__ import annotations

import extractor.graph as graph_mod
from extractor.graph import build_graph, initial_state
from extractor.schemas import (
    Block,
    ClassMappingResult,
    Evidence,
    ExtractedAttribute,
    ModelExecutionInfo,
    ParsedDocument,
    PropertyExtractionResult,
)
from extractor.validators import validate_class_mapping, validate_extraction

DOC = ParsedDocument(
    doc_id="DOC-T", source_file="t.pdf", file_type="pdf", parser="docling", status="success",
    blocks=[
        Block(block_id="B1", page=1, type="heading", text="DM-600 BLDC 모터\n제조사: XYZ Motors"),
        Block(block_id="B2", page=2, type="table", text="정격 전압 | 24 V\n정격 출력 | 0.6 kW",
              rows=[["정격 전압", "24 V"], ["정격 출력", "0.6 kW"]]),
    ],
)
PROPS = [
    {"id": "ex:manufacturer", "label": "제조사", "required": False},
    {"id": "ex:ratedVoltage", "label": "정격 전압", "required": True},
    {"id": "ex:ratedPower", "label": "정격 출력", "required": True},
    {"id": "ex:weight", "label": "중량", "required": False},
]
CANDS = [{"id": "ex:Motor"}, {"id": "ex:Bolt"}]


def attr(pid, v, bid=None, page=None, text=None, unit=None):
    ev = Evidence(block_id=bid, page=page, text=text) if bid else None
    return ExtractedAttribute(property_id=pid, raw_value=v, raw_unit=unit, evidence=ev)


def codes(issues):
    return {(i.code, i.property_id) for i in issues}


def test_class_not_in_candidates_and_bad_evidence():
    r = ClassMappingResult(class_id="ex:Pump", reason="", needs_review=False,
                           evidence=Evidence(block_id="B1", page=2, text="펌프"))
    c = {i.code for i in validate_class_mapping(r, CANDS, DOC)}
    assert {"CLASS_NOT_IN_CANDIDATES", "EVIDENCE_PAGE_MISMATCH", "EVIDENCE_TEXT_MISMATCH"} <= c


def test_extraction_checks():
    r = PropertyExtractionResult(class_id="ex:Motor", needs_review=False, warnings=[], attributes=[
        attr("ex:manufacturer", "XYZ Motors", "B1", 1, "제조사: XYZ Motors"),
        attr("ex:ratedVoltage", "24 V", "B2", 2, "정격 전압 24 V"),        # 공백·구분자 차이는 허용
        attr("ex:ratedPower", "600 W", "B2", 2, "정격 출력 | 0.6 kW"),     # 단위 변환한 값 → 오류
        attr("ex:color", "black", "B1", 1, "DM-600"),                       # 요청하지 않은 속성
    ])
    c = codes(validate_extraction(r, PROPS, DOC, "ex:Motor"))
    assert ("VALUE_NOT_IN_EVIDENCE", "ex:ratedPower") in c
    assert ("UNKNOWN_PROPERTY", "ex:color") in c
    assert ("PROPERTY_NOT_RETURNED", "ex:weight") in c
    assert not any(pid == "ex:ratedVoltage" for _, pid in c)


def test_missed_value_vs_absent_value():
    r = PropertyExtractionResult(class_id="ex:Motor", needs_review=False, warnings=[], attributes=[
        attr("ex:manufacturer", "XYZ Motors", "B1", 1, "제조사: XYZ Motors"),
        attr("ex:ratedVoltage", "24 V", "B2", 2, "정격 전압 | 24 V"),
        attr("ex:ratedPower", None),   # 원문에 '정격 출력'이 있음 → 재시도 대상
        attr("ex:weight", None),       # 원문에 없음 → 정상 null
    ])
    issues = validate_extraction(r, PROPS, DOC, "ex:Motor")
    assert codes(issues) == {("VALUE_POSSIBLY_MISSED", "ex:ratedPower")}
    assert issues[0].retryable


def test_graph_retries_only_failed_property(monkeypatch):
    calls = []

    def fake_call(*, node, schema, payload, **kw):
        calls.append((node, payload))
        info = ModelExecutionInfo(node=node, model="fake", reasoning_effort=None, latency_ms=1)
        if schema is ClassMappingResult:
            return ClassMappingResult(class_id="ex:Motor", reason="모터", needs_review=False,
                                      evidence=Evidence(block_id="B1", page=1, text="DM-600 BLDC 모터")), info
        if "retry_context" not in payload:
            return PropertyExtractionResult(class_id="ex:Motor", needs_review=False, warnings=[], attributes=[
                attr("ex:manufacturer", "XYZ Motors", "B1", 1, "제조사: XYZ Motors"),
                attr("ex:ratedVoltage", "24 V", "B2", 2, "정격 전압 | 24 V"),
                attr("ex:ratedPower", None),
                attr("ex:weight", None),
            ]), info
        assert [p["id"] for p in payload["properties"]] == ["ex:ratedPower"]
        return PropertyExtractionResult(class_id="ex:Motor", needs_review=False, warnings=[], attributes=[
            attr("ex:ratedPower", "0.6 kW", "B2", 2, "정격 출력 | 0.6 kW"),
        ]), info

    class FakeOntology:
        def class_candidates(self):
            return CANDS

        def properties_of(self, cid):
            return PROPS

    monkeypatch.setattr(graph_mod, "call_structured", fake_call)
    final = build_graph(FakeOntology()).invoke(initial_state("t.pdf", parsed=DOC))
    assert [n for n, _ in calls] == ["02_class_mapping", "03_property_extraction", "03_property_extraction_retry"]
    vals = {a["property_id"]: a["raw_value"] for a in final["extraction"]["attributes"]}
    assert vals == {"ex:manufacturer": "XYZ Motors", "ex:ratedVoltage": "24 V",
                    "ex:ratedPower": "0.6 kW", "ex:weight": None}
    assert final["status"] == "extracted" and final["next_node"] == "04_normalize"


def test_graph_stops_at_review_when_class_unclear(monkeypatch):
    def fake_call(*, node, **kw):
        return ClassMappingResult(class_id=None, reason="근거 부족", needs_review=True, evidence=None), \
            ModelExecutionInfo(node=node, model="fake", reasoning_effort=None, latency_ms=1)

    monkeypatch.setattr(graph_mod, "call_structured", fake_call)
    final = build_graph().invoke(initial_state("t.pdf", parsed=DOC))
    assert final.get("extraction") is None
    assert final["next_node"] == "07_review"
