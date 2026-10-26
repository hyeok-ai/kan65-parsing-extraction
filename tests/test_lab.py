"""실험 평가·에이전트 결과 병합/검증 테스트 (LLM 호출 없음)."""

from __future__ import annotations

from extractor.agent import AgentAttribute, AgentEvidence, AgentResult, validate_agent_result
from extractor.agent_tools import DocumentWorkspace
from extractor.lab import _merge_fallback, evaluate
from extractor.schemas import Block, ParsedDocument

GOLD = {"class_id": "ex:Motor", "attributes": {
    "ex:ratedVoltage": ["24 V"], "ex:ratedSpeed": ["3,000 rpm"], "ex:ratedTorque": ["1.91 N·m"],
    "ex:weight": [None]}}


def rec(class_id="ex:Motor", status="extracted", **vals):
    return {"class_id": class_id, "status": status, "issues": [],
            "attributes": [{"property_id": k, "raw_value": v[0], "raw_unit": v[1], "evidence": None}
                           for k, v in vals.items()]}


def test_evaluate_normalizes_and_counts():
    r = rec(**{"ex:ratedVoltage": ("24", "V"), "ex:ratedSpeed": ("3000 rpm", None),
               "ex:ratedTorque": ("1.91 N-m", None), "ex:weight": ("1 kg", None)})
    ev = evaluate(r, GOLD)
    assert ev["correct"] == 3 and ev["hallucinated"] == 1
    assert ev["false_accept"]  # 자동 통과했지만 틀린 값이 있음


def test_evaluate_wrong_class_scores_zero():
    ev = evaluate(rec(class_id="ex:Bolt", **{"ex:ratedVoltage": ("24 V", None)}), GOLD)
    assert not ev["class_correct"] and ev["correct"] == 1  # weight=null만 '맞음'으로 남는다
    assert not ev["fully_correct"]


def test_merge_fallback_replaces_only_targets():
    final = {"class_result": {"class_id": "ex:Motor", "reason": "r", "needs_review": False,
                              "evidence": {"block_id": "B1", "page": 1, "text": "모터"}},
             "extraction": {"attributes": [
                 {"property_id": "ex:ratedVoltage", "raw_value": "24 V", "raw_unit": None,
                  "evidence": {"block_id": "B2", "page": 1, "text": "정격 전압 | 24 V"}},
                 {"property_id": "ex:weight", "raw_value": None, "raw_unit": None, "evidence": None}]}}
    agent = AgentResult(class_id="ex:Bolt", class_reason="x", class_evidence=None, needs_review=False, warnings=[],
                        attributes=[AgentAttribute(property_id="ex:weight", raw_value="2.4 kg", raw_unit=None,
                                                   evidence=AgentEvidence(source="docling_ocr", block_id="B4", page=1, text="2.4 kg")),
                                    AgentAttribute(property_id="ex:ratedVoltage", raw_value="99 V", raw_unit=None, evidence=None)])
    merged = _merge_fallback("docling", final, agent, ["ex:weight"], class_undecided=False)
    vals = {a.property_id: a.raw_value for a in merged.attributes}
    assert merged.class_id == "ex:Motor"                      # 클래스는 baseline 유지
    assert vals == {"ex:ratedVoltage": "24 V", "ex:weight": "2.4 kg"}  # 대상 외 속성은 덮어쓰지 않음
    assert merged.attributes[0].evidence.source == "docling"


def test_agent_evidence_checked_against_named_strategy(tmp_path):
    ws = DocumentWorkspace(tmp_path / "x.pdf")
    ws.docs["docling_ocr"] = ParsedDocument(
        doc_id="D", source_file="x.pdf", file_type="pdf", parser="docling_ocr", status="success",
        blocks=[Block(block_id="B1", page=1, type="table", text="중량 | 2.4 kg")])
    res = AgentResult(class_id="ex:Motor", class_reason="r", needs_review=False, warnings=[],
                      class_evidence=AgentEvidence(source="docling_ocr", block_id="B1", page=1, text="중량"),
                      attributes=[AgentAttribute(property_id="ex:weight", raw_value="2.4 kg", raw_unit=None,
                                                 evidence=AgentEvidence(source="docling_ocr", block_id="B1", page=1, text="중량 | 2.4 kg")),
                                  AgentAttribute(property_id="ex:ratedPower", raw_value="90 W", raw_unit=None,
                                                 evidence=AgentEvidence(source="docling_ocr", block_id="B1", page=1, text="정격 출력 | 90 W"))])
    _, attrs, issues = validate_agent_result(res, ws)
    codes = {(i.code, i.property_id) for i in issues}
    assert attrs[0].evidence.block_id == "docling_ocr:B1"
    assert ("EVIDENCE_TEXT_MISMATCH", "ex:ratedPower") in codes   # 원문에 없는 근거는 잡힌다
    assert not any(pid == "ex:weight" for _, pid in codes)


def test_property_definitions_are_not_treated_as_document_text(tmp_path):
    ws = DocumentWorkspace(tmp_path / "x.pdf")
    ws.docs["docling"] = ParsedDocument(
        doc_id="D", source_file="x.pdf", file_type="pdf", parser="docling", status="success",
        blocks=[Block(block_id="B1", page=1, type="text", text="DM-600 모터 정격 전압 24 V")])
    # MCP로 받은 속성 정의에는 '중량' 같은 속성명이 들어 있다.
    ws.trace.append({"tool": "get_class_properties", "args": {}, "output_full": '[{"label": "중량"}]', "via": "mcp"})
    ev = AgentEvidence(source="docling", block_id="B1", page=1, text="정격 전압 24 V")
    res = AgentResult(class_id="ex:Motor", class_reason="r", needs_review=False, warnings=[],
                      class_evidence=AgentEvidence(source="docling", block_id="B1", page=1, text="DM-600 모터"),
                      attributes=[AgentAttribute(property_id="ex:ratedVoltage", raw_value="24 V", raw_unit=None, evidence=ev),
                                  AgentAttribute(property_id="ex:weight", raw_value=None, raw_unit=None, evidence=None)])
    _, _, issues = validate_agent_result(res, ws)
    assert ("VALUE_POSSIBLY_MISSED", "ex:weight") not in {(i.code, i.property_id) for i in issues}
