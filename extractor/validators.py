"""02·03 LLM 출력의 의미 검증 (설계안 9.4, 11.4, 12.6).

Structured Outputs는 형식만 보장하므로 ID·근거·누락 여부는 코드로 다시 확인한다.
"""

from __future__ import annotations

import re
import unicodedata

from .schemas import (
    ClassMappingResult,
    Evidence,
    ParsedDocument,
    PropertyExtractionResult,
    ValidationIssue,
)


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", s)).strip().casefold()


def _loose(s: str) -> str:
    """공백·표 구분자 차이를 무시한 비교용 문자열."""
    return re.sub(r"[\s|:：]+", "", unicodedata.normalize("NFKC", s)).casefold()


def text_contains(haystack: str, needle: str) -> bool:
    return _norm(needle) in _norm(haystack) or _loose(needle) in _loose(haystack)


def check_evidence(
    ev: Evidence | None, doc: ParsedDocument, property_id: str | None = None
) -> list[ValidationIssue]:
    if ev is None:
        return []
    block = doc.block_map().get(ev.block_id)
    if block is None:
        return [
            ValidationIssue(
                code="EVIDENCE_BLOCK_NOT_FOUND",
                message=f"근거 블록 {ev.block_id}가 파싱 결과에 없음",
                property_id=property_id,
                retryable=True,
            )
        ]
    issues = []
    if block.page != ev.page:
        issues.append(
            ValidationIssue(
                code="EVIDENCE_PAGE_MISMATCH",
                message=f"{ev.block_id}는 {block.page}페이지인데 {ev.page}페이지로 반환됨",
                property_id=property_id,
                retryable=True,
            )
        )
    if not text_contains(block.text, ev.text):
        issues.append(
            ValidationIssue(
                code="EVIDENCE_TEXT_MISMATCH",
                message=f"근거 문구가 {ev.block_id} 원문에 없음: {ev.text!r}",
                property_id=property_id,
                retryable=True,
            )
        )
    return issues


def validate_class_mapping(
    result: ClassMappingResult, candidates: list[dict], doc: ParsedDocument
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    candidate_ids = {c["id"] for c in candidates}
    if result.class_id is None:
        issues.append(ValidationIssue(code="CLASS_UNDECIDED", message="적합한 클래스를 판단하지 못함"))
    elif result.class_id not in candidate_ids:
        issues.append(
            ValidationIssue(
                code="CLASS_NOT_IN_CANDIDATES",
                message=f"후보에 없는 클래스 반환: {result.class_id}",
            )
        )
    if result.class_id is not None and result.evidence is None:
        issues.append(ValidationIssue(code="EVIDENCE_MISSING", message="클래스 선택 근거가 없음"))
    issues += check_evidence(result.evidence, doc)
    return issues


def _mentions(doc_text_norm: str, term: str) -> bool:
    term_n = _norm(term)
    if len(term_n) < 2:
        return False
    if term_n.isascii():
        return re.search(rf"(?<![a-z0-9]){re.escape(term_n)}(?![a-z0-9])", doc_text_norm) is not None
    return term_n in doc_text_norm


def validate_extraction(
    result: PropertyExtractionResult,
    properties: list[dict],
    doc: ParsedDocument,
    class_id: str,
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    prop_defs = {p["id"]: p for p in properties}

    if result.class_id != class_id:
        issues.append(
            ValidationIssue(
                code="CLASS_ID_CHANGED",
                message=f"요청 클래스 {class_id}와 다른 {result.class_id} 반환",
            )
        )

    seen: dict[str, int] = {}
    for a in result.attributes:
        seen[a.property_id] = seen.get(a.property_id, 0) + 1
    for pid in prop_defs.keys() - seen.keys():
        issues.append(
            ValidationIssue(
                code="PROPERTY_NOT_RETURNED",
                message="요청한 속성이 결과에 없음",
                property_id=pid,
                retryable=True,
            )
        )
    for pid in seen.keys() - prop_defs.keys():
        issues.append(
            ValidationIssue(
                code="UNKNOWN_PROPERTY",
                message="요청하지 않은 속성 ID가 추가됨",
                property_id=pid,
            )
        )
    for pid, n in seen.items():
        if n > 1:
            issues.append(
                ValidationIssue(
                    code="DUPLICATE_PROPERTY",
                    message=f"같은 속성이 {n}번 반환됨",
                    property_id=pid,
                    retryable=True,
                )
            )

    doc_text = _norm("\n".join(b.text for b in doc.blocks))
    for a in result.attributes:
        p = prop_defs.get(a.property_id)
        if p is None:
            continue
        if a.raw_value is None:
            if a.evidence is not None:
                issues.append(
                    ValidationIssue(
                        code="EVIDENCE_WITHOUT_VALUE",
                        message="값은 null인데 근거가 반환됨",
                        property_id=a.property_id,
                    )
                )
            terms = [p["label"], *p.get("synonyms", [])]
            if any(_mentions(doc_text, t) for t in terms):
                issues.append(
                    ValidationIssue(
                        code="VALUE_POSSIBLY_MISSED",
                        message="원문에 속성명이 보이지만 값이 추출되지 않음",
                        property_id=a.property_id,
                        retryable=True,
                    )
                )
            elif p.get("required"):
                issues.append(
                    ValidationIssue(
                        code="REQUIRED_VALUE_MISSING",
                        message="필수 속성값이 원문에서 확인되지 않음",
                        property_id=a.property_id,
                    )
                )
            continue

        if a.evidence is None:
            issues.append(
                ValidationIssue(
                    code="EVIDENCE_MISSING",
                    message="값은 있는데 근거가 없음",
                    property_id=a.property_id,
                    retryable=True,
                )
            )
            continue
        ev_issues = check_evidence(a.evidence, doc, a.property_id)
        issues += ev_issues
        if not ev_issues and not text_contains(a.evidence.text, a.raw_value):
            issues.append(
                ValidationIssue(
                    code="VALUE_NOT_IN_EVIDENCE",
                    message=f"값 {a.raw_value!r}가 근거 문구에 없음 (추정·변환 의심)",
                    property_id=a.property_id,
                    retryable=True,
                )
            )
        block = doc.block_map().get(a.evidence.block_id)
        if a.raw_unit and block and not text_contains(block.text, a.raw_unit):
            issues.append(
                ValidationIssue(
                    code="UNIT_NOT_IN_SOURCE",
                    message=f"단위 {a.raw_unit!r}가 근거 블록에 없음",
                    property_id=a.property_id,
                    retryable=True,
                )
            )
    return issues
