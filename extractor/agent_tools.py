"""에이전트가 쓰는 문서 도구 (설계안 7.2: 제공 도구와 호출 횟수를 제한).

같은 함수를 LangChain 도구(인프로세스)와 자체 MCP 서버(extractor.mcp_server) 양쪽에서 쓴다.
도구 출력은 모두 문자열이며, 토큰을 아끼도록 길이를 제한한다.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from .ontology import JsonOntology, OntologySource
from .parsing import PDF_STRATEGIES, parse_cached
from .schemas import Block, ParsedDocument
from .validators import _loose, _norm

STRATEGY_HELP = {
    "docling": "Docling 기본(OCR 끔). 텍스트 레이어가 있는 PDF에 적합, 문서당 수 초",
    "docling_ocr": "Docling + RapidOCR(한국어). 스캔·이미지 속 글자까지 읽음. 느리고 짧은 단어를 놓칠 수 있음",
    "docling_table_alt": "Docling, 표 셀 매칭 끔. 표 구조가 깨질 때 대안",
    "pdfplumber": "선으로 그린 표와 텍스트 줄을 빠르게 추출. 레이아웃 분석은 단순",
    "excel": "Excel 전용(openpyxl)",
}
FULL_TEXT_BUDGET = 6000  # parse_document가 블록 전문을 그대로 돌려주는 최대 글자 수
CACHE_ENV = "EXTRACT_PARSE_CACHE"  # "0"이면 파싱 캐시를 읽지 않는다


class DocumentWorkspace:
    """한 문서에 대한 파싱 결과와 도구 호출 기록."""

    def __init__(self, path: str | Path, ontology: OntologySource | None = None, use_cache: bool | None = None):
        self.path = Path(path)
        self.is_excel = self.path.suffix.lower() in (".xlsx", ".xlsm")
        self.ontology = ontology or JsonOntology()
        # MCP 서버 프로세스에는 환경변수로 전달된다.
        self.use_cache = os.environ.get(CACHE_ENV, "1") != "0" if use_cache is None else use_cache
        self.docs: dict[str, ParsedDocument] = {}
        self.trace: list[dict] = []

    @property
    def strategies(self) -> list[str]:
        return ["excel"] if self.is_excel else PDF_STRATEGIES

    def get_doc(self, strategy: str, read_cache: bool | None = None) -> ParsedDocument:
        if strategy not in self.docs:
            read = self.use_cache if read_cache is None else read_cache
            self.docs[strategy] = parse_cached(self.path, strategy, read=read)
        return self.docs[strategy]

    def _record(self, name: str, args: dict, out: str, t0: float, extra: dict | None = None) -> str:
        self.trace.append({"tool": name, "args": args, "output_chars": len(out),
                           "ms": int((time.perf_counter() - t0) * 1000), "output_preview": out[:300],
                           **(extra or {})})
        return out

    # ------------------------------------------------------------------ 도구
    def parse_document(self, strategy: str) -> str:
        t0 = time.perf_counter()
        if strategy not in self.strategies:
            return self._record("parse_document", {"strategy": strategy},
                                f"사용할 수 없는 전략. 가능: {self.strategies}", t0)
        load = "reused" if strategy in self.docs else "new"
        doc = self.get_doc(strategy)
        # 첫 줄의 parse_ms·cache·load는 실험 시간 집계(lab.parse_cost)가 읽는다. MCP로 호출해도 같다.
        lines = [f"strategy={strategy} status={doc.status} blocks={len(doc.blocks)} "
                 f"parse_ms={doc.elapsed_ms} cache={'hit' if doc.from_cache else 'miss'} load={load}"]
        cov = [f"p{p.page}:{p.coverage:.0%}" for p in doc.pages if p.coverage is not None]
        if cov:
            lines.append("텍스트 레이어 포함률: " + ", ".join(cov))
        for w in doc.warnings:
            lines.append(f"[{w.severity}] {w.code} p{w.page or '-'} {w.block_id or ''}: {w.message}")
        total = sum(len(b.text) for b in doc.blocks)
        full = total <= FULL_TEXT_BUDGET
        lines.append("블록 목록 (전문):" if full else "블록 목록 (앞부분만, 전문은 read_blocks):")
        for b in doc.blocks:
            text = b.text if full else b.text[:100].replace("\n", " / ")
            lines.append(f"- {b.block_id} p{b.page} {b.type}: {text}")
        return self._record("parse_document", {"strategy": strategy}, "\n".join(lines), t0)

    def read_blocks(self, strategy: str, block_ids: list[str]) -> str:
        t0 = time.perf_counter()
        doc = self.get_doc(strategy)
        bm = doc.block_map()
        out = []
        for bid in block_ids[:20]:
            b = bm.get(bid)
            out.append(f"- {bid}: (없음)" if b is None else f"- {b.block_id} p{b.page} {b.type}:\n{b.text}")
        return self._record("read_blocks", {"strategy": strategy, "block_ids": block_ids}, "\n".join(out), t0)

    def search_document(self, strategy: str, query: str) -> str:
        t0 = time.perf_counter()
        doc = self.get_doc(strategy)
        q_n, q_l = _norm(query), _loose(query)
        hits: list[Block] = [b for b in doc.blocks if q_n in _norm(b.text) or q_l in _loose(b.text)]
        lines = []
        for b in hits[:10]:
            rows = [ln for ln in b.text.split("\n") if q_n in _norm(ln) or q_l in _loose(ln)]
            lines.append(f"- {b.block_id} p{b.page} {b.type}: " + " / ".join(rows[:5] or [b.text[:150]]))
        out = "\n".join(lines) if lines else f"'{query}'를 포함한 블록 없음"
        return self._record("search_document", {"strategy": strategy, "query": query}, out, t0)

    def get_class_properties(self, class_id: str) -> str:
        t0 = time.perf_counter()
        try:
            out = json.dumps(self.ontology.properties_of(class_id), ensure_ascii=False)
        except KeyError as e:
            out = str(e)
        return self._record("get_class_properties", {"class_id": class_id}, out, t0)

    # ------------------------------------------------------------------ 검증용
    def virtual_document(self, extra_text: dict[str, str] | None = None) -> ParsedDocument:
        """여러 전략의 블록을 'strategy:block_id'로 합친 문서. 에이전트 근거 검증에 쓴다."""
        blocks = [Block(**{**b.model_dump(), "block_id": f"{s}:{b.block_id}"})
                  for s, d in self.docs.items() for b in d.blocks]
        for bid, text in (extra_text or {}).items():
            blocks.append(Block(block_id=bid, page=0, type="other", text=text))
        return ParsedDocument(doc_id="VIRTUAL", source_file=self.path.name,
                              file_type="xlsx" if self.is_excel else "pdf", parser="+".join(self.docs),
                              status="success", blocks=blocks)

    def langchain_tools(self) -> list:
        from langchain_core.tools import StructuredTool

        strategies = self.strategies

        def parse_document(strategy: str) -> str:
            """문서를 지정한 전략으로 파싱하고 상태·품질 경고·블록 목록을 돌려준다."""
            return self.parse_document(strategy)

        def read_blocks(strategy: str, block_ids: list[str]) -> str:
            """해당 전략 파싱 결과에서 블록 전문을 읽는다. (최대 20개)"""
            return self.read_blocks(strategy, block_ids)

        def search_document(strategy: str, query: str) -> str:
            """해당 전략 파싱 결과에서 query를 포함한 블록과 줄을 찾는다. 공백·구분자 차이는 무시."""
            return self.search_document(strategy, query)

        def get_class_properties(class_id: str) -> str:
            """온톨로지 클래스의 속성 정의(id, label, required, target_unit, synonyms)를 돌려준다."""
            return self.get_class_properties(class_id)

        desc = "전략: " + "; ".join(f"{s}={STRATEGY_HELP[s]}" for s in strategies)
        return [
            StructuredTool.from_function(parse_document, description=parse_document.__doc__ + " " + desc),
            StructuredTool.from_function(read_blocks),
            StructuredTool.from_function(search_document),
            StructuredTool.from_function(get_class_properties),
        ]
