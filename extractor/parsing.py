"""01 문서 파싱 (설계안 6·7·13절).

파서별 결과를 공통 ParsedDocument로 변환하고, 품질 검사 결과에 따라
규칙 기반으로 다음 파싱 전략을 고른다. (LLM 판단 없음)

전략
- docling            : Docling, OCR 끔, TableFormer ACCURATE
- docling_ocr        : Docling, RapidOCR(한국어) 전체 페이지 OCR — 텍스트 레이어가 없을 때
- docling_table_alt  : Docling, 표 셀 매칭 끔 — 표 구조가 비정상일 때
- pdfplumber         : 선 기반 표 추출 + 텍스트 줄 — Docling 실패·표 누락 시
- excel              : openpyxl
"""

from __future__ import annotations

import hashlib
import re
import time
import unicodedata
from functools import lru_cache
from pathlib import Path

from .schemas import Block, PageInfo, ParsedDocument, ParseWarning

PDF_STRATEGIES = ["docling", "docling_ocr", "docling_table_alt", "pdfplumber"]

# 품질 경고 코드 → 시도할 대체 전략 (앞에서부터 아직 안 해본 것)
FALLBACK_RULES: dict[str, list[str]] = {
    "PARSER_EXCEPTION": ["pdfplumber", "docling"],
    "NO_TEXT_LAYER": ["docling_ocr"],
    "PAGE_NO_TEXT_LAYER": ["docling_ocr"],
    "LOW_TEXT": ["docling_ocr", "pdfplumber"],
    "CONTENT_LOSS": ["pdfplumber", "docling_ocr"],
    "TABLE_MISSING": ["docling_table_alt", "pdfplumber"],
    "TABLE_SPARSE": ["docling_table_alt", "pdfplumber"],
    "TABLE_SINGLE_COLUMN": ["docling_table_alt", "pdfplumber"],
    "TABLE_LABEL_MISSING": ["docling_table_alt", "pdfplumber"],
}
OCR_STRATEGIES = {"docling_ocr"}
MAX_PARSE_ATTEMPTS = 3


def make_doc_id(path: Path) -> str:
    h = hashlib.sha1(path.read_bytes()).hexdigest()[:8].upper()
    return f"DOC-{h}"


def _clean(s: str | None) -> str:
    if s is None:
        return ""
    return unicodedata.normalize("NFC", str(s)).strip()


def _norm_bbox(l: float, t: float, r: float, b: float) -> list[float]:
    """좌표 순서를 [left, top, right, bottom]으로 보장한다. (회전 페이지·다단 배치 등에서 뒤집힐 수 있음)"""
    return [round(min(l, r), 1), round(min(t, b), 1), round(max(l, r), 1), round(max(t, b), 1)]


def _rows_to_text(rows: list[list[str]]) -> str:
    return "\n".join(" | ".join(c.replace("\n", " ") for c in r) for r in rows)


# ---------------------------------------------------------------------------
# Docling
# ---------------------------------------------------------------------------


@lru_cache(maxsize=4)
def _docling_converter(strategy: str):
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import (
        OcrMode,
        PdfPipelineOptions,
        RapidOcrOptions,
        TableFormerMode,
        TableStructureOptions,
    )
    from docling.document_converter import DocumentConverter, PdfFormatOption

    opts = PdfPipelineOptions(
        do_ocr=strategy == "docling_ocr",
        do_table_structure=True,
        table_structure_options=TableStructureOptions(
            mode=TableFormerMode.ACCURATE,
            do_cell_matching=strategy != "docling_table_alt",
        ),
    )
    if strategy == "docling_ocr":
        # 스캔 샘플 비교 결과 FULL_PAGE보다 DEFAULT 모드가 짧은 셀 글자를 더 많이 살렸다.
        opts.ocr_options = RapidOcrOptions(lang=["korean"], mode=OcrMode.DEFAULT)
    return DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)})


_DOCLING_LABEL_TYPE = {
    "title": "heading",
    "section_header": "heading",
    "text": "text",
    "paragraph": "text",
    "list_item": "list_item",
    "caption": "caption",
    "table": "table",
}


def _parse_docling(path: Path, strategy: str) -> tuple[list[Block], list[PageInfo]]:
    from docling_core.types.doc import TableItem, TextItem

    doc = _docling_converter(strategy).convert(str(path)).document
    pages = [
        PageInfo(page=no, width=p.size.width, height=p.size.height)
        for no, p in sorted(doc.pages.items())
    ]
    heights = {p.page: p.height for p in pages}

    blocks: list[Block] = []
    for item, _level in doc.iterate_items():
        if isinstance(item, TableItem):
            rows = [[_clean(c.text) for c in row] for row in item.data.grid]
            rows = [r for r in rows if any(r)]
            if not rows:
                continue
            text, btype = _rows_to_text(rows), "table"
        elif isinstance(item, TextItem):
            text = _clean(item.text)
            if not text:
                continue
            rows, btype = None, _DOCLING_LABEL_TYPE.get(item.label.value, "other")
        else:
            continue  # 그림 등은 현재 범위 밖

        page_no, bbox = None, None
        if item.prov:
            prov = item.prov[0]
            page_no = prov.page_no
            bb = prov.bbox.to_top_left_origin(page_height=heights[page_no])
            bbox = _norm_bbox(bb.l, bb.t, bb.r, bb.b)
        blocks.append(
            Block(
                block_id=f"B{len(blocks) + 1}",
                page=page_no or 0,
                type=btype,
                text=text,
                rows=rows,
                bbox=bbox,
                label=item.label.value,
            )
        )
    return blocks, pages


# ---------------------------------------------------------------------------
# pdfplumber
# ---------------------------------------------------------------------------


def _inside(obj: dict, boxes: list[tuple]) -> bool:
    cx, cy = (obj["x0"] + obj["x1"]) / 2, (obj["top"] + obj["bottom"]) / 2
    return any(x0 <= cx <= x1 and t <= cy <= b for x0, t, x1, b in boxes)


def _parse_pdfplumber(path: Path) -> tuple[list[Block], list[PageInfo]]:
    import pdfplumber

    blocks: list[Block] = []
    pages: list[PageInfo] = []
    with pdfplumber.open(str(path)) as pdf:
        for pno, page in enumerate(pdf.pages, start=1):
            pages.append(PageInfo(page=pno, width=float(page.width), height=float(page.height)))
            tables = page.find_tables()
            tboxes = [t.bbox for t in tables]
            items: list[tuple[float, Block]] = []

            for t in tables:
                rows = [[_clean(c) for c in r] for r in t.extract()]
                rows = [r for r in rows if any(r)]
                if rows:
                    x0, top, x1, bottom = t.bbox
                    items.append(
                        (top, Block(block_id="", page=pno, type="table", text=_rows_to_text(rows),
                                    rows=rows, bbox=_norm_bbox(x0, top, x1, bottom), label="table"))
                    )

            outside = page.filter(lambda o: o.get("object_type") != "char" or not _inside(o, tboxes))
            lines = outside.extract_text_lines(return_chars=False)
            para: list[dict] = []

            def flush():
                if not para:
                    return
                text = "\n".join(_clean(ln["text"]) for ln in para)
                bbox = _norm_bbox(min(l["x0"] for l in para), min(l["top"] for l in para),
                                  max(l["x1"] for l in para), max(l["bottom"] for l in para))
                items.append((bbox[1], Block(block_id="", page=pno, type="text", text=text,
                                             bbox=bbox, label="text_lines")))
                para.clear()

            for ln in lines:
                if para:
                    prev = para[-1]
                    gap = ln["top"] - prev["bottom"]
                    if gap > 0.6 * (prev["bottom"] - prev["top"]):
                        flush()
                para.append(ln)
            flush()

            for _, b in sorted(items, key=lambda x: x[0]):
                b.block_id = f"B{len(blocks) + 1}"
                blocks.append(b)
    return blocks, pages


# ---------------------------------------------------------------------------
# Excel
# ---------------------------------------------------------------------------


def _parse_excel(path: Path) -> tuple[list[Block], list[PageInfo]]:
    import openpyxl

    wb = openpyxl.load_workbook(str(path), data_only=True)
    blocks: list[Block] = []
    pages: list[PageInfo] = []
    for sno, ws in enumerate(wb.worksheets, start=1):
        pages.append(PageInfo(page=sno))
        grid = [[_clean(v) for v in row] for row in ws.iter_rows(values_only=True)]
        # 병합 셀은 좌상단 값으로 채워 속성명-값 대응이 끊기지 않게 한다.
        for rng in ws.merged_cells.ranges:
            v = grid[rng.min_row - 1][rng.min_col - 1]
            for r in range(rng.min_row - 1, rng.max_row):
                for c in range(rng.min_col - 1, rng.max_col):
                    grid[r][c] = v
        grid = [r for r in grid if any(r)]
        if not grid:
            continue
        keep = [c for c in range(len(grid[0])) if any(r[c] for r in grid)]
        rows = [[r[c] for c in keep] for r in grid]
        # 제목처럼 행 전체가 병합된 경우는 한 칸으로 줄인다.
        rows = [[r[0]] if len(r) > 1 and len(set(r)) == 1 else r for r in rows]
        blocks.append(
            Block(block_id=f"B{len(blocks) + 1}", page=sno, type="table",
                  text=_rows_to_text(rows), rows=rows, label="sheet", sheet=ws.title)
        )
    return blocks, pages


# ---------------------------------------------------------------------------
# 품질 검사 (설계안 6.3)
# ---------------------------------------------------------------------------

_TOKEN = re.compile(r"[0-9A-Za-z가-힣]+(?:[.,][0-9]+)?")


def _tokens(s: str) -> list[str]:
    return [t for t in _TOKEN.findall(unicodedata.normalize("NFKC", s).casefold()) if len(t) >= 2]


def _text_layer(path: Path) -> list[str]:
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(str(path))
    try:
        return [pdf[i].get_textpage().get_text_range() for i in range(len(pdf))]
    finally:
        pdf.close()


def _pdfplumber_table_counts(path: Path) -> dict[int, int]:
    import pdfplumber

    with pdfplumber.open(str(path)) as pdf:
        return {i: len(p.find_tables()) for i, p in enumerate(pdf.pages, start=1)}


def quality_check(doc: ParsedDocument, path: Path) -> list[ParseWarning]:
    ws: list[ParseWarning] = []
    parsed_chars = sum(len(re.sub(r"\s", "", b.text)) for b in doc.blocks)
    if parsed_chars < 30:
        ws.append(ParseWarning(code="LOW_TEXT", severity="error",
                               message=f"추출 텍스트가 {parsed_chars}자뿐임"))

    if doc.file_type == "pdf":
        layer = _text_layer(path)
        layer_chars = sum(len(re.sub(r"\s", "", t)) for t in layer)
        ocr_used = doc.parser == "docling_ocr"
        if layer_chars < 20 and not ocr_used:
            ws.append(ParseWarning(code="NO_TEXT_LAYER", severity="error",
                                   message="PDF에 텍스트 레이어가 없음 (스캔 문서로 추정)"))

        if layer_chars >= 20 and not ocr_used:
            for i, t in enumerate(layer, start=1):
                if not re.sub(r"\s", "", t):
                    ws.append(ParseWarning(code="PAGE_NO_TEXT_LAYER", page=i,
                                           message="이 페이지만 텍스트 레이어가 없음 (부분 스캔 의심)"))

        by_page: dict[int, list[str]] = {}
        for b in doc.blocks:
            by_page.setdefault(b.page, []).extend(_tokens(b.text))
        for pi in doc.pages:
            src = _tokens(layer[pi.page - 1]) if pi.page - 1 < len(layer) else []
            pi.text_layer_chars = len(re.sub(r"\s", "", layer[pi.page - 1])) if src else 0
            if not src or ocr_used:
                continue
            got = set(by_page.get(pi.page, []))
            pi.coverage = round(sum(t in got for t in src) / len(src), 3)
            if pi.coverage < 0.85:
                ws.append(ParseWarning(
                    code="CONTENT_LOSS", page=pi.page,
                    severity="error" if pi.coverage < 0.5 else "warning",
                    message=f"텍스트 레이어 대비 파싱 포함률 {pi.coverage:.0%}"))

        if doc.parser.startswith("docling"):
            expected = _pdfplumber_table_counts(path)
            for page, n in expected.items():
                found = sum(1 for b in doc.blocks if b.page == page and b.type == "table")
                if found < n:
                    ws.append(ParseWarning(
                        code="TABLE_MISSING", page=page,
                        message=f"선으로 그려진 표 {n}개 중 {found}개만 표로 인식됨"))

        for b in doc.blocks:
            if b.page == 0 or b.bbox is None:
                ws.append(ParseWarning(code="NO_POSITION", block_id=b.block_id, severity="info",
                                       message="페이지·위치 정보를 추적할 수 없는 블록"))

    for b in doc.blocks:
        if b.type != "table" or not b.rows:
            continue
        ncols = max(len(r) for r in b.rows)
        cells = [c for r in b.rows for c in r]
        empty = sum(1 for c in cells if not c) / max(len(cells), 1)
        if ncols == 1 and len(b.rows) > 1:
            ws.append(ParseWarning(code="TABLE_SINGLE_COLUMN", page=b.page, block_id=b.block_id,
                                   message="열이 1개뿐인 표 (속성명·값 분리 실패 의심)"))
        if empty > 0.5:
            ws.append(ParseWarning(code="TABLE_SPARSE", page=b.page, block_id=b.block_id,
                                   message=f"빈 셀 비율 {empty:.0%}"))
        unlabeled = sum(1 for r in b.rows[1:] if len(r) > 1 and not r[0] and any(r[1:]))
        if unlabeled:
            ws.append(ParseWarning(code="TABLE_LABEL_MISSING", page=b.page, block_id=b.block_id,
                                   message=f"값은 있으나 첫 열(속성명)이 빈 행 {unlabeled}개"))
        if len({len(r) for r in b.rows}) > 1:
            ws.append(ParseWarning(code="TABLE_RAGGED", page=b.page, block_id=b.block_id,
                                   severity="info", message="행마다 열 수가 다름"))
    return ws


# ---------------------------------------------------------------------------
# 실행 · 대체 전략 선택
# ---------------------------------------------------------------------------


def parse_with_strategy(path: str | Path, strategy: str) -> ParsedDocument:
    path = Path(path)
    file_type = "xlsx" if path.suffix.lower() in (".xlsx", ".xlsm") else "pdf"
    base = dict(doc_id=make_doc_id(path), source_file=path.name, file_type=file_type, parser=strategy)
    t0 = time.perf_counter()
    try:
        if strategy == "excel":
            blocks, pages = _parse_excel(path)
        elif strategy == "pdfplumber":
            blocks, pages = _parse_pdfplumber(path)
        elif strategy in ("docling", "docling_ocr", "docling_table_alt"):
            blocks, pages = _parse_docling(path, strategy)
        else:
            raise ValueError(f"알 수 없는 전략: {strategy}")
    except Exception as e:  # 파서 실패도 결과로 기록해 다음 전략을 고를 수 있게 한다.
        return ParsedDocument(
            **base, status="failed", elapsed_ms=int((time.perf_counter() - t0) * 1000),
            warnings=[ParseWarning(code="PARSER_EXCEPTION", severity="error",
                                   message=f"{type(e).__name__}: {e}")],
        )
    doc = ParsedDocument(**base, status="success", pages=pages, blocks=blocks)
    doc.elapsed_ms = int((time.perf_counter() - t0) * 1000)
    doc.warnings = quality_check(doc, path)
    sev = {w.severity for w in doc.warnings}
    doc.status = "failed" if "error" in sev else "degraded" if "warning" in sev else "success"
    return doc


# 파서·품질 검사 코드를 바꾸면 올려서 이전 캐시를 무효화한다.
PARSER_VERSION = "2"
CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache" / "parse"


def parse_cached(path: str | Path, strategy: str, read: bool = True) -> ParsedDocument:
    """실험용 디스크 캐시. elapsed_ms는 최초 실행 시간을 그대로 유지하고 from_cache로 구분한다.

    read=False이면 캐시를 읽지 않고 새로 파싱하되 결과는 캐시에 쓴다. (속도 측정 모드)
    """
    path = Path(path)
    key = CACHE_DIR / f"{make_doc_id(path)}_{strategy}_v{PARSER_VERSION}.json"
    if read and key.exists():
        doc = ParsedDocument.model_validate_json(key.read_text(encoding="utf-8"))
        doc.source_file = path.name
        doc.from_cache = True
        return doc
    doc = parse_with_strategy(path, strategy)
    if doc.warnings and doc.warnings[0].code == "PARSER_EXCEPTION":
        return doc  # 일시적 실패일 수 있으니 캐시하지 않는다.
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key.write_text(doc.model_dump_json(), encoding="utf-8")
    return doc


def initial_strategy(path: str | Path) -> str:
    return "excel" if Path(path).suffix.lower() in (".xlsx", ".xlsm") else "docling"


def next_strategy(doc: ParsedDocument, tried: list[str]) -> str | None:
    """품질 경고에 대응하는 대체 전략 중 아직 시도하지 않은 것을 고른다."""
    if doc.status == "success" or doc.file_type != "pdf" or len(tried) >= MAX_PARSE_ATTEMPTS:
        return None
    # OCR 결과의 문제는 텍스트 레이어가 없다는 뜻이므로 비-OCR 전략은 의미가 없다.
    allowed = OCR_STRATEGIES if doc.parser in OCR_STRATEGIES else set(PDF_STRATEGIES)
    for w in sorted(doc.warnings, key=lambda w: w.severity != "error"):
        for s in FALLBACK_RULES.get(w.code, []):
            if s not in tried and s in allowed:
                return s
    return None


def score(doc: ParsedDocument) -> tuple:
    errors = sum(w.severity == "error" for w in doc.warnings)
    warns = sum(w.severity == "warning" for w in doc.warnings)
    cov = [p.coverage for p in doc.pages if p.coverage is not None]
    return (-errors, -warns, sum(cov) / len(cov) if cov else 0.0, len(doc.blocks))


def parse_document(
    path: str | Path, strategy: str = "auto", use_cache: bool = False
) -> tuple[ParsedDocument, list[ParsedDocument]]:
    """단독 실행용. 그래프에서는 같은 규칙을 노드·조건부 엣지로 실행한다."""
    def run(p, s):
        return parse_cached(p, s, read=use_cache)

    if strategy != "auto":
        d = run(path, strategy)
        return d, [d]
    attempts = [run(path, initial_strategy(path))]
    while (s := next_strategy(attempts[-1], [a.parser for a in attempts])) is not None:
        attempts.append(run(path, s))
    return max(attempts, key=score), attempts
