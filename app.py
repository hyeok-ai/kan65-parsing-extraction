"""파싱·추출 결과 육안 확인 뷰어.

uv run streamlit run app.py
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from extractor.graph import build_graph, initial_state
from extractor.llm import LLMConfig
from extractor.ontology import JsonOntology
from extractor.parsing import PDF_STRATEGIES, parse_document
from extractor.schemas import ParsedDocument
from extractor.visualize import TYPE_COLORS, draw_overlay, render_pages

ROOT = Path(__file__).resolve().parent
SAMPLES = ROOT / "samples"
UPLOADS = ROOT / "uploads"
SCALE = 1.6
SEV_ICON = {"error": "🛑", "warning": "⚠️", "info": "ℹ️"}

st.set_page_config(page_title="문서 파싱·추출 확인", layout="wide")


# --------------------------------------------------------------------------- 캐시
@st.cache_data(show_spinner=False)
def cached_parse(path: str, mtime: float, strategy: str):
    best, attempts = parse_document(path, strategy)
    return best.model_dump(), [a.model_dump() for a in attempts]


@st.cache_data(show_spinner=False)
def cached_pages(path: str, mtime: float):
    return render_pages(path, SCALE)


@st.cache_resource
def graph():
    return build_graph()


def rgb_hex(c):
    return "#%02x%02x%02x" % c


# --------------------------------------------------------------------------- 사이드바
with st.sidebar:
    st.header("문서")
    samples = sorted(p.name for p in SAMPLES.glob("*") if p.suffix.lower() in (".pdf", ".xlsx"))
    uploaded = st.file_uploader("PDF / Excel 업로드", type=["pdf", "xlsx"])
    if uploaded:
        UPLOADS.mkdir(exist_ok=True)
        data = uploaded.getvalue()
        path = UPLOADS / f"{hashlib.sha1(data).hexdigest()[:8]}_{uploaded.name}"
        if not path.exists():
            path.write_bytes(data)
    else:
        default = samples.index("motor_spec.pdf") if "motor_spec.pdf" in samples else 0
        path = SAMPLES / st.selectbox("샘플 문서", samples, index=default)
    is_pdf = path.suffix.lower() == ".pdf"

    st.header("01 파싱")
    strategy = st.selectbox(
        "전략", ["auto", *PDF_STRATEGIES] if is_pdf else ["excel"],
        help="auto: Docling 1차 파싱 후 품질 경고에 따라 규칙 기반 대체 전략 적용",
    )

    st.header("02·03 LLM")
    cfg = LLMConfig()
    cfg.model = st.text_input("모델", cfg.model)
    efforts = ["none", "low", "medium", "high"]
    cfg.class_effort = st.selectbox("02 reasoning effort", efforts, index=efforts.index(cfg.class_effort or "low"))
    cfg.extract_effort = st.selectbox("03 reasoning effort", efforts, index=efforts.index(cfg.extract_effort or "low"))

    st.divider()
    st.caption("블록 종류 색상")
    st.markdown(" ".join(
        f"<span style='background:{rgb_hex(c)};color:white;padding:1px 6px;border-radius:3px;"
        f"font-size:12px'>{t}</span>" for t, c in TYPE_COLORS.items() if t != "other"),
        unsafe_allow_html=True)

mtime = path.stat().st_mtime
with st.spinner(f"{path.name} 파싱 중… (Docling 첫 실행은 모델 로딩으로 오래 걸림)"):
    best_d, attempts_d = cached_parse(str(path), mtime, strategy)
doc = ParsedDocument.model_validate(best_d)
attempts = [ParsedDocument.model_validate(a) for a in attempts_d]
pages = cached_pages(str(path), mtime) if is_pdf else []

st.title(path.name)
tab_parse, tab_cmp, tab_ext, tab_json = st.tabs(["① 파싱 결과", "② 파서 비교", "③ 클래스·속성 추출", "④ JSON"])


# --------------------------------------------------------------------------- 공통 렌더
def show_warnings(d: ParsedDocument):
    if not d.warnings:
        st.success("품질 검사 경고 없음 — 단, 검사로 못 잡는 오류가 있을 수 있으니 원문과 직접 비교할 것")
        return
    st.dataframe(pd.DataFrame([{"": SEV_ICON[w.severity], "코드": w.code, "페이지": w.page,
                                "블록": w.block_id, "내용": w.message} for w in d.warnings]),
                 hide_index=True, width="stretch")


def show_block(b, highlight=False):
    head = f"{'🔴 ' if highlight else ''}**{b.block_id}** · `{b.type}`" + (f" · {b.label}" if b.label else "")
    st.markdown(head)
    if b.rows:
        width = max(len(r) for r in b.rows)
        st.dataframe(pd.DataFrame([r + [""] * (width - len(r)) for r in b.rows]),
                     hide_index=True, width="stretch")
    else:
        st.text(b.text)


def page_view(d: ParsedDocument, page_no: int, highlight: set[str], key: str, only_hl=False):
    blocks = [b for b in d.blocks if b.page == page_no]
    if is_pdf and pages:
        st.image(draw_overlay(pages[page_no - 1], blocks, SCALE, highlight, dim_others=only_hl),
                 width="stretch")
    return blocks


# --------------------------------------------------------------------------- ① 파싱
with tab_parse:
    c = st.columns(5)
    c[0].metric("선택된 전략", doc.parser)
    c[1].metric("상태", doc.status)
    c[2].metric("블록 / 표", f"{len(doc.blocks)} / {sum(b.type == 'table' for b in doc.blocks)}")
    c[3].metric("파싱 시간", f"{(doc.elapsed_ms or 0) / 1000:.1f} s")
    cov = [p.coverage for p in doc.pages if p.coverage is not None]
    c[4].metric("텍스트 레이어 포함률", f"{min(cov):.0%}" if cov else "-",
                help="페이지별 최소값. PDF 텍스트 레이어 토큰 중 파싱 결과에 들어간 비율 (OCR·Excel은 해당 없음)")

    if len(attempts) > 1:
        st.caption("시도 순서: " + " → ".join(
            f"{a.parser}({a.status}, {(a.elapsed_ms or 0) / 1000:.1f}s)" for a in attempts))
    show_warnings(doc)

    page_ids = sorted({b.page for b in doc.blocks}) or [1]
    if is_pdf:
        page_ids = list(range(1, len(pages) + 1))
    pno = st.radio("페이지" if is_pdf else "시트", page_ids, horizontal=True, key="p_page",
                   format_func=lambda p: f"{p}" if is_pdf else next(
                       (b.sheet for b in doc.blocks if b.page == p and b.sheet), str(p)))
    blocks_on_page = [b for b in doc.blocks if b.page == pno]
    sel = st.multiselect("강조할 블록", [b.block_id for b in blocks_on_page], key="p_sel")

    left, right = st.columns([3, 2]) if is_pdf else (None, st.container())
    if left is not None:
        with left:
            page_view(doc, pno, set(sel), "p")
    with right:
        with st.container(height=900 if is_pdf else None):
            for b in blocks_on_page:
                if sel and b.block_id not in sel:
                    continue
                show_block(b, b.block_id in sel)
                st.divider()
    unplaced = [b for b in doc.blocks if b.page == 0]
    if unplaced:
        with st.expander(f"페이지 정보 없는 블록 {len(unplaced)}개"):
            for b in unplaced:
                show_block(b)


# --------------------------------------------------------------------------- ② 비교
with tab_cmp:
    if not is_pdf:
        st.info("Excel은 openpyxl 단일 파서만 사용합니다.")
    else:
        cc = st.columns(2)
        sa = cc[0].selectbox("왼쪽", PDF_STRATEGIES, index=0, key="cmp_a")
        sb = cc[1].selectbox("오른쪽", PDF_STRATEGIES, index=PDF_STRATEGIES.index("pdfplumber"), key="cmp_b")
        if st.button("두 전략으로 파싱해 비교", type="primary"):
            st.session_state["cmp"] = (sa, sb)
        if st.session_state.get("cmp") == (sa, sb):
            docs = []
            for s in (sa, sb):
                with st.spinner(f"{s} 파싱 중…"):
                    docs.append(ParsedDocument.model_validate(cached_parse(str(path), mtime, s)[0]))
            summary = pd.DataFrame([{
                "전략": d.parser, "상태": d.status, "시간(s)": round((d.elapsed_ms or 0) / 1000, 2),
                "블록": len(d.blocks), "표": sum(b.type == "table" for b in d.blocks),
                "포함률(최소)": min([p.coverage for p in d.pages if p.coverage is not None], default=None),
                "경고": ", ".join(w.code for w in d.warnings),
            } for d in docs])
            st.dataframe(summary, hide_index=True, width="stretch")
            cpno = st.radio("페이지", list(range(1, len(pages) + 1)), horizontal=True, key="cmp_page")
            cols = st.columns(2)
            for col, d in zip(cols, docs):
                with col:
                    st.subheader(d.parser)
                    page_view(d, cpno, set(), d.parser)
                    for b in [b for b in d.blocks if b.page == cpno and b.type == "table"]:
                        show_block(b)


# --------------------------------------------------------------------------- ③ 추출
with tab_ext:
    st.caption(f"현재 파싱 결과({doc.parser})를 그대로 02 → 03에 전달합니다. 모델: `{cfg.model}`")
    run_key = f"run::{path}::{mtime}::{strategy}"
    if st.button("02 클래스 매핑 + 03 속성 추출 실행", type="primary", disabled=doc.status == "failed"):
        final = None
        with st.status("LangGraph 실행 중", expanded=True) as status:
            stream = graph().stream(initial_state(str(path), cfg, doc), stream_mode=["updates", "values"])
            for mode, chunk in stream:
                if mode == "values":
                    final = chunk
                    continue
                for node, upd in chunk.items():
                    runs = upd.get("model_runs") or []
                    extra = (f" — {runs[0]['latency_ms']} ms, in {runs[0]['input_tokens']} / "
                             f"out {runs[0]['output_tokens']} tok") if runs else ""
                    st.write(f"✔ `{node}`{extra}")
            status.update(label="완료", state="complete")
        st.session_state[run_key] = final

    final = st.session_state.get(run_key)
    if final:
        status_color = {"extracted": "green", "needs_review": "orange", "failed": "red"}[final["status"]]
        st.markdown(f"### 결과: :{status_color}[{final['status']}] → 다음 노드 `{final['next_node']}`")
        for r in final.get("review_reasons", []):
            st.markdown(f"- {r}")

        cr = final.get("class_result")
        st.subheader("02 클래스 매핑")
        if cr:
            labels = {c["id"]: c["label"] for c in JsonOntology().class_candidates()}
            k = st.columns([1, 3])
            k[0].metric("class_id", f"{cr['class_id']}", labels.get(cr["class_id"], ""))
            k[1].markdown(f"**reason**: {cr['reason']}  \n**needs_review**: `{cr['needs_review']}`")
            if cr.get("evidence"):
                k[1].markdown(f"**근거** {cr['evidence']['block_id']} p{cr['evidence']['page']}: “{cr['evidence']['text']}”")
        for i in final.get("class_issues", []):
            st.warning(f"{i['code']}: {i['message']}")

        ex = final.get("extraction")
        if ex:
            st.subheader("03 속성 추출")
            prop_defs = {p["id"]: p for p in final.get("properties", [])}
            issues_by = {}
            for i in final.get("extraction_issues", []):
                issues_by.setdefault(i.get("property_id"), []).append(i["code"])
            rows = []
            for a in ex["attributes"]:
                p = prop_defs.get(a["property_id"], {})
                ev = a.get("evidence") or {}
                rows.append({
                    "속성": f"{p.get('label', '?')} ({a['property_id']})",
                    "필수": "●" if p.get("required") else "",
                    "raw_value": a["raw_value"],
                    "raw_unit": a.get("raw_unit"),
                    "기준 단위": p.get("target_unit"),
                    "블록": ev.get("block_id"), "페이지": ev.get("page"),
                    "근거 원문": ev.get("text"),
                    "검증": ", ".join(issues_by.get(a["property_id"], [])) or "✔",
                })
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
            if ex.get("warnings"):
                st.warning("LLM warnings: " + " / ".join(ex["warnings"]))

            ev_blocks = {a["evidence"]["block_id"] for a in ex["attributes"] if a.get("evidence")}
            if cr and cr.get("evidence"):
                ev_blocks.add(cr["evidence"]["block_id"])
            if is_pdf and ev_blocks:
                st.subheader("근거 위치")
                bm = doc.block_map()
                ev_pages = sorted({bm[b].page for b in ev_blocks if b in bm})
                epno = st.radio("페이지", ev_pages, horizontal=True, key="ev_page")
                page_view(doc, epno, ev_blocks, "ev", only_hl=st.toggle("근거 블록만 표시", True))

        runs = final.get("model_runs", [])
        if runs:
            st.subheader("모델 호출 기록")
            st.dataframe(pd.DataFrame(runs), hide_index=True, width="stretch")
            total = sum(r["cost_usd"] or 0 for r in runs)
            st.caption(f"총 {len(runs)}회 호출 · 예상 비용 ${total:.5f} (공식 단가 × 토큰, 캐싱 미반영)")


# --------------------------------------------------------------------------- ④ JSON
with tab_json:
    st.download_button("ParsedDocument JSON 다운로드", json.dumps(best_d, ensure_ascii=False, indent=2),
                       file_name=f"{path.stem}.parsed.json", mime="application/json")
    st.json(best_d, expanded=2)
    final = st.session_state.get(f"run::{path}::{mtime}::{strategy}")
    if final:
        st.download_button("파이프라인 State JSON 다운로드", json.dumps(final, ensure_ascii=False, indent=2, default=str),
                           file_name=f"{path.stem}.state.json", mime="application/json")
        st.json(final, expanded=1)
