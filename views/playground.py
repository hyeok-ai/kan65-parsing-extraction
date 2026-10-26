"""방법별 비교 실험 플레이그라운드.

실험 결과는 runs/<실험 ID>/records.jsonl 에 쌓이므로 나중에 다시 불러와 비교할 수 있다.
"""

from __future__ import annotations

import json

import altair as alt
import pandas as pd
import streamlit as st

from extractor.agent import AgentConfig
from extractor.lab import (
    METHODS,
    ExperimentConfig,
    list_experiments,
    list_samples,
    load_gold,
    load_records,
    run_experiment,
    summarize,
)
from extractor.llm import LLMConfig, PRICING_FILE, load_pricing

BAR = "#2a78d6"
SEQ = ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]  # 한 색상 단계 (값이 클수록 진함)
EFFORTS = ["none", "low", "medium", "high"]

st.title("실험 플레이그라운드")
st.caption("같은 문서·같은 온톨로지·같은 출력 스키마로 방법을 바꿔 가며 정확도·토큰·비용·시간을 비교한다. "
           "결과는 `runs/`에 저장되어 언제든 다시 불러올 수 있다.")

tab_run, tab_cmp, tab_detail, tab_cost = st.tabs(["① 실행", "② 결과 비교", "③ 실행 상세", "④ 과금 시뮬레이션"])
pricing = load_pricing()
models = list(pricing)


# =========================================================================== ① 실행
with tab_run:
    left, right = st.columns([1, 1])
    with left:
        st.subheader("대상")
        samples = list_samples()
        gold = load_gold()
        sel_samples = st.multiselect("문서", [p.name for p in samples], default=[p.name for p in samples],
                                     help="정답(data/eval/gold.json)이 있는 문서만 정확도가 계산된다")
        no_gold = [s for s in sel_samples if s not in gold]
        if no_gold:
            st.caption("정답 없음(비용·시간만 기록): " + ", ".join(no_gold))
        sel_methods = st.multiselect(
            "방법", list(METHODS), default=["code_single", "code_rules", "code_agent", "agent_tools"],
            format_func=lambda m: f"{METHODS[m][0]} — {METHODS[m][1]}")
        st.caption("F·G(MCP)는 문서마다 MCP 서버를 새로 띄우므로 순차 실행되고 느리다.")
    with right:
        st.subheader("모델 · 설정")
        c1, c2 = st.columns(2)
        ex_model = c1.selectbox("02·03 모델", models, index=models.index("gpt-6-luna") if "gpt-6-luna" in models else 0)
        ag_model = c2.selectbox("에이전트 모델", models, index=models.index("gpt-6-luna") if "gpt-6-luna" in models else 0)
        c1, c2, c3 = st.columns(3)
        class_effort = c1.selectbox("02 effort", EFFORTS, index=1)
        extract_effort = c2.selectbox("03 effort", EFFORTS, index=1)
        agent_effort = c3.selectbox("에이전트 effort", EFFORTS, index=1)
        c1, c2, c3 = st.columns(3)
        max_tools = c1.number_input("에이전트 최대 도구 호출", 2, 30, 8)
        repeats = c2.number_input("반복 횟수", 1, 10, 1, help="LLM 출력 편차를 보려면 2회 이상")
        workers = c3.number_input("동시 실행", 1, 8, 3,
                                  help="처리 시간을 정확히 비교하려면 1. 동시에 돌리면 CPU·API를 나눠 써서 시간이 늘어난다")
        use_cache = st.toggle("파싱 캐시 사용", True,
                              help="같은 문서·전략의 Docling 결과를 재사용. 처리 시간은 최초 파싱 시간을 더해 추정한다.")
        name = st.text_input("실험 이름", placeholder="예: luna-vs-sol")

    n_jobs = len(sel_samples) * len(sel_methods) * repeats
    st.info(f"실행 수: 문서 {len(sel_samples)} × 방법 {len(sel_methods)} × 반복 {repeats} = **{n_jobs}회**. "
            "실행 중 페이지를 벗어나면 중단되며, 그때까지의 결과는 저장된다.")

    if st.button("실험 실행", type="primary", disabled=n_jobs == 0):
        cfg = ExperimentConfig(
            extract=LLMConfig(model=ex_model, class_effort=class_effort, extract_effort=extract_effort),
            agent=AgentConfig(model=ag_model, effort=agent_effort, max_tool_calls=int(max_tools)),
            use_parse_cache=use_cache, repeats=int(repeats),
        )
        bar = st.progress(0.0, "준비 중 (파싱 캐시 생성)…")
        log = st.empty()
        live = st.empty()
        rows: list[dict] = []
        logs: list[str] = []

        def on_log(msg: str):
            logs.append(msg)
            log.code("\n".join(logs[-8:]))

        def on_progress(done: int, total: int, rec: dict):
            ev = rec.get("eval") or {}
            rows.append({"방법": METHODS[rec["method"]][0], "문서": rec["sample"], "상태": rec["status"],
                         "속성 정확도": ev.get("attr_accuracy"), "비용($)": rec["cost_usd"],
                         "시간(s)": round(rec["est_ms"] / 1000, 1), "오류": rec.get("error") or rec.get("agent_error")})
            bar.progress(done / total, f"{done}/{total} 완료")
            live.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

        exp_id = run_experiment([p for p in samples if p.name in sel_samples], sel_methods, cfg, name,
                                int(workers), on_progress, on_log)
        st.session_state["selected_experiments"] = [exp_id]
        st.success(f"완료: `{exp_id}` — ② 결과 비교 탭에서 확인")


# =========================================================================== ② 비교
def _hbar(df: pd.DataFrame, value: str, title: str, fmt: str):
    base = alt.Chart(df).encode(
        y=alt.Y("label:N", sort="-x", title=None),
        x=alt.X(f"{value}:Q", title=title),
        tooltip=[alt.Tooltip("label:N", title="방법"), alt.Tooltip(f"{value}:Q", title=title, format=fmt)],
    )
    bars = base.mark_bar(color=BAR, cornerRadiusEnd=4, height=14)
    text = base.mark_text(align="left", dx=4, color="gray").encode(text=alt.Text(f"{value}:Q", format=fmt))
    return (bars + text).properties(height=max(120, 34 * len(df)), title=title)


exps = list_experiments()
with tab_cmp:
    if not exps:
        st.info("아직 실험이 없습니다. ① 실행 탭에서 실험을 돌리세요.")
    else:
        labels = {e["experiment_id"]: f"{e['experiment_id']} · {e['n_records']}건 · {', '.join(e['methods'])}" for e in exps}
        default = [e for e in st.session_state.get("selected_experiments", []) if e in labels] or [exps[0]["experiment_id"]]
        chosen = st.multiselect("실험 선택 (여러 개 고르면 합쳐서 비교)", list(labels), default=default,
                                format_func=labels.get)
        records = load_records(chosen)
        summary, per_run = summarize(records)
        if summary.empty:
            st.warning("기록이 없습니다.")
        else:
            st.subheader("방법별 요약")
            st.dataframe(summary, hide_index=True, width="stretch", column_config={
                "클래스 정확도": st.column_config.NumberColumn(format="percent"),
                "속성 정확도": st.column_config.NumberColumn(format="percent"),
                "자동 통과율": st.column_config.NumberColumn(format="percent"),
                "에이전트 사용률": st.column_config.NumberColumn(format="percent"),
                "문서당 비용($)": st.column_config.NumberColumn(format="$%.5f"),
                "총 비용($)": st.column_config.NumberColumn(format="$%.5f"),
                "완전 정답 1건당 비용($)": st.column_config.NumberColumn(format="$%.5f"),
            })
            st.caption("**자동통과 오류** = 검토 없이 통과(extracted)했는데 틀린 문서 수. 0이어야 안전하다. "
                       "**완전 정답 1건당 비용** = 총 비용 ÷ 클래스·속성이 모두 맞은 문서 수 (설계안 12.3의 '성공 건당 총비용').")

            chart_df = summary.copy()
            used_models = set(chart_df["추출 모델"]) | (set(chart_df["에이전트 모델"]) - {"-"})
            multi_model = len(used_models) > 1
            suffix = (" · " + chart_df["추출 모델"] + "/" + chart_df["에이전트 모델"]) if multi_model else ""
            chart_df["label"] = chart_df["방법"] + suffix
            chart_df["short"] = chart_df["방법"].str.split().str[0] + suffix
            c1, c2 = st.columns(2)
            c1.altair_chart(_hbar(chart_df, "속성 정확도", "속성 정확도 (평균)", ".0%"), width="stretch")
            c2.altair_chart(_hbar(chart_df, "문서당 비용($)", "문서당 비용 (USD)", "$.5f"), width="stretch")
            c1, c2 = st.columns(2)
            c1.altair_chart(_hbar(chart_df, "평균 입력 토큰", "문서당 입력 토큰", ",.0f"), width="stretch")
            c2.altair_chart(_hbar(chart_df, "평균 처리 시간(s)", "문서당 처리 시간 (초)", ".1f"), width="stretch")

            st.subheader("정확도 대비 비용")
            pts = alt.Chart(chart_df).encode(
                x=alt.X("문서당 비용($):Q", title="문서당 비용 (USD)", scale=alt.Scale(zero=True)),
                y=alt.Y("속성 정확도:Q", title="속성 정확도", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
                tooltip=["label:N", alt.Tooltip("문서당 비용($):Q", format="$.5f"),
                         alt.Tooltip("속성 정확도:Q", format=".1%"), alt.Tooltip("평균 처리 시간(s):Q", format=".1f")],
            )
            st.altair_chart((pts.mark_circle(size=120, color=BAR, stroke="white", strokeWidth=2)
                             + pts.mark_text(align="left", dx=9, dy=-6, color="gray").encode(text="short:N")).properties(height=320),
                            width="stretch")
            st.caption("왼쪽 위(싸고 정확함)일수록 좋다. 점에 마우스를 올리면 방법 이름과 수치가 나온다.")

            st.subheader("문서 × 방법 속성 정확도")
            grid = per_run.copy()
            grid["label"] = grid["방법"].str.split().str[0] + (" · " + grid["추출 모델"] + "/" + grid["에이전트 모델"] if multi_model else "")
            grid = grid.groupby(["문서", "label"], as_index=False).agg(
                acc=("속성 정확도", "mean"), status=("상태", lambda s: "/".join(sorted(set(s)))),
                cost=("비용($)", "mean"))
            heat = alt.Chart(grid).encode(
                x=alt.X("label:N", title=None, axis=alt.Axis(labelAngle=-20)),
                y=alt.Y("문서:N", title=None),
                tooltip=["문서:N", alt.Tooltip("label:N", title="방법"), alt.Tooltip("acc:Q", title="속성 정확도", format=".0%"),
                         alt.Tooltip("status:N", title="상태"), alt.Tooltip("cost:Q", title="비용", format="$.5f")],
            )
            st.altair_chart((heat.mark_rect(stroke="white", strokeWidth=2).encode(
                color=alt.Color("acc:Q", title="정확도", scale=alt.Scale(domain=[0, 1], range=SEQ), legend=alt.Legend(format="%")))
                + heat.mark_text(fontSize=12).encode(
                    text=alt.Text("acc:Q", format=".0%"),
                    color=alt.condition("datum.acc > 0.55", alt.value("white"), alt.value("black"))))
                .properties(height=36 * grid["문서"].nunique() + 60), width="stretch")
            st.caption(" · ".join(label for label, _ in METHODS.values()))

            with st.expander("실행별 원자료"):
                st.dataframe(per_run, hide_index=True, width="stretch")
                st.download_button("CSV 다운로드", per_run.to_csv(index=False).encode("utf-8-sig"),
                                   file_name="runs.csv", mime="text/csv")


# =========================================================================== ③ 상세
with tab_detail:
    records = load_records([e["experiment_id"] for e in exps]) if exps else []
    if not records:
        st.info("실험 기록이 없습니다.")
    else:
        exp_ids = sorted({r["experiment_id"] for r in records}, reverse=True)
        c1, c2, c3 = st.columns(3)
        exp_id = c1.selectbox("실험", exp_ids)
        recs = [r for r in records if r["experiment_id"] == exp_id]
        method = c2.selectbox("방법", sorted({r["method"] for r in recs}), format_func=lambda m: METHODS.get(m, (m,))[0])
        recs = [r for r in recs if r["method"] == method]
        sample = c3.selectbox("문서", sorted({r["sample"] for r in recs}))
        recs = [r for r in recs if r["sample"] == sample]
        r = recs[0] if len(recs) == 1 else recs[st.number_input("반복 번호", 0, len(recs) - 1, 0)]

        ev = r.get("eval") or {}
        m = st.columns(6)
        m[0].metric("상태", r["status"])
        m[1].metric("클래스", r.get("class_id") or "null", "정답" if ev.get("class_correct") else ("오답" if ev else ""))
        m[2].metric("속성 정확도", f"{ev['attr_accuracy']:.0%}" if ev else "-")
        m[3].metric("비용", f"${r['cost_usd']:.5f}")
        m[4].metric("토큰 in/out", f"{r['tokens_in']:,} / {r['tokens_out']:,}")
        m[5].metric("처리 시간", f"{r['est_ms'] / 1000:.1f} s")
        if r.get("error") or r.get("agent_error"):
            st.error(r.get("error") or r.get("agent_error"))
        st.caption(f"파싱 전략: {' → '.join(r['parse_strategies']) or '-'} · 에이전트 사용: {r['agent_used']}"
                   + (" · 도구 호출 한도 도달" if r.get("hit_tool_limit") else ""))

        st.subheader("속성 결과 vs 정답")
        details = (ev or {}).get("details", {})
        attrs = {a["property_id"]: a for a in r.get("attributes", [])}
        issues_by: dict = {}
        for i in r.get("issues", []):
            issues_by.setdefault(i.get("property_id"), []).append(i["code"])
        icon = {"correct": "✅", "missed": "⬜ 누락", "wrong": "❌ 오답", "hallucinated": "⚠️ 환각"}
        rows = []
        for pid in dict.fromkeys([*details, *attrs]):
            a, d = attrs.get(pid, {}), details.get(pid, {})
            evd = a.get("evidence") or {}
            rows.append({"속성": pid, "판정": icon.get(d.get("verdict"), "-"), "예측": d.get("pred", a.get("raw_value")),
                         "정답": " | ".join("null" if g is None else g for g in d.get("gold", [])) if d else "-",
                         "근거 블록": evd.get("block_id"), "근거 원문": evd.get("text"),
                         "검증": ", ".join(issues_by.get(pid, [])) or "✔"})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        if r.get("review_reasons"):
            with st.expander(f"검토 사유 {len(r['review_reasons'])}개"):
                for x in r["review_reasons"]:
                    st.markdown(f"- {x}")

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("LLM 호출")
            st.dataframe(pd.DataFrame(r.get("model_calls", [])), hide_index=True, width="stretch")
        with c2:
            st.subheader("도구 호출")
            tc = pd.DataFrame(r.get("tool_calls", []))
            if tc.empty:
                st.caption("없음")
            else:
                tc["args"] = tc["args"].map(lambda a: json.dumps(a, ensure_ascii=False))
                st.dataframe(tc, hide_index=True, width="stretch")


# =========================================================================== ④ 과금
with tab_cost:
    st.caption(f"단가는 `{PRICING_FILE.relative_to(PRICING_FILE.parent.parent)}`에서 읽는다. 요금이 바뀌면 그 파일만 고치면 된다.")
    st.dataframe(pd.DataFrame([{"모델": k, "입력 $/1M": v["input"], "출력 $/1M": v["output"],
                                "캐시 입력 $/1M": v.get("cached_input", "-")} for k, v in pricing.items()]),
                 hide_index=True)
    if not exps:
        st.info("실험을 먼저 실행하세요.")
    else:
        st.subheader("월 처리량 기준 비용 추정")
        st.caption("선택한 실험의 방법별 평균 토큰을 그대로 두고 단가만 바꿔 계산한다. "
                   "모델을 바꾸면 실제 토큰 수와 정확도가 달라지므로 대략적인 비교용이다 — 정확히 보려면 그 모델로 실험을 다시 돌린다.")
        exp_for_cost = st.selectbox("기준 실험", [e["experiment_id"] for e in exps], key="cost_exp")
        docs_per_month = st.number_input("월 처리 문서 수", 10, 1_000_000, 1000, step=100)
        summary, _ = summarize(load_records([exp_for_cost]))
        rows = []
        for _, s in summary.iterrows():
            row = {"방법": s["방법"], "평균 입력 토큰": int(s["평균 입력 토큰"]), "평균 출력 토큰": int(s["평균 출력 토큰"])}
            for mname, p in pricing.items():
                per_doc = s["평균 입력 토큰"] / 1e6 * p["input"] + s["평균 출력 토큰"] / 1e6 * p["output"]
                row[f"{mname} 월 비용($)"] = round(per_doc * docs_per_month, 2)
            rows.append(row)
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
