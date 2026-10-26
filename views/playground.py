"""방법별 비교 실험 플레이그라운드.

실험 결과는 runs/<실험 ID>/records.jsonl 에 쌓이고, 불러올 때마다 현재 정답(gold.json)으로 다시 채점한다.
"""

from __future__ import annotations

import json
import time

import altair as alt
import pandas as pd
import streamlit as st

from extractor.lab import (
    CODE_ONLY,
    METHODS,
    SAMPLES_DIR,
    STAGES,
    build_variants,
    consistency_summary,
    estimate,
    list_experiments,
    list_samples,
    load_gold,
    load_records,
    property_summary,
    report_markdown,
    run_experiment,
    save_gold_entry,
    speed_summary,
    summarize,
)
from extractor.llm import PRICING_FILE, load_pricing
from extractor.ontology import JsonOntology

BAR = "#2a78d6"
SEQ = ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]  # 한 색상 단계 (값이 클수록 진함)
# 단계 색: 검증 스크립트 통과(대비 3:1 미만인 색이 있어 표를 함께 보여 준다)
STAGE_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
EFFORTS = ["none", "low", "medium", "high"]

st.title("실험 플레이그라운드")
st.caption("같은 문서·온톨로지·출력 스키마로 방법과 모델을 바꿔 가며 정확도·토큰·비용·시간을 비교한다. "
           "결과는 `runs/`에 쌓이고, 정답을 고치면 지난 실험도 다시 채점된다.")

tabs = st.tabs(["① 실행", "② 결과 비교", "③ 속도", "④ 속성·일관성", "⑤ 실행 상세", "⑥ 정답 관리", "⑦ 과금 시뮬레이션"])
tab_run, tab_cmp, tab_speed, tab_prop, tab_detail, tab_gold, tab_cost = tabs
pricing = load_pricing()
models = list(pricing)
default_model = "gpt-6-luna" if "gpt-6-luna" in models else models[0]
exps = list_experiments()
exp_labels = {e["experiment_id"]: f"{e['experiment_id']} · {e['n_records']}건 · {', '.join(e['methods'])}" for e in exps}


def pick_experiments(key: str) -> list[str]:
    if not exps:
        st.info("아직 실험이 없습니다. ① 실행 탭에서 실험을 돌리세요.")
        return []
    default = [e for e in st.session_state.get("selected_experiments", []) if e in exp_labels] or [exps[0]["experiment_id"]]
    return st.multiselect("실험 선택 (여러 개 고르면 합쳐서 비교)", list(exp_labels), default=default,
                          format_func=exp_labels.get, key=key)


def add_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    multi = df["설정"].nunique() > 1
    short = df["방법"].str.split().str[0]
    df["label"] = df["방법"] + (" · " + df["설정"] if multi else "")
    df["short"] = short + (" · " + df["설정"] if multi else "")
    return df


def hbar(df: pd.DataFrame, value: str, title: str, fmt: str):
    base = alt.Chart(df).encode(
        y=alt.Y("label:N", sort="-x", title=None),
        x=alt.X(f"{value}:Q", title=title),
        tooltip=[alt.Tooltip("label:N", title="방법"), alt.Tooltip(f"{value}:Q", title=title, format=fmt)],
    )
    bars = base.mark_bar(color=BAR, cornerRadiusEnd=4, height=14)
    text = base.mark_text(align="left", dx=4, color="gray").encode(text=alt.Text(f"{value}:Q", format=fmt))
    return (bars + text).properties(height=max(120, 34 * len(df)), title=title)


# =========================================================================== ① 실행
with tab_run:
    left, right = st.columns([1, 1])
    with left:
        st.subheader("대상")
        samples = list_samples()
        gold = load_gold()
        sel_samples = st.multiselect("문서", [p.name for p in samples], default=[p.name for p in samples],
                                     help="정답이 있는 문서만 정확도가 계산된다 (⑥ 정답 관리에서 추가)")
        no_gold = [s for s in sel_samples if s not in gold]
        if no_gold:
            st.caption("정답 없음(비용·시간만 기록): " + ", ".join(no_gold))
        sel_methods = st.multiselect(
            "방법", list(METHODS), default=["code_single", "code_rules", "code_agent", "agent_tools"],
            format_func=lambda m: f"{METHODS[m][0]} — {METHODS[m][1]}")
        st.caption("F·G(MCP)는 문서마다 MCP 서버를 새로 띄우므로 순차 실행되고 느리다.")
    with right:
        st.subheader("모델 · 설정")
        st.caption("여러 개 고르면 모든 조합을 한 실험에서 비교한다. A·B는 에이전트 설정과 무관해 02·03 설정별로 한 번만 돈다.")
        c1, c2 = st.columns(2)
        ex_models = c1.multiselect("02·03 모델", models, default=[default_model])
        ag_models = c2.multiselect("에이전트 모델", models, default=[default_model])
        c1, c2, c3 = st.columns(3)
        class_effort = c1.selectbox("02 effort", EFFORTS, index=1)
        extract_efforts = c2.multiselect("03 effort", EFFORTS, default=["low"])
        agent_efforts = c3.multiselect("에이전트 effort", EFFORTS, default=["low"])
        c1, c2, c3 = st.columns(3)
        max_tools = c1.number_input("에이전트 최대 도구 호출", 2, 30, 8)
        repeats = c2.number_input("반복 횟수", 1, 10, 1, help="LLM 출력 편차를 보려면 2회 이상 (④ 탭에서 일관성 확인)")
        workers = c3.number_input("동시 실행", 1, 8, 3,
                                  help="처리 시간을 정확히 비교하려면 1. 동시에 돌리면 CPU·API를 나눠 써서 시간이 늘어난다")
        speed_mode = st.toggle("⏱ 속도 측정 모드", False,
                               help="파싱 캐시를 읽지 않고 실제로 파싱하며, 동시 실행 1로 순서대로 돈다. "
                                    "시작 전에 Docling 모델을 미리 올려 첫 문서가 로딩 시간을 떠안지 않게 한다.")
        use_cache = not speed_mode and st.toggle("파싱 캐시 사용", True,
                                                  help="같은 문서·전략의 파싱 결과를 재사용. 처리 시간에는 최초 파싱 시간을 더해 추정한다.")
        name = st.text_input("실험 이름", placeholder="예: luna-vs-sol")

    variants = build_variants(ex_models or [default_model], class_effort, extract_efforts or ["low"],
                              ag_models or [default_model], agent_efforts or ["low"], int(max_tools),
                              use_cache, int(repeats))
    n_code = sum(m in CODE_ONLY for m in sel_methods)
    n_extract_variants = len({v.variant_label(False) for v in variants})
    n_jobs = len(sel_samples) * int(repeats) * (n_code * n_extract_variants + (len(sel_methods) - n_code) * len(variants))
    eff_workers = 1 if speed_mode else int(workers)

    est = estimate(sel_methods, len(sel_samples), int(repeats), variants, eff_workers) if n_jobs else None
    msg = f"실행 수 **{n_jobs}회** (문서 {len(sel_samples)} × 방법 {len(sel_methods)} × 설정 {len(variants)} × 반복 {repeats}, 중복 제외)"
    if est and est["per"]:
        msg += f" · 예상 비용 **${est['total_cost']:.4f}** · 예상 시간 **{est['total_s'] / 60:.1f}분** (지난 실험 기준 어림)"
    st.info(msg + ". 실행 중 페이지를 벗어나면 중단되며, 그때까지의 결과는 저장된다.")
    if est and est["missing"]:
        st.caption("기록이 없어 어림에서 빠진 방법: " + ", ".join(est["missing"]))
    if len(variants) > 8:
        st.warning(f"설정 조합이 {len(variants)}개입니다. 비용·시간이 커질 수 있습니다.")

    if st.button("실험 실행", type="primary", disabled=n_jobs == 0):
        bar = st.progress(0.0, "준비 중…")
        log = st.empty()
        live = st.empty()
        rows: list[dict] = []
        logs: list[str] = []

        def on_log(m: str):
            logs.append(m)
            log.code("\n".join(logs[-8:]))

        def on_progress(done: int, total: int, rec: dict):
            ev = rec.get("eval") or {}
            rows.append({"방법": METHODS[rec["method"]][0], "설정": rec["variant"], "문서": rec["sample"],
                         "상태": rec["status"], "속성 정확도": ev.get("attr_accuracy"), "비용($)": rec["cost_usd"],
                         "시간(s)": round(rec["est_ms"] / 1000, 1), "오류": rec.get("error") or rec.get("agent_error")})
            bar.progress(done / total, f"{done}/{total} 완료")
            live.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

        exp_id = run_experiment([p for p in samples if p.name in sel_samples], sel_methods, variants, name,
                                eff_workers, on_progress, on_log)
        st.session_state["selected_experiments"] = [exp_id]
        st.success(f"완료: `{exp_id}` — ② 결과 비교 탭에서 확인 (다른 탭에 반영하려면 새로고침)")


# =========================================================================== ② 비교
with tab_cmp:
    chosen = pick_experiments("cmp_exps")
    records = load_records(chosen) if chosen else []
    summary, per_run = summarize(records) if records else (pd.DataFrame(), pd.DataFrame())
    if records and not summary.empty:
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
        st.download_button("📝 보고서(Markdown) 다운로드", report_markdown(chosen).encode("utf-8"),
                           file_name="comparison_report.md", mime="text/markdown",
                           help="Jira·Confluence에 붙여 넣을 수 있는 표 형식")

        chart_df = add_labels(summary)
        c1, c2 = st.columns(2)
        c1.altair_chart(hbar(chart_df, "속성 정확도", "속성 정확도 (평균)", ".0%"), width="stretch")
        c2.altair_chart(hbar(chart_df, "문서당 비용($)", "문서당 비용 (USD)", "$.5f"), width="stretch")
        c1, c2 = st.columns(2)
        c1.altair_chart(hbar(chart_df, "평균 입력 토큰", "문서당 입력 토큰", ",.0f"), width="stretch")
        c2.altair_chart(hbar(chart_df, "완전 정답 1건당 비용($)", "완전 정답 1건당 비용 (USD)", "$.5f"), width="stretch")

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
        grid = add_labels(per_run)
        grid = grid.groupby(["문서", "short"], as_index=False).agg(
            acc=("속성 정확도", "mean"), status=("상태", lambda s: "/".join(sorted(set(s)))), cost=("비용($)", "mean"))
        heat = alt.Chart(grid).encode(
            x=alt.X("short:N", title=None, axis=alt.Axis(labelAngle=-20)), y=alt.Y("문서:N", title=None),
            tooltip=["문서:N", alt.Tooltip("short:N", title="방법"), alt.Tooltip("acc:Q", title="속성 정확도", format=".0%"),
                     alt.Tooltip("status:N", title="상태"), alt.Tooltip("cost:Q", title="비용", format="$.5f")])
        st.altair_chart((heat.mark_rect(stroke="white", strokeWidth=2).encode(
            color=alt.Color("acc:Q", title="정확도", scale=alt.Scale(domain=[0, 1], range=SEQ), legend=alt.Legend(format="%")))
            + heat.mark_text(fontSize=12).encode(text=alt.Text("acc:Q", format=".0%"),
                                                 color=alt.condition("datum.acc > 0.55", alt.value("white"), alt.value("black"))))
            .properties(height=36 * grid["문서"].nunique() + 60), width="stretch")
        st.caption(" · ".join(label for label, _ in METHODS.values()))

        with st.expander("실행별 원자료"):
            st.dataframe(per_run, hide_index=True, width="stretch")
            st.download_button("CSV 다운로드", per_run.to_csv(index=False).encode("utf-8-sig"),
                               file_name="runs.csv", mime="text/csv")


# =========================================================================== ③ 속도
with tab_speed:
    chosen = pick_experiments("speed_exps")
    records = load_records(chosen) if chosen else []
    if records:
        cfgs = {e["experiment_id"]: e for e in exps}
        unfair = [e for e in chosen if cfgs.get(e, {}).get("use_parse_cache", True) or cfgs.get(e, {}).get("workers", 3) > 1]
        if unfair:
            st.warning("다음 실험은 동시 실행 또는 파싱 캐시를 써서 시간 비교가 정확하지 않을 수 있다 "
                       f"(캐시 시 파싱 시간은 최초 측정값으로 추정): {', '.join(unfair)}. "
                       "정확히 비교하려면 ① 실행에서 **속도 측정 모드**로 다시 돌린다.")
        speed = speed_summary(records)
        st.subheader("처리 시간 분포")
        st.dataframe(speed, hide_index=True, width="stretch")
        st.caption("p50 = 중앙값, p95 = 느린 쪽 5%를 뺀 최댓값. 단계: 파싱(대체 파싱·OCR 포함) · LLM(모델 호출 합) · "
                   "도구(파싱 외 도구 호출) · MCP 기동(서버 프로세스 시작) · 기타(그래프·검증·오버헤드).")

        speed = add_labels(speed)
        long = speed.melt(id_vars=["label", "short"], value_vars=[f"평균 {s}(s)" for s in STAGES],
                          var_name="단계", value_name="초")
        long["단계"] = long["단계"].str.replace("평균 ", "").str.replace("(s)", "", regex=False)
        long["순서"] = long["단계"].map({s: i for i, s in enumerate(STAGES)})
        stacked = alt.Chart(long).mark_bar(stroke="white", strokeWidth=2).encode(
            y=alt.Y("short:N", title=None),
            x=alt.X("초:Q", title="문서당 평균 처리 시간 (초)", stack="zero"),
            color=alt.Color("단계:N", scale=alt.Scale(domain=STAGES, range=STAGE_COLORS),
                            legend=alt.Legend(orient="top", title=None)),
            order=alt.Order("순서:Q"),
            tooltip=[alt.Tooltip("label:N", title="방법"), "단계:N", alt.Tooltip("초:Q", format=".2f")],
        )
        st.subheader("단계별 평균 시간")
        st.altair_chart(stacked.properties(height=alt.Step(30)), width="stretch")
        with st.expander("단계별 시간 표"):
            st.dataframe(long.pivot_table(index="label", columns="단계", values="초")[STAGES], width="stretch")

        st.subheader("실행별 처리 시간")
        _, runs_df = summarize(records)
        runs = add_labels(runs_df)
        strip = alt.Chart(runs).mark_circle(size=70, color=BAR, opacity=0.7, stroke="white", strokeWidth=1).encode(
            y=alt.Y("short:N", title=None), x=alt.X("처리 시간(s):Q", title="처리 시간 (초)"),
            tooltip=[alt.Tooltip("label:N", title="방법"), "문서:N", alt.Tooltip("처리 시간(s):Q", format=".1f"), "상태:N"])
        st.altair_chart(strip.properties(height=alt.Step(30)), width="stretch")
        st.caption("점 하나가 문서 하나의 실행. 오른쪽으로 길게 퍼질수록 문서에 따라 시간이 크게 달라진다 (대체 파싱·OCR·에이전트 호출).")


# =========================================================================== ④ 속성·일관성
with tab_prop:
    chosen = pick_experiments("prop_exps")
    records = load_records(chosen) if chosen else []
    if records:
        props = property_summary(records)
        if props.empty:
            st.info("정답이 있는 실행이 없습니다.")
        else:
            st.subheader("속성별 정답률")
            st.caption("어떤 속성을 어느 방법이 자주 놓치는지 본다. 프롬프트·동의어·대체 파싱 규칙을 고칠 곳을 찾는 용도.")
            props = add_labels(props)
            heat = alt.Chart(props).encode(
                x=alt.X("short:N", title=None, axis=alt.Axis(labelAngle=-20)), y=alt.Y("속성:N", title=None),
                tooltip=["속성:N", alt.Tooltip("label:N", title="방법"), alt.Tooltip("정답률:Q", format=".0%"),
                         "실행수:Q", "누락:Q", "오답:Q", "환각:Q"])
            st.altair_chart((heat.mark_rect(stroke="white", strokeWidth=2).encode(
                color=alt.Color("정답률:Q", scale=alt.Scale(domain=[0, 1], range=SEQ), legend=alt.Legend(format="%")))
                + heat.mark_text(fontSize=11).encode(text=alt.Text("정답률:Q", format=".0%"),
                                                     color=alt.condition("datum['정답률'] > 0.55", alt.value("white"), alt.value("black"))))
                .properties(height=24 * props["속성"].nunique() + 60), width="stretch")
            with st.expander("속성별 표"):
                st.dataframe(props.drop(columns=["label", "short"]).sort_values("정답률"), hide_index=True, width="stretch")

        st.subheader("반복 일관성")
        cons, cons_detail = consistency_summary(records)
        if cons.empty:
            st.info("같은 설정을 2회 이상 돌린 실험에서 표시된다. ① 실행에서 반복 횟수를 2 이상으로 설정한다.")
        else:
            st.dataframe(cons, hide_index=True, width="stretch", column_config={
                "평균 속성값 일치율": st.column_config.NumberColumn(format="percent"),
                "상태 일치율": st.column_config.NumberColumn(format="percent")})
            st.caption("속성값 일치율 = 반복 실행에서 같은 값을 낸 속성의 비율. 낮을수록 실행마다 결과가 흔들린다.")
            with st.expander("문서별"):
                st.dataframe(cons_detail, hide_index=True, width="stretch")


# =========================================================================== ⑤ 상세
with tab_detail:
    records = load_records([e["experiment_id"] for e in exps]) if exps else []
    if not records:
        st.info("실험 기록이 없습니다.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        exp_id = c1.selectbox("실험", sorted({r["experiment_id"] for r in records}, reverse=True))
        recs = [r for r in records if r["experiment_id"] == exp_id]
        method = c2.selectbox("방법", sorted({r["method"] for r in recs}), format_func=lambda m: METHODS.get(m, (m,))[0])
        recs = [r for r in recs if r["method"] == method]
        variant = c3.selectbox("설정", sorted({r["variant"] for r in recs}))
        recs = [r for r in recs if r["variant"] == variant]
        sample = c4.selectbox("문서", sorted({r["sample"] for r in recs}))
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
        timing = r.get("timing") or {}
        st.caption("시간 구성: " + " · ".join(f"{s} {timing.get(s, 0) / 1000:.1f}s" for s in STAGES)
                   + f" · 파싱 전략: {' → '.join(r['parse_strategies']) or '-'} · 에이전트 사용: {r['agent_used']}"
                   + (" · 도구 호출 한도 도달" if r.get("hit_tool_limit") else ""))
        if r.get("error") or r.get("agent_error"):
            st.error(r.get("error") or r.get("agent_error"))

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


# =========================================================================== ⑥ 정답 관리
with tab_gold:
    st.caption("평가셋(설계안 15.1: 20~30건)을 늘리는 곳. 문서를 올리고 → 실행 결과로 미리 채운 뒤 → 고쳐서 저장한다. "
               "저장하면 지난 실험도 새 정답으로 다시 채점된다.")
    up = st.file_uploader("새 문서를 samples/에 추가", type=["pdf", "xlsx"], key="gold_upload")
    if up is not None:
        target = SAMPLES_DIR / up.name
        if target.exists():
            st.warning(f"`{up.name}`이 이미 있습니다. 다른 이름으로 올리세요.")
        elif st.button(f"`{up.name}` 추가"):
            target.write_bytes(up.getvalue())
            st.success("추가했습니다. 새로고침하면 목록에 나옵니다.")

    gold = load_gold()
    names = [p.name for p in list_samples()]
    gsample = st.selectbox("문서", names, format_func=lambda n: f"{n} {'✅ 정답 있음' if n in gold else '— 정답 없음'}")
    onto = JsonOntology()
    cands = onto.class_candidates()
    entry = gold.get(gsample) or {}

    # 미리 채우기: 기존 정답 → 없으면 가장 최근 실행 결과
    all_recs = load_records([e["experiment_id"] for e in exps], reevaluate=False) if exps else []
    recent = [r for r in all_recs if r["sample"] == gsample and r.get("class_id")]
    prefill_rec = None
    if not entry and recent:
        options = {f"{r['experiment_id']} · {METHODS[r['method']][0]} · {r['variant']}": r for r in recent}
        key = st.selectbox("미리 채울 실행 결과", list(options), help="에이전트 결과도 고를 수 있다. 반드시 원문과 대조해 고친다.")
        prefill_rec = options[key]

    class_opts = ["(없음: 온톨로지 밖 제품)"] + [c["id"] for c in cands]
    cur_class = entry.get("class_id") if entry else (prefill_rec or {}).get("class_id")
    gclass = st.selectbox("정답 클래스", class_opts, index=class_opts.index(cur_class) if cur_class in class_opts else 0,
                          format_func=lambda c: c if c.startswith("(") else f"{c} ({next(x['label'] for x in cands if x['id'] == c)})")
    gclass = None if gclass.startswith("(") else gclass

    if gclass:
        def fmt(vals):
            return " | ".join("null" if v is None else v for v in vals)

        pred = {a["property_id"]: a for a in (prefill_rec or {}).get("attributes", [])}
        rows = []
        for p in onto.properties_of(gclass):
            if entry and gclass == entry.get("class_id"):
                val = fmt(entry["attributes"].get(p["id"], [None]))
            else:
                a = pred.get(p["id"]) or {}
                v = a.get("raw_value")
                if v and a.get("raw_unit"):
                    v = f"{v} {a['raw_unit']}"
                val = v or "null"
            rows.append({"property_id": p["id"], "속성": p["label"], "필수": "●" if p.get("required") else "",
                         "정답 (허용 표기를 | 로 구분, 원문에 없으면 null)": val})
        edited = st.data_editor(pd.DataFrame(rows), hide_index=True, width="stretch",
                                disabled=["property_id", "속성", "필수"], key=f"gold_editor_{gsample}_{gclass}")
    else:
        edited = None
        st.caption("온톨로지 밖 제품은 class_id=null로 저장하며 속성은 비운다.")

    if st.button("정답 저장", type="primary"):
        attrs = {}
        if edited is not None:
            for _, row in edited.iterrows():
                vals = [v.strip() for v in str(row.iloc[3]).split("|")]
                attrs[row["property_id"]] = [None if v.lower() in ("null", "") else v for v in vals]
        save_gold_entry(gsample, gclass, attrs)
        st.success(f"`{gsample}` 정답을 저장했습니다 (data/eval/gold.json). 커밋해야 팀과 공유된다.")
        time.sleep(0.5)
        st.rerun()


# =========================================================================== ⑦ 과금
with tab_cost:
    st.caption(f"단가는 `{PRICING_FILE.relative_to(PRICING_FILE.parent.parent)}`에서 읽는다. 요금이 바뀌면 그 파일만 고치면 된다.")
    st.dataframe(pd.DataFrame([{"모델": k, "입력 $/1M": v["input"], "출력 $/1M": v["output"],
                                "캐시 입력 $/1M": v.get("cached_input", "-")} for k, v in pricing.items()]),
                 hide_index=True)
    if exps:
        st.subheader("월 처리량 기준 비용 추정")
        st.caption("선택한 실험의 방법별 평균 토큰을 그대로 두고 단가만 바꿔 계산한다. "
                   "모델을 바꾸면 실제 토큰 수와 정확도가 달라지므로 대략적인 비교용이다 — 정확히 보려면 ①에서 그 모델로 돌린다.")
        exp_for_cost = st.selectbox("기준 실험", [e["experiment_id"] for e in exps], key="cost_exp")
        docs_per_month = st.number_input("월 처리 문서 수", 10, 1_000_000, 1000, step=100)
        summary, _ = summarize(load_records([exp_for_cost]))
        rows = []
        for _, s in summary.iterrows():
            row = {"방법": s["방법"], "설정": s["설정"], "평균 입력 토큰": int(s["평균 입력 토큰"]),
                   "평균 출력 토큰": int(s["평균 출력 토큰"])}
            for mname, p in pricing.items():
                per_doc = s["평균 입력 토큰"] / 1e6 * p["input"] + s["평균 출력 토큰"] / 1e6 * p["output"]
                row[f"{mname} 월 비용($)"] = round(per_doc * docs_per_month, 2)
            rows.append(row)
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
