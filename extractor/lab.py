"""방법별 비교 실험 (설계안 14·15절).

  python -m extractor.lab run --methods code_single,code_rules,code_agent --samples all
  python -m extractor.lab summary <실험 ID>
  python -m extractor.lab list

결과는 runs/<실험 ID>/ 에 config.json, records.jsonl 로 쌓인다.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import threading
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Callable

from pydantic import BaseModel

from .agent import AgentAttribute, AgentConfig, AgentEvidence, AgentResult, AgentRun
from .agent import arun_mcp_agent, run_fallback_agent, run_full_agent, validate_agent_result
from .agent_tools import DocumentWorkspace
from .graph import build_graph, initial_state
from .llm import LLMConfig
from .parsing import initial_strategy, parse_cached, parse_document
from .schemas import ParsedDocument

ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "runs"
SAMPLES_DIR = ROOT / "samples"
GOLD_FILE = ROOT / "data" / "eval" / "gold.json"

METHODS: dict[str, tuple[str, str]] = {
    "code_single": ("A 코드 단일", "Docling 1회 파싱 → 02 → 03 (03 코드 재시도만)"),
    "code_rules": ("B 코드+규칙", "품질 경고 시 규칙 기반 대체 파싱 → 02 → 03"),
    "code_agent": ("C 코드→에이전트", "A가 검토 필요로 끝나면 남은 부분만 에이전트가 도구로 보완"),
    "code_rules_agent": ("D 코드+규칙→에이전트", "B가 검토 필요로 끝나면 에이전트가 보완"),
    "agent_tools": ("E 에이전트 단독", "처음부터 에이전트가 자체 도구(인프로세스)로 처리"),
    "agent_mcp_custom": ("F 에이전트+자체 MCP", "E와 같은 도구를 MCP(stdio)로 호출 — MCP 오버헤드 측정용"),
    "agent_mcp_docling": ("G 에이전트+Docling MCP", "docling-mcp 기본 도구(변환·구조·검색·텍스트) 사용"),
}
MCP_METHODS = {"agent_mcp_custom", "agent_mcp_docling"}


@dataclass
class ExperimentConfig:
    extract: LLMConfig = field(default_factory=LLMConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    use_parse_cache: bool = True
    repeats: int = 1


class RunRecord(BaseModel):
    experiment_id: str
    method: str
    sample: str
    repeat: int = 0
    model: str
    agent_model: str | None = None
    status: str
    class_id: str | None = None
    class_reason: str | None = None
    attributes: list[dict] = []
    issues: list[dict] = []
    review_reasons: list[str] = []
    parse_strategies: list[str] = []
    parse_ms: int = 0
    agent_used: bool = False
    agent_error: str | None = None
    hit_tool_limit: bool = False
    model_calls: list[dict] = []
    tool_calls: list[dict] = []
    wall_ms: int = 0
    est_ms: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    tokens_reasoning: int = 0
    cost_usd: float = 0.0
    error: str | None = None
    eval: dict | None = None


# ---------------------------------------------------------------------------
# 방법별 실행
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _graph():
    return build_graph()


@lru_cache(maxsize=1)
def _mcp_loop() -> asyncio.AbstractEventLoop:
    # langchain-openai가 비동기 HTTP 클라이언트를 전역으로 재사용하므로 asyncio.run을 여러 번 쓰면
    # 닫힌 루프에 묶인 클라이언트가 남는다. MCP 실행은 하나의 루프에서만 돌린다.
    return asyncio.new_event_loop()


def _code_pipeline(path: Path, cfg: ExperimentConfig, rules: bool):
    strategy = "auto" if rules else initial_strategy(path)
    best, attempts = parse_document(path, strategy, use_cache=cfg.use_parse_cache)
    final = _graph().invoke(initial_state(str(path), cfg.extract, best))
    return best, attempts, final


def _baseline_summary(best: ParsedDocument, final: dict) -> dict:
    return {
        "parse_strategy": best.parser,
        "parse_status": best.status,
        "parse_warnings": [{"code": w.code, "page": w.page, "message": w.message} for w in best.warnings],
        "class_result": final.get("class_result"),
        "attributes": (final.get("extraction") or {}).get("attributes", []),
        "issues": final.get("class_issues", []) + final.get("extraction_issues", []),
        "review_reasons": final.get("review_reasons", []),
    }


def _agent_status(class_id, issues, needs_review: bool) -> str:
    return "extracted" if class_id and not issues and not needs_review else "needs_review"


def _apply_agent(rec: dict, ws: DocumentWorkspace, run: AgentRun, merged: AgentResult | None):
    rec.update(agent_used=True, agent_error=run.error, hit_tool_limit=run.hit_tool_limit)
    rec["model_calls"] += [m.model_dump() for m in run.model_calls]
    rec["tool_calls"] = [{k: v for k, v in t.items() if k != "output_full"} for t in run.tool_calls]
    if merged is None:
        return
    class_res, attrs, issues = validate_agent_result(merged, ws)
    rec.update(class_id=class_res.class_id, class_reason=class_res.reason,
               attributes=[a.model_dump() for a in attrs], issues=[i.model_dump() for i in issues],
               status=_agent_status(class_res.class_id, issues, merged.needs_review),
               review_reasons=[f"{i.code} [{i.property_id}]: {i.message}" for i in issues]
               + (["에이전트가 검토 필요로 표시"] if merged.needs_review else []) + merged.warnings)
    for s, d in ws.docs.items():
        if s not in rec["parse_strategies"]:
            rec["parse_strategies"].append(s)
            rec["parse_ms"] += d.elapsed_ms or 0
            rec["_saved_ms"] += (d.elapsed_ms or 0) if d.from_cache else 0


def _merge_fallback(baseline_strategy: str, final: dict, agent: AgentResult, targets: list[str],
                    class_undecided: bool) -> AgentResult:
    if class_undecided:
        return agent
    cr = final["class_result"]
    new = {a.property_id: a for a in agent.attributes if a.property_id in targets}
    attrs = []
    for a in final["extraction"]["attributes"]:
        if a["property_id"] in new:
            attrs.append(new[a["property_id"]])
            continue
        ev = a.get("evidence")
        attrs.append(AgentAttribute(property_id=a["property_id"], raw_value=a["raw_value"], raw_unit=a.get("raw_unit"),
                                    evidence=AgentEvidence(source=baseline_strategy, **ev) if ev else None))
    cev = cr.get("evidence")
    return AgentResult(class_id=cr["class_id"], class_reason=cr["reason"],
                       class_evidence=AgentEvidence(source=baseline_strategy, **cev) if cev else None,
                       attributes=attrs, needs_review=agent.needs_review, warnings=agent.warnings)


def run_method(method: str, path: Path, cfg: ExperimentConfig) -> dict:
    t0 = time.perf_counter()
    rec: dict = dict(method=method, sample=path.name, model=cfg.extract.model, status="failed",
                     model_calls=[], tool_calls=[], parse_strategies=[], parse_ms=0, _saved_ms=0)
    uses_agent = method not in ("code_single", "code_rules")
    if uses_agent:
        rec["agent_model"] = cfg.agent.model

    if method in ("code_single", "code_rules", "code_agent", "code_rules_agent"):
        best, attempts, final = _code_pipeline(path, cfg, rules=method in ("code_rules", "code_rules_agent"))
        ex = final.get("extraction") or {}
        cr = final.get("class_result") or {}
        rec.update(
            status=final["status"], class_id=cr.get("class_id"), class_reason=cr.get("reason"),
            attributes=ex.get("attributes", []), review_reasons=final.get("review_reasons", []),
            issues=final.get("class_issues", []) + final.get("extraction_issues", []),
            parse_strategies=[a.parser for a in attempts],
            parse_ms=sum(a.elapsed_ms or 0 for a in attempts),
            _saved_ms=sum(a.elapsed_ms or 0 for a in attempts if a.from_cache),
            model_calls=final.get("model_runs", []),
        )
        if method in ("code_agent", "code_rules_agent") and final["status"] != "extracted":
            ws = DocumentWorkspace(path)
            for a in attempts:
                ws.docs[a.parser] = a
            class_undecided = not cr.get("class_id") or not final.get("extraction")
            # 오류가 난 속성 + 값이 비어 있는 속성 (선택 속성의 null은 오류가 안 나서 따로 넣는다)
            targets = sorted({i["property_id"] for i in final.get("extraction_issues", []) if i.get("property_id")}
                             | {a["property_id"] for a in ex.get("attributes", []) if a.get("raw_value") is None})
            if not class_undecided and not targets:
                targets = [a["property_id"] for a in ex.get("attributes", [])]
            run = run_fallback_agent(ws, cfg.agent, _baseline_summary(best, final), targets, class_undecided)
            merged = _merge_fallback(best.parser, final, run.result, targets, class_undecided) if run.result else None
            _apply_agent(rec, ws, run, merged)

    elif method == "agent_tools":
        ws = DocumentWorkspace(path)
        run = run_full_agent(ws, cfg.agent)
        rec["status"] = "failed"
        _apply_agent(rec, ws, run, run.result)

    elif method in MCP_METHODS:
        ws = DocumentWorkspace(path)
        server = "docling" if method == "agent_mcp_docling" else "custom"
        run = _mcp_loop().run_until_complete(arun_mcp_agent(ws, cfg.agent, server))
        _apply_agent(rec, ws, run, run.result)
    else:
        raise ValueError(f"알 수 없는 방법: {method}")

    calls = rec["model_calls"]
    rec["wall_ms"] = int((time.perf_counter() - t0) * 1000)
    rec["est_ms"] = rec["wall_ms"] + rec.pop("_saved_ms")
    rec["tokens_in"] = sum(c.get("input_tokens") or 0 for c in calls)
    rec["tokens_out"] = sum(c.get("output_tokens") or 0 for c in calls)
    rec["tokens_reasoning"] = sum(c.get("reasoning_tokens") or 0 for c in calls)
    rec["cost_usd"] = round(sum(c.get("cost_usd") or 0 for c in calls), 6)
    return rec


# ---------------------------------------------------------------------------
# 평가
# ---------------------------------------------------------------------------


def _cmp(s: str | None) -> str | None:
    if s is None:
        return None
    s = unicodedata.normalize("NFKC", s).casefold()
    s = s.replace("·", "-").replace("∙", "-").replace("⋅", "-")
    return re.sub(r"[\s,]", "", s)


def load_gold() -> dict:
    return json.loads(GOLD_FILE.read_text(encoding="utf-8"))


def evaluate(rec: dict, gold: dict | None) -> dict | None:
    if gold is None:
        return None
    class_ok = rec.get("class_id") == gold["class_id"]
    pred = {a["property_id"]: a for a in rec.get("attributes", [])} if class_ok else {}
    issue_props = {i.get("property_id") for i in rec.get("issues", []) if i.get("code", "").startswith(("EVIDENCE", "VALUE_NOT"))}
    res = dict(class_correct=class_ok, n_props=len(gold["attributes"]), correct=0, wrong=0, missed=0,
               hallucinated=0, details={})
    evid_total = evid_ok = 0
    for pid, accept in gold["attributes"].items():
        a = pred.get(pid)
        value = None if a is None else (a.get("raw_value") or None)
        if value is not None and a.get("raw_unit") and _cmp(a["raw_unit"]) not in _cmp(value):
            value = f"{value} {a['raw_unit']}"
        ok_vals = {_cmp(v) for v in accept}
        if _cmp(value) in ok_vals:
            verdict = "correct"
        elif value is None:
            verdict = "missed"
        elif ok_vals == {None}:
            verdict = "hallucinated"
        else:
            verdict = "wrong"
        res[verdict] += 1
        res["details"][pid] = {"pred": value, "gold": accept, "verdict": verdict}
        if value is not None:
            evid_total += 1
            evid_ok += pid not in issue_props
    n = res["n_props"]
    res["attr_accuracy"] = round(res["correct"] / n, 3) if n else (1.0 if class_ok else 0.0)
    res["fully_correct"] = class_ok and res["correct"] == n
    res["evidence_ok_rate"] = round(evid_ok / evid_total, 3) if evid_total else None
    # 자동 통과했는데 틀린 경우가 가장 위험하다 (검토 없이 등록됨).
    res["false_accept"] = rec.get("status") == "extracted" and not res["fully_correct"]
    return res


# ---------------------------------------------------------------------------
# 실행기
# ---------------------------------------------------------------------------


def list_samples() -> list[Path]:
    return sorted(p for p in SAMPLES_DIR.iterdir() if p.suffix.lower() in (".pdf", ".xlsx") and not p.name.startswith("_"))


def _prewarm(paths: list[Path], methods: list[str], log: Callable[[str], None]):
    """Docling은 스레드 안전하지 않으니 병렬 실행 전에 필요한 파싱을 순차로 캐시해 둔다."""
    agent = any(m in METHODS and m not in ("code_single", "code_rules") for m in methods)
    for p in paths:
        ws = DocumentWorkspace(p)
        strategies = ws.strategies if agent else [initial_strategy(p)]
        for s in strategies:
            t = time.perf_counter()
            d = parse_cached(p, s)
            if not d.from_cache:
                log(f"파싱 캐시 생성 {p.name} / {s} ({int((time.perf_counter() - t) * 1000)} ms)")
        if any(m in ("code_rules", "code_rules_agent") for m in methods):
            parse_document(p, "auto", use_cache=True)


def run_experiment(
    samples: list[Path], methods: list[str], cfg: ExperimentConfig, name: str = "",
    workers: int = 3, progress: Callable[[int, int, dict], None] | None = None,
    log: Callable[[str], None] = lambda m: print(m, file=sys.stderr),
) -> str:
    exp_id = datetime.now().strftime("%Y%m%d-%H%M%S") + (f"_{re.sub(r'[^0-9A-Za-z가-힣_-]+', '-', name)}" if name else "")
    out = RUNS_DIR / exp_id
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps({
        "experiment_id": exp_id, "name": name, "created_at": datetime.now().isoformat(timespec="seconds"),
        "methods": methods, "samples": [p.name for p in samples],
        "extract": asdict(cfg.extract), "agent": asdict(cfg.agent),
        "use_parse_cache": cfg.use_parse_cache, "repeats": cfg.repeats,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    if cfg.use_parse_cache:
        _prewarm(samples, methods, log)
    gold = load_gold()
    jobs = [(m, p, r) for r in range(cfg.repeats) for p in samples for m in methods]
    lock = threading.Lock()
    done = 0

    def one(m: str, p: Path, r: int) -> dict:
        try:
            rec = run_method(m, p, cfg)
        except Exception as e:
            rec = dict(method=m, sample=p.name, model=cfg.extract.model, status="failed", error=f"{type(e).__name__}: {e}")
        rec.update(experiment_id=exp_id, repeat=r)
        rec["eval"] = evaluate(rec, gold.get(p.name))
        return RunRecord.model_validate(rec).model_dump()

    def save(rec: dict):
        nonlocal done
        with lock:
            with (out / "records.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            done += 1
            if progress:
                progress(done, len(jobs), rec)

    parallel = [j for j in jobs if j[0] not in MCP_METHODS]
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for fut in as_completed([pool.submit(one, *j) for j in parallel]):
            save(fut.result())
    for j in jobs:  # MCP 방법은 서버 프로세스를 띄우므로 순차 실행
        if j[0] in MCP_METHODS:
            save(one(*j))
    return exp_id


# ---------------------------------------------------------------------------
# 결과 조회 · 요약
# ---------------------------------------------------------------------------


def list_experiments() -> list[dict]:
    out = []
    for d in sorted(RUNS_DIR.glob("*/config.json"), reverse=True):
        cfg = json.loads(d.read_text(encoding="utf-8"))
        recs = d.parent / "records.jsonl"
        cfg["n_records"] = sum(1 for _ in recs.open(encoding="utf-8")) if recs.exists() else 0
        out.append(cfg)
    return out


def load_records(exp_ids: list[str]) -> list[dict]:
    recs = []
    for e in exp_ids:
        f = RUNS_DIR / e / "records.jsonl"
        if f.exists():
            recs += [json.loads(line) for line in f.open(encoding="utf-8") if line.strip()]
    return recs


def summarize(records: list[dict]):
    import pandas as pd

    rows = []
    for r in records:
        ev = r.get("eval") or {}
        rows.append({
            "실험": r["experiment_id"], "방법": METHODS.get(r["method"], (r["method"],))[0], "method": r["method"],
            "추출 모델": r["model"], "에이전트 모델": r.get("agent_model") or "-", "문서": r["sample"],
            "상태": r["status"], "클래스 정답": ev.get("class_correct"), "속성 정확도": ev.get("attr_accuracy"),
            "완전 정답": ev.get("fully_correct"), "자동통과 오류": ev.get("false_accept"),
            "근거 일치율": ev.get("evidence_ok_rate"),
            "환각": ev.get("hallucinated"), "에이전트 사용": r.get("agent_used"),
            "LLM 호출": len(r.get("model_calls", [])), "도구 호출": len([t for t in r.get("tool_calls", []) if not t["tool"].startswith("(")]),
            "입력 토큰": r.get("tokens_in"), "출력 토큰": r.get("tokens_out"), "추론 토큰": r.get("tokens_reasoning"),
            "비용($)": r.get("cost_usd"), "처리 시간(s)": round((r.get("est_ms") or 0) / 1000, 2),
            "오류": r.get("error") or r.get("agent_error"),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df, df
    g = df.groupby(["방법", "추출 모델", "에이전트 모델"], sort=False)
    summary = g.agg(**{
        "문서 수": ("문서", "count"),
        "클래스 정확도": ("클래스 정답", "mean"),
        "속성 정확도": ("속성 정확도", "mean"),
        "완전 정답 문서": ("완전 정답", "sum"),
        "자동 통과율": ("상태", lambda s: (s == "extracted").mean()),
        "자동통과 오류": ("자동통과 오류", "sum"),
        "환각 합계": ("환각", "sum"),
        "에이전트 사용률": ("에이전트 사용", "mean"),
        "평균 LLM 호출": ("LLM 호출", "mean"),
        "평균 도구 호출": ("도구 호출", "mean"),
        "평균 입력 토큰": ("입력 토큰", "mean"),
        "평균 출력 토큰": ("출력 토큰", "mean"),
        "문서당 비용($)": ("비용($)", "mean"),
        "총 비용($)": ("비용($)", "sum"),
        "평균 처리 시간(s)": ("처리 시간(s)", "mean"),
    }).reset_index()
    summary["완전 정답 1건당 비용($)"] = summary["총 비용($)"] / summary["완전 정답 문서"].where(summary["완전 정답 문서"] > 0)
    return summary.round(4), df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    import pandas as pd

    ap = argparse.ArgumentParser(prog="extractor.lab")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--methods", default="code_single,code_rules,code_agent,agent_tools")
    r.add_argument("--samples", default="all", help="all 또는 쉼표로 구분한 파일명")
    r.add_argument("--model", default=None, help="02·03 모델")
    r.add_argument("--agent-model", default=None)
    r.add_argument("--agent-effort", default=None)
    r.add_argument("--max-tool-calls", type=int, default=None)
    r.add_argument("--repeats", type=int, default=1)
    r.add_argument("--workers", type=int, default=3)
    r.add_argument("--no-cache", action="store_true")
    r.add_argument("--name", default="")
    s = sub.add_parser("summary")
    s.add_argument("experiment_ids", nargs="+")
    sub.add_parser("list")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)

    if a.cmd == "list":
        for e in list_experiments():
            print(e["experiment_id"], e["n_records"], ",".join(e["methods"]))
        return
    if a.cmd == "summary":
        summary, _ = summarize(load_records(a.experiment_ids))
        print(summary.to_string(index=False))
        return

    cfg = ExperimentConfig(use_parse_cache=not a.no_cache, repeats=a.repeats)
    if a.model:
        cfg.extract.model = a.model
    if a.agent_model:
        cfg.agent.model = a.agent_model
    if a.agent_effort:
        cfg.agent.effort = a.agent_effort
    if a.max_tool_calls:
        cfg.agent.max_tool_calls = a.max_tool_calls
    methods = a.methods.split(",")
    for m in methods:
        if m not in METHODS:
            ap.error(f"알 수 없는 방법 {m}. 가능: {', '.join(METHODS)}")
    samples = list_samples() if a.samples == "all" else [SAMPLES_DIR / s for s in a.samples.split(",")]

    def progress(done, total, rec):
        ev = rec.get("eval") or {}
        print(f"[{done}/{total}] {rec['method']:18} {rec['sample']:28} {rec['status']:12} "
              f"acc={ev.get('attr_accuracy')} ${rec['cost_usd']:.5f} {rec['est_ms'] / 1000:.1f}s"
              + (f" ERR {rec.get('error') or rec.get('agent_error')}" if rec.get("error") or rec.get("agent_error") else ""),
              file=sys.stderr)

    exp_id = run_experiment(samples, methods, cfg, a.name, a.workers, progress)
    summary, _ = summarize(load_records([exp_id]))
    print(f"\n실험 {exp_id}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
