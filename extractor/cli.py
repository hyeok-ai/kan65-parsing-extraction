"""명령행 실행.

  python -m extractor.cli parse samples/motor_spec.pdf --overlay out/        # 01만, bbox 이미지 저장
  python -m extractor.cli run samples/motor_spec.pdf --out out/motor.json    # 01~03 그래프
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .parsing import PDF_STRATEGIES, parse_document


def _dump(obj, out: str | None):
    text = json.dumps(obj, ensure_ascii=False, indent=2)
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(text, encoding="utf-8")
        print(f"저장: {out}")
    else:
        print(text)


def cmd_parse(a):
    best, attempts = parse_document(a.file, a.strategy)
    for d in attempts:
        mark = "*" if d is best else " "
        print(f"{mark} {d.parser:18} {d.status:9} {d.elapsed_ms:>6} ms  blocks={len(d.blocks)}  "
              f"warnings={[w.code for w in d.warnings]}", file=sys.stderr)
    if a.overlay and best.file_type == "pdf":
        from .visualize import draw_overlay, render_pages

        out = Path(a.overlay)
        out.mkdir(parents=True, exist_ok=True)
        scale = 1.5
        for i, img in enumerate(render_pages(a.file, scale), start=1):
            path = out / f"{Path(a.file).stem}_p{i}_{best.parser}.png"
            draw_overlay(img, [b for b in best.blocks if b.page == i], scale).save(path)
            print(f"오버레이: {path}", file=sys.stderr)
    _dump(best.model_dump(), a.out)


def cmd_run(a):
    from .graph import build_graph, initial_state
    from .llm import LLMConfig

    cfg = LLMConfig()
    if a.model:
        cfg.model = a.model
    if a.class_effort:
        cfg.class_effort = a.class_effort
    if a.extract_effort:
        cfg.extract_effort = a.extract_effort
    graph = build_graph()
    final = None
    for chunk in graph.stream(initial_state(a.file, cfg), stream_mode="values"):
        final = chunk
    for step in final.get("model_runs", []):
        print(f"  {step['node']:28} {step['model']} effort={step['reasoning_effort']} "
              f"in={step['input_tokens']} out={step['output_tokens']} {step['latency_ms']} ms "
              f"${step['cost_usd']}" + (f" ERROR {step['error']}" if step["error"] else ""),
              file=sys.stderr)
    print(f"status={final['status']} next={final['next_node']}", file=sys.stderr)
    for r in final.get("review_reasons", []):
        print(f"  검토 사유: {r}", file=sys.stderr)
    _dump(final, a.out)


def main():
    ap = argparse.ArgumentParser(prog="extractor")
    sub = ap.add_subparsers(required=True)

    p = sub.add_parser("parse", help="01 문서 파싱")
    p.add_argument("file")
    p.add_argument("--strategy", default="auto", choices=["auto", *PDF_STRATEGIES, "excel"])
    p.add_argument("--out")
    p.add_argument("--overlay", help="bbox 오버레이 PNG 저장 폴더")
    p.set_defaults(func=cmd_parse)

    r = sub.add_parser("run", help="01~03 실행 (LangGraph)")
    r.add_argument("file")
    r.add_argument("--model")
    r.add_argument("--class-effort")
    r.add_argument("--extract-effort")
    r.add_argument("--out")
    r.set_defaults(func=cmd_run)

    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    a.func(a)


if __name__ == "__main__":
    main()
