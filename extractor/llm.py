"""LLM 호출 (설계안 12절): ChatOpenAI + Structured Outputs(JSON Schema, strict).

API 키는 환경변수 OPENAI_API_KEY(.env 가능)로만 읽는다.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from .schemas import ModelExecutionInfo

load_dotenv()

# 2026-10-26 기준 Standard 단가 (USD / 1M tokens): (입력, 출력). 캐싱 할인 미반영.
PRICING = {
    "gpt-6-luna": (0.10, 0.50),
    "gpt-6.1-sol": (2.00, 10.00),
}


@dataclass
class LLMConfig:
    model: str = os.getenv("LLM_MODEL", "gpt-6-luna")
    class_effort: str | None = os.getenv("CLASS_MAPPING_EFFORT", "low")
    extract_effort: str | None = os.getenv("EXTRACTION_EFFORT", "low")
    max_extraction_retries: int = int(os.getenv("MAX_EXTRACTION_RETRIES", "2"))
    timeout: float = 120.0


def _cost(model: str, inp: int | None, out: int | None) -> float | None:
    if model not in PRICING or inp is None or out is None:
        return None
    pi, po = PRICING[model]
    return round(inp / 1e6 * pi + out / 1e6 * po, 6)


def call_structured(
    *,
    node: str,
    schema: type[BaseModel],
    system: str,
    payload: dict,
    model: str,
    effort: str | None,
    timeout: float = 120.0,
) -> tuple[BaseModel | None, ModelExecutionInfo]:
    llm = ChatOpenAI(model=model, reasoning_effort=effort, timeout=timeout, max_retries=2)
    runnable = llm.with_structured_output(schema, method="json_schema", strict=True, include_raw=True)
    messages = [
        SystemMessage(system),
        HumanMessage(json.dumps(payload, ensure_ascii=False, indent=1)),
    ]
    t0 = time.perf_counter()
    try:
        out = runnable.invoke(messages)
    except Exception as e:
        return None, ModelExecutionInfo(
            node=node, model=model, reasoning_effort=effort,
            latency_ms=int((time.perf_counter() - t0) * 1000), error=f"{type(e).__name__}: {e}",
        )
    latency = int((time.perf_counter() - t0) * 1000)
    usage = getattr(out["raw"], "usage_metadata", None) or {}
    inp, outp = usage.get("input_tokens"), usage.get("output_tokens")
    info = ModelExecutionInfo(
        node=node, model=model, reasoning_effort=effort,
        input_tokens=inp, output_tokens=outp,
        reasoning_tokens=(usage.get("output_token_details") or {}).get("reasoning"),
        latency_ms=latency, cost_usd=_cost(model, inp, outp),
        error=str(out["parsing_error"]) if out.get("parsing_error") else None,
    )
    return out.get("parsed"), info
