"""LLM 호출 (설계안 12절): ChatOpenAI + Structured Outputs(JSON Schema, strict).

API 키는 환경변수 OPENAI_API_KEY(.env 가능)로만 읽는다.
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from .schemas import ModelExecutionInfo

load_dotenv()

PRICING_FILE = Path(__file__).resolve().parent.parent / "data" / "pricing.json"


def load_pricing() -> dict[str, dict]:
    return json.loads(PRICING_FILE.read_text(encoding="utf-8"))["models"]


@dataclass
class LLMConfig:
    model: str = os.getenv("LLM_MODEL", "gpt-6-luna")
    class_effort: str | None = os.getenv("CLASS_MAPPING_EFFORT", "low")
    extract_effort: str | None = os.getenv("EXTRACTION_EFFORT", "low")
    max_extraction_retries: int = int(os.getenv("MAX_EXTRACTION_RETRIES", "2"))
    timeout: float = 120.0


def cost_usd(model: str, inp: int | None, out: int | None, cached: int | None = None) -> float | None:
    price = load_pricing().get(model)
    if price is None or inp is None or out is None:
        return None
    cached = cached or 0
    cached_price = price.get("cached_input", price["input"])
    total = (inp - cached) / 1e6 * price["input"] + cached / 1e6 * cached_price + out / 1e6 * price["output"]
    return round(total, 6)


def usage_info(node: str, model: str, effort: str | None, usage: dict | None, latency_ms: int,
               error: str | None = None) -> ModelExecutionInfo:
    usage = usage or {}
    inp, out = usage.get("input_tokens"), usage.get("output_tokens")
    cached = (usage.get("input_token_details") or {}).get("cache_read")
    return ModelExecutionInfo(
        node=node, model=model, reasoning_effort=effort,
        input_tokens=inp, output_tokens=out,
        reasoning_tokens=(usage.get("output_token_details") or {}).get("reasoning"),
        latency_ms=latency_ms, cost_usd=cost_usd(model, inp, out, cached), error=error,
    )


class UsageRecorder(BaseCallbackHandler):
    """에이전트 내부의 모델 호출마다 토큰·지연을 기록한다."""

    def __init__(self, node: str, model: str, effort: str | None):
        self.node, self.model, self.effort = node, model, effort
        self.calls: list[ModelExecutionInfo] = []
        self._start: dict = {}
        self._lock = threading.Lock()

    def on_chat_model_start(self, serialized, messages, *, run_id, **kw):
        self._start[run_id] = time.perf_counter()

    def on_llm_end(self, response, *, run_id, **kw):
        t0 = self._start.pop(run_id, time.perf_counter())
        msg = response.generations[0][0].message if response.generations else None
        info = usage_info(self.node, self.model, self.effort, getattr(msg, "usage_metadata", None),
                          int((time.perf_counter() - t0) * 1000))
        with self._lock:
            self.calls.append(info)

    def on_llm_error(self, error, *, run_id, **kw):
        t0 = self._start.pop(run_id, time.perf_counter())
        with self._lock:
            self.calls.append(usage_info(self.node, self.model, self.effort, None,
                                         int((time.perf_counter() - t0) * 1000), f"{type(error).__name__}: {error}"))


def chat_model(model: str, effort: str | None, timeout: float = 120.0) -> ChatOpenAI:
    return ChatOpenAI(model=model, reasoning_effort=effort, timeout=timeout, max_retries=2)


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
    llm = chat_model(model, effort, timeout)
    runnable = llm.with_structured_output(schema, method="json_schema", strict=True, include_raw=True)
    messages = [
        SystemMessage(system),
        HumanMessage(json.dumps(payload, ensure_ascii=False, indent=1)),
    ]
    t0 = time.perf_counter()
    try:
        out = runnable.invoke(messages)
    except Exception as e:
        return None, usage_info(node, model, effort, None, int((time.perf_counter() - t0) * 1000),
                                f"{type(e).__name__}: {e}")
    info = usage_info(node, model, effort, getattr(out["raw"], "usage_metadata", None),
                      int((time.perf_counter() - t0) * 1000),
                      str(out["parsing_error"]) if out.get("parsing_error") else None)
    return out.get("parsed"), info
