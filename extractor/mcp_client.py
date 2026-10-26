"""MCP 서버(stdio)의 도구를 LangChain 도구로 감싼다.

langchain-mcp-adapters 0.3.x는 mcp 1.x 전용이고 docling-mcp 3.x는 mcp 2.x를 요구해서
mcp 2.x 클라이언트로 직접 감쌌다. 호출마다 지연·출력 길이를 trace에 남긴다.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

TOOL_OUTPUT_LIMIT = 8000
MCP_LOG = Path(__file__).resolve().parent.parent / ".cache" / "mcp_server.log"


def _venv_script(name: str) -> str:
    exe = Path(sys.executable).parent / (name + (".exe" if os.name == "nt" else ""))
    return str(exe) if exe.exists() else (shutil.which(name) or name)


def docling_mcp_params() -> StdioServerParameters:
    return StdioServerParameters(
        command=_venv_script("docling-mcp-server"),
        args=["--transport", "stdio", "conversion", "manipulation"],
        env={**os.environ, "DOCLING_MCP_CONVERSION_MODE": "local"},
    )


def product_mcp_params(path: str | Path) -> StdioServerParameters:
    root = Path(__file__).resolve().parent.parent
    return StdioServerParameters(command=sys.executable, args=["-m", "extractor.mcp_server", str(Path(path).resolve())],
                                 env={**os.environ, "PYTHONIOENCODING": "utf-8"}, cwd=str(root))


def _result_text(res) -> str:
    if res.structured_content:
        text = json.dumps(res.structured_content, ensure_ascii=False)
    else:
        text = "\n".join(getattr(c, "text", str(c)) for c in res.content)
    return ("ERROR: " if res.is_error else "") + text


@asynccontextmanager
async def mcp_tools(params: StdioServerParameters, trace: list[dict]):
    from langchain_core.tools import StructuredTool

    t0 = time.perf_counter()
    MCP_LOG.parent.mkdir(parents=True, exist_ok=True)
    errlog = MCP_LOG.open("a", encoding="utf-8")  # 서버 로그가 콘솔을 덮지 않게 파일로 보낸다.
    async with stdio_client(params, errlog=errlog) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            listed = (await session.list_tools()).tools
            startup_ms = int((time.perf_counter() - t0) * 1000)

            def make(tool):
                async def call(**kwargs):
                    t = time.perf_counter()
                    try:
                        text = _result_text(await session.call_tool(tool.name, kwargs))
                    except Exception as e:
                        text = f"ERROR: {type(e).__name__}: {e}"
                    trace.append({"tool": tool.name, "args": kwargs, "output_chars": len(text),
                                  "ms": int((time.perf_counter() - t) * 1000), "output_preview": text[:300],
                                  "output_full": text[:20000], "via": "mcp"})
                    return text[:TOOL_OUTPUT_LIMIT]

                return StructuredTool(name=tool.name, description=tool.description or tool.name,
                                      args_schema=tool.input_schema, coroutine=call)

            yield [make(t) for t in listed], startup_ms
