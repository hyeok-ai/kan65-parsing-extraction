"""자체 문서 도구 MCP 서버 (stdio).

  python -m extractor.mcp_server <문서 경로>

인프로세스 에이전트와 같은 도구를 MCP로 제공해, 'Python API 대신 MCP로 호출'했을 때의
차이(지연·토큰)만 따로 비교할 수 있게 한다. (설계안 14절)
stdout은 MCP 프로토콜이 쓰므로 여기서 print하지 않는다.
"""

from __future__ import annotations

import sys

from mcp.server.mcpserver import MCPServer

from .agent_tools import STRATEGY_HELP, DocumentWorkspace

ws = DocumentWorkspace(sys.argv[1])
server = MCPServer("product-extract-tools")
_STRATEGIES = "; ".join(f"{s}={STRATEGY_HELP[s]}" for s in ws.strategies)


@server.tool(description="문서를 지정한 전략으로 파싱하고 상태·품질 경고·블록 목록을 돌려준다. 전략: " + _STRATEGIES)
def parse_document(strategy: str) -> str:
    return ws.parse_document(strategy)


@server.tool(description="해당 전략 파싱 결과에서 블록 전문을 읽는다. (최대 20개)")
def read_blocks(strategy: str, block_ids: list[str]) -> str:
    return ws.read_blocks(strategy, block_ids)


@server.tool(description="해당 전략 파싱 결과에서 query를 포함한 블록과 줄을 찾는다. 공백·구분자 차이는 무시.")
def search_document(strategy: str, query: str) -> str:
    return ws.search_document(strategy, query)


@server.tool(description="온톨로지 클래스의 속성 정의(id, label, required, target_unit, synonyms)를 돌려준다.")
def get_class_properties(class_id: str) -> str:
    return ws.get_class_properties(class_id)


if __name__ == "__main__":
    server.run("stdio")
