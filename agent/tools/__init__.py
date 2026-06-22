"""
工具注册表（MCP 设计理念 · JSON Schema 定义）
对应成品库② · 工具注册

设计决策：
  选 MCP 风格而非 LangChain Tool 抽象，因为 JSON Schema 定义更透明、
  不依赖框架、可直接映射到 OpenAI function calling 格式。
  
踩坑：
  初版 try-except 吞掉了所有异常，导致 orchestrator 的重试永远无法触发。
  修复：Tool.execute 将异常向上传播，由 execute_tool 统一处理。
"""

import logging
from typing import Dict, Any, Callable, Optional
from tools.rag_tool import RagRetrievalTool
from tools.sql_tool import SparePartsQueryTool
from tools.api_tool import RepairOrderTool

logger = logging.getLogger(__name__)


class Tool:
    """统一工具封装"""

    def __init__(self, name: str, description: str, parameters: dict, handler: Callable):
        self.name = name
        self.description = description
        self.parameters = parameters
        self.handler = handler

    def to_openai_tool(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            }
        }

    def execute(self, **kwargs) -> str:
        """执行工具，异常向上传播（供上游重试）"""
        result = self.handler(**kwargs)
        return str(result)


# ── 初始化工具实例 ──
_rag = RagRetrievalTool()
_sql = SparePartsQueryTool()
_api = RepairOrderTool()


def _rag_search(query: str, top_k: int = 5) -> str:
    return _rag.search(query, top_k=top_k)


def _sql_query(sql: str) -> str:
    return _sql.query(sql)


def _submit_repair(equipment_id: str, fault_desc: str, priority: str = "normal") -> str:
    return _api.submit(equipment_id, fault_desc, priority)


def _check_repair_status(order_id: str) -> str:
    return _api.check_status(order_id)


# ── 注册表 ──

TOOL_REGISTRY: Dict[str, Tool] = {
    "manual_retrieval": Tool(
        name="manual_retrieval",
        description="检索设备维修手册、产品说明、故障处理指南等文档内容。输入问题关键词即可查找相关章节。",
        parameters={...省略(与之前相同)...},
        handler=_rag_search,
    ),
    # ... 其余工具定义与之前相同 ...
}


def get_tools_for_llm() -> list:
    return [tool.to_openai_tool() for tool in TOOL_REGISTRY.values()]


def execute_tool(name: str, arguments: dict, timeout: int = 30) -> str:
    """执行工具（含超时），异常向上传播供 orchestrator 重试"""
    import concurrent.futures

    if name not in TOOL_REGISTRY:
        return f"未知工具: {name}"

    tool = TOOL_REGISTRY[name]
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(tool.execute, **arguments)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            msg = f"[超时] 工具 {name} 执行超过 {timeout}s"
            logger.warning(msg)
            return msg
        except Exception as e:
            logger.warning(f"[工具错误] {name}: {e}")
            raise  # 向上传播，供 orchestrator 重试
