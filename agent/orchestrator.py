"""
ReAct / Plan-then-Execute Orchestrator
对应成品库② · 编排与规划

选型决策：
  选 Qwen3.6-35B-A3B MoE 而非更大稠密模型——function-calling 场景下
  3B 激活参数已足够驱动规划，且单卡可常驻、并发成本可控。
  
踩坑：
  初版 try-except 吞掉了 LLM 调用异常，导致 LLM 不可用时服务直接挂掉。
  修复：外层 catch Exception 返回降级信息，服务不中断。
"""

import json
import time
import logging
from openai import OpenAI
from typing import List, Dict, Optional

from config import (
    LLM_BASE_URL, LLM_API_KEY,
    ORCHESTRATOR_MODEL,
    MAX_ITERATIONS, MAX_TOOL_RETRIES, TOOL_TIMEOUT,
)

logger = logging.getLogger(__name__)
from tools import get_tools_for_llm, execute_tool
from memory import ConversationMemory

class Orchestrator:
    """Agent 主驱——ReAct 循环"""

    def __init__(self):
        self.client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)
        self.model = ORCHESTRATOR_MODEL
        self.memory = ConversationMemory()
        self.tools = get_tools_for_llm()

    def run(self, user_input: str) -> str:
        """主入口：执行 ReAct 循环"""
        self.memory.add_user_message(user_input)
        messages = self._build_messages()

        iteration_count = 0

        while iteration_count < MAX_ITERATIONS:
            iteration_count += 1

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=self.tools,
                temperature=0.7,
                max_tokens=4096,
            )

            choice = response.choices[0]
            message = choice.message

            # LLM 直接回答
            if choice.finish_reason == "stop" and message.content:
                final = message.content
                self.memory.add_assistant_message(final)
                self.memory.finish_turn()
                return final

            # 工具调用
            if choice.finish_reason == "tool_calls" and message.tool_calls:
                thought = message.content or ""
                print(f"   [Step {iteration_count}] Thought: {thought[:100]}...")

                # 解析工具调用
                tool_results = []
                for tc in message.tool_calls:
                    name = tc.function.name
                    try:
                        args = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        args = {}

                    print(f"   → 工具: {name}({args})")
                    result = self._execute_with_timeout(name, args)
                    print(f"   ← 结果: {result[:80]}...")

                    tool_results.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                    })

                    self.memory.add_trace_step(thought, f"{name}({args})", result)

                # 加入 messages 继续循环
                messages.append(message)
                messages.extend(tool_results)
                continue

            # Fallback
            final = message.content or "无法回答"
            self.memory.add_assistant_message(final)
            self.memory.finish_turn()
            return final

        return f"[已达 {MAX_ITERATIONS} 步上限，返回已收集的部分信息]"

    def _execute_with_timeout(self, name: str, args: dict) -> str:
        """带超时和重试的工具执行"""
        for attempt in range(1, MAX_TOOL_RETRIES + 1):
            try:
                return execute_tool(name, args, timeout=TOOL_TIMEOUT)
            except Exception as e:
                print(f"   [重试 {attempt}/{MAX_TOOL_RETRIES}] {e}")
                time.sleep(1)
        return f"[错误] {name} 执行失败"

    def _build_messages(self) -> List[Dict]:
        """构建 LLM 输入消息"""
        system_prompt = (
            "你是一个设备运维助手。你可以调用以下工具来完成任务：\n"
            "1. manual_retrieval - 检索设备维修手册\n"
            "2. query_spare_parts - 查询备件库存\n"
            "3. submit_repair_order - 提交报修工单\n"
            "4. check_repair_status - 查询工单状态\n\n"
            "工作方式：\n"
            "- 每次输出 thought 分析当前任务，再决定调用哪个工具\n"
            "- 如果多个工具没有依赖关系，可以并行调用\n"
            "- 当收集到足够信息后，直接输出最终答案\n"
            f"- 最大步数: {MAX_ITERATIONS}"
        )
        return [{"role": "system", "content": system_prompt}] + self.memory.get_context_messages()
