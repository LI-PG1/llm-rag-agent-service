"""
Critic 校验模块（DeepSeek-R1-Distill-Qwen-14B）
对应成品库② · Critic 校验
"""

from openai import OpenAI
import json

from config import LLM_BASE_URL, LLM_API_KEY, CRITIC_MODEL


class Critic:
    """
    Reflexion 式 Critic 校验器。
    仅在两个节点触发：工具异常信号 + 最终答案归总。
    """

    def __init__(self):
        self.client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)
        self.model = CRITIC_MODEL

    def check_observation(self, tool_name: str, observation: str) -> dict:
        """
        检查工具输出是否异常。
        触发点：工具返回 error、空结果、零相关等异常信号。
        """
        abnormal_signals = [
            "错误", "超时", "为空", "不存在", "未找到",
            "失败", "异常", "null", "None", "零相关",
        ]

        is_abnormal = any(signal in observation for signal in abnormal_signals)

        return {
            "tool": tool_name,
            "abnormal": is_abnormal,
            "needs_retry": is_abnormal,
            "observation_preview": observation[:100],
        }

    def verify_final_answer(self, query: str, answer: str, tool_results: list) -> dict:
        """
        最终答案校验：用 Critic 模型检查答案与工具结果是否一致。
        触发点：Orchestrator 给出最终 answer 之后。
        """
        if not answer:
            return {"valid": False, "reason": "空答案", "needs_regeneration": True}

        prompt = (
            f"原始问题: {query}\n\n"
            f"工具返回信息:\n{chr(10).join(tool_results[:5])}\n\n"
            f"候选答案: {answer}\n\n"
            "请检查：1. 答案是否使用了工具返回的数据 2. 答案是否有幻觉 3. 工具结果和答案是否一致\n"
            "输出 JSON: {\"valid\": bool, \"reason\": \"...\", \"needs_regeneration\": bool}"
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.1,
            )
            result = json.loads(response.choices[0].message.content)
        except Exception as e:
            result = {"valid": True, "reason": f"Critic 异常: {e}", "needs_regeneration": False}

        return result

    def needs_intervention(self, observation: str) -> bool:
        """快速判断是否需要 Critic 介入"""
        abnormal = self.check_observation("", observation)
        return abnormal["abnormal"]
