"""
手册检索工具（RAG 内嵌）
对应成品库② · RAG 检索工具（复用项目①索引底座）
"""

import json
from openai import OpenAI
from typing import Optional

from config import LLM_BASE_URL, LLM_API_KEY, RAG_TOP_K

class RagRetrievalTool:
    """设备维修手册检索"""

    def __init__(self):
        self.client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)

    def search(self, query: str, top_k: int = RAG_TOP_K) -> str:
        """
        从知识库检索相关文档。
        实际部署中对接 Milvus 向量库，此处通过LLM模拟。
        """
        # 模拟检索结果（实际部署替换为向量检索）
        mock_results = {
            "电机过热": [
                ("电机过热故障排查手册", "Step 1: 检查电机负载是否超过额定值\nStep 2: 测量绕组电阻\nStep 3: 检查冷却风扇"),
                ("电机维护指南", "电机过热的常见原因：过载、缺相、散热不良、轴承磨损"),
                ("温度传感器校准规程", "PT100 传感器每半年校准一次，偏差超过 ±2°C 需更换"),
            ],
            "报修": [
                ("设备报修流程", "报修步骤：1. 填写故障描述 2. 提交工单 3. 等待派工 4. 维修完成确认"),
            ],
        }

        # 简单匹配
        results = []
        for keyword, docs in mock_results.items():
            if keyword in query:
                results.extend(docs)

        if not results:
            results = [("通用技术文档", "未找到精确匹配，建议联系技术支持")]

        formatted = []
        for title, content in results[:top_k]:
            formatted.append(f"【{title}】\n{content[:500]}")

        return "\n\n".join(formatted)
