"""
多模型生成与 Corrective RAG 模块
对应成品库① · 生成与 Corrective RAG
"""

import json
import re
from typing import List, Tuple, Optional
from openai import OpenAI

from config import (
    LLM_BASE_URL, LLM_API_KEY,
    SIMPLE_MODEL, COMPLEX_MODEL,
    SUFFICIENT_CONTEXT_MIN_CHUNKS, MAX_RETRY,
)
from document_processor import DocumentChunk

class Generator:
    """多模型路由生成 + Citation 溯源 + Corrective RAG"""

    def __init__(self):
        self.client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)
        self.simple_model = SIMPLE_MODEL
        self.complex_model = COMPLEX_MODEL

    def is_complex_query(self, query: str) -> bool:
        """
        判断 query 复杂度，决定走哪个模型。
        实际部署中用 LLM 做分类，这里用简单规则。
        """
        complex_keywords = ["对比", "分析", "为什么", "如何", "流程", "原因", "方案", "计算", "比较"]
        return any(kw in query for kw in complex_keywords)

    def sufficient_context(self, chunks: List[DocumentChunk]) -> bool:
        """判断上下文是否足够生成"""
        total_text = sum(len(c.text) for c in chunks)
        return len(chunks) >= SUFFICIENT_CONTEXT_MIN_CHUNKS and total_text > 50

    def generate(
        self,
        query: str,
        chunks: List[DocumentChunk],
        history: Optional[List[dict]] = None,
    ) -> Tuple[str, List[dict]]:
        """
        多模型路由 → Sufficient Context 校验 → 生成 → Citation 验证 → 重试

        返回: (final_answer, citations)
        citations: [{"text": str, "source": str, "verified": bool}, ...]
        """
        if not self.sufficient_context(chunks):
            return "上下文不足，无法回答。请补充更多相关信息后重试。", []

        model = self.complex_model if self.is_complex_query(query) else self.simple_model
        messages = self._build_messages(query, chunks, history)

        for attempt in range(1, MAX_RETRY + 2):  # 首次 + max_retry 次重试
            print(f"  [生成] model={model}, attempt={attempt}/{MAX_RETRY + 1}")

            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.3,
                max_tokens=4096,
            )
            answer = response.choices[0].message.content or ""

            # Citation 验证
            citations = self._verify_citations(answer, chunks)
            verified = all(c["verified"] for c in citations)

            if verified or attempt > MAX_RETRY:
                return answer, citations

            # 未通过：二次检索（此处用补充检索信号）
            print(f"  [Citation 验证] 未通过，触发补充检索 (attempt {attempt}/{MAX_RETRY})")
            messages.append({"role": "user", "content": "请重新回答，注意严格引用提供文档中的信息。对没有来源支撑的陈述，请明确标注。"})

        return answer, citations

    def _build_messages(
        self, query: str, chunks: List[DocumentChunk], history: Optional[List[dict]] = None,
    ) -> List[dict]:
        """构建 prompt 消息列表"""
        context = "\n\n---\n\n".join([
            f"[来源: {c.metadata.get('source', 'unknown')}]\n{c.text[:2000]}"
            for c in chunks[:5]
        ])

        system_prompt = (
            "你是一个严谨的客服文档问答系统。请基于以下文档内容回答用户问题。\n\n"
            "规则：\n"
            "1. 只能根据提供的文档回答，不能使用自己的知识。\n"
            "2. 每个回答后必须附上引用来源，格式为【来源：文档名】。\n"
            "3. 如果文档中没有足够信息，请明确说'根据现有文档无法回答'。\n"
            "4. 保持回答简洁、准确。"
        )

        messages = [{"role": "system", "content": system_prompt}]

        if history:
            for h in history[-6:]:
                messages.append(h)

        messages.append({"role": "user", "content": f"文档内容：\n{context}\n\n问题：{query}"})
        return messages

    def _verify_citations(self, answer: str, chunks: List[DocumentChunk]) -> List[dict]:
        """
        LLM-as-Judge：验证回答中每个事实是否可归因到文档
        简单实现：检查是否包含引用标注
        """
        citations = []
        # 按句子分割
        sentences = re.split(r'(?<=[。！？])', answer)

        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue

            # 检查是否包含来源标注
            has_source = bool(re.search(r'【来源|\[来源\]|\[\\d+\]', sent))
            # 检查是否可在任何 chunk 中找到
            found_in_docs = any(
                any(phrase in chunk.text for phrase in sent.split("，")[:1])
                for chunk in chunks
            )
            citations.append({
                "text": sent[:100],
                "source": str([c.metadata.get("source") for c in chunks[:2]]),
                "verified": has_source or found_in_docs,
            })

        return citations
