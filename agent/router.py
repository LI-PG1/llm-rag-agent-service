"""
入口分流模块
对应成品库② · 入口分流：简单→RAG 直回，复杂→Agent
"""

from config import SIMPLE_KEYWORDS

class EntryRouter:
    """入口分流：判断 query 走 RAG 直回还是 Agent 循环"""

    def is_simple_query(self, query: str) -> bool:
        """
        判断是否为简单咨询。
        默认基于关键词，实际部署时替换为 LLM 分类。
        """
        q = query.lower()
        # 简单关键词匹配
        for kw in SIMPLE_KEYWORDS:
            if kw in q:
                return True

        # 短 query（<8 字）且非复合句 → 简单
        if len(q) < 8 and not any(c in q for c in ["和", "与", "并且", "同时"]):
            return True

        return False

    def route(self, query: str) -> str:
        """路由决策：'rag' 或 'agent'"""
        if self.is_simple_query(query):
            return "rag"
        return "agent"

if __name__ == "__main__":
    router = EntryRouter()
    test_queries = [
        ("查手册：电机过热故障代码", "rag"),
        ("电机编号 M-2024-0892 报修流程怎么走", "agent"),
        ("先查库存再发起报修", "agent"),
        ("什么是过载保护", "rag"),
    ]
    for q, expected in test_queries:
        result = router.route(q)
        status = "✅" if result == expected else "❌"
        print(f"{status} {q} → {result} (期望: {expected})")
