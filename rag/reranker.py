"""
重排序模块（Cross-encoder Reranker）
对应成品库① · bge-reranker-v2-m3
"""

from typing import List, Tuple
from config import TOP_K_RERANK, RERANK_SCORE_THRESHOLD
from document_processor import DocumentChunk

class Reranker:
    """Cross-encoder 重排序"""

    def __init__(self, model_name: str = "bge-reranker-v2-m3"):
        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self):
        """延迟加载模型（仅在首次使用时加载）"""
        try:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(self.model_name)
            print(f"  [Reranker] 已加载: {self.model_name}")
        except ImportError:
            print("  [Reranker] sentence-transformers 未安装，跳过")
        except Exception as e:
            print(f"  [Reranker] 加载失败: {e}，将使用检索原始分数")

    def rerank(
        self,
        query: str,
        candidates: List[Tuple[float, DocumentChunk]],
        top_k: int = TOP_K_RERANK,
        threshold: float = RERANK_SCORE_THRESHOLD,
    ) -> List[Tuple[float, DocumentChunk]]:
        """
        Cross-encoder 重排序：对候选块逐对打分
        返回：[(rerank_score, chunk), ...]
        """
        if self.model is None or len(candidates) == 0:
            return candidates[:top_k]

        # 准备 (query, doc) 对
        pairs = [(query, chunk.text) for _, chunk in candidates]
        scores = self.model.predict(pairs)

        # 按 rerank 分数排序
        scored = [(float(score), candidates[i][1]) for i, score in enumerate(scores)]
        scored.sort(key=lambda x: x[0], reverse=True)

        # 阈值过滤
        filtered = [(s, c) for s, c in scored if s >= threshold]

        print(f"  [Reranker] {len(candidates)}→{len(filtered)} (阈值={threshold}, top_k={top_k})")
        return filtered[:top_k]

    def score(self, query: str, doc_text: str) -> float:
        """单对打分"""
        if self.model is None:
            return 0.0
        return float(self.model.predict([(query, doc_text)])[0])
