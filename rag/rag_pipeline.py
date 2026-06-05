"""
端到端 RAG 管线编排
"""

import numpy as np
from openai import OpenAI
from typing import List, Optional

from config import (
    EMBEDDING_MODEL, LLM_BASE_URL, LLM_API_KEY,
    TOP_K_DENSE, HNSW_EF_SEARCH,
)
from document_processor import DocumentChunk, load_documents, split_documents, build_sample_docs
from vector_store import VectorStore
from retriever import SimpleBM25, rrf_fusion, QueryRewriter
from reranker import Reranker
from generator import Generator


class RAGPipeline:
    """端到端 RAG 管线"""

    def __init__(self, store_type: str = "chroma"):
        self.store_type = store_type
        self.vector_store: Optional[VectorStore] = None
        self.bm25: Optional[SimpleBM25] = None
        self.bm25_chunks: List[DocumentChunk] = []
        self.reranker = Reranker()
        self.generator = Generator()
        self.query_rewriter = QueryRewriter()
        self.client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)
        self._initialized = False
        self._all_texts: List[str] = []

    def initialize(self, with_sample_docs: bool = True):
        """初始化管线：加载 → 切分 → 向量化 → 入库"""
        print("=" * 50)
        print("RAG Pipeline 初始化")
        print("=" * 50)

        if with_sample_docs:
            build_sample_docs()

        print("\n[1/4] 加载文档...")
        docs = load_documents()
        if not docs:
            print("  无文档，跳过初始化")
            return

        print(f"\n[2/4] 切分文档 (chunk={512}, overlap={64})...")
        chunks = split_documents(docs)

        print(f"\n[3/4] 向量化 + 入库...")
        self.vector_store = VectorStore(self.store_type)
        embeddings = self._get_embeddings([c.text for c in chunks])
        self.vector_store.add_chunks(chunks, embeddings)

        print(f"\n[4/4] 构建 BM25 索引...")
        self._all_texts = [c.text for c in chunks]
        self.bm25_chunks = chunks
        self.bm25 = SimpleBM25(self._all_texts)

        self._initialized = True
        print(f"\n初始化完成。向量数: {self.vector_store.count()}")
        print("=" * 50)

    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """调用 Embedding 模型获取向量"""
        try:
            response = self.client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=texts,
            )
            return [r.embedding for r in response.data]
        except Exception as e:
            print(f"  [Embedding API 调用失败: {e}]")
            print("  使用随机向量占位（测试用）")
            np.random.seed(42)
            return [np.random.randn(1536).tolist() for _ in texts]

    def query(self, user_input: str, history: Optional[List[dict]] = None) -> dict:
        """
        端到端查询
        """
        if not self._initialized:
            return {"answer": "管线未初始化，请先调用 initialize()", "citations": []}

        print(f"\n[查询] {user_input}")
        result = {"query": user_input, "steps": {}}

        # ── 查询改写（长尾低频） ──
        queries = self.query_rewriter.rewrite(user_input)
        result["steps"]["rewritten_queries"] = queries
        print(f"  [改写] {queries}")

        # ── 向量检索 + BM25 + RRF ──
        all_dense_chunks = []
        all_scores_map = {}

        for q in queries:
            q_vec = self._get_embeddings([q])[0]
            dense_chunks = self.vector_store.search(q_vec, top_k=TOP_K_DENSE)
            all_dense_chunks.extend(dense_chunks)

            # BM25
            bm25_scores = self.bm25.search(q, top_k=TOP_K_DENSE)
            for score, idx in bm25_scores:
                if idx < len(self.bm25_chunks):
                    all_scores_map[idx] = max(all_scores_map.get(idx, 0), score)

        bm25_scores = [(s, i) for i, s in all_scores_map.items()]
        fused = rrf_fusion(all_dense_chunks, bm25_scores, self.bm25_chunks)
        result["steps"]["retrieval_count"] = len(fused)

        # ── Reranker ──
        reranked = self.reranker.rerank(user_input, fused, top_k=5)
        result["steps"]["reranked_count"] = len(reranked)

        # ── Sufficient Context 校验 ──
        chunks = [c for _, c in reranked]
        if not self.generator.sufficient_context(chunks):
            result["answer"] = "根据现有文档无法回答此问题。"
            result["insufficient_context"] = True
            return result

        # ── 生成 + Citation ──
        answer, citations = self.generator.generate(user_input, chunks, history)
        result["answer"] = answer
        result["citations"] = citations
        result["model"] = self.generator.simple_model if not self.generator.is_complex_query(user_input) else self.generator.complex_model

        return result

    def get_retrieval_only(self, user_input: str) -> List[DocumentChunk]:
        """仅检索（不生成），用于 /retrieve 接口"""
        if not self._initialized:
            return []
        q_vec = self._get_embeddings([user_input])[0]
        dense = self.vector_store.search(q_vec, top_k=TOP_K_DENSE)
        bm25 = self.bm25.search(user_input, top_k=TOP_K_DENSE)
        bm25_scored = [(s, i) for s, i in bm25 if i < len(self.bm25_chunks)]
        fused = rrf_fusion(dense, bm25_scored, self.bm25_chunks)
        reranked = self.reranker.rerank(user_input, fused, top_k=TOP_K_RERANK)
        return [c for _, c in reranked]


if __name__ == "__main__":
    # 测试运行
    pipeline = RAGPipeline(store_type="chroma")
    pipeline.initialize(with_sample_docs=True)

    test_queries = [
        "5G 畅享套餐 199 元档包含多少流量？",
        "宽带无法连接怎么处理？",
        "政企专线 100M 多少钱一个月？",
    ]

    for q in test_queries:
        result = pipeline.query(q)
        print(f"\nQ: {q}")
        print(f"A: {result['answer'][:200]}")
        print(f"  引文: {sum(1 for c in result.get('citations', []) if c.get('verified'))}/{len(result.get('citations', []))} 已验证")
        print("-" * 40)
