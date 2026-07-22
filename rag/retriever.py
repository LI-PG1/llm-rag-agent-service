"""
混合检索模块（Dense + BM25 + RRF）
对应成品库① · 检索/召回
"""

import math
import json
from typing import List, Tuple, Optional
from openai import OpenAI

from config import (
    TOP_K_DENSE, DENSE_WEIGHT, BM25_WEIGHT,
    LLM_BASE_URL, LLM_API_KEY, QUERY_REWRITE_MODEL, QUERY_REWRITE_ENABLED,
    LLM_TIMEOUT,
)
from document_processor import DocumentChunk

class SimpleBM25:
    """轻量 BM25 实现，用于混合检索"""

    def __init__(self, corpus: List[str]):
        self.corpus = corpus
        self.k1 = 1.5
        self.b = 0.75
        self._build_index()

    def _build_index(self):
        self.doc_freq = {}
        self.doc_lengths = []
        self.avgdl = 0

        for doc in self.corpus:
            terms = doc.lower().split()
            self.doc_lengths.append(len(terms))
            seen = set()
            for term in set(terms):
                self.doc_freq[term] = self.doc_freq.get(term, 0) + 1

        self.avgdl = sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 1

    def score(self, query: str, doc_idx: int) -> float:
        query_terms = query.lower().split()
        doc = self.corpus[doc_idx].lower().split()
        dl = len(doc)
        score = 0.0
        n = len(self.corpus)

        for term in query_terms:
            tf = doc.count(term)
            if tf == 0:
                continue
            df = self.doc_freq.get(term, 0)
            idf = math.log((n - df + 0.5) / (df + 0.5) + 1)
            score += idf * (tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl))

        return score

    def search(self, query: str, top_k: int = 50) -> List[Tuple[float, int]]:
        scores = [(self.score(query, i), i) for i in range(len(self.corpus))]
        scores.sort(key=lambda x: x[0], reverse=True)
        return scores[:top_k]

def rrf_fusion(
    dense_results: List[Tuple[float, DocumentChunk]],
    bm25_results: List[Tuple[float, int]],
    bm25_chunks: List[DocumentChunk],
    k: int = 60,
    top_k: int = 50,
) -> List[Tuple[float, DocumentChunk]]:
    """
    Reciprocal Rank Fusion 融合排序
    dense_results: [(score, chunk), ...]
    bm25_results: [(score, idx), ...]
    """
    rrf_scores = {}

    # Dense 排名
    for rank, (_, chunk) in enumerate(dense_results):
        chunk_id = id(chunk)
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + DENSE_WEIGHT * (1 / (k + rank + 1))
        rrf_scores[f"_chunk_{chunk_id}"] = chunk

    # BM25 排名
    for rank, (_, idx) in enumerate(bm25_results):
        if idx < len(bm25_chunks):
            chunk = bm25_chunks[idx]
            chunk_id = id(chunk)
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + BM25_WEIGHT * (1 / (k + rank + 1))
            rrf_scores[f"_chunk_{chunk_id}"] = chunk

    # 按 RRF 分数排序
    scored_chunks = []
    seen = set()
    for key, score in sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True):
        if isinstance(key, int) and key not in seen:  # chunk_id
            seen.add(key)
            chunk = rrf_scores.get(f"_chunk_{key}")
            if chunk:
                scored_chunks.append((score, chunk))

    return scored_chunks[:top_k]

class QueryRewriter:
    """查询改写：长尾低频 query 消歧（对应 Qwen3.5-122B-A10B）"""

    def __init__(self, model: str = QUERY_REWRITE_MODEL):
        self.client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY, timeout=LLM_TIMEOUT)
        self.model = model

    def rewrite(self, query: str) -> List[str]:
        """将原 query 改写为多个子查询"""
        if not QUERY_REWRITE_ENABLED:
            return [query]

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{
                    "role": "system",
                    "content": "将用户问题改写为多个角度的搜索查询，每行一个，直接输出无需解释。"
                }, {
                    "role": "user",
                    "content": query,
                }],
                temperature=0.3,
                max_tokens=256,
            )
            content = response.choices[0].message.content or query
            queries = [q.strip() for q in content.split("\n") if q.strip()]
            return queries[:5]  # 最多 5 个改写
        except Exception as e:
            print(f"  [查询改写] 失败: {e}")
            return [query]
