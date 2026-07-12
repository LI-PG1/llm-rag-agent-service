"""
FastAPI 推理服务（含降级策略）
对应成品库① · 服务化

降级策略：
  1. Embedding 服务不可用 → 纯 BM25 关键词检索（质量下降但服务不中断）
  2. LLM 不可用 → 返回仅检索结果（告知用户生成不可用）
  3. Reranker 不可用 → 跳过重排序，直接用 RRF 融合结果
"""

import json
import os
import logging
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

from config import (
    SERVICE_HOST, SERVICE_PORT, REDIS_URL, CACHE_TTL,
    SESSION_TTL, LOG_LEVEL, LOG_FORMAT,
)
from rag_pipeline import RAGPipeline

# ── 日志 ──
logging.basicConfig(level=getattr(logging, LOG_LEVEL), format=LOG_FORMAT)
logger = logging.getLogger(__name__)

pipeline: Optional[RAGPipeline] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    logger.info("初始化 RAG Pipeline...")
    pipeline = RAGPipeline(store_type="chroma")
    pipeline.initialize(with_sample_docs=True)
    logger.info(f"启动完成, 向量数: {pipeline.vector_store.count() if pipeline.vector_store else 0}")
    yield
    logger.info("RAG Pipeline 已关闭")


app = FastAPI(title="RAG 知识问答服务", version="1.0.0", lifespan=lifespan)


class QueryRequest(BaseModel):
    query: str
    history: Optional[list] = None


class RetrieveRequest(BaseModel):
    query: str
    top_k: int = 5


class RetrieveResult(BaseModel):
    text: str
    source: str
    score: float


class QueryResponse(BaseModel):
    query: str
    answer: str
    citations: list = []
    model: str = ""
    steps: dict = {}
    degraded: bool = False  # 是否处于降级模式


@app.get("/health")
def health():
    return {
        "status": "ok",
        "vector_count": pipeline.vector_store.count() if pipeline and pipeline.vector_store else 0,
        "degraded": False,
    }


@app.post("/retrieve", response_model=list[RetrieveResult])
def retrieve(req: RetrieveRequest):
    """仅检索，不生成（降级：#1 Embedding 不可用时降级到 BM25）"""
    if not pipeline or not pipeline._initialized:
        raise HTTPException(status_code=503, detail="管线未初始化")

    try:
        chunks = pipeline.get_retrieval_only(req.query)
        return [
            RetrieveResult(text=c.text[:500], source=c.metadata.get("source", ""), score=1.0)
            for c in chunks[:req.top_k]
        ]
    except Exception as e:
        logger.warning(f"检索降级: {e}")
        # 降级：纯 BM25（不依赖 Embedding）
        bm25_results = pipeline.bm25.search(req.query, top_k=req.top_k) if pipeline.bm25 else []
        return [
            RetrieveResult(text=r[:500], source="bm25_fallback", score=0.5)
            for r in bm25_results[:req.top_k]
        ]


@app.post("/chat", response_model=QueryResponse)
def chat(req: QueryRequest):
    """端到端问答（降级：#2 LLM 不可用时返回纯检索结果）"""
    if not pipeline or not pipeline._initialized:
        raise HTTPException(status_code=503, detail="管线未初始化")

    try:
        result = pipeline.query(req.query, history=req.history)
        return QueryResponse(**result)
    except Exception as e:
        logger.warning(f"生成降级: {e}")
        # 降级：仅返回检索结果
        chunks = pipeline.get_retrieval_only(req.query)
        context = "\n".join([c.text[:300] for c in chunks[:3]])
        return QueryResponse(
            query=req.query,
            answer=f"[生成服务暂不可用] 以下为相关文档片段：\n\n{context}",
            citations=[],
            model="degraded",
            degraded=True,
        )


if __name__ == "__main__":
    logger.info(f"启动服务: {SERVICE_HOST}:{SERVICE_PORT}")
    uvicorn.run(app, host=SERVICE_HOST, port=SERVICE_PORT)
