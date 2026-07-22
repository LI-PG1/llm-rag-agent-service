"""
向量数据库操作模块
支持 Milvus（生产）和 Chroma（开发/测试）两种后端
对应成品库① · 向量化入库
"""

import os
import pickle
from typing import List, Optional

from config import (
    VECTOR_STORE_TYPE, EMBEDDING_DIM,
    MILVUS_HOST, MILVUS_PORT, MILVUS_COLLECTION,
    CHROMA_PATH, HNSW_M, HNSW_EF_CONSTRUCTION, HNSW_EF_SEARCH,
)
from document_processor import DocumentChunk

class VectorStore:
    """统一向量存储接口"""

    def __init__(self, store_type: str = VECTOR_STORE_TYPE):
        self.store_type = store_type
        self._init_store()

    def _init_store(self):
        if self.store_type == "milvus":
            self._init_milvus()
        elif self.store_type == "chroma":
            self._init_chroma()
        else:
            # fallback: 内存存储（纯测试用）
            self._storage = []
            self._vectors = []

    def _init_milvus(self):
        """初始化 Milvus 连接"""
        try:
            from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility
            connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)

            # 检查 collection 是否存在
            if utility.has_collection(MILVUS_COLLECTION):
                self.collection = Collection(MILVUS_COLLECTION)
                self.collection.load()
                print(f"  [Milvus] 连接已有 collection: {MILVUS_COLLECTION}")
            else:
                # 创建 collection
                schema = CollectionSchema([
                    FieldSchema("id", DataType.INT64, is_primary=True, auto_id=True),
                    FieldSchema("vector", DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM),
                    FieldSchema("text", DataType.VARCHAR, max_length=65535),
                    FieldSchema("source", DataType.VARCHAR, max_length=255),
                    FieldSchema("metadata", DataType.VARCHAR, max_length=1024),
                ])
                self.collection = Collection(MILVUS_COLLECTION, schema)
                # 创建 HNSW 索引
                index_params = {
                    "metric_type": "IP",
                    "index_type": "HNSW",
                    "params": {"M": HNSW_M, "efConstruction": HNSW_EF_CONSTRUCTION},
                }
                self.collection.create_index("vector", index_params)
                self.collection.load()
                print(f"  [Milvus] 创建新 collection: {MILVUS_COLLECTION} (HNSW, IP, M={HNSW_M})")

            self.milvus_ready = True
        except ImportError:
            print("  [Milvus] pymilvus 未安装，回退到内存模式")
            self._storage = []
            self._vectors = []
        except Exception as e:
            print(f"  [Milvus] 连接失败: {e}，回退到内存模式")
            self._storage = []
            self._vectors = []

    def _init_chroma(self):
        """初始化 Chroma"""
        try:
            import chromadb
            os.makedirs(CHROMA_PATH, exist_ok=True)
            self.chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
            self.chroma_collection = self.chroma_client.get_or_create_collection(
                name="rag_docs",
                metadata={"hnsw:space": "ip", "hnsw:M": HNSW_M},
            )
            print(f"  [Chroma] 初始化完成: {CHROMA_PATH}")
            self.chroma_ready = True
        except ImportError:
            print("  [Chroma] chromadb 未安装，回退到内存模式")
            self._storage = []
            self._vectors = []

    def add_chunks(self, chunks: List[DocumentChunk], embeddings: List[List[float]]):
        """添加文档块及其向量"""
        if self.store_type == "chroma" and hasattr(self, "chroma_ready"):
            texts = [c.text for c in chunks]
            metadatas = [{"source": c.metadata.get("source", ""), "chunk_start": str(c.metadata.get("chunk_start", ""))} for c in chunks]
            ids = [f"chunk_{hash(c.text) % 10**8}" for c in chunks]
            self.chroma_collection.add(
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
                ids=ids,
            )
            print(f"  [Chroma] 添加 {len(chunks)} 个向量")
        elif self.store_type == "milvus" and hasattr(self, "milvus_ready"):
            texts = [c.text for c in chunks]
            sources = [c.metadata.get("source", "") for c in chunks]
            metadatas = [str(c.metadata) for c in chunks]
            self.collection.insert([
                [embeddings], [texts], [sources], [metadatas],
            ])
            print(f"  [Milvus] 添加 {len(chunks)} 个向量")
        else:
            self._storage.extend(chunks)
            self._vectors.extend(embeddings)
            print(f"  [内存] 添加 {len(chunks)} 个向量")

    def search(self, query_vector: List[float], top_k: int = 50) -> List[tuple]:
        """
        向量相似性搜索
        返回: [(score, DocumentChunk), ...]
        """
        if self.store_type == "chroma" and hasattr(self, "chroma_ready"):
            results = self.chroma_collection.query(
                query_embeddings=[query_vector],
                n_results=top_k,
                include=["documents", "metadatas", "distances"],
            )
            chunks = []
            for i in range(len(results["documents"][0])):
                chunk = DocumentChunk(
                    text=results["documents"][0][i],
                    metadata=eval(results["metadatas"][0][i]) if isinstance(results["metadatas"][0][i], str) else results["metadatas"][0][i],
                )
                score = 1 - results["distances"][0][i]  # IP 距离转分数
                chunks.append((score, chunk))
            return chunks

        elif self.store_type == "milvus" and hasattr(self, "milvus_ready"):
            search_params = {"metric_type": "IP", "params": {"ef": HNSW_EF_SEARCH}}
            results = self.collection.search(
                data=[query_vector],
                anns_field="vector",
                param=search_params,
                limit=top_k,
                output_fields=["text", "source", "metadata"],
            )
            chunks = []
            for hits in results:
                for hit in hits:
                    chunk = DocumentChunk(
                        text=hit.entity.get("text"),
                        metadata={"source": hit.entity.get("source"), **eval(hit.entity.get("metadata", "{}"))},
                    )
                    chunks.append((hit.score, chunk))
            return chunks

        else:
            # 内存模式：暴力搜索
            import numpy as np
            qv = np.array(query_vector)
            scores = []
            for i, vec in enumerate(self._vectors):
                score = np.dot(qv, np.array(vec))
                scores.append((score, self._storage[i]))
            scores.sort(key=lambda x: x[0], reverse=True)
            return scores[:top_k]

    def count(self) -> int:
        """返回向量总数"""
        if self.store_type == "chroma" and hasattr(self, "chroma_ready"):
            return self.chroma_collection.count()
        elif self.store_type == "milvus" and hasattr(self, "milvus_ready"):
            return self.collection.num_entities
        else:
            return len(self._storage)
