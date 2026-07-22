# llm-rag-agent-service

RAG 知识问答 + 多工具 Agent 的融合服务。共用推理底座。

---

## 目录

```
├── [README.md](README.md)
├── [USAGE.md](USAGE.md)                  — 快速使用指南
├── [requirements.txt](requirements.txt)
│
├── [core/](core/)
│   └── [memory.py](core/memory.py)              — 对话记忆管理
│
├── [rag/](rag/)  ← RAG 检索增强生成管线
│   ├── [document_processor.py](rag/document_processor.py)  — 文档加载 & 切分
│   ├── [vector_store.py](rag/vector_store.py)        — 向量库
│   ├── [retriever.py](rag/retriever.py)           — 混合检索
│   ├── [reranker.py](rag/reranker.py)            — 重排序
│   ├── [generator.py](rag/generator.py)           — 生成 + Citation 验证
│   └── [rag_pipeline.py](rag/rag_pipeline.py)       — 管线编排
│
├── [agent/](agent/)  ← Agent 引擎
│   ├── [orchestrator.py](agent/orchestrator.py)      — ReAct 规划循环
│   ├── [critic.py](agent/critic.py)             — Critic 校验
│   ├── [router.py](agent/router.py)             — 入口分流
│   └── [tools/](agent/tools/)
│       ├── [__init__.py](agent/tools/__init__.py)      — 工具注册表
│       ├── [rag_tool.py](agent/tools/rag_tool.py)      — 手册检索
│       ├── [sql_tool.py](agent/tools/sql_tool.py)      — 备件库查询
│       └── [api_tool.py](agent/tools/api_tool.py)      — 报修工单 API
│
├── [service/](service/)  ← FastAPI 服务入口
│   ├── [service.py](service/service.py)           — /rag/query + /agent/run
│   └── [config.yaml](service/config.yaml)          — 服务配置
│
└── [data/](data/)
    ├── [rag_bench.json](data/rag_bench.json)         — RAG 评估
    └── [agent_bench.json](data/agent_bench.json)       — Agent 评估
```

## 管线关系

```
         ┌── 简单查询 ──→ RAG 直回
用户请求 ─┤
         └── 复杂任务 ──→ Agent (ReAct) → 工具调用 → Critic → 回答
```

## 相关仓库

- [llm-deploy-playbook](https://github.com/LI-PG1/llm-deploy-playbook)
- [llm-model-optimization](https://github.com/LI-PG1/llm-model-optimization)
