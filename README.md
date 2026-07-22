# llm-rag-agent-service

RAG 知识问答 + 多工具 Agent 的融合服务。两个管线共用同一个推理底座。

---

## 目录结构

```
├── README.md                # 本文件
├── core/
│   └── memory.py            # 对话记忆管理器
├── rag/
│   ├── document_processor.py # 文档加载 & 切分
│   ├── vector_store.py       # 向量库（Chroma/Milvus）
│   ├── retriever.py          # 混合检索（Dense + BM25 + RRF）
│   ├── reranker.py           # Cross-encoder 重排序
│   ├── generator.py          # 多模型路由 + Corrective RAG
│   └── rag_pipeline.py       # 端到端管线编排
├── agent/
│   ├── orchestrator.py       # ReAct 规划循环
│   ├── critic.py             # Critic 校验
│   ├── router.py             # 入口分流
│   └── tools/
│       ├── __init__.py       # 工具注册表
│       ├── rag_tool.py       # 手册检索工具
│       ├── sql_tool.py       # 备件库查询
│       └── api_tool.py       # 报修工单 API
├── service/
│   ├── service.py            # FastAPI 入口
│   └── config.yaml           # 服务配置
└── data/
    ├── rag_bench.json        # RAG 检索评估数据
    └── agent_bench.json      # Agent 多步任务评估数据
```

## 快速导航

| 你想用什么 | 路径 |
|-----------|------|
| RAG 问答 | rag/rag_pipeline.py → service/service.py (/rag/query) |
| Agent 调度 | agent/orchestrator.py → service/service.py (/agent/run) |
| 混合检索 | rag/retriever.py + rag/reranker.py |
| 添加新工具 | agent/tools/ 下新建 + __init__.py 注册 |
| 改配置 | service/config.yaml |
| 改记忆策略 | core/memory.py |
| 容器化部署 | docker-compose.yml + requirements.txt |

## 管线关系

```
         ┌── 简单查询 ──→ RAG 直回（不进 Agent 循环）
用户请求 ─┤
         └── 复杂任务 ──→ Agent (ReAct) ──→ 工具调用 ──→ Critic 校验 ──→ 回答
                              │                   │
                              └── RAG 检索工具 ────┘（复用 rag/ 管线）
```

## 相关仓库

- [llm-deploy-playbook](https://github.com/LI-PG1/llm-deploy-playbook) — 底层部署方案和踩坑记录
- [llm-model-optimization](https://github.com/LI-PG1/llm-model-optimization) — 量化评估和微调模板
