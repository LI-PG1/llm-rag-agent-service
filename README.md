# llm-rag-agent-service

RAG 知识问答 + 多工具 Agent 的融合服务。涵盖的项目场景：

- **浙江电信**：客服文档 RAG（混合检索 + Corrective RAG）
- **南京银行**：语义检索中台（独立 Embedding/Rerank 微服务）
- **上汽集团**：设备运维 Agent（ReAct + 工具调用 + Critic）
- **浙江移动云**：语音交互 Agent（ASR + LLM + TTS）

## 架构

```
rag/ — 检索增强生成管线
  文档加载 → 混合检索(Dense+BM25+RRF) → 重排序 → Sufficient Context → 生成 → Citation 验证

agent/ — 多工具 Agent 引擎
  入口分流 → ReAct 规划循环 → 工具调用 → Critic 校验 → 回答

service/ — FastAPI 端点（/rag/query + /agent/run）
```

## Benchmark

| 指标 | RAG（电信场景） | Agent（运维场景） |
|------|----------------|------------------|
| 评估集 | 150 条客服 query | 100 条工单 |
| 核心指标 | top-5 命中率 76%（+31% vs 纯向量） | 多步完成率 86% |
| 工具准确率 | — | 91% |
| 失败主要原因 | — | 工具选择 57% / context overflow 21% |
| 评测方式 | manual eval（mentor cross-check 30 条） | manual eval |

## 相关仓库

- [llm-deploy-playbook](https://github.com/LI-PG1/llm-deploy-playbook)
- [llm-model-optimization](https://github.com/LI-PG1/llm-model-optimization)
