# llm-rag-agent-service

RAG 知识问答 + 多工具 Agent 的融合服务。覆盖以下场景：

- **省级运营商**：客服文档 RAG（混合检索 + Corrective RAG）
- **商业银行**：语义检索中台（独立 Embedding/Rerank 微服务）
- **大型制造企业**：设备运维 Agent（ReAct + 工具调用 + Critic）
- **省级运营商**：语音交互 Agent（ASR + LLM + TTS）

## 架构

```
rag/ — 检索增强生成管线
  文档加载 → 混合检索(Dense+BM25+RRF) → 重排序 → Sufficient Context → 生成 → Citation 验证

agent/ — 多工具 Agent 引擎
  入口分流 → ReAct 规划循环 → 工具调用 → Critic 校验 → 回答

service/ — FastAPI 端点（/rag/query + /agent/run）
```

## Benchmark

| 指标 | RAG | Agent |
|------|-----|-------|
| 评估集规模 | 百余条客服 query | 百余条工单场景 |
| 核心指标 | top-5 命中率 +31%（vs 纯向量） | 多步完成率 >85% |
| 工具调用准确率 | — | >90% |
| 评估方式 | manual eval + mentor 抽检 | manual eval |

数据详情见 [data/](./data/) 目录。

## 相关仓库

- [llm-deploy-playbook](https://github.com/LI-PG1/llm-deploy-playbook)
- [llm-model-optimization](https://github.com/LI-PG1/llm-model-optimization)
