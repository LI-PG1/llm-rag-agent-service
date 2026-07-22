# llm-rag-agent-service

RAG 知识问答 + 多工具 Agent 的融合服务。覆盖以下场景：

- **某省级运营商**：客服文档 RAG（混合检索 + Corrective RAG）
  语音交互 Agent（ASR + LLM + TTS）
- **某地区商业银行**：语义检索中台（独立 Embedding/Rerank 微服务）
- **大型制造业国企**：设备运维 Agent（ReAct + 工具调用 + Critic）

## 架构

rag/ — 检索增强生成：文档加载 -> 混合检索 -> 重排序 -> 生成 -> Citation 验证
agent/ — 多工具 Agent：入口分流 -> ReAct 循环 -> 工具调用 -> Critic 校验
service/ — FastAPI 端点（/rag/query + /agent/run）

## Benchmark

| 指标 | RAG | Agent |
|------|-----|-------|
| 评估集 | 百余条客服 query | 百余条工单场景 |
| 核心指标 | top-5 命中率 +31%（vs 纯向量） | 多步完成率 >85% |
| 工具调用准确率 | 不适用 | >90% |
| 评估方式 | manual eval + mentor 抽检 | manual eval |

## 相关仓库

- [llm-deploy-playbook](https://github.com/LI-PG1/llm-deploy-playbook) — 部署笔记和踩坑记录
- [llm-model-optimization](https://github.com/LI-PG1/llm-model-optimization) — 量化评估和微调模板
