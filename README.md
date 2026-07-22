# llm-rag-agent-service

RAG 知识问答 + 多工具 Agent 的融合服务。覆盖以下场景：

- **某省级运营商**：客服文档 RAG（混合检索 + Corrective RAG）
- **某地区商业银行**：语义检索中台（独立 Embedding/Rerank 微服务）
- **大型制造业国企**：设备运维 Agent（ReAct + 工具调用 + Critic）
- **某省级运营商（语音渠道）**：语音交互 Agent（ASR + LLM + TTS）

## 架构



## Benchmark

| 指标 | RAG | Agent |
|------|-----|-------|
| 评估集 | 百余条客服 query | 百余条工单场景 |
| 核心指标 | top-5 命中率 +31%（vs 纯向量） | 多步完成率 >85% |
| 工具调用准确率 | 不适用 | >90% |
| 评估方式 | manual eval + mentor 抽检 | manual eval |

数据详情见 [data/](./data/) 目录。

## 相关仓库

- [llm-deploy-playbook](https://github.com/LI-PG1/llm-deploy-playbook)
- [llm-model-optimization](https://github.com/LI-PG1/llm-model-optimization)
