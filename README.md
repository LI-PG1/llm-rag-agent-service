# llm-rag-agent-service

RAG 知识问答 + 多工具 Agent 的融合服务。共用推理底座。

## 架构

```
rag/ — 检索增强生成管线
  → 文档解析 → 混合检索(Dense+BM25+RRF) → 重排序 → Corrective RAG

agent/ — 多工具 Agent 引擎
  → 入口分流 → ReAct 规划 → 工具调用 → Critic 校验

core/ — 共享组件（对话记忆、LLM 路由）

service/ — FastAPI 入口
  → /rag/query → /agent/run
```

## 相关仓库

- [llm-deploy-playbook](https://github.com/LI-PG1/llm-deploy-playbook) — 底层部署方案
