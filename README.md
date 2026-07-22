# llm-rag-agent-service

RAG 知识问答 + 多工具 Agent 的融合服务。

## 架构

```
rag/ — 检索增强生成
  文档加载 → 混合检索(Dense+BM25+RRF) → 重排序 → Sufficient Context → 生成 → Citation 验证

agent/ — 多工具 Agent
  入口分流 → ReAct 规划循环 → 工具调用 → Critic 校验 → 回答

core/ — 共享组件（LLM 路由、对话记忆）
service/ — FastAPI 端点（/rag/query + /agent/run）
```

## API

```bash
# RAG 问答
curl -X POST http://localhost:8000/rag/query \
  -H "Content-Type: application/json" \
  -d '{"query":"5G 套餐包含多少流量？"}'

# Agent 任务
curl -X POST http://localhost:8000/agent/run \
  -H "Content-Type: application/json" \
  -d '{"query":"查 M-2024 轴承库存并提交报修"}'
```

## 相关仓库

- [llm-deploy-playbook](https://github.com/LI-PG1/llm-deploy-playbook)
- [llm-model-optimization](https://github.com/LI-PG1/llm-model-optimization)
