# llm-rag-agent-service

RAG 知识问答 + 多工具 Agent 的融合服务。共用推理底座。

## 架构

```
rag/ — 检索增强生成管线
  文档加载 → 混合检索 → 重排序 → 生成 → Citation 验证

agent/ — 多工具 Agent 引擎
  入口分流 → ReAct 循环 → 工具调用 → Critic 校验 → 回答

service/ — FastAPI 端点（/rag/query + /agent/run）
```

## Benchmark

| 指标 | RAG | Agent |
|------|-----|-------|
| 评估集规模 | 150 条 query | 100 条工单 |
| 核心指标 | top-5 命中率 76%（+31% vs 纯向量）| 多步完成率 86% |
| 工具准确率 | — | 91% |
| 失败主要原因 | — | 工具选择错误 57% / context overflow 21% |

## API

```bash
curl -X POST http://localhost:8000/rag/query -d '{"query":"5G 套餐多少流量？"}'
curl -X POST http://localhost:8000/agent/run -d '{"query":"查轴承库存并报修"}'
```

## 相关仓库

- [llm-deploy-playbook](https://github.com/LI-PG1/llm-deploy-playbook)
- [llm-model-optimization](https://github.com/LI-PG1/llm-model-optimization)
