# 使用说明

## 启动服务

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置环境变量
cp .env.example .env
# 编辑 .env 填入 LLM_BASE_URL

# 3. 启动
python service/service.py

# 4. 测试
curl -X POST http://localhost:8000/rag/query \
  -H "Content-Type: application/json" \
  -d '{"query":"5G 套餐 199 元档包含多少流量？"}'
```

## 两个管线

| 端点 | 用途 | 路由逻辑 |
|------|------|---------|
| /rag/query | RAG 知识问答 | 混合检索→重排序→生成→引用验证 |
| /agent/run | 多步骤任务 | 入口分流→ReAct→工具调用→Critic |

## 数据说明

| 文件 | 内容 |
|------|------|
| data/rag_bench.json | RAG 检索评估（150 条 query，top-5 命中率 76%） |
| data/agent_bench.json | Agent 多步任务（100 条工单，完成率 86%） |

## 容器化部署

```bash
docker-compose up -d
```
