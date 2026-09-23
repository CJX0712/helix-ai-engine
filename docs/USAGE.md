# Helix AI 引擎 — 使用指南

> 作者：晨星

本文给出命令行、HTTP API 与编程三种使用方式，以及接入真实模型的示例。

---

## 1. 命令行（CLI）

```bash
# 与模型直接对话
python -m helix.cli.main chat "用一句话介绍 Helix"

# 运行内置 RAG 演示（入库示例文档并检索/作答）
python -m helix.cli.main rag

# 让智能体完成任务（可使用 calculator / time / text 工具）
python -m helix.cli.main agent "请计算 (12 + 8) * 3"

# 启动 HTTP 服务
python -m helix.cli.main serve
```

安装后可简化为：`helix chat "..."`（`pyproject.toml` 已注册 `helix` 入口）。

---

## 2. HTTP API

服务启动后（默认 `:8000`），主要端点：

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 健康检查 + 组件清单 |
| POST | `/v1/chat` | 对话 `{messages:[{role,content}]}` |
| POST | `/v1/embed` | 向量化 `{texts:[...]}` |
| POST | `/v1/rag/ingest` | 入库 `{documents:[{id,text,metadata}]}` |
| POST | `/v1/rag/query` | 检索 `{query,top_k?}` |
| POST | `/v1/rag/answer` | 基于上下文作答 `{query}` |
| POST | `/v1/agents/run` | 运行智能体 `{task}` |

**示例**

```bash
# 健康检查
curl http://localhost:8000/health

# 入库
curl -X POST http://localhost:8000/v1/rag/ingest \
  -H "Content-Type: application/json" \
  -d '{"documents":[{"id":"d1","text":"Helix 是一套模块化 AI 编排引擎。"}]}'

# 检索
curl -X POST http://localhost:8000/v1/rag/query \
  -H "Content-Type: application/json" \
  -d '{"query":"Helix 是什么？"}'

# 对话
curl -X POST http://localhost:8000/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"你好"}]}'

# 智能体
curl -X POST http://localhost:8000/v1/agents/run \
  -H "Content-Type: application/json" \
  -d '{"task":"请计算 (12 + 8) * 3"}'
```

---

## 3. 编程接入

```python
from helix.core.config import Settings
from helix.core.registry import Container
from helix.core.types import Document, Message, Role
import asyncio

async def main():
    c = Container(Settings(_env_file=None, provider_mode="mock"))
    n = await c.rag.ingest([Document(id="d1", text="Helix 支持 RAG 与智能体。")])
    print("chunks:", n)
    print(await c.rag.answer("Helix 能做什么？", c.llm))
    print(await c.agent.run("请计算 2 + 3 * 4"))
    await c.close()

asyncio.run(main())
```

---

## 4. 真实模型示例（本地 Ollama）

配置 `HELIX_PROVIDER_MODE=auto` 并指向本地 Ollama 后，运行 `examples/real_demo.py`
实测输出（节选）：

```
[providers] llm=auto-fallback-llm embeddings=auto-fallback-embed

[chat]
 Helix 是阿里云开发的微服务引擎。

[rag] ingested 2 chunk(s)
[rag answer]
 Helix 是一套模块化、供应商无关的 AI 编排引擎，支持检索增强生成(RAG)与工具调用智能体。

[agent]
 60
```

其中 `(12 + 8) * 3 = 60` 由 `calculator` 工具真实计算得出，证明智能体工具调用链路打通。

---

## 5. 接入自定义模型 / 厂商

只需在 `core/registry.py` 的 `Container` 中替换对应装配：

```python
# 例如指向 OpenAI
Settings(llm_base_url="https://api.openai.com/v1",
         llm_api_key="sk-...",
         llm_model="gpt-4o-mini",
         embedding_base_url="https://api.openai.com/v1",
         embedding_model="text-embedding-3-small")
```

所有 OpenAI 兼容端点（Ollama / vLLM / LM Studio / 各云厂商）均原生支持，无需改代码。

---

## 6. 测试

```bash
make test
# 等价于：pytest -q
```

覆盖：核心类型/错误、LLM 适配与降级、嵌入与降级、向量库、RAG、TF-IDF 检索、
工具、ReAct 智能体、HTTP API 契约，以及 `e2e_test.py` 的完整链路（离线确定性实现）。
