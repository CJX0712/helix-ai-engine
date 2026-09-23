# Helix AI 引擎 — 部署指南

> 作者：晨星

Helix 可在**本地虚拟环境**、**Docker** 或任意容器平台上运行。所有外部能力通过
OpenAI 兼容协议接入（Ollama / OpenAI / vLLM / LM Studio 等），因此部署本身与模型推理解耦。

---

## 1. 本地部署（venv）

```bash
# 1) 创建隔离环境并安装锁版依赖
make setup            # Linux/macOS
# 或 Windows PowerShell:
./setup.ps1

# 2) 运行测试，确认链路完整
make test

# 3) 启动 HTTP 服务（默认 0.0.0.0:8000）
make serve
```

`make setup` 等价于：
```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.lock
.venv/Scripts/pip install -e .[dev]
```

---

## 2. Docker 部署

```bash
docker build -t helix-ai-engine .
docker run -p 8000:8000 \
  -e HELIX_PROVIDER_MODE=auto \
  -e HELIX_LLM_BASE_URL=http://host.docker.internal:11434/v1 \
  -e HELIX_EMBEDDING_BASE_URL=http://host.docker.internal:11434/v1 \
  helix-ai-engine
```

`Dockerfile` 已基于 `python:3.13-slim` 并使用 `requirements.lock` 安装，构建可复现。
镜像内 `CMD` 为 `python -m helix.cli.main serve`（uvicorn, `0.0.0.0:8000`）。

---

## 3. 环境变量（全部以 `HELIX_` 为前缀）

| 变量 | 默认 | 说明 |
|------|------|------|
| `HELIX_PROVIDER_MODE` | `auto` | `auto` / `openai` / `mock` |
| `HELIX_LLM_BASE_URL` | `http://localhost:11434/v1` | LLM 的 OpenAI 兼容端点 |
| `HELIX_LLM_API_KEY` | `ollama` | LLM 访问密钥（Ollama 可任意值） |
| `HELIX_LLM_MODEL` | `qwen2.5:0.5b` | 聊天模型名 |
| `HELIX_EMBEDDING_BASE_URL` | `http://localhost:11434/v1` | 嵌入端点 |
| `HELIX_EMBEDDING_API_KEY` | `ollama` | 嵌入密钥 |
| `HELIX_EMBEDDING_MODEL` | `nomic-embed-text` | 嵌入模型名 |
| `HELIX_VECTORSTORE_TYPE` | `memory` | 向量库类型 |
| `HELIX_TOP_K` | `4` | 检索返回条数 |
| `HELIX_MAX_AGENT_ITERATIONS` | `6` | 智能体最大步数 |
| `HELIX_LOG_LEVEL` | `INFO` | 日志级别 |

复制 `.env.example` 为 `.env` 后按需修改即可。

---

## 4. 接入本地 Ollama（真实模型）

```bash
# 安装并启动 Ollama 后拉取模型
ollama pull qwen2.5:1.5b-instruct     # 聊天
ollama pull nomic-embed-text          # 嵌入

# 让 Helix 使用真实模型（端点不可达时自动降级）
export HELIX_PROVIDER_MODE=auto
make serve
```

> 注意：若本机存在 `HTTP_PROXY` / `HTTPS_PROXY`，Helix 对 `localhost` / `127.0.0.1`
> 端点会自动绕过代理（`trust_env=False`），无需手动配置 `NO_PROXY`。

---

## 5. 可复现性保障

| 环节 | 机制 |
|------|------|
| 依赖 | `requirements.lock`（pip freeze 精确版本）+ `requirements.txt`（逻辑依赖） |
| 构建 | `Dockerfile` 多阶段思路 + 锁版安装 |
| 验证 | `tests/` 每模块独立用例 + `tests/e2e_test.py` 端到端 |
| 配置 | 全部经 `pydantic-settings` 由环境变量驱动，12-Factor 友好 |

---

## 6. 健康检查与可观测性

- `GET /health` 返回各组件名称与已注册工具列表，可用于探针 / 编排健康检查。
- 所有模块使用统一 `get_logger` 输出结构化日志（含组件名），便于接入集中日志。

---

## 7. 扩展与水平伸缩建议

- **向量库**：实现 `VectorStore` 接口即可接入 Chroma / Qdrant / PGVector，替换 `Container` 中一行装配。
- **嵌入/LLM**：实现 `EmbeddingProvider` / `LLMProvider` 接口即可接入新厂商；当前 `OpenAICompatible*` 已覆盖绝大多数。
- **API 进程**：多副本无状态部署，向量库外置；`/health` 作就绪探针。
- **推理算力**：将 `HELIX_LLM_BASE_URL` 指向带 GPU 的推理服务（vLLM / TGI）即可提升吞吐。
