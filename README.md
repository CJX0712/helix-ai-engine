# Helix AI Engine

<div align="center">

[![CI](https://github.com/CJX0712/helix-ai-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/CJX0712/helix-ai-engine/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/CJX0712/helix-ai-engine?sort=semver)](https://github.com/CJX0712/helix-ai-engine/releases)
[![License](https://img.shields.io/badge/License-MIT-green)](./LICENSE)
[![Author](https://img.shields.io/badge/author-晨星-1f6feb)](https://github.com/CJX0712)

**模块化、Provider 无关的 AI 编排引擎：RAG + 工具智能体**

</div>

每个模块通过清晰的接口契约连接，因此**可独立验证**；整套系统**可在干净环境复现**。默认零密钥离线可跑。

---

## 核心能力

| 能力 | 实现 |
| --- | --- |
| Provider 无关 | LLM 与嵌入适配器统一抽象：OpenAI 兼容 / Ollama / 本地确定性兜底 |
| 混合检索 | 内存 NumPy 向量库 + scikit-learn TF-IDF 词法检索 |
| 工具智能体 | ReAct Agent，采用与模型无关的文本化工具调用协议 |
| 双出口 | FastAPI 服务 + CLI |
| 可验证 | 每个模块自带 pytest；端到端测试在离线环境跑通完整链路 |

---

## 快速开始

```bash
make setup      # 建 venv 并安装锁定依赖
make test       # 跑全量测试
make serve      # 在 :8000 启动 HTTP API
```

Windows PowerShell：`./setup.ps1` 然后 `pytest`。

**Provider 模式**

| 环境变量 | 行为 |
| --- | --- |
| `HELIX_PROVIDER_MODE=mock` | 完全离线，使用确定性 provider |
| `HELIX_PROVIDER_MODE=auto`（默认） | 优先本地 Ollama 模型，不可用时优雅降级 |

---

## 架构

```
config → interfaces(契约) → [embeddings | retriever | rerank] → rag
                                                                  ↓
                                              agents/reactor(ReAct) → tools
                                                                  ↓
                                                        api(FastAPI) / cli
```

| 目录 | 职责 |
| --- | --- |
| `helix/core/` | 配置、错误、接口契约 |
| `helix/agents/` | ReAct 智能体与基类 |
| `helix/api/` | FastAPI 服务层 |
| `helix/cli/` | 命令行入口 |
| `docs/` | 架构 / 部署 / 使用文档 |
| `examples/` | 可运行示例 |

---

## 文档

| 文档 | 内容 |
| --- | --- |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | 分层设计与模块契约 |
| [docs/USAGE.md](docs/USAGE.md) | 使用方式 |
| [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) | 部署与容器发布（GHCR） |

---

MIT · 作者：**晨星 · Chen Xing**（[@CJX0712](https://github.com/CJX0712)）
