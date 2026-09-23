# Helix AI 引擎 — 系统架构

> 作者：晨星
> 版本：1.0.0

Helix 是一套**供应商无关、模块解耦、可独立验证、可端到端运行**的 AI 系统开发框架。
其设计目标：优先复用开源成果（FastAPI / Pydantic / NumPy / scikit-learn / Ollama·OpenAI
兼容协议），不自研底层模型，把所有复杂度收敛到**清晰的接口契约**与**依赖注入装配**上。

---

## 1. 设计原则

1. **单一职责**：每个模块只解决一件事，互不越界。
2. **面向接口编程**：模块之间只依赖 `core/interfaces.py` 里的 `Protocol`，不依赖具体实现。
3. **可独立验证**：每个模块都有对应的 `tests/*` 用例，可单独运行。
4. **可热插拔**：切换 LLM / 嵌入 / 向量库只需改 `core/registry.py` 一处。
5. **离线可跑 + 真实可用**：`auto` 模式优先用真实模型，端点不可达时透明降级到本地确定性实现。
6. **可复现**：`requirements.lock` 精确锁版，干净环境一条命令复现。

---

## 2. 模块划分与接口契约

| 模块 | 职责 | 接口（Protocol） | 关键实现 |
|------|------|------------------|----------|
| `core` | 配置 / 类型 / 错误 / 注册表(DI) | `LLMProvider` `EmbeddingProvider` `VectorStore` `Retriever` `Tool` `Agent` | `Container`（装配根） |
| `providers/llm` | LLM 适配 | `LLMProvider` | `OpenAICompatibleLLM` · `MockLLMProvider` · `AutoFallbackLLM` |
| `providers/embeddings` | 向量化适配 | `EmbeddingProvider` | `OpenAICompatibleEmbedding` · `DeterministicEmbedding` · `AutoFallbackEmbedding` |
| `vectorstore` | 向量库 | `VectorStore` | `MemoryVectorStore`（NumPy 余弦） |
| `rag` | 切分 / 检索 / 生成管线 | `Retriever` | `RecursiveCharacterTextSplitter` · `RAGPipeline` · `TfidfRetriever` |
| `agents` | ReAct 工具调用编排 | `Agent` | `ReActAgent` |
| `tools` | 原子工具 | `Tool` | `CalculatorTool` · `TimeTool` · `TextTool` |
| `api` | HTTP 服务 | REST | FastAPI `create_app` |
| `cli` | 命令行演示 | — | argparse `main` |

---

## 3. 调用关系（端到端链路）

```
                +-------------------+
   CLI / API -->|      Agent        |
                +--------+----------+
                         | 调用
          +--------------+--------------+
          |                             |
       Tool.run()                  LLM.generate() / .stream()
                                         |
                                    +----+-----+
                                    |    RAG    |
                                    +----+-----+
                                         | 检索
                              EmbeddingProvider.embed()
                                         |
                                    VectorStore.search()
```

- `Container`（`core/registry.py`）是唯一的装配根，把上述组件按接口注入。
- `Agent` 只认识 `LLMProvider` 与 `Tool` 接口；`RAG` 只认识 `EmbeddingProvider` 与 `VectorStore` 接口。
- 任何实现都可被替换（例如把 `MemoryVectorStore` 换成 Chroma/PGVector）而不动其他模块。

---

## 4. 接口契约（节选）

```python
# core/interfaces.py
@runtime_checkable
class LLMProvider(Protocol):
    name: str
    async def generate(self, messages: List[Message], **kwargs) -> str: ...
    async def stream(self, messages: List[Message], **kwargs) -> AsyncIterator[str]: ...
    async def close(self) -> None: ...

@runtime_checkable
class EmbeddingProvider(Protocol):
    name: str
    dimension: int
    async def embed(self, texts: List[str]) -> List[List[float]]: ...

@runtime_checkable
class VectorStore(Protocol):
    name: str
    async def add(self, chunks, vectors) -> None: ...
    async def search(self, vector, top_k) -> List[ScoredChunk]: ...
```

---

## 5. 数据流

1. **入库（RAG.ingest）**：`Document` → `Splitter` 切成 `Chunk` → `EmbeddingProvider.embed` → `VectorStore.add`。
2. **检索（RAG.retrieve）**：query → `embed` → `VectorStore.search(top_k)` → `List[ScoredChunk]`。
3. **生成（RAG.answer）**：检索结果拼成上下文 → `LLMProvider.generate` → 答案。
4. **智能体（Agent.run）**：任务 → `LLM` 产出 `ACTION/INPUT` 或 `FINAL` → 命中 `Tool` 执行并回灌观察 → 循环至 `FINAL`。

智能体使用**文本化工具调用协议**（`ACTION:` / `INPUT:` / `FINAL:`），因此兼容任何聊天模型，
无需依赖厂商私有 function-calling 接口。

---

## 6. 容错与可复现

- **自动降级**：`AutoFallbackLLM` / `AutoFallbackEmbedding` 在真实端点首次 `ProviderError` 时
  永久切换到本地确定性实现，保证整条链路始终可运行。
- **确定性离线实现**：`DeterministicEmbedding`（词哈希）与 `MockLLMProvider` 使 CI / 干净环境
  无需 GPU、无需密钥即可跑通全部测试与端到端链路。
- **代理绕过**：适配器对 `localhost` / `127.0.0.1` 端点自动 `trust_env=False`，避免本机 HTTP 代理
  拦截本地 Ollama 请求（已实测修复）。
- **锁版依赖**：`requirements.lock` 为 `pip freeze` 精确快照；`requirements.txt` 为人工维护的逻辑依赖。

---

## 7. 目录结构

```
helix-ai-engine/
├── helix/
│   ├── core/         # 配置/类型/接口/注册表/错误/日志
│   ├── providers/    # llm + embeddings 适配层
│   ├── vectorstore/  # 向量库
│   ├── rag/          # 切分器 + RAG 管线 + TF-IDF 检索
│   ├── agents/       # ReAct 编排
│   ├── tools/        # 原子工具
│   ├── api/          # FastAPI 服务
│   └── cli/          # 命令行入口
├── tests/            # 每模块独立测试 + 端到端
├── examples/         # 真实模型演示脚本
├── docs/             # 架构/部署/使用
├── requirements.lock # 精确锁版
├── Dockerfile
├── Makefile / setup.ps1
└── pyproject.toml
```
