# vfront

[English](./README.md)

由 [vLLM](https://github.com/vllm-project/vllm) 驱动的 OpenAI 兼容 API 服务。

它的目标是成为 OpenAI API 的直接替代：接口路径、请求字段、响应结构、流式事件都尽量对齐官方 SDK。把你的 `openai` 客户端 `base_url` 指向 vfront，现有调用通常无需改动。

## 特性

- **OpenAI 兼容的核心 API 面**：Chat Completions、Completions、Embeddings、Models、Responses、Files、Batches，以及已实现的存储对象生命周期接口
- **流式输出**：Chat/Completions 的 `[DONE]` 语义与官方保持一致，Responses 事件流完整
- **Responses API**：支持 background 模式、多轮上下文、取消、上下文截断，以及 `input_items` 查询
- **多 LoRA**：通过 `model` 字段按请求切换 LoRA
- **MCP 工具**：支持服务端工具调用与流式 agent loop
- **批处理**：支持 chat、completions、embeddings、responses 的异步 batch
- **持久化**：基于 LMDB 持久化 responses、batches 和文件元数据
- **类型安全**：Pydantic v2 协议模型，mypy clean
- **高性能**：Granian + uvloop + vLLM PagedAttention

## 云端专属能力的处理方式

vfront 面向 **本地 / 自托管推理**。依赖 OpenAI 云端基础设施的字段，会为了 SDK 兼容而被接受，但通常会被忽略，不会因为这些字段而在客户端报校验错误。

### 未实现的接口

以下 OpenAI API 家族依赖云端后端，因此当前不会提供：

| 接口家族 | 原因 |
|----------|------|
| `/v1/audio/*` | 需要 Whisper / TTS 服务 |
| `/v1/images/*` | 需要 DALL·E 服务 |
| `/v1/moderations` | 需要 OpenAI moderation 分类器 |
| `/v1/fine_tuning/*` | 需要云端训练流水线 |
| `/v1/vector_stores/*` | 需要云端向量数据库 |
| `/v1/assistants/*`, `/v1/threads/*` | 已废弃的云端状态型运行时 |
| `/v1/evals/*` | 需要云端评测流水线 |
| `/v1/organization/*`, `/v1/invites/*` | 云端账号管理能力 |

### Responses 内建工具

Responses API 中依赖 OpenAI 云端的内建工具当前不支持，建议改用 MCP：

| 内建工具 | 云端行为 | 本地替代 |
|----------|----------|----------|
| `web_search` | OpenAI 托管搜索 | MCP web-search server |
| `file_search` | 云端向量检索 | MCP filesystem server |
| `code_interpreter` | 云端 Python 沙箱 | MCP code execution server |
| `computer_use` | 云端桌面操作 | MCP desktop automation server |
| `image_generation` | DALL·E 集成 | MCP image generation server |

## 快速开始

### 前置条件

- Python 3.12+
- 支持 CUDA 的 GPU
- [uv](https://github.com/astral-sh/uv)

### 安装

```bash
git clone https://github.com/your-org/vfront.git
cd vfront

uv venv --python 3.12
source .venv/bin/activate
uv pip install -e .
```

### 配置

先确定部署模式，再复制对应示例配置：

```bash
cp embedded.yaml.example embedded.yaml
# 或
cp frontend.yaml.example frontend.yaml
# 或
cp engine.yaml.example engine.yaml
```

支持三种模式：

- `embedded`：单机部署，frontend、存储、jobs，以及本地 runtime 或本地 managed engine 一起运行
- `frontend`：纯控制面，只负责 OpenAI 兼容 API、存储、jobs 和路由
- `engine`：独立 vLLM worker，只服务一个基础模型及其 LoRA

CLI 推荐通过 `-c` 指定 YAML。vfront 会先把配置文件路径解析成绝对路径，再交给 `fastapiex.settings`。

### 运行

```bash
# 单机
uv run vfront embedded -c embedded.yaml

# 纯 frontend
uv run vfront frontend -c frontend.yaml

# 独立 engine
uv run vfront engine -c engine.yaml
```

默认 frontend 监听在 `http://0.0.0.0:8000`。

### 使用

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="any")

resp = client.chat.completions.create(
    model="Qwen/Qwen2.5-7B-Instruct",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(resp.choices[0].message.content)
```

## API 接口

### 核心接口

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/v1/chat/completions` | Chat Completions（支持流式、tools、`n>1`、`logit_bias`、`logprobs`） |
| `POST` | `/v1/completions` | 传统 Completions（支持流式、`suffix`、`best_of`） |
| `POST` | `/v1/embeddings` | Embeddings（支持 float / base64、`dimensions`） |
| `GET` | `/v1/models` | 列出模型（基础模型 + LoRA） |
| `GET` | `/v1/models/{model}` | 获取模型详情 |
| `DELETE` | `/v1/models/{model}` | 当前不支持运行时模型目录变更 |

### 已存储的 Chat Completions

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/v1/chat/completions/{id}` | 获取已存储的 Chat Completion |
| `POST` | `/v1/chat/completions/{id}` | 更新已存储 Chat Completion 的 metadata |
| `DELETE` | `/v1/chat/completions/{id}` | 删除已存储的 Chat Completion |
| `GET` | `/v1/chat/completions/{id}/messages` | 列出已存储 Chat Completion 的消息 |

### Responses API

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/v1/responses` | 创建 Response（支持流式事件、background、tools） |
| `GET` | `/v1/responses/{id}` | 获取已存储的 Response |
| `GET` | `/v1/responses/{id}/input_items` | 列出生成该 Response 时使用的输入项 |
| `DELETE` | `/v1/responses/{id}` | 删除 Response |
| `POST` | `/v1/responses/{id}/cancel` | 取消进行中的 Response |

`GET /v1/responses` 故意不提供，因为它不在当前仓库的 `openapi.yaml` 合同里。

### Files / Batches

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/v1/files` | 上传文件（multipart） |
| `GET` | `/v1/files` | 列出文件 |
| `GET` | `/v1/files/{id}` | 获取文件元数据 |
| `GET` | `/v1/files/{id}/content` | 下载文件内容 |
| `DELETE` | `/v1/files/{id}` | 删除文件 |
| `POST` | `/v1/batches` | 创建 batch |
| `GET` | `/v1/batches/{id}` | 获取 batch 状态 |
| `GET` | `/v1/batches` | 列出 batch |
| `POST` | `/v1/batches/{id}/cancel` | 取消 batch |

### 健康检查

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/health` | 存活检查 |
| `GET` | `/ready` | 就绪检查 |

## 部署模式

### Embedded

`embedded` 是最简单的模式，适合单机或开发环境。

使用 [embedded.yaml.example](./embedded.yaml.example) 作为起点。

按 `frontend.routing.transport` 分三种行为：

- `local`：vLLM 在 frontend 进程内，必须 `frontend.server.workers=1`
- `managed`：frontend 会拉起本地 engine 子进程，并通过 UDS 与之通信；frontend workers 可以独立扩展
- `remote`：也能工作，但一般不如直接使用 `frontend`

### Frontend

`frontend` 是控制面，负责：

- OpenAI 兼容 API
- files / batches / responses
- MCP
- job runner
- 把推理请求路由到 engine worker

使用 [frontend.yaml.example](./frontend.yaml.example) 作为起点。

`routing.backends[].base_url` 支持两种形式：

- 普通 HTTP：`http://10.0.0.11:9001`
- UDS：`unix:///tmp/vfront/qwen-7b.sock`

`frontend` 自己不负责托管远程 engine；它只根据配置路由。

### Engine

`engine` 是独立的 vLLM worker，负责：

- 监听 frontend 请求
- 服务基础模型
- 挂载 LoRA
- 内部 API 鉴权

使用 [engine.yaml.example](./engine.yaml.example) 作为起点。

监听说明：

- 独立 `engine` 模式在配置了 `engine.server.uds` 时优先监听 UDS；否则使用 `engine.server.host` / `engine.server.port`
- `embedded` 拉起的 managed engine 通过 UDS 监听
- 当 engine 是由 `embedded` 拉起的 managed worker 时，父进程注入的 UDS 路径优先级更高

## 配置说明

### `frontend.app` / `engine.app`

- `frontend.app.api_key` 用于保护 `/v1/*`
- 环境变量覆盖遵循 `fastapiex.settings` 的路径键规则，因此 `frontend.app.api_key` 对应 `FRONTEND__APP__API_KEY`
- 生产环境建议显式配置 `cors_origins`
- `log_dir` 控制日志目录

### `frontend.server` / `engine.server`

- `host` 和 `port` 控制 TCP 监听
- `uds` 是本地 Unix domain socket 文件路径，不是 URL
- 配置了 `uds` 时，它会优先于 `host` / `port`

### `frontend.routing`

- `transport=local` 只用于 `embedded`
- `transport=remote` 用于 `frontend`
- `transport=managed` 让 `embedded` 使用本地 UDS engine worker
- `models` 定义对外暴露的基础模型拓扑
- frontend 的模型项不再配置 `max_model_len` 这类运行时事实；这些信息以后端 health 为准
- `backends` 定义 `engine_key -> endpoint` 的映射
- `backends[].base_url` 支持 `http://...` 和 `unix:///path/to.sock`
- 当 `embedded` 使用 `managed` 时，UDS backend 会自动由 `engine.managed_workers.socket_dir` 推导

### `engine.runtime`

- `model` 是 Hugging Face 模型名或本地路径
- `served_model_name` 是对外暴露的 engine key
- `tensor_parallel_size`、`pipeline_parallel_size`、`dtype`、`gpu_memory_utilization` 等会传给 vLLM

### `adapters`

- `modules[].name` 是对外使用的 LoRA 模型名
- `modules[].base_model` 用于绑定到某个基础模型
- 请求里直接把 adapter `name` 作为 `model` 即可

### `frontend.jobs`

- background Responses 和 Batches 由私有 runner 执行
- `run_embedded=true` 只对 `embedded` 有意义
- `embedded + local` 时 runner 在 API 进程内，因此 workers 必须是 `1`
- `embedded + managed` 和 `frontend` 会使用私有 runner 子进程，因此 frontend workers 可以独立扩展

### `engine.managed_workers`

- 只在 `embedded + frontend.routing.transport=managed` 时使用
- `api_key` 是 frontend 调本地 engine worker 的内部凭证
- `socket_dir` 决定本地 UDS 文件放在哪里

### `engine.api`

- 用于 `engine` worker 自己的内部 API
- `api_key` 保护 engine 内部接口
- `log_access` 控制 engine 侧访问日志

## 传输矩阵

| 模式 | engine 放置方式 | frontend 到 engine 的传输方式 | frontend workers |
|------|------------------|-------------------------------|------------------|
| `embedded + frontend.routing.transport=local` | 进程内 | Python 直接调用 | 必须为 `1` |
| `embedded + frontend.routing.transport=managed` | 本地子进程 | HTTP over UDS | 可以大于 `1` |
| `frontend + frontend.routing.transport=remote` | 外部 engine 进程 | HTTP 或 HTTP over UDS | 可以大于 `1` |
| `engine` | 独立 worker | 配置了 `engine.server.uds` 时监听 UDS，否则监听 `engine.server.host/engine.server.port` | 固定为 `1` |

## 架构

```text
client / SDK
    -> vfront frontend
        -> routing + storage + jobs + MCP
        -> engine router
            -> vfront engine worker(s)
                -> vLLM runtime
```

目录分层也和这个边界一致：

- `vfront/frontend/*`：API、middleware、store、jobs、router、MCP
- `vfront/engine/*`：runtime 和 engine worker
- `vfront/shared/*`：共享协议、engine 抽象、共享配置片段
- `vfront/adapter/*`：协议到 engine 的适配逻辑
- `vfront/protocol/*`：OpenAI 兼容协议模型

设计原则：

- `protocol-first`：协议模型集中放在 `vfront/protocol/*`
- `settings by ownership`：`@Settings("path")` 放在实际 `GetSettings("path")` 的所有者附近
- `frontend / engine split`：DI 扫描范围和部署边界保持一致
- `adapter isolation`：请求和响应的转换逻辑不污染运行时
- `engine abstraction`：frontend 只依赖共享 engine contract，不直接依赖 vLLM 细节

## 开发

### 安装开发依赖

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e ".[dev]"
pre-commit install
```

### 真实联调 / Smoke Check

优先做真实端到端验证，而不是维护一套大量 mock 的历史单测：

```bash
# 单进程 embedded
uv run vfront embedded -c vfront-qwen35-local.yaml

# 分离部署 frontend / engine
uv run vfront engine -c vfront-qwen35-engine.yaml
uv run vfront frontend -c vfront-qwen35-frontend.yaml

# 静态检查
ruff check vfront/
mypy vfront/
```

### 静态检查

```bash
ruff check vfront/
ruff format vfront/
mypy vfront/
```

## License

MIT
