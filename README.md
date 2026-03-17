# vfront

[简体中文](./README.zh.md)

100% OpenAI-compatible API server powered by [vLLM](https://github.com/vllm-project/vllm).

Drop-in replacement for the OpenAI API — every endpoint, every field, every streaming event matches the official SDK behavior. Point your `openai` client at this server and it just works.

## Features

- **OpenAI-Compatible Surface** — Chat Completions, Completions, Embeddings, Models, Responses, Files, and Batches with stored-object lifecycle where implemented
- **Streaming** — SSE streaming with correct `[DONE]` sentinel for Chat/Completions, full Responses event protocol
- **Responses API** — Streaming events, background mode, multi-turn, cancellation, context truncation, and `input_items` retrieval
- **Multi-LoRA** — Hot-swap LoRA adapters per request via the `model` field
- **MCP Tools** — Server-side tool execution with streaming agentic loops (stdio + Streamable HTTP transports)
- **Batch Processing** — Async batch inference for chat, completions, embeddings, and responses
- **Persistence** — LMDB-backed store for responses, batches, and file metadata with TTL expiry
- **Type Safety** — Pydantic v2 protocol models with 100% SDK field coverage, mypy clean (0 errors)
- **High Performance** — Granian (Rust ASGI server) + uvloop + vLLM PagedAttention

## Cloud-Only Limitations

This server targets **local / self-hosted inference** powered by vLLM. Features that depend on OpenAI's proprietary cloud infrastructure are **accepted for SDK compatibility but silently ignored**. The SDK will never throw a validation error — your code just works, and cloud-only fields have no effect.

### Endpoints Not Implemented

These OpenAI API endpoint families require cloud-specific backends and are **not served**:

| Endpoint Family | Reason |
|-----------------|--------|
| `/v1/audio/*` (speech, transcriptions, translations) | Requires Whisper / TTS cloud service |
| `/v1/images/*` (generations, edits, variations) | Requires DALL·E cloud service |
| `/v1/moderations` | Requires OpenAI moderation classifier |
| `/v1/fine_tuning/*` | Cloud-managed training pipeline |
| `/v1/vector_stores/*` | Cloud-managed vector database |
| `/v1/assistants/*`, `/v1/threads/*` | Deprecated Assistants API (cloud-only stateful runtime) |
| `/v1/evals/*` | Cloud evaluation pipeline |
| `/v1/organization/*`, `/v1/invites/*` | Cloud account management |

### Chat Completions — Ignored Parameters

| Parameter | What It Does on OpenAI Cloud |
|-----------|------------------------------|
| `audio` | Real-time audio input/output modality |
| `modalities` | Multi-modal output selection (`["text", "audio"]`) |
| `prediction` | Predicted output / speculative decoding hint |
| `web_search_options` | Built-in web search grounding |
| `prompt_cache_key` / `prompt_cache_retention` | Server-side prompt caching |
| `safety_identifier` | Cloud safety filter profile |
| `verbosity` | Response verbosity control |

### Responses API — Ignored Parameters

| Parameter | What It Does on OpenAI Cloud |
|-----------|------------------------------|
| `include` | Selectively include extra fields (e.g. logprobs) |
| `prompt` | Advanced prompt template management |
| `context_management` | Cloud context window strategy |
| `conversation` | Server-side conversation ID tracking |
| `prompt_cache_key` / `prompt_cache_retention` | Server-side prompt caching |
| `safety_identifier` | Cloud safety filter profile |

### Responses API — Unsupported Built-In Tools

OpenAI's Responses API offers several built-in tools that run on cloud infrastructure. These are **not accepted** — use [MCP tools](#mcp) for equivalent functionality:

| Built-In Tool | Cloud Behavior | Local Alternative |
|---------------|----------------|-------------------|
| `web_search` | OpenAI-managed web search | MCP web-search server (e.g. Brave) |
| `file_search` | Cloud vector store retrieval | MCP filesystem server |
| `code_interpreter` | Cloud Python sandbox | MCP code execution server |
| `computer_use` | Cloud computer interaction | MCP desktop automation server |
| `image_generation` | DALL·E integration | MCP image generation server |

> **Tip:** MCP tools provide a fully open, extensible alternative. See the [MCP configuration](#mcp) section.

## Quick Start

### Prerequisites

- Python ≥ 3.12
- CUDA-capable GPU (for vLLM inference)
- [uv](https://github.com/astral-sh/uv) (recommended package manager)

### Installation

```bash
git clone https://github.com/your-org/vfront.git
cd vfront

uv venv --python 3.12
source .venv/bin/activate
uv pip install -e .
```

### Configuration

Pick the deployment mode first, then copy the matching example:

```bash
cp embedded.yaml.example embedded.yaml
# or
cp frontend.yaml.example frontend.yaml
# or
cp engine.yaml.example engine.yaml
```

Three modes are supported:

- `embedded`: single-node deployment; frontend, storage, jobs, and either an in-process runtime or managed local engine workers live together
- `frontend`: control-plane only; connects to remote engine workers over HTTP or HTTP-over-UDS
- `engine`: dedicated vLLM worker process serving one configured base model

The CLI `-c` flag is the recommended way to point to a config file. Internally,
`vfront` resolves the path and exports it to `fastapiex.settings`.

### Run

```bash
# Single-node / local development
uv run vfront embedded -c embedded.yaml

# Control-plane only
uv run vfront frontend -c frontend.yaml

# Dedicated engine worker
uv run vfront engine -c engine.yaml
```

The server starts on `http://0.0.0.0:8000` by default.

### Use

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="any")

# Chat Completions
response = client.chat.completions.create(
    model="Qwen/Qwen2.5-7B-Instruct",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)

# Streaming
for chunk in client.chat.completions.create(
    model="Qwen/Qwen2.5-7B-Instruct",
    messages=[{"role": "user", "content": "Tell me a story"}],
    stream=True,
):
    print(chunk.choices[0].delta.content or "", end="")

# Responses API
resp = client.responses.create(
    model="Qwen/Qwen2.5-7B-Instruct",
    input="What is 2+2?",
)
print(resp.output_text)

# Embeddings
emb = client.embeddings.create(
    model="Qwen/Qwen2.5-7B-Instruct",
    input="Hello world",
)
print(len(emb.data[0].embedding))
```

## API Endpoints

### Core

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/chat/completions` | Chat Completions (streaming, tools, n>1, logit_bias, logprobs) |
| `POST` | `/v1/completions` | Legacy Completions (streaming, suffix, best_of) |
| `POST` | `/v1/embeddings` | Embeddings (float / base64, dimensions) |
| `GET` | `/v1/models` | List models (base + LoRA adapters) |
| `GET` | `/v1/models/{model}` | Get model details |
| `DELETE` | `/v1/models/{model}` | Runtime model catalog mutation is currently unsupported |

### Stored Chat Completions

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/v1/chat/completions/{id}` | Retrieve stored Chat Completion |
| `POST` | `/v1/chat/completions/{id}` | Update stored Chat Completion metadata |
| `DELETE` | `/v1/chat/completions/{id}` | Delete stored Chat Completion |
| `GET` | `/v1/chat/completions/{id}/messages` | List stored Chat Completion messages |

### Responses API

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/responses` | Create Response (streaming events, background mode, tools) |
| `GET` | `/v1/responses/{id}` | Retrieve stored Response |
| `GET` | `/v1/responses/{id}/input_items` | List the items used to generate a stored Response |
| `DELETE` | `/v1/responses/{id}` | Delete Response |
| `POST` | `/v1/responses/{id}/cancel` | Cancel in-progress Response |

`GET /v1/responses` is intentionally **not** exposed because it is not part of the
current local OpenAI-compatible contract in `openapi.yaml`.

### Files & Batches

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/files` | Upload file (multipart) |
| `GET` | `/v1/files` | List files (with pagination) |
| `GET` | `/v1/files/{id}` | Get file metadata |
| `GET` | `/v1/files/{id}/content` | Download file content |
| `DELETE` | `/v1/files/{id}` | Delete file |
| `POST` | `/v1/batches` | Create batch job |
| `GET` | `/v1/batches/{id}` | Get batch status |
| `GET` | `/v1/batches` | List batches |
| `POST` | `/v1/batches/{id}/cancel` | Cancel batch |

### Health

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Liveness check |
| `GET` | `/ready` | Readiness check (engine loaded) |

## Deployment Modes

### Embedded

`embedded` is the simplest way to run vfront. It uses one config file and keeps
the OpenAI-compatible frontend, background jobs, persistence, and local vLLM
runtime in one deployment unit.

Use [embedded.yaml.example](./embedded.yaml.example) as the starting point.

Behavior by transport:

- `frontend.routing.transport=local`: keeps vLLM in-process; requires `frontend.server.workers=1`
- `frontend.routing.transport=managed`: spawns dedicated local engine worker processes over UDS; frontend workers can scale independently
- `frontend.routing.transport=remote`: valid but usually less useful than plain `frontend`

### Frontend

`frontend` is the control-plane. It serves the OpenAI-compatible API, owns
files, batches, responses, MCP integration, and job execution, and routes
inference to one or more engine workers.

Use [frontend.yaml.example](./frontend.yaml.example) as the starting point.

Relevant sections:

- `frontend.server`
- `frontend.app`
- `frontend.routing`
- `adapters`
- `frontend.store`
- `frontend.file_storage`
- `frontend.jobs`
- `frontend.mcp`

Transport notes:

- `routing.backends[].base_url` accepts normal HTTP endpoints such as `http://10.0.0.11:9001`
- it also accepts UDS endpoints such as `unix:///tmp/vfront/qwen-7b.sock`
- `frontend` does not start or supervise engine workers; it only routes to configured backends

### Engine

`engine` is a dedicated vLLM worker. It only cares about serving one configured
runtime model plus any attached LoRA adapters.

Use [engine.yaml.example](./engine.yaml.example) as the starting point.

Relevant sections:

- `engine.server`
- `engine.app`
- `engine.runtime`
- `adapters`
- `engine.api`

Listener notes:

- standalone `engine` mode listens on `engine.server.uds` when configured; otherwise it falls back to `engine.server.host` and `engine.server.port`
- managed workers spawned by `embedded` listen on UDS internally
- when `embedded` spawns managed workers, the parent process injects its own UDS path and that takes precedence

## Configuration Notes

### `frontend.app` / `engine.app`

- `frontend.app.api_key` protects `/v1/*`
- environment overrides follow `fastapiex.settings` path keys, so `frontend.app.api_key` maps to `FRONTEND__APP__API_KEY`
- `cors_origins` should be set explicitly in production
- `log_dir` controls file logging

### `frontend.server` / `engine.server`

- `host` and `port` control TCP listening
- `uds` is a local Unix domain socket path, not a URL
- when `uds` is configured, it takes precedence over `host` and `port`

### `frontend.routing`

- `transport=local` is only for `embedded`
- `transport=remote` is for `frontend`
- `transport=managed` lets `embedded` spawn local engine workers over UDS
- `models` defines the public base-model topology exposed by `/v1/models`
- frontend model entries intentionally do not carry runtime facts such as `max_model_len`; those come from engine health
- `backends` maps `engine_key -> endpoint` when `transport=remote`
- backend `base_url` may be `http://...` or `unix:///path/to/worker.sock`
- when `embedded` uses `transport=managed`, backend UDS endpoints are derived automatically from `engine.managed_workers.socket_dir`

### `engine.runtime`

- `model` is the Hugging Face model ID or local path
- `served_model_name` is the public engine key
- `tensor_parallel_size`, `pipeline_parallel_size`, `dtype`, and `gpu_memory_utilization` are passed to vLLM

### `adapters`

- `modules[].name` is the public LoRA model id
- `modules[].base_model` binds the adapter to one base model
- use the adapter `name` directly in OpenAI-compatible requests as `model`

### `frontend.jobs`

- background Responses and Batches run through the private runner
- `run_embedded=true` is only meaningful in `embedded`
- `frontend` can keep jobs enabled while using a private internal runner process
- `embedded + local` keeps the runner in-process, so API workers must stay at `1`
- `embedded + managed` and `frontend` run the runner in a private subprocess, so frontend workers can scale independently

### `engine.managed_workers`

- only used by `embedded` when `frontend.routing.transport=managed`
- `api_key` is the frontend-side credential for talking to managed local engine workers
- `socket_dir` controls where embedded mode places Unix domain sockets for local engine workers

### `engine.api`

- only used by `engine` workers, or by `embedded` when it spawns managed workers
- `api_key` protects the internal engine API
- `log_access` controls access logging on the engine worker listener

## Transport Matrix

| Mode | Engine placement | Frontend to engine transport | Frontend workers |
|------|------------------|------------------------------|------------------|
| `embedded` + `frontend.routing.transport=local` | in-process | direct Python calls | must be `1` |
| `embedded` + `frontend.routing.transport=managed` | local child processes | HTTP over UDS | can be `>1` |
| `frontend` + `frontend.routing.transport=remote` | external engine processes | HTTP or HTTP over UDS | can be `>1` |
| `engine` | dedicated worker | listens on `engine.server.uds` when set, otherwise `engine.server.host`/`engine.server.port` | always `1` |

## Architecture

```text
client / SDK
    -> vfront frontend
        -> routing + storage + jobs + MCP
        -> engine router
            -> vfront engine worker(s)
                -> vLLM runtime
```

Package layout follows the same boundary:

- `vfront/frontend/*`: API, middleware, storage, jobs, router, MCP
- `vfront/engine/*`: local runtime and worker server
- `vfront/shared/*`: shared protocol, engine abstractions, config fragments
- `vfront/adapter/*`: protocol-to-engine translation helpers
- `vfront/protocol/*`: OpenAI-compatible Pydantic models

Design principles:

- `protocol-first`: OpenAI-compatible shapes live in `vfront/protocol/*`
- `settings by ownership`: each `@Settings("path")` lives beside the code that calls `GetSettings("path")`
- `frontend / engine split`: DI scanning follows deployment boundaries instead of importing the whole app
- `adapter isolation`: request/response translation stays in `vfront/adapter/*`
- `engine abstraction`: frontend code only depends on shared engine contracts, not vLLM internals

## Development

### Setup

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e ".[dev]"
pre-commit install
```

### Smoke Checks

Prefer real end-to-end checks over mocked unit suites. The minimum supported
verification flows are:

```bash
# Embedded local stack
uv run vfront embedded -c vfront-qwen35-local.yaml

# Split frontend / engine
uv run vfront engine -c vfront-qwen35-engine.yaml
uv run vfront frontend -c vfront-qwen35-frontend.yaml

# Static checks
ruff check vfront/
mypy vfront/
```

### Linting

```bash
# Pre-commit hooks (staged files)
pre-commit run

# Run on all files
pre-commit run --all-files

# Individually
ruff check vfront/              # Lint
ruff format vfront/             # Format
mypy vfront/                    # Type check (0 errors)
```

## License

MIT
