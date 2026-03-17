"""Internal engine-worker HTTP routes."""

from __future__ import annotations

from fastapi import APIRouter, Header, HTTPException
from fastapiex.di import Inject
from fastapiex.settings import GetSettings
from starlette.responses import StreamingResponse

from vfront.engine.service.local import LocalEngineManager
from vfront.protocol.engine_internal import (
    DecodeTokensRequest,
    DecodeTokensResponse,
    EncodeRequest,
    EncodeResponse,
    EngineHealthResponse,
    GenerateOutputChunk,
    GenerateRequest,
    LoadLoRARequest,
    LoRAAdapterPayload,
    LoRAListResponse,
    PreparedInfillPromptResponse,
    PrepareInfillRequest,
    TokenIdsResponse,
    TokenizeChatRequest,
    TokenizeTextRequest,
    UnloadLoRARequest,
)

router = APIRouter()


def _require_worker_auth(authorization: str | None) -> None:
    settings = GetSettings("engine.api")
    if settings.api_key is None:
        return
    if authorization != f"Bearer {settings.api_key}":
        raise HTTPException(status_code=401, detail="Unauthorized")


@router.get("/health", response_model=EngineHealthResponse)
async def health(
    authorization: str | None = Header(default=None),
    backend: LocalEngineManager = Inject("engine_backend"),  # type: ignore[assignment]
) -> EngineHealthResponse:
    _require_worker_auth(authorization)
    return EngineHealthResponse.from_engine(await backend.health())


@router.get("/loras", response_model=LoRAListResponse)
async def list_loras(
    authorization: str | None = Header(default=None),
    backend: LocalEngineManager = Inject("engine_backend"),  # type: ignore[assignment]
) -> LoRAListResponse:
    _require_worker_auth(authorization)
    adapters = await backend.list_loras()
    return LoRAListResponse(data=[LoRAAdapterPayload.from_engine(item) for item in adapters])


@router.post("/loras/load", status_code=204)
async def load_lora(
    request: LoadLoRARequest,
    authorization: str | None = Header(default=None),
    backend: LocalEngineManager = Inject("engine_backend"),  # type: ignore[assignment]
) -> None:
    _require_worker_auth(authorization)
    await backend.load_lora(name=request.name, path=request.path)


@router.post("/loras/unload", status_code=204)
async def unload_lora(
    request: UnloadLoRARequest,
    authorization: str | None = Header(default=None),
    backend: LocalEngineManager = Inject("engine_backend"),  # type: ignore[assignment]
) -> None:
    _require_worker_auth(authorization)
    await backend.unload_lora(name=request.name)


@router.post("/tokenize/chat", response_model=TokenIdsResponse)
async def tokenize_chat(
    request: TokenizeChatRequest,
    authorization: str | None = Header(default=None),
    backend: LocalEngineManager = Inject("engine_backend"),  # type: ignore[assignment]
) -> TokenIdsResponse:
    _require_worker_auth(authorization)
    token_ids = await backend.tokenize_chat(
        request.messages,
        tools=request.tools,
        reasoning_effort=request.reasoning_effort,
    )
    return TokenIdsResponse(token_ids=token_ids)


@router.post("/tokenize/text", response_model=TokenIdsResponse)
async def tokenize_text(
    request: TokenizeTextRequest,
    authorization: str | None = Header(default=None),
    backend: LocalEngineManager = Inject("engine_backend"),  # type: ignore[assignment]
) -> TokenIdsResponse:
    _require_worker_auth(authorization)
    return TokenIdsResponse(token_ids=await backend.tokenize_text(request.text))


@router.post("/tokenize/infill", response_model=PreparedInfillPromptResponse)
async def prepare_infill(
    request: PrepareInfillRequest,
    authorization: str | None = Header(default=None),
    backend: LocalEngineManager = Inject("engine_backend"),  # type: ignore[assignment]
) -> PreparedInfillPromptResponse:
    _require_worker_auth(authorization)
    prompt = await backend.prepare_infill(
        request.prefix,
        request.suffix,
    )
    return PreparedInfillPromptResponse.from_engine(prompt)


@router.post("/decode", response_model=DecodeTokensResponse)
async def decode_tokens(
    request: DecodeTokensRequest,
    authorization: str | None = Header(default=None),
    backend: LocalEngineManager = Inject("engine_backend"),  # type: ignore[assignment]
) -> DecodeTokensResponse:
    _require_worker_auth(authorization)
    return DecodeTokensResponse(text=await backend.decode_tokens(request.token_ids))


@router.post("/encode", response_model=EncodeResponse)
async def encode(
    request: EncodeRequest,
    authorization: str | None = Header(default=None),
    backend: LocalEngineManager = Inject("engine_backend"),  # type: ignore[assignment]
) -> EncodeResponse:
    _require_worker_auth(authorization)
    output = await backend.encode(
        request.prompt_token_ids,
        request.params.to_engine(),
        request.request_id,
    )
    return EncodeResponse.from_engine(output)


@router.post("/generate")
async def generate(
    request: GenerateRequest,
    authorization: str | None = Header(default=None),
    backend: LocalEngineManager = Inject("engine_backend"),  # type: ignore[assignment]
) -> StreamingResponse:
    _require_worker_auth(authorization)

    async def _stream():
        async for output in backend.generate(
            request.prompt_token_ids,
            request.params.to_engine(),
            request.request_id,
        ):
            yield GenerateOutputChunk.from_engine(output).model_dump_json() + "\n"

    return StreamingResponse(_stream(), media_type="application/x-ndjson")
