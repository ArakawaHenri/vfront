"""HTTP engine client for remote engine workers."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass

import httpx

from vfront.protocol.engine_internal import (
    DecodeTokensRequest,
    DecodeTokensResponse,
    EncodeRequest,
    EncodeResponse,
    EngineHealthResponse,
    GenerateOutputChunk,
    GenerateRequest,
    InternalEncodeParams,
    InternalGenerateParams,
    LoadLoRARequest,
    LoRAListResponse,
    PreparedInfillPromptResponse,
    PrepareInfillRequest,
    TokenIdsResponse,
    TokenizeChatRequest,
    TokenizeTextRequest,
    UnloadLoRARequest,
)
from vfront.shared.engine.client import EngineClient
from vfront.shared.engine.errors import EngineInputError
from vfront.shared.engine.reasoning import ReasoningEffort
from vfront.shared.engine.types import (
    EmbeddingOutput,
    EncodeParams,
    EngineHealth,
    GenerateOutput,
    GenerateParams,
    LoRAAdapterInfo,
    PreparedInfillPrompt,
)


@dataclass(slots=True, frozen=True)
class HttpClientEndpoint:
    """Resolved HTTP client endpoint configuration."""

    base_url: str
    uds: str | None = None


def resolve_http_endpoint(endpoint: str) -> HttpClientEndpoint:
    """Resolve a configured endpoint into httpx base_url + optional UDS path."""
    if endpoint.startswith("unix://"):
        uds_path = endpoint.removeprefix("unix://")
        if not uds_path.startswith("/"):
            uds_path = f"/{uds_path}"
        if not uds_path:
            raise ValueError("unix endpoint must include a socket path")
        return HttpClientEndpoint(
            base_url="http://engine-worker",
            uds=uds_path,
        )
    return HttpClientEndpoint(base_url=endpoint.rstrip("/"))


class RemoteHttpEngineClient(EngineClient):
    """Engine client backed by a remote engine worker over HTTP."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str | None = None,
        timeout_seconds: float = 300.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        resolved_endpoint = resolve_http_endpoint(base_url)
        self._base_url = resolved_endpoint.base_url
        self._client = client or self._build_client(
            endpoint=resolved_endpoint,
            api_key=api_key,
            timeout_seconds=timeout_seconds,
        )
        self._owns_client = client is None
        self._runtime_model = ""
        self._max_model_len = 0

    @staticmethod
    def _build_headers(api_key: str | None) -> dict[str, str]:
        if api_key is None:
            return {}
        return {"Authorization": f"Bearer {api_key}"}

    @classmethod
    def _build_client(
        cls,
        *,
        endpoint: HttpClientEndpoint,
        api_key: str | None,
        timeout_seconds: float,
    ) -> httpx.AsyncClient:
        transport = None
        if endpoint.uds is not None:
            transport = httpx.AsyncHTTPTransport(uds=endpoint.uds)
        return httpx.AsyncClient(
            base_url=endpoint.base_url,
            timeout=timeout_seconds,
            headers=cls._build_headers(api_key),
            transport=transport,
        )

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def generate(
        self,
        prompt_token_ids: list[int],
        params: GenerateParams,
        request_id: str,
    ) -> AsyncIterator[GenerateOutput]:
        payload = GenerateRequest(
            prompt_token_ids=prompt_token_ids,
            params=InternalGenerateParams.from_engine(params),
            request_id=request_id,
        )
        async with self._client.stream(
            "POST",
            "/generate",
            json=payload.model_dump(),
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line:
                    continue
                yield GenerateOutputChunk.model_validate(json.loads(line)).to_engine()

    async def encode(
        self,
        prompt_token_ids: list[int],
        params: EncodeParams,
        request_id: str,
    ) -> EmbeddingOutput:
        response = await self._client.post(
            "/encode",
            json=EncodeRequest(
                prompt_token_ids=prompt_token_ids,
                params=InternalEncodeParams.from_engine(params),
                request_id=request_id,
            ).model_dump(),
        )
        response.raise_for_status()
        return EncodeResponse.model_validate(response.json()).to_engine()

    async def tokenize_chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        reasoning_effort: ReasoningEffort | None = None,
    ) -> list[int]:
        response = await self._client.post(
            "/tokenize/chat",
            json=TokenizeChatRequest(
                messages=messages,
                tools=tools,
                reasoning_effort=reasoning_effort,
            ).model_dump(),
        )
        self._raise_for_status_or_input_error(response)
        return TokenIdsResponse.model_validate(response.json()).token_ids

    async def tokenize_text(self, text: str) -> list[int]:
        response = await self._client.post(
            "/tokenize/text",
            json=TokenizeTextRequest(text=text).model_dump(),
        )
        self._raise_for_status_or_input_error(response)
        return TokenIdsResponse.model_validate(response.json()).token_ids

    async def prepare_infill(
        self,
        prefix: str,
        suffix: str,
    ) -> PreparedInfillPrompt:
        response = await self._client.post(
            "/tokenize/infill",
            json=PrepareInfillRequest(
                prefix=prefix,
                suffix=suffix,
            ).model_dump(),
        )
        self._raise_for_status_or_input_error(response)
        return PreparedInfillPromptResponse.model_validate(response.json()).to_engine()

    async def decode_tokens(self, token_ids: list[int]) -> str:
        response = await self._client.post(
            "/decode",
            json=DecodeTokensRequest(token_ids=token_ids).model_dump(),
        )
        response.raise_for_status()
        return DecodeTokensResponse.model_validate(response.json()).text

    async def health(self) -> EngineHealth:
        response = await self._client.get("/health")
        response.raise_for_status()
        health = EngineHealthResponse.model_validate(response.json()).to_engine()
        self._runtime_model = health.runtime_model
        self._max_model_len = health.max_model_len
        return health

    async def list_loras(self) -> list[LoRAAdapterInfo]:
        response = await self._client.get("/loras")
        response.raise_for_status()
        payload = LoRAListResponse.model_validate(response.json())
        return [item.to_engine() for item in payload.data]

    async def load_lora(self, *, name: str, path: str) -> None:
        response = await self._client.post(
            "/loras/load",
            json=LoadLoRARequest(name=name, path=path).model_dump(),
        )
        response.raise_for_status()

    async def unload_lora(self, *, name: str) -> None:
        response = await self._client.post(
            "/loras/unload",
            json=UnloadLoRARequest(name=name).model_dump(),
        )
        response.raise_for_status()

    @property
    def runtime_model(self) -> str:
        return self._runtime_model

    @property
    def max_model_len(self) -> int:
        return self._max_model_len

    @staticmethod
    def _raise_for_status_or_input_error(response: httpx.Response) -> None:
        if response.status_code == 400:
            try:
                payload = response.json()
            except ValueError:
                payload = None
            detail = (
                payload.get("detail")
                if isinstance(payload, dict) and isinstance(payload.get("detail"), str)
                else response.text
            )
            raise EngineInputError(detail or "Invalid engine input.")
        response.raise_for_status()
