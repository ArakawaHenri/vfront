"""EngineClient protocol — the contract between routers and inference engines.

Any engine backend (vLLM, SGLang, TensorRT-LLM, etc.) must implement this protocol.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol, runtime_checkable

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


@runtime_checkable
class EngineClient(Protocol):
    """Protocol that all engine backends must satisfy."""

    def generate(
        self,
        prompt_token_ids: list[int],
        params: GenerateParams,
        request_id: str,
    ) -> AsyncIterator[GenerateOutput]: ...

    async def encode(
        self,
        prompt_token_ids: list[int],
        params: EncodeParams,
        request_id: str,
    ) -> EmbeddingOutput: ...

    async def tokenize_chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        reasoning_effort: ReasoningEffort | None = None,
    ) -> list[int]: ...

    async def tokenize_text(self, text: str) -> list[int]: ...

    async def prepare_infill(
        self,
        prefix: str,
        suffix: str,
    ) -> PreparedInfillPrompt: ...

    async def decode_tokens(self, token_ids: list[int]) -> str: ...

    async def health(self) -> EngineHealth: ...

    async def list_loras(self) -> list[LoRAAdapterInfo]: ...

    async def load_lora(self, *, name: str, path: str) -> None: ...

    async def unload_lora(self, *, name: str) -> None: ...

    @property
    def runtime_model(self) -> str: ...

    @property
    def max_model_len(self) -> int: ...
