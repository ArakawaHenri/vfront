"""Local vLLM-backed engine manager and clients.

This module owns the in-process vLLM implementation. It is intentionally
separate from the higher-level router so future remote engine workers can
implement the same EngineClient contract without dragging vLLM imports into
the control plane.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from fastapiex.di import BaseService, Service
from vllm import AsyncLLMEngine, SamplingParams
from vllm.lora.request import LoRARequest

from vfront.engine.service.generation_orchestrator import LocalEngineOrchestrator
from vfront.engine.service.health_reporter import LocalEngineHealthReporter
from vfront.engine.service.lifecycle_coordinator import LocalEngineLifecycleCoordinator
from vfront.engine.service.lora_manager import LocalEngineAdapterManager
from vfront.engine.service.runtime_factory import LocalEngineRuntimeFactory
from vfront.engine.service.state_access import LocalEngineStateAccess
from vfront.engine.service.tokenizer_manager import LocalEngineTokenizerManager
from vfront.engine.service.vllm_adapter import (
    build_vllm_prompt,
    to_generate_output,
    to_sampling_params,
)
from vfront.shared.engine.client import EngineClient
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

if TYPE_CHECKING:
    from vllm.transformers_utils.tokenizer import AnyTokenizer

    from vfront.shared.config.adapters import AdapterSettings
    from vfront.shared.config.models import ReasoningConfig, RuntimeModelConfig

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class LocalEngineRuntimeState:
    """Runtime state for a single vLLM runtime model instance."""

    engine: AsyncLLMEngine | Any
    runtime_model: str
    weights: str
    max_model_len: int
    tokenizer: AnyTokenizer | Any | None = None
    tokenizer_id: str | None = None
    quantization: str | None = None
    dtype: str = "auto"
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    tokenizer_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    lora_requests: dict[str, LoRARequest] = field(default_factory=dict)
    tool_parser: str | None = None
    reasoning: ReasoningConfig | None = None
    reasoning_parser: str | None = None
    infill: Any | None = None
    fim: bool | None = None


class LocalEngineClient:
    """Engine client proxy bound to a specific local runtime model."""

    def __init__(self, backend: LocalEngineManager, runtime_model: str) -> None:
        self._backend = backend
        self._runtime_model = runtime_model

    async def generate(
        self,
        prompt_token_ids: list[int],
        params: GenerateParams,
        request_id: str,
    ) -> AsyncIterator[GenerateOutput]:
        async for output in self._backend._generate_on_engine(
            self._runtime_model,
            prompt_token_ids,
            params,
            request_id,
        ):
            yield output

    async def encode(
        self,
        prompt_token_ids: list[int],
        params: EncodeParams,
        request_id: str,
    ) -> EmbeddingOutput:
        return await self._backend._encode_on_engine(
            self._runtime_model,
            prompt_token_ids,
            params,
            request_id,
        )

    async def tokenize_chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        reasoning_effort: ReasoningEffort | None = None,
    ) -> list[int]:
        return await self._backend._tokenize_chat_on_engine(
            self._runtime_model,
            messages,
            tools,
            reasoning_effort,
        )

    async def tokenize_text(self, text: str) -> list[int]:
        return await self._backend._tokenize_text_on_engine(self._runtime_model, text)

    async def prepare_infill(
        self,
        prefix: str,
        suffix: str,
    ) -> PreparedInfillPrompt:
        return await self._backend._prepare_infill_on_engine(
            self._runtime_model,
            prefix,
            suffix,
        )

    async def decode_tokens(self, token_ids: list[int]) -> str:
        return await self._backend._decode_tokens_on_engine(self._runtime_model, token_ids)

    async def health(self) -> EngineHealth:
        return self._backend._health_for_engine(self._runtime_model)

    async def list_loras(self) -> list[LoRAAdapterInfo]:
        return self._backend._list_loras_for_engine(self._runtime_model)

    async def load_lora(self, *, name: str, path: str) -> None:
        await self._backend._load_lora_on_engine(self._runtime_model, name=name, path=path)

    async def unload_lora(self, *, name: str) -> None:
        await self._backend._unload_lora_on_engine(self._runtime_model, name=name)

    @property
    def runtime_model(self) -> str:
        return self._backend._get_runtime_state(self._runtime_model).runtime_model

    @property
    def max_model_len(self) -> int:
        return self._backend._get_runtime_state(self._runtime_model).max_model_len


@Service("engine_backend", eager=True)
class LocalEngineManager(BaseService):
    """Wrap multiple in-process vLLM engines behind the EngineClient contract."""

    def __init__(self) -> None:
        self._engine: AsyncLLMEngine | None = None
        self._tokenizer: AnyTokenizer | None = None
        self._tokenizer_lock = asyncio.Lock()
        self._runtime_model: str = ""
        self._max_model_len: int = 0
        self._lora_requests: dict[str, LoRARequest] = {}
        self._runtimes: dict[str, LocalEngineRuntimeState] = {}
        self._state_access = LocalEngineStateAccess(
            get_runtimes=lambda: self._runtimes,
            get_primary_engine=lambda: self._engine,
            get_primary_tokenizer=lambda: self._tokenizer,
            get_primary_runtime_model=lambda: self._runtime_model,
            get_primary_max_model_len=lambda: self._max_model_len,
            get_primary_lora_requests=lambda: self._lora_requests,
            set_primary_engine=self._set_primary_engine,
            set_primary_tokenizer=self._set_primary_tokenizer,
            set_primary_runtime_model=self._set_primary_runtime_model,
            set_primary_max_model_len=self._set_primary_max_model_len,
            set_primary_lora_requests=self._set_primary_lora_requests,
            create_engine_client=lambda runtime_model: self
            if not runtime_model or runtime_model == self._runtime_model
            else LocalEngineClient(self, runtime_model),
        )
        self._adapter_manager = LocalEngineAdapterManager(
            get_runtime_state=self._get_runtime_state,
            get_primary_runtime_model=lambda: self._runtime_model,
            get_primary_lora_requests=lambda: self._lora_requests,
            set_primary_lora_requests=self._set_primary_lora_requests,
        )
        self._health_reporter = LocalEngineHealthReporter(
            get_runtime_state=lambda runtime_model: self._get_runtime_state(runtime_model),
            has_runtime_states=lambda: bool(self._runtimes),
            get_primary_runtime_model=lambda: self._runtime_model,
            get_primary_max_model_len=lambda: self._max_model_len,
            get_primary_tokenizer=lambda: self._tokenizer,
            set_primary_tokenizer=self._set_primary_tokenizer,
            get_primary_lora_requests=lambda: self._lora_requests,
        )
        self._tokenizer_manager = LocalEngineTokenizerManager(
            get_runtime_state=lambda runtime_model: self._get_runtime_state(runtime_model),
            has_runtime_states=lambda: bool(self._runtimes),
            get_engine=self._get_engine,
            get_primary_runtime_model=lambda: self._runtime_model,
            get_primary_tokenizer=lambda: self._tokenizer,
            get_primary_tokenizer_lock=lambda: self._tokenizer_lock,
            set_primary_tokenizer=self._set_primary_tokenizer,
        )
        self._generation_orchestrator = LocalEngineOrchestrator(
            get_runtime_state=lambda runtime_model: self._get_runtime_state(runtime_model),
            has_runtime_states=lambda: bool(self._runtimes),
            get_engine=self._get_engine,
            get_primary_runtime_model=lambda: self._runtime_model,
            get_tokenizer_for_engine=self._get_tokenizer_for_engine,
            resolve_lora_for_runtime=self._resolve_lora_for_runtime,
            resolve_lora=self._resolve_lora,
            to_sampling_params=self._to_sampling_params,
            to_generate_output=self._to_generate_output,
            build_vllm_prompt=self._build_vllm_prompt,
        )
        self._runtime_factory = LocalEngineRuntimeFactory()
        self._lifecycle_coordinator = LocalEngineLifecycleCoordinator(
            get_runtimes=lambda: self._runtimes,
            get_primary_engine=lambda: self._engine,
            create_runtime_state=self._create_runtime_state,
            sync_primary_runtime=self._sync_primary_runtime,
            reset_primary_state=self._reset_primary_state,
        )

    @classmethod
    async def create(cls) -> LocalEngineManager:
        instance = cls()
        await instance.start()
        return instance

    @classmethod
    async def destroy(cls, instance: LocalEngineManager) -> None:
        await instance.shutdown()

    async def start(self) -> None:
        await self._lifecycle_coordinator.start()

    async def shutdown(self) -> None:
        await self._lifecycle_coordinator.shutdown()

    @property
    def runtime_model(self) -> str:
        return self._runtime_model

    @property
    def max_model_len(self) -> int:
        return self._max_model_len

    @property
    def runtime_models(self) -> list[str]:
        if self._runtimes:
            return list(self._runtimes)
        return [self._runtime_model] if self._runtime_model else []

    @property
    def lora_names(self) -> list[str]:
        return self._adapter_manager.names()

    def has_loaded_engines(self) -> bool:
        return self._state_access.has_loaded_engines()

    def for_runtime_model(self, runtime_model: str) -> EngineClient:
        return self._state_access.for_runtime_model(runtime_model)

    async def generate(
        self,
        prompt_token_ids: list[int],
        params: GenerateParams,
        request_id: str,
    ) -> AsyncIterator[GenerateOutput]:
        async for output in self._generate_on_engine(
            self._runtime_model,
            prompt_token_ids,
            params,
            request_id,
        ):
            yield output

    async def encode(
        self,
        prompt_token_ids: list[int],
        params: EncodeParams,
        request_id: str,
    ) -> EmbeddingOutput:
        return await self._encode_on_engine(
            self._runtime_model,
            prompt_token_ids,
            params,
            request_id,
        )

    async def tokenize_chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        reasoning_effort: ReasoningEffort | None = None,
    ) -> list[int]:
        return await self._tokenize_chat_on_engine(
            self._runtime_model,
            messages,
            tools,
            reasoning_effort,
        )

    async def tokenize_text(self, text: str) -> list[int]:
        return await self._tokenize_text_on_engine(self._runtime_model, text)

    async def prepare_infill(
        self,
        prefix: str,
        suffix: str,
    ) -> PreparedInfillPrompt:
        return await self._prepare_infill_on_engine(
            self._runtime_model,
            prefix,
            suffix,
        )

    async def decode_tokens(self, token_ids: list[int]) -> str:
        return await self._decode_tokens_on_engine(self._runtime_model, token_ids)

    async def health(self) -> EngineHealth:
        if not self.has_loaded_engines():
            return EngineHealth(
                ready=False,
                runtime_model=self._runtime_model,
                max_model_len=0,
                loaded_loras=(),
                transport="local",
            )
        return self._health_for_engine(self._runtime_model)

    async def list_loras(self) -> list[LoRAAdapterInfo]:
        return self._adapter_manager.list_loras(self._runtime_model)

    async def load_lora(self, *, name: str, path: str) -> None:
        await self._adapter_manager.load_lora(self._runtime_model, name=name, path=path)

    async def unload_lora(self, *, name: str) -> None:
        await self._adapter_manager.unload_lora(self._runtime_model, name=name)

    def _create_runtime_state(
        self,
        *,
        runtime_model: RuntimeModelConfig,
        lora_settings: AdapterSettings,
        default_fim_policy: bool | None,
    ) -> LocalEngineRuntimeState:
        return self._runtime_factory.create_runtime_state(
            runtime_model=runtime_model,
            lora_settings=lora_settings,
            default_fim_policy=default_fim_policy,
        )

    def _sync_primary_runtime(self, runtime: LocalEngineRuntimeState) -> None:
        self._state_access.sync_primary_runtime(runtime)

    def _reset_primary_state(self) -> None:
        self._state_access.reset_primary_state()

    def _get_runtime_state(
        self,
        runtime_model: str | None = None,
    ) -> LocalEngineRuntimeState:
        return self._state_access.get_runtime_state(runtime_model)

    def _get_engine(self) -> AsyncLLMEngine:
        return self._state_access.get_engine()

    async def _generate_on_engine(
        self,
        runtime_model: str,
        prompt_token_ids: list[int],
        params: GenerateParams,
        request_id: str,
    ) -> AsyncIterator[GenerateOutput]:
        async for output in self._generation_orchestrator.generate(
            runtime_model,
            prompt_token_ids,
            params,
            request_id,
        ):
            yield output

    async def _encode_on_engine(
        self,
        runtime_model: str,
        prompt_token_ids: list[int],
        params: EncodeParams,
        request_id: str,
    ) -> EmbeddingOutput:
        return await self._generation_orchestrator.encode(
            runtime_model,
            prompt_token_ids,
            params,
            request_id,
        )

    async def _tokenize_chat_on_engine(
        self,
        runtime_model: str,
        messages: list[dict],
        tools: list[dict] | None = None,
        reasoning_effort: ReasoningEffort | None = None,
    ) -> list[int]:
        return await self._tokenizer_manager.tokenize_chat_on_engine(
            runtime_model,
            messages,
            tools,
            reasoning_effort,
        )

    async def _tokenize_text_on_engine(self, runtime_model: str, text: str) -> list[int]:
        return await self._tokenizer_manager.tokenize_text_on_engine(runtime_model, text)

    async def _decode_tokens_on_engine(
        self,
        runtime_model: str,
        token_ids: list[int],
    ) -> str:
        return await self._tokenizer_manager.decode_tokens_on_engine(runtime_model, token_ids)

    async def _prepare_infill_on_engine(
        self,
        runtime_model: str,
        prefix: str,
        suffix: str,
    ) -> PreparedInfillPrompt:
        return await self._tokenizer_manager.prepare_infill_on_engine(
            runtime_model,
            prefix,
            suffix,
        )

    def _resolve_lora(self, lora_name: str) -> LoRARequest:
        return self._adapter_manager.resolve_lora(lora_name)

    def _resolve_lora_for_runtime(
        self,
        runtime: LocalEngineRuntimeState,
        lora_name: str | None,
    ) -> LoRARequest | None:
        return self._adapter_manager.resolve_lora_for_runtime(runtime, lora_name)

    def _health_for_engine(self, runtime_model: str) -> EngineHealth:
        return self._health_reporter.compute_health(runtime_model)

    def _list_loras_for_engine(self, runtime_model: str) -> list[LoRAAdapterInfo]:
        return self._adapter_manager.list_loras(runtime_model)

    async def _load_lora_on_engine(
        self,
        runtime_model: str,
        *,
        name: str,
        path: str,
    ) -> None:
        await self._adapter_manager.load_lora(runtime_model, name=name, path=path)

    async def _unload_lora_on_engine(self, runtime_model: str, *, name: str) -> None:
        await self._adapter_manager.unload_lora(runtime_model, name=name)

    def _set_primary_lora_requests(self, requests: dict[str, LoRARequest]) -> None:
        self._lora_requests = requests

    def _set_primary_engine(self, engine: AsyncLLMEngine | None) -> None:
        self._engine = engine

    def _set_primary_tokenizer(self, tokenizer: AnyTokenizer | None) -> None:
        self._tokenizer = tokenizer

    def _set_primary_runtime_model(self, runtime_model: str) -> None:
        self._runtime_model = runtime_model

    def _set_primary_max_model_len(self, max_model_len: int) -> None:
        self._max_model_len = max_model_len

    async def _get_tokenizer(self) -> AnyTokenizer:
        return await self._tokenizer_manager.get_tokenizer()

    async def _get_tokenizer_for_engine(self, runtime_model: str) -> AnyTokenizer:
        return await self._tokenizer_manager.get_tokenizer_for_engine(runtime_model)

    @staticmethod
    def _to_sampling_params(params: GenerateParams) -> SamplingParams:
        return to_sampling_params(params)

    @staticmethod
    def _to_generate_output(vllm_output: Any, request_id: str) -> GenerateOutput:
        return to_generate_output(vllm_output, request_id)

    @staticmethod
    def _build_vllm_prompt(
        engine: Any,
        prompt_token_ids: list[int],
        *,
        multimodal_data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return build_vllm_prompt(
            engine,
            prompt_token_ids,
            multimodal_data=multimodal_data,
        )
