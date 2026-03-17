"""Primary-state synchronization and runtime lookup helpers for local engines."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from vllm.lora.request import LoRARequest

from vfront.shared.engine.client import EngineClient

if TYPE_CHECKING:
    from vllm import AsyncLLMEngine
    from vllm.transformers_utils.tokenizer import AnyTokenizer

    from vfront.engine.service.local import LocalEngineRuntimeState

GetRuntimes = Callable[[], dict[str, "LocalEngineRuntimeState"]]
GetPrimaryEngine = Callable[[], "AsyncLLMEngine | None"]
GetPrimaryTokenizer = Callable[[], "AnyTokenizer | None"]
GetPrimaryRuntimeModel = Callable[[], str]
GetPrimaryMaxModelLen = Callable[[], int]
GetPrimaryLoRARequests = Callable[[], dict[str, LoRARequest]]
SetPrimaryEngine = Callable[["AsyncLLMEngine | None"], None]
SetPrimaryTokenizer = Callable[["AnyTokenizer | None"], None]
SetPrimaryRuntimeModel = Callable[[str], None]
SetPrimaryMaxModelLen = Callable[[int], None]
SetPrimaryLoRARequests = Callable[[dict[str, LoRARequest]], None]
CreateEngineClient = Callable[[str], EngineClient]


class LocalEngineStateAccess:
    """Centralize runtime lookup and primary-state synchronization."""

    def __init__(
        self,
        *,
        get_runtimes: GetRuntimes,
        get_primary_engine: GetPrimaryEngine,
        get_primary_tokenizer: GetPrimaryTokenizer,
        get_primary_runtime_model: GetPrimaryRuntimeModel,
        get_primary_max_model_len: GetPrimaryMaxModelLen,
        get_primary_lora_requests: GetPrimaryLoRARequests,
        set_primary_engine: SetPrimaryEngine,
        set_primary_tokenizer: SetPrimaryTokenizer,
        set_primary_runtime_model: SetPrimaryRuntimeModel,
        set_primary_max_model_len: SetPrimaryMaxModelLen,
        set_primary_lora_requests: SetPrimaryLoRARequests,
        create_engine_client: CreateEngineClient,
    ) -> None:
        self._get_runtimes = get_runtimes
        self._get_primary_engine = get_primary_engine
        self._get_primary_tokenizer = get_primary_tokenizer
        self._get_primary_runtime_model = get_primary_runtime_model
        self._get_primary_max_model_len = get_primary_max_model_len
        self._get_primary_lora_requests = get_primary_lora_requests
        self._set_primary_engine = set_primary_engine
        self._set_primary_tokenizer = set_primary_tokenizer
        self._set_primary_runtime_model = set_primary_runtime_model
        self._set_primary_max_model_len = set_primary_max_model_len
        self._set_primary_lora_requests = set_primary_lora_requests
        self._create_engine_client = create_engine_client

    def has_loaded_engines(self) -> bool:
        return bool(self._get_runtimes()) or self._get_primary_engine() is not None

    def for_runtime_model(self, runtime_model: str) -> EngineClient:
        runtimes = self._get_runtimes()
        primary_runtime_model = self._get_primary_runtime_model()

        if not runtimes:
            if primary_runtime_model and runtime_model != primary_runtime_model:
                raise RuntimeError(f"Runtime model '{runtime_model}' is not initialized.")
            return self._create_engine_client(primary_runtime_model)

        runtime = runtimes.get(runtime_model)
        if runtime is None:
            raise RuntimeError(f"Runtime model '{runtime_model}' is not initialized.")
        if len(runtimes) == 1 and runtime.runtime_model == primary_runtime_model:
            return self._create_engine_client(primary_runtime_model)
        return self._create_engine_client(runtime_model)

    def sync_primary_runtime(self, runtime: LocalEngineRuntimeState) -> None:
        self._set_primary_engine(runtime.engine)
        self._set_primary_tokenizer(runtime.tokenizer)
        self._set_primary_runtime_model(runtime.runtime_model)
        self._set_primary_max_model_len(runtime.max_model_len)
        self._set_primary_lora_requests(runtime.lora_requests)

    def reset_primary_state(self) -> None:
        self._set_primary_engine(None)
        self._set_primary_tokenizer(None)
        self._set_primary_runtime_model("")
        self._set_primary_max_model_len(0)
        self._set_primary_lora_requests({})

    def get_runtime_state(
        self,
        runtime_model: str | None = None,
    ) -> LocalEngineRuntimeState:
        runtimes = self._get_runtimes()
        if runtimes:
            selected_runtime = runtime_model or self._get_primary_runtime_model()
            runtime = runtimes.get(selected_runtime)
            if runtime is None:
                raise RuntimeError(f"Runtime model '{selected_runtime}' is not initialized.")
            return runtime

        engine = self._get_primary_engine()
        if engine is not None:
            selected_runtime = runtime_model or self._get_primary_runtime_model()
            return self._build_primary_runtime_state(engine, selected_runtime)
        raise RuntimeError("Engine runtimes are not initialized.")

    def get_engine(self) -> AsyncLLMEngine:
        engine = self._get_primary_engine()
        if engine is None:
            raise RuntimeError("Engine not initialized. Call start() first.")
        return engine

    def _build_primary_runtime_state(
        self,
        engine: AsyncLLMEngine,
        runtime_model: str,
    ) -> LocalEngineRuntimeState:
        from vfront.engine.service.local import LocalEngineRuntimeState

        return LocalEngineRuntimeState(
            engine=engine,
            runtime_model=runtime_model,
            weights=runtime_model,
            max_model_len=self._get_primary_max_model_len(),
            tokenizer=self._get_primary_tokenizer(),
            lora_requests=self._get_primary_lora_requests(),
            fim=None,
        )
