"""Health-reporting helpers for local engine runtimes."""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING, cast

from vfront.engine.service.infill import describe_infill_config
from vfront.engine.service.reasoning_parsing import (
    list_available_reasoning_parsers,
    resolve_effective_reasoning_parser,
)
from vfront.engine.service.tool_parsing import (
    list_available_tool_parsers,
    resolve_effective_tool_parser,
)
from vfront.shared.engine.reasoning import format_reasoning_effort_key
from vfront.shared.engine.reasoning_adapter import describe_reasoning_template_support
from vfront.shared.engine.types import EffectiveFIMMode, EngineHealth, InfillStrategy

if TYPE_CHECKING:
    from vllm.lora.request import LoRARequest
    from vllm.transformers_utils.tokenizer import AnyTokenizer

    from vfront.engine.service.local import LocalEngineRuntimeState


GetRuntimeState = Callable[[str], "LocalEngineRuntimeState"]
HasRuntimeStates = Callable[[], bool]
GetPrimaryRuntimeModel = Callable[[], str]
GetPrimaryMaxModelLen = Callable[[], int]
GetPrimaryTokenizer = Callable[[], "AnyTokenizer | None"]
SetPrimaryTokenizer = Callable[["AnyTokenizer"], None]
GetPrimaryLoRARequests = Callable[[], dict[str, "LoRARequest"]]


def get_vllm_version() -> str | None:
    try:
        return version("vllm")
    except PackageNotFoundError:
        return None


class LocalEngineHealthReporter:
    """Compute EngineHealth for local runtimes without owning manager state."""

    def __init__(
        self,
        *,
        get_runtime_state: GetRuntimeState,
        has_runtime_states: HasRuntimeStates,
        get_primary_runtime_model: GetPrimaryRuntimeModel,
        get_primary_max_model_len: GetPrimaryMaxModelLen,
        get_primary_tokenizer: GetPrimaryTokenizer,
        set_primary_tokenizer: SetPrimaryTokenizer,
        get_primary_lora_requests: GetPrimaryLoRARequests,
    ) -> None:
        self._get_runtime_state = get_runtime_state
        self._has_runtime_states = has_runtime_states
        self._get_primary_runtime_model = get_primary_runtime_model
        self._get_primary_max_model_len = get_primary_max_model_len
        self._get_primary_tokenizer = get_primary_tokenizer
        self._set_primary_tokenizer = set_primary_tokenizer
        self._get_primary_lora_requests = get_primary_lora_requests

    def compute_health(self, runtime_model: str) -> EngineHealth:
        configured_reasoning_effort_map: (
            dict[str, dict[str, bool | str | int | float | None] | None] | None
        ) = None
        configured_reasoning_parser = None
        effective_reasoning_parser = None
        reasoning_parser_resolution_error = None
        configured_fim_policy = None
        effective_fim_mode: EffectiveFIMMode | None = None
        available_infill_strategies: tuple[InfillStrategy, ...] = ()
        configured_native_infill_profile = None
        configured_fallback_infill_strategy = None

        if self._has_runtime_states():
            runtime = self._get_runtime_state(runtime_model)
            current_runtime_model = runtime.runtime_model
            weights = runtime.weights
            max_model_len = runtime.max_model_len
            tokenizer_id = runtime.tokenizer_id
            quantization = runtime.quantization
            loaded_loras = tuple(sorted(runtime.lora_requests))
            configured_tool_parser = runtime.tool_parser
            configured_reasoning_parser = runtime.reasoning_parser
            tokenizer = runtime.tokenizer
            if tokenizer is None:
                tokenizer = cast("AnyTokenizer", runtime.engine.get_tokenizer())
                runtime.tokenizer = tokenizer
                if runtime.runtime_model == self._get_primary_runtime_model():
                    self._set_primary_tokenizer(tokenizer)
            reasoning_diagnostics = describe_reasoning_template_support(
                tokenizer=tokenizer,
                config=runtime.reasoning,
            )
            configured_reasoning_effort_map = (
                {
                    format_reasoning_effort_key(effort): deepcopy(kwargs)
                    for effort, kwargs in reasoning_diagnostics.configured_effort_map.items()
                }
                if reasoning_diagnostics.configured_effort_map is not None
                else None
            )
            (
                raw_available_infill_strategies,
                configured_native_infill_profile,
                configured_fallback_infill_strategy,
                configured_fim_policy,
                raw_effective_fim_mode,
            ) = describe_infill_config(runtime.infill, fim_policy=runtime.fim)
            available_infill_strategies = cast(
                "tuple[InfillStrategy, ...]",
                raw_available_infill_strategies,
            )
            effective_fim_mode = cast("EffectiveFIMMode | None", raw_effective_fim_mode)
        else:
            runtime = self._get_runtime_state(self._get_primary_runtime_model())
            current_runtime_model = self._get_primary_runtime_model()
            weights = runtime.weights
            max_model_len = self._get_primary_max_model_len()
            tokenizer_id = runtime.tokenizer_id
            quantization = runtime.quantization
            loaded_loras = tuple(sorted(self._get_primary_lora_requests()))
            configured_tool_parser = runtime.tool_parser
            configured_reasoning_parser = runtime.reasoning_parser
            tokenizer = self._get_primary_tokenizer()
            if tokenizer is not None:
                reasoning_diagnostics = describe_reasoning_template_support(
                    tokenizer=tokenizer,
                    config=runtime.reasoning,
                )
                configured_reasoning_effort_map = (
                    {
                        format_reasoning_effort_key(effort): deepcopy(kwargs)
                        for effort, kwargs in reasoning_diagnostics.configured_effort_map.items()
                    }
                    if reasoning_diagnostics.configured_effort_map is not None
                    else None
                )
            (
                raw_available_infill_strategies,
                configured_native_infill_profile,
                configured_fallback_infill_strategy,
                configured_fim_policy,
                raw_effective_fim_mode,
            ) = describe_infill_config(runtime.infill, fim_policy=runtime.fim)
            available_infill_strategies = cast(
                "tuple[InfillStrategy, ...]",
                raw_available_infill_strategies,
            )
            effective_fim_mode = cast("EffectiveFIMMode | None", raw_effective_fim_mode)

        effective_tool_parser = None
        tool_parser_resolution_error = None
        try:
            effective_tool_parser = resolve_effective_tool_parser(
                configured_tool_parser,
                backend_model=weights,
            )
        except Exception as exc:
            tool_parser_resolution_error = str(exc)
        try:
            effective_reasoning_parser = resolve_effective_reasoning_parser(
                configured_reasoning_parser,
                backend_model=weights,
            )
        except Exception as exc:
            reasoning_parser_resolution_error = str(exc)

        return EngineHealth(
            ready=True,
            runtime_model=current_runtime_model,
            max_model_len=max_model_len,
            weights=weights,
            tokenizer_id=tokenizer_id,
            quantization=quantization,
            dtype=runtime.dtype,
            tensor_parallel_size=runtime.tensor_parallel_size,
            pipeline_parallel_size=runtime.pipeline_parallel_size,
            engine_version=get_vllm_version(),
            loaded_loras=loaded_loras,
            transport="local",
            available_tool_parsers=tuple(list_available_tool_parsers()),
            configured_tool_parser=configured_tool_parser,
            effective_tool_parser=effective_tool_parser,
            tool_parser_resolution_error=tool_parser_resolution_error,
            available_reasoning_parsers=tuple(list_available_reasoning_parsers()),
            configured_reasoning_parser=configured_reasoning_parser,
            effective_reasoning_parser=effective_reasoning_parser,
            reasoning_parser_resolution_error=reasoning_parser_resolution_error,
            configured_reasoning_effort_map=configured_reasoning_effort_map,
            configured_fim_policy=configured_fim_policy,
            effective_fim_mode=effective_fim_mode,
            available_infill_strategies=available_infill_strategies,
            configured_native_infill_profile=configured_native_infill_profile,
            configured_fallback_infill_strategy=configured_fallback_infill_strategy,
        )
