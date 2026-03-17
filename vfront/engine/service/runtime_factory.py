"""Runtime-state construction for local vLLM engine instances."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, cast

from vllm import AsyncEngineArgs, AsyncLLMEngine
from vllm.lora.request import LoRARequest

if TYPE_CHECKING:
    from vfront.engine.service.local import LocalEngineRuntimeState
    from vfront.shared.config.adapters import AdapterSettings
    from vfront.shared.config.models import RuntimeModelConfig

logger = logging.getLogger(__name__)


class LocalEngineRuntimeFactory:
    """Build LocalEngineRuntimeState instances from runtime settings."""

    def create_runtime_state(
        self,
        *,
        runtime_model: RuntimeModelConfig,
        lora_settings: AdapterSettings,
        default_fim_policy: bool | None,
    ) -> LocalEngineRuntimeState:
        from vfront.engine.service.local import LocalEngineRuntimeState

        engine_args = cast("Any", AsyncEngineArgs)(
            model=runtime_model.weights,
            tokenizer=runtime_model.tokenizer or runtime_model.weights,
            tensor_parallel_size=runtime_model.tensor_parallel_size,
            pipeline_parallel_size=runtime_model.pipeline_parallel_size,
            dtype=runtime_model.dtype,
            gpu_memory_utilization=runtime_model.gpu_memory_utilization,
            enforce_eager=runtime_model.enforce_eager,
            trust_remote_code=runtime_model.trust_remote_code,
            max_model_len=runtime_model.max_model_len,
            quantization=runtime_model.quantization,
        )

        if lora_settings.enabled:
            engine_args.enable_lora = True
            engine_args.max_lora_rank = lora_settings.max_rank
            engine_args.max_loras = lora_settings.max_loras
            if lora_settings.max_cpu_loras is not None:
                engine_args.max_cpu_loras = lora_settings.max_cpu_loras

        runtime = LocalEngineRuntimeState(
            engine=AsyncLLMEngine.from_engine_args(engine_args),
            runtime_model=runtime_model.id,
            weights=runtime_model.weights,
            max_model_len=runtime_model.max_model_len or 0,
            tokenizer_id=runtime_model.tokenizer,
            quantization=runtime_model.quantization,
            dtype=runtime_model.dtype,
            tensor_parallel_size=runtime_model.tensor_parallel_size,
            pipeline_parallel_size=runtime_model.pipeline_parallel_size,
            tool_parser=runtime_model.tool_parser,
            reasoning=runtime_model.reasoning,
            reasoning_parser=runtime_model.reasoning_parser,
            infill=runtime_model.infill,
            fim=runtime_model.fim if runtime_model.fim is not None else default_fim_policy,
        )

        if lora_settings.enabled:
            engine_modules = [
                module for module in lora_settings.modules if module.runtime_model == runtime_model.id
            ]
            for idx, module in enumerate(engine_modules, start=1):
                runtime.lora_requests[module.id] = LoRARequest(
                    lora_name=module.id,
                    lora_int_id=idx,
                    lora_path=module.path,
                )
                logger.info(
                    "Registered LoRA adapter: %s → %s (runtime=%s)",
                    module.id,
                    module.path,
                    runtime_model.id,
                )

        return runtime
