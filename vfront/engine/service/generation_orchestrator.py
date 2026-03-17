"""Generation and encoding orchestration for local engine runtimes."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, cast

from vllm import PoolingParams, SamplingParams
from vllm.lora.request import LoRARequest

from vfront.engine.service.output_parsing import EngineOutputParserSession
from vfront.engine.service.reasoning_parsing import resolve_effective_reasoning_parser
from vfront.engine.service.tool_parsing import resolve_effective_tool_parser
from vfront.shared.engine.types import EmbeddingOutput, EncodeParams, GenerateOutput, GenerateParams

if TYPE_CHECKING:
    from vllm import AsyncLLMEngine
    from vllm.transformers_utils.tokenizer import AnyTokenizer

    from vfront.engine.service.local import LocalEngineRuntimeState
    from vfront.shared.config.models import ReasoningConfig


logger = logging.getLogger(__name__)

GetRuntimeState = Callable[[str], "LocalEngineRuntimeState"]
HasRuntimeStates = Callable[[], bool]
GetEngine = Callable[[], "AsyncLLMEngine"]
GetPrimaryRuntimeModel = Callable[[], str]
GetTokenizerForEngine = Callable[[str], Awaitable["AnyTokenizer"]]
ResolveLoRA = Callable[[str], LoRARequest]
ResolveLoRAForRuntime = Callable[["LocalEngineRuntimeState", str | None], LoRARequest | None]
ToSamplingParams = Callable[[GenerateParams], SamplingParams]
ToGenerateOutput = Callable[[Any, str], GenerateOutput]
BuildVllmPrompt = Callable[..., dict[str, Any]]


@dataclass(slots=True, frozen=True)
class GenerateRuntimeContext:
    engine: Any
    lora_request: LoRARequest | None
    weights: str
    configured_tool_parser: str | None
    configured_reasoning_parser: str | None
    reasoning_config: ReasoningConfig | None


class LocalEngineOrchestrator:
    """Coordinate generation and encoding while leaving state ownership to the manager."""

    def __init__(
        self,
        *,
        get_runtime_state: GetRuntimeState,
        has_runtime_states: HasRuntimeStates,
        get_engine: GetEngine,
        get_primary_runtime_model: GetPrimaryRuntimeModel,
        get_tokenizer_for_engine: GetTokenizerForEngine,
        resolve_lora_for_runtime: ResolveLoRAForRuntime,
        resolve_lora: ResolveLoRA,
        to_sampling_params: ToSamplingParams,
        to_generate_output: ToGenerateOutput,
        build_vllm_prompt: BuildVllmPrompt,
    ) -> None:
        self._get_runtime_state = get_runtime_state
        self._has_runtime_states = has_runtime_states
        self._get_engine = get_engine
        self._get_primary_runtime_model = get_primary_runtime_model
        self._get_tokenizer_for_engine = get_tokenizer_for_engine
        self._resolve_lora_for_runtime = resolve_lora_for_runtime
        self._resolve_lora = resolve_lora
        self._to_sampling_params = to_sampling_params
        self._to_generate_output = to_generate_output
        self._build_vllm_prompt = build_vllm_prompt

    async def generate(
        self,
        runtime_model: str,
        prompt_token_ids: list[int],
        params: GenerateParams,
        request_id: str,
    ) -> AsyncIterator[GenerateOutput]:
        runtime = self._resolve_generate_runtime_context(runtime_model, params)
        sampling_params = self._to_sampling_params(params)
        prompt = self._build_vllm_prompt(
            runtime.engine,
            prompt_token_ids,
            multimodal_data=params.multimodal_data,
        )
        output_parser_session = await self._create_output_parser_session(
            runtime_model=runtime_model,
            params=params,
            runtime=runtime,
        )
        try:
            async for vllm_output in runtime.engine.generate(
                prompt=prompt,
                sampling_params=sampling_params,
                request_id=request_id,
                lora_request=runtime.lora_request,
            ):
                output = self._to_generate_output(vllm_output, request_id)
                if output_parser_session is not None:
                    output = self._parse_generation_output(output, output_parser_session)
                yield output
        except RuntimeError:
            raise
        except Exception as exc:
            logger.exception("Engine generate error for request %s", request_id)
            raise RuntimeError(f"Engine error: {exc}") from exc

    async def encode(
        self,
        runtime_model: str,
        prompt_token_ids: list[int],
        params: EncodeParams,
        request_id: str,
    ) -> EmbeddingOutput:
        engine = (
            self._get_runtime_state(runtime_model).engine
            if self._has_runtime_states()
            else self._get_engine()
        )
        pooling_params = cast("Any", PoolingParams)(
            **({"dimensions": params.dimensions} if params.dimensions else {})
        )
        try:
            final_output = None
            async for output in engine.encode(
                prompt=cast("Any", self._build_vllm_prompt(engine, prompt_token_ids)),
                pooling_params=pooling_params,
                request_id=request_id,
            ):
                final_output = output

            if final_output is None:
                raise RuntimeError("Engine returned no embedding output")

            return EmbeddingOutput(
                data=final_output.outputs.data.tolist(),
                prompt_token_ids=tuple(final_output.prompt_token_ids),
            )
        except RuntimeError:
            raise
        except Exception as exc:
            logger.exception("Engine encode error for request %s", request_id)
            raise RuntimeError(f"Engine error: {exc}") from exc

    def _resolve_generate_runtime_context(
        self,
        runtime_model: str,
        params: GenerateParams,
    ) -> GenerateRuntimeContext:
        if self._has_runtime_states():
            runtime = self._get_runtime_state(runtime_model)
            return GenerateRuntimeContext(
                engine=runtime.engine,
                lora_request=self._resolve_lora_for_runtime(runtime, params.lora_name),
                weights=runtime.weights,
                configured_tool_parser=runtime.tool_parser,
                configured_reasoning_parser=runtime.reasoning_parser,
                reasoning_config=runtime.reasoning,
            )

        return GenerateRuntimeContext(
            engine=self._get_engine(),
            lora_request=self._resolve_lora(params.lora_name) if params.lora_name else None,
            weights=self._get_primary_runtime_model(),
            configured_tool_parser=None,
            configured_reasoning_parser=None,
            reasoning_config=None,
        )

    async def _create_output_parser_session(
        self,
        *,
        runtime_model: str,
        params: GenerateParams,
        runtime: GenerateRuntimeContext,
    ) -> EngineOutputParserSession | None:
        effective_tool_parser = None
        effective_reasoning_parser = (
            resolve_effective_reasoning_parser(
                runtime.configured_reasoning_parser,
                backend_model=runtime.weights,
            )
            if params.chat_template_applied
            else None
        )
        if params.tools:
            effective_tool_parser = resolve_effective_tool_parser(
                params.tool_parser
                if params.tool_parser is not None
                else runtime.configured_tool_parser,
                backend_model=runtime.weights,
            )
        if effective_reasoning_parser is None and (
            effective_tool_parser is None or not params.tools
        ):
            return None

        tokenizer = cast("Any", await self._get_tokenizer_for_engine(runtime_model))
        return EngineOutputParserSession.create(
            tokenizer=tokenizer,
            num_choices=params.n,
            tools=params.tools,
            tool_choice=params.tool_choice,
            tool_parser_name=effective_tool_parser if params.tools else None,
            reasoning_parser_name=effective_reasoning_parser,
            reasoning_effort=params.reasoning_effort,
            reasoning_config=runtime.reasoning_config,
        )

    @staticmethod
    def _parse_generation_output(
        output: GenerateOutput,
        output_parser_session: EngineOutputParserSession,
    ) -> GenerateOutput:
        return GenerateOutput(
            request_id=output.request_id,
            outputs=tuple(
                replace(
                    delta,
                    text=parsed_state.text,
                    reasoning_text=(
                        parsed_state.reasoning_text
                        if parsed_state.reasoning_text is not None
                        else delta.reasoning_text
                    ),
                    semantic=parsed_state.semantic,
                    parsed_tool_calls=(
                        output_parser_session.parse_final(
                            choice_index=delta.index,
                            current_text=delta.text,
                        )
                        if delta.finish_reason is not None
                        else None
                    ),
                )
                for delta in output.outputs
                for parsed_state in (
                    output_parser_session.parse(
                        choice_index=delta.index,
                        current_text=delta.text,
                        current_token_ids=delta.token_ids,
                    ),
                )
            ),
            prompt_token_ids=output.prompt_token_ids,
            finished=output.finished,
        )
