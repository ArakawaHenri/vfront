"""Pure adapters between vfront engine types and vLLM APIs."""

from __future__ import annotations

import time
from typing import Any, cast

from vllm import SamplingParams
from vllm.inputs.data import token_inputs
from vllm.sampling_params import StructuredOutputsParams

from vfront.shared.engine.types import CompletionDelta, GenerateOutput, GenerateParams


def build_vllm_prompt(
    engine: Any,
    prompt_token_ids: list[int],
    *,
    multimodal_data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    prompt: dict[str, Any] = {"prompt_token_ids": prompt_token_ids}
    if multimodal_data:
        prompt["multi_modal_data"] = multimodal_data

    renderer = getattr(engine, "__dict__", {}).get("renderer")
    process_for_engine = getattr(renderer, "process_for_engine", None)
    if callable(process_for_engine):
        return cast("dict[str, Any]", process_for_engine(prompt, time.time()))

    if multimodal_data:
        return prompt
    return cast("dict[str, Any]", token_inputs(prompt_token_ids))


def to_sampling_params(params: GenerateParams) -> SamplingParams:
    kwargs: dict[str, object] = {
        "temperature": params.temperature,
        "top_p": params.top_p,
        "n": params.n,
        "max_tokens": params.max_tokens,
    }
    if params.stop:
        kwargs["stop"] = params.stop
    if params.presence_penalty != 0.0:
        kwargs["presence_penalty"] = params.presence_penalty
    if params.frequency_penalty != 0.0:
        kwargs["frequency_penalty"] = params.frequency_penalty
    if params.logprobs is not None:
        kwargs["logprobs"] = params.logprobs
    if params.prompt_logprobs is not None:
        kwargs["prompt_logprobs"] = params.prompt_logprobs
    if params.seed is not None:
        kwargs["seed"] = params.seed
    if params.logit_bias:
        kwargs["logit_bias"] = params.logit_bias

    if params.json_schema is not None:
        kwargs["structured_outputs"] = StructuredOutputsParams(json=params.json_schema)
    elif params.json_object:
        kwargs["structured_outputs"] = StructuredOutputsParams(json_object=True)
    return SamplingParams(**cast("dict[str, Any]", kwargs))


def to_generate_output(vllm_output: Any, request_id: str) -> GenerateOutput:
    return GenerateOutput(
        request_id=request_id,
        outputs=tuple(
            CompletionDelta(
                index=out.index,
                text=out.text,
                token_ids=tuple(out.token_ids),
                finish_reason=out.finish_reason,
                logprobs=tuple(out.logprobs) if out.logprobs is not None else None,
                cumulative_logprob=out.cumulative_logprob,
                reasoning_text=getattr(out, "reasoning_content", None),
            )
            for out in vllm_output.outputs
        ),
        prompt_token_ids=tuple(vllm_output.prompt_token_ids),
        finished=vllm_output.finished,
    )
