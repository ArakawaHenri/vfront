"""Shared legacy-completions use case."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import cast

from vfront.adapter.output import (
    generate_completion_id,
    request_output_to_completion_response,
)
from vfront.adapter.sampling import completion_request_to_generate_params
from vfront.frontend.api.v1.helper import (
    finalize_completion_generation,
    params_for_completion_prompt,
    prepare_completion_generation,
    prepare_completion_prompts,
    transform_completion_output_stream,
)
from vfront.frontend.api.v1.helper.completion_infill import PreparedCompletionPrompt
from vfront.frontend.compat.validators import (
    validate_completion_prompt_shape,
    validate_completion_request,
)
from vfront.frontend.middleware.exceptions import InvalidRequestError
from vfront.protocol.common import UsageInfo
from vfront.protocol.completion import CompletionRequest, CompletionResponse
from vfront.shared.engine.client import EngineClient
from vfront.shared.engine.generation import collect_final_stream_output
from vfront.shared.engine.types import GenerateParams

_COMPLETION_DEFAULT_PROMPT = "<|endoftext|>"


@dataclass(slots=True, frozen=True)
class PreparedCompletionUseCase:
    request_id: str
    params: GenerateParams
    runtime: EngineClient
    prepared_prompts: tuple[PreparedCompletionPrompt, ...]
    system_fingerprint: str | None
    adapter: str | None
    requested_n: int
    is_multi: bool


async def normalize_all_prompts(
    prompt: str | list[str] | list[int] | list[list[int]] | None,
    backend: EngineClient,
) -> list[tuple[list[int], str]]:
    """Normalize OpenAI completion prompt variants into (token_ids, text) pairs."""
    validate_completion_prompt_shape(prompt)

    if prompt is None:
        return [(await backend.tokenize_text(_COMPLETION_DEFAULT_PROMPT), _COMPLETION_DEFAULT_PROMPT)]

    if isinstance(prompt, str):
        return [(await backend.tokenize_text(prompt), prompt)]

    if isinstance(prompt[0], int):
        token_ids = cast("list[int]", prompt)
        return [(token_ids, await backend.decode_tokens(token_ids))]

    if isinstance(prompt[0], str):
        return [(await backend.tokenize_text(p), p) for p in cast("list[str]", prompt)]

    if isinstance(prompt[0], list):
        result = []
        for ids in cast("list[list[int]]", prompt):
            result.append((ids, await backend.decode_tokens(ids)))
        return result

    raise InvalidRequestError(
        "'prompt' must be a string, array of strings, array of integers, or array of token arrays.",
        param="prompt",
    )


async def prepare_completion_use_case(
    *,
    request: CompletionRequest,
    runtime: EngineClient,
    adapter: str | None,
    system_fingerprint: str | None,
) -> PreparedCompletionUseCase:
    request_id = generate_completion_id()
    prepared = prepare_completion_generation(
        request=request,
        params=completion_request_to_generate_params(request),
    )
    params = prepared.params
    params.lora_name = adapter

    all_prompts = await normalize_all_prompts(request.prompt, runtime)
    validate_completion_request(request, num_prompts=len(all_prompts))
    prepared_prompts = await prepare_completion_prompts(
        request=request,
        normalized_prompts=all_prompts,
        runtime=runtime,
    )
    _validate_prompt_context(prepared_prompts=prepared_prompts, runtime=runtime)

    return PreparedCompletionUseCase(
        request_id=request_id,
        params=params,
        runtime=runtime,
        prepared_prompts=tuple(prepared_prompts),
        system_fingerprint=system_fingerprint,
        adapter=adapter,
        requested_n=prepared.requested_n,
        is_multi=len(prepared_prompts) > 1,
    )


async def execute_non_streaming_completion(
    *,
    request: CompletionRequest,
    prepared: PreparedCompletionUseCase,
) -> CompletionResponse:
    all_choices = []
    total_prompt_tokens = 0
    total_completion_tokens = 0
    choice_offset = 0

    for prompt_index, prompt in enumerate(prepared.prepared_prompts):
        rid = prepared.request_id if not prepared.is_multi else f"{prepared.request_id}-p{prompt_index}"
        final_output = await collect_final_stream_output(
            transform_completion_output_stream(
                prepared.runtime.generate(
                    prompt.prompt_token_ids,
                    params_for_completion_prompt(base=prepared.params, prompt=prompt),
                    rid,
                ),
                prompt=prompt,
                suffix=request.suffix,
            ),
            empty_error_message="Engine returned no output",
        )

        finalized = finalize_completion_generation(
            request=request,
            output=final_output,
            requested_n=prepared.requested_n,
        )
        sub_response = request_output_to_completion_response(
            finalized.output,
            model=request.model,
            request_id=prepared.request_id,
            echo=request.echo or False,
            prompt_text=prompt.prompt_text,
            lora_name=prepared.adapter,
            system_fingerprint=prepared.system_fingerprint,
            completion_tokens_override=finalized.completion_tokens_override,
        )

        for choice in sub_response.choices:
            choice.index = choice_offset + choice.index
        choice_offset += len(sub_response.choices)

        all_choices.extend(sub_response.choices)
        if sub_response.usage:
            total_prompt_tokens += sub_response.usage.prompt_tokens
            total_completion_tokens += sub_response.usage.completion_tokens

    return CompletionResponse(
        id=prepared.request_id,
        created=int(time.time()),
        model=request.model,
        choices=all_choices,
        usage=UsageInfo(
            prompt_tokens=total_prompt_tokens,
            completion_tokens=total_completion_tokens,
            total_tokens=total_prompt_tokens + total_completion_tokens,
        ),
        system_fingerprint=prepared.system_fingerprint,
    )


def _validate_prompt_context(
    *,
    prepared_prompts: tuple[PreparedCompletionPrompt, ...] | list[PreparedCompletionPrompt],
    runtime: EngineClient,
) -> None:
    max_len = runtime.max_model_len
    if max_len <= 0:
        return
    for prompt in prepared_prompts:
        if len(prompt.prompt_token_ids) > max_len:
            raise InvalidRequestError(
                f"Prompt length {len(prompt.prompt_token_ids)} tokens exceeds model context window {max_len}.",
                code="context_length_exceeded",
            )
