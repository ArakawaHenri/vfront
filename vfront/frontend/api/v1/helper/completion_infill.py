"""Helpers for suffix-aware legacy completions."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, replace

from vfront.frontend.middleware.exceptions import InvalidRequestError
from vfront.protocol.completion import CompletionRequest
from vfront.shared.engine.client import EngineClient
from vfront.shared.engine.types import GenerateOutput, GenerateParams, PreparedInfillPrompt


@dataclass(slots=True, frozen=True)
class PreparedCompletionPrompt:
    prompt_token_ids: list[int]
    prompt_text: str
    infill: PreparedInfillPrompt | None = None


async def prepare_completion_prompts(
    *,
    request: CompletionRequest,
    normalized_prompts: list[tuple[list[int], str]],
    runtime: EngineClient,
) -> list[PreparedCompletionPrompt]:
    prompts: list[PreparedCompletionPrompt] = []
    suffix = request.suffix
    if suffix is None:
        return [
            PreparedCompletionPrompt(prompt_token_ids=token_ids, prompt_text=prompt_text)
            for token_ids, prompt_text in normalized_prompts
        ]
    for _token_ids, prompt_text in normalized_prompts:
        try:
            infill = await runtime.prepare_infill(
                prompt_text,
                suffix,
            )
        except Exception as exc:
            raise InvalidRequestError(str(exc), param="suffix") from exc

        if request.logprobs is not None and not infill.supports_logprobs:
            raise InvalidRequestError(
                "The configured suffix-completion fallback does not support 'logprobs'.",
                param="logprobs",
            )
        if request.stream and not infill.supports_streaming:
            raise InvalidRequestError(
                "The configured suffix-completion fallback does not support streaming.",
                param="stream",
            )

        prompts.append(
            PreparedCompletionPrompt(
                prompt_token_ids=list(infill.prompt_token_ids),
                prompt_text=prompt_text,
                infill=infill,
            )
        )
    return prompts


def params_for_completion_prompt(
    *,
    base: GenerateParams,
    prompt: PreparedCompletionPrompt,
) -> GenerateParams:
    internal_stop = list(prompt.infill.internal_stop) if prompt.infill is not None else []
    merged_stop = _merge_stop_sequences(base.stop, internal_stop)
    return replace(base, stop=merged_stop)


async def transform_completion_output_stream(
    output_stream: AsyncIterator[GenerateOutput],
    *,
    prompt: PreparedCompletionPrompt,
    suffix: str | None,
) -> AsyncIterator[GenerateOutput]:
    if prompt.infill is None or suffix is None:
        async for output in output_stream:
            yield output
        return

    async for output in output_stream:
        yield GenerateOutput(
            request_id=output.request_id,
            outputs=tuple(
                replace(
                    delta,
                    text=_extract_visible_text(
                        raw_text=delta.text,
                        prompt=prompt,
                        suffix=suffix,
                        finished=delta.finish_reason is not None or output.finished,
                    ),
                )
                for delta in output.outputs
            ),
            prompt_token_ids=output.prompt_token_ids,
            finished=output.finished,
        )


def _merge_stop_sequences(
    user_stop: list[str] | None,
    internal_stop: list[str],
) -> list[str] | None:
    if not internal_stop:
        return list(user_stop) if user_stop is not None else None
    merged: list[str] = []
    seen: set[str] = set()
    for item in (user_stop or []) + internal_stop:
        if item in seen:
            continue
        seen.add(item)
        merged.append(item)
    return merged


def _extract_visible_text(
    *,
    raw_text: str,
    prompt: PreparedCompletionPrompt,
    suffix: str,
    finished: bool,
) -> str:
    visible = raw_text
    infill = prompt.infill
    if infill is not None and infill.strategy == "chat_fallback":
        open_tag = infill.response_open_tag or ""
        close_tag = infill.response_close_tag or ""
        start_index = visible.find(open_tag) if open_tag else 0
        if start_index == -1:
            if not finished:
                return ""
        else:
            visible = visible[start_index + len(open_tag) :]
        end_index = visible.find(close_tag) if close_tag else -1
        if end_index != -1:
            visible = visible[:end_index]
    return _trim_suffix_overlap(visible, suffix)


def _trim_suffix_overlap(text: str, suffix: str) -> str:
    if not text or not suffix:
        return text
    max_overlap = min(len(text), len(suffix))
    for overlap in range(max_overlap, 0, -1):
        if text.endswith(suffix[:overlap]):
            return text[:-overlap]
    return text
