"""Shared completion-execution helpers for API v1 routes."""

from __future__ import annotations

from dataclasses import dataclass, replace

from vfront.frontend.middleware.exceptions import InvalidRequestError
from vfront.protocol.completion import CompletionRequest
from vfront.shared.engine.types import CompletionDelta, GenerateOutput, GenerateParams


@dataclass(slots=True)
class PreparedCompletionGeneration:
    params: GenerateParams
    requested_n: int


@dataclass(slots=True)
class FinalizedCompletionGeneration:
    output: GenerateOutput
    completion_tokens_override: int | None = None


def prepare_completion_generation(
    *,
    request: CompletionRequest,
    params: GenerateParams,
) -> PreparedCompletionGeneration:
    """Prepare engine params for completions-only wrapper semantics."""
    requested_n = request.n or 1
    params.n = requested_n

    best_of = request.best_of
    if best_of is not None and best_of > requested_n:
        params.n = best_of
        # vLLM 0.17.x only tracks cumulative logprob when logprobs are enabled.
        if params.logprobs is None:
            params.logprobs = 1

    return PreparedCompletionGeneration(
        params=params,
        requested_n=requested_n,
    )


def finalize_completion_generation(
    *,
    request: CompletionRequest,
    output: GenerateOutput,
    requested_n: int,
) -> FinalizedCompletionGeneration:
    """Apply OpenAI-style best_of filtering and hide internal ranking logprobs."""
    returned = list(output.outputs)
    completion_tokens_override: int | None = None
    best_of = request.best_of

    if best_of is not None and best_of > requested_n:
        completion_tokens_override = sum(len(item.token_ids) for item in returned)
        returned = _select_best_outputs(
            outputs=returned,
            requested_n=requested_n,
        )

    expose_logprobs = request.logprobs is not None and request.logprobs > 0
    normalized_outputs = tuple(
        replace(
            item,
            index=index,
            logprobs=item.logprobs if expose_logprobs else None,
        )
        for index, item in enumerate(returned)
    )
    return FinalizedCompletionGeneration(
        output=replace(output, outputs=normalized_outputs),
        completion_tokens_override=completion_tokens_override,
    )


def _select_best_outputs(
    *,
    outputs: list[CompletionDelta],
    requested_n: int,
) -> list[CompletionDelta]:
    scored: list[tuple[float, float, int, CompletionDelta]] = []
    for index, item in enumerate(outputs):
        if item.cumulative_logprob is None:
            raise RuntimeError(
                "Engine did not provide cumulative logprob required for best_of ranking."
            )
        token_count = max(len(item.token_ids), 1)
        average_logprob = item.cumulative_logprob / token_count
        scored.append((average_logprob, item.cumulative_logprob, -index, item))

    scored.sort(reverse=True)
    return [item for *_score, item in scored[:requested_n]]


def validate_completion_best_of(request: CompletionRequest) -> None:
    """Validate best_of semantics at the API layer."""
    best_of = request.best_of
    if best_of is None:
        return

    if best_of < 0:
        raise InvalidRequestError(
            f"'best_of' must be greater than or equal to 0, got {best_of}.",
            param="best_of",
        )
    if best_of == 0:
        return
    if best_of > 20:
        raise InvalidRequestError(
            f"'best_of' must be less than or equal to 20, got {best_of}.",
            param="best_of",
        )
    if request.stream and best_of > 1:
        raise InvalidRequestError(
            "'best_of' is not supported with streaming responses.",
            param="best_of",
        )

    requested_n = request.n or 1
    if best_of < requested_n:
        raise InvalidRequestError(
            f"'best_of' must be greater than or equal to 'n' ({requested_n}), got {best_of}.",
            param="best_of",
        )
