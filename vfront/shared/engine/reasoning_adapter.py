"""Explicit reasoning policy resolution for backend-specific chat templates."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING

from vfront.shared.config.models import (
    ReasoningConfig,
    ReasoningPolicyValue,
)
from vfront.shared.engine.reasoning import (
    ReasoningEffort,
    ReasoningEffortKey,
    format_reasoning_effort_key,
)
from vfront.shared.engine.signatures import inspect_keyword_argument_support

if TYPE_CHECKING:
    from vllm.transformers_utils.tokenizer import AnyTokenizer

ReasoningBindingValue = bool | str | int | float | None


@dataclass(frozen=True)
class ReasoningTemplateDiagnostics:
    configured_effort_map: dict[ReasoningEffortKey, ReasoningPolicyValue] | None


def supports_chat_template_kwarg(tokenizer: AnyTokenizer, name: str) -> bool:
    """Return True when the tokenizer/template actually consumes the kwarg."""
    support = inspect_keyword_argument_support(tokenizer.apply_chat_template, name)
    return support.explicit or support.via_var_keyword


def resolve_reasoning_template_kwargs(
    *,
    tokenizer: AnyTokenizer,
    reasoning_effort: ReasoningEffort | None,
    config: ReasoningConfig | None,
) -> dict[str, ReasoningBindingValue]:
    """Resolve chat-template kwargs from the explicit configured effort map."""
    if config is None:
        if reasoning_effort is None:
            return {}
        raise ValueError(
            "This model does not define reasoning policies. "
            "Remove 'reasoning_effort' or configure "
            "engine.runtime.models[].reasoning.effort_map."
        )

    if reasoning_effort not in config.effort_map:
        key = format_reasoning_effort_key(reasoning_effort)
        raise ValueError(
            f"Reasoning policy '{key}' is not configured for this model. "
            "Configure engine.runtime.models[].reasoning.effort_map explicitly."
        )

    return _resolve_explicit_kwargs(
        tokenizer=tokenizer,
        explicit_kwargs=config.effort_map[reasoning_effort],
    )


def describe_reasoning_template_support(
    *,
    tokenizer: AnyTokenizer,
    config: ReasoningConfig | None,
) -> ReasoningTemplateDiagnostics:
    del tokenizer
    return ReasoningTemplateDiagnostics(
        configured_effort_map=deepcopy(config.effort_map) if config is not None else None
    )


def _resolve_explicit_kwargs(
    *,
    tokenizer: AnyTokenizer,
    explicit_kwargs: ReasoningPolicyValue,
) -> dict[str, ReasoningBindingValue]:
    if explicit_kwargs is None:
        return {}

    kwargs: dict[str, ReasoningBindingValue] = {}
    for key, value in explicit_kwargs.items():
        if not supports_chat_template_kwarg(tokenizer, key):
            raise RuntimeError(
                f"Configured reasoning chat-template kwarg '{key}' is not "
                "supported by the active tokenizer/template."
            )
        kwargs[key] = value
    return kwargs
