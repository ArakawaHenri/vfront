"""Engine-side infill prompt preparation for native FIM and graceful fallback."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol, cast

from jinja2 import Environment

from vfront.engine.service.tokenizer import apply_chat_template_to_ids
from vfront.shared.config.models import InfillConfig, NativeInfillConfig
from vfront.shared.engine.types import PreparedInfillPrompt

if TYPE_CHECKING:
    from vllm.transformers_utils.tokenizer import AnyTokenizer

_JINJA_ENV = Environment()
_DEFAULT_NATIVE_TEMPLATE = "{{ prefix_token }}{{ prefix }}{{ suffix_token }}{{ suffix }}{{ middle_token }}"
FIMPolicyName = Literal["auto", "enabled", "disabled"]


class _TokenizerLike(Protocol):
    def encode(self, text: str) -> list[int]: ...


@dataclass(frozen=True)
class _ResolvedNativeProfile:
    profile: str
    prefix_token: str
    suffix_token: str
    middle_token: str
    template: str
    extra_stop: tuple[str, ...]


def prepare_infill_prompt(
    *,
    tokenizer: _TokenizerLike,
    prefix: str,
    suffix: str,
    config: InfillConfig | None,
    fim_policy: bool | None,
) -> PreparedInfillPrompt:
    if fim_policy is False:
        raise ValueError("Suffix-based completions are disabled for this runtime.")

    if config is not None and config.native is not None:
        native = _prepare_native_prompt(
            tokenizer=tokenizer,
            prefix=prefix,
            suffix=suffix,
            config=config.native,
        )
        return native

    if config is not None and fim_policy is True and config.fallback is not None:
        fallback = config.fallback
        system_prompt = fallback.system_prompt
        user_prompt = _JINJA_ENV.from_string(fallback.user_template).render(
            prefix=prefix,
            suffix=suffix,
            open_tag=fallback.open_tag,
            close_tag=fallback.close_tag,
        )
        prompt_token_ids = apply_chat_template_to_ids(
            cast("AnyTokenizer", tokenizer),
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            add_generation_prompt=True,
        )
        return PreparedInfillPrompt(
            prompt_token_ids=tuple(prompt_token_ids),
            strategy="chat_fallback",
            internal_stop=(fallback.close_tag,),
            response_open_tag=fallback.open_tag,
            response_close_tag=fallback.close_tag,
            supports_streaming=True,
            supports_logprobs=False,
        )

    if fim_policy is True:
        raise ValueError(
            "FIM is enabled for this runtime, but neither native support nor a fallback "
            "strategy is configured."
        )
    raise ValueError("This runtime does not support native suffix-based completions.")


def describe_infill_config(
    config: InfillConfig | None,
    *,
    fim_policy: bool | None,
) -> tuple[tuple[str, ...], str | None, str | None, FIMPolicyName, str]:
    if config is None:
        return (), None, None, describe_fim_policy(fim_policy), effective_fim_mode(
            config=None,
            fim_policy=fim_policy,
        )

    strategies: list[str] = []
    native_profile = None
    fallback_strategy = None
    if config.native is not None:
        strategies.append("native")
        native_profile = config.native.profile
    if config.fallback is not None:
        strategies.append("chat_fallback")
        fallback_strategy = config.fallback.strategy
    return (
        tuple(strategies),
        native_profile,
        fallback_strategy,
        describe_fim_policy(fim_policy),
        effective_fim_mode(config=config, fim_policy=fim_policy),
    )


def describe_fim_policy(fim_policy: bool | None) -> FIMPolicyName:
    if fim_policy is True:
        return "enabled"
    if fim_policy is False:
        return "disabled"
    return "auto"


def effective_fim_mode(*, config: InfillConfig | None, fim_policy: bool | None) -> str:
    if fim_policy is False:
        return "disabled"
    if config is not None and config.native is not None:
        return "native"
    if fim_policy is True and config is not None and config.fallback is not None:
        return "chat_fallback"
    return "disabled"


def _prepare_native_prompt(
    *,
    tokenizer: _TokenizerLike,
    prefix: str,
    suffix: str,
    config: NativeInfillConfig,
) -> PreparedInfillPrompt:
    profile = _resolve_native_profile(config)
    compiled = _JINJA_ENV.from_string(profile.template).render(
        prefix=prefix,
        suffix=suffix,
        prefix_token=profile.prefix_token,
        suffix_token=profile.suffix_token,
        middle_token=profile.middle_token,
    )
    prompt_token_ids = tokenizer.encode(compiled)
    return PreparedInfillPrompt(
        prompt_token_ids=tuple(prompt_token_ids),
        strategy="native",
        internal_stop=profile.extra_stop,
        supports_streaming=True,
        supports_logprobs=True,
    )


def _resolve_native_profile(config: NativeInfillConfig) -> _ResolvedNativeProfile:
    if config.profile == "starcoder":
        return _ResolvedNativeProfile(
            profile="starcoder",
            prefix_token="<fim_prefix>",
            suffix_token="<fim_suffix>",
            middle_token="<fim_middle>",
            template=_DEFAULT_NATIVE_TEMPLATE,
            extra_stop=tuple(config.extra_stop),
        )

    return _ResolvedNativeProfile(
        profile="custom",
        prefix_token=config.prefix_token or "",
        suffix_token=config.suffix_token or "",
        middle_token=config.middle_token or "",
        template=config.template or _DEFAULT_NATIVE_TEMPLATE,
        extra_stop=tuple(config.extra_stop),
    )
