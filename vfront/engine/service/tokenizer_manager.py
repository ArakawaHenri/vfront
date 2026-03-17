"""Tokenizer coordination for local engine runtimes."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from json import JSONDecodeError
from typing import TYPE_CHECKING, Any, cast

from jinja2 import TemplateError

from vfront.engine.service.infill import prepare_infill_prompt
from vfront.engine.service.tokenizer import apply_chat_template_to_ids
from vfront.shared.engine.errors import EngineInputError
from vfront.shared.engine.reasoning import ReasoningEffort
from vfront.shared.engine.types import PreparedInfillPrompt

if TYPE_CHECKING:
    from vllm import AsyncLLMEngine
    from vllm.transformers_utils.tokenizer import AnyTokenizer

    from vfront.engine.service.local import LocalEngineRuntimeState


logger = logging.getLogger(__name__)

GetRuntimeState = Callable[[str], "LocalEngineRuntimeState"]
HasRuntimeStates = Callable[[], bool]
GetEngine = Callable[[], "AsyncLLMEngine"]
GetPrimaryRuntimeModel = Callable[[], str]
GetPrimaryTokenizer = Callable[[], "AnyTokenizer | None"]
GetPrimaryTokenizerLock = Callable[[], asyncio.Lock]
SetPrimaryTokenizer = Callable[["AnyTokenizer"], None]


class LocalEngineTokenizerManager:
    """Manage tokenizer caching and tokenization across local runtimes."""

    def __init__(
        self,
        *,
        get_runtime_state: GetRuntimeState,
        has_runtime_states: HasRuntimeStates,
        get_engine: GetEngine,
        get_primary_runtime_model: GetPrimaryRuntimeModel,
        get_primary_tokenizer: GetPrimaryTokenizer,
        get_primary_tokenizer_lock: GetPrimaryTokenizerLock,
        set_primary_tokenizer: SetPrimaryTokenizer,
    ) -> None:
        self._get_runtime_state = get_runtime_state
        self._has_runtime_states = has_runtime_states
        self._get_engine = get_engine
        self._get_primary_runtime_model = get_primary_runtime_model
        self._get_primary_tokenizer = get_primary_tokenizer
        self._get_primary_tokenizer_lock = get_primary_tokenizer_lock
        self._set_primary_tokenizer = set_primary_tokenizer

    async def get_tokenizer(self) -> AnyTokenizer:
        async with self._get_primary_tokenizer_lock():
            tokenizer = self._get_primary_tokenizer()
            if tokenizer is None:
                tokenizer = cast("AnyTokenizer", self._get_engine().get_tokenizer())
                self._set_primary_tokenizer(tokenizer)
        return tokenizer

    async def get_tokenizer_for_engine(self, runtime_model: str) -> AnyTokenizer:
        if not self._has_runtime_states():
            return await self.get_tokenizer()

        runtime = self._get_runtime_state(runtime_model)
        async with runtime.tokenizer_lock:
            if runtime.tokenizer is None:
                runtime.tokenizer = cast("AnyTokenizer", runtime.engine.get_tokenizer())
                if runtime.runtime_model == self._get_primary_runtime_model():
                    self._set_primary_tokenizer(runtime.tokenizer)
        return runtime.tokenizer

    async def tokenize_chat_on_engine(
        self,
        runtime_model: str,
        messages: list[dict],
        tools: list[dict] | None = None,
        reasoning_effort: ReasoningEffort | None = None,
    ) -> list[int]:
        try:
            tokenizer = cast("Any", await self.get_tokenizer_for_engine(runtime_model))
            reasoning_config = (
                self._get_runtime_state(runtime_model).reasoning
                if self._has_runtime_states()
                else None
            )
            result = apply_chat_template_to_ids(
                tokenizer,
                messages,
                tools=tools,
                add_generation_prompt=True,
                reasoning_effort=reasoning_effort,
                reasoning_config=reasoning_config,
            )
            return result if isinstance(result, list) else list(result)
        except (TemplateError, JSONDecodeError, TypeError, ValueError) as exc:
            raise EngineInputError(str(exc)) from exc
        except Exception as exc:
            logger.exception("Tokenization error (chat)")
            raise RuntimeError(f"Tokenization error: {exc}") from exc

    async def tokenize_text_on_engine(self, runtime_model: str, text: str) -> list[int]:
        try:
            tokenizer = cast("Any", await self.get_tokenizer_for_engine(runtime_model))
            return tokenizer.encode(text)
        except (TypeError, ValueError) as exc:
            raise EngineInputError(str(exc)) from exc
        except Exception as exc:
            logger.exception("Tokenization error (text)")
            raise RuntimeError(f"Tokenization error: {exc}") from exc

    async def decode_tokens_on_engine(
        self,
        runtime_model: str,
        token_ids: list[int],
    ) -> str:
        try:
            tokenizer = cast("Any", await self.get_tokenizer_for_engine(runtime_model))
            return tokenizer.decode(token_ids)
        except Exception as exc:
            logger.exception("Token decoding error")
            raise RuntimeError(f"Token decoding error: {exc}") from exc

    async def prepare_infill_on_engine(
        self,
        runtime_model: str,
        prefix: str,
        suffix: str,
    ) -> PreparedInfillPrompt:
        try:
            runtime = self._get_runtime_state(runtime_model)
            tokenizer = cast("Any", await self.get_tokenizer_for_engine(runtime_model))
            return prepare_infill_prompt(
                tokenizer=tokenizer,
                prefix=prefix,
                suffix=suffix,
                config=runtime.infill,
                fim_policy=runtime.fim,
            )
        except (TemplateError, JSONDecodeError, TypeError, ValueError) as exc:
            raise EngineInputError(str(exc)) from exc
        except Exception as exc:
            logger.exception("Tokenization error (infill)")
            raise RuntimeError(f"Tokenization error: {exc}") from exc
