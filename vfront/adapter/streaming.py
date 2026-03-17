"""SSE streaming chunk generation for Chat Completions and Completions."""

from __future__ import annotations

import contextlib
import json
import logging
import time
from collections.abc import AsyncGenerator

from vfront.adapter.assistant_output import AssistantOutputStreamState
from vfront.adapter.output import _map_finish_reason, generate_system_fingerprint
from vfront.adapter.tool_calls import build_tool_calls_from_parsed
from vfront.adapter.usage import estimate_reasoning_tokens
from vfront.protocol.chat_completion import (
    ChatCompletionStreamChoice,
    ChatCompletionStreamResponse,
    ChoiceLogprobs,
    DeltaMessage,
)
from vfront.protocol.common import (
    ChatUsageInfo,
    CompletionTokensDetails,
    LogprobContent,
    PromptTokensDetails,
    ToolCall,
    TopLogprob,
    UsageInfo,
)
from vfront.protocol.completion import (
    CompletionStreamChoice,
    CompletionStreamResponse,
)
from vfront.shared.engine.types import GenerateOutput


def _build_streaming_logprobs(
    lp_slice: tuple[dict[str, float], ...],
) -> ChoiceLogprobs | None:
    """Build a ChoiceLogprobs from a per-token logprob slice (streaming delta)."""
    content: list[LogprobContent] = []
    for lp in lp_slice:
        if not lp:
            continue
        top_list = [
            TopLogprob(token=tok, logprob=val, bytes=list(tok.encode("utf-8")))
            for tok, val in lp.items()
        ]
        if top_list:
            content.append(
                LogprobContent(
                    token=top_list[0].token,
                    logprob=top_list[0].logprob,
                    bytes=top_list[0].bytes,
                    top_logprobs=top_list,
                )
            )
    return ChoiceLogprobs(content=content) if content else None


def _build_tool_call_chunks(
    request_id: str,
    created: int,
    model: str,
    fingerprint: str,
    choice_index: int,
    tool_calls: list[ToolCall],
) -> list[str]:
    chunks: list[str] = []
    for tc_idx, tc in enumerate(tool_calls):
        first_delta = {
            "index": tc_idx,
            "id": tc.id,
            "type": "function",
            "function": {
                "name": tc.function.name,
                "arguments": "",
            },
        }
        first_chunk = ChatCompletionStreamResponse(
            id=request_id,
            created=created,
            model=model,
            choices=[
                ChatCompletionStreamChoice(
                    index=choice_index,
                    delta=DeltaMessage(tool_calls=[first_delta]),
                    finish_reason=None,
                )
            ],
            system_fingerprint=fingerprint,
        )
        chunks.append(f"data: {first_chunk.model_dump_json()}\n\n")

        args_delta = {
            "index": tc_idx,
            "function": {
                "arguments": tc.function.arguments,
            },
        }
        args_chunk = ChatCompletionStreamResponse(
            id=request_id,
            created=created,
            model=model,
            choices=[
                ChatCompletionStreamChoice(
                    index=choice_index,
                    delta=DeltaMessage(tool_calls=[args_delta]),
                    finish_reason=None,
                )
            ],
            system_fingerprint=fingerprint,
        )
        chunks.append(f"data: {args_chunk.model_dump_json()}\n\n")

    return chunks


def _build_content_chunk(
    request_id: str,
    created: int,
    model: str,
    fingerprint: str,
    choice_index: int,
    text: str,
    logprobs: ChoiceLogprobs | None = None,
) -> str:
    chunk = ChatCompletionStreamResponse(
        id=request_id,
        created=created,
        model=model,
        choices=[
            ChatCompletionStreamChoice(
                index=choice_index,
                delta=DeltaMessage(content=text),
                finish_reason=None,
                logprobs=logprobs,
            )
        ],
        system_fingerprint=fingerprint,
    )
    return f"data: {chunk.model_dump_json()}\n\n"

async def stream_chat_completion(
    request_id: str,
    model: str,
    output_generator: AsyncGenerator[GenerateOutput, None],
    include_usage: bool = False,
    n: int = 1,
    has_tool_definitions: bool = False,
    semantic_tool_parsing: bool = False,
    emit_done: bool = True,
    lora_name: str | None = None,
    system_fingerprint: str | None = None,
) -> AsyncGenerator[str, None]:
    """Yield SSE-formatted chunks: `data: {json}\\n\\n` with `data: [DONE]\\n\\n`.

    When has_tool_definitions=True and the model finishes with tool_calls,
    the accumulated text is parsed into proper ChoiceDeltaToolCall objects
    and emitted in the SDK-expected incremental format.

    When emit_done=False, the `[DONE]` sentinel is NOT emitted — the caller
    is responsible for emitting it (used by the agentic streaming loop).
    """
    created = int(time.time())
    fingerprint = system_fingerprint or generate_system_fingerprint(lora_name=lora_name)

    # Initial role chunk
    for i in range(n):
        initial = ChatCompletionStreamResponse(
            id=request_id,
            created=created,
            model=model,
            choices=[
                ChatCompletionStreamChoice(
                    index=i,
                    delta=DeltaMessage(role="assistant", content=""),
                    finish_reason=None,
                )
            ],
            system_fingerprint=fingerprint,
        )
        yield f"data: {initial.model_dump_json()}\n\n"

    total_prompt_tokens = 0
    total_completion_tokens = 0
    prev_token_counts: dict[int, int] = {}
    prev_logprob_counts: dict[int, int] = {}
    prev_reasoning_texts: dict[int, str] = {}
    last_logprobs: dict[int, tuple[dict[str, float], ...]] = {}
    last_reasoning_texts: dict[int, str] = {}
    semantic_tool_calls_seen: set[int] = set()
    # Pre-populate all n states so every choice gets a finish chunk even if
    # the engine never emits output for a particular index (edge case with n>1).
    states: dict[int, AssistantOutputStreamState] = {
        i: AssistantOutputStreamState(has_tool_definitions=has_tool_definitions)
        for i in range(n)
    }
    stream_failed = False
    try:
        async for output in output_generator:
            total_prompt_tokens = len(output.prompt_token_ids)

            for completion in output.outputs:
                idx = completion.index
                state = states.setdefault(
                    idx,
                    AssistantOutputStreamState(
                        has_tool_definitions=has_tool_definitions,
                    ),
                )
                delta_text = state.update(completion.text, completion.finish_reason)

                if completion.semantic and completion.semantic.tool_call_deltas:
                    semantic_tool_calls_seen.add(idx)
                    if completion.semantic.content_delta:
                        yield _build_content_chunk(
                            request_id,
                            created,
                            model,
                            fingerprint,
                            idx,
                            completion.semantic.content_delta,
                        )
                    for tool_delta in completion.semantic.tool_call_deltas:
                        tool_call_payload: dict[str, object] = {"index": tool_delta.index}
                        if tool_delta.id is not None:
                            tool_call_payload["id"] = tool_delta.id
                        if tool_delta.type is not None:
                            tool_call_payload["type"] = tool_delta.type
                        function_payload: dict[str, str] = {}
                        if tool_delta.name is not None:
                            function_payload["name"] = tool_delta.name
                        if tool_delta.arguments_delta is not None:
                            function_payload["arguments"] = tool_delta.arguments_delta
                        if function_payload:
                            tool_call_payload["function"] = function_payload
                        chunk = ChatCompletionStreamResponse(
                            id=request_id,
                            created=created,
                            model=model,
                            choices=[
                                ChatCompletionStreamChoice(
                                    index=idx,
                                    delta=DeltaMessage(tool_calls=[tool_call_payload]),
                                    finish_reason=None,
                                )
                            ],
                            system_fingerprint=fingerprint,
                        )
                        yield f"data: {chunk.model_dump_json()}\n\n"
                    if completion.logprobs is not None:
                        last_logprobs[idx] = completion.logprobs
                    if completion.reasoning_text is not None:
                        last_reasoning_texts[idx] = completion.reasoning_text
                    continue

                reasoning_text = completion.reasoning_text or ""
                previous_reasoning = prev_reasoning_texts.get(idx, "")
                reasoning_delta = reasoning_text[len(previous_reasoning) :]
                prev_reasoning_texts[idx] = reasoning_text
                if reasoning_text:
                    last_reasoning_texts[idx] = reasoning_text
                if reasoning_delta:
                    reasoning_chunk = ChatCompletionStreamResponse(
                        id=request_id,
                        created=created,
                        model=model,
                        choices=[
                            ChatCompletionStreamChoice(
                                index=idx,
                                delta=DeltaMessage(reasoning=reasoning_delta),
                                finish_reason=None,
                            )
                        ],
                        system_fingerprint=fingerprint,
                    )
                    yield f"data: {reasoning_chunk.model_dump_json()}\n\n"

                cur_tokens = len(completion.token_ids)
                prev_tokens = prev_token_counts.get(idx, 0)
                total_completion_tokens += max(0, cur_tokens - prev_tokens)
                prev_token_counts[idx] = cur_tokens

                if delta_text:
                    lp_obj: ChoiceLogprobs | None = None
                    if completion.logprobs is not None:
                        prev_lp = prev_logprob_counts.get(idx, 0)
                        cur_lp = len(completion.logprobs)
                        lp_obj = _build_streaming_logprobs(completion.logprobs[prev_lp:cur_lp])
                        prev_logprob_counts[idx] = cur_lp
                    chunk = ChatCompletionStreamResponse(
                        id=request_id,
                        created=created,
                        model=model,
                        choices=[
                            ChatCompletionStreamChoice(
                                index=idx,
                                delta=DeltaMessage(content=delta_text),
                                finish_reason=None,
                                logprobs=lp_obj,
                            )
                        ],
                        system_fingerprint=fingerprint,
                    )
                    yield f"data: {chunk.model_dump_json()}\n\n"

                if completion.logprobs is not None:
                    last_logprobs[idx] = completion.logprobs

        # After streaming completes, check for tool calls in accumulated text
        for idx in sorted(states.keys()):
            state = states[idx]
            normalized, deferred_text = state.finalize()
            mapped_reason = _map_finish_reason(state.finish_reason)

            if idx in semantic_tool_calls_seen:
                mapped_reason = "tool_calls"
            elif semantic_tool_parsing and normalized.kind == "tool_calls":
                raise RuntimeError(
                    "Engine-side semantic parser was expected to provide streaming tool call deltas, "
                    "but only legacy post-hoc tool-call text was available."
                )
            elif not semantic_tool_parsing and normalized.kind == "tool_calls" and normalized.tool_calls:
                for sse_chunk in _build_tool_call_chunks(
                    request_id,
                    created,
                    model,
                    fingerprint,
                    idx,
                    normalized.tool_calls,
                ):
                    yield sse_chunk
                mapped_reason = "tool_calls"
            elif deferred_text:
                deferred_lp: ChoiceLogprobs | None = None
                lps = last_logprobs.get(idx)
                if lps is not None:
                    prev_lp = prev_logprob_counts.get(idx, 0)
                    deferred_lp = _build_streaming_logprobs(lps[prev_lp:])
                yield _build_content_chunk(
                    request_id,
                    created,
                    model,
                    fingerprint,
                    idx,
                    deferred_text,
                    logprobs=deferred_lp,
                )

            # Emit finish chunk
            finish = ChatCompletionStreamResponse(
                id=request_id,
                created=created,
                model=model,
                choices=[
                    ChatCompletionStreamChoice(
                        index=idx,
                        delta=DeltaMessage(),
                        finish_reason=mapped_reason,
                    )
                ],
                system_fingerprint=fingerprint,
            )
            yield f"data: {finish.model_dump_json()}\n\n"

        if include_usage:
            total_reasoning_tokens = sum(
                estimate_reasoning_tokens(reasoning_text)
                for reasoning_text in last_reasoning_texts.values()
            )
            usage_chunk = ChatCompletionStreamResponse(
                id=request_id,
                created=created,
                model=model,
                choices=[],
                usage=ChatUsageInfo(
                    prompt_tokens=total_prompt_tokens,
                    completion_tokens=total_completion_tokens,
                    total_tokens=total_prompt_tokens + total_completion_tokens,
                    completion_tokens_details=CompletionTokensDetails(
                        reasoning_tokens=total_reasoning_tokens
                    ),
                    prompt_tokens_details=PromptTokensDetails(cached_tokens=0),
                ),
                system_fingerprint=fingerprint,
            )
            yield f"data: {usage_chunk.model_dump_json()}\n\n"
    except Exception as exc:
        stream_failed = True
        logging.getLogger(__name__).exception("Error during chat completion streaming")
        # Emit a standard OpenAI error event before ending the stream
        error_payload = {
            "error": {
                "message": f"An error occurred during streaming: {exc}",
                "type": "server_error",
                "param": None,
                "code": "stream_error",
            }
        }
        yield f"data: {json.dumps(error_payload)}\n\n"
        raise
    except BaseException:
        stream_failed = True
        raise
    finally:
        # Ensure vLLM stops generating on client disconnect or error
        with contextlib.suppress(Exception):
            await output_generator.aclose()
        if emit_done and not stream_failed:
            yield "data: [DONE]\n\n"


async def stream_and_collect(
    request_id: str,
    model: str,
    output_generator: AsyncGenerator[GenerateOutput, None],
    n: int = 1,
    has_tool_definitions: bool = False,
    accumulated: list[str] | None = None,
    collected_tool_calls: list[ToolCall] | None = None,
    lora_name: str | None = None,
    system_fingerprint: str | None = None,
) -> AsyncGenerator[str, None]:
    """Stream a single generation round, collecting accumulated text.

    Yields SSE chunks (no [DONE]). Stores the final accumulated text for
    index 0 into `accumulated[0]` if provided — used by the agentic loop
    to detect tool calls after each round.
    """
    del n
    created = int(time.time())
    fingerprint = system_fingerprint or generate_system_fingerprint(lora_name=lora_name)
    states: dict[int, AssistantOutputStreamState] = {}
    prev_reasoning_texts: dict[int, str] = {}
    semantic_tool_calls_seen: set[int] = set()
    last_parsed_tool_calls: dict[int, list[ToolCall]] = {}

    try:
        async for output in output_generator:
            for completion in output.outputs:
                idx = completion.index
                state = states.setdefault(
                    idx,
                    AssistantOutputStreamState(
                        has_tool_definitions=has_tool_definitions,
                    ),
                )
                delta_text = state.update(completion.text, completion.finish_reason)

                if completion.parsed_tool_calls is not None:
                    parsed = build_tool_calls_from_parsed(completion.parsed_tool_calls)
                    if parsed:
                        last_parsed_tool_calls[idx] = parsed

                if completion.semantic and completion.semantic.tool_call_deltas:
                    semantic_tool_calls_seen.add(idx)
                    if completion.semantic.content_delta:
                        yield _build_content_chunk(
                            request_id,
                            created,
                            model,
                            fingerprint,
                            idx,
                            completion.semantic.content_delta,
                        )
                    for tool_delta in completion.semantic.tool_call_deltas:
                        tool_call_payload: dict[str, object] = {"index": tool_delta.index}
                        if tool_delta.id is not None:
                            tool_call_payload["id"] = tool_delta.id
                        if tool_delta.type is not None:
                            tool_call_payload["type"] = tool_delta.type
                        function_payload: dict[str, str] = {}
                        if tool_delta.name is not None:
                            function_payload["name"] = tool_delta.name
                        if tool_delta.arguments_delta is not None:
                            function_payload["arguments"] = tool_delta.arguments_delta
                        if function_payload:
                            tool_call_payload["function"] = function_payload
                        chunk = ChatCompletionStreamResponse(
                            id=request_id,
                            created=created,
                            model=model,
                            choices=[
                                ChatCompletionStreamChoice(
                                    index=idx,
                                    delta=DeltaMessage(tool_calls=[tool_call_payload]),
                                    finish_reason=None,
                                )
                            ],
                            system_fingerprint=fingerprint,
                        )
                        yield f"data: {chunk.model_dump_json()}\n\n"
                    continue

                reasoning_text = completion.reasoning_text or ""
                previous_reasoning = prev_reasoning_texts.get(idx, "")
                reasoning_delta = reasoning_text[len(previous_reasoning) :]
                prev_reasoning_texts[idx] = reasoning_text
                if reasoning_delta:
                    chunk = ChatCompletionStreamResponse(
                        id=request_id,
                        created=created,
                        model=model,
                        choices=[
                            ChatCompletionStreamChoice(
                                index=idx,
                                delta=DeltaMessage(reasoning=reasoning_delta),
                                finish_reason=None,
                            )
                        ],
                        system_fingerprint=fingerprint,
                    )
                    yield f"data: {chunk.model_dump_json()}\n\n"

                if delta_text:
                    chunk = ChatCompletionStreamResponse(
                        id=request_id,
                        created=created,
                        model=model,
                        choices=[
                            ChatCompletionStreamChoice(
                                index=idx,
                                delta=DeltaMessage(content=delta_text),
                                finish_reason=None,
                            )
                        ],
                        system_fingerprint=fingerprint,
                    )
                    yield f"data: {chunk.model_dump_json()}\n\n"
    finally:
        await output_generator.aclose()

    # Post-stream: emit tool call chunks if detected
    for idx in sorted(states.keys()):
        state = states[idx]
        normalized, deferred_text = state.finalize()
        mapped_reason = _map_finish_reason(state.finish_reason)

        semantic_tool_calls = last_parsed_tool_calls.get(idx)
        if semantic_tool_calls or idx in semantic_tool_calls_seen:
            mapped_reason = "tool_calls"
        elif normalized.kind == "tool_calls" and normalized.tool_calls:
            for sse_chunk in _build_tool_call_chunks(
                request_id,
                created,
                model,
                fingerprint,
                idx,
                normalized.tool_calls,
            ):
                yield sse_chunk
            mapped_reason = "tool_calls"
        elif deferred_text:
            yield _build_content_chunk(
                request_id,
                created,
                model,
                fingerprint,
                idx,
                deferred_text,
            )

        finish = ChatCompletionStreamResponse(
            id=request_id,
            created=created,
            model=model,
            choices=[
                ChatCompletionStreamChoice(
                    index=idx,
                    delta=DeltaMessage(),
                    finish_reason=mapped_reason,
                )
            ],
            system_fingerprint=fingerprint,
        )
        yield f"data: {finish.model_dump_json()}\n\n"

    # Store accumulated text for the caller
    if accumulated is not None:
        accumulated.clear()
        state0 = states.get(0)
        accumulated.append(state0.text if state0 is not None else "")
    if collected_tool_calls is not None:
        collected_tool_calls.clear()
        if semantic_calls := last_parsed_tool_calls.get(0):
            collected_tool_calls.extend(semantic_calls)
        else:
            state0 = states.get(0)
            if state0 is not None:
                normalized, _ = state0.finalize()
                if normalized.tool_calls:
                    collected_tool_calls.extend(normalized.tool_calls)


async def stream_completion(
    request_id: str,
    model: str,
    output_generator: AsyncGenerator[GenerateOutput, None],
    include_usage: bool = False,
    n: int = 1,
    lora_name: str | None = None,
    system_fingerprint: str | None = None,
) -> AsyncGenerator[str, None]:
    """Yield SSE-formatted chunks for legacy completions."""
    created = int(time.time())
    fingerprint = system_fingerprint or generate_system_fingerprint(lora_name=lora_name)
    total_prompt_tokens = 0
    total_completion_tokens = 0
    prev_texts: dict[int, str] = {}
    prev_token_counts: dict[int, int] = {}
    stream_failed = False

    try:
        async for output in output_generator:
            total_prompt_tokens = len(output.prompt_token_ids)

            for completion in output.outputs:
                idx = completion.index
                new_text = completion.text
                prev = prev_texts.get(idx, "")
                delta_text = new_text[len(prev) :]
                prev_texts[idx] = new_text

                cur_tokens = len(completion.token_ids)
                prev_tokens = prev_token_counts.get(idx, 0)
                total_completion_tokens += max(0, cur_tokens - prev_tokens)
                prev_token_counts[idx] = cur_tokens

                if delta_text:
                    chunk = CompletionStreamResponse(
                        id=request_id,
                        created=created,
                        model=model,
                        choices=[
                            CompletionStreamChoice(
                                index=idx,
                                text=delta_text,
                                finish_reason=None,
                            )
                        ],
                        system_fingerprint=fingerprint,
                    )
                    yield f"data: {chunk.model_dump_json()}\n\n"

                if completion.finish_reason is not None:
                    finish = CompletionStreamResponse(
                        id=request_id,
                        created=created,
                        model=model,
                        choices=[
                            CompletionStreamChoice(
                                index=idx,
                                text="",
                                finish_reason=_map_finish_reason(completion.finish_reason),
                            )
                        ],
                        system_fingerprint=fingerprint,
                    )
                    yield f"data: {finish.model_dump_json()}\n\n"

        if include_usage:
            usage_chunk = CompletionStreamResponse(
                id=request_id,
                created=created,
                model=model,
                choices=[],
                usage=UsageInfo(
                    prompt_tokens=total_prompt_tokens,
                    completion_tokens=total_completion_tokens,
                    total_tokens=total_prompt_tokens + total_completion_tokens,
                ),
                system_fingerprint=fingerprint,
            )
            yield f"data: {usage_chunk.model_dump_json()}\n\n"
    except Exception as exc:
        stream_failed = True
        logging.getLogger(__name__).exception("Error during completion streaming")
        # Emit a standard OpenAI error event before ending the stream
        error_payload = {
            "error": {
                "message": f"An error occurred during streaming: {exc}",
                "type": "server_error",
                "param": None,
                "code": "stream_error",
            }
        }
        yield f"data: {json.dumps(error_payload)}\n\n"
        raise
    except BaseException:
        stream_failed = True
        raise
    finally:
        with contextlib.suppress(Exception):
            await output_generator.aclose()
        if not stream_failed:
            yield "data: [DONE]\n\n"
