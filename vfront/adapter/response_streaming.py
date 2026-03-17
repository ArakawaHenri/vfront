"""Generate Responses API streaming events (SSE).

Follows the OpenAI Responses API streaming event protocol:
  response.created → response.in_progress →
  response.output_item.added → response.content_part.added →
  response.output_text.delta (×N) →
  response.output_text.done → response.content_part.done →
  response.output_item.done → response.completed
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Literal

from vfront.adapter.assistant_output import AssistantOutputStreamState
from vfront.protocol.response import (
    ContentPartAddedEvent,
    ContentPartDoneEvent,
    FunctionCallArgumentsDeltaEvent,
    FunctionCallArgumentsDoneEvent,
    FunctionToolCall,
    InputTokensDetails,
    OutputItemAddedEvent,
    OutputItemDoneEvent,
    OutputMessage,
    OutputText,
    OutputTokensDetails,
    ReasoningItem,
    ResponseCompletedEvent,
    ResponseCreatedEvent,
    ResponseCreateRequest,
    ResponseError,
    ResponseErrorEvent,
    ResponseFailedEvent,
    ResponseInProgressEvent,
    ResponseObject,
    ResponseUsage,
    TextDeltaEvent,
    TextDoneEvent,
)
from vfront.shared.engine.types import GenerateOutput


def _sse_line(event_type: str, data: str) -> str:
    """Format a single SSE event."""
    return f"event: {event_type}\ndata: {data}\n\n"


def build_response_shell(
    request: ResponseCreateRequest,
    response_id: str | None = None,
    created_at: float | None = None,
) -> ResponseObject:
    """Build an in-progress Response object shell."""
    now = created_at or time.time()
    resp_id = response_id or f"resp_{uuid.uuid4().hex[:24]}"
    return ResponseObject(
        id=resp_id,
        created_at=now,
        status="in_progress",
        model=request.model,
        output=[],
        metadata=request.metadata,
        instructions=request.instructions,
        temperature=request.temperature,
        top_p=request.top_p,
        max_output_tokens=request.max_output_tokens,
        stop=request.stop,
        frequency_penalty=request.frequency_penalty,
        presence_penalty=request.presence_penalty,
        text=request.text,
        tools=request.tools,
        tool_choice=request.tool_choice,
        parallel_tool_calls=(
            True if request.parallel_tool_calls is None else request.parallel_tool_calls
        ),
        store=request.store,
        previous_response_id=request.previous_response_id,
        truncation=request.truncation,
        user=request.user,
        background=request.background,
        top_logprobs=request.top_logprobs,
        reasoning=request.reasoning,
        service_tier=request.service_tier,
        max_tool_calls=request.max_tool_calls,
    )


def build_response_prelude_events(
    request: ResponseCreateRequest,
    response_id: str | None = None,
    created_at: float | None = None,
    initial_sequence_number: int = 0,
) -> tuple[list[str], int]:
    """Build the initial `response.created` and `response.in_progress` events."""
    response_shell = build_response_shell(
        request=request,
        response_id=response_id,
        created_at=created_at,
    )
    seq = initial_sequence_number
    events = [
        _sse_line(
            "response.created",
            ResponseCreatedEvent(
                response=response_shell,
                sequence_number=seq,
            ).model_dump_json(),
        ),
        _sse_line(
            "response.in_progress",
            ResponseInProgressEvent(
                response=response_shell,
                sequence_number=seq + 1,
            ).model_dump_json(),
        ),
    ]
    return events, seq + 2


def build_function_tool_call_events(
    tool_item: FunctionToolCall,
    output_index: int,
    initial_sequence_number: int,
    chunk_size: int = 32,
) -> tuple[list[str], int]:
    """Build the SSE event sequence for a function call output item.

    Emits: output_item.added → function_call_arguments.delta (×N) →
    function_call_arguments.done → output_item.done

    At least one delta event is always emitted (empty string for empty arguments),
    matching the OpenAI Responses streaming contract.
    """
    seq = initial_sequence_number
    item_id = tool_item.id or tool_item.call_id
    events = [
        _sse_line(
            "response.output_item.added",
            OutputItemAddedEvent(
                item=tool_item,
                output_index=output_index,
                sequence_number=seq,
            ).model_dump_json(),
        ),
    ]
    seq += 1

    # Emit argument chunks as delta events; always at least one delta.
    args = tool_item.arguments or ""
    chunk_starts = range(0, len(args), chunk_size) if args else range(0, 1)
    for i in chunk_starts:
        events.append(
            _sse_line(
                "response.function_call_arguments.delta",
                FunctionCallArgumentsDeltaEvent(
                    delta=args[i : i + chunk_size],
                    item_id=item_id,
                    output_index=output_index,
                    sequence_number=seq,
                ).model_dump_json(),
            )
        )
        seq += 1

    events.extend([
        _sse_line(
            "response.function_call_arguments.done",
            FunctionCallArgumentsDoneEvent(
                name=tool_item.name,
                arguments=tool_item.arguments,
                item_id=item_id,
                output_index=output_index,
                sequence_number=seq,
            ).model_dump_json(),
        ),
        _sse_line(
            "response.output_item.done",
            OutputItemDoneEvent(
                item=tool_item,
                output_index=output_index,
                sequence_number=seq + 1,
            ).model_dump_json(),
        ),
    ])
    return events, seq + 2


def build_response_failure_events(
    request: ResponseCreateRequest,
    message: str,
    response_id: str | None = None,
    created_at: float | None = None,
    initial_sequence_number: int = 0,
    code: str = "stream_error",
) -> tuple[list[str], int]:
    """Build `response.error` and `response.failed` events."""
    response_shell = build_response_shell(
        request=request,
        response_id=response_id,
        created_at=created_at,
    )
    seq = initial_sequence_number
    failed_response = response_shell.model_copy(
        update={
            "status": "failed",
            "error": ResponseError(
                code=code,
                message=message,
            ),
        }
    )
    events = [
        _sse_line(
            "response.error",
            ResponseErrorEvent(
                code=code,
                message=message,
                sequence_number=seq,
            ).model_dump_json(),
        ),
        _sse_line(
            "response.failed",
            ResponseFailedEvent(
                response=failed_response,
                sequence_number=seq + 1,
            ).model_dump_json(),
        ),
    ]
    return events, seq + 2


def build_response_completed_event(
    request: ResponseCreateRequest,
    output_items: list,
    prompt_tokens: int,
    output_tokens: int,
    sequence_number: int,
    response_id: str | None = None,
    created_at: float | None = None,
    reasoning_tokens: int = 0,
) -> tuple[str, ResponseObject]:
    """Build a `response.completed` event and return the response payload."""
    completed_response = build_response_shell(
        request=request,
        response_id=response_id,
        created_at=created_at,
    ).model_copy(
        update={
            "status": "completed",
            "output": output_items,
            "usage": ResponseUsage(
                input_tokens=prompt_tokens,
                output_tokens=output_tokens,
                total_tokens=prompt_tokens + output_tokens,
                input_tokens_details=InputTokensDetails(cached_tokens=0),
                output_tokens_details=OutputTokensDetails(reasoning_tokens=reasoning_tokens),
            ),
            "completed_at": time.time(),
        }
    )
    return (
        _sse_line(
            "response.completed",
            ResponseCompletedEvent(
                response=completed_response,
                sequence_number=sequence_number,
            ).model_dump_json(),
        ),
        completed_response,
    )


def format_response_completed_event(
    response: ResponseObject,
    sequence_number: int,
) -> str:
    """Format a `response.completed` SSE event for an existing response object."""
    return _sse_line(
        "response.completed",
        ResponseCompletedEvent(
            response=response,
            sequence_number=sequence_number,
        ).model_dump_json(),
    )


@dataclass(slots=True)
class _ActiveResponseItemState:
    kind: Literal["message", "tool_call"] | None = None
    output_index: int = 0
    item_id: str = ""
    content_index: int = 0
    text_parts: list[str] = field(default_factory=list)
    tool_call_index: int | None = None
    tool_call_name: str = ""
    tool_call_id: str = ""
    tool_call_argument_parts: list[str] = field(default_factory=list)


def _open_message_item(
    *,
    output_index: int,
    item_id: str,
    content_index: int,
    sequence_number: int,
) -> tuple[list[str], int]:
    events = [
        _sse_line(
            "response.output_item.added",
            OutputItemAddedEvent(
                item=OutputMessage(id=item_id, status="in_progress", content=[]),
                output_index=output_index,
                sequence_number=sequence_number,
            ).model_dump_json(),
        ),
        _sse_line(
            "response.content_part.added",
            ContentPartAddedEvent(
                part=OutputText(text="", annotations=[]),
                item_id=item_id,
                output_index=output_index,
                content_index=content_index,
                sequence_number=sequence_number + 1,
            ).model_dump_json(),
        ),
    ]
    return events, sequence_number + 2


def _close_message_item(
    *,
    output_index: int,
    item_id: str,
    content_index: int,
    text: str,
    sequence_number: int,
) -> tuple[list[str], OutputMessage, int]:
    final_part = OutputText(text=text, annotations=[])
    completed = OutputMessage(
        id=item_id,
        status="completed",
        content=[final_part],
    )
    events = [
        _sse_line(
            "response.output_text.done",
            TextDoneEvent(
                text=text,
                item_id=item_id,
                output_index=output_index,
                content_index=content_index,
                sequence_number=sequence_number,
            ).model_dump_json(),
        ),
        _sse_line(
            "response.content_part.done",
            ContentPartDoneEvent(
                part=final_part,
                item_id=item_id,
                output_index=output_index,
                content_index=content_index,
                sequence_number=sequence_number + 1,
            ).model_dump_json(),
        ),
        _sse_line(
            "response.output_item.done",
            OutputItemDoneEvent(
                item=completed,
                output_index=output_index,
                sequence_number=sequence_number + 2,
            ).model_dump_json(),
        ),
    ]
    return events, completed, sequence_number + 3


def _open_tool_call_item(
    *,
    output_index: int,
    item_id: str,
    call_id: str,
    name: str,
    sequence_number: int,
) -> tuple[list[str], int]:
    item = FunctionToolCall(
        id=item_id,
        call_id=call_id,
        name=name,
        arguments="",
        status="in_progress",
    )
    events = [
        _sse_line(
            "response.output_item.added",
            OutputItemAddedEvent(
                item=item,
                output_index=output_index,
                sequence_number=sequence_number,
            ).model_dump_json(),
        )
    ]
    return events, sequence_number + 1


def _close_tool_call_item(
    *,
    output_index: int,
    item_id: str,
    call_id: str,
    name: str,
    arguments: str,
    sequence_number: int,
) -> tuple[list[str], FunctionToolCall, int]:
    completed = FunctionToolCall(
        id=item_id,
        call_id=call_id,
        name=name,
        arguments=arguments,
        status="completed",
    )
    events = [
        _sse_line(
            "response.function_call_arguments.done",
            FunctionCallArgumentsDoneEvent(
                name=name,
                arguments=arguments,
                item_id=item_id,
                output_index=output_index,
                sequence_number=sequence_number,
            ).model_dump_json(),
        ),
        _sse_line(
            "response.output_item.done",
            OutputItemDoneEvent(
                item=completed,
                output_index=output_index,
                sequence_number=sequence_number + 1,
            ).model_dump_json(),
        ),
    ]
    return events, completed, sequence_number + 2


async def generate_response_stream_events(
    engine_stream: AsyncIterator[GenerateOutput],
    request: ResponseCreateRequest,
    response_id: str | None = None,
    created_at: float | None = None,
    prompt_tokens: int = 0,
    prefix_tool_calls: list[FunctionToolCall] | None = None,
    initial_sequence_number: int = 0,
    emit_prelude: bool = True,
    emit_prefix_tool_call_events: bool = True,
) -> AsyncIterator[str]:
    """Yield SSE-formatted streaming events from a vLLM engine stream.

    Each yielded string is a complete SSE event block (event + data + blank line).
    Supports reasoning items, text output, and function tool calls.
    """
    now = created_at or time.time()
    resp_id = response_id or f"resp_{uuid.uuid4().hex[:24]}"
    response_shell = build_response_shell(
        request=request,
        response_id=resp_id,
        created_at=now,
    )
    seq = initial_sequence_number

    if emit_prelude:
        prelude_events, seq = build_response_prelude_events(
            request=request,
            response_id=resp_id,
            created_at=now,
            initial_sequence_number=seq,
        )
        for event in prelude_events:
            yield event

    emitted_prefix_tool_calls = list(prefix_tool_calls or [])

    # Accumulate output for final response
    accumulated_text = ""
    accumulated_reasoning = ""
    total_completion_tokens = 0
    total_reasoning_tokens = 0
    previous_reasoning_len = 0
    generated_tool_items: list[FunctionToolCall] = []
    completed_output_items = list(emitted_prefix_tool_calls)
    output_index = len(emitted_prefix_tool_calls)
    has_tools = bool(request.tools)
    text_state = AssistantOutputStreamState(has_tool_definitions=has_tools)
    semantic_emitted = False
    active_item = _ActiveResponseItemState(output_index=output_index)

    # ── Reasoning output item (emitted first if present) ──
    reasoning_item_emitted = False
    reasoning_id = f"rs_{uuid.uuid4().hex[:24]}"

    # ── Text output item ──
    msg_id = f"msg_{uuid.uuid4().hex[:24]}"
    text_item_emitted = False
    content_index = 0

    if emit_prefix_tool_call_events:
        for idx, tool_item in enumerate(emitted_prefix_tool_calls):
            tool_events, seq = build_function_tool_call_events(
                tool_item=tool_item,
                output_index=idx,
                initial_sequence_number=seq,
            )
            for event in tool_events:
                yield event

    try:
        async for vllm_output in engine_stream:
            for completion in vllm_output.outputs:
                current_text = completion.text
                current_token_count = len(completion.token_ids)

                # ── Handle reasoning content ──
                reasoning_text = completion.reasoning_text
                if reasoning_text and reasoning_text != accumulated_reasoning:
                    if not reasoning_item_emitted:
                        # Emit reasoning output item
                        reasoning_item = ReasoningItem(id=reasoning_id, summary=[])
                        yield _sse_line(
                            "response.output_item.added",
                            OutputItemAddedEvent(
                                item=reasoning_item,
                                output_index=output_index,
                                sequence_number=seq,
                            ).model_dump_json(),
                        )
                        seq += 1
                        reasoning_item_emitted = True

                    # Emit reasoning text delta
                    reasoning_delta = reasoning_text[previous_reasoning_len:]
                    if reasoning_delta:
                        yield _sse_line(
                            "response.output_text.delta",
                            TextDeltaEvent(
                                delta=reasoning_delta,
                                item_id=reasoning_id,
                                output_index=output_index,
                                content_index=0,
                                sequence_number=seq,
                            ).model_dump_json(),
                        )
                        seq += 1

                    previous_reasoning_len = len(reasoning_text)
                    accumulated_reasoning = reasoning_text

                # ── Handle text content ──
                delta_text = text_state.update(current_text, completion.finish_reason)
                should_stream_text = text_state.kind == "text"

                semantic_delta = completion.semantic
                if semantic_delta and (
                    semantic_delta.content_delta or semantic_delta.tool_call_deltas
                ):
                    semantic_emitted = True

                    if reasoning_item_emitted:
                        reasoning_item = ReasoningItem(
                            id=reasoning_id,
                            summary=[{"type": "summary_text", "text": accumulated_reasoning}],
                        )
                        yield _sse_line(
                            "response.output_item.done",
                            OutputItemDoneEvent(
                                item=reasoning_item,
                                output_index=active_item.output_index,
                                sequence_number=seq,
                            ).model_dump_json(),
                        )
                        seq += 1
                        completed_output_items.append(reasoning_item)
                        active_item.output_index += 1
                        total_reasoning_tokens = max(1, len(accumulated_reasoning) // 4)
                        reasoning_item_emitted = False

                    for tool_delta in semantic_delta.tool_call_deltas:
                        if active_item.kind == "message":
                            close_events, completed_msg, seq = _close_message_item(
                                output_index=active_item.output_index,
                                item_id=active_item.item_id,
                                content_index=active_item.content_index,
                                text="".join(active_item.text_parts),
                                sequence_number=seq,
                            )
                            for event in close_events:
                                yield event
                            completed_output_items.append(completed_msg)
                            active_item = _ActiveResponseItemState(
                                output_index=active_item.output_index + 1
                            )

                        if (
                            active_item.kind == "tool_call"
                            and active_item.tool_call_index != tool_delta.index
                        ):
                            close_events, completed_tool, seq = _close_tool_call_item(
                                output_index=active_item.output_index,
                                item_id=active_item.item_id,
                                call_id=active_item.tool_call_id,
                                name=active_item.tool_call_name,
                                arguments="".join(active_item.tool_call_argument_parts),
                                sequence_number=seq,
                            )
                            for event in close_events:
                                yield event
                            completed_output_items.append(completed_tool)
                            active_item = _ActiveResponseItemState(
                                output_index=active_item.output_index + 1
                            )

                        if active_item.kind != "tool_call":
                            if tool_delta.name is None:
                                continue
                            item_id = f"fc_{uuid.uuid4().hex[:24]}"
                            call_id = f"call_{uuid.uuid4().hex[:24]}"
                            open_events, seq = _open_tool_call_item(
                                output_index=active_item.output_index,
                                item_id=item_id,
                                call_id=call_id,
                                name=tool_delta.name,
                                sequence_number=seq,
                            )
                            for event in open_events:
                                yield event
                            active_item = _ActiveResponseItemState(
                                kind="tool_call",
                                output_index=active_item.output_index,
                                item_id=item_id,
                                tool_call_index=tool_delta.index,
                                tool_call_name=tool_delta.name,
                                tool_call_id=call_id,
                            )

                        if tool_delta.arguments_delta:
                            yield _sse_line(
                                "response.function_call_arguments.delta",
                                FunctionCallArgumentsDeltaEvent(
                                    delta=tool_delta.arguments_delta,
                                    item_id=active_item.item_id,
                                    output_index=active_item.output_index,
                                    sequence_number=seq,
                                ).model_dump_json(),
                            )
                            seq += 1
                            active_item.tool_call_argument_parts.append(
                                tool_delta.arguments_delta
                            )

                    if semantic_delta.content_delta:
                        if active_item.kind == "tool_call":
                            close_events, completed_tool, seq = _close_tool_call_item(
                                output_index=active_item.output_index,
                                item_id=active_item.item_id,
                                call_id=active_item.tool_call_id,
                                name=active_item.tool_call_name,
                                arguments="".join(active_item.tool_call_argument_parts),
                                sequence_number=seq,
                            )
                            for event in close_events:
                                yield event
                            completed_output_items.append(completed_tool)
                            active_item = _ActiveResponseItemState(
                                output_index=active_item.output_index + 1
                            )

                        if active_item.kind != "message":
                            item_id = f"msg_{uuid.uuid4().hex[:24]}"
                            open_events, seq = _open_message_item(
                                output_index=active_item.output_index,
                                item_id=item_id,
                                content_index=0,
                                sequence_number=seq,
                            )
                            for event in open_events:
                                yield event
                            active_item = _ActiveResponseItemState(
                                kind="message",
                                output_index=active_item.output_index,
                                item_id=item_id,
                                content_index=0,
                            )

                        yield _sse_line(
                            "response.output_text.delta",
                            TextDeltaEvent(
                                delta=semantic_delta.content_delta,
                                item_id=active_item.item_id,
                                output_index=active_item.output_index,
                                content_index=active_item.content_index,
                                sequence_number=seq,
                            ).model_dump_json(),
                        )
                        seq += 1
                        active_item.text_parts.append(semantic_delta.content_delta)

                    accumulated_text = current_text
                    total_completion_tokens = current_token_count
                    continue

                if delta_text and not text_item_emitted:
                    # Close reasoning item if it was open
                    if reasoning_item_emitted:
                        yield _sse_line(
                            "response.output_item.done",
                            OutputItemDoneEvent(
                                item=ReasoningItem(
                                    id=reasoning_id,
                                    summary=[
                                        {"type": "summary_text", "text": accumulated_reasoning}
                                    ],
                                ),
                                output_index=output_index,
                                sequence_number=seq,
                            ).model_dump_json(),
                        )
                        seq += 1
                        output_index += 1
                        total_reasoning_tokens = max(1, len(accumulated_reasoning) // 4)

                    # Emit text output item
                    msg_item = OutputMessage(id=msg_id, status="in_progress", content=[])
                    yield _sse_line(
                        "response.output_item.added",
                        OutputItemAddedEvent(
                            item=msg_item,
                            output_index=output_index,
                            sequence_number=seq,
                        ).model_dump_json(),
                    )
                    seq += 1

                    empty_text_part = OutputText(text="", annotations=[])
                    yield _sse_line(
                        "response.content_part.added",
                        ContentPartAddedEvent(
                            part=empty_text_part,
                            item_id=msg_id,
                            output_index=output_index,
                            content_index=content_index,
                            sequence_number=seq,
                        ).model_dump_json(),
                    )
                    seq += 1
                    text_item_emitted = True

                if delta_text and should_stream_text and text_item_emitted:
                    yield _sse_line(
                        "response.output_text.delta",
                        TextDeltaEvent(
                            delta=delta_text,
                            item_id=msg_id,
                            output_index=output_index,
                            content_index=content_index,
                            sequence_number=seq,
                        ).model_dump_json(),
                    )
                    seq += 1

                accumulated_text = current_text
                total_completion_tokens = current_token_count
    except Exception as exc:
        logging.getLogger(__name__).exception("Error during response streaming")
        failure_events, _seq = build_response_failure_events(
            request=request,
            message=str(exc),
            response_id=resp_id,
            created_at=now,
            initial_sequence_number=seq,
        )
        for event in failure_events:
            yield event
        return
    finally:
        # Ensure vLLM stops generating on client disconnect or error
        if hasattr(engine_stream, "aclose"):
            await engine_stream.aclose()

    final_output_items: list = list(completed_output_items)
    if semantic_emitted:
        if reasoning_item_emitted:
            reasoning_item = ReasoningItem(
                id=reasoning_id,
                summary=[{"type": "summary_text", "text": accumulated_reasoning}],
            )
            yield _sse_line(
                "response.output_item.done",
                OutputItemDoneEvent(
                    item=reasoning_item,
                    output_index=active_item.output_index,
                    sequence_number=seq,
                ).model_dump_json(),
            )
            seq += 1
            final_output_items.append(reasoning_item)
            active_item.output_index += 1
            total_reasoning_tokens = max(1, len(accumulated_reasoning) // 4)

        if active_item.kind == "message":
            close_events, completed_msg, seq = _close_message_item(
                output_index=active_item.output_index,
                item_id=active_item.item_id,
                content_index=active_item.content_index,
                text="".join(active_item.text_parts),
                sequence_number=seq,
            )
            for event in close_events:
                yield event
            final_output_items.append(completed_msg)
        elif active_item.kind == "tool_call":
            close_events, completed_tool, seq = _close_tool_call_item(
                output_index=active_item.output_index,
                item_id=active_item.item_id,
                call_id=active_item.tool_call_id,
                name=active_item.tool_call_name,
                arguments="".join(active_item.tool_call_argument_parts),
                sequence_number=seq,
            )
            for event in close_events:
                yield event
            final_output_items.append(completed_tool)
    else:
        # ── Close reasoning item if it was the only output ──
        if reasoning_item_emitted and not text_item_emitted:
            reasoning_item = ReasoningItem(
                id=reasoning_id,
                summary=[{"type": "summary_text", "text": accumulated_reasoning}],
            )
            yield _sse_line(
                "response.output_item.done",
                OutputItemDoneEvent(
                    item=reasoning_item,
                    output_index=output_index,
                    sequence_number=seq,
                ).model_dump_json(),
            )
            seq += 1
            output_index += 1
            total_reasoning_tokens = max(1, len(accumulated_reasoning) // 4)

        next_output_index = output_index

        # ── Detect tool calls in accumulated text ──
        normalized, deferred_text = text_state.finalize()
        if request.tools and normalized.kind == "tool_calls" and normalized.tool_calls:
            raise RuntimeError(
                "Engine-side semantic parser was expected to provide tool-call deltas, "
                "but only legacy post-hoc tool-call text was available."
            )
        if normalized.kind == "tool_calls" and normalized.tool_calls:
            for tc in normalized.tool_calls:
                tc_id = f"fc_{uuid.uuid4().hex[:24]}"
                generated_tool_items.append(
                    FunctionToolCall(
                        id=tc_id,
                        name=tc.function.name,
                        arguments=tc.function.arguments,
                        call_id=tc.id,
                        status="completed",
                    )
                )
        elif deferred_text and not text_item_emitted:
            msg_item = OutputMessage(id=msg_id, status="in_progress", content=[])
            yield _sse_line(
                "response.output_item.added",
                OutputItemAddedEvent(
                    item=msg_item,
                    output_index=next_output_index,
                    sequence_number=seq,
                ).model_dump_json(),
            )
            seq += 1

            final_text_part = OutputText(text="", annotations=[])
            yield _sse_line(
                "response.content_part.added",
                ContentPartAddedEvent(
                    part=final_text_part,
                    item_id=msg_id,
                    output_index=next_output_index,
                    content_index=content_index,
                    sequence_number=seq,
                ).model_dump_json(),
            )
            seq += 1

            yield _sse_line(
                "response.output_text.delta",
                TextDeltaEvent(
                    delta=deferred_text,
                    item_id=msg_id,
                    output_index=next_output_index,
                    content_index=content_index,
                    sequence_number=seq,
                ).model_dump_json(),
            )
            seq += 1

            final_text_part = OutputText(text=deferred_text, annotations=[])
            yield _sse_line(
                "response.output_text.done",
                TextDoneEvent(
                    text=deferred_text,
                    item_id=msg_id,
                    output_index=next_output_index,
                    content_index=content_index,
                    sequence_number=seq,
                ).model_dump_json(),
            )
            seq += 1

            yield _sse_line(
                "response.content_part.done",
                ContentPartDoneEvent(
                    part=final_text_part,
                    item_id=msg_id,
                    output_index=next_output_index,
                    content_index=content_index,
                    sequence_number=seq,
                ).model_dump_json(),
            )
            seq += 1

            completed_msg = OutputMessage(
                id=msg_id,
                status="completed",
                content=[final_text_part],
            )
            yield _sse_line(
                "response.output_item.done",
                OutputItemDoneEvent(
                    item=completed_msg,
                    output_index=next_output_index,
                    sequence_number=seq,
                ).model_dump_json(),
            )
            seq += 1
            next_output_index += 1

        # ── Close text output item ──
        if text_item_emitted:
            yield _sse_line(
                "response.output_text.done",
                TextDoneEvent(
                    text=accumulated_text,
                    item_id=msg_id,
                    output_index=output_index,
                    content_index=content_index,
                    sequence_number=seq,
                ).model_dump_json(),
            )
            seq += 1

            final_text_part = OutputText(text=accumulated_text, annotations=[])
            yield _sse_line(
                "response.content_part.done",
                ContentPartDoneEvent(
                    part=final_text_part,
                    item_id=msg_id,
                    output_index=output_index,
                    content_index=content_index,
                    sequence_number=seq,
                ).model_dump_json(),
            )
            seq += 1

            completed_msg = OutputMessage(
                id=msg_id,
                status="completed",
                content=[final_text_part],
            )
            yield _sse_line(
                "response.output_item.done",
                OutputItemDoneEvent(
                    item=completed_msg,
                    output_index=output_index,
                    sequence_number=seq,
                ).model_dump_json(),
            )
            seq += 1
            next_output_index += 1

        # ── Emit function call items (if detected) ──
        for i, tool_item in enumerate(generated_tool_items):
            fc_output_index = next_output_index + i
            tool_events, seq = build_function_tool_call_events(
                tool_item=tool_item,
                output_index=fc_output_index,
                initial_sequence_number=seq,
            )
            for event in tool_events:
                yield event

        if reasoning_item_emitted:
            final_output_items.append(
                ReasoningItem(
                    id=reasoning_id,
                    summary=[{"type": "summary_text", "text": accumulated_reasoning}],
                )
            )
        if text_item_emitted or deferred_text:
            final_text_part = OutputText(text=deferred_text or accumulated_text, annotations=[])
            final_output_items.append(
                OutputMessage(
                    id=msg_id,
                    status="completed",
                    content=[final_text_part],
                )
            )
        final_output_items.extend(generated_tool_items)

    # ── response.completed ──
    usage = ResponseUsage(
        input_tokens=prompt_tokens,
        output_tokens=total_completion_tokens,
        total_tokens=prompt_tokens + total_completion_tokens,
        input_tokens_details=InputTokensDetails(cached_tokens=0),
        output_tokens_details=OutputTokensDetails(reasoning_tokens=total_reasoning_tokens),
    )
    completed_response = response_shell.model_copy(
        update={
            "status": "completed",
            "output": final_output_items,
            "usage": usage,
            "completed_at": time.time(),
        }
    )
    yield _sse_line(
        "response.completed",
        ResponseCompletedEvent(response=completed_response, sequence_number=seq).model_dump_json(),
    )
