"""Translate between Responses API format and engine format.

Handles:
- Responses input → chat messages
- GenerateOutput → ResponseObject
- ResponseCreateRequest → GenerateParams
"""

from __future__ import annotations

import json
import time
import uuid
from copy import deepcopy
from typing import Any

from vfront.adapter.assistant_output import normalize_assistant_output
from vfront.adapter.tool_calls import limit_parallel_tool_calls, serialize_tool_call_arguments
from vfront.frontend.service.mcp.executor import ToolExecutionResult
from vfront.protocol.common import ToolCall
from vfront.protocol.response import (
    FunctionCallOutput,
    FunctionToolCall,
    FunctionToolDefinition,
    IncompleteDetails,
    InputContent,
    InputTokensDetails,
    OutputMessage,
    OutputText,
    OutputTokensDetails,
    ReasoningItem,
    ResponseCreateRequest,
    ResponseFunctionToolCallOutputResource,
    ResponseInputImageContent,
    ResponseInputItem,
    ResponseInputMessageResource,
    ResponseInputTextContent,
    ResponseObject,
    ResponseOutputTextContent,
    ResponseTextConfig,
    ResponseUsage,
)
from vfront.shared.engine.client import EngineClient
from vfront.shared.engine.reasoning import ReasoningEffort, extract_reasoning_effort
from vfront.shared.engine.tokenization import tokenize_chat_messages
from vfront.shared.engine.types import GenerateOutput, GenerateParams

RESPONSE_CONTEXT_NAMESPACE = "response_context"
_INSTRUCTIONS_MESSAGE_KEY = "__vfront_instructions__"


def input_to_messages(
    request: ResponseCreateRequest,
    previous_response: ResponseObject | None = None,
    previous_input_messages: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Convert Responses API input to a list of chat messages for the tokenizer.

    Returns a list of {"role": ..., "content": ...} dicts suitable for
    tokenizer.apply_chat_template().
    """
    messages: list[dict[str, Any]] = []

    # Prepend history from previous_response if provided
    if previous_input_messages is not None:
        messages.extend(deepcopy(previous_input_messages))
    elif previous_response is not None:
        messages.extend(_response_to_history_messages(previous_response, include_instructions=False))

    # System / developer instructions
    if request.instructions:
        messages.append(
            {
                "role": "system",
                "content": request.instructions,
                _INSTRUCTIONS_MESSAGE_KEY: True,
            }
        )

    # Parse input
    if isinstance(request.input, str):
        messages.append({"role": "user", "content": request.input})
    elif isinstance(request.input, list):
        for item in request.input:
            if isinstance(item, ResponseInputItem):
                messages.append(_input_item_to_message(item))
            elif isinstance(item, FunctionCallOutput):
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": item.call_id,
                        "content": item.output,
                    }
                )
            elif isinstance(item, dict):
                msg = _dict_input_to_message(item)
                if msg:
                    messages.append(msg)
    return messages


def merge_response_tools(
    request_tools: list[FunctionToolDefinition] | None,
    mcp_tool_defs: list[dict[str, Any]] | None = None,
) -> list[FunctionToolDefinition] | None:
    """Merge user-provided tools with server-injected MCP tools."""
    all_tools = list(request_tools) if request_tools else []
    for mcp_tool in mcp_tool_defs or []:
        fn = mcp_tool["function"]
        all_tools.append(
            FunctionToolDefinition(
                name=fn["name"],
                description=fn.get("description", ""),
                parameters=fn.get("parameters"),
            )
        )
    return all_tools or None


def response_tools_to_chat_tools(
    tools: list[FunctionToolDefinition] | None,
) -> list[dict[str, Any]] | None:
    if not tools:
        return None
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            },
        }
        for tool in tools
    ]


def response_tool_choice_to_chat_tool_choice(tool_choice: Any) -> Any:
    if not isinstance(tool_choice, dict):
        return tool_choice
    if tool_choice.get("type") != "function":
        return tool_choice
    if "function" in tool_choice:
        return tool_choice
    name = tool_choice.get("name")
    if not name:
        return tool_choice
    return {"type": "function", "function": {"name": name}}


def build_generation_messages(
    conversation_messages: list[dict[str, Any]],
    tools: list[FunctionToolDefinition] | None,
) -> list[dict[str, Any]]:
    """Build model-facing messages from persisted conversation messages."""
    generation_messages: list[dict[str, Any]] = []
    for message in deepcopy(conversation_messages):
        message.pop(_INSTRUCTIONS_MESSAGE_KEY, None)
        generation_messages.append(message)
    inject_tools_prompt(generation_messages, tools)
    return generation_messages


async def truncate_messages_for_context(
    messages: list[dict[str, Any]],
    max_input_tokens: int,
    engine_client: EngineClient,
    tools: list[FunctionToolDefinition] | None,
    reasoning_effort: ReasoningEffort | None = None,
) -> list[dict[str, Any]]:
    """Remove oldest non-system messages until the generation input fits."""
    if max_input_tokens <= 0:
        return messages
    token_ids = await tokenize_chat_messages(
        engine_client,
        build_generation_messages(messages, tools),
        reasoning_effort=reasoning_effort,
    )
    if len(token_ids) <= max_input_tokens:
        return messages

    system = [message for message in messages if message["role"] == "system"]
    conversation = [message for message in messages if message["role"] != "system"]

    while conversation:
        conversation.pop(0)
        trial = system + conversation
        token_ids = await tokenize_chat_messages(
            engine_client,
            build_generation_messages(trial if trial else system, tools),
            reasoning_effort=reasoning_effort,
        )
        if len(token_ids) <= max_input_tokens:
            return trial if trial else system

    return system


def messages_for_response_continuation(
    conversation_messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Strip request-level instructions before persisting continuation state."""
    continuation_messages: list[dict[str, Any]] = []
    for message in deepcopy(conversation_messages):
        if message.pop(_INSTRUCTIONS_MESSAGE_KEY, False):
            continue
        continuation_messages.append(message)
    return continuation_messages


def _response_item_id(prefix: str, response_id: str, index: int) -> str:
    return f"{prefix}_{response_id}_{index}"


def _stringify_tool_output(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return json.dumps(content, ensure_ascii=False)
    return str(content)


def _normalize_input_item_content(content: Any) -> list[object]:
    if content is None:
        return []
    if isinstance(content, str):
        return [ResponseInputTextContent(text=content)]
    if not isinstance(content, list):
        return [ResponseInputTextContent(text=str(content))]

    normalized: list[object] = []
    for item in content:
        if isinstance(item, str):
            normalized.append(ResponseInputTextContent(text=item))
            continue
        if isinstance(
            item,
            ResponseInputTextContent | ResponseInputImageContent | ResponseOutputTextContent,
        ):
            normalized.append(item)
            continue
        if isinstance(item, dict):
            item_type = item.get("type")
            if item_type == "input_image":
                normalized.append(ResponseInputImageContent.model_validate(item))
            elif item_type == "output_text":
                normalized.append(ResponseOutputTextContent.model_validate(item))
            else:
                normalized.append(ResponseInputTextContent.model_validate(item))
            continue
        normalized.append(ResponseInputTextContent(text=str(item)))
    return normalized


def _normalize_output_item_content(content: Any) -> list[object]:
    if content is None:
        return []
    if isinstance(content, str):
        return [OutputText(text=content)]
    if not isinstance(content, list):
        return [OutputText(text=str(content))]

    normalized: list[object] = []
    for item in content:
        if isinstance(item, str):
            normalized.append(OutputText(text=item))
            continue
        if isinstance(item, OutputText):
            normalized.append(item)
            continue
        if isinstance(item, dict):
            if item.get("type") == "output_text":
                normalized.append(OutputText.model_validate(item))
            elif item.get("type") == "input_text":
                normalized.append(OutputText(text=str(item.get("text") or "")))
            else:
                normalized.append(OutputText(text=json.dumps(item, ensure_ascii=False)))
            continue
        normalized.append(OutputText(text=str(item)))
    return normalized


def _conversation_messages_to_input_items(
    *,
    response_id: str,
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for message in messages:
        role = message.get("role")
        if role == "tool":
            items.append(
                ResponseFunctionToolCallOutputResource(
                    id=_response_item_id("fco", response_id, len(items)),
                    call_id=str(message.get("tool_call_id") or ""),
                    output=_stringify_tool_output(message.get("content")),
                ).model_dump()
            )
            continue

        tool_calls = message.get("tool_calls") or []
        for tool_call in tool_calls:
            function = tool_call.get("function") or {}
            items.append(
                FunctionToolCall(
                    id=_response_item_id("fc", response_id, len(items)),
                    call_id=str(
                        tool_call.get("id") or _response_item_id("call", response_id, len(items))
                    ),
                    name=str(function.get("name") or ""),
                    arguments=serialize_tool_call_arguments(function.get("arguments")),
                    status="completed",
                ).model_dump()
            )

        content = message.get("content")
        if role == "assistant" and content is not None:
            items.append(
                OutputMessage(
                    id=_response_item_id("msg", response_id, len(items)),
                    content=_normalize_output_item_content(content),
                ).model_dump()
            )
            continue

        if role in {"user", "system", "developer"} and content is not None:
            items.append(
                ResponseInputMessageResource(
                    id=_response_item_id("msg", response_id, len(items)),
                    role=role,
                    content=_normalize_input_item_content(content),
                ).model_dump()
            )
    return items


def inject_tools_prompt(
    messages: list[dict[str, Any]],
    tools: list[FunctionToolDefinition] | None,
) -> list[dict[str, Any]]:
    """Append the function-calling prompt to the system message when tools exist."""
    if not tools:
        return messages

    tools_prompt = build_tools_prompt(tools)
    if not tools_prompt:
        return messages

    has_system = any(message["role"] == "system" for message in messages)
    if has_system:
        for message in messages:
            if message["role"] == "system":
                message["content"] = message["content"] + "\n\n" + tools_prompt
                break
    else:
        messages.insert(0, {"role": "system", "content": tools_prompt})

    return messages


def _input_item_to_message(item: ResponseInputItem) -> dict[str, Any]:
    """Convert a ResponseInputItem to a chat message dict."""
    content = _extract_content_parts(item.content)
    role = "system" if item.role == "developer" else item.role
    message: dict[str, Any] = {"role": role, "content": content}
    if item.model_extra:
        tool_calls = item.model_extra.get("tool_calls")
        if tool_calls is not None:
            message["tool_calls"] = deepcopy(tool_calls)
        tool_call_id = item.model_extra.get("tool_call_id")
        if tool_call_id is not None:
            message["tool_call_id"] = tool_call_id
    return message


def _dict_input_to_message(item: dict) -> dict[str, Any] | None:
    """Convert a raw dict input item to a chat message dict."""
    item_type = item.get("type", "message")
    if item_type == "function_call_output":
        return {
            "role": "tool",
            "tool_call_id": item.get("call_id"),
            "content": item.get("output", ""),
        }
    role = item.get("role", "user")
    if role == "developer":
        role = "system"
    content = item.get("content", "")
    if isinstance(content, list):
        content = _extract_content_parts(content)
    elif content is not None and not isinstance(content, str):
        content = str(content)
    message: dict[str, Any] = {"role": role, "content": content}
    tool_calls = item.get("tool_calls")
    if tool_calls is not None:
        message["tool_calls"] = deepcopy(tool_calls)
    tool_call_id = item.get("tool_call_id")
    if tool_call_id is not None:
        message["tool_call_id"] = tool_call_id
    return message


def _extract_content_parts(
    content: str | list[InputContent | Any],
) -> str | list[dict[str, Any]]:
    """Extract content as plain text or an OpenAI content-part list.

    Returns a plain string when the content contains no images (backwards
    compatible with text-only tokenizer paths).  Returns a list of
    ``{"type": "text"|"image_url", ...}`` dicts when images are present so
    that ``apply_chat_template`` can process multimodal inputs.
    """
    if isinstance(content, str):
        return content

    has_image = any(
        isinstance(part, ResponseInputImageContent)
        or (isinstance(part, dict) and part.get("type") == "image_url")
        for part in content
    )

    if not has_image:
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif hasattr(part, "text"):
                parts.append(part.text)
            elif isinstance(part, dict):
                parts.append(part.get("text", ""))
        return "".join(parts)

    result: list[dict[str, Any]] = []
    for part in content:
        if isinstance(part, str):
            result.append({"type": "text", "text": part})
        elif isinstance(part, ResponseInputImageContent):
            if part.image_url is None:
                continue
            image_url_obj: dict[str, Any] = {"url": part.image_url}
            if part.detail is not None:
                image_url_obj["detail"] = part.detail
            result.append({"type": "image_url", "image_url": image_url_obj})
        elif isinstance(part, dict):
            if part.get("type") == "image_url":
                result.append(part)
            else:
                result.append({"type": "text", "text": part.get("text", "")})
        elif hasattr(part, "text"):
            result.append({"type": "text", "text": part.text})
    return result


def _extract_content_text(content: str | list[InputContent | Any]) -> str:
    """Extract plain text from content (string or content part list)."""
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for part in content:
        if isinstance(part, str):
            parts.append(part)
        elif hasattr(part, "text"):
            parts.append(part.text)
        elif isinstance(part, dict):
            parts.append(part.get("text", ""))
    return "".join(parts)


def _response_to_history_messages(
    response: ResponseObject,
    include_instructions: bool = True,
) -> list[dict[str, Any]]:
    """Extract message history from a stored ResponseObject for multi-turn."""
    messages: list[dict[str, Any]] = []
    if include_instructions and response.instructions:
        messages.append({"role": "system", "content": response.instructions})
    pending_tool_calls: list[dict[str, Any]] = []
    for item in response.output:
        if isinstance(item, FunctionToolCall):
            pending_tool_calls.append(
                {
                    "id": item.call_id,
                    "type": "function",
                    "function": {
                        "name": item.name,
                        "arguments": item.arguments,
                    },
                }
            )
            continue
        if isinstance(item, OutputMessage):
            text_parts: list[str] = []
            for content in item.content:
                if isinstance(content, OutputText):
                    text_parts.append(content.text)
            if text_parts or pending_tool_calls:
                assistant_message: dict[str, Any] = {
                    "role": "assistant",
                    "content": "".join(text_parts) if text_parts else None,
                }
                if pending_tool_calls:
                    assistant_message["tool_calls"] = pending_tool_calls
                messages.append(assistant_message)
                pending_tool_calls = []
    if pending_tool_calls:
        messages.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": pending_tool_calls,
            }
        )
    return messages


def response_request_to_generate_params(
    request: ResponseCreateRequest,
) -> GenerateParams:
    """Convert a ResponseCreateRequest to GenerateParams."""
    params = GenerateParams(chat_template_applied=True)

    params.temperature = request.temperature if request.temperature is not None else 1.0
    if request.top_p is not None:
        params.top_p = request.top_p
    if request.max_output_tokens is not None:
        params.max_tokens = request.max_output_tokens
    if request.stop is not None:
        params.stop = request.stop if isinstance(request.stop, list) else [request.stop]
    if request.presence_penalty is not None:
        params.presence_penalty = request.presence_penalty
    if request.frequency_penalty is not None:
        params.frequency_penalty = request.frequency_penalty
    if request.top_logprobs is not None:
        params.logprobs = request.top_logprobs
    params.reasoning_effort = extract_reasoning_effort(request.reasoning)

    # Structured outputs via text.format
    if request.text is not None:
        _apply_text_format(request.text, params)

    return params


def _apply_text_format(text_config: ResponseTextConfig, params: GenerateParams) -> None:
    """Apply text.format settings to GenerateParams."""
    fmt = text_config.format
    if fmt.type == "json_object":
        params.json_object = True
    elif fmt.type == "json_schema" and fmt.json_schema is not None:
        params.json_schema = fmt.json_schema.schema_ or {}


def request_output_to_response(
    engine_output: GenerateOutput,
    request: ResponseCreateRequest,
    response_id: str | None = None,
    created_at: float | None = None,
) -> ResponseObject:
    """Convert a GenerateOutput to a ResponseObject."""
    now = created_at or time.time()
    resp_id = response_id or f"resp_{uuid.uuid4().hex[:24]}"

    output_items = []
    total_completion_tokens = 0
    total_reasoning_tokens = 0

    for completion in engine_output.outputs:
        total_completion_tokens += len(completion.token_ids)
        has_tools = bool(request.tools)
        text = completion.text

        # Emit reasoning item if present
        if completion.reasoning_text:
            reasoning_id = f"rs_{uuid.uuid4().hex[:24]}"
            output_items.append(
                ReasoningItem(
                    id=reasoning_id,
                    summary=[{"type": "summary_text", "text": completion.reasoning_text}],
                )
            )
            # Estimate reasoning tokens (heuristic: chars / 4)
            total_reasoning_tokens += max(1, len(completion.reasoning_text) // 4)

        # Check for function tool calls
        if completion.parsed_tool_calls is not None:
            for tool in limit_parallel_tool_calls(
                completion.parsed_tool_calls,
                parallel_tool_calls=request.parallel_tool_calls,
            ):
                output_items.append(
                    FunctionToolCall(
                        name=tool.name,
                        arguments=tool.arguments,
                        call_id=tool.id or f"call_{uuid.uuid4().hex[:24]}",
                        status="completed",
                    )
                )
            continue
        if has_tools:
            normalized = normalize_assistant_output(
                text,
                has_tool_definitions=True,
            )
            if normalized.kind == "tool_calls":
                raise RuntimeError(
                    "Engine-side semantic parser was expected to provide parsed tool calls, "
                    "but only legacy post-hoc tool-call text was available."
                )
        # Regular message output
        msg = OutputMessage(
            status="completed",
            content=[OutputText(text=text, annotations=[])],
        )
        output_items.append(msg)

    prompt_tokens = len(engine_output.prompt_token_ids)
    usage = ResponseUsage(
        input_tokens=prompt_tokens,
        output_tokens=total_completion_tokens,
        total_tokens=prompt_tokens + total_completion_tokens,
        input_tokens_details=InputTokensDetails(cached_tokens=0),
        output_tokens_details=OutputTokensDetails(reasoning_tokens=total_reasoning_tokens),
    )

    # Detect truncation: if any completion hit max tokens, mark incomplete
    is_truncated = any(c.finish_reason == "length" for c in engine_output.outputs)
    status = "incomplete" if is_truncated else "completed"
    incomplete_details = IncompleteDetails(reason="max_output_tokens") if is_truncated else None

    return ResponseObject(
        id=resp_id,
        created_at=now,
        completed_at=now if status in ("completed", "incomplete") else None,
        status=status,
        model=request.model,
        output=output_items,
        usage=usage,
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
        incomplete_details=incomplete_details,
        max_tool_calls=request.max_tool_calls,
    )


def prepend_tool_executions(
    response: ResponseObject,
    tool_executions: list[ToolExecutionResult],
) -> ResponseObject:
    """Prepend server-executed MCP tool calls to a Response output."""
    if not tool_executions:
        return response

    executed_items = [
        FunctionToolCall(
            name=execution.tool_name,
            arguments=serialize_tool_call_arguments(execution.arguments),
            call_id=execution.tool_call_id,
            status="completed",
        )
        for execution in tool_executions
    ]
    return response.model_copy(update={"output": executed_items + list(response.output)})


def replace_response_tool_calls(
    response: ResponseObject,
    tool_calls: list[ToolCall],
) -> ResponseObject:
    """Replace function call items in a response, preserving non-tool outputs."""
    replacement_items = [
        FunctionToolCall(
            name=tool_call.function.name,
            arguments=serialize_tool_call_arguments(tool_call.function.arguments),
            call_id=tool_call.id,
            status="completed",
        )
        for tool_call in limit_parallel_tool_calls(
            tool_calls,
            parallel_tool_calls=response.parallel_tool_calls,
        )
    ]
    preserved_items = [item for item in response.output if not isinstance(item, FunctionToolCall)]
    return response.model_copy(update={"output": preserved_items + replacement_items})


def build_tools_prompt(tools: list[FunctionToolDefinition]) -> str:
    """Build a tools/functions description to append to the system prompt."""
    if not tools:
        return ""
    lines = ["You have access to the following functions:\n"]
    for tool in tools:
        lines.append(f"- {tool.name}")
        if tool.description:
            lines.append(f"  Description: {tool.description}")
        if tool.parameters:
            lines.append(f"  Parameters: {json.dumps(tool.parameters)}")
        lines.append("")
    lines.append(
        "When you want to call a function, respond with a JSON array of objects "
        'with "name" and "arguments" keys.'
    )
    return "\n".join(lines)
