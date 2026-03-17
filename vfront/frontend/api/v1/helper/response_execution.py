"""Shared response-execution helpers for API v1 routes."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from vfront.adapter.response import (
    build_generation_messages,
    prepend_tool_executions,
    replace_response_tool_calls,
    request_output_to_response,
    response_request_to_generate_params,
    response_tool_choice_to_chat_tool_choice,
    response_tools_to_chat_tools,
)
from vfront.adapter.sampling import extract_multimodal_data
from vfront.frontend.middleware.exceptions import InvalidRequestError
from vfront.frontend.service.mcp.client import MCPClientManager
from vfront.frontend.service.mcp.executor import (
    ToolExecutionResult,
    append_final_output_to_messages,
    run_agentic_loop,
)
from vfront.protocol.common import ToolCall
from vfront.protocol.response import (
    FunctionToolDefinition,
    ResponseCreateRequest,
    ResponseObject,
)
from vfront.shared.engine.generation import collect_final_stream_output
from vfront.shared.engine.reasoning import ReasoningEffort
from vfront.shared.engine.tokenization import tokenize_chat_messages
from vfront.shared.engine.types import GenerateOutput


@dataclass(slots=True)
class ResponseGenerationResult:
    final_output: GenerateOutput
    final_messages: list[dict[str, Any]]
    tool_executions: list[ToolExecutionResult]
    pending_client_calls: list[ToolCall]


@dataclass(slots=True)
class PreparedResponseGeneration:
    generation_messages: list[dict[str, Any]]
    params: Any


async def tokenize_response_messages(
    runtime: Any,
    generation_messages: list[dict[str, Any]],
    *,
    reasoning_effort: ReasoningEffort | None = None,
) -> list[int]:
    """Tokenize a Responses prompt and enforce model context limits."""
    try:
        prompt_token_ids = await tokenize_chat_messages(
            runtime,
            generation_messages,
            reasoning_effort=reasoning_effort,
        )
    except ValueError as exc:
        raise InvalidRequestError(str(exc), param="reasoning.effort") from exc
    max_len = getattr(runtime, "max_model_len", 0)
    if max_len > 0 and len(prompt_token_ids) > max_len:
        raise InvalidRequestError(
            f"Prompt length {len(prompt_token_ids)} tokens exceeds model context window {max_len}.",
            code="context_length_exceeded",
        )
    return prompt_token_ids


def prepare_response_generation(
    *,
    request: ResponseCreateRequest,
    conversation_messages: list[dict[str, Any]],
    tools: list[FunctionToolDefinition] | None,
    lora_name: str | None,
    tool_parser: str | None,
) -> PreparedResponseGeneration:
    """Prepare generation messages and params for a Responses request."""
    generation_messages = build_generation_messages(conversation_messages, tools)
    params = response_request_to_generate_params(request)
    params.lora_name = lora_name
    params.multimodal_data = extract_multimodal_data(generation_messages)
    params.tools = response_tools_to_chat_tools(tools)
    params.tool_choice = response_tool_choice_to_chat_tool_choice(request.tool_choice)
    params.tool_parser = tool_parser
    return PreparedResponseGeneration(
        generation_messages=generation_messages,
        params=params,
    )


def build_response_from_generation(
    *,
    engine_output: GenerateOutput,
    request: ResponseCreateRequest,
    response_id: str,
    created_at: float,
    tool_executions: list[ToolExecutionResult] | None = None,
    pending_client_calls: list[ToolCall] | None = None,
) -> ResponseObject:
    """Build a public response object from engine output and tool execution state."""
    response = request_output_to_response(
        engine_output=engine_output,
        request=request,
        response_id=response_id,
        created_at=created_at,
    )
    if pending_client_calls:
        return replace_response_tool_calls(response, list(pending_client_calls))
    return prepend_tool_executions(response, list(tool_executions or []))


async def execute_response_generation(
    *,
    conversation_messages: list[dict[str, Any]],
    generation_messages: list[dict[str, Any]],
    tools: list[FunctionToolDefinition] | None,
    parallel_tool_calls: bool | None,
    max_tool_calls: int | None,
    runtime: Any,
    params: Any,
    response_id: str,
    use_agentic_tools: bool,
    mcp_manager: MCPClientManager | None,
    max_tool_rounds: int,
    timeout: int,
) -> ResponseGenerationResult:
    """Execute a non-streaming Responses request with shared tool-loop semantics."""
    if use_agentic_tools:
        if mcp_manager is None:
            raise RuntimeError("MCP manager is required for agentic response generation.")
        tools_json = [tool.model_dump() for tool in tools] if tools else None
        result = await run_agentic_loop(
            messages=conversation_messages,
            tools_json=tools_json,
            mcp_manager=mcp_manager,
            generate_fn=runtime.generate,
            tokenize_fn=lambda loop_messages, loop_tools: tokenize_response_messages(
                runtime,
                build_generation_messages(loop_messages, tools),
                reasoning_effort=getattr(params, "reasoning_effort", None),
            ),
            max_rounds=max_tool_rounds,
            max_tool_calls=max_tool_calls,
            timeout=timeout,
            generate_params_factory=lambda: deepcopy(params),
            request_id_prefix=response_id,
            parallel_tool_calls=parallel_tool_calls,
        )
        return ResponseGenerationResult(
            final_output=result.final_output,
            final_messages=result.final_messages,
            tool_executions=result.tool_executions,
            pending_client_calls=result.pending_client_calls,
        )

    prompt_token_ids = await tokenize_response_messages(
        runtime,
        generation_messages,
        reasoning_effort=getattr(params, "reasoning_effort", None),
    )
    final_output = await collect_final_stream_output(
        runtime.generate(prompt_token_ids, params, response_id),
        empty_error_message="Engine returned no output",
    )

    return ResponseGenerationResult(
        final_output=final_output,
        final_messages=append_final_output_to_messages(
            conversation_messages,
            final_output,
            has_tool_definitions=bool(tools),
            parallel_tool_calls=parallel_tool_calls,
        ),
        tool_executions=[],
        pending_client_calls=[],
    )
