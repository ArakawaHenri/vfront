"""Tool executor — agentic loop for server-side MCP tool execution."""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable, Iterable
from copy import deepcopy
from typing import Any

from loguru import logger

from vfront.adapter.assistant_output import parse_tool_calls_if_present
from vfront.adapter.tool_calls import limit_parallel_tool_calls
from vfront.frontend.service.mcp.client import MCPClientManager
from vfront.protocol.common import ToolCall
from vfront.shared.engine.generation import collect_final_stream_output
from vfront.shared.engine.types import GenerateOutput


class ToolExecutionResult:
    """Result of a single tool execution round."""

    __slots__ = ("tool_name", "tool_call_id", "arguments", "result_text", "is_error")

    def __init__(
        self,
        tool_name: str,
        tool_call_id: str,
        arguments: dict[str, Any],
        result_text: str,
        is_error: bool = False,
    ) -> None:
        self.tool_name = tool_name
        self.tool_call_id = tool_call_id
        self.arguments = arguments
        self.result_text = result_text
        self.is_error = is_error


def parse_tool_call_arguments(raw_arguments: str) -> dict[str, Any]:
    """Best-effort JSON decode for tool-call arguments."""
    try:
        return json.loads(raw_arguments)
    except (json.JSONDecodeError, KeyError, TypeError):
        return {}


def requires_explicit_tool_parser_for_tools(
    tools: Iterable[Any] | None,
    *,
    mcp_manager: MCPClientManager | None,
    name_getter: Callable[[Any], str | None],
) -> bool:
    """Return True when tool use must rely on an explicit semantic parser."""
    tool_names = [name for tool in tools or [] if (name := name_getter(tool))]
    if not tool_names:
        return False
    if mcp_manager is None:
        return True
    is_mcp_tool = getattr(mcp_manager, "is_mcp_tool", None)
    if callable(is_mcp_tool):
        return any(not is_mcp_tool(name) for name in tool_names)

    get_openai_tools = getattr(mcp_manager, "get_openai_tools", None)
    if callable(get_openai_tools):
        mcp_tool_names = {
            function.get("name")
            for tool in get_openai_tools() or []
            if isinstance(tool, dict)
            and isinstance(function := tool.get("function"), dict)
            and isinstance(function.get("name"), str)
        }
        return any(name not in mcp_tool_names for name in tool_names)

    return True


async def execute_mcp_tool_call(
    mcp_manager: MCPClientManager,
    tool_call: ToolCall,
    *,
    timeout: int,
) -> ToolExecutionResult:
    """Execute one MCP tool call with shared argument parsing and error handling."""
    args = parse_tool_call_arguments(tool_call.function.arguments)
    try:
        result_text = await mcp_manager.execute_tool(
            tool_call.function.name,
            args,
            timeout=timeout,
        )
        is_error = False
    except Exception as exc:
        logger.warning("MCP tool '{}' execution failed: {}", tool_call.function.name, exc)
        result_text = f"Error: {exc}"
        is_error = True

    return ToolExecutionResult(
        tool_name=tool_call.function.name,
        tool_call_id=tool_call.id,
        arguments=args,
        result_text=result_text,
        is_error=is_error,
    )


class AgenticLoopResult:
    """Final result after all agentic loop rounds complete."""

    __slots__ = (
        "final_output",
        "tool_executions",
        "rounds",
        "final_messages",
        "pending_client_calls",
    )

    def __init__(
        self,
        final_output: GenerateOutput,
        tool_executions: list[ToolExecutionResult],
        rounds: int,
        final_messages: list[dict[str, Any]],
        pending_client_calls: list[ToolCall] | None = None,
    ) -> None:
        self.final_output = final_output
        self.tool_executions = tool_executions
        self.rounds = rounds
        self.final_messages = final_messages
        self.pending_client_calls = list(pending_client_calls or [])


def append_final_output_to_messages(
    messages: list[dict[str, Any]],
    final_output: GenerateOutput,
    has_tool_definitions: bool = False,
    parallel_tool_calls: bool | None = True,
) -> list[dict[str, Any]]:
    """Append the final assistant turn to a conversation history."""
    updated_messages = deepcopy(messages)
    if not final_output.outputs:
        return updated_messages

    completion = final_output.outputs[0]

    # Prioritize parsed tool calls from semantic engine parser
    if completion.parsed_tool_calls is not None:
        parsed_tool_calls = limit_parallel_tool_calls(
            completion.parsed_tool_calls,
            parallel_tool_calls=parallel_tool_calls,
        )
        updated_messages.append(
            {
                "role": "assistant",
                "content": completion.semantic.content_delta if completion.semantic else None,
                "tool_calls": [
                    {
                        "id": tc.id or f"call_{uuid.uuid4().hex[:24]}",
                        "type": tc.type,
                        "function": {"name": tc.name, "arguments": tc.arguments},
                    }
                    for tc in parsed_tool_calls
                ],
            }
        )
        return updated_messages

    content = completion.text
    if content is not None:
        updated_messages.append(
            {
                "role": "assistant",
                "content": content,
            }
        )

    return updated_messages


async def run_agentic_loop(
    *,
    messages: list[dict[str, Any]],
    tools_json: list[dict[str, Any]] | None,
    mcp_manager: MCPClientManager,
    generate_fn: Any,
    tokenize_fn: Any,
    max_rounds: int = 10,
    max_tool_calls: int | None = None,
    timeout: int = 30,
    generate_params_factory: Callable[..., Any] | None = None,
    request_id_prefix: str = "agentic",
    parallel_tool_calls: bool | None = True,
) -> AgenticLoopResult:
    """Run the agentic loop: generate → detect tool calls → execute → repeat.

    Args:
        messages: Chat messages (will be mutated with tool results).
        tools_json: OpenAI-format tool definitions (client + MCP merged).
        mcp_manager: The MCP client manager for executing server-side tools.
        generate_fn: Async generator function(prompt_token_ids, params, request_id) → GenerateOutput.
        tokenize_fn: Async function(messages, tools) → token IDs.
        max_rounds: Maximum agentic loop iterations.
        max_tool_calls: Maximum total tool calls across all rounds (None = unlimited).
        timeout: Per-tool-call timeout in seconds.
        generate_params_factory: Callable that returns GenerateParams for each round.

    Returns:
        AgenticLoopResult with final output and all tool execution records.
    """
    all_tool_executions: list[ToolExecutionResult] = []
    current_messages = deepcopy(messages)
    round_num = 0
    total_tool_calls_used = 0

    while round_num < max_rounds:
        round_num += 1

        # Tokenize and generate
        prompt_token_ids = await tokenize_fn(current_messages, tools_json)
        params = generate_params_factory() if generate_params_factory else None
        if params is None:
            from vfront.shared.engine.types import GenerateParams

            params = GenerateParams()

        request_id = f"{request_id_prefix}-{round_num}"

        # Collect full output
        final_output = await collect_final_stream_output(
            generate_fn(prompt_token_ids, params, request_id),
            empty_error_message="Engine returned no output in agentic loop",
        )

        # Check for tool calls in the output
        completion = final_output.outputs[0]
        if completion.parsed_tool_calls is not None:
            tool_calls = [
                ToolCall(
                    id=tc.id or f"call_{uuid.uuid4().hex[:24]}",
                    type=tc.type,
                    function={"name": tc.name, "arguments": tc.arguments},
                )
                for tc in completion.parsed_tool_calls
            ]
        else:
            tool_calls = list(parse_tool_calls_if_present(completion.text or "") or [])
        tool_calls = limit_parallel_tool_calls(
            tool_calls,
            parallel_tool_calls=parallel_tool_calls,
        )

        if not tool_calls:
            # No tool calls — we're done
            return AgenticLoopResult(
                final_output=final_output,
                tool_executions=all_tool_executions,
                rounds=round_num,
                final_messages=append_final_output_to_messages(
                    current_messages,
                    final_output,
                    has_tool_definitions=bool(tools_json),
                    parallel_tool_calls=parallel_tool_calls,
                ),
                pending_client_calls=[],
            )

        # Enforce max_tool_calls: trim if adding all calls would exceed the limit
        if max_tool_calls is not None:
            remaining = max_tool_calls - total_tool_calls_used
            if remaining <= 0:
                # Limit already reached — treat as if no tools were called (final answer)
                logger.info("max_tool_calls ({}) reached; stopping tool loop", max_tool_calls)
                break
            if len(tool_calls) > remaining:
                tool_calls = tool_calls[:remaining]

        # Check which tool calls are MCP (server-side) vs client-side
        mcp_calls: list[ToolCall] = []
        client_calls: list[ToolCall] = []
        for tc in tool_calls:
            fn_name = tc.function.name
            if mcp_manager.is_mcp_tool(fn_name):
                mcp_calls.append(tc)
            else:
                client_calls.append(tc)

        if client_calls and not mcp_calls:
            # All tool calls are client-side — return to client
            current_messages.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [tc.model_dump() for tc in tool_calls],
                }
            )
            return AgenticLoopResult(
                final_output=final_output,
                tool_executions=all_tool_executions,
                rounds=round_num,
                final_messages=deepcopy(current_messages),
                pending_client_calls=client_calls,
            )

        # Add assistant message with tool calls
        current_messages.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [tc.model_dump() for tc in tool_calls],
            }
        )

        # Execute MCP tool calls
        for tc in mcp_calls:
            execution = await execute_mcp_tool_call(
                mcp_manager,
                tc,
                timeout=timeout,
            )
            all_tool_executions.append(execution)

            # Add tool result to messages
            current_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": execution.tool_call_id,
                    "content": execution.result_text,
                }
            )

        # If there are also client-side calls, we can't execute them
        if client_calls:
            logger.info(
                "Mixed MCP/client tool calls in round {}; "
                "executed {} MCP calls, {} client calls pending",
                round_num,
                len(mcp_calls),
                len(client_calls),
            )
            return AgenticLoopResult(
                final_output=final_output,
                tool_executions=all_tool_executions,
                rounds=round_num,
                final_messages=deepcopy(current_messages),
                pending_client_calls=client_calls,
            )

        logger.debug("Agentic round {}: executed {} MCP tool calls", round_num, len(mcp_calls))
        total_tool_calls_used += len(mcp_calls)

        if max_tool_calls is not None and total_tool_calls_used >= max_tool_calls:
            logger.info("max_tool_calls ({}) reached after round {}", max_tool_calls, round_num)
            break

    # Max rounds or max_tool_calls reached — do one final generation without tools
    logger.warning("Agentic loop stopped (rounds={}, tool_calls_used={})", round_num, total_tool_calls_used)
    # Do one final generation without tools to get a text response
    prompt_token_ids = await tokenize_fn(current_messages, tools_json)
    params = generate_params_factory() if generate_params_factory else None
    if params is None:
        from vfront.shared.engine.types import GenerateParams

        params = GenerateParams()

    final_output = await collect_final_stream_output(
        generate_fn(
            prompt_token_ids,
            params,
            f"{request_id_prefix}-final",
        ),
        empty_error_message="Engine returned no output in final agentic round",
    )

    return AgenticLoopResult(
        final_output=final_output,
        tool_executions=all_tool_executions,
        rounds=round_num,
        final_messages=append_final_output_to_messages(
            current_messages,
            final_output,
            has_tool_definitions=bool(tools_json),
            parallel_tool_calls=parallel_tool_calls,
        ),
        pending_client_calls=[],
    )
